import os
import sys
sys.path.append('../src')

import torch
from torch import nn
import torch.utils.benchmark as benchmark
from pathlib import Path

from transformer import Transformer
from trainer import run_one_epoch, init_metrics
from utils import get_loss
from utils.get_data import get_data_loader, get_dataset

torch.set_num_threads(10)

device = 'cuda:0'
dataset_name = 'tracking-60k'
batch_size = 1

def get_regions(num_regions: int, num_or_hashes: int, num_heads: int, num_and_hashes: int = 2) -> torch.Tensor:
    """Get the regions for the E2LSH hashing.

    Args:
        num_regions (int): The number of regions.
        num_or_hashes (int): The number of OR hashes.
        num_heads (int): The number of heads.
        num_and_hashes (int): The number of AND hashes.

    Returns:
        torch.Tensor: The regions for the E2LSH hashing, of shape (num_and_hashes, num_heads * num_or_hashes).
    """
    lower = 2
    upper = 2 * num_regions ** (1 / num_and_hashes) - lower

    regions = torch.tensor([
        [torch.rand(1).item() * (upper - lower) + lower for _ in range(num_and_hashes)]
        for _ in range(num_heads * num_or_hashes)
    ])

    regions = (num_regions / regions.prod(dim=1, keepdim=True)) ** (1 / num_and_hashes) * regions
    regions = torch.round(regions * 3) / 3
    return regions.permute(1, 0)


def quantile_partition(sorted_indices: torch.Tensor, regions: torch.Tensor) -> torch.Tensor:
    """Partition the sorted indices into regions.

    Args:
        sorted_indices (torch.Tensor): The sorted indices, of shape (num_elements).
        regions (torch.Tensor): The regions, of shape (num_heads * n_hashes, 1).

    Returns:
        torch.Tensor: The partitioned indices, of shape (num_heads * n_hashes, num_elements).
    """
    num_elements = sorted_indices.shape[-1]
    region_size = torch.ceil(num_elements / regions)
    inverse_indices = torch.argsort(sorted_indices, dim=-1)

    base = torch.arange(num_elements, device=sorted_indices.device)[None]
    region_indices = base // region_size + 1
    reassigned_regions = region_indices[:, inverse_indices]
    return reassigned_regions


def bit_shift(base: torch.Tensor, shift_idx: torch.Tensor) -> torch.Tensor:
    """Perform a bitwise shift and combination of two tensors.

    Args:
        base (torch.Tensor): The base tensor, of shape (num_heads * n_hashes, num_elements).
        shift_idx (torch.Tensor): The tensor to be shifted, of shape (num_heads * n_hashes, num_elements).

    Returns:
        torch.Tensor: The combined tensor after bitwise shift, of shape (num_heads * n_hashes, num_elements).
    """
    max_base = base.max(dim=1, keepdim=True).values
    num_bits = torch.ceil(torch.log2(max_base + 1)).long()
    return (shift_idx << num_bits) | base


def pad_and_unpad(
        batch: torch.Tensor, 
        block_size: int, 
        region_indices: torch.Tensor, 
        raw_sizes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate the padding and unpadding sequences for the input tensors.

    Args:
        batch (torch.Tensor): The batch vector, of shape (num_elements).
        block_size (int): The block size.
        region_indices (torch.Tensor): The region indices, of shape (num_elements).
        raw_sizes (torch.Tensor): The raw sizes of the graphs, of shape (num_graphs).

    Returns:
        tuple: A tuple containing the padding and unpadding sequences.
    """
    padded_sizes = ((raw_sizes + block_size - 1) // block_size) * block_size
    pad_sizes = padded_sizes - raw_sizes

    pad_cumsum = padded_sizes.cumsum(0)
    pad_seq = torch.arange(pad_cumsum[-1], device=batch.device)
    unpad_seq = torch.ones(pad_cumsum[-1], device=batch.device).bool()

    sorted_region_indices = region_indices.argsort()
    for i in range(len(raw_sizes)):
        idx_to_fill = pad_cumsum[i] - block_size - pad_sizes[i] + torch.arange(pad_sizes[i], device=batch.device)
        if i >= 1:
            pad_seq[pad_cumsum[i - 1] :] -= pad_sizes[i - 1]
            idx_to_fill -= pad_sizes[:i].sum()
        pad_seq[pad_cumsum[i] - pad_sizes[i] : pad_cumsum[i]] = sorted_region_indices[idx_to_fill]
        unpad_seq[pad_cumsum[i] - pad_sizes[i] : pad_cumsum[i]] = False
    
    return pad_seq, unpad_seq


def prepare_input(
        x: torch.Tensor, 
        coords: torch.Tensor, 
        batch: torch.Tensor, 
        num_regions: int, 
        num_heads: int, 
        n_hashes: int,
        block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare the input for the Transformer.

    Args:
        x (torch.Tensor): The input features, of shape (num_elements, in_dim).
        coords (torch.Tensor): The coordinates of the nodes, of shape (num_elements, coords_dim).
        batch (torch.Tensor): The batch vector, of shape (num_elements).
        num_regions (int): The number of regions.
        num_heads (int): The number of heads.
        n_hashes (int): The number of OR hashes.
        block_size (int): The block size.

    Returns:
        tuple: A tuple containing:
            - The padded input features, of shape (padded_size, in_dim).
            - The combined shifts, of shape (n_hashes, num_heads, padded_size).
            - The padded coordinates, of shape (padded_size, coords_dim).
            - The unpadding sequence, of shape (padded_size).
    """
    regions = get_regions(num_regions, n_hashes, num_heads)
    graph_sizes = batch.bincount()

    graph_size_cumsum = graph_sizes.cumsum(0)

    region_indices_eta, region_indices_phi = [], []
    for graph_idx in range(len(graph_size_cumsum)):
        start_idx = 0 if graph_idx == 0 else graph_size_cumsum[graph_idx - 1]
        end_idx = graph_size_cumsum[graph_idx]

        sorted_eta_idx = torch.argsort(coords[start_idx:end_idx, 0], dim=-1)
        sorted_phi_idx = torch.argsort(coords[start_idx:end_idx, 1], dim=-1)

        region_indices_eta.append(quantile_partition(sorted_eta_idx, regions[0][:, None]))
        region_indices_phi.append(quantile_partition(sorted_phi_idx, regions[1][:, None]))

    region_indices_eta = torch.cat(region_indices_eta, dim=-1) # (num_heads * n_hashes, num_elements)
    region_indices_phi = torch.cat(region_indices_phi, dim=-1) # (num_heads * n_hashes, num_elements)

    combined_shifts = bit_shift(region_indices_eta.long(), region_indices_phi.long())
    combined_shifts = bit_shift(combined_shifts, batch[None])
    combined_shifts = combined_shifts.view(n_hashes, num_heads, -1)

    pad_seq, unpad_seq = pad_and_unpad(batch, block_size, combined_shifts[0, 0], graph_sizes)
    x = x[pad_seq]
    combined_shifts = combined_shifts[..., pad_seq]
    coords = coords[pad_seq]

    return x, combined_shifts, coords, unpad_seq


class Transformer(nn.Module):
    def __init__(self, in_dim, coords_dim, **kwargs):
        super().__init__()
        self.coords_dim = coords_dim
        self.n_layers = 2
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]
        #self.out_dim = kwargs['out_dim']
        self.block_size = kwargs["block_size"]
        self.n_hashes = kwargs["n_hashes"]
        self.num_w_per_dist = kwargs["num_w_per_dist"]
        self.mlp_out_hdim = kwargs['h_dim']//2

        # Feature encoder
        self.feat_encoder = nn.Sequential(
            nn.Linear(in_dim, self.dim_per_head),
            nn.ReLU(),
            nn.Linear(self.dim_per_head, self.dim_per_head),
        )

        # Attention layer 1
        self.norm1_1 = nn.LayerNorm(self.dim_per_head)
        self.norm1_2 = nn.LayerNorm(self.dim_per_head)
        self.w_q1 = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_k1 = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_v1 = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_rpe1 = nn.Parameter(torch.randn(self.num_heads * self.dim_per_head, self.num_w_per_dist * (self.coords_dim - 1)))
        self.alpha1 = nn.Parameter(torch.normal(0, 1, (self.num_heads, self.dim_per_head + self.coords_dim, self.n_hashes)))
        self.alpha1.requires_grad = False
        self.out_linear1 = nn.Linear(self.num_heads * self.dim_per_head, self.dim_per_head)
        self.ff1 = nn.Sequential(
            nn.Linear(self.dim_per_head, self.dim_per_head),
            nn.ReLU(),
            nn.Linear(self.dim_per_head, self.dim_per_head),
        )

        # Attention layer 2
        self.norm2_1 = nn.LayerNorm(self.dim_per_head)
        self.norm2_2 = nn.LayerNorm(self.dim_per_head)
        self.w_q2 = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_k2 = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_v2 = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_rpe2 = nn.Parameter(torch.randn(self.num_heads * self.dim_per_head, self.num_w_per_dist * (self.coords_dim - 1)))
        self.alpha2 = nn.Parameter(torch.normal(0, 1, (self.num_heads, self.dim_per_head + self.coords_dim, self.n_hashes)))
        self.alpha2.requires_grad = False
        self.out_linear2 = nn.Linear(self.num_heads * self.dim_per_head, self.dim_per_head)
        self.ff2 = nn.Sequential(
            nn.Linear(self.dim_per_head, self.dim_per_head),
            nn.ReLU(),
            nn.Linear(self.dim_per_head, self.dim_per_head),
        )

        # After attentions
        self.W = nn.Linear(self.dim_per_head * (self.n_layers + 1), int(self.dim_per_head // 2), bias=False)

        self.mlp_out = nn.Sequential(
            nn.Linear(int(self.dim_per_head // 2), self.mlp_out_hdim),
            nn.LayerNorm(self.mlp_out_hdim),
            nn.Tanh(),
            nn.Linear(self.mlp_out_hdim, self.mlp_out_hdim),
            nn.LayerNorm(self.mlp_out_hdim),
            nn.Tanh(),
            nn.Linear(self.mlp_out_hdim, self.mlp_out_hdim),
            nn.LayerNorm(self.mlp_out_hdim),
            nn.Tanh(),
            nn.Linear(self.mlp_out_hdim, self.mlp_out_hdim),
            nn.LayerNorm(self.mlp_out_hdim),
            nn.Tanh(),
            nn.Linear(self.mlp_out_hdim, int(self.dim_per_head // 2)),
        )
        #self.out_proj = nn.Linear(int(self.dim_per_head // 2), num_classes)
        #self.out_proj = nn.Linear(int(self.dim_per_head // 2), self.out_dim)

    def forward(self, x, combined_shifts, coords, unpad_seq):
        # Encode features


        x = self.feat_encoder(x) # (padded_size, dim_per_head)

        all_encoded_x = x

        ###########################################################################################
        # Attention layer 1
        x_normed = self.norm1_1(x)

        q, k, v = self.w_q1(x_normed), self.w_k1(x_normed), self.w_v1(x_normed)



        q = q.view(-1, self.num_heads, self.dim_per_head)
        k = k.view(-1, self.num_heads, self.dim_per_head)
        v = v.view(-1, self.num_heads, self.dim_per_head)

        # Prep q, k
        w = self.w_rpe1.view(self.num_heads, self.dim_per_head, -1, self.num_w_per_dist)
        qw = w.sum(dim=1).clamp(max=50).exp().sum(dim=-1)
        new_qw_expand_dim = torch.cat([qw[:, :1], qw], dim=-1) # (num_heads, coords_dim)
        
        sqrt_w_r = torch.sqrt(2 * new_qw_expand_dim).unsqueeze(0) * coords.unsqueeze(1) # (padded_size, num_heads, coords_dim)
        
        q_hat = torch.cat([q, sqrt_w_r], dim=-1).permute(1, 0, 2) # (num_heads, padded_size, dim_per_head + coords_dim)
        k_hat = torch.cat([k, sqrt_w_r], dim=-1).permute(1, 0, 2) # (num_heads, padded_size, dim_per_head + coords_dim)
        value = v.permute(1, 0, 2) # (num_heads, padded_size, dim_per_head)

        # E2LSH
        with torch.no_grad():
            q_hashed = torch.bmm(q_hat, self.alpha1).permute(2, 0, 1) # (n_hashes, num_heads, padded_size)
            k_hashed = torch.bmm(k_hat, self.alpha1).permute(2, 0, 1) # (n_hashes, num_heads, padded_size)
            max_hash_shift = torch.max(q_hashed.max(-1, keepdim=True).values, k_hashed.max(-1, keepdim=True).values)
            min_hash_shift = torch.min(q_hashed.min(-1, keepdim=True).values, k_hashed.min(-1, keepdim=True).values)
            hash_shift = max_hash_shift - min_hash_shift # (n_hashes, num_heads, 1)
        
        combined_shifts = combined_shifts * hash_shift # (n_hashes, num_heads, padded_size)
        q_hashed = q_hashed + combined_shifts
        k_hashed = k_hashed + combined_shifts

        q_positions = q_hashed.argsort(dim=-1)
        k_positions = k_hashed.argsort(dim=-1)

        # Sort to buckets
        q_positions_expanded = q_positions.unsqueeze(-1).expand(q_positions.shape + (self.dim_per_head + self.coords_dim,))
        q_hat_expanded = q_hat.unsqueeze(0).expand(q_positions_expanded.shape[:-2] + q_hat.shape[-2:]) # (n_hashes, num_heads, padded_size, dim_per_head + coords_dim)
        q_batch_selected = q_hat_expanded.gather(-2, q_positions_expanded)
        s_query = q_batch_selected.view(self.n_hashes, self.num_heads, -1, self.block_size, self.dim_per_head + self.coords_dim)

        k_positions_unsqueezed = k_positions.unsqueeze(-1)
        k_positions_expanded = k_positions_unsqueezed.expand(k_positions.shape + (self.dim_per_head + self.coords_dim,))
        k_hat_expanded = k_hat.unsqueeze(0).expand(k_positions_expanded.shape[:-2] + k_hat.shape[-2:]) # (n_hashes, num_heads, padded_size, dim_per_head + coords_dim)
        k_batch_selected = k_hat_expanded.gather(-2, k_positions_expanded)
        s_key = k_batch_selected.view(self.n_hashes, self.num_heads, -1, self.block_size, self.dim_per_head + self.coords_dim)

        v_positions_expanded = k_positions_unsqueezed.expand(k_positions.shape + (self.dim_per_head,))
        v_hat_expanded = value.unsqueeze(0).expand(v_positions_expanded.shape[:-2] + value.shape[-2:]) # (n_hashes, num_heads, padded_size, dim_per_head)
        v_batch_selected = v_hat_expanded.gather(-2, v_positions_expanded)
        s_value = v_batch_selected.view(self.n_hashes, self.num_heads, -1, self.block_size, self.dim_per_head)

        # Compute attention
        q_sq_05 = -0.5 * (s_query**2).sum(dim=-1, keepdim=True)
        k_sq_05 = -0.5 * (s_key**2).sum(dim=-1, keepdim=True)

        clustered_dists = torch.einsum("...id,...jd->...ij", s_query, s_key) # (n_hashes, num_heads, -1, block_size, block_size)
        qk = (clustered_dists + q_sq_05 + k_sq_05.transpose(-1, -2)).clamp(max=0.0).exp()

        denom = qk.sum(dim=-1, keepdim=True) + (1e-20) # (n_hashes, num_heads, -1, block_size, 1)
        so = torch.einsum("...ij,...jd->...id", qk, s_value) # (n_hashes, num_heads, -1, block_size, dim_per_head)

        # Unsort from buckets
        arange = torch.arange(q_positions.shape[-1], device=q_positions.device).expand_as(q_positions)
        q_rev_positions = torch.empty_like(q_positions).scatter(-1, q_positions, arange)

        so_squeezed = so.view(self.n_hashes, self.num_heads, -1, self.dim_per_head)
        q_rev_positions_expanded = q_rev_positions.unsqueeze(-1).expand(q_rev_positions.shape + (self.dim_per_head,))
        so_squeezed_expanded = so_squeezed.expand(q_rev_positions_expanded.shape[:-2] + so_squeezed.shape[-2:])
        o = so_squeezed_expanded.gather(-2, q_rev_positions_expanded)

        denom_squeezed = denom.view(self.n_hashes, self.num_heads, -1, 1)
        q_rev_positions_expanded = q_rev_positions.unsqueeze(-1)
        denom_squeezed_expanded = denom_squeezed.expand(q_rev_positions_expanded.shape[:-2] + denom_squeezed.shape[-2:])
        logits = denom_squeezed_expanded.gather(-2, q_rev_positions_expanded)

        aggr_out = o.sum(dim=0) / logits.sum(dim=0) # (num_heads, padded_size, dim_per_head)
        aggr_out = aggr_out.view(-1, self.num_heads * self.dim_per_head)
        aggr_out = self.out_linear1(aggr_out) # (padded_size, dim_per_head)

        x = x + aggr_out
        x = x + self.ff1(self.norm1_2(x))

        all_encoded_x = torch.cat([all_encoded_x, x], dim=-1)

        ###########################################################################################
        # Attention layer 2
        x_normed = self.norm2_1(x)

        q, k, v = self.w_q2(x_normed), self.w_k2(x_normed), self.w_v2(x_normed)
        q = q.view(-1, self.num_heads, self.dim_per_head)
        k = k.view(-1, self.num_heads, self.dim_per_head)
        v = v.view(-1, self.num_heads, self.dim_per_head)

        # Prep q, k
        w = self.w_rpe2.view(self.num_heads, self.dim_per_head, -1, self.num_w_per_dist)
        qw = w.sum(dim=1).clamp(max=50).exp().sum(dim=-1)
        new_qw_expand_dim = torch.cat([qw[:, :1], qw], dim=-1) # (num_heads, coords_dim)
        sqrt_w_r = torch.sqrt(2 * new_qw_expand_dim).unsqueeze(0) * coords.unsqueeze(1) # (padded_size, num_heads, coords_dim)
        q_hat = torch.cat([q, sqrt_w_r], dim=-1).permute(1, 0, 2) # (num_heads, padded_size, dim_per_head + coords_dim)
        k_hat = torch.cat([k, sqrt_w_r], dim=-1).permute(1, 0, 2) # (num_heads, padded_size, dim_per_head + coords_dim)
        value = v.permute(1, 0, 2) # (num_heads, padded_size, dim_per_head)

        # E2LSH
        with torch.no_grad():
            q_hashed = torch.bmm(q_hat, self.alpha2).permute(2, 0, 1) # (n_hashes, num_heads, padded_size)
            k_hashed = torch.bmm(k_hat, self.alpha2).permute(2, 0, 1) # (n_hashes, num_heads, padded_size)
            max_hash_shift = torch.max(q_hashed.max(-1, keepdim=True).values, k_hashed.max(-1, keepdim=True).values)
            min_hash_shift = torch.min(q_hashed.min(-1, keepdim=True).values, k_hashed.min(-1, keepdim=True).values)
            hash_shift = max_hash_shift - min_hash_shift # (n_hashes, num_heads, 1)
        
        combined_shifts = combined_shifts * hash_shift # (n_hashes, num_heads, padded_size)
        q_hashed = q_hashed + combined_shifts
        k_hashed = k_hashed + combined_shifts

        q_positions = q_hashed.argsort(dim=-1)
        k_positions = k_hashed.argsort(dim=-1)

        # Sort to buckets
        q_positions_expanded = q_positions.unsqueeze(-1).expand(q_positions.shape + (self.dim_per_head + self.coords_dim,))
        q_hat_expanded = q_hat.unsqueeze(0).expand(q_positions_expanded.shape[:-2] + q_hat.shape[-2:]) # (n_hashes, num_heads, padded_size, dim_per_head + coords_dim)
        q_batch_selected = q_hat_expanded.gather(-2, q_positions_expanded)
        s_query = q_batch_selected.view(self.n_hashes, self.num_heads, -1, self.block_size, self.dim_per_head + self.coords_dim)

        k_positions_expanded = k_positions.unsqueeze(-1).expand(k_positions.shape + (self.dim_per_head + self.coords_dim,))
        k_hat_expanded = k_hat.unsqueeze(0).expand(k_positions_expanded.shape[:-2] + k_hat.shape[-2:]) # (n_hashes, num_heads, padded_size, dim_per_head + coords_dim)
        k_batch_selected = k_hat_expanded.gather(-2, k_positions_expanded)
        s_key = k_batch_selected.view(self.n_hashes, self.num_heads, -1, self.block_size, self.dim_per_head + self.coords_dim)

        v_positions_expanded = k_positions.unsqueeze(-1).expand(k_positions.shape + (self.dim_per_head,))
        v_hat_expanded = value.unsqueeze(0).expand(v_positions_expanded.shape[:-2] + value.shape[-2:]) # (n_hashes, num_heads, padded_size, dim_per_head)
        v_batch_selected = v_hat_expanded.gather(-2, v_positions_expanded)
        s_value = v_batch_selected.view(self.n_hashes, self.num_heads, -1, self.block_size, self.dim_per_head)

        # Compute attention
        q_sq_05 = -0.5 * (s_query**2).sum(dim=-1, keepdim=True)
        k_sq_05 = -0.5 * (s_key**2).sum(dim=-1, keepdim=True)

        clustered_dists = torch.einsum("...id,...jd->...ij", s_query, s_key) # (n_hashes, num_heads, -1, block_size, block_size)
        qk = (clustered_dists + q_sq_05 + k_sq_05.transpose(-1, -2)).clamp(max=0.0).exp()

        denom = qk.sum(dim=-1, keepdim=True) + (1e-20) # (n_hashes, num_heads, -1, block_size, 1)
        so = torch.einsum("...ij,...jd->...id", qk, s_value) # (n_hashes, num_heads, -1, block_size, dim_per_head)

        # Unsort from buckets
        arange = torch.arange(q_positions.shape[-1], device=q_positions.device).expand_as(q_positions)
        q_rev_positions = torch.empty_like(q_positions).scatter(-1, q_positions, arange)

        so_squeezed = so.view(self.n_hashes, self.num_heads, -1, self.dim_per_head)
        q_rev_positions_expanded = q_rev_positions.unsqueeze(-1).expand(q_rev_positions.shape + (self.dim_per_head,))
        so_squeezed_expanded = so_squeezed.expand(q_rev_positions_expanded.shape[:-2] + so_squeezed.shape[-2:])
        o = so_squeezed_expanded.gather(-2, q_rev_positions_expanded)

        denom_squeezed = denom.view(self.n_hashes, self.num_heads, -1, 1)
        q_rev_positions_expanded = q_rev_positions.unsqueeze(-1)
        denom_squeezed_expanded = denom_squeezed.expand(q_rev_positions_expanded.shape[:-2] + denom_squeezed.shape[-2:])
        logits = denom_squeezed_expanded.gather(-2, q_rev_positions_expanded)

        aggr_out = o.sum(dim=0) / logits.sum(dim=0) # (num_heads, padded_size, dim_per_head)
        aggr_out = aggr_out.view(-1, self.num_heads * self.dim_per_head)
        aggr_out = self.out_linear2(aggr_out) # (padded_size, dim_per_head)

        x = x + aggr_out
        x = x + self.ff2(self.norm2_2(x))

        all_encoded_x = torch.cat([all_encoded_x, x], dim=-1)

        ###########################################################################################

        x = torch.tanh(self.W(all_encoded_x))
        x = x + self.mlp_out(x)

        #out = self.out_proj(torch.mean(x[unpad_seq], dim=0))

        return x


model_configs = {'block_size': 100, 'n_hashes': 3, 'num_regions': 150, 'num_heads': 8, 'h_dim': 24, 'n_layers': 4, 'num_w_per_dist': 10}
torch.cuda.set_device(device)

dataset_dir = Path('../data/') / dataset_name.split("-")[0]
dataset = get_dataset(dataset_name, dataset_dir)

loaders = get_data_loader(dataset, dataset.idx_split, batch_size=batch_size)

model = Transformer(in_dim=dataset.x_dim, coords_dim=dataset.coords_dim, **model_configs)

for batch in loaders["train"]:
    x, combined_shifts, coords, unpad_seq = prepare_input(batch.x,batch.coords, batch.batch, model_configs["num_regions"],model_configs["num_heads"], model_configs["n_hashes"],model_configs['block_size'])
    result = model(x, combined_shifts, coords, unpad_seq)
    print (result)
