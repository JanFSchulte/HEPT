import torch
import torch.nn as nn
from typing import List

from einops import rearrange
from ..model_utils.hash_utils import lsh_mapping, batched_index_select, invert_permutation, E2LSH

class FullAttention(nn.Module):
    def __init__(self, hash_dim, **kwargs):
        super().__init__()
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]
        self.out_linear = nn.Linear(self.num_heads * self.dim_per_head, self.dim_per_head)

        self.block_size = kwargs["block_size"]
        self.n_hashes = kwargs["n_hashes"]
        self.num_w_per_dist = kwargs["num_w_per_dist"]
        self.e2lsh = E2LSH(n_hashes=self.n_hashes, n_heads=self.num_heads, dim=hash_dim)

    def forward(self, query, key, value, padding_mask,**kwargs):
        
        query = query.view(-1, self.num_heads, self.dim_per_head)
        key = key.view(-1, self.num_heads, self.dim_per_head)
        value = value.view(-1, self.num_heads, self.dim_per_head)

        w = rearrange(
            kwargs["w_rpe"].weight,
            "(h d) (r k) -> h d r k",
            h=self.num_heads,
            d=self.dim_per_head,
            k=self.num_w_per_dist,
        )
        q_hat, k_hat = prep_qk(query, key, w, kwargs["coords"])
        q_hat = rearrange(q_hat, "n h d -> h n d")
        k_hat = rearrange(k_hat, "n h d -> h n d")
        value = rearrange(value, "n h d -> h n d")

        out =  attention(q_hat, k_hat, value, padding_mask)
        return self.out_linear(rearrange(out, "h n d -> n (h d)"))


def prep_qk(query, key, w, coords):
    qw = w.sum(dim=1).clamp(max=50).exp().sum(dim=-1)
    new_qw_expand_dim = torch.cat([qw[:, :1], qw], dim=-1)
    sqrt_w_r = torch.sqrt(2 * new_qw_expand_dim)[None] * coords[:, None]
    q_hat = torch.cat([query, sqrt_w_r], dim=-1)
    k_hat = torch.cat([key, sqrt_w_r], dim=-1)
    return q_hat, k_hat




def attention(Q, K, V, padding_mask):
    """
    Q, K, V: Tensors of shape (h, N*B, d)
      - h = num_heads
      - N*B = flattened dimension for sequence_length x batch_size
      - d = embedding dimension per head
    padding_mask: (B, N) boolean tensor
      - True indicates a real (non-padded) token
      - False indicates a padded position

    Returns:
      out: Tensor of shape (h, N*B, d), same layout as Q/K/V.
    """
    h, NB, d_model = Q.shape
    # We assume NB = N * B
    # You must know B and N from context (e.g., your data loader).
    # For this example, let's pass them in or compute them:
    #print(padding_mask.shape)
    B = 1  # number of batches
    N = padding_mask.size(0)  # max sequence length
    
    #print(padding_mask.shape)
    assert NB == N * B, f"Mismatch: NB={NB} != N*B={N*B}"

    # 1. Reshape from (h, N*B, d) -> (h, B, N, d) -> permute to (B, h, N, d)
    Q_4d = Q.view(h, B, N, d_model)  # (h, B, N, d)
    K_4d = K.view(h, B, N, d_model)
    V_4d = V.view(h, B, N, d_model-6)

    # permute so that batch dimension is first: (B, h, N, d)
    Q_bhnd = Q_4d.permute(1, 0, 2, 3).contiguous()
    K_bhnd = K_4d.permute(1, 0, 2, 3).contiguous()
    V_bhnd = V_4d.permute(1, 0, 2, 3).contiguous()

    # 2. Compute pairwise squared distances => (B, h, N, N)
    dist_sq = (Q_bhnd.unsqueeze(3) - K_bhnd.unsqueeze(2)).pow(2).sum(dim=-1)
    # dist_sq shape: (B, h, N, N)

    # 3. Apply RBF kernel
    kernel = torch.exp(-0.5 * dist_sq)  # (B, h, N, N)

    # 4. Build a (B, N, N) mask from padding_mask  (N). B = 1 in this case
    #    attn_mask[b, u, v] = 1 if both (u,v) are valid for that batch
    attn_mask = padding_mask.unsqueeze(0) # (B, N)
    attn_mask = attn_mask.unsqueeze(2) & attn_mask.unsqueeze(1)  # (B, N, N)
    attn_mask = attn_mask.unsqueeze(1)  # (B, 1, N, N) to broadcast over h
    attn_mask_f = attn_mask.float()

    # 5. Zero out kernel entries where padded
    kernel_masked = kernel * attn_mask_f  # (B, h, N, N)

    # 6. Normalize row-wise (across N dimension for keys)
    eps = 1e-8
    denom = kernel_masked.sum(dim=-1, keepdim=True) + eps  # (B, h, N, 1)
    attn_weights = kernel_masked / denom  # (B, h, N, N)


    # 7. Weighted sum => (B, h, N, d)
    out_bhnd = torch.matmul(attn_weights.float(), V_bhnd.float())

    # 8. Reshape back to (h, N*B, d)
    # permute (B, h, N, d) -> (h, B, N, d) -> flatten B*N
    out_hbnd = out_bhnd.permute(1, 0, 2, 3).contiguous()  # (h, B, N, d)
    out = out_hbnd.view(h, NB, d_model-6)  # (h, N*B, d)

    return out
