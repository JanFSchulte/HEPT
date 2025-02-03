from torch_geometric.loader import DataLoader
from datasets import Tracking, TrackingTransform, Pileup, PileupTransform

import torch
#import torch.utils.data.DataLoader as DataLoaderTorch

from torch_geometric.data import Data, Batch

def pad_collated_batch(batch, max_nodes):
    """Pads node features in a batched PyG Data object and returns a padding mask."""
    x_padded_list = []
    coords_padded_list = []
    padding_mask_list = []
    num_nodes_orig = []
    batch_indices = batch.batch  # Batch indices for nodes
    num_graphs = batch_indices.max().item() + 1  # Get number of graphs

    for graph_idx in range(num_graphs):
        mask = batch_indices == graph_idx  # Select nodes for this graph
        x = batch.x[mask]  # Extract node features
        coords = batch.coords[mask]  # Extract node features
        # Get number of nodes and create mask
        num_nodes = x.shape[0]
        num_nodes_orig.append(torch.tensor([num_nodes]))
        padding_mask = torch.ones(max_nodes, dtype=torch.bool)  # Start with all "real" nodes
        if num_nodes < max_nodes:
            pad_size = max_nodes - num_nodes
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))  # Pad nodes
            coords = torch.nn.functional.pad(coords, (0, 0, 0, pad_size))  # Pad nodes
           
            # Mark padding indices as False
            padding_mask[num_nodes:] = False  

        x_padded_list.append(x)
        coords_padded_list.append(coords)
        padding_mask_list.append(padding_mask)

    # Stack padded tensors
    batch.x = torch.cat(x_padded_list, dim=0)
    batch.coords = torch.cat(coords_padded_list, dim=0)
    batch.num_nodes_orig = torch.cat(num_nodes_orig, dim=0)
    batch.padding_mask = torch.cat(padding_mask_list, dim=0)  # Store padding mask in batch

    return batch

def custom_collate_fn(batch):
    """Custom collate function to pad node features in each batch."""
    batch = Batch.from_data_list(batch)  # Create a batched Data object

    # Determine max nodes in the batch
    batch_indices = batch.batch
    num_graphs = batch_indices.max().item() + 1
    #max_nodes = max((batch_indices == i).sum().item() for i in range(num_graphs))
    max_nodes = 1500
    # Apply padding
    return pad_collated_batch(batch, max_nodes)

def get_data_loader(dataset, idx_split, batch_size, pad=False):

    if pad:
        train_loader = torch.utils.data.DataLoader(
            dataset[idx_split["train"]],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset[idx_split["valid"]],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn
        )
        test_loader = torch.utils.data.DataLoader(
            dataset[idx_split["test"]],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn
        )
    else:
        train_loader = DataLoader(
            dataset[idx_split["train"]],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn
        )
        valid_loader = DataLoader(
            dataset[idx_split["valid"]],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn
        )
        test_loader = DataLoader(
            dataset[idx_split["test"]],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn
        )


    return {"train": train_loader, "valid": valid_loader, "test": test_loader}


def get_dataset(dataset_name, data_dir):
    if "tracking" in dataset_name:
        dataset = Tracking(data_dir, transform=TrackingTransform(), dataset_name=dataset_name, pad=True)
    elif dataset_name == "pileup":
        dataset = Pileup(data_dir, transform=PileupTransform())
    else:
        raise NotImplementedError
    dataset.dataset_name = dataset_name
    return dataset




