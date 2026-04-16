"""
OGB node classification dataset loaders (ogbn-arxiv, ogbn-proteins).
"""
from ...registry import register_dataset
from ... import ExperimentConfig


@register_dataset('ogbn-arxiv')
def build_ogbn_arxiv(cfg: ExperimentConfig):
    """
    Build ogbn-arxiv dataset from OGB.

    Large-scale citation network with OGB train/val/test node splits.
    - 169343 nodes, 1166243 edges, 40 classes
    - Node features: 128-dimensional
    """
    import torch
    from ogb.nodeproppred import PygNodePropPredDataset
    from torch_geometric.loader import DataLoader

    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data/OGB')
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    # Build boolean masks from OGB node-index splits
    num_nodes = data.num_nodes
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[split_idx['train']] = True
    data.val_mask[split_idx['valid']]   = True
    data.test_mask[split_idx['test']]   = True

    # Flatten labels from [N, 1] to [N]
    if data.y.dim() == 2 and data.y.shape[1] == 1:
        data.y = data.y.squeeze(1)

    train_loader = DataLoader([data], batch_size=1, shuffle=False)
    val_loader   = DataLoader([data], batch_size=1, shuffle=False)
    test_loader  = DataLoader([data], batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


@register_dataset('ogbn-proteins')
def build_ogbn_proteins(cfg: ExperimentConfig):
    """
    Build ogbn-proteins dataset from OGB.

    Protein-protein association network.
    - 132534 nodes, 39561252 edges, 112 classes (multi-label)
    - No node features (use node degree or random init)Protein-protein association network with OGB train/val/test node splits.
    - 132534 nodes, 39561252 edges, 112 binary labels (multi-label)
    - No raw node features; edge_attr (8-dim) is mean-aggregated to obtain node features.
    - Task: Node-Multilabel-Classification  Metric: ROCAUC-multilabel
    """
    import torch
    from ogb.nodeproppred import PygNodePropPredDataset
    from torch_geometric.loader import DataLoader

    dataset = PygNodePropPredDataset(name='ogbn-proteins', root='./data/OGB')

    data = dataset[0]
    split_idx = dataset.get_idx_split()

    # Aggregate edge_attr (8-dim) to node features via mean pooling
    if data.x is None or data.x.numel() == 0:
        edge_feat_dim = data.edge_attr.shape[1]
        try:
            from torch_geometric.utils import scatter
            data.x = scatter(data.edge_attr, data.edge_index[0], dim=0,
                             dim_size=data.num_nodes, reduce='mean')
        except Exception:
            import torch
            data.x = torch.zeros((data.num_nodes, edge_feat_dim))

    # Build boolean masks from OGB node-index splits
    num_nodes = data.num_nodes
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[split_idx['train']] = True
    data.val_mask[split_idx['valid']]   = True
    data.test_mask[split_idx['test']]   = True

    # Labels are [N, 112] float (0/1 multi-label)
    data.y = data.y.float()

    train_loader = DataLoader([data], batch_size=1, shuffle=False)
    val_loader   = DataLoader([data], batch_size=1, shuffle=False)
    test_loader  = DataLoader([data], batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
