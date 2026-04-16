"""
Planetoid dataset loaders (Cora, CiteSeer, PubMed).
"""
from ...registry import register_dataset
from ... import ExperimentConfig


@register_dataset('Cora')
@register_dataset('CiteSeer')
@register_dataset('PubMed')
def build_planetoid(cfg: ExperimentConfig):
    """
    Build Planetoid datasets (Cora, CiteSeer, PubMed).

    These are citation network datasets for node classification.
    - Cora: 2708 nodes, 5429 edges, 7 classes, 1433 features
    - CiteSeer: 3327 nodes, 4732 edges, 6 classes, 3703 features
    - PubMed: 19717 nodes, 44338 edges, 3 classes, 500 features

    Note: For node classification, the entire graph is used as a single batch.
    The train/val/test splits are handled via masks in the data object.
    """
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures
    from torch_geometric.loader import DataLoader

    transform = NormalizeFeatures()
    dataset = Planetoid(root='./data/Planetoid', name=cfg.dataset_name, transform=transform)

    # For Planetoid, we return the same dataset for all loaders
    # The train/val/test splits are determined by masks in the data object
    # Each loader returns the full graph; masking is applied during training/eval
    data = dataset[0]

    # Create a wrapper dataset that can be used with DataLoader
    class NodeClassificationDataset:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return 1  # Single graph

        def __getitem__(self, idx):
            return self.data

    node_dataset = NodeClassificationDataset(data)

    # For node classification, batch size is always 1 (the entire graph)
    train_loader = DataLoader([data], batch_size=1, shuffle=False)
    val_loader = DataLoader([data], batch_size=1, shuffle=False)
    test_loader = DataLoader([data], batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
