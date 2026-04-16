"""
Amazon co-purchase and Coauthor dataset loaders.
"""
from ...registry import register_dataset
from ... import ExperimentConfig


@register_dataset('AmazonPhoto')
@register_dataset('AmazonComputers')
def build_amazon(cfg: ExperimentConfig):
    """
    Build Amazon co-purchase datasets.

    - AmazonPhoto: 7650 nodes, 119081 edges, 8 classes
    - AmazonComputers: 13752 nodes, 245861 edges, 10 classes
    """
    from torch_geometric.datasets import Amazon
    from torch_geometric.transforms import NormalizeFeatures, RandomNodeSplit
    from torch_geometric.loader import DataLoader

    # Map dataset name to internal name
    name_map = {
        'AmazonPhoto': 'Photo',
        'AmazonComputers': 'Computers'
    }
    internal_name = name_map[cfg.dataset_name]

    # Apply transforms including random split for train/val/test
    transform = NormalizeFeatures()
    dataset = Amazon(root='./data/Amazon', name=internal_name, transform=transform)

    # Get the single graph and add train/val/test masks
    data = dataset[0]

    # Create random node split if masks don't exist
    if not hasattr(data, 'train_mask'):
        split_transform = RandomNodeSplit(
            split='train_rest',
            num_val=int(0.2 * data.num_nodes),
            num_test=int(0.2 * data.num_nodes)
        )
        data = split_transform(data)

    # For node classification, batch size is always 1 (the entire graph)
    train_loader = DataLoader([data], batch_size=1, shuffle=False)
    val_loader = DataLoader([data], batch_size=1, shuffle=False)
    test_loader = DataLoader([data], batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


@register_dataset('CoauthorCS')
@register_dataset('CoauthorPhysics')
def build_coauthor(cfg: ExperimentConfig):
    """
    Build Coauthor datasets.

    - CoauthorCS: 18333 nodes, 81894 edges, 15 classes
    - CoauthorPhysics: 34493 nodes, 247962 edges, 5 classes
    """
    from torch_geometric.datasets import Coauthor
    from torch_geometric.transforms import NormalizeFeatures, RandomNodeSplit
    from torch_geometric.loader import DataLoader

    # Map dataset name to internal name
    name_map = {
        'CoauthorCS': 'CS',
        'CoauthorPhysics': 'Physics'
    }
    internal_name = name_map[cfg.dataset_name]

    transform = NormalizeFeatures()
    dataset = Coauthor(root='./data/Coauthor', name=internal_name, transform=transform)

    # Get the single graph and add train/val/test masks
    data = dataset[0]

    # Create random node split if masks don't exist
    if not hasattr(data, 'train_mask'):
        split_transform = RandomNodeSplit(
            split='train_rest',
            num_val=int(0.2 * data.num_nodes),
            num_test=int(0.2 * data.num_nodes)
        )
        data = split_transform(data)

    # For node classification, batch size is always 1 (the entire graph)
    train_loader = DataLoader([data], batch_size=1, shuffle=False)
    val_loader = DataLoader([data], batch_size=1, shuffle=False)
    test_loader = DataLoader([data], batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
