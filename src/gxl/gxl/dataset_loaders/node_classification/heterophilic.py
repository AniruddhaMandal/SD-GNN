"""
Heterophilic graph dataset loaders.
"""
from ...registry import register_dataset
from ... import ExperimentConfig


@register_dataset('Chameleon')
@register_dataset('Squirrel')
@register_dataset('Actor')
@register_dataset('Cornell')
@register_dataset('Texas')
@register_dataset('Wisconsin')
def build_heterophilic(cfg: ExperimentConfig):
    """
    Build heterophilic graph datasets.

    These datasets have low homophily, where connected nodes often have different labels.
    - Chameleon: 2277 nodes, 36101 edges, 5 classes
    - Squirrel: 5201 nodes, 217073 edges, 5 classes
    - Actor: 7600 nodes, 33544 edges, 5 classes
    - Cornell, Texas, Wisconsin: ~180 nodes each, 5 classes (WebKB)
    """
    from torch_geometric.datasets import WikipediaNetwork, Actor as ActorDataset, WebKB
    from torch_geometric.transforms import NormalizeFeatures, RandomNodeSplit
    from torch_geometric.loader import DataLoader

    transform = NormalizeFeatures()

    if cfg.dataset_name in ['Chameleon', 'Squirrel']:
        dataset = WikipediaNetwork(root='./data/WikipediaNetwork', name=cfg.dataset_name.lower(), transform=transform)
    elif cfg.dataset_name == 'Actor':
        dataset = ActorDataset(root='./data/Actor', transform=transform)
    elif cfg.dataset_name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root='./data/WebKB', name=cfg.dataset_name, transform=transform)
    else:
        raise ValueError(f"Unknown heterophilic dataset: {cfg.dataset_name}")

    # Get the single graph and add train/val/test masks
    data = dataset[0]

    # Handle multi-split masks (e.g., WikipediaNetwork has 10 splits for CV)
    # Select the first split (index 0) if masks are 2D
    if hasattr(data, 'train_mask') and data.train_mask.dim() == 2:
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]

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


@register_dataset('RomanEmpire')
@register_dataset('AmazonRatings')
@register_dataset('Minesweeper')
@register_dataset('Tolokers')
@register_dataset('Questions')
def build_heterophilic_platonov(cfg: ExperimentConfig):
    """
    Build heterophilic datasets from Platonov et al. (2023).

    These are more challenging heterophilic benchmarks.
    - RomanEmpire: Wikipedia article network
    - AmazonRatings: Amazon product co-review network
    - Minesweeper: Synthetic grid-based dataset
    - Tolokers: Crowdsourcing worker network
    - Questions: Question-answering user network
    """
    from torch_geometric.datasets import HeterophilousGraphDataset
    from torch_geometric.transforms import NormalizeFeatures
    from torch_geometric.loader import DataLoader

    # Map dataset names
    name_map = {
        'RomanEmpire': 'Roman-empire',
        'AmazonRatings': 'Amazon-ratings',
        'Minesweeper': 'Minesweeper',
        'Tolokers': 'Tolokers',
        'Questions': 'Questions'
    }
    internal_name = name_map[cfg.dataset_name]

    transform = NormalizeFeatures()
    dataset = HeterophilousGraphDataset(root='./data/HeterophilousGraph', name=internal_name, transform=transform)

    # Get the single graph - these datasets typically have train/val/test masks
    data = dataset[0]

    # HeterophilousGraphDataset provides 2D masks [N, 10] for 10 pre-computed splits;
    # select the first split (index 0) to get 1D boolean masks.
    if hasattr(data, 'train_mask') and data.train_mask.dim() == 2:
        data.train_mask = data.train_mask[:, 0]
        data.val_mask   = data.val_mask[:, 0]
        data.test_mask  = data.test_mask[:, 0]

    # For node classification, batch size is always 1 (the entire graph)
    train_loader = DataLoader([data], batch_size=1, shuffle=False)
    val_loader = DataLoader([data], batch_size=1, shuffle=False)
    test_loader = DataLoader([data], batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
