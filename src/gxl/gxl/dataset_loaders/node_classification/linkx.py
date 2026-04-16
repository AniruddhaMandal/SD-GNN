"""
LINKX heterophilous benchmark dataset loaders (Lim et al., 2021).
"""
from ...registry import register_dataset
from ... import ExperimentConfig


@register_dataset('ArxivYear')
@register_dataset('SnapPatents')
@register_dataset('Penn94')
@register_dataset('Pokec')
@register_dataset('TwitchGamers')
@register_dataset('Genius')
def build_linkx(cfg: ExperimentConfig):
    """
    Build LINKX heterophilous benchmark datasets (Lim et al., 2021).

    All datasets provide pre-split train/val/test masks (10 splits; first is used).
    - ArxivYear:    169343 nodes, 128 features,  5 classes (year prediction)
    - SnapPatents: 2923922 nodes, 269 features,  5 classes
    - Penn94:        41554 nodes,   5 features,  2 classes (gender)
    - Pokec:       1632803 nodes,  65 features,  2 classes (gender)
    - TwitchGamers: 168114 nodes,   7 features,  2 classes
    - Genius:       421961 nodes,  12 features,  2 classes
    """
    from torch_geometric.datasets import LinkXDataset
    from torch_geometric.loader import DataLoader

    name_map = {
        'ArxivYear':    'arxiv-year',
        'SnapPatents':  'snap-patents',
        'Penn94':       'penn94',
        'Pokec':        'pokec',
        'TwitchGamers': 'twitch-gamers',
        'Genius':       'genius',
    }
    internal_name = name_map[cfg.dataset_name]

    dataset = LinkXDataset(root='./data/LINKX', name=internal_name)
    data = dataset[0]

    # LinkXDataset provides 2D masks (N x num_splits); use the first split
    if hasattr(data, 'train_mask') and data.train_mask.dim() == 2:
        data.train_mask = data.train_mask[:, 0]
        data.val_mask   = data.val_mask[:, 0]
        data.test_mask  = data.test_mask[:, 0]

    train_loader = DataLoader([data], batch_size=1, shuffle=False)
    val_loader   = DataLoader([data], batch_size=1, shuffle=False)
    test_loader  = DataLoader([data], batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
