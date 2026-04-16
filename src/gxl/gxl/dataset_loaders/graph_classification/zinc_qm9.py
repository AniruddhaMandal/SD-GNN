"""
ZINC and QM9 dataset loaders.
"""
from ...registry import register_dataset
from ... import ExperimentConfig
from ...utils.split_and_loader import build_dataloaders_from_dataset


@register_dataset('ZINC')
def build_zinc(cfg: ExperimentConfig):
    from torch_geometric.datasets import ZINC
    # No pre_transform: ZINC already stores edges in both directions.
    # Raw integer features: x in [0,20], edge_attr in [1,3].
    # Atom/bond embeddings are learned inside the model.
    train_dataset = ZINC(root='./data/ZINC', subset=True, split='train')
    test_dataset = ZINC(root='./data/ZINC', subset=True, split='test')
    val_dataset = ZINC(root='./data/ZINC', subset=True, split='val')
    dataset = (train_dataset, test_dataset, val_dataset)
    return build_dataloaders_from_dataset(dataset, cfg)


@register_dataset('QM9')
def build_qm9(cfg: ExperimentConfig):
    from torch_geometric.datasets import QM9
    from sklearn.model_selection import train_test_split
    from ...encoder import FilterTarget
    from ...encoder import NormaliseTarget

    if cfg.task == 'Single-Target-Regression':
        if cfg.train.dataloader_kwargs.get('target_idx', None) is None:
            raise NameError(f'set target_idx in config.')
        target_idx = int(cfg.train.dataloader_kwargs.get('target_idx'))
        transforms = FilterTarget(target_idx)
    dataset = QM9('./data/QM9', transform=transforms)
    if cfg.task == "All-Target-Regression":
        transforms = None

    return build_dataloaders_from_dataset(dataset, cfg)
