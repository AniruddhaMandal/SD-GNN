"""
Molecule graph classification dataset loaders (BBBP, Tox21, ogbg-ppa).
"""
from ...registry import register_dataset
from ... import ExperimentConfig
from ...utils.split_and_loader import build_dataloaders_from_dataset


@register_dataset('ogbg-ppa')
def build_ogbg_ppa(cfg: ExperimentConfig):
    """
    Build OGB graph classification dataset: ogbg-ppa.

    - ogbg-ppa: Protein-protein association prediction (37 classes)
    """
    from ogb.graphproppred import PygGraphPropPredDataset

    dataset = PygGraphPropPredDataset(name='ogbg-ppa', root='./data/OGB')

    # OGB provides its own splits via get_idx_split() - handled by build_dataloaders_from_dataset
    return build_dataloaders_from_dataset(dataset, cfg)


@register_dataset('BBBP')
def build_bbbp(cfg: ExperimentConfig):
    """
    Build BBBP (Blood-Brain Barrier Penetration) dataset from MoleculeNet.

    Task: Binary classification
    Size: ~2039 molecules
    Node features: 9 (atom features from RDKit)
    """
    from torch_geometric.datasets import MoleculeNet
    from torch_geometric.transforms import ToUndirected, Compose

    transforms = Compose([ToUndirected()])
    dataset = MoleculeNet(root='data/MoleculeNet', name='BBBP', transform=transforms)

    return build_dataloaders_from_dataset(dataset, cfg)


@register_dataset('Tox21')
def build_tox21(cfg: ExperimentConfig):
    """
    Build Tox21 dataset from MoleculeNet.

    Task: Multi-label binary classification (12 toxicity assays)
    Size: ~7831 molecules
    Node features: 9 (atom features from RDKit)
    Note: Labels contain NaN values for missing assays
    """
    from torch_geometric.datasets import MoleculeNet
    from torch_geometric.transforms import ToUndirected, Compose

    transforms = Compose([ToUndirected()])
    dataset = MoleculeNet(root='data/MoleculeNet', name='Tox21', transform=transforms)

    return build_dataloaders_from_dataset(dataset, cfg)
