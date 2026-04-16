"""
CSL (Circular Skip Link) dataset loader.
"""
from ...registry import register_dataset
from ... import ExperimentConfig
from ...utils.split_and_loader import build_dataloaders_from_dataset
from .._base import _ListDataset


def _make_csl_graphs(n: int = 41, skip_values=None, graphs_per_class: int = 15, seed: int = 0):
    """
    Build Circular Skip Link (CSL) graphs.
    Each class r defines C(n, r): a cycle on n nodes plus skip edges i -- (i+r) % n.
    Within each class, graphs_per_class isomorphic copies are produced by
    random node relabellings (different permutations → structurally identical but
    different-looking graphs, challenging for permutation-invariant GNNs).
    """
    import torch
    from torch_geometric.data import Data

    if skip_values is None:
        # Standard CSL-10 benchmark (Murphy et al. 2019)
        skip_values = [2, 3, 4, 5, 6, 9, 11, 12, 13, 16]

    rng = torch.Generator()
    rng.manual_seed(seed)

    data_list = []
    for label, r in enumerate(skip_values):
        # Build canonical edge set (undirected, no duplicates)
        edge_set = set()
        for i in range(n):
            for j in [(i + 1) % n, (i - 1) % n,   # cycle
                      (i + r) % n, (i - r) % n]:    # skip
                if i != j:
                    edge_set.add((min(i, j), max(i, j)))
        src = torch.tensor([u for u, v in edge_set] + [v for u, v in edge_set], dtype=torch.long)
        dst = torch.tensor([v for u, v in edge_set] + [u for u, v in edge_set], dtype=torch.long)
        base_ei = torch.stack([src, dst], dim=0)

        for _ in range(graphs_per_class):
            perm = torch.randperm(n, generator=rng)
            ei   = perm[base_ei]   # remap node indices
            y    = torch.tensor([label], dtype=torch.long)
            data_list.append(Data(edge_index=ei, y=y, num_nodes=n))

    return data_list


@register_dataset('CSL')
def build_csl(cfg: ExperimentConfig):
    """
    CSL (Circular Skip Link) dataset — generated on-the-fly, no external dependency.
    Standard 10-class benchmark: C(41, r) for r in {2,3,4,5,6,9,11,12,13,16},
    15 randomly-relabelled isomorphic copies per class (150 graphs total).
    """
    from torch_geometric.transforms import Compose
    from ...utils.data_transform import SetNodeFeaturesOnes, AddLaplacianPE

    f_type     = cfg.model_config.kwargs.get('node_feature_type')
    lap_pe_dim = cfg.model_config.kwargs.get('lap_pe_dim', 8)

    assert f_type is not None, \
        "CSL requires `node_feature_type` in model_config.kwargs."

    if f_type == "all_one":
        node_dim  = cfg.model_config.node_feature_dim
        transform = Compose([SetNodeFeaturesOnes(dim=node_dim, cat=False)])
    elif f_type == "lap_pe":
        transform = Compose([AddLaplacianPE(k=lap_pe_dim, cat=False)])
    elif f_type == "all_one_with_lap_pe":
        node_dim  = cfg.model_config.node_feature_dim
        transform = Compose([
            SetNodeFeaturesOnes(dim=node_dim, cat=False),
            AddLaplacianPE(k=lap_pe_dim, cat=True),
        ])
    else:
        raise ValueError(f"Unsupported node_feature_type for CSL: {f_type}")

    graphs  = _make_csl_graphs()           # fixed seed → reproducible dataset
    dataset = _ListDataset(graphs, transform=transform)
    return build_dataloaders_from_dataset(dataset, cfg)
