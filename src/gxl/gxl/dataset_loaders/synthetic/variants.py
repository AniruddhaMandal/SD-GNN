"""
Synthetic graph dataset builders: K4, Triangle-Parity, Clique-* variants.
Uses local synthetic data classes instead of the `synthetic_dataset` package.
"""
from ...registry import register_dataset
from ... import ExperimentConfig
from ...utils.split_and_loader import build_dataloaders_from_dataset


@register_dataset('K4')
@register_dataset('Triangle-Parity')
@register_dataset('Clique-Detection')
@register_dataset('Multi-Clique-Detection')
@register_dataset('Clique-Detection-Controlled')
@register_dataset('Sparse-Clique-Detection')
def build_synthetic(cfg: ExperimentConfig):
    from .clique_detection import CliqueDetectionDataset, MultiCliqueDetectionDataset
    from .clique_detection_controlled import DensityControlledCliqueDetectionDataset
    from .sparse_clique_detection import SparseCliqueDetectionDataset
    from .triangles import ParityTriangleGraphDataset
    from .cliques import K4ParityDataset
    import os
    import json
    import hashlib
    import tempfile
    import torch
    from torch_geometric.data import InMemoryDataset, Data
    from torch_geometric.transforms import ToUndirected, Compose
    from ...utils.data_transform import ClipOneHotDegree, ClipDegreeEmbed, SetNodeFeaturesOnes, AddLaplacianPE

    f_type = cfg.model_config.kwargs.get('node_feature_type')
    max_degree = cfg.model_config.kwargs.get('max_degree')
    lap_pe_dim = cfg.model_config.kwargs.get('lap_pe_dim', 8)

    assert f_type is not None, \
        "for data with no feature type requires `node_feature_type` in model keywords."

    if f_type == "all_one":
        node_dim = cfg.model_config.node_feature_dim
        transforms = Compose([SetNodeFeaturesOnes(dim=node_dim, cat=False)])
    elif f_type == "lap_pe":
        transforms = Compose([AddLaplacianPE(k=lap_pe_dim, cat=False)])
    elif f_type == "all_one_with_lap_pe":
        node_dim = cfg.model_config.node_feature_dim
        transforms = Compose([
            SetNodeFeaturesOnes(dim=node_dim, cat=False),
            AddLaplacianPE(k=lap_pe_dim, cat=True)
        ])
    elif f_type == "one_hot_degree":
        assert max_degree is not None, \
            "`max_degree` in model keywords. "
        transforms = Compose([
            ToUndirected(),
            ClipOneHotDegree(max_degree=max_degree, cat=False)
        ])
    elif f_type == "degree_embed":
        assert max_degree is not None, \
            "`max_degree` in model keywords. "
        node_dim = cfg.model_config.node_feature_dim
        transforms = Compose([
            ToUndirected(),
            ClipDegreeEmbed(max_degree=max_degree, embed_dim=node_dim, cat=False)
        ])
    else:
        raise ValueError(f"Unknown `node_feature_type`({f_type})")

    # Build a SyntheticGraphData-compatible factory using local classes
    registry = {
        "Triangle-Parity": ParityTriangleGraphDataset,
        "K4": K4ParityDataset,
        "Clique-Detection": CliqueDetectionDataset,
        "Multi-Clique-Detection": MultiCliqueDetectionDataset,
        "Clique-Detection-Controlled": DensityControlledCliqueDetectionDataset,
        "Sparse-Clique-Detection": SparseCliqueDetectionDataset,
    }

    # ---------- helpers (inlined from SyntheticGraphData factory) ----------
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "SYNTHETIC-DATA")

    def _stable_hash(d):
        s = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

    def _atomic_save(obj, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_", suffix=".pt")
        os.close(fd)
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    def _serialize(params):
        def _clean(v):
            if isinstance(v, (str, bool, int, float, type(None))):
                return v
            if isinstance(v, dict):
                return {k: _clean(v[k]) for k in sorted(v)}
            if isinstance(v, (list, tuple)):
                return [_clean(x) for x in v]
            return str(v)
        return _clean(params)

    def _get(name, cache=True, **kwargs):
        constructor = registry[name]
        spec = {k: v for k, v in kwargs.items()
                if k not in ['transform', 'pre_transform', 'seed', 'store_on_device', 'device']}
        params_clean = _serialize(spec)
        safe_name = name.replace(os.sep, "_")
        short_hash = _stable_hash(params_clean)
        cache_file = os.path.join(cache_dir, f"{safe_name}_{short_hash}.pt")

        if cache and os.path.exists(cache_file):
            loaded = torch.load(cache_file, weights_only=False)
            if isinstance(loaded, InMemoryDataset):
                return loaded
            if isinstance(loaded, list):
                transform = kwargs.get('transform', None)
                # Return list with transform applied lazily via a thin wrapper
                class _Wrap:
                    def __init__(self, lst, tx):
                        self._lst = lst
                        self._tx = tx
                    def __len__(self):
                        return len(self._lst)
                    def __getitem__(self, i):
                        g = self._lst[i]
                        return self._tx(g) if self._tx is not None else g
                    def __add__(self, other):
                        combined = list(self._lst) + list(other._lst if hasattr(other, '_lst') else other)
                        return _Wrap(combined, self._tx)
                return _Wrap(loaded, transform)
            return loaded

        instance = constructor(**kwargs)
        if hasattr(instance, "__len__") and hasattr(instance, "__getitem__"):
            data_list = [instance[i] for i in range(len(instance))]
        elif isinstance(instance, list):
            data_list = instance
        else:
            data_list = list(instance)

        if cache:
            _atomic_save(data_list, cache_file)

        transform = kwargs.get('transform', None)
        class _Wrap:
            def __init__(self, lst, tx):
                self._lst = lst
                self._tx = tx
            def __len__(self):
                return len(self._lst)
            def __getitem__(self, i):
                g = self._lst[i]
                return self._tx(g) if self._tx is not None else g
            def __add__(self, other):
                combined = list(self._lst) + list(other._lst if hasattr(other, '_lst') else other)
                return _Wrap(combined, self._tx)
        return _Wrap(data_list, transform)

    # ---------- per-dataset logic ----------
    if cfg.dataset_name == 'Clique-Detection':
        dataset = _get(cfg.dataset_name,
                       cache=True,
                       num_graphs=2000,
                       k=4,
                       node_range=(20, 40),
                       p_no_clique=0.04,
                       p_with_clique=0.08,
                       transform=transforms)
    elif cfg.dataset_name == 'Clique-Detection-Controlled':
        dataset = _get(cfg.dataset_name,
                       cache=True,
                       num_graphs=2000,
                       k=4,
                       node_range=(20, 30),
                       p_no_clique=0.08,
                       p_with_clique=0.06,
                       transform=transforms)
    elif cfg.dataset_name == 'Multi-Clique-Detection':
        dataset = _get(cfg.dataset_name,
                       cache=True,
                       num_graphs=2000,
                       k=4,
                       node_range=(25, 45),
                       p_base=0.08,
                       transform=transforms)
    elif cfg.dataset_name == 'Sparse-Clique-Detection':
        dataset = _get(cfg.dataset_name,
                       cache=True,
                       num_graphs=2000,
                       k=4,
                       node_range=(30, 50),
                       p_base=0.015,
                       transform=transforms)
    elif cfg.dataset_name == 'Triangle-Parity':
        # Original triangle parity dataset
        data_even = _get(cfg.dataset_name,
                         cache=True,
                         num_graphs=1000,
                         node_range=(20, 40),
                         desired_parity=0,
                         p=0.1,
                         transform=transforms)
        data_odd = _get(cfg.dataset_name,
                        cache=True,
                        num_graphs=1000,
                        node_range=(20, 40),
                        desired_parity=1,
                        p=0.1,
                        transform=transforms)
        dataset = data_even + data_odd
    else:
        # Default for K4 and other datasets
        dataset = _get(cfg.dataset_name,
                       cache=True,
                       num_graphs=2000,
                       node_range=(50, 60),
                       desired_parity=[0, 1],
                       p=0.051,
                       transform=transforms)

    return build_dataloaders_from_dataset(dataset, cfg)
