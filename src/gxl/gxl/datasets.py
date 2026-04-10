from .registry import register_dataset
from . import ExperimentConfig
from .utils.split_and_loader import build_dataloaders_from_dataset

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


class _ListDataset:
    """Minimal dataset wrapper around a list of PyG Data objects."""
    def __init__(self, data_list, transform=None):
        self._data      = data_list
        self._transform = transform

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        g = self._data[idx]
        if self._transform is not None:
            g = self._transform(g)
        return g


@register_dataset('CSL')
def build_csl(cfg: ExperimentConfig):
    """
    CSL (Circular Skip Link) dataset — generated on-the-fly, no external dependency.
    Standard 10-class benchmark: C(41, r) for r in {2,3,4,5,6,9,11,12,13,16},
    15 randomly-relabelled isomorphic copies per class (150 graphs total).
    """
    from torch_geometric.transforms import Compose
    from .utils.data_transform import SetNodeFeaturesOnes, AddLaplacianPE

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


@register_dataset('K4')
@register_dataset('Triangle-Parity')
@register_dataset('Clique-Detection')
@register_dataset('Multi-Clique-Detection')
@register_dataset('Clique-Detection-Controlled')
@register_dataset('Sparse-Clique-Detection')
def build_synthetic(cfg: ExperimentConfig):
    from synthetic_dataset import SyntheticGraphData
    from torch_geometric.transforms import ToUndirected, Compose
    from .utils.data_transform import ClipOneHotDegree, ClipDegreeEmbed, SetNodeFeaturesOnes, AddLaplacianPE

    f_type = cfg.model_config.kwargs.get('node_feature_type')
    max_degree = cfg.model_config.kwargs.get('max_degree')
    lap_pe_dim = cfg.model_config.kwargs.get('lap_pe_dim', 8)

    assert  f_type is not None,  \
        "for data with no feature type requires `node_feature_type` in model keywords."

    if f_type == "all_one":
        node_dim = cfg.model_config.node_feature_dim
        transforms = Compose([SetNodeFeaturesOnes(dim=node_dim,cat=False)])
    elif f_type == "lap_pe":
        transforms = Compose([AddLaplacianPE(k=lap_pe_dim, cat=False)])
    elif f_type == "all_one_with_lap_pe":
        node_dim = cfg.model_config.node_feature_dim
        transforms = Compose([
            SetNodeFeaturesOnes(dim=node_dim, cat=False),
            AddLaplacianPE(k=lap_pe_dim, cat=True)
        ])
    elif f_type == "one_hot_degree":
        assert max_degree is not None,  \
            "`max_degree` in model keywords. "
        transforms = Compose([
            ToUndirected(),
            ClipOneHotDegree(max_degree=max_degree,cat=False)
            ])
    elif f_type == "degree_embed":
        assert max_degree is not None,  \
            "`max_degree` in model keywords. "
        node_dim = cfg.model_config.node_feature_dim
        transforms = Compose([
            ToUndirected(),
            ClipDegreeEmbed(max_degree=max_degree,embed_dim=node_dim,cat=False)
            ])
    else:
        raise ValueError(f"Unknown `node_feature_type`({f_type})")
    if cfg.dataset_name == 'Clique-Detection':
        dataset = syn_data.get(cfg.dataset_name,
                               cache=True,
                               num_graphs=2000,
                               k=4,
                               node_range=(20, 40),
                               p_no_clique=0.04,
                               p_with_clique=0.08,
                               transform=transforms)
    elif cfg.dataset_name == 'Clique-Detection-Controlled':
        dataset = syn_data.get(cfg.dataset_name,
                               cache=True,
                               num_graphs=2000,
                               k=4,
                               node_range=(20, 30),
                               p_no_clique=0.08,
                               p_with_clique=0.06,
                               transform=transforms)
    elif cfg.dataset_name == 'Multi-Clique-Detection':
        dataset = syn_data.get(cfg.dataset_name,
                               cache=True,
                               num_graphs=2000,
                               k=4,
                               node_range=(25, 45),
                               p_base=0.08,
                               transform=transforms)
    elif cfg.dataset_name == 'Sparse-Clique-Detection':
        dataset = syn_data.get(cfg.dataset_name,
                               cache=True,
                               num_graphs=2000,
                               k=4,
                               node_range=(30, 50),
                               p_base=0.015,
                               transform=transforms)
    elif cfg.dataset_name == 'Triangle-Parity':
        # Original triangle parity dataset
        data_even = syn_data.get(cfg.dataset_name,
                               cache=True,
                               num_graphs=1000,
                               node_range=(20,40),
                               desired_parity=0,
                               p=0.1,
                               transform=transforms)
        data_odd = syn_data.get(cfg.dataset_name,
                               cache=True,
                               num_graphs=1000,
                               node_range=(20,40),
                               desired_parity=1,
                               p=0.1,
                               transform=transforms)
        dataset = data_even + data_odd
    else:
        # Default for K4 and other datasets
        dataset = syn_data.get(cfg.dataset_name,
                               cache=True,
                               num_graphs=2000,
                               node_range=(50,60),
                               desired_parity=[0,1],
                               p=0.051,
                               transform=transforms)

    return build_dataloaders_from_dataset(dataset, cfg)
    

@register_dataset('PascalVOC-SP')
@register_dataset('COCO-SP')
@register_dataset('PCQM-Contact')
@register_dataset('Peptides-func')
@register_dataset('Peptides-struct')
def build_lrgb(cfg: ExperimentConfig):
    from torch_geometric.datasets import LRGBDataset
    train_data = LRGBDataset("data/LRGB",cfg.dataset_name, "train")
    test_data = LRGBDataset("data/LRGB", cfg.dataset_name, "test")
    val_data = LRGBDataset("data/LRGB", cfg.dataset_name, "val")
    dataset = (train_data, test_data, val_data)
    return build_dataloaders_from_dataset(dataset, cfg)

@register_dataset('MUTAG')
@register_dataset('ENZYMES')
@register_dataset('PROTEINS')
@register_dataset('COLLAB')
@register_dataset('IMDB-BINARY')
@register_dataset('REDDIT-BINARY')
@register_dataset('PTC_MR')
@register_dataset('AIDS')
def build_tudata(cfg: ExperimentConfig):
    from torch_geometric.datasets import TUDataset
    from torch_geometric.transforms import ToUndirected, Compose
    from .utils.data_transform import ClipOneHotDegree, ClipDegreeEmbed
    transforms = Compose([ToUndirected()])
    dataset = TUDataset(root="data/TUDataset",name=cfg.dataset_name,transform=transforms)
    needs_x = (getattr(dataset[0],'x', None) is None) or (dataset.num_node_features == 0)
    if needs_x:
        f_type = cfg.model_config.kwargs.get('node_feature_type')
        max_degree = cfg.model_config.kwargs.get('max_degree')
        assert f_type is not None, \
            "Dataset has no node features — set `node_feature_type` in model_config.kwargs."
        assert max_degree is not None, \
            "Dataset has no node features — set `max_degree` in model_config.kwargs."
        if f_type == "one_hot_degree":
            transforms = Compose([ToUndirected(),ClipOneHotDegree(max_degree=max_degree,
                                                                  cat=False)])
        elif f_type == "degree_embed":
            node_dim = cfg.model_config.node_feature_dim
            transforms = Compose([ToUndirected(),ClipDegreeEmbed(max_degree=max_degree,
                                                                 embed_dim=node_dim,
                                                                 cat=False)])
        else:
            raise ValueError(f"Unknown `node_feature_type`({f_type})")
        dataset = TUDataset(root="data/TUDataset",name=cfg.dataset_name,transform=transforms)

    return build_dataloaders_from_dataset(dataset, cfg)

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
    from .encoder import FilterTarget
    from .encoder import NormaliseTarget


    if cfg.task == 'Single-Target-Regression':
        if cfg.train.dataloader_kwargs.get('target_idx', None) is None:
            raise NameError(f'set target_idx in config.')
        target_idx = int(cfg.train.dataloader_kwargs.get('target_idx'))
        transforms = FilterTarget(target_idx)
    dataset = QM9('./data/QM9',transform=transforms)
    if cfg.task == "All-Target-Regression":
        transforms = None

    return build_dataloaders_from_dataset(dataset, cfg)

@register_dataset('PPI-BP')
@register_dataset('HPO-METAB')
@register_dataset('HPO-NEURO')
@register_dataset('EM-USER')
def build_subgnn(cfg: ExperimentConfig):
    """
    Build SubGNN datasets (Alsentzer et al., NeurIPS 2020)

    Datasets:
    - PPI-BP: Protein-protein interaction - Biological process (6 classes)
    - HPO-METAB: Metabolic disorder classification (6 classes)
    - HPO-NEURO: Neurological disorder classification (10 classes, multi-label)
    - EM-USER: Endomondo user gender classification (2 classes)
    """
    from .subgnn_dataset import SubGNNDataset
    from torch_geometric.transforms import Compose
    from .utils.data_transform import AddLaplacianPE, SetNodeFeaturesOnes

    # Map dataset name to internal format
    dataset_name_map = {
        'PPI-BP': 'ppi_bp',
        'HPO-METAB': 'hpo_metab',
        'HPO-NEURO': 'hpo_neuro',
        'EM-USER': 'em_user'
    }

    internal_name = dataset_name_map[cfg.dataset_name]

    # Optional transforms
    transforms = None
    if hasattr(cfg.model_config, 'kwargs'):
        f_type = cfg.model_config.kwargs.get('node_feature_type', None)
        lap_pe_dim = cfg.model_config.kwargs.get('lap_pe_dim', 8)

        if f_type == "lap_pe":
            transforms = Compose([AddLaplacianPE(k=lap_pe_dim, cat=False)])
        elif f_type == "all_one_with_lap_pe":
            node_dim = cfg.model_config.node_feature_dim
            transforms = Compose([
                SetNodeFeaturesOnes(dim=node_dim, cat=False),
                AddLaplacianPE(k=lap_pe_dim, cat=True)
            ])

    # Load dataset
    dataset = SubGNNDataset(
        root='./data/SubGNN',
        name=internal_name,
        transform=transforms
    )

    # Get splits
    splits = dataset.get_idx_split()
    train_dataset = dataset[splits['train']]
    val_dataset = dataset[splits['valid']]
    test_dataset = dataset[splits['test']]

    # Return as tuple for build_dataloaders_from_dataset
    return build_dataloaders_from_dataset((train_dataset, test_dataset, val_dataset), cfg)

@register_dataset('ogbg-molhiv')
@register_dataset('molhiv')
def build_molhiv(cfg: ExperimentConfig):
    """
    Build OGB-MolHIV dataset.

    Uses RDKit for SMILES-to-graph conversion with OGB-style features.
    Scaffold splitting (80/10/10) is applied.

    Task: Binary classification (HIV activity prediction)
    Metric: ROC-AUC
    """
    from .dataset_loaders.molhiv import MolHIVDataset
    from .encoder import OGBAtomEncoder, OGBBondEncoder
    from torch_geometric.transforms import Compose

    emb_dim = cfg.model_config.node_feature_dim
    edge_emb_dim = getattr(cfg.model_config, 'edge_feature_dim', None) or emb_dim

    transforms = Compose([
        OGBAtomEncoder(emb_dim=emb_dim),
        OGBBondEncoder(emb_dim=edge_emb_dim),
    ])

    dataset = MolHIVDataset(
        root='./data/OGB/molhiv',
        transform=transforms
    )

    # Dataset has get_idx_split() - will be used automatically by build_dataloaders_from_dataset
    return build_dataloaders_from_dataset(dataset, cfg)


# ============ Node Classification Datasets ============

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


# ============ Graph Classification Datasets ============

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
