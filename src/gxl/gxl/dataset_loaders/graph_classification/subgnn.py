"""
SubGNN dataset loaders (Alsentzer et al., NeurIPS 2020).
"""
from ...registry import register_dataset
from ... import ExperimentConfig
from ...utils.split_and_loader import build_dataloaders_from_dataset


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
    from ...subgnn_dataset import SubGNNDataset
    from torch_geometric.transforms import Compose
    from ...utils.data_transform import AddLaplacianPE, SetNodeFeaturesOnes

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
