"""
TU (TUDataset benchmark) graph classification loaders.
"""
from ...registry import register_dataset
from ... import ExperimentConfig
from ...utils.split_and_loader import build_dataloaders_from_dataset


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
    from ...utils.data_transform import ClipOneHotDegree, ClipDegreeEmbed
    transforms = Compose([ToUndirected()])
    dataset = TUDataset(root="data/TUDataset", name=cfg.dataset_name, transform=transforms)
    needs_x = (getattr(dataset[0], 'x', None) is None) or (dataset.num_node_features == 0)
    if needs_x:
        f_type = cfg.model_config.kwargs.get('node_feature_type')
        max_degree = cfg.model_config.kwargs.get('max_degree')
        assert f_type is not None, \
            "Dataset has no node features — set `node_feature_type` in model_config.kwargs."
        assert max_degree is not None, \
            "Dataset has no node features — set `max_degree` in model_config.kwargs."
        if f_type == "one_hot_degree":
            transforms = Compose([ToUndirected(), ClipOneHotDegree(max_degree=max_degree,
                                                                   cat=False)])
        elif f_type == "degree_embed":
            node_dim = cfg.model_config.node_feature_dim
            transforms = Compose([ToUndirected(), ClipDegreeEmbed(max_degree=max_degree,
                                                                  embed_dim=node_dim,
                                                                  cat=False)])
        else:
            raise ValueError(f"Unknown `node_feature_type`({f_type})")
        dataset = TUDataset(root="data/TUDataset", name=cfg.dataset_name, transform=transforms)

    return build_dataloaders_from_dataset(dataset, cfg)
