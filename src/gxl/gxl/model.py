import torch.nn as nn
from gxl.models.head import ClassifierHead, LinkPredictorHead
from gxl.models.amplified_head import build_amplified_head
from gxl.registry import get_model
from .registry import register_model
from . import ExperimentConfig
from . import SubgraphFeaturesBatch

@register_model('VANILLA')
def VANILLA(cfg: ExperimentConfig):
    from gxl.models.vanilla import VanillaGNNClassifier
    # For node-level tasks, skip global pooling
    pooling = 'off' if cfg.task == 'Node-Classification' else cfg.model_config.pooling
    residual = cfg.model_config.kwargs.get('residual', True)
    jk_mode = cfg.model_config.kwargs.get('jk_mode', 'cat')
    gcnii_alpha = cfg.model_config.kwargs.get('gcnii_alpha', 0.1)
    gcnii_theta = cfg.model_config.kwargs.get('gcnii_theta', 0.5)
    model = VanillaGNNClassifier(in_channels=cfg.model_config.node_feature_dim,
                            edge_dim= cfg.model_config.edge_feature_dim,
                            hidden_dim=cfg.model_config.hidden_dim,
                            out_dim=cfg.model_config.hidden_dim,
                            num_layers=cfg.model_config.mpnn_layers,
                            dropout=cfg.model_config.dropout,
                            conv_type=cfg.model_config.mpnn_type,
                            pooling=pooling,
                            residual=residual,
                            gcnii_alpha=gcnii_alpha,
                            gcnii_theta=gcnii_theta,
                            jk_mode=jk_mode)
    return model

@register_model('SS-GNN')
def SSGNN(cfg: ExperimentConfig):
    from gxl.models.ss_gnn import SSGNNGraphEncoder, SSGNNNodeEncoder

    is_node_level = cfg.task in ('Node-Classification', 'Link-Prediction')

    if is_node_level:
        model = SSGNNNodeEncoder(
            in_channels=cfg.model_config.node_feature_dim,
            edge_dim=cfg.model_config.edge_feature_dim,
            hidden_dim=cfg.model_config.hidden_dim,
            num_layers=cfg.model_config.mpnn_layers,
            dropout=cfg.model_config.dropout,
            conv_type=cfg.model_config.mpnn_type,
            aggregator=cfg.model_config.pooling,
            temperature=cfg.model_config.temperature,
            pooling=cfg.model_config.subgraph_param.pooling)
    else:
        model = SSGNNGraphEncoder(
            in_channels=cfg.model_config.node_feature_dim,
            edge_dim=cfg.model_config.edge_feature_dim,
            hidden_dim=cfg.model_config.hidden_dim,
            num_layers=cfg.model_config.mpnn_layers,
            dropout=cfg.model_config.dropout,
            conv_type=cfg.model_config.mpnn_type,
            aggregator=cfg.model_config.pooling,
            temperature=cfg.model_config.temperature,
            pooling=cfg.model_config.subgraph_param.pooling)

    return model

@register_model('SD-GNN')
def SDGNN(cfg: ExperimentConfig):
    from gxl.models.sd_gnn import SDGNNEncoder
    kw = cfg.model_config.kwargs
    return SDGNNEncoder(
        in_channels   = cfg.model_config.node_feature_dim,
        edge_dim      = cfg.model_config.edge_feature_dim,
        hidden_dim    = cfg.model_config.hidden_dim,
        num_layers    = cfg.model_config.mpnn_layers,
        dropout       = cfg.model_config.dropout,
        conv_type     = cfg.model_config.mpnn_type,
        pooling       = cfg.model_config.pooling,
        sub_pooling   = cfg.model_config.subgraph_param.pooling,
        aggregator    = kw.get('aggregator', 'mean'),
        temperature   = cfg.model_config.temperature or 0.5,
        sub_num_layers= kw.get('sub_num_layers', None),
        sub_conv_type = kw.get('sub_conv_type', None),
    )

class ExperimentModel(nn.Module):
    def __init__(self,
                 cfg: ExperimentConfig):
        super().__init__()
        self.is_link_prediction = (cfg.task == 'Link-Prediction')

        self.encoder = get_model(cfg.model_name)(cfg) if cfg.model_name else None

        # Determine encoder output dimension
        # For SS-GNN-WL, use combined_dim; otherwise use hidden_dim
        if hasattr(self.encoder, 'combined_dim'):
            encoder_out_dim = self.encoder.combined_dim
        else:
            encoder_out_dim = cfg.model_config.hidden_dim

        if self.is_link_prediction:
            self.model_head =  LinkPredictorHead(in_dim=encoder_out_dim,
                                                 mlp_hidden=cfg.model_config.hidden_dim,
                                                 score_fn=cfg.model_config.kwargs['head_score_fn'],
                                                 dropout=cfg.model_config.dropout)
        else:
            # Check if amplified head is requested
            head_type = cfg.model_config.kwargs.get('classifier_head_type', 'standard')
            head_scale = cfg.model_config.kwargs.get('classifier_scale', 10.0)

            if head_type == 'standard':
                self.model_head = ClassifierHead(in_dim=encoder_out_dim,
                                                 hidden_dim=cfg.model_config.hidden_dim,
                                                 num_classes=cfg.model_config.out_dim,
                                                 dropout=cfg.model_config.dropout)
            else:
                # Use amplified head
                self.model_head = build_amplified_head(
                    head_type=head_type,
                    in_dim=encoder_out_dim,
                    num_classes=cfg.model_config.out_dim,
                    hidden_dim=cfg.model_config.hidden_dim,
                    dropout=cfg.model_config.dropout,
                    scale=head_scale
                )
    def forward(self, batch: SubgraphFeaturesBatch):
        if self.is_link_prediction:
            encoding = self.encoder(batch)
            output = self.model_head(encoding, batch.edge_label_index)
        else:
            encoding = self.encoder(batch)
            output = self.model_head(encoding)
            
        return output

def build_model(cfg: ExperimentConfig)-> nn.Module:
    model = ExperimentModel(cfg)
    return model