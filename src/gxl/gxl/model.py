import torch.nn as nn
from gxl.models.head import ClassifierHead, LinkPredictorHead
from gxl.models.amplified_head import build_amplified_head
from gxl.registry import get_model
from . import ExperimentConfig
from . import SubgraphFeaturesBatch

# Import model modules to trigger their @register_model decorators
import gxl.models.vanilla  # noqa: F401
import gxl.models.ss_gnn   # noqa: F401
import gxl.models.sd_gnn   # noqa: F401
import gxl.models.arch_2    # noqa: F401
import gxl.models.arch_2_v2 # noqa: F401
import gxl.models.arch_3    # noqa: F401
import gxl.models.arch_4    # noqa: F401
import gxl.models.arch_5    # noqa: F401
import gxl.models.arch_6    # noqa: F401
import gxl.models.arch_7    # noqa: F401


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