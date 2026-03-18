import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv, GINConv, GCNConv, SAGEConv, GATv2Conv, SGConv, GCN2Conv, PNAConv,
    global_mean_pool, global_add_pool, global_max_pool
)
from torch_geometric.nn.norm import BatchNorm
from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import register_model


def make_mlp(in_dim, hidden_dim, out_dim, num_layers=2, activate_last=False):
    layers = []
    if num_layers == 1:
        layers += [nn.Linear(in_dim, out_dim)]
        if activate_last: layers += [nn.ReLU()]
    else:
        layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        if activate_last: layers += [nn.ReLU()]
    return nn.Sequential(*layers)


class VanillaGNNClassifier(nn.Module):
    """
    A flexible graph-level classifier:
      - conv_type: 'gine' (uses edge_attr), 'gin', 'gcn', 'sage', 'gatv2', 'sgc', 'gcnii', 'pna', 'jknet'
      - For Peptides-func (multi-label), uses BCE-with-logits style outputs.
    """
    SUPPORTED_CONVS = {'gine', 'gin', 'gcn', 'sage', 'gatv2', 'sgc', 'gcnii', 'pna', 'jknet'}
    def __init__(self,
                 in_channels: int,
                 edge_dim: int,
                 hidden_dim: int,
                 out_dim: int = 10,
                 num_layers: int = 5,
                 mlp_layers: int = 2,
                 dropout: float = 0.1,
                 conv_type: str = 'gine',
                 pooling: str = 'mean',
                 residual: bool = True,
                 # PNA parameters
                 deg: torch.Tensor = None,
                 # GCNII parameters
                 gcnii_alpha: float = 0.1,
                 gcnii_theta: float = 0.5,
                 # JKNet parameters
                 jk_mode: str = 'cat'):
                 
        super().__init__()
        conv_type = conv_type.lower()
        assert conv_type in self.SUPPORTED_CONVS, f"Unknown conv_type '{conv_type}', expected one of {self.SUPPORTED_CONVS}"
        assert pooling in {'mean', 'add', 'max', 'sum', 'off'}
        
        # JKNet uses GCN as the base convolution
        self.conv_type = 'gcn' if conv_type == 'jknet' else conv_type
        self.is_jknet = (conv_type == 'jknet')
        self.jk_mode = jk_mode if self.is_jknet else None
        self.num_layers = num_layers
        self.dropout = dropout
        self.out_dim = out_dim
        self.pooling = pooling
        self.residual = residual
        self.hidden_dim = hidden_dim
        self.gcnii_alpha = gcnii_alpha
        self.gcnii_theta = gcnii_theta

        # PNA default degree histogram if not provided
        if conv_type == 'pna':
            self.deg = deg if deg is not None else torch.ones(128, dtype=torch.long)
        else:
            self.deg = None

        # Project inputs to hidden_dim once so all layers are width-matched.
        self.node_proj = nn.Linear(in_channels, hidden_dim)

        # If using edge features (only GINE uses them directly)
        self.use_edges = (conv_type == 'gine')
        if self.use_edges:
            self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # Build layers
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for layer_idx in range(num_layers):
            self.convs.append(self._make_conv(hidden_dim, hidden_dim, mlp_layers, layer_idx=layer_idx))
            self.bns.append(BatchNorm(hidden_dim))

        # JKNet aggregation
        if self.is_jknet:
            if jk_mode == 'cat':
                self.jk_linear = nn.Linear(hidden_dim * num_layers, hidden_dim)
            elif jk_mode == 'max':
                self.jk_linear = nn.Linear(hidden_dim, hidden_dim)
            elif jk_mode == 'lstm':
                self.jk_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
                self.jk_linear = nn.Linear(2 * hidden_dim, hidden_dim)

        if self.pooling == 'mean':
            self.pooling_fn = global_mean_pool
        elif self.pooling in ['add', 'sum']:
            self.pooling_fn = global_add_pool
        elif self.pooling == 'max':
            self.pooling_fn = global_max_pool
        elif self.pooling == 'off':
            self.pooling_fn = lambda h, b: h
    
    def _make_conv(self, in_dim, out_dim, mlp_layers, layer_idx=0):
        if self.conv_type == 'gine':
            mlp = make_mlp(in_dim, in_dim, out_dim, num_layers=mlp_layers)
            return GINEConv(nn=mlp, train_eps=True)
        if self.conv_type == 'gin':
            mlp = make_mlp(in_dim, in_dim, out_dim, num_layers=mlp_layers)
            return GINConv(nn=mlp, train_eps=True)
        if self.conv_type == 'gcn':
            return GCNConv(in_dim, out_dim, cached=False, normalize=True)
        if self.conv_type == 'sage':
            return SAGEConv(in_dim, out_dim)
        if self.conv_type == 'gatv2':
            return GATv2Conv(in_dim, out_dim, heads=1, concat=True)
        if self.conv_type == 'sgc':
            return SGConv(in_dim, out_dim, K=1)
        if self.conv_type == 'gcnii':
            return GCN2Conv(out_dim, alpha=self.gcnii_alpha, theta=self.gcnii_theta,
                            layer=layer_idx + 1, shared_weights=True, cached=False, normalize=True)
        if self.conv_type == 'pna':
            return PNAConv(in_dim, out_dim,
                           aggregators=['mean', 'min', 'max', 'std'],
                           scalers=['identity', 'amplification', 'attenuation'],
                           deg=self.deg)
        raise ValueError(f"Unknown conv_type: {self.conv_type}")
        raise ValueError("Unknown conv_type")

    # ---------- FORWARD ----------
    def forward(self, batch: SubgraphFeaturesBatch):
        """
        x: [N, in_channels]
        edge_index: [2, E]
        batch: [N]
        edge_attr: [E, edge_dim]  (required if conv_type='gine')
        """
        x = batch.x
        edge_attr = batch.edge_attr
        edge_index = batch.edge_index
        
        h = self.node_proj(x)
        h_0 = h  # Initial features for GCNII
        if self.use_edges:
            if edge_attr is None:
                raise ValueError("edge_attr is required for conv_type='gine'.")
            e = self.edge_proj(edge_attr)
        
        layer_outputs = []
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h_res = h
            if self.conv_type == 'gcnii':
                h = conv(h, h_0, edge_index)
            elif self.use_edges:
                h = conv(h, edge_index, e)
            else:
                h = conv(h, edge_index)
            h = conv(h, edge_index) if not self.use_edges else conv(h, edge_index, e)
            h = bn(h)
            h = F.relu(h)
            if self.residual:
                h = h + h_res
            if i < self.num_layers - 1:  # Skip dropout on last layer
                h = F.dropout(h, p=self.dropout, training=self.training)
            layer_outputs.append(h)

        # JKNet aggregation
        if self.is_jknet:
            if self.jk_mode == 'cat':
                h = torch.cat(layer_outputs, dim=-1)
                h = self.jk_linear(h)
            elif self.jk_mode == 'max':
                h = torch.stack(layer_outputs, dim=0).max(dim=0)[0]
                h = self.jk_linear(h)
            elif self.jk_mode == 'lstm':
                h_stack = torch.stack(layer_outputs, dim=1)  # [N, L, H]
                h_lstm, _ = self.jk_lstm(h_stack)
                h = h_lstm[:, -1, :]
                h = self.jk_linear(h)
        
        g = self.pooling_fn(h, batch.batch)
        return g


@register_model('VANILLA')
def build_vanilla(cfg: ExperimentConfig):
    pooling = 'off' if cfg.task == 'Node-Classification' else cfg.model_config.pooling
    residual = cfg.model_config.kwargs.get('residual', True)
    jk_mode = cfg.model_config.kwargs.get('jk_mode', 'cat')
    gcnii_alpha = cfg.model_config.kwargs.get('gcnii_alpha', 0.1)
    gcnii_theta = cfg.model_config.kwargs.get('gcnii_theta', 0.5)
    return VanillaGNNClassifier(
        in_channels=cfg.model_config.node_feature_dim,
        edge_dim=cfg.model_config.edge_feature_dim,
        hidden_dim=cfg.model_config.hidden_dim,
        out_dim=cfg.model_config.hidden_dim,
        num_layers=cfg.model_config.mpnn_layers,
        dropout=cfg.model_config.dropout,
        conv_type=cfg.model_config.mpnn_type,
        pooling=pooling,
        residual=residual,
        gcnii_alpha=gcnii_alpha,
        gcnii_theta=gcnii_theta,
        jk_mode=jk_mode,
    )