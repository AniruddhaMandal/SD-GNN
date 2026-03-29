"""
ARCH-7-V5: SUN terms, single local GNN + role-differentiated global GNNs.

Sits between V2 (4 GNNs, slow) and V4 (2 GNNs, quality drop):
  - 1 local GNN  on S*k flat nodes  (bottleneck — keep single)
  - 2 global GNNs on N_total nodes  (cheap — restore role split)

The global GNN runs on collapsed canonical-node representations and is
O(N_total) not O(S*k), so the extra global GNN costs almost nothing.
Restoring root/non-root differentiation in the global branch recovers
most of V2's quality without the full 2× overhead of V2.

Per-layer update:
    h_skip = skip_proj(h)
    h1     = BN(local_conv(h, intra_edges))              # single local
    h2     = BN(global_conv[role](x_sum, orig_edges))[v] # role-split global
    x_vv   = vv_proj(canonical_root[v])
    x_kk   = kk_proj(subgraph_root → all in subgraph)
    out    = ReLU(h_skip + h1 + h2 + x_vv + x_kk)
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import scatter

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import register_model
from gxl.models.arch_2_v2 import make_mlp, _flatten_subgraphs
from gxl.models.arch_7 import _make_conv
from gxl.models.arch_7_v4 import _encode


# ---------------------------------------------------------------------------
# Core layer
# ---------------------------------------------------------------------------

class Arch7V5Layer(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        edge_dim:   int,
        mlp_layers: int   = 2,
        conv_type:  str   = 'gine',
        dropout:    float = 0.0,
    ):
        super().__init__()
        H = hidden_dim
        self.use_edge_attr = (conv_type == 'gine')
        self.dropout = dropout

        # Single local GNN (the expensive one — keep one)
        self.local_conv = _make_conv(conv_type, H, H, edge_dim, mlp_layers)
        self.local_bn   = BatchNorm(H)

        # Role-differentiated global GNNs (cheap — runs on N_total only)
        self.global_conv      = _make_conv(conv_type, H, H, edge_dim, mlp_layers)
        self.global_conv_root = _make_conv(conv_type, H, H, edge_dim, mlp_layers)
        self.global_bn        = BatchNorm(H)
        self.global_bn_root   = BatchNorm(H)

        self.skip_proj = nn.Linear(H, H)
        self.vv_proj   = nn.Linear(H, H)
        self.kk_proj   = nn.Linear(H, H)

    def forward(
        self,
        h_flat:        torch.Tensor,
        intra_ei:      torch.Tensor,
        ea_flat:       torch.Tensor,
        valid:         torch.Tensor,
        node_ids:      torch.Tensor,
        N_total:       int,
        edge_index:    torch.Tensor,
        edge_attr:     torch.Tensor,
        sub_batch:     torch.Tensor,
        S:             int,
        k:             int,
        root_flat_idx: torch.Tensor,
        is_root:       torch.Tensor,
    ) -> torch.Tensor:

        valid_f     = valid.float().unsqueeze(-1)
        is_root_f   = is_root.unsqueeze(-1)
        clamped_ids = node_ids.clamp(min=0)
        valid_mask  = node_ids >= 0

        # ── skip ─────────────────────────────────────────────────────────
        h_skip = self.skip_proj(h_flat) * valid_f

        # ── local GNN (single) ───────────────────────────────────────────
        if self.use_edge_attr and ea_flat is not None:
            h1 = self.local_conv(h_flat, intra_ei, ea_flat)
        else:
            h1 = self.local_conv(h_flat, intra_ei)
        h1 = self.local_bn(h1) * valid_f

        # ── global GNN (role-split, cheap) ───────────────────────────────
        x_sum = scatter(
            h_flat[valid_mask], node_ids[valid_mask],
            dim=0, reduce='mean', dim_size=N_total,
        )
        if self.use_edge_attr and edge_attr is not None:
            h2_nr = self.global_conv(x_sum, edge_index, edge_attr)
            h2_r  = self.global_conv_root(x_sum, edge_index, edge_attr)
        else:
            h2_nr = self.global_conv(x_sum, edge_index)
            h2_r  = self.global_conv_root(x_sum, edge_index)
        h2_nr = self.global_bn(h2_nr)[clamped_ids]      * valid_f
        h2_r  = self.global_bn_root(h2_r)[clamped_ids]  * valid_f
        h2 = torch.where(is_root_f, h2_r, h2_nr)

        # ── x_vv ─────────────────────────────────────────────────────────
        root_ids   = node_ids[root_flat_idx]
        root_valid = root_ids >= 0
        x_vv_canonical = scatter(
            h_flat[root_flat_idx][root_valid], root_ids[root_valid],
            dim=0, reduce='mean', dim_size=N_total,
        )
        x_vv = self.vv_proj(x_vv_canonical[clamped_ids]) * valid_f

        # ── x_kk ─────────────────────────────────────────────────────────
        x_kk = self.kk_proj(h_flat[root_flat_idx[sub_batch]]) * valid_f

        # ── combine ───────────────────────────────────────────────────────
        out = F.relu(h_skip + h1 + h2 + x_vv + x_kk)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out * valid_f


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch7V5GraphEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_dim:  int,
        edge_dim:    int,
        num_layers:  int   = 6,
        mlp_layers:  int   = 2,
        dropout:     float = 0.0,
        conv_type:   str   = 'gine',
    ):
        super().__init__()
        self.atom_encoder = nn.Embedding(in_channels, hidden_dim)
        self.bond_encoder = nn.Embedding(edge_dim,    hidden_dim)
        self.role_encoder = nn.Embedding(2,           hidden_dim)

        self.layers = nn.ModuleList([
            Arch7V5Layer(hidden_dim, hidden_dim, mlp_layers, conv_type, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        h_flat, node_ids, N_total = _encode(
            sf, self.atom_encoder, self.bond_encoder, self.role_encoder, self.layers
        )
        valid_mask = node_ids >= 0
        node_embs = scatter(
            h_flat[valid_mask], node_ids[valid_mask],
            dim=0, reduce='mean', dim_size=N_total,
        )
        return global_add_pool(node_embs, sf.batch)


# ---------------------------------------------------------------------------
# Node-level encoder
# ---------------------------------------------------------------------------

class Arch7V5NodeEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_dim:  int,
        edge_dim:    int,
        num_layers:  int   = 6,
        mlp_layers:  int   = 2,
        dropout:     float = 0.0,
        conv_type:   str   = 'gine',
    ):
        super().__init__()
        self.atom_encoder = nn.Embedding(in_channels, hidden_dim)
        self.bond_encoder = nn.Embedding(edge_dim,    hidden_dim)
        self.role_encoder = nn.Embedding(2,           hidden_dim)

        self.layers = nn.ModuleList([
            Arch7V5Layer(hidden_dim, hidden_dim, mlp_layers, conv_type, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        h_flat, node_ids, N_total = _encode(
            sf, self.atom_encoder, self.bond_encoder, self.role_encoder, self.layers
        )
        valid_mask = node_ids >= 0
        return scatter(
            h_flat[valid_mask], node_ids[valid_mask],
            dim=0, reduce='mean', dim_size=N_total,
        )


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-7-V5')
def build_arch7_v5(cfg: ExperimentConfig):
    kw = cfg.model_config.kwargs
    is_node_level = cfg.task in ('Node-Classification', 'Link-Prediction')

    common = dict(
        in_channels = cfg.model_config.node_feature_dim,
        edge_dim    = cfg.model_config.edge_feature_dim,
        hidden_dim  = cfg.model_config.hidden_dim,
        num_layers  = cfg.model_config.mpnn_layers,
        mlp_layers  = kw.get('mlp_layers', 2),
        dropout     = cfg.model_config.dropout,
        conv_type   = cfg.model_config.mpnn_type,
    )

    return Arch7V5NodeEncoder(**common) if is_node_level else Arch7V5GraphEncoder(**common)
