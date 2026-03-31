"""
ARCH-20: SubgraphFormer-inspired architecture with graphlet sampler.

Key design changes vs original version
----------------------------------------
1. SubgraphFormer computes all aggregations in parallel from the SAME h_flat,
   then concatenates → MLP → residual (cat_encoder).  The original ARCH-20
   used sequential addition, which is wrong.

2. uL and vL+vv now both read from h_flat independently; outputs are
   concatenated [S*k, 2H] → Linear(2H, H) → GELU → LayerNorm → + residual.

3. LayerNorm replaces BatchNorm at the combination step for more stable
   training on variable-size molecular graphs.

SubgraphFormer layer → ARCH-20 equivalent
------------------------------------------
  uL  within-subgraph GINEConv          (replicates SubgraphFormer's uL)
  vL  global GINEConv on root reps      (approximates vL/vu cross-subgraph MP)
  vv  root broadcast to subgraph nodes  (replicates vv scatter)
  cat_encoder: cat[uL_out, vv_out] → MLP → LN → residual

PE
--
  Global RWSE on full batched graph (on-the-fly, fast for small molecules).
  Approximates SubgraphFormer's LapPE + APSP.

Requirements
------------
  - Per-node sampling → 'ARCH-20' in _build_all_node_targets.
  - graphlet sampler.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.nn.norm import BatchNorm

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import register_model
from gxl.models.arch_2_v2 import make_mlp, _flatten_subgraphs
from gxl.models.arch_19 import _global_rwse


# ---------------------------------------------------------------------------
# SubgraphormerLayer: parallel uL + (vL→vv), concat → MLP → residual
# ---------------------------------------------------------------------------

class SubgraphormerLayer(nn.Module):
    """
    Parallel computation of:
      uL   Local GINEConv on intra-subgraph edges          → h_local  [S*k, H]
      vL   Global GINEConv on root reps (original edges)   → h_global [N_total, H]
      vv   Broadcast h_global back to each subgraph pos    → h_bcast  [S*k, H]

    Combination (SubgraphFormer cat_encoder style):
      h_cat = cat(h_local, h_bcast)                        [S*k, 2H]
      h_new = GELU(Linear(2H → H))                         [S*k, H]
      h_new = LayerNorm(h_new)                             [S*k, H]
      h_out = (h_in + h_new) * valid_f                     [S*k, H]

    Both uL and vL read from h_flat (same input), not sequentially.
    """

    def __init__(self, hidden_dim: int, mlp_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        H = hidden_dim

        # uL: within-subgraph GINE
        self.local_conv = GINEConv(
            make_mlp(H, H, H, mlp_layers), train_eps=True, edge_dim=H)
        self.local_bn   = BatchNorm(H)

        # vL: global GINE on root reps
        self.global_conv = GINEConv(
            make_mlp(H, H, H, mlp_layers), train_eps=True, edge_dim=H)
        self.global_bn   = BatchNorm(H)

        # vv: projection before broadcast
        self.broadcast_proj = nn.Linear(H, H, bias=False)

        # cat_encoder: [2H → H] + LayerNorm
        self.cat_encoder = nn.Sequential(
            nn.Linear(2 * H, H),
            nn.GELU(),
            nn.Linear(H, H),
        )
        self.norm = nn.LayerNorm(H)

        self.dropout = dropout

    def forward(
        self,
        h_flat:        torch.Tensor,   # [S*k, H]
        intra_ei:      torch.Tensor,   # [2, E_sub]
        intra_ea:      torch.Tensor,   # [E_sub, H]
        valid_f:       torch.Tensor,   # [S*k, 1]
        global_ei:     torch.Tensor,   # [2, E_orig]
        global_ea:     torch.Tensor,   # [E_orig, H]
        root_flat_idx: torch.Tensor,   # [S]
        node_assign:   torch.Tensor,   # [S]    subgraph s → global node id
        sub_ids:       torch.Tensor,   # [S*k]  flat pos → subgraph id
        N_total:       int,
        S:             int,
    ) -> torch.Tensor:

        h_in = h_flat

        # ── uL: local GINE (reads from h_flat) ────────────────────────────
        h_local = self.local_bn(F.relu(self.local_conv(h_flat, intra_ei, intra_ea)))
        h_local = F.dropout(h_local, p=self.dropout, training=self.training)

        # ── vL: global GINE on mean-pooled root reps (reads from h_flat) ──
        h_root  = h_flat[root_flat_idx]                       # [S, H]
        h_node  = scatter(h_root, node_assign, dim=0,
                          reduce='mean', dim_size=N_total)    # [N_total, H]
        h_cross = self.global_bn(F.relu(
            self.global_conv(h_node, global_ei, global_ea)))  # [N_total, H]
        h_cross = F.dropout(h_cross, p=self.dropout, training=self.training)
        h_node  = h_node + h_cross                            # node-level residual

        # ── vv: broadcast updated node rep to each subgraph position ──────
        h_bcast = self.broadcast_proj(h_node[node_assign])   # [S, H]
        h_bcast = h_bcast[sub_ids]                           # [S*k, H]

        # ── cat_encoder: concat → MLP → LN → residual ────────────────────
        h_cat = torch.cat([h_local, h_bcast], dim=-1)        # [S*k, 2H]
        h_new = self.norm(self.cat_encoder(h_cat))            # [S*k, H]
        return (h_in + h_new) * valid_f                       # [S*k, H]


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch20GraphEncoder(nn.Module):

    def __init__(
        self,
        in_channels:  int,
        hidden_dim:   int,
        edge_dim:     int,
        gnn_layers:   int   = 6,
        mlp_layers:   int   = 2,
        rwse_steps:   int   = 16,
        dropout:      float = 0.0,
    ):
        super().__init__()
        H = hidden_dim
        self.H          = H
        self.rwse_steps = rwse_steps

        self.atom_encoder = nn.Embedding(in_channels, H)
        self.bond_encoder = nn.Embedding(edge_dim,    H)
        self.rwse_proj    = nn.Sequential(nn.Linear(rwse_steps, H), nn.ReLU())

        self.layers = nn.ModuleList([
            SubgraphormerLayer(H, mlp_layers, dropout)
            for _ in range(gnn_layers)
        ])

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        sf.x         = self.atom_encoder(sf.x.long().squeeze(-1))
        sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)

        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        S, k   = sf.nodes_sampled.shape
        m      = S // N_total
        device = x_flat.device

        # Global RWSE PE
        rwse      = _global_rwse(sf.edge_index, sf.ptr, N_total, self.rwse_steps, device)
        rwse_flat = self.rwse_proj(rwse)[node_ids]            # [S*k, H]
        valid_f   = valid.float().unsqueeze(-1)
        h_flat    = (x_flat + rwse_flat) * valid_f

        # Pre-compute index helpers
        root_flat_idx = torch.arange(S, device=device) * k
        node_assign   = torch.arange(N_total, device=device).repeat_interleave(m)
        sub_ids       = torch.arange(S * k, device=device) // k
        global_ei     = sf.edge_index.to(device)
        global_ea     = sf.edge_attr

        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid_f,
                global_ei, global_ea,
                root_flat_idx, node_assign, sub_ids,
                N_total, S,
            )

        # Readout: scatter_mean within subgraph → mean over m → global add pool
        valid_mask = valid
        h_sub  = scatter(h_flat[valid_mask], sub_batch[valid_mask],
                         dim=0, reduce='mean', dim_size=S)    # [S, H]
        h_node = h_sub.view(N_total, m, self.H).mean(dim=1)  # [N_total, H]
        return global_add_pool(h_node, sf.batch)              # [B, H]


# ---------------------------------------------------------------------------
# Node-level encoder
# ---------------------------------------------------------------------------

class Arch20NodeEncoder(nn.Module):

    def __init__(
        self,
        in_channels:  int,
        hidden_dim:   int,
        edge_dim:     int,
        gnn_layers:   int   = 6,
        mlp_layers:   int   = 2,
        rwse_steps:   int   = 16,
        dropout:      float = 0.0,
    ):
        super().__init__()
        H = hidden_dim
        self.H          = H
        self.rwse_steps = rwse_steps

        self.atom_encoder = nn.Embedding(in_channels, H)
        self.bond_encoder = nn.Embedding(edge_dim,    H)
        self.rwse_proj    = nn.Sequential(nn.Linear(rwse_steps, H), nn.ReLU())

        self.layers = nn.ModuleList([
            SubgraphormerLayer(H, mlp_layers, dropout)
            for _ in range(gnn_layers)
        ])

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        sf.x         = self.atom_encoder(sf.x.long().squeeze(-1))
        sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)

        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        S, k   = sf.nodes_sampled.shape
        m      = S // N_total
        device = x_flat.device

        rwse      = _global_rwse(sf.edge_index, sf.ptr, N_total, self.rwse_steps, device)
        rwse_flat = self.rwse_proj(rwse)[node_ids]
        valid_f   = valid.float().unsqueeze(-1)
        h_flat    = (x_flat + rwse_flat) * valid_f

        root_flat_idx = torch.arange(S, device=device) * k
        node_assign   = torch.arange(N_total, device=device).repeat_interleave(m)
        sub_ids       = torch.arange(S * k, device=device) // k
        global_ei     = sf.edge_index.to(device)
        global_ea     = sf.edge_attr

        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid_f,
                global_ei, global_ea,
                root_flat_idx, node_assign, sub_ids,
                N_total, S,
            )

        valid_mask = valid
        h_sub  = scatter(h_flat[valid_mask], sub_batch[valid_mask],
                         dim=0, reduce='mean', dim_size=S)
        h_node = h_sub.view(N_total, m, self.H).mean(dim=1)
        return h_node                                         # [N_total, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-20')
def build_arch20(cfg: ExperimentConfig):
    kw            = cfg.model_config.kwargs
    is_node_level = cfg.task in ('Node-Classification', 'Link-Prediction')

    common = dict(
        in_channels = cfg.model_config.node_feature_dim,
        edge_dim    = cfg.model_config.edge_feature_dim,
        hidden_dim  = cfg.model_config.hidden_dim,
        gnn_layers  = kw.get('gnn_layers',  6),
        mlp_layers  = kw.get('mlp_layers',  2),
        rwse_steps  = kw.get('rwse_steps',  16),
        dropout     = cfg.model_config.dropout,
    )

    return Arch20NodeEncoder(**common) if is_node_level else Arch20GraphEncoder(**common)
