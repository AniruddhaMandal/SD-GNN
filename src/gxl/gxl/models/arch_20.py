"""
ARCH-20: SubgraphFormer-inspired architecture with graphlet sampler.

SubgraphFormer uses N ego-graphs (one per node), N² virtual nodes, and
precomputed Kronecker-product edge indices.  We reproduce its three key
message-passing patterns per layer on top of our per-node graphlet samples.

SubgraphFormer layer → ARCH-20 equivalent
------------------------------------------
  uL  within-subgraph MP via original edges  →  LocalGINE  (same edge set)
  vL  cross-subgraph MP via original edges   →  GlobalGINE on root reps
  vv  root broadcast within subgraph         →  root rep added to every position

All three run once per layer; outputs summed with residual.
No Transformer — purely message-passing, closest to the SubgraphFormer spirit.

PE
--
SubgraphFormer: LapPE(Kronecker graph) + APSP distance.
ARCH-20:        Global RWSE (computed on full batched graph on-the-fly).
                RWSE captures ring membership just like LapPE, and is fast
                to compute for small molecules.

Pooling (uG → global pool)
--------------------------
SubgraphFormer: scatter_mean within subgraph → global_mean_pool.
ARCH-20:        scatter_mean within subgraph → mean over m → global_add_pool.

Pipeline
--------
  Stage 1  – Encode      atom_encoder, bond_encoder
  Stage 2  – Global RWSE rwse_proj(RWSE) added to every flat node
  Stage 3×L – SubgraphormerLayer:
    uL    LocalGINE(h_flat, intra_ei, intra_ea)     [S*k, H]
    vL    root reps → mean per node →
          GlobalGINE(h_node, global_ei, global_ea)  [N_total, H]
    vv    h_flat += broadcast(h_node_updated)[sub_ids]
          BN + ReLU + residual
  Stage 4  – Readout
    scatter_mean within subgraph (uG)               [S, H]
    view + mean over m                              [N_total, H]
    global_add_pool                                 [B, H]

Requirements
------------
  - Per-node sampling → 'ARCH-20' in _build_all_node_targets in experiment.py.
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
# One SubgraphormerLayer: uL + vL + vv
# ---------------------------------------------------------------------------

class SubgraphormerLayer(nn.Module):
    """
    One layer with three coupled message-passing operations:

      uL  Local GINE within each subgraph  (encodes subgraph topology)
      vL  Global GINE on root reps via original graph edges
              (cross-subgraph communication between neighboring nodes)
      vv  Root broadcast: updated root rep added to every position
              in its subgraph (propagates global context into subgraph)

    All three share the same hidden dimension H.
    Each has its own GINEConv + BatchNorm.
    A final residual wraps the combined update.
    """

    def __init__(self, hidden_dim: int, mlp_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        H = hidden_dim

        # uL: within-subgraph GINE
        self.local_conv = GINEConv(
            make_mlp(H, H, H, mlp_layers), train_eps=True, edge_dim=H)
        self.local_bn   = BatchNorm(H)

        # vL: global GINE on root reps (original graph edges)
        self.global_conv = GINEConv(
            make_mlp(H, H, H, mlp_layers), train_eps=True, edge_dim=H)
        self.global_bn   = BatchNorm(H)

        # vv: broadcast projection (root → subgraph positions)
        self.broadcast_proj = nn.Linear(H, H, bias=False)

        # Combined update norm
        self.update_bn = BatchNorm(H)

        self.dropout = dropout

    def forward(
        self,
        h_flat:         torch.Tensor,   # [S*k, H]
        intra_ei:       torch.Tensor,   # [2, E_sub]
        intra_ea:       torch.Tensor,   # [E_sub, H]
        valid_f:        torch.Tensor,   # [S*k, 1]
        global_ei:      torch.Tensor,   # [2, E_orig]
        global_ea:      torch.Tensor,   # [E_orig, H]
        root_flat_idx:  torch.Tensor,   # [S]      flat indices of roots
        node_assign:    torch.Tensor,   # [S]      global node id per subgraph
        sub_ids:        torch.Tensor,   # [S*k]    subgraph id per flat pos
        N_total:        int,
        S:              int,
    ) -> torch.Tensor:

        h_in = h_flat  # keep for residual

        # ── uL: local GINE ─────────────────────────────────────────────────
        h_local = self.local_bn(F.relu(self.local_conv(h_flat, intra_ei, intra_ea)))
        h_local = F.dropout(h_local, p=self.dropout, training=self.training)

        # ── vL: cross-subgraph GINE on root reps ──────────────────────────
        h_root = h_flat[root_flat_idx]                        # [S, H]
        # Mean-pool roots belonging to the same node  → [N_total, H]
        h_node = scatter(h_root, node_assign, dim=0,
                         reduce='mean', dim_size=N_total)
        h_cross = self.global_bn(F.relu(
            self.global_conv(h_node, global_ei, global_ea)))  # [N_total, H]
        h_cross = F.dropout(h_cross, p=self.dropout, training=self.training)
        h_node  = h_node + h_cross                            # residual at node level

        # ── vv: broadcast root update to every subgraph position ──────────
        # Each subgraph s gets node_assign[s]'s updated rep → [S, H]
        h_broadcast = self.broadcast_proj(h_node[node_assign])  # [S, H]
        # Expand: flat position p belongs to subgraph sub_ids[p]
        h_broadcast_flat = h_broadcast[sub_ids]                 # [S*k, H]

        # ── Combine: local + broadcast, then residual ──────────────────────
        h_new = self.update_bn(h_local + h_broadcast_flat)
        h_flat = (h_in + h_new) * valid_f                       # [S*k, H]

        return h_flat


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch20GraphEncoder(nn.Module):

    def __init__(
        self,
        in_channels:  int,
        hidden_dim:   int,
        edge_dim:     int,
        gnn_layers:   int   = 4,
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
        # ── Stage 1: encode ────────────────────────────────────────────────
        sf.x         = self.atom_encoder(sf.x.long().squeeze(-1))
        sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)

        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        S, k   = sf.nodes_sampled.shape
        m      = S // N_total
        device = x_flat.device

        # ── Stage 2: global RWSE PE ────────────────────────────────────────
        rwse     = _global_rwse(sf.edge_index, sf.ptr, N_total, self.rwse_steps, device)
        rwse_emb = self.rwse_proj(rwse)                          # [N_total, H]

        # Add RWSE to every flat position (lookup by global node id)
        # node_ids[p] = global node index for flat position p
        rwse_flat = rwse_emb[node_ids]                           # [S*k, H]
        valid_f   = valid.float().unsqueeze(-1)
        h_flat    = (x_flat + rwse_flat) * valid_f               # [S*k, H]

        # ── Pre-compute index helpers ──────────────────────────────────────
        root_flat_idx = torch.arange(S, device=device) * k      # [S]
        # For per-node sampling: subgraph s belongs to global node s // m
        node_assign   = torch.arange(N_total, device=device).repeat_interleave(m)  # [S]
        # sub_ids[p] = which subgraph flat position p belongs to
        sub_ids       = torch.arange(S * k, device=device) // k  # [S*k]

        # Original graph edges and attrs (for vL cross-subgraph MP)
        global_ei = sf.edge_index.to(device)                     # [2, E_orig]
        global_ea = sf.edge_attr                                  # [E_orig, H]

        # ── Stage 3: L SubgraphormerLayers ────────────────────────────────
        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid_f,
                global_ei, global_ea,
                root_flat_idx, node_assign, sub_ids,
                N_total, S,
            )

        # ── Stage 4: readout (uG → mean over m → global pool) ─────────────
        # scatter_mean within each subgraph (uG)
        valid_mask = valid                                        # [S*k] bool
        h_sub = scatter(
            h_flat[valid_mask], sub_batch[valid_mask],
            dim=0, reduce='mean', dim_size=S,
        )                                                         # [S, H]

        h_3d      = h_sub.view(N_total, m, self.H)               # [N_total, m, H]
        h_node    = h_3d.mean(dim=1)                              # [N_total, H]
        return global_add_pool(h_node, sf.batch)                  # [B, H]


# ---------------------------------------------------------------------------
# Node-level encoder (for node/link tasks)
# ---------------------------------------------------------------------------

class Arch20NodeEncoder(nn.Module):

    def __init__(
        self,
        in_channels:  int,
        hidden_dim:   int,
        edge_dim:     int,
        gnn_layers:   int   = 4,
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

        rwse     = _global_rwse(sf.edge_index, sf.ptr, N_total, self.rwse_steps, device)
        rwse_emb = self.rwse_proj(rwse)
        rwse_flat = rwse_emb[node_ids]
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
        h_sub = scatter(
            h_flat[valid_mask], sub_batch[valid_mask],
            dim=0, reduce='mean', dim_size=S,
        )
        h_3d   = h_sub.view(N_total, m, self.H)
        return h_3d.mean(dim=1)                                   # [N_total, H]


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
        gnn_layers  = kw.get('gnn_layers',  4),
        mlp_layers  = kw.get('mlp_layers',  2),
        rwse_steps  = kw.get('rwse_steps',  16),
        dropout     = cfg.model_config.dropout,
    )

    return Arch20NodeEncoder(**common) if is_node_level else Arch20GraphEncoder(**common)
