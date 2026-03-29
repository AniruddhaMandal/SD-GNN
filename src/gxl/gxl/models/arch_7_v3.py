"""
ARCH-7-V3: SUN-complete MP with Horvitz-Thompson Node Pooling.

Identical to ARCH-7-V2 in all message-passing layers.
The only change is the three-stage readout:

  Stage 1 — subgraph pooling
    h_sub[s] = sum_pool(h_flat nodes belonging to subgraph s)   [S, H]

  Stage 2 — HT-weighted node pooling
    Each canonical node v has m subgraphs s_1 … s_m.
    Standard mean weights each equally regardless of how likely it was sampled.
    The Horvitz-Thompson correction up-weights rare subgraphs (low p → large 1/p):

        w_s = softmax_m( -α · logP[s] )     (over the m subgraphs of node v)
        h_v = Σ_s  w_s · h_sub[s]           [N_total, H]

    With α = 1 this is the exact HT correction; α is learnable so the model can
    reduce or increase the correction.  α = 0 recovers uniform mean.

  Stage 3 — graph pooling
    h_graph = global_add_pool(h_v, batch)                       [B, H]

Motivation:
  If rare subgraphs carry more discriminative structural information (e.g. they
  encode unusual local motifs), the standard mean readout under-represents them
  relative to their importance.  HT pooling corrects this bias directly at the
  node representation level, before graph pooling, giving every canonical node a
  richer, unbiased structural summary.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import scatter

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import register_model
from gxl.models.arch_2_v2 import _flatten_subgraphs
from gxl.models.arch_7_v2 import Arch7V2Layer   # identical MP layer


# ---------------------------------------------------------------------------
# Shared HT-weighted pooling helper
# ---------------------------------------------------------------------------

def _ht_node_pool(
    h_sub:   torch.Tensor,   # [S, H]  — one vector per subgraph
    lp:      torch.Tensor,   # [S]     — sanitised log-probs
    m:       int,
    N_total: int,
    alpha:   torch.Tensor,   # scalar learnable
) -> torch.Tensor:           # [N_total, H]
    """
    HT-weighted aggregation of m subgraph vectors per canonical node.

        w = softmax(-α · lp.view(N_total, m), dim=-1)   [N_total, m]
        h = (w.unsqueeze(-1) * h_sub.view(N_total, m, H)).sum(1)
    """
    H = h_sub.shape[-1]
    lp_2d   = lp.view(N_total, m)                          # [N_total, m]
    w       = F.softmax(-alpha * lp_2d, dim=-1)            # [N_total, m]
    h_sub_2d = h_sub.view(N_total, m, H)                   # [N_total, m, H]
    return (w.unsqueeze(-1) * h_sub_2d).sum(dim=1)         # [N_total, H]


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch7V3GraphEncoder(nn.Module):
    """
    Same as Arch7V2GraphEncoder, readout replaced by 3-stage HT pooling.
    """

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
            Arch7V2Layer(hidden_dim, hidden_dim, mlp_layers, conv_type, dropout)
            for _ in range(num_layers)
        ])

        # α = 1: full HT correction from epoch 0; model can learn to adjust
        self.ht_alpha = nn.Parameter(torch.ones(1))

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        if not sf.x.is_floating_point():
            sf.x = self.atom_encoder(sf.x.long().squeeze(-1))
        if sf.edge_attr is not None and not sf.edge_attr.is_floating_point():
            sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)

        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        S, k   = sf.nodes_sampled.shape
        m      = S // N_total
        device = x_flat.device

        # ── root helpers ──────────────────────────────────────────────────
        root_flat_idx = torch.arange(S, device=device) * k    # [S]
        is_root = torch.zeros(S * k, dtype=torch.bool, device=device)
        is_root[root_flat_idx] = True

        # ── initial features ──────────────────────────────────────────────
        role_emb = self.role_encoder(is_root.long())
        valid_f  = valid.float().unsqueeze(-1)
        h_flat   = (x_flat + role_emb) * valid_f

        # ── message passing ───────────────────────────────────────────────
        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid,
                node_ids, N_total, sf.edge_index, sf.edge_attr,
                sub_batch, S, k, root_flat_idx, is_root,
            )

        # ── sanitise log-probs ────────────────────────────────────────────
        if sf.log_probs is not None:
            lp = sf.log_probs.clone()
            lp[~torch.isfinite(lp)] = 0.0
        else:
            lp = torch.zeros(S, device=device)

        # ── Stage 1: subgraph pooling — mean over valid node positions ───
        # Mean (not sum) keeps magnitudes consistent regardless of how many
        # valid nodes a subgraph has, so Stage 2 HT weights are not distorted.
        valid_mask = node_ids >= 0
        h_sub = scatter(
            h_flat[valid_mask], sub_batch[valid_mask],
            dim=0, reduce='mean', dim_size=S,
        )   # [S, H]

        # ── Stage 2: HT-weighted node pooling ────────────────────────────
        # Subgraphs are m-contiguous per canonical node (graphlet_sampler Phase 1b)
        node_embs = _ht_node_pool(h_sub, lp, m, N_total, self.ht_alpha)  # [N_total, H]

        # ── Stage 3: graph pooling ────────────────────────────────────────
        return global_add_pool(node_embs, sf.batch)            # [B, H]


# ---------------------------------------------------------------------------
# Node-level encoder
# ---------------------------------------------------------------------------

class Arch7V3NodeEncoder(nn.Module):
    """Node-level variant: returns [N_total, H] via HT-weighted subgraph pooling."""

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
            Arch7V2Layer(hidden_dim, hidden_dim, mlp_layers, conv_type, dropout)
            for _ in range(num_layers)
        ])

        self.ht_alpha = nn.Parameter(torch.ones(1))

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        if not sf.x.is_floating_point():
            sf.x = self.atom_encoder(sf.x.long().squeeze(-1))
        if sf.edge_attr is not None and not sf.edge_attr.is_floating_point():
            sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)

        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        S, k   = sf.nodes_sampled.shape
        m      = S // N_total
        device = x_flat.device

        root_flat_idx = torch.arange(S, device=device) * k
        is_root = torch.zeros(S * k, dtype=torch.bool, device=device)
        is_root[root_flat_idx] = True

        role_emb = self.role_encoder(is_root.long())
        valid_f  = valid.float().unsqueeze(-1)
        h_flat   = (x_flat + role_emb) * valid_f

        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid,
                node_ids, N_total, sf.edge_index, sf.edge_attr,
                sub_batch, S, k, root_flat_idx, is_root,
            )

        if sf.log_probs is not None:
            lp = sf.log_probs.clone()
            lp[~torch.isfinite(lp)] = 0.0
        else:
            lp = torch.zeros(S, device=device)

        valid_mask = node_ids >= 0
        h_sub = scatter(
            h_flat[valid_mask], sub_batch[valid_mask],
            dim=0, reduce='mean', dim_size=S,
        )   # [S, H]

        return _ht_node_pool(h_sub, lp, m, N_total, self.ht_alpha)  # [N_total, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-7-V3')
def build_arch7_v3(cfg: ExperimentConfig):
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

    if is_node_level:
        return Arch7V3NodeEncoder(**common)
    else:
        return Arch7V3GraphEncoder(**common)
