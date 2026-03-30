"""
ARCH-15: Root-Rep Readout (ARCH-10 with root-node token instead of sum-pool).

Identical to ARCH-10 in all message-passing layers.  The only change is how
each subgraph is collapsed to a single token before the HT-corrected MHA:

  ARCH-10:  h_sub[s] = sum( h_u  for u in subgraph s )
  ARCH-15:  h_sub[s] = h_flat[ root_flat_idx[s] ]   ← root node only

Motivation
----------
After L GNN layers within a k-node subgraph (k=10, L=6), the root node's
representation has aggregated information from ALL other nodes via multi-hop
message passing (since 6 hops > the diameter of a 10-node subgraph).  The
sum-pool over all k nodes is therefore largely redundant — it blends in k-1
extra representations that are already encoded in the root's rep, introducing
noise rather than signal.

Using just the root rep gives a cleaner token:
  "What does node v look like when situated inside subgraph s?"

This is how SubgraphFormer and DS-framework papers build per-subgraph tokens,
and is the main design choice that separates them from naive pooling baselines.

Everything else (Arch9Layer, BFS dist PE, logP PE, HT-corrected MHA readout,
global_add_pool) is unchanged from ARCH-10.

Requirements:
  - Per-node subgraph sampling (_build_all_node_targets), S = N_total * m.
  - graphlet sampler (sf.log_probs must be populated).
"""

import torch
from torch import nn
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.norm import BatchNorm

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import register_model
from gxl.models.arch_2_v2 import make_mlp, _flatten_subgraphs
from gxl.models.arch_8_b import _bfs_distances
from gxl.models.arch_9 import Arch9Layer
from gxl.models.arch_10 import _ht_attn_bias


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch15GraphEncoder(nn.Module):

    MAX_DIST = 32

    def __init__(
        self,
        in_channels: int,
        hidden_dim:  int,
        edge_dim:    int,
        num_layers:  int   = 6,
        mlp_layers:  int   = 2,
        dropout:     float = 0.0,
        num_heads:   int   = 4,
    ):
        super().__init__()

        self.atom_encoder = nn.Embedding(in_channels, hidden_dim)
        self.bond_encoder = nn.Embedding(edge_dim,    hidden_dim)
        self.dist_encoder = nn.Embedding(self.MAX_DIST + 1, hidden_dim)
        self.logp_proj    = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())

        self.layers = nn.ModuleList([
            Arch9Layer(hidden_dim, hidden_dim, mlp_layers, dropout)
            for _ in range(num_layers)
        ])

        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.ht_alpha  = nn.Parameter(torch.ones(1))
        self.readout_mha  = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.readout_norm = BatchNorm(hidden_dim)

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        sf.x         = self.atom_encoder(sf.x.long().squeeze(-1))
        sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)

        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        S, k   = sf.nodes_sampled.shape
        m      = S // N_total
        device = x_flat.device

        root_flat_idx = torch.arange(S, device=device) * k   # [S]

        if sf.log_probs is not None:
            lp = sf.log_probs.clone()
            lp[~torch.isfinite(lp)] = 0.0
        else:
            lp = torch.zeros(S, device=device)

        # ── initialise flat representations ───────────────────────────────────
        dist    = _bfs_distances(intra_ei, S, k).clamp(max=self.MAX_DIST)
        dist_pe = self.dist_encoder(dist)
        logp_pe = self.logp_proj(lp[sub_batch].unsqueeze(-1))

        valid_f = valid.float().unsqueeze(-1)
        h_flat  = (x_flat + dist_pe + logp_pe) * valid_f

        # ── message-passing ───────────────────────────────────────────────────
        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid,
                node_ids, N_total, sf.edge_index, sf.edge_attr,
                sub_batch, S, k, root_flat_idx,
            )

        # ── readout: root-node token (ARCH-15 change vs ARCH-10) ──────────────
        # After L GNN layers the root already aggregates its full k-hop
        # neighbourhood.  Extract it directly instead of sum-pooling all nodes.
        h_sub    = h_flat[root_flat_idx]                          # [S, H]
        h_sub_2d = h_sub.view(N_total, m, -1)                    # [N_total, m, H]

        # ── HT-corrected self-attention ────────────────────────────────────────
        attn_mask = _ht_attn_bias(lp, m, N_total, self.ht_alpha, self.num_heads)
        h_attn, _ = self.readout_mha(
            h_sub_2d, h_sub_2d, h_sub_2d, attn_mask=attn_mask,
        )
        h_attn = h_attn + h_sub_2d                               # residual

        # ── mean → BN → sum-pool per graph ────────────────────────────────────
        node_emb = self.readout_norm(h_attn.mean(dim=1))         # [N_total, H]
        return global_add_pool(node_emb, sf.batch)                # [B, H]


# ---------------------------------------------------------------------------
# Node-level encoder
# ---------------------------------------------------------------------------

class Arch15NodeEncoder(nn.Module):

    MAX_DIST = 32

    def __init__(
        self,
        in_channels: int,
        hidden_dim:  int,
        edge_dim:    int,
        num_layers:  int   = 6,
        mlp_layers:  int   = 2,
        dropout:     float = 0.0,
        num_heads:   int   = 4,
    ):
        super().__init__()
        self.atom_encoder = nn.Embedding(in_channels, hidden_dim)
        self.bond_encoder = nn.Embedding(edge_dim,    hidden_dim)
        self.dist_encoder = nn.Embedding(self.MAX_DIST + 1, hidden_dim)
        self.logp_proj    = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())

        self.layers = nn.ModuleList([
            Arch9Layer(hidden_dim, hidden_dim, mlp_layers, dropout)
            for _ in range(num_layers)
        ])

        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.ht_alpha  = nn.Parameter(torch.ones(1))
        self.readout_mha  = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.readout_norm = BatchNorm(hidden_dim)

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        sf.x         = self.atom_encoder(sf.x.long().squeeze(-1))
        sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)

        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        S, k   = sf.nodes_sampled.shape
        m      = S // N_total
        device = x_flat.device
        root_flat_idx = torch.arange(S, device=device) * k

        if sf.log_probs is not None:
            lp = sf.log_probs.clone()
            lp[~torch.isfinite(lp)] = 0.0
        else:
            lp = torch.zeros(S, device=device)

        dist    = _bfs_distances(intra_ei, S, k).clamp(max=self.MAX_DIST)
        dist_pe = self.dist_encoder(dist)
        logp_pe = self.logp_proj(lp[sub_batch].unsqueeze(-1))

        valid_f = valid.float().unsqueeze(-1)
        h_flat  = (x_flat + dist_pe + logp_pe) * valid_f

        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid,
                node_ids, N_total, sf.edge_index, sf.edge_attr,
                sub_batch, S, k, root_flat_idx,
            )

        # Root-node token
        h_sub    = h_flat[root_flat_idx]                          # [S, H]
        h_sub_2d = h_sub.view(N_total, m, -1)

        attn_mask = _ht_attn_bias(lp, m, N_total, self.ht_alpha, self.num_heads)
        h_attn, _ = self.readout_mha(
            h_sub_2d, h_sub_2d, h_sub_2d, attn_mask=attn_mask,
        )
        h_attn = h_attn + h_sub_2d

        return self.readout_norm(h_attn.mean(dim=1))              # [N_total, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-15')
def build_arch15(cfg: ExperimentConfig):
    kw            = cfg.model_config.kwargs
    is_node_level = cfg.task in ('Node-Classification', 'Link-Prediction')

    common = dict(
        in_channels = cfg.model_config.node_feature_dim,
        edge_dim    = cfg.model_config.edge_feature_dim,
        hidden_dim  = cfg.model_config.hidden_dim,
        num_layers  = cfg.model_config.mpnn_layers,
        mlp_layers  = kw.get('mlp_layers', 2),
        dropout     = cfg.model_config.dropout,
        num_heads   = kw.get('num_heads', 4),
    )

    return Arch15NodeEncoder(**common) if is_node_level else Arch15GraphEncoder(**common)
