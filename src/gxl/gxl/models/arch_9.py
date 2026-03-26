"""
ARCH-9: Role-Differentiated Subgraph GNN with Subgraph-Level Attention Readout.

Design rationale:
  Root and non-root nodes play structurally different roles inside a graphlet.
  The root is the canonical "centre of attention" — it receives inter-subgraph
  global context. Non-root nodes receive local neighbourhood context plus an
  explicit signal from their subgraph's root. The readout collapses each
  subgraph to a single vector and then runs self-attention over the m subgraph
  vectors for each canonical node before final pooling.

Initialisation (once, before layer 0):
  h = atom_encoder(x)                    # [S*k, H]  embedded atom type
    + dist_encoder(bfs_dist_from_root)   # [S*k, H]  distance-to-root PE
    + logp_proj(log_probs[sub_batch])    # [S*k, H]  log-sampling-prob PE

Per-layer update (L layers, Arch9Layer):

  Non-root node v (in subgraph s with root r):
    h1     = GINE(h_v, intra_neighbors, ea)          # local neighbourhood
    h_role = W_self * h_v + W_root * h_{r,s}         # self + root influence
    h_v'   = ReLU( BN(h1) + h_role )

  Root node r (in subgraph s):
    h1       = GINE_intra(h_r, intra_neighbors, ea)  # local neighbourhood
    h_r_can  = scatter_mean(h_flat[roots], root_ids) # [N, H] canonical root reps
    h_inter  = GINE_inter(h_r_can, orig_edges, ea)   # inter-root global MP
    h_r'     = ReLU( BN(h1) + BN(h_inter)[r] )

Readout:
  h_sub   = sum_pool(h_flat, per_subgraph)            # [S,   H]
  h_sub_2d= h_sub.view(N_total, m, H)  (sorted by root)
  z       = MHA(h_sub_2d) + h_sub_2d                 # [N_total, m, H]
  node_emb= BN(mean(z, dim=1))                        # [N_total, H]
  graph   = global_add_pool(node_emb, batch)          # [B,     H]

Requirements:
  - Per-node subgraph sampling (_build_all_node_targets), so S = N_total * m.
  - graphlet sampler (sf.log_probs must be populated).
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv, GINConv, global_add_pool
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import scatter

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import register_model
from gxl.models.arch_2_v2 import make_mlp, _flatten_subgraphs
from gxl.models.arch_8_b import _bfs_distances   # reuse BFS helper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gine(hidden_dim: int, edge_dim: int, mlp_layers: int) -> GINEConv:
    return GINEConv(
        make_mlp(hidden_dim, hidden_dim, hidden_dim, mlp_layers),
        train_eps=True,
        edge_dim=edge_dim,
    )


# ---------------------------------------------------------------------------
# Core layer
# ---------------------------------------------------------------------------

class Arch9Layer(nn.Module):
    """
    One ARCH-9 message-passing layer.

    Root nodes receive:
      (1) intra-subgraph GINE  (same local neighbours as non-roots)
      (2) inter-root GINE on the original graph (canonical-node-level)

    Non-root nodes receive:
      (1) intra-subgraph GINE
      (2) self-projection  (W_self · h_v)
      (3) root-influence   (W_root · h_root_of_same_subgraph)

    Both branches share the same intra_conv weights.
    """

    MAX_DIST = 32   # cap for distance embedding index

    def __init__(
        self,
        hidden_dim: int,
        edge_dim:   int,
        mlp_layers: int   = 2,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.dropout = dropout

        # Shared intra-subgraph GINE (used by both root and non-root)
        self.intra_conv = _make_gine(hidden_dim, edge_dim, mlp_layers)
        self.intra_bn   = BatchNorm(hidden_dim)

        # Non-root branch: self + root influence
        self.self_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.root_proj  = nn.Linear(hidden_dim, hidden_dim)

        # Root branch: inter-root GINE on original graph
        self.inter_conv = _make_gine(hidden_dim, edge_dim, mlp_layers)
        self.inter_bn   = BatchNorm(hidden_dim)

    def forward(
        self,
        h_flat:       torch.Tensor,   # [S*k, H]
        intra_ei:     torch.Tensor,   # [2, E_sub]
        ea_flat:      torch.Tensor,   # [E_sub, H]
        valid:        torch.Tensor,   # [S*k] bool
        node_ids:     torch.Tensor,   # [S*k] int, -1 = padding
        N_total:      int,
        edge_index:   torch.Tensor,   # [2, E] original graph
        edge_attr:    torch.Tensor,   # [E, H] original edge attrs
        sub_batch:    torch.Tensor,   # [S*k] subgraph index per position
        S:            int,
        k:            int,
        root_flat_idx: torch.Tensor,  # [S] flat index of each subgraph's root
    ) -> torch.Tensor:                # [S*k, H]

        valid_f     = valid.float().unsqueeze(-1)
        clamped_ids = node_ids.clamp(min=0)

        # ── root position mask ────────────────────────────────────────────────
        is_root = torch.zeros(S * k, dtype=torch.bool, device=h_flat.device)
        is_root[root_flat_idx] = True
        is_root_f = is_root.float().unsqueeze(-1)   # [S*k, 1]

        # ── intra-subgraph GINE (shared, all positions) ───────────────────────
        h1 = self.intra_conv(h_flat, intra_ei, ea_flat)   # [S*k, H]
        h1 = self.intra_bn(h1) * valid_f

        # ── non-root branch: self + root-of-subgraph influence ────────────────
        # For flat position i in subgraph s, the root is at flat position s*k.
        # sub_batch[i] = s  →  root position = sub_batch[i] * k
        h_root_bcast = h_flat[sub_batch * k]              # [S*k, H]
        h_non_root   = self.self_proj(h_flat) + self.root_proj(h_root_bcast)

        # ── root branch: inter-root GINE on original graph ────────────────────
        # Canonical root representation = mean of each canonical node's root positions
        h_roots          = h_flat[root_flat_idx]           # [S, H]
        root_ids         = node_ids[root_flat_idx]         # [S]
        h_root_canonical = scatter(
            h_roots, root_ids,
            dim=0, reduce='mean', dim_size=N_total,
        )   # [N_total, H]

        h_inter      = self.inter_conv(h_root_canonical, edge_index, edge_attr)
        h_inter      = self.inter_bn(h_inter)              # [N_total, H]
        h_inter_bcast = h_inter[clamped_ids] * valid_f    # [S*k, H]

        # ── role-differentiated combine ───────────────────────────────────────
        # root:     ReLU(h1  +  h_inter_bcast)
        # non-root: ReLU(h1  +  h_non_root)
        out = is_root_f * (h1 + h_inter_bcast) + (1.0 - is_root_f) * (h1 + h_non_root)
        out = F.relu(out) * valid_f
        out = F.dropout(out, p=self.dropout, training=self.training) * valid_f
        return out


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch9GraphEncoder(nn.Module):
    """
    Full ARCH-9 graph-level encoder.

    Initialisation → L Arch9Layers → subgraph sum-pool →
    per-root-node self-attention over m subgraphs → mean → add-pool → [B, H]
    """

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

        # Node / edge encoders
        self.atom_encoder = nn.Embedding(in_channels, hidden_dim)
        self.bond_encoder = nn.Embedding(edge_dim,    hidden_dim)

        # Distance-to-root positional encoding (BFS distance within subgraph)
        self.dist_encoder = nn.Embedding(self.MAX_DIST + 1, hidden_dim)

        # Log-sampling-probability encoding: scalar logP → H-dim
        self.logp_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
        )

        # Message-passing layers
        self.layers = nn.ModuleList([
            Arch9Layer(hidden_dim, hidden_dim, mlp_layers, dropout)
            for _ in range(num_layers)
        ])

        # Readout: self-attention over m subgraph representations per root node
        assert hidden_dim % num_heads == 0
        self.readout_mha  = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.readout_norm = BatchNorm(hidden_dim)

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        # ── embed atom / bond types ───────────────────────────────────────────
        sf.x         = self.atom_encoder(sf.x.long().squeeze(-1))          # [N, H]
        sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)  # [E, H]

        # ── flatten into subgraph space ───────────────────────────────────────
        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        S, k = sf.nodes_sampled.shape
        m    = S // N_total
        device = x_flat.device

        # Root positions: flat index s*k for subgraph s
        root_flat_idx = torch.arange(S, device=device) * k   # [S]

        # ── distance-to-root PE ───────────────────────────────────────────────
        dist    = _bfs_distances(intra_ei, S, k).clamp(max=self.MAX_DIST)
        dist_pe = self.dist_encoder(dist)                      # [S*k, H]

        # ── log-sampling-probability PE ───────────────────────────────────────
        if sf.log_probs is not None:
            # sf.log_probs: [S], one logP per subgraph
            logp_per_pos = sf.log_probs[sub_batch].unsqueeze(-1)   # [S*k, 1]
            logp_pe      = self.logp_proj(logp_per_pos)             # [S*k, H]
        else:
            logp_pe = torch.zeros_like(x_flat)

        # ── initialise flat node representations ──────────────────────────────
        valid_f = valid.float().unsqueeze(-1)
        h_flat  = (x_flat + dist_pe + logp_pe) * valid_f      # [S*k, H]

        # ── L message-passing layers ──────────────────────────────────────────
        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid,
                node_ids, N_total, sf.edge_index, sf.edge_attr,
                sub_batch, S, k, root_flat_idx,
            )

        # ── readout ───────────────────────────────────────────────────────────
        # Step 1: sum-pool each subgraph → [S, H]
        valid_mask = node_ids >= 0
        h_sub = scatter(
            h_flat[valid_mask], sub_batch[valid_mask],
            dim=0, reduce='sum', dim_size=S,
        )   # [S, H]

        # Step 2: group m subgraph reps per canonical root node → [N_total, m, H]
        root_ids = node_ids[root_flat_idx]                     # [S] canonical root IDs
        order    = torch.argsort(root_ids, stable=True)
        h_sub_2d = h_sub[order].view(N_total, m, h_sub.shape[-1])   # [N_total, m, H]

        # Step 3: self-attention over m subgraph reps + residual
        h_attn, _ = self.readout_mha(h_sub_2d, h_sub_2d, h_sub_2d)  # [N_total, m, H]
        h_attn    = h_attn + h_sub_2d

        # Step 4: mean over m subgraphs → per-node embedding
        node_emb = h_attn.mean(dim=1)                          # [N_total, H]
        node_emb = self.readout_norm(node_emb)

        # Step 5: sum-pool canonical nodes per graph
        return global_add_pool(node_emb, sf.batch)             # [B, H]


# ---------------------------------------------------------------------------
# Node-level encoder
# ---------------------------------------------------------------------------

class Arch9NodeEncoder(nn.Module):
    """Node-level variant: returns [N_total, H]."""

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

        S, k  = sf.nodes_sampled.shape
        m     = S // N_total
        device = x_flat.device
        root_flat_idx = torch.arange(S, device=device) * k

        dist    = _bfs_distances(intra_ei, S, k).clamp(max=self.MAX_DIST)
        dist_pe = self.dist_encoder(dist)

        if sf.log_probs is not None:
            logp_pe = self.logp_proj(sf.log_probs[sub_batch].unsqueeze(-1))
        else:
            logp_pe = torch.zeros_like(x_flat)

        valid_f = valid.float().unsqueeze(-1)
        h_flat  = (x_flat + dist_pe + logp_pe) * valid_f

        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid,
                node_ids, N_total, sf.edge_index, sf.edge_attr,
                sub_batch, S, k, root_flat_idx,
            )

        valid_mask = node_ids >= 0
        h_sub = scatter(
            h_flat[valid_mask], sub_batch[valid_mask],
            dim=0, reduce='sum', dim_size=S,
        )

        root_ids = node_ids[root_flat_idx]
        order    = torch.argsort(root_ids, stable=True)
        h_sub_2d = h_sub[order].view(N_total, m, h_sub.shape[-1])

        h_attn, _ = self.readout_mha(h_sub_2d, h_sub_2d, h_sub_2d)
        h_attn    = h_attn + h_sub_2d

        node_emb  = self.readout_norm(h_attn.mean(dim=1))
        return node_emb   # [N_total, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-9')
def build_arch9(cfg: ExperimentConfig):
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

    if is_node_level:
        return Arch9NodeEncoder(**common)
    else:
        return Arch9GraphEncoder(**common)
