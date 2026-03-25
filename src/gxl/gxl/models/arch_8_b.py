"""
ARCH-8-B: Intra-Subgraph Transformer with Distance-to-Root Positional Encoding.

Architecture option B from .notes/transformer_arch_options.md.

Key changes over ARCH-7:
  1. Distance-to-root PE: each flat position receives an embedding of its BFS
     distance from the subgraph root (nodes_sampled[s, 0]), computed on-the-fly
     from the intra-subgraph edge index. This is the most critical missing signal.

  2. Intra-subgraph transformer: replaces the local GINEConv with a full
     self-attention layer over all k nodes in each subgraph simultaneously,
     reshaped to [S, k, H]. Padding positions are masked out via
     src_key_padding_mask. Each node attends to every real node in its subgraph
     (O(k²) per subgraph, cheap for k=10).

Per-layer update:
  dist_pe     = dist_encoder(bfs_dist_from_root)            # [S*k, H]
  h_in        = h_flat + dist_pe                            # [S*k, H]
  h1          = BN(SubgraphTransformer(h_in.view(S,k,H),
                                        pad_mask).view(S*k,H))# [S*k, H]
  x_sum       = scatter_mean(h_flat[valid], node_ids[valid]) # [N,   H]
  h2          = BN(GlobalGNN(x_sum, orig_edges))             # [N,   H]
  h_sub       = BN(MLP(scatter_mean(h_flat, sub_ids)))       # [S,   H]
  h_flat      = ReLU(skip(h_flat) + h1
                     + h2[node_ids]
                     + h_sub[sub_ids]) * valid_mask

Readout (same as ARCH-7):
  node_embs = scatter_mean(h_flat[valid], node_ids[valid])   # [N, H]
  graph_emb = global_add_pool(node_embs, batch)              # [B, H]
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import (
    GINEConv, GINConv, GCNConv, SAGEConv,
    global_add_pool,
)
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import scatter

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import register_model
from gxl.models.arch_2_v2 import make_mlp, _flatten_subgraphs


# ---------------------------------------------------------------------------
# Conv factory  (same as ARCH-7)
# ---------------------------------------------------------------------------

def _make_conv(conv_type, in_dim, out_dim, edge_dim, mlp_layers):
    if conv_type == 'gine':
        return GINEConv(
            make_mlp(in_dim, in_dim, out_dim, mlp_layers),
            train_eps=True, edge_dim=edge_dim,
        )
    if conv_type == 'gin':
        return GINConv(make_mlp(in_dim, in_dim, out_dim, mlp_layers), train_eps=True)
    if conv_type == 'gcn':
        return GCNConv(in_dim, out_dim)
    if conv_type == 'sage':
        return SAGEConv(in_dim, out_dim)
    raise ValueError(f"Unknown conv_type: {conv_type}")


# ---------------------------------------------------------------------------
# BFS distance computation
# ---------------------------------------------------------------------------

def _bfs_distances(intra_ei: torch.Tensor, S: int, k: int) -> torch.Tensor:
    """
    Compute BFS distance from each subgraph's root (flat position s*k) to every
    other flat position in the same subgraph, using intra-subgraph edges.

    Algorithm: iterative edge relaxation (Bellman-Ford style for unweighted
    graphs = BFS). Runs at most k-1 rounds; terminates early if converged.

    Returns:
        dist  [S*k]  long tensor, values in [0, k].
               Root positions = 0. Disconnected positions = k (clamped).
    """
    device = intra_ei.device
    n      = S * k
    INF    = k  # anything >= k signals "unreachable"

    dist = torch.full((n,), INF, dtype=torch.long, device=device)
    # Roots are the first node in each subgraph
    root_idx = torch.arange(S, device=device) * k
    dist[root_idx] = 0

    if intra_ei.shape[1] == 0:
        return dist

    # Make undirected by adding reverse edges
    src = torch.cat([intra_ei[0], intra_ei[1]])   # [2*E_sub]
    dst = torch.cat([intra_ei[1], intra_ei[0]])   # [2*E_sub]

    for _ in range(k - 1):
        prop = (dist[src] + 1).clamp(max=INF)     # [2*E_sub] proposed distances
        prev = dist.clone()
        # In-place min-scatter: dist[dst] = min(dist[dst], prop)
        dist.scatter_reduce_(0, dst, prop, reduce='amin', include_self=True)
        if torch.equal(dist, prev):               # early stop if converged
            break

    return dist   # [S*k], values in [0, k]


# ---------------------------------------------------------------------------
# Core layer
# ---------------------------------------------------------------------------

class Arch8BLayer(nn.Module):
    """
    One ARCH-8-B layer.

    Replaces ARCH-7's local GINEConv with an intra-subgraph transformer
    that operates on [S, k, H] with distance-to-root positional encoding.
    The global GNN branch and subgraph broadcast are retained from ARCH-7.

    Four terms:
      h_skip    — learnable skip connection (unchanged)
      h1        — intra-subgraph transformer with distance PE (replaces local GNN)
      h2_bcast  — global GNN on scatter_mean(h_flat), broadcast back
      h_sub     — subgraph-level mean → MLP broadcast (SUN's readout term)
    """

    # Maximum distance embedding index. Distances are clamped to [0, MAX_DIST].
    # k ≤ MAX_DIST is safe for any reasonable subgraph size.
    MAX_DIST = 32

    def __init__(
        self,
        hidden_dim:   int,
        edge_dim:     int,
        mlp_layers:   int   = 2,
        conv_type:    str   = 'gine',
        num_heads:    int   = 4,
        dropout:      float = 0.0,
    ):
        super().__init__()
        self.use_edge_attr = (conv_type == 'gine')
        self.dropout       = dropout

        # Distance-to-root positional encoding
        # Index 0 = root, 1..MAX_DIST-1 = reachable nodes, MAX_DIST = unreachable/padding
        self.dist_encoder = nn.Embedding(self.MAX_DIST + 1, hidden_dim)

        # Intra-subgraph transformer (Pre-LN for stability)
        # norm_first=True: LayerNorm before attention (Pre-LN)
        self.sub_transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.sub_bn = BatchNorm(hidden_dim)

        # Global branch (unchanged from ARCH-7)
        self.global_conv = _make_conv(conv_type, hidden_dim, hidden_dim, edge_dim, mlp_layers)
        self.global_bn   = BatchNorm(hidden_dim)

        # Skip connection
        self.skip_proj   = nn.Linear(hidden_dim, hidden_dim)

        # Subgraph-level broadcast (SUN's readout term, from ARCH-7)
        self.sub_readout_mlp = make_mlp(hidden_dim, hidden_dim, hidden_dim, mlp_layers)
        self.sub_readout_bn  = BatchNorm(hidden_dim)

    def forward(
        self,
        h_flat:     torch.Tensor,   # [S*k, H]
        intra_ei:   torch.Tensor,   # [2, E_sub]
        ea_flat:    torch.Tensor,   # [E_sub, H] or None
        valid:      torch.Tensor,   # [S*k] bool
        node_ids:   torch.Tensor,   # [S*k] int, -1 = padding
        N_total:    int,
        edge_index: torch.Tensor,   # [2, E] original graph
        edge_attr:  torch.Tensor,   # [E, H]
        sub_batch:  torch.Tensor,   # [S*k] int, subgraph index per flat position
        S:          int,
        k:          int,
    ) -> torch.Tensor:              # [S*k, H]

        valid_f     = valid.float().unsqueeze(-1)
        clamped_ids = node_ids.clamp(min=0)
        valid_mask  = node_ids >= 0

        # ── skip ────────────────────────────────────────────────────────────
        h_skip = self.skip_proj(h_flat) * valid_f

        # ── distance-to-root PE + intra-subgraph transformer ────────────────
        # Compute BFS distances from subgraph roots (root = flat pos s*k)
        dist      = _bfs_distances(intra_ei, S, k)              # [S*k] long, in [0, k]
        dist      = dist.clamp(max=self.MAX_DIST)               # cap at embedding size
        dist_pe   = self.dist_encoder(dist)                     # [S*k, H]

        h_in      = h_flat + dist_pe                            # [S*k, H]

        # Reshape to [S, k, H] for the transformer
        h_2d      = h_in.view(S, k, -1)                        # [S, k, H]

        # Mask padding positions (True = ignore in attention)
        pad_mask  = ~valid.view(S, k)                          # [S, k]

        # Intra-subgraph self-attention
        h1_2d     = self.sub_transformer(h_2d, src_key_padding_mask=pad_mask)
        h1        = h1_2d.view(S * k, -1) * valid_f            # [S*k, H]
        h1        = self.sub_bn(h1) * valid_f

        # ── global GNN on plain scatter_mean(h_flat) ─────────────────────────
        x_sum = scatter(
            h_flat[valid_mask], node_ids[valid_mask],
            dim=0, reduce='mean', dim_size=N_total,
        )   # [N_total, H]

        if self.use_edge_attr and edge_attr is not None:
            h2 = self.global_conv(x_sum, edge_index, edge_attr)
        else:
            h2 = self.global_conv(x_sum, edge_index)
        h2       = self.global_bn(h2)
        h2_bcast = h2[clamped_ids] * valid_f                   # [S*k, H]

        # ── subgraph-level broadcast ─────────────────────────────────────────
        h_sub = scatter(
            h_flat[valid_mask], sub_batch[valid_mask],
            dim=0, reduce='mean', dim_size=S,
        )   # [S, H]
        h_sub       = self.sub_readout_bn(self.sub_readout_mlp(h_sub))
        h_sub_bcast = h_sub[sub_batch] * valid_f               # [S*k, H]

        # ── combine ──────────────────────────────────────────────────────────
        out = F.relu(h_skip + h1 + h2_bcast + h_sub_bcast)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = out * valid_f
        return out


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch8BGraphEncoder(nn.Module):
    """
    embed → flatten → L Arch8B layers → scatter_mean per canonical node → sum_pool → [B, H]
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
        num_heads:   int   = 4,
    ):
        super().__init__()
        self.atom_encoder = nn.Embedding(in_channels, hidden_dim)
        self.bond_encoder = nn.Embedding(edge_dim,    hidden_dim)

        self.layers = nn.ModuleList([
            Arch8BLayer(hidden_dim, hidden_dim, mlp_layers, conv_type, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        sf.x         = self.atom_encoder(sf.x.long().squeeze(-1))
        sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)

        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        h_flat = x_flat
        S, k   = sf.nodes_sampled.shape

        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid,
                node_ids, N_total, sf.edge_index, sf.edge_attr,
                sub_batch, S, k,
            )

        valid_mask = node_ids >= 0
        node_embs  = scatter(
            h_flat[valid_mask], node_ids[valid_mask],
            dim=0, reduce='mean', dim_size=N_total,
        )   # [N_total, H]

        return global_add_pool(node_embs, sf.batch)   # [B, H]


# ---------------------------------------------------------------------------
# Node-level encoder
# ---------------------------------------------------------------------------

class Arch8BNodeEncoder(nn.Module):
    """Node-level variant: returns per-canonical-node embeddings [N_total, H]."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim:  int,
        edge_dim:    int,
        num_layers:  int   = 6,
        mlp_layers:  int   = 2,
        dropout:     float = 0.0,
        conv_type:   str   = 'gine',
        num_heads:   int   = 4,
    ):
        super().__init__()
        self.atom_encoder = nn.Embedding(in_channels, hidden_dim)
        self.bond_encoder = nn.Embedding(edge_dim,    hidden_dim)

        self.layers = nn.ModuleList([
            Arch8BLayer(hidden_dim, hidden_dim, mlp_layers, conv_type, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        sf.x         = self.atom_encoder(sf.x.long().squeeze(-1))
        sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)

        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        h_flat = x_flat
        S, k   = sf.nodes_sampled.shape

        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid,
                node_ids, N_total, sf.edge_index, sf.edge_attr,
                sub_batch, S, k,
            )

        valid_mask = node_ids >= 0
        return scatter(
            h_flat[valid_mask], node_ids[valid_mask],
            dim=0, reduce='mean', dim_size=N_total,
        )   # [N_total, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-8-B')
def build_arch8b(cfg: ExperimentConfig):
    kw            = cfg.model_config.kwargs
    is_node_level = cfg.task in ('Node-Classification', 'Link-Prediction')

    common = dict(
        in_channels = cfg.model_config.node_feature_dim,
        edge_dim    = cfg.model_config.edge_feature_dim,
        hidden_dim  = cfg.model_config.hidden_dim,
        num_layers  = cfg.model_config.mpnn_layers,
        mlp_layers  = kw.get('mlp_layers', 2),
        dropout     = cfg.model_config.dropout,
        conv_type   = cfg.model_config.mpnn_type,
        num_heads   = kw.get('num_heads', 4),
    )

    if is_node_level:
        return Arch8BNodeEncoder(**common)
    else:
        return Arch8BGraphEncoder(**common)
