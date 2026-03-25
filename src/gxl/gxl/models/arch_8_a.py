"""
ARCH-8-A: View Attention across subgraph appearances of each canonical node.

Architecture option A from .notes/transformer_arch_options.md.

Key change over ARCH-7:
  For each canonical node v, its m root appearances (one per subgraph centred at v)
  attend to each other via multi-head self-attention, replacing the plain
  scatter_mean used in ARCH-7's cross-subgraph branch.

  Requires per-node subgraph sampling (experiment.py activates
  _build_all_node_targets for this model), which guarantees:
      S = N_total * m   (exactly m subgraphs per canonical node)
  so the m root views can be reshaped into a clean [N_total, m, H] tensor.

Per-layer update:
  h_skip        = skip_proj(h_flat)                             # [S*k, H]
  h1            = BN(LocalGNN(h_flat, intra_edges))             # [S*k, H]
  h_attn_node   = ViewMHA(h_roots)                             # [N,   H]  ← NEW
  h2            = BN(GlobalGNN(h_attn_node, orig_edges))        # [N,   H]
  h_sub         = BN(MLP(scatter_mean(h_flat, sub_ids)))        # [S,   H]
  h_flat        = ReLU(h_skip + h1
                       + h_attn_node[node_ids]                  # attended view summary
                       + h2[node_ids]                           # global-GNN context
                       + h_sub[sub_ids]) * valid_mask           # subgraph broadcast

Readout (same as ARCH-7):
  node_embs = scatter_mean(h_flat[valid], node_ids[valid])      # [N, H]
  graph_emb = global_add_pool(node_embs, batch)                 # [B, H]

Root position convention:
  _flatten_subgraphs lays h_flat as [sub0_node0, ..., sub0_node_{k-1},
                                      sub1_node0, ..., sub_{S-1}_node_{k-1}]
  so the root of subgraph s (nodes_sampled[s, 0]) is at flat index s * k.
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
# View attention module
# ---------------------------------------------------------------------------

class ViewAttentionAggregator(nn.Module):
    """
    Multi-head self-attention over the m root views of each canonical node.

    Input:
        h_roots   [S, H]   — h_flat at root positions (flat index s*k for subgraph s)
        root_ids  [S]      — global canonical node ID for each root (0..N_total-1)
        N_total   int      — number of canonical nodes in the batch
        m         int      — subgraphs per canonical node (S = N_total * m)

    Processing:
        1. Sort roots by canonical node id → [N_total, m, H]  (exactly m per node)
        2. Multi-head self-attention across m views with residual
        3. Mean over m views → [N_total, H]

    Output:
        h_node       [N_total, H]   — attended per-node embedding
        restore_order [S]           — inverse permutation (for optional use)
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        self.mha  = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = BatchNorm(hidden_dim)

    def forward(
        self,
        h_roots:  torch.Tensor,   # [S, H]
        root_ids: torch.Tensor,   # [S]   global canonical node IDs
        N_total:  int,
        m:        int,
    ) -> torch.Tensor:            # [N_total, H]

        H = h_roots.shape[-1]

        # Sort so canonical node v's m views are consecutive
        order    = torch.argsort(root_ids, stable=True)   # [S]
        h_sorted = h_roots[order]                          # [S, H]

        # Reshape to [N_total, m, H] — valid because exactly m roots per node
        h_2d = h_sorted.view(N_total, m, H)               # [N_total, m, H]

        # Multi-head self-attention + residual
        h_attn, _ = self.mha(h_2d, h_2d, h_2d)           # [N_total, m, H]
        h_attn    = h_attn + h_2d

        # Mean-pool the m attended views → per-node summary
        h_node = h_attn.mean(dim=1)                        # [N_total, H]
        h_node = self.norm(h_node)
        return h_node


# ---------------------------------------------------------------------------
# Core layer
# ---------------------------------------------------------------------------

class Arch8ALayer(nn.Module):
    """
    One ARCH-8-A layer.

    Five terms combined:
      h_skip        — learnable skip connection
      h1            — local GNN on intra-subgraph edges
      h_attn_bcast  — view-attended per-node summary, broadcast to all appearances
      h2_bcast      — global GNN on h_attn_node, broadcast to flat space
      h_sub_bcast   — subgraph-level mean → MLP broadcast (SUN's readout term)
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim:   int,
        mlp_layers: int   = 2,
        conv_type:  str   = 'gine',
        num_heads:  int   = 4,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.use_edge_attr = (conv_type == 'gine')
        self.dropout       = dropout

        # Local branch
        self.local_conv      = _make_conv(conv_type, hidden_dim, hidden_dim, edge_dim, mlp_layers)
        self.local_bn        = BatchNorm(hidden_dim)

        # View attention (the core addition over ARCH-7)
        self.view_attn       = ViewAttentionAggregator(hidden_dim, num_heads, dropout)

        # Global branch — takes attended node summaries as input
        self.global_conv     = _make_conv(conv_type, hidden_dim, hidden_dim, edge_dim, mlp_layers)
        self.global_bn       = BatchNorm(hidden_dim)

        # Skip connection
        self.skip_proj       = nn.Linear(hidden_dim, hidden_dim)

        # Subgraph-level broadcast (SUN's readout term, carried over from ARCH-7)
        self.sub_readout_mlp = make_mlp(hidden_dim, hidden_dim, hidden_dim, mlp_layers)
        self.sub_readout_bn  = BatchNorm(hidden_dim)

    def forward(
        self,
        h_flat:       torch.Tensor,   # [S*k, H]
        intra_ei:     torch.Tensor,   # [2, E_sub]
        ea_flat:      torch.Tensor,   # [E_sub, H] or None
        valid:        torch.Tensor,   # [S*k] bool
        node_ids:     torch.Tensor,   # [S*k] int, -1 = padding
        N_total:      int,
        edge_index:   torch.Tensor,   # [2, E] original graph
        edge_attr:    torch.Tensor,   # [E, H]
        sub_batch:    torch.Tensor,   # [S*k] int, subgraph index per flat position
        S:            int,
        root_flat_idx: torch.Tensor,  # [S]  flat position of each subgraph's root (s*k)
        m:            int,            # subgraphs per canonical node
    ) -> torch.Tensor:                # [S*k, H]

        valid_f     = valid.float().unsqueeze(-1)
        clamped_ids = node_ids.clamp(min=0)
        valid_mask  = node_ids >= 0

        # ── skip ────────────────────────────────────────────────────────────
        h_skip = self.skip_proj(h_flat) * valid_f

        # ── local GNN ───────────────────────────────────────────────────────
        if self.use_edge_attr and ea_flat is not None:
            h1 = self.local_conv(h_flat, intra_ei, ea_flat)
        else:
            h1 = self.local_conv(h_flat, intra_ei)
        h1 = h1 * valid_f
        h1 = self.local_bn(h1) * valid_f

        # ── view attention ───────────────────────────────────────────────────
        # Root of subgraph s is at flat position s*k (guaranteed by _flatten_subgraphs)
        h_roots  = h_flat[root_flat_idx]               # [S, H]
        root_ids = node_ids[root_flat_idx]             # [S] canonical node per root
        # Attend over m root views per canonical node → [N_total, H]
        h_attn_node  = self.view_attn(h_roots, root_ids, N_total, m)
        # Broadcast attended summary to every flat position of its canonical node
        h_attn_bcast = h_attn_node[clamped_ids] * valid_f   # [S*k, H]

        # ── global GNN on attended node summaries ────────────────────────────
        if self.use_edge_attr and edge_attr is not None:
            h2 = self.global_conv(h_attn_node, edge_index, edge_attr)
        else:
            h2 = self.global_conv(h_attn_node, edge_index)
        h2       = self.global_bn(h2)                        # [N_total, H]
        h2_bcast = h2[clamped_ids] * valid_f                 # [S*k, H]

        # ── subgraph-level broadcast ─────────────────────────────────────────
        h_sub = scatter(
            h_flat[valid_mask], sub_batch[valid_mask],
            dim=0, reduce='mean', dim_size=S,
        )   # [S, H]
        h_sub       = self.sub_readout_bn(self.sub_readout_mlp(h_sub))
        h_sub_bcast = h_sub[sub_batch] * valid_f             # [S*k, H]

        # ── combine ──────────────────────────────────────────────────────────
        out = F.relu(h_skip + h1 + h_attn_bcast + h2_bcast + h_sub_bcast)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = out * valid_f
        return out


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch8AGraphEncoder(nn.Module):
    """
    embed → flatten → L Arch8A layers → scatter_mean per canonical node → sum_pool → [B, H]
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
            Arch8ALayer(hidden_dim, hidden_dim, mlp_layers, conv_type, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        sf.x         = self.atom_encoder(sf.x.long().squeeze(-1))
        sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)

        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        h_flat = x_flat
        S, k   = sf.nodes_sampled.shape          # S = N_total * m, k = subgraph size
        m      = S // N_total                     # subgraphs per canonical node

        # Root of subgraph s sits at flat position s*k
        root_flat_idx = torch.arange(S, device=sf.x.device) * k   # [S]

        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid,
                node_ids, N_total, sf.edge_index, sf.edge_attr,
                sub_batch, S, root_flat_idx, m,
            )

        # Readout: mean over all appearances per canonical node, then sum-pool
        valid_mask = node_ids >= 0
        node_embs  = scatter(
            h_flat[valid_mask], node_ids[valid_mask],
            dim=0, reduce='mean', dim_size=N_total,
        )   # [N_total, H]

        return global_add_pool(node_embs, sf.batch)   # [B, H]


# ---------------------------------------------------------------------------
# Node-level encoder
# ---------------------------------------------------------------------------

class Arch8ANodeEncoder(nn.Module):
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
            Arch8ALayer(hidden_dim, hidden_dim, mlp_layers, conv_type, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        sf.x         = self.atom_encoder(sf.x.long().squeeze(-1))
        sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)

        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        h_flat = x_flat
        S, k   = sf.nodes_sampled.shape
        m      = S // N_total
        root_flat_idx = torch.arange(S, device=sf.x.device) * k

        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid,
                node_ids, N_total, sf.edge_index, sf.edge_attr,
                sub_batch, S, root_flat_idx, m,
            )

        valid_mask = node_ids >= 0
        return scatter(
            h_flat[valid_mask], node_ids[valid_mask],
            dim=0, reduce='mean', dim_size=N_total,
        )   # [N_total, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-8-A')
def build_arch8a(cfg: ExperimentConfig):
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
        return Arch8ANodeEncoder(**common)
    else:
        return Arch8AGraphEncoder(**common)
