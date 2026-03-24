"""
ARCH-7: Interleaved Subgraph + Global GNN (SUN-inspired)

Three components, each addressing a specific gap vs SUN:

  1. Per-layer interleaved local + global GNN:
       h1  = local_gnn(intra-subgraph edges)          # [S*k, H]
       x_sum = scatter_mean(h_flat, node_ids)          # [N, H] — per canonical node
       h2  = global_gnn(x_sum, original_edges)         # [N, H]
       h_skip = skip_proj(h_flat)                      # [S*k, H] — SUN's x_kv term
       h_sub = scatter_mean(h_flat, sub_batch)         # [S, H] — SUN's readout term
       h_flat = ReLU(h_skip + h1 + h2[node_ids] + readout_mlp(h_sub)[sub_batch])

  2. All-node subgraph broadcast (h_sub): pools all nodes per subgraph and
     broadcasts back, bypassing edge sparsity in random graphlets. This is
     SUN's "readout" term that lets disconnected nodes in the same graphlet
     share information.

  3. Correct readout:
       node_embs = scatter_mean(h_flat, node_ids)      # [N, H]
       h_graph   = global_add_pool(node_embs, batch)   # [B, H]  — SUM not mean
     Sum pooling over canonical node embeddings matches the standard GNN readout
     (vanilla GIN uses sum pooling; mean-of-mean-of-subgraphs loses additive structure).
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
# Conv factory
# ---------------------------------------------------------------------------

def _make_conv(conv_type, in_dim, out_dim, edge_dim, mlp_layers):
    if conv_type == 'gine':
        return GINEConv(
            make_mlp(in_dim, in_dim, out_dim, mlp_layers),
            train_eps=True, edge_dim=edge_dim,
        )
    if conv_type == 'gin':
        return GINConv(
            make_mlp(in_dim, in_dim, out_dim, mlp_layers),
            train_eps=True,
        )
    if conv_type == 'gcn':
        return GCNConv(in_dim, out_dim)
    if conv_type == 'sage':
        return SAGEConv(in_dim, out_dim)
    raise ValueError(f"Unknown conv_type: {conv_type}")


# ---------------------------------------------------------------------------
# Core layer
# ---------------------------------------------------------------------------

class Arch7Layer(nn.Module):
    """
    One interleaved local + global GNN layer with subgraph broadcast and skip.

    Combines four terms:
      h_skip  — learnable transform of current h_flat (SUN's x_kv: skip connection)
      h1      — local GNN on intra-subgraph edges (captures local graphlet structure)
      h2      — global GNN on x_sum broadcast back to flat space (full-graph context)
      h_sub   — subgraph-level mean broadcast back to all nodes in the same subgraph
                (SUN's readout term; lets disconnected graphlet nodes share info)
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim:   int,
        mlp_layers: int   = 2,
        conv_type:  str   = 'gine',
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.use_edge_attr = (conv_type == 'gine')
        self.dropout = dropout

        # Local branch
        self.local_conv = _make_conv(conv_type, hidden_dim, hidden_dim, edge_dim, mlp_layers)
        self.local_bn   = BatchNorm(hidden_dim)

        # Global branch (runs on x_sum = per-canonical-node mean of h_flat)
        self.global_conv = _make_conv(conv_type, hidden_dim, hidden_dim, edge_dim, mlp_layers)
        self.global_bn   = BatchNorm(hidden_dim)

        # Skip connection (SUN's x_kv — learned transform of current node rep)
        self.skip_proj = nn.Linear(hidden_dim, hidden_dim)

        # Subgraph-level broadcast (SUN's readout — mean of all nodes in subgraph → each node)
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
        S:          int,            # total number of subgraphs in batch
    ) -> torch.Tensor:               # [S*k, H]

        valid_f     = valid.float().unsqueeze(-1)
        clamped_ids = node_ids.clamp(min=0)
        valid_mask  = node_ids >= 0   # alias for valid, but avoids recomputing

        # ── skip (x_kv) ────────────────────────────────────────────────────
        h_skip = self.skip_proj(h_flat) * valid_f          # [S*k, H]

        # ── local GNN ──────────────────────────────────────────────────────
        if self.use_edge_attr and ea_flat is not None:
            h1 = self.local_conv(h_flat, intra_ei, ea_flat)
        else:
            h1 = self.local_conv(h_flat, intra_ei)
        h1 = h1 * valid_f
        h1 = self.local_bn(h1)
        h1 = h1 * valid_f

        # ── x_sum: per-canonical-node mean (only valid positions) ──────────
        x_sum = scatter(
            h_flat[valid_mask], node_ids[valid_mask],
            dim=0, reduce='mean', dim_size=N_total,
        )   # [N_total, H]

        # ── global GNN on x_sum ────────────────────────────────────────────
        if self.use_edge_attr and edge_attr is not None:
            h2 = self.global_conv(x_sum, edge_index, edge_attr)
        else:
            h2 = self.global_conv(x_sum, edge_index)
        h2 = self.global_bn(h2)                            # [N_total, H]
        h2_bcast = h2[clamped_ids] * valid_f               # [S*k, H]

        # ── subgraph-level broadcast (SUN's readout term) ──────────────────
        # Pool all valid nodes per subgraph, apply MLP, broadcast back
        h_sub = scatter(
            h_flat[valid_mask], sub_batch[valid_mask],
            dim=0, reduce='mean', dim_size=S,
        )   # [S, H]
        h_sub = self.sub_readout_bn(self.sub_readout_mlp(h_sub))  # [S, H]
        h_sub_bcast = h_sub[sub_batch] * valid_f           # [S*k, H]

        # ── combine all four terms ──────────────────────────────────────────
        out = F.relu(h_skip + h1 + h2_bcast + h_sub_bcast)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = out * valid_f
        return out


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch7GraphEncoder(nn.Module):
    """
    embed → flatten → L interleaved layers → scatter_mean per node → sum_pool → [B, H]

    Readout: per-canonical-node mean of all subgraph appearances, then sum_pool.
    This matches the standard GNN readout (vanilla GIN uses sum_pool), avoiding
    the double-mean that was hurting the previous version.
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
        self.bond_encoder = nn.Embedding(edge_dim, hidden_dim)

        self.layers = nn.ModuleList([
            Arch7Layer(hidden_dim, hidden_dim, mlp_layers, conv_type, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        device = sf.x.device

        sf.x         = self.atom_encoder(sf.x.long().squeeze(-1))                      # [N, H]
        sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)          # [E, H]

        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        h_flat = x_flat
        S      = sf.nodes_sampled.shape[0]
        B      = sf.sample_ptr.size(0) - 1

        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid,
                node_ids, N_total, sf.edge_index, sf.edge_attr,
                sub_batch, S,
            )

        # ── Readout: mean over subgraph appearances per canonical node ──────
        valid_mask = node_ids >= 0
        node_embs = scatter(
            h_flat[valid_mask], node_ids[valid_mask],
            dim=0, reduce='mean', dim_size=N_total,
        )   # [N_total, H]

        # Sum-pool canonical node embeddings per graph → [B, H]
        return global_add_pool(node_embs, sf.batch)


# ---------------------------------------------------------------------------
# Node-level encoder
# ---------------------------------------------------------------------------

class Arch7NodeEncoder(nn.Module):
    """Node-level variant: returns per-node embeddings [N_total, H]."""

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
        self.bond_encoder = nn.Embedding(edge_dim, hidden_dim)

        self.layers = nn.ModuleList([
            Arch7Layer(hidden_dim, hidden_dim, mlp_layers, conv_type, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        device = sf.x.device

        sf.x         = self.atom_encoder(sf.x.long().squeeze(-1))
        sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)

        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        h_flat = x_flat
        S      = sf.nodes_sampled.shape[0]

        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid,
                node_ids, N_total, sf.edge_index, sf.edge_attr,
                sub_batch, S,
            )

        valid_mask = node_ids >= 0
        return scatter(
            h_flat[valid_mask], node_ids[valid_mask],
            dim=0, reduce='mean', dim_size=N_total,
        )   # [N_total, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-7')
def build_arch7(cfg: ExperimentConfig):
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
        return Arch7NodeEncoder(**common)
    else:
        return Arch7GraphEncoder(**common)
