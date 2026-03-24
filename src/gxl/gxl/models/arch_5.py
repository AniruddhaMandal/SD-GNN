"""
ARCH-5: Independent Subgraph GNN + Post-Aggregation Global GNN

Two-phase pipeline that separates local and global processing:

  Phase 1 — Subgraph (local):
    L1 layers of independent GNN within each subgraph (no cross-subgraph MP).
    Extract root node embedding per subgraph → [S, H].
    Aggregate m root embeddings per node with log_prob weighted_mean → node_embs [N, H].

  Phase 2 — Global:
    L2 layers of standard GNN on node_embs using the ORIGINAL graph edge structure.
    Full inter-node communication; every node can reach every other node.
    Readout: sum-pool over all nodes → graph_emb [G, H].

Why this fixes the ARCH-4 coverage problem without ARCH-3 over-smoothing:
  - Phase 1 computes DIVERSE per-node embeddings (each subgraph is an independent view).
    Unlike ARCH-3's per-layer scatter-mean which blurs subgraph diversity, here
    aggregation only happens ONCE after all L1 layers → diversity is preserved.
  - Phase 2 starts from those diverse node_embs and runs standard GNN
    (no different from vanilla GIN in this phase), giving full coverage via the
    real edge structure. No convergence-to-mean because each node starts with
    a different, subgraph-enriched representation.
  - Net effect: node_embs[v] encodes "what v's sampled neighborhoods look like",
    and the global GNN then lets those richer features propagate to all neighbours.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import (
    GINEConv, GINConv, GCNConv, SAGEConv,
    global_add_pool, global_mean_pool, global_max_pool,
)
from torch_geometric.nn.norm import BatchNorm

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import get_aggregator, register_model

from gxl.models.arch_2_v2 import make_mlp
from gxl.models.arch_4 import IndependentSubgraphEncoder

from typing import Literal


# ---------------------------------------------------------------------------
# Global GNN layer  (standard MP on original graph)
# ---------------------------------------------------------------------------

class GlobalGNNLayer(nn.Module):
    """One GNN layer operating on the original graph's edge structure."""

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        edge_dim:     int,
        mlp_layers:   int   = 2,
        conv_type:    str   = 'gin',
        dropout:      float = 0.0,
        residual:     bool  = True,
    ):
        super().__init__()
        self.conv_type = conv_type
        self.dropout   = dropout
        self.residual  = residual
        self.use_ea    = (conv_type == 'gine')
        self.edge_dim  = edge_dim

        self.conv = self._make_conv(in_channels, out_channels, mlp_layers)
        self.bn   = BatchNorm(out_channels)

        if residual and in_channels != out_channels:
            self.res_proj = nn.Linear(in_channels, out_channels)
        else:
            self.res_proj = nn.Identity()

    def forward(
        self,
        h:          torch.Tensor,  # [N, in_channels]
        edge_index: torch.Tensor,  # [2, E]
        edge_attr:  torch.Tensor,  # [E, edge_dim] or None
    ) -> torch.Tensor:              # [N, out_channels]
        h_res = h

        if self.use_ea and edge_attr is not None:
            h_new = self.conv(h, edge_index, edge_attr)
        else:
            h_new = self.conv(h, edge_index)

        h_new = self.bn(h_new)
        if self.residual:
            h_new = h_new + self.res_proj(h_res)
        h_new = F.dropout(h_new, p=self.dropout, training=self.training)
        return h_new

    def _make_conv(self, in_dim, out_dim, mlp_layers):
        if self.conv_type == 'gine':
            return GINEConv(make_mlp(in_dim, in_dim, out_dim, mlp_layers),
                            train_eps=True, edge_dim=self.edge_dim)
        if self.conv_type == 'gin':
            return GINConv(make_mlp(in_dim, in_dim, out_dim, mlp_layers),
                           train_eps=True)
        if self.conv_type == 'gcn':
            return GCNConv(in_dim, out_dim)
        if self.conv_type == 'sage':
            return SAGEConv(in_dim, out_dim)
        raise ValueError(f"Unknown conv_type: {self.conv_type}")


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch5GraphEncoder(nn.Module):
    """
    Phase 1: L1-layer independent subgraph GNN → aggregate root embs per node [N, H]
    Phase 2: L2-layer global GNN on original graph → [N, H]
    Readout: sum-pool per graph → [G, H]
    """

    def __init__(
        self,
        in_channels:   int,
        hidden_dim:    int,
        edge_dim:      int,
        sub_layers:    int   = 4,    # L1: independent subgraph GNN layers
        global_layers: int   = 4,    # L2: global GNN layers
        mlp_layers:    int   = 2,
        dropout:       float = 0.0,
        conv_type:     str   = 'gin',
        residual:      bool  = True,
        init_mode:     Literal['concat', 'add'] = 'concat',
        aggregator:    str   = 'weighted_mean',
        temperature:   float = 0.5,
        graph_pool:    str   = 'sum',
    ):
        super().__init__()

        # Learnable atom/bond embeddings (in_channels = atom vocab, edge_dim = bond vocab)
        # ZINC: atom types 0-20, bond types 1-3 (shift to 0-2)
        self.atom_encoder = nn.Embedding(in_channels, hidden_dim)
        self.bond_encoder = nn.Embedding(edge_dim, hidden_dim)

        # Phase 1: subgraph encoder — operates on hidden_dim features after embedding
        self.sub_encoder = IndependentSubgraphEncoder(
            in_channels = hidden_dim,
            hidden_dim  = hidden_dim,
            edge_dim    = hidden_dim,
            num_layers  = sub_layers,
            mlp_layers  = mlp_layers,
            dropout     = dropout,
            conv_type   = conv_type,
            residual    = residual,
            init_mode   = init_mode,
        )

        # Node-level aggregator: m root embeddings → 1 node embedding
        try:
            self.aggregator = get_aggregator(aggregator)(
                hidden_dim=hidden_dim, temperature=temperature)
        except Exception:
            if aggregator in ('sum', 'add'):
                self.aggregator = global_add_pool
            elif aggregator == 'max':
                self.aggregator = global_max_pool
            else:
                self.aggregator = global_mean_pool

        # Phase 2: global GNN on original graph — edge features also hidden_dim after embedding
        self.global_layers = nn.ModuleList([
            GlobalGNNLayer(
                in_channels  = hidden_dim,
                out_channels = hidden_dim,
                edge_dim     = hidden_dim,
                mlp_layers   = mlp_layers,
                conv_type    = conv_type,
                dropout      = dropout,
                residual     = residual,
            )
            for _ in range(global_layers)
        ])

        # Graph readout
        if graph_pool == 'sum':
            self.graph_pool_fn = global_add_pool
        elif graph_pool == 'mean':
            self.graph_pool_fn = global_mean_pool
        elif graph_pool == 'max':
            self.graph_pool_fn = global_max_pool
        else:
            raise ValueError(f"Unknown graph_pool: {graph_pool}")

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        device = sf.x.device

        # Encode raw integer atom/bond types to hidden_dim vectors
        sf.x = self.atom_encoder(sf.x.long().squeeze(-1))                      # [N, H]
        sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)  # [E, H]

        # ── Phase 1: subgraph GNN ──────────────────────────────────────────
        root_embs, target_batch, log_probs, T, N_total = self.sub_encoder(sf)
        # root_embs: [S, H],  target_batch: [S] → 0..T-1

        # Aggregate m root embeddings → node_embs [T, H]
        if getattr(self.aggregator, 'needs_log_probs', False):
            node_embs = self.aggregator(root_embs, target_batch,
                                        log_probs=log_probs)
        else:
            node_embs = self.aggregator(root_embs, target_batch)
        # node_embs: [T, H] where T = N_total (all-node-targets)

        # ── Phase 2: global GNN ────────────────────────────────────────────
        # node_embs is ordered by target_nodes (= arange(N_total) via
        # _build_all_node_targets), so it maps directly to sf.edge_index.
        h = node_embs
        for layer in self.global_layers:
            h = layer(h, sf.edge_index, sf.edge_attr)

        # ── Graph readout ──────────────────────────────────────────────────
        return self.graph_pool_fn(h, sf.batch)   # [G, H]


# ---------------------------------------------------------------------------
# Node-level encoder
# ---------------------------------------------------------------------------

class Arch5NodeEncoder(nn.Module):
    """
    Phase 1: subgraph GNN → aggregate root embs per node → node_embs [T, H]
    Phase 2: global GNN on original graph → h [N, H]
    Returns h for node classification / link prediction.
    """

    def __init__(
        self,
        in_channels:   int,
        hidden_dim:    int,
        edge_dim:      int,
        sub_layers:    int   = 4,
        global_layers: int   = 4,
        mlp_layers:    int   = 2,
        dropout:       float = 0.0,
        conv_type:     str   = 'gin',
        residual:      bool  = True,
        init_mode:     Literal['concat', 'add'] = 'concat',
        aggregator:    str   = 'weighted_mean',
        temperature:   float = 0.5,
    ):
        super().__init__()
        self.atom_encoder = nn.Embedding(in_channels, hidden_dim)
        self.bond_encoder = nn.Embedding(edge_dim, hidden_dim)
        self.sub_encoder = IndependentSubgraphEncoder(
            in_channels = hidden_dim,
            hidden_dim  = hidden_dim,
            edge_dim    = hidden_dim,
            num_layers  = sub_layers,
            mlp_layers  = mlp_layers,
            dropout     = dropout,
            conv_type   = conv_type,
            residual    = residual,
            init_mode   = init_mode,
        )
        try:
            self.aggregator = get_aggregator(aggregator)(
                hidden_dim=hidden_dim, temperature=temperature)
        except Exception:
            if aggregator in ('sum', 'add'):
                self.aggregator = global_add_pool
            elif aggregator == 'max':
                self.aggregator = global_max_pool
            else:
                self.aggregator = global_mean_pool

        self.global_layers = nn.ModuleList([
            GlobalGNNLayer(
                in_channels  = hidden_dim,
                out_channels = hidden_dim,
                edge_dim     = hidden_dim,
                mlp_layers   = mlp_layers,
                conv_type    = conv_type,
                dropout      = dropout,
                residual     = residual,
            )
            for _ in range(global_layers)
        ])

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        sf.x = self.atom_encoder(sf.x.long().squeeze(-1))                      # [N, H]
        sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)  # [E, H]
        root_embs, target_batch, log_probs, T, N_total = self.sub_encoder(sf)

        if getattr(self.aggregator, 'needs_log_probs', False):
            node_embs = self.aggregator(root_embs, target_batch, log_probs=log_probs)
        else:
            node_embs = self.aggregator(root_embs, target_batch)

        h = node_embs
        for layer in self.global_layers:
            h = layer(h, sf.edge_index, sf.edge_attr)

        return h   # [N_total, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-5')
def build_arch5(cfg: ExperimentConfig):
    kw = cfg.model_config.kwargs
    is_node_level = cfg.task in ('Node-Classification', 'Link-Prediction')

    # mpnn_layers splits evenly between subgraph and global phases by default.
    # Override with kwargs 'sub_layers' / 'global_layers' for asymmetric splits.
    total = cfg.model_config.mpnn_layers
    sub_layers    = kw.get('sub_layers',    total // 2)
    global_layers = kw.get('global_layers', total - sub_layers)

    base_kwargs = dict(
        in_channels   = cfg.model_config.node_feature_dim,
        edge_dim      = cfg.model_config.edge_feature_dim,
        hidden_dim    = cfg.model_config.hidden_dim,
        sub_layers    = sub_layers,
        global_layers = global_layers,
        dropout       = cfg.model_config.dropout,
        conv_type     = cfg.model_config.mpnn_type,
        residual      = kw.get('residual', True),
        init_mode     = kw.get('init_mode', 'concat'),
        aggregator    = kw.get('aggregator', 'weighted_mean'),
        temperature   = cfg.model_config.temperature or 0.5,
    )

    if is_node_level:
        return Arch5NodeEncoder(**base_kwargs)
    else:
        return Arch5GraphEncoder(
            **base_kwargs,
            graph_pool = kw.get('graph_pool', 'sum'),
        )
