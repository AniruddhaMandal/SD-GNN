"""
ARCH-7: Interleaved Subgraph + Global GNN (SUN-inspired)

Implements SUN's three core ideas in the SD-GNN framework:

  1. All-node subgraph pooling — all k nodes per subgraph contribute to the final
     readout, not just the root node (which ARCH-5 discards 9/10 of computation).

  2. Per-layer interleaved local + global GNN — at every layer:
       h1 = local_gnn(intra-subgraph edges)
       x_sum = mean of h over all canonical node appearances
       h2 = global_gnn(x_sum, original_graph_edges)
       h_flat = ReLU(BN(h1) + BN(h2)[node_ids])
     Both branches run in parallel with independent weights. Each subgraph node
     gets global graph context at every layer via h2[node_ids].

  3. Mean-of-subgraphs final readout — mean pool all nodes per subgraph, then
     mean over subgraphs per graph (matches SUN exactly).
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import (
    GINEConv, GINConv, GCNConv, SAGEConv,
)
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import scatter

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import register_model
from gxl.models.arch_2_v2 import make_mlp, _flatten_subgraphs

from typing import Literal


# ---------------------------------------------------------------------------
# Conv factory
# ---------------------------------------------------------------------------

def _make_conv(
    conv_type:  str,
    in_dim:     int,
    out_dim:    int,
    edge_dim:   int,
    mlp_layers: int,
):
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
    One interleaved local + global GNN layer.

    Local branch:  GNN on flat [S*k] space using intra-subgraph edges only.
    Global branch: GNN on x_sum (per-canonical-node mean of h_flat) using
                   the original full-graph edges.
    Combine:       h_flat = ReLU(BN(h_local) + BN(h_global)[node_ids])
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

        self.local_conv  = _make_conv(conv_type, hidden_dim, hidden_dim, edge_dim, mlp_layers)
        self.local_bn    = BatchNorm(hidden_dim)

        self.global_conv = _make_conv(conv_type, hidden_dim, hidden_dim, edge_dim, mlp_layers)
        self.global_bn   = BatchNorm(hidden_dim)

    def forward(
        self,
        h_flat:     torch.Tensor,   # [S*k, H]
        intra_ei:   torch.Tensor,   # [2, E_sub] — intra-subgraph edges in flat index space
        ea_flat:    torch.Tensor,   # [E_sub, H] or None
        valid:      torch.Tensor,   # [S*k] bool — False for padding positions
        node_ids:   torch.Tensor,   # [S*k] int  — canonical node id, -1 for padding
        N_total:    int,
        edge_index: torch.Tensor,   # [2, E] original graph edges
        edge_attr:  torch.Tensor,   # [E, H] original edge attrs
    ) -> torch.Tensor:               # [S*k, H]

        valid_f     = valid.float().unsqueeze(-1)   # [S*k, 1]
        clamped_ids = node_ids.clamp(min=0)         # [S*k], padding clamped to 0

        # ── local GNN ──────────────────────────────────────────────────────
        if self.use_edge_attr and ea_flat is not None:
            h1 = self.local_conv(h_flat, intra_ei, ea_flat)
        else:
            h1 = self.local_conv(h_flat, intra_ei)
        h1 = h1 * valid_f               # zero out padding before BN
        h1 = self.local_bn(h1)
        h1 = h1 * valid_f               # re-zero after BN (BN can shift zeros)

        # ── x_sum: mean of h_flat over all appearances of each canonical node ──
        valid_mask = node_ids >= 0      # [S*k] — same as valid, but clearer intent
        x_sum = scatter(
            h_flat[valid_mask],
            node_ids[valid_mask],
            dim=0, reduce='mean', dim_size=N_total,
        )   # [N_total, H]

        # ── global GNN on x_sum ────────────────────────────────────────────
        if self.use_edge_attr and edge_attr is not None:
            h2 = self.global_conv(x_sum, edge_index, edge_attr)
        else:
            h2 = self.global_conv(x_sum, edge_index)
        h2 = self.global_bn(h2)        # [N_total, H]

        # ── broadcast global context back to flat subgraph positions ───────
        h2_bcast = h2[clamped_ids] * valid_f    # [S*k, H], padding zeroed

        # ── combine ────────────────────────────────────────────────────────
        out = F.relu(h1 + h2_bcast)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = out * valid_f             # ensure padding stays zero after dropout
        return out


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch7GraphEncoder(nn.Module):
    """
    embed → flatten → L interleaved layers → mean subgraph pool → mean graph pool → [B, H]
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

        # Embed raw integer atom/bond types
        sf.x        = self.atom_encoder(sf.x.long().squeeze(-1))                      # [N, H]
        sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)         # [E, H]

        # Flatten into [S*k] space; ea_flat uses the already-embedded sf.edge_attr
        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        h_flat = x_flat   # [S*k, H], padding zeroed by _flatten_subgraphs

        S = sf.nodes_sampled.shape[0]

        # Subgraph → graph batch mapping via sample_ptr [B+1]
        B = sf.sample_ptr.size(0) - 1
        subgraph_graph_batch = torch.repeat_interleave(
            torch.arange(B, device=device),
            sf.sample_ptr[1:] - sf.sample_ptr[:-1],
        )   # [S]

        # Interleaved local + global layers
        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid,
                node_ids, N_total, sf.edge_index, sf.edge_attr,
            )

        # Mean-pool all valid nodes per subgraph → [S, H]
        valid_mask = node_ids >= 0
        h_sub = scatter(
            h_flat[valid_mask],
            sub_batch[valid_mask],
            dim=0, reduce='mean', dim_size=S,
        )

        # Mean over subgraphs per graph → [B, H]
        h_graph = scatter(
            h_sub, subgraph_graph_batch,
            dim=0, reduce='mean', dim_size=B,
        )

        return h_graph   # [B, H]


# ---------------------------------------------------------------------------
# Node-level encoder
# ---------------------------------------------------------------------------

class Arch7NodeEncoder(nn.Module):
    """
    Same pipeline but returns per-node embeddings for node classification /
    link prediction tasks.
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

        sf.x        = self.atom_encoder(sf.x.long().squeeze(-1))
        sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)

        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        h_flat = x_flat

        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid,
                node_ids, N_total, sf.edge_index, sf.edge_attr,
            )

        # Aggregate: mean over all subgraph appearances per canonical node → [N_total, H]
        valid_mask = node_ids >= 0
        node_embs = scatter(
            h_flat[valid_mask],
            node_ids[valid_mask],
            dim=0, reduce='mean', dim_size=N_total,
        )

        return node_embs   # [N_total, H]


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
