"""
ARCH-7-V2: SUN-complete Interleaved Subgraph + Global GNN.

Extends ARCH-7 with the four SUN components that were missing:

  1. Root indicator — binary Embedding(2, H) added to initial node features so
     the model knows which position is the subgraph root from the start.

  2. x_vv term — canonical root representation broadcast to ALL copies of the
     same canonical node v across every subgraph.  Gives each node v a global
     view of its own "average root state" across all m subgraph contexts.

  3. x_kk term — current subgraph's root representation broadcast to all nodes
     in that subgraph, replacing ARCH-7's mean-of-all broadcast.  Root identity
     (not average node) is the correct SUN "anchor".

  4. Role differentiation — separate GNN weights for root vs non-root positions
     (4 GNNs per layer: local × {root, non-root}, global × {root, non-root}).

ARCH-7's better readout is retained:
    scatter_mean(h_flat, node_ids) → global_add_pool
vs SUN's double mean (scatter_mean subgraphs → scatter_mean graphs).

Per-layer update (5 terms, roles differentiated):
    h_skip = skip_proj(h)                                    # x_kv
    h1     = BN(local_conv[role](h, intra_edges))           # role-aware local GNN
    h2     = BN(global_conv[role](x_sum, orig_edges))[v]    # role-aware global GNN
    x_vv   = vv_proj(canonical_root[v])                     # global root context
    x_kk   = kk_proj(subgraph_root → all in same subgraph)  # local root context
    out    = ReLU(h_skip + h1 + h2 + x_vv + x_kk)
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv, GINConv, GCNConv, SAGEConv, global_add_pool
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import scatter

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import register_model
from gxl.models.arch_2_v2 import make_mlp, _flatten_subgraphs
from gxl.models.arch_7 import _make_conv


# ---------------------------------------------------------------------------
# Core layer
# ---------------------------------------------------------------------------

class Arch7V2Layer(nn.Module):
    """
    One interleaved layer with 5 terms and role-differentiated GNNs.

    Args:
        hidden_dim: feature dimension H
        edge_dim:   edge feature dimension (used by GINEConv)
        mlp_layers: depth of GINEConv MLP
        conv_type:  'gine' | 'gin' | 'gcn' | 'sage'
        dropout:    dropout probability
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
        H = hidden_dim
        self.use_edge_attr = (conv_type == 'gine')
        self.dropout = dropout

        # ── local GNN: separate weights per role ──────────────────────────
        self.local_conv      = _make_conv(conv_type, H, H, edge_dim, mlp_layers)
        self.local_conv_root = _make_conv(conv_type, H, H, edge_dim, mlp_layers)
        self.local_bn        = BatchNorm(H)
        self.local_bn_root   = BatchNorm(H)

        # ── global GNN: separate weights per role ─────────────────────────
        self.global_conv      = _make_conv(conv_type, H, H, edge_dim, mlp_layers)
        self.global_conv_root = _make_conv(conv_type, H, H, edge_dim, mlp_layers)
        self.global_bn        = BatchNorm(H)
        self.global_bn_root   = BatchNorm(H)

        # ── skip (x_kv), x_vv, x_kk projections ──────────────────────────
        self.skip_proj = nn.Linear(H, H)
        self.vv_proj   = nn.Linear(H, H)
        self.kk_proj   = nn.Linear(H, H)

    def forward(
        self,
        h_flat:        torch.Tensor,   # [S*k, H]
        intra_ei:      torch.Tensor,   # [2, E_sub]
        ea_flat:       torch.Tensor,   # [E_sub, H] or None
        valid:         torch.Tensor,   # [S*k] bool
        node_ids:      torch.Tensor,   # [S*k] int, -1 = padding
        N_total:       int,
        edge_index:    torch.Tensor,   # [2, E] original graph
        edge_attr:     torch.Tensor,   # [E, H]
        sub_batch:     torch.Tensor,   # [S*k] int, subgraph index per flat position
        S:             int,
        k:             int,
        root_flat_idx: torch.Tensor,   # [S] — root flat index per subgraph
        is_root:       torch.Tensor,   # [S*k] bool
    ) -> torch.Tensor:                  # [S*k, H]

        valid_f     = valid.float().unsqueeze(-1)          # [S*k, 1]
        is_root_f   = is_root.unsqueeze(-1)                # [S*k, 1] bool
        clamped_ids = node_ids.clamp(min=0)                # safe index into [N_total, H]
        valid_mask  = node_ids >= 0

        # ── skip (x_kv) ──────────────────────────────────────────────────
        h_skip = self.skip_proj(h_flat) * valid_f          # [S*k, H]

        # ── local GNN (role-differentiated) ──────────────────────────────
        if self.use_edge_attr and ea_flat is not None:
            h1_nr = self.local_conv(h_flat, intra_ei, ea_flat)
            h1_r  = self.local_conv_root(h_flat, intra_ei, ea_flat)
        else:
            h1_nr = self.local_conv(h_flat, intra_ei)
            h1_r  = self.local_conv_root(h_flat, intra_ei)
        h1_nr = self.local_bn(h1_nr)      * valid_f
        h1_r  = self.local_bn_root(h1_r)  * valid_f
        h1 = torch.where(is_root_f, h1_r, h1_nr)          # [S*k, H]

        # ── global GNN (role-differentiated) ─────────────────────────────
        x_sum = scatter(
            h_flat[valid_mask], node_ids[valid_mask],
            dim=0, reduce='mean', dim_size=N_total,
        )   # [N_total, H]

        if self.use_edge_attr and edge_attr is not None:
            h2_nr = self.global_conv(x_sum, edge_index, edge_attr)
            h2_r  = self.global_conv_root(x_sum, edge_index, edge_attr)
        else:
            h2_nr = self.global_conv(x_sum, edge_index)
            h2_r  = self.global_conv_root(x_sum, edge_index)
        h2_nr = self.global_bn(h2_nr)
        h2_r  = self.global_bn_root(h2_r)
        # Broadcast [N_total, H] → [S*k, H] then select by role
        h2_nr_bcast = h2_nr[clamped_ids] * valid_f
        h2_r_bcast  = h2_r[clamped_ids]  * valid_f
        h2 = torch.where(is_root_f, h2_r_bcast, h2_nr_bcast)  # [S*k, H]

        # ── x_vv: canonical root → all copies of same canonical node ─────
        # root_ids[s] = canonical node ID of subgraph s's root (-1 if degenerate)
        root_ids   = node_ids[root_flat_idx]               # [S]
        root_valid = root_ids >= 0
        h_roots    = h_flat[root_flat_idx]                 # [S, H]
        x_vv_canonical = scatter(
            h_roots[root_valid], root_ids[root_valid],
            dim=0, reduce='mean', dim_size=N_total,
        )   # [N_total, H]
        x_vv = self.vv_proj(x_vv_canonical[clamped_ids]) * valid_f  # [S*k, H]

        # ── x_kk: subgraph root → all nodes in same subgraph ─────────────
        # root_flat_idx[sub_batch[i]] = flat index of subgraph i's root
        x_kk = self.kk_proj(h_flat[root_flat_idx[sub_batch]]) * valid_f  # [S*k, H]

        # ── combine ───────────────────────────────────────────────────────
        out = F.relu(h_skip + h1 + h2 + x_vv + x_kk)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out * valid_f


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch7V2GraphEncoder(nn.Module):
    """
    embed → role-init → L interleaved layers → scatter_mean per node → sum_pool → [B, H]
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
        self.role_encoder = nn.Embedding(2,           hidden_dim)  # 0=non-root, 1=root

        self.layers = nn.ModuleList([
            Arch7V2Layer(hidden_dim, hidden_dim, mlp_layers, conv_type, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        if not sf.x.is_floating_point():
            sf.x = self.atom_encoder(sf.x.long().squeeze(-1))
        if sf.edge_attr is not None and not sf.edge_attr.is_floating_point():
            sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)
        # else: already encoded to hidden_dim by dataset transform (e.g. OGBAtomEncoder)

        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        S, k   = sf.nodes_sampled.shape
        device = x_flat.device

        # Root positions: subgraph s has root at flat index s*k
        root_flat_idx = torch.arange(S, device=device) * k    # [S]

        # is_root mask — safe: root_flat_idx ⊂ [0, S*k)
        is_root = torch.zeros(S * k, dtype=torch.bool, device=device)
        is_root[root_flat_idx] = True

        # Initial features + role indicator
        role_emb = self.role_encoder(is_root.long())           # [S*k, H]
        valid_f  = valid.float().unsqueeze(-1)
        h_flat   = (x_flat + role_emb) * valid_f

        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid,
                node_ids, N_total, sf.edge_index, sf.edge_attr,
                sub_batch, S, k, root_flat_idx, is_root,
            )

        # Readout: mean over subgraph appearances per canonical node → sum-pool
        valid_mask = node_ids >= 0
        node_embs = scatter(
            h_flat[valid_mask], node_ids[valid_mask],
            dim=0, reduce='mean', dim_size=N_total,
        )   # [N_total, H]
        return global_add_pool(node_embs, sf.batch)            # [B, H]


# ---------------------------------------------------------------------------
# Node-level encoder
# ---------------------------------------------------------------------------

class Arch7V2NodeEncoder(nn.Module):
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
        self.bond_encoder = nn.Embedding(edge_dim,    hidden_dim)
        self.role_encoder = nn.Embedding(2,           hidden_dim)

        self.layers = nn.ModuleList([
            Arch7V2Layer(hidden_dim, hidden_dim, mlp_layers, conv_type, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        if not sf.x.is_floating_point():
            sf.x         = self.atom_encoder(sf.x.long().squeeze(-1))
            sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)

        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        S, k   = sf.nodes_sampled.shape
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

        valid_mask = node_ids >= 0
        return scatter(
            h_flat[valid_mask], node_ids[valid_mask],
            dim=0, reduce='mean', dim_size=N_total,
        )   # [N_total, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-7-V2')
def build_arch7_v2(cfg: ExperimentConfig):
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
        return Arch7V2NodeEncoder(**common)
    else:
        return Arch7V2GraphEncoder(**common)
