"""
ARCH-11: Graphlet-View Transformer.

Replaces scatter-mean x_vv aggregation with a pre-norm multi-head
self-attention block (SubgraphViewAttn) that attends over all m graphlet
views of each canonical node.

The graphlet sampler gives exactly m subgraphs per canonical node, enabling
a clean [N_total, m, H] reshape. The transformer learns content-adaptive
pooling over subgraph views — distinguishing "this node is a triangle hub"
from "this node sits on a long path" — a strict generalisation of the
uniform mean used in SUN/ARCH-7-V2.

Per-layer update:
    h_skip = skip_proj(h)
    h1     = BN(local_conv(h, intra_edges, ea))         # local GNN on S*k
    x_sum  = scatter_mean(h[valid], node_ids[valid])    # collapse to N_total
    h2     = BN(global_conv(x_sum, orig_edges))[v]      # global GNN on N_total
    x_vv   = vv_proj(ViewAttn(roots.view(N,m,H)).mean)  # attended x_vv
    x_kk   = kk_proj(subgraph_root → all in subgraph)
    out    = ReLU(h_skip + h1 + h2 + x_vv + x_kk)

Requires: _build_all_node_targets must be active (ARCH-11 in the model list
in experiment.py) so that S = N_total * m is guaranteed.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import scatter

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import register_model
from gxl.models.arch_2_v2 import make_mlp, _flatten_subgraphs
from gxl.models.arch_7 import _make_conv


# ---------------------------------------------------------------------------
# Subgraph-view attention
# ---------------------------------------------------------------------------

class SubgraphViewAttn(nn.Module):
    """
    Pre-norm Transformer encoder block that attends over m subgraph views.

    Input:  [N_total, m, H]
    Output: [N_total, H]   (mean-pool over m attended tokens)

    Uses pre-norm (norm before attention/FFN) for training stability.
    FFN uses GELU with 4× expansion, matching standard Transformer practice.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        H = hidden_dim
        self.norm1 = nn.LayerNorm(H)
        self.attn  = nn.MultiheadAttention(
            H, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(H)
        self.ffn   = nn.Sequential(
            nn.Linear(H, 4 * H),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * H, H),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N_total, m, H]
        r = self.norm1(x)
        r, _ = self.attn(r, r, r, need_weights=False)
        x = x + r
        x = x + self.ffn(self.norm2(x))
        return x.mean(dim=1)   # [N_total, H]


# ---------------------------------------------------------------------------
# Core layer
# ---------------------------------------------------------------------------

class Arch11Layer(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        edge_dim:   int,
        num_heads:  int   = 4,
        mlp_layers: int   = 2,
        conv_type:  str   = 'gine',
        dropout:    float = 0.0,
    ):
        super().__init__()
        H = hidden_dim
        self.use_edge_attr = (conv_type == 'gine')
        self.dropout = dropout

        # Local GNN (runs on S*k — keep single for speed)
        self.local_conv = _make_conv(conv_type, H, H, edge_dim, mlp_layers)
        self.local_bn   = BatchNorm(H)

        # Global GNN (runs on N_total — cheap)
        self.global_conv = _make_conv(conv_type, H, H, edge_dim, mlp_layers)
        self.global_bn   = BatchNorm(H)

        # Subgraph-view attention (replaces x_vv scatter-mean)
        self.view_attn = SubgraphViewAttn(H, num_heads, dropout)
        self.vv_proj   = nn.Linear(H, H)

        self.skip_proj = nn.Linear(H, H)
        self.kk_proj   = nn.Linear(H, H)

    def forward(
        self,
        h_flat:        torch.Tensor,   # [S*k, H]
        intra_ei:      torch.Tensor,   # [2, E_sub]
        ea_flat:       torch.Tensor,   # [E_sub, H] or None
        valid:         torch.Tensor,   # [S*k] bool
        node_ids:      torch.Tensor,   # [S*k] global node ID (-1=padding)
        N_total:       int,
        edge_index:    torch.Tensor,   # [2, E]
        edge_attr:     torch.Tensor,   # [E, H] or None
        sub_batch:     torch.Tensor,   # [S*k] subgraph index
        m:             int,            # subgraphs per canonical node
        root_flat_idx: torch.Tensor,   # [S] flat indices of subgraph roots
    ) -> torch.Tensor:

        valid_f     = valid.float().unsqueeze(-1)
        clamped_ids = node_ids.clamp(min=0)
        valid_mask  = node_ids >= 0

        # ── skip ─────────────────────────────────────────────────────────
        h_skip = self.skip_proj(h_flat) * valid_f

        # ── local GNN ────────────────────────────────────────────────────
        if self.use_edge_attr and ea_flat is not None:
            h1 = self.local_conv(h_flat, intra_ei, ea_flat)
        else:
            h1 = self.local_conv(h_flat, intra_ei)
        h1 = self.local_bn(h1) * valid_f

        # ── global GNN ───────────────────────────────────────────────────
        x_sum = scatter(
            h_flat[valid_mask], node_ids[valid_mask],
            dim=0, reduce='mean', dim_size=N_total,
        )
        if self.use_edge_attr and edge_attr is not None:
            h_global = self.global_conv(x_sum, edge_index, edge_attr)
        else:
            h_global = self.global_conv(x_sum, edge_index)
        h2 = self.global_bn(h_global)[clamped_ids] * valid_f

        # ── x_vv via view-attention ───────────────────────────────────────
        # Gather root embedding for each subgraph, reshape to [N_total, m, H],
        # attend over m views, project back. This is content-adaptive x_vv.
        h_roots   = h_flat[root_flat_idx]              # [S, H]
        h_roots3d = h_roots.view(N_total, m, -1)       # [N_total, m, H]
        x_vv_node = self.view_attn(h_roots3d)          # [N_total, H]
        x_vv      = self.vv_proj(x_vv_node[clamped_ids]) * valid_f

        # ── x_kk: subgraph root → all nodes in same subgraph ─────────────
        x_kk = self.kk_proj(h_flat[root_flat_idx[sub_batch]]) * valid_f

        # ── combine ───────────────────────────────────────────────────────
        out = F.relu(h_skip + h1 + h2 + x_vv + x_kk)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out * valid_f


# ---------------------------------------------------------------------------
# Shared encoder body
# ---------------------------------------------------------------------------

def _encode_arch11(
    sf:           SubgraphFeaturesBatch,
    atom_encoder: nn.Module,
    bond_encoder: nn.Module,
    role_encoder: nn.Embedding,
    layers:       nn.ModuleList,
) -> tuple:
    """Run embedding + MP, return (h_flat, node_ids, N_total)."""
    if not sf.x.is_floating_point():
        sf.x = atom_encoder(sf.x.long().squeeze(-1))
    if sf.edge_attr is not None and not sf.edge_attr.is_floating_point():
        sf.edge_attr = bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)

    x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
        _flatten_subgraphs(sf)

    S, k   = sf.nodes_sampled.shape
    device = x_flat.device

    # m subgraphs per canonical node — required for view(N_total, m, H).
    # This holds iff _build_all_node_targets is active for this model name.
    assert S % N_total == 0, (
        f"ARCH-11 requires S ({S}) % N_total ({N_total}) == 0. "
        "Add 'ARCH-11' to the _build_all_node_targets list in experiment.py."
    )
    m = S // N_total

    root_flat_idx = torch.arange(S, device=device) * k
    is_root = torch.zeros(S * k, dtype=torch.bool, device=device)
    is_root[root_flat_idx] = True

    role_emb = role_encoder(is_root.long())
    h_flat   = (x_flat + role_emb) * valid.float().unsqueeze(-1)

    for layer in layers:
        h_flat = layer(
            h_flat, intra_ei, ea_flat, valid,
            node_ids, N_total, sf.edge_index, sf.edge_attr,
            sub_batch, m, root_flat_idx,
        )

    return h_flat, node_ids, N_total


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch11GraphEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_dim:  int,
        edge_dim:    int,
        num_layers:  int   = 6,
        num_heads:   int   = 4,
        mlp_layers:  int   = 2,
        dropout:     float = 0.0,
        conv_type:   str   = 'gine',
    ):
        super().__init__()
        self.atom_encoder = nn.Embedding(in_channels, hidden_dim)
        self.bond_encoder = nn.Embedding(edge_dim,    hidden_dim)
        self.role_encoder = nn.Embedding(2,           hidden_dim)

        self.layers = nn.ModuleList([
            Arch11Layer(hidden_dim, hidden_dim, num_heads, mlp_layers, conv_type, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        h_flat, node_ids, N_total = _encode_arch11(
            sf, self.atom_encoder, self.bond_encoder, self.role_encoder, self.layers
        )
        valid_mask = node_ids >= 0
        node_embs = scatter(
            h_flat[valid_mask], node_ids[valid_mask],
            dim=0, reduce='mean', dim_size=N_total,
        )
        return global_add_pool(node_embs, sf.batch)


# ---------------------------------------------------------------------------
# Node-level encoder
# ---------------------------------------------------------------------------

class Arch11NodeEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_dim:  int,
        edge_dim:    int,
        num_layers:  int   = 6,
        num_heads:   int   = 4,
        mlp_layers:  int   = 2,
        dropout:     float = 0.0,
        conv_type:   str   = 'gine',
    ):
        super().__init__()
        self.atom_encoder = nn.Embedding(in_channels, hidden_dim)
        self.bond_encoder = nn.Embedding(edge_dim,    hidden_dim)
        self.role_encoder = nn.Embedding(2,           hidden_dim)

        self.layers = nn.ModuleList([
            Arch11Layer(hidden_dim, hidden_dim, num_heads, mlp_layers, conv_type, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        h_flat, node_ids, N_total = _encode_arch11(
            sf, self.atom_encoder, self.bond_encoder, self.role_encoder, self.layers
        )
        valid_mask = node_ids >= 0
        return scatter(
            h_flat[valid_mask], node_ids[valid_mask],
            dim=0, reduce='mean', dim_size=N_total,
        )


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-11')
def build_arch11(cfg: ExperimentConfig):
    kw = cfg.model_config.kwargs
    is_node_level = cfg.task in ('Node-Classification', 'Link-Prediction')

    common = dict(
        in_channels = cfg.model_config.node_feature_dim,
        edge_dim    = cfg.model_config.edge_feature_dim,
        hidden_dim  = cfg.model_config.hidden_dim,
        num_layers  = cfg.model_config.mpnn_layers,
        num_heads   = kw.get('num_heads', 4),
        mlp_layers  = kw.get('mlp_layers', 2),
        dropout     = cfg.model_config.dropout,
        conv_type   = cfg.model_config.mpnn_type,
    )

    return Arch11NodeEncoder(**common) if is_node_level else Arch11GraphEncoder(**common)
