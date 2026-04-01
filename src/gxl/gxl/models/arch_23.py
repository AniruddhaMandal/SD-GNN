"""
ARCH-23: Improved ARCH-10 — Role-Differentiated GNN + RWSE + Root-Token + Transformer Readout.

Three targeted improvements over ARCH-10 (which achieved 0.11 MAE on ZINC):

1. Global RWSE replaces BFS distance PE
   BFS distance cannot detect rings (all ring members are equidistant from root).
   RWSE returns-to-origin probability at step t captures ring structure: a node on
   a length-5 ring has non-zero p(return at t=5).  It is also faster to compute
   because it runs on the full batched graph once (no per-subgraph BFS loops).

2. Root-token readout replaces sum-pool
   ARCH-10 sums all valid positions in each subgraph → loses the distinction
   between "what is the root?" and "what is the context?".  Root-token readout
   directly lifts h_flat[root_flat_idx] → [S, H] as the subgraph representation,
   exactly the final GNN state of the most-informed node.  This mirrors ARCH-15/16.

3. 2-layer pre-norm Transformer (self-attn + FFN) replaces single MHA
   A single MHA with a linear residual has no position-wise non-linearity.
   Two Transformer layers (each: LayerNorm → MHA → residual, LayerNorm → FFN → residual)
   give the model expressive capacity to integrate patterns from all m subgraphs.
   HT correction (learnable α) still applied in every layer's self-attention.

Message passing (Arch9Layer) is unchanged — role differentiation between root and
non-root nodes with inter-root GINEConv on the original graph is kept intact.

Requirements:
  - Per-node subgraph sampling (_build_all_node_targets), so S = N_total * m.
  - graphlet sampler (sf.log_probs must be populated).
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
from gxl.models.arch_9 import Arch9Layer
from gxl.models.arch_19 import _global_rwse
from gxl.models.arch_10 import _ht_attn_bias


# ---------------------------------------------------------------------------
# Pre-norm Transformer layer with HT attention bias support
# ---------------------------------------------------------------------------

class HTTransformerLayer(nn.Module):
    """
    Pre-norm Transformer layer: LN → MHA(+HT bias) → residual → LN → FFN → residual.

    Using pre-norm (LN before sub-layer) is more stable than post-norm for
    deep Transformers and avoids the gradient vanishing issue with BatchNorm.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0,
                 ffn_multiplier: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mha   = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, ffn_multiplier * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_multiplier * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with optional HT key bias
        h, _ = self.mha(self.norm1(x), self.norm1(x), self.norm1(x),
                        attn_mask=attn_bias)
        x = x + h
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch23GraphEncoder(nn.Module):

    def __init__(
        self,
        in_channels:    int,
        hidden_dim:     int,
        edge_dim:       int,
        num_layers:     int   = 6,
        mlp_layers:     int   = 2,
        rwse_steps:     int   = 16,
        dropout:        float = 0.0,
        num_heads:      int   = 4,
        readout_layers: int   = 2,
    ):
        super().__init__()
        H = hidden_dim
        self.H          = H
        self.rwse_steps = rwse_steps
        self.num_heads  = num_heads

        self.atom_encoder = nn.Embedding(in_channels, H)
        self.bond_encoder = nn.Embedding(edge_dim,    H)
        self.rwse_proj    = nn.Sequential(nn.Linear(rwse_steps, H), nn.ReLU())

        self.layers = nn.ModuleList([
            Arch9Layer(H, H, mlp_layers, dropout)
            for _ in range(num_layers)
        ])

        # Learnable HT scale — same as ARCH-10
        self.ht_alpha = nn.Parameter(torch.ones(1))

        # 2-layer pre-norm Transformer readout
        assert H % num_heads == 0
        self.readout_layers = nn.ModuleList([
            HTTransformerLayer(H, num_heads, dropout)
            for _ in range(readout_layers)
        ])
        self.readout_norm = nn.LayerNorm(H)

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        sf.x         = self.atom_encoder(sf.x.long().squeeze(-1))
        sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)

        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        S, k   = sf.nodes_sampled.shape
        m      = S // N_total
        device = x_flat.device

        root_flat_idx = torch.arange(S, device=device) * k   # [S]

        # Sanitise log-probs
        if sf.log_probs is not None:
            lp = sf.log_probs.clone()
            lp[~torch.isfinite(lp)] = 0.0
        else:
            lp = torch.zeros(S, device=device)

        # ── Global RWSE (fast, ring-aware, computed once) ─────────────────────
        rwse      = _global_rwse(sf.edge_index, sf.ptr, N_total, self.rwse_steps, device)
        rwse_flat = self.rwse_proj(rwse)[node_ids.clamp(min=0)]  # [S*k, H]

        valid_f = valid.float().unsqueeze(-1)
        h_flat  = (x_flat + rwse_flat) * valid_f

        # ── Role-differentiated message passing ───────────────────────────────
        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid,
                node_ids, N_total, sf.edge_index, sf.edge_attr,
                sub_batch, S, k, root_flat_idx,
            )

        # ── Root-token readout ────────────────────────────────────────────────
        # Take the final root representation as the subgraph's token.
        # Root at flat position s*k always has node_ids[s*k] >= 0 (non-padding).
        h_tok    = h_flat[root_flat_idx]              # [S, H]
        h_tok_2d = h_tok.view(N_total, m, self.H)     # [N_total, m, H]

        # ── 2-layer HT-corrected Transformer ─────────────────────────────────
        attn_bias = _ht_attn_bias(lp, m, N_total, self.ht_alpha, self.num_heads)
        for layer in self.readout_layers:
            h_tok_2d = layer(h_tok_2d, attn_bias)

        # ── Per-node embedding → graph embedding ──────────────────────────────
        node_emb = self.readout_norm(h_tok_2d.mean(dim=1))   # [N_total, H]
        return global_add_pool(node_emb, sf.batch)            # [B, H]


# ---------------------------------------------------------------------------
# Node-level encoder
# ---------------------------------------------------------------------------

class Arch23NodeEncoder(nn.Module):

    def __init__(
        self,
        in_channels:    int,
        hidden_dim:     int,
        edge_dim:       int,
        num_layers:     int   = 6,
        mlp_layers:     int   = 2,
        rwse_steps:     int   = 16,
        dropout:        float = 0.0,
        num_heads:      int   = 4,
        readout_layers: int   = 2,
    ):
        super().__init__()
        H = hidden_dim
        self.H          = H
        self.rwse_steps = rwse_steps
        self.num_heads  = num_heads

        self.atom_encoder = nn.Embedding(in_channels, H)
        self.bond_encoder = nn.Embedding(edge_dim,    H)
        self.rwse_proj    = nn.Sequential(nn.Linear(rwse_steps, H), nn.ReLU())

        self.layers = nn.ModuleList([
            Arch9Layer(H, H, mlp_layers, dropout)
            for _ in range(num_layers)
        ])

        self.ht_alpha = nn.Parameter(torch.ones(1))

        assert H % num_heads == 0
        self.readout_layers = nn.ModuleList([
            HTTransformerLayer(H, num_heads, dropout)
            for _ in range(readout_layers)
        ])
        self.readout_norm = nn.LayerNorm(H)

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

        rwse      = _global_rwse(sf.edge_index, sf.ptr, N_total, self.rwse_steps, device)
        rwse_flat = self.rwse_proj(rwse)[node_ids.clamp(min=0)]
        valid_f   = valid.float().unsqueeze(-1)
        h_flat    = (x_flat + rwse_flat) * valid_f

        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid,
                node_ids, N_total, sf.edge_index, sf.edge_attr,
                sub_batch, S, k, root_flat_idx,
            )

        h_tok    = h_flat[root_flat_idx]
        h_tok_2d = h_tok.view(N_total, m, self.H)

        attn_bias = _ht_attn_bias(lp, m, N_total, self.ht_alpha, self.num_heads)
        for layer in self.readout_layers:
            h_tok_2d = layer(h_tok_2d, attn_bias)

        node_emb = self.readout_norm(h_tok_2d.mean(dim=1))
        return node_emb   # [N_total, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-23')
def build_arch23(cfg: ExperimentConfig):
    kw            = cfg.model_config.kwargs
    is_node_level = cfg.task in ('Node-Classification', 'Link-Prediction')

    common = dict(
        in_channels    = cfg.model_config.node_feature_dim,
        edge_dim       = cfg.model_config.edge_feature_dim,
        hidden_dim     = cfg.model_config.hidden_dim,
        num_layers     = cfg.model_config.mpnn_layers,
        mlp_layers     = kw.get('mlp_layers',     2),
        rwse_steps     = kw.get('rwse_steps',      16),
        dropout        = cfg.model_config.dropout,
        num_heads      = kw.get('num_heads',       4),
        readout_layers = kw.get('readout_layers',  2),
    )

    return Arch23NodeEncoder(**common) if is_node_level else Arch23GraphEncoder(**common)
