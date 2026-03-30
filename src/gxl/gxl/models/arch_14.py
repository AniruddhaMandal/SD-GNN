"""
ARCH-14: Global GIN + Per-Node Subgraph View Attention

Combines the key insight from ARCH-13 (use full-graph GIN node reps to build
graphlet tokens) with the correct sampling mode from ARCH-10 (per-node
sampling, S = N_total × m).

Pipeline
--------
  Stage 1  – Embed          atom/bond features → sf.x [N_total, H]
  Stage 2  – Full-graph GIN  L GINEConv layers on all edges
                             → h_v [N_total, H]
  Stage 3  – Graphlet pool   For each subgraph s rooted at node v:
                               z_s = mean({h_u : u ∈ N^S_s})  → [S, H]
  Stage 4  – Token           token_s = Linear(z_s) + LogProbMLP(log P_s)
  Stage 5  – Reshape         [S, H] → [N_total, m, H]
  Stage 6  – View attention  T-layer Transformer over m tokens per node
                             with learnable HT correction in attention scores:
                               score(q, k) ∝ exp(Q·K/√d − α·log P_k)
                             → mean over m attended tokens → [N_total, H]
  Stage 7  – Readout         global_add_pool(node_emb, batch) → [B, H]

Why each component matters
--------------------------
* Per-node sampling (S = N_total × m):  every canonical node v gets m
  subgraphs rooted at it.  The final node embedding is computed from these
  m views, preserving the node-additive inductive bias required for ZINC.

* Full-graph GIN tokens:  h_u from a 6-layer GIN already encodes the whole
  molecular graph via multi-hop MP.  Pooling these over graphlet members
  gives z_s a token that carries both local motif structure AND global
  context — far richer than a local k-node GNN (ARCH-12).

* HT-corrected attention:  rare subgraphs (low p → large −α·log p boost)
  are up-weighted in the attention scores.  This is the approximate
  Horvitz-Thompson correction: it counteracts the under-sampling of
  structurally distinctive, low-probability graphlets.

* global_add_pool over nodes:  ZINC's target (penalised logP) is additive
  over atoms.  Summing canonical node embeddings preserves this structure,
  unlike summing graphlet embeddings (ARCH-12/13 mistake).

Requires _build_all_node_targets to be active (add 'ARCH-14' to the list
in experiment.py) so that S = N_total × m is guaranteed.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.norm import BatchNorm

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import register_model
from gxl.models.arch_7 import _make_conv


# ---------------------------------------------------------------------------
# Full-graph GIN  (stage 2)
# ---------------------------------------------------------------------------

class FullGraphGIN(nn.Module):
    """
    GINEConv stack on the full batched graph.
    Residual connection after every layer for stable deep training.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim:   int,
        num_layers: int   = 6,
        mlp_layers: int   = 2,
        conv_type:  str   = 'gine',
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.use_edge_attr = (conv_type == 'gine')
        self.dropout       = dropout
        self.convs = nn.ModuleList([
            _make_conv(conv_type, hidden_dim, hidden_dim, edge_dim, mlp_layers)
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers)])

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor = None,
    ) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            h = conv(x, edge_index, edge_attr) \
                if (self.use_edge_attr and edge_attr is not None) \
                else conv(x, edge_index)
            h = bn(F.relu(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h
        return x   # [N_total, H]


# ---------------------------------------------------------------------------
# Log-probability encoder  (stage 4)
# ---------------------------------------------------------------------------

class LogProbEncoder(nn.Module):
    """Scalar log P → H-dim embedding (clamped to [−30, 0])."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        H = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, H),
            nn.ReLU(),
            nn.Linear(H, H),
        )

    def forward(self, log_probs: torch.Tensor) -> torch.Tensor:
        return self.mlp(log_probs.float().clamp(-30.0, 0.0).unsqueeze(-1))


# ---------------------------------------------------------------------------
# Node-view Transformer  (stage 6)
# ---------------------------------------------------------------------------

class NodeViewTransformerLayer(nn.Module):
    """
    One pre-norm Transformer layer over [N, m, H] per-node view tokens.

    Keeps the m dimension intact so layers can be stacked.
    The HT correction is an additive key bias: −α · log P_k.
    This makes rare subgraphs (low probability, distinctive structure)
    dominate the attention for every query within the same canonical node.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        H = hidden_dim
        assert H % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = H // num_heads
        self.dropout   = dropout

        self.norm1 = nn.LayerNorm(H)
        self.qkv   = nn.Linear(H, 3 * H, bias=False)
        self.out   = nn.Linear(H, H)
        self.norm2 = nn.LayerNorm(H)
        self.ffn   = nn.Sequential(
            nn.Linear(H, 2 * H),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * H, H),
        )

    def forward(
        self,
        x:      torch.Tensor,
        ht_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x       : [N, m, H]
        ht_bias : [N, 1, 1, m]  additive key bias (−α · log P), or None
        Returns : [N, m, H]
        """
        N, m, H = x.shape
        nh, hd  = self.num_heads, self.head_dim

        r       = self.norm1(x)
        q, k, v = self.qkv(r).chunk(3, dim=-1)
        q = q.view(N, m, nh, hd).transpose(1, 2)   # [N, nh, m, hd]
        k = k.view(N, m, nh, hd).transpose(1, 2)
        v = v.view(N, m, nh, hd).transpose(1, 2)

        attn_drop = self.dropout if self.training else 0.0
        r = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=ht_bias,   # [N, 1, 1, m] broadcasts over heads & queries
            dropout_p=attn_drop,
        )
        r = r.transpose(1, 2).reshape(N, m, H)
        x = x + self.out(r)
        x = x + self.ffn(self.norm2(x))
        return x   # [N, m, H]


class NodeViewAttention(nn.Module):
    """
    Stack of Transformer layers over m subgraph views per canonical node.

    Each layer refines the m token representations with HT-corrected
    self-attention.  A final mean-pool over m produces the node embedding.

    ht_alpha is a shared learnable scalar (initialised to 1 = full HT
    correction).  Setting it to 0 recovers uniform mean pooling, so the
    model can degrade gracefully if the correction is not useful.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads:  int,
        num_layers: int   = 2,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.layers   = nn.ModuleList([
            NodeViewTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.ht_alpha = nn.Parameter(torch.ones(1))   # learnable HT scale

    def forward(
        self,
        x:       torch.Tensor,
        lp_2d:   torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x     : [N_total, m, H]
        lp_2d : [N_total, m]  log-probabilities per view (optional)
        Returns [N_total, H]
        """
        ht_bias = None
        if lp_2d is not None:
            # Key-only bias: [N, 1, 1, m] → broadcasts to [N, nh, m, m]
            ht_bias = (
                -self.ht_alpha
                * lp_2d.float().clamp(-30.0, 0.0)
                .unsqueeze(1).unsqueeze(2)          # [N, 1, 1, m]
            )

        for layer in self.layers:
            x = layer(x, ht_bias)   # [N, m, H]

        return x.mean(dim=1)        # [N_total, H]


# ---------------------------------------------------------------------------
# Full graph encoder
# ---------------------------------------------------------------------------

class Arch14GraphEncoder(nn.Module):

    def __init__(
        self,
        in_channels:  int,
        hidden_dim:   int,
        edge_dim:     int,
        gin_layers:   int   = 6,
        attn_layers:  int   = 2,
        num_heads:    int   = 4,
        mlp_layers:   int   = 2,
        dropout:      float = 0.0,
        conv_type:    str   = 'gine',
    ):
        super().__init__()
        H = hidden_dim
        self.hidden_dim = hidden_dim

        # Input encoders
        self.atom_encoder = nn.Embedding(in_channels, H)
        self.bond_encoder = nn.Embedding(edge_dim,    H)

        # Stage 2: global GIN on the full graph
        self.gin = FullGraphGIN(H, H, gin_layers, mlp_layers, conv_type, dropout)

        # Stage 4: token construction
        self.token_proj   = nn.Linear(H, H)
        self.logp_encoder = LogProbEncoder(H)

        # Stage 6: per-node view attention
        self.view_attn = NodeViewAttention(H, num_heads, attn_layers, dropout)

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        device = sf.x.device

        # ── Stage 1: encode atoms and bonds ──────────────────────────────────
        if not sf.x.is_floating_point():
            sf.x = self.atom_encoder(sf.x.long().squeeze(-1))
        if sf.edge_attr is not None and (
            not sf.edge_attr.is_floating_point()
            or sf.edge_attr.dim() < 2
            or sf.edge_attr.shape[-1] != self.hidden_dim
        ):
            sf.edge_attr = self.bond_encoder(sf.edge_attr.long().view(-1) - 1)

        # ── Stage 2: full-graph GIN ───────────────────────────────────────────
        h_v = self.gin(sf.x, sf.edge_index, sf.edge_attr)   # [N_total, H]

        N_total = h_v.shape[0]

        # ── Stage 3: pool GIN node reps over each subgraph's member nodes ─────
        # sf.nodes_sampled: [S, k], global indices (-1 = padding)
        nodes_t  = sf.nodes_sampled                          # [S, k]
        v_nodes  = nodes_t.clamp(min=0)                      # safe gather
        v_mask   = (nodes_t >= 0).float()                    # [S, k]

        gathered = h_v[v_nodes]                              # [S, k, H]
        gathered = gathered * v_mask.unsqueeze(-1)
        count    = v_mask.sum(dim=1, keepdim=True).clamp(min=1)
        z        = gathered.sum(dim=1) / count               # [S, H]

        S = nodes_t.shape[0]
        assert S % N_total == 0, (
            f"ARCH-14 requires S ({S}) % N_total ({N_total}) == 0. "
            "Add 'ARCH-14' to _build_all_node_targets in experiment.py."
        )
        m = S // N_total

        # ── Stage 4: token construction ───────────────────────────────────────
        log_probs = (
            sf.log_probs.float()
            if sf.log_probs is not None
            else torch.zeros(S, device=device)
        )
        tokens = self.token_proj(z) + self.logp_encoder(log_probs)   # [S, H]

        # ── Stage 5: reshape to per-node view matrix ──────────────────────────
        tokens_3d = tokens.view(N_total, m, -1)      # [N_total, m, H]
        lp_2d     = log_probs.view(N_total, m)        # [N_total, m]

        # ── Stage 6: per-node view attention → node embeddings ───────────────
        node_emb = self.view_attn(tokens_3d, lp_2d)  # [N_total, H]

        # ── Stage 7: sum-pool to graph level ─────────────────────────────────
        return global_add_pool(node_emb, sf.batch)    # [B, H]


# ---------------------------------------------------------------------------
# Node-level encoder (for node-classification / link-prediction)
# ---------------------------------------------------------------------------

class Arch14NodeEncoder(nn.Module):

    def __init__(
        self,
        in_channels:  int,
        hidden_dim:   int,
        edge_dim:     int,
        gin_layers:   int   = 6,
        attn_layers:  int   = 2,
        num_heads:    int   = 4,
        mlp_layers:   int   = 2,
        dropout:      float = 0.0,
        conv_type:    str   = 'gine',
    ):
        super().__init__()
        H = hidden_dim
        self.hidden_dim = hidden_dim
        self.atom_encoder = nn.Embedding(in_channels, H)
        self.bond_encoder = nn.Embedding(edge_dim,    H)
        self.gin          = FullGraphGIN(H, H, gin_layers, mlp_layers, conv_type, dropout)
        self.token_proj   = nn.Linear(H, H)
        self.logp_encoder = LogProbEncoder(H)
        self.view_attn    = NodeViewAttention(H, num_heads, attn_layers, dropout)

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        device = sf.x.device

        if not sf.x.is_floating_point():
            sf.x = self.atom_encoder(sf.x.long().squeeze(-1))
        if sf.edge_attr is not None and (
            not sf.edge_attr.is_floating_point()
            or sf.edge_attr.dim() < 2
            or sf.edge_attr.shape[-1] != self.hidden_dim
        ):
            sf.edge_attr = self.bond_encoder(sf.edge_attr.long().view(-1) - 1)

        h_v     = self.gin(sf.x, sf.edge_index, sf.edge_attr)
        N_total = h_v.shape[0]

        nodes_t  = sf.nodes_sampled
        v_nodes  = nodes_t.clamp(min=0)
        v_mask   = (nodes_t >= 0).float()
        gathered = h_v[v_nodes] * v_mask.unsqueeze(-1)
        z        = gathered.sum(1) / v_mask.sum(1, keepdim=True).clamp(min=1)

        S  = nodes_t.shape[0]
        m  = S // N_total
        log_probs = (
            sf.log_probs.float() if sf.log_probs is not None
            else torch.zeros(S, device=device)
        )
        tokens   = self.token_proj(z) + self.logp_encoder(log_probs)
        node_emb = self.view_attn(tokens.view(N_total, m, -1), log_probs.view(N_total, m))
        return node_emb   # [N_total, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-14')
def build_arch14(cfg: ExperimentConfig):
    kw            = cfg.model_config.kwargs
    is_node_level = cfg.task in ('Node-Classification', 'Link-Prediction')

    common = dict(
        in_channels = cfg.model_config.node_feature_dim,
        edge_dim    = cfg.model_config.edge_feature_dim,
        hidden_dim  = cfg.model_config.hidden_dim,
        gin_layers  = kw.get('gin_layers',  6),
        attn_layers = kw.get('attn_layers', 2),
        num_heads   = kw.get('num_heads',   4),
        mlp_layers  = kw.get('mlp_layers',  2),
        dropout     = cfg.model_config.dropout,
        conv_type   = cfg.model_config.mpnn_type,
    )
    return Arch14NodeEncoder(**common) if is_node_level else Arch14GraphEncoder(**common)
