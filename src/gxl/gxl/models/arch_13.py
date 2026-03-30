"""
ARCH-13: Global GIN + Graphlet Token Transformer

Separates the two concerns that were conflated in ARCH-12:

  1. Node representation learning  — standard GIN on the full graph gives
     every node h_v that encodes its multi-hop neighbourhood.

  2. Structural motif reasoning    — graphlet tokens are formed by pooling
     these already-rich node reps over the sampled graphlet members, then
     a Transformer reasons about which combinations of motifs are predictive.

Pipeline
--------
  Stage 1 – Embed          atom/bond features → [N_total, H]
  Stage 2 – Full-graph GIN  L layers of GINEConv on sf.edge_index
                            → h_v [N_total, H]
  Stage 3 – Graphlet pool   z_j = mean({h_v : v ∈ N^S_j})  → [S, H]
  Stage 4 – Token           h_j = Linear(z_j) + LogProbMLP(log P_j)
  Stage 5 – Reshape         [S, H] → [B, m, H]
  Stage 6 – Transformer     pre-norm, Flash Attention → [B, m, H]
  Stage 7 – Readout         w_j = softmax_m(MLP(log P_j))
                            graph_emb = Σ_j  w_j · h_j^final → [B, H]

Why this design works
---------------------
* Each h_v from the deep GIN already "sees" the whole molecular graph via
  multi-hop message passing.  Pooling h_v over a graphlet's node set gives
  a token that encodes both internal topology AND global context — far
  richer than pooling a local graphlet GNN (ARCH-12).

* The Transformer enables cross-graphlet reasoning: it can learn that
  "aromatic ring next to carbonyl → specific logP contribution" by attending
  between the ring token and the carbonyl token.

* Importance-weighted readout (softmax over MLP(log P)) lets the model
  focus on structurally distinctive graphlets and down-weight redundant
  or low-information samples.

* The model needs NO subgraph edge indices (_flatten_subgraphs is not used);
  only sf.nodes_sampled and sf.log_probs are required from the sampler,
  keeping the code simple.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import scatter

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import register_model
from gxl.models.arch_7 import _make_conv


# ---------------------------------------------------------------------------
# Full-graph GIN  (stage 2)
# ---------------------------------------------------------------------------

class FullGraphGIN(nn.Module):
    """
    Standard GINEConv stack operating on the full batched graph.

    Each layer: h = BN(ReLU(conv(x, edge_index, edge_attr)));  x = x + h
    Residual connections give stable gradients with deeper stacks.
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
    """Maps scalar log_prob → H-dim vector (clamped to [-30, 0])."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        H = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, H),
            nn.ReLU(),
            nn.Linear(H, H),
        )

    def forward(self, log_probs: torch.Tensor) -> torch.Tensor:
        lp = log_probs.float().clamp(-30.0, 0.0).unsqueeze(-1)
        return self.mlp(lp)


# ---------------------------------------------------------------------------
# Graphlet Transformer  (stages 5-6)
# ---------------------------------------------------------------------------

class GraphletTransformerLayer(nn.Module):
    """
    Pre-norm Transformer over [B, m, H] graphlet tokens.
    Uses F.scaled_dot_product_attention (Flash Attention when available).
    2× FFN expansion to keep memory usage low.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, m, H = x.shape
        nh, hd  = self.num_heads, self.head_dim

        r       = self.norm1(x)
        q, k, v = self.qkv(r).chunk(3, dim=-1)
        q = q.view(B, m, nh, hd).transpose(1, 2)
        k = k.view(B, m, nh, hd).transpose(1, 2)
        v = v.view(B, m, nh, hd).transpose(1, 2)

        attn_drop = self.dropout if self.training else 0.0
        r = F.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)
        r = r.transpose(1, 2).reshape(B, m, H)
        x = x + self.out(r)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Importance-weighted readout  (stage 7)
# ---------------------------------------------------------------------------

class ImportanceReadout(nn.Module):
    """
      w_j = softmax_m( score_MLP(log P_j) )
      out  = Σ_j  w_j · h_j^final
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        H = hidden_dim
        self.score_mlp = nn.Sequential(
            nn.Linear(1, H // 2),
            nn.ReLU(),
            nn.Linear(H // 2, 1),
        )

    def forward(self, h: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        # h: [B, m, H];  log_probs: [B, m]
        lp = log_probs.float().clamp(-30.0, 0.0).unsqueeze(-1)  # [B, m, 1]
        w  = self.score_mlp(lp).softmax(dim=1)                   # [B, m, 1]
        return (w * h).sum(dim=1)                                 # [B, H]


# ---------------------------------------------------------------------------
# Full encoder
# ---------------------------------------------------------------------------

class Arch13GraphEncoder(nn.Module):

    def __init__(
        self,
        in_channels:        int,
        hidden_dim:         int,
        edge_dim:           int,
        gin_layers:         int   = 6,
        transformer_layers: int   = 4,
        num_heads:          int   = 4,
        mlp_layers:         int   = 2,
        dropout:            float = 0.0,
        conv_type:          str   = 'gine',
    ):
        super().__init__()
        H = hidden_dim
        self.hidden_dim = hidden_dim

        # Input encoders
        self.atom_encoder = nn.Embedding(in_channels, H)
        self.bond_encoder = nn.Embedding(edge_dim,    H)

        # Stage 2: full-graph GIN
        self.gin = FullGraphGIN(H, H, gin_layers, mlp_layers, conv_type, dropout)

        # Stage 4: token construction
        self.token_proj   = nn.Linear(H, H)
        self.logp_encoder = LogProbEncoder(H)

        # Stage 6: Transformer over graphlet tokens
        self.tf_layers = nn.ModuleList([
            GraphletTransformerLayer(H, num_heads, dropout)
            for _ in range(transformer_layers)
        ])

        # Stage 7: importance-weighted readout
        self.readout = ImportanceReadout(H)

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

        # ── Stage 2: full-graph GIN → node representations ───────────────────
        h_v = self.gin(sf.x, sf.edge_index, sf.edge_attr)   # [N_total, H]

        # ── Stage 3: pool node reps over graphlet members ────────────────────
        # sf.nodes_sampled: [S, k], global node indices (-1 = padding)
        nodes_t  = sf.nodes_sampled                          # [S, k]
        v_nodes  = nodes_t.clamp(min=0)                      # safe gather
        v_mask   = (nodes_t >= 0).float()                    # [S, k]

        gathered = h_v[v_nodes]                              # [S, k, H]
        gathered = gathered * v_mask.unsqueeze(-1)           # zero padding
        count    = v_mask.sum(dim=1, keepdim=True).clamp(min=1)
        z        = gathered.sum(dim=1) / count               # [S, H]

        S, k = nodes_t.shape
        B    = int(sf.batch.max().item()) + 1
        assert S % B == 0, (
            f"ARCH-13 requires S ({S}) % B ({B}) == 0. "
            "Ensure subgraph_param.m is fixed and the graphlet sampler is used."
        )
        m = S // B

        # ── Stage 4: token construction ───────────────────────────────────────
        log_probs = (
            sf.log_probs.float()
            if sf.log_probs is not None
            else torch.zeros(S, device=device)
        )
        tokens = self.token_proj(z) + self.logp_encoder(log_probs)   # [S, H]

        # ── Stages 5-6: reshape and Transformer ──────────────────────────────
        h    = tokens.view(B, m, -1)      # [B, m, H]
        lp3d = log_probs.view(B, m)       # [B, m]
        for layer in self.tf_layers:
            h = layer(h)

        # ── Stage 7: importance-weighted readout ──────────────────────────────
        return self.readout(h, lp3d)      # [B, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-13')
def build_arch13(cfg: ExperimentConfig):
    kw = cfg.model_config.kwargs
    if cfg.task in ('Node-Classification', 'Link-Prediction'):
        raise ValueError(
            "ARCH-13 is a graph-level model and does not support "
            "node-classification or link-prediction tasks."
        )
    return Arch13GraphEncoder(
        in_channels        = cfg.model_config.node_feature_dim,
        edge_dim           = cfg.model_config.edge_feature_dim,
        hidden_dim         = cfg.model_config.hidden_dim,
        gin_layers         = kw.get('gin_layers',         6),
        transformer_layers = kw.get('transformer_layers', 4),
        num_heads          = kw.get('num_heads',          4),
        mlp_layers         = kw.get('mlp_layers',         2),
        dropout            = cfg.model_config.dropout,
        conv_type          = cfg.model_config.mpnn_type,
    )
