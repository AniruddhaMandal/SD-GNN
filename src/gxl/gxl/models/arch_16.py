"""
ARCH-16: Local GINE + Root-Rep Token + T-layer Subgraph Transformer.

The SubgraphFormer-inspired clean design — no global/inter-root message
passing at all.  The Transformer itself handles all cross-subgraph reasoning.

Pipeline
--------
  Stage 1  – Embed         atom_encoder(x)  +  bond_encoder(edge_attr)
  Stage 2  – Init PE       h_flat = embed + dist_pe(bfs_dist) + logp_pe(logP)
  Stage 3  – Local GINE    L × GINEConv on intra-subgraph edges only
                           (residual + BN, no inter-root global MP)
  Stage 4  – Root token    h_token[s] = h_flat[root_flat_idx[s]]   [S, H]
  Stage 5  – Reshape       [S, H] → [N_total, m, H]
  Stage 6  – Transformer   T × pre-norm (LayerNorm → MHA → res →
                                         LayerNorm → FFN → res)
                           HT correction: additive key bias −α·logP[k]
  Stage 7  – Mean-pool     [N_total, m, H] → [N_total, H]
  Stage 8  – Readout       global_add_pool  →  [B, H]

Why this is new
---------------
Every previous arch with local GNN (ARCH-9/10/11/15) also runs inter-root
global GINEConv per layer — expensive and not needed if a deep Transformer
handles cross-subgraph context.  ARCH-16 removes that entirely:

  Local GINE layer:  O(S·k·E_sub)   — encodes neighbourhood topology
  Transformer:       O(N_total·m²)  — reasons over views per node

The Transformer is the sole source of cross-subgraph communication,
matching SubgraphFormer's design philosophy.

BFS distance PE keeps position-in-subgraph awareness (who is the root,
how far am I from it).  LogP PE injects sampling-probability signal.
Root-rep token gives the transformer a single clean per-subgraph vector
that, after L GINE layers, already aggregates the full k-hop neighbourhood.

Requirements
------------
  - Per-node subgraph sampling (_build_all_node_targets), S = N_total × m.
  - graphlet sampler (sf.log_probs must be populated).
  - Add 'ARCH-16' to the _build_all_node_targets list in experiment.py.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.nn.norm import BatchNorm

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import register_model
from gxl.models.arch_2_v2 import make_mlp, _flatten_subgraphs
from gxl.models.arch_8_b import _bfs_distances


# ---------------------------------------------------------------------------
# Stage 3: simple local GINE layer (no global MP)
# ---------------------------------------------------------------------------

class LocalGINELayer(nn.Module):
    """
    One GINEConv layer on intra-subgraph edges with BN + residual.
    No inter-root global message passing — deliberately kept cheap.
    """

    def __init__(self, hidden_dim: int, edge_dim: int,
                 mlp_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.conv    = GINEConv(
            make_mlp(hidden_dim, hidden_dim, hidden_dim, mlp_layers),
            train_eps=True, edge_dim=edge_dim,
        )
        self.bn      = BatchNorm(hidden_dim)
        self.dropout = dropout

    def forward(
        self,
        h:       torch.Tensor,   # [S*k, H]
        ei:      torch.Tensor,   # [2, E_sub]
        ea:      torch.Tensor,   # [E_sub, H]
        valid_f: torch.Tensor,   # [S*k, 1]
    ) -> torch.Tensor:
        h_new = self.bn(F.relu(self.conv(h, ei, ea)))
        h_new = F.dropout(h_new, p=self.dropout, training=self.training)
        return (h + h_new) * valid_f   # residual, zero-out padding


# ---------------------------------------------------------------------------
# Stage 6: T-layer pre-norm Transformer over m subgraph tokens per node
# ---------------------------------------------------------------------------

class SubgraphTransformerLayer(nn.Module):
    """
    One pre-norm Transformer layer over [N, m, H] subgraph-view tokens.

    Follows the standard LLM Transformer block:
        x ← x + Attention( LayerNorm(x), attn_bias )
        x ← x + FFN( LayerNorm(x) )

    The HT bias is a key-only additive mask: bias[q, k] = −α · logP[k],
    passed in as attn_mask to F.scaled_dot_product_attention.
    Flash Attention is used automatically when available (PyTorch ≥ 2.0).

    FFN uses 4× hidden expansion (standard for LLMs) with GELU.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        H              = hidden_dim
        self.num_heads = num_heads
        self.head_dim  = H // num_heads
        self.dropout   = dropout

        # Attention
        self.norm1 = nn.LayerNorm(H)
        self.qkv   = nn.Linear(H, 3 * H, bias=False)
        self.out   = nn.Linear(H, H)

        # FFN (4× expansion like standard Transformer / LLM)
        self.norm2 = nn.LayerNorm(H)
        self.ffn   = nn.Sequential(
            nn.Linear(H, 4 * H),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * H, H),
        )

    def forward(
        self,
        x:       torch.Tensor,        # [N, m, H]
        ht_bias: torch.Tensor = None, # [N, 1, m, m]  or  None
    ) -> torch.Tensor:
        N, m, H = x.shape
        nh, hd  = self.num_heads, self.head_dim

        # ── pre-norm self-attention ───────────────────────────────────────────
        r       = self.norm1(x)
        q, k, v = self.qkv(r).chunk(3, dim=-1)
        q = q.view(N, m, nh, hd).transpose(1, 2)   # [N, nh, m, hd]
        k = k.view(N, m, nh, hd).transpose(1, 2)
        v = v.view(N, m, nh, hd).transpose(1, 2)

        attn_drop = self.dropout if self.training else 0.0
        r = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=ht_bias,   # [N, 1, m, m] broadcasts over heads
            dropout_p=attn_drop,
        )                                           # [N, nh, m, hd]
        r = r.transpose(1, 2).reshape(N, m, H)
        x = x + self.out(r)

        # ── pre-norm FFN ──────────────────────────────────────────────────────
        x = x + self.ffn(self.norm2(x))
        return x   # [N, m, H]


class SubgraphTransformer(nn.Module):
    """
    Stack of T SubgraphTransformerLayers.
    Computes and caches the HT bias once, reuses across all layers.
    Final output is mean-pool over m → [N_total, H].
    """

    def __init__(
        self,
        hidden_dim:  int,
        num_heads:   int,
        num_layers:  int   = 4,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.layers   = nn.ModuleList([
            SubgraphTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.ht_alpha = nn.Parameter(torch.ones(1))   # learnable HT scale
        self.norm_out = nn.LayerNorm(hidden_dim)       # final norm before mean-pool

    def forward(
        self,
        x:    torch.Tensor,        # [N_total, m, H]
        lp:   torch.Tensor = None, # [N_total, m]  log-probs (optional)
    ) -> torch.Tensor:
        # Build key-only HT bias once: [N, 1, 1, m] → broadcasts to [N, 1, m, m]
        ht_bias = None
        if lp is not None:
            # Clamp to [-30, 0]; unsqueeze for [N, 1, 1, m] broadcast
            ht_bias = (
                -self.ht_alpha
                * lp.float().clamp(-30.0, 0.0)
                .unsqueeze(1).unsqueeze(2)   # [N, 1, 1, m]
            )

        for layer in self.layers:
            x = layer(x, ht_bias)

        x = self.norm_out(x)
        return x.mean(dim=1)   # [N_total, H]


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch16GraphEncoder(nn.Module):

    MAX_DIST = 32

    def __init__(
        self,
        in_channels:        int,
        hidden_dim:         int,
        edge_dim:           int,
        gnn_layers:         int   = 4,
        transformer_layers: int   = 4,
        num_heads:          int   = 4,
        mlp_layers:         int   = 2,
        dropout:            float = 0.0,
    ):
        super().__init__()
        H = hidden_dim

        # Embeddings
        self.atom_encoder = nn.Embedding(in_channels, H)
        self.bond_encoder = nn.Embedding(edge_dim,    H)
        self.dist_encoder = nn.Embedding(self.MAX_DIST + 1, H)
        self.logp_proj    = nn.Sequential(nn.Linear(1, H), nn.ReLU())

        # Stage 3: local GINE layers (no global MP)
        self.gnn_layers = nn.ModuleList([
            LocalGINELayer(H, H, mlp_layers, dropout)
            for _ in range(gnn_layers)
        ])

        # Stage 6: subgraph Transformer
        self.transformer = SubgraphTransformer(H, num_heads, transformer_layers, dropout)

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
            lp = sf.log_probs.float().clone()
            lp[~torch.isfinite(lp)] = 0.0
        else:
            lp = torch.zeros(S, device=device)

        # ── Stage 2: positional encodings ─────────────────────────────────────
        dist    = _bfs_distances(intra_ei, S, k).clamp(max=self.MAX_DIST)
        dist_pe = self.dist_encoder(dist)                      # [S*k, H]
        logp_pe = self.logp_proj(lp[sub_batch].unsqueeze(-1))  # [S*k, H]

        valid_f = valid.float().unsqueeze(-1)
        h_flat  = (x_flat + dist_pe + logp_pe) * valid_f      # [S*k, H]

        # ── Stage 3: local GINE (no global MP) ────────────────────────────────
        for layer in self.gnn_layers:
            h_flat = layer(h_flat, intra_ei, ea_flat, valid_f)

        # ── Stage 4-5: root token → [N_total, m, H] ───────────────────────────
        h_tokens = h_flat[root_flat_idx]           # [S, H]
        h_3d     = h_tokens.view(N_total, m, -1)   # [N_total, m, H]
        lp_2d    = lp.view(N_total, m)             # [N_total, m]

        # ── Stage 6-7: Transformer → mean-pool ────────────────────────────────
        node_emb = self.transformer(h_3d, lp_2d)   # [N_total, H]

        # ── Stage 8: graph readout ─────────────────────────────────────────────
        return global_add_pool(node_emb, sf.batch)  # [B, H]


# ---------------------------------------------------------------------------
# Node-level encoder
# ---------------------------------------------------------------------------

class Arch16NodeEncoder(nn.Module):

    MAX_DIST = 32

    def __init__(
        self,
        in_channels:        int,
        hidden_dim:         int,
        edge_dim:           int,
        gnn_layers:         int   = 4,
        transformer_layers: int   = 4,
        num_heads:          int   = 4,
        mlp_layers:         int   = 2,
        dropout:            float = 0.0,
    ):
        super().__init__()
        H = hidden_dim
        self.atom_encoder = nn.Embedding(in_channels, H)
        self.bond_encoder = nn.Embedding(edge_dim,    H)
        self.dist_encoder = nn.Embedding(self.MAX_DIST + 1, H)
        self.logp_proj    = nn.Sequential(nn.Linear(1, H), nn.ReLU())

        self.gnn_layers = nn.ModuleList([
            LocalGINELayer(H, H, mlp_layers, dropout)
            for _ in range(gnn_layers)
        ])
        self.transformer = SubgraphTransformer(H, num_heads, transformer_layers, dropout)

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
            lp = sf.log_probs.float().clone()
            lp[~torch.isfinite(lp)] = 0.0
        else:
            lp = torch.zeros(S, device=device)

        dist    = _bfs_distances(intra_ei, S, k).clamp(max=self.MAX_DIST)
        dist_pe = self.dist_encoder(dist)
        logp_pe = self.logp_proj(lp[sub_batch].unsqueeze(-1))

        valid_f = valid.float().unsqueeze(-1)
        h_flat  = (x_flat + dist_pe + logp_pe) * valid_f

        for layer in self.gnn_layers:
            h_flat = layer(h_flat, intra_ei, ea_flat, valid_f)

        h_tokens = h_flat[root_flat_idx]
        h_3d     = h_tokens.view(N_total, m, -1)
        lp_2d    = lp.view(N_total, m)

        return self.transformer(h_3d, lp_2d)   # [N_total, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-16')
def build_arch16(cfg: ExperimentConfig):
    kw            = cfg.model_config.kwargs
    is_node_level = cfg.task in ('Node-Classification', 'Link-Prediction')

    common = dict(
        in_channels        = cfg.model_config.node_feature_dim,
        edge_dim           = cfg.model_config.edge_feature_dim,
        hidden_dim         = cfg.model_config.hidden_dim,
        gnn_layers         = kw.get('gnn_layers',         4),
        transformer_layers = kw.get('transformer_layers', 4),
        num_heads          = kw.get('num_heads',          4),
        mlp_layers         = kw.get('mlp_layers',         2),
        dropout            = cfg.model_config.dropout,
    )

    return Arch16NodeEncoder(**common) if is_node_level else Arch16GraphEncoder(**common)
