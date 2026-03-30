"""
ARCH-12: Graphlet Token Transformer with Node-Level Readout

Each graph → m graphlets (subgraphs of size k).  The architecture has two
phases:

Phase A — Graphlet encoding (graph-level reasoning)
  Stage 1 – Embed       atom/bond features → [N_total, H]
  Stage 2 – Graphlet GNN  GINEConv × gnn_layers on intra-subgraph edges
                          with residual connections → h_flat [S*k, H]
  Stage 3 – Pool         mean-pool per graphlet → z_j ∈ [S, H]
  Stage 4 – Token        h_j = Linear(z_j) + LogProbMLP(log P_j)
  Stage 5 – Reshape      [S, H] → [B, m, H]
  Stage 6 – Transformer  pre-norm, Flash Attention → [B, m, H]

Phase B — Node reconstruction (node-level readout, critical for performance)
  Stage 7 – Unpool       broadcast each graphlet's Transformer output back
                         to every flat position belonging to that graphlet
                         → h_ctx [S*k, H]
  Stage 8 – Combine      node_out = h_flat + h_ctx  (local + global context)
  Stage 9 – Scatter      scatter_mean to canonical nodes → [N_total, H]
  Stage 10 – Global GNN  optional: 1–2 GINE layers on original graph edges
  Stage 11 – Readout     global_add_pool → [B, H]

Why Phase B matters
-------------------
ZINC's target (penalised logP) decomposes additively over atoms, so
global_add_pool(node_embeddings) is the right inductive bias.  The earlier
ARCH-12 attempt read out from graphlet-level tokens (weighted sum of m
embeddings) which loses node identity and breaks the additive structure,
producing ~0.40 MAE.  Phase B restores the node-level readout while keeping
the cross-graphlet Transformer as a source of global context.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import scatter

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import register_model
from gxl.models.arch_2_v2 import _flatten_subgraphs
from gxl.models.arch_7 import _make_conv


# ---------------------------------------------------------------------------
# Laplacian PE  (optional — set lap_pe_dim=0 to disable)
# ---------------------------------------------------------------------------

def _compute_batch_lap_pe(
    edge_index: torch.Tensor,
    batch:      torch.Tensor,
    pe_dim:     int,
    training:   bool,
) -> torch.Tensor:
    """
    Online per-batch normalised-Laplacian eigenvector PE.
    Returns [N_total, pe_dim] (zeros for graphs with < 2 nodes).
    """
    from scipy.sparse.linalg import eigsh
    from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian

    device  = edge_index.device
    N_total = int(batch.shape[0])
    pe      = torch.zeros(N_total, pe_dim, device=device)

    B      = int(batch.max().item()) + 1
    counts = torch.bincount(batch, minlength=B)
    ptr    = torch.zeros(B + 1, dtype=torch.long, device=device)
    ptr[1:] = counts.cumsum(0)

    for g in range(B):
        s, e  = int(ptr[g].item()), int(ptr[g + 1].item())
        n_g   = e - s
        dim_g = min(pe_dim, n_g - 1)
        if dim_g <= 0:
            continue

        mask  = (batch[edge_index[0]] == g)
        ei_g  = edge_index[:, mask] - s

        L_idx, L_val = get_laplacian(ei_g, normalization='sym', num_nodes=n_g)
        L_sp = to_scipy_sparse_matrix(L_idx, L_val, n_g).tocsc()

        try:
            k_req      = min(dim_g + 1, n_g)
            vals, vecs = eigsh(L_sp, k=k_req, which='SM')
            order      = np.argsort(vals)
            vecs       = vecs[:, order[1: dim_g + 1]]
        except Exception:
            continue

        vecs_t = torch.from_numpy(vecs.astype(np.float32)).to(device)

        if training:
            sign = torch.randint(0, 2, (1, vecs_t.shape[1]),
                                 device=device).float() * 2 - 1
        else:
            amax = vecs_t.abs().argmax(dim=0)
            sign = vecs_t[amax, torch.arange(vecs_t.shape[1])].sign()
            sign = sign.unsqueeze(0).clamp(min=1)

        pe[s:e, :dim_g] = vecs_t * sign

    return pe


# ---------------------------------------------------------------------------
# Graphlet GNN  (phase A, stage 2)
# ---------------------------------------------------------------------------

class GraphletGNN(nn.Module):
    """
    GINEConv stack over the flat [S*k, H] subgraph space.
    Residual connections after each layer for training stability.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim:   int,
        num_layers: int   = 3,
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
        return x


# ---------------------------------------------------------------------------
# Log-probability encoder  (phase A, stage 4)
# ---------------------------------------------------------------------------

class LogProbEncoder(nn.Module):
    """Maps scalar log_prob → H-dim vector via 2-layer MLP."""

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
# Graphlet Transformer  (phase A, stages 5-6)
# ---------------------------------------------------------------------------

class GraphletTransformerLayer(nn.Module):
    """
    Pre-norm Transformer over [B, m, H] graphlet tokens.
    Flash Attention via F.scaled_dot_product_attention.
    2× FFN expansion to save memory.
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
# Global GNN  (phase B, stage 10 — optional)
# ---------------------------------------------------------------------------

class GlobalGNN(nn.Module):
    """
    1–2 GINE layers on the original (full) graph edge set.
    Runs on canonical node embeddings [N_total, H] after unpool,
    letting nodes that were never co-sampled still exchange information.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim:   int,
        num_layers: int   = 1,
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
        return x


# ---------------------------------------------------------------------------
# Full graph encoder
# ---------------------------------------------------------------------------

class Arch12GraphEncoder(nn.Module):

    def __init__(
        self,
        in_channels:        int,
        hidden_dim:         int,
        edge_dim:           int,
        gnn_layers:         int   = 3,
        transformer_layers: int   = 4,
        global_gnn_layers:  int   = 1,
        num_heads:          int   = 4,
        mlp_layers:         int   = 2,
        dropout:            float = 0.0,
        conv_type:          str   = 'gine',
        lap_pe_dim:         int   = 0,
    ):
        super().__init__()
        H = hidden_dim
        self.hidden_dim = hidden_dim
        self.lap_pe_dim = lap_pe_dim

        # Input encoders
        self.atom_encoder = nn.Embedding(in_channels, H)
        self.bond_encoder = nn.Embedding(edge_dim,    H)

        # Optional LapPE
        if lap_pe_dim > 0:
            self.pe_lin  = nn.Linear(lap_pe_dim, H, bias=False)
            token_in_dim = 2 * H
        else:
            self.pe_lin  = None
            token_in_dim = H

        # Phase A: Graphlet encoding
        self.graphlet_gnn = GraphletGNN(H, H, gnn_layers, mlp_layers,
                                        conv_type, dropout)
        self.token_proj   = nn.Linear(token_in_dim, H)
        self.logp_encoder = LogProbEncoder(H)
        self.tf_layers    = nn.ModuleList([
            GraphletTransformerLayer(H, num_heads, dropout)
            for _ in range(transformer_layers)
        ])

        # Phase B: Node reconstruction
        # Combine local (h_flat) and global context (h_ctx from transformer)
        self.node_proj = nn.Linear(2 * H, H)   # fuse h_flat + h_ctx
        # Optional global GNN over the full graph after unpool
        self.global_gnn = GlobalGNN(H, H, global_gnn_layers, mlp_layers,
                                    conv_type, dropout) \
                          if global_gnn_layers > 0 else None

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        device = sf.x.device

        # ── Stage 1: encode atoms and bonds ─────────────────────────────────
        if not sf.x.is_floating_point():
            sf.x = self.atom_encoder(sf.x.long().squeeze(-1))
        # Robust edge-attr check: handles 1-D float tensors (some PyG/ZINC
        # versions) as well as the usual [E, 1] long or [E, H] float cases.
        if sf.edge_attr is not None and (
            not sf.edge_attr.is_floating_point()
            or sf.edge_attr.dim() < 2
            or sf.edge_attr.shape[-1] != self.hidden_dim
        ):
            sf.edge_attr = self.bond_encoder(sf.edge_attr.long().view(-1) - 1)

        # ── Stage 1b: (optional) LapPE ───────────────────────────────────────
        if self.lap_pe_dim > 0:
            lap_pe = _compute_batch_lap_pe(
                sf.edge_index, sf.batch, self.lap_pe_dim, self.training
            )

        # ── Stage 2: flatten subgraphs ────────────────────────────────────────
        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        S, k = sf.nodes_sampled.shape
        B    = int(sf.batch.max().item()) + 1
        assert S % B == 0, (
            f"ARCH-12 requires S ({S}) % B ({B}) == 0. "
            "Ensure subgraph_param.m is fixed and the graphlet sampler is used."
        )
        m = S // B

        # ── Stage 2: Graphlet GNN ─────────────────────────────────────────────
        h_flat = self.graphlet_gnn(x_flat, intra_ei, ea_flat)  # [S*k, H]

        # ── Stage 3: Pool each graphlet ──────────────────────────────────────
        z = scatter(
            h_flat[valid], sub_batch[valid],
            dim=0, reduce='mean', dim_size=S,
        )   # [S, H]

        # ── Stage 4a: (optional) LapPE per graphlet ───────────────────────────
        if self.lap_pe_dim > 0:
            nodes_t     = sf.nodes_sampled
            v_nodes     = nodes_t.clamp(min=0)
            v_mask      = (nodes_t >= 0).float()
            pe_gathered = lap_pe[v_nodes] * v_mask.unsqueeze(-1)
            count       = v_mask.sum(1, keepdim=True).clamp(min=1)
            pe_j        = pe_gathered.sum(1) / count
            pe_j        = self.pe_lin(pe_j)
            tokens      = self.token_proj(torch.cat([z, pe_j], dim=-1))
        else:
            tokens = self.token_proj(z)   # [S, H]

        # ── Stage 4b: log-prob token ──────────────────────────────────────────
        log_probs = (
            sf.log_probs.float()
            if sf.log_probs is not None
            else torch.zeros(S, device=device)
        )
        tokens = tokens + self.logp_encoder(log_probs)   # [S, H]

        # ── Stage 5-6: Transformer over graphlet tokens ───────────────────────
        h = tokens.view(B, m, -1)          # [B, m, H]
        for layer in self.tf_layers:
            h = layer(h)

        # ── Stage 7-8: Unpool and combine with node-level features ───────────
        # Each graphlet's Transformer output is broadcast to every flat node
        # position in that graphlet, then fused with the local graphlet-GNN
        # output.  This gives every node access to cross-graphlet context
        # while preserving its individual representation.
        h_ctx   = h.view(S, -1)[sub_batch]          # [S*k, H]: context per position
        node_out = self.node_proj(
            torch.cat([h_flat, h_ctx], dim=-1)
        )   # [S*k, H]  (fuse local + global)

        # ── Stage 9: Scatter-mean to canonical nodes ──────────────────────────
        node_emb = scatter(
            node_out[valid], node_ids[valid],
            dim=0, reduce='mean', dim_size=N_total,
        )   # [N_total, H]

        # ── Stage 10: (optional) global GNN on full graph ────────────────────
        if self.global_gnn is not None:
            node_emb = node_emb + self.global_gnn(
                node_emb, sf.edge_index, sf.edge_attr
            )

        # ── Stage 11: Sum-pool to graph level ────────────────────────────────
        return global_add_pool(node_emb, sf.batch)   # [B, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-12')
def build_arch12(cfg: ExperimentConfig):
    kw = cfg.model_config.kwargs
    if cfg.task in ('Node-Classification', 'Link-Prediction'):
        raise ValueError(
            "ARCH-12 is a graph-level model and does not support "
            "node-classification or link-prediction tasks."
        )
    return Arch12GraphEncoder(
        in_channels        = cfg.model_config.node_feature_dim,
        edge_dim           = cfg.model_config.edge_feature_dim,
        hidden_dim         = cfg.model_config.hidden_dim,
        gnn_layers         = kw.get('gnn_layers',         3),
        transformer_layers = kw.get('transformer_layers', 4),
        global_gnn_layers  = kw.get('global_gnn_layers',  1),
        num_heads          = kw.get('num_heads',          4),
        mlp_layers         = kw.get('mlp_layers',         2),
        dropout            = cfg.model_config.dropout,
        conv_type          = cfg.model_config.mpnn_type,
        lap_pe_dim         = kw.get('lap_pe_dim',         0),
    )
