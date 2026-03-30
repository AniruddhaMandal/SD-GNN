"""
ARCH-12: Graphlet Token Transformer

Each graph is represented by m sampled graphlets. Each graphlet is encoded
as a single Transformer token; a standard Transformer reasons over the m
tokens and an importance-weighted readout produces the graph embedding.

Pipeline
--------
  Stage 1 – Embedding       atom/bond features → [N_total, H];
                             (optional) LapPE     → [N_total, p]
  Stage 2 – Graphlet GNN    GINEConv × gnn_layers on intra-subgraph edges
                             with residual connections → [S*k, H]
  Stage 3 – Pool            mean-pool per graphlet (valid nodes only)
                             → z_j ∈ [S, H]
  Stage 4 – Token           pe_j = mean LapPE of member nodes   (if p > 0)
                             lp_j = LogProbMLP(log P_j)
                             h_j  = Linear(cat(z_j, pe_j)) + lp_j
                                  = Linear(z_j) + lp_j            (if p = 0)
  Stage 5 – Reshape         [S, H]  →  [B, m, H]
  Stage 6 – Transformer     pre-norm, Flash Attention, 2× FFN
                             → [B, m, H]
  Stage 7 – Readout         w_j = softmax_m( MLP(log P_j) )
                             graph_emb = Σ_j  w_j · h_j^final
                             → [B, H]

Notes
-----
* ARCH-12 is graph-level only (no NodeEncoder variant).
* Does NOT require _build_all_node_targets in experiment.py.
* Requires the graphlet sampler (sampler: "graphlet" in config) with a
  fixed m (subgraph_param.m) so that S = B × m always holds.
* LapPE is computed online per batch via scipy eigsh (fast for small
  molecules; disable with lap_pe_dim=0 for large graphs).
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
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

    Sign ambiguity handling:
      training  — random ±1 flip per eigenvector (data augmentation)
      inference — flip so the largest-magnitude element is positive

    Returns [N_total, pe_dim] (zeros for graphs with fewer than 2 nodes).
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
        s, e    = int(ptr[g].item()), int(ptr[g + 1].item())
        n_g     = e - s
        dim_g   = min(pe_dim, n_g - 1)
        if dim_g <= 0:
            continue

        # Remap edges to local node indices [0, n_g)
        mask  = (batch[edge_index[0]] == g)
        ei_g  = edge_index[:, mask] - s

        # Normalised Laplacian L_sym
        L_idx, L_val = get_laplacian(ei_g, normalization='sym', num_nodes=n_g)
        L_sp = to_scipy_sparse_matrix(L_idx, L_val, n_g).tocsc()

        try:
            k_req        = min(dim_g + 1, n_g)
            vals, vecs   = eigsh(L_sp, k=k_req, which='SM')
            order        = np.argsort(vals)
            vecs         = vecs[:, order[1: dim_g + 1]]   # skip trivial zero
        except Exception:
            continue

        vecs_t = torch.from_numpy(vecs.astype(np.float32)).to(device)

        if training:
            # Random sign flip for augmentation
            sign = torch.randint(0, 2, (1, vecs_t.shape[1]),
                                 device=device).float() * 2 - 1
        else:
            # Largest-magnitude element → positive
            amax = vecs_t.abs().argmax(dim=0)           # [dim_g]
            sign = vecs_t[amax, torch.arange(vecs_t.shape[1])].sign()
            sign = sign.unsqueeze(0).clamp(min=1)       # guard zeros

        pe[s:e, :dim_g] = vecs_t * sign

    return pe   # [N_total, pe_dim]


# ---------------------------------------------------------------------------
# Graphlet GNN  (stage 2)
# ---------------------------------------------------------------------------

class GraphletGNN(nn.Module):
    """
    GINEConv stack that runs over the flat [S*k, H] subgraph space.

    Each layer applies convolution → BN → ReLU → dropout, then adds a
    residual to the input.  This stable training dynamic outperforms plain
    GIN stacks in our experiments.
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
            x = x + h   # residual
        return x


# ---------------------------------------------------------------------------
# Log-probability encoder  (stage 4)
# ---------------------------------------------------------------------------

class LogProbEncoder(nn.Module):
    """
    Maps each graphlet's scalar sampling log-probability → H-dim embedding.

    The log-probs from the seed-expansion sampler are always ≤ 0 and can be
    very negative for rare graphlets.  We clamp to [-30, 0] for numerical
    stability — practically all useful variation sits in this range.
    """

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
        return self.mlp(lp)   # [S, H]


# ---------------------------------------------------------------------------
# Graphlet Transformer  (stage 6)
# ---------------------------------------------------------------------------

class GraphletTransformerLayer(nn.Module):
    """
    Pre-norm Transformer layer over [B, m, H] graphlet tokens.

    Uses F.scaled_dot_product_attention (Flash Attention when available) so
    the [B, heads, m, m] attention weight matrix is never stored in memory.
    FFN uses 2× expansion (not 4×) to halve intermediate activations.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        H = hidden_dim
        assert H % num_heads == 0, \
            f"hidden_dim ({H}) must be divisible by num_heads ({num_heads})"
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
        # x: [B, m, H]
        B, m, H = x.shape
        nh, hd  = self.num_heads, self.head_dim

        # Pre-norm self-attention
        r       = self.norm1(x)
        q, k, v = self.qkv(r).chunk(3, dim=-1)
        q = q.view(B, m, nh, hd).transpose(1, 2)   # [B, nh, m, hd]
        k = k.view(B, m, nh, hd).transpose(1, 2)
        v = v.view(B, m, nh, hd).transpose(1, 2)

        attn_drop = self.dropout if self.training else 0.0
        r = F.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)
        r = r.transpose(1, 2).reshape(B, m, H)
        x = x + self.out(r)

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Importance-weighted readout  (stage 7)
# ---------------------------------------------------------------------------

class ImportanceReadout(nn.Module):
    """
    Learns importance scores from each graphlet's log-probability, then
    produces a soft-weighted sum over the m output tokens.

      w_j = softmax_m( score_MLP(log P_j) )
      out = Σ_j  w_j · h_j^final
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
        lp = log_probs.float().clamp(-30.0, 0.0).unsqueeze(-1)   # [B, m, 1]
        w  = self.score_mlp(lp).softmax(dim=1)                    # [B, m, 1]
        return (w * h).sum(dim=1)                                  # [B, H]


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
            token_in_dim = 2 * H   # cat(z_j, pe_j)
        else:
            self.pe_lin  = None
            token_in_dim = H

        # Graphlet GNN (runs inside each graphlet)
        self.graphlet_gnn = GraphletGNN(H, H, gnn_layers, mlp_layers,
                                        conv_type, dropout)

        # Token construction
        self.token_proj   = nn.Linear(token_in_dim, H)
        self.logp_encoder = LogProbEncoder(H)

        # Transformer (over m tokens per graph)
        self.tf_layers = nn.ModuleList([
            GraphletTransformerLayer(H, num_heads, dropout)
            for _ in range(transformer_layers)
        ])

        # Readout
        self.readout = ImportanceReadout(H)

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        device = sf.x.device

        # ── Stage 1: encode atoms and bonds ─────────────────────────────────
        if not sf.x.is_floating_point():
            sf.x = self.atom_encoder(sf.x.long().squeeze(-1))
        # Robust edge-attr check: also encode when the tensor is 1D (which
        # happens in some PyG/ZINC versions where bond types are float scalars)
        # or when the last dimension doesn't match hidden_dim yet.
        if sf.edge_attr is not None and (
            not sf.edge_attr.is_floating_point()
            or sf.edge_attr.dim() < 2
            or sf.edge_attr.shape[-1] != self.hidden_dim
        ):
            sf.edge_attr = self.bond_encoder(sf.edge_attr.long().view(-1) - 1)

        # ── Stage 1b: (optional) LapPE on original graph ────────────────────
        if self.lap_pe_dim > 0:
            lap_pe = _compute_batch_lap_pe(
                sf.edge_index, sf.batch, self.lap_pe_dim, self.training
            )   # [N_total, lap_pe_dim]

        # ── Stage 2/3: flatten subgraphs, run Graphlet GNN, pool ────────────
        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        S, k = sf.nodes_sampled.shape
        B    = int(sf.batch.max().item()) + 1
        assert S % B == 0, (
            f"ARCH-12 requires S ({S}) % B ({B}) == 0. "
            "Ensure subgraph_param.m is fixed and the graphlet sampler is used."
        )
        m = S // B

        h_flat = self.graphlet_gnn(x_flat, intra_ei, ea_flat)  # [S*k, H]

        # Mean-pool valid positions per graphlet
        z = scatter(
            h_flat[valid], sub_batch[valid],
            dim=0, reduce='mean', dim_size=S,
        )   # [S, H]

        # ── Stage 4a: (optional) LapPE per graphlet ──────────────────────────
        if self.lap_pe_dim > 0:
            nodes_t     = sf.nodes_sampled                       # [S, k]
            v_nodes     = nodes_t.clamp(min=0)                   # safe gather
            v_mask      = (nodes_t >= 0).float()                 # [S, k]
            pe_gathered = lap_pe[v_nodes]                        # [S, k, p]
            pe_gathered = pe_gathered * v_mask.unsqueeze(-1)
            count       = v_mask.sum(1, keepdim=True).clamp(min=1)
            pe_j        = pe_gathered.sum(1) / count             # [S, p]
            pe_j        = self.pe_lin(pe_j)                      # [S, H]
            tokens      = self.token_proj(torch.cat([z, pe_j], dim=-1))
        else:
            tokens = self.token_proj(z)                          # [S, H]

        # ── Stage 4b: add log-prob embedding ─────────────────────────────────
        log_probs = (
            sf.log_probs.float()
            if sf.log_probs is not None
            else torch.zeros(S, device=device)
        )
        tokens = tokens + self.logp_encoder(log_probs)           # [S, H]

        # ── Stage 5: reshape to [B, m, H] ───────────────────────────────────
        h    = tokens.view(B, m, -1)                             # [B, m, H]
        lp3d = log_probs.view(B, m)                              # [B, m]

        # ── Stage 6: Transformer ─────────────────────────────────────────────
        for layer in self.tf_layers:
            h = layer(h)

        # ── Stage 7: importance-weighted readout ──────────────────────────────
        return self.readout(h, lp3d)                             # [B, H]


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
        num_heads          = kw.get('num_heads',          4),
        mlp_layers         = kw.get('mlp_layers',         2),
        dropout            = cfg.model_config.dropout,
        conv_type          = cfg.model_config.mpnn_type,
        lap_pe_dim         = kw.get('lap_pe_dim',         0),
    )
