"""
ARCH-10: Role-Differentiated Subgraph GNN with Horvitz-Thompson Attention Readout.

Identical to ARCH-9 in all message-passing layers.  The only change is in the
readout attention: the pre-softmax scores are biased by  -α · logP[k], turning
the standard MHA into an (approximate) Horvitz-Thompson-corrected estimator.

Horvitz-Thompson motivation
---------------------------
Subgraph s for root v is drawn with probability p_s = exp(logP_s).  A naive
mean over m samples under-represents rare neighbourhoods relative to their true
prevalence.  The HT estimator corrects this by weighting each sample by 1/p_s.

In the attention framework this maps to an additive key bias:

  attention[q, k]  ∝  exp( Q_q · K_k / sqrt(d)  −  α · logP[k] )
                    =  exp( Q_q · K_k / sqrt(d) ) / p[k]^α

With α = 1 this is the exact HT correction; the model can learn a different α.
Rare subgraphs (low p, very negative logP, large −logP boost) dominate the
softmax, making the readout focus on the atypical, information-rich graphlets.

α is a learnable scalar initialised to 1.0 (full HT from epoch 0).  Setting it
to 0 recovers standard uniform attention, so the model can fall back if HT hurts.

All other components (GINE layer, dist PE, logP node PE, inter-root MP) are
unchanged from ARCH-9; see arch_9.py for detailed commentary.

Requirements:
  - Per-node subgraph sampling (_build_all_node_targets), so S = N_total * m.
  - graphlet sampler (sf.log_probs must be populated).
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import scatter

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import register_model
from gxl.models.arch_2_v2 import make_mlp, _flatten_subgraphs
from gxl.models.arch_8_b import _bfs_distances
from gxl.models.arch_9 import Arch9Layer   # identical MP layer


# ---------------------------------------------------------------------------
# HT-corrected readout attention
# ---------------------------------------------------------------------------

def _ht_attn_bias(lp: torch.Tensor, m: int, N_total: int,
                  alpha: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Build the Horvitz-Thompson attention bias for nn.MultiheadAttention.

    The bias is key-only: bias[q, k] = -alpha * logP[k], so every query
    attends proportionally to 1/p[k]^alpha for each key k.

    Args:
        lp       [S]       sanitised log-probs (finite, 0 for degenerate)
        m        int       subgraphs per canonical node
        N_total  int       total canonical nodes in batch
        alpha    scalar    learnable HT scale (nn.Parameter)
        num_heads int      number of MHA heads

    Returns:
        [N_total * num_heads, m, m]  — attn_mask for nn.MultiheadAttention.
        PyTorch MHA (batch_first=True) expects this shape with batch-major,
        head-minor ordering: indices [n*num_heads + h, :, :] for batch n, head h.
    """
    logp_2d = lp.view(N_total, m)                              # [N_total, m]
    # key-only bias: broadcast across queries → [N_total, m, m]
    bias = (-alpha * logp_2d).unsqueeze(1).expand(N_total, m, m)
    # tile for each head: [N_total, num_heads, m, m] → [N_total*num_heads, m, m]
    bias = bias.unsqueeze(1).expand(N_total, num_heads, m, m)
    return bias.reshape(N_total * num_heads, m, m).contiguous()


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch10GraphEncoder(nn.Module):
    """
    Full ARCH-10 graph-level encoder.

    Identical to Arch9GraphEncoder except the readout MHA uses an
    HT-corrected attention bias:  scores += -α · logP[key].
    """

    MAX_DIST = 32

    def __init__(
        self,
        in_channels: int,
        hidden_dim:  int,
        edge_dim:    int,
        num_layers:  int   = 6,
        mlp_layers:  int   = 2,
        dropout:     float = 0.0,
        num_heads:   int   = 4,
    ):
        super().__init__()

        self.atom_encoder = nn.Embedding(in_channels, hidden_dim)
        self.bond_encoder = nn.Embedding(edge_dim,    hidden_dim)
        self.dist_encoder = nn.Embedding(self.MAX_DIST + 1, hidden_dim)
        self.logp_proj    = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())

        self.layers = nn.ModuleList([
            Arch9Layer(hidden_dim, hidden_dim, mlp_layers, dropout)
            for _ in range(num_layers)
        ])

        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        # α = 1: full HT correction from epoch 0; model can learn to adjust.
        self.ht_alpha  = nn.Parameter(torch.ones(1))
        self.readout_mha  = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.readout_norm = BatchNorm(hidden_dim)

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        sf.x         = self.atom_encoder(sf.x.long().squeeze(-1))
        sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)

        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        S, k   = sf.nodes_sampled.shape
        m      = S // N_total
        device = x_flat.device

        root_flat_idx = torch.arange(S, device=device) * k   # [S]

        # Sanitise log-probs once; reuse for node PE and readout bias.
        if sf.log_probs is not None:
            lp = sf.log_probs.clone()
            lp[~torch.isfinite(lp)] = 0.0
        else:
            lp = torch.zeros(S, device=device)

        # ── initialise flat representations ───────────────────────────────────
        dist    = _bfs_distances(intra_ei, S, k).clamp(max=self.MAX_DIST)
        dist_pe = self.dist_encoder(dist)                          # [S*k, H]
        logp_pe = self.logp_proj(lp[sub_batch].unsqueeze(-1))     # [S*k, H]

        valid_f = valid.float().unsqueeze(-1)
        h_flat  = (x_flat + dist_pe + logp_pe) * valid_f

        # ── message-passing ───────────────────────────────────────────────────
        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid,
                node_ids, N_total, sf.edge_index, sf.edge_attr,
                sub_batch, S, k, root_flat_idx,
            )

        # ── readout ───────────────────────────────────────────────────────────
        # Step 1: sum-pool each subgraph → [S, H]
        valid_mask = node_ids >= 0
        h_sub = scatter(
            h_flat[valid_mask], sub_batch[valid_mask],
            dim=0, reduce='sum', dim_size=S,
        )

        # Step 2: view as [N_total, m, H]
        # Subgraphs are already in m-contiguous order per target node
        # (graphlet_sampler.cpp Phase 1b), so direct view is correct.
        h_sub_2d = h_sub.view(N_total, m, h_sub.shape[-1])       # [N_total, m, H]

        # Step 3: HT-corrected self-attention.
        # attn_mask adds  -α · logP[k]  to every pre-softmax score for key k,
        # so attention[q,k] ∝ exp(Q·K/sqrt(d)) / p[k]^α.
        # Rare subgraphs (low p, large −logP) receive more attention weight.
        attn_mask = _ht_attn_bias(lp, m, N_total, self.ht_alpha, self.num_heads)
        h_attn, _ = self.readout_mha(
            h_sub_2d, h_sub_2d, h_sub_2d, attn_mask=attn_mask,
        )
        h_attn = h_attn + h_sub_2d                               # residual

        # Step 4: mean → BN → sum-pool per graph
        node_emb = self.readout_norm(h_attn.mean(dim=1))         # [N_total, H]
        return global_add_pool(node_emb, sf.batch)                # [B, H]


# ---------------------------------------------------------------------------
# Node-level encoder
# ---------------------------------------------------------------------------

class Arch10NodeEncoder(nn.Module):
    """Node-level variant: returns [N_total, H]."""

    MAX_DIST = 32

    def __init__(
        self,
        in_channels: int,
        hidden_dim:  int,
        edge_dim:    int,
        num_layers:  int   = 6,
        mlp_layers:  int   = 2,
        dropout:     float = 0.0,
        num_heads:   int   = 4,
    ):
        super().__init__()
        self.atom_encoder = nn.Embedding(in_channels, hidden_dim)
        self.bond_encoder = nn.Embedding(edge_dim,    hidden_dim)
        self.dist_encoder = nn.Embedding(self.MAX_DIST + 1, hidden_dim)
        self.logp_proj    = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())

        self.layers = nn.ModuleList([
            Arch9Layer(hidden_dim, hidden_dim, mlp_layers, dropout)
            for _ in range(num_layers)
        ])

        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.ht_alpha  = nn.Parameter(torch.ones(1))
        self.readout_mha  = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.readout_norm = BatchNorm(hidden_dim)

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

        dist    = _bfs_distances(intra_ei, S, k).clamp(max=self.MAX_DIST)
        dist_pe = self.dist_encoder(dist)
        logp_pe = self.logp_proj(lp[sub_batch].unsqueeze(-1))

        valid_f = valid.float().unsqueeze(-1)
        h_flat  = (x_flat + dist_pe + logp_pe) * valid_f

        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid,
                node_ids, N_total, sf.edge_index, sf.edge_attr,
                sub_batch, S, k, root_flat_idx,
            )

        valid_mask = node_ids >= 0
        h_sub = scatter(
            h_flat[valid_mask], sub_batch[valid_mask],
            dim=0, reduce='sum', dim_size=S,
        )

        h_sub_2d  = h_sub.view(N_total, m, h_sub.shape[-1])
        attn_mask = _ht_attn_bias(lp, m, N_total, self.ht_alpha, self.num_heads)
        h_attn, _ = self.readout_mha(
            h_sub_2d, h_sub_2d, h_sub_2d, attn_mask=attn_mask,
        )
        h_attn = h_attn + h_sub_2d

        return self.readout_norm(h_attn.mean(dim=1))              # [N_total, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-10')
def build_arch10(cfg: ExperimentConfig):
    kw            = cfg.model_config.kwargs
    is_node_level = cfg.task in ('Node-Classification', 'Link-Prediction')

    common = dict(
        in_channels = cfg.model_config.node_feature_dim,
        edge_dim    = cfg.model_config.edge_feature_dim,
        hidden_dim  = cfg.model_config.hidden_dim,
        num_layers  = cfg.model_config.mpnn_layers,
        mlp_layers  = kw.get('mlp_layers', 2),
        dropout     = cfg.model_config.dropout,
        num_heads   = kw.get('num_heads', 4),
    )

    if is_node_level:
        return Arch10NodeEncoder(**common)
    else:
        return Arch10GraphEncoder(**common)
