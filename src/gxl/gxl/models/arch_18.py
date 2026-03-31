"""
ARCH-18: ARCH-16 with RWSE replacing BFS-distance PE inside subgraphs.

The only architectural change from ARCH-16:
  Stage 2 PE:  BFS distance embedding  →  RWSE (Random Walk Structural Encoding)

Why RWSE beats BFS distance for molecular graphs
-------------------------------------------------
BFS distance from root gives every non-root node an integer 1..k.  Two nodes
at the same BFS depth are indistinguishable even if one is in a ring and the
other is on a chain — exactly the case for aromatic rings in molecules.

RWSE encodes the *p-step return probability* of a random walk starting at each
node:
    rw[v, t]  =  (A^t / D)[v, v]   for t = 1 … p

where A is the adjacency and D is the diagonal degree matrix.
This is a p-dimensional vector per node.  Ring nodes have non-zero values at
small t (walks easily return through the ring), chain-end nodes do not.
RWSE is therefore a provably stronger structural descriptor than BFS distance
for distinguishing local graph topology (Dwivedi et al., 2022).

Computation inside subgraphs
----------------------------
RWSE is computed independently for each subgraph on its intra-subgraph edges
(same edges used by local GINE).  There is no cross-subgraph leakage.

Implementation: matrix-power iteration on the normalised adjacency.
For small subgraphs (k ≤ 16) this is very cheap — far cheaper than BFS
distance propagation which requires k−1 scatter rounds.

Pipeline (same as ARCH-16 except Stage 2)
------------------------------------------
  Stage 1  – Embed         atom_encoder(x)  +  bond_encoder(edge_attr)
  Stage 2  – Init PE       h_flat = embed + rwse_proj(RWSE) + logp_pe(logP)
               RWSE: [S*k, p] → Linear(p, H) → [S*k, H]
               logP: same as ARCH-16
  Stage 3  – Local GINE    L × GINEConv on intra-subgraph edges
  Stage 4  – Root token    h_token[s] = h_flat[root_flat_idx[s]]
  Stage 5  – Reshape       [S, H] → [N_total, m, H]
  Stage 6  – Transformer   T × pre-norm (LN → MHA + HT bias → res → LN → FFN → res)
  Stage 7  – Mean-pool     [N_total, m, H] → [N_total, H]
  Stage 8  – Readout       global_add_pool  →  [B, H]

Requirements
------------
  - Per-node subgraph sampling, 'ARCH-18' in _build_all_node_targets.
  - graphlet sampler (sf.log_probs populated).
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.nn.norm import BatchNorm

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import register_model
from gxl.models.arch_2_v2 import make_mlp, _flatten_subgraphs
from gxl.models.arch_16 import LocalGINELayer, SubgraphTransformer


# ---------------------------------------------------------------------------
# RWSE inside subgraphs
# ---------------------------------------------------------------------------

def _rwse(intra_ei: torch.Tensor, S: int, k: int, steps: int) -> torch.Tensor:
    """
    Compute RWSE (p-step return probabilities) for each node in each subgraph.

    For each node v and walk length t:
        rw[v, t]  =  prob. that a t-step random walk starting from v returns to v

    Computed by iterating the row-normalised adjacency T = D^{-1} A:
        P_0  =  I                (one-hot; each node starts at itself)
        P_t  =  P_{t-1} · T     (one step of random walk)
        rw[v, t]  =  P_t[v, v]  (diagonal element = return probability)

    We keep the k×k matrix P per subgraph and read its diagonal at each step.

    Args:
        intra_ei : [2, E]   intra-subgraph edge index (flat node indices in [0, S*k))
        S        : int      number of subgraphs
        k        : int      nodes per subgraph (including padding)
        steps    : int      number of walk steps p

    Returns:
        rw  [S*k, steps]  float32, values in [0, 1].  Padding positions get 0.
    """
    device = intra_ei.device
    n      = S * k

    # Build per-subgraph row-normalised adjacency as a dense [S, k, k] matrix.
    # Each subgraph s has flat indices [s*k .. s*k+k-1].
    rw_out = torch.zeros(n, steps, device=device)

    if intra_ei.shape[1] == 0:
        return rw_out  # no edges → all zeros

    # Convert flat indices to (subgraph, local) pairs
    src_flat = intra_ei[0]    # [E]
    dst_flat = intra_ei[1]    # [E]
    sub_id   = src_flat // k  # [E]  subgraph index
    src_loc  = src_flat % k   # [E]  local row index in subgraph
    dst_loc  = dst_flat % k   # [E]  local col index in subgraph

    # Build dense adjacency [S, k, k]
    A = torch.zeros(S, k, k, device=device)
    A[sub_id, src_loc, dst_loc] = 1.0

    # Row-normalise: T = D^{-1} A  (rows sum to 1 for connected nodes)
    deg = A.sum(dim=-1, keepdim=True).clamp(min=1.0)   # [S, k, 1]
    T   = A / deg                                       # [S, k, k]

    # Iterate: P starts as identity, walk probability matrix
    P = torch.eye(k, device=device).unsqueeze(0).expand(S, -1, -1).clone()  # [S, k, k]

    for t in range(steps):
        P = torch.bmm(P, T)                             # [S, k, k]
        diag = P.diagonal(dim1=1, dim2=2)               # [S, k]
        rw_out[:, t] = diag.reshape(n)                  # [S*k, 1] stored in column t

    return rw_out   # [S*k, steps]


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch18GraphEncoder(nn.Module):

    def __init__(
        self,
        in_channels:        int,
        hidden_dim:         int,
        edge_dim:           int,
        gnn_layers:         int   = 4,
        transformer_layers: int   = 4,
        num_heads:          int   = 4,
        mlp_layers:         int   = 2,
        rwse_steps:         int   = 16,
        dropout:            float = 0.0,
    ):
        super().__init__()
        H = hidden_dim

        # Stage 1: embeddings
        self.atom_encoder = nn.Embedding(in_channels, H)
        self.bond_encoder = nn.Embedding(edge_dim,    H)

        # Stage 2: RWSE projection  (p → H)  +  logP projection
        self.rwse_proj = nn.Sequential(nn.Linear(rwse_steps, H), nn.ReLU())
        self.logp_proj = nn.Sequential(nn.Linear(1, H), nn.ReLU())
        self.rwse_steps = rwse_steps

        # Stage 3: local GINE layers
        self.gnn_layers = nn.ModuleList([
            LocalGINELayer(H, H, mlp_layers, dropout)
            for _ in range(gnn_layers)
        ])

        # Stage 6: subgraph Transformer (reused from ARCH-16)
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

        # ── Stage 2: RWSE + logP PE ───────────────────────────────────────────
        rw      = _rwse(intra_ei, S, k, self.rwse_steps)       # [S*k, p]
        rwse_pe = self.rwse_proj(rw)                            # [S*k, H]
        logp_pe = self.logp_proj(lp[sub_batch].unsqueeze(-1))  # [S*k, H]

        valid_f = valid.float().unsqueeze(-1)
        h_flat  = (x_flat + rwse_pe + logp_pe) * valid_f       # [S*k, H]

        # ── Stage 3: local GINE ───────────────────────────────────────────────
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

class Arch18NodeEncoder(nn.Module):

    def __init__(
        self,
        in_channels:        int,
        hidden_dim:         int,
        edge_dim:           int,
        gnn_layers:         int   = 4,
        transformer_layers: int   = 4,
        num_heads:          int   = 4,
        mlp_layers:         int   = 2,
        rwse_steps:         int   = 16,
        dropout:            float = 0.0,
    ):
        super().__init__()
        H = hidden_dim
        self.atom_encoder = nn.Embedding(in_channels, H)
        self.bond_encoder = nn.Embedding(edge_dim,    H)
        self.rwse_proj    = nn.Sequential(nn.Linear(rwse_steps, H), nn.ReLU())
        self.logp_proj    = nn.Sequential(nn.Linear(1, H), nn.ReLU())
        self.rwse_steps   = rwse_steps

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

        rw      = _rwse(intra_ei, S, k, self.rwse_steps)
        rwse_pe = self.rwse_proj(rw)
        logp_pe = self.logp_proj(lp[sub_batch].unsqueeze(-1))

        valid_f = valid.float().unsqueeze(-1)
        h_flat  = (x_flat + rwse_pe + logp_pe) * valid_f

        for layer in self.gnn_layers:
            h_flat = layer(h_flat, intra_ei, ea_flat, valid_f)

        h_tokens = h_flat[root_flat_idx]
        h_3d     = h_tokens.view(N_total, m, -1)
        lp_2d    = lp.view(N_total, m)

        return self.transformer(h_3d, lp_2d)   # [N_total, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-18')
def build_arch18(cfg: ExperimentConfig):
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
        rwse_steps         = kw.get('rwse_steps',         16),
        dropout            = cfg.model_config.dropout,
    )

    return Arch18NodeEncoder(**common) if is_node_level else Arch18GraphEncoder(**common)
