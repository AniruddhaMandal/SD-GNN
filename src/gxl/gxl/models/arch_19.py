"""
ARCH-19: GPM-style architecture with graphlet sampler.

The closest reproduction of GPM (Graph Pattern Machine) within our framework.

GPM vs ARCH-19
--------------
GPM:
  - Random walks (nodes revisit → self-visit adjacency captures rings)
  - Sequential Transformer on walk nodes (no GNN)
  - Global RWSE node PE (computed on full graph, looked up per node)
  - Standard inter-pattern Transformer (no bias)
  - m=16 train / m=128 eval (random subset of 128 pre-sampled)

ARCH-19 (with graphlet sampler):
  - Graphlets: k-node induced subgraphs (no revisitation, so no self-visit PE)
  - Sequential Transformer on graphlet nodes (no GINEConv) ← key GPM idea
  - Global RWSE node PE (same as GPM, computed on full graph on-the-fly)
  - Intra-subgraph adjacency PE: k×k adj row → Linear → H (ring proxy for graphlets)
  - Bond features: scatter-mean of bond embeddings incident to each position
  - Standard inter-pattern Transformer (no HT/overlap bias) ← key GPM idea
  - m=32 train / m=128 eval (configured in subgraph_param)

Pipeline
--------
  Stage 1  – Encode      atom_encoder(x) → [N_total, H]
                         bond_encoder(edge_attr) → [E, H]
  Stage 2  – Global PE   RWSE on full batched graph → [N_total, rwse_steps]
                         projected: rwse_proj(RWSE) → [N_total, H]
  Stage 3  – Gather      For each subgraph s, gather k positions:
                         x_seq    = x[nodes_sampled[s,:]]         [S, k, H]
                         rw_seq   = rwse_proj[nodes_sampled[s,:]] [S, k, H]
                         bond_seq = scatter_mean(bond on intra-edges→dst) [S, k, H]
                         input_proj(cat[x_seq, rw_seq, bond_seq]) → [S, k, H]
  Stage 4  – Struct PE   Intra-subgraph k×k adjacency → adj_pe_proj [S, k, H]
             Pos PE      Learned position embedding 0..k-1 → [1, k, H]
  Stage 5  – Intra Tfm   L_intra × pre-norm Transformer on k-length sequence
                         mean-pool over valid positions → [S, H]
  Stage 6  – Reshape     [S, H] → [B, m, H]
  Stage 7  – Inter Tfm   L_inter × pre-norm standard Transformer (no bias)
                         mean-pool over m → [B, H]
  Stage 8  – Head        Linear → prediction

m_train vs m_eval
-----------------
Set `m_eval` in subgraph_param config to use a different (larger) m at eval:
    "subgraph_param": { "k": 10, "m": 32, "m_eval": 128 }
experiment.py reads m_eval when model.training is False.
The Transformer handles variable m natively (no positional PE over patterns).

Requirements
------------
  - Per-graph sampling (NOT in _build_all_node_targets).
  - graphlet sampler (sf.edge_src_global must be populated).
  - 'ARCH-19' NOT added to _build_all_node_targets in experiment.py.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter
from torch_geometric.nn import global_add_pool

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import register_model
from gxl.models.arch_2_v2 import make_mlp
from gxl.models.arch_16 import SubgraphTransformerLayer


# ---------------------------------------------------------------------------
# Global RWSE
# ---------------------------------------------------------------------------

@torch.no_grad()
def _global_rwse(edge_index: torch.Tensor,
                 ptr:        torch.Tensor,
                 N_total:    int,
                 steps:      int,
                 device:     torch.device) -> torch.Tensor:
    """
    Compute p-step return probabilities for every node in the batched graph.

    For node v and step t:  rw[v, t] = Prob(t-step walk from v returns to v)
                                      = (A D^{-1})^t [v, v]

    Computed independently per graph in the batch (molecules are small).

    Returns: [N_total, steps] float32
    """
    rwse    = torch.zeros(N_total, steps, device=device)
    B       = ptr.shape[0] - 1
    ei      = edge_index.to(device)
    ptr_dev = ptr.to(device)

    for g in range(B):
        start = ptr_dev[g].item()
        end   = ptr_dev[g + 1].item()
        n     = end - start
        if n == 0:
            continue

        mask     = (ei[0] >= start) & (ei[0] < end)
        local_ei = ei[:, mask] - start          # [2, E_g]

        A = torch.zeros(n, n, device=device)
        if local_ei.shape[1] > 0:
            A[local_ei[0], local_ei[1]] = 1.0

        deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)
        T   = A / deg                           # row-normalised [n, n]

        P = torch.eye(n, device=device)
        for t in range(steps):
            P = P @ T
            rwse[start:end, t] = P.diagonal()

    return rwse


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch19GraphEncoder(nn.Module):

    def __init__(
        self,
        in_channels:   int,
        hidden_dim:    int,
        edge_dim:      int,
        k_max:         int   = 10,
        rwse_steps:    int   = 16,
        intra_layers:  int   = 2,
        inter_layers:  int   = 2,
        num_heads:     int   = 4,
        dropout:       float = 0.0,
    ):
        super().__init__()
        H              = hidden_dim
        self.H         = H
        self.k_max     = k_max
        self.rwse_steps = rwse_steps

        # ── Stage 1: atom + bond encoders ──────────────────────────────────
        self.atom_encoder = nn.Embedding(in_channels, H)
        self.bond_encoder = nn.Embedding(edge_dim,    H)

        # ── Stage 2: global RWSE projection ────────────────────────────────
        self.rwse_proj = nn.Sequential(nn.Linear(rwse_steps, H), nn.ReLU())

        # ── Stage 3: input projection (atom + rwse + bond → H) ─────────────
        self.input_proj = nn.Linear(3 * H, H)

        # ── Stage 4: structural + position PE ──────────────────────────────
        # Adjacency row PE: for position i, adj[s, i, :] is a k-dim binary
        # vector showing which other positions are connected → project to H
        self.adj_pe_proj = nn.Linear(k_max, H, bias=False)
        # Learnable position embedding (0 = root, rest = BFS expansion order)
        self.pos_emb = nn.Embedding(k_max, H)

        # ── Stage 5: intra-subgraph Transformer (treats k nodes as sequence)
        self.intra_tfm = nn.ModuleList([
            SubgraphTransformerLayer(H, num_heads, dropout)
            for _ in range(intra_layers)
        ])
        self.intra_norm = nn.LayerNorm(H)

        # ── Stage 7: inter-pattern standard Transformer ─────────────────────
        self.inter_tfm = nn.ModuleList([
            SubgraphTransformerLayer(H, num_heads, dropout)
            for _ in range(inter_layers)
        ])
        self.inter_norm = nn.LayerNorm(H)

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        # ── Stage 1: encode ────────────────────────────────────────────────
        x  = self.atom_encoder(sf.x.long().squeeze(-1))                 # [N_total, H]
        ea = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)     # [E, H]

        device  = x.device
        N_total = x.shape[0]
        S, k    = sf.nodes_sampled.shape                                # k == k_max
        B       = int(sf.batch.max().item()) + 1
        m       = S // B

        # ── Stage 2: global RWSE ───────────────────────────────────────────
        rwse     = _global_rwse(sf.edge_index, sf.ptr, N_total, self.rwse_steps, device)
        rwse_emb = self.rwse_proj(rwse)                                 # [N_total, H]

        # ── Stage 3: gather per-position features ─────────────────────────
        valid        = sf.nodes_sampled >= 0                            # [S, k]
        nodes_safe   = sf.nodes_sampled.clamp(min=0)                   # [S, k]
        flat_nodes   = nodes_safe.view(-1)                              # [S*k]

        x_seq  = x[flat_nodes].view(S, k, self.H)                      # [S, k, H]
        rw_seq = rwse_emb[flat_nodes].view(S, k, self.H)               # [S, k, H]

        # Bond features: scatter-mean of bond embeddings to each flat position
        bond_seq = torch.zeros(S * k, self.H, device=device)
        ei_sub   = sf.edge_index_sampled.to(device)
        esg      = sf.edge_src_global.to(device)
        if ei_sub.shape[1] > 0:
            bond_feat = ea[esg]                                         # [E_sub, H]
            dst_flat  = ei_sub[1]                                       # [E_sub]
            bond_seq  = scatter(bond_feat, dst_flat, dim=0,
                                reduce='mean', dim_size=S * k)
        bond_seq = bond_seq.view(S, k, self.H)                         # [S, k, H]

        h = self.input_proj(torch.cat([x_seq, rw_seq, bond_seq], dim=-1))  # [S, k, H]

        # ── Stage 4: adjacency PE + position PE ───────────────────────────
        # Build k×k adjacency for each subgraph from intra-subgraph edges
        A = torch.zeros(S, k, k, device=device)
        if ei_sub.shape[1] > 0:
            sub_id  = ei_sub[0] // k
            src_loc = ei_sub[0] % k
            dst_loc = ei_sub[1] % k
            A[sub_id, src_loc, dst_loc] = 1.0                          # [S, k, k]

        adj_pe = self.adj_pe_proj(A)                                    # [S, k, H]
        pos_pe = self.pos_emb(torch.arange(k, device=device)).unsqueeze(0)  # [1, k, H]

        valid_f = valid.float().unsqueeze(-1)                           # [S, k, 1]
        h = (h + adj_pe + pos_pe) * valid_f                            # [S, k, H]

        # ── Stage 5: intra-subgraph Transformer ──────────────────────────
        for layer in self.intra_tfm:
            h = layer(h, ht_bias=None)
        h = self.intra_norm(h)

        # Masked mean pool → [S, H]
        h_sub = (h * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1.0)

        # ── Stage 6: reshape to [B, m, H] ────────────────────────────────
        h_3d = h_sub.view(B, m, self.H)                                # [B, m, H]

        # ── Stage 7: inter-pattern Transformer ───────────────────────────
        for layer in self.inter_tfm:
            h_3d = layer(h_3d, ht_bias=None)
        h_3d = self.inter_norm(h_3d)

        # ── Stage 8: readout ──────────────────────────────────────────────
        return h_3d.mean(dim=1)                                         # [B, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-19')
def build_arch19(cfg: ExperimentConfig):
    sp = cfg.model_config.subgraph_param
    kw = cfg.model_config.kwargs

    return Arch19GraphEncoder(
        in_channels  = cfg.model_config.node_feature_dim,
        edge_dim     = cfg.model_config.edge_feature_dim,
        hidden_dim   = cfg.model_config.hidden_dim,
        k_max        = sp.k,
        rwse_steps   = kw.get('rwse_steps',   16),
        intra_layers = kw.get('intra_layers',  2),
        inter_layers = kw.get('inter_layers',  2),
        num_heads    = kw.get('num_heads',     4),
        dropout      = cfg.model_config.dropout,
    )
