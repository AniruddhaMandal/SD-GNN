"""
ARCH-21: Faithful GPM (Graph Pattern Machine) reproduction.

Reference: "Graph Pattern Machine" — random walks + sequential encoding +
           inter-pattern Transformer.

Architecture
------------
  Sampler     Random walks (rw_sampler), k nodes per walk (walk_length = k-1).
              m=16 train, m_eval=128 (randomly selected subsets at train time,
              all at eval — matches GPM pre-sample 128, use 16 strategy).

  Stage 1     atom_encoder + bond_encoder  →  embeddings in R^H
  Stage 2     Global RWSE (rwse_steps, default 8 to match GPM node_pe_dim=8)
              rwse_proj → R^H
  Stage 3     Gather per-position features:
                node_feat  [S, k, H] = atom_emb[nodes_sampled]
                rwse_feat  [S, k, H] = rwse_emb[nodes_sampled]
                bond_feat  [S, k, H] = [zeros | bond_emb[edge_src_global.view(S,k-1)]]
              input_proj(cat(node, rwse, bond)) → [S, k, H]
  Stage 4     Pattern PE (GPM self-adjacency):
                adj[s,i,j] = 1 if nodes_sampled[s,i] == nodes_sampled[s,j]
                pe_gru(adj)  [S, k, H]  → mean → pe_proj → [S, H]
                Added to walk embedding after intra-walk encoding (× pe_weight).
  Stage 5     Intra-walk encoder (configurable: 'gru' or 'transformer'):
                GRU:         GRU(H,H) on [S,k,H] → masked mean → [S,H]
                Transformer: L×pre-norm TfmLayer  → masked mean → [S,H]
              h_sub += pe * pe_weight
  Stage 6     Reshape [S,H] → [B, m, H]
  Stage 7     Inter-pattern Transformer (L layers, pre-norm, no bias)
              → [B, m, H]
  Stage 8     mean(dim=1) → [B, H]   (head is in ExperimentModel)

Key differences vs ARCH-19
---------------------------
  * Uses rw_sampler (ordered walks, revisits allowed) → correct self-adj PE
  * Bond features gathered sequentially (GPM-style) not scatter-mean
  * GRU intra-walk encoder option (preserves walk direction)
  * hidden_dim=256, rwse_steps=8 to match GPM defaults
  * NOT in _build_all_node_targets (graph-level, not per-node)
"""

import torch
from torch import nn
from torch.nn import functional as F

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import register_model
from gxl.models.arch_19 import _global_rwse
from gxl.models.arch_16 import SubgraphTransformerLayer


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch21GraphEncoder(nn.Module):

    def __init__(
        self,
        in_channels:  int,
        hidden_dim:   int,
        edge_dim:     int,
        k:            int   = 9,
        rwse_steps:   int   = 8,
        intra_layers: int   = 1,
        inter_layers: int   = 1,
        num_heads:    int   = 4,
        pattern_enc:  str   = 'transformer',
        pe_weight:    float = 1.0,
        dropout:      float = 0.1,
    ):
        super().__init__()
        H               = hidden_dim
        self.H          = H
        self.k          = k
        self.rwse_steps = rwse_steps
        self.pe_weight  = pe_weight
        self.pattern_enc = pattern_enc

        # ── Stage 1: atom + bond encoders ──────────────────────────────────
        self.atom_encoder = nn.Embedding(in_channels, H)
        self.bond_encoder = nn.Embedding(edge_dim,    H)

        # ── Stage 2: RWSE projection ────────────────────────────────────────
        self.rwse_proj = nn.Sequential(nn.Linear(rwse_steps, H), nn.ReLU())

        # ── Stage 3: input projection (node + rwse + bond → H) ─────────────
        self.input_proj = nn.Linear(3 * H, H)

        # ── Stage 4: Pattern PE via self-adjacency GRU ──────────────────────
        # adj is [S, k, k]; treat k×k row as input sequence of length k
        # each "token" is the k-dim binary row of the adjacency matrix
        self.pe_gru  = nn.GRU(input_size=k, hidden_size=H,
                               num_layers=1, batch_first=True)
        self.pe_proj = nn.Linear(H, H)

        # ── Stage 5: intra-walk encoder ─────────────────────────────────────
        if pattern_enc == 'gru':
            self.intra_gru  = nn.GRU(input_size=H, hidden_size=H,
                                      num_layers=intra_layers, batch_first=True)
            self.intra_proj = nn.Linear(H, H)
        elif pattern_enc == 'transformer':
            # Pre-norm Transformer layers operating on [S, k, H]
            # Reuse SubgraphTransformerLayer but it expects [N, m, H] → same shape
            self.intra_tfm = nn.ModuleList([
                SubgraphTransformerLayer(H, num_heads, dropout)
                for _ in range(intra_layers)
            ])
            self.intra_norm = nn.LayerNorm(H)
        else:
            raise ValueError(f"Unknown pattern_enc: {pattern_enc!r}. Use 'gru' or 'transformer'.")

        # ── Stage 7: inter-pattern Transformer ──────────────────────────────
        self.inter_tfm = nn.ModuleList([
            SubgraphTransformerLayer(H, num_heads, dropout)
            for _ in range(inter_layers)
        ])
        self.inter_norm = nn.LayerNorm(H)

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        # ── Stage 1 ────────────────────────────────────────────────────────
        x  = self.atom_encoder(sf.x.long().squeeze(-1))              # [N_total, H]
        ea = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)  # [E, H]

        device  = x.device
        N_total = x.shape[0]
        S, k    = sf.nodes_sampled.shape
        B       = int(sf.batch.max().item()) + 1
        m       = S // B

        # ── Stage 2 ────────────────────────────────────────────────────────
        rwse     = _global_rwse(sf.edge_index, sf.ptr, N_total, self.rwse_steps, device)
        rwse_emb = self.rwse_proj(rwse)                               # [N_total, H]

        # ── Stage 3: gather features ────────────────────────────────────────
        valid      = sf.nodes_sampled.to(device) >= 0                 # [S, k]
        nodes_safe = sf.nodes_sampled.to(device).clamp(min=0)         # [S, k]
        flat_nodes = nodes_safe.reshape(-1)                            # [S*k]

        x_seq  = x[flat_nodes].view(S, k, self.H)                     # [S, k, H]
        rw_seq = rwse_emb[flat_nodes].view(S, k, self.H)              # [S, k, H]

        # Sequential bond gather: walk_length = k-1 edges per walk
        walk_length = k - 1
        if walk_length > 0 and sf.edge_src_global is not None:
            esg = sf.edge_src_global.to(device)                        # [S*(k-1)]
            bond_gathered = ea[esg].view(S, walk_length, self.H)       # [S, k-1, H]
            bond_seq = torch.cat([
                torch.zeros(S, 1, self.H, device=device),
                bond_gathered
            ], dim=1)                                                   # [S, k, H]
        else:
            bond_seq = torch.zeros(S, k, self.H, device=device)

        valid_f = valid.float().unsqueeze(-1)                          # [S, k, 1]
        h = self.input_proj(torch.cat([x_seq, rw_seq, bond_seq], dim=-1)) * valid_f  # [S, k, H]

        # ── Stage 4: pattern PE (self-adjacency of walk node IDs) ───────────
        # adj[s, i, j] = 1 if nodes_safe[s, i] == nodes_safe[s, j]
        adj = (nodes_safe.unsqueeze(-1) == nodes_safe.unsqueeze(-2)).float()  # [S, k, k]
        pe_h, _ = self.pe_gru(adj)                                    # [S, k, H]
        # masked mean over valid positions
        pe_mean = (pe_h * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1.0)
        pe = self.pe_proj(pe_mean)                                     # [S, H]

        # ── Stage 5: intra-walk encoding ────────────────────────────────────
        if self.pattern_enc == 'gru':
            h_enc, _ = self.intra_gru(h)                              # [S, k, H]
            h_enc = h_enc * valid_f
            h_sub = self.intra_proj(
                h_enc.sum(dim=1) / valid_f.sum(dim=1).clamp(min=1.0)  # masked mean
            )                                                          # [S, H]
        else:  # transformer
            # SubgraphTransformerLayer expects [N, m, H] → here N=S, m=k
            for layer in self.intra_tfm:
                h = layer(h, ht_bias=None)                            # [S, k, H]
            h = self.intra_norm(h)
            h_sub = (h * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1.0)  # [S, H]

        h_sub = h_sub + pe * self.pe_weight                           # [S, H]

        # ── Stage 6: reshape to [B, m, H] ───────────────────────────────────
        h_3d = h_sub.view(B, m, self.H)

        # ── Stage 7: inter-pattern Transformer ──────────────────────────────
        for layer in self.inter_tfm:
            h_3d = layer(h_3d, ht_bias=None)
        h_3d = self.inter_norm(h_3d)

        # ── Stage 8: mean over patterns ─────────────────────────────────────
        return h_3d.mean(dim=1)                                        # [B, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-21')
def build_arch21(cfg: ExperimentConfig):
    sp = cfg.model_config.subgraph_param
    kw = cfg.model_config.kwargs

    return Arch21GraphEncoder(
        in_channels  = cfg.model_config.node_feature_dim,
        edge_dim     = cfg.model_config.edge_feature_dim,
        hidden_dim   = cfg.model_config.hidden_dim,
        k            = sp.k,
        rwse_steps   = kw.get('rwse_steps',      8),
        intra_layers = kw.get('intra_layers',    1),
        inter_layers = kw.get('inter_layers',    1),
        num_heads    = kw.get('num_heads',       4),
        pattern_enc  = kw.get('pattern_encoder', 'transformer'),
        pe_weight    = kw.get('pe_weight',       1.0),
        dropout      = cfg.model_config.dropout,
    )
