"""
ARCH-22: Faithful GPM (Graph Pattern Machine) reproduction for ZINC.

Fixes every architectural divergence found in ARCH-21.

Why ARCH-21 was stuck at 0.98
------------------------------
1. **Wrong inter-transformer semantics** (critical):
   ARCH-21 reshaped to [B, m, H] and did *within-graph* attention over m=16
   patterns.  GPM's actual design is the opposite: pattern features are shaped
   [h=m, n=B, H] and TransformerEncoderLayer(batch_first=True) treats h as
   *batch* and n=B as *sequence*, producing cross-graph attention.  Each of the
   16 pattern positions independently attends over all B=1024 graphs in the
   batch, giving the model cross-graph context.  mean(dim=0) → [B, H].

2. **Warmup 5× too slow**:
   ARCH-21 warmed up over 50 epochs.  GPM uses 100 warmup *steps* ≈ 10 epochs
   with batch_size=1024.  LR=0.01/50=0.0002 at epoch 1 effectively froze the
   model for the first ~50 epochs, creating the "stuck from the beginning" look.

3. **Batch size 16× too small**:
   GPM uses 1024; ARCH-21 used 64.  Smaller batches hurt both gradient quality
   and the cross-graph attention (fewer graphs → less context).

4. **No multiscale walk masking**:
   GPM's default multiscale=[2,4,6,8] applies progressive masks in the
   intra-walk encoder: patterns 0-3 see 3 positions, 4-7 see 5, 8-11 see 7,
   12-15 see all 9.  This creates a curriculum and forces the model to learn
   useful representations at multiple walk lengths.

5. **Intra-walk encoder with 4 heads instead of GPM's 1 head**:
   GPM: pattern_encoder_heads=1.

6. **768-dim → 256 input projection instead of GPM's 40 → 256**:
   GPM uses OGB-style 16-dim atom/bond encoders + 8-dim RWSE = 40 dims total.
   Concatenating 256+256+256=768 and projecting makes the model harder to train
   at the start.

GPM ZINC hyperparameters (config/main.yaml + utils/args.py)
------------------------------------------------------------
  hidden_dim=256   atom_emb_dim=16   bond_emb_dim=16   node_pe_dim=8
  k=9 (walk_length=8)   m=16 train / m_eval=128
  multiscale=[2,4,6,8]
  intra: TransformerEncoderLayer nhead=1  layers=1  batch_first=True  post-norm
  inter: TransformerEncoderLayer nhead=4  layers=1  batch_first=True  post-norm
         + shared outer LayerNorm + manual outer residual (GPM's transformer_encode)
  pe_encoder=gru   pe_weight=1.0   dropout=0.1   grad_clip=1.0
  optimizer=Adam   lr=0.01   warmup_steps=100   scheduler=warmup
  batch_size=1024  epochs=1000

Architecture stages
-------------------
  Stage 1  atom_encoder(21, 16) + bond_encoder(3, 16)
  Stage 2  _global_rwse → [N_total, 8]
  Stage 3  Gather per walk-position: cat(node[16], pe[8], bond[16]) → Linear(40, H)
  Stage 4  Pattern PE: self-adjacency GRU on [S, k, k] → [S, H]  (GPM _encode_pe)
  Stage 5  Multiscale masking [2,4,6,8] + intra-walk TransformerEncoder (nhead=1)
           → simple mean over k → [S, H]
  Stage 6  Add PE × pe_weight;  reshape to [m, B, H]  (GPM [h, n, H] layout)
  Stage 7  Inter-pattern transformer (GPM cross-batch):
             for layer in inter_layers:
                 last = x
                 x = layer(inter_norm(x))   # outer pre-norm + post-norm layer
                 x = last + x               # manual outer residual
           → mean(dim=0) → [B, H]
  Head     ClassifierHead in ExperimentModel (Linear(256,256) → ReLU → Linear(256,1))
           GPM uses a single Linear(256,1); the extra hidden layer is a minor difference.
"""

import copy
import torch
from torch import nn

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import register_model
from gxl.models.arch_19 import _global_rwse


class Arch22GraphEncoder(nn.Module):

    def __init__(
        self,
        atom_types:   int   = 21,
        bond_types:   int   = 3,
        hidden_dim:   int   = 256,
        node_pe_dim:  int   = 8,
        atom_emb_dim: int   = 16,
        bond_emb_dim: int   = 16,
        k:            int   = 9,
        multiscale:   list  = None,   # default [2, 4, 6, 8]
        intra_layers: int   = 1,
        inter_layers: int   = 1,
        intra_heads:  int   = 1,      # GPM: pattern_encoder_heads=1
        inter_heads:  int   = 4,      # GPM: heads=4
        pe_weight:    float = 1.0,
        dropout:      float = 0.1,
    ):
        super().__init__()
        if multiscale is None:
            multiscale = [2, 4, 6, 8]

        H = hidden_dim
        self.H          = H
        self.k          = k
        self.multiscale = multiscale
        self.pe_weight  = pe_weight
        self.node_pe_dim = node_pe_dim
        self.rwse_steps  = node_pe_dim   # 8 RWSE steps = GPM node_pe_dim=8

        # ── Stage 1: OGB-style 16-dim atom / bond encoders ──────────────────
        # GPM: AtomEncoder(emb_dim=16)  → effectively Embedding over unique atom types
        #      BondEncoder(emb_dim=16)  → effectively Embedding over unique bond types
        self.atom_encoder = nn.Embedding(atom_types, atom_emb_dim)
        self.bond_encoder = nn.Embedding(bond_types, bond_emb_dim)

        # ── Stage 3: input projection ────────────────────────────────────────
        # atom(16) + node_pe(8) + bond(16) = 40  →  H
        # GPM: self.pre_projection = nn.Linear(input_dim + edge_dim + node_pe_dim, hidden_dim)
        #                           = Linear(16 + 16 + 8, 256) = Linear(40, 256)
        input_dim = atom_emb_dim + node_pe_dim + bond_emb_dim  # 40
        self.input_proj = nn.Linear(input_dim, H)

        # ── Stage 4: pattern PE via self-adjacency GRU ───────────────────────
        # GPM: pe_rnn = GRU(input_size=k, hidden_size=H), pe_projection = Linear(H, H)
        # adj is [S, k, k]; GRU treats k steps each of size k (adjacency rows)
        self.pe_gru  = nn.GRU(input_size=k, hidden_size=H,
                               num_layers=1, batch_first=True)
        self.pe_proj = nn.Linear(H, H)

        # ── Stage 5: intra-walk encoder ──────────────────────────────────────
        # GPM: TransformerEncoder(nhead=1, dim_ffn=4H, batch_first=True, post-norm)
        intra_layer = nn.TransformerEncoderLayer(
            d_model        = H,
            nhead          = intra_heads,   # 1 (GPM default)
            dim_feedforward= 4 * H,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = False,         # post-norm (GPM default)
        )
        self.intra_encoder = nn.TransformerEncoder(
            intra_layer, num_layers=intra_layers
        )

        # ── Stage 7: inter-pattern transformer (GPM cross-batch) ─────────────
        # GPM:  self.norm = LayerNorm(hidden_dim)   ← shared outer norm
        #       self.encoder = ModuleList([deepcopy(encoder_layer) ...])
        #       transformer_encode:
        #           for layer in self.encoder:
        #               last_x = x
        #               x = layer(self.norm(x))
        #               x = last_x + x
        #
        # batch_first=True: batch-dim = m (pattern index), seq-dim = B (graphs).
        # → attention is *across graphs* for each pattern position.
        self.inter_norm = nn.LayerNorm(H)     # shared outer norm (GPM: self.norm)

        _inter_layer = nn.TransformerEncoderLayer(
            d_model        = H,
            nhead          = inter_heads,   # 4 (GPM default)
            dim_feedforward= 4 * H,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = False,         # post-norm (GPM default)
        )
        # GPM uses deepcopy per layer
        self.inter_layers_list = nn.ModuleList([
            copy.deepcopy(_inter_layer) for _ in range(inter_layers)
        ])

    # -------------------------------------------------------------------------
    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:

        # ── Stage 1 ──────────────────────────────────────────────────────────
        x  = self.atom_encoder(sf.x.long().squeeze(-1))              # [N, atom_emb_dim]
        ea = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)  # [E, bond_emb_dim]

        device  = x.device
        N_total = x.shape[0]
        S, k    = sf.nodes_sampled.shape          # S = B * m
        B       = int(sf.batch.max().item()) + 1
        m       = S // B

        # ── Stage 2: RWSE (same formula as T.AddRandomWalkPE) ────────────────
        rwse = _global_rwse(sf.edge_index, sf.ptr, N_total,
                            self.rwse_steps, device)                  # [N, 8]

        # ── Stage 3: gather per-walk-position features ────────────────────────
        nodes_safe = sf.nodes_sampled.to(device).clamp(min=0)         # [S, k]
        flat_nodes = nodes_safe.reshape(-1)                            # [S*k]

        # GPM: feat_gathered = feat[nids_flat]  →  atom embeddings per position
        node_feat = x[flat_nodes].view(S, k, x.shape[-1])             # [S, k, 16]

        # GPM: node_pe_gathered  →  RWSE per position
        pe_feat   = rwse[flat_nodes].view(S, k, self.node_pe_dim)     # [S, k, 8]

        # GPM: e_feat_gathered = [zeros | bond_emb[eids]] (sequential)
        walk_length = k - 1
        ed = ea.shape[-1]  # bond_emb_dim = 16
        if walk_length > 0 and sf.edge_src_global is not None:
            esg      = sf.edge_src_global.to(device)                  # [S*(k-1)]
            bond_seq = torch.cat([
                torch.zeros(S, 1, ed, device=device),
                ea[esg].view(S, walk_length, ed),
            ], dim=1)                                                  # [S, k, 16]
        else:
            bond_seq = torch.zeros(S, k, ed, device=device)

        # GPM: feat_gathered = cat([atom, pe, bond], dim=-1) → pre_projection
        # cat → [S, k, 40]  → Linear(40, H) → [S, k, H]
        h_input = self.input_proj(
            torch.cat([node_feat, pe_feat, bond_seq], dim=-1)
        )                                                              # [S, k, H]

        # ── Stage 4: pattern PE (GPM _encode_pe with pe_encoder='gru') ───────
        # GPM layout: [h=m, n=B, k] – remap from [S=B*m, k]
        # nodes_safe ordering: s = b*m + i  → view(B, m, k).T(0,1) → [m, B, k]
        patterns_gpm = nodes_safe.view(B, m, k).transpose(0, 1)       # [m, B, k]

        # GPM: adj = patterns.unsqueeze(-1) == patterns.unsqueeze(-2) → [h,n,k,k]
        adj      = (patterns_gpm.unsqueeze(-1) == patterns_gpm.unsqueeze(-2)).float()
        adj_flat = adj.reshape(S, k, k)                               # [S, k, k]

        # GPM: pe, _ = self.pe_rnn(adj);  pe = pe_projection(pe.mean(dim=1))
        pe_h, _  = self.pe_gru(adj_flat)                              # [S, k, H]
        pe_vec   = self.pe_proj(pe_h.mean(dim=1))                     # [S, H]
        pe_vec   = pe_vec.view(m, B, self.H)                          # [m, B, H]

        # ── Stage 5: multiscale masking + intra-walk encoding ─────────────────
        # Reorder h_input to GPM [h=m, n=B] layout so mask aligns with pattern groups
        h_gpm_flat = (
            h_input.view(B, m, k, self.H)
                   .transpose(0, 1)          # [m, B, k, H]
                   .reshape(S, k, self.H)    # [S=m*B, k, H]
        )

        # Build GPM multiscale mask:  True = valid position, False = padded
        # GPM:  mask[start:, :, :scale+1] = True  (cumulative over scales)
        if self.multiscale:
            mask = torch.zeros(m, B, k, device=device, dtype=torch.bool)
            for i, scale in enumerate(self.multiscale):
                start = int(i * m / len(self.multiscale))
                mask[start:, :, :scale + 1] = True
            # TransformerEncoderLayer src_key_padding_mask: True = IGNORE
            # GPM:  mask = ~mask  before passing to encoder
            padding_mask = (~mask).view(S, k)                         # [S, k]
        else:
            padding_mask = None

        # GPM: embeddings = self.encoder(feat_gathered, src_key_padding_mask=mask)
        #      embeddings = embeddings.mean(dim=1)   ← simple mean, not masked mean
        h_enc = self.intra_encoder(h_gpm_flat,
                                   src_key_padding_mask=padding_mask)  # [S, k, H]
        h_sub = h_enc.mean(dim=1)                                      # [S, H]

        # ── Stage 6: add PE, reshape to GPM [m, B, H] layout ─────────────────
        # GPM: pattern_feat = pattern_feat + pe * pe_weight
        pattern_feat = h_sub.view(m, B, self.H) + pe_vec * self.pe_weight  # [m, B, H]

        # ── Stage 7: inter-pattern transformer (GPM cross-batch attention) ────
        # pattern_feat: [m=16, B=batch_size, H]
        # TransformerEncoderLayer(batch_first=True):
        #   → treats dim-0 (m) as batch, dim-1 (B) as *sequence*
        #   → attention is across the B graphs for each of the m pattern positions
        #
        # GPM transformer_encode:
        #   for layer in self.encoder:
        #       last_x = x
        #       x = layer(self.norm(x))   # outer LayerNorm then post-norm layer
        #       x = last_x + x            # manual outer residual
        for layer in self.inter_layers_list:
            last = pattern_feat
            pattern_feat = layer(self.inter_norm(pattern_feat))        # [m, B, H]
            pattern_feat = last + pattern_feat

        # GPM: instance_emb = pattern_feat.mean(dim=0)  →  [n=B, H]
        return pattern_feat.mean(dim=0)                                # [B, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-22')
def build_arch22(cfg: ExperimentConfig):
    sp = cfg.model_config.subgraph_param
    kw = cfg.model_config.kwargs

    return Arch22GraphEncoder(
        atom_types   = cfg.model_config.node_feature_dim,
        bond_types   = cfg.model_config.edge_feature_dim,
        hidden_dim   = cfg.model_config.hidden_dim,
        node_pe_dim  = kw.get('node_pe_dim',   8),
        atom_emb_dim = kw.get('atom_emb_dim',  16),
        bond_emb_dim = kw.get('bond_emb_dim',  16),
        k            = sp.k,
        multiscale   = kw.get('multiscale',    [2, 4, 6, 8]),
        intra_layers = kw.get('intra_layers',  1),
        inter_layers = kw.get('inter_layers',  1),
        intra_heads  = kw.get('intra_heads',   1),
        inter_heads  = kw.get('inter_heads',   4),
        pe_weight    = kw.get('pe_weight',     1.0),
        dropout      = cfg.model_config.dropout,
    )
