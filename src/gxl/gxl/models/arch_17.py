"""
ARCH-17: Per-Graph Graphlet Sampling + Local GINE + Relative-Bias Transformer.

The SubgraphFormer-inspired design with two key differences:
  1. Graphlet sampling (random k-node subgraphs) instead of full ego-graphs.
  2. Graphormer-style relative attention bias instead of GNN on Kronecker graph.

Pipeline
--------
  Stage 1  – Embed          atom_encoder(x) + bond_encoder(edge_attr)
  Stage 2  – Init PE        h_flat = embed + BFS_dist_pe + logP_pe      [S*k, H]
  Stage 3  – Local GINE     L × GINEConv on intra-subgraph edges only
                            (no inter-subgraph / global MP)
  Stage 4  – Root token     z_s = h_flat[ root_flat_idx[s] ]            [S,   H]
              After L GNN layers the root has aggregated its full k-hop
              neighbourhood — same design as ARCH-16.
  Stage 5  – Reshape        [S, H]  →  [B, m, H]
  Stage 6  – Relative bias  attn_bias[b, i, j]  =
                              overlap_emb( |V_i ∩ V_j| )   ← structural proximity
                              − α · logP[b, j]              ← HT key correction
              Bias shape [B, 1, m, m] broadcast over heads.
  Stage 7  – Transformer    T × pre-norm block (LN → MHA+bias → res → LN → FFN → res)
                            4× FFN expansion, Flash Attention.
  Stage 8  – Readout        HT-weighted mean over m tokens → [B, H]

Relative bias design
--------------------
overlap_emb  : nn.Embedding(k+1, 1)   maps |V_i ∩ V_j| ∈ {0,..,k} → scalar
               High overlap  ↔ subgraphs share many atoms  ↔ structurally close.
               This is the graphlet analogue of Graphormer's spatial encoding.

logP HT bias : −α · logP[j] (key-only, like ARCH-10/16).
               Rare subgraphs (low p, large −logP) dominate each query's attention.
               α is a shared learnable scalar, initialised to 1.

The two biases are additive, giving the Transformer a full picture:
  - WHICH subgraphs are structurally nearby (overlap)
  - WHICH subgraphs are information-rich rare events (HT)

Per-graph vs per-node sampling
-------------------------------
ARCH-17 uses per-GRAPH sampling: S = B × m (m subgraphs per graph).
This does NOT require _build_all_node_targets.
The readout goes directly to graph level, with no intermediate node embeddings.
The HT-weighted mean is the graph-level aggregation.

Root distance (future work)
---------------------------
SPD(root_i, root_j) in the original molecular graph would add a second structural
relative bias (Graphormer-style spatial encoding).  Not included here to avoid
CPU↔GPU BFS overhead per forward pass; overlap is already a strong proxy.

Requirements
------------
  - graphlet sampler (sf.log_probs populated, sf.nodes_sampled available).
  - subgraph_param.m set in config (fixed m per graph).
  - Do NOT add 'ARCH-17' to _build_all_node_targets in experiment.py.
"""

import torch
from torch import nn
from torch.nn import functional as F

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import register_model
from gxl.models.arch_2_v2 import _flatten_subgraphs
from gxl.models.arch_8_b import _bfs_distances
from gxl.models.arch_16 import LocalGINELayer     # reuse: GINE + BN + residual


# ---------------------------------------------------------------------------
# Overlap computation  (Stage 6, part 1)
# ---------------------------------------------------------------------------

def _compute_overlap(nodes_sampled: torch.Tensor, B: int, m: int, k: int) -> torch.Tensor:
    """
    Compute pairwise node-overlap between subgraphs within each graph.

    For subgraphs i and j of graph b:
        overlap[b, i, j]  =  |V_i ∩ V_j|   (count of shared valid nodes)

    Padding entries (node index == -1) are excluded.

    Algorithm: loop over k positions in subgraph i, check membership in j.
    Memory: O(B · m · m · k) per iteration — no large intermediate tensors.

    Args:
        nodes_sampled : [S=B*m, k]  global node indices (-1 = padding)
        B, m, k       : batch size, subgraphs per graph, nodes per subgraph

    Returns:
        [B, m, m]  long tensor, values in {0, …, k}
    """
    nodes = nodes_sampled.view(B, m, k)            # [B, m, k]
    valid = nodes >= 0                              # [B, m, k] bool

    overlap = torch.zeros(B, m, m, device=nodes.device, dtype=torch.long)

    for ki in range(k):
        node_ki  = nodes[:, :, ki]                 # [B, m]  node index at position ki
        valid_ki = valid[:, :, ki]                 # [B, m]  is position ki valid?

        # Does node_ki[b, i] appear anywhere in nodes[b, j, :]?
        # [B, m, 1] == [B, 1, m, k]  →  [B, m, m, k]  →  any(dim=-1)  →  [B, m, m]
        in_j = (node_ki.unsqueeze(2).unsqueeze(3)
                == nodes.unsqueeze(1)).any(dim=-1)  # [B, m, m]

        # Only count if position ki in subgraph i is a real node
        overlap += (in_j & valid_ki.unsqueeze(2)).long()

    return overlap   # [B, m, m]


# ---------------------------------------------------------------------------
# Transformer layer with relative additive bias  (Stage 7)
# ---------------------------------------------------------------------------

class RelativeBiasTransformerLayer(nn.Module):
    """
    Pre-norm Transformer layer (LLM-style) that accepts an additive attention bias.

    Bias is added to the raw QK scores before softmax — identical to how
    Graphormer adds spatial encoding and ALiBi adds positional bias in LLMs.

        attn_score[b, h, i, j]  +=  bias[b, 0, i, j]

    The single-head bias broadcasts over all attention heads.

    FFN uses 4× expansion (standard LLM ratio).
    Flash Attention via F.scaled_dot_product_attention (PyTorch ≥ 2.0).
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        H               = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = H // num_heads
        self.dropout    = dropout

        self.norm1 = nn.LayerNorm(H)
        self.qkv   = nn.Linear(H, 3 * H, bias=False)
        self.out   = nn.Linear(H, H)

        self.norm2 = nn.LayerNorm(H)
        self.ffn   = nn.Sequential(
            nn.Linear(H, 4 * H),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * H, H),
        )

    def forward(
        self,
        x:    torch.Tensor,         # [B, m, H]
        bias: torch.Tensor = None,  # [B, 1, m, m]  additive attn bias
    ) -> torch.Tensor:
        B, m, H = x.shape
        nh, hd  = self.num_heads, self.head_dim

        # ── pre-norm attention ────────────────────────────────────────────────
        r       = self.norm1(x)
        q, k, v = self.qkv(r).chunk(3, dim=-1)
        q = q.view(B, m, nh, hd).transpose(1, 2)   # [B, nh, m, hd]
        k = k.view(B, m, nh, hd).transpose(1, 2)
        v = v.view(B, m, nh, hd).transpose(1, 2)

        attn_drop = self.dropout if self.training else 0.0
        r = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=bias,    # [B, 1, m, m] → broadcasts over heads
            dropout_p=attn_drop,
        )                                           # [B, nh, m, hd]
        r = r.transpose(1, 2).reshape(B, m, H)
        x = x + self.out(r)

        # ── pre-norm FFN ──────────────────────────────────────────────────────
        x = x + self.ffn(self.norm2(x))
        return x   # [B, m, H]


class RelativeBiasTransformer(nn.Module):
    """
    Stack of T RelativeBiasTransformerLayers with shared relative bias.

    The bias is computed ONCE before the layer stack and reused across all T layers:
        bias[b, 0, i, j] = overlap_emb( |V_i ∩ V_j| ) − α · logP[b, j]

    Final output: HT-weighted mean over m tokens → [B, H].
    HT weights: softmax( −logP ) concentrates mass on rare/informative subgraphs.
    """

    def __init__(
        self,
        hidden_dim:  int,
        num_heads:   int,
        num_layers:  int   = 4,
        k_max:       int   = 10,   # max subgraph size (for overlap embedding)
        dropout:     float = 0.0,
    ):
        super().__init__()
        H = hidden_dim

        self.layers      = nn.ModuleList([
            RelativeBiasTransformerLayer(H, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm_out    = nn.LayerNorm(H)

        # Relative positional bias components
        self.overlap_emb = nn.Embedding(k_max + 1, 1)   # overlap count → scalar
        self.ht_alpha    = nn.Parameter(torch.ones(1))  # learnable HT scale

    def forward(
        self,
        x:            torch.Tensor,        # [B, m, H]
        overlap:      torch.Tensor,        # [B, m, m]  overlap counts (long)
        lp:           torch.Tensor = None, # [B, m]     log-probs (float)
    ) -> torch.Tensor:
        B, m, H = x.shape

        # ── build shared additive bias  [B, 1, m, m] ─────────────────────────
        # overlap_emb: [B, m, m, 1] → squeeze → [B, m, m]
        ov_bias = self.overlap_emb(overlap.clamp(max=self.overlap_emb.num_embeddings - 1))
        ov_bias = ov_bias.squeeze(-1)                   # [B, m, m]

        ht_bias = torch.zeros(B, m, m, device=x.device)
        if lp is not None:
            # Key-only HT bias: −α · logP[j] (broadcast over query dim)
            # [B, m] → [B, 1, m] → [B, m, m]
            ht_bias = -self.ht_alpha * lp.float().clamp(-30.0, 0.0).unsqueeze(1)
            ht_bias = ht_bias.expand(B, m, m)

        bias = (ov_bias + ht_bias).unsqueeze(1)         # [B, 1, m, m]

        # ── T Transformer layers ──────────────────────────────────────────────
        for layer in self.layers:
            x = layer(x, bias)

        x = self.norm_out(x)                            # [B, m, H]

        # ── HT-weighted mean readout ──────────────────────────────────────────
        # Rare subgraphs (low p, large −logP) get more weight.
        # When lp is None (no sampler) this degrades to uniform mean.
        if lp is not None:
            w = F.softmax(-lp.float().clamp(-30.0, 0.0), dim=1)  # [B, m]
            return (w.unsqueeze(-1) * x).sum(dim=1)               # [B, H]
        else:
            return x.mean(dim=1)                                   # [B, H]


# ---------------------------------------------------------------------------
# Full encoder
# ---------------------------------------------------------------------------

class Arch17GraphEncoder(nn.Module):

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
        k_max:              int   = 10,
        dropout:            float = 0.0,
    ):
        super().__init__()
        H = hidden_dim

        # Stage 1: embeddings
        self.atom_encoder = nn.Embedding(in_channels, H)
        self.bond_encoder = nn.Embedding(edge_dim,    H)

        # Stage 2: positional encodings
        self.dist_encoder = nn.Embedding(self.MAX_DIST + 1, H)
        self.logp_proj    = nn.Sequential(nn.Linear(1, H), nn.ReLU())

        # Stage 3: local GINE (no global MP)
        self.gnn_layers = nn.ModuleList([
            LocalGINELayer(H, H, mlp_layers, dropout)
            for _ in range(gnn_layers)
        ])

        # Stage 7: relative-bias Transformer
        self.transformer = RelativeBiasTransformer(
            H, num_heads, transformer_layers, k_max, dropout,
        )

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        sf.x         = self.atom_encoder(sf.x.long().squeeze(-1))
        sf.edge_attr = self.bond_encoder(sf.edge_attr.long().squeeze(-1) - 1)

        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        S, k   = sf.nodes_sampled.shape
        B      = int(sf.batch.max().item()) + 1
        m      = S // B
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
        dist_pe = self.dist_encoder(dist)                       # [S*k, H]
        logp_pe = self.logp_proj(lp[sub_batch].unsqueeze(-1))  # [S*k, H]

        valid_f = valid.float().unsqueeze(-1)
        h_flat  = (x_flat + dist_pe + logp_pe) * valid_f       # [S*k, H]

        # ── Stage 3: local GINE ───────────────────────────────────────────────
        for layer in self.gnn_layers:
            h_flat = layer(h_flat, intra_ei, ea_flat, valid_f)

        # ── Stage 4: root token ───────────────────────────────────────────────
        z     = h_flat[root_flat_idx]           # [S, H]

        # ── Stage 5: reshape to [B, m, H] ─────────────────────────────────────
        h_3d  = z.view(B, m, -1)               # [B, m, H]
        lp_2d = lp.view(B, m)                  # [B, m]

        # ── Stage 6: overlap bias ─────────────────────────────────────────────
        overlap = _compute_overlap(sf.nodes_sampled, B, m, k)   # [B, m, m]

        # ── Stages 7-8: Transformer + HT-weighted readout ────────────────────
        return self.transformer(h_3d, overlap, lp_2d)   # [B, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-17')
def build_arch17(cfg: ExperimentConfig):
    kw = cfg.model_config.kwargs

    if cfg.task in ('Node-Classification', 'Link-Prediction'):
        raise NotImplementedError("ARCH-17 is graph-level only.")

    k_max = cfg.model_config.subgraph_param.get('k', 10)

    return Arch17GraphEncoder(
        in_channels        = cfg.model_config.node_feature_dim,
        edge_dim           = cfg.model_config.edge_feature_dim,
        hidden_dim         = cfg.model_config.hidden_dim,
        gnn_layers         = kw.get('gnn_layers',         4),
        transformer_layers = kw.get('transformer_layers', 4),
        num_heads          = kw.get('num_heads',          4),
        mlp_layers         = kw.get('mlp_layers',         2),
        k_max              = k_max,
        dropout            = cfg.model_config.dropout,
    )
