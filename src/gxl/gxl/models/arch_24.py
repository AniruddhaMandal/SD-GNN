"""
ARCH-24: Role-Differentiated Subgraph GNN with Configurable HT Correction.

Based on ARCH-9. Adds two orthogonal HT-correction flags for the ablation study
on *where* Horvitz-Thompson correction helps:

  use_ht_pool  (default False):
    Replaces the MHA subgraph readout with a self-normalised HT-weighted pool.
    Each subgraph's contribution to node v is weighted by softmax(-α · logP_s),
    amplifying rare subgraphs at the final aggregation step.

        ARCH-9:   h_v = mean( MHA(h_sub_2d) + h_sub_2d,  dim=m )
        HT-pool:  w   = softmax(-α · logP)     [N_total, m]
                  h_v = (w ⊙ h_sub_2d).sum(dim=1)

  use_ht_inter (default False):
    Replaces the uniform scatter_mean in inter-root aggregation with a
    HT-weighted scatter.  For each canonical node v, its representative
    across m subgraphs is a 1/p-weighted mean of its m root reps, pushing
    rare structural contexts into the cross-subgraph message passing.

        ARCH-9:   h_can[v] = mean_{s: root(s)=v}  h_root[s]
        HT-inter: w_s      = exp(-β · logP_s) / Σ_t exp(-β · logP_t)
                  h_can[v] = Σ_{s: root(s)=v}  w_s · h_root[s]

Ablation matrix (compare all four to ARCH-9 baseline):
  use_ht_pool=F, use_ht_inter=F  ←  equivalent to ARCH-9 (use as sanity check)
  use_ht_pool=T, use_ht_inter=F  ←  HT at readout only
  use_ht_pool=F, use_ht_inter=T  ←  HT at inter-root MP only
  use_ht_pool=T, use_ht_inter=T  ←  Full HT (both sites corrected)

Both α (pool) and β (inter) are learnable scalars initialised to 1.0.
Setting them to 0 recovers uniform weights, so the model can down-regulate
HT if it hurts on a given dataset.

Requirements:
  - Per-node subgraph sampling (_build_all_node_targets), S = N_total × m.
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
from gxl.models.arch_9 import Arch9Layer, _make_gine


# ---------------------------------------------------------------------------
# Layer: Arch9Layer + optional HT-weighted inter-root aggregation
# ---------------------------------------------------------------------------

class Arch24Layer(nn.Module):
    """
    Arch9Layer extended with optional HT-weighted inter-root aggregation.

    When `use_ht_inter=True` the encoder pre-computes per-subgraph HT weights
    and passes them as `ht_inter_w` to each forward call.  The layer then uses
    a weighted scatter instead of scatter_mean for the canonical root rep.

    When `ht_inter_w is None` (or use_ht_inter=False) the behaviour is
    identical to Arch9Layer.
    """

    def __init__(
        self,
        hidden_dim:     int,
        edge_dim:       int,
        mlp_layers:     int   = 2,
        dropout:        float = 0.0,
        use_inter_conv: bool  = True,
    ):
        super().__init__()
        self.dropout        = dropout
        self.use_inter_conv = use_inter_conv

        self.intra_conv = _make_gine(hidden_dim, edge_dim, mlp_layers)
        self.intra_bn   = BatchNorm(hidden_dim)

        self.self_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.root_proj  = nn.Linear(hidden_dim, hidden_dim)

        self.inter_conv = _make_gine(hidden_dim, edge_dim, mlp_layers)
        self.inter_bn   = BatchNorm(hidden_dim)

    def forward(
        self,
        h_flat:        torch.Tensor,            # [S*k, H]
        intra_ei:      torch.Tensor,            # [2, E_sub]
        ea_flat:       torch.Tensor,            # [E_sub, H]
        valid:         torch.Tensor,            # [S*k] bool
        node_ids:      torch.Tensor,            # [S*k] int, -1=padding
        N_total:       int,
        edge_index:    torch.Tensor,            # [2, E] original graph
        edge_attr:     torch.Tensor,            # [E, H]
        sub_batch:     torch.Tensor,            # [S*k] subgraph index
        S:             int,
        k:             int,
        root_flat_idx: torch.Tensor,            # [S]
        ht_inter_w:    torch.Tensor = None,     # [S_valid] pre-normalised HT weights
        root_valid:    torch.Tensor = None,     # [S] bool mask (passed with ht_inter_w)
    ) -> torch.Tensor:

        valid_f     = valid.float().unsqueeze(-1)
        clamped_ids = node_ids.clamp(min=0)

        is_root   = torch.zeros(S * k, dtype=torch.bool, device=h_flat.device)
        is_root[root_flat_idx] = True
        is_root_f = is_root.float().unsqueeze(-1)

        # ── intra-subgraph GINE ───────────────────────────────────────────────
        if ea_flat is None:
            ea_flat = torch.zeros(intra_ei.shape[1], h_flat.shape[-1], device=h_flat.device)
        h1 = self.intra_conv(h_flat, intra_ei, ea_flat)
        h1 = self.intra_bn(h1) * valid_f

        # ── non-root branch ───────────────────────────────────────────────────
        h_root_bcast = h_flat[sub_batch * k]
        h_non_root   = self.self_proj(h_flat) + self.root_proj(h_root_bcast)

        # ── inter-root branch ─────────────────────────────────────────────────
        if self.use_inter_conv:
            h_roots  = h_flat[root_flat_idx]        # [S, H]
            root_ids = node_ids[root_flat_idx]       # [S]
            rv       = root_ids >= 0                 # [S] bool (may differ from passed root_valid)

            if ht_inter_w is not None:
                # HT-weighted scatter: pre-normalised weights [S_valid]
                h_root_canonical = scatter(
                    h_roots[rv] * ht_inter_w.unsqueeze(-1),
                    root_ids[rv],
                    dim=0, reduce='sum', dim_size=N_total,
                )
            else:
                h_root_canonical = scatter(
                    h_roots[rv], root_ids[rv],
                    dim=0, reduce='mean', dim_size=N_total,
                )

            if edge_attr is None:
                edge_attr = torch.zeros(edge_index.shape[1], h_flat.shape[-1], device=h_flat.device)
            h_inter       = self.inter_conv(h_root_canonical, edge_index, edge_attr)
            h_inter       = self.inter_bn(h_inter)
            h_inter_bcast = h_inter[clamped_ids] * valid_f
        else:
            h_inter_bcast = torch.zeros_like(h1)

        # ── role-differentiated combine ───────────────────────────────────────
        out = is_root_f * (h1 + h_inter_bcast) + (1.0 - is_root_f) * (h1 + h_non_root)
        out = F.relu(out) * valid_f
        out = F.dropout(out, p=self.dropout, training=self.training) * valid_f
        return out


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch24GraphEncoder(nn.Module):
    """
    Full ARCH-24 graph-level encoder.

    Base: Arch9 pipeline (role-differentiated MP + MHA readout).
    Extensions: HT-weighted pooling and/or HT-weighted inter-root aggregation.
    """

    MAX_DIST = 32

    def __init__(
        self,
        in_channels:    int,
        hidden_dim:     int,
        edge_dim:       int,
        num_layers:     int   = 6,
        mlp_layers:     int   = 2,
        dropout:        float = 0.0,
        num_heads:      int   = 4,
        use_inter_conv: bool  = True,
        use_ht_pool:    bool  = False,
        use_ht_inter:   bool  = False,
        use_bfs_pe:     bool  = True,
        use_logp_pe:    bool  = True,
    ):
        super().__init__()
        self.use_ht_pool  = use_ht_pool
        self.use_ht_inter = use_ht_inter
        self.use_bfs_pe   = use_bfs_pe
        self.use_logp_pe  = use_logp_pe
        self.hidden_dim   = hidden_dim

        self.atom_encoder = nn.Embedding(in_channels, hidden_dim)
        self.bond_encoder = nn.Embedding(edge_dim,    hidden_dim)
        # Linear projections for datasets with multi-dim continuous/categorical features
        # (e.g. Peptides-func/struct which have [N,9] node and [E,3] edge features)
        self.node_proj = nn.Linear(in_channels, hidden_dim)
        self.bond_proj = nn.Linear(edge_dim,    hidden_dim)
        self.dist_encoder = nn.Embedding(self.MAX_DIST + 1, hidden_dim)
        self.logp_proj    = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())

        self.layers = nn.ModuleList([
            Arch24Layer(hidden_dim, hidden_dim, mlp_layers, dropout,
                        use_inter_conv=use_inter_conv)
            for _ in range(num_layers)
        ])

        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads

        # MHA readout (used when use_ht_pool=False)
        self.readout_mha  = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.readout_norm = BatchNorm(hidden_dim)

        # HT correction scalars — both initialised to 1.0 (full HT from epoch 0)
        # α: pool readout weight   β: inter-root aggregation weight
        self.ht_alpha_pool  = nn.Parameter(torch.ones(1))
        self.ht_alpha_inter = nn.Parameter(torch.ones(1))

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        # ── embed ─────────────────────────────────────────────────────────────
        # Defensive encoding: skip if already [*, H] float (already embedded, e.g. molhiv)
        if (not sf.x.is_floating_point()
                or sf.x.dim() < 2
                or sf.x.shape[-1] != self.hidden_dim):
            if sf.x.dim() <= 1 or sf.x.shape[-1] == 1:
                # Scalar integer per node (ZINC: [N,1] or [N]) → embedding lookup
                sf.x = self.atom_encoder(sf.x.long().view(-1))
            else:
                # Multi-dim features (Peptides: [N,9], PROTEINS: [N,3]) → linear proj
                sf.x = self.node_proj(sf.x.float())
        if sf.edge_attr is not None and (
            not sf.edge_attr.is_floating_point()
            or sf.edge_attr.dim() < 2
            or sf.edge_attr.shape[-1] != self.hidden_dim
        ):
            if sf.edge_attr.dim() <= 1 or sf.edge_attr.shape[-1] == 1:
                # Scalar integer per edge (ZINC: [E]) → embedding lookup (1-indexed)
                sf.edge_attr = self.bond_encoder(sf.edge_attr.long().view(-1) - 1)
            else:
                # Multi-dim edge features (Peptides: [E,3]) → linear proj
                sf.edge_attr = self.bond_proj(sf.edge_attr.float())

        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        S, k   = sf.nodes_sampled.shape
        m      = S // N_total
        device = x_flat.device

        root_flat_idx = torch.arange(S, device=device) * k   # [S]

        # ── sanitise log-probs ────────────────────────────────────────────────
        if sf.log_probs is not None:
            lp = sf.log_probs.clone()
            lp[~torch.isfinite(lp)] = 0.0
        else:
            lp = torch.zeros(S, device=device)

        # ── positional encodings ──────────────────────────────────────────────
        dist    = _bfs_distances(intra_ei, S, k).clamp(max=self.MAX_DIST)
        dist_pe = self.dist_encoder(dist) if self.use_bfs_pe else torch.zeros_like(x_flat)
        logp_pe = self.logp_proj(lp[sub_batch].unsqueeze(-1)) if self.use_logp_pe else torch.zeros_like(x_flat)

        valid_f = valid.float().unsqueeze(-1)
        h_flat  = (x_flat + dist_pe + logp_pe) * valid_f

        # ── pre-compute HT inter-root weights (constant across layers) ────────
        ht_inter_w = None
        if self.use_ht_inter:
            root_ids_all = node_ids[root_flat_idx]          # [S]
            root_valid   = root_ids_all >= 0                # [S] bool
            rv_ids       = root_ids_all[root_valid]         # [S_valid]
            w_unnorm     = torch.exp(-self.ht_alpha_inter * lp[root_valid])  # [S_valid]
            w_sum        = scatter(w_unnorm, rv_ids, reduce='sum',
                                   dim_size=N_total)         # [N_total]
            ht_inter_w   = w_unnorm / (w_sum[rv_ids] + 1e-16)  # [S_valid] normalised

        # ── L message-passing layers ──────────────────────────────────────────
        for layer in self.layers:
            h_flat = layer(
                h_flat, intra_ei, ea_flat, valid,
                node_ids, N_total, sf.edge_index, sf.edge_attr,
                sub_batch, S, k, root_flat_idx,
                ht_inter_w=ht_inter_w,
            )

        # ── readout: sum-pool subgraphs → [S, H] → [N_total, m, H] ──────────
        valid_mask = node_ids >= 0
        h_sub = scatter(
            h_flat[valid_mask], sub_batch[valid_mask],
            dim=0, reduce='sum', dim_size=S,
        )
        h_sub_2d = h_sub.view(N_total, m, h_sub.shape[-1])   # [N_total, m, H]

        # ── HT-pool or MHA readout ────────────────────────────────────────────
        if self.use_ht_pool:
            # Self-normalised HT: softmax(-α · logP) weighted sum over m subgraphs
            lp_2d = lp.view(N_total, m)                       # [N_total, m]
            w     = torch.softmax(-self.ht_alpha_pool * lp_2d, dim=1)  # [N_total, m]
            node_emb = (w.unsqueeze(-1) * h_sub_2d).sum(dim=1)         # [N_total, H]
        else:
            # Standard MHA readout (ARCH-9)
            h_attn, _ = self.readout_mha(h_sub_2d, h_sub_2d, h_sub_2d)
            node_emb  = h_attn + h_sub_2d
            node_emb  = node_emb.mean(dim=1)                  # [N_total, H]

        node_emb = self.readout_norm(node_emb)
        return global_add_pool(node_emb, sf.batch)            # [B, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-24')
def build_arch24(cfg: ExperimentConfig):
    kw = cfg.model_config.kwargs

    return Arch24GraphEncoder(
        in_channels    = cfg.model_config.node_feature_dim,
        edge_dim       = cfg.model_config.edge_feature_dim,
        hidden_dim     = cfg.model_config.hidden_dim,
        num_layers     = cfg.model_config.mpnn_layers,
        mlp_layers     = kw.get('mlp_layers',     2),
        dropout        = cfg.model_config.dropout,
        num_heads      = kw.get('num_heads',       4),
        use_inter_conv = kw.get('use_inter_conv',  True),
        use_ht_pool    = kw.get('use_ht_pool',     False),
        use_ht_inter   = kw.get('use_ht_inter',    False),
        use_bfs_pe     = kw.get('use_bfs_pe',      True),
        use_logp_pe    = kw.get('use_logp_pe',     True),
    )
