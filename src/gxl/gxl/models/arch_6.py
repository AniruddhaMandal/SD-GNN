"""
ARCH-6 / GPM-gSWARD: Transformer-based Subgraph GNN with gSWARD sampling.

Inspired by GPM (Wang et al., 2025) but replaces random-walk sampling with
gSWARD (graphlet sampling with log probabilities).

Pipeline:
  Phase 1 — Local (Base) Transformer:
    Each gSWARD graphlet of k nodes is treated as a sequence of k tokens.
    Edge features act as per-head additive attention bias (Graphormer-style)
    so the transformer is aware of graph topology within each subgraph.
    Root token embedding is extracted → subgraph_emb [S, H].

  Aggregation:
    Inverse-probability weighted mean of m subgraph embeddings per node
    (WeightedMeanAggregator with log_probs) → node_embs [N, H].
    Rarer subgraphs (low log-prob) get higher weight.

  Phase 2 — Global Transformer:
    All N nodes of each graph packed into dense tensors [G, max_N, H]
    with a src_key_padding_mask for variable-size graphs.
    Edge features again used as per-head additive attention bias.
    Sum-pool valid tokens → graph_emb [G, H].
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import get_aggregator, register_model
from gxl.models.arch_2_v2 import _flatten_subgraphs, LogProbNodeFeatureInitializer

from typing import Literal


# ---------------------------------------------------------------------------
# Edge attention bias builders
# ---------------------------------------------------------------------------

def _build_local_edge_bias(
    edge_index_sampled: torch.Tensor,  # [2, E_sub] LOCAL coords (0..k-1)
    ea_flat:            torch.Tensor,  # [E_sub, edge_dim]
    edge_ptr:           torch.Tensor,  # [S+1]
    edge_proj:          nn.Linear,     # edge_dim → num_heads
    S: int, k: int, num_heads: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Additive attention bias for local transformer: [S*num_heads, k, k].

    bias[s*num_heads + h, src, dst] += edge_proj(ea)[h]
    for each edge (src→dst) in subgraph s, for each head h.
    """
    total = S * num_heads * k * k
    bias_flat = torch.zeros(total, device=device)

    if ea_flat is None or ea_flat.size(0) == 0:
        return bias_flat.reshape(S * num_heads, k, k)

    edges_per_sub = edge_ptr[1:] - edge_ptr[:-1]              # [S]
    sub_for_edge  = torch.repeat_interleave(
        torch.arange(S, device=device), edges_per_sub)         # [E_sub]

    src_local = edge_index_sampled[0]   # [E_sub]  in [0, k-1]
    dst_local = edge_index_sampled[1]   # [E_sub]  in [0, k-1]

    ea_proj = edge_proj(ea_flat)        # [E_sub, num_heads]
    E_sub    = sub_for_edge.size(0)
    heads_t  = torch.arange(num_heads, device=device)  # [num_heads]

    # flat index into [S, num_heads, k, k] treated as 1-D:
    #   idx = (sub * num_heads + h) * k*k + src * k + dst
    sh_flat   = (sub_for_edge.unsqueeze(1) * num_heads
                 + heads_t.unsqueeze(0)) * (k * k)              # [E_sub, num_heads]
    local_pos = (src_local * k + dst_local).unsqueeze(1).expand(-1, num_heads)  # [E_sub, num_heads]
    full_idx  = (sh_flat + local_pos).reshape(-1)               # [E_sub * num_heads]

    bias_flat.scatter_add_(0, full_idx, ea_proj.reshape(-1))
    return bias_flat.reshape(S * num_heads, k, k)


def _build_global_edge_bias(
    edge_index:  torch.Tensor,   # [2, E] global node indices
    edge_attr:   torch.Tensor,   # [E, edge_dim]
    edge_proj:   nn.Linear,      # edge_dim → num_heads
    ptr:         torch.Tensor,   # [G+1] node pointer
    batch_vec:   torch.Tensor,   # [N] graph index per node
    max_N:       int,
    num_heads:   int,
    G:           int,
    device:      torch.device,
) -> torch.Tensor:
    """
    Additive attention bias for global transformer: [G*num_heads, max_N, max_N].
    """
    total    = G * num_heads * max_N * max_N
    bias_flat = torch.zeros(total, device=device)

    if edge_attr is None or edge_attr.size(0) == 0:
        return bias_flat.reshape(G * num_heads, max_N, max_N)

    src_global = edge_index[0]               # [E]
    dst_global = edge_index[1]               # [E]
    g          = batch_vec[src_global]        # [E]

    src_local  = src_global - ptr[g]          # local idx in [0, N_g-1]
    dst_local  = dst_global - ptr[g]

    ea_proj   = edge_proj(edge_attr)           # [E, num_heads]
    heads_t   = torch.arange(num_heads, device=device)

    sh_flat   = (g.unsqueeze(1) * num_heads
                 + heads_t.unsqueeze(0)) * (max_N * max_N)      # [E, num_heads]
    local_pos = (src_local * max_N + dst_local).unsqueeze(1).expand(-1, num_heads)
    full_idx  = (sh_flat + local_pos).reshape(-1)               # [E * num_heads]

    bias_flat.scatter_add_(0, full_idx, ea_proj.reshape(-1))
    return bias_flat.reshape(G * num_heads, max_N, max_N)


# ---------------------------------------------------------------------------
# Local (Base) Transformer
# ---------------------------------------------------------------------------

class LocalSubgraphTransformer(nn.Module):
    """
    Transformer on k tokens of each gSWARD graphlet.

    Input:  h [S*k, H], valid [S*k] bool
    Output: sub_embs [S, H]  (mean-pool of valid tokens per subgraph)

    Edge features add a per-head additive attention bias (Graphormer-style).
    Padded tokens masked via src_key_padding_mask.
    """

    def __init__(
        self,
        hidden_dim:    int,
        num_heads:     int   = 4,
        num_layers:    int   = 4,
        dropout:       float = 0.0,
        edge_dim:      int   = 0,
        use_edge_bias: bool  = True,
        ffn_dim:       int   = None,
    ):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.num_heads     = num_heads
        self.use_edge_bias = use_edge_bias and (edge_dim > 0)

        if self.use_edge_bias:
            self.edge_proj = nn.Linear(edge_dim, num_heads)

        ffn = ffn_dim or (hidden_dim * 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=ffn, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h:                  torch.Tensor,   # [S*k, H]
        valid:              torch.Tensor,   # [S*k] bool
        edge_index_sampled: torch.Tensor,   # [2, E_sub] LOCAL coords (0..k-1)
        ea_flat:            torch.Tensor,   # [E_sub, edge_dim] or None
        edge_ptr:           torch.Tensor,   # [S+1]
        S: int, k: int,
    ) -> torch.Tensor:                      # [S, H]
        device = h.device
        H      = self.hidden_dim

        h_3d     = h.view(S, k, H)
        pad_mask = ~valid.view(S, k)        # True → ignore (PyTorch convention)

        attn_mask = None
        if self.use_edge_bias and ea_flat is not None and ea_flat.size(0) > 0:
            attn_mask = _build_local_edge_bias(
                edge_index_sampled, ea_flat, edge_ptr,
                self.edge_proj, S, k, self.num_heads, device,
            )   # [S*num_heads, k, k]

        # Convert bool pad_mask → float (-inf for masked, 0 for valid)
        # so it matches the float attn_mask type and avoids PyTorch deprecation warning.
        pad_mask_f = pad_mask.float().masked_fill(pad_mask, float('-inf'))

        out = self.transformer(
            h_3d,
            mask=attn_mask,
            src_key_padding_mask=pad_mask_f,
        )   # [S, k, H]
        out = self.norm(out)

        # Mean-pool over ALL valid tokens (matching GPM's mean(dim=1) extraction).
        # Masking padded positions to zero before summing avoids polluting the mean.
        valid_3d  = valid.view(S, k).float().unsqueeze(-1)          # [S, k, 1]
        valid_sum = valid_3d.sum(dim=1).clamp(min=1.0)              # [S, 1]
        sub_embs  = (out * valid_3d).sum(dim=1) / valid_sum         # [S, H]
        return sub_embs


# ---------------------------------------------------------------------------
# Global Transformer
# ---------------------------------------------------------------------------

class GlobalGraphTransformer(nn.Module):
    """
    Transformer on all N nodes of each graph.

    Input:  node_embs [N, H], batch_vec [N], ptr [G+1], edge_index [2,E], edge_attr [E,D]
    Output: graph_embs [G, H]  (sum-pool over valid nodes)
    """

    def __init__(
        self,
        hidden_dim:    int,
        num_heads:     int   = 4,
        num_layers:    int   = 4,
        dropout:       float = 0.0,
        edge_dim:      int   = 0,
        use_edge_bias: bool  = True,
        ffn_dim:       int   = None,
    ):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.num_heads     = num_heads
        self.use_edge_bias = use_edge_bias and (edge_dim > 0)

        if self.use_edge_bias:
            self.edge_proj = nn.Linear(edge_dim, num_heads)

        ffn = ffn_dim or (hidden_dim * 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=ffn, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_embs:  torch.Tensor,   # [N, H]
        batch_vec:  torch.Tensor,   # [N] graph index
        ptr:        torch.Tensor,   # [G+1]
        edge_index: torch.Tensor,   # [2, E]
        edge_attr:  torch.Tensor,   # [E, edge_dim] or None
    ) -> torch.Tensor:               # [G, H]
        device = node_embs.device
        G      = ptr.size(0) - 1

        dense_x, valid_mask = to_dense_batch(node_embs, batch_vec)  # [G, max_N, H], [G, max_N]
        max_N = dense_x.size(1)

        attn_mask = None
        if self.use_edge_bias and edge_attr is not None and edge_attr.size(0) > 0:
            attn_mask = _build_global_edge_bias(
                edge_index, edge_attr, self.edge_proj,
                ptr, batch_vec, max_N, self.num_heads, G, device,
            )

        pad_mask   = ~valid_mask                                               # [G, max_N] bool
        pad_mask_f = pad_mask.float().masked_fill(pad_mask, float('-inf'))    # float version

        out = self.transformer(
            dense_x,
            mask=attn_mask,
            src_key_padding_mask=pad_mask_f,
        )   # [G, max_N, H]
        out = self.norm(out)

        # Masked sum-pool over valid nodes
        graph_emb = (out * valid_mask.unsqueeze(-1)).sum(dim=1)   # [G, H]
        return graph_emb


# ---------------------------------------------------------------------------
# Shared forward logic
# ---------------------------------------------------------------------------

def _gpm_encode(
    sf: SubgraphFeaturesBatch,
    initializer: LogProbNodeFeatureInitializer,
    local_transformer: LocalSubgraphTransformer,
    aggregator,
) -> torch.Tensor:
    """
    Shared phase-1 logic: SubgraphFeaturesBatch → node_embs [N, H].

    Steps:
      1. Flatten subgraphs, build initial node embeddings [S*k, H]
      2. Local transformer on each graphlet → sub_embs [S, H] (mean-pool)
      3. mean aggregation per node → node_embs [N, H]
    """
    device = sf.x.device
    S, k   = sf.nodes_sampled.shape
    T      = sf.target_nodes.size(0)
    m      = S // T

    # Flatten
    x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
        _flatten_subgraphs(sf)

    # Target-to-subgraph mapping: subgraph s belongs to target node target_batch[s]
    target_batch = torch.arange(T, device=device).repeat_interleave(m)

    # Root positions (in local coords, 0..k-1) — used as root indicator in init
    root_global   = sf.target_nodes[target_batch]
    matches       = (sf.nodes_sampled == root_global.unsqueeze(1))
    root_local    = matches.long().argmax(dim=1)                     # [S]
    root_mask     = torch.zeros(S * k, dtype=torch.long, device=device)
    root_flat_idx = torch.arange(S, device=device) * k + root_local
    root_mask[root_flat_idx] = 1

    # Log-prob vector [S*k, 1]
    if sf.log_probs is not None:
        lp = sf.log_probs.clone()
        lp[~torch.isfinite(lp)] = 0.0
        lp_flat = lp.unsqueeze(1).expand(S, k).reshape(S * k, 1)
    else:
        lp_flat = torch.zeros(S * k, 1, device=device)
    lp_flat = lp_flat * valid.float().unsqueeze(-1)

    # Initial embeddings [S*k, H]
    h = initializer(x_flat, lp_flat, root_mask)

    # Phase 1: Local Transformer — mean-pool over all valid tokens → [S, H]
    # Pass sf.edge_index_sampled (local 0..k-1 coords), not intra_ei (global S*k coords)
    sub_embs = local_transformer(
        h, valid,
        sf.edge_index_sampled, ea_flat, sf.edge_ptr,
        S, k,
    )   # [S, H]

    # Aggregation: m sub_embs → node_embs [T=N_total, H]
    log_probs = sf.log_probs
    if getattr(aggregator, 'needs_log_probs', False):
        node_embs = aggregator(sub_embs, target_batch, log_probs=log_probs)
    else:
        node_embs = aggregator(sub_embs, target_batch)

    return node_embs   # [T, H]


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class GPMgSWARDGraphEncoder(nn.Module):
    def __init__(
        self,
        in_channels:   int,
        hidden_dim:    int,
        edge_dim:      int,
        sub_layers:    int   = 4,
        global_layers: int   = 4,
        num_heads:     int   = 4,
        dropout:       float = 0.0,
        init_mode:     Literal['concat', 'add'] = 'concat',
        aggregator:    str   = 'weighted_mean',
        temperature:   float = 0.5,
        use_edge_bias: bool  = True,
        ffn_dim:       int   = None,
    ):
        super().__init__()
        self.initializer = LogProbNodeFeatureInitializer(in_channels, hidden_dim, mode=init_mode)
        self.local_transformer = LocalSubgraphTransformer(
            hidden_dim=hidden_dim, num_heads=num_heads, num_layers=sub_layers,
            dropout=dropout, edge_dim=edge_dim, use_edge_bias=use_edge_bias,
            ffn_dim=ffn_dim,
        )
        try:
            self.aggregator = get_aggregator(aggregator)(
                hidden_dim=hidden_dim, temperature=temperature)
        except Exception:
            from torch_geometric.nn import global_mean_pool
            self.aggregator = global_mean_pool
        self.global_transformer = GlobalGraphTransformer(
            hidden_dim=hidden_dim, num_heads=num_heads, num_layers=global_layers,
            dropout=dropout, edge_dim=edge_dim, use_edge_bias=use_edge_bias,
            ffn_dim=ffn_dim,
        )

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        node_embs = _gpm_encode(sf, self.initializer,
                                self.local_transformer, self.aggregator)

        ptr = sf.ptr
        if ptr is None:
            G   = int(sf.batch.max().item()) + 1
            ptr = torch.zeros(G + 1, dtype=torch.long, device=sf.x.device)
            for g in range(G):
                ptr[g + 1] = (sf.batch == g).sum() + ptr[g]

        return self.global_transformer(
            node_embs, sf.batch, ptr, sf.edge_index, sf.edge_attr,
        )   # [G, H]


# ---------------------------------------------------------------------------
# Node-level encoder
# ---------------------------------------------------------------------------

class GPMgSWARDNodeEncoder(nn.Module):
    def __init__(
        self,
        in_channels:   int,
        hidden_dim:    int,
        edge_dim:      int,
        sub_layers:    int   = 4,
        global_layers: int   = 4,
        num_heads:     int   = 4,
        dropout:       float = 0.0,
        init_mode:     Literal['concat', 'add'] = 'concat',
        aggregator:    str   = 'weighted_mean',
        temperature:   float = 0.5,
        use_edge_bias: bool  = True,
        ffn_dim:       int   = None,
    ):
        super().__init__()
        self.initializer = LogProbNodeFeatureInitializer(in_channels, hidden_dim, mode=init_mode)
        self.local_transformer = LocalSubgraphTransformer(
            hidden_dim=hidden_dim, num_heads=num_heads, num_layers=sub_layers,
            dropout=dropout, edge_dim=edge_dim, use_edge_bias=use_edge_bias,
            ffn_dim=ffn_dim,
        )
        try:
            self.aggregator = get_aggregator(aggregator)(
                hidden_dim=hidden_dim, temperature=temperature)
        except Exception:
            from torch_geometric.nn import global_mean_pool
            self.aggregator = global_mean_pool
        self.global_transformer = GlobalGraphTransformer(
            hidden_dim=hidden_dim, num_heads=num_heads, num_layers=global_layers,
            dropout=dropout, edge_dim=edge_dim, use_edge_bias=use_edge_bias,
            ffn_dim=ffn_dim,
        )

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        node_embs = _gpm_encode(sf, self.initializer,
                                self.local_transformer, self.aggregator)

        ptr = sf.ptr
        if ptr is None:
            G   = int(sf.batch.max().item()) + 1
            ptr = torch.zeros(G + 1, dtype=torch.long, device=sf.x.device)
            for g in range(G):
                ptr[g + 1] = (sf.batch == g).sum() + ptr[g]

        return self.global_transformer(
            node_embs, sf.batch, ptr, sf.edge_index, sf.edge_attr,
        )   # [N, H] — global transformer returns dense_x sum-pooled... actually [G,H]
        # For node-level tasks, we want per-node embeddings.
        # Return node_embs before global pooling instead.


# Node encoder should return [N, H], not [G, H].
# Override forward to return per-node embeddings from the global transformer output.
class GPMgSWARDNodeEncoder(nn.Module):  # noqa: F811  (redefine cleanly)
    def __init__(
        self,
        in_channels:   int,
        hidden_dim:    int,
        edge_dim:      int,
        sub_layers:    int   = 4,
        global_layers: int   = 4,
        num_heads:     int   = 4,
        dropout:       float = 0.0,
        init_mode:     Literal['concat', 'add'] = 'concat',
        aggregator:    str   = 'weighted_mean',
        temperature:   float = 0.5,
        use_edge_bias: bool  = True,
        ffn_dim:       int   = None,
    ):
        super().__init__()
        self.initializer = LogProbNodeFeatureInitializer(in_channels, hidden_dim, mode=init_mode)
        self.local_transformer = LocalSubgraphTransformer(
            hidden_dim=hidden_dim, num_heads=num_heads, num_layers=sub_layers,
            dropout=dropout, edge_dim=edge_dim, use_edge_bias=use_edge_bias,
            ffn_dim=ffn_dim,
        )
        try:
            self.aggregator = get_aggregator(aggregator)(
                hidden_dim=hidden_dim, temperature=temperature)
        except Exception:
            from torch_geometric.nn import global_mean_pool
            self.aggregator = global_mean_pool

        # Global transformer layers (without final pool — returns node-level)
        ffn = ffn_dim or (hidden_dim * 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=ffn, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.global_transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=global_layers)
        self.global_norm = nn.LayerNorm(hidden_dim)
        if use_edge_bias and edge_dim > 0:
            self.global_edge_proj = nn.Linear(edge_dim, num_heads)
        else:
            self.global_edge_proj = None
        self.num_heads    = num_heads
        self.use_edge_bias = use_edge_bias and (edge_dim > 0)
        self.edge_dim      = edge_dim

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        device    = sf.x.device
        node_embs = _gpm_encode(sf, self.initializer,
                                self.local_transformer, self.aggregator)  # [N, H]

        ptr = sf.ptr
        if ptr is None:
            G   = int(sf.batch.max().item()) + 1
            ptr = torch.zeros(G + 1, dtype=torch.long, device=device)
            for g in range(G):
                ptr[g + 1] = (sf.batch == g).sum() + ptr[g]

        G = ptr.size(0) - 1
        dense_x, valid_mask = to_dense_batch(node_embs, sf.batch)  # [G, max_N, H]
        max_N = dense_x.size(1)

        attn_mask = None
        if self.use_edge_bias and sf.edge_attr is not None:
            attn_mask = _build_global_edge_bias(
                sf.edge_index, sf.edge_attr, self.global_edge_proj,
                ptr, sf.batch, max_N, self.num_heads, G, device,
            )

        pad_mask_n   = ~valid_mask
        pad_mask_nf  = pad_mask_n.float().masked_fill(pad_mask_n, float('-inf'))
        out = self.global_transformer_enc(
            dense_x, mask=attn_mask,
            src_key_padding_mask=pad_mask_nf,
        )   # [G, max_N, H]
        out = self.global_norm(out)

        # Un-pad: extract each node's embedding back from the dense tensor
        # valid_mask [G, max_N] → out[valid_mask] gives [N, H] in correct order
        node_out = out[valid_mask]   # [N_total, H]
        return node_out


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-6')
def build_arch6(cfg: ExperimentConfig):
    kw            = cfg.model_config.kwargs
    is_node_level = cfg.task in ('Node-Classification', 'Link-Prediction')

    total         = cfg.model_config.mpnn_layers
    sub_layers    = kw.get('sub_layers',    total // 2)
    global_layers = kw.get('global_layers', total - sub_layers)

    base_kwargs = dict(
        in_channels   = cfg.model_config.node_feature_dim,
        hidden_dim    = cfg.model_config.hidden_dim,
        edge_dim      = cfg.model_config.edge_feature_dim,
        sub_layers    = sub_layers,
        global_layers = global_layers,
        num_heads     = kw.get('num_heads', 4),
        dropout       = cfg.model_config.dropout,
        init_mode     = kw.get('init_mode', 'concat'),
        aggregator    = kw.get('aggregator', 'weighted_mean'),
        temperature   = cfg.model_config.temperature or 0.5,
        use_edge_bias = kw.get('use_edge_bias', True),
        ffn_dim       = kw.get('ffn_dim', None),
    )

    if is_node_level:
        return GPMgSWARDNodeEncoder(**base_kwargs)
    else:
        return GPMgSWARDGraphEncoder(**base_kwargs)
