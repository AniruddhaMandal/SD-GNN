"""
    Arch-2-v2: Log probability node encoder + Symmetric Subgraph GNN Layer
        + Arbitary Readout + MLP
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import scatter

from torch_geometric.nn import (
    GINEConv,
    GINConv,
    GCNConv,
    SAGEConv,
    GATv2Conv,
    SGConv,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)
from torch_geometric.nn.norm import BatchNorm

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import get_aggregator, register_model

from typing import Literal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mlp(in_dim, hidden_dim, out_dim, num_layers=2, activate_last=False):
    layers = []
    if num_layers == 1:
        layers += [nn.Linear(in_dim, out_dim)]
        if activate_last: layers += [nn.ReLU()]
    else:
        layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        if activate_last: layers += [nn.ReLU()]
    return nn.Sequential(*layers)


def _flatten_subgraphs(sf: SubgraphFeaturesBatch):
    """
    Expand the batch's original node/edge features into the flat [S*k] subgraph space.

    Returns:
        x_flat      [S*k, F]          gathered node features (padded positions zeroed)
        ea_flat     [E_sub, edge_dim] gathered edge attributes (or None)
        intra_ei    [2, E_sub]        edge index in [S*k] node space
        sub_batch   [S*k]             maps each position → subgraph index (0..S-1)
        node_ids    [S*k]             global node ID per position (-1 for padding)
        valid       [S*k] bool        True for non-padded positions
        N_total     int               total nodes in the original batched graph
    """
    x            = sf.x                  # [N, F]
    edge_attr    = sf.edge_attr          # [E, edge_dim] or None
    nodes_t      = sf.nodes_sampled      # [S, k]  ← no trailing comma!
    edge_index_t = sf.edge_index_sampled # [2, E_sub]
    edge_ptr_t   = sf.edge_ptr           # [S+1]
    edge_src_t   = sf.edge_src_global    # [E_sub]

    device = x.device
    S, k = nodes_t.shape
    N_total = x.size(0)

    node_ids = nodes_t.flatten()                          # [S*k]
    valid    = node_ids >= 0                              # [S*k]
    clamped  = node_ids.clamp(min=0)

    x_flat = x[clamped]
    if not valid.all():
        x_flat = x_flat * valid.float().unsqueeze(-1)

    ea_flat = edge_attr[edge_src_t] if edge_attr is not None else None

    # Remap local edge indices (0..k-1) → global flat indices (0..S*k-1)
    edges_per_sub = edge_ptr_t[1:] - edge_ptr_t[:-1]     # [S]
    offsets = torch.repeat_interleave(
        torch.arange(S, device=device), edges_per_sub
    ) * k                                                  # [E_sub]
    intra_ei = offsets.unsqueeze(0) + edge_index_t        # [2, E_sub]

    sub_batch = torch.repeat_interleave(
        torch.arange(S, device=device), k
    )                                                      # [S*k]

    return x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total


# ---------------------------------------------------------------------------
# Log-Prob Node Feature Initializer
# ---------------------------------------------------------------------------

class LogProbNodeFeatureInitializer(nn.Module):
    """
    Projects raw node features + log-sampling-probability + root indicator
    into a single hidden-dim embedding for each node position in the [S*k] space.

    Args:
        in_channels: raw node feature dimension
        H:           hidden dimension
        mode:        'concat' (3-way projection) or 'add' (element-wise sum)
    """
    def __init__(self, in_channels: int, H: int,
                 mode: Literal['concat', 'add'] = 'concat'):
        super().__init__()
        self.mode = mode
        self.node_proj = nn.Linear(in_channels, H)
        self.logp_proj = nn.Linear(1, H)
        # Fixed: Embedding(2, H) — root is a scalar 0/1 index, not a 2-vector
        self.root_emb  = nn.Embedding(2, H)
        if mode == 'concat':
            self.init_proj = nn.Linear(3 * H, H)

    def forward(self,
                x_flat:    torch.Tensor,   # [S*k, in_channels]
                lp_flat:   torch.Tensor,   # [S*k, 1]
                root_mask: torch.Tensor,   # [S*k] long, 1 at root positions
                ) -> torch.Tensor:         # [S*k, H]
        h_x    = self.node_proj(x_flat)    # [S*k, H]
        h_lp   = self.logp_proj(lp_flat)   # [S*k, H]
        h_root = self.root_emb(root_mask)  # [S*k, H]
        if self.mode == 'concat':
            return self.init_proj(torch.cat([h_x, h_lp, h_root], dim=-1))
        else:  # add
            return h_x + h_lp + h_root  


# ---------------------------------------------------------------------------
# Symmetric Subgraph GNN Layer  (MP-1 + MP-2)
# ---------------------------------------------------------------------------

class SubgraphGNNLayer(nn.Module):
    """
    One layer of symmetric message passing:
      MP-1 (in_subgraph_mp):  GNN on intra-subgraph edges         [S*k, H] → [S*k, H]
      MP-2 (out_subgraph_mp): cross-subgraph scatter → broadcast → fuse

    Forward Pass: [S*k, H] → [S*k, H]
    """
    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        edge_dim:     int,
        mlp_layers:   int  = 2,
        conv_type:    str  = 'gin',
        dropout:      float = 0.1,
        residual:     bool  = True,
        batch_norm:   bool  = True,
    ):
        super().__init__()
        # store hyperparams needed in forward / _make_conv
        self.conv_type   = conv_type
        self.dropout     = dropout
        self.residual    = residual
        self.use_ea      = (conv_type == 'gine')
        self.edge_dim    = edge_dim   # needed by GINEConv for its internal edge projection

        # MP-1: intra-subgraph GNN
        self.in_subgraph_mp = self._make_conv(in_channels, out_channels, mlp_layers)

        # MP-2: fuse self-embedding with cross-subgraph mean  (2H → H)
        self.out_subgraph_mp = nn.Linear(in_channels + out_channels, out_channels)

        self.bn = BatchNorm(out_channels) if batch_norm else nn.Identity()

        # residual projection when dims differ
        if residual and in_channels != out_channels:
            self.res_proj = nn.Linear(in_channels, out_channels)
        else:
            self.res_proj = nn.Identity()

    def forward(
        self,
        h:            torch.Tensor,  # [S*k, in_channels]
        intra_ei:     torch.Tensor,  # [2, E_sub]
        edge_attr:    torch.Tensor,  # [E_sub, edge_dim] or None
        node_ids:     torch.Tensor,  # [S*k] global node IDs (-1 = padding)
        valid:        torch.Tensor,  # [S*k] bool
        N_total:      int,
    ) -> torch.Tensor:               # [S*k, out_channels]
        h_res = h
        valid_f = valid.float().unsqueeze(-1)   # [S*k, 1]
        clamped = node_ids.clamp(min=0)

        # MP-1: within-subgraph
        if self.use_ea and edge_attr is not None:
            h = self.in_subgraph_mp(h, intra_ei, edge_attr)
        else:
            h = self.in_subgraph_mp(h, intra_ei)
        # Bug 1 fix: GIN MLP has biases → MLP(0) ≠ 0, padded positions become
        # non-zero. Re-zero them before scatter so they don't corrupt x_agg.
        h = h * valid_f

        # MP-2: aggregate all copies of each global node, broadcast back.
        # Bug 2 fix: reduce='mean' counts padded positions (all clamped to 0)
        # in the denominator, diluting global node 0. Use sum/valid_count instead.
        h_sum  = scatter(h, clamped, dim=0, dim_size=N_total, reduce='sum')
        v_count = scatter(valid.float(), clamped, dim=0,
                          dim_size=N_total, reduce='sum').clamp(min=1).unsqueeze(-1)
        x_agg   = h_sum / v_count                           # [N_total, H]
        x_cross = x_agg[clamped] * valid_f                  # [S*k, H]
        h = F.relu(self.out_subgraph_mp(torch.cat([h, x_cross], dim=-1)))

        # BN + residual + dropout
        h = self.bn(h)
        if self.residual:
            h = h + self.res_proj(h_res)
        h = F.dropout(h, p=self.dropout, training=self.training)
        # Bug 1 fix (continued): out_subgraph_mp and res_proj also have biases.
        # Zero padded positions at end of layer so the next layer starts clean.
        h = h * valid_f
        return h

    def _make_conv(self, in_dim, out_dim, mlp_layers):
        if self.conv_type == 'gine':
            # edge_dim tells GINEConv to project ea_flat (edge_feature_dim) → in_dim
            # internally before adding to neighbor features. Without this, GINEConv
            # requires edge_attr.size(-1) == in_dim which fails for ZINC (4 vs 64).
            return GINEConv(nn=make_mlp(in_dim, in_dim, out_dim, mlp_layers),
                            train_eps=True, edge_dim=self.edge_dim)
        if self.conv_type == 'gin':
            return GINConv(nn=make_mlp(in_dim, in_dim, out_dim, mlp_layers), train_eps=True)
        if self.conv_type == 'gcn':
            return GCNConv(in_dim, out_dim, cached=False, normalize=True)
        if self.conv_type == 'sage':
            return SAGEConv(in_dim, out_dim)
        if self.conv_type == 'gatv2':
            return GATv2Conv(in_dim, out_dim, heads=1, concat=True)
        if self.conv_type == 'sgc':
            return SGConv(in_dim, out_dim, K=1)
        raise ValueError(f"Unknown conv_type: {self.conv_type}")


# ---------------------------------------------------------------------------
# Core encoder: [S*k, F] → [S*k, H]
# ---------------------------------------------------------------------------

class SubgraphGNNEncoder(nn.Module):
    """
    Encodes every node of every subgraph with L layers of symmetric MP
    (in-subgraph GNN + cross-subgraph scatter-mean).

    Forward Pass: SubgraphFeaturesBatch → h [S*k, H], node_ids, valid, sub_batch, N_total
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim:  int,
        edge_dim:    int,
        num_layers:  int   = 5,
        mlp_layers:  int   = 2,
        dropout:     float = 0.1,
        conv_type:   str   = 'gin',
        residual:    bool  = True,
        init_mode:   Literal['concat', 'add'] = 'concat',
    ):
        super().__init__()
        H = hidden_dim
        self.initializer = LogProbNodeFeatureInitializer(in_channels, H, mode=init_mode)
        self.layers = nn.ModuleList([
            SubgraphGNNLayer(
                in_channels  = H,
                out_channels = H,
                edge_dim     = edge_dim,
                mlp_layers   = mlp_layers,
                conv_type    = conv_type,
                dropout      = dropout,
                residual     = residual,
            )
            for _ in range(num_layers)
        ])

    def forward(self, sf: SubgraphFeaturesBatch):
        """
        Returns:
            h           [S*k, H]
            node_ids    [S*k]    global node IDs (-1 = padding)
            valid       [S*k]    bool mask
            sub_batch   [S*k]    subgraph index per position
            N_total     int
        """
        device = sf.x.device
        S, k   = sf.nodes_sampled.shape
        T      = sf.target_nodes.size(0)
        m      = S // T

        # 1. Flatten batch → [S*k] space
        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        # 2. Build root mask: mark which position in [S*k] is the root node
        target_batch  = torch.arange(T, device=device).repeat_interleave(m)  # [S]
        root_global   = sf.target_nodes[target_batch]                          # [S]
        matches       = (sf.nodes_sampled == root_global.unsqueeze(1))        # [S, k]
        root_local    = matches.long().argmax(dim=1)                           # [S]
        root_flat_idx = torch.arange(S, device=device) * k + root_local       # [S]
        root_mask     = torch.zeros(S * k, dtype=torch.long, device=device)
        root_mask[root_flat_idx] = 1                                           # [S*k]

        # 3. Build log-prob vector [S*k, 1]
        # Sanitize: replace -inf/nan (placeholder subgraphs) with 0 before the linear.
        if sf.log_probs is not None:
            lp = sf.log_probs.clone()
            lp[~torch.isfinite(lp)] = 0.0
            lp_flat = lp.unsqueeze(1).expand(S, k).reshape(S * k, 1)
        else:
            lp_flat = torch.zeros(S * k, 1, device=device)
        # Zero out padded positions — x_flat is already zeroed by valid mask in
        # _flatten_subgraphs, lp_flat must be too so padded h stays at zero.
        lp_flat = lp_flat * valid.float().unsqueeze(-1)

        # 4. Initial node embeddings
        h = self.initializer(x_flat, lp_flat, root_mask)  # [S*k, H]

        # 5. L layers of MP-1 + MP-2
        for layer in self.layers:
            h = layer(h, intra_ei, ea_flat, node_ids, valid, N_total)

        return h, node_ids, valid, sub_batch, N_total


# ---------------------------------------------------------------------------
# Base encoder (shared graph/node pooling setup)
# ---------------------------------------------------------------------------

class _SSGNNBaseEncoder(nn.Module):
    """
    Holds a SubgraphGNNEncoder and the subgraph-level pooling function.
    Subclasses add graph-level or node-level aggregation on top.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim:  int,
        edge_dim:    int,
        num_layers:  int   = 5,
        mlp_layers:  int   = 2,
        dropout:     float = 0.1,
        conv_type:   str   = 'gin',
        residual:    bool  = True,
        sub_pooling: str   = 'mean',
        init_mode:   Literal['concat', 'add'] = 'concat',
    ):
        super().__init__()
        self.encoder = SubgraphGNNEncoder(
            in_channels = in_channels,
            hidden_dim  = hidden_dim,
            edge_dim    = edge_dim,
            num_layers  = num_layers,
            mlp_layers  = mlp_layers,
            dropout     = dropout,
            conv_type   = conv_type,
            residual    = residual,
            init_mode   = init_mode,
        )
        if sub_pooling not in ('mean', 'sum', 'add', 'max'):
            raise ValueError(f"Unknown sub_pooling: {sub_pooling}")
        self.sub_pooling = sub_pooling


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class SSGNNGraphEncoder(_SSGNNBaseEncoder):
    """
    Generates graph representation from sampled subgraphs using symmetric
    message passing and arbitrary readout.

    Returns [G, H].
    """
    def __init__(
        self,
        in_channels:  int,
        hidden_dim:   int,
        edge_dim:     int,
        num_layers:   int   = 5,
        mlp_layers:   int   = 2,
        dropout:      float = 0.1,
        conv_type:    str   = 'gin',
        residual:     bool  = True,
        sub_pooling:  str   = 'mean',
        aggregator:   str   = 'mean',
        temperature:  float = 0.5,
        init_mode:    Literal['concat', 'add'] = 'concat',
    ):
        super().__init__(in_channels, hidden_dim, edge_dim, num_layers,
                         mlp_layers, dropout, conv_type, residual,
                         sub_pooling, init_mode)
        try:
            self.aggregator = get_aggregator(aggregator)(
                hidden_dim=hidden_dim, temperature=temperature)
        except Exception:
            if aggregator in ('sum', 'add'):
                self.aggregator = global_add_pool
            elif aggregator == 'max':
                self.aggregator = global_max_pool
            else:
                self.aggregator = global_mean_pool

    def forward(self, sf: SubgraphFeaturesBatch):
        device = sf.x.device

        h, node_ids, valid, sub_batch, N_total = self.encoder(sf)  # [S*k, H]

        # Subgraph pool: [S*k, H] → [S, H]
        # h is already zero at padded positions (enforced by SubgraphGNNLayer).
        # Dispatch on configured pooling mode; each branch counts only valid positions.
        valid_f = valid.float().unsqueeze(-1)
        if self.sub_pooling in ('sum', 'add'):
            # Sum only valid node embeddings within each subgraph.
            sub_emb = global_add_pool(h * valid_f, sub_batch)              # [S, H]
        elif self.sub_pooling == 'max':
            # Set padded positions to -inf so they never win the max.
            h_masked = h.masked_fill(~valid.unsqueeze(-1), float('-inf'))
            sub_emb  = global_max_pool(h_masked, sub_batch)                # [S, H]
        else:  # 'mean'
            # Divide by valid count, not total k, to exclude padded positions.
            sub_sum   = global_add_pool(h * valid_f, sub_batch)            # [S, H]
            sub_count = global_add_pool(valid_f, sub_batch).clamp(min=1)   # [S, 1]
            sub_emb   = sub_sum / sub_count                                 # [S, H]

        # Graph pool: [S, H] → [G, H]
        sample_ptr      = sf.sample_ptr                       # [G+1]
        G               = sample_ptr.size(0) - 1
        samples_per_g   = sample_ptr[1:] - sample_ptr[:-1]   # [G]
        log_probs       = sf.log_probs                        # [S] or None

        if (samples_per_g == 0).any():
            zero_emb      = torch.zeros(1, sub_emb.size(1), device=device)
            new_embs      = []
            new_lp        = []
            new_graph_ids = []
            for g in range(G):
                ns = samples_per_g[g].item()
                if ns > 0:
                    s, e = sample_ptr[g].item(), sample_ptr[g + 1].item()
                    new_embs.append(sub_emb[s:e])
                    if log_probs is not None:
                        new_lp.append(log_probs[s:e])
                    new_graph_ids.extend([g] * ns)
                else:
                    new_embs.append(zero_emb)
                    if log_probs is not None:
                        new_lp.append(torch.tensor([float('-inf')], device=device))
                    new_graph_ids.append(g)
            sub_emb   = torch.cat(new_embs, dim=0)
            log_probs = torch.cat(new_lp, dim=0) if log_probs is not None else None
            graph_ptr = torch.tensor(new_graph_ids, dtype=torch.long, device=device)
        else:
            graph_ptr = torch.repeat_interleave(
                torch.arange(G, device=device), samples_per_g)

        if getattr(self.aggregator, 'needs_log_probs', False):
            return self.aggregator(sub_emb, graph_ptr, log_probs=log_probs)
        return self.aggregator(sub_emb, graph_ptr)   # [G, H]


# ---------------------------------------------------------------------------
# Node-level encoder
# ---------------------------------------------------------------------------

class SSGNNNodeEncoder(_SSGNNBaseEncoder):
    """
    Generates node representations from sampled subgraphs. Each global node v
    gets an embedding by mean-pooling its representations across all subgraphs.

    Returns [N_total, H].
    """
    def forward(self, sf: SubgraphFeaturesBatch):
        h, node_ids, valid, sub_batch, N_total = self.encoder(sf)  # [S*k, H]

        # Aggregate each global node's embedding across all subgraphs it appears in.
        # Padded positions have node_ids=-1, clamped to 0 → would pollute node 0's mean.
        # Use sum/count so padded contributions (h=0, valid=0) are excluded from denominator.
        valid_f  = valid.float().unsqueeze(-1)
        clamped  = node_ids.clamp(min=0)
        h_sum    = scatter(h * valid_f, clamped, dim=0,
                           dim_size=N_total, reduce='sum')          # [N_total, H]
        v_count  = scatter(valid.float(), clamped, dim=0,
                           dim_size=N_total, reduce='sum'
                           ).clamp(min=1).unsqueeze(-1)             # [N_total, 1]
        node_emb = h_sum / v_count                                  # [N_total, H]
        return node_emb


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-2-V2')
def build_arch2_v2(cfg: ExperimentConfig):
    kw = cfg.model_config.kwargs
    is_node_level = cfg.task in ('Node-Classification', 'Link-Prediction')
    base_kwargs = dict(
        in_channels = cfg.model_config.node_feature_dim,
        edge_dim    = cfg.model_config.edge_feature_dim,
        hidden_dim  = cfg.model_config.hidden_dim,
        num_layers  = cfg.model_config.mpnn_layers,
        dropout     = cfg.model_config.dropout,
        conv_type   = cfg.model_config.mpnn_type,
        residual    = kw.get('residual', True),
        init_mode   = kw.get('init_mode', 'concat'),
    )
    if is_node_level:
        return SSGNNNodeEncoder(
            **base_kwargs,
            sub_pooling = cfg.model_config.subgraph_param.pooling,
        )
    else:
        return SSGNNGraphEncoder(
            **base_kwargs,
            sub_pooling = cfg.model_config.subgraph_param.pooling,
            aggregator  = kw.get('aggregator', 'mean'),
            temperature = cfg.model_config.temperature or 0.5,
        )
