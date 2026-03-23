"""
ARCH-3: Separate Local+Global Track Subgraph GNN

Problem with ARCH-2-V2:
    Per-layer MP-2 fused [h_local, x_cross] via Linear(2H→H), which could learn
    to suppress h_local in favour of the cross-subgraph mean. Over L layers, all
    copies of node v converge to the same value → subgraph diversity is lost.

Fix:
    Maintain two separate tensors per layer:
      h_local  [S*k, H]    — subgraph-specific, updated ONLY by intra-subgraph GNN
      h_global [N_total, H] — cross-subgraph consensus, updated by scatter-mean of h_local

    At the start of MP-1 we inject h_global additively into h_local (so the GNN has
    global context), but the OUTPUT of MP-1 is the new h_local — the global mean never
    overwrites local structure, it only informs it.

Final readout:
    'global': sum-pool h_global over graph nodes  (clean, standard-GNN-style)
    'local' : root-node embeddings from h_local, aggregated over subgraphs
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import scatter
from torch_geometric.nn import (
    GINEConv, GINConv, GCNConv, SAGEConv,
    global_add_pool, global_mean_pool, global_max_pool,
)
from torch_geometric.nn.norm import BatchNorm

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import get_aggregator, register_model

# Re-use stable helpers from arch_2_v2
from gxl.models.arch_2_v2 import (
    make_mlp,
    _flatten_subgraphs,
    LogProbNodeFeatureInitializer,
)

from typing import Literal


# ---------------------------------------------------------------------------
# Separate-track layer
# ---------------------------------------------------------------------------

class SeparateTrackLayer(nn.Module):
    """
    One layer of separate-track symmetric message passing.

    Flow:
        h_in     = (h_local + h_global[nodes]) * valid   # inject global, zero padding
        h_local  = MP-1(h_in, intra_ei)                  # intra-subgraph GNN
        h_global = scatter_mean(h_local)                 # cross-subgraph consensus
        (+ BN, residual, dropout on each track independently)

    The local track is never overwritten by the global mean — it only
    receives global context as an additive offset at the MP-1 INPUT.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        edge_dim:     int,
        mlp_layers:   int   = 2,
        conv_type:    str   = 'gin',
        dropout:      float = 0.0,
        residual:     bool  = True,
    ):
        super().__init__()
        self.conv_type = conv_type
        self.dropout   = dropout
        self.residual  = residual
        self.use_ea    = (conv_type == 'gine')
        self.edge_dim  = edge_dim

        self.mp1       = self._make_conv(in_channels, out_channels, mlp_layers)
        self.bn_local  = BatchNorm(out_channels)
        self.bn_global = BatchNorm(out_channels)

        if residual and in_channels != out_channels:
            self.res_local  = nn.Linear(in_channels, out_channels)
            self.res_global = nn.Linear(in_channels, out_channels)
        else:
            self.res_local  = nn.Identity()
            self.res_global = nn.Identity()

    def forward(
        self,
        h_local:  torch.Tensor,   # [S*k, in_channels]
        h_global: torch.Tensor,   # [N_total, in_channels]
        intra_ei: torch.Tensor,   # [2, E_sub]
        ea_flat:  torch.Tensor,   # [E_sub, edge_dim] or None
        node_ids: torch.Tensor,   # [S*k]  global IDs (-1 = padding)
        valid:    torch.Tensor,   # [S*k] bool
        N_total:  int,
    ):
        valid_f  = valid.float().unsqueeze(-1)   # [S*k, 1]
        clamped  = node_ids.clamp(0)             # padded positions → 0

        h_local_res  = h_local
        h_global_res = h_global

        # ── MP-1 ────────────────────────────────────────────────────────────
        # Inject global context additively at the INPUT so MP-1 has full
        # context, but the output is purely a function of local subgraph
        # structure — global mean doesn't overwrite local.
        h_in = (h_local + h_global[clamped]) * valid_f   # [S*k, H]

        if self.use_ea and ea_flat is not None:
            h_local_new = self.mp1(h_in, intra_ei, ea_flat)
        else:
            h_local_new = self.mp1(h_in, intra_ei)

        # GIN MLP has biases → re-zero padding after every biased op
        h_local_new = h_local_new * valid_f

        # ── MP-2: update global from local ─────────────────────────────────
        # Use sum/count (not scatter mean) to exclude padded positions from
        # the denominator — same fix as in ARCH-2-V2.
        h_sum    = scatter(h_local_new, clamped, dim=0,
                           dim_size=N_total, reduce='sum')                   # [N_total, H]
        v_count  = scatter(valid.float(), clamped, dim=0,
                           dim_size=N_total, reduce='sum'
                           ).clamp(min=1).unsqueeze(-1)                      # [N_total, 1]
        h_global_new = h_sum / v_count                                       # [N_total, H]

        # ── Local track: BN → residual → dropout → re-zero ─────────────────
        h_local_new = self.bn_local(h_local_new)
        if self.residual:
            h_local_new = h_local_new + self.res_local(h_local_res)
        h_local_new = F.dropout(h_local_new, p=self.dropout, training=self.training)
        h_local_new = h_local_new * valid_f

        # ── Global track: BN → residual → dropout ──────────────────────────
        h_global_new = self.bn_global(h_global_new)
        if self.residual:
            h_global_new = h_global_new + self.res_global(h_global_res)
        h_global_new = F.dropout(h_global_new, p=self.dropout, training=self.training)

        return h_local_new, h_global_new

    def _make_conv(self, in_dim, out_dim, mlp_layers):
        if self.conv_type == 'gine':
            return GINEConv(make_mlp(in_dim, in_dim, out_dim, mlp_layers),
                            train_eps=True, edge_dim=self.edge_dim)
        if self.conv_type == 'gin':
            return GINConv(make_mlp(in_dim, in_dim, out_dim, mlp_layers),
                           train_eps=True)
        if self.conv_type == 'gcn':
            return GCNConv(in_dim, out_dim)
        if self.conv_type == 'sage':
            return SAGEConv(in_dim, out_dim)
        raise ValueError(f"Unknown conv_type: {self.conv_type}")


# ---------------------------------------------------------------------------
# Core encoder
# ---------------------------------------------------------------------------

class SeparateTrackEncoder(nn.Module):
    """
    Runs L layers of separate-track MP on a SubgraphFeaturesBatch.

    Returns:
        h_local       [S*k, H]    subgraph-specific node embeddings
        h_global      [N_total, H] cross-subgraph consensus node embeddings
        node_ids      [S*k]
        valid         [S*k] bool
        sub_batch     [S*k]       subgraph index per flat position
        root_flat_idx [S]         flat index of root node per subgraph
        N_total       int
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim:  int,
        edge_dim:    int,
        num_layers:  int   = 4,
        mlp_layers:  int   = 2,
        dropout:     float = 0.0,
        conv_type:   str   = 'gin',
        residual:    bool  = True,
        init_mode:   Literal['concat', 'add'] = 'concat',
    ):
        super().__init__()
        H = hidden_dim
        self.initializer = LogProbNodeFeatureInitializer(in_channels, H, mode=init_mode)
        self.layers = nn.ModuleList([
            SeparateTrackLayer(
                in_channels  = H,
                out_channels = H,
                edge_dim     = edge_dim,
                mlp_layers   = mlp_layers,
                conv_type    = conv_type,
                dropout      = dropout,
                residual      = residual,
            )
            for _ in range(num_layers)
        ])

    def forward(self, sf: SubgraphFeaturesBatch):
        device = sf.x.device
        S, k   = sf.nodes_sampled.shape
        T      = sf.target_nodes.size(0)
        m      = S // T

        # 1. Flatten batch → [S*k] space
        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        # 2. Root mask [S*k]: 1 at the root position of each subgraph
        target_batch  = torch.arange(T, device=device).repeat_interleave(m)
        root_global   = sf.target_nodes[target_batch]
        matches       = (sf.nodes_sampled == root_global.unsqueeze(1))
        root_local    = matches.long().argmax(dim=1)
        root_flat_idx = torch.arange(S, device=device) * k + root_local
        root_mask     = torch.zeros(S * k, dtype=torch.long, device=device)
        root_mask[root_flat_idx] = 1

        # 3. Log-prob vector [S*k, 1] — sanitize -inf/nan from failed samples
        if sf.log_probs is not None:
            lp = sf.log_probs.clone()
            lp[~torch.isfinite(lp)] = 0.0
            lp_flat = lp.unsqueeze(1).expand(S, k).reshape(S * k, 1)
        else:
            lp_flat = torch.zeros(S * k, 1, device=device)
        lp_flat = lp_flat * valid.float().unsqueeze(-1)

        # 4. Initialise h_local from features + log-prob + root indicator
        h_local = self.initializer(x_flat, lp_flat, root_mask)   # [S*k, H]

        # 5. Initialise h_global as scatter-mean of initial h_local
        clamped  = node_ids.clamp(0)
        valid_f  = valid.float().unsqueeze(-1)
        h_sum    = scatter(h_local * valid_f, clamped, dim=0,
                           dim_size=N_total, reduce='sum')
        v_count  = scatter(valid.float(), clamped, dim=0,
                           dim_size=N_total, reduce='sum').clamp(1).unsqueeze(-1)
        h_global = h_sum / v_count                                # [N_total, H]

        # 6. L layers of separate-track MP
        for layer in self.layers:
            h_local, h_global = layer(
                h_local, h_global, intra_ei, ea_flat, node_ids, valid, N_total
            )

        return h_local, h_global, node_ids, valid, sub_batch, root_flat_idx, N_total


# ---------------------------------------------------------------------------
# Base (holds encoder + readout config)
# ---------------------------------------------------------------------------

class _SymGNNBase(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim:  int,
        edge_dim:    int,
        num_layers:  int   = 4,
        mlp_layers:  int   = 2,
        dropout:     float = 0.0,
        conv_type:   str   = 'gin',
        residual:    bool  = True,
        init_mode:   Literal['concat', 'add'] = 'concat',
    ):
        super().__init__()
        self.encoder = SeparateTrackEncoder(
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


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class SymGNNGraphEncoder(_SymGNNBase):
    """
    Graph representation from separate-track subgraph GNN.

    readout='global' (default):
        Sum-pool h_global over all nodes in each graph.
        h_global[v] after L layers encodes v's local subgraph structure
        (via MP-1) + cross-subgraph consensus (via scatter-mean). Summing
        over all N nodes gives a graph embedding that scales with graph size,
        exactly like standard GNN sum-readout but with richer node reps.

    readout='local':
        Take h_local[root_flat_idx[s]] for each subgraph s (the root node's
        local embedding), then aggregate over subgraphs per graph. Preserves
        subgraph-specific diversity since h_local is never overwritten by the
        global mean.

    Returns [G, H].
    """

    def __init__(
        self,
        in_channels:  int,
        hidden_dim:   int,
        edge_dim:     int,
        num_layers:   int   = 4,
        mlp_layers:   int   = 2,
        dropout:      float = 0.0,
        conv_type:    str   = 'gin',
        residual:     bool  = True,
        init_mode:    Literal['concat', 'add'] = 'concat',
        readout:      Literal['global', 'local'] = 'global',
        graph_pool:   str   = 'sum',    # for readout='global': sum/mean/max over nodes
        aggregator:   str   = 'sum',    # for readout='local':  aggregator over subgraphs
        temperature:  float = 0.5,
    ):
        super().__init__(in_channels, hidden_dim, edge_dim, num_layers,
                         mlp_layers, dropout, conv_type, residual, init_mode)
        self.readout = readout

        # readout='global': simple pool over nodes
        if graph_pool == 'sum':
            self.graph_pool_fn = global_add_pool
        elif graph_pool == 'mean':
            self.graph_pool_fn = global_mean_pool
        elif graph_pool == 'max':
            self.graph_pool_fn = global_max_pool
        else:
            raise ValueError(f"Unknown graph_pool: {graph_pool}")

        # readout='local': aggregator over subgraph root embeddings
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

        h_local, h_global, node_ids, valid, sub_batch, root_flat_idx, N_total = \
            self.encoder(sf)

        if self.readout == 'global':
            # h_global: [N_total, H] — sum over nodes per graph
            # sf.batch maps each node → its graph index
            return self.graph_pool_fn(h_global, sf.batch)   # [G, H]

        # readout == 'local': root embeddings from h_local → aggregate over subgraphs
        root_embs = h_local[root_flat_idx]                  # [S, H]

        sample_ptr    = sf.sample_ptr                        # [G+1]
        G             = sample_ptr.size(0) - 1
        samples_per_g = sample_ptr[1:] - sample_ptr[:-1]    # [G]
        log_probs     = sf.log_probs                         # [S] or None

        if (samples_per_g == 0).any():
            zero_emb      = torch.zeros(1, root_embs.size(1), device=device)
            new_embs, new_lp, new_graph_ids = [], [], []
            for g in range(G):
                ns = samples_per_g[g].item()
                if ns > 0:
                    s, e = sample_ptr[g].item(), sample_ptr[g + 1].item()
                    new_embs.append(root_embs[s:e])
                    if log_probs is not None:
                        new_lp.append(log_probs[s:e])
                    new_graph_ids.extend([g] * ns)
                else:
                    new_embs.append(zero_emb)
                    if log_probs is not None:
                        new_lp.append(torch.tensor([float('-inf')], device=device))
                    new_graph_ids.append(g)
            root_embs = torch.cat(new_embs, dim=0)
            log_probs = torch.cat(new_lp,   dim=0) if log_probs is not None else None
            graph_ptr = torch.tensor(new_graph_ids, dtype=torch.long, device=device)
        else:
            graph_ptr = torch.repeat_interleave(
                torch.arange(G, device=device), samples_per_g)

        if getattr(self.aggregator, 'needs_log_probs', False):
            return self.aggregator(root_embs, graph_ptr, log_probs=log_probs)
        return self.aggregator(root_embs, graph_ptr)         # [G, H]


# ---------------------------------------------------------------------------
# Node-level encoder
# ---------------------------------------------------------------------------

class SymGNNNodeEncoder(_SymGNNBase):
    """
    Node representations from separate-track subgraph GNN.

    h_global[v] after L layers IS the per-node representation — no additional
    scatter/mean needed. Returns [N_total, H].
    """

    def forward(self, sf: SubgraphFeaturesBatch):
        _, h_global, *_ = self.encoder(sf)
        return h_global   # [N_total, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-3')
def build_arch3(cfg: ExperimentConfig):
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
        return SymGNNNodeEncoder(**base_kwargs)
    else:
        return SymGNNGraphEncoder(
            **base_kwargs,
            readout     = kw.get('readout', 'global'),
            graph_pool  = kw.get('graph_pool', 'sum'),
            aggregator  = kw.get('aggregator', 'sum'),
            temperature = cfg.model_config.temperature or 0.5,
        )
