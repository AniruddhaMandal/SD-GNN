"""
ARCH-4: Independent Subgraph GNN with Root-Node Readout

Architecture:
    1. Run L layers of GNN independently within each subgraph (no cross-subgraph MP)
    2. Take the ROOT node embedding h[root_flat_idx[s]] for each subgraph s → [S, H]
    3. Aggregate m root embeddings per node using log_prob weighted aggregator → [T, H]
    4. Pool T node embeddings per graph → [G, H]

Key insight: After L GNN layers, the root node already aggregated information from
all k-1 sampled neighbours. Taking only the root embedding (not a mean over all k)
preserves the structural information specific to that subgraph.  Aggregating m such
embeddings per node with the log_prob aggregator gives a rich node representation.

No cross-subgraph MP means no over-smoothing, no coverage-bleed, and each subgraph
is treated as an independent view of the node's local neighbourhood — exactly the
principle behind DS-GNN / ESAN but applied to random graphlet samples.
"""

import torch
from torch import nn
from torch.nn import functional as F
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
# Intra-subgraph GNN layer  (no cross-subgraph scatter)
# ---------------------------------------------------------------------------

class IntraSubgraphLayer(nn.Module):
    """
    Standard GNN layer operating on the flat [S*k] space.
    Subgraphs are independent — no MP-2 cross-subgraph scatter.
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

        self.conv = self._make_conv(in_channels, out_channels, mlp_layers)
        self.bn   = BatchNorm(out_channels)

        if residual and in_channels != out_channels:
            self.res_proj = nn.Linear(in_channels, out_channels)
        else:
            self.res_proj = nn.Identity()

    def forward(
        self,
        h:        torch.Tensor,  # [S*k, in_channels]
        intra_ei: torch.Tensor,  # [2, E_sub]
        ea_flat:  torch.Tensor,  # [E_sub, edge_dim] or None
        valid:    torch.Tensor,  # [S*k] bool
    ) -> torch.Tensor:            # [S*k, out_channels]
        valid_f = valid.float().unsqueeze(-1)
        h_res   = h

        if self.use_ea and ea_flat is not None:
            h_new = self.conv(h, intra_ei, ea_flat)
        else:
            h_new = self.conv(h, intra_ei)

        # GIN MLP has biases → re-zero padded positions
        h_new = h_new * valid_f

        h_new = self.bn(h_new)
        if self.residual:
            h_new = h_new + self.res_proj(h_res)
        h_new = F.dropout(h_new, p=self.dropout, training=self.training)
        # Keep padded positions at zero after residual/BN
        h_new = h_new * valid_f
        return h_new

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
# Core encoder: runs independent subgraph GNNs, returns root embeddings [S, H]
# ---------------------------------------------------------------------------

class IndependentSubgraphEncoder(nn.Module):
    """
    Runs L layers of intra-subgraph GNN on a SubgraphFeaturesBatch.

    Returns:
        root_embs    [S, H]   root node embedding for each subgraph
        target_batch [S]      target node index for each subgraph (0..T-1)
        log_probs    [S]      log sampling probability per subgraph (or None)
        T            int      number of target nodes in the batch
        N_total      int      total nodes in the original batched graph
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
            IntraSubgraphLayer(
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
        device = sf.x.device
        S, k   = sf.nodes_sampled.shape
        T      = sf.target_nodes.size(0)
        m      = S // T

        # 1. Flatten subgraphs → [S*k] space
        x_flat, ea_flat, intra_ei, sub_batch, node_ids, valid, N_total = \
            _flatten_subgraphs(sf)

        # 2. Root positions in [S*k] flat space
        target_batch  = torch.arange(T, device=device).repeat_interleave(m)  # [S]
        root_global   = sf.target_nodes[target_batch]                          # [S]
        matches       = (sf.nodes_sampled == root_global.unsqueeze(1))        # [S, k]
        root_local    = matches.long().argmax(dim=1)                           # [S]
        root_flat_idx = torch.arange(S, device=device) * k + root_local       # [S]
        root_mask     = torch.zeros(S * k, dtype=torch.long, device=device)
        root_mask[root_flat_idx] = 1

        # 3. Log-prob vector [S*k, 1] — sanitise -inf/nan from failed samples
        if sf.log_probs is not None:
            lp = sf.log_probs.clone()
            lp[~torch.isfinite(lp)] = 0.0
            lp_flat = lp.unsqueeze(1).expand(S, k).reshape(S * k, 1)
        else:
            lp_flat = torch.zeros(S * k, 1, device=device)
        lp_flat = lp_flat * valid.float().unsqueeze(-1)

        # 4. Initial embeddings: node features + log-prob + root indicator
        h = self.initializer(x_flat, lp_flat, root_mask)   # [S*k, H]

        # 5. L layers of independent intra-subgraph GNN
        for layer in self.layers:
            h = layer(h, intra_ei, ea_flat, valid)

        # 6. Extract root node embedding for each subgraph
        root_embs = h[root_flat_idx]   # [S, H]

        return root_embs, target_batch, sf.log_probs, T, N_total


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------

class Arch4GraphEncoder(nn.Module):
    """
    Two-level pooling:
        1. Aggregate m root embeddings per node with log_prob aggregator → [T, H]
        2. Pool node embeddings per graph                                 → [G, H]
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
        aggregator:   str   = 'weighted_mean',
        temperature:  float = 0.5,
        graph_pool:   str   = 'sum',
    ):
        super().__init__()
        self.encoder = IndependentSubgraphEncoder(
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

        # Node-level aggregator: m root embeddings → 1 node embedding
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

        # Graph pool: node embeddings → graph embedding
        if graph_pool == 'sum':
            self.graph_pool_fn = global_add_pool
        elif graph_pool == 'mean':
            self.graph_pool_fn = global_mean_pool
        elif graph_pool == 'max':
            self.graph_pool_fn = global_max_pool
        else:
            raise ValueError(f"Unknown graph_pool: {graph_pool}")

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        device = sf.x.device

        root_embs, target_batch, log_probs, T, N_total = self.encoder(sf)
        # root_embs:    [S, H]
        # target_batch: [S]   — index in 0..T-1

        # ── Step 1: aggregate m root embeddings → node embeddings ──────────
        if getattr(self.aggregator, 'needs_log_probs', False):
            node_embs = self.aggregator(root_embs, target_batch,
                                        log_probs=log_probs)   # [T, H]
        else:
            node_embs = self.aggregator(root_embs, target_batch)   # [T, H]

        # ── Step 2: pool node embeddings → graph embedding ─────────────────
        # target_ptr [G+1] gives CSR boundaries of target nodes per graph
        target_ptr      = sf.target_ptr                            # [G+1]
        G               = target_ptr.size(0) - 1
        nodes_per_graph = target_ptr[1:] - target_ptr[:-1]         # [G]
        node_graph_ptr  = torch.repeat_interleave(
            torch.arange(G, device=device), nodes_per_graph)       # [T]

        return self.graph_pool_fn(node_embs, node_graph_ptr)        # [G, H]


# ---------------------------------------------------------------------------
# Node-level encoder
# ---------------------------------------------------------------------------

class Arch4NodeEncoder(nn.Module):
    """
    Aggregate m root embeddings per target node → [T, H].
    For node classification and link prediction.
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
        aggregator:   str   = 'weighted_mean',
        temperature:  float = 0.5,
    ):
        super().__init__()
        self.encoder = IndependentSubgraphEncoder(
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

    def forward(self, sf: SubgraphFeaturesBatch) -> torch.Tensor:
        root_embs, target_batch, log_probs, T, N_total = self.encoder(sf)

        if getattr(self.aggregator, 'needs_log_probs', False):
            return self.aggregator(root_embs, target_batch, log_probs=log_probs)
        return self.aggregator(root_embs, target_batch)   # [T, H]


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model('ARCH-4')
def build_arch4(cfg: ExperimentConfig):
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
        aggregator  = kw.get('aggregator', 'weighted_mean'),
        temperature = cfg.model_config.temperature or 0.5,
    )

    if is_node_level:
        return Arch4NodeEncoder(**base_kwargs)
    else:
        return Arch4GraphEncoder(
            **base_kwargs,
            graph_pool = kw.get('graph_pool', 'sum'),
        )
