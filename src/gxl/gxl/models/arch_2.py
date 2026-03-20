"""
Architecture 2: LP-NF (Log-Probability Node Features) with Layerwise MP-1 + MP-2

Node features are initialized as:  W_x * x_v || W_p * log(P_i) || W_r * 1[v = root(S_i)]

Per layer:
  MP-1: GIN within each subgraph S_i
  MP-2: cross-subgraph mean aggregation of all copies of each node v (SUN-aligned)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import scatter

from gxl import SubgraphFeaturesBatch, ExperimentConfig
from gxl.registry import get_aggregator, register_model


def _make_mlp(in_dim, hidden_dim, out_dim, num_layers=2):
    layers = []
    if num_layers == 1:
        layers += [nn.Linear(in_dim, out_dim)]
    else:
        layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)


class _LPNFBase(nn.Module):
    """
    Shared base for LP-NF encoders.

    Handles:
      - Initial feature construction: x_v || log_p_i || root_indicator → hidden_dim
      - MP-1 (within-subgraph GIN) per layer
      - MP-2 (cross-subgraph mean scatter) per layer
    """

    def __init__(self,
                 in_channels: int,
                 edge_dim: int,
                 hidden_dim: int,
                 num_layers: int = 5,
                 mlp_layers: int = 2,
                 dropout: float = 0.1,
                 conv_type: str = 'gin',
                 residual: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual

        H = hidden_dim

        # Initial feature projections
        self.node_proj = nn.Linear(in_channels, H)
        self.logp_proj = nn.Linear(1, H)
        self.root_emb  = nn.Embedding(2, H)           # 0=non-root, 1=root
        self.init_proj = nn.Linear(3 * H, H)

        # Per-layer MP-1: within-subgraph GIN
        self.mp1_convs = nn.ModuleList([
            GINConv(_make_mlp(H, H, H, mlp_layers), train_eps=True)
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([BatchNorm(H) for _ in range(num_layers)])

        # Per-layer MP-2: fuse self-embedding + cross-subgraph mean
        self.mp2_update = nn.ModuleList([
            nn.Linear(2 * H, H) for _ in range(num_layers)
        ])

    def _find_root_positions(self, nodes_t, target_nodes, target_batch):
        """
        Find local index of the root node within each subgraph.

        Args:
            nodes_t:      [S, k] global node IDs (padded with -1)
            target_nodes: [T] global root node IDs
            target_batch: [S] maps each subgraph → target index in [0..T-1]

        Returns:
            root_local: [S] local index (0..k-1) of root in each subgraph
        """
        root_global = target_nodes[target_batch]              # [S]
        matches = (nodes_t == root_global.unsqueeze(1))       # [S, k]
        root_local = matches.long().argmax(dim=1)             # [S]
        return root_local

    def _build_initial_features(self, x, edge_attr, nodes_t, log_probs,
                                 root_local, edge_ptr, edge_index_t,
                                 edge_src_global):
        """
        Build the initial node embedding tensor h [S*k, H] and all related
        index tensors needed for MP-1 and MP-2.

        Returns:
            h              [S*k, H]
            ea_flat        [E_total, edge_dim] or None
            intra_ei       [2, E_total]  edge index in the S*k node space
            subgraph_batch [S*k]         maps node → subgraph index
            root_flat_idx  [S]           flat index of root node per subgraph
            node_ids_flat  [S*k]         global node ID per position
            valid          [S*k] bool    True for non-padded positions
            N_total        int           total nodes in the original graph
        """
        device = x.device
        S, k = nodes_t.shape
        N_total = x.size(0)

        # Flatten node IDs
        node_ids_flat = nodes_t.flatten()                          # [S*k]
        valid = node_ids_flat >= 0                                 # [S*k]
        node_ids_clamped = node_ids_flat.clamp(min=0)

        # Gather global node features; zero out padding
        global_x = x[node_ids_clamped]                            # [S*k, in_channels]
        if not valid.all():
            global_x = global_x * valid.float().unsqueeze(-1)

        # Log probability broadcast: one log_p per subgraph → all k nodes
        if log_probs is not None:
            lp = log_probs.unsqueeze(1).expand(S, k).reshape(S * k, 1)
        else:
            lp = torch.zeros(S * k, 1, device=device)

        # Root indicator per position
        root_flat_idx = torch.arange(S, device=device) * k + root_local  # [S]
        root_mask = torch.zeros(S * k, dtype=torch.long, device=device)
        root_mask[root_flat_idx] = 1                               # 1 at root positions

        # Initial projection
        h_node = self.node_proj(global_x)                         # [S*k, H]
        h_logp = self.logp_proj(lp)                               # [S*k, H]
        h_root = self.root_emb(root_mask)                         # [S*k, H]
        h = self.init_proj(torch.cat([h_node, h_logp, h_root], dim=-1))  # [S*k, H]

        # Build intra-subgraph edge index in [S*k] node space
        # edge_ptr: [S+1], edge_index_t: [2, E_total] in local coords [0..k-1]
        num_edges_per_subgraph = edge_ptr[1:] - edge_ptr[:-1]     # [S]
        offsets = torch.repeat_interleave(
            torch.arange(S, device=device), num_edges_per_subgraph
        ) * k                                                      # [E_total]
        intra_ei = offsets.unsqueeze(0) + edge_index_t            # [2, E_total]

        # Edge attributes in flat space
        if edge_attr is not None:
            ea_flat = edge_attr[edge_src_global]
        else:
            ea_flat = None

        # Subgraph batch vector
        subgraph_batch = torch.repeat_interleave(
            torch.arange(S, device=device), k
        )                                                          # [S*k]

        return h, ea_flat, intra_ei, subgraph_batch, root_flat_idx, node_ids_flat, valid, N_total

    def _run_layers(self, h, intra_ei, ea_flat, node_ids_flat, valid, N_total):
        """
        Run num_layers of MP-1 (intra-subgraph) + MP-2 (cross-subgraph).

        Args:
            h             [S*k, H]
            intra_ei      [2, E_total]
            ea_flat       [E_total, edge_dim] or None
            node_ids_flat [S*k]
            valid         [S*k] bool
            N_total       int

        Returns:
            h [S*k, H]
        """
        valid_f = valid.float().unsqueeze(-1)   # [S*k, 1]
        node_ids_clamped = node_ids_flat.clamp(min=0)

        for l in range(self.num_layers):
            h_res = h

            # MP-1: within-subgraph GIN
            h = self.mp1_convs[l](h, intra_ei)                    # [S*k, H]

            # MP-2: aggregate all copies of each global node v
            x_agg = scatter(
                h * valid_f,
                node_ids_clamped, dim=0,
                dim_size=N_total, reduce='mean'
            )                                                      # [N_total, H]
            x_cross = x_agg[node_ids_clamped]                     # [S*k, H]
            x_cross = x_cross * valid_f                            # zero padding

            h = F.relu(self.mp2_update[l](torch.cat([h, x_cross], dim=-1)))  # [S*k, H]

            # BN + residual + dropout
            h = self.bns[l](h)
            if self.residual:
                h = h + h_res
            if l < self.num_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        return h


# ---------------------------------------------------------------------------
# Graph-level encoder
# ---------------------------------------------------------------------------
class LPNFGraphEncoder(_LPNFBase):
    """
    LP-NF encoder for graph classification.

    Pipeline:
      1. Build initial node features with log-prob injection
      2. L layers of MP-1 (intra) + MP-2 (cross-subgraph)
      3. Subgraph pool: mean/add/max over nodes in each subgraph
      4. Graph pool: aggregate subgraph embeddings per graph

    Returns [G, H].
    """

    def __init__(self,
                 in_channels, edge_dim, hidden_dim,
                 num_layers=5, mlp_layers=2, dropout=0.1,
                 conv_type='gin', residual=True,
                 sub_pooling='mean', aggregator='mean', temperature=0.5):
        super().__init__(in_channels, edge_dim, hidden_dim, num_layers,
                         mlp_layers, dropout, conv_type, residual)

        if sub_pooling == 'mean':
            self.sub_pool_fn = global_mean_pool
        elif sub_pooling in ('add', 'sum'):
            self.sub_pool_fn = global_add_pool
        elif sub_pooling == 'max':
            self.sub_pool_fn = global_max_pool
        else:
            raise ValueError(f"Unknown sub_pooling: {sub_pooling}")

        try:
            self.aggregator = get_aggregator(aggregator)(
                hidden_dim=hidden_dim, temperature=temperature)
        except Exception:
            if aggregator in ('sum', 'add'):
                self.aggregator = global_add_pool
            elif aggregator == 'mean':
                self.aggregator = global_mean_pool
            elif aggregator == 'max':
                self.aggregator = global_max_pool
            else:
                self.aggregator = global_mean_pool

    def forward(self, batch: SubgraphFeaturesBatch):
        device = batch.x.device
        S, k = batch.nodes_sampled.shape
        T = batch.target_nodes.size(0)
        m = S // T

        target_batch = torch.arange(T, device=device).repeat_interleave(m)  # [S]

        root_local = self._find_root_positions(
            batch.nodes_sampled, batch.target_nodes, target_batch)

        h, ea_flat, intra_ei, sub_batch, root_flat_idx, node_ids_flat, valid, N_total = \
            self._build_initial_features(
                batch.x, batch.edge_attr,
                batch.nodes_sampled, batch.log_probs,
                root_local, batch.edge_ptr,
                batch.edge_index_sampled, batch.edge_src_global)

        h = self._run_layers(h, intra_ei, ea_flat, node_ids_flat, valid, N_total)  # [S*k, H]

        # Subgraph pool: [S*k, H] → [S, H]
        sub_emb = self.sub_pool_fn(h, sub_batch)                  # [S, H]

        # Graph pool: [S, H] → [G, H]
        sample_ptr = batch.sample_ptr                              # [G+1]
        G = sample_ptr.size(0) - 1
        samples_per_graph = sample_ptr[1:] - sample_ptr[:-1]      # [G]
        log_probs = batch.log_probs                                # [S] or None

        if (samples_per_graph == 0).any():
            zero_emb = torch.zeros(1, sub_emb.size(1), device=device)
            new_sub_embs = []
            new_log_probs = []
            new_graph_ids = []
            for g in range(G):
                ns = samples_per_graph[g].item()
                if ns > 0:
                    s, e = sample_ptr[g].item(), sample_ptr[g + 1].item()
                    new_sub_embs.append(sub_emb[s:e])
                    if log_probs is not None:
                        new_log_probs.append(log_probs[s:e])
                    new_graph_ids.extend([g] * ns)
                else:
                    new_sub_embs.append(zero_emb)
                    if log_probs is not None:
                        new_log_probs.append(torch.tensor([float('-inf')], device=device))
                    new_graph_ids.append(g)
            sub_emb = torch.cat(new_sub_embs, dim=0)
            if log_probs is not None:
                log_probs = torch.cat(new_log_probs, dim=0)
            global_graph_ptr = torch.tensor(new_graph_ids, dtype=torch.long, device=device)
        else:
            global_graph_ptr = torch.repeat_interleave(
                torch.arange(G, device=device), samples_per_graph)

        if getattr(self.aggregator, 'needs_log_probs', False):
            return self.aggregator(sub_emb, global_graph_ptr, log_probs=log_probs)
        return self.aggregator(sub_emb, global_graph_ptr)          # [G, H]


# ---------------------------------------------------------------------------
# Node-level encoder
# ---------------------------------------------------------------------------
class LPNFNodeEncoder(_LPNFBase):
    """
    LP-NF encoder for node classification and link prediction.

    Pipeline:
      1. Build initial node features with log-prob injection
      2. L layers of MP-1 (intra) + MP-2 (cross-subgraph)
      3. Scatter: aggregate all copies of each global node → [N_total, H]

    Returns [N_total, H].
    """

    def forward(self, batch: SubgraphFeaturesBatch):
        device = batch.x.device
        S, k = batch.nodes_sampled.shape
        T = batch.target_nodes.size(0)
        m = S // T

        target_batch = torch.arange(T, device=device).repeat_interleave(m)  # [S]

        root_local = self._find_root_positions(
            batch.nodes_sampled, batch.target_nodes, target_batch)

        h, ea_flat, intra_ei, sub_batch, root_flat_idx, node_ids_flat, valid, N_total = \
            self._build_initial_features(
                batch.x, batch.edge_attr,
                batch.nodes_sampled, batch.log_probs,
                root_local, batch.edge_ptr,
                batch.edge_index_sampled, batch.edge_src_global)

        h = self._run_layers(h, intra_ei, ea_flat, node_ids_flat, valid, N_total)  # [S*k, H]

        # Scatter all copies of each node → mean embedding
        valid_f = valid.float().unsqueeze(-1)
        node_emb = scatter(
            h * valid_f,
            node_ids_flat.clamp(min=0), dim=0,
            dim_size=N_total, reduce='mean'
        )                                                          # [N_total, H]
        return node_emb


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------
@register_model('ARCH-2')
def build_arch2(cfg: ExperimentConfig):
    kw = cfg.model_config.kwargs
    is_node_level = cfg.task in ('Node-Classification', 'Link-Prediction')
    base_kwargs = dict(
        in_channels=cfg.model_config.node_feature_dim,
        edge_dim=cfg.model_config.edge_feature_dim,
        hidden_dim=cfg.model_config.hidden_dim,
        num_layers=cfg.model_config.mpnn_layers,
        dropout=cfg.model_config.dropout,
        conv_type=cfg.model_config.mpnn_type,
        residual=kw.get('residual', True),
    )
    if is_node_level:
        return LPNFNodeEncoder(**base_kwargs)
    else:
        return LPNFGraphEncoder(
            **base_kwargs,
            sub_pooling=cfg.model_config.subgraph_param.pooling,
            aggregator=kw.get('aggregator', 'mean'),
            temperature=cfg.model_config.temperature or 0.5,
        )
