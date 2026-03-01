import math
from typing import overload
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import (
    GINEConv, GINConv, GCNConv, SAGEConv, GATv2Conv, SGConv, GCN2Conv, PNAConv,
    global_mean_pool, global_add_pool, global_max_pool, global_sort_pool
)
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import scatter
from gxl import SubgraphFeaturesBatch
from gxl import aggregator
from gxl.registry import get_aggregator


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


class SubgraphGNNEncoder(nn.Module):
    '''
        Standard GNN encoder for individual subgraphs.
        Runs message-passing on batched subgraph nodes and pools each
        subgraph into a single embedding vector.
    '''
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 edge_dim: int,
                 hidden_dim: int,
                 num_layers: int = 5,
                 mlp_layers: int = 4,
                 dropout: float = 0.1,
                 conv_type: str = "gine",
                 pooling: str = "mean",
                 res_connect: bool = True,
                 batch_norm: bool = True,
                 # GCNII parameters
                 gcnii_alpha: float = 0.1,
                 gcnii_theta: float = 0.5,
                 # PNA parameters
                 deg: torch.Tensor = None):

        super().__init__()
        conv_type = conv_type.lower()
        assert conv_type in {'gine', 'gin', 'gcn', 'sage', 'gatv2', 'sgc', 'gcnii', 'pna'}
        self.conv_type = conv_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.out_channels = out_channels
        self.gcnii_alpha = gcnii_alpha
        self.gcnii_theta = gcnii_theta
        if conv_type == 'pna' and deg is None:
            deg = torch.ones(128, dtype=torch.long)
        self.deg = deg
        if pooling == 'mean':
            self.pooling_fn = global_mean_pool
        elif pooling in ['add', 'sum']:
            self.pooling_fn = global_add_pool
        elif pooling == 'max':
            self.pooling_fn = global_max_pool
        else:
            raise ValueError(f"unknown value of subgraph pooling: {pooling}")
        self.res_connect = res_connect
        self.batch_norm = batch_norm

        self.node_proj = nn.Linear(in_channels, hidden_dim)

        # If using edge features (only GINE uses them directly)
        self.use_edges = (conv_type == 'gine')
        if self.use_edges:
            # Project raw edge_attr -> hidden_dim once (works for all layers)
            self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # Build layers
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for layer_idx in range(num_layers):
            self.convs.append(self._make_conv(hidden_dim, hidden_dim, mlp_layers, layer_idx=layer_idx))
            if self.batch_norm:
                self.bns.append(BatchNorm(hidden_dim))

    def _make_conv(self, in_dim, out_dim, mlp_layers, layer_idx=0):
        if self.conv_type == 'gine':
            mlp = make_mlp(in_dim, in_dim, out_dim, num_layers=mlp_layers)
            return GINEConv(nn=mlp, train_eps=True)  # we project edge_attr ourselves
        if self.conv_type == 'gin':
            mlp = make_mlp(in_dim, in_dim, out_dim, num_layers=mlp_layers)
            return GINConv(nn=mlp, train_eps=True)
        if self.conv_type == 'gcn':
            return GCNConv(in_dim, out_dim, cached=False, normalize=True)
        if self.conv_type == 'sage':
            return SAGEConv(in_dim, out_dim)
        if self.conv_type == 'gatv2':
            return GATv2Conv(in_dim, out_dim, heads=1, concat=True)  # keeps dim = out_dim
        if self.conv_type == 'sgc':
            return SGConv(in_dim, out_dim, K=1)
        if self.conv_type == 'gcnii':
            return GCN2Conv(out_dim, alpha=self.gcnii_alpha, theta=self.gcnii_theta,
                            layer=layer_idx + 1, shared_weights=True, cached=False, normalize=True)
        if self.conv_type == 'pna':
            return PNAConv(in_dim, out_dim,
                           aggregators=['mean', 'min', 'max', 'std'],
                           scalers=['identity', 'amplification', 'attenuation'],
                           deg=self.deg)
        raise ValueError(f"Unknown conv_type: {self.conv_type}")

    # ---------- FORWARD ----------
    def forward(self,
                x:torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor,
                edge_attr: torch.Tensor=None)->torch.Tensor:
        """
        SubgraphGNNEncoder forward pass.
        Args:
            x: node attribute
            edge_index: based on global edge index
            batch: node's parent subgraph index
            edge_attr: edge attribute
        Returns:
            [torch.Tensor]: encoding of each subgraph
        """
        h = self.node_proj(x)
        h_0 = h  # Initial features for GCNII
        if self.use_edges:
            if edge_attr is None:
                raise ValueError("edge_attr is required for conv_type='gine'.")
            e = self.edge_proj(edge_attr)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h_res = h
            if self.conv_type == 'gcnii':
                h = conv(h, h_0, edge_index)
            elif self.use_edges:
                h = conv(h, edge_index, e)
            else:
                h = conv(h, edge_index)
            if self.batch_norm:
                h = bn(h)
            h = F.relu(h)
            if self.res_connect:
                h = h + h_res
            if i < self.num_layers - 1:  # Skip dropout on last layer
                h = F.dropout(h, p=self.dropout, training=self.training)

        g = self.pooling_fn(h, batch)

        return g


# ---------------------------------------------------------------------------
# Shared mixin: subgraph encoding from sampled batch data
# ---------------------------------------------------------------------------
class _SubgraphEncodeMixin:
    """Shared encode_subgraphs logic used by both graph-level and node-level encoders."""

    def encode_subgraphs(
        self,
        x,                 # (N, IN_dim)
        edge_attr,         # (E_global, E_dim)
        nodes_t,           # (B_total, k)  global node ids; -1 padded
        edge_index_t,      # (2, E_total)  in local/sample mode(in (0,1,..k-1)), concatenated
        edge_ptr_t,        # (B_total+1,)
        edge_src_global_t  # (E_total,) index of edges from `edge_index_t` in batch.edge_index
    ):
        """
        Returns:
            sample_emb: (B_total, H_dim)
        """
        device = x.device
        num_subgraphs, k = nodes_t.shape

        # gather global node attrib
        stacked_nodes = nodes_t.flatten()

        # Handle -1 padding: clamp to 0 and create mask
        valid_mask = (stacked_nodes >= 0)
        stacked_nodes_clamped = stacked_nodes.clamp(min=0)

        global_x = x[stacked_nodes_clamped]

        # Zero out features for padded positions
        if not valid_mask.all():
            global_x = global_x * valid_mask.unsqueeze(-1).float()

        # gather global edge atrribute
        if edge_attr is not None:
            global_edge_attr = edge_attr[edge_src_global_t]
        else:
            global_edge_attr = None

        # convert edge index to (subgraph)batch level
        global_edge_index = torch.repeat_interleave(torch.arange(0,num_subgraphs,device=device),edge_ptr_t[1:]-edge_ptr_t[:-1])*k
        global_edge_index = global_edge_index + edge_index_t

        # global batch pointer
        global_batch_ptr = torch.repeat_interleave(torch.arange(0,num_subgraphs,device=device),k)

        # send to device
        x = self.encoder(x=global_x,edge_attr=global_edge_attr,edge_index=global_edge_index,batch=global_batch_ptr)

        return x


# ---------------------------------------------------------------------------
# Graph-level encoder: subgraphs → graph embeddings [G, H]
# ---------------------------------------------------------------------------
class SSGNNGraphEncoder(nn.Module, _SubgraphEncodeMixin):
    """
    Graph-level SS-GNN encoder.

    Encodes sampled subgraphs with a GNN backbone, then aggregates all
    subgraph representations per graph into a single graph embedding.

    Returns:
        [G, H] graph-level embeddings.
    """
    def __init__(self,
                 in_channels: int,
                 edge_dim: int,
                 hidden_dim: int,
                 num_layers: int = 5,
                 mlp_layers: int = 2,
                 dropout: float = 0.1,
                 conv_type: str = 'gine',
                 aggregator: str = 'mean',
                 temperature: float = 0.5,
                 pooling: str = 'mean'):

        super().__init__()

        self.encoder = SubgraphGNNEncoder(in_channels=in_channels,
                                          out_channels=hidden_dim,
                                          hidden_dim=hidden_dim,
                                          edge_dim=edge_dim,
                                          num_layers=num_layers,
                                          conv_type=conv_type,
                                          mlp_layers=mlp_layers,
                                          dropout=dropout,
                                          pooling=pooling)

        # set aggregator
        try:
            self.aggregator = get_aggregator(aggregator)\
                (hidden_dim=hidden_dim, temperature=temperature)
        except:
            if aggregator in ['sum', 'add']:
                self.aggregator = global_add_pool
            if aggregator == 'mean':
                self.aggregator = global_mean_pool
            if aggregator == 'max':
                self.aggregator = global_max_pool

    def forward(self, batch: SubgraphFeaturesBatch):
        """
        Returns:
            graph_emb: [G, H] graph-level embeddings.
        """
        sample_emb = self.encode_subgraphs(
            x=batch.x,
            edge_attr=batch.edge_attr,
            nodes_t=batch.nodes_sampled,
            edge_index_t=batch.edge_index_sampled,
            edge_ptr_t=batch.edge_ptr,
            edge_src_global_t=batch.edge_src_global)  # [B_total, H]

        device = sample_emb.device
        sample_ptr_t = batch.sample_ptr
        num_graphs = sample_ptr_t.size(0) - 1
        samples_per_graph = sample_ptr_t[1:] - sample_ptr_t[:-1]
        log_probs = batch.log_probs  # [B_total] or None

        if (samples_per_graph == 0).any():
            zero_emb = torch.zeros(1, sample_emb.size(1), device=device)
            new_sample_embs = []
            new_log_probs = []
            new_graph_ids = []

            for g in range(num_graphs):
                num_samples = samples_per_graph[g].item()
                if num_samples > 0:
                    start_idx = sample_ptr_t[g]
                    end_idx = sample_ptr_t[g + 1]
                    new_sample_embs.append(sample_emb[start_idx:end_idx])
                    if log_probs is not None:
                        new_log_probs.append(log_probs[start_idx:end_idx])
                    new_graph_ids.extend([g] * num_samples)
                else:
                    new_sample_embs.append(zero_emb)
                    if log_probs is not None:
                        new_log_probs.append(torch.tensor([float('-inf')], device=device))
                    new_graph_ids.append(g)

            sample_emb = torch.cat(new_sample_embs, dim=0)
            if log_probs is not None:
                log_probs = torch.cat(new_log_probs, dim=0)
            global_graph_ptr = torch.tensor(new_graph_ids, dtype=torch.long, device=device)
        else:
            global_graph_ptr = torch.repeat_interleave(
                torch.arange(0, num_graphs, device=device), samples_per_graph)

        if getattr(self.aggregator, 'needs_log_probs', False):
            return self.aggregator(sample_emb, global_graph_ptr, log_probs=log_probs)
        return self.aggregator(sample_emb, global_graph_ptr)  # [G, H]


# ---------------------------------------------------------------------------
# Node-level encoder: subgraphs → per-node embeddings [N_total, H]
# ---------------------------------------------------------------------------
class SSGNNNodeEncoder(nn.Module, _SubgraphEncodeMixin):
    """
    Node-level SS-GNN encoder for node classification and link prediction.

    Encodes sampled subgraphs with a GNN backbone, then aggregates
    the m subgraph representations centered on each target node into
    a per-node embedding using the same aggregator interface as the
    graph-level encoder.  Non-target nodes receive zero embeddings
    (they are masked out downstream).

    Requires ``batch.target_nodes`` and ``batch.target_ptr`` to be set
    (populated by ``_build_node_targets`` or ``_build_link_targets``
    in experiment.py).

    Returns:
        [N_total, H] per-node embeddings.
    """
    def __init__(self,
                 in_channels: int,
                 edge_dim: int,
                 hidden_dim: int,
                 num_layers: int = 5,
                 mlp_layers: int = 2,
                 dropout: float = 0.1,
                 conv_type: str = 'gine',
                 aggregator: str = 'mean',
                 temperature: float = 0.5,
                 pooling: str = 'mean'):

        super().__init__()

        self.encoder = SubgraphGNNEncoder(in_channels=in_channels,
                                          out_channels=hidden_dim,
                                          hidden_dim=hidden_dim,
                                          edge_dim=edge_dim,
                                          num_layers=num_layers,
                                          conv_type=conv_type,
                                          mlp_layers=mlp_layers,
                                          dropout=dropout,
                                          pooling=pooling)

        # Set aggregator — same pattern as SSGNNGraphEncoder
        try:
            self.aggregator = get_aggregator(aggregator)\
                (hidden_dim=hidden_dim, temperature=temperature)
        except:
            if aggregator in ['sum', 'add']:
                self.aggregator = global_add_pool
            if aggregator == 'mean':
                self.aggregator = global_mean_pool
            if aggregator == 'max':
                self.aggregator = global_max_pool

    def forward(self, batch: SubgraphFeaturesBatch):
        """
        Returns:
            node_emb: [N_total, H] per-node embeddings.
        """
        sample_emb = self.encode_subgraphs(
            x=batch.x,
            edge_attr=batch.edge_attr,
            nodes_t=batch.nodes_sampled,
            edge_index_t=batch.edge_index_sampled,
            edge_ptr_t=batch.edge_ptr,
            edge_src_global_t=batch.edge_src_global)  # [B_total, H]

        device = sample_emb.device
        H = sample_emb.size(1)
        N_total = batch.x.size(0)

        target_nodes = batch.target_nodes   # [T] global node IDs
        target_ptr = batch.target_ptr       # [G+1]
        sample_ptr = batch.sample_ptr       # [G+1]
        log_probs = batch.log_probs         # [B_total] or None
        G = sample_ptr.size(0) - 1
        T = target_nodes.size(0)

        # Infer m: total samples / total targets
        total_samples = sample_ptr[-1].item()
        m = total_samples // T if T > 0 else 1

        # Build target_batch: maps each sample → target index [0..T-1]
        # Samples are m-contiguous per target within each graph's block.
        target_batch_parts = []
        for g in range(G):
            t_start = target_ptr[g].item()
            t_end = target_ptr[g + 1].item()
            T_g = t_end - t_start
            if T_g == 0:
                continue
            # T_g targets, m samples each → repeat each target index m times
            target_batch_parts.append(
                torch.arange(t_start, t_end, device=device).repeat_interleave(m))

        if not target_batch_parts:
            return sample_emb.new_zeros(N_total, H)

        target_batch = torch.cat(target_batch_parts)  # [B_total]

        # Aggregate m subgraphs per target using the registered aggregator
        if getattr(self.aggregator, 'needs_log_probs', False):
            target_embs = self.aggregator(sample_emb, target_batch, log_probs=log_probs)
        else:
            target_embs = self.aggregator(sample_emb, target_batch)  # [T, H]

        # Scatter target embeddings into full node tensor (differentiable)
        node_emb = scatter(target_embs, target_nodes, dim=0,
                           dim_size=N_total, reduce='sum')
        return node_emb


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------
SubgraphSamplingGNNClassifier = SSGNNGraphEncoder
