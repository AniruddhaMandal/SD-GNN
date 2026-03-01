import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from gxl import SubgraphFeaturesBatch
from gxl.registry import get_aggregator
from gxl.models.ss_gnn import SubgraphGNNEncoder, _SubgraphEncodeMixin


class SDGNNEncoder(nn.Module, _SubgraphEncodeMixin):
    """
    Two-stage SD-GNN encoder.

    Stage 1: Encode m subgraphs per node → aggregate into per-node enrichment
             vectors → concatenate with original features → project to H dims.
    Stage 2: Run a full-graph MPNN on the enriched node features → global
             pooling → graph embedding [G, H].

    The mixin's encode_subgraphs() hard-codes ``self.encoder``, so Stage 1
    backbone MUST be stored as ``self.encoder``.  Stage 2 is ``self.mpnn``.
    """

    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int = 5,
        mlp_layers: int = 2,
        dropout: float = 0.1,
        conv_type: str = 'gin',
        pooling: str = 'mean',
        sub_pooling: str = 'mean',
        aggregator: str = 'mean',
        temperature: float = 0.5,
        sub_num_layers: int = None,
        sub_conv_type: str = None,
    ):
        super().__init__()

        # --- Stage 1 backbone (MUST be self.encoder for _SubgraphEncodeMixin) ---
        self.encoder = SubgraphGNNEncoder(
            in_channels=in_channels,
            out_channels=hidden_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=sub_num_layers if sub_num_layers is not None else num_layers,
            mlp_layers=mlp_layers,
            dropout=dropout,
            conv_type=sub_conv_type if sub_conv_type is not None else conv_type,
            pooling=sub_pooling,
        )

        # --- Per-node aggregator: m subgraph embs → 1 node emb ---
        try:
            self.aggregator = get_aggregator(aggregator)(
                hidden_dim=hidden_dim, temperature=temperature
            )
        except Exception:
            if aggregator in ('sum', 'add'):
                self.aggregator = global_add_pool
            elif aggregator == 'mean':
                self.aggregator = global_mean_pool
            elif aggregator == 'max':
                self.aggregator = global_max_pool
            else:
                self.aggregator = global_mean_pool

        # --- Injection projection: [F + H] → H ---
        self.node_proj = nn.Sequential(
            nn.Linear(in_channels + hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # --- Stage 2: full-graph MPNN (receives H-dim enriched features) ---
        self.mpnn = SubgraphGNNEncoder(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mlp_layers=mlp_layers,
            dropout=dropout,
            conv_type=conv_type,
            pooling=pooling,
        )

    def forward(self, batch: SubgraphFeaturesBatch) -> torch.Tensor:
        """
        Args:
            batch: SubgraphFeaturesBatch with target_nodes set to all nodes
                   (populated by _build_all_node_targets in experiment.py).

        Returns:
            graph_emb: [G, hidden_dim] graph-level embeddings.
        """
        N = batch.x.size(0)
        S = batch.nodes_sampled.size(0)  # N * m
        m = S // N

        # --- Stage 1: encode N*m subgraphs → [N*m, H] ---
        sample_emb = self.encode_subgraphs(
            batch.x,
            batch.edge_attr,
            batch.nodes_sampled,
            batch.edge_index_sampled,
            batch.edge_ptr,
            batch.edge_src_global,
        )

        # --- Aggregate m samples per node → [N, H] ---
        # target_batch[i] = which node (0..N-1) sample i belongs to
        target_batch = torch.arange(N, device=batch.x.device).repeat_interleave(m)

        if getattr(self.aggregator, 'needs_log_probs', False):
            node_emb = self.aggregator(sample_emb, target_batch, log_probs=batch.log_probs)
        else:
            node_emb = self.aggregator(sample_emb, target_batch)  # [N, H]

        # --- Inject enrichment into node features → [N, H] ---
        x_enriched = self.node_proj(torch.cat([batch.x, node_emb], dim=-1))

        # --- Stage 2: full-graph MPNN + pooling → [G, H] ---
        return self.mpnn(
            x=x_enriched,
            edge_index=batch.edge_index,
            batch=batch.batch,
            edge_attr=batch.edge_attr,
        )
