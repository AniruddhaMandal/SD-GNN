"""
    Arch-2-v2: Log probability node encoder + Symmetric Subgraph GNN Layer 
        + Arbitary Readout + MLP
"""

from torch import nn

from gxl import SubgraphFeaturesBatch, ExperimentConfig

from typing import Literal

class LogProbNodeFeatureInitializer(nn.Module):
    """
    Adds log prpbabilities and root node identifier to node features.
    Args:
        H: [int] hidden dimension
        type: 'concat' or 'add'
    """
    def __init__(self, H: int, type: Literal['concat', 'add']):
        super(LogProbNodeFeatureInitializer, self).__init__()
        self.logP_proj = nn.Linear(1, H)
        self.root_proj = nn.Linear(2, H) # root indicator 0/1

    def forward(self, sf: SubgraphFeaturesBatch):
        lp = self.logP_proj(sf.log_probs)
        rt = self.root_proj()
        
        pass

class SubgraphGNNLayer(nn.Module):
    """
    Symmetric message passing layer that passes messages both insubgraph and out-of-subgraph.
    Forward Pass: [in_dim, N_global] -> [out_dim, N_global]
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        hidden_dim: int,
        mlp_layers: int,
        dropout: float,
        conv_type: str,
        pooling: str,
    ):
        super(SubgraphGNNLayer, self).__init__()


class SubgraphGNNEncoder(nn.Module):
    """
    Encodes every node of every subgraph with symmetric message passing (insubgraph and out-of-subgraph 
    message passing).
    Forward Pass: [in_dim, N_global] -> [out_dim, N_global]"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        mlp_layers: int,
        dropout: float,
        conv_type: str,
        pooling: str,
    ):
        super(SubgraphGNNEncoder, self).__init__()

class SSGNNGraphEncoder(nn.Module):
    """
        Generates graph representation from sampled subgraphs using symmetric message passing and 
        arbitrary readout.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        mlp_layers: int,
        dropout: float,
        conv_type: str,
        pooling: str,
    ):
        super(SSGNNGraphEncoder, self).__init__()

class SSGNNNodeEncoder(nn.Module):
    """
        Generates node representations from sampled subgraphs. Encode node features of subgraphs 
        with `SubgraphGNNEncoder`. Node v gets encoding by pooling from the represntations of v in 
        all subgraphs rooted at v.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        mlp_layers: int,
        dropout: float,
        conv_type: str,
        pooling: str,
    ):
        super(SSGNNNodeEncoder, self).__init__()
