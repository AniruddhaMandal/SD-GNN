import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.transforms import Compose, ToUndirected
from torch_geometric.datasets import TUDataset

class ClipOneHotDegree:
    def __init__(self, max_degree, cat=False):
        self.max_degree = max_degree
        self.cat = cat
    def __call__(self, data):
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes).to(torch.long)
        deg = deg.clamp_max(self.max_degree)  # overflow → last bin
        x_oh = F.one_hot(deg, num_classes=self.max_degree + 1).float()
        data.x = x_oh if not self.cat else torch.cat([data.x, x_oh], dim=-1)
        return data

class ClipDegreeEmbed:
    def __init__(self, max_degree: int, embed_dim: int, cat: bool = False):
        """
        max_degree: int
            Maximum degree bucket. Degrees above this are clipped.
        embed_dim: int
            Dimension of learnable degree embedding.
        cat: bool
            If True, concatenate embedding to existing node features.
            If False, replace existing node features with embedding.
        """
        self.max_degree = max_degree
        self.embed_dim = embed_dim
        self.cat = cat

        # Create the learnable embedding table
        self.embedding = torch.nn.Embedding(max_degree + 1, embed_dim)
        self.embedding.weight.requires_grad_(False)

    def __call__(self, data):
        # Compute node degree
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes).to(torch.long)
        deg = deg.clamp_max(self.max_degree)  # overflow → last bin

        # Lookup learnable embedding for each degree
        deg_emb = self.embedding(deg)

        # Assign to data.x
        if self.cat and data.x is not None:
            data.x = torch.cat([data.x, deg_emb], dim=-1)
        else:
            data.x = deg_emb
        return data

class SetNodeFeaturesOnes:
    def __init__(self, dim: int, cat: bool = False):
        """
        Set node features to all ones with specified dimension.

        Automatically handles dimension mismatches from caching by always
        ensuring the output has the correct dimension.

        Parameters
        ----------
        dim: int
            Dimension of the node features (all ones vector).
        cat: bool
            If True, concatenate ones to existing node features.
            If False, replace existing node features with ones.
        """
        self.dim = dim
        self.cat = cat

    def __call__(self, data):
        num_nodes = data.num_nodes
        ones_features = torch.ones((num_nodes, self.dim), dtype=torch.float)

        # Always replace with correct dimension when not concatenating
        # This handles caching issues where cached features may have wrong dim
        if not self.cat:
            data.x = ones_features
        elif data.x is not None:
            data.x = torch.cat([data.x, ones_features], dim=-1)
        else:
            data.x = ones_features

        return data

class AddLaplacianPE:
    def __init__(self, k: int = 8, cat: bool = True):
        """
        Add Laplacian Positional Encoding to node features.

        Computes the k smallest non-trivial eigenvectors of the graph Laplacian
        and uses them as positional features.

        Parameters
        ----------
        k: int
            Number of eigenvectors to use (default: 8)
        cat: bool
            If True, concatenate PE to existing node features.
            If False, replace existing node features with PE.
        """
        self.k = k
        self.cat = cat

    def __call__(self, data):
        from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
        import scipy.sparse as sp
        import numpy as np

        num_nodes = data.num_nodes
        edge_index = data.edge_index

        # Compute Laplacian
        edge_index_lap, edge_weight_lap = get_laplacian(
            edge_index, normalization='sym', num_nodes=num_nodes
        )

        # Convert to scipy sparse matrix
        L = to_scipy_sparse_matrix(edge_index_lap, edge_weight_lap, num_nodes)

        # Compute eigenvectors
        # We want k smallest eigenvalues (excluding the trivial 0 eigenvalue)
        try:
            eigenvalues, eigenvectors = sp.linalg.eigsh(
                L, k=self.k + 1, which='SM', return_eigenvectors=True
            )
            # Skip the first eigenvector (corresponding to eigenvalue ~0)
            pe = eigenvectors[:, 1:self.k + 1]
        except:
            # Fallback if eigsh fails: use random features
            pe = np.random.randn(num_nodes, self.k) * 0.01

        pe = torch.from_numpy(pe).float()

        # Concatenate or replace
        if self.cat and data.x is not None:
            data.x = torch.cat([data.x, pe], dim=-1)
        else:
            data.x = pe

        return data


if __name__ == "__main__":
    transforms = Compose([ToUndirected(), ClipOneHotDegree(max_degree=512, cat=False)])  # pick a roomy cap
    dataset = TUDataset(root="data/TUDataset", name="COLLAB", transform=transforms)
