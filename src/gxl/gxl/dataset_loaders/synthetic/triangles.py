# Copied from SS-GNN (github.com/YOURUSERNAME/SS-GNN) — see LICENSE
import random
from typing import Optional, Tuple, List, Callable

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from tqdm import tqdm


class ParityTriangleGraphDataset(Dataset):
    """
    Dataset that generates graphs with a desired triangle-count parity.

    For each graph:
      1. Choose n ~ Uniform(min_n, max_n) (inclusive).
      2. Sample G ~ Erdos-Renyi(n, p).
      3. Count triangles (trace(A^3)/6).
      4. While triangle_count % 2 != desired_parity:
           - Toggle a random undirected edge (add/remove)
           - Recount triangles
      5. Store edge_index (COO 2 x E, both directions), num_nodes, triangle_count, label.

    __getitem__(i) -> PyG_Data(edge_index: torch.LongTensor(2, E),
                      num_nodes: int,
                      triangle_count: int,
                      label: torch.LongTensor(()) )  # 0 or 1
    """

    def __init__(
        self,
        num_graphs: int,
        node_range: Tuple[int, int],
        p: float = 0.3,
        desired_parity: int = 0,
        seed: Optional[int] = None,
        device: str = "cpu",
        max_attempts_per_graph: int = 100000,
        store_on_device: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None
    ):
        assert desired_parity in (0, 1, [0,1])
        assert 0.0 <= p <= 1.0
        assert isinstance(node_range, (tuple, list)) and len(node_range) == 2
        min_n, max_n = int(node_range[0]), int(node_range[1])
        assert 1 <= min_n <= max_n, "node_range must satisfy 1 <= min_n <= max_n"

        self.num_graphs = int(num_graphs)
        self.min_n = min_n
        self.max_n = max_n
        self.p = float(p)
        self.desired_parity = int(desired_parity) if not isinstance(desired_parity,list) else desired_parity
        self.device = torch.device(device)
        self.max_attempts = int(max_attempts_per_graph)
        self.store_on_device = bool(store_on_device)
        self.transform = transform
        self.pre_transform = pre_transform

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # storage lists
        # each entry: dict with keys: 'edge_index', 'num_nodes', 'triangles', 'label'
        self.graphs: List[dict] = []

        # generate all graphs up-front
        self._generate_all_graphs()

    # ---------------------------
    # --- helper methods -------
    # ---------------------------
    def _sample_erdos_renyi_adj(self, n: int) -> torch.Tensor:
        """
        Return symmetric adjacency matrix (n x n) dtype=uint8 on CPU with zeros on diagonal.
        """
        probs = torch.rand((n, n))
        upper = torch.triu((probs < self.p).to(dtype=torch.uint8), diagonal=1)
        adj = upper + upper.t()
        return adj  # 0/1 uint8

    def _count_triangles(self, adj: torch.Tensor) -> int:
        """
        Count triangles using trace(A^3) / 6.
        adj: square 0/1 tensor (any dtype). Computation done on self.device with float64.
        """
        n = adj.shape[0]
        A = adj.to(dtype=torch.float64, device=self.device)
        A2 = A.matmul(A)
        A3 = A2.matmul(A)
        tri = torch.diagonal(A3).sum().item() / 6.0
        return int(round(tri))

    def _toggle_random_edge(self, adj: torch.Tensor) -> None:
        """
        Toggle a random undirected edge (i <-> j) in-place. adj on CPU expected.
        """
        n = adj.shape[0]
        i = random.randrange(n)
        j = random.randrange(n - 1)
        if j >= i:
            j += 1
        # toggle both symmetric positions
        if adj.dtype == torch.uint8:
            newval = 1 - adj[i, j].item()
            adj[i, j] = newval
            adj[j, i] = newval
        else:
            newval = 1.0 - float(adj[i, j].item())
            adj[i, j] = newval
            adj[j, i] = newval

    def _adj_to_edge_index(self, adj: torch.Tensor, add_both_directions: bool = True) -> torch.LongTensor:
        """
        Convert symmetric adjacency (uint8 or float 0/1) to edge_index (2, E).
        If add_both_directions=True, each undirected edge i<j appears as (i,j) and (j,i).
        If False, each undirected edge appears once with i<j ordering.
        """
        # find upper-triangular entries (i < j)
        n = adj.shape[0]
        # use triu to avoid duplicates, diagonal excluded
        upper = torch.triu(adj, diagonal=1)
        idx = torch.nonzero(upper, as_tuple=False)  # rows [num_edges, 2] with (i,j)
        if idx.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long)
        i = idx[:, 0].long()
        j = idx[:, 1].long()
        if add_both_directions:
            row = torch.cat([i, j], dim=0)
            col = torch.cat([j, i], dim=0)
            edge_index = torch.stack([row, col], dim=0)  # shape (2, 2*E)
        else:
            edge_index = torch.stack([i, j], dim=0)  # shape (2, E)
        return edge_index

    # ---------------------------
    # --- generation logic ------
    # ---------------------------
    def _generate_single_graph(self) -> Data:
        """
        Generate one graph and return a PyG Data with edge_index, num_nodes, triangles, label.
        """
        n = random.randint(self.min_n, self.max_n)
        adj = self._sample_erdos_renyi_adj(n)  # CPU uint8
        tri = self._count_triangles(adj)

        attempts = 0
        while ((tri % 2) != self.desired_parity) and (self.desired_parity != [0,1]):
            if attempts >= self.max_attempts:
                raise RuntimeError(
                    f"Exceeded max_attempts ({self.max_attempts}) while trying to fix parity for a graph with n={n}."
                )
            self._toggle_random_edge(adj)
            tri = self._count_triangles(adj)
            attempts += 1

        # convert to edge_index (COO). We'll return bidirectional edges (PyG-friendly).
        edge_index = self._adj_to_edge_index(adj, add_both_directions=True)  # LongTensor
        # optionally move storage to device to save future transfers
        if self.store_on_device:
            edge_index = edge_index.to(self.device)

        y = torch.tensor([tri % 2], dtype=torch.long)

        data = Data(
            edge_index=edge_index,
            num_nodes=n,
            triangles=torch.tensor([tri], dtype=torch.long),
            y=y,
        )
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        return data


    def _generate_all_graphs(self) -> None:

        for _ in tqdm(range(self.num_graphs),ncols=50, desc="[Triangle-Parity] building:"):
            g = self._generate_single_graph()
            self.graphs.append(g)

    # ---------------------------
    # --- Dataset interface -----
    # ---------------------------
    def __len__(self) -> int:
        return self.num_graphs

    def __getitem__(self, idx: int) -> Data:
        data = self.graphs[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
