# Copied from SS-GNN (github.com/YOURUSERNAME/SS-GNN) — see LICENSE
import random
from typing import Optional, Tuple, List
import itertools

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class K4ParityDataset(Dataset):
    """
    Generate graphs with two classes:
      - Class 0: ER(n,p) then remove one edge from every K4 (so no K4 remains).
      - Class 1: Start from a class-0 graph and add `num_cliques` random K4s (make those 4-sets complete).

    Each item is a torch_geometric.data.Data with:
        .edge_index (2, E) (bidirectional),
        .num_nodes (int),
        .y (label: 0 or 1),
        .k4_count (tensor([k]) number of K4s in the generated graph)

    Parameters
    ----------
    num_graphs : int
        Total number of graphs to create (by default half Class0 half Class1).
    node_range : Tuple[int,int]
        Inclusive range (min_n, max_n) for number of nodes; n is sampled uniformly per graph.
    p : float
        Edge probability for ER graph (used for class0 base).
    num_cliques : int
        Number of K4s to insert for each class-1 example.
    class0_count : Optional[int]
        If provided, number of class0 graphs. Otherwise uses floor(num_graphs/2).
    class1_count : Optional[int]
        If provided, number of class1 graphs. Otherwise uses num_graphs - class0_count.
    seed : Optional[int]
        Random seed for reproducibility.
    max_attempts : int
        Safety bound for operations that might loop (not usually needed here).
    store_on_device : bool
        If True move edge_index tensors to `device` (device is cpu/cuda).
    device : str
        Device for numeric ops (mainly used if you later add GPU acceleration).
    """

    def __init__(
        self,
        num_graphs: int,
        node_range: Tuple[int, int] = (8, 14),
        p: float = 0.2,
        num_cliques: int = 2,
        class0_count: Optional[int] = None,
        class1_count: Optional[int] = None,
        seed: Optional[int] = None,
        max_attempts: int = 100000,
        store_on_device: bool = False,
        device: str = "cpu",
    ):
        assert 0.0 <= p <= 1.0
        assert len(node_range) == 2
        min_n, max_n = int(node_range[0]), int(node_range[1])
        assert 4 <= min_n <= max_n, "node_range min must be >=4 (to have 4-node subsets)"

        self.num_graphs = int(num_graphs)
        self.min_n = min_n
        self.max_n = max_n
        self.p = float(p)
        self.num_cliques = int(num_cliques)
        self.max_attempts = int(max_attempts)
        self.device = torch.device(device)
        self.store_on_device = bool(store_on_device)

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        if class0_count is None:
            class0_count = self.num_graphs // 2
        if class1_count is None:
            class1_count = self.num_graphs - class0_count

        self.class0_count = int(class0_count)
        self.class1_count = int(class1_count)
        assert self.class0_count + self.class1_count == self.num_graphs

        self.graphs: List[Data] = []
        self._generate_all()

    # ----------------- low level helpers -----------------

    def _sample_erdos_renyi_adj(self, n: int) -> torch.Tensor:
        """Return symmetric adjacency (n,n) uint8, zero diagonal."""
        probs = torch.rand((n, n))
        upper = torch.triu((probs < self.p).to(torch.uint8), diagonal=1)
        adj = upper + upper.t()
        return adj

    def _adj_to_edge_index(self, adj: torch.Tensor) -> torch.LongTensor:
        """Return bidirectional edge_index (2, E) as LongTensor."""
        upper = torch.triu(adj, diagonal=1)
        idx = torch.nonzero(upper, as_tuple=False)
        if idx.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long)
        i = idx[:, 0]
        j = idx[:, 1]
        row = torch.cat([i, j], dim=0).long()
        col = torch.cat([j, i], dim=0).long()
        return torch.stack([row, col], dim=0)

    def _is_k4(self, adj: torch.Tensor, nodes4: Tuple[int, int, int, int]) -> bool:
        """Return True if the induced subgraph on nodes4 is K4."""
        i, j, k, l = nodes4
        # list all 6 unordered pairs
        pairs = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
        for (u, v) in pairs:
            if adj[u, v].item() == 0:
                return False
        return True

    def _count_k4s(self, adj: torch.Tensor) -> int:
        """Count number of K4s by brute force enumeration of 4-node subsets."""
        n = adj.size(0)
        cnt = 0
        for comb in itertools.combinations(range(n), 4):
            if self._is_k4(adj, comb):
                cnt += 1
        return cnt

    # ----------------- class 0 generation -----------------

    def _remove_one_edge_from_k4(self, adj: torch.Tensor, nodes4: Tuple[int, int, int, int]) -> None:
        """
        Remove one random edge among the 6 edges of the 4-set to break K4.
        Modifies adj in-place.
        """
        i, j, k, l = nodes4
        pairs = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
        # choose a pair that currently has an edge (should be all 6 in K4 case)
        existing = [p for p in pairs if adj[p[0], p[1]].item() == 1]
        if not existing:
            return  # nothing to remove (defensive)
        u, v = random.choice(existing)
        adj[u, v] = 0
        adj[v, u] = 0

    def _make_class0(self, n: int) -> torch.Tensor:
        """
        Create a class-0 adjacency: start ER(n,p) and remove one edge from every K4 found.
        After completion, there should be 0 K4s.
        """
        adj = self._sample_erdos_renyi_adj(n)
        # Enumerate all 4-subsets and remove one edge if K4
        # Note: because removing edges may cause earlier K4 tests to fail for later subsets,
        # we simply iterate combinations once in deterministic order.
        for comb in itertools.combinations(range(n), 4):
            if self._is_k4(adj, comb):
                self._remove_one_edge_from_k4(adj, comb)
        # verification: ensure no K4 remains
        remaining = self._count_k4s(adj)
        if remaining != 0:
            # extremely unlikely, but try again by scanning until none remain (safety)
            # This loop is defensive; the above should remove all K4s.
            attempts = 0
            while remaining != 0 and attempts < self.max_attempts:
                # find one K4 (first encountered) and remove one edge
                for comb in itertools.combinations(range(n), 4):
                    if self._is_k4(adj, comb):
                        self._remove_one_edge_from_k4(adj, comb)
                        break
                remaining = self._count_k4s(adj)
                attempts += 1
            if remaining != 0:
                raise RuntimeError("Failed to eliminate all K4s after max attempts.")
        return adj

    # ----------------- class 1 generation -----------------

    def _make_class1_from_class0(self, adj0: torch.Tensor, num_cliques: int) -> torch.Tensor:
        """
        Given a K4-free adj0, return a new adjacency where `num_cliques` 4-subsets
        are turned into K4s by adding all 6 edges for each subset.
        """
        n = adj0.size(0)
        adj = adj0.clone()
        all_subsets = list(itertools.combinations(range(n), 4))
        # If num_cliques is larger than number of subsets, sample with replacement
        if num_cliques <= len(all_subsets):
            chosen = random.sample(all_subsets, num_cliques)
        else:
            # sample with replacement
            chosen = [random.choice(all_subsets) for _ in range(num_cliques)]

        for comb in chosen:
            i, j, k, l = comb
            pairs = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
            for (u, v) in pairs:
                adj[u, v] = 1
                adj[v, u] = 1
        return adj

    # ----------------- top-level generation -----------------

    def _generate_one_pair(self) -> List[Data]:
        """
        Generate one class0 graph and one class1 derived graph.
        Returns [Data_class0, Data_class1].
        """
        n = random.randint(self.min_n, self.max_n)
        adj0 = self._make_class0(n)
        k4_0 = self._count_k4s(adj0)
        assert k4_0 == 0, "Class0 graph unexpectedly contains K4 after processing."

        # create class0 Data
        ei0 = self._adj_to_edge_index(adj0)
        if self.store_on_device:
            ei0 = ei0.to(self.device)
        data0 = Data(
            edge_index=ei0,
            num_nodes=n,
            y=torch.tensor([0], dtype=torch.long),
            k4_count=torch.tensor([0], dtype=torch.long),
        )

        # create class1 by injecting K4s
        adj1 = self._make_class1_from_class0(adj0, self.num_cliques)
        k4_1 = self._count_k4s(adj1)
        ei1 = self._adj_to_edge_index(adj1)
        if self.store_on_device:
            ei1 = ei1.to(self.device)
        data1 = Data(
            edge_index=ei1,
            num_nodes=n,
            y=torch.tensor([1], dtype=torch.long),
            k4_count=torch.tensor([k4_1], dtype=torch.long),
        )
        return [data0, data1]

    def _generate_all(self):
        # Strategy: generate class0_count class-0 graphs and class1_count class-1 graphs derived from them.
        # If imbalance, we create extra class0/class1 as needed.
        created0 = 0
        created1 = 0
        # We'll produce pairs until one of counts is satisfied; then produce remaining singles as needed.
        while created0 < self.class0_count or created1 < self.class1_count:
            # produce one pair (class0 + its class1)
            pair = self._generate_one_pair()
            if created0 < self.class0_count:
                self.graphs.append(pair[0])
                created0 += 1
            if created1 < self.class1_count:
                self.graphs.append(pair[1])
                created1 += 1
        # if we created slightly more because of pair ordering, trim to exact num_graphs
        if len(self.graphs) > self.num_graphs:
            self.graphs = self.graphs[: self.num_graphs]

    # ----------------- Dataset API -----------------

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]
