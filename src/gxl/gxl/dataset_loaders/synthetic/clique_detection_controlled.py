# Copied from SS-GNN (github.com/YOURUSERNAME/SS-GNN) — see LICENSE
"""
Density-Controlled Clique Detection Dataset

Key difference from original: Both classes have SAME expected number of edges.
This forces models to detect the clique structure itself, not density as a proxy.

Class 0 (no clique):  Higher p, no clique planted
Class 1 (has clique): Lower p, but with planted k-clique

Result: Same edge count on average, but different structure.
- Vanilla GNN should struggle (1-WL cannot detect cliques)
- SS-GNN should succeed (k+1 subgraphs can detect k-cliques)
"""

import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx
from typing import Optional, Tuple, Callable
from itertools import combinations
from tqdm import tqdm


class DensityControlledCliqueDetectionDataset(Dataset):
    """
    Binary classification with controlled density:
    - Class 0: ER(n, p_high), no k-clique
    - Class 1: ER(n, p_low) + planted k-clique

    Where p_high > p_low is chosen such that both have same expected edges.
    """

    def __init__(
        self,
        num_graphs: int = 2000,
        k: int = 4,
        node_range: Tuple[int, int] = (20, 30),
        p_no_clique: float = 0.08,
        p_with_clique: float = 0.06,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        """
        Args:
            num_graphs: Total number of graphs (split 50/50)
            k: Clique size to detect
            node_range: (min_nodes, max_nodes)
            p_no_clique: Edge probability for graphs WITHOUT cliques (higher)
            p_with_clique: Base edge probability for graphs WITH cliques (lower)
                          The planted k-clique compensates to match edge count
            transform: Transform to apply to each graph
            pre_transform: Pre-transform to apply during generation
        """
        self.num_graphs = num_graphs
        self.k = k
        self.node_range = node_range
        self.p_no_clique = p_no_clique
        self.p_with_clique = p_with_clique

        super().__init__(None, transform, pre_transform)

        # Generate dataset
        self.data_list = self._generate_dataset()

    def _has_k_clique(self, G: nx.Graph, k: int) -> bool:
        """Check if graph has a k-clique."""
        try:
            cliques = list(nx.find_cliques(G))
            return any(len(c) >= k for c in cliques)
        except:
            return False

    def _plant_clique(self, G: nx.Graph, k: int) -> nx.Graph:
        """Plant a k-clique in the graph."""
        nodes = list(G.nodes())
        if len(nodes) < k:
            return G

        # Choose k random nodes
        clique_nodes = np.random.choice(nodes, size=k, replace=False)

        # Add all edges between them
        for i, j in combinations(clique_nodes, 2):
            G.add_edge(i, j)

        return G

    def _generate_graph_no_clique(self) -> nx.Graph:
        """Generate a graph WITHOUT a k-clique."""
        max_attempts = 100

        for _ in range(max_attempts):
            n = np.random.randint(self.node_range[0], self.node_range[1] + 1)
            G = nx.erdos_renyi_graph(n, self.p_no_clique)

            # Check it doesn't naturally have a k-clique
            if not self._has_k_clique(G, self.k):
                return G

        # Fallback: return graph even if it has a clique (rare)
        return G

    def _generate_graph_with_clique(self) -> nx.Graph:
        """Generate a graph WITH a planted k-clique."""
        n = np.random.randint(self.node_range[0], self.node_range[1] + 1)

        # Start with lower density
        G = nx.erdos_renyi_graph(n, self.p_with_clique)

        # Plant k-clique
        G = self._plant_clique(G, self.k)

        return G

    def _generate_dataset(self) -> list:
        """Generate the full dataset."""
        data_list = []

        half = self.num_graphs // 2

        # Generate graphs WITHOUT k-cliques (Class 0)
        print(f"Generating {half} graphs WITHOUT {self.k}-cliques...")
        for _ in tqdm(range(half), desc="[No Clique]"):
            G = self._generate_graph_no_clique()

            # Convert to PyG Data
            data = from_networkx(G)
            data.y = torch.tensor([0], dtype=torch.long)
            data.num_nodes = G.number_of_nodes()
            data.has_clique = torch.tensor([0], dtype=torch.long)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        # Generate graphs WITH k-cliques (Class 1)
        print(f"Generating {self.num_graphs - half} graphs WITH {self.k}-cliques...")
        for _ in tqdm(range(self.num_graphs - half), desc="[Has Clique]"):
            G = self._generate_graph_with_clique()

            # Convert to PyG Data
            data = from_networkx(G)
            data.y = torch.tensor([1], dtype=torch.long)
            data.num_nodes = G.number_of_nodes()
            data.has_clique = torch.tensor([1], dtype=torch.long)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        # Statistics
        edges_class0 = [d.num_edges for d in data_list[:half]]
        edges_class1 = [d.num_edges for d in data_list[half:]]

        print(f"\n✓ Generated {self.num_graphs} graphs")
        print(f"  Class 0 (no {self.k}-clique): {half}")
        print(f"  Class 1 (has {self.k}-clique): {self.num_graphs - half}")
        print(f"  Average nodes: {np.mean([d.num_nodes for d in data_list]):.1f}")
        print(f"  Average edges:")
        print(f"    Class 0: {np.mean(edges_class0):.1f} ± {np.std(edges_class0):.1f}")
        print(f"    Class 1: {np.mean(edges_class1):.1f} ± {np.std(edges_class1):.1f}")
        print(f"    Difference: {abs(np.mean(edges_class0) - np.mean(edges_class1)):.1f}")

        return data_list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        data = self.data_list[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
