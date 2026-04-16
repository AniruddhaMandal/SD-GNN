# Copied from SS-GNN (github.com/YOURUSERNAME/SS-GNN) — see LICENSE
"""
Clique Detection Dataset for SS-GNN Validation

Binary classification: Does the graph contain a k-clique?
- Class 0: Graphs WITHOUT any k-cliques
- Class 1: Graphs WITH at least one k-clique

This task satisfies SS-GNN theory because:
- μ_0(k-clique) = 0 (no k-cliques in distribution)
- μ_1(k-clique) > 0 (k-cliques present in distribution)
- Clear distributional difference: ||μ_0 - μ_1|| >> 0 ✓
"""

import random
from typing import Optional, Tuple, List, Callable
from itertools import combinations

import torch
import networkx as nx
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from tqdm import tqdm


def has_k_clique(G: nx.Graph, k: int) -> bool:
    """
    Check if graph G contains a k-clique.

    Uses NetworkX's find_cliques which is reasonably efficient.
    For k=4, we can also use specific algorithms.
    """
    if k <= 0:
        return True
    if k == 1:
        return G.number_of_nodes() >= 1
    if k == 2:
        return G.number_of_edges() >= 1

    # For k >= 3, use clique finder
    try:
        cliques = nx.find_cliques(G)
        for clique in cliques:
            if len(clique) >= k:
                return True
        return False
    except Exception:
        # Fallback: brute force for small graphs
        if G.number_of_nodes() < k:
            return False
        for nodes in combinations(G.nodes(), k):
            if G.subgraph(nodes).number_of_edges() == k * (k - 1) // 2:
                return True
        return False


def plant_clique(G: nx.Graph, k: int, existing_nodes: bool = True) -> List[int]:
    """
    Plant a k-clique in graph G.

    Parameters:
    - G: NetworkX graph to modify
    - k: clique size
    - existing_nodes: if True, use existing nodes; if False, add new nodes

    Returns:
    - List of nodes forming the planted clique
    """
    n = G.number_of_nodes()

    if existing_nodes:
        if n < k:
            raise ValueError(f"Graph has only {n} nodes but need {k} for clique")
        # Select k random nodes
        clique_nodes = random.sample(list(G.nodes()), k)
    else:
        # Add k new nodes
        clique_nodes = list(range(n, n + k))
        G.add_nodes_from(clique_nodes)

    # Connect all pairs in the clique
    for i in clique_nodes:
        for j in clique_nodes:
            if i < j:
                G.add_edge(i, j)

    return clique_nodes


class CliqueDetectionDataset(Dataset):
    """
    Binary classification dataset: graphs with/without k-cliques.

    Class 0 (no_clique): Sparse Erdős-Rényi graphs without k-cliques
    Class 1 (has_clique): Erdős-Rényi graphs with planted k-cliques

    Parameters:
    - num_graphs: Total number of graphs (split evenly between classes)
    - k: Clique size to detect (default 4)
    - node_range: (min_n, max_n) for graph size
    - p_no_clique: Edge probability for class 0 graphs (sparse enough to avoid cliques)
    - p_with_clique: Edge probability for class 1 base graphs (before planting)
    - seed: Random seed
    - max_attempts: Max attempts to generate valid graphs
    - transform: Optional PyG transform
    - pre_transform: Optional PyG pre_transform
    """

    def __init__(
        self,
        num_graphs: int = 2000,
        k: int = 4,
        node_range: Tuple[int, int] = (20, 40),
        p_no_clique: float = 0.04,
        p_with_clique: float = 0.08,
        seed: Optional[int] = None,
        max_attempts: int = 10000,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        device: str = "cpu",
        store_on_device: bool = False,
    ):
        super().__init__()

        assert k >= 3, "k must be at least 3"
        assert num_graphs > 0
        assert isinstance(node_range, (tuple, list)) and len(node_range) == 2
        min_n, max_n = int(node_range[0]), int(node_range[1])
        assert min_n >= k, f"min_n ({min_n}) must be >= k ({k})"
        assert min_n <= max_n

        self.num_graphs = int(num_graphs)
        self.k = int(k)
        self.min_n = min_n
        self.max_n = max_n
        self.p_no_clique = float(p_no_clique)
        self.p_with_clique = float(p_with_clique)
        self.max_attempts = int(max_attempts)
        self.transform = transform
        self.pre_transform = pre_transform
        self.device = torch.device(device)
        self.store_on_device = bool(store_on_device)

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # Storage
        self.graphs: List[Data] = []

        # Generate dataset
        self._generate_all_graphs()

    def _generate_class_0_graph(self) -> nx.Graph:
        """
        Generate a graph WITHOUT k-cliques.

        Strategy: Use sparse Erdős-Rényi and verify no k-clique exists.
        """
        for attempt in range(self.max_attempts):
            n = random.randint(self.min_n, self.max_n)
            G = nx.erdos_renyi_graph(n, self.p_no_clique)

            # Ensure connected (optional, but helps)
            if not nx.is_connected(G):
                # Add minimum edges to make connected
                components = list(nx.connected_components(G))
                for i in range(len(components) - 1):
                    u = random.choice(list(components[i]))
                    v = random.choice(list(components[i + 1]))
                    G.add_edge(u, v)

            # Verify no k-clique
            if not has_k_clique(G, self.k):
                return G

        # Fallback: create a tree (guaranteed no cliques for k >= 3)
        n = random.randint(self.min_n, self.max_n)
        G = nx.random_tree(n)
        return G

    def _generate_class_1_graph(self) -> nx.Graph:
        """
        Generate a graph WITH at least one k-clique.

        Strategy: Create ER graph and plant a k-clique.
        """
        n = random.randint(self.min_n, self.max_n)
        G = nx.erdos_renyi_graph(n, self.p_with_clique)

        # Ensure connected
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                u = random.choice(list(components[i]))
                v = random.choice(list(components[i + 1]))
                G.add_edge(u, v)

        # Plant k-clique
        plant_clique(G, self.k, existing_nodes=True)

        # Verify it has a k-clique (sanity check)
        assert has_k_clique(G, self.k), "Failed to plant k-clique!"

        return G

    def _nx_to_pyg(self, G: nx.Graph, label: int) -> Data:
        """
        Convert NetworkX graph to PyG Data object.

        Parameters:
        - G: NetworkX graph
        - label: 0 (no clique) or 1 (has clique)
        """
        # Convert to PyG
        data = from_networkx(G)

        # Add label
        data.y = torch.tensor([label], dtype=torch.long)

        # Add metadata
        data.num_nodes = G.number_of_nodes()
        data.num_edges = G.number_of_edges()
        data.has_clique = torch.tensor([label], dtype=torch.long)

        # Move to device if requested
        if self.store_on_device:
            data = data.to(self.device)

        # Apply transforms
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        return data

    def _generate_all_graphs(self) -> None:
        """Generate all graphs for the dataset."""

        num_per_class = self.num_graphs // 2

        print(f"Generating {num_per_class} graphs WITHOUT {self.k}-cliques...")
        for _ in tqdm(range(num_per_class), ncols=60, desc="[No Clique]"):
            G = self._generate_class_0_graph()
            data = self._nx_to_pyg(G, label=0)
            self.graphs.append(data)

        print(f"Generating {num_per_class} graphs WITH {self.k}-cliques...")
        for _ in tqdm(range(num_per_class), ncols=60, desc="[Has Clique]"):
            G = self._generate_class_1_graph()
            data = self._nx_to_pyg(G, label=1)
            self.graphs.append(data)

        # Shuffle
        random.shuffle(self.graphs)

        # Statistics
        print(f"\n✓ Generated {len(self.graphs)} graphs")
        print(f"  Class 0 (no {self.k}-clique): {sum(1 for g in self.graphs if g.y.item() == 0)}")
        print(f"  Class 1 (has {self.k}-clique): {sum(1 for g in self.graphs if g.y.item() == 1)}")

        # Sample statistics
        avg_nodes = sum(g.num_nodes for g in self.graphs) / len(self.graphs)
        avg_edges = sum(g.num_edges for g in self.graphs) / len(self.graphs)
        print(f"  Average nodes: {avg_nodes:.1f}")
        print(f"  Average edges: {avg_edges:.1f}")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        data = self.graphs[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data


class MultiCliqueDetectionDataset(Dataset):
    """
    Multi-class version: classify by number of k-cliques.

    Classes:
    - 0: No k-cliques
    - 1: Exactly 1 k-clique
    - 2: Exactly 2 k-cliques
    - 3: 3 or more k-cliques

    This is more challenging and tests distributional differences more thoroughly.
    """

    def __init__(
        self,
        num_graphs: int = 2000,
        k: int = 4,
        node_range: Tuple[int, int] = (25, 45),
        p_base: float = 0.08,
        seed: Optional[int] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        device: str = "cpu",
        store_on_device: bool = False,
    ):
        super().__init__()

        self.num_graphs = int(num_graphs)
        self.k = int(k)
        self.min_n, self.max_n = int(node_range[0]), int(node_range[1])
        self.p_base = float(p_base)
        self.transform = transform
        self.pre_transform = pre_transform
        self.device = torch.device(device)
        self.store_on_device = bool(store_on_device)

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.graphs: List[Data] = []
        self._generate_all_graphs()

    def _count_k_cliques(self, G: nx.Graph) -> int:
        """Count number of k-cliques in graph."""
        count = 0
        try:
            cliques = list(nx.find_cliques(G))
            for clique in cliques:
                if len(clique) == self.k:
                    count += 1
                elif len(clique) > self.k:
                    # Count all k-subsets
                    from itertools import combinations
                    count += len(list(combinations(clique, self.k)))
        except Exception:
            pass
        return count

    def _generate_graph_with_n_cliques(self, target_cliques: int) -> nx.Graph:
        """Generate graph with approximately target_cliques k-cliques."""
        n = random.randint(self.min_n, self.max_n)
        G = nx.erdos_renyi_graph(n, self.p_base)

        # Plant cliques
        for _ in range(target_cliques):
            if G.number_of_nodes() >= self.k:
                plant_clique(G, self.k, existing_nodes=True)

        return G

    def _generate_all_graphs(self) -> None:
        """Generate graphs for each class."""
        graphs_per_class = self.num_graphs // 4

        for target in range(4):
            desc = f"[{target} cliques]" if target < 3 else "[3+ cliques]"
            for _ in tqdm(range(graphs_per_class), ncols=60, desc=desc):
                G = self._generate_graph_with_n_cliques(target if target < 3 else target + 1)

                # Determine actual class
                count = self._count_k_cliques(G)
                label = min(count, 3)  # 0, 1, 2, or 3+

                # Convert to PyG
                data = from_networkx(G)
                data.y = torch.tensor([label], dtype=torch.long)
                data.num_cliques = torch.tensor([count], dtype=torch.long)
                data.num_nodes = G.number_of_nodes()

                if self.store_on_device:
                    data = data.to(self.device)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                self.graphs.append(data)

        random.shuffle(self.graphs)

        print(f"\n✓ Generated {len(self.graphs)} graphs")
        for label in range(4):
            count = sum(1 for g in self.graphs if g.y.item() == label)
            print(f"  Class {label}: {count} graphs")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        data = self.graphs[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
