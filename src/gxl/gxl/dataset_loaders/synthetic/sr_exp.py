"""
SR and EXP synthetic graph dataset loaders.

Also includes SR25-i datasets: pairs from the BREC strongly-regular benchmark
where graphs have exactly 25 nodes.  Each SR25-i is an independent binary
classification task — 32 relabelings of G₀ (label=0) vs 32 of G₁ (label=1).
"""
from ...registry import register_dataset
from ... import ExperimentConfig
from ...utils.split_and_loader import build_dataloaders_from_dataset
from .._base import _ListDataset

# Global BREC pair indices of the 7 SR25 pairs (n=25, strongly-regular, pairs 110-159).
# Determined empirically by checking num_nodes==25 in str.npy.
SR25_PAIR_INDICES = [111, 112, 113, 114, 115, 116, 117]


def _make_sr_graphs(n_per_class: int = 20, seed: int = 0):
    """
    SR dataset: Shrikhande graph vs 4×4 Rook's graph.
    Both are strongly regular SR(16, 6, 2, 2): same (n,k,λ,μ) parameters,
    non-isomorphic, indistinguishable by ≤2-WL. Binary classification:
    class 0 = Shrikhande, class 1 = Rook's graph.
    """
    import torch
    from torch_geometric.data import Data

    rng = torch.Generator()
    rng.manual_seed(seed)
    n = 16

    def idx(a, b):
        return a * 4 + b

    # Shrikhande graph: nodes (a,b) in Z4×Z4
    # Neighbors: (a±1,b), (a,b±1), (a+1,b+1), (a-1,b-1) all mod 4
    shrikhande_edges = set()
    for a in range(4):
        for b in range(4):
            u = idx(a, b)
            for da, db in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]:
                v = idx((a + da) % 4, (b + db) % 4)
                if u != v:
                    shrikhande_edges.add((min(u, v), max(u, v)))

    # 4×4 Rook's graph: (a,b) connects to all same-row and same-column nodes
    rooks_edges = set()
    for a in range(4):
        for b in range(4):
            u = idx(a, b)
            for b2 in range(4):
                if b2 != b:
                    v = idx(a, b2)
                    rooks_edges.add((min(u, v), max(u, v)))
            for a2 in range(4):
                if a2 != a:
                    v = idx(a2, b)
                    rooks_edges.add((min(u, v), max(u, v)))

    data_list = []
    for label, edges in [(0, shrikhande_edges), (1, rooks_edges)]:
        edges = list(edges)
        src_b = torch.tensor([u for u, v in edges], dtype=torch.long)
        dst_b = torch.tensor([v for u, v in edges], dtype=torch.long)
        for _ in range(n_per_class):
            perm = torch.randperm(n, generator=rng)
            src = perm[src_b]
            dst = perm[dst_b]
            ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
            data_list.append(Data(edge_index=ei, y=torch.tensor([label], dtype=torch.long),
                                  num_nodes=n))
    return data_list


def _make_exp_graphs(n_per_class: int = 60, seed: int = 0):
    """
    EXP-like dataset: pairs of 1-WL-equivalent but non-isomorphic graphs.

    For any k-regular graph with all-ones node features, 1-WL assigns the same
    colour to every node at every round (all nodes have degree k; all neighbours
    have the same colour), so any two k-regular n-node graphs are WL-equivalent.
    A model more expressive than 1-WL can still distinguish them by counting
    substructures such as triangles and cycles.

    Three families (each pair shares n and k):
      Family 1 – n=6,  k=3: K_{3,3} (bipartite, no triangles)
                             vs Triangular Prism (has triangles)
      Family 2 – n=8,  k=3: Cubical / Q3 graph (bipartite, girth 4)
                             vs Wagner graph (non-bipartite, girth 3)
      Family 3 – n=10, k=3: Petersen graph (girth 5)
                             vs Pentagonal Prism (girth 4)

    Label 0 = first graph in pair, label 1 = second graph.
    """
    import torch
    from torch_geometric.data import Data

    rng = torch.Generator()
    rng.manual_seed(seed)

    def _relabel_copies(edges, n, label, count):
        edges = list(edges)
        src_b = torch.tensor([u for u, v in edges], dtype=torch.long)
        dst_b = torch.tensor([v for u, v in edges], dtype=torch.long)
        out = []
        for _ in range(count):
            perm = torch.randperm(n, generator=rng)
            src = perm[src_b];  dst = perm[dst_b]
            ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
            out.append(Data(edge_index=ei, y=torch.tensor([label], dtype=torch.long),
                            num_nodes=n))
        return out

    families = []

    # ── Family 1: K_{3,3} vs Triangular Prism (n=6, k=3) ────────────────────
    k33 = {(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)}
    prism3 = {(0,1),(1,2),(0,2),(3,4),(4,5),(3,5),(0,3),(1,4),(2,5)}
    families.append((6, k33, prism3))

    # ── Family 2: Cube (Q3) vs Wagner graph (n=8, k=3) ───────────────────────
    # Cube: nodes 0-7, adjacent if binary representations differ by exactly 1 bit
    cube = {(i, j) for i in range(8) for j in range(i+1, 8)
            if bin(i ^ j).count('1') == 1}
    # Wagner: 8-cycle 0-1-2-3-4-5-6-7-0 plus diagonals 0-4, 1-5, 2-6, 3-7
    wagner = {(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(0,7),
              (0,4),(1,5),(2,6),(3,7)}
    families.append((8, cube, wagner))

    # ── Family 3: Petersen vs Pentagonal Prism (n=10, k=3) ───────────────────
    petersen = {
        (0,1),(1,2),(2,3),(3,4),(0,4),   # outer 5-cycle
        (5,7),(7,9),(9,6),(6,8),(8,5),   # inner pentagram
        (0,5),(1,6),(2,7),(3,8),(4,9),   # spokes
    }
    pent_prism = {
        (0,1),(1,2),(2,3),(3,4),(0,4),   # outer pentagon
        (5,6),(6,7),(7,8),(8,9),(5,9),   # inner pentagon
        (0,5),(1,6),(2,7),(3,8),(4,9),   # rungs
    }
    families.append((10, petersen, pent_prism))

    data_list = []
    for n, edges0, edges1 in families:
        data_list.extend(_relabel_copies(edges0, n, 0, n_per_class))
        data_list.extend(_relabel_copies(edges1, n, 1, n_per_class))
    return data_list


@register_dataset('SR')
def build_sr(cfg: ExperimentConfig):
    """
    SR dataset: Shrikhande graph vs 4×4 Rook's graph.
    Both are strongly regular SR(16, 6, 2, 2) — same parameters, non-isomorphic,
    indistinguishable by ≤2-WL. Binary classification (0=Shrikhande, 1=Rook's).
    40 graphs total: 20 random relabellings of each.
    """
    from torch_geometric.transforms import Compose
    from ...utils.data_transform import SetNodeFeaturesOnes

    node_dim  = cfg.model_config.node_feature_dim
    transform = Compose([SetNodeFeaturesOnes(dim=node_dim, cat=False)])

    graphs  = _make_sr_graphs(n_per_class=20, seed=0)
    dataset = _ListDataset(graphs, transform=transform)
    return build_dataloaders_from_dataset(dataset, cfg)


def _make_sr25_pair(pair_local_idx: int, raw_dir: str, num_relabel: int = 32, seed: int = 0):
    """
    Build a binary classification dataset for the i-th SR25 pair.

    Loads str.npy from raw_dir, extracts the pair at
    SR25_PAIR_INDICES[pair_local_idx], generates num_relabel random relabelings
    of each graph, and returns a flat list of 2*num_relabel Data objects:
      - label 0 → G₀ relabelings
      - label 1 → G₁ relabelings
    """
    import os
    import numpy as np
    import networkx as nx
    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils.convert import from_networkx

    rng = torch.Generator()
    rng.manual_seed(seed)

    strongly_reg = np.load(os.path.join(raw_dir, "str.npy"), allow_pickle=True)

    # str.npy covers BREC pairs 110-159.  Pair at global index p → local index p-110.
    global_idx  = SR25_PAIR_INDICES[pair_local_idx]
    local_idx   = global_idx - 110          # offset within str.npy
    g6_0 = strongly_reg[local_idx * 2]
    g6_1 = strongly_reg[local_idx * 2 + 1]

    def g6_to_edges(g6):
        if isinstance(g6, str):
            g6 = g6.encode()
        G   = nx.from_graph6_bytes(g6)
        pyg = from_networkx(G)
        return pyg.edge_index, G.number_of_nodes()

    def relabeled_copies(g6, label, count):
        ei_orig, n = g6_to_edges(g6)
        copies = []
        for _ in range(count):
            perm   = torch.randperm(n, generator=rng)
            ei_new = perm[ei_orig]
            x      = torch.ones(n, 1, dtype=torch.float)
            ea     = torch.ones(ei_new.shape[1], 1, dtype=torch.float)
            copies.append(Data(
                x=x, edge_index=ei_new, edge_attr=ea,
                y=torch.tensor([label], dtype=torch.long),
                num_nodes=n,
            ))
        return copies

    data_list  = relabeled_copies(g6_0, label=0, count=num_relabel)
    data_list += relabeled_copies(g6_1, label=1, count=num_relabel)
    return data_list


def _build_sr25_i(pair_local_idx: int, cfg: ExperimentConfig):
    raw_dir = getattr(cfg, "raw_dir", "data/BREC/raw")
    graphs  = _make_sr25_pair(pair_local_idx, raw_dir=raw_dir)
    dataset = _ListDataset(graphs)
    return build_dataloaders_from_dataset(dataset, cfg)


# Register SR25-0 … SR25-6 (one per SR25 pair)
for _i in range(len(SR25_PAIR_INDICES)):
    register_dataset(f"SR25-{_i}")(
        (lambda i: lambda cfg: _build_sr25_i(i, cfg))(_i)
    )


@register_dataset('EXP')
def build_exp(cfg: ExperimentConfig):
    """
    EXP-like dataset: three families of 1-WL-equivalent but non-isomorphic
    regular graph pairs.  All-ones node features → 1-WL assigns the same colour
    to every node in any k-regular graph, so any model distinguishing pairs is
    strictly more expressive than 1-WL.

    Families:
      - K_{3,3} vs Triangular Prism  (n=6,  k=3)
      - Cube (Q3) vs Wagner graph     (n=8,  k=3)
      - Petersen vs Pentagonal Prism  (n=10, k=3)

    360 graphs total: 60 relabellings × 2 graphs × 3 families.
    """
    from torch_geometric.transforms import Compose
    from ...utils.data_transform import SetNodeFeaturesOnes

    node_dim  = cfg.model_config.node_feature_dim
    transform = Compose([SetNodeFeaturesOnes(dim=node_dim, cat=False)])

    graphs  = _make_exp_graphs(n_per_class=60, seed=0)
    dataset = _ListDataset(graphs, transform=transform)
    return build_dataloaders_from_dataset(dataset, cfg)
