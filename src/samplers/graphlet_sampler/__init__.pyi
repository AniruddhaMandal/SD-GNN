from typing import Tuple
import torch

def sample_batch(
    edge_index: torch.Tensor,
    ptr: torch.Tensor,
    m_per_graph: int,
    k: int,
    mode: str = "sample",
    seed: int = 42,
    target_nodes: torch.Tensor = ...,
    target_ptr: torch.Tensor = ...,
) -> Tuple[
    torch.Tensor,   # nodes_t:          [B_total, k]  int64 global node IDs (-1 on failure)
    torch.Tensor,   # edge_index_t:     [2, E_total]  int64 local/global edge indices
    torch.Tensor,   # edge_ptr_t:       [B_total + 1] int64 cumulative edge counts per sample
    torch.Tensor,   # sample_ptr_t:     [G + 1]       int64 cumulative sample counts per graph
    torch.Tensor,   # edge_src_global_t:[E_total]     int64 original edge index in input edge_index
    torch.Tensor,   # log_probs_t:      [B_total]     float32 log sampling probability (-inf on failure)
]:
    """
    GPU-parallel seed-expansion (Lifting) k-graphlet sampler.

    Args:
        edge_index: [2, E] int64 edge connectivity tensor (global node IDs).
        ptr:        [G+1] int64 graph node pointer (CSR-like boundaries).
        m_per_graph: Number of subgraph samples per graph (or per target if targets given).
        k:          Target subgraph size (max MAX_K=32).
        mode:       "sample" for sample-local edge indices [0, k-1],
                    "global" for global node IDs in edge_index_t.
        seed:       RNG seed for reproducibility.
        target_nodes: Optional [T] int64 global node IDs to seed subgraphs on.
                      When provided, each target gets m_per_graph subgraphs seeded on it.
        target_ptr:   Optional [G+1] int64 CSR boundaries of targets per graph.
                      Required when target_nodes is provided.

    Returns:
        6-tuple of (nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_global_t, log_probs_t).

    When target_nodes is provided, subgraphs are ordered m-contiguous per target within
    each graph's block:
        Graph g has T_g targets, m subgraphs each:
          sample_ptr[g] ... sample_ptr[g+1] spans T_g * m subgraphs:
            [target_0: sub_0..sub_{m-1}] [target_1: sub_0..sub_{m-1}] ...
    """
    ...
