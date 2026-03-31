"""
Python random walk sampler with the same API as graphlet_sampler.sample_batch.

Returns walks as ordered node sequences [S, k] and edge IDs along the walk in
sequential order [S*(k-1)], so models can gather edge features by walk position
(GPM-style: ea[edge_src_global.view(S, k-1)]).

API
---
sample_batch(edge_index, ptr, m, k, mode='sample', seed=42, **kwargs)
  -> (nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_global_t, log_probs_t)

  nodes_t          [S, k]       global node IDs; -1 if walk hit dead end
  edge_index_t     [2, S*(k-1)] sequential pairs in flat S*k space (API compat)
  edge_ptr_t       [S+1]        uniform: edge_ptr_t[s] = s*(k-1)
  sample_ptr_t     [G+1]        cumulative sample counts per graph
  edge_src_global_t[S*(k-1)]   original edge indices in batched edge_index,
                                 in walk order: [s0_e0, s0_e1, ..., s1_e0, ...]
  log_probs_t      [S]          zeros (uniform walk)
"""

import torch
from torch_cluster import random_walk


def sample_batch(
    edge_index: torch.Tensor,   # [2, E] CPU
    ptr:        torch.Tensor,   # [G+1] CPU
    m:          int,
    k:          int,
    mode:       str  = 'sample',
    seed:       int  = 42,
    **kwargs,
):
    G           = ptr.shape[0] - 1
    walk_length = k - 1

    all_nodes       = []
    all_edge_global = []
    sample_ptr_list = [0]

    rng = torch.Generator()
    rng.manual_seed(seed)

    # Precompute edge → graph mapping for efficient slicing
    # edges are in node order so we can use ptr to derive edge ranges
    # We scan once to find edge boundaries per graph.
    edge_start_per_graph = torch.zeros(G + 1, dtype=torch.long)
    for g in range(G):
        s_n = ptr[g].item()
        e_n = ptr[g + 1].item()
        edge_start_per_graph[g + 1] = (
            edge_start_per_graph[g]
            + int(((edge_index[0] >= s_n) & (edge_index[0] < e_n)).sum().item())
        )

    for g in range(G):
        start_node = int(ptr[g].item())
        end_node   = int(ptr[g + 1].item())
        n_g        = end_node - start_node
        e_start    = int(edge_start_per_graph[g].item())
        e_end      = int(edge_start_per_graph[g + 1].item())

        if n_g == 0:
            sample_ptr_list.append(sample_ptr_list[-1])
            continue

        # Local edges for this graph
        row_g = edge_index[0, e_start:e_end] - start_node   # [E_g]
        col_g = edge_index[1, e_start:e_end] - start_node   # [E_g]

        # Random start nodes
        start_nodes = torch.randint(0, n_g, (m,), generator=rng)

        if e_end == e_start:
            # Isolated graph: repeat start node, no edges
            walk_nodes = start_nodes.unsqueeze(1).expand(m, k).clone()
            walk_edges = torch.zeros(m, walk_length, dtype=torch.long)
        else:
            if walk_length == 0:
                walk_nodes = start_nodes.unsqueeze(1)
                walk_edges = torch.zeros(m, 0, dtype=torch.long)
            else:
                walks, local_eids = random_walk(
                    row_g, col_g,
                    start=start_nodes,
                    walk_length=walk_length,
                    p=1.0, q=1.0,
                    return_edge_indices=True,
                )
                # walks:      [m, k] local node IDs
                # local_eids: [m, walk_length] indices into (row_g, col_g); -1 at dead ends

                walk_nodes = walks                    # [m, k] local
                walk_edges = local_eids               # [m, k-1] local eid or -1

        # Convert to global node IDs
        walk_nodes_global = walk_nodes + start_node  # [m, k]

        # Convert local edge IDs → global edge IDs
        walk_edges_global = walk_edges.clone()
        valid_e = walk_edges >= 0
        if valid_e.any():
            walk_edges_global[valid_e] = e_start + walk_edges[valid_e]
        # dead-end edges: keep 0 as a safe fallback index (model will mask via node validity)
        walk_edges_global[~valid_e] = 0

        all_nodes.append(walk_nodes_global)
        all_edge_global.append(walk_edges_global)
        sample_ptr_list.append(sample_ptr_list[-1] + m)

    if len(all_nodes) == 0:
        # Edge case: empty batch
        nodes_t          = torch.zeros(0, k, dtype=torch.long)
        edge_index_t     = torch.zeros(2, 0, dtype=torch.long)
        edge_ptr_t       = torch.zeros(1, dtype=torch.long)
        sample_ptr_t     = torch.zeros(G + 1, dtype=torch.long)
        edge_src_global_t = torch.zeros(0, dtype=torch.long)
        log_probs_t      = torch.zeros(0)
        return nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_global_t, log_probs_t

    nodes_t   = torch.cat(all_nodes,       dim=0)           # [S, k]
    esg_flat  = torch.cat(all_edge_global, dim=0).reshape(-1) # [S*(k-1)]

    S = nodes_t.shape[0]

    # edge_index_t: sequential pairs in flat S*k space
    # position i within sample s maps to flat index s*k + i
    if walk_length > 0:
        s_idx    = torch.arange(S).unsqueeze(1).expand(S, walk_length)  # [S, k-1]
        i_src    = torch.arange(walk_length).unsqueeze(0).expand(S, -1) # [S, k-1]
        src_flat = (s_idx * k + i_src).reshape(-1)
        dst_flat = (s_idx * k + i_src + 1).reshape(-1)
        edge_index_t = torch.stack([src_flat, dst_flat], dim=0)          # [2, S*(k-1)]
    else:
        edge_index_t = torch.zeros(2, 0, dtype=torch.long)

    edge_ptr_t       = torch.arange(S + 1, dtype=torch.long) * walk_length
    sample_ptr_t     = torch.tensor(sample_ptr_list, dtype=torch.long)
    edge_src_global_t = esg_flat
    log_probs_t      = torch.zeros(S)

    return nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_global_t, log_probs_t
