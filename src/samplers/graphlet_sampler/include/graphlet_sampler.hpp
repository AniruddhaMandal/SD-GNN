#pragma once
#include <cstdint>

// ---- Limits for per-thread local arrays in the CUDA kernel ----
#define MAX_K 32
#define MAX_BOUNDARY 512

// Host-side wrapper that launches the CUDA kernel.
// All pointer arguments must reside in GPU memory.
void launch_graphlet_kernel(
    // CSR graph (flat, all graphs concatenated)
    const int32_t* row_ptr,      // [N_total + 1]
    const int32_t* col_idx,      // [nnz]
    // Per-sample metadata
    const int32_t* graph_ids,    // [B_total]  graph index for each sample
    const int32_t* seed_nodes,   // [B_total]  seed node (graph-local), -1 = random
    const int32_t* graph_n,      // [G]        node count per graph
    const int32_t* graph_offsets,// [G]        ptr[g] (CSR row offset per graph)
    // Outputs (preallocated)
    int32_t* out_nodes,          // [B_total * k]  graph-local node IDs, -1 on failure
    float*   out_log_probs,      // [B_total]      log sampling probability
    // Scalar parameters
    int B_total,
    int k,
    uint64_t rng_seed
);
