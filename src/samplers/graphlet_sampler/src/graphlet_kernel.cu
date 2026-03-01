// graphlet_kernel.cu
// GPU kernel for seed-expansion (Lifting) k-graphlet sampling.
// One CUDA thread per sample.  Philox RNG.

#include "graphlet_sampler.hpp"
#include <curand_kernel.h>
#include <math_constants.h>
#include <cfloat>
#include <cmath>

// -----------------------------------------------------------------------
// Kernel: each thread produces one k-graphlet sample
// -----------------------------------------------------------------------
__global__ void graphlet_sample_kernel(
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_idx,
    const int32_t* __restrict__ graph_ids,
    const int32_t* __restrict__ seed_nodes,
    const int32_t* __restrict__ graph_n,
    const int32_t* __restrict__ graph_offsets,
    int32_t* __restrict__ out_nodes,
    float*   __restrict__ out_log_probs,
    int B_total,
    int k,
    uint64_t rng_seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B_total) return;

    // ---- RNG init (Philox) ----
    curandStatePhilox4_32_10_t rng;
    curand_init(rng_seed, (unsigned long long)tid, 0, &rng);

    int gid      = graph_ids[tid];
    int n        = graph_n[gid];
    int g_offset = graph_offsets[gid];   // global node offset for this graph

    // Pointer into output row for this sample
    int32_t* my_nodes = out_nodes + (int64_t)tid * k;

    // ---- Degenerate: graph too small ----
    if (n < k || n == 0) {
        for (int i = 0; i < k; ++i) my_nodes[i] = -1;
        out_log_probs[tid] = -CUDART_INF_F;
        return;
    }

    // ---- Per-thread local arrays (registers / local mem) ----
    int32_t S[MAX_K];                       // chosen node set (graph-local IDs)
    int32_t boundary_nodes[MAX_BOUNDARY];   // boundary node IDs (graph-local)
    float   boundary_weights[MAX_BOUNDARY]; // weight of each boundary node
    int     s_size = 0;
    int     b_size = 0;

    float log_prob = 0.0f;

    // ---- Helper: check if node is already in S ----
    auto in_S = [&](int32_t v) -> bool {
        for (int i = 0; i < s_size; ++i)
            if (S[i] == v) return true;
        return false;
    };

    // ---- Helper: find node in boundary, return index or -1 ----
    auto find_in_boundary = [&](int32_t v) -> int {
        for (int i = 0; i < b_size; ++i)
            if (boundary_nodes[i] == v) return i;
        return -1;
    };

    // ---- Helper: add neighbors of `node` (graph-local) to boundary ----
    auto expand_boundary = [&](int32_t node) {
        int global_node = g_offset + node;
        int row_start = row_ptr[global_node];
        int row_end   = row_ptr[global_node + 1];
        for (int e = row_start; e < row_end; ++e) {
            int32_t nbr_global = col_idx[e];
            int32_t nbr_local  = nbr_global - g_offset;
            if (in_S(nbr_local)) continue;  // already chosen
            int idx = find_in_boundary(nbr_local);
            if (idx >= 0) {
                boundary_weights[idx] += 1.0f;  // increment weight
            } else if (b_size < MAX_BOUNDARY) {
                boundary_nodes[b_size]   = nbr_local;
                boundary_weights[b_size] = 1.0f;
                b_size++;
            }
            // else: silently drop (boundary overflow, very rare)
        }
    };

    // ---- Step 1: pick seed node ----
    int32_t seed;
    if (seed_nodes[tid] >= 0) {
        seed = seed_nodes[tid];
        // deterministic seed: log_prob contribution = 0
    } else {
        // uniform random seed
        seed = curand(&rng) % (unsigned)n;
        log_prob += -logf((float)n);
    }
    S[s_size++] = seed;

    // ---- Step 2: build initial boundary from seed ----
    expand_boundary(seed);

    // ---- Step 3: iteratively expand ----
    for (int j = 1; j < k; ++j) {
        if (b_size == 0) {
            // Boundary empty before collecting k nodes -> failure
            for (int i = s_size; i < k; ++i) S[i] = -1;
            log_prob = -CUDART_INF_F;
            s_size = k;  // break out
            break;
        }

        // Compute total weight W
        float W = 0.0f;
        for (int i = 0; i < b_size; ++i) W += boundary_weights[i];

        // Weighted sample via linear prefix-sum scan
        float u = curand_uniform(&rng) * W;  // uniform in (0, W]
        float cumsum = 0.0f;
        int chosen_idx = b_size - 1;  // fallback to last
        for (int i = 0; i < b_size; ++i) {
            cumsum += boundary_weights[i];
            if (cumsum >= u) {
                chosen_idx = i;
                break;
            }
        }

        float w_j = boundary_weights[chosen_idx];
        log_prob += logf(w_j) - logf(W);

        int32_t chosen_node = boundary_nodes[chosen_idx];

        // Swap-remove chosen from boundary
        b_size--;
        if (chosen_idx < b_size) {
            boundary_nodes[chosen_idx]   = boundary_nodes[b_size];
            boundary_weights[chosen_idx] = boundary_weights[b_size];
        }

        // Add chosen to S
        S[s_size++] = chosen_node;

        // Expand boundary with chosen node's neighbors
        expand_boundary(chosen_node);
    }

    // ---- Write output ----
    for (int i = 0; i < k; ++i) my_nodes[i] = S[i];
    out_log_probs[tid] = log_prob;
}

// -----------------------------------------------------------------------
// Host wrapper
// -----------------------------------------------------------------------
void launch_graphlet_kernel(
    const int32_t* row_ptr,
    const int32_t* col_idx,
    const int32_t* graph_ids,
    const int32_t* seed_nodes,
    const int32_t* graph_n,
    const int32_t* graph_offsets,
    int32_t* out_nodes,
    float*   out_log_probs,
    int B_total,
    int k,
    uint64_t rng_seed)
{
    if (B_total == 0) return;
    const int threads = 256;
    const int blocks  = (B_total + threads - 1) / threads;
    graphlet_sample_kernel<<<blocks, threads>>>(
        row_ptr, col_idx, graph_ids, seed_nodes,
        graph_n, graph_offsets,
        out_nodes, out_log_probs,
        B_total, k, rng_seed);
    // Synchronize to ensure kernel completion before CPU reads results
    cudaDeviceSynchronize();
}
