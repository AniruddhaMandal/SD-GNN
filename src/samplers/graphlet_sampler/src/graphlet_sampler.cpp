// graphlet_sampler.cpp
// CPU preprocessing + GPU dispatch + CPU edge extraction for the graphlet sampler.
// Returns 6-tuple: (nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_global_t, log_probs_t)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>
#include <cstdint>
#include <omp.h>

#include "graphlet_sampler.hpp"

namespace py = pybind11;
using i64 = int64_t;
using i32 = int32_t;

py::tuple sample_batch(
    const torch::Tensor& edge_index,   // [2, E] int64
    const torch::Tensor& ptr,          // [G+1] int64
    int m_per_graph,
    int k,
    const std::string& mode = "sample",
    uint64_t seed = 42,
    const c10::optional<torch::Tensor>& target_nodes = c10::nullopt,
    const c10::optional<torch::Tensor>& target_ptr   = c10::nullopt
) {
    TORCH_CHECK(edge_index.dtype() == torch::kInt64, "edge_index must be int64");
    TORCH_CHECK(ptr.dtype() == torch::kInt64, "ptr must be int64");
    TORCH_CHECK(k >= 1 && k <= MAX_K, "k must be in [1, MAX_K]");

    auto input_device = edge_index.device();

    // ---- CPU copies ----
    auto ei_cpu  = edge_index.cpu().contiguous();
    auto ptr_cpu = ptr.cpu().contiguous();

    auto ei_acc  = ei_cpu.accessor<i64, 2>();
    auto ptr_acc = ptr_cpu.accessor<i64, 1>();

    const i64 G        = ptr_acc.size(0) - 1;   // number of graphs
    const i64 E        = ei_cpu.size(1);         // total edges
    const i64 N_total  = ptr_acc[G];             // total nodes across all graphs

    bool has_targets = target_nodes.has_value() && target_nodes.value().numel() > 0;
    torch::Tensor tn_cpu, tp_cpu;
    if (has_targets) {
        tn_cpu = target_nodes.value().cpu().contiguous();
        tp_cpu = target_ptr.value().cpu().contiguous();
        TORCH_CHECK(tp_cpu.size(0) == G + 1, "target_ptr must have G+1 entries");
    }

    // ==================================================================
    // Phase 1: CSR construction
    // ==================================================================

    // Count degrees for CSR row_ptr
    std::vector<i32> csr_row_ptr((size_t)(N_total + 1), 0);
    for (i64 e = 0; e < E; ++e) {
        i64 u = ei_acc[0][e];
        csr_row_ptr[(size_t)(u + 1)]++;
    }
    // Prefix sum
    for (i64 i = 1; i <= N_total; ++i)
        csr_row_ptr[(size_t)i] += csr_row_ptr[(size_t)(i - 1)];

    const i64 nnz = csr_row_ptr[(size_t)N_total];

    // Fill col_idx and csr_to_edge_col (maps CSR position -> original edge index column)
    std::vector<i32> csr_col_idx((size_t)nnz);
    std::vector<i32> csr_to_edge_col((size_t)nnz);
    std::vector<i32> write_offset(csr_row_ptr.begin(), csr_row_ptr.end()); // copy for insertion

    for (i64 e = 0; e < E; ++e) {
        i64 u = ei_acc[0][e];
        i64 v = ei_acc[1][e];
        i32 pos = write_offset[(size_t)u]++;
        csr_col_idx[(size_t)pos]     = (i32)v;
        csr_to_edge_col[(size_t)pos] = (i32)e;
    }

    // graph_n and graph_offsets
    std::vector<i32> graph_n_vec((size_t)G);
    std::vector<i32> graph_offsets_vec((size_t)G);
    for (i64 g = 0; g < G; ++g) {
        graph_offsets_vec[(size_t)g] = (i32)ptr_acc[g];
        graph_n_vec[(size_t)g]      = (i32)(ptr_acc[g + 1] - ptr_acc[g]);
    }

    // ==================================================================
    // Phase 1b: Per-sample arrays
    // ==================================================================

    i64 B_total;
    std::vector<i32> graph_ids_vec;
    std::vector<i32> seed_nodes_vec;
    std::vector<i64> sample_ptr_vec((size_t)(G + 1), 0);

    if (!has_targets) {
        // Without targets: B_total = G * m, all seeds = -1
        B_total = G * (i64)m_per_graph;
        graph_ids_vec.resize((size_t)B_total);
        seed_nodes_vec.resize((size_t)B_total, -1);

        for (i64 g = 0; g < G; ++g) {
            i64 base = g * (i64)m_per_graph;
            for (int s = 0; s < m_per_graph; ++s) {
                graph_ids_vec[(size_t)(base + s)] = (i32)g;
            }
            sample_ptr_vec[(size_t)(g + 1)] = sample_ptr_vec[(size_t)g] + (i64)m_per_graph;
        }
    } else {
        // With targets: B_total = T * m. Subgraphs grouped m-contiguous per target.
        auto tn_acc = tn_cpu.accessor<i64, 1>();
        auto tp_acc = tp_cpu.accessor<i64, 1>();

        i64 T = tn_cpu.size(0);
        B_total = T * (i64)m_per_graph;
        graph_ids_vec.resize((size_t)B_total);
        seed_nodes_vec.resize((size_t)B_total);

        i64 sample_idx = 0;
        for (i64 g = 0; g < G; ++g) {
            i64 t_start = tp_acc[g];
            i64 t_end   = tp_acc[g + 1];
            i64 T_g = t_end - t_start;
            for (i64 ti = t_start; ti < t_end; ++ti) {
                i64 target_global = tn_acc[ti];
                i32 target_local  = (i32)(target_global - ptr_acc[g]);
                for (int s = 0; s < m_per_graph; ++s) {
                    graph_ids_vec[(size_t)sample_idx]  = (i32)g;
                    seed_nodes_vec[(size_t)sample_idx]  = target_local;
                    sample_idx++;
                }
            }
            sample_ptr_vec[(size_t)(g + 1)] = sample_ptr_vec[(size_t)g] + T_g * (i64)m_per_graph;
        }
    }

    // ==================================================================
    // Phase 2: GPU dispatch
    // ==================================================================

    TORCH_CHECK(torch::cuda::is_available(), "CUDA not available");

    auto gpu_opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto gpu_opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    // Upload CSR
    auto gpu_row_ptr = torch::from_blob(csr_row_ptr.data(), {N_total + 1},
                                        torch::kInt32).to(torch::kCUDA);
    auto gpu_col_idx = torch::from_blob(csr_col_idx.data(), {nnz},
                                        torch::kInt32).to(torch::kCUDA);

    // Upload per-sample arrays
    auto gpu_graph_ids  = torch::from_blob(graph_ids_vec.data(), {B_total},
                                           torch::kInt32).to(torch::kCUDA);
    auto gpu_seed_nodes = torch::from_blob(seed_nodes_vec.data(), {B_total},
                                           torch::kInt32).to(torch::kCUDA);
    auto gpu_graph_n    = torch::from_blob(graph_n_vec.data(), {G},
                                           torch::kInt32).to(torch::kCUDA);
    auto gpu_graph_off  = torch::from_blob(graph_offsets_vec.data(), {G},
                                           torch::kInt32).to(torch::kCUDA);

    // Allocate outputs
    auto gpu_out_nodes     = torch::full({B_total, (i64)k}, -1, gpu_opts_i32);
    auto gpu_out_log_probs = torch::full({B_total}, -std::numeric_limits<float>::infinity(), gpu_opts_f32);

    // Launch kernel
    launch_graphlet_kernel(
        gpu_row_ptr.data_ptr<i32>(),
        gpu_col_idx.data_ptr<i32>(),
        gpu_graph_ids.data_ptr<i32>(),
        gpu_seed_nodes.data_ptr<i32>(),
        gpu_graph_n.data_ptr<i32>(),
        gpu_graph_off.data_ptr<i32>(),
        gpu_out_nodes.data_ptr<i32>(),
        gpu_out_log_probs.data_ptr<float>(),
        (int)B_total, k, seed);

    // Download to CPU
    auto cpu_out_nodes = gpu_out_nodes.cpu();       // [B_total, k] int32
    auto cpu_log_probs = gpu_out_log_probs.cpu();   // [B_total] float32

    // ==================================================================
    // Phase 3: Edge extraction (OpenMP parallel)
    // ==================================================================

    auto local_nodes_acc = cpu_out_nodes.accessor<i32, 2>();

    // Per-sample edge collection (parallel)
    struct EdgeRec { i64 u_local; i64 v_local; i64 src_edge; };
    std::vector<std::vector<EdgeRec>> per_sample_edges((size_t)B_total);

    #pragma omp parallel for schedule(dynamic)
    for (i64 s = 0; s < B_total; ++s) {
        int gid      = graph_ids_vec[(size_t)s];
        int g_offset = graph_offsets_vec[(size_t)gid];

        // Collect valid nodes for this sample
        int valid_count = 0;
        i32 sample_nodes[MAX_K];
        for (int i = 0; i < k; ++i) {
            i32 nl = local_nodes_acc[s][i];
            if (nl < 0) break;
            sample_nodes[valid_count++] = nl;
        }
        if (valid_count == 0) continue;

        // For each chosen node, iterate CSR neighbors and check membership
        auto& edges = per_sample_edges[(size_t)s];
        for (int i = 0; i < valid_count; ++i) {
            i32 u_local = sample_nodes[i];
            i32 u_global = g_offset + u_local;
            int row_start = csr_row_ptr[(size_t)u_global];
            int row_end   = csr_row_ptr[(size_t)(u_global + 1)];

            for (int e = row_start; e < row_end; ++e) {
                i32 v_global = csr_col_idx[(size_t)e];
                i32 v_local  = v_global - g_offset;

                // Linear membership scan
                for (int j = 0; j < valid_count; ++j) {
                    if (sample_nodes[j] == v_local) {
                        // Map to sample-local indices for mode="sample"
                        i64 u_out, v_out;
                        if (mode == "sample") {
                            u_out = (i64)i;
                            v_out = (i64)j;
                        } else {
                            u_out = (i64)(g_offset + u_local);
                            v_out = (i64)(g_offset + v_local);
                        }
                        edges.push_back({u_out, v_out, (i64)csr_to_edge_col[(size_t)e]});
                        break;
                    }
                }
            }
        }
    }

    // Build edge_ptr and concatenate edges
    auto cpu_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true);

    i64 total_edges = 0;
    for (i64 s = 0; s < B_total; ++s)
        total_edges += (i64)per_sample_edges[(size_t)s].size();

    torch::Tensor edge_index_t    = torch::empty({2, total_edges}, cpu_opts);
    torch::Tensor edge_src_global = torch::empty({total_edges}, cpu_opts);
    torch::Tensor edge_ptr_t      = torch::empty({B_total + 1}, cpu_opts);

    auto ei_out   = edge_index_t.data_ptr<i64>();
    auto es_out   = edge_src_global.data_ptr<i64>();
    auto ep_out   = edge_ptr_t.accessor<i64, 1>();

    i64 offset = 0;
    ep_out[0] = 0;
    for (i64 s = 0; s < B_total; ++s) {
        const auto& edges = per_sample_edges[(size_t)s];
        for (size_t j = 0; j < edges.size(); ++j) {
            ei_out[offset]               = edges[j].u_local;   // row 0
            ei_out[total_edges + offset]  = edges[j].v_local;   // row 1
            es_out[offset]               = edges[j].src_edge;
            offset++;
        }
        ep_out[s + 1] = offset;
    }

    // ==================================================================
    // Phase 4: Build nodes_t (globalize) and output tensors
    // ==================================================================

    // Globalize node IDs: graph-local -> global
    torch::Tensor nodes_t = torch::full({B_total, (i64)k}, -1, cpu_opts);
    auto nodes_acc = nodes_t.accessor<i64, 2>();

    #pragma omp parallel for
    for (i64 s = 0; s < B_total; ++s) {
        int gid      = graph_ids_vec[(size_t)s];
        int g_offset = graph_offsets_vec[(size_t)gid];
        for (int i = 0; i < k; ++i) {
            i32 nl = local_nodes_acc[s][i];
            if (nl >= 0)
                nodes_acc[s][i] = (i64)(g_offset + nl);
        }
    }

    // sample_ptr
    torch::Tensor sample_ptr_t = torch::empty({G + 1}, cpu_opts);
    auto sp_acc = sample_ptr_t.accessor<i64, 1>();
    for (i64 g = 0; g <= G; ++g)
        sp_acc[g] = sample_ptr_vec[(size_t)g];

    // log_probs as float32 -> int64 tensor not needed, keep as float
    // Convert from float32 to a proper torch tensor on CPU (pinned)
    auto log_probs_t = cpu_log_probs.to(torch::kFloat32).pin_memory();

    // Move to input device if needed
    if (!input_device.is_cpu()) {
        nodes_t        = nodes_t.to(input_device);
        edge_index_t   = edge_index_t.to(input_device);
        edge_ptr_t     = edge_ptr_t.to(input_device);
        sample_ptr_t   = sample_ptr_t.to(input_device);
        edge_src_global = edge_src_global.to(input_device);
        log_probs_t    = log_probs_t.to(input_device);
    }

    return py::make_tuple(nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_global, log_probs_t);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sample_batch", &sample_batch,
          "GPU-parallel seed-expansion (Lifting) k-graphlet sampler.\n"
          "Returns 6-tuple: (nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_global_t, log_probs_t)",
          py::arg("edge_index"),
          py::arg("ptr"),
          py::arg("m_per_graph"),
          py::arg("k"),
          py::arg("mode") = "sample",
          py::arg("seed") = 42,
          py::arg("target_nodes") = py::none(),
          py::arg("target_ptr") = py::none());
}
