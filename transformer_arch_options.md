# Transformer-Based Subgraph Aggregation: Three Architecture Options

Comparison axis: **bag of subgraphs → graph representation pipeline**.
Starting point is the flat tensor `h_flat [S*k, H]` produced after subgraph flattening.
All three options differ in *where* and *what kind* of attention is applied.

---

## Current SD-GNN (ARCH-7) Baseline

```
h_flat [S*k, H]  — m subgraphs per node, k nodes per subgraph
    │
    ├─ LocalGNN(intra-subgraph edges)             → h1  [S*k, H]
    ├─ scatter_mean(h_flat, node_ids) → GlobalGNN → h2  [N,   H] → broadcast [S*k, H]
    ├─ skip_proj(h_flat)                          → h_skip [S*k, H]
    └─ scatter_mean(h_flat, sub_ids) → MLP        → h_sub  [S*k, H]

h_new = ReLU(h_skip + h1 + h2_broadcast + h_sub)          # sum of terms

Readout:
    scatter_mean(h_flat, node_ids) → node_embs [N, H]
    global_add_pool(node_embs)     → graph_emb [B, H]
```

Cross-subgraph signal travels only through `scatter_mean → GlobalGNN → broadcast`.
No attention weights — every node's contribution to a canonical node is equal.

---

## Option A: Attention Across Subgraph Views of the Same Node

**Core idea:** For each canonical node v, its m subgraph appearances are a *set*.
Replace `scatter_mean` with multi-head self-attention over that set.
This is the "vu" aggregation in Subgraphormer.

### Pipeline

```
h_flat [S*k, H]
    │
    ├─ LocalGNN(intra-subgraph edges)    → h1 [S*k, H]   (unchanged)
    │
    ├─ NODE-VIEW ATTENTION (new):
    │     For each canonical node v, gather its m appearances:
    │         views_v = h_flat[node_ids == v]             # [m_v, H]
    │     Self-attention over views_v:
    │         z_v = MHA(Q=views_v, K=views_v, V=views_v) # [m_v, H]
    │     Scatter back:
    │         h_attn[node_ids == v] = z_v                 # [S*k, H]
    │     Then per-canonical-node summary for GlobalGNN:
    │         x_sum[v] = mean(z_v)                        # [N, H]
    │         h2 = GlobalGNN(x_sum, original_edges)       # [N, H]
    │         h2_bcast = h2[node_ids]                     # [S*k, H]
    │
    └─ h_new = ReLU(h_skip + h1 + h_attn + h2_bcast)

Readout:
    x_sum = scatter(h_attn, node_ids, reduce='mean')  → [N, H]
    global_add_pool(x_sum)                            → [B, H]
```

### Implementation Note

The key operation is variable-length grouped self-attention. Two approaches:

**Approach A1 — Padded tensor:**
Pack views per node into `[N, m, H]`, run `nn.MultiheadAttention` with key_padding_mask for nodes with fewer than m appearances. Clean but wastes memory when m is uneven.

**Approach A2 — Block-diagonal attention:**
Build a block-diagonal attention mask in `[S*k, S*k]` where block `[i,j]=1` iff `node_ids[i] == node_ids[j]`. Pass as `attn_mask` to `TransformerConv` or `torch.nn.functional.scaled_dot_product_attention`. Memory-efficient for large S*k.

### vs Subgraphormer
Subgraphormer's `vu` aggregation does exactly this, using `TransformerConv` with `index_vu` precomputed edges. The difference: Subgraphormer operates on the full N² node set (all subgraphs), so `vu` connects each (u,v) to all (u',v) — attention over all N subgraphs for node v. SD-GNN connects each appearance to only its m sampled subgraphs.

### vs Current SD-GNN
Replaces uniform `scatter_mean` with learned attention weights. Nodes that appear in more informative subgraph contexts get higher weight. Adds `O(m²·H)` per canonical node per layer.

---

## Option B: Transformer Within Each Subgraph (Replace Local GNN)

**Core idea:** Within each subgraph, treat the k nodes as a sequence and run full self-attention (with distance-to-root PE). Replace the local GINEConv with a transformer encoder.

### Pipeline

```
h_flat [S*k, H]  — reshape to [S, k, H]
    │
    ├─ INTRA-SUBGRAPH TRANSFORMER (new):
    │     Add distance-to-root positional encoding:
    │         pos_enc[i] = embed(distance(root, node_i))  # [S, k, H]
    │         h_in = h_flat + pos_enc                     # [S*k, H]
    │     Reshape: [S, k, H]
    │     Self-attention (full, or masked by subgraph edges):
    │         h1 = TransformerEncoder(h_in)               # [S, k, H]
    │     Flatten back: [S*k, H]
    │
    ├─ scatter_mean(h_flat, node_ids) → GlobalGNN         → h2_bcast [S*k, H]
    │
    └─ h_new = ReLU(h_skip + h1 + h2_bcast)

Readout: (unchanged from ARCH-7)
    scatter_mean(h_new, node_ids) → [N, H]
    global_add_pool               → [B, H]
```

### Distance-to-Root PE

Critical addition absent from current SD-GNN. The graphlet sampler guarantees `nodes_sampled[sub_idx, 0]` is the root. Compute BFS distance from root to each node in the subgraph at preprocessing time and embed:

```python
self.dist_embed = nn.Embedding(max_dist + 2, hidden_dim)  # +2 for padding/-1
# In forward:
dist_pe = self.dist_embed(distances_flat)   # [S*k, H]
h_in    = h_flat + dist_pe
```

### Attention Mask Options
- **Full attention** (no mask): every node attends to all k nodes in subgraph. O(k²) per subgraph but k is small (e.g., 10).
- **Edge-masked attention**: attention only along subgraph edges (equivalent to GAT within subgraph). Less expressive but respects graph structure.

### vs Subgraphormer
Subgraphormer's `uL` aggregation uses GINEConv (not transformer) for within-subgraph local message passing. Its `vL` is a "transposed" view — node (u,v) aggregates from its neighbors in subgraph v, i.e., it sees itself as *embedded in another subgraph*. Option B does not implement `vL`.
Subgraphormer uses APSP (all-pairs shortest paths) for distance PE. Option B only needs distance from root (cheaper: BFS once per subgraph vs full Floyd-Warshall).

### vs Current SD-GNN
Provides global within-subgraph context: every node sees every other node in its subgraph at once, not just 1-hop neighbors. Key for sparse/disconnected graphlets where local GNN cannot propagate far. Cost: `O(k²)` per subgraph per layer (fine for k=10).

---

## Option C: Transformer Over Subgraph-Level Representations

**Core idea:** After collapsing each subgraph to a single vector, treat the m subgraphs of a node as a sequence and run a transformer over them before final pooling. Attention is at the *subgraph level*, not the node level.

### Pipeline

```
h_flat [S*k, H]
    │
    ├─ LocalGNN + GlobalGNN layers (ARCH-7 unchanged)
    │
    └─ SUBGRAPH-LEVEL READOUT (replaces final scatter_mean + add_pool):

    Step 1 — Per-subgraph summary:
        h_sub[s] = mean(h_flat[sub_ids == s])          # [S, H]
                   (or: h_flat[sub_ids == s, 0] for root-only)

    Step 2 — Group by graph node:
        For canonical node v with subgraphs s_1..s_m:
            sub_set_v = h_sub[s_1..s_m]                # [m, H]

    Step 3 — Subgraph-set transformer:
        z_v = MHA(sub_set_v) + sub_set_v               # [m, H]  (self-attn + residual)
        node_emb_v = mean(z_v)                         # [H]     (pool over m)

    Step 4 — Graph-level pool:
        global_add_pool(node_emb_v for all v)          → [B, H]
```

### Variant C1: Cross-Node Subgraph Attention (Graph Transformer)

Extend Step 3 to attend not just within a node's own subgraphs but across all subgraphs in the graph:

```
All subgraph summaries: h_sub [S, H]  (S = N*m total subgraphs in batch)
Run graph-transformer:
    h_sub_new = TransformerLayer(h_sub, graph-level attention or sparse attention)
Then:
    node_embs = scatter_mean(h_sub_new, subgraph_to_node_map)  → [N, H]
    global_add_pool(node_embs)                                  → [B, H]
```

This is the most expressive variant: subgraphs can exchange information globally, not just within the same canonical node.

### vs Subgraphormer
Subgraphormer never explicitly forms per-subgraph summary vectors as a separate stage. Its `uG` aggregation (sum over v-dimension) is conceptually similar to Step 1, but then immediately goes to `global_mean_pool`. There is no inter-subgraph transformer after that collapse. Option C is more modular: run any GNN for within-subgraph message passing, then apply a transformer purely at the subgraph-representation level.

### vs Current SD-GNN
Current ARCH-7 uses `scatter_mean(h_flat, node_ids) → global_add_pool` — effectively mean-of-subgraph-means with no weighting. Option C adds learned attention weights over which subgraphs matter most for each canonical node, and (in C1) allows subgraphs to communicate with each other before final pooling.

---

## Summary Comparison

| | ARCH-7 (current) | Option A | Option B | Option C |
|---|---|---|---|---|
| **Attention location** | None | Between views of same node | Within each subgraph | Between subgraph summaries |
| **What attends to what** | — | h(v,sub_i) ↔ h(v,sub_j) | node_i ↔ node_j within subgraph | sub_i_of_v ↔ sub_j_of_v |
| **Distance-to-root PE** | No | No (optional) | Yes (key ingredient) | Optional |
| **Replaces** | — | scatter_mean in cross-subgraph branch | LocalGNN | Final readout |
| **Complexity added** | — | O(m²H) per node | O(k²H) per subgraph | O(m²H) per node |
| **Closest Subgraphormer term** | — | `vu` aggregation | `uL` + distance embedding | No direct equivalent |
| **Key benefit** | Simple | Learns which subgraph contexts are informative | Full within-subgraph context + position awareness | Learns which subgraphs represent a node best |

### Recommended Priority

1. **Option B first** — distance-to-root PE alone (without even adding transformer) is likely the single biggest missing ingredient. Add PE to the existing LocalGNN, measure impact before adding attention.
2. **Option A second** — replaces scatter_mean with attention; surgically improves the weakest link in ARCH-7.
3. **Option C last** — most expensive; mainly useful if A+B plateau and you want a larger architectural change.
