# Attribute Filtering in Approximate Nearest Neighbor Search: An In-depth Experimental Study

**作者**：Mocheng Li, Xiao Yan, Baotong Lu, Yue Zhang, James Cheng, Chenhao Ma  
**机构**：The Chinese University of Hong Kong (Shenzhen), Wuhan University, Microsoft Research  
**发表**：ACM SIGMOD 2026 | arXiv:2508.16263  
**代码**：https://github.com/lmccccc/FANNBench

---

## Abstract

With the growing integration of structured and unstructured data, new methods have emerged for performing similarity searches on vectors while honoring structured attribute constraints, i.e., a process known as Filtering Approximate Nearest Neighbor (Filtering ANN) search. Since many of these algorithms have only appeared in recent years and are designed to work with a variety of base indexing methods and filtering strategies, there is a pressing need for a unified analysis that identifies their core techniques and enables meaningful comparisons.

In this work, we present a unified Filtering ANN search interface that encompasses the latest algorithms and evaluate them extensively from multiple perspectives. First, we propose a comprehensive taxonomy of existing Filtering ANN algorithms based on attribute types and filtering strategies. Next, we analyze their key components, i.e., index structures, pruning strategies, and entry point selection, to elucidate design differences and tradeoffs. We then conduct a broad experimental evaluation on 10 algorithms and 12 methods across 4 datasets (each with up to 10 million items), incorporating both synthetic and real attributes and covering selectivity levels from 0.1% to 100%. Finally, an in-depth component analysis reveals the influence of pruning, entry point selection, and edge filtering costs on overall performance.

---

## 1 Introduction

The advent of Transformer architectures and modern embedding techniques has led to the widespread use of vector representations for unstructured data such as text, video, audio, and images. ANN search is integral to vector databases, serving as the fundamental system for recommendation systems, pattern matching, and retrieval-augmented generation (RAG).

Various indexing methods have been developed for efficient ANN search, including graph-based, quantization-based, hashing-based, and tree-based approaches. Graph-based algorithms construct neighborhood graphs that offer high query efficiency and recall by traversing from an entry point to successively closer neighbors. Quantization-based methods compress vectors via dimensionality reduction or lower-bit encoding, resulting in low memory usage and good GPU parallelism compatibility.

### 1.1 Filtering ANN Search

Building upon the foundation of ANN search, there is a growing need for hybrid ANN search scenarios that integrate vector similarity search with relational database-style functionalities. We refer to this hybrid search task as **Filtering Approximate Nearest Neighbor (Filtering ANN) search**.

**Filtering Attributes.** Current systems or methods predominantly target two attribute types:

- **Numerical Attributes and Range Filtering.** Range filtering retrieves nearest neighbors whose attribute values fall within a specified interval. Methods: SeRF, iRangeGraph, UNIFY.
- **Categorical Attributes and Label Filtering.** Label filtering finds nearest neighbors that share one or more categorical tags. Example: Filtered DiskANN.
- **Arbitrary Filtering.** Some systems support more flexible filtering via user-defined filtering functions. Examples: VBASE, Milvus, Faiss, ACORN.

**Filtering Strategies.**

- **Pre-filtering**: applies filtering prior to the similarity search to reduce the search space; effective for IVF-based indexes but can disrupt graph traversal paths.
- **Post-filtering**: performs ANN search first then removes candidates that do not meet constraints; efficient at high selectivity, costly at low selectivity.
- **Joint-filtering**: integrates attribute filtering directly into the ANN search, restricting traversal to valid nodes; especially effective at moderate selectivity.

### 1.2 Contributions

1. **Taxonomy Study (Section 3)**: systematic review of 12 systems/algorithms, two-axis taxonomy by filtering type × filtering strategy.
2. **Comprehensive Experimental Evaluation (Section 5.2, 5.3, 5.5)**: unified interface, 12 methods from 10 papers, selectivity 0.1%–100%, 4 datasets up to 10M.
3. **Component Analysis (Section 4, 5.6, 5.7, 5.8)**: pruning strategies, entry point selection, edge filtering overhead.
4. **Usage and Development Guidelines (Section 6)**: practical selection guide and open problems.

---

## 2 Preliminaries

**Table 2: Notations**

| Notation | Description |
|----------|-------------|
| D = {O} | Vector dataset with attributes |
| O = (v, a) | Vector and numerical attribute |
| O = (v, A) | Vector and categorical attribute set |
| Q = (q, r) | Label query vector and its restriction |
| r = (l, u) | Lower/upper bound for range query |
| r = f | Label for label query |
| n | Size of D |
| D_r | Subset of D filtered by restriction r |
| d | Dimension per vector |
| M | Maximum degree for graph ANN index |
| φ(vi, vj) | Similarity between vector vi and vj |

### 2.1 Vector Quantization

Product Quantization (PQ) compresses vectors by clustering dimensions into 256 clusters, encoding each with an 8-bit code. Advanced variants include OPQ, SQ, and RabitQ.

### 2.2 Inverted File

IVF divides vectors into partitions using k-means clustering. During search, only the nearest partitions to the query are processed. Commonly combined with PQ (IVFPQ).

### 2.3 Graph-Based ANN Search

- **KGraph**: iteratively refines neighbors; efficient construction but sacrifices connectivity (multiple disconnected components).
- **RNG (Relative Neighbor Graph)**: prunes edge e(v₁, v₃) if v₃ is closer to v₂ than to v₁, where e(v₁, v₂) already exists. Foundation of modern graph indexes.
- **Vamana Graph (VG)**: starts from random graph, performs one-pass ANN search, prunes with RNG strategy. Retains edges to remote vectors for connectivity (DiskANN).
- **HNSW**: multi-layer graph index; each higher layer is subsampled. Single entry point at top layer; RNG-inspired pruning. Efficient construction and query.

### 2.4 Filtering ANN Search: Formal Definition

$$NN(q|r) = \arg\min_{o \in D, r(o.a)} ||q - o.v||$$

where NN(q|r) denotes the exact nearest neighbor whose attribute satisfies predicate r.

---

## 3 Overview of Filtering ANN Algorithms

**Table 3: Filtering ANN Algorithms**

| Algorithm | Filtering Type | ANN Index | Filtering Strategy | Attribute Index |
|-----------|---------------|-----------|-------------------|-----------------|
| SeRF | Range | HNSW | Joint-filtering | Segmented edges |
| DSG | Range | HNSW | Joint-filtering | Segmented edges |
| β-WST | Range | VG | Pre-filtering | Segmented subgraphs (BST) |
| UNIFY | Range | HNSW | Pre/Post/Joint-filtering | Segmented subgraphs (Cross-subgraph edges) + Skip list |
| iRangeGraph | Range | RNG | Pre-filtering | Segmented subgraphs (BST) |
| FDiskANN-VG | Label | VG | Joint-filtering | Labeled edges |
| FDiskANN-SVG | Label | VG | Joint-filtering | Labeled edges + Stitched graph |
| NHQ | Label | NSW/KGraph | None | Joint distance |
| Milvus | Arbitrary | HNSW/IVF | Pre-filtering | Partition |
| Faiss-HNSW | Arbitrary | HNSW | Post-filtering | Subset identification |
| Faiss-IVFPQ | Arbitrary | IVF | Pre-filtering | Subset identification |
| ACORN | Arbitrary | HNSW | Joint-filtering | Subset identification + Two-hop scan |

### 3.1 Range Filtering ANN Search

- **SeRF**: constructs range-aware edges by sorting dataset by attribute values and incrementally inserting vectors. Creates edges covering all possible query ranges via progressive search (largest to smallest). Uses three entry points within queried range.
- **DSG**: builds on SeRF with dynamic vector insertion (no pre-sorting), enabling real-time index updates. Adds one extra range per edge for guidance.
- **β-WST**: binary search tree (BST) over attribute ranges; each BST node hosts a dedicated subgraph. log(n) layers. OptPostFiltering variant introduces controlled subgraph overlap with post-filtering.
- **iRangeGraph**: BST with log(n) layers; instead of merging results from subgraphs, collects entry points from all matched subgraphs and traverses them as unified. Avoids post-filtering.
- **UNIFY**: fixed disjoint subgraphs linked via cross-subgraph edges (mask list per edge). Adaptive strategy:
  - selectivity < Sel_low: skip list pre-filter scan
  - selectivity Sel_low~Sel_high: joint-filtering within pertinent subgraph
  - selectivity > Sel_high: post-filtering over full HNSW

### 3.2 Label Filtering ANN Search

- **Filtered-DiskANN**: extends Vamana Graph with labeled edges. FDiskANN-SVG introduces Stitched graph: builds subgraphs per categorical value and merges into unified graph, improving connectivity at the cost of slower construction.
- **NHQ**: multi-attribute queries; joint distance metric:

$$\phi'(O_i, O_j) = w_1 \cdot \phi(O_i.v, O_j.v) + w_2 \cdot \sum_{t=1}^{|O.A|} \mathbf{1}\{O_i.A_t = O_j.A_t\}$$

Efficiently guides search but may not perfectly match filter criteria at low selectivity.

### 3.3 Arbitrary Filtering ANN Search

- **Faiss**: supports HNSW, LSH, IVF plus quantization. Uses is_member(id) for subset identification during search; post-filtering in HNSW, pre-filtering in IVFPQ.
- **Milvus**: built on Faiss; partitions dataset into 64 subsets by attribute to improve low-selectivity performance.
- **ACORN**: predicate-agnostic; two-hop rule during index construction and two-hop neighbor expansion during search when immediate neighbors are insufficient.

---

## 4 Detailed Analysis of Key Components

### 4.1 Attribute Index

**Segmented edges** (SeRF, DSG): annotate edges with range indicators covering all possible query ranges. Progressive search from largest to smallest range. SeRF requires pre-sorted insertion; DSG enables dynamic updates.

**Segmented subgraphs**:
- **BST approach** (β-WST, iRangeGraph): up to log(n) layers → 2n-1 subgraphs, supporting large spectrum of range combinations.
- **Fixed disjoint subgraphs** (UNIFY): fewer combinations; relies on post-filtering for narrow ranges.

### 4.2 Pruning Techniques

Standard RNG pruning ignores attribute diversity. Specialized variants:

- **Two-hop pruning** (ACORN): restricts pruning to two-hop neighbors without caring about relative distances. Retains more potentially useful connections.
- **Label-covered pruning** (Filtered-DiskANN): edge e(v₁, v₃) is pruned only if both RNG condition AND label coverage condition hold (v₂'s attribute set covers those of both v₁ and v₃). Preserves crucial connectivity for categorical filtering.

### 4.3 Entry Point Strategies

- **Unrestricted entry points**: Faiss-HNSW, post-filtering mode in ACORN/NHQ/UNIFY; use original graph entry points.
- **Specialized entry points**: SeRF and DSG bypass hierarchical navigation, directly select multiple bottom-layer entry points at evenly spaced intervals within valid range.
- **iRangeGraph**: dynamically combines entry points from multiple matched subgraphs.
- **UNIFY joint-filtering**: single top-level entry point from selected subgraphs, connected via cross-subset edges.

---

## 5 Experiments

### 5.1 Setup

**Platform**: Ubuntu 24.04 LTS, Intel® Xeon® Platinum 8358 @ 2.60GHz, 2TB RAM, 128 physical cores. 128 threads for construction; 1 thread for queries.

**Metrics**: QPS (higher = better); Comparisons per query for graph-based methods (fewer = better).

**Datasets**:

| Dataset | Dim | Labels | Size | Query Size |
|---------|-----|--------|------|------------|
| SIFT | 128 | Synthetic | 10M | 10K |
| SpaceV | 100 | Synthetic | 10M | 10K |
| Redcaps | 512 | Real/Synthetic | 1M | 10K |
| Youtube-RGB | 1024 | Real/Synthetic | 1M | 10K |

Synthetic: numerical 0–100,000; categorical 500 distinct values. Selectivity controlled by adjusting bounds or assignment probabilities.

**Common settings**: all graph-based methods use M=40, ef_construction=1000. Evaluated at 90% recall@10.

### 5.2 Range Filtering: Results and Analysis

Key observations:

1. **Partitioning ensures query availability at low selectivity.** Milvus achieves reliable service at 0.1% selectivity; Faiss-HNSW fails completely. Partitioning raises effective selectivity within each subset.
2. **IVF-based methods show reliable recall.** Faiss-IVFPQ consistently handles 0.1% selectivity; HNSW's monotonic search restricts exploration at convergence, while IVF's non-monotonic nature enables broader exploration.
3. **Attribute-aware indexing benefits diminish at high selectivity.** Above 50%, even Faiss-HNSW achieves competitive performance.
4. **Range filtering methods achieve similar performance at moderate selectivity.** UNIFY-hybrid, WST-opt, iRangeGraph, SeRF and DSG consistently yield highest QPS at 1%–10% selectivity.
5. **BST excels at low selectivity vs. cross-subgraph edges.** iRangeGraph and WST-Vamana excel at 0.1%; UNIFY-hybrid's coarse 8-subset partitioning suffers.
6. **Segmented edges fail at low selectivity.** SeRF and DSG fail at 0.1% — graph-based monotonicity issue. DSG underperforms SeRF due to significant edge filtering overhead (Section 5.8).

### 5.3 Label Filtering: Results and Analysis

Key observations:

1. **Robustness of the stitched method.** FDiskANN-SVG generally performs well across datasets and selectivity levels. Fails to achieve 90% recall on SIFT (underlying Vamana graph quality limitation).
2. **Limitations of joint distance at low selectivity.** Both NHQ-KGraph and NHQ-NSW suffer performance drops below 1% selectivity; sparse same-label nodes cause connectivity failure.
3. **Simpler pruning outperforms in label filtering.** NHQ-KGraph (retains all nearest neighbors) outperforms NHQ-NSW (RNG pruning). In Filtering ANN contexts, straightforward connectivity rules can outperform complex geometric pruning.

### 5.4 Arbitrary Filtering Analysis

Two-predicate hybrid filtering (categorical + numerical). Key findings:

1. **Faiss-HNSW and ACORN show competitive performance at high selectivity.** ACORN and Faiss-HNSW achieve higher QPS at >10% selectivity; Faiss-HNSW best at >50%.
2. **Partitioned subset scan: simple yet effective at low selectivity.** Milvus-HNSW strong at low selectivity; ef_search has no impact (direct scans within partitions). Faiss-HNSW and ACORN struggle at 0.1%.
3. **Multiple predicate filtering performs similarly to single predicate** under uniform distribution for Faiss and ACORN.

### 5.5 Index Analysis

**Table 5: Index size and construction time** (selected highlights, SIFT 4.80GB base size):

| Algorithm | Size (GB) | Mem (GB) | Time (s) |
|-----------|-----------|----------|----------|
| Faiss-HNSW | 7.90 | 7.99 | 1733 |
| Faiss-IVFPQ | 0.68 | 2.90 | 1354 |
| ACORN | 10.45 | 109.33 | 3859 |
| FDiskANN-VG | 6.34 | 7.24 | 505 |
| FDiskANN-SVG | 6.35 | 7.11 | 370 |
| NHQ-KGraph | 5.71 | 12.90 | 266 |
| SeRF | 6.96 | 10.44 | 364 |
| DSG | 39.35 | 40.35 | 2905 |
| WST-opt | 40.22 | 97.69 | 8048 |
| iRangeGraph | 24.27 | 44.72 | 69801 |
| UNIFY | 30.25 | 36.96 | 17721 |

Key observations:
- Range filtering methods have theoretical index size O(Mn log(n)); SeRF's practical memory is lower due to small M and skipped ranges.
- WST-opt: most memory-intensive (vector duplication across overlapping BST subgraphs).
- iRangeGraph: longest construction time due to single-threaded, sequential subgraph building.
- ACORN: disproportionate memory usage (boolean flag per vector per query).

### 5.6 Pruning Strategy Evaluation

Experimental comparison of RNG pruning vs. KGraph-style pruning vs. ACORN two-hop pruning:

- ACORN two-hop pruning performs best for its architecture at low selectivity; KGraph achieves similar results; RNG better at >50% selectivity.
- SeRF_kg (KGraph pruning on SeRF) improves performance below 5% selectivity but still fails at 0.1%.
- **Conclusion**: RNG-style pruning preserves quality at moderate-to-high selectivity but is less effective at extremely low selectivity. The right pruning depends on the selectivity regime.

### 5.7 Entry Point Selection Analysis

**Table 6: Comparisons variation w.r.t. entry point count (SIFT)**

- UNIFY: hierarchical structure has minimal impact at low selectivity but meaningful impact at >50% selectivity.
- SeRF/DSG: increasing entry points from 3 to 30 or 300 consistently reduces comparisons (e.g., SeRF with 300 ep → 5.53% reduction at 1% selectivity, 3.85% at 50%).
- Bottom-layer entry point selection with 30 or 300 points surpasses default hierarchical settings.
- **Conclusion**: More entry points improve performance without recall loss; hierarchy is only beneficial at high selectivity.

### 5.8 Edge Filtering Overhead Analysis

DSG performs fewer comparisons per query than SeRF but achieves lower QPS. Root cause: DSG spends significant time filtering valid edges during neighbor selection. As ef_search increases, time for edge filtering also increases. SeRF spends almost no time on this step.

---

## 6 Lessons Learned

### 6.1 Insights

- **I1.** Fine-grained segmented subgraphs are highly effective for range Filtering ANN (iRangeGraph, β-WST, UNIFY).
- **I2.** Segmented edge-based methods fail at low selectivity due to incomplete search during index construction (SeRF, DSG).
- **I3.** Partitioning is effective for low-selectivity queries (Milvus, stitched/segmented subgraph methods).
- **I4.** Label filtering algorithms remain underdeveloped; no method performs reliably across all settings.
- **I5.** Hybrid distance filtering supports only high-selectivity queries; multi-cardinality strict matching is rarely practical.
- **I6.** Hierarchical structures offer limited benefits; only effective at >50% selectivity.
- **I7.** Larger entry point sets improve performance consistently across algorithms and selectivity levels.

### 6.2 Tool Selection Guide

- **Numerical attributes**: UNIFY or iRangeGraph (best QPS). SeRF if construction time/memory is critical. Avoid WST-opt (excessive memory).
- **Label filtering** (cardinality=1): FDiskANN-SVG. If cardinality>1: NHQ-KGraph.
- **Arbitrary filtering**: ACORN (most flexible). For >50% selectivity: Faiss-HNSW. For <1% selectivity: Milvus-HNSW (stable scan).

### 6.3 Open Problems

- **O1.** Arbitrary Filtering ANN remains an open challenge; ACORN struggles with robustness. IVF-based techniques promising.
- **O2.** Hyperparameter sensitivity; need adaptive tuning strategies.
- **O3.** Integration with vector databases: specialized architectures create barriers.
- **O4.** Dynamic Filtering ANN: only DSG supports updates; others require sorted orders.
- **O5.** Attribute distribution analysis: uniform assumption may not reflect real-world; vector-attribute interaction poorly understood.

---

## 7 Related Work

- **AnalyticDB-V (ADBV)**: Voronoi graph + PQ; cost estimator selects from brute-force scan, PQ pre-filtering, VoGPQ pre-filtering, VoGPQ post-filtering.
- **VBASE**: PostgreSQL extension; Relaxed Monotonicity termination signal; softens constraints to expand search space for filtering queries.
- **RII**: introduced subset search concept on IVFPQ; two scanning strategies based on subset size threshold.
- **AIRSHIP**: HNSW-based label filtering; samples diverse-label vectors for entry point coverage; effective under skewed distributions but no open-source code.
