# Efficient Multivector Retrieval with Token-Aware Clustering and Hierarchical Indexing

Silvio Martinico∗   
University of Pisa & ISTI–CNR Pisa, Italy   
silvio.martinico@phd.unipi.it   
Cosimo Rulli   
ISTI–CNR   
Pisa, Italy   
cosimo.rulli@isti.cnr.it

Franco Maria Nardini ISTI–CNR Pisa, Italy francomaria.nardini@isti.cnr.it

Rossano Venturini   
University of Pisa   
Pisa, Italy   
rossano.venturini@unipi.it

## Abstract

Multivector retrieval models achieve state-of-the-art effectiveness through fine-grained token-level representations, but their deployment incurs substantial computational and memory costs. Current solutions—based on the well-known k-means clustering algorithm— group similar vectors together to enable both effective compression and efficient retrieval. However, standard k-means scales poorly with the number of clusters and dataset size, and favours frequent tokens during training while underrepresenting rare, discriminative ones. In this work, we introduce Tachiom, a multivector retrieval system that exploits token-level structure to significantly accelerate both clustering and retrieval. By accounting for tokens' distribution during centroid allocation, Tachiom easily scales to millions of centroids, enabling highly accurate document scoring using only centroids, avoiding expensive token-level computation. Tachiom combines a graph-based index over centroids with an optimized Product Quantization layout for efficient final scoring. Experiments on Ms Marco-v1 and LoTTE show that Tachiom achieves up to 247× faster clustering than k-means and up to 9.8× retrieval speedup over state-of-the-art systems while maintaining comparable or superior effectiveness.

## Keywords

Multivector Retrieval, Late Interaction, Clustering, Efficiency.

## 1 Introduction

Transformer-based language models have fundamentally reshaped Information Retrieval (IR), shifting the paradigm from lexical-based to representation-based retrieval. Within this landscape, multivector models, such as ColBERT, have emerged as a gold standard for effectiveness. By encoding documents as sets of token-level vectors and computing relevance via late interaction, i.e., allowing every query token to interact with every document token, these models capture fine-grained semantic nuances that single-vector dense retrievers often miss. However, the real-world deployment of multivector models remains severely constrained by their computational demands. Storing and searching large sets of token vectors creates massive memory footprints and computational overhead. To mitigate this, state-of-the-art solutions adopt a gather-and-refine strategy combined with token quantization. In the gather phase, a lightweight search over token vectors identifies promising candidates; in the refine phase, full MaxSim scores are computed only for the selected candidates. Both phases critically depend on a coarse quantizer—a set of centroid vectors that approximate the token embedding space. During gathering, centroids serve as efficient proxies for locating relevant tokens. For compression, each token vector is decomposed into a centroid assignment and a residual vector, which is then compressed using more precise Vector Quantization (VQ) techniques, like, for instance, Product Quantization (PQ). This two-level scheme substantially improves approximation: centroid assignments distinguish distant vectors at minimal cost, while residual compression captures local nuances.

ColBERTv2 pioneered this centroid-based approach, compressing the residual (elementwise difference between the vector and the centroid) rather than the vector itself. PLAID extended this framework with centroid interaction and inverted-file filtering over clustered token embeddings for efficient retrieval. Subsequent work refined this framework: EMVB introduced bitvector prefiltering to accelerate candidate selection and optimized PQ variants to improve residuals' compression, achieving significant speedups over PLAID. WARP combined PLAID's centroid interaction with XTR's lightweight architecture; the authors showed that their method also generalizes to ColBERTv2 embeddings. IGP proposed building a graph index on the set of centroids to avoid exhaustive centroids scoring.

Across all these methods, both retrieval quality and compression effectiveness scale directly with the number of centroids: finer-grained centroids enable more precise approximations and more selective gathering. However, standard k-means clustering—the predominant method for centroids selection—creates a severe bottleneck: it scales poorly with both dataset size and number of clusters, and critically, it ignores the token structure of multivector embeddings. Training even moderately large sets of centroids (e.g., 262K) on multivector collections can require tens of hours on multi-core CPUs or expensive GPU resources, fundamentally limiting achievable granularity. In this paper, we address this bottleneck by exploiting a key structural property of multivector embeddings: token identity. Unlike generic vector sets, multivector collections exhibit strong correlations between token type and embedding distribution. Common tokens (e.g., stopwords) dominate k-means' loss during training, consuming the majority of centroids, while rare domain-specific tokens—often more discriminative for retrieval—receive minimal representation. We introduce Tachiom, a multivector retrieval system that exploits this structure at both the clustering and retrieval stages. At its core, Tachiom employs Token-Aware Clustering (Tac), which explicitly allocates centroids based on token frequency and semantic variance. By partitioning the clustering workload across token groups, Tac achieves significantly faster centroids computation over standard k-means. This efficiency enables scaling to millions of centroids, far beyond prior work. Leveraging these high-resolution centroids, Tachiom performs gathering using only centroid-level interactions via an HNSW graph index, so as to bypass expensive token-level computations entirely. For refinement, a cache-optimized PQ layout tailored for late interaction enables efficient final scoring.

In summary, the novel contributions of this work are:

- We propose Tac, a clustering algorithm that leverages token identity to decompose the global clustering problem into independent per-token subproblems, improving scalability while producing centroids better aligned with retrieval objectives. We further derive a theoretical lower bound on the computational speed-up achieved by Tac over k-means.
- We introduce Tachiom, a retrieval architecture that leverages the clustering granularity enabled by Tac to perform centroids-only gathering via graph-based search and efficient full scoring with late interaction-tailored PQ.
- We conduct comprehensive experiments on Ms Marco-v1 and LoTTE, demonstrating that our proposed architectures offer superior efficiency-effectiveness trade-offs. Our results show that Tac achieves up to 247× faster training than standard k-means, while Tachiom delivers up to 9.8× faster end-to-end retrieval compared to state-of-the-art systems. To favor reproducibility, we release our implementation of Tac and Tachiom in Rust on GitHub at https://github.com/TusKANNy/tachiom.

## 2 Methodology

## 2.1 Token-Aware Clustering

Multivector models such as ColBERT represent each document as a set of contextualized token embeddings. State-of-the-art retrieval methods rely on centroids to approximate these large collections of token vectors, typically using k-means clustering to select representative centroids. However, standard k-means treats token embeddings as a generic vector set, discarding the structural information provided by token identity. Token-Aware Clustering (Tac) explicitly exploits the token structure of multivector representations. Tac serves two purposes: (i) it significantly accelerates clustering by partitioning the workload into independent per-token subproblems, crucial for the massive vector collections produced by ColBERT-like models; (ii) it balances centroid allocation toward rare tokens, which are underrepresented by standard k-means despite their importance for retrieval accuracy.

k-means aims to find a set of clusters C = {C_1, ..., C_κ} that partitions the dataset and minimizes the Within-Cluster Sum of Squares (WCSS) or Inertia:

$$\sum_{i=1}^{\kappa} \sum_{\mathbf{t} \in C_i} \|\mathbf{t} - \mathbf{c}_i\|^2$$

where **c**_i denotes the centroid of cluster C_i. This objective treats all vectors equally, but multivector collections exhibit severe frequency imbalance. Common tokens (e.g., stopwords) may appear millions of times, while domain-specific terms have way lower frequencies. Analysing two ColBERTv2 multivector collections, we observed that the top 100 most frequent tokens account for over 40% of all vectors, specifically 41% on Ms Marco-v1 and 45% on LoTTE-pooled. Consequently, the WCSS loss is dominated by high-frequency tokens, which consume the majority of centroids despite contributing little to retrieval quality. Meanwhile, rare tokens—which often carry the most discriminative semantic signal—receive minimal representation, degrading approximation quality precisely where it matters most.

Drawing from classic Information Retrieval literature, where Inverse Document Frequency (IDF) has long been employed to favor rare, discriminative terms, we propose Tac to address this imbalance in the clustering context. Tac decouples centroid allocation from uniform WCSS minimization by explicitly weighting tokens based on their frequency and semantic variance, so as to favor rare tokens. Tac first computes a weight for each token type representing its share of the total centroid budget, then performs independent k-means clustering for each token type.

Tac distributes the centroid budget through a four-stage pipeline: (1) Tail Handling isolates very rare tokens to prevent their complete marginalization; (2) Damped Scoring computes an allocation weight for the remaining tokens; (3) Bounding enforces strict limits to avoid both under- and over-allocation; and (4) Budget Reconciliation adjusts the final assignments to exactly match the global centroid budget.

**Phase 1: Tail Handling.** We partition token types into three categories based on their frequency n: Micro tokens (n < μ) receive 1 centroid each, Small tokens (μ ≤ n < τ) receive 2 centroids each, and Active tokens (n ≥ τ) are allocated centroids dynamically based on their frequency and semantic variance. This fixed allocation prevents the complete marginalization of extremely rare tokens.

**Phase 2: Damped Scoring.** For active tokens, we compute a spread measure s_j that quantifies semantic variance; we adopt componentwise variance for computational efficiency:

$$s_j := \frac{1}{n_j} \sum_{i=1}^{n_j} \|\mathbf{t}_{j,i} - \bar{\mathbf{t}}_j\|^2$$

where n_j is the frequency of token j, **t**_{j,i} is the i-th occurrence of token j, and **t̄**_j is its mean embedding. We then compute a damped weight, **w**_j = √(n_j) · s_j. The square-root damping applies diminishing returns to highly frequent tokens, preventing them from monopolizing centroids budget; the spread coefficient s_j favors tokens with high semantic variance. The number of centroids κ_j of token j is allocated proportionally to its weight:

$$\kappa_j = \left\lfloor \frac{w_j}{\sum_{i=1}^{N_T} w_i} \cdot B \right\rfloor$$

where N_T is the number of different tokens in our collection and B is the remaining centroid budget after tail handling.

**Phase 3: Bounding.** We enforce a hard floor ε such that κ_j ≥ ε for all active tokens to guarantee minimum representation. Additionally, we impose an upper bound to prevent over-allocation: each token must satisfy n_j/κ_j ≥ θ, ensuring at least θ vectors per centroid on average. This dual bounding prevents both under-representation of rare tokens in low-budget regimes and over-allocation to already well-approximated tokens in high-budget regimes.

**Phase 4: Budget Reconciliation.** We redistribute surplus or deficit centroids to exactly match the target budget B. Once centroids are allocated, we perform independent κ_j-means clustering for each token j with its assigned κ_j centroids.

In summary, Tac fundamentally differs from standard k-means in two key aspects: 1) it prioritizes retrieval quality over pure reconstruction error: by allocating centroids based on token frequency and semantic variance rather than uniform WCSS contribution, Tac ensures that rare, discriminative tokens receive adequate representation in terms of assigned centroids; 2) instead of optimizing a single global objective over all N vectors with κ centroids, Tac solves independent subproblems for each token, clustering the n_j vectors of token j into κ_j centroids. This decomposition substantially reduces the per-iteration computational cost and enables effective parallelization, as each token can be processed independently, whereas standard k-means requires shared access to the full dataset. Moreover, the damped centroid allocation mitigates the bottleneck caused by highly frequent tokens, further improving both total cost and parallel efficiency.

**Theoretical Lower Bound for Speed-up.** Standard k-means requires O(I · N · κ · d) operations for I iterations over N vectors with κ centroids in d dimensions. Tac reduces this to O(I · Σ_{j=1}^{N_T} n_j · κ_j · d). The speedup is lower-bounded by the ratio of the total weight to the maximum single token weight. Because the cumulative weight across the vocabulary vastly exceeds that of any single token, this formulation guarantees a substantial baseline speedup. Furthermore, the damped scoring mechanism not only balances centroid allocation toward rare tokens, but actively suppresses the dominance of the most common ones. By applying square-root scaling to frequencies, Tac directly reduces the maximum single-token weight, strictly improving the speedup guarantee.

**Residuals Compression.** After computing the centroids, we compress the residuals, i.e., the difference between each token vector in the dataset and its assigned centroid, using PQ. Because residual magnitudes may vary across tokens due to the non-uniform centroid allocation of Tac, we normalize the residual vectors and store their norms separately. The resulting normalized residuals are more homogeneous and therefore easier to compress effectively.

## 2.2 Hierarchical Index for Optimized Multivector Retrieval

The speedups achieved by Tac over k-means allow us to significantly scale up the number of centroids used. This fine-grained clustering significantly improves the approximation provided by centroids alone, thus eliminating the need for full token scoring (centroid + residual) during the gather phase, differently from existing approaches. Furthermore, by decoupling the gather phase from the residual scores calculation, we can specialize the refinement stage with a Product Quantization layout optimized specifically for document-level MaxSim evaluation. Building on these observations, we introduce Tachiom (Token-Aware Clustering and Hierarchical Indexing for Optimized Multivector retrieval), a retrieval architecture that combines an HNSW proximity graph over centroids for efficient gathering with a cache-optimized PQ layout for efficient refinement. Tachiom operates in two main phases: first, it explores centroids to select a set of candidate documents; in the second phase, it uses both centroids and residuals to compute the full MaxSim scores of the candidate documents and provide the final ranking.

**Index Structure.** We build an HNSW graph over the centroid set; each centroid **c**_i maintains an inverted list L_i containing the document IDs of all tokens assigned to **c**_i. This design enables document-level scoring during the gather phase without accessing individual token vectors.

**Gather Phase: Centroid-based Document Scoring.** For each query token **q**_i of a query Q:

1. We traverse the graph to identify the top-κ_c nearest centroids;
2. For each retrieved centroid **c**_j, we iterate over its inverted list L_j;
3. For documents appearing in multiple lists, we retain only the highest centroid similarity: s̃_i(d) = max_{j: d∈L_j} ⟨**q**_i, **c**_j⟩
4. We accumulate partial scores across all query tokens: S̃(q, d) = Σ_{i=1}^{n_q} s̃_i(d)

This produces an approximate MaxSim score using only centroid-level interactions. Documents appearing in inverted lists for multiple query tokens naturally accumulate higher scores, capturing the multi-token matching signal without decompressing token vectors. Before the next phase, we truncate the candidate set to the top κ_d documents by partial score S̃(q, ·). We also apply adaptive filtering to further reduce the candidate set, adopting the Candidates Pruning (CP) strategy, regulated by a parameter α.

![](images/9140efa909f641a33fe24ffb16fc512d868ac6655b24b644ba24695a10aef0be.jpg)  
*Figure 1: Distance tables layout with 4 centroids per subspace. Evaluating code c = 2 for subspace m = 2, document token d.*

**Refine Phase: PQ-Optimized MaxSim Computation.** For the pruned candidate set, we compute full MaxSim scores using both centroids and PQ-compressed residuals. We adopt specialized data layouts optimized for cache locality and SIMD vectorization.

*Document Layout:* For each document d, we store all tokens' centroid IDs contiguously, followed by all PQ codes. This enables streaming evaluation: we first accumulate all centroid contributions in one pass, then refine with residuals in a second pass.

*Distance Table Layout:* PQ requires precomputing distance tables between query tokens and PQ centroids for each PQ subspace. A naive multivector PQ implementation processes each query token independently, resulting in a separate distance table per token. However, because a document token's quantization is fixed, all n_q query tokens must evaluate their distance against the same PQ centroid for any given subspace. Consequently, the standard memory layout necessitates n_q scattered memory accesses across n_q disjoint tables, creating a severe memory-bandwidth bottleneck. To eliminate this overhead, we reorganize the precomputed distances into a cache-optimized three-level hierarchy:

- **Macro-blocks**: indexed by PQ subspace (M blocks for M subspaces);
- **Centroid blocks**: indexed by PQ centroid ID (2^b blocks for b-bit codes);
- **Query token micro-blocks**: containing n_q consecutive distances between each query token and the same PQ centroid on that subspace.

When evaluating a document token's PQ code c for subspace m, our layout enables retrieving the distances to all n_q query tokens as a single contiguous micro-block (Figure 1). This avoids the n_q scattered memory lookups of the standard layout, yielding up to a 3.8× speedup in residual distance computation compared to a standard PQ implementation.

## 3 Experiments

We perform experiments on Ms Marco-v1 (8.8M passages, 598M token vectors, 6,980 dev.small queries) for in-domain evaluation and LoTTE-pooled (2.4M passages, 266M token vectors, 2,931 search/dev queries) for out-of-domain evaluation. We use ColBERTv2 as the encoder with a maximum of 180 tokens per document for LoTTE. We evaluate the retrieval results using the standard metrics for each dataset: MRR@10 for Ms Marco-v1 and Success@5 for LoTTE. Experiments run on an Intel Xeon Silver 4314 CPU @ 2.40GHz with 64 threads. Our implementation is written in Rust 1.92.0-nightly on top of the kANNolo library. Clustering exploits all 64 threads in parallel, while retrieval experiments are executed sequentially on a single core.

## 3.1 Clustering Evaluation

We first evaluate Tac's efficiency and effectiveness on Ms Marco-v1 against two k-means baselines: Faiss, which is one of the most widely-adopted k-means implementations, and FastKMeans-rs, an efficient Rust implementation of the FastKMeans library. Since our implementation uses kANNolo's AVX2-optimized single-vector distance kernels without production-ready external libraries such as Intel MKL, we evaluate two Faiss configurations: generic AVX2 (comparable to our setup) and MKL-enabled.

For Tac, we set tail handling thresholds μ = 128 and τ = 256, and bounding parameters ε = 4 and θ = 39 based on preliminary experiments. We compress residuals using PQ with M = 32 subspaces and 8-bit codes per subspace. The number of k-means iterations is set to 10 for both standard k-means and Tac.

Clustering time measurements include (i) centroids computation and (ii) centroid assignments for all vectors in the dataset. The sample size for k-means is set to 10M vectors; we assume a 24-hour budget for algorithms' running time. We assess clustering quality by measuring retrieval effectiveness when using the compressed representations to rerank candidates. In order to isolate vector approximation from index approximation, we do not rely on Tachiom for retrieval. Following the methodology of [19], we rerank the top 1,000 documents retrieved by Splade CoCondenser using the RerankIndex with all optimizations disabled—specifically removing early exit and pruning mechanisms—and return the top 10 results for each query. This approach is designed to emulate an exhaustive search while maintaining feasible query times, allowing us to precisely assess our centroids' approximation quality.

The results of this evaluation are reported in Figure 2. As shown in the bottom plot, Tac achieves training times orders of magnitude lower than all competing methods, even when they benefit from Intel MKL acceleration. Specifically, Tac yields speedups up to 84× (Faiss with MKL), 247× (Faiss), and 230× (FastKMeans-rs) over the competitors. Our method clusters nearly 600 million 128-dimensional vectors into 262K clusters in just 8 minutes, potentially benefiting prior methods such as PLAID, EMVB, WARP, and IGP, which all rely on this same centroid budget. Furthermore, Tac scales seamlessly to over 4M centroids in only 102 minutes, while the 24-hour timeout prevents the competitors to surpass 131K centroids (Faiss and FastKMeans-rs) and 524K (Faiss with MKL).

The top plot of Figure 2 compares the MRR@10 achieved by Tac and k-means on Ms Marco-v1 across different centroid budgets. This plot highlights that Tac not only offers substantial speedups but also delivers equal or superior approximation quality at a fixed centroid budget, confirming that our allocation strategy effectively improves centroid assignment and validating the benefits of token-aware design for retrieval effectiveness.

![](images/7302d807e82b824ddbde8582d3174e377804fd174086a2ea7946719ed49a12dd.jpg)

![](images/9ad7bc7fd338df96a70a52f1acbb3d01bac74d24888e56d091008e50b42e62c8.jpg)  
*Figure 2: Ms Marco-v1. Top plot: MRR@10 for 1,000 candidates reranking. Bottom plot: clustering time in minutes.*

## 3.2 Retrieval Evaluation

We evaluate the efficiency-effectiveness trade-off offered by Tachiom by comparing it to three state-of-the-art multivector retrieval methods: EMVB, WARP and IGP. We report the best (lowest) average query time for Tachiom and all selected competitors at different metrics cut-off points.

For Tachiom, we set the number of centroids to 2^21 ≈ 2M for LoTTE and 2^22 ≈ 4M for Ms Marco-v1. The HNSW graphs over centroids are built with M = 32 neighbors and ef_c = 1500. At search time, we perform grid search over: number of retrieved centroids per query token κ_c ∈ {15, 20, 40, 80, 100, 120}, maximum candidate documents κ_d ∈ {250, 500, 1000, 2000, 4000}, and candidate pruning threshold α ∈ {0.35, 0.4, 0.45, 0.5}, with HNSW search parameter ef_s = 3/2 · κ_c. For competitors, we follow the grid search procedures from [19]. All the methods use the same residuals size, i.e. 32 bytes per token vector. This is achieved with PQ32 or its variants for Tachiom and EMVB, and with 2 bits per component for the Scalar Quantization employed by WARP and IGP.

**Results (Table 1):** Tachiom exhibits superior efficiency across all metric cutoffs, achieving speedups ranging from 2.5× to 9.8× compared to the baselines.

| | Ms MARCO-V1 (MRR@10) | | | | LoTTE (Success@5) | | | |
|---|---|---|---|---|---|---|---|---|
| | 39.0 | 39.3 | 39.6 | | 67.5 | 68.0 | | 68.5 |
| WARP | 98ms (9.8×) | 98ms (7.0×) | − | | 49ms (4.4×) | − | | − |
| IGP | 72ms (7.2×) | − | − | | 48ms (4.3×) | − | | − |
| EMVB | 55ms (5.5×) | 56ms (4.0×) | 57ms (3.8×) | | 54ms (4.9×) | 55ms (3.9×) | | 59ms (2.5×) |
| TACHIOM | 10ms | 14ms | 15ms | | 11ms | 14ms | 23ms | |

Tachiom is up to 5.5× faster than the best available data structure EMVB and matches its peak effectiveness despite employing standard Product Quantization (PQ). EMVB relies on computationally expensive PQ variants, namely JMPQ—a supervised method that jointly trains centroids, PQ, and the query encoder—on Ms Marco-v1 and OPQ on LoTTE. These results demonstrate that Tachiom, through its careful centroid selection and ability to scale centroids via Tac, rivals the effectiveness of both JMPQ and OPQ without incurring their additional training overhead.

## 4 Conclusions and Future Work

In this work, we presented Tac, a clustering algorithm that decomposes the global clustering problem into independent per-token subproblems and allocates centroids to tokens based on their frequency and semantic variance, achieving up to 247× speedup over k-means and improving effectiveness. On top of this, we developed Tachiom, a retrieval architecture that leverages high-granularity centroids to efficiently gather candidates, bypassing expensive token-level interactions. Tachiom achieves a 9.8× speedup over state-of-the-art methods. Together, these contributions unlock the practical utility of multivector retrieval for large-scale datasets. Future work will assess the generalizability of Tac beyond ColBERTv2 and evaluate Tachiom on datasets substantially larger than Ms Marco-v1. Additionally, by leveraging the superior approximation quality of Tac centroids, we intend to investigate higher residual compression ratios.
