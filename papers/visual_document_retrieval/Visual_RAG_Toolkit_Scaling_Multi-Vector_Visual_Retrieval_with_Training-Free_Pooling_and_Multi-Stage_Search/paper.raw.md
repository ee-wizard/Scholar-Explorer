# Visual RAG Toolkit: Scaling Multi-Vector Visual Retrieval with Training-Free Pooling and Multi-Stage Search

Ara Yeroyan   
Independent Researcher   
Armenia   
ar23yeroyan@gmail.com

# Abstract

Multi-vector visual retrievers (e.g., ColPali-style late interaction models) deliver strong accuracy, but scale poorly because each page yields thousands of vectors, making indexing and search increasingly expensive. We present Visual RAG Toolkit, a practical system for scaling visual multi-vector retrieval with training-free, modelaware pooling and multi-stage retrieval. Motivated by Matryoshka Embeddings, our method performs static spatial pooling–including a lightweight sliding-window averaging variant–over patch embeddings to produce compact tile-level and global representations for fast candidate generation, followed by exact MaxSim reranking using full multi-vector embeddings.

Our design yields a quadratic reduction in vector-to-vector comparisons by reducing stored vectors per page from thousands to dozens, notably without requiring post-training, adapters, or distillation. Across experiments with interaction-style models such as ColPali and ColSmol-500M, we observe that over the limited (size) ViDoRe v2 benchmark corpus 2-stage retrieval typically preserves NDCG and Recall $@ 5 / 1 0$ with minimal degradation, while substantially improving throughput $( \sim 4 \times \mathrm { Q P S } )$ ; with sensitivity mainly at very large $k$ . The toolkit additionally provides robust preprocessing–high resolution PDF to image conversion, optional margin/empty-region cropping and token hygiene (indexing only visual tokens)–and a reproducible evaluation pipeline, enabling rapid exploration of two-, three-, and cascaded retrieval variants. By emphasizing efficiency at common cutoffs (e.g., $k \leq 1 0 \}$ ), the toolkit lowers hardware barriers and makes state-of-the-art visual retrieval more accessible in practice.

# 1 Introduction

Document retrieval has traditionally relied on text extraction pipelines– OCR, layout analysis, and text-based indexing–that discard the rich visual structure of real-world documents: charts, tables, infographics, coloured headings, and complex layouts. Recent advances in vision-language models (VLMs) have opened an alternative: visual document retrieval, where pages are embedded directly from their rendered images, bypassing OCR entirely [2].

The ColPali family of models [2] adapts the late-interaction paradigm pioneered by ColBERT [3] to the visual domain. Each document page is encoded into a set of patch embeddings by a VLM backbone (PaliGemma-3B for ColPali-v1.3, Qwen2-VL for ColQwen2.5 [1], SmolVLM for ColSmol-500M [7]), and query– document relevance is scored via MaxSim aggregation over all query–patch pairs. This approach has achieved state-of-the-art results on the ViDoRe benchmark [5], significantly outperforming text-only pipelines on visually rich documents.

The scaling bottleneck. Late-interaction accuracy comes at a steep computational cost. A single page from ColPali (v1.3) produces $D = 1 0 2 4$ vectors of dimension $d = 1 2 8$ ; ColQwen2.5-v0.2 accepts dynamic resolutions and produces up to $\sim 7 6 8$ visual tokens per page (we observe $D \approx 7 0 0 { - } 7 5 0$ in practice).

A typical query contains 8–12 words–for example, “What is the ESG risk assessment methodology?” (8 words, $Q \approx 1 0$ tokens1). Scoring one query–document pair requires $Q \times D$ inner products, each of dimension $d$ ; scanning $N$ pages multiplies this:

$$
\mathrm { C o s t } _ { \mathrm { s e a r c h } } = Q \times D \times N \times d \quad \mathrm { ( m u l t i p l y - a d d s ) }
$$

Example. For ColPali with $N = 1 0 , 0 0 0$ , $D = 1 0 2 4$ , $Q = 1 0$ , $d = 1 2 8$ :

$$
\underbrace { 1 0 \times 1 0 2 4 \times 1 0 , 0 0 0 \times 1 2 8 } _ { = 1 . 3 1 \times 1 0 ^ { 1 0 } }
$$

Reducing $D$ from 1024 to 32 pooled vectors (our row-mean pooling):

$1 0 \times 3 2 \times 1 0 , 0 0 0 \times 1 2 8$ multiply-adds per query $ 3 2 \times$ reduction. =4.10×108

The $d$ factor cancels in the ratio, so the saving grows quadratically with the compression ratio $D / D ^ { \prime }$ , regardless of dimension.

Beyond search: index construction. The above concerns search alone. For systems using HNSW indexing, building the graph (with typical $M { = } 1 6$ , $e f _ { \mathrm { c } } = 1 2 8 )$ ) requires $O \big ( N { \cdot } e f _ { \mathrm { c } } { \cdot } \log N { \cdot } M \big )$ pairwise comparisons, where each comparison between two multi-vector points costs $O ( D ^ { 2 } \cdot d )$ . For $N = 1 0 { , } 0 0 0$ pages with $D = 1 0 2 4$ and $d = 1 2 8$ , this amounts to trillions of floating-point operations. Reducing $D$ alleviates both search and index construction costs simultaneously.

Our approach vs. existing work. Existing efficiency techniques focus on quantization [6] (e.g., binary vectors) or engine-level optimisations like PLAID’s centroid pruning, which reduce the cost per comparison but not the number of comparisons. We target an orthogonal axis: reducing the number of stored vectors per page through training-free spatial pooling, then leveraging the compact representations for fast multi-stage retrieval.

# 2 Methodology

We present Visual RAG Toolkit, an end-to-end open-source system for visual document retrieval that covers the full pipeline from PDF ingestion through indexed multi-stage search. The toolkit is designed so that researchers, students, and practitioners can build and evaluate a complete visual RAG system on consumer hardware (see $\ S 4$ ). It provides three core contributions:

(1) Training-free, model-aware spatial pooling. Inspired by the idea behind Matryoshka Representation Learning [4]– that useful representations can be extracted at multiple granularities from a single embedding without additional training–we compress the full set of patch embeddings into compact multi-vector summaries (e.g., from ${ \sim } 1 0 2 4$ to ${ \sim } 3 2$ vectors per page) using static spatial operations tailored to each model’s architecture.

(2) Multi-stage retrieval over Qdrant [6] named vectors, using the compact pooled representations for fast candidate generation and the full patch embeddings for exact MaxSim reranking–all executed server-side in a single API call.

(3) Robust preprocessing and reproducible evaluation: PDF-to-image conversion, optional empty-region cropping, token hygiene, and benchmark scripts that enable systematic ablation across models, pooling strategies, and retrieval configurations.

# 2.1 Token Hygiene

VLMs emit several categories of non-visual tokens alongside the visual patch tokens (the embeddings that each correspond to a spatial region of the input image). These include: (i) special tokens such as CLS, BOS/EOS; (ii) prompt/instruction tokens, e.g., “⟨bos⟩Describe the image” prepended by ColPali (v1.3); and (iii) padding tokens introduced during batch processing, where images of different sizes are padded to a uniform sequence length within a batch, producing trailing zero vectors.

Standard MaxSim treats all tokens equally, allowing non-visual tokens to act as spurious high-similarity attractors that inflate scores. While filtering seems straightforward, it is notably not done by raw ViDoRe benchmark2 submissions–models are evaluated with all tokens, including padding.

We detect and strip non-visual tokens at index time. In practice, ColPali (v1.3) retains 1024 of 1030 total tokens; ColQwen2.5 retains 720−768 (mean 743). This reduces inner products (Eq. 1) and improves quality–our “clean” 1-stage baseline sometimes exceeds published ViDoRe v2 leaderboard scores, because non-visual tokens no longer distort MaxSim. Cleaner vectors also make downstream pooling more reliable.

# 2.2 Empty-Region Cropping

Document pages frequently contain blank margins, headers, and page numbers. We optionally detect and remove low-variance border regions using row/column standard-deviation thresholds, with configurable page-number strip removal. The tighter crop focuses encoder capacity on content, benefiting models with fixed input resolution (ColPali). For dynamic-resolution models (ColSmol, ColQwen2.5), the benefit is twofold: not only does the encoder see more informative pixels, but a smaller cropped image also produces fewer patches and tiles, directly reducing the number of stored vectors per page and the inner-product count at search time.

# 2.3 Spatial Pooling Strategies

Our core insight is that the spatial structure of patch embeddings can be exploited to produce compact summaries without any training. We developed model-specific strategies iteratively, driven by the observation that different backbones require different approaches. We describe them in the order we developed them.

# 2.3.1 ColSmol-500M.

Tile-level mean pooling. ColSmol’s processor resizes each page to $5 1 2 \times 5 1 2$ pixels, partitions it into an $n _ { \mathrm { r o w s } } { \times } n _ { \mathrm { c o l s } }$ grid of tiles (each producing $P = 6 4$ patch tokens), and appends one global tile–a squeezed version of the entire original image–yielding $n _ { \mathrm { r o w s } } { \cdot } n _ { \mathrm { c o l s } } { + } 1$ tile groups (typically $1 2 + 1 = 1 3$ ) and ${ \sim } 8 3 2$ total patch tokens. We mean-pool within each tile group to obtain one vector per tile:

$$
\mathbf { t } _ { i } = \frac { 1 } { P } \sum _ { p = 1 } ^ { P } \mathbf { x } _ { ( i , p ) } \in \mathbb { R } ^ { d } , \quad i = 1 , \dots , n _ { \mathrm { t i l e s } }
$$

Result: ${ \sim } 8 3 2 \to { \sim } 1 3$ vectors–a ${ \bf 6 4 \times }$ compression.

# 2.3.2 ColPali (v1.3).

Row-wise mean pooling. ColPali uses a fixed $3 2 { \times } 3 2$ patch grid (1024 visual tokens, $d = 1 2 8$ ). We reshape tokens to a 2D grid and mean-pool across columns:

$$
\mathbf { r } _ { h } = \frac { 1 } { W } \sum _ { w = 1 } ^ { W } \operatorname { g r i d } [ h , w ] \in \mathbb { R } ^ { d } , \quad h = 1 , . . . , H
$$

Result: $1 0 2 4  3 2$ row vectors–a $\mathbf { 3 2 x }$ reduction.

Conv1d experimental pooling. On top of these row vectors, we apply a uniform sliding window of size $k = 3$ with boundary extension, producing $N { \pm 2 }$ output vectors from $N$ input rows:

$$
y _ { i } = \frac { 1 } { | W _ { i } | } \sum _ { j \in W _ { i } } \mathbf { r } _ { j } , \quad W _ { i } = \left\{ j : | j - ( i { - } r ) | \leq r , \ 0 \leq j < N \right\}
$$

This adds inter-row context at negligible cost. For ColPali, where no learned local mixing exists in the backbone, uniform averaging works well.

2.3.3 ColQwen2.5 (v0.2). ColQwen2.5 [1] builds on Qwen2-VL, which accepts images at variable aspect ratios and applies a learned PatchMerger: each $2 { \times } 2$ block of patch tokens is fused via Layer-$\mathrm { N o r m }  .$ concatenation $ \mathrm { M L P }$ , reducing the grid from $H { \times } W$ to $H _ { \mathrm { e f f } } { \times } W _ { \mathrm { e f f } }$ $\left( H _ { \mathrm { e f f } } \approx \lceil H / 2 \rceil \right)$ ). Because PatchMerger is a learned spatial mixing (not a simple average), each output token already encodes its $2 { \times } 2$ neighbourhood.

Why ColPali’s conv1d failed here. We initially applied the same conv1d approach that worked for ColPali (§2.3.2). It degraded ColQwen2.5’s retrieval quality. The cause: uniform averaging over already-mixed representations double-smooths spatial information, washing out discriminative features; the $N + 2$ border extension further introduces artifacts where the backbone does not expect them. This failure motivated a distinct pooling strategy.

Weighted same-length smoothing (Gaussian / Triangular). Instead of conv1d, we designed same-length smoothing $( N \to N )$ with non-uniform weights:

$$
y _ { i } = \frac { 1 } { Z _ { i } } \sum _ { j = i - r } ^ { i + r } w _ { | j - i | } { \bf r } _ { j } , Z _ { i } = \sum _ { j = i - r \atop 0 \leq j < N } ^ { i + r } w _ { | j - i | }
$$

Boundary indices outside $[ 0 , N )$ are skipped and weights re-normalised. With $k = 3 \ ( r = 1 )$ :

• Gaussian: $w _ { \delta } ~ = ~ \exp ( - \delta ^ { 2 } / 2 \sigma ^ { 2 } )$ , $\sigma = \operatorname* { m a x } ( 0 . 5 , r / 2 )$ ; weights $\approx$ [0.61, 1.0, 0.61]. • Triangular: $w _ { \delta } = \left( r + 1 \right) - \delta$ ; weights $= [ 1 , 2 , 1 ]$ .

Since PatchMerger already provides learned local context, only gentle smoothing is needed; Gaussian $( \sigma \approx 0 . 5 )$ works best as its rapid decay preserves center-row identity.

Adaptive row-mean pooling for dynamic resolution. Since ColQwen2.5’s grid $H _ { \mathrm { e f f } } { \times } W _ { \mathrm { e f f } }$ varies per image, we mean over columns (preserving vertical/reading-order structure), then adaptively downsample rows to at most $T$ vectors (default $T = 3 2$ ) using evenlyspaced bins. Pages with $H _ { \mathrm { e f f } } < T$ are not upsampled.

# 2.4 Multi-Stage Retrieval

Multi-stage retrieve-then-rerank pipelines are a well-established pattern in information retrieval: a cheap first stage (e.g., BM25 or a bi-encoder) retrieves a broad candidate set, and a more expensive model reranks it [3]. We apply the same principle within the multivector paradigm: the cheap stage uses our compact pooled vectors, and the expensive stage uses the full patch embeddings–both stored in the same Qdrant collection as named vectors.

Concretely, each page is stored with: initial (full multi-vector, ${ \sim } 7 0 0 { - } 1 0 2 4 $ vectors), mean_pooling (row/tile-pooled, ${ \sim } 1 3 { - } 3 2 )$ , experimental smoothed variants, and global_pooling (single vector). 2-stage retrieval prefetches top- $K$ candidates via MaxSim on a compact named vector, then reranks with exact MaxSim on initial–executed entirely server-side via Qdrant’s prefetch $^ +$ query API, minimising round-trips. A 3-stage cascade adds a globalpooling prefetch before the pooled-vector stage. The toolkit exposes all hyperparameters (prefetch- $K$ , stage-1 vector choice, top- $k$ , cascade depth) for systematic exploration of the accuracy–latency trade-off.

# 3 Data and Evaluation Protocol

We evaluate on ViDoRe v2 [5], a page-level visual document retrieval benchmark that emphasises realistic, non-extractive queries and diverse document types (charts, tables, infographics, multilingual content). ViDoRe v2 provides four English-language datasets. We select three that are topically distinct: ESG Reports ( $1 5 3 8 \mathrm { \ m u l }$ - tilingual pages, 227 queries), Biomedical Lectures (1016 pages, 639 queries), and Economics Reports (452 pages, 231 queries)– 3006 pages total. The 4th dataset covers the same ESG domain in English only; we exclude it to avoid topical redundancy and keep the distractor analysis clean.

Evaluation scopes and the distractor experiment. We evaluate each configuration in two scopes:

Table 1: Official ViDoRe v2 leaderboard (per-dataset, no preprocessing).   

<table><tr><td></td><td></td><td>N@5</td><td>N@10</td><td>R@5</td><td>R@10</td><td>R@100</td></tr><tr><td>ColPali</td><td>ESG</td><td>.549</td><td>.574</td><td>.539</td><td>.643</td><td>.921</td></tr><tr><td>v1.3</td><td>Bio</td><td>.546</td><td>.588</td><td>.575</td><td>.702</td><td>.914</td></tr><tr><td></td><td>Econ</td><td>.486</td><td>.484</td><td>.272</td><td>.400</td><td>.878</td></tr><tr><td>ColQwen ESG</td><td></td><td>.664</td><td>.682</td><td>.718</td><td>.776</td><td>.925</td></tr><tr><td>2.5</td><td>Bio</td><td>.592</td><td>.631</td><td>.612</td><td>.729</td><td>.901</td></tr><tr><td></td><td>Econ</td><td>.533</td><td>.528</td><td>.297</td><td>.420</td><td>.904</td></tr></table>

(i) Per-dataset: each query searches only its own corpus (comparable to the official ViDoRe leaderboard). This is the natural starting point–it establishes that 2-stage retrieval preserves accuracy on each domain independently.

(ii) Union (distractor): all 3006 pages are merged into a single Qdrant collection, so each query must find its relevant pages among cross-dataset distractors. Since 1-stage cost scales linearly with $N$ (Eq. 1) while 2-stage reranking is capped at $K$ candidates, this scope directly tests whether the speedup advantage grows with corpus size.

Metrics. NDCG and Recall $ @ k \in \{ 5 , 1 0 , 1 0 0 \}$ ; throughput (QPS).

# 4 Experiments

The toolkit exposes dozens of configurable parameters: number of retrieval stages, prefetch- $K$ , stage-1 vector choice, pooling kernel, cropping thresholds, and more. We explore representative configurations that isolate the effect of our pooling and multi-stage retrieval, keeping other variables fixed.

Model and hardware selection. We deliberately select models that are practical on consumer hardware: ColSmol-500M [7] (500M parameters, tile grid, ${ \sim } 8 3 2$ patches), ColPali-v1.3 [2] (3B, fixed $3 2 { \times } 3 2$ grid, 1024 patches), and ColQwen2.5-v0.2 [1] (3B, dynamic resolution, PatchMerger). All vectors are stored in FP16; Qdrant collections are fully in-RAM with no HNSW index. The 500M–3B parameter range is chosen intentionally: these models run comfortably on a single consumer GPU or Apple Silicon Mac, making state-of-the-art late-interaction visual retrieval accessible to researchers, students, and practitioners without datacenter resources. Combined with our pooling and multi-stage retrieval, a complete visual RAG system–from PDF ingestion through indexed search–can be built and evaluated on a laptop.

Baselines. Two baselines are relevant. The first is the official ViDoRe v2 leaderboard score for each model (Table 1), which uses the raw model output (all tokens, including padding and special tokens) without any preprocessing. The second–and our primary comparison–is 1-stage full: exact MaxSim over all stored patch embeddings in our collection, after applying our token hygiene and optional cropping (§2.1–2.2). Notably, even though we store vectors in FP16 (lower precision than the leaderboard’s FP32), our 1-stage baseline occasionally exceeds the official scores (compare Tables 1 and 2)–demonstrating that token hygiene and cropping can matter more than numerical precision. We report all 2-stage results (prefetch $K = 2 5 6$ , top-100) relative to this stronger baseline.

Table 2: Our results (union scope, 3006 pages, with token hygiene).   

<table><tr><td></td><td></td><td>N@5</td><td>N@10</td><td>R@5</td><td>R@10</td><td>R@100</td><td>QPS</td></tr><tr><td>Crriis 1stage base</td><td>ESG</td><td>.551</td><td>.573</td><td>.570</td><td>.661</td><td>.908</td><td>0.28</td></tr><tr><td></td><td>Bio</td><td>.566</td><td>.601</td><td>.611</td><td>.713</td><td>.911</td><td>0.27</td></tr><tr><td></td><td>Econ</td><td>.488</td><td>.490</td><td>.244</td><td>.401</td><td>.855</td><td>0.31</td></tr><tr><td>2stage Conv1d ESG</td><td></td><td>.559+.01</td><td>.576.00</td><td>.576+.01</td><td>.662.00</td><td>.818-.09</td><td>1.27</td></tr><tr><td></td><td>Bio</td><td>.566.00</td><td>.601.00</td><td>.610.00</td><td>.712.00</td><td>.902-.01</td><td>1.37</td></tr><tr><td></td><td>Econ</td><td>.486.00</td><td>.489.00</td><td>.243.00</td><td>.401.00</td><td>.834-.02</td><td>1.39</td></tr><tr><td></td><td>1stage base ESG</td><td>.509</td><td>.564</td><td>.538</td><td>.704</td><td>.856</td><td>0.31</td></tr><tr><td></td><td>Bio</td><td>.575</td><td>.612</td><td>.605</td><td>.718</td><td>.880</td><td>0.32</td></tr><tr><td></td><td>Econ</td><td>.572</td><td>.551</td><td>.308</td><td>.432</td><td>.873</td><td>0.33</td></tr><tr><td>2stage Gauss.</td><td>ESG</td><td>.513.00</td><td>.560.00</td><td>.549+.01</td><td>.690-.01</td><td>.788-.07</td><td>1.15</td></tr><tr><td></td><td>Bio</td><td>.574.00</td><td>.611.00</td><td>.605.00</td><td>.716.00</td><td>.865-.02</td><td>1.28</td></tr><tr><td></td><td>Econ</td><td>.572.00</td><td>.552.00</td><td>.309.00</td><td>.433.00</td><td>.821-.05</td><td>1.25</td></tr><tr><td>1stage base</td><td>ESG</td><td>.404</td><td>.457</td><td>.460</td><td>.620</td><td>.961</td><td>0.50</td></tr><tr><td></td><td>Bio</td><td>.390</td><td>.420</td><td>.431</td><td>.519</td><td>.778</td><td>0.50</td></tr><tr><td></td><td>Econ</td><td>.323</td><td>.327</td><td>.165</td><td>.268</td><td>.692</td><td>0.52</td></tr><tr><td>2stage tiles</td><td>ESG</td><td>.369-.04</td><td>：.401-.06.405-.05</td><td></td><td>.503-.12</td><td>.658-.30</td><td>1.32</td></tr><tr><td></td><td>Bio</td><td>.381-.01</td><td>.406-.01</td><td>.413-.02</td><td>.490-.03</td><td>.677-.10</td><td>1.50</td></tr><tr><td></td><td>Econ</td><td>.329+.01</td><td>.325.00</td><td>.171+.01</td><td>.261-.01</td><td>.636-.06</td><td>1.54</td></tr><tr><td>3stage casc.</td><td>ESG</td><td>.371-.03</td><td></td><td>3.408-.05.406-.05</td><td>.520-.10</td><td>.678-.28</td><td>0.86</td></tr><tr><td></td><td>Bio</td><td></td><td>.371-.02 .397-.02 .399-.03</td><td></td><td>.477-.04</td><td>.655-.12</td><td>0.88</td></tr><tr><td></td><td>Econ</td><td>.332+.01</td><td>.330.00</td><td>.173+.01</td><td>.266.00</td><td>.688.00</td><td>0.88</td></tr></table>

# 5 Results

Table 2 reports our union-scope results $( \mathrm { N } = \mathrm { N D C G }$ , $\mathrm { R } = \mathrm { R e c a l l }$ in the table headers). We conducted many more experiments than shown–varying pooling kernels, prefetch- $K$ , cascade depth, and cropping–and report a representative subset per model.

ColPali and ColQwen2.5: ${ \sim } 4 \times$ faster, nearly lossless. For both 3B models, 2-stage retrieval achieves $3 . 8 \mathrm { - } 4 . 5 \times \mathrm { Q P S }$ while preserving N@5, $\mathrm { N } @ 1 0$ , R@5, and $\mathrm { R @ 1 0 }$ within $\pm 0 . 0 1$ of the 1-stage baseline. Degradation appears only at $\mathbb { R } \ @ \mathbb { 1 } 0 0$ ( $- 0 . 0 2$ to $- 0 . 0 9 )$ ), where the 256-candidate prefetch window limits coverage–acceptable in practice, since RAG applications typically use $k \leq 1 0$ .

ColSmol-500M: small models degrade more. ColSmol shows larger drops (up to $- 0 . 3 0 \mathrm { \ R } \ @ 1 0 0 )$ , suggesting that sub-1B models lack sufficient representational capacity for pooling to remain lossless. The 3-stage cascade recovers some recall but at lower QPS.

Throughput. In per-dataset evaluation (each dataset searched independently, 452–1538 pages), 2-stage yields $\sim 2 \times$ QPS. In the union setting (all 3006 pages combined as distractors), speedup grows to ${ \sim } 4 \times$ . This is consistent with the quadratic cost reduction from $\ S 1$ (Eq. 1): as $N$ grows, 1-stage cost increases linearly while 2-stage reranking is capped at $K = 2 5 6$ candidates. Our corpus is modest; the $2 \times \to 4 \times$ trend with just a $3 \times$ increase in $N$ suggests that larger collections will see even greater gains.

Pooling kernel selection. For ColQwen2.5, conv1d degraded quality (§2.3.3); Gaussian slightly outperformed Triangular. For ColPali, conv1d and tile-based pooling performed comparably.

# 6 Demo System

A Streamlit demo3 lets users upload PDFs, index into a free Qdrant Cloud account using ColPali-family models, and query with visual retrieval (PDF images cropping→pooling indexing multistage search→results). The open-source package4 (pip install visual-rag-toolkit) provides a CLI, modular SDK, and benchmark scripts to reproduce all experiments.

# 7 Conclusion

We presented Visual RAG Toolkit, an end-to-end open-source system that makes multi-vector visual document retrieval practical on consumer hardware. Our central contribution is training-free, model-aware spatial pooling that reduces stored vectors per page from ${ \sim } 1 0 2 4$ to ${ \sim } 3 2$ , yielding a quadratic reduction in innerproduct computations (Eq. 1). Combined with multi-stage retrieval– cheap prefetch on pooled vectors, exact MaxSim rerank on full embeddings–the toolkit achieves up to ${ \sim } 4 \times$ throughput improvement with negligible quality loss at practical cutoffs $( k \leq 1 0 )$ .

The quality degradation we observe concentrates at Recall@100, a regime rarely needed in production RAG systems and chatbots where $k \leq 1 0$ is standard. For the 3B models (ColPali-v1.3 and ColQwen2.5), retrieval metrics at $k \leq 1 0$ remain within $\pm 0 . 0 1$ of the uncompressed baseline–a strong indication that spatial pooling preserves the information most relevant for practical retrieval. The sub-1B model (ColSmol-500M) shows larger drops, pointing to a representational capacity threshold below which aggressive pooling becomes lossy.

Crucially, the speedup advantage grows with corpus size: from ${ \sim } 2 \times$ in per-dataset evaluation to ${ \sim } 4 \times$ in our union (distractor) setting with just a $3 \times$ increase in $N$ . This trend is consistent with the quadratic cost analysis of $\ S 1$ and suggests that larger real-world collections will see even greater efficiency gains.

Limitations and future work. Our pooling strategies are modelspecific: each backbone’s tokenisation and spatial processing requires a tailored approach. Specifically, fixed-grid models (Col-Pali) use conv1d sliding-window pooling, PatchMerger models (ColQwen) require weighted same-length smoothing (Gaussian/Triangular), and tile-based models (ColSmol) use tile-level mean pooling. However, most current visual retrieval models share one of these three architectural patterns, so the existing strategies cover the majority of the ecosystem. For entirely new architectures, the toolkit’s modular design makes it straightforward to implement a new pooling function without changing the retrieval or evaluation pipeline.

Additional directions for future work include: (i) improving pooling quality to the point where 1-stage retrieval on pooled embeddings alone becomes viable–achieving not only speed but also significant storage reduction by eliminating the need to store full patch vectors; (ii) exploring learned pooling (e.g., lightweight adapters) that could close the quality gap for small models; and (iii) combining our vector-count reduction with orthogonal techniques such as quantization and HNSW pruning for multiplicative efficiency gains.

# References

[1] Manuel Faysse, Hugues Sibille, and Tony Wu. 2024. ColQwen2.5-v0.2: A Qwen2.5- VL-based Late-Interaction Retriever. https://huggingface.co/vidore/colqwen2.5- v0.2 Accessed: 2026-02-12.   
[2] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2025. ColPali: Efficient Document Retrieval with Vision Language Models. In International Conference on Learning Representations (ICLR). https://arxiv.org/abs/2407.01449   
[3] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. Association for Computing Machinery, New York, NY, USA, 39–48. doi:10.1145/3397271.3401075   
[4] Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, and Ali Farhadi. 2022. Matryoshka Representation Learning. In Advances in Neural Information Processing Systems (NeurIPS). https://proceedings.neurips.cc/paper_files/paper/2022/hash/ c32319f4868da7613d78af9993100e42-Abstract-Conference.html   
[5] Quentin Macé, António Loison, and Manuel Faysse. 2025. ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval. arXiv preprint arXiv:2505.17166 (2025). https://arxiv.org/abs/2505.17166   
[6] Qdrant Team. 2024. Qdrant: Open-Source Vector Search Engine. https://qdrant. tech Accessed: 2026-02-12.   
[7] Manu Romero. 2024. ColSmol-500M: A Compact Vision–Language Retrieval Model. https://huggingface.co/vidore/colSmol-500M Accessed: 2026-02-12.