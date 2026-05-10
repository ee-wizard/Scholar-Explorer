# Sculpting the Vector Space: Towards Efficient Multi-VectorCannot capturefine-grained info Visual Document Retrieval via PRUNE-THEN-MERGE Framework

Yibo $\mathbf { Y a n ^ { 1 , 2 , 3 } }$ , Mingdong $\mathbf { O u } ^ { 2 , * }$ , $\mathbf { { V i } \thinspace { C a o ^ { 2 } } }$ , Xin $\mathbf { Z o u } ^ { 1 , 3 }$ , Jiahao $\mathbf { H u o } ^ { 1 , 2 }$ , Shuliang $\mathbf { L i u ^ { 1 , 3 } }$ , James Kwok3, Xuming $\mathbf { H } \mathbf { u } ^ { 1 , 3 , }$ †

1Hong Kong University of Science and Technology (Guangzhou), 2Alibaba Cloud Computing, 3Hong Kong University of Science and Technology yanyibo70@gmail.com, xuminghu@hkust-gz.edu.cn

# Abstract

Visual Document Retrieval (VDR), which aims to retrieve relevant pages within vast corpora of visually-rich documents, is of significance in current multimodal retrieval applications. The state-of-the-art multi-vector paradigm excels in performance but suffers from prohibitive overhead, a problem that current efficiency methods like pruning and merging address imperfectly, creating a difficult trade-off between compression rate and feature fidelity. To overcome this dilemma, we introduce PRUNE-THEN-MERGE, a novel two-stage framework that synergizes these complementary approaches. Our method first employs an adaptive pruning stage to filter out low-information patches, creating a refined, high-signal set of embeddings. Subsequently, a hierarchical merging stage compresses this prefiltered set, effectively summarizing semantic content without the noise-induced feature dilution seen in single-stage methods. Extensive experiments on 29 VDR datasets demonstrate that our framework consistently outperforms existing methods, significantly extending the near-lossless compression range and providing robust performance at high compression ratios.

# 1 Introduction

Visual Document Retrieval (VDR) is a critical task that involves retrieving relevant document pages from a vast corpus based on queries that often combine textual and visual cues1. In an era dominated by visually-rich documents such as reports, slides, and academic papers, VDR has become paramount for applications ranging from enterprise search to domain-specific Retrieval-Augmented Generation (RAG) (Zheng et al., 2025; Zhao et al., 2023; Gao et al., 2025). Unlike traditional text retrieval systems that rely on Optical Character Recognition (OCR) to extract textual content (Zhang et al.,

![](images/d40b64dadfb4d8ef6b8fe933237ba2690ac5c617d4af8e2aaa77ecdb8d365978.jpg)  
Figure 1: Comparison of single-vec vs. multi-vec VDR.

2024a), current approaches leverage Large Vision-Language Models (LVLMs) to process entire document pages as images (Jiang et al., 2024b; Meng et al., 2025), as illustrated in Figure 1 (Left). This preserves vital structural and layout information, enabling a more holistic, "what-you-see-is-whatyou-get" understanding that is crucial for accurately interpreting tables, figures, and complex layouts.

The field has rapidly evolved from brittle OCRbased pipelines and page-level (single-vector) models to the current state-of-the-art: patch-level (multi-vector) retrieval, as illustrated in Figure (Right). Pioneered by seminal works like Col-Pali (Faysse et al., 2024), this paradigm represents each document page as a collection of patchlevel embeddings (NomicAI, 2025; Teiletche et al., 2025). This fine-grained representation allows for a late-interaction mechanism, such as MaxSim, which computes nuanced, token-to-patch similarities (Khattab and Zaharia, 2020). This capability to match a query’s specific terms to precise regions within a document has proven to deliver superior retrieval performance, as it captures localized details that single-vector representations often miss.

Despite their exceptional performance, the widespread adoption of multi-vector models is hindered by a critical efficiency bottleneck: prohibitive storage and computational overhead (Santhanam et al., 2022; Chen et al., 2024; Scheerer et al., 2025). Storing hundreds or even thousands of vectors for every single document page makes largescale deployment impractical and costly. Consequently, recent research has focused on optimizing the efficiency, largely converging on two distinct paradigms. The first is pruning-based, exemplified by DocPruner (Yan et al., 2025c), which adaptively discards less informative patches based on intra-document attention, as shown in Suffer at Figure 1 (Left). This approach can achieve a near-Compress%lossless performance with a moderate pruning rate but suffers from a sharp performance decline at higher compression ratios. The second is mergingbased, as explored in Light-ColPali/ColQwen series (Ma et al., 2025), which directly merges multiple patches into fewer vectors, as shown in Figure 1 (Right). This method offers more graceful performance degradation at high compression rates and is simple to implement. However, its crude merging process can dilute discriminative features, leading to an unstable lossless performance range.

![](images/2269a2bfb9bf831c153702ade4e5f9eaea413071969241c60e2a434b6434c936.jpg)  
Figure 2: Comparison of pruning-based vs. merging-based efficient VDR paradigms.

To leverage the strengths of these complementary approaches, we propose PRUNE-THEN-MERGE, a novel two-stage framework for efficient multi-vector VDR. Our core logic is to first refine, then compress. In the first stage, we employ an adaptive pruning mechanism to intelligently filter out low-information patches, such as empty spaces or decorative elements, leveraging pruning’s ability to precisely identify and remove noisy vectors. In the second stage, we apply hierarchical clustering to merge the remaining set of semantically rich patches. By performing a more sophisticated merging operation on a pre-filtered, high-quality set of vectors, our method avoids the feature dilution pitfalls of naive merging and pushes beyond the compression limits of standalone pruning.

We conducted extensive experiments to validate our approach on 29 mainstream VDR datasets, integrating PRUNE-THEN-MERGE with three leading multi-vector models: ColQwen2.5 (Faysse et al., 2024), ColNomic (NomicAI, 2025), and Jina-v4 (Günther et al., 2025). Our results yield the following key findings: $\bullet$ Compared to state-of-the-art pruning work (DocPruner), PRUNE-THEN-MERGE extends the near-lossless compression range by an average of 10 percentage points, from $[ 5 0 - 6 0 \% ]$ to $[ 6 0 - 7 0 \% ]$ . $\pmb { \varrho }$ At high compression rates (e.g., $80 \%$ and above), our method circumvents the sharp performance cliff seen in pruning-only methods, consistently outperforming all baselines.

# 2 Related Work

# 2.1 Visual Document Retrieval

The evolution of VDR can be broadly categorized into three successive paradigms. Early approaches were OCR-based (Xiao et al., 2024; Wang et al., 2023; Karpukhin et al., 2020), relying on OCR tools to extract text from document images, which was then indexed by conventional text retrievers. For a query $q$ and a document image $d$ , this can be formulated as scoring based on the extracted text $T ( d )$ , i.e., $s ( q , T ( d ) )$ . However, this process is often brittle and discards critical layout and nontextual information, limiting its effectiveness on complex documents. The advent of LVLMs catalyzed a shift towards OCR-free methods. The first wave, page-level retrieval (or single-vector retrieval) (Ma et al., 2024a; Zhang et al., 2024b; Liu et al., 2025e), encodes an entire document page $d$ and a query $q$ into single, holistic embeddings. Using an encoder $\Phi ( \cdot )$ , the relevance is typically computed as the similarity between these two vectors: $\boldsymbol { s } ( \boldsymbol { q } , d ) = \sin ( \Phi _ { \boldsymbol { q } } ( \boldsymbol { q } ) , \Phi _ { d } ( d ) )$ where $\Phi _ { q } ( q ) \in \mathbb { R } ^ { D }$ and $\Phi _ { d } ( d ) \in \mathbb { R } ^ { D }$ . While this paradigm preserves the document’s visual integrity, the single-vector representation often fails to capture fine-grained details necessary for precise matching.

To overcome this, the current paradigm has converged on patch-level retrieval (or multi-vector retrieval) (Masry, 2024; NomicAI, 2025; Günther et al., 2025; Masry et al., 2025). Pioneered by ColPali (Faysse et al., 2024), this approach represents a document page as a set of multiple patchlevel embeddings $\mathbf { \bar { D } } = \{ \mathbf { d } _ { j } \} _ { j = 1 } ^ { N _ { p } }$ , where $\mathbf { d } _ { j } \in \mathbb { R } ^ { D }$ and the query as a set of token-level embeddings $\mathbf { Q } = \{ \mathbf { q } _ { i } \} _ { i = 1 } ^ { N _ { q } }$ . Relevance is then computed via a late-interaction mechanism like MaxSim:

$$
s ( q , d ) = \sum _ { i = 1 } ^ { N q } \operatorname* { m a x } _ { j = 1 } ^ { N _ { p } } \mathbf { q } _ { i } ^ { \top } \mathbf { d } _ { j } .
$$

# 2.2 Efficient VDR

Multi-vector VDR models are hampered by a significant efficiency bottleneck: prohibitive storage overhead. Storing a set of $N _ { p }$ embeddings for each document page results in a storage cost of $O ( N _ { p } \times D )$ per page, which is orders of magnitude greater than the $O ( D )$ cost of single-vector models. This challenge has spurred research into two primary efficiency optimization paradigms.

The first is pruning-based strategies, which aim to discard redundant or less informative patch embeddings during the offline indexing stage (Zong and Piwowarski, 2025; Lassance et al., 2022). DocPruner (Yan et al., 2025c) is a representative work in this area, introducing an adaptive mechanism that leverages intra-document attention to identify and remove patches. However, pruning methods often face a sharp performance drop at higher rates. The second paradigm is mergingbased strategies, which reduce the number of embeddings by combining multiple patches into a smaller set of vectors (Clavié et al., 2024; MacAvaney et al., 2025). Light-ColPali (Ma et al., 2025) explores this by applying techniques like spatial pooling or semantic clustering. However, the averaging process inherent in merging can dilute the distinctive features of highly salient patches, often leading to a less stable performance. Furthermore, recent MetaEmbed (Xiao et al., 2025b) compresses tokens via a budget-based compression rate (e.g., Matryoshka embeddings), but it requires welldesigned training and architecture change. Therefore, we propose PRUNE-THEN-MERGE, a hybrid framework that synergizes these two paradigms to overcome their individual limitations.

See more related works in Appendix B.

# 3 Methodology

In this section, we first formalize the task setting of multi-vector VDR. We then introduce PRUNE-THEN-MERGE, detailing its two-stage mechanism.

# 3.1 Task Setting

The task of VDR is to retrieve a ranked list of relevant document pages from a corpus $\begin{array} { r l } { \mathcal { C } } & { { } = } \end{array}$ $\{ d _ { 1 } , d _ { 2 } , \ldots , d _ { | C | } \}$ for a given textual query $q$ . In the multi-vector retrieval paradigm, both queries and documents are represented as sets of embedding vectors. A LVLM-based encoder, denoted as $\Phi ( \cdot )$ , maps a textual query $q$ into a set of $N _ { q }$ token-level embeddings $\mathbf { Q } = \{ \mathbf { q } _ { i } \} _ { i = 1 } ^ { N _ { q } }$ , where each $\mathbf { q } _ { i } \in \mathbb { R } ^ { D }$ and $D$ is the embedding dimension. Similarly, a document page $d$ , rendered as an image, is processed by the same encoder $\Phi ( \cdot )$ to produce a set of $N _ { p }$ patch-level embeddings $\mathbf { D } = \{ \mathbf { d } _ { j } \} _ { j = 1 } ^ { N _ { p } }$ where each $\mathbf { d } _ { j } \in \mathbb { R } ^ { D }$ .

Following the late-interaction mechanism, the relevance score $s ( q , d )$ between the query and the document is computed via the MaxSim operation as defined in Equation (1). The primary challenge addressed in this work is the storage overhead of $O ( N _ { p } \times D )$ per page. Our objective is to generate a compressed set of document embeddings $\mathbf { D } ^ { \prime \prime }$ with size $N _ { p } ^ { \prime \prime } = | \mathbf { D } ^ { \prime \prime } |$ , such that $N _ { p } ^ { \prime \prime } \ll N _ { p }$ , thereby substantially reducing storage costs while minimizing degradation in retrieval performance.

# 3.2 The PRUNE-THEN-MERGE Framework

PRUNE-THEN-MERGE is a query-agnostic, offline compression framework designed to synergize the precision of pruning with the high-ratio compression capability of merging. The framework operates in two sequential stages: (1) an adaptive pruning stage to filter out low-information patches, followed by (2) a hierarchical merging stage to compress the remaining semantically rich patches. See our pseudo-code in Appendix C for reference.

# 3.2.1 Stage 1: Adaptive Pruning

The first stage aims to create an intermediate, refined set of patch embeddings by discarding those with low informational content, such as background or decorative elements. This is achieved by leveraging the LVLM’s internal attention mechanism as a proxy for patch importance.

Formally, during the forward pass of a document image $d$ through the encoder $\Phi ( \cdot )$ , we extract the attention weights $\mathbf { A } ^ { ( L ) } \in \mathbb { R } ^ { H \times S \times S }$ from the final transformer layer, where $H$ is the number of attention heads and $S$ is the sequence length. We average the weights across all heads to obtain a smoothed attention map $\bar { \mathbf { A } } ^ { ( L ) } \in \mathbb { R } ^ { S \times S }$ . The importance score $I ( \mathbf { d } _ { j } )$ for the $j$ -th patch is defined as the attention it receives from a global token (e.g., the [EOS] token): vector of importan $I ( \mathbf { d } _ { j } ) = \bar { \mathbf { A } } _ { \mathrm { e o s } , j } ^ { ( L ) }$ d s a. $\mathscr { T } _ { d } = \overset { \vartriangle } { \{ I ( { \bf d } _ { j } ) \} } _ { j = 1 } ^ { N _ { p } }$

We then compute a document-specific adaptive threshold $\tau _ { d }$ based on the statistical properties of these scores: $\begin{array} { r c l } { \mu _ { d } } & { = } & { \frac { 1 } { N _ { p } } \sum _ { j = 1 } ^ { N _ { p } } { I ( \mathbf { \bar { d } } _ { j } ) } } \end{array}$ ; $\begin{array} { r } { \sigma _ { d } = \sqrt { \frac { 1 } { N _ { p } } \sum _ { j = 1 } ^ { N _ { p } } ( I ( \mathbf { d } _ { j } ) - \mu _ { d } ) ^ { 2 } } } \end{array}$ ; $\tau _ { d } = \mu _ { d } + k \cdot \sigma _ { d }$ where $k$ is a hyperparameter controlling the pruning strictness. The intermediate set of pruned embeddings, $\mathbf { D ^ { \prime } }$ , is formed by retaining only patches whose importance exceeds this threshold: $\mathbf { D } ^ { \prime } =$ $\{ \mathbf { d } _ { j } \in \mathbf { D } \mid I ( \mathbf { d } _ { j } ) > \tau _ { d } \}$ . To ensure robustness, if this process yields an empty set, we retain the single patch with the highest importance score. The resulting set $\mathbf { D ^ { \prime } }$ contains $N _ { p } ^ { \prime } = | \mathbf { D } ^ { \prime } |$ embeddings, where $N _ { p } ^ { \prime } < N _ { p }$ .

# 3.2.2 Stage 2: Hierarchical Merging

The second stage takes the filtered set of highquality embeddings $\mathbf { D ^ { \prime } }$ and further compresses it through semantic clustering. This avoids the feature dilution that occurs when merging is applied to the full, noisy set of patches.

Given the intermediate set D′ ⊂ RN′p×D and a merging factor $m$ , we first determine the target number of clusters $N _ { p } ^ { \prime \prime }$ $^ { \prime } \colon N _ { p } ^ { \prime \prime } = \operatorname* { m a x } ( 1 , \lfloor N _ { p } ^ { \prime } / m \rfloor )$ We then perform hierarchical agglomerative clustering on $\mathbf { D ^ { \prime } }$ . The process involves:

1. Normalization: All embeddings in $\mathbf { D ^ { \prime } }$ are L2-normalized.

2. Distance Matrix: A pairwise distance matrix $\pmb { \Delta } \in \mathbb { R } ^ { N _ { p } ^ { \prime } \times N _ { p } ^ { \prime } }$ is computed based on cosine distance between the normalized embeddings.

3. Clustering: A linkage algorithm (e.g., Ward’s method) is applied to $\pmb { \Delta }$ to build a hierarchy, which is then partitioned into $N _ { p } ^ { \prime \prime }$ clusters. This yields a cluster assignment label for each embedding in $\mathbf { D ^ { \prime } }$ .

For each of the $N _ { p } ^ { \prime \prime }$ clusters, a new representative embedding is generated by computing the centroid (mean) of all member embeddings. This results in the final, compressed set of embeddings $\mathbf { D } ^ { \prime \prime } = \{ \mathbf { d } _ { c } ^ { \prime \prime } \} _ { c = 1 } ^ { N _ { p } ^ { \prime \prime } }$ , where each centroid $\mathbf { d } _ { c } ^ { \prime \prime } \in \mathbb { R } ^ { D }$ This two-stage process effectively reduces the initial $N _ { p }$ embeddings to a much smaller set of $N _ { p } ^ { \prime \prime }$ semantically rich representations.

# 3.2.3 Scoring with Pruned-and-Merged Embeddings

During online retrieval, the relevance score is computed using the final compressed embedding set $\mathbf { D } ^ { \prime \prime }$ . The query $q$ is encoded into $\mathbf { Q }$ as usual, and the MaxSim ssearch space: $\begin{array} { r } { s ^ { \prime \prime } ( q , d ) = \sum _ { i = 1 } ^ { N _ { q } } \operatorname* { m a x } _ { c = 1 } ^ { N _ { p } ^ { \prime \prime } } \mathbf { q } _ { i } ^ { \top } \mathbf { d } _ { c } ^ { \prime \prime } } \end{array}$ ced. By performing the expensive MaxSim operation on a significantly smaller set of embeddings $( N _ { p } ^ { \prime \prime } \ \ll$ $N _ { p , \astrosun }$ ), our framework drastically reduces online computational costs and offline storage requirements, while preserving high retrieval accuracy.

# 3.3 Theoretical Guarantee

The empirical success of the PRUNE-THEN-MERGE framework is underpinned by a sound theoretical basis. We posit that its superiority stems from decomposing a single, complex compression problem into a sequence of two more tractable, specialized sub-problems: (1) query-agnostic information filtering via pruning, and (2) redundancy reduction via merging. This decomposition allows for a more effective approximation of the ideal, yet intractable, Information Bottleneck (IB) objective.

# 3.3.1 The Optimization Problem as an IB

The overarching goal is to compress the full patch embedding set $\mathbf { D }$ into a compact set $\mathbf { D } ^ { \prime \prime }$ that retains maximum relevance to any potential query. Following the IB principle, this can be formulated as: $\mathrm { { \ m a x } _ { D ^ { \prime \prime } } }$ $\mathbb { E } _ { q \sim P ( q ) } [ I ( \mathbf { D } ^ { \prime \prime } ; s ( q , \mathbf { D } ) ) ] - \beta I ( \mathbf { D } ^ { \prime \prime } ; \mathbf { D } )$ where $s ( q , \mathbf { D } )$ is the relevance score and $P ( q )$ is the unknown query distribution. This problem is intractable. Our framework approximates this by decomposing the overall compression mapping $g : \mathbf { D }  \mathbf { D } ^ { \prime \prime }$ into a composition of two functions, $g = g _ { m } \circ g _ { p }$ , where $g _ { p }$ is the pruning map and $g _ { m }$ is the merging map.

# 3.3.2 Pruning as Query-Agnostic Information Filtering

The first stage, pruning, acts as an information filter. We assume the full patch set $\mathbf { D }$ can be partitioned into a high-information signal set $\mathbf { D } _ { \mathrm { s i g } }$ and a lowinformation noise set $\mathbf { D } _ { \mathrm { n o i } }$ . The key assumption is that for any patch $\mathbf { d } _ { \mathrm { n o i } } \in \mathbf { D } _ { \mathrm { n o i } }$ , its contribution to the document’s global semantics (represented by the [EOS] token’s hidden state $\mathbf { h } _ { \mathrm { e o s . } }$ ) is negligible: $I ( { \bf d } _ { \mathrm { n o i } } ; { \bf h } _ { \mathrm { e o s } } ) \approx 0$ , $\forall \mathbf { d } _ { \mathrm { n o i } } \ \in \ \mathbf { D } _ { \mathrm { n o i } }$ . The objective of the pruning stage $g _ { p }$ is to produce an intermediate set $\mathbf { D ^ { \prime } }$ that approximates the true signal set, ${ \bf D } ^ { \prime } \approx { \bf D } _ { \mathrm { s i g } }$ , by maximizing the preserved information about the document’s global meaning: $\begin{array} { r } { \mathbf { D } ^ { \prime } = g _ { p } ( \mathbf { D } ) = \arg \operatorname* { m a x } _ { \hat { \mathbf { D } } \subset \mathbf { D } } \quad I ( \hat { \mathbf { D } } ; \mathbf { h } _ { \mathrm { e o s } } ) . } \end{array}$ subject to a compression constraint. Our adaptive thresholding mechanism, which leverages the attention scores $I ( { \bf d } _ { j } ) \propto I ( { \bf d } _ { j } ; { \bf h } _ { \mathrm { e o s } } )$ , serves as a practical solver for this objective. This filtering step is crucial as it transforms the input for the next stage from a low Signal-to-Noise Ratio (SNR) set $\mathbf { D }$ to a high-SNR set $\mathbf { D ^ { \prime } }$ .

# 3.3.3 Merging as Rate-Distortion Optimization

The second stage, merging, addresses the semantic redundancy within the filtered set $\mathbf { D ^ { \prime } }$ This is a classic vector quantization problem, which can be framed under Rate-Distortion theory. The goal is to find a "codebook" $\mathbf { D } ^ { \prime \prime }$ of size $N _ { p } ^ { \prime \prime }$ (the "rate") that minimizes the information loss, or "distortion," with respect

to the high-SNR signal $\mathbf { D ^ { \prime } }$ . The distortion is quantified by Mean Squared Error (MSE). The objective of merging map $g _ { m }$ is: $\mathbf { D } ^ { \prime \prime } = g _ { m } ( \mathbf { D } ^ { \prime } ) =$ arg min $\begin{array} { r } { \underset { : N _ { p } ^ { \prime \prime } } { \mathbb { E } } \mathbf { d } _ { j } \in \mathbf { D } ^ { \prime } \left[ \operatorname* { m i n } _ { \hat { \mathbf { d } } \in \hat { \mathbf { D } } ^ { \prime \prime } } | | \mathbf { d } _ { j } - \hat { \mathbf { d } } | | _ { 2 } ^ { 2 } \right] } \end{array}$ .   
Dˆ ′′⊂RD ,|Dˆ ′′|=   
Hierarchical clustering followed by centroid   
computation is a highly effective algorithm for   
solvingclusters or a given partition, the optimal centroid $\mathbf { D ^ { \prime } }$ into each $\{ C _ { c } \} _ { c = 1 } ^ { N _ { p } ^ { \prime \prime } }$ $\mathbf { d } _ { c } ^ { \prime \prime }$   
its mean: $\begin{array} { r } { \mathbf { d } _ { c } ^ { \prime \prime } = \underset { \mathbf { v } \in \mathbb { R } ^ { D } } { \arg \operatorname* { m i n } } \sum _ { \mathbf { d } _ { j } \in C _ { c } } | | \mathbf { d } _ { j } - \mathbf { v } | | _ { 2 } ^ { 2 } = } \end{array}$ $\begin{array} { r } { \frac { 1 } { | C _ { c } | } \sum _ { \mathbf { d } _ { j } \in C _ { c } } \mathbf { d } _ { j } } \\ { \frac { 1 } { | C _ { c } | } \sum _ { \mathbf { d } _ { j } \in C _ { c } } \mathbf { d } _ { j } } \end{array}$ . This stage efficiently reduces   
redundancy by creating a compact, summary   
representation of the core semantic concepts.

# 3.3.4 The Synergistic Advantage

The efficacy of PRUNE-THEN-MERGE arises from the synergy between these two stages. Let $\mathbf { d } _ { c } ^ { * } ( S )$ denote the optimal centroid for a cluster within a set $s$ . A naive, single-stage merging approach operates directly on the noisy set $\mathbf { D } = \mathbf { D } _ { \mathrm { s i g } } \cup \mathbf { D } _ { \mathrm { n o i } }$ . Its resulting centroids are inherently biased estimators of the true semantic concepts:

$$
\mathbf { d } _ { c } ^ { * } ( \mathbf { D } ) = \frac { 1 } { | C _ { c } | } \left( \sum _ { \mathbf { d } _ { j } \in C _ { c } \cap \mathbf { D } _ { \mathrm { s i g } } } \mathbf { d } _ { j } + \sum _ { \mathbf { d } _ { k } \in C _ { c } \cap \mathbf { D } _ { \mathrm { n o i } } } \mathbf { d } _ { k } \right) .
$$

The noise vectors $\mathbf { d } _ { k }$ pull the centroids away from the true signal’s center of mass. In contrast, our framework first isolates $\begin{array} { r l } { \mathbf { D } ^ { \prime } } & { { } \approx } \end{array}$ $\mathbf { D } _ { \mathrm { s i g } }$ , so the subsequent merging stage computes centroids $\mathbf { d } _ { c } ^ { * } ( \mathbf { D } ^ { \prime } )$ that are largely unbiased by noise. Consequently, the distortion of the final representation with respect to the true signal is significantly lower for our method. Let $\mathbf { D } _ { \mathrm { o u r s } } ^ { \prime \prime } = g _ { m } ( g _ { p } ( \mathbf { D } ) )$ and $\mathbf { D } _ { \mathrm { n a i v e } } ^ { \prime \prime } = g _ { m } ( \mathbf { D } )$ . We can assert: Edj∈Dsig $\begin{array} { r } { \mathbb { E } _ { \mathbf { d } _ { j } \in \mathbf { D } _ { \mathrm { s i g } } } \left[ \operatorname* { m i n } _ { \hat { \mathbf { d } } \in \mathbf { D } _ { \mathrm { o u r s } } ^ { \prime \prime } } | | \mathbf { d } _ { j } - \hat { \mathbf { d } } | | _ { 2 } ^ { 2 } \right] \ \ll \ } \end{array}$ $\begin{array} { r } { \mathbb { E } _ { \mathbf { d } _ { j } \in \mathbf { D } _ { \mathrm { s i g } } } \left[ \operatorname* { m i n } _ { \hat { \mathbf { d } } \in \mathbf { D } _ { \mathrm { n a i v e } } ^ { \prime \prime } } \mathbf { \bar { \Pi } } | | \mathbf { d } _ { j } - \hat { \mathbf { d } } | | _ { 2 } ^ { 2 } \right] } \end{array}$ . By decomposing the problem, PRUNE-THEN-MERGE ensures that the final compact representation is a more faithful summary of the document’s essential information, leading to a superior performancecompression trade-off.

See more theoretical analysis in Appendix D.

# 4 Experiments and Analysis

# 4.1 Experimental Setup

Benchmarks & Evaluation. Our experimental validation is performed on a comprehensive suite of six representative VDR benchmarks, totaling 29 distinct datasets (more details in Appendix E). These include ViDoRe-V1 (Faysse et al., 2024), ViDoRe-V2 (Macé et al., 2025), JinaVDR-Bench (Günther et al., 2025), REAL-MM-RAG (Wasserman et al., 2025), ViDoSeek (Wang et al., 2025b), and MMLongBench-Doc (Ma et al., 2024b). We integrate our framework with three leading multi-vector models to serve as our base models: ColQwen2.5 (Faysse et al., 2024), Col-Nomic (NomicAI, 2025), and Jina Embeddings v4 (Günther et al., 2025). Following standard VDR practices (Faysse et al., 2024; Günther et al., 2025; Xu et al., 2025a), we adopt nDCG $\boldsymbol { @ 5 }$ as the primary evaluation metric.

Baselines. We benchmark our PRUNE-THEN-MERGE against three categories of baselines. See detailed elaboration of baselines in Appendix F.

(I) Base Models. These are the original, uncompressed multi-vector models. They serve as the performance ceiling and storage-cost upper bound.

(II) Pruning-based Methods. We compare against four strategies that reduce vector count:

▶ Random: A naive baseline that randomly removes a fixed fraction of patch embeddings. Its hyperparameter is the pruning_ratio.

▶ Attention-plus-Similarity: An adaptive approach that prunes patches based on a combined score of importance ([EOS] attention) and representativeness ([EOS] similarity), following Wen et al. (2025). Key hyperparameters are an adaptation factor k and a weighting factor alpha.

▶ Pivot-Threshold: A two-stage adaptive method inspired by VisPruner (Zhang et al., 2025), which first filters patches via an adaptive attention threshold and then de-duplicates the resulting set based on similarity to pivot tokens. Hyperparameters include an importance factor k, a de-duplication factor k_dup, and num_pivots.

▶ DocPruner: A state-of-the-art adaptive method that discards less informative patches based on the intra-document attention distribution, following Yan et al. (2025c). Its primary hyperparameter is the adaptation factor $\mathsf { k }$ .

(III) Merging-based Methods. Following Ma et al. (2025), we compare three merging strategies:

Sem-Cluster: Applies hierarchical clustering to patch embeddings and uses cluster centroids as the merged representation. The tunable hyperparameter is the merging_factor.

![](images/9b11ffd9e3ce0d06cf771e268b894d263d6ad57077ebdbb1a0ededee35da4119.jpg)  
Figure 3: Performance comparison $( \mathrm { n D C G } @ 5 )$ between PRUNE-THEN-MERGE and baselines on ViDoRe-V1 (Faysse et al., 2024) across Jina-v4 (Left), ColQwen2.5 (Middle), and ColNomic (Right). solid lines denote adaptive methods, whereas dashed lines denote non-adaptive ones; circular nodes represent pruning methods, whereas square nodes represent merging ones.

▶ 1D-Pooling: Groups sequential patch embeddings and reduces them via 1D average pooling. The hyperparameter is the merging_factor, defining the pooling window size.

▶ 2D-Pooling: Organizes embeddings into a 2D grid and applies 2D average pooling. The merging_factor must be a perfect square.

Implementation Details. To ensure fair comparisons, we reproduced the results for all base models in alignment with their official implementations. Our evaluation framework is built upon the official ViDoRe Benchmark repository2. For PRUNE-THEN-MERGE, we explore a range of hyperparameters: the adaptation factor $k$ for the pruning stage is selected from $\{ - 1 , - 0 . 7 5 , - 0 . 5 \}$ , and the merging factor $m$ for the merging stage is chosen from $\{ 2 , 4 \}$ . Detailed hyperparameter settings for all baselines are provided in Appendix F. All experiments were conducted on a cluster of NVIDIA A100 (80GB) GPUs. The complete codebased will be made publicly available upon acceptance.

# 4.2 Experimental Analysis

In this section, we conduct a comprehensive experimental analysis to rigorously evaluate PRUNE-THEN-MERGE framework3, which is guided by five research questions (RQs): (RQ1) How does PRUNE-THEN-MERGE maintain its superior performance across a wide variety of visual document types? (RQ2) Does the performance advantage of PRUNE-THEN-MERGE generalize robustly to multilingual retrieval scenarios? (RQ3) Can the framework’s effectiveness extend to more complex, realworld settings that require semantic understanding instead of keyword matching? (RQ4) What is the performance gap between PRUNE-THEN-MERGE framework and its variants? (RQ5) What are the quantifiable gains in storage efficiency that PRUNE-THEN-MERGE achieves?

# 4.2.1 VDR Performance Comparison

Our PRUNE-THEN-MERGE framework consistently demonstrates near-lossless performance at high compression rates, maintaining base performance up to approximately a $70 \%$ pruning rate across a diverse array of 16 datasets from four major VDR benchmarks (We only leave ViDoRe-V1 illustration in main text due to space limit. See the rest in Appendix G.2.). For instance, on the comprehensive ViDoRe-V1 (Figure 3), our method maintains the base $\mathrm { n D C G } @ 5$ of 0.87 with ColQwen2.5, even after compressing the embeddings by $68 \%$ .

At extremely high compression rates $( 8 0 - 9 0 \% )$ , where the performance of baselines typically collapses, PRUNE-THEN-MERGE consistently establishes its superiority over both pruning- and merging-only baselines. Pruning-only methods like DocPruner suffer a sharp performance cliff beyond a $70 \%$ pruning rate. For example, on ViDoRe-V1 with ColQwen2.5 at an ${ \sim } 8 4 { \cdot } 8 7 \%$ pruning rate, PRUNE-THEN-MERGE achieves an $\mathrm { n D C G } @ 5$ of 0.86, whereas DocPruner drops sharply to 0.77. While merging-based methods like Sem-Cluster exhibit more graceful degradation, our framework still frequently matches or outperforms them. This advantage stems from our method’s ability to avoid two failure modes: it circumvents the drastic information loss of aggressive pruning by first identifying and then summarizing semantic clusters, and it avoids feature dilution of naive merging by operating on a pre-filtered, high-signal set of embeddings.

A notable observation is that PRUNE-THEN-MERGE appears to have a slightly extended nearlossless compression range when integrated with the Jina-v4 model. For example, on both the Vi-DoSeek and MMLongBench-Doc, our framework maintains the full baseline performance of the Jinav4 model up to a $7 5 \%$ pruning rate. In contrast, a pure pruning approach like DocPruner experiences a significant performance drop at a similar compression level on these datasets (e.g., from a baseline of 0.54 to 0.48 on MMLongBench-Doc at a $7 7 \%$ rate). We speculate this is linked to Jina-v4’s unique training paradigm, which simultaneously optimizes for both single-vector (pooled) and multi-vector representations (Günther et al., 2025). This inherent training for pooling likely makes its embeddings more amenable to merging operations.

![](images/a143f2158fe1470df1be9e1596eb7da815e2e29daa70bdb4fbee424843af5dbf.jpg)  
Figure 4: Performance comparison $\mathrm { ( n D C G } @ 5 \mathrm { ) }$ between PRUNE-THEN-MERGE and baselines on JinaVDR (Günther et al., 2025) across Jina-v4 (Left), ColQwen2.5 (Middle), and ColNomic (Right). solid lines denote adaptive methods, whereas dashed lines denote non-adaptive ones; circular nodes represent pruning methods, whereas square nodes represent merging ones.

![](images/da867e1ad4c57500262ac02117595bb6330680b5826b216a5db370e4c1c624c1.jpg)  
Figure 5: Performance comparison $\mathrm { ( n D C G } @ 5 \mathrm { ) }$ between PRUNE-THEN-MERGE and baselines on REAL-MM-RAG (Wasserman et al., 2025) across Jina-v4 $( L e f t )$ , ColQwen2.5 (Middle) & ColNomic (Right). solid lines denote adaptive methods, whereas dashed lines denote non-adaptive ones; circular nodes represent pruning methods, whereas square nodes represent merging ones.

# 4.2.2 Generalization to Multilingual Scenarios

Our framework’s effectiveness generalizes robustly across the nine diverse languages in the JinaVDR, proving its utility in global-scale retrieval systems. The results, shown in Figure 4, demonstrate that PRUNE-THEN-MERGE consistently maintains a superior performance-compression trade-off regardless of the language. For example, when paired with ColQwen2.5, our method achieves an overall $\mathrm { n D C G } @ 5$ of 0.52 at an $84 \%$ compression rate, outperforming DocPruner, which scores 0.46 at a lower $80 \%$ rate. This language-agnostic capability stems from our reliance on the VLM’s universal, pre-trained understanding of visual and semantic structures, allowing the framework to identify and compress information without being biased by language-specific features. Due to space limit, see more discussion in Appendix G.3.

# 4.2.3 Generalization to Complex Settings

The superiority of PRUNE-THEN-MERGE extends to the challenging REAL-MM-RAG benchmark, which is designed to test deep semantic understanding through non-extractive, rephrased queries. Across all three base models, our framework consistently outperforms baselines, especially at high compression rates, as shown in Figure 5. With the ColQwen2.5 model, for instance, PRUNE-THEN-MERGE achieves an $\mathrm { n D C G } @ 5$ of 0.65 at an aggressive $84 \%$ compression rate, clearly surpassing DocPruner (0.56) and Sem-Cluster (0.61) at similar or lower rates. This proves that by distilling documents into a set of core semantic centroids, our method creates a representation that is robust to abstract, rephrased queries.

Our framework’s advantage is especially critical for dense, text-heavy document formats, where pruning-only methods falter dramatically under high compression. On the dense Financial Report dataset with Jina-v4, DocPruner’s performance plummets from a 0.69 baseline to just 0.44 at an $84 \%$ pruning rate, marking a $36 \%$ relative drop. In stark contrast, PRUNE-THEN-MERGE gracefully degrades to only 0.66 at a comparable $83 \%$ rate, demonstrating remarkable stability. This suggests that dense documents contain high levels of semantic redundancy that are poorly handled by aggressive pruning; our method’s merging stage excels in this scenario by summarizing semantically related text chunks into robust centroids, preserving the document’s holistic meaning far more effectively. See more discussion in Appendix G.4.

# 4.2.4 Variant Analysis

![](images/f855dcc10e4bef24d26e4430d217b6d1a70902a64ab8b54bc85411a7f03ea81b.jpg)  
Figure 6: Variant comparisons of Jina-v4 on ViDoRe-V2.

We conduct a variant analysis on the ViDoRe-V2 benchmark using the Jina-v4, as shown in Figure 6. We compare PRUNE-THEN-MERGE against several variants: (I) attention-ratio, a non-adaptive baseline that removes a fixed proportion of patches with the lowest attention scores; (II) attention-threshold, a static variant that discards patches based on a single, globally-defined attention threshold; (III) attention-threshold-nfp, which augments the static threshold approach by prepending a noisefiltering prompt to guide the model’s focus; and (IV) attention-threshold-adap, our adaptive pruning stage which is identical to DocPruner.

The complete PRUNE-THEN-MERGE framework consistently outperforms its standalone pruning component (attention-threshold-adap), underscoring the critical contribution of the subsequent merging stage, especially at high compression rates. While our framework and its pruningonly counterpart perform identically at lower compression rates, a clear performance gap emerges as compression becomes more aggressive. For example, at an approximately $80 \%$ compression rate, PRUNE-THEN-MERGE maintains a robust $\mathrm { n D C G } @ 5$ of 0.55, whereas the adaptive pruningonly variant drops to 0.51. This confirms our central hypothesis: the initial pruning stage creates a high-quality, refined set of embeddings that allows the subsequent merging stage to operate more effectively, summarizing semantic concepts without the distortion caused by noise, a limitation inherent to pruning-only approaches at high compression. See more discussion in Appendix G.5.

Table 1: Relative improvement of PRUNE-THEN-MERGE wrt. performance, storage, and latency to base models on ViDoRe-V1 (We set adaptation factor as -0.75 and merging factor as 4; orange denotes better and green denotes worse).   

<table><tr><td>△</td><td>ColQwen</td><td>ColNomic</td><td>JinaV4</td></tr><tr><td>nDCG@5</td><td>↓0.27%</td><td>↓0.51%</td><td>↓0.57%</td></tr><tr><td> Storage</td><td>↓58.88%</td><td>↓52.16%</td><td>↓52.77%</td></tr><tr><td>Latency</td><td>↑56.10%</td><td>↑51.11%</td><td>↑47.06%</td></tr></table>

# 4.2.5 Efficiency Analysis

PRUNE-THEN-MERGE achieves a compelling balance between compression ratio and retrieval fidelity, effectively reducing storage costs by over half with minimal impact on accuracy. As detailed in Table 1 (See details in Appendix G.6), the framework yields a substantial overall storage reduction of $5 4 . 6 0 \%$ across the three base models, reaching a peak reduction of $5 8 . 8 8 \%$ for ColQwen2.5. Despite this aggressive compression, retrieval performance remains remarkably robust, with the average $\mathrm { n D C G } @ 5$ decreasing by only a marginal $0 . 4 5 \%$ relative to the base models. While our approach increases the average per-document encoding latency from 0.46s to $0 . 6 9 s$ , this overhead is incurred exclusively during the offline indexing stage and remains well within a perfectly acceptable range for real-world applications, especially considering that traditional OCR-based retrieval pipelines (e.g., OCR paired with BGE-M3 (Chen et al., 2023)) can often exceed 7 seconds per page.

# 5 Conclusion

In this work, we addressed the critical efficiency bottleneck in multi-vector VDR by proposing PRUNE-THEN-MERGE, a novel two-stage compression framework. Our framework uniquely synergizes the precision of pruning with the high-ratio compression of merging, following a ‘first refine, then compress’ paradigm. Through extensive experiments, we demonstrated that our approach not only extends the near-lossless compression range but also maintains superior performance at aggressive compression rates. Ultimately, PRUNE-THEN-MERGE provides a blueprint for advancing the practical applicability of multi-vector models.

# Limitations

• The efficacy of the pruning stage is inherently tied to the reliability of the base LVLM’s internal attention mechanism as a proxy for patch importance. We plan to investigate more sophisticated, query-independent ranking metrics, such as gradient-based importance, to provide a more accurate signal for information filtering.

• The framework currently relies on predefined hyperparameters, such as the adaptation factor and merging factor, to balance the compression-performance trade-off. We will work toward developing more automated, data-driven strategies that can self-adapt these parameters based on document complexity and layout characteristics.

# Acknowledgments

This work was supported by the Alibaba Innovative Research (AIR) Program (Grant No.64575662); Alibaba Research Intern Program; National Natural Science Foundation of China (Grant No.62506318); Guangdong Provincial Department of Education Project (Grant No.2024KQNCX028); Scientific Research Projects for the Highereducational Institutions (Grant No.2024312096), Education Bureau of Guangzhou Municipality; Guangzhou-HKUST(GZ) Joint Funding Program (Grant No.2025A03J3957), Education Bureau of Guangzhou Municipality.

