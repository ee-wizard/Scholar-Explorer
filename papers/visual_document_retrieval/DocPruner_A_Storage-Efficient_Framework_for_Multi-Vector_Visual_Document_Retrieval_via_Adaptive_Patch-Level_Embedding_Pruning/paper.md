# DocPruner: A STORAGE-EFFICIENT FRAMEWORK FOR MULTI-VECTOR VISUAL DOCUMENT RETRIEVAL VIA ADAPTIVE PATCH-LEVEL EMBEDDING PRUNING

Yibo $\mathbf { Y a n } ^ { 1 , 2 , 3 }$ , Guangwei $\mathbf { X } \mathbf { u } ^ { 2 }$ , Xin $\mathbf { Z o u } ^ { 1 , 3 }$ , Shuliang ${ \bf L i u ^ { 1 , 3 } }$ , James Kwok3, Xuming $\mathbf { H } \mathbf { u } ^ { 1 , 3 , * }$

1Hong Kong University of Science and Technology (Guangzhou), 2Alibaba Cloud Computing, 3Hong Kong University of Science and Technology yanyibo70@gmail.com, kunka.xgw@alibaba-inc.com, xuminghu@hkust-gz.edu.cn

# ABSTRACT

Visual Document Retrieval (VDR), the task of retrieving visually-rich document pages using queries that combine visual and textual cues, is crucial for numerous real-world applications. Recent state-of-the-art methods leverage Large Vision-Language Models (LVLMs) in a multi-vector paradigm, representing each document as patch-level embeddings to capture fine-grained details. While highly effective, this approach introduces a critical challenge: prohibitive storage overhead, as storing hundreds of vectors per page makes large-scale deployment costly and impractical. To address this, we introduce DocPruner, the first framework to employ adaptive patch-level embedding pruning for VDR to effectively reduce the storage overhead. DocPruner leverages the intra-document patch attention distribution to dynamically identify and discard redundant embeddings for each document. This adaptive mechanism enables a significant $50 \%$ reduction in storage for leading multi-vector VDR models with negligible degradation in document retrieval performance. Extensive experiments across more than ten representative datasets validate that DocPruner offers a robust, flexible, and effective solution for building storage-efficient, large-scale VDR systems.

# 1 INTRODUCTION

Visual Document Retrieval (VDR), the task of retrieving relevant document pages based on a query that leverages both visual and textual cues, is of paramount importance in numerous real-world applications, from e-commerce product searches to educational resource discovery (Ding et al., 2024; Zheng et al., 2025; Wang et al., 2025b). In contrast to traditional text retrieval, VDR presents a greater challenge as it must interpret not only the textual content but also the complex layouts, tables, figures, and other visual elements that convey critical information (Abootorabi et al., 2025; Mei et al., 2025). Consequently, this intricate task has garnered increasing attention within the information retrieval community in recent years, driving innovation beyond text-centric paradigms.

The methodology for VDR has undergone a significant paradigm shift. Early approaches were predominantly OCR-based, involving the extraction of text from document images, which was then indexed by conventional text retrievers (Zhang et al., 2024a; Hegghammer, 2022), as shown in Figure 1 (a). However, these methods are often brittle and error-prone, frequently failing to preserve the vital layout and structural relationship inherent in the visual representation (Most et al., 2025; Guo et al., 2025). With the recent advent of Large Vision-Language Models (LVLMs) and their dramatically enhanced visual understanding capabilities (Caffagni et al., 2024), the research community has begun to explore LVLM-based methods, which have demonstrated state-of-the-art retrieval performance (Mace et al. ´ , 2025; Gunther et al. ¨ , 2025; Dong et al., 2025; Tanaka et al., 2025). These methods generally fall into two categories: one that encodes an entire document page and the query into single, holistic embeddings (i.e., page-level retrieval) (Zhang et al., 2024c; Liu et al., 2025b; Jiang et al., 2024b; Meng et al., 2025), and another that represents a document as multiple patch-level embeddings and the query as multiple token-level embeddings, as illustrated in Figure 1 (b). The former approach, while simple, often fails to capture the fine-grained details necessary for understanding complex documents, leading to suboptimal performance. As a result, the latter patch-level retrieval has emerged as the preferred paradigm for leading models.

![](images/5e45b6d19e24ed9c78cba47a611cfae61308ee6284551b18ff798a4f241f0990.jpg)  
Figure 1: The illustration of comparison between OCR-based (a) & LVLM-based (b) paradigms for VDR, and DocPruner (c), a novel framework to adaptively prune the patch-level embeddings for diverse document types.

The ascent of the patch-level retrieval paradigm is primarily attributed to the advantages of multivector retrieval, a technique pioneered by ColBERT-style late interaction (Khattab & Zaharia, 2020). The core mechanism of this approach involves a MaxSim operation, where for each query token embedding, the maximum similarity score against all patch embeddings of a document is computed, and these scores are then aggregated to determine relevance. The VDR field first witnessed the successful application of this paradigm with ColPali (Faysse et al., 2024), which spurred a wave of subsequent works that further refined and enhanced the performance of multi-vector VDR (NomicAI, 2025; Gunther et al. ¨ , 2025; Xu et al., 2025a; Karlinsky et al., 2025). However, despite its effectiveness, the multi-vector approach suffers from a critical efficiency bottleneck: prohibitive storage overhead. Storing hundreds or even thousands of embedding vectors for every single document page makes large-scale deployment costly and challenging (Ma et al., 2025).

To address this critical challenge, we introduce DocPruner, the first framework to successfully employ adaptive pruning in the context of VDR to significantly alleviate storage overhead, as shown in Figure 1 (c). The core of DocPruner is an elegant yet powerful mechanism that leverages the patch-level attention score distribution within a single document to perform adaptive pruning of its patch embeddings. This allows the framework to dynamically adjust the pruning ratio for different documents, achieving a $50 \%$ reduction in patch embeddings for several state-of-theart multi-vector models with negligible performance degradation. While some prior works have explored efficiency optimizations for multi-vector VDR, they are often constrained by pre-defined pruning rates or fixed thresholds (Cheng et al., 2024a; Ma et al., 2025; Tmamna et al., 2024), which lack the adaptability required for diverse, real-world visual documents. We believe that the design philosophy of DocPruner, which enables robust performance even for diverse models and datasets, ensures its flexibility and extensibility for practical, large-scale multimodal retrieval applications.

Our contributions can be summarized as follows:

❶ Pioneering Pruning for VDR. We propose DocPruner, the first framework to introduce an adaptive pruning mechanism to the VDR domain. It achieves a substantial $50 \%$ average patch pruning rate with near-lossless performance, effectively mitigating the storage overhead of topperforming multi-vector VDR models.

❷ Adaptive Property for Diverse Documents. The adaptive nature of DocPruner allows it to dynamically tailor the pruning ratio for different types of visual documents, a feature that is particularly crucial in real-world scenarios where document formats and information densities vary widely.

❸ Extensive Experimental Validation. We conduct comprehensive experiments on diverse and even multilingual VDR benchmarks, demonstrating the effectiveness and robustness of DocPruner when integrated with multiple leading multi-vector retrieval models in the community.

# 2 RELATED WORK

# 2.1 VISUAL DOCUMENT RETRIEVAL

Visual Document Retrieval (VDR) aims to retrieve relevant visually-rich documents based on visual representations, a paradigm that has garnered significant attention from the research community (Zheng et al., 2025; Mei et al., 2025; Zhang et al., 2025b). Previous OCR-based methods rely on document parsing to extract textual content (Xiao et al., 2024; Wang et al., 2023; Karpukhin et al., 2020), a process that can lose critical layout information and fail to interpret non-textual components. Consequently, the field has rapidly evolved from early OCR-plus-retriever pipelines to a paradigm leveraging powerful VLMs as OCR-free retriever backbones. By treating documents as images, VDR systems can preserve this vital structural and visual integrity, enabling a “what-yousee-is-what-you-get” retrieval mechanism that aligns with human perception (Ma et al., 2024).

These VLM-based methods primarily fall into two categories: efficient but less detailed page-level retrieval, and the more powerful patch-level retrieval. Page-level retrievers, such as DSE (Ma et al., 2024), GME (Zhang et al., 2024c) and UniSE (Liu et al., 2025b), encode an entire document page into a single, compact embedding. While efficient, this approach may lose fine-grained details crucial for specific queries. State-of-the-art patch-level retrievers (e.g., ColPali (Faysse et al., 2024), ColQwen (Faysse et al., 2024), ColNomic (NomicAI, 2025), Jina Embeddings v4 (Gunther et al. ¨ , 2025), and Llama Nemoretriever Colembed (Xu et al., 2025a)) achieve superior performance by generating fine-grained, multi-vector representations per page, yet this introduces a critical bottleneck due to prohibitive storage and computational overhead. Our proposed DocPruner directly addresses this pain point by proposing a solution to adaptively reduce the storage footprint of patchlevel embeddings, thereby making high-performance VDR more practical and scalable.

# 2.2 MULTI-VECTOR RETRIEVAL

Multi-vector retrievers, also known as late-interaction models (Khattab & Zaharia, 2020; Ji et al., 2024), computes relevance by first independently encoding queries and documents into sets of tokenlevel embeddings and then performing fine-grained similarity calculations. Formally, given a query $q$ and a document $d$ with $L _ { 1 }$ and $L _ { 2 }$ tokens respectively, they are encoded into embedding matrices $\mathbf { Q } = ( \mathbf { q } _ { 1 } , \dots , \mathbf { q } _ { L _ { 1 } } ) \in \mathbb { R } ^ { P \times L _ { 1 } }$ and $\mathbf { D } = ( \bar { \bf d _ { 1 } } , \dots , \bar { \bf d _ { { L _ { 2 } } } } ) \ \bar { \in } \ \mathbb { R } ^ { P \times { L _ { 2 } } }$ , where $P$ is the embedding dimension. The final score is derived from their token-wise similarity matrix $\mathbf { S } = \mathbf { Q } ^ { \top } \mathbf { D }$ . For instance, ColBERT model (Khattab & Zaharia, 2020) computes the score via a MaxSim operation:

$$
s ( q , d ) = \sum _ { i = 1 } ^ { L _ { 1 } } \operatorname* { m a x } _ { j = 1 } ^ { L _ { 2 } } \mathbf { q } _ { i } ^ { \top } \mathbf { d } _ { j } .
$$

Building on this foundation, ColBERTv2 (Santhanam et al., 2021) introduced a centroid-based method to compress token embeddings for greater storage efficiency. PLAID (Santhanam et al., 2022) further optimized this by using centroid interactions for efficient pruning of low-scoring documents. Other approaches have focused on reducing the number of stored vectors: XTR (Lee et al., 2023) trains the model to prioritize and retrieve only key document tokens, Acquavia et al. (2023) remove embeddings of less impactful tokens, and Clavie et al. ´ (2024) cluster similar token embeddings at indexing time to reduce the total vector count. Recently, MUVERA (Jayaram et al., 2024) proposed using Fixed Dimensional Encodings (FDEs) to approximate the multi-vector similarities, enabling efficient retrieval. Despite their effectiveness, a primary limitation of these text-based multi-vector models is their significant storage overhead, which scales linearly with the number of document tokens $\left( L _ { 2 } \right)$ , resulting in a storage cost of $O ( P \times L _ { 2 } )$ per document, a substantial increase compared to $O ( P )$ cost of single-vector models (MacAvaney et al., 2025; Ji et al., 2024).

The concept of multi-vector retrieval has been extended to VDR, leveraging the fine-grained interaction capabilities to better align textual queries with visual content (Plale et al., 2025; Xu et al., 2025b). Pioneering this direction, ColPali (Faysse et al., 2024) adapted the ColBERT framework by using PaliGemma-3B model (Beyer et al., 2024) to generate multi-vector embeddings directly from document images. Subsequently, Llama Nemoretriever Colembed (Xu et al., 2025a) further advanced this paradigm by modifying Llama-3.2-3B (Grattafiori et al., 2024) with bidirectional attention and employing a two-stage training strategy to achieve state-of-the-art performance on the ViDoRe benchmark. More recently, Jina Embeddings v4 (Gunther et al. ¨ , 2025) proposed a unified Qwen2.5-VL (Bai et al., 2025) architecture that supports both single-vector and multi-vector outputs, utilizing LoRA adapters for task-specific optimization. However, the storage overhead problem persists in these visual models, which remains a critical challenge that DocPruner aims to address.

More related work can be seen in Appendix A.

# 3 METHODOLOGY

In this section, we first formalize the setting of multi-vector VDR in $D$ Section 3.1). We then introduce our proposed framework, DocPruner, detailing its mechanism for adaptive patch-level embedding pruning in $D$ Section 3.2). Finally, we establish a theoretical foundation rooted in information theory to justify its efficacy in $D$ Section 3.3).

# 3.1 TASK FORMULATION

The task of VDR is to retrieve a ranked list of relevant document pages from a large corpus ${ \mathcal { C } } =$ $\{ d _ { 1 } , d _ { 2 } , \ldots , d _ { | C | } \}$ for a given textual query $q$ . In the context of multi-vector VDR (Faysse et al., 2024), both queries and documents are represented by sets of embedding vectors.

Let a query $q$ be a sequence of $L _ { q }$ textual tokens. A VLM-based encoder, denoted as $\Phi ( \cdot )$ , maps this query into a set of token-level embeddings $\mathbf { Q } = \{ \mathbf { q } _ { i } \} _ { i = 1 } ^ { L _ { q } }$ , where each $\mathbf { q } _ { i } \in \mathbb { R } ^ { P }$ and $P$ is the embedding dimension. Similarly, a document page $d$ is first rendered as an image and then processed by the VLM encoder $\Phi ( \cdot )$ , which divides the image into a grid of patches. This process yields a set of $L _ { d }$ patch-level embeddings $\mathbf { D } = \{ \mathbf { d } _ { j } \} _ { j = 1 } ^ { L _ { d } }$ , where each $\mathbf { d } _ { j } \in \mathbb { R } ^ { P }$ .

Following the late-interaction paradigm (Khattab & Zaharia, 2020; Santhanam et al., 2021), the relevance score $s ( q , d )$ is computed via a MaxSim operation as defined in Equation 1. The primary challenge is the storage overhead associated with this representation. Storing the full set of embeddings $\mathbf { D }$ for every document results in a cost of $O ( L _ { d } \times P )$ per page, which is prohibitive for large-scale corpora. Our objective is to generate a pruned set of document embeddings $\mathbf { D } ^ { \prime } \subset \mathbf { D }$ such that its size, $L _ { d } ^ { \prime } = | \mathbf { D } ^ { \prime } |$ , is significantly smaller than $L _ { d }$ $( L _ { d } ^ { \prime } \ll L _ { d } )$ , thereby substantially reducing the storage cost to $\mathcal { O } ( L _ { d } ^ { \prime } \times P )$ while preserving retrieval performance.

# 3.2 THE DocPruner FRAMEWORK

DocPruner is a lightweight, plug-and-play framework applied during the offline indexing phase. It is designed around two core principles: being query-agnostic to enable offline processing and document-adaptive to handle the diverse nature of visual documents. The framework systematically identifies and discards redundant or less informative patch embeddings without requiring any model retraining. The process involves three main steps: quantifying patch importance, applying an adaptive threshold, and scoring with the pruned embeddings. See pseudocode in Section B.

# 3.2.1 QUANTIFYING PATCH IMPORTANCE VIA GLOBAL TOKEN ATTENTION

The central challenge of offline pruning is to estimate the importance of each patch without access to a query. We need a reliable, intrinsic signal of salience. Our key insight is that a VLM, in the process of understanding a document image, already computes such a signal. Specifically, we leverage the attention mechanism directed towards a global token. A global token is a special token whose final hidden state is trained to aggregate and summarize information from the entire input sequence. Its representation must encapsulate the document’s overall semantics.

In our framework, we use the end-of-sequence [EOS] token as the default global token, a common and effective choice in many VLM architectures. We extract the attention weights from the final Transformer layer, as this layer captures the most abstract and semantically rich relationships.

Formally, let $\mathbf { A } ^ { ( L ) }$ be the attention weights from the final layer $L$ . After averaging across all $H$ attention heads to create a smooth, robust attention map $\begin{array} { r } { ( \bar { \mathbf { A } } _ { i , j } ^ { ( L ) } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \mathbf { A } _ { h , i , j } ^ { ( L ) } ) } \end{array}$ , we define the importance score $I ( \mathbf { d } _ { j } )$ for the $j$ -th patch as the attention it receives from the global token:

$$
I ( \mathbf { d } _ { j } ) = \bar { \mathbf { A } } _ { \mathrm { g l o b a l } , j } ^ { ( L ) } .
$$

This process yields a vector of importance scores ${ \mathcal { T } } _ { d } = \{ I ( \mathbf { d } _ { j } ) \} _ { j = 1 } ^ { L _ { d } }$ for each document, which serves as the foundation for our adaptive pruning.

# 3.2.2 ADAPTIVE THRESHOLDING FOR PRUNING

Naive pruning strategies, such as using a fixed pruning ratio or a global threshold, are ill-suited for VDR. Visual documents exhibit vast heterogeneity in information density—a sparse title page has very different characteristics from a dense, text-filled page. A fixed strategy would either overprune the dense page, losing critical information, or under-prune the sparse page, retaining useless background patches. DocPruner’s adaptive thresholding directly addresses this by tailoring the pruning decision to the statistical properties of each individual document.

For a given document $d$ with $L _ { d }$ patch embeddings, we have a corresponding vector of importance scores $\mathcal { T } _ { d } = \{ I ( \mathbf { d } _ { j } ) \} _ { j = 1 } ^ { L _ { d } }$ . Our method computes a document-specific threshold by leveraging the first two statistical moments of these scores. First, we define the mean importance $\mu _ { d }$ , which establishes a baseline salience level for the document’s patches. A high mean suggests the document is generally information-rich. It is formally calculated as:

$$
 \mu _ { d } = \frac { 1 } { L _ { d } } \sum _ { j = 1 } ^ { L _ { d } } I ( { \bf d } _ { j } ) .
$$

Second, we compute the standard deviation $\sigma _ { d }$ , which measures the dispersion of importance scores. A high standard deviation indicates that a few patches are exceptionally important compared to the rest, a hallmark of sparse but salient content. It is calculated as:

$$
\sigma _ { d } = \sqrt { \frac { 1 } { L _ { d } } \sum _ { j = 1 } ^ { L _ { d } } ( I ( \mathbf { d } _ { j } ) - \mu _ { d } ) ^ { 2 } } .
$$

The adaptive pruning threshold $\tau _ { d }$ for document $d$ is then defined as a linear combination of these two statistics: $\tau _ { d } = \mu _ { d } + k \cdot \sigma _ { d }$ , where $k$ is a hyperparameter that acts as a adaptation factor. It determines how many standard deviations above the mean a patch’s importance score must be considered significant. We define the preliminary pruned set of patch embeddings $\hat { \mathbf { D } } _ { d } ^ { \prime }$ as:

$$
\hat { \mathbf { D } } _ { d } ^ { \prime } = \{ \mathbf { d } _ { j } \in \mathbf { D } _ { d } \mid I ( \mathbf { d } _ { j } ) > \tau _ { d } \} .
$$

To handle the edge case where overly aggressive pruning might discard all embeddings (i.e., $\hat { \mathbf { D } } _ { d } ^ { \prime } = $ $\varnothing$ ), we guarantee that at least one embedding is preserved. The final pruned set $\mathbf { D } _ { d } ^ { \prime }$ is defined as:

$$
\begin{array} { r } { \mathbf { D } _ { d } ^ { \prime } = \left\{ \begin{array} { l l } { \hat { \mathbf { D } } _ { d } ^ { \prime } } & { \mathrm { i f } \hat { \mathbf { D } } _ { d } ^ { \prime } \neq \emptyset } \\ { \{ \mathbf { d } _ { j ^ { * } } \} \mathrm { ~ w h e r e ~ } j ^ { * } = \underset { j \in \{ 1 , \dots , L _ { d } \} } { \mathrm { a r g } \operatorname* { m a x } } I ( \mathbf { d } _ { j } ) } & { \mathrm { i f } \hat { \mathbf { D } } _ { d } ^ { \prime } = \emptyset . } \end{array} \right. } \end{array}
$$

# 3.2.3 SCORING WITH PRUNED EMBEDDINGS

The ultimate goal of pruning is to reduce storage and, by extension, accelerate online retrieval, without compromising ranking quality. At query time, the retrieval process remains identical to the standard late-interaction paradigm, with one crucial difference: the search space for the $\mathtt { M a x S i m }$ operation is significantly reduced. Instead of comparing each query token embedding against the full set of document embeddings $\mathbf { D }$ , we use the compact, pruned set $\mathbf { D ^ { \prime } }$ . The pruned relevance score, $s ^ { \prime } ( q , d )$ , is computed as: $\begin{array} { r } { \bar { s ^ { \prime } } ( q , d ) = \sum _ { i = 1 } ^ { L _ { q } } \operatorname* { m a x } _ { \mathbf { d } _ { j } \in \mathbf { D ^ { \prime } } } \bar { \mathbf { q } _ { i } ^ { \top } } \mathbf { d } _ { j } } \end{array}$ . For a given query $q$ , we compute $s ^ { \prime } ( q , d _ { k } )$ for all documents $d _ { k }$ in the corpus to obtain a ranked list. The effectiveness of this ranking is then evaluated using Normalized Discounted Cumulative Gain at rank 5 $\mathrm { ( n D C G } @ 5 \mathrm { ) }$ .

# 3.3 THEORETICAL FOUNDATION

The efficacy of DocPruner can be rigorously analyzed through the Information Bottleneck $\mathbf { ( I B ) }$ principle (Tishby et al., 2000; Saxe et al., 2019; Tishby & Zaslavsky, 2015). The IB framework aims to learn a compressed representation $\mathbf { Z }$ of an input random variable $\mathbf { X }$ that is maximally informative about a target variable $\mathbf { Y }$ . This is formulated as the following optimization problem:

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { \mathbf { Z } } } & { { } \mathcal { L } _ { I B } ( \mathbf { Z } ) = I ( \mathbf { Z } ; \mathbf { Y } ) - \beta I ( \mathbf { Z } ; \mathbf { X } ) , } \end{array}
$$

where $I ( \cdot ; \cdot )$ denotes mutual information and $\beta$ is a Lagrangian multiplier balancing compression and information preservation.

The Intractable Ideal. In our VDR task, $\mathbf { X }$ is the full set of document embeddings $\mathbf { D }$ , $\mathbf { Z }$ is the pruned set $\mathbf { D ^ { \prime } }$ , and the target $\mathbf { Y }$ is the relevance score $s ( q , d )$ , which depends on a future, unknown query $q$ . The ideal objective is to maximize the expected information about relevance over the distribution of all possible queries $P ( q )$ :

$$
\operatorname* { m a x } _ { \mathbf { D } ^ { \prime } } \quad \mathbb { E } _ { q \sim P ( q ) } [ I ( \mathbf { D } ^ { \prime } ; s ( q , d ) ) ] \quad \mathrm { s . t . } \quad | \mathbf { D } ^ { \prime } | \ll | \mathbf { D } | .
$$

This objective is intractable due to the unknown query distribution $P ( q )$ .

DocPruner as a Tractable Approximation. DocPruner offers a principled, tractable approximation to this problem.

▶ Global Token as Relevance Proxy. The hidden state of global token, $\mathbf { h } _ { g l o b a l }$ , serves as a sufficient statistic for document’s relevance to an arbitrary query. That is, $I ( \mathbf { D } ; s ( q , d ) ) \approx I ( \mathbf { D } ; \mathbf { h } _ { g l o b a l } )$ . This axiom posits that the global token’s representation, which summarizes the entire document, captures the necessary information for determining relevance. The attention scores $I ( \mathbf { d } _ { j } )$ directly measure the information flow from each patch to this summary. Therefore, by selecting patches that maximize $I ( { \bf D } ^ { \prime } ; { \bf h } _ { \mathrm { g l o b a l } } )$ , we are effectively approximating the ideal, intractable objective.

▶ Entropy-Aware Pruning. The adaptive threshold $\tau _ { d }$ dynamically adjusts the pruning ratio based on the information entropy of the document’s attention distribution. Let the normalized attention scores form a probability distribution $\begin{array} { r } { p _ { d } ( j ) = \frac { I ( \mathbf { d } _ { j } ) } { \sum _ { i } I ( \mathbf { d } _ { i } ) } } \end{array}$ over the patches. The information content of the document is captured by its Shannon entropy $\begin{array} { r } { \dot { H } ( p _ { d } ) = - \sum _ { j } p _ { d } ( j ) \log p _ { d } ( j ) } \end{array}$ .

1. Low-Entropy Documents: For documents with low information entropy (e.g., title pages), $p _ { d }$ is a sparse, peaky distribution. A few patches have very high attention scores, while most have near-zero scores. The term $k \cdot \sigma _ { d }$ dominates, setting a high threshold $\tau _ { d }$ that isolates only the highly informative “outlier” patches, resulting in aggressive pruning.

2. High-Entropy Documents: For documents with high information entropy (e.g., dense text pages), $p _ { d }$ is more uniform. Attention scores are distributed more evenly across many patches. The threshold $\tau _ { d }$ is more lenient, preserving a larger pathc number that collectively contribute to the document’s meaning.

# 4 EXPERIMENT

4.1 EXPERIMENTAL SETUP

Benchmarks & Evaluation. We conduct our experiments on recent representative VDR benchmarks: ViDoRe-V2 (Mace et al. ´ , 2025) and JinaVDR-Bench (Gunther et al. ¨ , 2025) (More details in Appendix C). We use three state-of-the-art multi-vector VDR models as our base models: ColQwen2.5 (Faysse et al., 2024), ColNomic (NomicAI, 2025), and Jina Embeddings V4 (Gunther et al. ¨ , 2025). Following standard practice in VDR domain (Faysse et al., 2024; Gunther ¨ et al., 2025; NomicAI, 2025; Xu et al., 2025a), we use nDCG $\boldsymbol { \ @ 5 }$ as the primary evaluation metric.

Baselines. We compare DocPruner against three categories of baselines.

(I) Base Models. This represents the original multi-vector models without any pruning or merging.   
They serve as the performance upper bound of storage cost.

(II) Merging-based Methods. Following Ma et al. (2025), the only work focused on VDR storage optimization via merging, we implement three merging strategies:

▶ Sem-Cluster: Merges patch embeddings by performing hierarchical clustering and representing each cluster by its centroid. The tunable hyperparameter is the merging factor, which determines the target number of clusters.

$\blacktriangleright$ 1D-Pooling: Applies 1D average pooling over sequential groups of patch embeddings to reduce their count. The hyperparameter is merging factor, which defines the pooling window size.

▶ 2D-Pooling: Arranges patch embeddings into a 2D grid and applies 2D average pooling. The hyperparameter is merging factor, which must be a perfect square.

(III) Pruning-based Methods: We compare three pruning strategies adapted to VDR context:

![](images/510757be71c3a4ef48ac6d61af74b9cafbf6741c458730bf9de8dc68e5c5c954.jpg)  
Figure 2: Performance comparison $( \mathrm { n D C G } @ 5 )$ between DocPruner and baselines on ViDoRe-V2 benchmark (Mace et al. ´ , 2025) across ColQwen2.5 (Left), ColNomic (Middle), and Jina Embedding V4 (Right). Here, solid lines denote adaptive methods, whereas dashed lines denote non-adaptive ones; circular nodes represent pruning methods, whereas square nodes represent merging methods.

▶ Random: Randomly discards a fixed fraction of patch embeddings, serving as a naive baseline. The hyperparameter is the pruning ratio.

▶ Attention-plus-Similarity: An adaptive method that computes a combined score from both the [EOS] attention (importance) and the embedding similarity to the [EOS] token (representativeness), then prunes patches below a dynamically calculated threshold, following Wen et al. (2025). Hyperparameters include an adaptive factor $\mathrm { k }$ and a weighting factor alpha.

▶ Pivot-Threshold: A two-stage adaptive method that first filters an “important set” of patches using an adaptive attention threshold, and then de-duplicates this set by pruning patches that are too similar to selected “pivot”, following VisPruner (Zhang et al., 2025c). Hyperparameters include an adaptive factor for importance $\mathrm { k }$ , a de-duplication factor k dup, and num pivots.

Implementation Details. To ensure fair and reproducible comparisons, we replicated the base results of three base models in aligned with their respective official implementation. Our evaluation codebase is adapted from the official ViDoRe Benchmark repository1. The complete code for our experiments, including all baseline implementations and the DocPruner framework, will be made publicly available upon acceptance. For DocPruner, the adaptation factor $k$ has a range of $\{ - 0 . 5 , - 0 . 2 5 , 0 , 0 . 2 5 , 0 . 5 , 1 \}$ . The details of hyperparameters for all baseline methods is detailed in the Appendix D. All experiments were conducted on a NVIDIA A100 (80GB) GPU cluster.

# 4.2 EXPERIMENTAL ANALYSIS

In this section, we conduct a comprehensive experimental analysis to answer four key research questions (RQs). (RQ1) How effectively does DocPruner maintain retrieval performance on diverse visual document types while achieving significant storage compression? (RQ2) Can DocPruner’s robust performance generalize to multilingual retrieval scenarios? (RQ3) What is the difference between DocPruner framework and its variants? (RQ4) What are the quantifiable relative improvements in storage efficiency and latency of implementing DocPruner?

# 4.2.1 RETRIEVAL PERFORMANCE COMPARISON (RQ1)

To answer RQ1, we evaluate DocPruner’s performance against a comprehensive set of baselines on the ViDoRe-V2 benchmark. The results, visualized in Figure 2, demonstrate the effectiveness and robustness of our approach across three leading multi-vector models. See more results in Sec.E.1.

Observation ❶: DocPruner achieves near-lossless retrieval performance while pruning 50- $60 \%$ of embeddings, demonstrating remarkable robustness across different base models. As illustrated in Figure 2, DocPruner consistently operates near the performance ceiling set by the unpruned base models (i.e., dashed black line) even when aournd $60 \%$ of embeddings are pruned. For instance, when applied to ColQwen2.5, DocPruner removes $5 1 . 6 \%$ of patch embeddings with a mere 0.0038 drop in nDCG $\textcircled { a } 5$ (from 0.5508 to 0.5470). This high efficiency is mirrored on Jina Embedding V4, where it prunes $5 4 . 1 \%$ of embeddings while the nDCG $\textcircled { \alpha } 5$ only decreases from 0.5687 to 0.5608. Even on the high-performing ColNomic model, DocPruner achieves a $4 3 . 6 \%$ pruning ratio with a negligible performance change (0.5960 vs. the base’s 0.5946), showcasing a remarkable balance between efficiency and accuracy. This robustness stems from DocPruner’s mechanism, which leverages intra-document attention to create a document-specific importance score for each patch, effectively retaining the most semantically salient information necessary for retrieval.

![](images/abe8cdc12390fb004a9823d83280b053b85b74aa7c5c9fd8560da52036297d55.jpg)  
Figure 4: Performance comparison $( \mathrm { n D C G } @ 5 )$ between DocPruner and baselines on JinaVDR benchmark (Gunther et al. ¨ , 2025) across ColQwen2.5 (Left), ColNomic (Middle), and Jina Embedding V4 (Right). Here, solid lines denote adaptive methods, whereas dashed lines denote non-adaptive ones; circular nodes represent pruning methods, whereas square nodes represent merging methods.

Observation $\otimes$ : Pruning-based strategies are generally more effective at preserving retrieval performance than merging-based strategies. This trend is evident across all three models, where methods marked with circles (pruning) consistently form a higher-performance frontier than those with squares (merging). For instance, on the ColNomic model at a around $7 5 \%$ compression ratio, DocPruner achieves an nDCG $\textcircled { \alpha } 5$ of 0.5730 (at a $7 2 . 1 \%$ ratio), whereas the strongest merging baseline, sem-cluster, drops to 0.5426 (at a $7 5 \%$ ratio). The reason for this disparity is that merging, by averaging feature vectors, can dilute the distinctiveness of highly salient patches, blurring important signals. In contrast, pruning preserves the original, high-fidelity embeddings of the most critical patches, vital for the late-interaction mechanism’s ability to find precise query-patch matches.

Observation $\pmb { \otimes }$ : Adaptive pruning methods generally exhibit a superior performance-compression trade-off compared to non-adaptive, fixed-ratio approaches. The solid lines in Figure 2, representing adaptive methods like DocPruner, consistently maintain higher nDCG $\textcircled { \alpha } 5$ scores than their non-adaptive counterparts (dashed lines) at similar compression levels (esp., below $60 \%$ ratio). For example, on the ColNomic model, DocPruner achieves a high $\mathrm { n D C G } @ 5$ of

![](images/07b509cd6ef1638649ac139100ee8fb0f3d8965ec2752bc950f3063268b47bd3.jpg)  
Figure 3: Adaptive pruning ratio values of four different datasets in ViDoRe-V2 across difference k (More in Sec.E.1).

0.5960 with a $4 3 . 6 \%$ pruning ratio, outperforming all non-adaptive baselines. The superiority of adaptive methods is because they intelligently account for the heterogeneity of visual documents (validated by Figure 3); they prune more aggressively on information-sparse pages and more conservatively on information-dense ones, whereas fixed-ratio methods apply a one-size-fits-all strategy that can be suboptimal.

Observation ❹: Notably, merging-based methods exhibit uncharacteristically strong performance on the Jina Embedding V4, in some cases surpassing DocPruner. This phenomenon can likely be attributed to JinaV4’s unique training architecture; its technical report (Gunther et al. ¨ , 2025) reveals that the model is explicitly co-trained to produce a single-vector embedding via mean pooling over its token-level representations. This training paradigm encourages the model to learn patch embeddings that are inherently more aggregable and robust to averaging, making post-hoc merging strategies unusually effective as they align with the model’s intrinsic properties.

# 4.2.2 GENERALIZATION TO MULTILINGUAL SCENARIOS (RQ2)

To answer RQ2, we evaluate DocPruner’s generalization capability on the multilingual JinaVDR benchmark, where we choose documents in German, Russian, Chinese, and Japanese. The overall and per-language results, presented in Figure 4 and Sec.E.2, lead to the following observations.

Observation ❺: DocPruner demonstrates strong and consistent performance across diverse multilingual datasets, maintaining near-lossless retrieval accuracy while achieving substantial storage savings (i.e., around $5 0 \%$ ). For instance, on the ColNomic model, DocPruner achieves a remarkable $6 1 . 0 \%$ overall pruning ratio with a slight increase in $\mathrm { n D C G } @ 5$ from the base’s 0.8151 to 0.8191. Similarly, when applied to ColQwen2.5, it prunes $5 4 . 0 \%$ of embeddings while improving the $\mathrm { n D C G } @ 5$ score from 0.6877 to 0.6958. This robust generalization stems from DocPruner’s core mechanism, which relies on the language-agnostic visual attention patterns within the VLM.

Observation $\pmb { \circledcirc }$ : DocPruner’s adaptive nature is particularly evident in its ability to dynamically adjust pruning ratios for documents in different languages, reflecting varying information densities. This tailored approach is clearly visible in the per-language pruning statistics shown in Section E.2. Using the ColNomic model as an example (with ${ \bf k } = - 0 . 5$ ), DocPruner applies a modest pruning ratio of $9 . 0 \%$ for German documents $( \mathrm { n D C G } @ 5 $ of 0.6022 vs. base 0.5975) and $7 . 0 \%$ for Spanish documents $\mathrm { ( n D C G } @ 5$ of 0.7896 vs. base 0.7927). In contrast, it identifies greater redundancy in other languages, pruning $3 6 . 3 \%$ for Japanese and $3 7 . 6 \%$ for Chinese documents while maintaining high performance. This demonstrates that DocPruner is not applying a uniform rule but is sensitive to the intrinsic properties of the documents themselves, automatically allocating the storage budget proportional to each document’s information entropy.

# 4.2.3 VARIANT STUDY (RQ3)

To answer RQ3, we conduct a variant study comparing DocPruner against pruning-based variants (shown in Figure 5), which are: (I) attention-ratio, a non-adaptive method that prunes a fixed percentage of patches with the lowest attention scores; $\mathbf { \Pi } ^ { ( \mathbf { I I } ) }$ attention-threshold, which uses a fixed, global attention value as the pruning threshold; and (III) attention-threshold-nfp, which enhances the static threshold method with a noise-filtering-prompt (nfp) to guide the model’s focus.

Observation ❼: The document-adaptive statistical thresholding of DocPruner consistently achieves a superior performancecompression trade-off compared to simpler pruning variants that rely on fixed ratios or static thresholds. While all methods leverage attention scores, their pruning criteria differ fundamentally: attention-ratio enforces a uniform compression rate, whereas attentionthreshold and attention-threshold-nfp apply a one-size-fits-all importance cutoff. At a significant pruning ratio of approximately $60 \%$ , DocPruner sustains a high $\mathrm { n D C G } @ 5$ of 0.54; but the performance of the static attentionthreshold variant collapses to below 0.45, and

![](images/3ecbcaf061e1ec6a750409739fd945eecd73380486d8439baf589d8b99c44e75.jpg)  
Figure 5: Overall comparison between DocPruner & variants (See per-dataset analysis in Sec.E.3).

even the improved attention-threshold-nfp and fixed-ratio attention-ratio methods lag considerably.

# 4.2.4 EFFICIENCY ANALYSIS (RQ4)

Observation ❽: DocPruner achieves a substantial storage footprint reduction of approximately $50 \%$ on average with near-lossless retrieval performance, at the cost of an acceptable increase in offline encoding latency. As detailed in Table 1, DocPruner reduces storage footprints by $5 1 . 5 5 \%$ for ColQwen, $4 3 . 6 2 \%$ for ColNomic, and $5 4 . 0 9 \%$ for JinaV4, while the $\mathrm { n D C G } @ 5$ performance changes are minimal $( \downarrow 0 . 6 9 \%$ , $\uparrow 0 . 2 4 \%$ , and $\downarrow 1 . 3 9 \%$ , respectively). This specific setting of ${ \bf k } = - 0 . 2 5$ consistently delivers an optimal trade-off between performance and storage across all multi-vector models. Although DocPruner introduces an overhead that increases offline latency by $6 0 \%$ due to the extra steps of attention score extraction and filtering, the practical impact is modest. The average per-document encoding time increases from a baseline of $0 . 4 7 s$ to only $0 . 7 7 s$ , a duration that is acceptable for an offline indexing phase and vastly superior to 7.22s required by OCR-based method (i.e., OCR+BGE-M3 (Chen et al., 2024a)).

Table 1: Relative improvement of performance, storage, and latency to base models on ViDoRe-V2 (adaptation factor k as -0.25; orange denotes better and green denotes worse).   

<table><tr><td>△</td><td>ColQwen</td><td>ColNomic</td><td>JinaV4</td></tr><tr><td>nDCG@5</td><td>↓0.69%</td><td>↑0.24%</td><td>↓1.39%</td></tr><tr><td> Storage</td><td>↓51.55%</td><td>↓43.62%</td><td>↓54.09%</td></tr><tr><td>Latency</td><td>↑60.00%</td><td>↑65.96%</td><td>↑66.00%</td></tr></table>

# 5 CONCLUSION

In this paper, we addressed the critical challenge of prohibitive storage overhead in state-of-the-art multi-vector VDR systems. We introduced DocPruner, a novel and adaptive framework for patchlevel embedding pruning, which leverages the attention paid by a global token to each image patch to derive a query-agnostic importance score. Crucially, DocPruner employs a document-specific statistical threshold, allowing it to dynamically adjust the pruning ratio for documents of varying information density and complexity. Through extensive experiments across more than ten benchmark datasets, we have demonstrated that DocPruner can achieve a substantial $50 \%$ reduction in stored patch embeddings with only negligible degradation in retrieval accuracy. Future work could explore integrating this pruning mechanism directly into the model training process or extending the adaptive principle to other modalities. Ultimately, DocPruner charts a path toward fine-grained multimodal understanding as practical, real-world applications at an unprecedented scale.

We elaborate the broader impact of DocPruner in Section F.

