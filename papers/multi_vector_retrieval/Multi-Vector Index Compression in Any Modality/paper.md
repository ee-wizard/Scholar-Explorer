# Multi-Vector Index Compression in Any Modality

Hanxiang Qin∗   
hqin14@jhu.edu   
Johns Hopkins University   
Baltimore, MD, USA   
Alexander Martin∗   
amart233@jhu.edu   
Johns Hopkins University   
Baltimore, MD, USA   
Rohan Jha   
rjha5@jhu.edu   
Johns Hopkins University   
Baltimore, MD, USA   
Chunsheng Zuo   
czuo3@jhu.edu   
Johns Hopkins University   
Baltimore, MD, USA   
Reno Kriz   
rkriz1@jhu.edu   
Johns Hopkins University   
Baltimore, MD, USA   
Benjamin Van Durme   
vandurme@jhu.edu   
Johns Hopkins University   
Baltimore, MD, USA

# Abstract

We study efficient multi-vector retrieval for late interaction in any modality. Late interaction has emerged as a dominant paradigm for information retrieval in text, images, visual documents, and videos, but its computation and storage costs grow linearly with document length, making it costly for image-, video-, and audiorich corpora. To address this limitation, we explore query-agnostic methods for compressing multi-vector document representations under a constant vector budget. We introduce four approaches for index compression: sequence resizing, memory tokens, hierarchical pooling, and a novel attention-guided clustering (AGC). AGC uses an attention-guided mechanism to identify the most semantically salient regions of a document as cluster centroids and to weight token aggregation. Evaluating these methods on retrieval tasks spanning text (BEIR), visual-document (ViDoRe), and video (MSR-VTT, MultiVENT 2.0), we show that attention-guided clustering consistently outperforms other parameterized compression methods (sequence resizing and memory tokens), provides greater flexibility in index size than non-parametric hierarchical clustering, and achieves competitive or improved performance compared to a full, uncompressed index.1

# CCS Concepts

• Information systems Search index compression.

# Keywords

Multi-vector representations, Index compression, Late interaction, Omni-modal retrieval

# ACM Reference Format:

Hanxiang Qin, Alexander Martin, Rohan Jha, Chunsheng Zuo, Reno Kriz, and Benjamin Van Durme. 2026. Multi-Vector Index Compression in Any

![](images/05dd112667e3240a8b85aac68baea561b5db5d50406a4852a00d844c1d8876cf.jpg)  
Figure 1: We explore index compression in any modality. We introduce SeqResize, projection-based, MemTok, tokenbased, H-Pool, heuristic-based, and AGC (Ours), hybrid attention-similarity. AGC better utilizes index tokens while maintaining performance $\mathbf { ( n D C G } @ 1 0 )$ at high compression.

Modality. In Preprint. ACM, New York, NY, USA, 12 pages. https://doi.org/ 10.1145/nnnnnnn.nnnnnnn

# 1 Introduction

Online information is increasingly multimodal, including videos, articles with figures or images, podcasts, and interactive web content. It is thus essential that information retrieval systems be able to index over large multimodal collections. However, indexing multimodal content at scale requires an incredible amount of storage,2 limiting the ability of search providers to build truly multimodal indices. While recent advances in multi/omnimodal retrieval [7, 12, 27, 33, 42] have begun to make performance gains, significant progress is still needed to achieve scalable performance in real-world settings.

In this work, we focus on multi-vector late interaction [21], which has shown promise in multimodal domains [12, 13, 38, 42, 50, 55]. Optimizations like ColBERTv2 [44] have improved efficiency through a two-stage retrieval pipeline that uses document cluster centroids to avoid scoring every document, enabling sub-linear scaling in collection size. However, the computation and storage cost still grow linearly with document length [43, 53]. This linear scaling presents a prohibitive barrier for multimodal corpora, where a single multimodal document could easily reach thousands of tokens. Additionally, these thousand token representations are often underutilized in late-interaction, making the indices built from them largely unnecessary in practice (Figure 1). We find that most multimodal late interaction models use only about $1 \%$ of their index during a full evaluation pass. To address this gap, we propose learning compact, query-agnostic, multi-vector, multimodal document representations under a constant vector budget. By bounding the document representation to a constant size, we ensure that both index storage and query-time costs remain manageable and fully customizable to storage or compute constraints, while retaining the benefits of fine-grained late interaction.

We adapt three strong performing multi-vector compression methods from textual domain to multimodal: (1) Sequence Resizing (SeqResize) [e.g., 35], where a full multi-vector document is projected down along the sequence dimension by an MLP; (2) Memory Tokens (MemTok) [e.g., 32, 53], where learnable vectors are appended to the document context and used as the representation; and (3) Hierarchical Pooling (H-Pool) [e.g., 9], which iteratively groups similar vectors and replaces them with their mean. However, these methods are ill-suited for multimodal compression as they struggle to handle redundant and noisy inputs, or suffer from representation collapse. To address these limitations, we introduce a novel attention-guided clustering, where learnable universal query tokens are used to guide the attention to select centroids and weight the aggregation for clustering (AGC).

We evaluate these methods across four tasks and three modalities: BEIR [48], a document retrieval benchmark (text), ViDoRe [36], a visual document retrieval benchmark (vision), MSR-VTT [54], a video-retrieval benchmark (vision), and MultiVENT 2.0 [23], a video-retrieval benchmark (audiovisual). On these benchmarks, we provide an extensive set of experiments and introduce new state-ofthe-art results on ViDoRe and MSR-VTT. We find that AGC is the strongest compression technique in any modality, offering the best performance at learned compression rates and better transferability between sizes than non-parametric compression (H-Pool). Additionally, we find that training with a compression objective can improve performance over an uncompressed multi-vector index on ViDoRe and MSR-VTT, highlighting that compression reduces the redundancy and noise of multimodal inputs.

Our contributions are summarized as follows:

(1) We introduce four methods for index compression in any modality: SeqResize, MemTok, H-Pool, and AGC. (2) AGC presents a novel approach, in which learnable universal query tokens select centroids and weight cluster pooling. (3) We present a series of experiments demonstrating the strong performance and flexibility of AGC across document, visual document, and video retrieval settings.

# 2 Related Work

Multimodal Retrieval. Many works have introduced benchmarks for evaluating representations in information retrieval. In the textonly setting, evaluation suites such as MS MARCO [3] and BEIR [48], have become standard for measuring retrieval effectiveness across diverse domains, tasks, and query types. Video retrieval has been extensively studied using benchmarks that use natural language descriptions to retrieve videos, such as, MSR-VTT [54], VATEX [51], DiDeMo [15], and ActivityNet Captions [22]. More recently, MultiVENT 2.0 [23] has provided a large-scale, multilingual benchmark for real world video retrieval. Visual document retrieval has also recently emerged as another challenging multimodal task, requiring strong optical character recognition ability and visual understanding of layout and graphics, e.g., ViDoRe [36] and MMDocIR [10]. Complementary efforts have introduced modality-specific embedding benchmarks, including MTEB for text embeddings [39], MSEB for audio embeddings [14], and MMEB for vision-language embeddings [20]. In this work, we focus on the following settings: text, visual document, video (vision only) and video (audiovisual). We believe this covers the most challenging combinations of modalities and model capabilities.

Multi-Vector Index Compression. Multi-vector embeddings offer a number of distinct axes by which to compress the index. Naturally, being just a collection of vectors, it is amenable to the same quantization [11, 18] and truncation [19, 24] methods as singlevector retrieval. It is also the norm in multi-vector text retrieval to down-project the encoder’s large hidden dimension to a more manageable dimension (e.g. $^ { 7 6 8 } \to 1 2 8$ ) [21]. Furthermore, as is the focus of this paper, multi-vector indices can be compressed along the sequence dimension as well in order to end up with fewer tokens. This is generally split between methods that prune tokens according to simple corpus-level or attentional heuristics [1, 61], pool them implicitly via special tokens that aggregate meaning or explicitly via heuristic representation merging [9, 16, 25, 41, 49], or simply project the sequence length into a fixed quantity of embeddings [35]. Finally, index methods like PLAID [43, 44] cluster the document token vectors and represent each as its nearest cluster centroid plus a low-bit-quantized version of its residual.

Attention-based Compression. To address the computational burden of long contexts for language models, prior work explores token compression via KV cache eviction. While query-aware methods effectively prune tokens based on prompt attention [8, 17, 28, 30, 47, 58, 59], they are incompatible with retrieval indexing, which requires document representations to be computed before the query is known. On the query-agnostic side, approaches instead leverage self-attention scores or learnable parameters to determine token importance [5, 6, 41, 56, 60]. However, a critical gap remains: these methods optimize for generative tasks by preserving the global "gist," whereas retrieval requires retaining discriminative details needed for distinguishing hard negatives from positive documents.

# 3 Preliminaries

Retrieval. Given a collection of documents $\mathcal { D }$ , where each document $d \in \mathcal { D }$ contains one or more modalities (e.g., text, audio,

visual), and a text query $q$ , we look to provide a ranking of documents in $\mathcal { D }$ based on their relevance to $q$ .

Late Interaction. To calculate the similarity score $s$ between two multi-vector representations, we adopt ColBERT-style Late Interaction [21].

Given a query representation Q and a document representation C, we compute the relevance score $s ( q , d )$ via the MaxSim operation:

$$
s ( q , d ) = \sum _ { i = 1 } ^ { n _ { q } } \operatorname* { m a x } _ { 1 \leq j \leq m } \langle \mathbf { q } _ { i } , \mathbf { c } _ { j } \rangle
$$

where $\langle \cdot , \cdot \rangle$ denotes the dot product. By summing the maximum similarities for each $q _ { i }$ , we get a score for $d$ ’s relevance to $q$ .

# 3.1 Problem Formulation

We formulate the compression of multimodal documents as the process of generating optimal representations under constraints for scalable late-interaction retrieval.

Query and Document Representation. We employ a query encoder $\phi$ that maps a query $q$ to a sequence of token embeddings ${ \bf Q _ { \alpha } } \in \mathbf { \Xi }$ $\mathbb { R } ^ { n _ { q } \times h }$ , where $n _ { q }$ is the sequence length and $h$ is the embedding dimension. We do not constrain the query length.

For documents, we define a mapping $\pi$ that transforms a document $d$ into a sequence of $m$ vectors:

$$
\pi : d \mapsto \mathbf { C } = \left[ \mathbf { c } _ { 1 } , \hdots , \mathbf { c } _ { m } \right] \in \mathbb { R } ^ { m \times h }
$$

Here, $m$ is a fixed budget of vectors independent of the document’s original length. The mapping $\pi$ represents the full representation generation pipeline which may involve direct encoding, parametric or heuristic compression, or a combination of these operations.

A critical constraint in retrieval is that this mapping must be applied during indexing, where the query $q$ remains unknown. Consequently, $\pi$ must compress the document in a query-agnostic manner while preserving information likely to be relevant for future queries. In this work, we explore both unparameterized and parameterized compression techniques, denoting parameterized formulations as $\pi _ { \theta }$ , where $\theta$ denotes learnable weights.

Objective. Our goal is to define $\pi$ such that a scoring function $S ( \mathbf { Q } , \mathbf { C } )$ assigns higher scores to relevant query-document pairs compared to less relevant ones. For parameterized mappings, this becomes an optimization problem where we seek to maximize retrieval accuracy within the fixed storage budget $m$ .

# 4 Multi-Vector Compression

We introduce three methods for multi-vector compression based on prior work (pictured in Figure 2): sequence resizing (SeqResize), memory tokens (MemTok), and hierarchical pooling (H-Pool).

# 4.1 Sequence Resizing

SeqResize is a parameterized compression method that projects the output of an encoder along the sequence dimension to a compressed representation with a fixed number of tokens. Prior work first used this method in document compression for text retrieval [35]. SeqResize is an intuitive approach, allowing an encoder to fully contextualize all representations in a document and then parameterize compression separately (but trained jointly) from the encoder.

To perform SeqResize, the tokens of the document $\mathbf { X } \in \mathbb { R } ^ { n \times h }$ are passed into the bidirectional encoder $F _ { \mathrm { e n c } }$ , which is an $L$ -layer transformer. Let $\mathbf { Z } ^ { ( L ) } = F _ { \mathrm { e n c } } ( \mathbf { X } ; \theta ) \in \mathbb { R } ^ { n \times h }$ be the last-layer hidden states. Since $n$ varies across documents, we first pad or truncate ${ \mathbf Z } ^ { ( L ) }$ to a fixed length $n _ { 0 }$ :

$$
\bar { \mathbf { Z } } ^ { ( L ) } = \mathrm { P a d T r u n c } ( \mathbf { Z } ^ { ( L ) } , n _ { 0 } ) \in \mathbb { R } ^ { n _ { 0 } \times h } .
$$

We then resize along the sequence dimension to produce the compressed multi-vector representation $\mathbf { C } \in \mathbb { R } ^ { m \times h }$ using a 2-layer MLP:

$$
\mathbf { C } = \left( \sigma \big ( \bar { \mathbf { Z } } ^ { ( L ) \top } \mathbf { W } _ { 1 } ^ { \top } \big ) \mathbf { W } _ { 2 } ^ { \top } \right) ^ { \top } , \qquad \mathbf { W } _ { 1 } \in \mathbb { R } ^ { d \times n _ { 0 } } , \ \mathbf { W } _ { 2 } \in \mathbb { R } ^ { m \times d } .
$$

Here, $h$ is the hidden dimension, $\theta$ are the parameters of the encoder, and $\sigma$ is a nonlinearity (e.g., ReLU). The transpose in the MLP form indicates that the same MLP maps each hidden channel’s length- $n _ { 0 }$ sequence to a length- $m$ sequence, i.e., it operates over the sequence dimension. The parameters of $\theta$ , $\mathbf { W } _ { 1 } , \mathbf { W } _ { 2 }$ are the learnable parameters of the compression function.

# 4.2 Memory Tokens

MemTok is a parameterized compression method that appends learnable "memory tokens" to a document context to use as the document representation. Prior works have used this method in document compression for retrieval [53] and generation [32]. MemTok approaches are a straightforward compression method, allowing for document compression to leverage a single encoder, instead of parameterizing compression in a different network (e.g., SeqResize). Following these works, we aim to learn memory tokens for any document representation.

To perform compression with MemTok, we append a set of $m$ memory tokens $\mathbf { M } ^ { \bar { \mathbf { \alpha } } } \in \mathbb { R } ^ { m \times h }$ to the document tokens $\mathbf { X } \in \mathbb { R } ^ { n \times h }$ , and feed the concatenated sequence into the encoder $F _ { \mathrm { e n c } }$ , which is an $L$ -layer transformer. After bidirectional self-attention, each memory token attends over the entire document, and we discard all non-memory positions. The final states of the $m$ memory tokens form the compressed multi-vector representation of the document.

Let $\mathbf { Z } ^ { ( L ) } \bar { \mathbf { \Psi } } = [ \mathbf { Z } _ { \mathrm { X } } , \mathbf { Z } _ { \mathrm { M } } ] \mathbf { \Psi } \in \mathbb { R } ^ { ( n + m ) \times \bar { h } }$ be the hidden states of the last layer of the encoder, where $\mathbf { Z } _ { \mathbf { X } } \in \mathbb { R } ^ { n \times h }$ and $\mathbf { Z _ { M } } \in \mathbb { R } ^ { m \times h }$ are the hidden states of the document and memory tokens, respectively. Formally, the compressed multi-vector representation of the document $\mathbf { C } \in \mathbb { R } ^ { m \times h }$ is given by:

$$
\begin{array} { c } { { { \displaystyle [ { \bf Z } _ { \bf X } ^ { ( L ) } , { \bf Z } _ { \bf M } ^ { ( L ) } ] = F _ { \mathrm { e n c } } ( { \bf \bar { \omega } } [ { \bf X } , { \bf M } ] ; \theta ) , } } } \\ { { { \displaystyle { \bf C } = [ { \bf c } _ { 1 } , { \bf c } _ { 2 } , . . . , { \bf c } _ { m } ] = { \bf Z } _ { \bf M } ^ { ( L ) } . } } } \end{array}
$$

where $h$ is the hidden dimension, $\theta$ are the parameters of the encoder, which are the learnable parameters of the compression function. In our experiments, the memory tokens $\mathbf { M }$ are initialized as learnable parameters and updated during training.

# 4.3 Hierarchical Pooling

H-Pool is a non-parametric compression method that iteratively groups similar vectors and replaces them with their mean. Prior work has introduced this method in document compression in the text domain [9]. Unlike the other approaches, H-Pool does not require the model to be trained for compression and allows for a simple heuristic-driven approach agnostic to modality or model.

![](images/6eaf96c2add312ca008bcdcd6780e17d8c9de4b8a21faa13fb37fe6c2196660f.jpg)  
Figure 2: Overview of multi-vector index compression techniques. (a) AGC uses universal query tokens to guide attention-based centroid selection and weight the aggregation of clustering. (b) MemTok appends tokens to the document context to act as the final representation. (c) SeqResize down projects a document representation along the sequence dimension. (d) H-Pool iteratively groups similar vectors and replaces them with their mean.

For each sequence, H-Pool starts from a sequence of token embeddings $\mathbf { X } \in \bar { \mathbb { R } } ^ { n \times h }$ . We compute a cosine distance matrix $\mathbf { R } \in \mathbb { R } ^ { n \times n }$ with entries

$$
r _ { i j } = 1 - { \frac { x _ { i } ^ { \top } x _ { j } } { \| x _ { i } \| _ { 2 } \| x _ { j } \| _ { 2 } } } ,
$$

which we use to run agglomerative hierarchical pooling with Ward linkage [52]. At any step of the algorithm, we maintain clusters as index sets $\{ \mathcal { A } _ { 1 } , . . . , \mathcal { A } _ { k } \}$ ; each cluster ${ \mathcal { A } } _ { a }$ has centroid

$$
\mu _ { a } = \frac { 1 } { | \mathcal { A } _ { a } | } \sum _ { i \in \mathcal { A } _ { a } } x _ { i } .
$$

Ward’s method iteratively merges the pair of clusters $( \mathcal { A } _ { a } , \mathcal { A } _ { b } )$ that minimizes the increase in within-cluster squared error,

$$
\Delta _ { a , b } = \frac { \left| \mathcal { R } _ { a } \right| \left| \mathcal { R } _ { b } \right| } { \left| \mathcal { R } _ { a } \right| + \left| \mathcal { R } _ { b } \right| } \left\| \mu _ { a } - \mu _ { b } \right\| _ { 2 } ^ { 2 } ,
$$

until exactly $m - m ^ { \prime }$ clusters remain, yielding a partition $\{ \mathcal { A } _ { 1 } , . . . , \mathcal { A } _ { m - m ^ { \prime } } \}$ . The pooled token embeddings are then defined as the mean of each cluster:

$$
c _ { j } = \frac { 1 } { | \mathcal { A } _ { j } | } \sum _ { i \in \mathcal { A } _ { j } } x _ { i } , \quad j = 1 , . . . , m - m ^ { \prime } ,
$$

In implementation, we provide the option to keep $m ^ { \prime }$ tokens as protected tokens and concatenate them back to the pooled token embeddings to form the final compressed sequence $\mathbf { C } = \left[ c _ { 1 } , \ldots , c _ { m } \right]$ .

# 4.4 Limitations

SeqResize, MemTok, and H-Pool reveal limitations when applied to multimodal data under a fixed token budget constraint. First, parametric methods like SeqResize exhibit a modeling failure in which many tokens remain unused during a single evaluation pass; consequently, it fails to scale effectively with the token budget (see subsection 6.4). MemTok suffers from information collapse: its architecture inherently smooths over distinct features, impeding the effective utilization of multi-vector representations. (See subsection 6.5). Second, neither SeqResize nor MemTok possesses the necessary heuristics to eliminate redundant information. Audio and visual signals are often semantically empty or redundant, such as silent audio segments, static backgrounds, and unchanged temporal sequences [4, 26]. These methods waste their limited token budgets on encoding repetitive and noisy signals rather than capturing key semantic content. Finally, while H-Pool actively removes redundancy, the reliance on greedy iterative merging makes them vulnerable to noisy outliers, like the aforementioned noise in multimodal data.

# 5 Attention-Guided Clustering

We now describe Attention-Guided Clustering (AGC), a compression technique designed to maximize the utility of a fixed token budget for document compression in any modality. AGC (shown in Figure 2 (a)) combines three main components: (i) Attention-based Centroid Selection, which utilizes learned universal query tokens to identify semantically salient information; (ii) Hard Clustering, which uses hard assignment to group tokens to reduce redundancy while preserving distinct semantic details; and (iii) Weighted Aggregation, which constructs the final compressed representations by averaging tokens within each cluster weighted by their saliency to mitigate the optimization challenges of hard operations.

# 5.1 Attention-based Centroid Selection

The first component of our approach is to identify information-rich regions within a document. To estimate token importance without specific user queries, we introduce learned “universal queries,” special tokens that probe the document for significant content.

Formally, we append a set of trainable tokens ${ \bf X } _ { \Psi } \in \mathbb { R } ^ { | \Psi | \times h }$ where $\Psi$ denotes the set of indices for these tokens, to the document tokens X and pass the concatenated sequence into the bidirectional encoder $F _ { \mathrm { e n c } }$ , an $L$ -layer transformer:

$$
[ \mathbf { Z } _ { \mathbf { X } } ^ { ( L ) } , \mathbf { Z } _ { \Psi } ^ { ( L ) } ] = F _ { \mathrm { e n c } } ( [ \mathbf { X } , \mathbf { X } _ { \Psi } ] ; \theta )
$$

where $\theta$ represents the encoder parameters and ${ \mathbf Z } ^ { ( L ) }$ denotes the hidden states at the last layer.

We then leverage the attention mechanism to quantify the importance of each document token. Let $\mathsf { A t t n } _ { i } ^ { ( L , \eta ) } \in \mathbb { R } ^ { n }$ denote the attention weights from the universal query token $i \in \Psi$ to all document tokens at the last layer $L$ and head $\eta$ . To obtain a global measure of importance, we average over heads and across universal query tokens to compute the saliency scores $\pmb { \alpha } \in \mathbb { R } ^ { n }$ :

$$
\pmb { \alpha } = \frac { 1 } { | \Psi | } \sum _ { i \in \Psi } \sum _ { \eta = 1 } ^ { H } \mathrm { A t t n } _ { i } ^ { ( L , \eta ) } .
$$

The aim of $\pmb { \alpha }$ is to capture high-level semantic relevance, allowing the model to distinguish signal from noise before clustering begins. Using these saliency scores, we then select cluster centroids. We select the top- $m$ tokens with the highest saliency scores, where $m$ is the target budget. Let ${ \cal T } \subset \{ 1 , \ldots , n \}$ denote the indices corresponding to the $m$ largest values in $\pmb { \alpha }$ . We define the cluster centroids as $\mathcal { M } = \{ \pmb { \mu } _ { k } \} _ { k = 1 } ^ { m }$ , where each centroid corresponds to a selected token representation $\mu _ { k } = \mathrm { Z } _ { \mathrm { X } , j } ^ { ( L ) }$ for some $j \in \mathcal { I }$ .

# 5.2 Clustering

Next, we need to organize the rest of the tokens. We group every other token with the centroid it is most similar to, effectively gathering related context into coherent clusters. This ensures that even if a word isn’t selected as a centroid, its information is preserved by being associated with a relevant cluster.

Formally, we assign every document token to its nearest centroid based on cosine similarity:

$$
\mathcal G _ { k } = \{ j \in \{ 1 , \dots , n \} | k = \underset { k ^ { \prime } \in \{ 1 , \dots , m \} } { \mathrm { a r g m a x ~ } } \cos ( \mathbf Z _ { \mathbf { X } , j } ^ { ( L ) } , \pmb \mu _ { k ^ { \prime } } ) \} .
$$

Similar to H-Pool, this clustering step reduces redundancy by grouping semantically similar tokens. However, unlike H-Pool, which relies on iterative agglomerative merging, our approach anchors clusters around the globally salient centroids identified by the universal queries. This ensures that the compression is guided by semantic importance rather than just local geometric proximity. Furthermore, by employing hard assignment rather than fully soft operations (e.g., MemTok), we ensure that distinct semantic concepts remain separated in the latent space, alleviating the risk of over-smoothing.

# 5.3 Weighted Aggregation

With clusters established, we aggregate the tokens in each group into a compact representation. Naive averaging treats all inputs equally, ignoring the varying information density typical of multimodal data. Just as P-frames in video are compressed more heavily than I-frames [26], or text outweighs margins in a document, we must distinguish signal from redundancy. We therefore employ Weighted Aggregation, where the saliency score $\pmb { \alpha }$ naturally serves as a learnable importance weight to prioritize critical content.

Formally, we construct the document’s compressed representation $\mathbf { C } = [ \bar { \mathbf { c } } _ { 1 } , \hdots , \mathbf { c } _ { m } ] \in \mathbb { R } ^ { m \times h }$ . We compute each cluster vector $\mathbf { c } _ { k }$ as the weighted average of the document tokens assigned to it:

$$
\mathbf { c } _ { k } = \frac { \sum _ { j \in { \mathcal { G } } _ { k } } \alpha _ { j } \mathbf { Z } _ { \mathrm { X } , j } ^ { ( L ) } } { \sum _ { j \in { \mathcal { G } } _ { k } } \alpha _ { j } } .
$$

This also ensures that while the structure is discrete, the contribution of each token remains continuous, allowing gradients to flow back to the feature encoder and capture fine-grained semantic variations.

# 6 Experiments

# 6.1 Datasets

We evaluate the multi-vector compression methods on several datasets spanning text, image, and video.

• BEIR [text, 48] is a collection of text retrieval tasks. For our evaluation we chose the set of publicly available datasets with corpora of fewer than 1M documents. We further exclude Quora, as it is a duplicate retrieval task in which the "documents" are duplicate questions, and therefore not in need or amenable to compression. The remaining datasets span medical, financial, and argument domains. • ViDoRe v2 [visual document, 36] is a visual document retrieval benchmark designed to evaluate systems on visually rich PDFs where information is conveyed through both text and layout (e.g., figures, tables). It consists of four datasets spanning insurance, biomedical, economics, and ESG domains, featuring long-form and cross-document queries that require multimodal understanding.

<table><tr><td rowspan="2">Method</td><td colspan="2">BEIR</td><td colspan="2">VIDoRE</td><td colspan="2">MSR-VTT</td><td colspan="2">MULTIVENT2.0</td></tr><tr><td>R@10</td><td>nDCG@10</td><td>R@1</td><td>nDCG@5</td><td>R@1</td><td>nDCG@10</td><td>R@10</td><td>nDCG@10</td></tr><tr><td>Baseline</td><td>37.1</td><td>46.2</td><td>27.7</td><td>60.0</td><td>55.7</td><td>71.9</td><td>／*</td><td>/*</td></tr><tr><td rowspan="2">SEQRESIZE</td><td>35.8</td><td>43.9</td><td>23.5</td><td>51.7</td><td>53.3</td><td>69.9</td><td>41.1</td><td>38.5</td></tr><tr><td>96.5%</td><td>95.0%</td><td>84.8%</td><td>86.2%</td><td>95.7%</td><td>96.9%</td><td>N/A</td><td>N/A</td></tr><tr><td rowspan="2">MEMTOK</td><td>36.3</td><td>45.0</td><td>25.0</td><td>54.4</td><td>54.2</td><td>69.9</td><td>48.7</td><td>44.8</td></tr><tr><td>97.8%</td><td>97.4%</td><td>90.3%</td><td>90.7%</td><td>97.3%</td><td>96.9%</td><td>N/A</td><td>N/A</td></tr><tr><td rowspan="2">H-PoOL</td><td>35.5</td><td>41.2</td><td>26.0</td><td>56.4</td><td>54.1</td><td>70.4</td><td>49.2</td><td>46.5</td></tr><tr><td>95.7%</td><td>89.2%</td><td>93.9%</td><td>94.0%</td><td>97.1%</td><td>97.6%</td><td>N/A</td><td>N/A</td></tr><tr><td rowspan="2">AGC (Ours)</td><td>37.0</td><td>45.0</td><td>26.3</td><td>56.7</td><td>56.9</td><td>71.5</td><td>49.6</td><td>46.3</td></tr><tr><td>99.7%</td><td>97.4%</td><td>94.9%</td><td>94.5%</td><td>102.1%</td><td>99.2%</td><td>N/A</td><td>N/A</td></tr></table>

Table 1: Results of index compression on each retrieval benchmark. Compression budgets: BEIR 32, ViDoRe 64, MSR-VTT 32, MultiVENT $2 . 0 6 4 . *$ means baseline was unable to build due to compute. Second row of each method shows percent of baseline.

• MSR-VTT [vision-only video, 54] is a video captioning dataset converted to text-to-video retrieval, where each query is a sentence description of a video. There is only one relevant video per query and 1000 query-video pairs in test with no additional irrelevant videos.

• MultiVENT 2.0 [audiovisual video, 23] is a text-to-video retrieval dataset with queries that target visual and audio information. There are ten relevant videos per query with 2546 queries and 109,800 videos in test.

# 6.2 Experimental Setup

BEIR. We begin finetuning from an already finetuned singlevector encoder [29, 57].3 We train for 10,000 steps with a distillation loss on 16-way MSMARCO [3] hard negatives scored by a reranker.4 We train with a batch size of 20, learning rate of $1 0 ^ { - 4 }$ , and bfloat16 precision. In both the training and evaluation settings, we use a maximum query length of 32 (with the usual ColBERT-style query augmentation mechanism), maximum document length of 300, and no query/document marker tokens. For retrieval, document embeddings are indexed and retrieved in a FastPlaid [43, 46] index with 4-bit residuals.

ViDoRe v2. We enable bidirectional attention and initialize the pretrained weights from Qwen2.5-VL-3B. We train on the ColPali train set5 for 2 epochs with a global batch size of 112 and gradient accumulation step of 4, learning rate of $1 0 ^ { - 5 }$ , and bfloat16 precision. We prepend "Passage: " and "Query: " for document and query respectively. Due to the combination of the large embedding dimension (2048) and quantity of embeddings $( > 1 0 0 0$ per document, uncompressed), we cannot fit our full uncompressed data in a FastPlaid index, and therefore resort to a brute-force search over a flat index. For the compressed methods, we are able to use a Fast-Plaid index, but we continue to use the flat index for compression methods to fairly compare with the baselines.

MSR-VTT. We again enable bidirectional attention and initialize the pretrained weights from Qwen2.5-VL-3B, Qwen2.5-VL-7B, and Qwen3-VL-4B for different variants. We use a fixed number of frames of 24. We train on the MSR-VTT Train 9k split for 2 epochs with a global batch size of 28 and gradient accumulation step of 4, learning rate of $1 0 ^ { - 5 }$ , and bfloat16 precision. We build a flat index of each method to gather the matching positions and strengths for analysis in subsection 6.5.

MultiVENT 2.0. We enable bidirectional attention and initialize the pretrained weights from Qwen2.5-Omni-3B. We train on a combination of the human written queries and synthetically generated queries [45]. We sample frames at most 24 frames and audio at 4KHz. We train for 2 epochs with a global batch size of 8 and gradient accumulation step of 4, learning rate of $1 0 ^ { - 5 }$ , and bfloat16 precision. We cannot build any index over the uncompressed representations, and only utilize FastPlaid for the compressed representations.

Evaluation. For each dataset, we report appropriate recall at k $( \mathrm { R } @ \mathrm { k } )$ and normalized discounted cumulative gain at k $( \mathrm { n D C G } @ \mathrm { k } )$ . We also report the percentage of base performance for each compression method calculated as ( compression scorebase score ) and report the compression ratio as $\textstyle 1 - { \bigl ( } { \frac { \mathrm { b u d g e t } } { \mathrm { a v g ~ t o k s ~ p e r ~ d o c } } } { \bigr ) }$

# 6.3 Results

In Table 1, we summarize the retrieval performance of the index compression methods in retrieval settings across each modality. Our main finding is that AGC performs the best across the modalities compared to the other compression techniques, maintaining $9 7 \%$ of the uncompressed model performance at $\scriptstyle \mathrm { { n D C G @ 1 0 } }$ . We also find that H-Pool performs well for non-parametric compression, often outperforming the other learned methods SeqResize and MemTok on non-text benchmarks.

Across the datasets, we find that the only method able to outperform the base model, which builds a full index with a one-to-one mapping between document tokens and vectors, is AGC $\operatorname { R @ 1 }$ on MSR-VTT). This highlights two main takeaways. (1) Training multimodal retrieval methods with a compression objective can be beneficial. Multimodal (audio and visual) tokens do not always provide new information to the document representations and are often semantically redundant6, meaning that information density does not scale linearly with document length, an assumption reasonably held in text. (2) The base model underutilizes its full document representation. In subsection 6.5, we show that the base model only utilizes about $\sim 1 \%$ of the representations. Practical results imply that full indices for multimodal collections offer diminishing returns relative to their cost.

<table><tr><td>Meth.</td><td>Avg</td><td>NF FQA</td><td>SciF</td><td>SciD</td><td>TC</td><td>Tou</td><td>Arg</td></tr><tr><td>Base</td><td>46.2</td><td>37.1</td><td>43.3 74.4</td><td>18.6</td><td>78.5</td><td>27.5</td><td>44.1</td></tr><tr><td>SEQ</td><td>43.9 95.0</td><td>31.0 83.6</td><td>36.8 70.3 85.0 94.5</td><td>18.7 100.5</td><td>75.8 96.6</td><td>24.8 90.2</td><td>50.0 113.4</td></tr><tr><td>MEM</td><td>45.0 97.4</td><td>35.2 94.9</td><td>39.8 71.7 91.9 96.4</td><td>17.6 94.6</td><td>77.6 98.9</td><td>27.7 100.7</td><td>45.1 102.3</td></tr><tr><td>H-P</td><td>41.2 89.1</td><td>33.7 90.8</td><td>36.2 72.5 83.6 97.4</td><td>18.4 98.9</td><td>62.0 79.0</td><td>17.6 64.0</td><td>48.3 109.5</td></tr><tr><td>AGC</td><td>45.0 97.4</td><td>36.0 97.0</td><td>38.9 72.4 89.8 97.3</td><td>18.1 97.3</td><td>78.7 100.3</td><td>23.3 84.7</td><td>47.6 107.9</td></tr></table>

Table 2: nDCG $\bf { \Pi } ( \ @ { ' } \mathbf { \Pi } ^ { \omega 1 0 }$ (top) and percent performance relative to uncompressed baseline (bottom) on BEIR datasets. NF: NFCorpus, FQA: FiQA, SciF: SciFact, SciD: SciDocs, TC: TREC-COVID, Tou: Touche, Arg: arguana.

BEIR. In Table 2, we explore the methods’ performance for text retrieval on a subset of BEIR datasets, reporting both the absolute $\mathrm { \ n D C G } @ 1 0$ and relative performance compared to the uncompressed ColBERT baseline. With a budget of 32 tokens, documents in BEIR are compressed by around $8 0 \%$ .7 We find that there is not much difference between MemTok and AGC. Both methods compress the text document representations well and maintain stable performance across the different datasets. Additionally, we find that there is a larger gap between H-Pool and the learned methods, as the performance varies more significantly depending on the task.

ViDoRe. In Table 3, we break down the performance of each method on the ViDoRe topic splits. Comparing the runs with the same training configurations, we see that AGC and H-Pool significantly outperform SeqResize and MemTok. H-Pool and AGC appear to be relatively equivalent, with their averages only differing by 0.002 in nDCG@5. However, when looking at the breakdown by topic, we see that AGC is more stable across domains than H-Pool. We also compare to another learned compression method, MetaEmbed [53]. We find that AGC and H-Pool have comparable or better performance to MetaEmbed. This highlights the strengths of both AGC and H-Pool, even when training at smaller scales.8

MSR-VTT. In Table 4, we report more detailed comparisons of our baseline and method on MSR-VTT. We find that every compression method sets a new state-of-the-art on MSR-VTT over previous multi-vector approaches (ColQwen-Omni, Video-ColBERT) and dense approaches (OmniEmbed), even when compressing the index to only 5 vectors per document. We even see at budgets of 32 and 128, AGC performs better at $\mathrm { R @ 1 }$ than the base model, again showing that training for compression in multimodal retrieval may not only lead to efficient indices, but also best performance.

<table><tr><td>Method</td><td>Tok</td><td>Avg</td><td>Bio</td><td>Econ</td><td>ESG-R</td><td>ESG-H</td></tr><tr><td>Base</td><td>1297</td><td>60.0</td><td>61.4</td><td>53.9</td><td>57.0</td><td>67.6</td></tr><tr><td>ColPali</td><td>-</td><td>53.3</td><td>56.5</td><td>49.9</td><td>55.7</td><td>51.1</td></tr><tr><td>CQO</td><td>1</td><td>56.5</td><td>56.5</td><td>53.2</td><td>54.2</td><td>62.2</td></tr><tr><td>MEMBED</td><td>64</td><td>58.8</td><td>58.7</td><td>55.5</td><td>57.4</td><td>63.7</td></tr><tr><td>SEQRESIZE</td><td>64</td><td>51.7</td><td>54.7</td><td>53.5</td><td>45.2</td><td>53.5</td></tr><tr><td>MEMTOK</td><td>64</td><td>54.3</td><td>56.8</td><td>53.0</td><td>46.4</td><td>61.4</td></tr><tr><td>H-POOL</td><td>64</td><td>56.4</td><td>59.6</td><td>52.1</td><td>53.4</td><td>60.6</td></tr><tr><td>AGC</td><td>64</td><td>56.7</td><td>59.0</td><td>54.5</td><td>55.8</td><td>57.3</td></tr></table>

Table 3: nDCG@5 breakdown by domain on ViDoRe v2 for the multilingual subsets. Bio: Biomedical, Econ: Economics, ESG-R: ESG Reports, ESG-H: ESG Human. ESGs are English. CQO: ColQwenOmni, MEmbed: MetaEmbed.

<table><tr><td>Tok</td><td>Method</td><td>R@1</td><td>R@10</td><td>nDCG@10</td></tr><tr><td>1</td><td>OmniEmbed-7B [33]</td><td>51.5</td><td>83.2</td><td>67.1</td></tr><tr><td>26</td><td>Video-ColBERT[42]</td><td>51.5</td><td>85.5</td><td>67.7</td></tr><tr><td>1702</td><td>ColQwen-Omni 3B</td><td>40.8</td><td>73.8</td><td>56.3</td></tr><tr><td>1318</td><td>Baseline 3B (Ours)</td><td>55.7</td><td>88.3</td><td>71.9</td></tr><tr><td rowspan="4">5</td><td>SEQRESIZE MEMTOK</td><td>53.4</td><td>86.7</td><td>69.5</td></tr><tr><td></td><td>52.7</td><td>87.3</td><td>69.3</td></tr><tr><td>H-PoOL</td><td>52.6</td><td>86.1</td><td>68.9</td></tr><tr><td>AGC</td><td>53.9</td><td>85.8</td><td>69.2</td></tr><tr><td rowspan="4">32</td><td>SEQRESIZE</td><td>53.3</td><td>86.9</td><td>69.9</td></tr><tr><td>MEMToK</td><td>54.2</td><td>86.4</td><td>69.9</td></tr><tr><td>H-PoOL</td><td>54.1</td><td>87.3</td><td>70.4</td></tr><tr><td>AGC</td><td>56.9</td><td>87.0</td><td>71.5</td></tr><tr><td rowspan="4">128</td><td>SEQRESIZE</td><td>52.6</td><td>87.6</td><td>69.7</td></tr><tr><td>MEMTOK</td><td></td><td></td><td></td></tr><tr><td>H-PoOL</td><td>54.9</td><td>86.7</td><td>70.5</td></tr><tr><td>AGC</td><td>54.4 56.4</td><td>87.2 87.6</td><td>70.9 71.6</td></tr></table>

Table 4: Retrieval performance on MSR-VTT. We compare compressed methods against uncompressed baselines at budgets of 5, 32, and 128 tokens, alongside SOTA models.

MultiVENT 2.0. In Table 1, we report the retrieval results for MultiVENT 2.0 on the compression methods. We are not able to build the index for the full model as comparable vision indices use a hidden dimension of 2048 [38, 42, 53, 55], demonstrating that for large multimodal indices compression is necessary. Unlike ViDoRe and MSR-VTT, evaluating on MultiVENT 2.0 requires leveraging the audio information to retrieve the relevant videos. We found inefficient audio sampling to be a major limitation of the Qwen-Omni model, suggesting interesting future work on how to pass audiovisual signal efficiently to MLLMs. For example, when training on the MultiVENT 2.0 data, we found that in order to fit a batch size of 8, the audio signal needed to be reduced from 16KHz (Qwen-Omni’s training rate) to 4KHz to fit on the devices.9

Table 5: Retrieval performance metrics of AGC across varying budgets and appending tokens on MSR-VTT. Each combination is used with the same configurations of training. Appn Tok: number of appending tokens.   

<table><tr><td>Tok</td><td>Appn Tok</td><td>R@1</td><td>R@10</td><td>nDCG@10</td></tr><tr><td rowspan="2">5</td><td>5</td><td>53.9</td><td>85.8</td><td>69.2</td></tr><tr><td>32</td><td>53.1</td><td>86.7</td><td>69.6</td></tr><tr><td rowspan="3">32</td><td>5</td><td>54.2</td><td>87.2</td><td>70.6</td></tr><tr><td>32</td><td>56.9</td><td>87.0</td><td>71.5</td></tr><tr><td>128</td><td>55.4</td><td>87.6</td><td>71.2</td></tr><tr><td rowspan="2">128</td><td>32</td><td>55.2</td><td>86.8</td><td>70.8</td></tr><tr><td>128</td><td>56.4</td><td>87.6</td><td>71.6</td></tr></table>

<table><tr><td>Method</td><td>Train</td><td>Test</td><td>R@1</td><td>R@10</td><td>nDCG@10</td></tr><tr><td>Baseline</td><td>1318</td><td>1318</td><td>55.7</td><td>88.3</td><td>71.9</td></tr><tr><td rowspan="3">AGC</td><td>32</td><td>5</td><td>53.6</td><td>87.4</td><td>70.1</td></tr><tr><td>32</td><td>32</td><td>56.9</td><td>87.0</td><td>71.5</td></tr><tr><td>32</td><td>128</td><td>56.4</td><td>87.5</td><td>71.7</td></tr><tr><td rowspan="3">H-POOL</td><td>1318</td><td>5</td><td>52.6</td><td>86.1</td><td>68.9</td></tr><tr><td>1318</td><td>32</td><td>54.1</td><td>87.3</td><td>70.4</td></tr><tr><td>1318</td><td>128</td><td>54.4</td><td>87.2</td><td>70.9</td></tr></table>

Table 6: Generalizability of AGC and H-Pool compression methods on MSR-VTT.

# 6.4 Compression Ranges and Stability

Compression Ranges. In Table 4, we explore learning different ranges of compression on MSR-VTT for 5 tokens $( 9 9 . 6 2 \% )$ , 32 tokens $( 9 7 . 5 7 \% )$ and 128 tokens $( 9 0 . 2 9 \% )$ . For SeqResize, we see that performance seems to be relatively flat across the compression ratios, with similar $\mathrm { R @ 1 }$ and $\mathrm { \ n D C G } @ 1 0$ results. This is an interesting finding, largely suggesting that SeqResize may underutilize the budget and lead to a suboptimal index (see further evidence in subsection 6.5). For all other methods, we see an increase in performance from the most extreme compression to lighter ratios. Additionally, we find H-Pool’s performance impressive as a non-parametric technique at a budget of 5. However, AGC continues to have strongest performance at each ratio, demonstrating its robustness to a variety of compression ratios.

Budget Sweeps. In Table 5, we analyze the impact of varying token budgets and the number of appending tokens of AGC on retrieval performance on MSR-VTT. We observe that performance scales positively with both the size of the token budget and the quantity of appending tokens. Notably, even under the most extreme compression setting (a budget of 5), AGC maintains robust performance, outperforming the single dense vector encoder, OmniEmbed-7B, despite using a smaller 3B backbone. When looking at the number of appended query tokens and the budget, we find that it is generally optimal to align the number of appended query tokens and the budget size. Additionally, we see that 32 appended query tokens and a budget of 5 outperforms 5 appended query tokens at the same budget, but we don’t see the same pattern for 128 appended query tokens and 32 budget. This suggests that it is important to avoid a low number of query tokens, but that performance doesn’t necessarily scale to the number of appended query tokens at any budget.

<table><tr><td>Model Variant</td><td>R@1</td><td>R@10</td><td>nDCG@10</td></tr><tr><td>Qwen2.5-VL-3B</td><td>56.9</td><td>87.0</td><td>71.5</td></tr><tr><td>Qwen2.5-VL-7B</td><td>58.0</td><td>89.0</td><td>73.0</td></tr><tr><td>Qwen3-VL-4B</td><td>58.5</td><td>88.4</td><td>73.0</td></tr></table>

Table 7: Generalizability of AGC across different model sizes and variants. Results indicate consistent performance scaling with larger and newer model backbones.

Compression Transferability. In Table 6, we again explore different ranges of compression on MSR-VTT, but only train the model for a single compression ratio (i.e., training AGC for 32). We find that AGC provides the best ability to generalize to an unseen compression ratio after training. We attribute this finding to a weakness in the heuristic of H-Pool, redundant visual tokens should not be equally merged. By leveraging attention to select centroids and weight the merging, AGC is better able to preserve salient semantic concepts and reduce the redundancy along the temporal dimension. We do not compare to MemTok and SeqResize in this experiment as they do not generalize to new compression ratios.

Looking closer at the AGC results, we observe very close performance between methods trained for budgets of 5 and 128 and the method trained for 32 and tested on those budgets. This demonstrates that our model can transfer abilities between budgets at performance near training AGC for that compression ratio.

Model Size and Backbone. In Table 7, we also provide an experiment on the stability of our method with a different model size and with a different underlying model. When scaling AGC to a 7B parameter model, initializing from Qwen2.5-VL-7B, we find that performance is significantly improved. Additionally, we initialize our model from Qwen3-VL-4B and find that performance again improves over both Qwen2.5 versions. These results demonstrate that our method benefits from models with stronger representations and stronger encoding abilities, and should scale to any backbone or model size.

# 6.5 Index Utilization

In Figure 3, we visualize the difference in index utilization for full uncompressed indices and the four compression techniques and the similarity between document vectors in each index on MSR-VTT. To calculate the matching strength, we sum the maximum similarity scores across all relevant query-document pairs in MSR-VTT and normalize by the total number of match records at that query position. For the heatmaps, we calculate the cosine similarity between token vectors within each document, averaged across documents in the index of MSR-VTT.

Token Utilization Analysis. Our analysis of index statistics reveals that of the 1.3 million unique document tokens, only $\sim 1 \%$ are active during a single evaluation pass, with the base model primarily utilizing the first $2 \%$ . This again stresses that building full indices for multimodal collections is unnecessary and that these indices can greatly benefit from compression. For SeqResize, we see a unique trend amongst the compression methods, only selecting a few tokens from the document representation in late interaction. This underutilization of the budget corroborates the findings of Table 4, where SeqResize’s performance seemed to plateau across the compression ratios. For MemTok and AGC, we see that both methods attempt to utilize their full document representations. Because MemTok’s representations are appended to the document context, we see a significant bias towards the first few tokens in the representation. This result largely follows the trend in dense encoding with causal language models to append a token at the end of the sequence to use as the document representation [31, 34]. Unlike MemTok, AGC and H-Pool use representations from the document leading to a better utilization of the compressed representations in late interaction.

![](images/58e38e7bed0646c47b33a44292c30548c60222db1926f715b2a157547c943ce1.jpg)  
Figure 3: Index utilization and inter-position similarity analysis on MSR-VTT. Top row: Per-position matching strength for each method, computed by summing the maximum similarity matches between query tokens and document tokens across all relevant query-document pairs, averaged over query positions. Bottom row: Pairwise cosine similarity between document vectors within each document, averaged across all documents in the index.

![](images/796a74aa607b0f57a1c1e3f7a55d09081e76c1921ca0d0815883679c3e1b0ea4.jpg)  
Figure 4: Correlation between retrieval performance metrics and distribution evenness measures on MSR-VTT dataset. Dashed lines indicate linear regression fits. All correlations are statistically significant $( p \leq 0 . 0 1 )$ , with Pearson’s r ranging from 0.959 to 0.996.

Token Similarity Analysis. In the heatmaps of Figure 3, we further investigate the internal structure of these indices by visualizing the cosine similarity between document tokens. For the full model, we observe that the first few tokens, which we previously noted dominate late interaction, exhibit a consistently high similarity $( \sim 0 . 7 )$ to nearly all other tokens in the document. Additionally, these initial tokens are highly similar to one another. This similarity explains the significant imbalance in matching strength observed in the corresponding bar chart. SeqResize presents a distinct pattern where tokens that are never used in interaction display negative similarity. We interpret this as a modeling failure; tokens derived from the same document context should theoretically maintain a baseline degree of positive similarity, which SeqResize fails to capture. Conversely, MemTok demonstrates an over-smoothing problem, where the heatmap is dominated by high similarity scores. This lack of diversity restricts the expressive power of the index, as the tokens fail to capture distinct semantic nuances. H-Pool, by design, merges similar tokens and consequently produces the most diverse set of representations, as evidenced by the lower offdiagonal similarities. However, this suggests that similarity-based heuristics alone are not sufficient for optimal performance, as H-Pool does not perform as well as learned methods in many settings despite its high diversity. Finally, AGC shows a trend similar to H-Pool but maintains decent inter-token similarities. This balance highlights the efficacy of our approach: it avoids the representation collapse seen in MemTok while preserving necessary semantic overlaps that H-Pool lacks, resulting in a robust compressed index.

Table 8: Pearson correlation analysis between retrieval metrics and inverse evenness metrics (1/evenness) on MSR-VTT, testing the hyperbolic relationship retrieval $\sim$ 1/evenness. All variants use a fixed budget of 32. Evenness metrics measured on matching strength of (document position, query position) pairs. CV: Coefficient of Variation, Gini: Gini Coefficient. Significance levels: $^ { * * } p < 0 . 0 1$ , ${ } ^ { * } p < 0 . 0 5$ .   

<table><tr><td></td><td colspan="2">CV</td><td colspan="2">Gini</td></tr><tr><td>Metric</td><td>Pearson r</td><td> p-value</td><td>Pearson r</td><td>p-value</td></tr><tr><td>R@1</td><td>0.974**</td><td>0.005</td><td>0.959**</td><td>0.010</td></tr><tr><td>nDCG@10</td><td>0.978**</td><td>0.004</td><td>0.943*</td><td>0.016</td></tr><tr><td>MRR</td><td>0.996**</td><td>&lt;0.001</td><td>0.962**</td><td>0.009</td></tr></table>

Predicting Performance with Utilization. Following from the above observations, we explore if it is possible to predict compression performance by only looking at how evenly distributed the strength of maximum similarity matches are in a document representation. We calculate the Coefficient of Variation (CV) and the Gini coefficient. The CV assesses relative variability standardized by the mean $\begin{array} { r } { C V = \frac { \sigma } { \mu } \times 1 0 0 } \end{array}$ , while the Gini coefficient quantifies distributional concentration $\begin{array} { r } { G = \frac { 2 \sum i x _ { i } } { n \sum x _ { i } } - \frac { n + 1 } { n } } \end{array}$ . For both metrics, lower values indicate a more evenly distributed activations.

In Table 8 and Figure 4, we show the Pearson’s correlation [40] between the retrieval metrics $( \mathbb { R } @ 1 , \mathbb { n } \mathrm { D C G } @ 1 0 .$ , and MRR) and the inverse evenness metrics (Coefficient of Variation (CV) and Gini Coefficient).10 We find a rough correlation between the evenness of the distribution of maximum similarity matches and retrieval performance. These results suggest that training late interaction methods to maximize the utility of each token in its document representations will lead to strong performance, which we leave for future work to explore. Additionally, this finding suggests that during development it is satisfactory to estimate the downstream performance of a compression method with the distribution of maximum similarity matches on a small set of queries. This is especially beneficial for multimodal tasks, where building indices is storage and time expensive.

# 6.6 Method Ablation

In Table 9, we perform an ablation analysis on the modeling choices in AGC using MSR-VTT. We examine the contribution of three components: Attention-based Centroid Selection, Clustering, and Weighted Aggregation. First, removing the attention weights (w/o Attn Weight) from the aggregation step leads to a decline in performance. As discussed in subsection 5.3, weighting the contribution of tokens by their saliency scores is beneficial for balancing the hard assignment operation with optimization stability. Without these weights, the contribution of individual tokens becomes less continuous, rendering the optimization landscape rougher and less effective. Second, we analyze the impact of attention-based selection $( \mathrm { w } / \mathrm { o }$ Attn Select) by replacing the learned universal query tokens with a random selection strategy. This prevents the model from distinguishing signal from noise. While this randomness helps maintain some diversity, it forces the model to compress mixed information into each vector. This saturates the capacity of each vector and weakens its ability to capture complex, discriminative semantics. Finally, we evaluate the model without clustering (w/o Cluster), relying solely on attention selection. Without the clustering operation to reduce redundancy and aggregate context, the resulting representations lack diversity and expressiveness. Consequently, token matches in one evaluation pass become highly concentrated on a narrow set of tokens, leading to worse performance. As shown in Figure 4, the progressive removal of Weighted Aggregation and Clustering leads to a simultaneous decline in retrieval performance and evenness of MaxSim matches. This confirms that AGC effectively incentivizes balanced utilization while improving retrieval.

<table><tr><td>Method</td><td>R@1</td><td>R@10</td><td>nDCG@10</td></tr><tr><td>AGC</td><td>56.9</td><td>87.0</td><td>71.5</td></tr><tr><td>w/o Attn Weight</td><td>55.7</td><td>86.5</td><td>71.0</td></tr><tr><td>w/o Attn Select</td><td>54.1</td><td>86.8</td><td>70.0</td></tr><tr><td>w/o Cluster</td><td>52.9</td><td>87.3</td><td>69.8</td></tr></table>

Table 9: Ablation Study Results on MSR-VTT. We observe a drop in performance as components are removed (individually, not in sequence) from the full AGC model.

# 7 Conclusion

In this work, we explore multi-vector index compression in any modality. We adapt three strong text-based compression methods to multimodal documents: SeqResize (projection-based), MemTok (token-based), and H-Pool (heuristic-based), and introduce AGC, a novel approach to multi-vector compression. AGC uses three main aspects—attention-based centroid selection, clustering, and weighted aggregation—to maximize the utility of a fixed document token budget. We find that AGC is a strong and robust method to compress documents in any modality under a variety of compression ratios, domains, and model specifications. AGC consistently outperforms the other compression methods across the modalities and even sets a new state-of-the-art on MSR-VTT. Finally, we visualize how each method utilizes its index, demonstrating that AGC and H-Pool properly utilize their budgets, and show that downstream retrieval performance roughly correlates to how well utilized a document representation is in late interaction.

Future Work. While our work corroborates the understanding that multi-vector embeddings can be compressed with a high compression ratio while retaining most retrieval performance, the budget is applied statically or potentially linearly in the case of H-Pool. A natural extension would be to develop compression mechanisms that allocate the budget in proportion to a document’s inherent informational content, perhaps by using lightweight features like our document token utilization to calibrate the level of compression.

# Acknowledgments

This material is based upon work supported by the NSF Graduate Research Fellowship under Grant No. DGE2139757. Any opinion, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
