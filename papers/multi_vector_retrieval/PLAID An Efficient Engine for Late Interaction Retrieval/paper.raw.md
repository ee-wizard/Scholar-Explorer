# PLAID: An Efficient Engine for Late Interaction Retrieval

Keshav Santhanam∗ keshav2@stanford.edu Stanford University United States

Omar Khattab∗   
okhattab@stanford.edu   
Stanford University   
United States

Christopher Potts Stanford University United States

Matei Zaharia Stanford University United States

# ABSTRACT

Pre-trained language models are increasingly important components across multiple information retrieval (IR) paradigms. Late interaction, introduced with the ColBERT model and recently refined in ColBERTv2, is a popular paradigm that holds state-of-the-art status across many benchmarks. To dramatically speed up the search latency of late interaction, we introduce the Performance-optimized Late Interaction Driver (PLAID). Without impacting quality, PLAID swiftly eliminates low-scoring passages using a novel centroid interaction mechanism that treats every passage as a lightweight bag of centroids. PLAID uses centroid interaction as well as centroid pruning, a mechanism for sparsifying the bag of centroids, within a highly-optimized engine to reduce late interaction search latency by up to $7 \times$ on a GPU and $4 5 \times$ on a CPU against vanilla ColBERTv2, while continuing to deliver state-of-the-art retrieval quality. This allows the PLAID engine with ColBERTv2 to achieve latency of tens of milliseconds on a GPU and tens or just few hundreds of milliseconds on a CPU at large scale, even at the largest scales we evaluate with 140M passages.

# 1 INTRODUCTION

Recent advances in neural information retrieval (IR) have led to notable gains on retrieval benchmarks and retrieval-based NLP tasks. Late interaction, introduced in ColBERT [22], is a paradigm that delivers state-of-the-art quality in many of these settings, including passage ranking [14, 42, 48], open-domain question answering [21, 24], conversational tasks [35, 38], and beyond [20, 54]. ColBERT and its variants encode queries and documents into tokenlevel vectors and conduct scoring via scalable yet fine-grained interactions at the level of tokens (Figure 1), alleviating the dotproduct bottleneck of single-vector representations. The recent ColBERTv2 [42] model demonstrates that late interaction models often considerably outperform recent single-vector and sparse representations within and outside the training domain, a finding echoed in several recent studies [26, 29, 43, 44, 51, 53].

Despite its strong retrieval quality, late interaction requires special infrastructure [22, 25] for low-latency retrieval as it encodes each query and each document as a full matrix. Most IR models represent documents as a single vector, either sparse (e.g., BM25 [41]; SPLADE [11]) or dense (e.g., DPR [18]; ANCE [49]), and thus mature sparse retrieval strategies like WAND [5] or dense kNN methods like HNSW [31] cannot be applied directly or optimally to late interaction. While recent work [28, 42, 45] has explored optimizing individual components of ColBERT’s pipeline, an end-to-end optimized engine has never been studied to our knowledge.

We study how to optimize late-interaction search latency at a large scale, taking all steps of retrieval into account. We build on the state-of-the-art ColBERTv2 model. Besides improving quality with denoised supervision, ColBERTv2 aggressively compresses the storage footprint of late interaction. It reduces the index size by up to an order of magnitude using residual representations (§3.1). In those, each vector in a passage is encoded using the ID of its nearest centroid that approximates its token semantics—among tens or hundreds of thousands of centroids obtained through $k$ -means clustering—and a quantized residual vector.

We introduce the Performance-optimized Late Interaction Driver (PLAID),1 an efficient retrieval engine that reduces late interaction search latency by $2 . 5 \mathrm { - } 7 \times$ on GPU and $9 - 4 5 \times$ on CPU against vanilla ColBERTv2 while retaining high quality. This allows the PLAID implementation of ColBERTv2, PLAID ColBERTv2, to achieve CPUonly latency of tens or just few hundreds of milliseconds and GPU latency of few tens of milliseconds at very large scale, even on 140M passages. Crucially, PLAID ColBERTv2 does so while continuing to deliver state-of-the-art retrieval quality.

To dramatically speed up search, PLAID leverages the centroid component of the ColBERTv2 representations, which is a compact integer ID per token. Instead of exhaustively scoring all passages found with nearest-neighbor search, PLAID uses the centroids to identify high-scoring passages and eliminate weaker candidates without loading their larger residuals. We conduct this in a multistage pipeline and introduce centroid interaction, a scoring mechanism that treats every passage as a lightweight bag of centroid $\mathit { I D s }$ . We show that this centroid-only multi-vector search exhibits high recall without using the vector residuals (§3.3), allowing us to reserve full scoring to a very small number of candidate passages. Because the centroids come from a fixed set (i.e., constitute a discrete vocabulary), the distance between the query vectors and all centroids can be computed once during search and re-used across all bag-of-centroids passage representations. This allows us to further leverage the centroid scores for centroid pruning, which sparsifies the bag of centroid representations in the earlier stages of retrieval by skipping centroid IDs that are distant from all query vectors.

In the PLAID engine, we implement centroid interaction and centroid pruning and implement optimized yet modular kernels for the data movement, decompression, and scoring components of late interaction with the residual representations of ColBERTv2 (§4.5). We extensively evaluate the quality and efficiency of PLAID within and outside the training domain (on MS MARCO v1 [36] and v2 [6], Wikipedia, and LoTTE [42]) and across a wide range of corpus sizes (2M–140M passages), search depths $( k { = } 1 0 , 1 0 0 , 1 0 0 0 )$ ), and hardware settings with single- and multi-threaded CPU and with a GPU (§5.2). We also conduct a detailed ablation study to understand the empirical sources of gains among centroid interaction, centroid pruning, and our faster kernels (§5.3).

![](images/a601c18466ba516d58e4479cb6a97b22c1cca3a51ed1514cd0b66da23f75160a.jpg)  
Figure 1: The late interaction architecture, given a query and a passage. Diagram from Khattab et al. [21] with permission.

In summary, we make the following contributions:

(1) We analyze centroid-only retrieval with ColBERTv2, showing that a pruned bag-of-centroids representation can support high-recall candidate generation (§3).   
(2) We propose PLAID, a retrieval engine that introduces centroid interaction and centroid pruning as well as optimized implementations of these techniques for dramatically improving the latency of late-interaction search (§4).   
(3) We extensively evaluate PLAID and conduct a large-scale evaluation up to 140M passages, the largest to our knowledge with late-interaction retrievers (§5).

# 2 RELATED WORK

# 2.1 Neural IR

The IR community has introduced many neural IR models based on pre-trained Transformers. Whereas early models were primarily cross-encoders [27, 37] that attend jointly to queries and passages, many subsequent models target higher efficiency by producing independent representations for queries and passages. Some of those produce sparse term weights [7, 32], whereas others encode each passage or query into a single vector [18, 39, 49] or multi-vector representation (the class we study; [12, 15, 21, 22, 42]). These choices make different tradeoffs about efficiency and quality: whereas sparse term weights and single-vector models can be particularly lightweight in some settings, multi-vector late interaction [22] can often result in considerably stronger quality and robustness. Orthogonal to the choice of modeling query–document interactions, researchers have improved the supervision for neural models with harder negatives [21, 49, 52] as well as distillation and denoising [13, 39, 40], among other approaches. Our work extends

![](images/8fbe340c486b635bb62389721619fce76f84077ebfaa1533555c5df021d7a990.jpg)  
Figure 2: Latency breakdown of MS MARCO v1 dev queries run with vanilla ColBERTv2 and PLAID ColBERTv2 on a TI-TAN V GPU. Vanilla ColBERTv2 is overwhelmingly bottlenecked with the cost of index lookup and decompression, a challenge that PLAID addresses.

ColBERTv2 [42], which combines late interaction modeling with hard negative and denoising supervision to achieve state-of-the-art quality among standalone retrievers.

# 2.2 Pruning for Sparse and Dense Retrieval

For sparse retrieval models, traditional IR has a wealth of work on fast strategies for skipping documents for top- $k$ search. Strategies often keep metadata like term score upper bounds to skip lowerscoring candidates and most follow a Document-At-A-Time (DAAT) scoring approach [5, 8, 9, 19, 33, 47]. Refer to Tonellotto et al. [46] for a detailed treatment of recent methods. A key difference to our settings is that these all strategies expect a set of precomputed scores (particularly, useful upper bounds on every term–document pair), whereas with late interaction the term–document interaction (i.e., the MaxSim score) is only known at query time after a matrix-vector multiplication. Our observations about the utility of centroids for accelerating late interaction successfully moves the problem closer to classical IR, but poses the challenge that the query-to-centroid scores are only known at query time.

For dense retrieval models that use single-vector representations, approximate $k$ -nearest neighbor (ANN) search is a well-studied problem [1, 16, 17, 31]. Our focus extends such work from a single vector to the late interaction of two matrices.

# 3 ANALYSIS OF COLBERTV2 RETRIEVAL

We begin by a preliminary investigation of the latency (§3.2) and scoring patterns (§3.3) of ColBERTv2 retrieval that motivates our work on PLAID. To make this section self-contained, $\ S 3 . 1$ reviews the modeling, storage, and supervision of ColBERTv2.

![](images/54835cb7db7aa17c9ba7695083365e086736453f239d499760963cbf194ac75e.jpg)  
Figure 3: Recall of passages retrieved by a centroid-only version of ColBERTv2 with respect to the top $k$ passages retrieved by vanilla ColBERTv2. Centroids alone can identify virtually all of the top- $\mathbf { \nabla } \cdot k$ passages retrieved with the full ColBERTv2 pipeline, within $1 0 \cdot k$ or fewer candidates, motivating our centroid interaction strategy.

# 3.1 Modeling, Storage, and Retrieval

PLAID optimizes retrieval for models using the late interaction architecture of ColBERT, which includes systems like ColBERTv2, Baleen [20], Hindsight [38], and DrDecr [24], among others. As depicted in Figure 1, a Transformer encodes queries and passages independently into vectors at the token level. For scalability, passage representations are pre-computed offline. At search time, the similarity between a query $q$ and a passage $d$ is computed as the summation of “MaxSim” operations, namely, the largest cosine similarity between each vector in the query matrix and all of the passage vectors:

$$
S _ { q , d } = \sum _ { i = 1 } ^ { | Q | } \underset { j = 1 } { \overset { | D | } { \operatorname* { m a x } } } Q _ { i } \cdot D _ { j } ^ { T }
$$

where $Q$ and $D$ are the matrix representations of the query and passage, respectively. In doing so, this scoring function aligns each query token with the “most similar” passage token and estimates relevance as the sum of these term-level scores. Refer to Khattab and Zaharia [22] for a more complete discussion of late interaction.

For storing the passage representations, we adopt the ColBERTv2 residual compression strategy, which reduces the index size by up to an order of magnitude over naive storage of late-interaction embeddings as vectors of 16-bit floating-point numbers. Instead, ColBERTv2’s compression strategy efficiently clusters all tokenlevel vectors and encodes each vector using the ID of its nearest cluster centroid as well as a quantized residual vector, wherein each dimension is 1- or 2-bit encoding of the delta between the centroid and the original uncompressed vector. Decompressing a vector requires locating its centroid ID, encoded using 4 bytes, and its residual, which consume 16 or 32 bytes for 1- or 2-bit residuals, assuming the default 128-dimensional vectors.

While we adopt ColBERTv2’s compression, we improve its retrieval strategy. We refer to the original retrieval strategy as “vanilla” ColBERTv2 retrieval. We refer to Santhanam et al. [42] for details of compression and retrieval in ColBERTv2.

![](images/8e2ca9121de5d1a793db7b1f101fc500e9658644439064a4725cac1eab0a3983.jpg)  
Figure 4: Centroid score distribution for each query among a random sample of 15 MS MARCO v1 dev queries evaluated with ColBERTv2.

# 3.2 ColBERTv2 Latency Breakdown

Figure 2 presents a breakdown of query latency on MS MARCO Passage Ranking (v1) on a GPU, showing results for vanilla Col-BERTv2 (Figure 2a) against the new PLAID ColBERTv2 (Figure 2b). Latency is divided between query encoding, candidate generation, index lookups (i.e., to gather the compressed vector representations for candidate passages), residual decompression, and finally scoring (i.e., the final MaxSim computations).

For vanilla ColBERTv2, index lookup and residual decompression are overwhelming bottlenecks. Gathering vectors from the index is expensive because it consumes significant memory bandwidth: each vector in this setting is encoded with a 4-bit centroid ID and 32-byte residuals, each passage contains tens of vectors, and there can be up to $2 ^ { 1 6 }$ candidate passages. Moreover, index lookup in vanilla ColBERTv2 also constructs padded tensors on the fly to deal with the variable length of passages. Decompression of residuals is comprised of several non-trivial operations such as unpacking bits and computing large sums, which can be expensive when ColBERTv2 produces a large initial candidate set $( \sim 1 0 ^ { - 4 0 \mathrm { k } }$ passages) as is the case for MS MARCO v1. While it is possible to use a smaller candidate set, doing so reduces recall (§5).

# 3.3 Centroids Alone Identify Strong Candidates

This breakdown in Figure 2b demonstrates that exhaustively scoring a large number of candidates passages, particularly gathering and decompressing their residuals, can amount to a considerable cost. Whereas ColBERTv2 [42] exploits centroids to reduce the space footprint, our work demonstrates that the centroids can also accelerate search, while maintaining quality, by serving as proxies for the passage embeddings. Because of this, we can skip low-scoring passages without having to look up or decompress their residuals, adding some additional candidate generation overhead to achieve substantial savings in the subsequent stages (Figure 2b).

Effectively, we hypothesize that centroid-only retrieval can find the high-scoring passages otherwise retrieved by vanilla ColBERTv2. We test this hypothesis by comparing the top- $k$ passages retrieved by vanilla ColBERTv2 to a modified implementation that conducts retrieval using only the centroids and no residuals. We present the results in Figure 3. At $k \in \{ 1 0 , 1 0 0 , 1 0 0 0 \}$ , the figure plots the average recall of the top- $k$ passages of vanilla ColBERTv2 within the passages retrieved by centroid-only ColBERTv2 at various depths. In other words, we report the fraction of the top- $k$ passages of vanilla ColBERTv2 that appear within the top- $k ^ { \prime }$ passages of centroid-only ColBERTv2, for $k ^ { \prime } \geq k$ .

The results support our hypothesis, both in domain for MS MARCO v1 and out of domain using the LoTTE Pooled (dev) search queries [42]. For instance, if we retrieve $1 0 \cdot k$ passages using only centroids, those $1 0 \cdot k$ passages still contain $9 9 + \%$ of the top $k$ passages retrieved by the vanilla ColBERTv2 full pipeline.

# 3.4 Not All Centroids Are Important Per Query

We further hypothesize that for a given query a small subset of the passage embedding clusters tend to be far more important than others in determining relevance. If this were in fact the case, then we could prioritize computation over these highly weighted centroids and discard the rest since we know they will not contribute significantly to the final ranking. We test this theory by randomly sampling 15 MS MARCO v1 queries and plotting an empirical CDF of each centroid’s maximum relevance score observed across all query tokens, as shown in Figure 4. We do find that there is a small tail of highly weighted centroids whose relevance scores have far higher magnitude than all other centroids. While not shown in Figure 4, we also repeated this experiment with LoTTE pooled queries and found a very similar score distribution.

# 4 PLAID

Figure 5 illustrates the PLAID scoring pipeline, which consists of multiple consecutive stages for retrieval, filtering, and ranking. The first stage produces an initial candidate set by computing relevance scores for each centroid with respect to the query embeddings. In the intermediate stages, PLAID uses the novel techniques of centroid interaction and centroid pruning to aggressively yet effectively filter the candidate passages. Finally, PLAID ranks the final candidate set using fully reconstructed passage embeddings. We discuss each of these modules in more depth as follows.

# 4.1 Candidate Generation

Given the query embedding matrix $Q$ and the list of centroid vectors $C$ in the index, PLAID computes the token-level query–centroid relevance scores $S _ { c , q }$ as a matrix multiplication:

$$
\boldsymbol { S _ { c , q } } = \boldsymbol { C } \cdot \boldsymbol { Q } ^ { T }
$$

and then identifies the passages “close” to the top- $^ { \cdot } t$ centroids per query token as the initial candidate set. A passage is close to a centroid $i f$ one or more of its tokens are assigned to that centroid by $k$ -means clustering during indexing. This value $t$ is referred to as nprobe in vanilla ColBERTv2 and we retain that terminology in PLAID ColBERTv2.

The initial candidate generation in PLAID ColBERTv2 differs from the corresponding vanilla ColBERTv2 stage in two key aspects. First, while vanilla ColBERTv2 saves an inverted list mapping centroids to their corresponding embedding IDs, PLAID ColBERTv2 instead structures the inverted list as a map from centroids to the corresponding unique passage $I D s .$ Storing passage IDs is advantageous over storing embedding IDs since there are far fewer passages than embeddings, meaning the inverted list has to store less information overall. This also enables PLAID ColBERTv2 to use 32-bit integers in the inverted list rather than potentially 64-bit longs.2 In practice, this translates to a space savings of $2 . 7 \times$ in the MS MARCO v2 [6] inverted list (71 GB to 27 GB, with 140M passages).

Second, and relatedly, if the initial candidate set was too large (as specified by the ncandidates hyperparameter) vanilla ColBERTv2 would prune it by scoring and ranking a subset of the candidate embedding vectors—in particular, the embeddings listed within the vanilla mapping from centroid IDs to embedding IDs—with full residual decompression, which is quite costly as we discuss in $\ S 3 . 2$ In contrast, PLAID ColBERTv2 does not impose any limit on the initial candidate size because the subsequent stages can cheaply filter the candidate passages with centroid interaction and pruning.

# 4.2 Centroid Interaction

Centroid interaction cheaply approximates per-passage relevance by substituting each token’s embedding vector with its nearest centroid in the standard MaxSim formulation. By applying centroid interaction as an additional filtering stage, the scoring pipeline can skip the expensive embedding reconstruction process for a large fraction of the candidate passages. This results in significantly faster end-to-end retrieval. Intuitively, centroid interaction enables PLAID to emulate traditional bag-of-words retrieval wherein the centroid relevance scores take the role of the term relevance scores used in systems like BM25. However, because of its vector representations (of the query in particular), PLAID computes the centroid relevance scores at query time in contrast to the more traditional pre-computed term relevance scores.

The procedure works as follows. Recall that $S _ { c , q }$ from Equation 2 stores the relevance scores for each centroid with respect to the query tokens. Suppose $I$ is the list of the centroid indices mapped to each of the tokens in the candidate set. Furthermore, let $S _ { c , q } [ i ]$ denote the $i$ -th row of $S _ { c , q }$ . Then PLAID constructs the centroidbased approximate scores $\tilde { D }$ as

![](images/1878aeb26aa3b03024195750f6e256f387545de150b1b1698ce6239fe399ef82.jpg)  
Figure 5: The PLAID scoring pipeline. The first stage generates an initial set of candidate passages using the centroids. Next the second and third stages leverage centroid pruning and centroid interaction respectively to refine the candidate set. Then the last stage performs full residual decompression to obtain the final passage ranking. We use the hyperparameter ndocs to specify the number of candidates returned by Stage 2, and in our experiments we have Stage 3 output $\frac { \boldsymbol { \mathsf { n d o c s } } } { 4 }$ passages.

$$
\tilde { D } = \left[ \begin{array} { c } { S _ { c , q } [ I _ { 1 } ] } \\ { S _ { c , q } [ I _ { 2 } ] } \\ { \dots } \\ { S _ { c , q } [ I _ { | \tilde { D } | } ] } \end{array} \right]
$$

Then to rank the candidate passages using $\tilde { D }$ , PLAID computes the MaxSim scores $s _ { \tilde { D } }$ as

$$
S _ { \tilde { D } } = \sum _ { i } ^ { | Q | } \frac { | \tilde { D } | } { j = 1 } \tilde { D } { i , j }
$$

The top $k$ most relevant passages drawn from $S _ { \tilde { D } }$ serve as the filtered candidate passage set.

PLAID includes optimized kernels to efficiently deploy centroid interaction (and more generally MaxSim operations); we discuss these in $\ S 4 . 5$ .

# 4.3 Centroid Pruning

As an additional optimization, PLAID leverages the observation from $\ S 3 . 3$ to first prune low-magnitude centroid scores before constructing $\tilde { D }$ . In this filtering phase PLAID will only score tokens whose maximum corresponding centroid score meets the given threshold $t _ { c s }$ . Concretely, $\tilde { D }$ will only be comprised of tokens whose corresponding centroid (suppose centroid $i$ ) meets the following condition:

$$
\operatorname* { m a x } _ { j = 1 } ^ { | Q | } S _ { c , q _ { i , j } } \geq t _ { c s }
$$

We introduce the hyperparameter ndocs to refer to the number of candidate documents selected by Stage 2. We then found empirically that choosing $\frac { \mathsf { n d o c s } } { 4 }$ candidates from Stage 3 produced good results; we use this heuristic for all the results presented in $\ S 5$ .

# 4.4 Scoring

As in vanilla ColBERTv2, PLAID will reconstruct the original embeddings of the final candidate passage set via residual decompression and rank these using MaxSim. Let $D$ be the reconstructed embedding vectors for the final candidate set after decompression. Then the final scores $S _ { q , d }$ are computed using Equation 1.

Section $\ S 4 . 5$ discusses fast kernels for accelerating the MaxSim and decompression operations.

# 4.5 Fast Kernels: Padding-Free MaxSim & Optimized Decompression

Figure 2a shows that index lookup operations are a large source of overhead for vanilla ColBERTv2. One reason these lookups are expensive is that they require reshaping and padding the 2D index tensors with an extra dimension representing the maximum passage length. The resulting 3D tensors facilitate batched MaxSim operations over ragged lists of token vectors. To avoid this padding, we instead implement custom ${ \mathrm { C } } { + } { + }$ code that directly computes the MaxSim scores over the packed 2D index tensors (i.e., one where many 2D sub-tensors of various lengths are concatenated along the same dimension). Our kernel loops over each passage’s corresponding token vectors to compute the per-passage maximum scores with respect to each query token and then sums the perpassage maximum scores across all query tokens. This design is trivial to parallelize across passages, and also enables $O ( | Q | )$ perthread memory usage by allocating a single output vector to store the maximum scores per query token and repeatedly updating this vector in-place. In contrast, the padding-based approach requires $O ( | D | \cdot | Q | )$ space. We have incorporated this design into optimized implementations of centroid interaction as well as the final MaxSim operation (stage 4 in Figure 5). PLAID only implements these kernels for CPU execution. Adding corresponding GPU kernels remains future work.

<table><tr><td>Dataset</td><td># Passages</td><td># Tokens</td><td># Queries</td><td colspan="2">ColBERTv2 Index Size (GiB)</td></tr><tr><td></td><td></td><td></td><td></td><td>Vanilla</td><td>PLAID</td></tr><tr><td>MS MARCO v1 [36]</td><td>8.8M</td><td>597.9M</td><td>6980</td><td>24.6</td><td>21.6</td></tr><tr><td>Wikipedia [18]</td><td>21.0M</td><td>2.6B</td><td>8757</td><td>105.2</td><td>92.0</td></tr><tr><td>LoTTE pooled [42]</td><td>2.4M</td><td>339.4M</td><td>2931</td><td>14.0</td><td>12.3</td></tr><tr><td>MS MARCO v2 [6]</td><td>138.4M</td><td>9.4B</td><td>3903</td><td>246.0</td><td>202.2</td></tr></table>

Table 1: List of benchmarks used for evaluation with relevant statistics.

ColBERTv2’s residual decompression scheme computes a list of centroid vectors, determines a fixed set of $2 ^ { b }$ possible deltas from these centroids, and then stores the index into the set of deltas corresponding to each embedding vector. In particular, each compressed 8-bit value stores $\frac { 8 } { b }$ indices in the range $[ 0 , 2 ^ { b } )$ . Col-BERTv2 incurs significant overhead due to residual decompression, as shown in Figure 2a. This is partially due to the naïve decompression implementation, which required explicitly unpacking bits from the compressed representation and performing expensive bit shift and sum operations to recover the original values. Instead, PLAID pre-computes all $2 ^ { 8 }$ possible lists of indices encoded by an 8-bit packed value. These outputs are stored in a lookup table so that the decompression function can simply retrieve the indices from the table rather than manually unpacking the bits. We include optimized implementations of this lookup-based decompression for both CPU and GPU execution. The GPU implementation uses a custom CUDA kernel that allocates a separate thread to decompress each individual byte in the compressed residual tensor (the thread block size is computed as ?? ·?? for ??-dimensional embedding vectors). The CPU implementation instead parallelizes decompression at the granularity of individual passages.

# 5 EVALUATION

Our evaluation seeks to answer the following research questions:

(1) How does PLAID affect end-to-end latency and retrieval quality across IR benchmarks? (§5.2)   
(2) How much do each of PLAID’s optimizations contribute to the performance speedups? (§5.3)   
(3) How well does PLAID scale with respect to the corpus size and the parallelism degree? (§5.4)

# 5.1 Setup

PLAID Implementation. The PLAID engine subsumes centroid interaction as well as optimizations for residual decompression. We implement PLAID modularly as an extension to ColBERTv2’s

Table 2: PLAID hyperparameter configuration.   

<table><tr><td>k</td><td>nprobe</td><td></td><td>ndocs</td></tr><tr><td>10</td><td>1</td><td>0.5</td><td>256</td></tr><tr><td>100</td><td>2</td><td>0.45</td><td>1024</td></tr><tr><td>1000</td><td>4</td><td>0.4</td><td>4096</td></tr></table>

PyTorch-based implementation, particularly its search components. For CPU execution, we implement the centroid interaction and decompression operations entirely in multithreaded ${ \mathrm { C } } { + } { + }$ code. For GPUs, we implement centroid interaction in PyTorch and provide a CUDA kernel for fast decompression. Overall, PLAID constitutes roughly 300 lines of additional Python code and 700 lines of ${ \mathrm { C } } { + } { + }$ .

Datasets. Our evaluation includes results from four different IR benchmarks, as listed in Table 1. We perform in-domain evaluation on the MS MARCO v1 and Wikipedia Open QA benchmarks, with retrievers trained specifically for these tasks, and out-of-domain evaluation on the StackExchange-based LoTTE Santhanam et al. [42] and the TREC 2021 Deep Learning Track [6] MS MARCO v2 benchmarks, with the ColBERTv2 retriever [42] trained on MS MARCO v1. For evaluation on Wikipedia we use the December 2018 dump [18] with queries from the NaturalQuestions (NQ) dataset [23]. Our LoTTE [42] evaluation uses the “pooled” dev dataset with “search”- style queries. For MS MARCO v2, we use the augmented passage version of the data [2] and include passage titles while ignoring headings. As we evaluate several configurations of the models, all of our evaluation is performed using development set queries.

Systems and hyperparameters. We report results for several systems for end-to-end results: vanilla ColBERTv2 and PLAID Col-BERTv2 as well as ColBERT (v1) [22], BM25 [41], SPLADEv2 [10], and DPR [18]. For vanilla ColBERTv2, we use the specific hyperparameters reported in the ColBERTv2 paper for each benchmark dataset. We indicate these in the result tables with p (nprobe) and c (ncandidates). For PLAID ColBERTv2, we evaluate three different settings: $k = 1 0$ , $k = 1 0 0$ , and $k = 1 0 0 0$ . The $k$ parameter controls the final number of scored documents as well as the retrieval hyperparameters described in $\ S 4$ . Table 2 lists these hyperparameter configurations for each $k$ setting. We find empirically that ranking ndocs4 documents for the final scoring stage produces strong results. For both vanilla ColBERTv2 and PLAID ColBERTv2, we compress all datasets to 2 bits per dimension, with the exception of MS MARCO v2 where we compress to 1 bit.

Hardware. We conduct all experiments on servers with 28 Intel Xeon Gold 6132 2.6 GHz CPU cores (2 threads per core for a total of 56 threads) and 4 NVIDIA TITAN V GPUs each. Every server has two NUMA sockets with roughly 92 ns intra-socket memory latency, 142 ns inter-socket memory latency, 72 GBps intra-socket memory bandwidth, and 33 GBps inter-socket memory bandwidth. Each TITAN V GPU has 12 GB of high-bandwidth memory.

Latency measurements. When measuring latency for end-to-end results, we compute the average latency of all queries (see Table 1 for query totals), and then report the minimum average latency across 3 trials. For other results we describe the specific measurement procedure in the relevant section. We discard the query encoding latency for neural models (ColBERTv1 [22], vanilla Col-BERTv2 [42], PLAID ColBERTv2, and SPLADEv2 [10]) following Mackenzie et al. [30]; prior work has shown that the cost of running the BERT model can be made negligible with standard techniques such as quantization, distillation, etc. [4]. We measure latency on an otherwise idle machine. We prepend commands with numactl --membind 0 to ensure intra-socket I/O operations. We do not do this for MS MARCO v2, since its large index may require both NUMA nodes. For GPU results we allow full usage of all 56 threads, but for CPU-only results we restrict usage to either 1 or 8 threads using torch.set_num_threads. For non-ColBERT systems we use the single-threaded latency numbers reported by Mackenzie et al. [30]. Note that these numbers were measured on a different hardware setup and using a different implementation and are therefore simply meant to establish PLAID ColBERTv2’s competitive performance rather than serving as absolute comparisons.

# 5.2 End-to-end Results

Table 3: End-to-end in-domain evaluation on the MS MARCO v1 benchmark. Numbers marked with an asterisk are copied from Formal et al. [11] for SPLADEv2 quality and Mackenzie et al. [30] for latencies.   

<table><tr><td rowspan="2">System</td><td rowspan="2">MRR@10 R@100 R@1k</td><td rowspan="2"></td><td colspan="2">Latency (ms)</td></tr><tr><td></td><td>1-CPU 8-CPU GPU</td></tr><tr><td>BM25 (PISA [34]; k = 1000) SPLADEv2 (PISA; k = 1000)</td><td>18.7* 36.8*</td><td></td><td>8.3* 97.9* 220.3*</td><td>-</td></tr><tr><td>ColBERTv1</td><td>36.1</td><td>87.3</td><td>95.2</td><td>- -</td></tr><tr><td>Vanilla ColBERTv2 (p=2, c=213)</td><td>39.7</td><td>90.4</td><td>96.6 3485.1</td><td>921.8</td></tr><tr><td>Vanilla ColBERTv2 (p=4, c=216)</td><td>39.7</td><td>91.4</td><td>98.3</td><td>53.4 - 4568.5 259.6</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PLAID ColBERTv2 (k = 10)</td><td>39.4</td><td></td><td>185.5</td><td>31.5</td></tr><tr><td>PLAID ColBERTv2 (k = 100)</td><td>39.8</td><td>90.6</td><td>- 222.3</td><td>52.9 20.2</td></tr><tr><td>PLAID ColBERTv2 (k = 1000)</td><td>39.8</td><td>91.3</td><td>97.5 352.3</td><td>101.3 38.4</td></tr></table>

Table 4: End-to-end in-domain retrieval evaluation on the Wikipedia open-domain question answering benchmark. We use the NQ checkpoint of ColBERT-QA [21], and apply ColBERTv2 compression. We compare vanilla ColBERTv2 retrieval against PLAID ColBERTv2 retrieval. DPR results from Karpukhin et al. [18]. We refer to Khattab et al. [21] for details on OpenQA retrieval evaluation.   

<table><tr><td rowspan="2">System</td><td rowspan="2">Success@5 Success@100</td><td rowspan="2"></td><td colspan="2">Latency (ms)</td></tr><tr><td>CPU (8) GPU</td><td></td></tr><tr><td>DPR</td><td>66.8</td><td>85.0</td><td></td><td></td></tr><tr><td>ColBERT-QA Retrieval (uncompressed)</td><td>75.3</td><td>89.2</td><td></td><td>-</td></tr><tr><td colspan="5">ColBERT-QA [21] Retriever with ColBERTv2 [42] residual compression</td></tr><tr><td>Vanilla ColBERT-QA Retrieval (p=4, c=215)</td><td>74.3</td><td>89.0</td><td>5077.9</td><td>204.1</td></tr><tr><td>PLAID ColBERT-QA Retrieval (k = 10)</td><td>73.3</td><td>-</td><td>67.1</td><td>13.6</td></tr><tr><td>PLAID ColBERT-QA Retrieval (k = 100)</td><td>74.1</td><td>88.0</td><td>120.1</td><td>26.9</td></tr><tr><td>PLAID ColBERT-QA Retrieval (k = 1000)</td><td>74.4</td><td>88.9</td><td>228.4</td><td>55.3</td></tr></table>

Table 5: End-to-end out-of-domain evaluation on the (dev) pooled dataset of the LoTTE benchmark. Numbers marked with an asterisk were taken from Santhanam et al. [42].   

<table><tr><td rowspan="2">System</td><td rowspan="2">Success@5</td><td rowspan="2">Success@100</td><td colspan="2">Latency (ms)</td></tr><tr><td>CPU (8)</td><td>GPU</td></tr><tr><td>BM25</td><td>47.8*</td><td>77.6*</td><td>,</td><td></td></tr><tr><td>SPLADEv2</td><td>67.0*</td><td>89.0*</td><td>-</td><td>-</td></tr><tr><td>Vanilla ColBERTv2 (p=2, c=213)</td><td>69.3</td><td>90.3</td><td>1508.4</td><td>66.9</td></tr><tr><td>PLAID ColBERTv2 (k = 10)</td><td>69.1</td><td>-</td><td>35.5</td><td>9.2</td></tr><tr><td>PLAID ColBERTv2 (k = 100)</td><td>69.4</td><td>89.9</td><td>64.8</td><td>17.4</td></tr><tr><td>PLAID ColBERTv2 (k = 1000)</td><td>69.6</td><td>90.5</td><td>163.1</td><td>27.3</td></tr></table>

<table><tr><td>System</td><td rowspan="2">MRR@100</td><td rowspan="2">R@1k R@100</td><td rowspan="2"></td><td>Latency (ms) GPU</td></tr><tr><td></td><td>8-CPU</td></tr><tr><td>BM25 (Anserini [50]; Augmented)</td><td>8.7</td><td>40.3</td><td>69.3</td><td>- -</td></tr><tr><td>Vanilla ColBERTv2 (p=4, c=216)</td><td>18.0</td><td>68.2</td><td>88.1</td><td>5228.5 OOM</td></tr><tr><td>PLAID ColBERTv2 (k = 10)</td><td>-</td><td>-</td><td>-</td><td>136.4 47.1</td></tr><tr><td>PLAID ColBERTv2 (k = 100)</td><td>17.9</td><td>67.0</td><td>-</td><td>181.9 96.1</td></tr><tr><td>PLAID ColBERTv2 (k = 1000)</td><td>18.0</td><td>68.4</td><td>85.7</td><td>251.3 OOM</td></tr></table>

Table 6: End-to-end out-of-domain evaluation on the MS MARCO v2 benchmark. BM25 results from [3].

Table 3 presents in-domain results for the MS MARCO v1 benchmark. We observe that in the most conservative setting $( k = 1 0 0 0 )$ , PLAID ColBERTv2 is able to match the $\mathbf { M R R } @ 1 0$ and Recall@100 achieved by vanilla ColBERTv2 while delivering speedups of $6 . 8 \times$ on GPU and $4 5 \times$ on CPU. For some minimal reduction in quality, PLAID ColBERTv2 can further increase the speedups over vanilla ColBERTv2 to $1 2 . 9 \mathrm { - } 2 2 . 6 \times$ on GPU and $8 6 . 4 \substack { - 1 4 5 \times }$ on CPU. PLAID ColBERTv2 also achieves competitive latency compared to other systems (within $1 . 6 \times$ of SPLADEv2) while outperforming them on retrieval quality.

We observe a similar trend with in-domain evaluation on the Wikipedia OpenQA benchmark as shown in Table 4. PLAID Col-BERTv2 achieves speedups of $3 . 7 \times$ on GPU and $2 2 \times$ on CPU with no quality loss compared to vanilla ColBERTv2, and speedups of $7 . 6 \mathrm { - } 1 5 \times$ on GPU and $4 2 . 3 \mathrm { - } 7 5 . 7 \times$ on CPU with minimal quality loss.

We confirm PLAID works well in out-of-domain settings, as well, as demonstrated by our results on the LoTTE “pooled” dataset. We see in Table 5 that PLAID ColBERTv2 outperforms vanilla Col-BERTv2 by $2 . 5 \times$ on GPU and $9 . 2 \times$ on CPU with $k = 1 0 0 0$ ; furthermore, this setting actually improves quality compared to vanilla ColBERTv2. With some quality loss PLAID ColBERTv2 can achieve speedups of $3 . 8 \mathrm { - } 7 . 3 \mathrm { \times }$ on GPU and $2 3 . 2 \mathrm { - } 4 2 . 5 \times$ on CPU. Note that the CPU latencies achieved on LoTTE with PLAID ColBERTv2 are larger than those achieved on MS MARCO v1 because the average LoTTE passage length is roughly $2 \times$ that of MS MARCO v1.

Finally, Table 6 shows that PLAID ColBERTv2 scales effectively to MS MARCO v2, which is a large-scale dataset with 138M passages and 9.4B tokens (approximately $1 6 \times$ bigger than MS MARCO v1). Continuing the trend we observe with other datasets, we find that PLAID ColBERTv2 is $2 0 . 8 \times$ faster than vanilla ColBERTv2 on CPU with no quality loss up to 100 passages. We do find that when $k = 1 0 0 0$ both vanilla ColBERTv2 and PLAID ColBERTv2 run out of memory on GPU; we believe we can address this in PLAID by implementing custom padding-free MaxSim kernels for GPU execution as discussed in $\ S 4 . 5$ .

![](images/7515274f7a1f117206f6f1b97bd622976f43d8deebfd67468d726418ffae2c11.jpg)  
Figure 6: Ablation of performance optimizations included in PLAID.

# 5.3 Ablation

Figure 6 presents an ablation analysis to break down PLAID’s performance improvements for both GPU and CPU execution. Our measurements are taken from evaluation on a random sample of 500 MS MARCO v1 queries (note that this results in minor differences in the absolute numbers reported in Table 3). We consider vanilla ColBERTv2 as a baseline, and then add one stage of centroid interaction without pruning (stage 3 in Figure 5), followed by another stage of centroid interaction with centroid pruning (stage 2 in Figure 5), and then finally the optimized kernels described in $\ S 4 . 5$ . When applicable we use hyperparameters corresponding to the $k = 1 0 0 0$ setting described in Table 2 (i.e., the most conservative setting).

We find that both the algorithmic improvements to the scoring pipeline as well as the implementation optimizations are key to PLAID’s performance. In particular, the centroid interaction stages alone deliver speedups of $5 . 2 \times$ on GPU and $8 . 6 \times$ on CPU, but adding the implementation optimizations result in additional speedups of $1 . 3 \times$ on GPU and $4 . 9 \times$ on CPU. Only enabling optimized ${ \mathrm { C } } { + } { + }$ kernels on CPU without centroid interaction (not shown in Figure 6) results in an end-to-end speedup of just $3 \times$ compared to $4 2 . 4 \times$ with the complete PLAID.

# 5.4 Scalability

We evaluate PLAID’s scalability with respect to both the dataset size as well as the parallelism degree (on CPU).

First, Figure 7 plots the end-to-end PLAID ColBERTv2 latencies we measured for each benchmark dataset versus the size of each dataset (measured in number of embeddings). While latencies across different datasets are not necessarily directly comparable (e.g due to different passage lengths), we nevertheless aim to analyze high-level trends from this figure. We find that in general, PLAID ColBERTv2 latencies appear to scale with respect to the square root of dataset size. This intuitively follows from the fact that ColBERTv2 sets the number of centroids proportionally to the square root of the number of embeddings, and the overhead of candidate generation is inversely correlated with the number of partitions.

![](images/8ee0cfc0f84faf7f975c8389d1e962601c6ae0404370ecb21b6f6b57167b43c5.jpg)  
Figure 7: End-to-end latency versus dataset size (as measured in number of embeddings) for each setting of $k$ (note the log-log scale). Dataset sizes are taken from Table 1, and latency numbers are taken from Tables 3, 4, 5, and 6.

![](images/7ad00a97a25dcb389dd26e73e9f651700de67e4295da044c6e034f7ce86d4993.jpg)  
Figure 8: PLAID scaling behavior with respect to the number of available CPU threads.

Next, Figure 8 plots the latency achieved by PLAID ColBERTv2 versus the number of available CPU threads, repeated for $k \in$ $\{ 1 0 , 1 0 0 , 1 0 0 0 \}$ . We evaluate a random sample of 500 MS MARCO v1 queries to obtain the latency measurements. We observe that PLAID is able to take advantage of additional threads; in particular, executing with 16 threads results in a speedup of $4 . 9 \times$ compared to single-threaded execution when $k = 1 0 0 0$ . While PLAID does not achieve perfect linear scaling, we speculate that possible explanations could include remaining inefficiencies in the existing vanilla ColBERTv2 candidate generation step (which we do not optimize at a low level for this work) or suboptimal load balancing between threads due to the non-uniform passage lengths. We defer more extensive profiling and potential solutions to future work.

# 6 CONCLUSION

In this work, we presented PLAID, an efficient engine for late interaction that accelerates retrieval by aggressively and cheaply filtering candidate passages. We showed that retrieval with only ColBERTv2 centroids retains high recall compared to vanilla Col-BERTv2, and the distribution of centroid relevance scores skews toward lower magnitude scores. Using these insights, we introduced the technique of centroid interaction and incorporated centroid interaction into multiple stages of the PLAID ColBERTv2 scoring pipeline. We also described our highly optimized implementation of PLAID that includes custom kernels for padding-free MaxSim and residual decompression operations. We found in our evaluation across several IR benchmarks that PLAID ColBERTv2 provides speedups of $2 . 5 \substack { - 6 . 8 \times }$ on GPU and $9 . 2 \mathrm { - } 4 5 \times$ on CPU with virtually no quality loss compared to vanilla ColBERTv2 while scaling effectively to a dataset of 140 million passages.

# REFERENCES

[1] Firas Abuzaid, Geet Sethi, Peter Bailis, and Matei Zaharia. 2019. To index or not to index: Optimizing exact maximum inner product search. In 2019 IEEE 35th International Conference on Data Engineering (ICDE). IEEE, 1250–1261. [2] Anserini GitHub Repo Authors. 2021. Passage Collection (Augmented). https://github.com/castorini/anserini/blob/master/docs/experiments-msmarcov2.md#passage-collection-augmented   
[3] Anserini GitHub Repo Authors. 2022. Anserini Regressions: MS MARCO (V2) Passage Ranking. https://github.com/castorini/anserini/blob/master/docs/ regressions-msmarco-v2-passage-augmented.md   
[4] Jo Kristian Bergum. 2021. Pretrained Transformer Language Models for Search - part 3. https://blog.vespa.ai/pretrained-transformer-language-models-forsearch-part-3/ [5] Andrei Z Broder, David Carmel, Michael Herscovici, Aya Soffer, and Jason Zien. 2003. Efficient query evaluation using a two-level retrieval process. In CIKM.   
[6] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Jimmy Lin. 2022. Overview of the TREC 2021 deep learning track. In Text REtrieval Conference (TREC). TREC. https://www.microsoft.com/en-us/research/publication/ overview-of-the-trec-2021-deep-learning-track/   
[7] Zhuyun Dai and Jamie Callan. 2020. Context-Aware Term Weighting For First Stage Passage Retrieval. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020, Jimmy Huang, Yi Chang, Xueqi Cheng, Jaap Kamps, Vanessa Murdock, Ji-Rong Wen, and Yiqun Liu (Eds.). ACM, 1533–1536. https://doi.org/10.1145/3397271.3401204   
[8] Constantinos Dimopoulos, Sergey Nepomnyachiy, and Torsten Suel. 2013. Optimizing top-k document retrieval strategies for block-max indexes. In WSDM.   
[9] Shuai Ding and Torsten Suel. 2011. Faster top-k document retrieval using blockmax indexes. In SIGIR.   
[10] Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant. 2021. SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval. arXiv preprint arXiv:2109.10086 (2021). https://arxiv.org/abs/2109.10086   
[11] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021. SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2288–2292.   
[12] Luyu Gao, Zhuyun Dai, and Jamie Callan. 2021. COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, Online, 3030–3042. https://doi.org/10.18653/v1/2021.naacl-main.241   
[13] Sebastian Hofstätter, Sophia Althammer, Michael Schröder, Mete Sertkan, and Allan Hanbury. 2020. Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation. arXiv preprint arXiv:2010.02666 (2020). https://arxiv.org/abs/2010.02666   
[14] Sebastian Hofstätter, Omar Khattab, Sophia Althammer, Mete Sertkan, and Allan Hanbury. 2022. Introducing Neural Bag of Whole-Words with ColBERTer: Contextualized Late Interactions using Enhanced Reduction. arXiv preprint arXiv:2203.13088 (2022).   
[15] Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, and Jason Weston. 2020. Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net. https://openreview.net/forum?id=SkxgnnNFvH   
[16] Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence 33, 1 (2010), 117–128.   
[17] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with gpus. IEEE Transactions on Big Data (2019).   
[18] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, Online, 6769–6781. https://doi.org/10.18653/v1/2020.emnlp-main.550   
[19] Omar Khattab, Mohammad Hammoud, and Tamer Elsayed. 2020. Finding the best of both worlds: Faster and more robust top-k document retrieval. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 1031–1040.   
[20] Omar Khattab, Christopher Potts, and Matei Zaharia. 2021. Baleen: Robust Multi-Hop Reasoning at Scale via Condensed Retrieval. In Thirty-Fifth Conference on Neural Information Processing Systems.   
[21] Omar Khattab, Christopher Potts, and Matei Zaharia. 2021. Relevance-guided Supervision for OpenQA with ColBERT. Transactions of the Association for Computational Linguistics 9 (2021), 929–944.   
[22] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020, Jimmy Huang, Yi Chang, Xueqi Cheng, Jaap Kamps, Vanessa Murdock, Ji-Rong Wen, and Yiqun Liu (Eds.). ACM, 39–48. https://doi.org/10.1145/3397271.3401075   
[23] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: A Benchmark for Question Answering Research. Transactions of the Association for Computational Linguistics 7 (2019), 452–466. https://doi.org/10.1162/tacl_a_00276   
[24] Yulong Li, Martin Franz, Md Arafat Sultan, Bhavani Iyer, Young-Suk Lee, and Avirup Sil. 2021. Learning Cross-Lingual IR from an English Retriever. arXiv preprint arXiv:2112.08185 (2021).   
[25] Jimmy Lin. 2022. A proposed conceptual framework for a representational approach to information retrieval. In ACM SIGIR Forum, Vol. 55. ACM New York, NY, USA, 1–29.   
[26] Simon Lupart and Stéphane Clinchant. 2022. Toward A Fine-Grained Analysis of Distribution Shifts in MSMARCO. arXiv preprint arXiv:2205.02870 (2022).   
[27] Sean MacAvaney, Andrew Yates, Arman Cohan, and Nazli Goharian. 2019. CEDR: Contextualized Embeddings for Document Ranking. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. 1101–1104. https://doi.org/10.1145/3331184.3331317   
[28] Craig Macdonald and Nicola Tonellotto. 2021. On approximate nearest neighbour selection for multi-stage dense retrieval. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 3318–3322.   
[29] Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. 2021. On Single and Multiple Representations in Dense Passage Retrieval. arXiv preprint arXiv:2108.06279 (2021).   
[30] Joel Mackenzie, Andrew Trotman, and Jimmy Lin. 2021. Wacky Weights in Learned Sparse Representations and the Revenge of Score-at-a-Time Query Evaluation. arXiv preprint arXiv:2110.11540 (2021).   
[31] Yu A Malkov and Dmitry A Yashunin. 2018. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE transactions on pattern analysis and machine intelligence 42, 4 (2018), 824–836.   
[32] Antonio Mallia, Omar Khattab, Torsten Suel, and Nicola Tonellotto. 2021. Learning passage impacts for inverted indexes. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1723–1727.   
[33] Antonio Mallia, Giuseppe Ottaviano, Elia Porciani, Nicola Tonellotto, and Rossano Venturini. 2017. Faster blockmax wand with variable-sized blocks. In SIGIR.   
[34] Antonio Mallia, Michal Siedlaczek, Joel Mackenzie, and Torsten Suel. 2019. PISA: Performant indexes and search for academia. Proceedings of the Open-Source IR Replicability Challenge (2019).   
[35] Antonios Minas Krasakis, Andrew Yates, and Evangelos Kanoulas. 2022. Zeroshot Query Contextualization for Conversational Search. arXiv e-prints (2022), arXiv–2204.   
[36] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. MS MARCO: A Human-Generated MAchine Reading COmprehension Dataset. arXiv preprint arXiv:1611.09268 (2016). https: //arxiv.org/abs/1611.09268   
[37] Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT. arXiv preprint arXiv:1901.04085 (2019). https://arxiv.org/abs/1901.04085   
[38] Ashwin Paranjape, Omar Khattab, Christopher Potts, Matei Zaharia, and Christopher D Manning. 2022. Hindsight: Posterior-guided Training of Retrievers for Improved Open-ended Generation. In International Conference on Learning Representations. https://openreview.net/forum?id=Vr_BTpw3wz   
[39] Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. 2021. RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, Online, 5835–5847. https: //doi.org/10.18653/v1/2021.naacl-main.466   
[40] Ruiyang Ren, Yingqi Qu, Jing Liu, Wayne Xin Zhao, Qiaoqiao She, Hua Wu, Haifeng Wang, and Ji-Rong Wen. 2021. RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking. arXiv preprint arXiv:2110.07367 (2021). https://arxiv.org/abs/2110.07367   
[41] Stephen E Robertson, Steve Walker, Susan Jones, Micheline M Hancock-Beaulieu, Mike Gatford, et al. 1995. Okapi at TREC-3. NIST Special Publication (1995).   
[42] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2021. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. arXiv preprint arXiv:2112.01488 (2021).   
[43] Katherine Thai, Yapei Chang, Kalpesh Krishna, and Mohit Iyyer. 2022. RELIC: Retrieving Evidence for Literary Claims. arXiv preprint arXiv:2203.10053 (2022).   
[44] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2). https://openreview. net/forum?id=wCu6T5xFjeJ   
[45] Nicola Tonellotto and Craig Macdonald. 2021. Query embedding pruning for dense retrieval. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 3453–3457.   
[46] Nicola Tonellotto, Craig Macdonald, Iadh Ounis, et al. 2018. Efficient Query Processing for Scalable Web Search. Foundations and Trends® in Information Retrieval (2018).   
[47] Howard Turtle and James Flood. 1995. Query evaluation: strategies and optimizations. IP & M (1995).   
[48] Xiao Wang, Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. 2021. Pseudorelevance feedback for multiple representation dense retrieval. In Proceedings of the 2021 ACM SIGIR International Conference on Theory of Information Retrieval. 297–306.   
[49] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N Bennett, Junaid Ahmed, and Arnold Overwijk. 2020. Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. In International Conference on Learning Representations.   
[50] Peilin Yang, Hui Fang, and Jimmy Lin. 2018. Anserini: Reproducible ranking baselines using Lucene. Journal of Data and Information Quality (JDIQ) 10, 4 (2018), 1–20.   
[51] Hansi Zeng, Hamed Zamani, and Vishwa Vinay. 2022. Curriculum Learning for Dense Retrieval Distillation. arXiv preprint arXiv:2204.13679 (2022).   
[52] Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Min Zhang, and Shaoping Ma. 2020. Learning To Retrieve: How to Train a Dense Retrieval Model Effectively and Efficiently. arXiv preprint arXiv:2010.10469 (2020). https://arxiv.org/abs/2010.10469   
[53] Jingtao Zhan, Xiaohui Xie, Jiaxin Mao, Yiqun Liu, Min Zhang, and Shaoping Ma. 2022. Evaluating Extrapolation Performance of Dense Retrieval. arXiv preprint arXiv:2204.11447 (2022).   
[54] Wei Zhong, Jheng-Hong Yang, and Jimmy Lin. 2022. Evaluating Token-Level and Passage-Level Dense Retrieval Models for Math Information Retrieval. arXiv preprint arXiv:2203.11163 (2022).