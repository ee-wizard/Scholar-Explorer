# ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data

Liana Patel

Stanford University

Stanford, USA

lianapat@stanford.edu

Carlos Guestrin

Stanford University

Stanford, USA

guestrin@stanford.edu

Peter Kraft

DBOS, Inc.

USA

peter.kraft@dbos.dev

Matei Zaharia

UC Berkeley

Berkeley, USA

matei@berkeley.edu

# ABSTRACT

Applications increasingly leverage mixed-modality data, and must jointly search over vector data, such as embedded images, text and video, as well as structured data, such as attributes and keywords. Proposed methods for this hybrid search setting either suffer from poor performance or support a severely restricted set of search predicates (e.g., only small sets of equality predicates), making them impractical for many applications. To address this, we present ACORN, an approach for performant and predicate-agnostic hybrid search. ACORN builds on Hierarchical Navigable Small Worlds (HNSW), a state-of-the-art graph-based approximate nearest neighbor index, and can be implemented efficiently by extending existing HNSW libraries. ACORN introduces the idea of predicate subgraph traversal to emulate a theoretically ideal, but impractical, hybrid search strategy. ACORN’s predicate-agnostic construction algorithm is designed to enable this effective search strategy, while supporting a wide array of predicate sets and query semantics. We systematically evaluate ACORN on both prior benchmark datasets, with simple, low-cardinality predicate sets, and complex multi-modal datasets not supported by prior methods. We show that ACORN achieves state-of-the-art performance on all datasets, outperforming prior methods with $2 \mathrm { - } 1 \mathrm { , } 0 0 0 \mathrm { \times }$ higher throughput at a fixed recall.

# CCS CONCEPTS

• Information systems Information retrieval query processing; Data structures.

# KEYWORDS

Vector Search, Approximate Nearest Neighbor Search, Hybrid Search

# 1 INTRODUCTION

Due to the representation strength of modern deep learning models, vector embeddings have become a powerful first-class datatype for wide-ranging applications that use retrieval-augmented generation [3, 65] or similarity-based search [18, 21, 42]. As a result, vector databases and indices are seeing increasing adoption in many production use cases. These systems provide an efficient approximate-nearest-neighbor (ANN) search interface over embedded unstructured data e.g., images, text, video, or user profiles.

However, many applications must jointly query both unstructured and structured data, requiring ANN search in combination with

predicate filtering. For example, customers on an e-commerce site can search for t-shirts similar to a reference image, while filtering on price [64]. Similarly, researchers performing a literature review may search with both natural language queries and filters on publication date, keywords or topics [54]. Likewise, a data scientist working on outlier detection can find misclassified images by retrieving those that look similar to a reference dog but have the label "cat" [2, 7].

To leverage diverse data modalities, applications need data management systems that effectively support hybrid search queries, i.e., similarity search with structured predicates. Such systems require (1) query performance, i.e., efficient and accurate search despite variance in workload characteristics, such as selectivity, attribute correlations, and scale, and (2) expressive query semantics: support for diverse query predicates that may not be known a priori (e.g., user-entered keywords, range searches, or regex matching).

Unfortunately, existing systems fall short of these goals. Three commonly used methods are pre-filtering [62, 64], post-filtering [1, 5, 62, 64, 67], and specialized data structures for low-cardinality predicate sets [25, 49, 63, 66]. Pre-filtering first finds all records in the dataset that pass the query predicate, then performs brute force similarity-search over the filtered vector set. This approach scales poorly, becoming inefficient for medium to high selectivity predicates on large datasets. Alternatively, post-filtering first searches an ANN index, then removes results that fail the query predicate. Since the database vectors closest to the query vector may not pass the predicate, post-filtering methods must typically expand the search scope. This is often expensive, particularly for search predicates with low selectivity or low correlation to the query vector, as we show in Figure 2. Milvus [62], Weaviate [1], AnalyticDB-V [64], and FAISS-IVF [5] build systems using these two core methods, and suffer from their performance limitations.

Recognizing these limitations, recent works construct specialized indices designed for hybrid search workloads with low-cardinality predicate sets consisting of equality predicate operators. For example, Filtered-DiskANN [25] outperforms prior baselines, but restricts the cardinality of the predicate set to about 1,000 and only supports equality predicates. HQANN [66] and NHQ [12] similarly constrain the predicate set to a small number of equality filters and in addition allow only a single structured attributes per dataset entry. These methods are often impractical since many applications have large, or unbounded predicate sets that are unknown a priori. In general, the possible predicate set’s cardinality grows

exponentially with each attribute’s cardinality, which itself may be large. Thus, we instead propose a predicate-agnostic index, which can support arbitrary and unbounded predicate sets.

In this paper, we propose ACORN (ANN Constraint-Optimized Retrieval Network), a novel approach for performant and predicateagnostic hybrid search that can serve high-cardinality and unbounded predicate sets. We propose two indices: $A C O R N { - } \gamma$ , designed for high-efficiency search, and ACORN-1, designed for low construction overhead in resource-constrained settings. Both methods modify the hierarchical navigable small world (HNSW) index, a state-of-the-art graph-based index for ANN search, and are easy to implement in existing HNSW libraries.

ACORN tackles both the performance limitations of pre- and post- filtering, as well as the semantic limitations of specialized indices. ACORN proposes the idea of predicate subgraph traversal during search. As the name implies, the search strategy traverses the subgraph of the ACORN index induced by the set of nodes that pass the query predicate. ACORN designs the index such that these arbitrary predicate subgraphs approximate an HNSW index. Unlike pre- and post-filtering, this allows ACORN to provide sublinear retrieval times despite variance in correlation between query vectors and predicates, which we find to be a major challenge for existing hybrid search systems. ACORN also serves wide-ranging predicate sets by employing a predicate-agnostic construction that alters HNSW’s algorithm to create a denser graph. Specifically, we introduce a predicate-agnostic neighbor expansion strategy in ACORN- $\gamma$ based on target predicate selectivity thresholds, which can be estimated empirically with or without knowing the predicate set. In conjunction, we propose a predicate-agnostic compression heuristic to efficiently manage the index space footprint while maintaining efficient search. We also explore the trade-off space between search performance and construction overhead, designing ACORN-1 to approximate ACORN-??’s search performance while further reducing the time-to-index (TTI) for resource-constrained settings.

We systematically evaluate ACORN- $\gamma$ and ACORN-1 on four datasets: SIFT1M [35], Paper [63], LAION [55], and TripClick [54]. Our evaluation includes both prior benchmark datasets, with simple, low-cardinality predicate sets, which prior specialized indices can serve, as well as more complex datasets with millions of possible predicates, which existing indices cannot to handle. On each, ACORN- $\gamma$ achieves state-of-the-art hybrid search performance with $2 { - } 1 0 0 0 \times$ higher queries per second (QPS) at 0.9 recall compared to prior methods. Specifically, ACORN achieves $2 \mathrm { - } 1 0 \times$ higher QPS on prior benchmarks, over $3 0 \times$ higher QPS on new benchmarks, and over $1 { , } 0 0 0 \mathrm { x }$ higher QPS at scale, on a 25-million-vector dataset. We find that ACORN-1 empirically approximates ACORN- $\gamma$ , attaining at most $5 \times$ lower QPS at fixed recall but $9 - 5 3 \times$ lower TTI compared to ACORN- $\gamma$ . Our detailed evaluation demonstrates the effectiveness of ACORN’s predicate-subgraph traversal strategy and predicate-agnostic construction techniques.

# 2 BACKGROUND

Existing methods for Approximate Nearest Neighbor (ANN) search can be broadly categorized as tree-based [15–17, 19, 28, 45, 50, 56], hashing-based [9–11, 24, 26, 29, 30, 40, 41, 44, 46, 52, 59, 69], quantization-based [23, 27, 34, 35, 39], and graph-based [22, 25, 32,

![](images/59fdd41cde59ef34b7fa051862f681281cf0af0741fcf37099bdb114dc34dd87.jpg)  
Figure 1: Schematic drawing of search over an HNSW index. The search path is shown by blue arrows, beginning on level 2 and ending on level 0 at the query point, shown in green.

47, 48, 58, 68]. In this work build on HNSW, a graph-based method that is empirically one of the best-performing on high-dimensional datasets, and we adapt it to support hybrid search.

Graph-based ANN methods have gained popularity due their state-of-the-art performance on varied ANN benchmarks [13, 57]. These methods typically perform search using a greedy routing strategy that traverses a graph index, starting from a pre-defined entry point. The index itself forms a proximity graph $G ( V , E )$ , such that each dataset point is represented by a vertex and edges connect nearby points. The index construction algorithm typically aims to approximate subgraphs of the Delaunay graph [38]. While the Delaunay graph guarantees convergence of a greedy routing algorithm, it is impossible to efficiently construct for arbitrary metric spaces [51]. Thus graph methods focus on more tractable approximations of Delaunay subgraphs, such as the Relative Neighbor Graph (RNG) [37, 60], and the Nearest-Neighbor Graph (NNG) [8, 20].

# 2.1 Hierarchical Navigable Small Worlds

As Figure 1 illustrates, HNSW forms a hierarchical, multi-level graph index with bounded degree. Below we briefly summarize the HNSW search and construction algorithm.

The HNSW construction algorithm iteratively inserts each point into the graph index, to construct a navigable graph with bounded degree, specified by parameter $M$ . For each inserted element ??, first, a maximum layer index ?? is stochastically chosen using an exponentially decaying probability distribution, normalized by the constant $m _ { L } = 1 / l n ( M )$ . The level assignment probability ensures that the expected characteristic path length increases with the level index. Intuitively, the upper-most level contains the longestrange links, which will be traversed first by the search algorithm, and the bottom-most level contains the shortest-range links, which are traversed last by the search algorithm. The insertion procedure then proceeds in two phases. In the first phase, a greedy search is performed iteratively from the top layer, beginning at a pre-defined entrypoint down to the $( l + 1 ) \mathrm { t h }$ layer. At each of these levels, the greedy subroutine chooses a single node that becomes the entrypoint into the next layer. In the second phase, the greedy search iterates over level ?? to level 0. The greedy search at each level now chooses efc nodes as candidate edges. Of these candidates, at most ?? are selected to become neighbors of ?? according to an RNG-based pruning algorithm [31]. At level 0, the degree bound is increased $2 \times M$ , which is shown to empirically improve performance.

The HNSW search algorithm begins its traversal from a predefined entry point at the upper-most layer of the multilayer graph,

$\mathbf { \Delta A l g o r i t h m 1 : H N S W - A N N - S E A R C H } ( x _ { q } , K , e f s )$   
Input: query vector $x_{q}$ , number of nearest neighbors to return $K$ size of dynamic candidate list efs   
Output: $K$ nearest elements to $x_{q}$ $e\gets$ entry-point to hnsw graph $W\gets \emptyset / /$ set of current nearest $L\gets$ level(e) // Top hnsw level   
for $l\gets L\dots 1$ do   
| $e\gets$ SEARCH-LAYER(xq,e,ef=1,l)   
end $W\gets$ SEARCH-LAYER(xq,e,ef=efs,l=0)   
return $K$ nearest elements from W to $x_{q}$

illustrated in Figure 1. The traversal then follows an iterative search strategy from the top level downwards. At each level a greedy search is used to choose a single node, which becomes the entrypoint into the next level. Once the bottom level is reached, rather than greedily choosing a single node, the search algorithm greedily chooses $K$ nearest elements to return. We outline this process in Algorithm 1. The search parameter efs provides a tradeoff between search quality and efficiency by controlling the size of the dynamic candidate list stored during the bottom level’s greedy search.

# 3 PROBLEM DEFINITION AND CHALLENGES

In this section we formally define the hybrid search setting and then analyze the performance challenges that existing predicate-agnostic methods, i.e., pre- and post-filtering, face. Our analysis leads us to explore several important workload characteristics. Specifically, we will consider predicate selectivity, the dataset size, and query correlation, which we introduce, formally define and find to be a major challenge for post-filtering methods.

We will later leverage our understanding of existing performance challenges in Section 4 to formulate a theoretically ideal hybrid search solution. Then, in Section 7, we will revisit the workload characteristics discussed in this section to rigorously evaluate ACORN’s search performances.

# 3.1 Hybrid Search Definitions

Let $D = \{ e _ { 1 } , e _ { 2 } , . . . , e _ { n } \} = \{ ( x _ { 1 } , a _ { 1 } ) , ( x _ { 2 } , a _ { 2 } ) , . . . , ( x _ { n } , a _ { n } ) \}$ be a dataset consisting of $n$ entities, each with a vector component, $x _ { i } \in \mathbb { R } ^ { d }$ , and a structured attribute-tuple, $a _ { i }$ , associated with entity $e _ { i }$ . Let $X = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { n } \}$ denote the set of vectors in the dataset, and $d i s t ( a , b )$ is the metric distance between any two points. Let $A =$ $\{ a _ { 1 } , a _ { 2 } , . . . , a _ { n } \}$ be the set of structured attributes in the dataset. We will denote $X _ { p } ~ \subseteq ~ X$ as the subset of vectors corresponding to entities in the dataset that pass a given predicate $\mathcal { P }$ . We refer to the selectivity (??) of predicate $\mathcal { P }$ as the fraction of entities from $D$ that satisfy the predicate, where $0 \leq s \leq 1$ .

We consider the hybrid search problem, described as follows. Given a dataset $D$ , target $K$ , and query $\boldsymbol { q } = ( x _ { q } , p _ { q } )$ , where $x _ { q } \in$ $\mathbb { R } ^ { d }$ , and $\mathit { p } _ { q }$ is a predicate, retrieve $x _ { q }$ ’s $K$ nearest neighbors that pass the predicate $\mathit { p } _ { q }$ . We will specifically focus on the problem of approximate nearest neighbor search w.r.t $x _ { q }$ . Here, our goal is to maximize both search accuracy and search emeasure accuracy by recall@?? = ??∩???? , where $G$ iency. We willis the ground

![](images/4ba09844101c7f9cf90a17869301a6835525421ba2df5fa4bab0eec476cab2be.jpg)  
Figure 2: Schematic drawing of a dataset with no predicate clustering (top), a dataset with predicate clustering and positive query correlation (middle), and a dataset with predicate clustering and negative query correlation (bottom). Dark blue circles show points that pass the predicate, and light gray circles show points that fail the predicate. The query vectors are shown in green.

truth set of $K$ nearest neighbors to $x _ { q }$ that satisfy $\hbar q$ , and $R$ is the retrieved set.

# 3.2 Search Performance of Baseline Methods

We now analyze the search complexity of two predominant baseline methods, pre- and post-filtering. We will consider how varied workload characteristics impact the search behavior of these methods. Through our analysis, we will make the standard assumption that distance computations dominate search performance. We note that HNSW’s unfiltered search complexity is $O ( \log ( n ) + K )$ .

Pre-filtering linearly scans $X _ { p }$ , computing distances over each point that passes the search predicate. This yields a hybrid search complexity of $O ( | X _ { p } | ) = O ( s n + K )$ . While pre-filtering always achieves perfect recall, its search complexity scales poorly for large dataset sizes or selectivities, growing linearly in either variable.

Post-filtering, by contrast, performs ANN-search over $X$ to find the closet query vector to $x _ { q }$ , then expands the search scope to find $K$ vectors that pass the query predicate, $\boldsymbol { p }$ . Intuitively, search performance varies depending on correlation between the query vector and the vectors in $X _ { p }$ . When the vectors of $X _ { p }$ are close to the query vector, post-filtering over HNSW has a search complexity of $O ( \log ( n ) + K )$ . If the vectors in $X _ { p }$ are uniformly distributed within $X$ , then post-filtering’s expected search complexity is $O ( \log ( n ) +$ $K / s )$ . However, vectors of $X _ { p }$ may be far away from the query vector, leading to a worst case of $O ( n )$ search performance.

We see that the search performance of either baseline is not robust to variations in selectivity, dataset size, and query correlation. We empirically verify these limitations in section 7 (figures 9, 10).

3.2.1 Formalizing Query Correlation. We will now formalize the notion of query correlation, which we find is key challenge for post-filtering-based systems. As Figure 2 shows, query correlation occurs when the vectors of $X _ { p }$ are non-uniformly distributed in $X$ and instead cluster together relative to the vectors in X. We refer to this phenomenon as predicate clustering. When predicate clustering occurs, a query vector may be either close or far away from the predicate cluster containing its search targets, inducing query correlation.

Definition: Query Correlation. We will consider the queryto-target distances for the given dataset compared to the expected query-to-target distances for a hypothetical dataset, under which no clustering is present. Formally, we define the query correlation of the hybrid search workload $Q$ over dataset $D$ as:

$$
C (D, Q) = \mathbb {E} _ {\left(x _ {i}, p _ {i}\right) \in Q} \left[ \mathbb {E} _ {R _ {i}} \left[ g \left(x _ {i}, R _ {i}\right) \right] - g \left(x _ {i}, X _ {p _ {i}}\right) \right]
$$

We let $R _ { i }$ be a random set variable of $| X _ { p _ { i } } |$ vectors drawn uniformly from $X$ , defined for each hybrid query $( x _ { i } , p _ { i } ) \ \in \ Q$ . We define $g ( x , S ) = \mathrm { m i n } _ { y \in S } d i s t ( x , y )$ to be the function mapping the query vector $x$ to the minimum distance of neighbors from the given vector set $S \subseteq R ^ { d }$ . Note that $g ( x _ { i } , X _ { p _ { i } } )$ is the ground-truth hybrid-search target of the query $( x _ { i } , p _ { i } )$ .

If, on average, query vectors are closer to their targets in $X _ { p _ { i } }$ , the true dataset of hybrid search targets, than in $R _ { i }$ , the no-clustering dataset, then the workload has positive query correlation. If the reverse is true, the workload has negative query correlation. We may also consider nearest-neighbor distance rather than the metric distance in the above definition. We also note that we can easily extend this definition to consider $K$ targets of the hybrid search, rather than one, by summing distances over the $K$ search targets.

# 4 THEORETICAL IDEAL HYBRID SEARCH PERFORMANCE WITH HNSW

For a given hybrid search query, we define the theoretically ideal search performance using HNSW data structures as the performance attainable if we knew the search predicate $\mathit { p } _ { q }$ during construction. In this case, we could construct an HNSW index over $X _ { p }$ . We call this the oracle partition index for that query. The complexity of searching this index is $O s ( \log s n + K )$ . Notably, the search performance of the oracle partition index outperforms both pre- and post-filtering across variations in predicate selectivity, data size, and query correlation. While pre-filtering’s search scales in $| X _ { p } |$ , search over the oracle partition scales sublinearly in $| X _ { p } |$ . The oracle partition is also robust under variations in query correlation: it does not require the search scope expansion used in post-filtering.

Despite its ideal search performance, the oracle partition index requires us to know all search predicates in advance and to create a full HNSW index per predicate. In practice, the oracle partition index is not possible to construct because query predicate sets are often unknown during construction and have high or unbounded cardinality. Building an HNSW per predicate would require prohibitive amounts of space and time. Thus, in this work, we will instead approximate search over the oracle partition index for a particular query, without ever explicitly constructing this index.

# 5 ACORN OVERVIEW

We now describe ACORN, a predicate-agnostic approach for stateof-the-art hybrid search. We propose two variants, which we refer to as ACORN- $\gamma$ (5.1, 5.2) and ACORN-1 (5.3). We design ACORN- $\gamma$ to achieve efficient search performance, and we design ACORN-1 to approximate ACORN-??’s search performance while further reducing the algorithm’s time to index (TTI) and space footprint for resource-constrained settings.

ACORN’s core idea is to search over the index’s predicate subgraph, i.e., the subgraph induced by $X _ { p }$ for a given search predicate

![](images/477e2635a5ff628d115ca1d433fafeb6cd98d60bb68268a3fd51d3bf98022710.jpg)  
Figure 3: An illustration of the predicate subgraph, shown by the green nodes. ACORN searches over the predicate subraph to emulate search over an oracle partition index.

Table 1: Summary of Notation   

<table><tr><td>Symbol</td><td>Description</td></tr><tr><td>γ</td><td>neighbor expansion factor for ACORN index</td></tr><tr><td>Mβ</td><td>compression parameter for ACORN index</td></tr><tr><td>ef</td><td>size of dynamic candidate list in ACORN greedy search</td></tr><tr><td>M</td><td>degree bound for traversed nodes during ACORN search</td></tr><tr><td>mL=1/ln M</td><td>level normalization constant for ACORN index</td></tr><tr><td>e</td><td>entry-point to ACORN index</td></tr><tr><td>ep</td><td>entry-point to ACORN&#x27;s predicate p&#x27;s subgraph</td></tr><tr><td>l(v)</td><td>maximum level index of node v in ACORN index</td></tr><tr><td>Nl(v)</td><td>neighbor list of node v at level l</td></tr><tr><td>Nl(p)(v)</td><td>filtered neighbors of node v at level l under predicate p</td></tr><tr><td>Xp</td><td>vector dataset that passes predicate p</td></tr><tr><td>s</td><td>selectivity</td></tr><tr><td>n</td><td>size of dataset</td></tr></table>

$\boldsymbol { p }$ , as shown in Figure 3. We modify the HNSW construction algorithms so that arbitrary predicate subgraphs emulate an HNSW oracle partition index without the need to explicitly construct one. ACORN- $\gamma$ achieves this by constructing a denser version of HNSW, which we parameterize by a neighbor list expansion factor, ??, a compression factor, $M _ { \beta }$ , and the HNSW parameters, efc and ??. Then by adding a filter step during search to ignore neighbors that fail the predicate, we find ACORN-??’s search can efficiently navigate to and traverse over the predicate subgraph, even under variations in query correlation. Meanwhile, ACORN-1 expands neighbor lists during search rather than during construction to approximate ACORN-??’s dense graph structure without building it.1

Overall, ACORN prescribes a simple and general framework for performant hybrid search based on the idea of predicate-subgraph traversal. The core techniques we propose are predicate-agnostic neighbor-list expansions and pruning during construction in combination with predicate-based filtering during search. While this framework can be applied to a variety of graph-based ANN indices, in this work we focus on HNSW due their state-of-the-art performance and widespread use.

# 5.1 ACORN- $\gamma$ Search Algorithm

Algorithm 2 outlines the greedy search algorithm ACORN uses at each level, beginning from the top level at a pre-defined entrypoint. The main difference between ACORN’s search algorithm and that of HNSW is how neighbor look-ups (line 9) are performed at

![](images/a176374fa5eafb7f160a23cff12031758b3058e91cfec1fe633851bd77750d1e.jpg)  
Figure 4: Diagram of ACORN’s neighbor selection strategies. Blue nodes represent neighbors that pass the query predicate. Sub-figure (a) shows the simple predicate-based filter applied to uncompressed edge lists of size $M \cdot \gamma ,$ followed by truncation to size $M = 3$ . Sub-figure (b) shows the compression-based heuristic. Sub-figure (c) shows the neighbor expansion strategy used in ACORN-1.

$\overline { { \mathrm { \ A l g o r i t h m 2 : \ A C O R N - S E A R C H - L A Y E R } ( x _ { q } , p _ { q } , e , e f , l ) } }$   
Input: query vector $x_{q}$ query predicate $p q$ ,entry-point $e$ number of nearest neighbors to return ef, level to search l Output: ef nearest elements to $x_{q}$ 1 $T\gets e / /$ visited set   
2 $C\gets e / /$ candidate set   
3 W $\leftarrow$ e// dynamic list of found nearest neighbors   
4 while $|C|\ge 0$ do   
5 $c\gets$ extract arg $\min_{x\in C}\| x_q - x\|$ 6 $f\gets$ get arg $\max_{x\in W}\| x_q - x\|$ 7 if dist(c,xq)>dist(f,xq) and $|W|\ge$ efc break   
8 neighborhood $\leftarrow$ GET-NEIGHBORS(c,l,pq)   
for each v E neighborhood[1:M]   
11 if v T   
12 $T\gets T\cup v$ 13 $f\gets \arg \max_{x\in W}\| x_q - x\|$ 14 if dist(v,xq)<dist(f,xq) or $|W| <   ef$ C C U v W W U v if $|W| > ef$ remove furthest element from W to $x_{q}$ 16   
17   
18

each visited node, ??. While HNSW simply checks the neighbor list, $N ^ { l } ( c )$ , ACORN performs additional steps to recover an appropriate neighborhood for the given search predicates.

Specifically, ACORN- $\gamma$ uses two neighbor look-up strategies, a simple filter method, shown in Figure 4(a), and a compression-based heuristic, shown in Figure 4(b), which is compatible with the compression strategy we optionally apply during construction, detailed in Section 5.2. For each visited node, ??, the filter-based neighbor look-ups simply scan the neighbor list $N ^ { l } ( \boldsymbol { v } )$ to find the sub-list of neighbors that pass the predicate, $N _ { \mathscr { p } } ^ { l } ( \upsilon )$ . If $N _ { \mathscr { p } } ^ { l } ( v )$ contains more than $M$ nodes, we take the first $M$ and return this as ??’s neighborhood. The compression-based neighbor look-ups instead partially expand the neighbor set $N ^ { l } ( \boldsymbol { v } )$ to include a subset of ??’s two-hop

neighbors, before performing filtering and truncation. This procedure entails two phases. The first phase iterates through the first $M _ { \beta }$ nodes of $N ^ { l } ( v )$ , simply filtering as in the previous strategy. The second phase iterates over the remainder of the neighbor list, expanding the search neighborhood to include neighbors of neighbors, before again filtering according to the query predicate. $M _ { \beta }$ is a construction parameter which we will discuss in the next section.

# 5.2 ACORN- $\gamma$ Construction Algorithm

We construct the ACORN- $\gamma$ index by applying two core modifications to the HNSW indexing algorithm: first, we expand each node’s neighbor list, and then we apply a novel predicate-agnostic pruning method to compress the index. Both of these steps are summarized in Figure 5.

Neighbor List Expansion. While HNSW collects $M$ approximate nearest neighbors as candidate edges for each node in the index, ACORN collects $M \cdot \gamma$ approximate nearest neighbors as candidate edges per node. To find these candidates during construction, ACORN uses a metadata-agnostic search over its graph index. Specifically, the neighbor lookup strategy at each node, ??, on level ??, simply accesses the neighbor list $N ^ { l } ( v )$ and returns the first $M$ nodes. Note that although each node contains up to $M \cdot \gamma$ neighbors, we assume by construction that $M$ neighbors per node are sufficient for maintaining navigability of the graph index. Thus, considering truncated neighbor lists while traversing the graph allows us to avoid unnecessary distance computations and TTI slowdowns.

One simple choice for $\gamma$ is $\frac { 1 } { s _ { m i n } }$ , where $s _ { m i n }$ is the minimum pred-?????? icate selectivity we plan to serve before resorting to pre-filtering. As we discuss in Section 6, ACORN’s indexing time and space footprint increase proportionally to ??. Meanwhile, pre-filtering becomes a competitive baseline at low predicate selectivity values, as we show in Figure 9a. Thus, ACORN is able to balance construction and search efficiency by using pre-filtering as a fall-back for queries with low-selectivity predicates. This leads to a simple cost-based model during search: if the estimated predicate selectivity of a given query is greater than $1 / \gamma _ { ; }$ , search the ACORN- $\gamma$ index, otherwise pre-filter. We note that leveraging pre-filtering in this way may degrade search efficiency, but not result quality, when errors occur in selectivity estimates. If a query’s true predicate selectivity is above $1 / \gamma$ , but the estimate is below, the system will mistakenly pre-filter,

![](images/0c920038d5626c7717aaab09ac57a6da5d6ab42403e7357a2b8a068f85a09eac.jpg)

![](images/045a7ccaf7265b353b3f8f1a0d7e99f748f6ef7884467f89f57784fd320ce2ad.jpg)  
Figure 5: A comparison of HNSW and ACORN-??’s strategies for (a) selecting candidate edges, shown for $M _ { = 3 }$ , and (b) pruning candidate edges for each inserted node $v$ , shown for $M _ { = 3 }$ , $M _ { \beta } { = } 2$ , $\gamma = 2$ .

obtaining perfect recall at possibly lower QPS than if the ACORN index was instead searched. If the reverse is true, the system will mistakenly search the ACORN index, whereas pre-filtering would have offered similar QPS and perfect recall.

Compression. A key challenge with ACORN- $\gamma$ s neighbor expansion step is that it increases index size and TTI. The increased index size poses a significant issue particularly for memory-resident graph indices, like HNSW. To address this, we introduce a predicateagnostic pruning technique. While we could apply compression to the full index, as discussed in Section 6.1, we specifically target the bottom level’s neighbor lists since they contribute most significantly to the indexing overhead. This follows from the exponentially decaying level assignment probability ACORN uses.

The core idea of the pruning procedure is to precisely retain each node’s nearby neighbors in the index, while approximating farther away neighbor during search. We use the tunable compression parameter, $M _ { \beta }$ , where $0 \leq M _ { \beta } , \leq M \cdot \gamma .$ . During construction, ACORN chooses each node’s final neighbor list by automatically retaining the nearest $M _ { \beta }$ candidate edges and aggressively pruning the remaining candidates. During search we can recover the first $M _ { \beta }$ neighbors of each node ?? directly from the neighbor list $N ^ { l } ( \boldsymbol { v } )$ , and approximate remaining neighbors by looking at 2-hop neighbors during search, as we described in Section 5.1.

Figure 5 outlines this pruning procedure applied to node ??’s candidate neighbor list. The algorithm iterates over the ordered candidate edge list and keeps the first $M _ { \beta }$ candidates. Over the remaining sub-list of candidates, the algorithm applies the following pruning procedure at each node. Let H be the dynamic set of ??’s chosen two-hop neighbors, initialized to ∅. We prune candidate ?? if it is contained in $H$ ; otherwise, we keep ?? and add all of its neighbors to $H$ . The pruning procedure stops after iterating over all candidates, or if $| H |$ plus the number of chosen edges exceeds $M \cdot \gamma$ . The pruned and ordered neighbor list is then stored in the ACORN index and H is discarded.

We highlight that the neighbor expansion during search, described in Section 5.1, can recover pruned neighbors regardless of the query predicate. It follows from ACORN’s pruning rule that any node $x$ that was pruned from some node ??’s neighbor list, $N ^ { l } ( \boldsymbol { v } )$ , must be in the neighbor list $N ^ { l } ( y )$ such that $y$ is a neighbor of ?? with index greater than $M _ { \beta }$ . During search, the neighbor lookup at ?? on level ?? will perform a neighbor-list expansion for all neighbors with an index greater than $M _ { \beta }$ , thus checking $N ^ { l } ( y )$ and finding $x$ .

We now briefly describe why HNSW’s pruning, a metadatablind mechanism, is insufficient for hybrid search. Consider the simple scenario shown in Figure 5. For a node, ??, inserted into the HNSW index at an arbitrary level ??, the algorithm generates candidates neighbors $a , b$ and ??. HNSW’s pruning rule iterates over ??’s candidate neighbor list in order of nearest to farthest neighbors. Node $^ { b }$ is pruned since there exists a neighbor ?? such that $^ { b }$ is closer to ?? than to ??. This RNG-approximation strategy corresponds to pruning the longest edge of the triangle formed by a triplet ??, ??, ??. In this case, we can prune the edge $v - b$ and expect a search path to traverse from ?? to $^ { b }$ via ??. The problem with this technique arises when we consider the hybrid search setting for an arbitrary predicate. Say ?? and $^ { b }$ pass a given query predicate $\mathit { p } _ { q }$ , but $a$ does not. Then ??, ??, ?? do not form a triangle in the predicate subgraph, and we cannot expect to find the path from ?? to $^ { b }$ through ??. As a result, HNSW’s pruning mechanism will falsely prune edge $v - b$ If we had complete knowledge of all possible query predicates, we could ensure that we only prune edges of triangles such that all three vertices always exist in the same subset of possible predicate subgraphs. FilteredDiskANN [25] takes this approach by restricting the set of possible query predicates. However, for arbitrary query predicates, ensuring this property holds becomes intractable.

# 5.3 ACORN-1

We now describe ACORN-1, an alternative approach which aims to approximate ACORN- $\cdot \gamma$ ’s search performance, while further minimizing index size and TTI. ACORN-1 achieves this by performing the neighbor expansion step solely during search, rather than during construction, as ACORN-?? does. ACORN-1’s construction corresponds to the original HNSW index without pruning. This construction corresponds to ACORN-??’s construction algorithm, with fixed parameters $\gamma = 1$ and $M _ { \beta } = M$ .

ACORN-1’s main difference from ACORN-?? during search, is its neighbor lookup strategy. Specifically, at each visited node, ??, during greedy search, ACORN-1 uses a full neighbor list expansion to consider all one-hop and two-hop neighbors of ??, before applying the predicate filter and truncating the resulting neighbor list to size ??. Figure 4(c) outlines this procedure.

# 6 DISCUSSION

In this section we analyze the ACORN index’s space complexity, construction complexity and search performance. We focus our attention on ACORN-??, since ACORN-1’s index construction represents a special case of ACORN-?? for fixed parameters $( \gamma = 1$ , $M _ { \beta } = M )$ , and we empirically show that ACORN-1 search approximates ACORN- $\gamma$ in Section 7. We note that our analysis in Sections 6.2 and 6.3 considers the complexity scaling of the search procedure under the assumption that we build the exact Delaunay graphs rather than approximate ones.

# 6.1 Index Size

The average memory consumption per node of the ACORN-?? index is $O ( M _ { \beta } + M + m _ { L } \cdot M \cdot \gamma )$ , assuming the number of bytes per edge is constant. For comparison, average memory consumption per node for the HNSW index scales $O ( M + m _ { L } \cdot M )$ . Overall, ACORN- $\gamma$ increases the bottom-level’s memory consumption by $O ( M _ { \beta } )$ per

node, and increases the higher levels memory consumption by a factor of $\gamma$ per node.

To understand ACORN’s memory consumption we evaluate the average number of neighbors stored per node. At level 0, compression is applied to the candidate edge lists of size $M \ast \gamma$ resulting in neighbor sets of length $M _ { \beta }$ plus a compressed set which scales $O ( M )$ . We show this empirically in figure 12. On higher levels, nodes have at most $M \ast \gamma$ edges. We multiply this by the average number of levels that an element is added to, given by $\begin{array} { r } { \mathbb { E } [ l + 1 ] = E [ - \ln ( \mathrm { u n i f } ( 0 , 1 ) ) * m _ { L } ] = m _ { L } + 1 . } \end{array}$ .

While we specifically target compression to level 0 in this work, because it uses the most space, compression could be applied to more levels in bottom-up order to further reduce the index size for large datasets. Denoting ???? as the chosen number of compressed levels, the average memory consumption per node in this generalized case is $O ( n c ( M _ { \beta } + M ) + ( m _ { L } - n c ) ( M \cdot \gamma )$ .

# 6.2 Construction Complexity

For fixed parameters $M , M _ { \beta }$ and efc, ACORN-??’s overall expected construction complexity scales $O ( n \cdot \gamma \cdot \log ( n ) \cdot \log ( \gamma ) )$ . Compared to HNSW, which has $O ( n \cdot \log ( n ) )$ expected construction complexity, ACORN- $\gamma$ increases TTI by a factor of $\gamma { \cdot } \mathrm { l o g } ( \gamma ) )$ due to the expanded edge lists it generates.

We now describe ACORN’s construction complexity in detail by decomposing it into the following three factors (i) the number of nodes in the dataset, given by $n$ (ii) the expected number of levels searched to insert each node into the index, and (iii) the expected complexity of searching on each level. By design, ACORN’s expected maximum level index scales $O ( \log n )$ according to its levelassignment probability, which is the same as HNSW. This provides our bound on (ii).

Turning our attention to (iii), we will first consider the length of the search path and then consider the computation cost incurred at each visited node. For the HNSW level probability assignment, it is known that the expected greedy search path length is bounded by a constant $\begin{array} { r } { S = \frac { \mathrm { ~ \bar { ~ } 1 ~ } } { 1 - \exp \left( - m _ { L } \right) } } \end{array}$ [48]. We can bound ACORN’s expected search path length by $O ( \gamma )$ since the path reaches a greedy minima in a constant number of steps and proceeds to expand the search scope by at most ??·?? nodes to collect up to ??·?? candidate neighbors during construction.

The computation complexity at each visited node along the search path is $O ( \log ( \gamma ) )$ , seen as follows. For each node visited, we first check its neighbor list to find at most ?? un-visited nodes, on which we perform distance computations in $O ( M \cdot d )$ time. Then, we update the sorted lists of candidate nodes and results in $O ( M \cdot d \cdot l o g ( \gamma \cdot M ) )$ time. Treating $M$ and $\gamma$ as constants, we see that at each visited node the computation complexity is $O ( \log \gamma )$ and for greedy search at each level, the complexity is $O ( \gamma \cdot \log ( \gamma ) )$ . Multiplying by $n \cdot \log ( n )$ yields ACORN’s final expected construction complexity, $O ( n \cdot \gamma \cdot \log ( n ) \cdot \log ( \gamma )$ .

# 6.3 Search Analysis

Turning our attention to ACORN-??’s search algorithm, we will first point out several properties of HNSW that ACORN’s predicate subgraphs aim to emulate. In Figure 7 we empirically show that ACORN’s search performance approximates that of the HNSW

oracle partition index. We will then describe ACORN’s expected search complexity. We define $l : X \to \mathbb { N }$ to be the mapping of nodes to there maximum level index in ACORN-??.

6.3.1 Index and Search Properties. Intuitively, for a given query, ACORN’s predicate subgraph will emulate the HNSW oracle partition index when the predicate subgraph forms a hierarchical structure, each node in the subgraph has degree close to $M$ , the subgraph has a fixed entrypoint at its maximum level index that we can efficiently find during search, and the subgraph is connected. We will examine each of these properties separately and consider when they hold. We also note one main difference between ACORN’s predicate subgraphs and HNSW that arises due to ACORN’s predicateagnostic pruning: each level of ACORN approximates a KNN graph, while each level of HNSW approximates a RNG graph. While this difference does not affect ACORN’s expected search complexity in Section 6.3.2, Malkov et al. [48] demonstrated that the RNG-based pruning empirically improves performance.

Hierarchy. First, we observe that the arbitrary predicate subgraph $G ( X _ { \mathfrak { p } } )$ forms a controllable hierarchy similar to the HNSW oracle partition index built over $X _ { p }$ with parameter $M$ . This is by design. ACORN- $\gamma$ ’s construction fixes $M$ , and consequently $m _ { L }$ , the level normalization constant. As a result, nodes of $X _ { p }$ in the ACORN- $\gamma$ index are sampled at rates equal to the level probabilities of the HNSW partition. Ensuring this level sampling holds allows us to bound the expected greedy search path length at each level by a constant, $S$ , as Malkov et al. [48] previously show.

Bounded Degree. Next, we will describe degree bounds, an important factor that impacts greedy search efficiency and convergence. While HNSW upper bounds the degree of each node by M during construction, ACORN- $\gamma$ enforces this upper bound during search. This ensures ACORN’s search performs a constant number of distance computations per visited node. We now focus our attention on lower bounding the degree of nodes visited during ACORN- $\cdot \gamma$ ’s search over the predicate subgraph.

If a node in the predicate subgraph has degree much lower than ??, this could adversely impact the search convergence and thus recall. For a dataset and query predicate that exhibit no predicate clustering, for any node $v$ in $G ( X _ { \mathfrak { p } } )$ ,

$$
\mathbb {E} \left[ \left| N _ {p} ^ {l} (v) \right| \right] = \left| N ^ {l} (v) \right| \cdot s = \gamma \cdot M \cdot s > M, \forall s > s _ {\min }
$$

This also holds as a lower bounds for datasets with predicate clustering, in which case $P r ( x \in N _ { p } ^ { l } ( v ) ) > s , \forall x \in N ^ { l } \bar { ( } v )$ where ?? is a node in the predicate cluster. Thus we will continue our lower bound analysis of node degrees under the worst case assumption of no predicate clustering. Using the binomial concentration inequality with parameter ??, and union-bounding over the expected search path length, we show that for the search path $\mathcal { P } = v _ { 1 } - . . . - v _ { y }$ over an arbitrary predicate subgraph:

$$
P r \Big [ \bigcup_ {v \in \mathcal {P}} \big (| N _ {p} (v) | \leq (1 - \delta) M \big) \Big ] \leq O \big (\log n \cdot \exp (- \delta^ {2} \gamma M s / 2) \big)
$$

We also analyze the probability that the subgraph traversal gets disconnected, which we bound by:

$$
P r \left[ \bigcup_ {v \in \mathcal {P}} \left(| N _ {p} (v) | \leq 0\right) \right] \leq O \left(\log n \cdot (1 - s) ^ {M \cdot \gamma}\right)
$$

We see that both bounds decay exponentially in ??.

Fixed Entry-point. Similar to HNSW, ACORN’s search begins from a fixed entry-point, chosen during construction. This predefined entry-point provides a simple and effective strategy that is also predicate-independent and robust to variations in query correlation, as we empirically show in Figure 10.

Intuitively, we expect the search to successfully navigate from ACORN’s fixed entry-point, ??, to the predicate-subgraph entrypoint, $e _ { p }$ , when we find a node that passes the predicate on an upper level of the index that is fully connected. In this case, there will exist a one-hop path from ?? to $e _ { p }$ . We consider $e _ { p }$ to be an arbitrary node that passes a given predicate $\boldsymbol { p }$ and is on the maximum level of the predicate subgraph. The index’s neighbor expansion parameter, ??, causes the index’s upper levels to be denser and, specifically those with less than $M \cdot \gamma$ nodes, to be fully connected. When these fully connected levels contain at least one node that passes the predicate, the search is guaranteed to route from ?? to $e _ { p }$ . Since ACORN samples all nodes with equal probability at each level, the probability that nodes passing a given predicate, $\boldsymbol { p }$ , exist on some level is proportional to the predicate’s selectivity, which takes a lower bound of $s _ { m i n } = 1 / \gamma$ .

Connectivity. We note that neither HNSW nor ACORN provides theoretical guarantees on connectivity over its level graphs for arbitrary datasets. Thus we instead rely primarily on empirical results for our analysis. However, for some cases, we can expect ACORN’s predicate subgraph to be connected when the HNSW oracle partition is connected. Two such cases are when $X _ { p }$ exhibits no predicate clustering, or $X _ { p }$ is clustered around a single region. In either case, each node has an expected degree of at least $M$ and each level approximates a KNN graph, which is connected when $K > > \log n$ . We empirically show in Figure 13a that ACORN’s predicate subgraphs exhibit connectivity for real datasets and hybrid search queries. To analyze potential connectivity problems, we recommend benchmarking ACORN’s hybrid search performance against HNSW’s ANN search performance using equivalent $M$ and efc parameters. If a significant gap in accuracy exists, we recommend incrementally increasing ?? from its its initial value of $1 / s _ { m i n }$ .

6.3.2 Search Complexity. ACORN- $\gamma ^ { \mathrm { i } }$ ’s expected search complexity scales:

$$
O \left(\left(d + \gamma\right) \cdot \log (s \cdot n) + \log (1 / s)\right)
$$

This approximates the HNSW oracle partition’s expected search complexity, $O ( d { \cdot } \log ( s { \cdot } n ) )$ . Intuitively, ACORN- $\gamma$ ’s search path performs some filtering at the upper levels before likely entering and traversing the predicate sub-graph, during which ACORN incurs a small overhead compared to HNSW search in order to perform the predicate filtering step over each neighbor list.

We derive ACORN- $\gamma$ ’s search complexity by considering two stages of its search traversal. In the first stage, search begins from a pre-defined entry-point ??, which need not pass the query predicate. In this stage, the search performs filtering only, dropping down each level on which the filtered neighbor list, $N _ { p } ( e )$ , is found to be empty. Once the traversal reaches the first node, $e _ { p }$ that passes the predicate, it enters the second stage, beginning its traversal over the predicate subgraph $G ( X _ { \mathfrak { p } } )$ .

In stage 1 the greedy search path on each layer has length 1, and occurs over $O ( \log n - \log ( s \cdot n ) )$ expected levels, yielding the

complexity $O ( \log ( 1 / s )$ . We see this because the expected maximum level index of the full ACORN index graph scales $O ( \log n )$ based on its level-assignment probability [48]. Meanwhile, the predicate subgraph $G ( X _ { p } )$ of size $s \cdot n$ has an expected maximum level index of $O ( \log ( s \cdot n ) )$ , once again according to its level sampling procedure.

The second stage of the search traverses the predicate subgraph in expected $O ( ( d + \gamma ) \cdot \log ( s \cdot n ) )$ complexity. As we previously describe, the expected maximum level index of the predicate subgraph scales $O ( \log ( s \cdot n ) )$ . At each level, the expected greedy path length can be bounded by a constant ?? due to the index level sampling procedure employed during construction. For each node visited along the greedy path, we perform distance computations in $O ( d )$ time on at most $M$ neighbors, and perform a constant-time predicate evaluations over at most $M \cdot \gamma$ neighbors.

# 7 EVALUATION

We evaluate ACORN through a series of experiments on real and synthetic datasets. Overall, our results show the following:

• ACORN- $\gamma$ achieves state-of-the-art hybrid search performance, outperforming existing methods by $2 \substack { - 1 , 0 0 0 \times }$ higher QPS at 0.9 recall on both prior benchmark datasets with simple, low-cardinality predicate sets, and more complex datasets with high-cardinality predicate sets. Specifically, ACORN achieves $2 \mathrm { - } 1 0 \times$ higher QPS on prior benchmarks, over $3 0 \times$ higher QPS on new benchmarks, and over $1 { , } 0 0 0 \mathrm { x }$ higher QPS at scale on a 25-million-vector dataset.   
• ACORN- $\gamma$ and ACORN-1 are predicate-agnostic methods, providing robust search performance under variations in predicate operators, predicate selectivity, query correlation, and dataset size.   
• ACORN-1 and ACORN-?? exhibit trade-offs between search performance and construction overhead. While ACORN-?? achieves up to $5 \times$ higher QPS than ACORN-1 at fixed recalls, ACORN-1 can be constructed with $9 \mathrm { - } 5 3 \mathrm { \times }$ lower time-to-index (TTI).

We now discuss our results in detail. We first describe the datasets (7.1) and baselines (7.2) we use. Then, we present a systematic evaluation of ACORN’s search peformance (7.3). Finally, we assess ACORN’s construction efficiency (7.4). We run all experiments on an AWS m5d.24xlarge instance with 370 GB of RAM, 96 vCPUs, and 196 threads.

# 7.1 Datasets

We conduct our experiments on two datasets with low-cardinality predicate sets (LCPS) and two datasets with high-cardinality predicate sets (HCPS). The LCPS datasets allow us to benchmark prior works that only support a constrained set of query predicates. The HCPS datasets consist of more complex and realistic query workloads, allowing us to more rigorously evaluate ACORN’s search performance. Table 2 provides a concise summary of all datasets.

Table 2: Datasets   

<table><tr><td rowspan="2"></td><td colspan="4">Base Data</td><td colspan="3">Query Workload</td></tr><tr><td># Vectors</td><td>Vector Dim</td><td>Vector Source Data</td><td>Structured Data</td><td>Predicate Operators</td><td>Avg. Query Selectivity</td><td>Predicate Cardinality</td></tr><tr><td>SIFT1M</td><td>1,000,000</td><td>128</td><td>images</td><td>random int.</td><td>equals(y)</td><td>0.083</td><td>12</td></tr><tr><td>Paper</td><td>2,029,997</td><td>200</td><td>passages</td><td>random int.</td><td>equals(y)</td><td>0.083</td><td>12</td></tr><tr><td>TripClick</td><td>1,055,976</td><td>768</td><td>passages</td><td>clinical area list &amp; publication date</td><td>contains(y1 ∨ y2 ∨ ...) &amp; between(y1, y2)</td><td>0.17, 0.362</td><td>&gt;108</td></tr><tr><td>LAION (1M)</td><td>1,000,448</td><td>512</td><td>images</td><td>text captions &amp; keyword list</td><td>regex-match(y) &amp; contains(y1 ∨ y2 ∨ ...)</td><td>0.056 - 0.133</td><td>&gt;1011</td></tr><tr><td>LAION (25M)</td><td>24,653,427</td><td>512</td><td>same as above</td><td>same as above</td><td>same as above</td><td>same as above</td><td>same as above</td></tr></table>

7.1.1 Datasets with Low Cardinality Predicate Sets. We use SIFT1M [35] and Paper [63], the two largest publically-available datasets used to evaluate recent specialized indices [25, 63]. For both datasets, we follow related works [25, 62, 63] to generate structured attributes and query predicates: for each base vector, we assign a random integer in the range 1 − 12 to represent structured attributes; and for each query vector, the associated query predicate performs an exact match with a randomly chosen integer in the attribute value domain. The resulting query predicate set has a cardinality of 12.

SIFT1M: The SIFT1M dataset was introduced by Jegou et al. in 2011 for ANN search. It consists of a collection of 1M base vectors, and 10K query vectors. All of the vectors are 128-dimensional local SIFT descriptors [43] from INRIA Holidays images [33].

Paper: Introduced by Wang et al. in 2022, the Paper dataset consists of about 2M base vectors and 10K query vectors. The dataset is generated by extracting and embedding the textual content from an in-house corpus of academic papers.

7.1.2 Datasets with High Cardinality Predicate Sets. We use the TripClick and LAION datasets in our experiments with HCPS datasets.

TripClick: The TripClick dataset, introduced by Rekabsaz et al. in 2021 for text retrieval, contains a real hybrid search query workload and base dataset from the click logs of a health web search engine. Each query consists of natural language search terms along with optional filters on clinical areas (e.g. "cardiology", "infectious disease", "surgery") and publication years. Each entity in the base dataset consists of a text passage, with a list of associated clinical areas and a publication date. The dataset contains 28 unique clinical areas and publication dates ranging from 1900 to 2020, resulting in over $2 ^ { 2 8 }$ possible query predicates total. We construct two query workloads, one consisting of queries that used date filters (dates) and another consisting of queries that used clincal area filters (areas). We generate 768-dimensional vectors from the query texts and passage texts using DPR [36], a widely-used, pre-trained encoder for open-domain Q&A. The resulting dataset has about 1?? base vectors, and we use a random sample of 1K queries for each query workload.

LAION: The LAION dataset [55] consists of 400M image embeddings and captions describing each image. The vector embeddings are generated from web-scraped images using CLIP [53], a multimodal language-vision model. In our evaluation, we construct two base datasets using 1M and 25M LAION subsets, both consisting of image vectors and text captions as a structured attribute. We also generate an additional structured attribute consisting of a keyword

list. We assign each image embedding its keyword list by taking the 3 words with highest text-to-image CLIP scores from a candidate list of 30 common adjectives and nouns (e.g., "animal", "scary").

To evaluate a series of micro-benchmarks, we generate four query workloads. For each query workload, we sample 1K vectors from the dataset as query vectors. We construct the regex query workload with predicates that perform regex-matching over the image captions. For each query predicate, we randomly choose strings of 2-10 regex tokens (e.g., "^[0-9]"). In addition, we construct three query workloads with predicates, similar to TripClick, that take a keyword list and filter out entities that do not have at least one matching keyword. Using this setup, we are able to easily control for correlation in the workload, and we generate a no correlation (no-cor), positive correlation (pos-cor), and negative correlation (neg-cor) workload. Figure 6 demonstrates some example queries and multi-modal retrieval results taken from each.

# 7.2 Benchmarked Methods

We briefly overview the methods we benchmark along with tested parameters. We implement ACORN-??, ACORN-1, pre-filtering, and HNSW post-filtering in ${ \mathrm { C } } { + } { + }$ in the FAISS codebase [5].

HNSW Post-filtering: To implement HNSW post-filtering, for each hybrid query with predicate selectivity ??, we over-search the HNSW index, gathering $K / s$ candidate results before applying the query filter. We note that this differes to some prior work [25], where HNSW post-filtering is implemented by collecting only $K$ candidate results, leading to significantly worse baseline query performance than ours. For the SIFT1M, Paper and LAION datasets, we use the FAISS’s default HNSW construction parameters: $M =$ 32, efc = 40. For the TripClick dataset, we find that the HNSW index for these parameters is unable to obtain high recalls for the standard ANN search task, thus we perform parameter tuning, as is standard. We perform a grid search for $M \in \{ 3 2 , 6 4 , 1 2 8 \}$ and $e f c \in \{ 4 0 , 8 0 , 1 2 0 , 1 6 0 , 2 0 0 \}$ and choose the pair the obtains the highest QPS at 0.9 Recall for ANN search. For TripClick, we choose $M = 1 2 8$ , $e f c = 2 0 0$ . We generate each recall-QPS curve by varying the search parameter efs from 10 to 800 in step sizes of 50.

Pre-filtering: We implement pre-filtering by first generating a list of dataset entries that pass the query predicate and then performing brute force search using FAISS’s optimized implementation for distance comparisons. We also efficiently implement all contains predicate evaluations using bitsets since the corresponding structured attributes have low cardinality.

![](images/7fba848f4e3181bac19667901023e4d70f05f539df569720446dbdd18113c5b2.jpg)

![](images/26905cc639190e9ce9ba0f4df79065ca75393bae6bf498511e7238e023760ba8.jpg)  
Figure 6: The figure contrasts retrieval results using vector-only similarity search (bottom left) versus hybrid search (right) on the LAION dataset. Both use the same query image (top left), and the hybrid search queries also include a structured query filter consisting of a keyword list, here containing a single keyword. The table on the right shows examples from three hybrid search query workloads: positive query correlation (top), no query correlation (middle), and negative query correlation (bottom).

Filtered-DiskANN: We evaluate both algorithms implemented in FilteredDiskANN [4], namely FilteredVamana and StitchedVamana. For both, we follow the recommended construction and search parameters according to the hyper-parameter tuning procedure described by Gollapudi et al. [25]. For FilteredVamana, we use construction parameters $L \ = \ 9 0 , R \ = \ 9 6$ , which generated the Pareto-Optimal recall-QPS curve from a parameter sweep over $R \in$ $\{ 3 2 , 6 4 , 9 6 \}$ and L between 50 and 100. For StitchedVamana, we use construction parameters $R _ { s m a l l } = 3 2 , L _ { s m a l l } = 1 0 0 , R _ { s t i t c h e d } = 6 4$ and $\alpha = 1 . 2$ , which generated the Pareto-Optimal recall-QPS curve from a parameter sweep over $R _ { s m a l l } , R _ { s t i t c h e d } \in \{ 3 2 , 6 4 , 9 6 \}$ and $L _ { s m a l l }$ between 50 and 100. To generate the recall-QPS curves we vary $L$ from 10 to 650 in increments of 20 for FilteredVamana, and $L _ { s m a l l }$ from 10 to 330 in increments of 20 for StitchedVamana.

NHQ: We evaluate the two algorithms, NHQ-NPG_NSW and NHQ-NPG_KGraph, proposed in [63]. For both we use the recommended parameters in the released codebase [12]. These parameters were selected using a hyperparameter grid search in order to generate the Pareto-optimal recall-QPS curve for either algorithm on the SIFT1M and Paper datasets. We generate the recall-QPS curve by varying ?? between 10 and 310 in steps of 20. In Figures 8b and 7b, we show the query performance of KGraph, the more performant of the two algorithms.

Milvus: We test four Milvus algorithms: IVF-Flat, IVF-SQ8, HNSW, and IVF-PQ [6]. For each we test the same parameters as Gollapudi et al. [25]. Since we find that the four Milvus algorithms achieve similar search performance, for simplicity, Figures 8b and 7b show only the method with Pareto-Optimal recall-QPS performance.

Oracle Partition Index: We implement this method by constructing an HNSW index for each possible query predicate in the LCPS datasets. For a given hybrid query, we search the HNSW partition corresponding to the query’s predicate. To construct each HNSW partition and generate the recall-QPS curve, we use the same parameters as the HNSW post-filtering method, described above.

ACORN-??: We choose the construction parameters ?? and efc to be the same as the HNSW post-filtering baseline, described above. We find that ACORN-??’s search performance is relatively insensitive to the choice of the construction parameter $M _ { \beta }$ , as Figure 12c shows. Thus, to maintain modest construction overhead, we choose $M _ { \beta }$ to be a small multiple of ?? , i.e., $M _ { \beta } = M$ or $M _ { \beta } = 2 M$ , picking the parameter for each dataset that obtains higher QPS at 0.9 Recall. Specifically, we constrain the memory budget of the index

to be no larger than the Vamana indices on the LCPS datasets and no larger than twice the size of the flat indices for HCPS datasets. We use $M _ { \beta }$ values of 32 for LAION-1M and LAION-25M, 64 for SIFT1M, Paper, and 128 for TripClick. We choose the construction parameter ?? according to the expected minimum selectivity query predicates of each dataset i.e., $\gamma = 1 2$ for SIFT1M and Paper, $\gamma = 3 0$ for LAION, and $\gamma = 8 0$ for TripClick. To generate the recall-QPS curve, we follow the same procedure described above for HNSW post-filtering.

ACORN-1: We construct ACORN-1 and generate the recall-QPS curve following the same procedure we use for ACORN-??, except that we fix $\gamma = 1$ and $M _ { \beta } = M$ .

# 7.3 Search Performance Results

We will begin our evaluation with benchmarks on the LCPS datasets, on which we are able to run all baseline methods as well as the oracle partition method. We will then present an evaluation on the HCPS datasets. On these datasets, the FilteredDiskANN and NHQ algorithms fail because they assume are unable to handle the high cardinality query predicate sets and non-equality predicate operators. As of this writing, we also find that Milvus cannot support regex-match predicates and contains predicates over variable length lists. As a result, we instead focus on comparing ACORN to the pre- and post-filtering baselines for the HCPS datasets. We report QPS averaged over 50 trials.

7.3.1 Benchmarks on LCPS Datasets. Figure 7 shows that ACORN-?? achieves state-of-the-art hybrid search performance and best approximates the theoretically ideal oracle partition strategy on the SIFT1M and Paper datasets. Notably, even compared to NHQ and FilteredDiskANN, which specialize for LCPS datasets, ACORN-?? consistently achieves $2 \mathrm { - } 1 0 \times$ higher QPS at fixed recall values, while maintaining generality. Additionally, we see ACORN-1 approximates ACORN-??’s search performance, attaining about $1 . 5 – 5 \times$ lower QPS than ACORN- $\gamma$ across a range of recall values.

To further investigate the relative search efficiency of ACORN-?? and ACORN-1, we turn our attention to Table 3, which shows the number of distance computations required of either method to obtain Recall@10 equal to 0.8. We see that the oracle partition method is the most efficient, requiring the fewest number of distance computations on both datasets. ACORN-?? is the next most efficient according to number of distance computations. While

![](images/1aabaa71ff7c37f4d800924510682525baa7bad799f8b693eaa9f36c917ee175.jpg)

![](images/d14fe1620e31fb6dcb4f84043dbecc35752e124a319a919183f9ccdae8a1196d.jpg)  
(a) SIFT1M Dataset

![](images/6ca69b17f728afbc8aa5982435ded316ce57123f022fb5e4b9a02be837090b99.jpg)  
(b) Paper Dataset   
Figure 7: Recall@10 vs QPS on SIFT1M and Paper

ACORN-?? approximates the oracle partition method, it’s predicateagnostic design precludes the same RNG-based pruning used to construct the oracle partitions. Rather than approximating RNGgraphs, ACORN-??’s levels approximate KNN-graphs, which are less efficient to search over explaining the performance gap. The table additionally shows that ACORN-1 is less efficient than ACORN-??, which is explained by the candidate edge generation used in ACORN-1. While the ACORN- $\gamma$ index stores up to $M \times \gamma$ edges per node during construction, ACORN-1 stores only up to $M$ edges per node during construction, and approximates an edge list of size $M \ast \gamma$ for each node during search using its neighbor expansion strategy. This approximation results in slight degradation to neighbor list quality and thus search performance. Finally, we see from the table, that HNSW post-filtering is the least efficient of the listed methods. This is because while ACORN-1 and ACORN- $\gamma$ almost exclusively traverse over nodes that pass the query predicates, the post-filtering algorithm is less discriminating and wastes distance computations on nodes failing the query predicate.

Returning to Figure 7, we see that the relative search efficiency, measured by QPS versus recall, of the oracle partition method, ACORN- $\gamma ,$ and ACORN-1 is not only affected by distance computations, but is also affected by vector dimensionality. We see that both ACORN-1 and ACORN-?? perform closer to the oracle partition method on the Paper dataset, while the performance gap grows slightly on SIFT1M. This is due to the cost of performing the filtering step over neighbor lists during search, which, relative to the cost of distance computations, is higher on SIFT1M than Paper since SIFT1M uses slightly lower-dimensional vectors.

7.3.2 Benchmarks on HCPS Datasets. Figure 8 shows that ACORN outperforms the baselines by $3 0 - 5 0 \times$ higher QPS at 0.9 recall on TripClick and LAION-1M, and as before, ACORN-1 approximates ACORN-??’s search performance. On both datasets, pre-filtering is prohibitively expensive, obtaining perfect recall at the price of efficiency. Meanwhile, post-filtering fails to obtain high recall, likely

Table 3: # Distance Computations to Achieve 0.8 Recall   

<table><tr><td></td><td>SIFT 1M</td><td>Paper</td></tr><tr><td>Oracle Partition</td><td>398.0</td><td>281.1</td></tr><tr><td>ACORN-γ</td><td>611.0 (+53.5%)</td><td>383.7 (+36.6%)</td></tr><tr><td>ACORN-1</td><td>999.6 (+151.0%)</td><td>567.8 (+101.2%)</td></tr><tr><td>HNSW Post-filter</td><td>1837.8 (+362.6%)</td><td>1425.5 (+406.2%)</td></tr></table>

* Percentage difference is shown in parenthesis and is relative to oracle partition method

![](images/fc56b6cc9ca86ba7d56ae4d78e63dcc06f92b8fb3ae531669dafc1436e9153e4.jpg)

![](images/94edc895d530333bed12ede9bfcc6f2a30d65b3323d39f69045c94b9f3eaafbf.jpg)

![](images/0c24aa669f1e3458b38f2e625324eac75d4210a890857eee676331ac4ec4aad3.jpg)  
(a) TripClick (areas)   
(b) TripClick (dates)   
(c) LAION1M (regex)   
Figure 8: Recall@10 vs QPS on TripClick and LAION-1M

due to the presence of varied query correlation and predicate selectivity, which we further explore further next.

Varied Predicate Selectivity: We use the Tripclick dataset to evaluate ACORN’s search performance across a range of realistic predicate selectivities. Figure 9 demonstrates that for each predicate selectivity percentile, ACORN-?? achieves 5-50x higher QPS at 0.9 recall compared to the next best-performing baseline. Once again ACORN-1 trails behind ACORN-??. We see that for low selectivity predicates, the pre-filtering method is most competitive, while the post-filtering baselines suffers from over $1 0 \times$ lower QPS than ACORN at fixed recall. However, for high selectivity predicates, prefiltering becomes less competitive while the post-filtering baseline obtains higher throughput, although its recall remains low.

Varied Query Correlation: Next we control for query correlation and evaluate ACORN on three different query workloads using the LAION-1M dataset. Figure 10 demonstrates that ACORN- $\gamma$ is robust to variations in query correlation and attains $2 8 \mathrm { - } 1 0 0 \times$ higher QPS at 0.9 recall than the next best baseline in each case. In the negative correlation case, the performance gap between postfiltering and the ACORN methods is the largest since post-filtering cannot successfully route towards nodes that pass the predicate. In the positive correlation case, ACORN- $\gamma$ once again outperforms the baselines, but post-filtering become more competitive, although it is still unable to attain recall above 0.9. The pre-filtering method’s QPS remains relatively unchanged, and is only affected by small variations in predicate selectivity for each query workload. As before, ACORN-1 approaches ACORN- $\gamma$ ’s search performance.

Scaling Dataset Size: Figure 11 shows ACORN’s search performance on LAION-25M with the no-correlation query workload, demonstrating that the performance gap between ACORN and existing baselines only grows as the dataset size scales. At 0.9 recall, ACORN- $\gamma$ achieves over three orders of magnitude higher QPS than the next best-performing baseline. As before, ACORN-1’s search performance approximates that of ACORN-??.

# 7.4 Index Construction

We will now evaluate ACORN’s construction procedure, including its indexing time and space footprint, ACORN-??’s compression procedure, and the predicate subgraph quality resulting from ACORN-??’s neighbor expansion approach.

7.4.1 TTI and Space Footprint. First, we analyze ACORN’s space footprint and indexing time. Table 4 and 5 show the time-to-index and index size of ACORN-?? and ACORN-1 compared to the bestperforming baselines. The reported index sizes for each method show the total space footprint of both vector storage and the index

![](images/d4cae3086852f920df0c49628fc9aa4efb85b99196f446ed250a946a9b590bd4.jpg)  
(a) 1p Sel (s=0.0127)

![](images/3f18eb7baa6f322b241b0c4e2ed9c5ce1825b9b9bb258c005adf67d9c90f3be7.jpg)  
(b) 25p Sel (s=0.0485)

![](images/1ddffb0c3a03aed7da83c6d23df00a7d85d0f549df418d2997e33e297331c040.jpg)  
(c) 50p Sel (s=0.1215)

![](images/572c95347a26290b4739ca8fb7193fdfa32bd771647b0dab859dd98a4e369e60.jpg)  
(d) 75p Sel (s=0.2529)

![](images/b81c69ac790f8b38fd7b1cead6e98304b756c653f4ba9250f4fb884ae6d61bd5.jpg)  
(e) 99p Sel (s=0.6164)   
Figure 9: Recall@10 vs QPS for Varied Selectivity Query Filters on TripClick

![](images/7e70326c133368ee7d52c83a891a70c60ab9386effe3b187c9ac6633684a18a7.jpg)  
(a) Neg. Correlation

![](images/9b72bd3c550a2303eac2c7ab41371515ecab193618817b812f426cabfdfa4284.jpg)  
(b) No Correlation

![](images/b893407a675c30c21e64b4c1802ec965a54c01340af8086509ccc97ecdc73713.jpg)  
(c) Pos. Correlation

![](images/2f3ad67ceec9c1f45f2a74856f2b29b68ffc7d969fb5d2b35abc35c63280e37b.jpg)  
Figure 10: Recall@10 vs QPS on LAION1M   
Figure 11: Recall@10 vs QPS on LAION-25M

itself. All methods are measured using the parameters reported in Section 7.2.

We first consider ACORN-??’s construction overhead. Table 4 shows that across all datasets, ACORN-??’s TTI is at most $1 1 \times$ higher than HNSW’s, and at most $2 . 1 5 \times$ higher than that of StitchedVamana, the best performing specialized index. Table 5 shows that ACORN-??’s index size is at most $1 . 3 \times$ larger than that of HNSW, and at least $2 5 \%$ smaller than that of StitchedVamana. The reason for ACORN- $\cdot \gamma ^ { : }$ ’s increased index size and TTI compared to HNSW is it’s candidate-edge generation step during construction, which expands each neighbor list. Meanwhile, ACORN-1 achieves the lowest TTI of all listed baselines in table 4, and its index size is at most $1 . 2 5 \times$ HNSW’s index size and at least $2 5 \%$ smaller than StitchedVamana’s index size. We see that while ACORN-?? achieves superior search performance by leveraging a neighbor-list expansion during construction, ACORN-1 provides a close approximation at lower TTI and space footprint by instead performing the neighbor-list expansion during search. The two algorithms exhibit a trade-off between search performance and construction overhead.

7.4.2 ACORN-?? Pruning. Given ACORN- $\cdot \gamma$ ’s higher construction overhead, we investigate the efficiency of its predicate-agnostic compression strategy in reducing index construction costs while maintaining search performance. First, Table 6 shows ACORN-??’s average out-degree per level for each dataset, confirming that compression on level 0 leads to significantly smaller neighbor lists, compared to level without compression, which may have neighbor lists as large as $M \cdot \gamma$ .

Turning our attention to Figure 12, we evaluate three different pruning strategies applied to ACORN-??’s neighbor lists during

Table 4: TTI (s)   

<table><tr><td></td><td>TripClick</td><td>LAION-1M</td><td>LAION-25M</td><td>Sift1M</td><td>Paper</td></tr><tr><td>ACORN-γ</td><td>9902.9</td><td>835.8</td><td>38,007.5</td><td>148.9</td><td>255.6</td></tr><tr><td>ACORN-1</td><td>322.9</td><td>25.9</td><td>705.3</td><td>8.6</td><td>27.0</td></tr><tr><td>HNSW</td><td>891.0</td><td>32.9</td><td>1,147.2</td><td>11.3</td><td>29.2</td></tr><tr><td>FilteredVamana</td><td>NA</td><td>NA</td><td>NA</td><td>18.3</td><td>51.9</td></tr><tr><td>StitchedVamana</td><td>NA</td><td>NA</td><td>NA</td><td>69.2</td><td>189.7</td></tr></table>

Table 5: Index Size (GB)   

<table><tr><td></td><td>TripClick</td><td>LAION-1M</td><td>LAION-25M</td><td>Sift1M</td><td>Paper</td></tr><tr><td>ACORN-γ</td><td>4.9</td><td>2.4</td><td>59</td><td>0.98</td><td>2.5</td></tr><tr><td>ACORN-1</td><td>4.6</td><td>2.3</td><td>59</td><td>0.93</td><td>2.4</td></tr><tr><td>HNSW</td><td>4.1</td><td>2.2</td><td>54</td><td>.75</td><td>2.1</td></tr><tr><td>Flat Index</td><td>3.1</td><td>1.9</td><td>47</td><td>.51</td><td>1.6</td></tr><tr><td>FilteredVamana</td><td>NA</td><td>NA</td><td>NA</td><td>.61</td><td>1.8</td></tr><tr><td>StitchedVamana</td><td>NA</td><td>NA</td><td>NA</td><td>1.3</td><td>3.5</td></tr></table>

Table 6: ACORN-?? Average Out Degree   

<table><tr><td></td><td>TripClick</td><td>LAION-1M</td><td>LAION-25M</td><td>Sift1M</td><td>Paper</td></tr><tr><td>Level 0 (compressed)</td><td>191</td><td>50.1</td><td>49.4</td><td>87.5</td><td>86.0</td></tr><tr><td>Level 1</td><td>8,075</td><td>960</td><td>960</td><td>384</td><td>384</td></tr><tr><td>Level 2</td><td>54.0</td><td>919</td><td>937</td><td>363</td><td>359</td></tr><tr><td>Level 3</td><td>0</td><td>25.3</td><td>689</td><td>25.3</td><td>57.4</td></tr><tr><td>Level 4</td><td>NA</td><td>0</td><td>16</td><td>0</td><td>1.0</td></tr><tr><td>M·γ</td><td>10,240</td><td>960</td><td>960</td><td>384</td><td>384</td></tr><tr><td>Mβ</td><td>128</td><td>32</td><td>32</td><td>64</td><td>64</td></tr></table>

construction: i) ACORN’s predicate-agnostic pruning strategy at varied levels of compression indicated by different $M _ { \beta }$ (Mb) values, where $M b = 7 6 8$ represents no pruning, and lower values represent more aggressive pruning, ii) a metadata-aware RNG-based pruning approach, which is employed by FilteredDiskANN’s algorithms, and iii) HNSW’s metadata-blind pruning. We consider TTI, space footprint, the number of candidates edges pruned per node and search performance. The figure represents space footprint measured by the average out degree of nodes on on level 0, the level on which each pruning strategy is applied. In addition, the figure shows search performance measured by recall at 20,000 QPS. We note that the recall ranges of the recall-QPS curve generated by different pruning methods varied significantly, leading us to choose a QPS threshold rather than a recall threshold. Interestingly, Figure 12 shows that ACORN’s pruning can significantly reduce both the TTI and space footprint by aggressively pruning candidate edges, while maintaining search performance. In comparison, applying HNSW pruning to the index results in significantly degraded hybrid search performance. Meanwhile the metadata-aware RNG-base pruning results in similar search performance to ACORN- $\gamma$ ’s pruning, but it is less efficient by TTI and space footprint than ACORN’s pruning for small values of $M _ { \beta }$ (e.g., $M _ { \beta } = 3 2 , 6 4 ,$ ).

ACORN-Gamma pruning Metadata-aware RNG-based pruning HNsW pruning (metadata-blind)

![](images/d11919f93fcec33a769bd54b8c89d5ce5bf2fc4a418dd4e670a7c88c266f48f1.jpg)  
(a) TTI

![](images/9ef279c3a522c761da9c6c1ae0ad975fe886b63faffcb59d6e2e59cb947dc133.jpg)  
(b) Space footprint

![](images/0e24225887a1a309e6941d75d62744ec55761e1a4d0dd2aee08dd5ea49bd9167.jpg)  
(c) # Edges Pruned

![](images/96417bbf11eddf006309f1cfd7b8da09ca8f3e04f4ee185fc73598d91161d3ae.jpg)  
(d) Search Perf.   
Figure 12: Comparison of pruning methods on SIFT1M and their impact on TTI (a), space footprint of the index (b), the number of candidate edges pruned (c) and search performance (d). $M _ { \beta }$ values used for ACORN- $\gamma$ are shown along the x-axis.

7.4.3 Graph Quality. Finally, we investigate the graph quality of ACORN-??’s predicate subgraphs. Figure 13 compares graph connectivity, graph height, and out degrees for HNSW oracle partitions and ACORN-?? predicate subgraphs across varied predicate selectivities on the TripClick dataset’s real hybrid search queries.

From Figure 13a, we see ACORN-??’s predicate-subgraph connectivity empirically matches or exceeds that of the HNSW oracle partition across selectivities, demonstrating the effectiveness of ACORN-??’s neighbor expansion strategy. Next, Figure 13b shows that the controlled hierarchy of ACORN-??’s predicate subgraphs emulate that of the HNSW oracle partitions. Malkov et al. show that HNSW search performance is sensitive to graph height [48]; thus, this result helps explain ACORN-??’s ability to emulate the search efficiency of the oracle partition. Lastly, Figure 13c examines the average out degree resulting from performing the search-time filtering, described in Figure 4(a), over the ACORN-?? index. We note that sufficiently high, but bounded, out-degrees are important for emulating HNSW’s navigability properties, as discussed in Section 6.3. The figure confirms that ACORN’s predicate subgraphs have average out-degrees consistently close to and bounded by ??. As expected, the HNSW oracle partition has significantly lower average out-degrees than nodes on ACORN-??’s uncompressed levels because HNSW applies RNG-based pruning. We also note, that the ACORN predicate subgraph with 1 percentile selectivity has lower average out degrees than the other predicate subgraphs because the low selectivity predicates result in fewer than 128 nodes on the largest uncompressed levels, thus capping the maximum out degree per node below $M = 1 2 8$ . Overall, we observe that ACORN-?? produces high quality predicate subgraphs, which empirically emulate several HNSW properties related to search efficiency.

# 8 RELATED WORK

Pre- & Post-filtering-based Systems. Many hybrid search systems rely on pre- and post-filtering. While several systems have developed pre-processing methods to perform faster filtering during search, these systems fail to reduce the excessive and expensive distance computations which bottleneck performance. Weaviate [1] creates an inverted index for structured data ahead of time, then uses it at query time to create a bitmap of eligible candidates during

![](images/82727b81d2b12460a26440f30ca23b5793cac3a56ee20ba25c0229712c385b9a.jpg)  
(a) # SCC

![](images/81cf16f49486cd45612300777e4c5be7b57f4d0f4bc57bb06f00d387bdb7b7c3.jpg)  
(b) Graph Height

![](images/39d20e7f8a5d1a5acd643b869528ca2c6196a1f05a326ecec3339b6072ecae71.jpg)  
(c) Avg Out Degree   
Figure 13: Graph quality of ACORN- $\gamma$ predicate subgraph evaluated by (a) average number of strongly connected components per level, (b) graph height, and (c) average out degree of nodes across uncompressed levels. Results are shown for the TripClick dataset with 1, 25, 50, 75, and 99 percentile selectivity predicates to generate the predicate subgraph and HNSW oracle partition.

post-filtering. Milvus [62] likewise creates an approved list of points by maintaining a distribution of attributes over the dataset in order to map commonly used query filters to a list of approved points before performing pre- or post-filtering. Several space-partitioning indices like FAISS-IVF [14, 34] and LSH [10] store metadata information in the index, allowing them to rapidly filter entities during post-filtering. Despite the optimized filtering steps in each of these approaches, the core problems of pre- and post-filtering remain, particularly for low correlation or selectivity predicates.

Specialized Indices. Alternatively, several recent works develop novel graph-based algorithms for hybrid search, often improving performance for a constrained set of predicates. NHQ [63] encodes attributes alongside vectors, and then uses a "fusion distance" during search that accounts for vector distances as well as attribute matches. This approach supports only equality query predicates and assumes each dataset entity has only one structured attribute. Filtered-DiskANN [25] proposes two algorithms: FilteredVamana and StitchedVamana. Both methods constrain the query filter cardinality to about 1, 000 with only equality predicates so that the index construction steps can use this knowledge to appropriately generate and prune candidate edge lists. Similarly HQI [49] optimizes batch query-processing by assuming a limited cardinality of 20 query predicates to design an efficient partitioning scheme. On the otherhand, Qdrant [61] proposes to densify an HNSW graph and perform a filtered greedy search. While this approach aligns intuitively with ACORN’s neighbor-list expansions during construction, Qdrant’s proposal inadvertently flattens the graph by directly increasing the HNSW parameter ??, which impacts HNSW’s level normalization constant. Malkov et al. show that HNSW’s performance is sensitive to its number of levels, and flattening the graph degrades search performance [48]. In addition, Qdrant’s proposed method does not provide a solution for dealing with the increased memory overhead after creating a denser HNSW.

# 9 CONCLUSION

We proposed ACORN, the first approach for efficient hybrid search across vectors and structured data that supports large and diverse sets of query predicates. ACORN uses a simple, yet effective, search strategy based on the core idea of predicate subgraph traversal. We presented two indices, ACORN- $ { \gamma }$ and ACORN-1, that implement this search strategy by modifying the HNSW indexing algorithm. Our results show that ACORN achieves state-of-the-art hybrid search performance on both prior benchmarks, involving simple,

low-cardinality query predicate sets, as well as more complex benchmarks involving new predicate operators and high cardinality predicate sets. Across both types of benchmarks, ACORN- $\gamma$ achieves $2 \mathrm { - } 1 \mathrm { , } 0 0 0 \mathrm { \times }$ higher QPS at 0.9 recall than prior methods, and ACORN-1 approximates ACORN-??’s search performance with $9 - 5 3 \times$ lower TTI for resource-constrained settings.

# ACKNOWLEDGMENTS

The authors would like to thank Peter Bailis for his valuable feedback on this work.

This research was supported in part by affiliate members and other supporters of the Stanford DAWN project, including Meta, Google, and VMware, as well as Cisco, SAP, and a Sloan Fellowship. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the sponsors.

