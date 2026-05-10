# Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval

Roi Pony 1 Adi Raz 1 Oshri Naparstek 1 Idan Friedman 1 Udi Barzelay 1

# Abstract

Multi-vector late-interaction retrievers such as ColBERT achieve state-of-the-art retrieval quality, but their query-time cost is dominated by exhaustively computing token-level MaxSim interactions for every candidate document. While approximating late interaction with single-vector representations reduces cost, it often incurs substantial accuracy loss. We introduce Col-Bandit, a query-time pruning algorithm that reduces this computational burden by casting reranking as a finite-population Top- $K$ identification problem. Col-Bandit maintains uncertainty-aware bounds over partially observed document scores and adaptively reveals only the (document, query token) MaxSim entries needed to determine the top results under statistical decision bounds with a tunable relaxation. Unlike coarse-grained approaches that prune entire documents or tokens offline, Col-Bandit sparsifies the interaction matrix on the fly. It operates as a zero-shot, drop-in layer over standard multi-vector systems, requiring no index modifications, offline preprocessing, or model retraining. Experiments on textual (BEIR) and multimodal (REAL-MM-RAG) benchmarks show that Col-Bandit preserves ranking fidelity while reducing MaxSim FLOPs by up to $5 \times$ , indicating that dense late-interaction scoring contains substantial redundancy that can be identified and pruned efficiently at query time.

# 1. Introduction

Multi-vector late-interaction retrievers, such as Col-BERT (Khattab & Zaharia, 2020), have emerged as a powerful alternative to single-vector dense retrieval. By representing each query and document as a set of token embeddings, these models capture fine-grained semantic matches that single-vector representations miss (Wang et al., 2023; Formal et al., 2021). This paradigm has been widely adopted in recent text and multimodal systems (Faysse et al., 2024; Team, 2025a; Warner et al., 2025; Team, 2025b; Xu et al., 2025; Gunther et al.¨ , 2025), becoming a standard foundation for high-accuracy neural retrieval. However, this granularity comes with a cost. Unlike single-vector retrieval, where scoring is a cheap dot product, exact late interaction requires evaluating a grid of token-level operations (MaxSim) for every document. Consequently, this computation often becomes the bottleneck in modern pipelines, motivating methods that reduce these operations without sacrificing ranking fidelity (Santhanam et al., 2022a; Engels et al., 2023).

The ”Hiring” Analogy. To build intuition for the inefficiency of standard late interaction, consider a manager hiring the top- $K$ candidates from $N$ applicants. Each applicant takes $T$ short tests, and their final score is the sum of the $T$ results. Administering all $T$ tests to every applicant guarantees the correct shortlist, but it is wasteful. A resource-efficient manager would allocate tests adaptively, giving a few to everyone and focusing the remaining budget on the candidates whose ranking is unclear, stopping once the top- $K$ is statistically certain. Standard late-interaction retrieval mirrors this wasteful strategy. It sums $T$ tokenwise interactions for every document, even though partial evaluation often suffices to rule documents out (or in).

The Opportunity: Removing Redundancy. In the vectorset setting, the total score is a sum of independent components. Na¨ıvely, systems evaluate the full sum for every candidate in the pool $\mathcal { D }$ . However, for any specific query, we do not need to know the exact score of a document that is clearly irrelevant, nor do we need perfect precision for a clear winner. We only need enough information to distinguish the true Top- $K$ documents from the rest. This implies that the computational budget should be spent asymmetrically, heavily on the “borderline” cases and sparsely on everything else.

Our Approach: Col-Bandit. We propose viewing this resource allocation problem as progressive matrix completion. We treat the token-level scores as values in a table that can be revealed on-demand. Our objective is to reveal just enough cells to confidently identify the Top- $K$ set, minimizing computation while maintaining a user-defined level of statistical reliability (Figure 1).

To this end, we introduce Col-Bandit, a purely query-time algorithm that operates directly on vanilla ColBERT. Unlike prior acceleration methods that require quantization or distilling document representations, Col-Bandit works on top of standard indices and model weights, requiring no index-time changes and no retraining. We formulate the task as a finite-population Top- $K$ identification problem. By exploiting the fact that document token sequences are finite, we utilize Serfling-type concentration inequalities (Bardenet & Maillard, 2015) to construct tighter confidence intervals than standard bandit approaches. We further introduce a calibration parameter to optimize the trade-off between theoretical certification and practical FLOP reduction.

![](images/c560c1164979dc85f049a7cfaabd46b518e166fa8b218f37ad9c66d8858b0912.jpg)  
Figure 1. Schematic of Col-Bandit. Given a query (e.g., ”human mobility...”) and a set of candidate documents (e.g., Nature, Auto), the goal is to identify the Top-2 relevant documents. (A) Full ColBERT determines the exact score for every document by summing all interaction cells (MaxSims), requiring $100 \%$ of the compute budget. (B) Col-Bandit approximates these sums using partial cell observations. By adaptively revealing informative cells (green) and skipping others (hatched), it maintains confidence intervals for the total score. The algorithm terminates as soon as a positive separation gap emerges: the Lower Bound of the weakest winner (Sports) is strictly higher than the Upper Bound of the strongest loser (Auto). This enables the identification of the correct Top- $K$ ranking while saving $60 \%$ of the query-time computations.

# Contributions.

• Formulation: We cast late-interaction reranking as a finite-population Top- $K$ identification problem using a progressive scoring framework.

• Algorithm: We introduce Col-Bandit, a Lower-Upper Confidence Bound (LUCB) (Kalyanakrishnan et al., 2012) style algorithm that leverages variance-adaptive Serfling bounds for tighter estimation and a tunable relaxation parameter for efficiency.

• Drop-in Acceleration: We demonstrate substantial FLOP reductions on standard benchmarks without requiring any offline index modifications or model retraining.

# 2. Background and Related Work

# 2.1. Preliminaries: Late Interaction Retrieval

ColBERT Late Interaction Scoring. Consider a query $Q$ and a document $d$ from a collection $\mathcal { D }$ of size $N$ . ColBERT represents the query as a set of $T$ token embeddings

$$
Q = \{ \mathbf { q } _ { 1 } , \mathbf { q } _ { 2 } , \dots , \mathbf { q } _ { T } \} \subset \mathbb { R } ^ { M } ,
$$

and each document $d$ as a set of $L _ { d }$ token embeddings

$$
\mathbf { E } ( d ) = \{ \mathbf { e } _ { d , 1 } , \mathbf { e } _ { d , 2 } , \dots , \mathbf { e } _ { d , L _ { d } } \} \subset \mathbb { R } ^ { M } ,
$$

where $M$ is the embedding dimension, $T$ is the query length, and $L _ { d }$ is the length of document $d$ .

The ColBERT scoring function computes relevance through a late interaction mechanism. For each query token $t \in [ T ]$ (where $[ T ] \triangleq \{ 1 , 2 , \dots , T \} )$ , ColBERT identifies the most similar document token using the MaxSim operation:

$$
h ( d , t ) \triangleq \operatorname* { m a x } _ { j \in [ L _ { d } ] } \mathrm { s i m } ( \mathbf { e } _ { d , j } , \mathbf { q } _ { t } ) ,
$$

where $\mathrm { s i m } ( \cdot , \cdot )$ is a similarity function (typically cosine similarity).1 The final query-document score aggregates

![](images/e83ab3d5d503631720728e0f839537ee5caf5dfaa836f0d8112bf5447efd70d6.jpg)  
Figure 2. Cost-Accuracy trade-off for Col-Bandit compared to Random Reveal (Doc-Uniform) and Greedy Top-Margin (Doc-TopMargin) across three retrieval settings (text and multimodal). Each star marker denotes a Col-Bandit operating point obtained by sweeping the relaxation parameter $\alpha _ { \mathrm { e f } }$ . The top-right corner (Overlap $\textcircled { a } 5 = 1 . 0$ , $\mathrm { C o s t } = 1 0 0 \%$ ) corresponds to full exhaustive scoring.

these per-query-token maxima:

$$
S ( d ; Q ) \triangleq \sum _ { t = 1 } ^ { T } h ( d , t ) .
$$

Top- $K$ Retrieval. The retrieval objective is to identify the set of $K$ documents from a search set $\mathcal { D }$ (e.g., a candidate pool produced by an ANN stage) with the highest scores:

$$
{ \mathcal { T } } _ { K } ^ { \star } ( Q ) \triangleq \operatorname * { a r g t o p K } _ { d \in { \mathcal { D } } } S ( d ; Q ) .
$$

Index-Time vs. Query-Time. Retrieval systems separate index-time (offline) processing, which extracts representations and builds index structures, from query-time (online) computation, which encodes the query and ranks candidates. In late-interaction systems, the full similarity computation performed at query time is typically treated as a reranking stage.

Atomic Cost. We define the atomic cost unit of query-time work as computing a single MaxSim value $h ( d , t )$ in Eq. 1. Standard exact reranking evaluates all $N \times T$ such values, which can dominate query-time cost even after candidate retrieval.

# 2.2. Related Work

We categorize related work by when and what they prune (Figure 3). Col-Bandit is the first to prune at the MaxSim operation level during query-time scoring.

Index-Time Compression & Token Pruning. Approaches like PLAID (Santhanam et al., 2022a), Col-BERTv2 (Santhanam et al., 2022b), and MUVERA (Dhulipala et al., 2024) accelerate retrieval via centroid-based compression, quantization, or fixed-dimensional encodings, improving the practicality of late-interaction methods that were initially constrained by considerable storage requirements. Additional system and indexing advances such as WARP (Scheerer et al., 2025) further improve scalability and usability. Similarly, token pruning methods (Lassance et al., 2021; Tonellotto & Macdonald, 2021) permanently discard non-informative tokens to reduce the index size $( N )$ or query length $( T )$ , including near-lossless vector count reduction (Clavie et al. ´ , 2024) and approaches that use a fixed number of representative tokens (MacAvaney et al., 2025). While effective, these methods are fixed at index-time and typically require offline modifications. Col-Bandit is orthogonal to these approaches, it operates purely at query-time on standard indices, dynamically pruning the atomic interaction matrix $H$ during scoring.

Efficient Systems & Bound-Based Pruning. Systemlevel optimizations like DESSERT (Engels et al., 2023) use approximate retrieval to speed up candidate generation. In sparse retrieval, algorithms like WAND (Broder et al., 2003) and BMW (Ding & Suel, 2011) use score upper bounds to skip documents. Col-Bandit bridges these concepts, applying bound-based early stopping to dense late-interaction. Unlike WAND, which prunes inverted list pointers, we prune atomic MaxSim operations $h ( d , t )$ to certify the Top- $K$ set with statistical guarantees.

MaxSim-Level Pruning (Our Approach). To our knowledge, no prior work adaptively prunes interactions within the exact scoring loop. Existing methods reduce the number of candidates $( N )$ or tokens $( T )$ before scoring. Col-Bandit frames the scoring process itself as a finite-population Top-$K$ identification problem, progressively revealing only the subset of MaxSim entries needed to certify the ranking.

Finite-Population Bandits and Top- $k$ Arm Identification. Our method is inspired by fixed-confidence Top- $K$ Arm

![](images/c02f710d053ff7750165e18875f307edf4e92568702fe3e51533ec980958a756.jpg)  
Figure 3. Taxonomy of efficient late-interaction retrieval. Methods are classified by when they prune (index-time vs. query-time) and what they prune. Col-Bandit is the first to dynamically prune the atomic interaction matrix $H$ during query-time scoring.

Identification in bandits (Kalyanakrishnan et al., 2012; Chen et al., 2014). The multi-armed bandit (MAB) framework has been extensively studied for resource-constrained selection problems, with two main paradigms: fixed-budget and fixed-confidence best arm identification (BAI). We focus on the fixed-confidence setting, where the goal is to identify the best arms with high probability while minimizing sample complexity, in our setting, we treat each document as an arm and connect reranking to Top- $K$ identification. Standard algorithms include UCB (Auer et al., 2002) and UCB-E (Audibert & Bubeck, 2010), which typically assume infinite sampling with replacement. The LUCB (Lower-Upper Confidence Bound) framework (Kalyanakrishnan et al., 2012) provides an efficient strategy for Top- $K$ identification by adaptively sampling arms based on confidence intervals, motivating our interval-driven reveal policy and stopping criterion.

Late-interaction retrieval has a fundamentally different structure: each document has a finite set of $T$ token scores that can be sampled without replacement. Recent work has explored MAB techniques in related contexts, including prompt learning (Shi et al., 2024), LLM evaluation (Zhou et al., 2024), and approximate $\mathbf { k }$ -NN search (Indyk & Wagner, 2019). We adapt the fixed-confidence Top- $K$ framework to exploit this finite-population structure. Specifically, we employ the Bernstein–Serfling inequality (Bardenet & Maillard, 2015) to derive variance-adaptive confidence bounds that shrink deterministically as a document is fully scored, providing tighter guarantees than standard infinitepopulation bandit bounds.

# 3. Problem Formulation

We formalize the efficient retrieval problem as a sequential decision process over a sparsely observed matrix, mapping the task to a fixed-confidence Multi-Armed Bandit (MAB) setting with finite populations.

# 3.1. The MaxSim Matrix and Observation Model

Consider a query $Q$ with $T$ tokens and a search set of $N$ documents, $\mathcal { D } = \{ d _ { 1 } , \ldots , d _ { N } \}$ . We define the implicit MaxSim Matrix, $H \in \mathbb { R } ^ { N \times T }$ , where each entry corresponds to the maximum similarity (1) of a query token with a document’s tokens:

$$
H _ { i , t } \triangleq h ( d _ { i } , t ) = \operatorname* { m a x } _ { j \in [ L _ { d _ { i } } ] } \operatorname { s i m } ( \mathbf { e } _ { d _ { i } , j } , \mathbf { q } _ { t } ) .
$$

The total late-interaction score for document $i$ is the rowsum:

$$
S _ { i } \triangleq \sum _ { t = 1 } ^ { T } H _ { i , t } .
$$

Our objective is to identify the set of indices $\mathcal { T } _ { K } ^ { \star }$ corresponding to the $K$ documents with the highest scores $S _ { i }$ .

At any time step, the algorithm has access to an observed set of entries $\Omega \subseteq [ N ] \times [ T ]$ . For each document $i$ , we denote the set of observed token indices as $\mathcal { O } _ { i } \triangleq \{ t : ( i , t ) \in \Omega \}$ and the unobserved counterpart as $\mathcal { U } _ { i } \triangleq [ T ] \backslash \mathcal { O } _ { i }$ . Revealing a new entry $( i , t ) \notin \Omega$ incurs a unit cost, returns the exact value $H _ { i , t }$ , and updates $\Omega  \Omega \cup \{ ( i , t ) \}$ .

We measure computational cost via coverage, defined as the fraction of the matrix revealed. At any time step of our algorithm the cost is:

$$
\gamma ( \Omega ) \triangleq \frac { | \Omega | } { N \times T } = \frac { 1 } { N T } \sum _ { i = 1 } ^ { N } | \mathcal { O } _ { i } | .
$$

# 3.2. Mapping to Finite-Population Bandits

This formulation mirrors the Best- $K$ Identification problem in stochastic bandits, where each document $i$ is an arm and $S _ { i }$ is its mean reward (up to a scaling factor $T$ ). However, two key structural properties distinguish our setting from standard literature: (i) Finite Population (Sampling without Replacement): Standard bandits assume that pulling arm $i$ reveals a sample from an infinite distribution $X \sim P _ { i }$ . In contrast, our “arm” $i$ consists of a fixed, finite population of $T$ values $\{ H _ { i , 1 } , \ldots , H _ { i , T } \}$ . Repeatedly querying the same document samples these values without replacement. This implies that as $| \mathcal { O } _ { i } |  T$ , the uncertainty about $S _ { i }$ collapses to zero deterministically. (ii) Bounded Support: The similarity function is bounded (e.g., cosine similarity in $[ - 1 , 1 ] )$ , providing a strict support $[ a , b ]$ for all unobserved entries $H _ { i , t }$ .

# 3.3. Objective: $\delta$ -PAC Identification

We seek an adaptive policy $\pi$ that decides which entry $( i , t )$ to reveal next, and a stopping rule $\tau$ . The algorithm must satisfy the Probably Approximately Correct (PAC) condition:

$$
\mathbb { P } \left( \hat { \mathcal { T } } _ { K } = \mathcal { T } _ { K } ^ { \star } \right) \ge 1 - \delta ,
$$

where $\hat { \mathcal { T } } _ { K }$ is the returned set and $\delta \in ( 0 , 1 )$ is a user-defined error tolerance. Among all $\delta$ -PAC policies, we aim to minimize the expected coverage $\mathbb { E } [ \gamma ( \Omega _ { \tau } ) ]$ .

# 4. Method: Col-Bandit

We view this problem as a competitive matrix completion task: entries of the $N \times T$ matrix $H$ are revealed adaptively until the Top- $K$ documents can be separated.

Col-Bandit maintains per-document decision bounds $[ \mathrm { L C B } _ { i } , \mathrm { U C B } _ { i } ]$ that guide (i) where to allocate additional computation and (ii) when to stop. At each iteration, the algorithm compares the weakest current Top- $K$ candidate against the strongest current non-Top- $K$ candidate and continues revealing entries until they are separated under the maintained decision bounds. The decision radius we use follows a finite-population, variance-adaptive template (empirical Bernstein–Serfling style) and is calibrated empirically The complete procedure is summarized in Algorithm 1.

# 4.1. Inputs and Exploration Strategies

Col-Bandit takes as input the document set $\mathcal { D }$ , target $K$ , and a user-specified tolerance knob $\delta$ that controls the conservativeness of the decision radius. Optionally, it utilizes token-wise bounds $H _ { i , t } \in [ a _ { i , t } , b _ { i , t } ]$ for unrevealed entries; when unavailable, we default to a global similarity range (e.g., [0, 1] or $[ - 1 , 1 ] )$ ). To ensure robust variance estimation and avoid premature stopping early in the process, we evaluate two exploration strategies.

Static warm-up. We initialize with a uniform random sample $\Omega _ { 0 } \subseteq [ N ] \times [ T ]$ of size $| \Omega _ { 0 } | = \lceil \gamma _ { \mathrm { i n i t } } N T \rceil$ , drawn without replacement. All entries $( i , t ) \in \Omega _ { 0 }$ are revealed to populate the initial interaction matrix, and the adaptive procedure starts from $\Omega  \Omega _ { 0 }$ .

Dynamic $\epsilon$ -greedy. We integrate an $\epsilon$ -greedy policy (Sutton et al., 1998) directly into the refinement step (Algorithm 1, lines 10–16). At each iteration, with probability $\epsilon$ we reveal a random unobserved token from the selected document to encourage exploration; otherwise, we select the token with the highest heuristic utility (exploitation). We ablate this policy against static warm-up and study sensitivity to $\gamma _ { \mathrm { i n i t } }$ and $\epsilon$ in Section 5.3. Empirically, dynamic $\epsilon$ -greedy consistently outperforms static warm-up by adapting more

# Algorithm 1 Col-Bandit (Adaptive Late-Interaction Pruning)

Require: Docs $\mathcal { D }$ , Query $Q , K , \delta$ , Relaxation $\alpha _ { e f }$ , Bounds   
$[ a , b ]$ , Exploration $\epsilon$   
1: Init: $\Omega  \emptyset$ , $H \in \mathbb { R } ^ { N \times T }$ (sparse)   
2: Compute initial $\mathrm { L C B } _ { i }$ , $\mathrm { U C B } _ { i }$ for all $i \in [ N ]$ using   
bounds   
3: while True do   
4: $\begin{array} { r l } & { \hat { \mathcal { T } } _ { K } \gets \arg \log \mathrm { K } _ { i \in [ N ] } \hat { S } _ { i } } \\ & { i ^ { + } \gets \arg \operatorname* { m i n } _ { i \in \hat { \mathcal { T } } _ { K } } \mathrm { L C B } _ { i } } \\ & { i ^ { - } \gets \arg \operatorname* { m a x } _ { i \notin \hat { \mathcal { T } } _ { K } } \mathrm { U C B } _ { i } } \\ & { \mathbf { i f } \mathrm { L C B } _ { i ^ { + } } \geq \mathrm { U C B } _ { i ^ { - } } \mathbf { t h e n } } \\ & { \quad \quad \mathbf { r e t u r n } \hat { \mathcal { T } } _ { K } } \end{array}$   
5: $\triangleright$ Weakest Winner   
6: $\triangleright$ Strongest Loser   
7:   
8: ▷ Top- $K$ separated   
9: end if   
10: $i ^ { \star } \gets \mathrm { a r g } \operatorname* { m a x } _ { i \in \{ i ^ { + } , i ^ { - } \} } ( \mathrm { U C B } _ { i } - \mathrm { L C B } _ { i } )$   
11: Sample $r \sim \mathrm { U n i f o r m } ( 0 , 1 )$   
12: if $r < \epsilon$ then   
13: Select uniform random $t ^ { \star } \in \mathcal { U } _ { i }$ ⋆ ▷ Exploration   
14: else   
15: $t ^ { \star } \gets \arg \operatorname* { m a x } _ { t \in \mathcal { U } _ { i ^ { \star } } } ( b _ { i ^ { \star } , t } - a _ { i ^ { \star } , t } ) \triangleright$ Max-Width   
16: end if   
17: $\begin{array} { r } { H _ { i ^ { \star } , t ^ { \star } }  h ( d _ { i ^ { \star } } , t ^ { \star } ) } \\ { \Omega  \Omega \cup \{ ( i ^ { \star } , t ^ { \star } ) \} } \end{array}$ ▷ Reveal MaxSim   
18:   
19: Update $\hat { \mu } _ { i ^ { \star } } , \hat { \sigma } _ { i ^ { \star } }$ using $H _ { i ^ { \star } , t ^ { \star } }$   
20: Update $\mathrm { L C B } _ { i ^ { \star } }$ , $\mathrm { U C B } _ { i ^ { \star } }$ via Eq. 13,14   
21: end while

effectively to instance-specific sparsity.

# 4.2. Ranking Proxy and Decision Bounds

Let $n _ { i } = | O _ { i } |$ denote the number of observed query-token and $\widehat { \mu } _ { i }$ be the empirical mean of observed token scores:

$$
\widehat { \mu } _ { i } = \frac { 1 } { n _ { i } } \sum _ { t \in \mathcal { O } _ { i } } H _ { i , t } .
$$

We define the estimated total score

$$
{ \widehat { S } } _ { i } \triangleq T \cdot { \widehat { \mu } } _ { i } .
$$

This estimate is used to order candidates and form the tentative set $\widehat { \tau } _ { \kappa }$ inside LUCB.

Deterministic (Hard) Bounds. Using the known range of unrevealed entries, we compute bounds that are always valid:

$$
\begin{array} { l } { { \displaystyle { \cal L } B _ { i } ^ { \mathrm { h a r d } } = \sum _ { t \in \mathcal { O } _ { i } } H _ { i , t } + \sum _ { t \in \mathcal { U } _ { i } } a _ { i , t } } , }  \\ { { \displaystyle { U B _ { i } ^ { \mathrm { h a r d } } = \sum _ { t \in \mathcal { O } _ { i } } H _ { i , t } + \sum _ { t \in \mathcal { U } _ { i } } b _ { i , t } } . } } \end{array}
$$

Variance-Adaptive Decision Radius. To adapt sampling to the variability of token interactions, we use an empirical Bernstein–Serfling style radius (Bardenet & Maillard, 2015):

where ${ \widehat { \sigma } } _ { i }$ is the empirical standard deviation over $\{ H _ { i , t } \} _ { t \in { \mathcal { O } } _ { i } }$ and $\rho _ { n _ { i } }$ is a finite-population correction satisfying $\rho _ { n _ { i } } \to 0$ as $n _ { i } \to T$ (definitions in Appendix A). This functional form has three useful properties in our setting: it scales with observed variability $( \widehat { \sigma } _ { i } )$ , shrinks roughly as $1 / \sqrt { n _ { i } }$ with additional reveals, and collapses to zero as a row becomes fully observed through $\rho _ { n _ { i } }$ . We treat $\alpha _ { \mathrm { e f } } \in ( 0 , 1 ]$ as a calibration parameter controlling conservativeness: $\alpha _ { \mathrm { e f } } = 1$ uses the unshrunk form, while $\alpha _ { \mathrm { e f } } < 1$ tightens the radius and improves the quality–coverage tradeoff in practice.

Hybrid Decision Interval. We combine deterministic hard bounds with the variance-adaptive decision radius:

$$
\begin{array} { r } { \mathrm { L C B } _ { i } = \operatorname* { m a x } \Big ( L B _ { i } ^ { \mathrm { h a r d } } , ~ \widehat { S } _ { i } - r _ { i } ^ { \mathrm { e f f } } \Big ) , } \\ { \mathrm { U C B } _ { i } = \operatorname* { m i n } \Big ( U B _ { i } ^ { \mathrm { h a r d } } , ~ \widehat { S } _ { i } + r _ { i } ^ { \mathrm { e f f } } \Big ) . } \end{array}
$$

Clipping to $[ L B _ { i } ^ { \mathrm { h a r d } } , U B _ { i } ^ { \mathrm { h a r d } } ]$ prevents excessive extrapolation from partial observations.

# 4.3. LUCB-Based Refinement Policy

We adopt the LUCB framework for Top- $K$ identification, summarized in Algorithm 1. Let $\widehat { \tau } _ { \kappa }$ be the $K$ documents with largest $\widehat { S } _ { i }$ , and define the weakest winner and strongest loser as

$$
i ^ { + } \in \arg \operatorname* { m i n } _ { i \in \widehat { \mathcal { T } } _ { K } } \mathrm { L C B } _ { i } , \qquad i ^ { - } \in \arg \operatorname* { m a x } _ { i \notin \widehat { \mathcal { T } } _ { K } } \mathrm { U C B } _ { i } .
$$

If $\mathrm { L C B } _ { i ^ { + } } \geq \mathrm { U C B } _ { i ^ { - } }$ , we terminate with a separated Top- $K$ set under the maintained decision bounds (as illustrated in Figure 1). Otherwise, we first pick the more ambiguous document

$$
i ^ { \star } = \arg \operatorname* { m a x } _ { i \in \{ i ^ { + } , i ^ { - } \} } \bigl ( \mathrm { U C B } _ { i } - \mathrm { L C B } _ { i } \bigr ) ,
$$

and then reveal one additional token for $i ^ { \star }$ using the dynamic $\epsilon$ -greedy strategy (Section 4.1): with probability $\epsilon$ we sample $t$ uniformly from $\mathcal { U } _ { i ^ { \star } }$ (exploration), otherwise we select

$$
t ^ { \star } = \arg \operatorname* { m a x } _ { t \in \mathcal { U } _ { i ^ { \star } } } \bigl ( b _ { i ^ { \star } , t } - a _ { i ^ { \star } , t } \bigr ) ,
$$

which targets the unrevealed token with the largest remaining deterministic uncertainty.

Uniform-within-row mode. In a non-adaptive variant where, once a document is selected, the next revealed token is sampled uniformly from the remaining unrevealed tokens in that row, setting $\alpha _ { \mathrm { e f } } = 1$ matches the conditions of the empirical Bernstein–Serfling bound (Appendix C). Empirically, the corresponding high-coverage endpoint attains exact agreement with full scoring (Fig. 8).

# 4.4. Practical Calibration

In practice, $\alpha _ { \mathrm { e f } }$ governs the aggressiveness of pruning: smaller values tighten the decision radius and reduce coverage, while larger values are more conservative. We therefore select $\alpha _ { \mathrm { e f } }$ based on a desired quality–coverage trade-off (Section 5.3). Unless stated otherwise, our default configuration uses dynamic $\epsilon$ -greedy refinement with an empirically calibrated $\alpha _ { \mathrm { e f } } < 1$ and reports both retrieval quality and achieved coverage.

# 5. Experiments

We evaluate Col-Bandit on five text retrieval datasets from BEIR (Thakur et al., 2021) using ColBERTv2 (Santhanam et al., 2022b) and Jina-ColBERT-v2 (Jha et al., 2024), and four multimodal datasets from REAL-MM-RAG (Wasserman et al., 2025) using Granite Vision Embedding 3.2 (Team, 2025a) multimodal embedding model. Dataset statistics are in Appendix Table 3.

Baselines We compare Col-Bandit against two baselines. The first is a naive random strategy, denoted as Doc-Uniform, which reveals MaxSim cells uniformly at random within each document (row) under a given coverage budget. The second is a greedy heuristic method, denoted as Doc-TopMargin, which reveals the MaxSim cells with the largest support (Section 3.2) within each row, subject to the same coverage budget. We describe the full configurations for two baselines in the appendix A.3 Algorithm 2, and 3.

# 5.1. Experimental Setup

Our evaluation follows the standard two-stage lateinteraction retrieval pipeline (Khattab & Zaharia, 2020). We evaluate all approaches at $K \in \{ 1 , 5 , 1 0 \}$ . In the first stage, an approximate nearest-neighbor (ANN) index is leveraged to retrieve a candidate set $\mathcal { D }$ for each query (in our experiments, we instantiate this stage using precomputed exact kNN per query token for reproducibility). In the second stage, the candidates are re-ranked using late interaction (e.g., MaxSim aggregation). In our evaluation, Col-Bandit operates in this second stage, adaptively revealing only a subset of MaxSim interactions within the query–document table. The first-stage retrieval provides informative bounds that can initialize Col-Bandit (Section 4.1). For each query

Table 1. Universal Efficiency Analysis (Text and Multimodal). We report the mean coverage budget (std)across BEIR and REAL-MM-RAG datasets required to achieve $90 \%$ (White) and $95 \%$ (Gray) Overlap $@ 1$ and Overlap $\ @ 5$ . Savings (vs. Full) is the compute reduction factor relative to full ColBERT reranking (i.e., $1 0 0 \% / \mathrm { M e a n } )$ ).   

<table><tr><td rowspan="3">Method</td><td colspan="4">Overlap@1</td><td colspan="4">Overlap@5</td></tr><tr><td>90%</td><td>95%</td><td>90%</td><td>95%</td><td>90%</td><td>95%</td><td>90%</td><td>95%</td></tr><tr><td colspan="2">Mean Coverage (std)</td><td colspan="2">Savings</td><td colspan="2">Mean Coverage (std)</td><td colspan="2">Savings</td></tr><tr><td colspan="9">ColBERTv2 (BEIR)</td></tr><tr><td>Doc-Uniform</td><td>65% (35.4)</td><td>71% (36.8)</td><td>1.5×</td><td>1.4×</td><td>98% (1.2)</td><td>100% (0.0)</td><td>1.0×</td><td>1.0×</td></tr><tr><td>Doc-TopMargin</td><td>56% (33.0)</td><td>63% (36.2)</td><td>1.8×</td><td>1.6×</td><td>79% (5.5)</td><td>91% (4.4)</td><td>1.3×</td><td>1.1×</td></tr><tr><td>Col-Bandit (Ours)</td><td>13% (10.2)</td><td>14% (10.7)</td><td>7.7×</td><td>7.1×</td><td>28% (6.3)</td><td>33% (8.3)</td><td>3.6×</td><td>3.1×</td></tr><tr><td colspan="9">Jina-ColBERT-V2 (BEIR)</td></tr><tr><td>Doc-Uniform</td><td>80% (37.6)</td><td>83% (35.8)</td><td>1.2×</td><td>1.2×</td><td>99% (1.2)</td><td>100% (0.0)</td><td>1.0×</td><td>1.0×</td></tr><tr><td>Doc-TopMargin</td><td>44% (26.2)</td><td>57% (36)</td><td>2.3×</td><td>1.7×</td><td>61% (8.6)</td><td>76% (7.0)</td><td>1.6×</td><td>1.3×</td></tr><tr><td>Col-Bandit (Ours)</td><td>11% (5.3)</td><td>14% (7.6)</td><td>9.1×</td><td>7.1×</td><td>26% (4.6)</td><td>34% (8.4)</td><td>3.8×</td><td>3.0×</td></tr><tr><td colspan="9">Granite Vision Embedding (REAL-MM-RAG)</td></tr><tr><td>Doc-Uniform</td><td>91% (9.5)</td><td>98% (3.9)</td><td>1.1×</td><td>1.0×</td><td>96% (0.0)</td><td>100% (0.0)</td><td>1.0×</td><td>1.0×</td></tr><tr><td>Doc-TopMargin</td><td>77% (12.4)</td><td>89% (8.6)</td><td>1.3×</td><td>1.1×</td><td>86% (3.5)</td><td>93% (2.5)</td><td>1.2×</td><td>1.1×</td></tr><tr><td>Col-Bandit (Ours)</td><td>16% (6.6)</td><td>18% (6.7)</td><td>6.3×</td><td>5.9×</td><td>31% (3.1)</td><td>41% (3.8)</td><td>3.2×</td><td>2.5×</td></tr></table>

token $\mathbf { q } _ { t }$

$$
a _ { i , t } = 0 , \quad b _ { i , t } = \left\{ { \begin{array} { l l l } { h ( d _ { i } , t ) } & { { \mathrm { i f ~ } } d _ { i } { \mathrm { ~ r e t r i e v e d ~ f o r ~ t o k e n ~ } } t } \\ { s _ { k ^ { \prime } } ^ { ( t ) } } & { { \mathrm { o t h e r w i s e } } } \end{array} } \right.
$$

where $h ( d _ { i } , t )$ is the actual MaxSim value computed during ANN retrieval (if $d _ { i }$ was retrieved for token $t$ ), and $s _ { k ^ { \prime } } ^ { ( t ) }$ is the similarity of the $k ^ { \prime }$ -th neighbor for token $t$ . These tokenlevel bounds translate into row-wise bounds for Col-Bandit’s confidence intervals (Eq. 10,11) enable faster convergence. Appendix A.1 details our two-stage retrieval pipeline.

Metrics. All results are measured relative to full lateinteraction scoring over the entire candidate set, which serves as the non-pruned reference. The ranking fidelity is measured by Overlap $@ K$ : Intersection between the approximate Top- $K$ set and the exact Top- $K$ set returned by full candidate set scoring.

$$
\mathbf { O v e r l a p @ } K = { \frac { | T _ { K } ^ { \star } ( Q ) \cap { \hat { T } } _ { K } ( Q ) | } { K } }
$$

Overlap $@ K$ measures how faithfully pruning methods recover the ranking produced by full late-interaction scoring. In addition, we evaluate retrieval effectiveness, which reflects end-task performance. We report standard IR metrics - Recall $@ K$ , $\mathbf { M R R } @ K$ , and $\mathrm { n D C G } @ K$ , computed against relevance labels. These metrics allow us to assess whether computational savings come at the cost of end-task quality. These perspectives answer different questions: the first evaluates approximation quality (can we reproduce Full Col-BERT cheaply?), while the second evaluates task quality (do we hurt retrieval performance?)

We evaluate all the methods along two complementary dimensions – quality and coverage. For visualization, we plot quality metrics ( $\mathbf { \dot { x } }$ -axis) against the resulting coverage $\gamma$ (y-axis). For Col-Bandit, operating points are generated by sweeping the relaxation parameter $\alpha _ { \mathrm { e f } } \in [ 1 0 ^ { - 3 } , 1 ]$ with fixed confidence $\delta = 0 . 0 1$ . For exploration, Col-Bandit employs $\epsilon$ -greedy2 with $\epsilon = 0 . 1$ . Baselines are evaluated at fixed coverage budgets $\gamma \in \{ 0 . 0 5 , 0 . 1 , \ldots , 1 . 0 \}$ .

Table 2. Retrieval effectiveness at different coverage levels on both REAL-MM-RAG and BEIR. Results are averaged across datasets and models. Full reranking at $100 \%$ coverage serves as the reference.   

<table><tr><td>Method</td><td>Coverage</td><td>Recall@5</td><td>nDCG@5</td><td>MRR@5</td></tr><tr><td>Full ColBERT</td><td>100%</td><td>0.66</td><td>0.58</td><td>0.61</td></tr><tr><td>Col-Bandit (Ours)</td><td>20%</td><td>0.60</td><td>0.54</td><td>0.57</td></tr><tr><td>Col-Bandit (Ours)</td><td>40%</td><td>0.65</td><td>0.57</td><td>0.60</td></tr><tr><td>Doc-TopMargin</td><td>40%</td><td>0.61</td><td>0.54</td><td>0.56</td></tr><tr><td>Doc-Uniform</td><td>40%</td><td>0.54</td><td>0.46</td><td>0.48</td></tr><tr><td colspan="5">Relative Retention at 20% Coverage (vs. Full ColBERT)</td></tr><tr><td>Col-Bandit (Ours)</td><td>−</td><td>90.9%</td><td>93.1%</td><td>93.4%</td></tr><tr><td colspan="5">Relative Retention at 40% Coverage (vs. Full ColBERT)</td></tr><tr><td>Col-Bandit (Ours)</td><td></td><td>98.8%</td><td>98.9%</td><td>99.1%</td></tr><tr><td>Doc-TopMargin</td><td></td><td>93.1%</td><td>92.3%</td><td>92.7%</td></tr><tr><td>Doc-Uniform</td><td>−</td><td>82.6%</td><td>79.1%</td><td>78.9%</td></tr></table>

# 5.2. Main Results

# Ranking Fidelity: Cost-Accuracy Trade-off.

We measure the cost–accuracy trade-off via Top- $K$ ranking recovery as a function of coverage $\gamma$ . Varying the relaxation parameter $\alpha _ { \mathrm { e f } }$ yields a tunable efficiency frontier (Fig. 2; summarized in Table 1). At matched coverage, Col-Bandit consistently attains higher ranking fidelity than all non-adaptive baselines. Table 1 reports the mean coverage required to reach $9 0 \%$ and $9 5 \%$ overlap at $K { = } 1$ and $K { = } 5$ (averaged over BEIR and REAL-MM-RAG; per-dataset results and additional plots for $K = 1 , 5 , 1 0$ are in Appendix B.1, 6, 7). Overall, Col-Bandit reaches target fidelity with substantially lower coverage, with the largest gains at small $K$ (Top-1) and still sizable savings at $K { = } 5$ These trends hold for both text-only retrievers (ColBERTv2, Jina-ColBERTv2) and multimodal embeddings (Granite Vision Embedding on REAL-MM-RAG), indicating that the adaptive reveal framework is model- and modality-agnostic.

![](images/62c732ff3f8c11b3c44af90235e8e1a71d13b1eb00b886d12f05b70d72101ec4.jpg)  
Figure 4. Exploration Strategy Ablation. Trade-off on Jina ColBERTv2 / HotPotQA. The dynamic $\epsilon$ -greedy policy (purple) consistently dominates static warm-up schedules (green), avoiding wasteful reveals on easy queries.

Retrieval Effectiveness: Impact on End-Task Performance. We test whether adaptive pruning harms retrieval by reporting Recall $\textcircled { a } 5$ , $\mathrm { n D C G } @ 5$ , and MRR $\textcircled { a } 5$ under different coverage budgets (Table 2; $K { = } 1$ in Appendix 8). Col-Bandit preserves relevance quality under substantial compute reduction (e.g., at $4 0 \%$ coverage it nearly matches full scoring), while heuristic baselines degrade more sharply. Even at $2 0 \%$ coverage, Col-Bandit remains competitive, showing graceful quality degradation as compute decreases.

# 5.3. Ablation Studies

Impact of Exploration Strategy. We compare our dynamic $\epsilon \cdot$ -greedy policy with a static Warm-up baseline that reveals a fixed fraction $\gamma$ upfront. As shown in Fig. 4, $\epsilon$ -greedy yields a better efficiency frontier by avoiding irreducible fixed costs on easy queries and allocating exploration only when rankings are ambiguous. We therefore use $\epsilon$ -greedy in all main experiments.

Benefit of ANN-Based Bounds. In realistic deployments, Col-Bandit can leverage bounds derived from the ANN retrieval stage (Section 5.1). However, Col-Bandit can also operate without external bounds, using only generic similaritymetric bounds (e.g., [0, 1] for normalized embeddings).

![](images/eb411f9dd8301c578bb5b34874af9e9f896a4be36e222ffb80634a567d4f35c6.jpg)  
Figure 5. Effect of ANN-derived bounds. Col-Bandit (purple) outperforms the corresponding baseline (gray) in both settings: with retrieval bounds (solid) and without (dashed). Granite Vision Embedding / TechSlides.

Figure 5 compares these settings (see Appendix B for additional datasets). Using ANN bounds consistently improves the accuracy-coverage trade-off, enabling Col-Bandit to achieve higher ranking fidelity at the same compute budget. For example, on the Granite Vision Embedding / TechSlides setting, achieving 0.9 Overlap $\textcircled { \omega } 5$ requires only $30 \%$ coverage when using ANN-derived bounds, compared to $50 \%$ for the generic-bounds variant. Importantly, even without ANNbased initialization, Col-Bandit still substantially outperforms Doc-Uniform (0.9 vs. 0.65 at $50 \%$ coverage), which similarly operates without ANN-derived bounds, demonstrating that the adaptive reveal strategy provides value beyond the availability of strong initial bounds.

# 6. Conclusion

We presented Col-Bandit, an adaptive framework for accelerating late-interaction reranking at query time by selectively revealing MaxSim computations until the Top- $K$ set stabilizes. Across BEIR and REAL-MM-RAG, Col-Bandit consistently exposes substantial redundancy in dense lateinteraction scoring, reducing MaxSim FLOPs by up to $5 \times$ while preserving high overlap with exhaustive reranking. A single calibration knob, $\alpha _ { \mathrm { e f } }$ (Eq. 12), provides a practical control over the quality–compute trade-off and yields strong Pareto frontiers against uniform and greedy reveal baselines. Col-Bandit is a drop-in reranking layer that requires no retraining or offline index changes, making it easy to deploy on top of standard search pipelines.

Limitations. Col-Bandit is designed for precision-oriented tasks with small $K$ ; as $K$ grows, more candidates cluster near the decision boundary, reducing efficiency gains. Our strongest empirical configuration uses adaptive token selection, for which the variance-based radius should be viewed as a calibrated decision heuristic rather than a formal certificate. Finally, our evaluation measures FLOP reductions; realizing wall-clock speedups requires batched implementations to amortize GPU kernel overheads.

Future Work. We plan to develop a batched implementation that reveals blocks of high-uncertainty cells simultaneously, enabling efficient parallel execution on modern GPU hardware.