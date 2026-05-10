# CITADEL: Conditional Token Interaction via Dynamic Lexical Routing for Efficient and Effective Multi-Vector Retrieval

Minghan $\mathbf { L i } ^ { 1 \ast }$ , Sheng-Chieh $\mathbf { L i n } ^ { 1 }$ , Barlas $\mathbf { 0 } \mathbf { g } \mathbf { u } \mathbf { z } ^ { 2 }$ , Asish Ghoshal2, Jimmy $\mathbf { L i n } ^ { 1 }$ , Yashar Mehdad2, Wen-tau $\mathbf { Y i h ^ { 2 } }$ , and Xilun Chen2†

University of Waterloo1, Meta AI2

{m692li,s269lin,jimmylin}@uwaterloo.ca

{barlaso,aghoshal,mehdad,scottyih,xilun}@meta.com

# Abstract

Multi-vector retrieval methods combine the merits of sparse (e.g. BM25) and dense (e.g. DPR) retrievers and have achieved state-ofthe-art performance on various retrieval tasks. These methods, however, are orders of magnitude slower and need much more space to store their indices compared to their singlevector counterparts. In this paper, we unify different multi-vector retrieval models from a token routing viewpoint and propose conditional token interaction via dynamic lexical routing, namely CITADEL, for efficient and effective multi-vector retrieval. CITADEL learns to route different token vectors to the predicted lexical “keys” such that a query token vector only interacts with document token vectors routed to the same key. This design significantly reduces the computation cost while maintaining high accuracy. Notably, CITADEL achieves the same or slightly better performance than the previous state of the art, ColBERT-v2, on both in-domain (MS MARCO) and out-of-domain (BEIR) evaluations, while being nearly 40 times faster. Code and data are available at https://github. com/facebookresearch/dpr-scale.

# 1 Introduction

The goal of information retrieval (Manning et al., 2008) is to find a set of related documents from a large data collection given a query. Traditional bagof-words systems (Robertson and Zaragoza, 2009; Lin et al., 2021a) calculate the ranking scores based on the query terms appearing in each document, which have been widely adopted in many applications such as web search (Nguyen et al., 2016; Noy et al., 2019) and open-domain question answering (Chen et al., 2017; Lee et al., 2019). Recently, dense retrieval (Karpukhin et al., 2020) based on pre-trained language models (Devlin et al., 2019;

![](images/9a1a2cb8e28aa2ea4d2ddaefd080f9b16b1a55485f52aab5be40f5bf0315ab0d.jpg)  
Figure 1: GPU latency vs ranking performance $( \mathbf { M } \mathbf { R } \mathbf { R } @ 1 0 )$ on MS MARCO passages with an A100 GPU. The circle size represents the index storage on disk. All models are trained without hard-negative mining, distillation, or further pre-training.

Liu et al., 2019) has been shown to be very effective. It circumvents the term mismatch problem in bag-of-words systems by encoding the queries and documents into low-dimensional embeddings and using their dot product as the similarity score (Figure 2a). However, dense retrieval is less robust on entity-heavy questions (Sciavolino et al., 2021) and out-of-domain datasets (Thakur et al., 2021), therefore calling for better solutions (Formal et al., 2021b; Gao and Callan, 2022).

In contrast, another family of retrieval models encodes input tokens into vectors first and performs fine-grained token interaction between queries and documents, known as multi-vector retrieval. Multi-vector retrieval has shown strong performance on both in-domain and out-of-domain evaluations. Among them, ColBERT (Khattab and Zaharia, 2020) is arguably the most celebrated method that has been the state of the art on multiple datasets so far. However, its wider application is hindered by its large index size and high retrieval latency. This problem results from the redundancy in the token interaction of ColBERT where many tokens might not contribute to the sentence semantics at all. To improve this, COIL (Gao et al., 2021a) imposes an exact match constraint on ColBERT for conditional token interaction, where only token embeddings with the same token id could interact with each other. Although reducing the latency, the word mismatch problem reoccurs and the model may fail to match queries and passages that use different words to express the same meaning.

![](images/400cdaece4aa83abdd955736077eb835753a53ab6201fd0ae10548107e6c3d46.jpg)  
Figure 2: A unified token routing view over different multi-vector and single-vector retrieval models. The cylinder in (d) represents the lexical key for token routing. The grey tokens in (c) and (d) represent the tokens that do not contribute to the final similarity. For (d), CITADEL routes “light”, “theory”, and “relativity” to the “Einstein” key to avoid term mismatch using a learned routing function.

In this paper, we first give a unified view of existing multi-vector retrieval methods based on token routing (Section 2), providing a new lens through which we expose the limitations of current models. Under the token routing view, ColBERT could be seen as all-to-all routing, where each query token exhaustively interacts with all passage tokens (Figure 2b). COIL, on the other hand, could be seen as static lexical routing using an exact match constraint, as each query token only interacts with the passage tokens that have the same token id as the query token (Figure 2c). As mentioned above, this exact match heuristic improves the latency over ColBERT but negatively affects the accuracy.

In contrast, we propose a novel conditional token interaction method using dynamic lexical routing called CITADEL as shown in Figure 2d. Instead of relying on static heuristics such as exact match, we train our model to dynamically moderate token interaction so that each query token only interacts with the most relevant tokens in the passage. This is achieved by using a lexical router, trained endto-end with the rest of the model, to route each contextualized token embedding to a set of activated lexical “keys” in the vocabulary. In this way, each query token embedding only interacts with the passage token embeddings that have the same activated key, which is dynamically determined during computation. As we shall see in Section 5.1, this learning-based routing does not lose any accuracy compared to the exhaustive all-to-all routing while using even fewer token interactions than COIL (Section 3.4), leading to a highly effective and efficient retriever.

Experiments on MS MARCO passages (Nguyen et al., 2016) and TREC DeepLearning 2019/2020 show that CITADEL achieves the same level of accuracy as ColBERT-v2 and sometimes even slightly better. We further test CITADEL on BEIR (Thakur et al., 2021) for out-of-domain evaluation and CITADEL still manages to keep up with ColBERTv2 (Santhanam et al., 2022b) which is the current state of the art. As for the latency, CITADEL can yield an average latency of $3 . 2 1 ~ \mathrm { m s }$ /query on MS MARCO passages using an A100 GPU, which is nearly $4 0 \times$ faster than ColBERT-v2. By further combining with product quantization, CITADEL’s index only takes $1 3 . 3 \mathrm { G B }$ on MS MARCO passages and reduces the latency to $0 . 9 \mathrm { m s } /$ query as shown in Figure 1.

# 2 A Unified Token Routing View of Multi-Vector Retrievers

In this section, we outline a unified view for understanding various neural retrievers, especially multi-vector ones, using the concept of token routing that dictates which query and passage tokens can interact with each other.

# 2.1 Single-Vector Retrieval

Given a collection of documents and a set of queries, single-vector models (Karpukhin et al., 2020; Izacard et al., 2022) use a bi-encoder structure where its query encoder $\eta _ { Q } ( \cdot )$ and document encoder $\eta _ { D } ( \cdot )$ are independent functions that map the input to a low-dimensional vector. Specifically, the similarity score $s$ between the query $q$ and document $d$ is defined by the dot product between their encoded vectors $v _ { q } = \eta _ { Q } ( q )$ and $v _ { d } = \eta _ { D } ( d )$ :

$$
s ( q , d ) = v _ { q } ^ { T } v _ { d } .
$$

As all the token embeddings are pooled (e.g., mean pooling, [CLS] pooling) before calculating the similarity score, no routing is committed as shown in Figure 2a.

# 2.2 Multi-Vector Retrieval

ColBERT (Khattab and Zaharia, 2020) proposes late interaction between the tokens in a query $\textit { q } = \{ q _ { 1 } , q _ { 2 } , \cdot \cdot \cdot , q _ { N } \}$ and a document $d \_ =$ $\{ d _ { 1 } , d _ { 2 } , \dotsb , d _ { M } \}$ :

$$
s ( \boldsymbol { q } , d ) = \sum _ { i = 1 } ^ { N } \operatorname* { m a x } _ { j } v _ { \boldsymbol { q } _ { i } } ^ { T } v _ { \boldsymbol { d } _ { j } } ,
$$

where $v _ { q _ { i } }$ and $v _ { d _ { j } }$ denotes the last-layer contextualized token embeddings of BERT. This is known as the MaxSim operation where it first takes the max over the dot products between each query and document token embedding, and then sums over the score for each query token. As this operation needs to exhaustively compare each query token to all document tokens, we refer to this as all-toall routing as shown in Figure 2b. The latency of ColBERT is inflated by the redundancy in the allto-all routing, as many tokens do not contribute to the sentence semantics. This also drastically increases the storage, requiring complex engineering schemes and low-level optimization to make it more practical (Santhanam et al., 2022b,a).

Another representative multi-vector approach known as COIL (Gao et al., 2021a) proposes an exact match constraint on the MaxSim operation where only the embeddings with the same token id could interact with each other. Let $\mathcal { T } _ { i } = \{ j \ \vert \ d _ { j } =$ $q _ { i } , 1 \leq j \leq M \}$ be the subset of document tokens $\{ d _ { j } \} _ { j = 1 } ^ { M }$ that have the same token ID as query token $q _ { i }$ , then we have:

$$
s ( \boldsymbol { q } , d ) = \sum _ { i = 1 } ^ { N } \operatorname* { m a x } _ { j \in \mathcal { T } _ { i } } v _ { \boldsymbol { q } _ { i } } ^ { T } v _ { \boldsymbol { d } _ { j } } ,
$$

It could be further combined with Equation (1) to improve the effectiveness if there’s no word overlap between the query and documents.

$$
s ( \boldsymbol { q } , \boldsymbol { d } ) = v _ { \boldsymbol { q } } ^ { T } v _ { d } + \sum _ { i = 1 } ^ { N } \operatorname* { m a x } _ { \boldsymbol { j } \in \mathcal { T } _ { i } } v _ { \boldsymbol { q } _ { i } } ^ { T } v _ { \boldsymbol { d } _ { j } } ,
$$

We refer to this token interaction as static lexical routing as shown in Figure 2c. As mentioned in Section 1, the word mismatch problem could happen if $\mathcal { I } _ { i } = \emptyset$ for all $q _ { i }$ , which affects the retrieval accuracy. Moreover, common tokens such as “the” and “a” will be frequently routed, which will create much larger token indices during indexing compared to those rare words. This bottlenecks the latency as COIL needs to frequently search overly large token indices.

# 3 The CITADEL Method

# 3.1 Dynamic Lexical Routing

Instead of using the wasteful all-to-all routing or the inflexible heuristics-based static routing, we would like our model to dynamically select which query and passage tokens should interact with each other based on their contextualized representation, which we refer to as dynamic lexical routing. Formally, the routing function (or router) routes each token to a set of lexical keys in the vocabulary, and is defined as $\phi ~ : ~ \mathbb { R } ^ { c } ~  ~ \mathbb { R } ^ { | \mathcal { V } | }$ where $c$ is the embedding dimension and $\nu$ is the lexicon of keys. For each contextualized token embedding, the router predicts a scalar score for each key in the lexicon indicating how relevant each token is to that key. Given a query token embedding $v _ { q _ { i } }$ and a document token vector $v _ { d _ { j } }$ , the token level router representations are $w _ { q _ { i } } \stackrel { \cdot } { = } \phi ( v _ { q _ { i } } )$ and $w _ { d _ { j } } = \phi ( v _ { d _ { j } } )$ , respectively. The elements in the router representations are then sorted in descending order and truncated by selecting the top-$K$ query keys and top- $L$ document keys, which are $\{ E _ { q _ { i } } ^ { ( 1 ) } , E _ { q _ { i } } ^ { ( 2 ) } , \cdot \cdot \cdot , E _ { q _ { i } } ^ { ( K ) } \}$ and $\{ E _ { d _ { j } } ^ { ( 1 ) } , E _ { d _ { j } } ^ { ( 2 ) } , \cdot \cdot \cdot , E _ { d _ { j } } ^ { ( L ) } \}$ for $q _ { i }$ and $d _ { j }$ , respectively. In practice, we use $K { = } 1$ and $L { = } 5$ as the default option which will be discussed in Section 3.5 and Section 6. The corresponding routing weights for $q _ { i }$ and $d _ { j }$ are $\{ w _ { q _ { i } } ^ { ( 1 ) } , w _ { q _ { i } } ^ { ( 2 ) } , \cdot \cdot \cdot , w _ { q _ { i } } ^ { ( K ) } \}$ and $\{ w _ { d _ { j } } ^ { ( 1 ) } , w _ { d _ { j } } ^ { ( 2 ) } , \cdot \cdot \cdot , w _ { d _ { j } } ^ { ( L ) } \}$ w(L)dj }, respectively. The final similarity score is similar to Equation (3), but we substitute the static lexical routing subset $\mathcal { I } _ { i }$ with a dynamic key set predicted by the router: $\mathcal { E } _ { i } ^ { ( k ) } = \{ j , l \ | \ E _ { d _ { j } } ^ { ( l ) } = E _ { q _ { i } } ^ { ( k ) } , 1 \le j \ \le$ M, 1 ≤ l ≤ L} for each key E(k)qi o f the query token :

$$
s ( \boldsymbol { q } , \boldsymbol { d } ) = \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \operatorname* { m a x } _ { \boldsymbol { j } , \boldsymbol { l } \in \mathcal { E } _ { i } ^ { ( k ) } } ( \boldsymbol { w } _ { \boldsymbol { q } _ { i } } ^ { ( k ) } \cdot \boldsymbol { v } _ { \boldsymbol { q } _ { i } } ) ^ { T } ( \boldsymbol { w } _ { \boldsymbol { d } _ { j } } ^ { ( l ) } \cdot \boldsymbol { v } _ { \boldsymbol { d } _ { j } } ) ,
$$

Optionally, all [CLS] tokens can be routed to an additional semantic key to complement our learned lexical routing. We then follow DPR (Karpukhin et al., 2020) to train the model contrastively. Given a query $q$ , a positive document $d ^ { + }$ , and a set of negative documents $D ^ { - }$ , the constrastive loss is:

$$
\mathcal { L } _ { \mathrm { e } } = - \log \frac { \exp ( s ( q , d ^ { + } ) ) } { \exp ( s ( q , d ^ { + } ) ) + \displaystyle \sum _ { d ^ { - } \in D ^ { - } } \exp ( s ( q , d ^ { - } ) ) } ,
$$

such that the distance from the query to the positive document $d ^ { + }$ is smaller than the query to the negative document $d ^ { - }$ .

# 3.2 Router Optimization

To train the router representation $\phi ( q )$ and $\phi ( d )$ , we adopt a contrastive loss such that the number of overlapped keys between a query and documents are large for positive $( q , d ^ { + } )$ pairs and small for negative pairs $( q , d ^ { - } )$ . We first pool the router representation for each query and document over the tokens. Given a sequence of token-level router representations $\{ \phi ( v _ { 1 } ) , \phi ( v _ { 2 } ) , \cdot \cdot \cdot , \phi ( v _ { M } ) \}$ , the sequence-level representation is defined as:

$$
\Phi = \operatorname* { m a x } _ { j = 1 } ^ { M } \phi ( v _ { j } ) .
$$

Similar to (Formal et al., 2021a), We find max pooling works the best in practice compared to other pooling methods. Subsequently, the contrastive loss for training the router is:

$$
\mathcal { L } _ { \mathrm { r } } = - \log \frac { \exp ( \Phi _ { q } ^ { T } \Phi _ { d ^ { + } } ) } { \exp ( \Phi _ { q } ^ { T } \Phi _ { d ^ { + } } ) + \displaystyle \sum _ { d ^ { - } \in D ^ { - } } \exp ( \Phi _ { q } ^ { T } \Phi _ { d ^ { - } } ) } ,
$$

# 3.3 Sparsely Activated Router Design

Softmax activation is commonly used for computing the routing weights in conditional computation models (Fedus et al., 2022; Mustafa et al., 2022). However, softmax often yields a small probability over a large number of dimensions (in our case, about 30,000) and the product of two probability values are even smaller, which makes it not suitable for yielding the routing weights $w _ { q _ { i } } ^ { ( k ) }$ and $w _ { d _ { j } } ^ { ( l ) }$ in Equation (5) as the corresponding gradients are too small. Instead, we use the activation from SPLADE (Formal et al., 2021b,a) to compute the router representation for a token embedding $v _ { j }$ :

$$
\phi ( v _ { j } ) = \log ( 1 + \mathrm { R e L U } ( W ^ { T } v _ { j } + b ) ) ,
$$

where $W$ and $b$ are the weights and biases of the Masked Language Modeling (MLM) layer of BERT. The SPLADE activation brings extra advantages as the ReLU activation filters irrelevant keys while the log-saturation suppresses overly large “wacky” weights (Mackenzie et al., 2021).

# 3.4 Regularization for Routing

$\ell _ { 1 }$ Regularization. Routing each token to more than one key increases the overall size of the index. Therefore, we propose to use $\ell _ { 1 }$ regularization on the router representation to encourage the router to only keep the most meaningful token interaction by pushing more routing weights to 0:

$$
\mathcal { L } _ { \mathrm { s } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \sum _ { j = 1 } ^ { T } \sum _ { k = 1 } ^ { | \mathcal { V } | } \phi ( v _ { i j } ) ^ { ( k ) } ,
$$

where $| \nu |$ is the number of keys, $B$ is the batch size, and $T$ is the sequence length. As shown in Figure 5, CITADEL has a sparsely activated set of keys, by routing important words to multiple lexical keys while ignoring many less salient words, leading to effective yet efficient retrieval.

Load Balancing. As mentioned in Section 2.2, the retrieval latency of COIL is bottlenecked by frequently searching overly large token indexes. This results from the static lexical routing where common “keys” have a larger chance to be activated, which results in large token indices during indexing. Therefore, a vital point for reducing the latency of multi-vector models is to evenly distribute each token embedding to different keys. Inspired by Switch Transformers (Fedus et al., 2022), we propose to minimize the load balancing loss that approximates the expected “evenness” of the number of tokens being routed to each key:

$$
\mathcal { L } _ { \mathfrak { b } } = \sum _ { k = 1 } ^ { | \mathcal { V } | } f _ { k } \cdot p _ { k } ,
$$

$p _ { k }$ is the batch approximation of the marginal probability of a token embedding being routed to the $k$ -th key:

$$
p _ { k } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \sum _ { j = 1 } ^ { T } \frac { \exp ( W _ { k } ^ { T } v _ { i j } + b _ { k } ) } { \sum _ { k ^ { \prime } } \exp ( W _ { k ^ { \prime } } ^ { T } v _ { i j } + b _ { k ^ { \prime } } ) } ,
$$

where $W$ and $b$ are the weights and bias of the routing function in Equation (9) and $v _ { i j }$ is the $j$ -th token embedding in sample $i$ of the batch. $f _ { k }$ is the batch approximation of the total number of tokens being dispatched to the $k$ -th key:

$$
f _ { k } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \sum _ { j = 1 } ^ { T } \mathbb { I } \{ \operatorname { a r g m a x } ( p _ { i j } ) = k \} ,
$$

where $p _ { i j } = \mathrm { s o f t m a x } ( W ^ { T } v _ { i j } + b )$ . Finally, we obtain the loss for training CITADEL:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { e } } + \mathcal { L } _ { \mathrm { r } } + \alpha \cdot \mathcal { L } _ { \mathrm { b } } + \beta \cdot \mathcal { L } _ { \mathrm { s } } ,
$$

where $\alpha \geq 0$ and $\beta \geq 0$ . The $\ell _ { 1 }$ and load balancing regularization is applied on both queries and documents.

# 3.5 Inverted Index Retrieval

As we store the token embeddings according to their lexical keys, this is essentially inverted index retrieval like BM25 but instead of a scalar for each token, we use a low-dimensional vector and the doc product as the term relevance.

Indexing and Post-hoc Pruning. To further reduce index storage, we can prune the vectors with routing weights less than a certain threshold $\tau$ after training. For a key $E$ in the lexicon $\nu$ , the token index $\mathcal { T } _ { E }$ consists of token embeddings $v _ { d _ { j } }$ and the routing weight $w _ { d _ { j } } ^ { E }$ for all documents $d$ in the corpus $\mathcal { C }$ is:

$$
\mathcal { T } _ { E } = \{ w _ { d _ { j } } ^ { E } \cdot v _ { d _ { j } } \ | w _ { d _ { j } } ^ { E } > \tau , 1 \leq j \leq M , \forall d \in \mathcal { C } \} .
$$

We will discuss the impact of post-hoc pruning in Section 5.3, where we find that post-hoc pruning can reduce the index size by $3 \times$ without significant accuracy loss. The final search index is defined as $\mathcal { T } = \{ \mathcal { T } _ { E } | E \in \mathcal { V } \}$ , where the load-balancing loss in Equation (11) will encourage the size distribution over $\mathcal { T } _ { E }$ to be as even as possible. In practice, we set the number of maximal routing keys of each token to 5 for the document and 1 for the query. The intuition is that the documents usually contain more information and need more key capacity, which is detailedly discussed in Section 6. Moreover, we find that using more than 1 routing key for the query does not improve the accuracy but increases the latency rapidly.

Token Retrieval. Given a query $q$ , CITADEL first encodes it into a sequence of token vectors $\{ v _ { q _ { i } } \} _ { i = 1 } ^ { N }$ , and then route each vector to its top-1 key $E$ with a routing weight $w _ { q _ { i } } ^ { E }$ . The final representation wEq · $w _ { q _ { i } } ^ { E } \cdot v _ { q _ { i } }$ will be sent to the corresponding token index $\mathcal { T } _ { E }$ for vector search. The final ranking list will be merged from each query token’s document ranking according to Equation (5).

# 4 Experiments

# 4.1 MS MARCO Passages Retrieval

We evaluate MS MARCO passages (Nguyen et al., 2016) and its shared tasks, TREC DeepLearning (DL) 2019/2020 passage ranking tasks (Craswell et al., 2020). The MS MARCO passages corpus has around 8.8 million passages with an average length of 60 words. TREC DL 2019 and 2020 contain 43 and 54 test queries whose relevance sets are densely labelled with scores from 0 to 4. Following ColBERT-v2 (Santhanam et al., 2022b) and SPLADE-v2 (Formal et al., 2021a), we train CITADEL and other baseline models on MS MARCO passages and report the results on its devsmall set and TREC DL 2019/2020 test queries. The evaluation metrics are MRR $@ 1 0$ , $\mathrm { n D C G } @ 1 0$ , and Recall $@ 1 0 0 0$ (i.e., ${ \mathrm { R @ 1 K } }$ . We provide a detailed implementation of CITADEL and other baselines in Appendix A.1, A.2, and A.3.

Table 1: In-domain evaluation on MS MARCO passages and TREC DL 2019/2020. CITADEL $^ +$ is trained with cross-encoder distillation and hard-negative mining. The red region means CITADEL/CITADEL $^ +$ is better than the method while the green region means that there’s no statistical significance $( p \ > \ 0 . 0 5 )$ . “ $\mathbf { \nabla } \times \mathbf { \vec { \mathbf { \nabla } } } \mathbf { \vec { \mathbf { \nabla } } }$ means not implemented and “-” means not practical to evaluate on a single CPU.   

<table><tr><td rowspan="2">Models</td><td colspan="2">MARCO Dev</td><td colspan="2">TREC DL19</td><td colspan="2">TREC DL20</td><td colspan="2">Index Storage</td><td colspan="3">Latency (ms/query)</td></tr><tr><td>MRR@10 R@1k</td><td></td><td>nDCG@10</td><td>R@1k</td><td>nDCG@10</td><td>R@ 1k</td><td>Disk (GB)</td><td>Factor1</td><td>Encode (GPU)</td><td>Search (GPU)</td><td>Search (CPU)</td></tr><tr><td colspan="10">Models trained with only BM25 hard negatives</td></tr><tr><td>BM25</td><td>0.188</td><td>0.858</td><td>0.506</td><td>0.739</td><td>0.488</td><td>0.733</td><td>0.67</td><td>×0.22</td><td>×</td><td>×</td><td>40.1</td></tr><tr><td>DPR-128</td><td>0.285</td><td>0.937</td><td>0.576</td><td>0.702</td><td>0.603</td><td>0.757</td><td>4.33</td><td>×1.42</td><td>7.09</td><td>0.63</td><td>430</td></tr><tr><td>DPR-768</td><td>0.319</td><td>0.941</td><td>0.611</td><td>0.742</td><td>0.591</td><td>0.796</td><td>26.0</td><td>×8.52</td><td>7.01</td><td>1.28</td><td>2015</td></tr><tr><td>SPLADE</td><td>0.340</td><td>0.965</td><td>0.683</td><td>0.813</td><td>0.671</td><td>0.823</td><td>2.60</td><td>×0.85</td><td>7.13</td><td>×</td><td>475</td></tr><tr><td>COIL-tok</td><td>0.350</td><td>0.964</td><td>0.660</td><td>0.809</td><td>0.679</td><td>0.825</td><td>52.5</td><td>× 17.2</td><td>10.7</td><td>46.8</td><td>1295</td></tr><tr><td>COIL-full</td><td>0.353</td><td>0.967</td><td>0.704</td><td>0.835</td><td>0.688</td><td>0.841</td><td>78.5</td><td>×25.7</td><td>10.8</td><td>47.9</td><td>3258</td></tr><tr><td>ColBERT CITADEL</td><td>0.360</td><td>0.968</td><td>0.694</td><td>0.830</td><td>0.676</td><td>0.837</td><td>154</td><td>×50.5</td><td>10.9</td><td>178</td><td>-</td></tr><tr><td></td><td>0.362</td><td>0.975</td><td>0.687</td><td>0.829</td><td>0.661</td><td>0.830</td><td>78.3</td><td>×25.7</td><td>10.8</td><td>3.95</td><td>520</td></tr><tr><td colspan="10">Models trained with further pre-training/hard-negative mining/distillation</td><td></td><td></td></tr><tr><td>coCondenser</td><td>0.382</td><td>0.984</td><td>0.674</td><td>0.820</td><td>0.684</td><td>0.839</td><td>26.0</td><td>×8.52</td><td>7.01</td><td>1.28</td><td>2015</td></tr><tr><td>SPLADE-v2</td><td>0.368</td><td>0.979</td><td>0.729</td><td>0.865</td><td>0.718</td><td>0.890</td><td>4.12</td><td>× 1.35</td><td>7.13</td><td>×</td><td>2710</td></tr><tr><td>ColBERT-v2</td><td>0.397</td><td>0.985</td><td>0.744</td><td>0.882</td><td>0.750</td><td>0.894</td><td>29.0</td><td>×9.51</td><td>10.9</td><td>122</td><td>3275</td></tr><tr><td>ColBERT-PLAID2</td><td>0.397</td><td>0.984</td><td>0.744</td><td>0.882</td><td>0.749</td><td>0.894</td><td>22.1</td><td>×7.25</td><td>10.9</td><td>55.0</td><td>370</td></tr><tr><td>CITADEL+</td><td>0.399</td><td>0.981</td><td>0.703</td><td>0.830</td><td>0.702</td><td>0.859</td><td>81.3</td><td>×26.7</td><td>10.8</td><td>3.21</td><td>635</td></tr></table>

1 Factor: Ratio of index size to plain text size. 2 The PLAID implementation of ColBERT contains complex engineering schemes and low-level optimization such as centroid interaction and fast kernels.

<table><tr><td>Methods</td><td>AA</td><td>CF</td><td>DB</td><td>Fe</td><td>FQ</td><td>HQ</td><td>NF</td><td>NQ</td><td>Qu</td><td>SF</td><td>SD</td><td>TC</td><td>T2</td><td>Avg.</td></tr><tr><td colspan="10">Models trained with only BM25 hard negatives</td><td colspan="7"></td></tr><tr><td>BM25</td><td>0.315</td><td>0.213</td><td>0.313</td><td>0.753</td><td>0.236</td><td>0.603</td><td>0.325</td><td>0.329</td><td>0.789</td><td>0.665</td><td>0.158</td><td></td><td>0.656</td><td>0.367</td><td>0.440</td></tr><tr><td>DPR-768</td><td>0.323</td><td>0.167</td><td>0.295</td><td>0.651</td><td>0.224</td><td>0.441</td><td>0.244</td><td>0.410</td><td>0.75</td><td>0.479</td><td></td><td>0.103</td><td>0.604</td><td>0.185</td><td>0.375</td></tr><tr><td>SPLADE</td><td>0.445</td><td>0.201</td><td>0.370</td><td>0.740</td><td>0.289</td><td>0.640</td><td>0.322</td><td>0.469</td><td>0.834</td><td>0.633</td><td></td><td>0.149</td><td>0.661</td><td>0.201</td><td>0.453</td></tr><tr><td>COIL-full</td><td>0.295</td><td>0.216</td><td>0.398</td><td>0.840</td><td>0.313</td><td>0.713</td><td>0.331</td><td>0.519</td><td>0.838</td><td>0.707</td><td></td><td>0.155</td><td>0.668</td><td>0.281</td><td>0.483</td></tr><tr><td>ColBERT</td><td>0.233</td><td>0.184</td><td>0.392</td><td>0.771</td><td>0.317</td><td>0.593</td><td>0.305</td><td>0.524</td><td>0.854</td><td></td><td>0.671</td><td>0.165</td><td>0.677</td><td>0.202</td><td>0.453</td></tr><tr><td>CITADEL</td><td>0.503</td><td>0.191</td><td>0.406</td><td>0.784</td><td>0.298</td><td>0.653</td><td>0.324</td><td>0.510</td><td>0.844</td><td></td><td>0.674</td><td>0.152</td><td>0.687</td><td>0.294</td><td>0.486</td></tr><tr><td colspan="10">Models with further pre-training/hard-negative mining/distillation</td><td colspan="7"></td></tr><tr><td>coCondenser</td><td>0.440</td><td>0.133</td><td>0.347</td><td>0.511</td><td>0.281</td><td>0.533</td><td>0.319</td><td>0.467</td><td></td><td>0.863</td><td>0.591</td><td>0.130</td><td>0.708</td><td>0.143</td><td>0.420</td></tr><tr><td>SPLADE-v2</td><td>0.479</td><td>0.235</td><td>0.435</td><td>0.786</td><td>0.336</td><td>0.684</td><td>0.334</td><td>0.521</td><td>0.838</td><td>0.693</td><td></td><td>0.158</td><td>0.710</td><td>0.272</td><td>0.499</td></tr><tr><td>ColBERT-v2</td><td>0.463</td><td>0.176</td><td>0.446</td><td>0.785</td><td>0.356</td><td>0.667</td><td>0.338</td><td>0.562</td><td>0.854</td><td></td><td>0.693</td><td>0.165</td><td>0.738</td><td>0.263</td><td>0.500</td></tr><tr><td>CITADEL+</td><td>0.490</td><td>0.181</td><td>0.420</td><td>0.747</td><td>0.332</td><td>0.652</td><td>0.337</td><td>0.539</td><td>0.852</td><td></td><td>0.695</td><td>0.147</td><td>0.680</td><td>0.340</td><td>0.493</td></tr><tr><td>CITADEL+ (w/o reg.)</td><td>0.511</td><td>0.182</td><td>0.422</td><td>0.765</td><td>0.330</td><td>0.664</td><td>0.337</td><td>0.540</td><td>0.853</td><td></td><td>0.690</td><td>0.159</td><td>0.715</td><td>0.342</td><td>0.501</td></tr></table>

Table 2: Out-of-domain evaluation on BEIR benchmark. nDCG $@ 1 0$ score is reported. Dataset Legend (Chen et al., 2022): TC $\varprojlim =$ TREC-COVID, NF=NFCorpus, $\mathrm { N Q = } ]$ NaturalQuestions, $\mathrm { H Q = }$ HotpotQA, FQ=FiQA, AA $\underline { { \underline { { \mathbf { \Pi } } } } }$ ArguAna, ${ \bf T } 2 { = } ^ { \prime }$ Touché-2020 (v2), Qu=Quora, DB $\vDash$ DBPedia, SD=SCIDOCS, Fe=FEVER, $\mathrm { C F } { = }$ Climate-FEVER, $\mathrm { S F = }$ SciFact.

Table 1 shows the in-domain evaluation results on MS MARCO passage and TREC DL 2019/2020. We divide the models into two classes: ones trained with only labels and BM25 hard negatives and the others trained with further pretraining (Gao and Callan, 2022), hard negative mining (Xiong et al., 2021), or distillation from a cross-encoder1. CITADEL is trained only with BM25 hard negatives, while CITADEL $^ +$ is trained with cross-encoder distillation and one-round hard negative mining. The default pruning threshold is $\tau = 0 . 9$ . As shown in Section 5.3, $\tau$ can be adjusted to strike different balances between latency, index size and accuracy. In both categories, CITADEL/CITADEL+ outperforms the baseline models on the MS MARCO passages dev set and greatly reduces the search latency on both GPU and CPU. For example, CITADEL+ achieves an average latency of $3 . 2 1 \mathrm { m s } /$ query which is close to DPR-768 (1.28 ms/query) on GPU, while having a $2 5 \%$ higher MRR $@ 1 0$ score. CITADEL also maintains acceptable index sizes on disk, which can be further reduced using product quantization (Section 6). Although not able to outperform several baselines on TREC DL 2019/2020, we perform t-test $( \mathfrak { p } <$ 0.05) on CITADEL and CITADEL+ against other state-of-the-art baselines in their sub-categories and show there is no statistical significance. The inconsistency is probably due to the small test set of TREC DL as it is very expensive to label hundreds of passages paired with each query.

Table 3: Maximal number of dot products per query on MS MARCO passages. # DP: number of dot products.   

<table><tr><td>Models</td><td>MRR@10</td><td>#DP ×106</td></tr><tr><td>ColBERT</td><td>0.360</td><td>4213</td></tr><tr><td>COIL-full</td><td>0.353</td><td>45.6</td></tr><tr><td>CITADEL</td><td>0.362</td><td>10.5</td></tr><tr><td>DPR-128</td><td>0.285</td><td>8.84</td></tr></table>

# 4.2 BEIR: Out-of-Domain Evaluation

We evaluate on BEIR benchmark (Thakur et al., 2021) which consists of a diverse set of 18 retrieval tasks across 9 domains. Following previous works (Santhanam et al., 2022b; Formal et al., 2021a), we only evaluate 13 datasets for license reasons. Table 2 shows the zero-shot evaluation results on BEIR. Without any pre-training or distillation, CITADEL manages to outperform all baselines in their sub-categories in terms of the average score. Compared with the distilled/pre-trained models, CITADEL+ still manages to achieve comparable performance. Interestingly, we find that if no regularization like load balancing and L1 is applied during training, CITADEL+ can reach a much higher average score that even outperforms ColBERT-v2. Our conjecture is that the regularization reduces the number of token interactions and the importance of such token interaction is learned from training data. It is hence not surprising that the more aggressively we prune token interaction, the more likely that it would hurt out-of-domain accuracy that’s not covered by the training data.

# 5 Performance Analysis

# 5.1 Number of Token Interactions

The actual latency is often impacted by engineering details and therefore FLOPS is often considered for comparing efficiency agnostic to the actual implementation. In our case, however, FLOPS is impacted by the vector dimension in the nearest neighbour search which is different across models. Therefore, we only compare the maximal number of dot products needed as a proxy for token interaction per query during retrieval as shown in Table 3. The number of dot products per query in CITADEL with pruning threshold $\tau = 0 . 9$ is comparable to DPR-128 and much lower than ColBERT and COIL, which is consistent with the latency numbers in Table 1. The reason is that CITADEL has a balanced and small inverted index credited to the $\ell _ { 1 }$ regularization and the load balancing loss.

![](images/0beb175ccf72715e3e56416d1a1c1385c7f301764ce582134df7c968c3285289.jpg)  
Figure 3: Normalized disk storage over token indices in an inverted list of CITADEL and COIL.

![](images/fb18ad0455d5645704b5ca762569f0a1824d4882d1c507ce102b3fd9aa0a1adb.jpg)  
Figure 4: Latency-memory-accuracy tradeoff on MS MARCO passages using post-hoc pruning. $\tau$ is the pruning threshold.

# 5.2 Effects of Load Balancing

Figure 3 shows why a balanced inverted index greatly speeds up the inverted index search. As mentioned in Section 3.5, both CITADEL and COIL organize their token embeddings in an inverted list, except that COIL uses the input id of the token embedding as the key while CITADEL uses the router predictions. This makes COIL’s index distribution predetermined by the token distribution of the corpus, where frequent words like “the” and “a” will have much larger index sizes than those rare words. In contrast, by applying the load balancing loss on the router prediction, CITADEL yields a more balanced and even index distribution where its largest index is $8 ~ \times$ smaller than COIL’s after normalization. We also provide a detailed latency breakdown in A.3.

Table 4: Product Quantization. Pruning threshold is set to 0.9. nbits: ratio of total bytes to the vector dimension.   

<table><tr><td>Condition</td><td>MRR@10</td><td>Storage (GB)</td><td>Latency (ms)</td></tr><tr><td>original</td><td>0.362</td><td>78.3</td><td>3.95</td></tr><tr><td>nbits=2</td><td>0.361</td><td>13.3</td><td>0.90</td></tr><tr><td>nbits=1</td><td>0.356</td><td>11.0</td><td>0.92</td></tr></table>

# 5.3 Latency-Memory-Accuracy Trade-Off

Effects of Post-hoc Pruning. Figure 4 shows the tradeoff among latency, memory, and MRR $@ 1 0$ on MS MARCO passages with post-hoc pruning. We try the pruning thresholds [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]. We could see that the $\mathbf { M R R } @ 1 0$ score barely decreases when we increase the threshold to from 0.5 to 1.1, but the latency decreases by a large margin, from about $1 8 \mathrm { m s } /$ query to $0 . 6 1 \mathrm { m s }  { \vert }$ /query. The sweet spot $\tau = 1 . 1$ ) is around $( 0 . 3 5 9 \mathrm { M R R } @ 1 0$ , 49.3GB, 0.61 ms/query) which has a similar latency as DPR-128 while being as effective as ColBERT. Another sweet spot $\mathit { \check { \tau } } = \ 0 . 9 $ ) is around (0.362, 78.5GB, 3.95 ms/query), which has an even better MRR $@ 1 0$ score than ColBERT and only increases the storage and latency by a small gap. This simple pruning strategy is extremely effective and we shall see later in Section 5.4 it also yields interpretable document representations.

Combination with Product Quantization. We could further reduce the latency and storage with product quantization (Jégou et al., 2011) (PQ) as shown in Table 4. For nbits $^ { = 2 }$ , we divide the vectors into sets of 4-dimensional sub-vectors and use 256 centroids for clustering the sub-vectors, while for nbits $^ { - 1 }$ we set the sub-vector dim to 8 and the same for the rest. With only 2 bits per dimension, the $\mathbf { M R R } @ 1 0$ score on MS MARCO Dev only drops $4 \%$ but the storage is reduced by $83 \%$ and latency is reduced by $76 \%$ .

# 5.4 Token Routing Analysis of CITADEL

Qualitative Analysis. We further analyze the token representations of CITADEL to better understand its predictions. Figure 5 shows an example from MS MARCO passages using the default pruning threshold $\tau = 0 . 9$ . We can see that a lot of redundant words that do not contribute to the final semantics are deactivated, meaning all their routing weights are 0. For the activated tokens, we could see the routed keys are contextualized as many of them are related to emoji which is the theme of the document. We could further control the representations with post-hoc pruning as shown in Figure 7. By increasing the pruning threshold, more keywords are pruned and finally leaving the most central word “arrhythmia” activated.

Figure 5: Attribution analysis of CITADEL. Grey tokens are deactivated, while bold tokens are routed to at least one activated key (in blue).

![](images/3f4b147304f62b3ff9ea796925b6014b79c7d1722ef36f4437a5f83818aab90d.jpg)  
Figure 6: Token number distribution over number of activated experts per passage. $\tau$ is the pruning threshold.

Quantitative Analysis. We also perform a quantitative analysis on the token distribution over the number of activated routing keys as shown in Figure 6. We plot the histogram of the number of tokens over the number of activated keys for the whole corpus. With $\ell _ { 1 }$ regularization, we already have around 50 tokens per passage in average deactivated, which means that all the routing weights of these 50 tokens are 0. As the pruning threshold increases, more and more tokens are deactivated, yielding a sparse representation for interpreting CITADEL’s behaviours.

Figure 7: Tokens in grey (pruned) have zero activated keys while bold tokens have at least one activated key. We leave out the expanded terms and routing weights due to space limit.   

<table><tr><td rowspan=1 colspan=1>Threshold </td><td rowspan=1 colspan=1>Threshold  Sample documents from MS MARCO Passages</td></tr><tr><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>Allmedications have side effectdindrugtreat arrhythmiaside effectseriousiwhenangeicatiopd fectrushiadmitted to the hospital to begin the medication. Medications for Arrhythmia</td></tr><tr><td rowspan=1 colspan=1>0.9</td><td rowspan=1 colspan=1>medications ide effeciru arrhythmiaside effectriousi feadmitted to the hospital to begin the medication. Medications for Arrhythmia</td></tr><tr><td rowspan=1 colspan=1>1.3</td><td rowspan=1 colspan=1>A the hospital to begin the medication. Medications for Arrhythmia</td></tr></table>

Table 5: [CLS] ablation on MS MARCO passage. We set the pruning threshold to 0.9. $\mathbf { M R R } @ 1 0$ is reported for Dev and $\mathrm { n D C G } @ 1 0$ is reported for DL19.   

<table><tr><td>Models</td><td>Dev</td><td>DL19</td><td>Latency (ms)</td></tr><tr><td>COIL-full</td><td>0.353</td><td>0.704</td><td>47.9</td></tr><tr><td>COIL-tok</td><td>0.350</td><td>0.660</td><td>46.8</td></tr><tr><td>CITADEL</td><td>0.362</td><td>0.687</td><td>3.95</td></tr><tr><td>CITADEL-tok</td><td>0.360</td><td>0.665</td><td>1.64</td></tr></table>

# 6 Ablation Study

Impact of [CLS] Table 5 shows the influence of removing the [CLS] vector for CITADEL on MS MARCO passage. Although removing [CLS] improves the latency by a large margin, the in-domain effectiveness is also adversely affected, especially on TREC DL 2019. Nevertheless, CITADELtok (w/o [CLS]) still outperforms its counterpart COIL-tok in both precision and latency.

Number of Routed Experts. Table 6 shows the influence of changing the maximum number of keys that each document token can be routed to during training and inference on MS MARCO passage. As the number of routing keys increases, the index storage also increases rapidly but so does the $\mathbf { M R R } @ 1 0$ score which plateaus after reaching 7 keys. The latency does not increase as much after reaching 3 routing keys which is probably due to the load balancing loss.

# 7 Related Works

Dense Retrieval. Supported by multiple approximate nearest neighbour search libraries (John-

<table><tr><td>#Keys</td><td>MRR@10</td><td>Storage (GB)</td><td>Latency (ms)</td></tr><tr><td>1</td><td>0.347</td><td>53.6</td><td>1.28</td></tr><tr><td>3</td><td>0.360</td><td>136</td><td>14.7</td></tr><tr><td>5</td><td>0.364</td><td>185</td><td>18.6</td></tr><tr><td>7</td><td>0.370</td><td>196</td><td>20.4</td></tr><tr><td>9</td><td>0.367</td><td>221</td><td>19.6</td></tr></table>

Table 6: Number of routing keys for documents during training. No post-hoc pruning is applied.

son et al., 2021; Guo et al., 2020), dense retrieval (Karpukhin et al., 2020) gained much popularity due to its efficiency and flexibility. To improve effectiveness, techniques such as hard negative mining (Xiong et al., 2021; Zhan et al., 2021) and knowledge distillation (Lin et al., 2021b; Hofstätter et al., 2021) are often deployed. Recently, retrieval-oriented pre-training(Gao et al., 2021b; Lu et al., 2021; Gao and Callan, 2021; Izacard et al., 2022; Gao and Callan, 2022) also draws much attention as they could substantially improve the fine-tuning performance of downstream tasks.

Sparse Retrieval. Traditional sparse retrieval systems such as BM25 (Robertson and Zaragoza, 2009) and tf–idf (Salton and Buckley, 1988) represent the documents as a bag of words with static term weights. Recently, many works leverage pretrained language models to learn contextualized term importance (Bai et al., 2020; Mallia et al., 2021; Formal et al., 2021b; Lin and Ma, 2021). These models could utilize existing inverted index libraries such as Pyserini (Lin et al., 2021a) to perform efficient sparse retrieval or even hybrid with dense retrieval (Hofstätter et al., 2022; Shen et al., 2022; Lin and Lin, 2022).

Multi-Vector Retrieval. As mentioned in previous sections, ColBERT (Khattab and Zaharia,

2020; Santhanam et al., 2022b,a; Hofstätter et al., 2022) is probably the most renowned family in multi-vector retrieval. Tons of engineering and optimization has been made to accelerate the retrieval latency and reduce the index storage such as centroid pruning and fast kernels. COIL (Gao et al., 2021a) shows another possibility to accelerate token-level retrieval combined with the exact match prior and inverted vector search. ME-BERT (Luan et al., 2021) and MVR (Zhang et al., 2022) propose to use a fixed number of token embeddings for late interaction (e.g., top-k positions or special tokens). Concurrently to this work, ALIGNER (Qian et al., 2022) proposes to frame multi-vector retrieval as a sparse alignment problem between query tokens and document tokens. They use entropy-regularized linear programming to find the best alignment scheme to trade index size off against effectiveness. Our 110M model achieves higher in-domain and out-of-domain accuracy than their base and even large variants.

# 8 Conclusion

This paper proposes a novel multi-vector retrieval method that achieves state-of-the-art performance on several benchmark datasets while being $4 0 \times$ faster than ColBERT-v2 and $1 7 \times$ faster than the most efficient multi-vector retrieval library to date, PLAID. By jointly optimizing for the token index size and load balancing, our new dynamic lexical routing scheme greatly reduces the redundancy in all-to-all token interaction of ColBERT while bridging the word-mismatch problem in COIL. Experiments on both in-domain and out-of-domain datasets demonstrate the effectiveness and efficiency of our model.