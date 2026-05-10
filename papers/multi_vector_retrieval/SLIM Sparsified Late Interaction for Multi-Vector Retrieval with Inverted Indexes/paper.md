# SLIM: Sparsified Late Interaction for Multi-Vector Retrieval with Inverted Indexes

Minghan Li   
University of Waterloo   
Waterloo, Canada   
m692li@uwaterloo.ca

Sheng-Chieh Lin University of Waterloo Waterloo, Canada s269lin@uwaterloo.ca

# ABSTRACT

This paper introduces Sparsified Late Interaction for Multi-vector (SLIM) retrieval with inverted indexes. Multi-vector retrieval methods have demonstrated their effectiveness on various retrieval datasets, and among them, ColBERT is the most established method based on the late interaction of contextualized token embeddings of pre-trained language models. However, efficient ColBERT implementations require complex engineering and cannot take advantage of off-the-shelf search libraries, impeding their practical use. To address this issue, SLIM first maps each contextualized token vector to a sparse, high-dimensional lexical space before performing late interaction between these sparse token embeddings. We then introduce an efficient two-stage retrieval architecture that includes inverted index retrieval followed by a score refinement module to approximate the sparsified late interaction, which is fully compatible with off-the-shelf lexical search libraries such as Lucene. SLIM achieves competitive accuracy on MS MARCO Passages and BEIR compared to ColBERT while being much smaller and faster on CPUs. To our knowledge, we are the first to explore using sparse token representations for multi-vector retrieval. Source code and data are integrated into the Pyserini IR toolkit.

# CCS CONCEPTS

• Information systems Retrieval models and ranking.

# KEYWORDS

Neural IR, Late Interaction, Inverted Indexes, Sparse Retrieval

# ACM Reference Format:

Minghan Li, Sheng-Chieh Lin, Xueguang Ma, and Jimmy Lin. 2023. SLIM: Sparsified Late Interaction for Multi-Vector Retrieval with Inverted Indexes. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’23), July 23–27, 2023, Taipei, Taiwan. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3539618. 3591977

# 1 INTRODUCTION

Pairwise token interaction [12, 19, 45] has been widely used in information retrieval tasks [31]. Interaction-based methods enable

Xueguang Ma   
University of Waterloo   
Waterloo, Canada   
x93ma@uwaterloo.ca   
Jimmy Lin   
University of Waterloo   
Waterloo, Canada   
jimmylin@uwaterloo.ca

deep coupling between queries and documents, which often outperform representation-based methods [2, 34] when strong text encoders are absent. However, with the rise of pre-trained transformer models [4, 27], representation-based methods such as DPR [17] and SPLADE [6] gain more popularity as the pre-trained representations capture rich semantics of the input texts. Moreover, these methods can leverage established search libraries such as FAISS [16] and Pyserini [22] for efficient retrieval. To combine the best of both worlds, models such as ColBERT [18] and COIL [9] that leverage late interaction are proposed, where the token interaction between queries and documents only happens at the last layer of contextualized embeddings. Their effectiveness and robustness are demonstrated on various retrieval and question-answering benchmarks. However, these models require token-level retrieval and aggregation, which results in large indexes and high retrieval latency. Therefore, different optimization schemes are proposed to improve efficiency in both time and space [13, 37], making multi-vector retrieval difficult to fit into off-the-shelf search libraries.

In this paper, we propose an efficient and compact approach called Sparsified Late Interaction for Multi-vector retrieval with inverted indexes, or SLIM. SLIM is as effective as the state-of-the-art multi-vector model ColBERT on MS MARCO Passages [32] and BEIR [40] without any distillation or hard negative mining while being more efficient. More importantly, SLIM is fully compatible with inverted indexes in existing search toolkits such as Pyserini, with only a few extra lines of code in the pre-processing and postprocessing steps. In contrast, methods such as ColBERT and COIL require custom implementations and optimizations [13, 37] in order to be practical to use. This presents a disadvantage because some researchers and practitioners may prefer neural retrievers that are compatible with existing infrastructure based on inverted indexes so that the models can be easily deployed in production.

In order to leverage standard inverted index search, SLIM projects each contextualized token embedding to a high-dimensional, sparse lexical space [6] before performing the late interaction operation. Ideally, the trained sparse representations of documents can be compressed and indexed into an inverted file system for search. Our method is the first to explore using sparse representations for multi-vector retrieval with inverted indexes to our knowledge. However, unlike previous supervised sparse retrieval methods such as uniCOIL [21] or SPLADE [6], which compute sequence-level representations for each query/document, deploying token-level sparse vectors into inverted indexes is problematic. As shown in Figure 1a, the high-level idea is to convert each sparse token embedding into a sub-query/document and perform token-level inverted list retrieval before aggregation (e.g., scatter sum and max). However, this is incredibly slow in practice as latency increases rapidly when the posting lists are too long.

![](images/963c1e0761d3e37481ef97b35d2f27dae4ae1dd9c8a180d97c7252049f82e1e0.jpg)  
Figure 1: (a) The naive SLIM implementation takes the expansion of each token as sub-queries/documents. The token-level rankings are merged using scatter operations. (b) The approximate SLIM implementation first fuses the lower- and upperbounds of scores for retrieval using Equations (8) and (9); the candidate list is then refined according to Equation (4).

![](images/41425e66f18d6e0f41fdd89668b9a389855000c0aee732ca4f87e4c0c7b4dd69.jpg)  
(b) Approximate SLIM implementation.

Therefore, instead of using the naive approach, a more economical way is to approximate the sparsified late interaction with a two-stage system. We first calculate the upper and lower bounds of the sparsified late-interaction scores by swapping the order of the operators in late interaction (see Section 2). In this way, the token-level sparse vectors are pooled before indexing to obtain sequence-level representations for the inverted index, yielding fast retrieval speed. After retrieval, a lightweight score refinement step is applied, where we use the sparse token vectors stored using Scipy [41] to rerank the candidate lists. As shown in Figure 1b, this two-stage design allows us to optimize the latency of the retrieval stage without worrying about accuracy loss as the score refinement will compensate for errors from the first stage. Experiments on MS MARCO Passages show that SLIM is able to achieve a similar ranking accuracy compared to ColBERT-v2, while using $4 0 \%$ less storage and achieving an $8 3 \%$ decrease in latency. To sum up, our contributions in this paper are three-fold:

• We are the first to use sparse representations for multi-vector retrieval with standard inverted indexes. • We provide a two-stage implementation of SLIM by approximating late interaction followed by a score refinement step. • SLIM is fully compatible with off-the-shelf search toolkits such as Pyserini.

# 2 METHODOLOGY

ColBERT [18] proposes late interaction between the tokens in a query $q = \{ q _ { 1 } , q _ { 2 } , \cdots , q _ { N } \}$ and a document $d = \{ d _ { 1 } , d _ { 2 } , \dotsb , d _ { M } \}$ :

$$
s ( q , d ) = \sum _ { i = 1 } ^ { N } \operatorname* { m a x } _ { j = 1 } ^ { M } v _ { q _ { i } } ^ { T } v _ { d _ { j } } ,
$$

where $v _ { q _ { i } }$ and $v _ { d _ { j } }$ denote the last-layer contextualized token embeddings of BERT. This operation exhaustively compares each query token to all document tokens. The latency and storage of ColBERT are bloated as many tokens do not contribute to either query or document semantics, and thus complex engineering optimizations are needed to make the model practical [37, 38].

Unlike ColBERT, which only uses contextualized token embeddings for computing similarity, SPLADE [5, 6] further utilizes the pre-trained Mask Language Modeling (MLM) layer to project each $c$ - dimensional token embedding to a high-dimensional, lexical space $V$ . Each dimension corresponds to a token and has a non-negative value due to the following activation:

$$
\phi _ { d _ { j } } = \log ( 1 + \mathrm { R e L U } ( W ^ { T } v _ { d _ { j } } + b ) ) ,
$$

where $\phi _ { d _ { j } } \in \mathbb { R } ^ { | V | } , v _ { d _ { j } }$ is the token embedding of the $j$ th token of document $d$ ; ?? and $^ { b }$ are the weights and biases of the MLM layer. To compute the final similarity score, SPLADE pools all the token embeddings into a sequence-level representation and uses the dot product between a query and a document as the similarity:

$$
s ( q , d ) = ( \operatorname* { m a x } _ { i = 1 } ^ { N } \phi _ { q _ { i } } ) ^ { T } ( \operatorname* { m a x } _ { j = 1 } ^ { M } \phi _ { d _ { j } } ) .
$$

Here, max pooling is an element-wise operation and is feasible for sparse representations as the dimensions of all the token vectors are aligned (lexical space) and non-negative. These two properties play a vital role in the success of SPLADE and are also important for making SLIM efficient, as we shall see later.

Similar to ColBERT’s late interaction, sparsified late interaction also takes advantages of the contextualized embeddings of BERT’s last layer. But different from ColBERT, we first apply the sparse activation in Equation (2) before calculating the similarity in Equation (1):

$$
s ( \boldsymbol { q } , d ) = \sum _ { i = 1 } ^ { N } \operatorname* { m a x } _ { j = 1 } ^ { M } \phi _ { \boldsymbol { q } _ { i } } ^ { T } \phi _ { d _ { j } } ,
$$

where $\phi _ { d _ { j } }$ and $\phi _ { q _ { i } }$ are the representations in Equation (2). However, as shown in Figure 1a, to keep the token-level interaction, we must convert the sparse representation of each token in a query to a subquery and retrieve the sub-documents from the index. Moreover, the token-level rankings need to be merged to yield the final ranked list using scatter operations, which further increases latency.

Due to the impracticality of the naive implementation, a natural solution would be to approximate Equation (4) during retrieval. We first unfold the dot product in Equation (4):

<table><tr><td>Models</td><td colspan="2">MARCO Dev MRR@10 R@1k</td><td colspan="2">TREC DL19 nDCG@10 R@1k</td><td colspan="2">TREC DL20 nDCG@10 R@1k</td><td>BEIR (13 tasks) nDCG@10</td><td colspan="2">Space&amp;Time Efficiency Disk (GB) Latency (ms/query)</td></tr><tr><td colspan="10">Models trained with only BM25 hard negatives from MS MARCO Passages</td></tr><tr><td>BM25</td><td>0.188</td><td>0.858</td><td>0.506</td><td>0.739</td><td>0.488</td><td>0.733</td><td>0.440</td><td>0.7</td><td>40</td></tr><tr><td>DPR</td><td>0.319</td><td>0.941</td><td>0.611</td><td>0.742</td><td>0.591</td><td>0.796</td><td>0.375</td><td>26.0</td><td>2015</td></tr><tr><td>SPLADE</td><td>0.340</td><td>0.965</td><td>0.683</td><td>0.813</td><td>0.671</td><td>0.823</td><td>0.453</td><td>2.6</td><td>475</td></tr><tr><td>COIL</td><td>0.353</td><td>0.967</td><td>0.704</td><td>0.835</td><td>0.688</td><td>0.841</td><td>0.483</td><td>78.5</td><td>3258</td></tr><tr><td>ColBERT</td><td>0.360</td><td>0.968</td><td>0.694</td><td>0.830</td><td>0.676</td><td>0.837</td><td>0.453</td><td>154.3</td><td>-</td></tr><tr><td>SLIM</td><td>0.358</td><td>0.962</td><td>0.701</td><td>0.824</td><td>0.640</td><td>0.854</td><td>0.451</td><td>18.2</td><td>580</td></tr><tr><td colspan="10">Models trained with further pre-training/hard-negative mining/distillation</td></tr><tr><td>coCondenser</td><td>0.382</td><td>0.984</td><td>0.674</td><td>0.820</td><td>0.684</td><td>0.839</td><td>0.420</td><td>26.0</td><td>2015</td></tr><tr><td>SPLADE-v2</td><td>0.368</td><td>0.979</td><td>0.729</td><td>0.865</td><td>0.718</td><td>0.890</td><td>0.499</td><td>4.1</td><td>2710</td></tr><tr><td>ColBERT-v2</td><td>0.397</td><td>0.985</td><td>0.744</td><td>0.882</td><td>0.750</td><td>0.894</td><td>0.500</td><td>29.0</td><td>3275</td></tr><tr><td>SLIM+</td><td>0.404</td><td>0.968</td><td>0.714</td><td>0.842</td><td>0.702</td><td>0.855</td><td>0.490</td><td>17.3</td><td>550</td></tr></table>

Table 1: In-domain and out-of-domain evaluation on MS MARCO Passages, TREC DL 2019/2020, and BEIR. “-” means not practical to evaluate on a single CPU. Latency is benchmarked on a single CPU and query encoding time is excluded.

$$
s ( q , d ) = \sum _ { i = 1 } ^ { N } \operatorname* { m a x } _ { j = 1 } ^ { M } \sum _ { k = 1 } ^ { \lvert V \rvert } \phi _ { q _ { i } } ^ { ( k ) } \phi _ { d _ { j } } ^ { ( k ) } ,
$$

where $\phi _ { q _ { i } } ^ { ( k ) }$ is the $k$ th elements of $\phi _ { q _ { i } }$ . As each $\phi _ { q _ { i } }$ across different $q _ { i }$ all shares the same lexical space and the values are non-negative, we can easily derive its upper-bound and lower-bound:

$$
\sum _ { i = 1 } ^ { N } \operatorname* { m a x } _ { j = 1 } ^ { M } \sum _ { k = 1 } ^ { | V | } e _ { q _ { i } } ^ { ( k ) } \phi _ { d _ { j } } ^ { ( k ) } \leq s ( q , d ) \leq \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { | V | } \operatorname* { m a x } _ { j = 1 } ^ { M } \phi _ { q _ { i } } ^ { ( k ) } \phi _ { d _ { j } } ^ { ( k ) } ,
$$

$$
e _ { q _ { i } } ^ { ( k ) } = \left\{ \begin{array} { c c } { \phi _ { q _ { i } } ^ { ( k ) } , } & { \mathrm { i f } k = \mathrm { a r g m a x } \phi _ { q _ { i } } ^ { ( k ) } } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right.
$$

Then, the lower-bound score $s _ { l } ( q , d )$ and upper-bound score $s _ { h } ( q , d )$ can be further factorized as:

$$
s _ { l } ( q , d ) = \sum _ { i = 1 } ^ { N } \operatorname* { m a x } _ { j = 1 } ^ { M } \sum _ { k = 1 } ^ { | V | } e _ { q _ { i } } ^ { ( k ) } \phi _ { d _ { j } } ^ { ( k ) } = ( \sum _ { i = 1 } ^ { N } e _ { q _ { i } } ) ^ { T } ( \operatorname* { m a x } _ { j = 1 } ^ { M } \phi _ { d _ { j } } ) ;
$$

$$
s _ { h } ( q , d ) = \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { \vert V \vert } \operatorname* { m a x } _ { j = 1 } ^ { M } \phi _ { q _ { i } } ^ { ( k ) } \phi _ { d _ { j } } ^ { ( k ) } = ( \sum _ { i = 1 } ^ { N } \phi _ { q _ { i } } ) ^ { T } ( \operatorname* { m a x } _ { j = 1 } ^ { M } \phi _ { d _ { j } } )
$$

where both the sum and max operations will be element-wise if the targets are vectors. We see that these two equations resemble the form of SPLADE in Equation (3), where queries and documents are encoded into sequence-level representations independently. In this way, $\operatorname* { m a x } _ { j = 1 } ^ { M } \phi _ { d _ { j } }$ in Equation (8) can be pre-computed and indexed offline. To approximate $s ( q , d )$ in Equation (5), we use the linear interpolation between the lower- and the upper-bound:

$$
\begin{array} { l } { \displaystyle s _ { a } ( { q } , d ) = \beta \cdot s _ { l } ( { q } , d ) + ( 1 - \beta ) \cdot s _ { h } ( { q } , d ) } \\ { \displaystyle \qquad = ( \sum _ { i = 1 } ^ { N } ( \beta \cdot e _ { { q } _ { i } } + ( 1 - \beta ) \cdot \phi _ { { q } _ { i } } ) ) ^ { T } ( \underset { j = 1 } { \overset { M } { \operatorname* { m a x } } } \phi _ { { d } _ { j } } ) } \end{array}
$$

where $s _ { a } ( q , d )$ is the approximate score of SLIM and $\beta \in \left[ 0 , 1 \right]$ is the interpolation coefficient. As shown in Equation (9), we can first fuse the query representation before retrieval and fine-tune the coefficient $\beta$ on a validation set. It is worth mentioning that the approximation is applied after the SLIM model is trained using Equation (4). During indexing,sequence level-representation $\operatorname* { m a x } _ { j = 1 } ^ { M } \phi _ { d _ { j } }$ serini [22] to index theand use Scipy [41] to store the sparse token vectors. During retrieval, we use Equation (9) to retrieve the top- $\boldsymbol { \cdot } \boldsymbol { k }$ candidates with the LuceneImpactSearcher in Pyserini. For score refinement, we extract the stored sparse token vectors for the top- $\boldsymbol { \cdot } \boldsymbol { k }$ documents and use Equation (4) to refine the candidate list. This two-stage breakdown yields a good effectiveness-efficiency trade-off, as we can aggressively reduce the first-stage retrieval latency without too much accuracy loss due to the score refinement.

To further reduce the memory footprint and improve latency for the first-stage retrieval, we apply two post-hoc pruning strategies on the inverted index to remove: (1) tokens that have term importance below a certain threshold; (2) postings that exceed a certain length (i.e., the inverse document frequency is below a threshold). Since all the term importance values are non-negative, tokens with smaller weights contribute less to the final similarity. Moreover, tokens with long postings mean that they frequently occur in different documents, which yield low inverse document frequencies (IDFs) and therefore can be safely discarded.

# 3 EXPERIMENTS

We evaluate our model and baselines on MS MARCO Passages [32] and its shared tasks, TREC DL 2019/2020 passage ranking tasks [3]. We train the baseline models on MS MARCO Passages and report results on its dev-small set and TREC DL 2019/2020 test queries following the same setup in CITADEL [20]. We further evaluate the models on the BEIR benchmark [40], which consists of a diverse set of 18 retrieval tasks across 9 domains. Following previous work [5, 38], we only evaluate 13 datasets due to license restrictions.1 The evaluation metrics are MRR@10, nDCG@10, and

![](images/8844e0574c36e7c5694a9fbef09ab80ed2724df66c09239634495585b58d4463.jpg)  
Figure 2: Effectiveness-Efficiency trade-offs of SLIM (w/ and w/o score refinement) on MS MARCO Passages.

Recall@1000 (i.e., $\operatorname { R @ 1 K }$ . Latency is benchmarked on a single Intel(R) Xeon(R) Platinum 8275CL CPU $\textcircled { a } 3 . 0 0 \mathrm { G H z }$ and both the batch size and the number of threads are set to 1.

We follow the training procedure of CITADEL [20] to train SLIM and apply $\ell _ { 1 }$ regularization on the sparse token representations for sparsity control. SLIM is trained only with BM25 hard negatives in MS MARCO Passages, while $\mathrm { S L I M ^ { + + } }$ is trained with cross-encoder distillation and hard negative mining. For indexing and pruning, the default token weight pruning threshold is 0.5 and the IDF threshold is 3. We use Pyserini to index and retrieve the documents and use Scipy to store the sparse token vectors (CSR matrix). For retrieval, the fusion weight $\beta$ in Equation (9) is set to 0.01. We first retrieve the top-4000 candidates using Pyserini’s LuceneImpactSearcher and then use the original sparse token vectors stored by Scipy to refine the ranked list and output the top-1000 documents.

Table 1 shows the in-domain evaluation results on MS MARCO Passages and TREC DL 2019/2020. SLIM and $\mathrm { S L I M ^ { + + } }$ manage to achieve accuracy comparable to ColBERT and ColBERT-v2 on MS MARCO Passages. However, the results seem unstable on TREC DL, which is expected as TREC DL has far fewer queries. For out-ofdomain evaluation, the results are a little bit worse than the current state of the art but still close to ColBERT. For latency and storage, SLIM has a much smaller disk requirement and lower latency compared to ColBERT. For example, $\mathrm { S L I M ^ { + + } }$ achieves effectiveness comparable to ColBERT-v2 with an $8 3 \%$ decrease in latency on a single CPU and using $4 0 \%$ less storage.

Next, we show that a two-stage implementation of SLIM provides a good trade-off between effectiveness and efficiency. Figure 2 plots CPU latency vs. MRR $@ 1 0 ,$ /Recall@1000 on MS MARCO Passages using IDF thresholding for SLIM (w/o hard-negative mining and distillation). We set the minimum IDF threshold to {0, 0.5, 1, 1.5, 2, 2.5, 3} and first-stage retrieval candidates top- $\mathbf { \nabla } \cdot k$ to {1000, 1500, $2 0 0 0 , 2 5 0 0 , 3 0 0 0 , 3 5 0 0 , 4 0 0 0 \}$ before score refinement (if any). Without the score refinement, the MRR $@ 1 0$ and Recall@1000 scores are high when the IDF threshold is small, but effectiveness drops rapidly when we increase the IDF threshold to trade accuracy for latency. In contrast, with the score refinement step, we see that with about 0.003 loss in MRR@10 and 0.01 loss in Recall@1000, latency improves drastically from about $1 8 0 0 ~ \mathrm { m s } ,$ /query to $5 0 0 ~ \mathrm { m s }$ /query. The reason is that the score refinement step, which takes a small amount of time compared to retrieval, compensates for the errors from the aggressive pruning at the first stage, which shows that our novel lower- and upper-bound fusion provides a good approximation of SLIM when the IDF threshold is low, and that the score refinement step is important as it allows more aggressive pruning for the first-stage retrieval.

# 4 RELATED WORK

Dense retrieval [17] gains much popularity as it is supported by multiple approximate nearest neighbor search libraries [11, 16]. To improve effectiveness, hard negative mining [42, 43] and knowledge distillation [14, 26] are often deployed. Recently, further pretraining for retrieval [7, 8, 10, 15, 28] is proposed to improve the fine-tuned effectiveness of downstream tasks.

Sparse retrieval systems such as BM25 [35] and tf–idf [36] encode documents into bags of words. Recently, pre-trained language models are used to learn contextualized term importance [1, 6, 21, 23, 30]. These models leverage current search toolkits such as Pyserini [22] to perform sparse retrieval, or contribute to hybrid approaches with dense retrieval [13, 24, 25, 39].

Besides ColBERT [18], COIL [9] accelerates retrieval by combining exact match and inverted index search. CITADEL [20] further introduces lexical routing to avoid the lexical mismatch issue in COIL. ME-BERT [29] and MVR [44] propose to use a portion of token embeddings for late interaction. ALIGNER [33] frames multi-vector retrieval as an alignment problem and uses entropy-regularized linear programming to solve it.

# 5 CONCLUSION

In this paper, we propose an efficient yet effective implementation for multi-vector retrieval using sparsified late interaction, which yields fast retrieval speed, small index size, and full compatibility with off-the-shelf search toolkits such as Pyserini. The key ingredients of SLIM include sparse lexical projection using the MLM layer, computing the lower- and upper-bounds of the late interaction, as well as token weight and postings pruning. Experiments on both indomain and out-of-domain information retrieval datasets show that SLIM achieves comparable accuracy to ColBERT-v2 while being more efficient in terms of both space and time.

# ACKNOWLEDGEMENTS

This research was supported in part by the Natural Sciences and Engineering Research Council (NSERC) of Canada; computational resources were provided by Compute Canada.