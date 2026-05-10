# Spike Hijacking in Late-Interaction Retrieval

Karthik Suresh $\jmath$ , Tushar Vatsa2, Tracy King3, Asim Kadav3 and Michael Friedrich3

1Adobe, 345 Park Ave, San Jose, CA 95110, USA

# Abstract

Late-interaction retrieval models rely on hard maximum similarity (MaxSim) to aggregate token-level similarities. Although effective, this winner-take-all pooling rule may structurally bias training dynamics. We provide a mechanistic study of gradient routing and robustness in MaxSim-based retrieval. In a controlled synthetic environment with in-batch contrastive training, we demonstrate that MaxSim induces significantly higher patchlevel gradient concentration than smoother alternatives such as Top-k pooling and softmax aggregation. While sparse routing can improve early discrimination, it also increases sensitivity to document length: as the number of document patches grows, MaxSim degrades more sharply than mild smoothing variants.

We corroborate these findings on a real-world multi-vector retrieval benchmark, where controlled documentlength sweeps reveal similar brittleness under hard max pooling. Together, our results isolate pooling-induced gradient concentration as a structural property of late-interaction retrieval and highlight a sparsity–robustness tradeoff. These findings motivate principled alternatives to hard max pooling in multi-vector retrieval systems.

# Keywords

Late-Interaction Retrieval, MaxSim Pooling, Contrastive Learning, Gradient Concentration, Robustness to Document Length

# 1. Introduction

Late-interaction retrieval models [1] have emerged as a powerful alternative to single-vector embedding methods [2, 3]. Instead of compressing a query and document into single representations, late-interaction approaches retain token or patch-level embeddings and compute relevance by aggregating fine-grained similarity scores. A widely adopted aggregation rule in such models is hard maximum similarity (MaxSim), where each query token contributes the maximum similarity it achieves over document patches. This winner-take-all formulation has been shown to be both effective and computationally attractive in multi-vector retrieval systems [1, 4, 5].

Despite its empirical success, the structural implications of MaxSim remain underexplored. MaxSim introduces a non-smooth routing mechanism: for each query token, only the highest-scoring document patch contributes to the final score. Under contrastive training with in-batch negatives [6], this induces a highly selective gradient pathway. While sparsity can enhance discrimination, it can concentrate learning signal on a small subset of document patches, affecting robustness and generalization.

We provide a mechanistic analysis of pooling in late-interaction retrieval, answering questions:

1. How does MaxSim based pooling shape gradient routing under contrastive learning?   
2. Does this structural property affect robustness, particularly as document length increases?

To isolate the effect of pooling, we construct a controlled synthetic retrieval environment in which queries and documents are generated from a shared concept dictionary. Within this setting, we train a small encoder using in-batch InfoNCE [6] and compare three aggregation strategies: MaxSim, Top- $k$ averaging, and temperature-controlled softmax pooling. This controlled setup allows us to measure patch-level gradient concentration and retrieval behavior as document length varies.

Our synthetic experiments reveal three findings. First, MaxSim induces significantly higher patchlevel gradient concentration than smoother alternatives. Second, although MaxSim achieves competitive retrieval quality in moderate-length documents, it exhibits greater sensitivity to increasing document length. Third, mild smoothing (e.g., Top- $k$ pooling) preserves retrieval performance while reducing gradient concentration and improving length robustness. These results expose a sparsity–robustness tradeoff inherent in pooling design. To validate that this is not confined to the synthetic regime, we conduct controlled document-length sweeps in a real-world multi-vector retrieval benchmark. The empirical trends mirror those observed in the synthetic environment, providing corroborating evidence that hard max pooling induces increased brittleness with respect to document length.

Our contributions are as follows:

• A mechanistic study of gradient routing under hard max pooling in late-interaction retrieval.   
• Introduction of patch-level gradient concentration metrics to quantify pooling-induced sparsity.   
• Demonstration, in a controlled synthetic setting, that MaxSim leads to increased sensitivity to document length compared to mild smoothing alternatives.   
• Empirical validation on a real-world benchmark showing that the sparsity-smoothness tradeoff governs pooling robustness across both adversarial and non-adversarial document-length increases.

# 2. Background and Problem Formulation

# 2.1. Late-Interaction Retrieval

Let a query $q$ consist of $T$ token embeddings

$$
q = \{ \mathbf q _ { 1 } , \dots , \mathbf q _ { T } \} , \quad \mathbf q _ { i } \in \mathbb { R } ^ { d } ,
$$

and let a document $d$ consist of $M$ patch embeddings

$$
d = \{ \mathbf { d } _ { 1 } , \hdots , \mathbf { d } _ { M } \} , \quad \mathbf { d } _ { j } \in \mathbb { R } ^ { d } .
$$

Late-interaction retrieval models [1] compute a similarity matrix

$$
S _ { i j } = { \bf q } _ { i } ^ { \top } { \bf d } _ { j } ,
$$

and aggregate these fine-grained similarities into a scalar relevance score

$$
\operatorname { s c o r e } ( q , d ) = \sum _ { i = 1 } ^ { T } \phi { \bigl ( } S _ { i 1 } , \ldots , S _ { i M } { \bigr ) } ,
$$

where $\phi ( \cdot )$ is a pooling operator applied independently for each query token.

# 2.2. Pooling Operators

We consider three commonly used pooling strategies.

Hard Max (MaxSim). Each query token contributes only its highest similarity over document patches [1].

$$
\phi _ { \operatorname* { m a x } } ( S _ { i 1 } , \ldots , S _ { i M } ) = \operatorname* { m a x } _ { 1 \leq j \leq M } S _ { i j } .
$$

Top- $k$ Averaging. Top- $k ( i )$ denotes the indices of the $k$ largest values in $\{ S _ { i j } \} _ { j = 1 } ^ { M }$ .

$$
\phi _ { k } ( S _ { i 1 } , \ldots , S _ { i M } ) = \frac { 1 } { k } \sum _ { j \in \mathrm { { T o p } } \cdot k ( i ) } S _ { i j } ,
$$

Softmax Pooling. $\tau > 0$ controls smoothness. As $\tau  0$ , softmax pooling approaches hard max.

$$
\phi _ { \tau } ( S _ { i 1 } , \dots , S _ { i M } ) = \sum _ { j = 1 } ^ { M } w _ { i j } S _ { i j } , \quad w _ { i j } = { \frac { \exp ( S _ { i j } / \tau ) } { \sum _ { m = 1 } ^ { M } \exp ( S _ { i m } / \tau ) } } ,
$$

# 2.3. Contrastive Training Objective

We train models using in-batch InfoNCE [6]. For a batch of $B$ query-document pairs $\{ ( q _ { i } , d _ { i } ^ { + } ) \} _ { i = 1 } ^ { B }$ , the score matrix is defined as

$$
{ \bf Z } _ { i j } = \mathrm { s c o r e } ( q _ { i } , d _ { j } ^ { + } ) .
$$

The objective is

$$
\mathcal { L } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } - \log \frac { \exp ( \mathbf { Z } _ { i i } ) } { \sum _ { j = 1 } ^ { B } \exp ( \mathbf { Z } _ { i j } ) } .
$$

Each query is trained to assign higher score to its positive document relative to in-batch negatives.

# 2.4. Patch-Level Gradient Concentration

To quantify pooling-induced sparsity, we analyze how gradient mass is distributed across document patches. For a fixed query-document pair $( q , d )$ and loss $\mathcal { L }$ , let

$$
g _ { j } = { \left\| \frac { \partial { \mathcal { L } } } { \partial \mathbf { d } _ { j } } \right\| }
$$

denote the gradient norm for patch $j$ .

We define patch-level gradient concentration using the Gini coefficient [7] over $\{ g _ { j } \} _ { j = 1 } ^ { M }$ :

$$
\mathrm { G i n i } ( g ) = 1 - { \frac { 2 } { M \sum _ { j = 1 } ^ { M } g _ { j } } } \sum _ { j = 1 } ^ { M } \left( \sum _ { m = 1 } ^ { j } g _ { ( m ) } \right) ,
$$

where $g _ { ( m ) }$ denotes the sorted gradient norms in ascending order. A higher Gini value indicates that gradient mass is concentrated on fewer patches. This metric directly compares how different pooling operators route learning signal during contrastive training.

# 2.5. Problem Statement

Our goal is to understand how the choice of pooling operator $\phi$ affects: (1) Patch-level gradient concentration under contrastive training, and (2) Retrieval robustness as document length $M$ increases. By isolating pooling within a controlled setting, we aim to characterize the structural tradeoffs between selectivity and smoothness in late-interaction retrieval.

# 3. Synthetic Experimental Setup

To isolate the structural effect of pooling, we construct a controlled synthetic retrieval environment. The design allows precise control over query–document alignment, document length, and negative hardness while keeping the training objective fixed.

Concept Dictionary and Embedding Space We generate a fixed dictionary of $C$ concept vectors

$$
\{ \mathbf { c } _ { 1 } , \hdots , \mathbf { c } _ { C } \} \subset \mathbb { R } ^ { d } ,
$$

sampled from a standard Gaussian distribution and $\ell _ { 2 }$ -normalized. Each query consists of $T$ token embeddings. A query is generated by sampling $T$ concept indices uniformly at random and adding Gaussian noise:

$$
\begin{array} { r } { \mathbf q _ { i } = \mathrm { n o r m a l i z e } ( \mathbf c _ { k _ { i } } + \epsilon ) , \quad \epsilon \sim \mathcal N ( 0 , \sigma _ { q } ^ { 2 } \mathbf I ) . } \end{array}
$$

Positive Documents A positive document of length $M$ consists of $M$ patch embeddings. To control task difficulty, only $K \leq T$ patches correspond to query concepts (distributed support), while the remaining patches are sampled from random concepts:

$$
\mathbf { d } _ { j } ^ { + } = \left\{ \begin{array} { l l } { \mathrm { n o r m a l i z e } ( \mathbf { c } _ { k _ { i } } + \epsilon ) , } & { \mathrm { i f ~ } j \leq K , } \\ { \mathrm { n o r m a l i z e } ( \mathbf { c } _ { r } + \epsilon ) , } & { \mathrm { o t h e r w i s e , } } \end{array} \right.
$$

where $\epsilon \sim \mathcal { N } ( 0 , \sigma _ { d } ^ { 2 } \mathbf { I } )$ and $r$ is a random concept index. This design ensures that positive relevance is distributed across multiple patches rather than concentrated in a single location.

Hard Negatives To simulate realistic confusers, we construct hard negative documents that: Share a small number of query concepts (partial overlap) and contain a “spike” patch formed by averaging several query concepts. The spike patch increases similarity across multiple query tokens and creates a potential winner-take-all scenario under hard max pooling. The remaining patches are sampled from random concepts. While synthetic by construction, spike patches model a practically relevant scenario: in retrieval-augmented generation, adversarially crafted or SEO-optimized passages may aggregate multiple query-relevant terms into a short span, creating a similar winner-take-all vulnerability. Our setup isolates this mechanism as a controlled stress test rather than a claim about attack frequency.

Training Protocol We train a small three-layer feed-forward encoder $f _ { \theta }$ that maps raw embeddings to normalized representations:

$$
\tilde { \mathbf { q } } _ { i } = f _ { \theta } ( \mathbf { q } _ { i } ) , \quad \tilde { \mathbf { d } } _ { j } = f _ { \theta } ( \mathbf { d } _ { j } ) .
$$

Training uses in-batch InfoNCE [6] with batch size $B = 6 4$ , where each query competes against $B - 1$ in-batch negatives. The dataset is fixed across epochs to ensure learnability. Unless otherwise stated, we train for 100 epochs with Adam [8] and evaluate every 5 epochs.

# Evaluation Metrics We report:

• Recall $@ k$ : fraction of queries whose positive document is ranked in the top- $k$ .   
• Patch-level Gradient Gini: inequality of gradient mass across document patches (Section 2).

Document Length Sweep To study robustness, we vary the number of document patches $M \in$ $\{ 3 2 , 6 4 , 1 2 8 , 2 5 6 \}$ while keeping: Query tokens fixed; Concept assignments fixed; Training procedure unchanged. Only the number of document patches changes. This controlled sweep isolates the effect of document length on retrieval quality under different pooling operators.

Compared Pooling Variants We compare three variants. This setup enables controlled comparison of gradient routing behavior and retrieval robustness across pooling strategies.

• MaxSim, • Top- $k$ averaging (with fixed $k$ ), • Softmax pooling with temperature $\tau = 1 . 0$ .

# 4. Results

We evaluate how pooling choice affects (i) gradient routing during training, (ii) retrieval performance, and (iii) robustness to document length. All synthetic results are summarized in Figure 1.

![](images/0c1c401c6f08c87a194a40c339f1d762df62b707489d90b64de88359381389cc.jpg)  
Figure 1: Synthetic training dynamics and document-length sweep. (A) Patch-level gradient concentration (Gini) vs. training. (B) Retrieval quality (Recal $\begin{array} { r } { \left. \left( \overline { { \omega } } \right. ^ { \cdot } \right. } \end{array}$ ) vs. training. (C) Retrieval quality vs. document length $M$ under fixed queries.

# 4.1. Synthetic Analysis

Experimental Setup. We generate a fixed synthetic dataset of queries and positive documents built from $C = 1 0 0$ latent concepts in $\mathbb { R } ^ { d }$ b $( d = 1 6 )$ . Each query contains $T = 1 6$ concept tokens sampled with Gaussian noise $\sigma _ { q } = 0 . 1$ , and each positive document contains $M$ patches with noise $\sigma _ { d } = 0 . 1$ , of which a subset correspond to query concepts and the remainder are distractors. Training is performed with in-batch negatives (batch size $B = 6 4$ ) under three pooling strategies: hard MaxSim, Top- $k$ averaging $( k = 4 )$ , and softmax aggregation $\mathit { \check { \tau } } = 1 . 0 $ ).

To study robustness under adversarial length expansion, we inject $K \in \{ 0 , 2 5 , 5 0 , 1 0 0 , 2 0 0 , 4 0 0 \}$ token-targeted hard-negative patches into non-gold documents only. We measure (i) patch-level gradient concentration using the Gini coefficient and (ii) retrieval quality using Recall $@ 1$ .

Gradient Concentration During Training. Figure 1A shows the patch-level gradient Gini coefficient over training epochs. MaxSim produces substantially higher gradient concentration than the alternatives. Throughout training, MaxSim maintains a Gini coefficient near 0.78, indicating that gradient mass is concentrated on a small subset of document patches. Top-4 pooling yields moderate concentration $( \sim ~ 0 . 4 5 )$ , while softmax pooling distributes gradients far more uniformly $( \sim \ 0 . 1 8 )$ . Importantly, this separation emerges early and remains stable across epochs, suggesting that gradient concentration is a structural property of the pooling operator, not a transient optimization artifact.

Retrieval Performance During Training. Figure 1B reports Recall $@ 1$ over training. MaxSim and Top-4 pooling achieve comparable retrieval performance $( \sim 0 . 3 3 - 0 . 3 8 \mathrm { R e c a l l } @ 1 )$ , while softmax pooling performs substantially worse. Thus, sparse routing appears beneficial for discrimination, but extreme sparsity (hard max) is not uniquely necessary for strong retrieval performance.

Robustness to Document Length. To evaluate brittleness, we conduct a controlled document-length sweep. Queries and concept assignments are fixed, while only the number of document patches $M$ varies. Figure 1C shows Recall@1 as a function of $M$ . All pooling strategies degrade as document length increases, reflecting increased opportunities for spurious matches. However, MaxSim exhibits sharper degradation compared to Top-4 pooling. While MaxSim performs strongly at small $M$ , its performance drops more rapidly as $M$ grows. Top-4 pooling degrades more smoothly and outperforms MaxSim at larger document lengths.

Summary of Synthetic Findings. The synthetic results reveal a consistent pattern:

• Hard MaxSim induces the highest gradient concentration.

• Moderate sparsity (Top- $k$ pooling) achieves similar retrieval quality with lower concentration.   
• Higher concentration correlates with increased sensitivity to document length.

Together, these results isolate pooling-induced gradient routing as a key structural factor influencing robustness in late-interaction retrieval. These results do not imply that lower concentration is uniformly desirable. In our synthetic setting, softmax pooling distributes gradients most evenly but yields the lowest retrieval quality (Figure 1B), suggesting that some degree of selective routing is necessary for discrimination. The practical tradeoff is regime-dependent: MaxSim excels on short, well-targeted documents, while moderate smoothing (e.g., Top- $k$ ) offers a better sparsity–robustness balance when document length is variable or adversarial content is a concern.

# 4.2. Real-World Corroboration on ColQwen ${ \bf 2 . 5 + }$ ViDoRe

We validate our synthetic findings using ColQwen2.5 [9] on the ViDoRe biomedical benchmark [10, 11]. Our goal is to test whether pooling-induced gradient concentration and spike hijacking manifest in practice.

Without retraining, we measure doc-patch gradient concentration (Gini) across 160 queries. MaxSim produces the highest concentration (0.983), followed by Top-5 (0.951) and Softmax (0.883), with paired $t$ -tests confirming significant differences $( p < 1 0 ^ { - 8 0 } )$ ). This ordering matches the synthetic results, indicating that concentration is structurally induced by pooling rather than an artifact of the training regime.

To test robustness, we inject $K \in \{ 0 , 2 5 , 5 0 , 1 0 0 , 2 0 0 , 4 0 0 \}$ token-targeted hard-negative patches into non-gold documents. Even at moderate injection levels ( $K = 1 0 0 ,$ , MaxSim and Top-4 retain only $2 7 . 8 \%$ and $2 9 . 3 \%$ of baseline recall respectively, while Softmax retains $6 7 . 6 \%$ . Spike hijack rates reach approximately 0.70 for both MaxSim and Top- $k$ , meaning most query tokens select injected distractors. Notably, increasing $k$ (Top-1, Top-4, Top-16) does not restore robustness: all spike-based operators retain only ${ \approx } 2 8 { - } 3 0 \%$ recall, because even at higher $k$ the injected patches dominate the top- $k$ set and poison the average.

A Gaussian control experiment confirms this degradation is semantic rather than length-driven: replacing hard negatives with random Gaussian patches preserves recall ${ \approx } 9 7 \%$ retained, hijack $< 0 . 1 5$ ). We note that these experiments swap pooling operators at inference time on a MaxSim-trained model; end-to-end retraining with each pooling objective may further amplify the benefit of smoother operators and is an important direction for future work.

Figure 2 illustrates this mechanism at the token level: under hard-negative injection, ${ \sim } 8 3 \%$ of tokenwise argmax selections shift to injected patches, whereas Gaussian injection produces minimal routing disruption.

Summary. Real-world experiments confirm that (i) gradient concentration is structural, (ii) semantic distractors induce spike hijacking and recall collapse, and (iii) larger $k$ does not mitigate brittleness, closely mirroring synthetic findings.

