# Semantics-aware Dataset Discovery from Data Lakes with Contextualized Column-based Representation Learning

Grace Fan

Northeastern University

United States

fan.gr@northeastern.edu

Jin Wang

Megagon Labs

United States

jin@megagon.ai

Yuliang Li

Megagon Labs

United States

yuliang@megagon.ai

Dan Zhang

Megagon Labs

United States

dan_z@megagon.ai

# ABSTRACT

Dataset discovery from data lakes is essential in many real application scenarios. In this paper, we propose Starmie, an end-toend framework for dataset discovery from data lakes (with table union search as the main use case). Our proposed framework features a contrastive learning method to train column encoders from pre-trained language models in a fully unsupervised manner. The column encoder of Starmie captures the rich contextual semantic information within tables by leveraging a contrastive multi-column pre-training strategy. We utilize the cosine similarity between column embedding vectors as the column unionability score and propose a filter-and-verification framework that allows exploring a variety of design choices to compute the unionability score between two tables accordingly. Empirical results on real table benchmarks show that Starmie outperforms the best-known solutions in the effectiveness of table union search by 6.8 in MAP and recall. Moreover, Starmie is the first to employ the HNSW (Hierarchical Navigable Small World) index to accelerate query processing of table union search which provides a 3,000X performance gain over the linear scan baseline and a 400X performance gain over an LSH index (the state-of-the-art solution for data lake indexing).

# PVLDB Reference Format:

Grace Fan, Jin Wang, Yuliang Li, Dan Zhang, and Renée Miller.

Semantics-aware Dataset Discovery from Data Lakes with Contextualized Column-based Representation Learning. PVLDB, 14(1): 50 - 60, 2021.

doi:10.14778/3421424.3421431

# PVLDB Artifact Availability:

The source code, data, and/or other artifacts have been made available at https://github.com/megagonlabs/starmie.

# 1 INTRODUCTION

The growing number of open datasets from governments, academic institutions, and companies have brought new opportunities for innovation, economic growth, and societal benefits. To integrate

Renée Miller

Northeastern University

United States

miller@northeastern.edu

and analyze such datasets, researchers in both academia and industry have built a number of dataset search engines to support the application of dataset discovery [3, 7, 16, 18, 31, 38, 43]. One popular example is Google’s dataset search [3] which provides keyword search on the metadata. However, for open datasets, simple keyword search might suffer from data quality issues of incomplete and inconsistent metadata across different datasets and publishers [1, 15, 39, 40]. Thus it is essential to support table search over open datasets, and more generally data lake tables (including private enterprise data lakes), to boost dataset discovery applications, such as finding related tables, domain discovery, and column clustering.

Finding related tables from data lakes [37, 44, 55] has a wide spectrum of real application scenarios. There are two sub-tasks of finding related tables, namely table union search and joinable table search. In this paper, we mainly focus on the problem of table union search, which has been recognized as a crucial task in dataset discovery from data lakes [2, 23, 37, 39, 40, 55, 59]. Given a query table and a collection of data lake tables, table union search aims to find all tables that are unionable with the query table. To determine whether two tables are unionable, existing solutions first identify all pairs of unionable columns from the two tables based on column representations, such as bag of tokens or bag of word embeddings. They then devise some mechanism to aggregate the column-level results to compute the table unionability score.

State-of-the-art: Early work on finding unionable tables used table clustering followed by simple syntactic measures such as the difference in column mean string length and cosine similarities to determine if two tables are unionable [4]. Table union search [40] improved on this by applying a rich collection of column representations including syntactic, semantic (leveraging ontologies), and natural language (based on word-embeddings) column representations. Two important innovations of this work were the modeling of data lake context to create an ensemble unionability score which models the surprisingness of a score given the score distributions within a data lake and the use of LSH indices to make table union search fast over large data lakes [40]. More recently $D ^ { 3 } L$ [2] added additional column representations based on regular expression matching and SANTOS [23] added to the column representations, representations of binary relationships. In parallel to these search-based approaches, the mighty hammer of deep learning has been applied to the problem of column matching (determining the semantic type of a column) [21, 54]. Since these

<table><tr><td rowspan="4">Table A:</td><td>Name</td><td>Mode of Travel</td><td>Purpose</td><td>Destination</td><td>Day</td><td>Month</td><td>Year</td><td>Expense</td></tr><tr><td>Philip Duffy</td><td>Air</td><td>Regional Meeting</td><td>London</td><td>10</td><td>April</td><td>2019</td><td>189.06</td></tr><tr><td>Jeremy Oppenheim</td><td>Taxi</td><td>Exchange Visit</td><td>Ottawa</td><td>30</td><td>Jul</td><td>2019</td><td>8.08</td></tr><tr><td>Mark Sedwill</td><td>Air</td><td>Evening Meal</td><td>Bristol</td><td>02</td><td>September</td><td>2019</td><td>50</td></tr></table>

Table B:   

<table><tr><td>Name</td><td>Date</td><td>Destination</td><td>Purpose</td></tr><tr><td>Clark</td><td>23/07</td><td>France</td><td>Discuss EU</td></tr><tr><td>Gyimah</td><td>03/09</td><td>Belgium</td><td>Build Relations</td></tr><tr><td>Harrington</td><td>05/08</td><td>China</td><td>Discuss Productivity</td></tr></table>

Table C:   

<table><tr><td>Bird Name</td><td>Scientific Name</td><td>Date</td><td>Location</td></tr><tr><td>Pine Siskin</td><td>Carduelis Pinus</td><td>2019</td><td>Ottawa</td></tr><tr><td>American Robin</td><td>Turdus migratorius</td><td>2019</td><td>Ottawa</td></tr><tr><td>Northern Flicker</td><td>Colaptes auratus</td><td>2019</td><td>London</td></tr></table>

# Figure 1: An example of table union search on Open Data.

approaches are supervised, they can only be applied to finding a limited set of semantic types (78 in their experiments), and while not a general solution for unionability in data lakes, they can be used in an offline fashion to find unionable tables containing the types on which they are trained.

However, there are still plenty of opportunities to further improve the performance of table union search. One important issue is to learn sufficient contextual information between columns in tables so as to determine the unionability. This point can be illustrated in the following motivation example.

Example 1.1. Figure 1 shows an example of finding unionable tables. Given the query Table A, existing approaches first find unionable columns. In this example, the column Destination in Table A will be deemed more unionable with Location from Table C than with Destination from Table B. This is because the syntactic similarity score, e.g. overlap and containment Jaccard, between the two Destination columns is 0; while the average word embedding of cities (Table A) is also not as close to that of nations (Table B). Similarly, if an ontology is used, Table A and Table C shares the same class while the values in B are in different (though related) classes. Meanwhile, looking at the tables as a whole we observe that Table A is actually irrelevant to Table C. But as existing solutions only look at the pair of single columns when calculating column unionability score, the columns Year/Date and Destination/Location of the two tables might be wrongly aligned together. Even techniques that look at relationships [23] can be fooled by the value overlap in this relationship and determine the relationship Year-Destination in Table A to be unionable with Date-Location in Table C. This kind of mistake can be avoided by looking at a table’s context, i.e. information carried by other columns within a table. Looking at the table as a whole, a method should be able to recognize that the Year in Table A is part of a travel date while in Table C it is the date of discovery of a bird; and Destination in Table A refers to the cities to which the officers are traveling; whereas Location in Table C is the city where a bird is found.

From the above example, we focus on the following challenges in proposing a new solution. Firstly, it is essential to learn richer semantics of columns based on natural language domain. To this end, we require a more powerful approach to learn the column representation so as to capture richer information instead of relying on simple methods like the average over bag of word embeddings utilized in previous studies [2, 13] or even the similarity of the word embedding distributions [40]. Secondly, we argue that it is crucial to utilize the contextual information within a table to learn the representation of each column, which is ignored by previous studies. Even proposals for capturing relationship semantics do not use contextual information to learn column representations [23].

Finally, due to the large volume of data lake tables, it is also a great challenge to develop a scalable and memory-efficient solution.

We propose Starmie, an end-to-end framework for dataset discovery from data lakes with table union search as the main use case. Starmie uses pre-trained language models (LMs) such as BERT [12] to obtain semantics-aware representations for columns of data lake tables. While pre-trained LMs have been shown to achieve state-ofthe-art results in table understanding applications [11, 29, 45], their good performance heavily relies on high-quality labeled training data. For the problem setting of table union search [39, 40], we must come up with a fully unsupervised approach in order to apply pre-trained LMs to such applications, something not yet supported by previous studies. Starmie addresses this issue by leveraging contrastive representation learning [10] to learn column representations in a self-supervised manner. An innovation of this approach is to assume that two randomly selected columns in a data lake can be used as negative training examples. For positive examples, we propose and use novel data augmentation methods. The framework defines a learning objective that connects the same or similar columns in the representation space while separating distinct columns. As such, Starmie can apply the pre-trained representation model in downstream tasks such as table union search without requiring any labels. We also propose to combine the learning algorithm with a novel multi-column table transformer model to learn contextualized column embeddings that model the column semantics depending on not only the column values, but also their context within a table. While a recent study SANTOS [23] can reach a similar goal by employing a knowledge base, our proposed methods can automatically capture such contextual information from tables in an unsupervised manner without relying on any external knowledge or labels.

Based on the proposed column encoders, we use cosine similarity between column embeddings as the column unionability score and develop a bipartite matching based method to calculate the table unionability score. We propose a filter-and-verification framework that enables the use of different indexing and pruning techniques to reduce the number of computations of the expensive bipartite matching. While most previous studies employed LSH index to improve the search performance, we also make use of HNSW (Hierarchical Navigable Small World) index [34] to accelerate query processing. Experimental results show that HNSW can significantly improve the query time while only slightly reducing the MAP/recall scores. Besides table union search, we further conduct two case studies to show that Starmie can also support other dataset discovery applications such as joinable table search and column clustering. We believe these results show great promise in the use of contextualized, self-supervised embeddings for many table understanding tasks.

Our contributions can be summarized as the following.

• We propose Starmie, an end-to-end framework to support dataset discovery over data lakes with table union search as the main use case.   
• We develop a contrastive learning framework to learn contextualized column representations for data lake tables without requiring labeled training instances. Starmie achieves an improvement of $6 . 8 \%$ in both MAP and recall compared with the best state-of-the-art method, with a MAP of $9 9 \%$ , a significant margin compared with previous studies.

• We design and implement a filter-and-verification based framework for computing the table-level unionability score which can accommodate multiple design choices of indexing and pruning to accelerate the overall query processing. By leveraging the HNSW index, Starmie achieves up to three orders of magnitude in performance gain for query time relative to the linear scan baseline.   
• We conduct an extensive set of experiments over two real world data lake corpora. Experimental results demonstrate that the proposed Starmie framework significantly outperforms existing solutions in effectiveness. It also shows good scalability and memory efficiency.   
• We further conduct case studies to show the flexibility and generality of our proposed framework in other dataset discovery applications.

# 2 OVERVIEW

# 2.1 Problem definition

A data lake consists of a collection of tables $\mathcal { T }$ . Each table $T \in { \mathcal { T } }$ consists of several columns $\{ t _ { 1 } , \ldots , t _ { m } \}$ where each column $t _ { i }$ can be from different domains. Here $m$ is the number of columns in table ?? (denoted as $\left. T \right. = m )$ ). We will use the notation $T$ to denote both the table and its set of columns if there is no ambiguity. To determine the unionability between two columns, following previous studies, we employ column encoders to generate the representations of columns. Then the column unionability score can be computed to measure the relevance between those representations. A column encoder M takes a column ?? as input and outputs $M ( t )$ as the representation. Given two columns $t _ { i }$ and $t _ { j }$ , the column unionability score is computed as $\mathcal { F } ( M ( t _ { i } ) , M ( t _ { j } ) )$ , where $\mathcal { F }$ is a scoring function between two column representations.

Based on the column unionability scores, we compute the table unionability score between two tables, which is obtained by aggregating the column unionability scores introduced above. Given two tables $S$ and $T$ , we define a table unionability scoring mechanism as $U = \{ \mathcal { F } , M , \mathcal { A } \}$ , where $\mathcal { M }$ and $\mathcal { F }$ are the column encoder and scoring function for two column representations, respectively. Here $\mathcal { A }$ is a mechanism to aggregate the column unionability scores between all pairs of columns from the two tables. We will introduce the details of $\mathcal { A }$ later in Section 4.

Following the above discussions, we can formally define the table union search problem as a top- $\mathbf { \nabla } \cdot \mathbf { k }$ search problem as Definition 2.1:

Definition 2.1 (Table Union Search). Given a collection of data lake tables $\mathcal { T }$ and a query table ??, top-k table union search aims at finding a subset $s \subseteq { \mathcal { T } }$ where $| S | = k$ and $\forall T \in S$ and $T ^ { \prime } \in \mathcal { T } - S$ , we have $U ( S , T ) \ge U ( S , T ^ { \prime } )$ .

# 2.2 System architecture

Figure 2 shows the overall architecture of Starmie that solves table union search in two stages: offline and online.

During the offline stage, Starmie pre-trains a column representation model that encodes columns of data lake tables into dense high-dimensional vectors (i.e., column embeddings). Then, we apply the trained model to all data lake tables to obtain the column embeddings via model inference. We store the embedding vectors in

![](images/8c96efd37a6dbdd64ec6134682c0a06fa81b6b20323ea9bf1bd3a3d0d3e3cac4.jpg)  
Figure 2: During the offline phase, Starmie pre-trains a multicolumn table encoder using contrastive learning and stores the embeddings of data lake columns in vector indices like HNSW. During online processing, Starmie retrieves candidate tables with similar contextualized column embeddings then verifies their table-level unionability scores using column alignment algorithms.

efficient vector indices for online retrieval. A key challenge for the offline stage is to train high-quality column encoders that capture the semantics of tabular data. In Starmie, we follow a recent trend [11, 29, 45] of table representation learning that encodes tabular data using pre-trained language models (LMs). Pre-trained LMs have achieved state-of-the-art performance on table understanding tasks such as column type and relation type annotation [45]. However, the good performance of pre-trained LMs requires fine-tuning on high-quality labeled datasets, which are always not available in table search applications such as table union search. Using pre-trained LMs off-the-shelf is also problematic as the column embeddings cannot capture (ir-)relevance between columns or the contextual information within tables. To this end, in Section 3, we propose a contrastive learning framework for learning high-dimensional column representations in fully unsupervised manner. We combine the framework with a multi-column table model that captures column semantics from the column values while taking the table context into account. Then we apply the column encoder to all tables to convert each table into a collection of embedding vectors.

During the online stage, given an input query table, we retrieve a set of candidate tables from the vector indices by searching for data lake column embeddings of high column-level similarity with the input columns. Starmie then applies a verification step for checking and ranking the candidates for the top- $\mathbf { \nabla } \cdot k$ tables with the highest table-level unionability scores. The first challenge for the online stage is how to efficiently search for unionable columns. This is not a trivial task due to the massive size of data lakes. We address this challenge by allowing different design choices of state-of-the-art high-dimensional vector indices. Yet another challenge is designing a table unionability function that can effectively aggregate the column unionability scores. As in other studies, we employ weighted bipartite graph matching. To address its limitation of high computation complexity, we introduce a novel algorithm to reduce the number of expensive calls to the exact matching algorithm by deducing lower and upper bounds of the matching score (Section 4).

![](images/5e84804eb7347c5998a81911da42ca80edf7a43c75ad449c33b5397bfd283991.jpg)  
Figure 3: Contrastive learning with single-column input.

# 3 LEARNING CONTEXTUALIZED COLUMN EMBEDDINGS

We now describe the offline stage for training high-quality column encoders. The encoder pre-processes tables into sequenced inputs and uses a pre-trained LM to encode each column into a highdimensional vector. We first introduce background knowledge in Section 3.1. We describe a novel contrastive learning approach for table encoders in Section 3.2 and generalize it to multi-column encoders for contextualized embeddings in Section 3.3. Finally, we describe the table pre-processing approaches to generate the input for such learning processes in Section 3.4.

# 3.1 Background

Contrastive learning is a self-supervision approach that learns data representations where similar data items are close while distinct data items are far apart. In Starmie, we adopt SimCLR [10] which was recently shown to be effective in Vision and NLP applications. Figure 3 illustrates the high-level idea of the algorithm. The goal is to learn an encoder M (e.g., a column encoder) that takes a data item (e.g., a column) as input and encodes it into a high-dimensional vector. To train the encoder in a self-supervised manner without labels, SimCLR relies on (1) a data augmentation operator generating semantic-preserving views (in our context this means $X _ { \mathsf { o r i } }$ and $X _ { a \mathbf { u } \mathbf { g } }$ that are unionable) of the same data item and (2) a sampling method (e.g., uniform sampling from a large collection) that returns pairs of data items (i.e., $X$ and ?? ) that are distinct (meaning non-unionable) with high probability. SimCLR then applies a contrastive loss function that connects the representations of the semantic-preserving (unionable) views meanwhile separating those of the sampled distinct (non-unionable) items. Next, we illustrate how we apply the algorithm for training a single-column encoder.

# 3.2 Contrastive Learning Framework

The goal is to connect representations of the same or unionable columns in their representation space while separating representations of distinct columns. To achieve the first goal, Algorithm 1 leverages a data augmentation operator op (Line 5). Given a batch of columns $X = \{ x _ { 1 } , . . . , x _ { N } \}$ where $N$ is the batch size, op transforms $X$ into a semantics-preserving view $X _ { \mathsf { a u g } }$ . We design the augmentation operator to be uniform sampling of the values from the original column. By doing so, we can generate diverse views of the same column while all views preserve the original semantic types. Then $M$ can encode the batches $X$ (also $X _ { \mathsf { o r i } }$ which is a copy

Algorithm 1: SimCLR pre-training   
Input: A collection $D$ of data lake columns Variables :Number of training epochs n_epoch; Data augmentation operator op; Learning rate $\eta$ Output: An embedding model $\mathcal{M}$ 1 Initialize $\mathcal{M}$ using a pre-trained LM;   
for $\mathrm{ep} = 1$ to n_epoch do   
3 Randomly split $D$ into batches $\{B_1,\dots B_n\}$ . for $B\in \{B_1,\ldots B_n\}$ do / \* augment and encode every item \*/ $B_{\mathrm{ori}},B_{\mathrm{aug}}\gets$ augment(B,op); Zori,Zaug $\leftarrow$ M(Bori),M(Baug); / \* Equation (1) and (2) \*/ L $\leftarrow$ Lcontrast(Zori,Zaug); / \* Back-prop to update M \*/ M $\leftarrow$ back-propagate(M, $\eta ,\partial \mathcal{L} / \partial \mathcal{M})$ 9 return M;

of $X$ in the figure) and $X _ { \mathsf { a u g } }$ into column embedding vectors $\vec { Z } _ { \mathrm { o r i } }$ and $\vec { Z } _ { \mathrm { a u g } }$ respectively. Note that $\vec { Z } _ { \mathrm { o r i } }$ and $\vec { Z } _ { \mathrm { a u g } }$ are both matrices with size $N$ times the dimension of embedding vector (e.g., 768 for BERT). 5

Next, the algorithm leverages a contrastive loss function to connect the semantics-preserving views of columns and separate representations of distinct columns (Line 6). More specifically, let $\vec { Z } = \{ \vec { z } _ { i } \} _ { 1 \leq i \leq 2 N }$ be the concatenation of the two encoded views $\vec { Z } _ { \mathrm { o r i } }$ and $\vec { Z } _ { \mathrm { a u g } }$ of batch $X$ introduced above. Here $\vec { z } _ { i }$ is the $i$ -th element of $\vec { Z } _ { \mathrm { o r i } }$ for $i \leq N$ and the $( i - N )$ -th element of $\vec { Z } _ { \mathrm { a u g } }$ for $i > N$ We first define a single-pair loss $\ell ( i , j )$ for an element pair $( \vec { z } _ { i } , \vec { z } _ { j } )$ to be Equation 1.

$$
\ell (i, j) = - \log \frac {\exp (\operatorname {s i m} (\vec {z} _ {i} , \vec {z} _ {j}) / \tau)}{\sum_ {k = 1} ^ {2 N} \mathbb {1} _ {[ k \neq i , k \neq j ]} \exp (\operatorname {s i m} (\vec {z} _ {i} , \vec {z} _ {k}) / \tau)} \tag {1}
$$

where sim is a similarity function such as cosine and $\tau$ is a temperature hyper-parameter in the range (0, 1]. We fix $\tau$ to be 0.07 empirically. Intuitively, by minimizing this loss for a pair $( \vec { z } _ { i } , \vec { z } _ { j } )$ that are views of the same columns, we (i) maximize the similarity score sim $\left( \vec { z } _ { i } , \vec { z } _ { j } \right)$ in the numerator and (ii) minimize $\vec { z } _ { i }$ ’s similarities with all the other elements in the denominator.

Next, we can obtain the contrastive loss by averaging all matching pairs shown in Equation 2 (Line 7):

$$
\mathcal {L} _ {\text {c o n t r a s t}} = \frac {1}{2 N} \sum_ {k = 1} ^ {N} [ \ell (k, k + N) + \ell (k + N, k) ] \tag {2}
$$

where each term $\ell ( k , k + N )$ and $\ell ( k + N , k )$ refers to pairs of views generated from the same column.

# 3.3 Multi-column Table Encoder

While the method shown in Algorithm 1 learns column representations based on values within a column itself, it cannot take the contextual information of a table into account. For example, the single-column model can understand that a column consisting of values “1997 1998 . . . ” is a column about years, but depending on the context of other columns present in the same table, the same

![](images/72a0ebfbc1b9309681c563acf70328120800201b13912f9eadffaaeddcdea548.jpg)  
Figure 4: Multi-column table encoder.

column can represent “years in which a species of bird was observed in a specific area” or “years of car production”, etc. As illustrated in the example in Figure 1, such understanding is important for deciding whether two tables are unionable or not.

To address this problem, Starmie combines contrastive learning with a multi-column table encoder illustrated in Figure 4. The model starts with serializing an input table into a string by concatenating cell values from each column. Following the implementation of tokenizers in the HuggingFace library, it also adds a special separator token ${ } ^ { * * } < s > { } ^ { * }$ to indicate the start of each column. Next, we feed the sequence as the input to a pre-trained LM such as RoBERTa [33]. Since the special token ${ } ^ { * * } < s > { } ^ { * }$ at the start a sequence in RoBERTa is pre-trained to capture the sequence representations, we also expect it to capture representations of columns given the table context.

The pre-trained LM first converts the input sequence into a sequence of token embeddings independent of their context then applies 12 or more Transformer layers [46] on top. The self-attention mechanism in the Transformer layers convert the word embeddings into a sequence of contextualized embeddings. These vector representations depend not only on the tokens themselves (e.g., “1797”) but also their context (e.g., “Albany”). As such, we can extract the representations of the separator tokens (i.e., $^ { * } < s > ^ { * } )$ ) to be the contextualized column embeddings.

To apply contrastive learning using the multi-column model, we adapt the SimCLR algorithm (Algorithm 1) as follows. First, we create the batches of columns (Line 3) by uniformly sampling a batch of tables from all data lake tables and form each batch of columns ?? using all columns from the sampled tables. To augment the batch ??, instead of transforming each column independently, we apply table-level augmentation operators such as row sampling and column sampling (Line 5). Note that in the multi-column setting, the augmentation operators produce views of tables with pairs of columns that align with each other. These pairs form the positive pairs in the contrastive loss as we illustrate in Figure 5.

We summarize the supported augmentation operators in Table 1. While there is a large design space of the operators, we summarize them by the levels (e.g., cell, row, column) of the table to which the operators apply. The cell-level operators are general transformations also used in related tasks such as Entity Matching [29]. The row and column-level operators cover different ways for creating samples of rows/columns. One can also perform more complex transformations by applying multiple operators simultaneously. In

![](images/17e3d81c45473eb5a6c40c9473671d5ac601da5c503bbce6554bb29680d47b40.jpg)

![](images/5d4ab601c931fe742f95179a5909fe97e043532fe60e42770b14a600d1731565.jpg)  
Figure 5: Table-level augmentation and column alignment.   
Figure 6: Contrastive learning positive and negative pairs.

our ablation study (see Appendix B.1), we find that the simple column sampling operator (drop_col) provides the best performance.

We then apply the multi-column model on the original and augmented views of tables to obtain the contextualized column embeddings $\vec { Z } _ { \mathrm { o r i } }$ and $\vec { Z } _ { \mathrm { a u g } }$ (Line 6) and compute the contrastive loss (Line 7). Note that in the multi-column setting, the positive pairs (for which we maximize the similarity) consist of the aligned pairs of columns generated by the augmentation operators. We minimize the similarity of all other pairs which include (i) pairs of unaligned columns from the same table and (ii) all pairs of columns from two distinct tables. By doing so, the algorithm learns representations that can distinguish columns with the same/different table contexts, thus creating the positive and negative pairs shown in Figure 6. More formally, let $P$ be the set of indices of all aligned pairs of columns in the batch ??, we minimize the multi-column contrastive loss shown in Equation 3:

$$
\mathcal {L} _ {\text {m u l t i - c o l u m n}} = \frac {1}{2 | P |} \sum_ {(i, j) \in P} [ \ell (i, j) + \ell (j, i) ]. \tag {3}
$$

# 3.4 Table Preprocessing

Typical pre-trained LMs like BERT support an input length of at most 512 sub-word tokens, while a column in real-world tables such as those in Open Data may contain thousands or even millions of tokens. To apply the proposed techniques in Section 3.2 and 3.3 on data lake tables, we must preprocess the columns to reduce the input length to fit the token limit of LMs, while preserving their semantics. The procedure is outlined in Algorithm 2, while the full

Table 1: Data augmentation operators at different levels.   

<table><tr><td>Level</td><td>Operators</td><td>Description</td></tr><tr><td>Cell</td><td>drop_cell, drop_token, swap_token, repl_token</td><td>Dropping a random cell; Drop- ping/swapping tokens within cells</td></tr><tr><td>Row</td><td>sample_row, shuffle_row</td><td>Sampling x% (e.g., 50) of rows; Shuf- fling the row order</td></tr><tr><td>Col</td><td>drop_col, drop_num_col, shuffle_col</td><td>DroppingX (numeric) columns; Shuffling column order</td></tr></table>

Algorithm 2: Table Preprocessing   
Input: A table $T$ ; A token scoring function such as TF-IDF TF-IDF(); The max #tokens $m$ . Variables :Preprocessing mode $\in$ {"row", "cell", "token"} Output: The table $T'$ with selected rows, cells, or tokens  
1 foreach cell $c \in T$ do  
    /* Sum over token scores */  
2 cell_score(c) $\leftarrow \sum_{\text{token} t \in c}$ TF-IDF(c);  
3 foreach row $r \in T$ do  
    /* Sum over cell scores */  
4 row_score(c) $\leftarrow \sum_{\text{cell} c \in r}$ cell_score(c);  
5 if mode = "row" then  
6 return Top- $n$ rows with highest row_score up to length $m$ ;  
7 if mode = "cell" then  
8 return Top- $n$ cells with highest cell_score for each column up to length $m/|T|$ ; // |T|: number of columns  
9 if mode = "token" then  
10 return Top- $n$ tokens with highest TF-IDF for each column up to length $m/|T|$ ; // |T|: number of columns

details with design choices (scoring functions, row/column orders, and alignment rules) are in the appendix due to the space limitation.

Algorithm 2 illustrates the steps of table pre-processing. It first assigns an importance score for each cell by first computing the TF-IDF scores of every token in a cell and then averaging the TF-IDF scores of all tokens. Then it ranks the average cell-level scores of rows and then selects the rows to be included in the serialization result. Here we finish this step in a deterministic way: by ranking in the descending order of the importance score, until we reach the token budget for each column.

# 4 ONLINE QUERY PROCESSING

In this section, we introduce how to find unionable tables based on contextualized column embeddings. We first introduce the table unionability scores and the overall workflow of online query processing in Section 4.1. Then we discuss the design choices for reducing the number of candidates using vector indices and deducing bounds for more efficient verification in Sections 4.2 and 4.3, respectively. Note that the online processing techniques explored here are not limited to any specific column encoders, they are also applicable to other dense-vector column representation methods [21, 54].

# 4.1 Table-level Matching Score

After training a column encoder $\mathcal { M }$ using techniques from Section 3, we can then obtain the embedding vectors for all columns in data lake tables via model inference. The column unionability score between two columns ?? and ?? can be calculated by using cosine similarity as $\mathcal { F }$ between those embedding vectors. Next, we define the function $\mathcal { A }$ for aggregating the column unionability scores to compute the table unionability. Motivated by the idea of $c$ -alignment [40] that aims to find a maximum set of one-to-one alignment between columns in two tables, we propose modeling table unionability as a weighted bipartite graph matching problem. More formally, given two tables ?? and $T$ with $m$ and $n$ columns respectively, we construct a bipartite graph $G = \langle S , T , E \rangle$ where

![](images/c2673aa8c10f98978d3cbf88cb426d592707bfda152aed977aa7c171b6eb24dd.jpg)  
Figure 7: Example of table unionability score via maximum bipartite matching. Solid (red) lines denote the edges belonging to the maximum matching.

the nodes $S$ and $T$ are the two sets of columns. The edges in $E$ denote the column unionability score between each pair of columns. Then table unionability score $U ( S , T )$ can be calculated by finding the maximum bipartite matching of graph ??. In order to remove the noise caused by dissimilar pairs of columns, we follow the de-noising strategy from fuzzy string matching [49] by introducing a hyper-parameter $\tau$ as the similarity lower bound: given two columns $s \in S$ and $t \in T$ , there is an edge $\langle s , t \rangle \in E$ iff $\mathcal { F } ( s , t ) \ge \tau$ .

Example 4.1. We show an example of computing the table unionability score in Figure 7. Suppose there are two tables $S$ and $T$ with 4 and 3 columns respectively and the threshold $\tau$ for column unionablity score is 0.5. Since the cosine similarity between $s _ { 3 }$ and $t _ { 3 }$ is $0 . 3 \ : ( < \tau )$ , the edge between them is discarded (denoted with a dash line). For the ease of presentation, we omit the remaining dash lines between other nodes in the figure. The maximum bipartite matching of this graph consists of the edges in red (solid lines), which are $\left. s _ { 1 } , t _ { 1 } \right.$ , $\left. s _ { 2 } , t _ { 2 } \right.$ and $\langle s _ { 4 } , t _ { 3 } \rangle$ with a score of 2.15.

In order to find the tables with top-k highest table unionability scores with the given query table $S$ , a straightforward method is to conduct a linear scan: we use a min-heap with cardinality of $k$ to keep the results of top-k search, then for each table ?? in the data lake, we directly compute $U ( S , T )$ ; and if the score is higher than the top element of the min-heap, we replace the top element with it and adjust the min-heap accordingly. However, since the time complexity of weighted bipartite matching is $O ( n ^ { 3 } \log n )$ , where $n$ is the total number of columns in two tables, it is rather expensive to traverse all tables in a data lake. A scalable solution requires reducing (i) the number of accessed tables and (ii) the computational overhead of verifying each pair of tables.

We propose a filter-and-verification framework to address this issue as illustrated in Algorithm 3. Instead of doing a linear scan over all data lake tables, it employs filter mechanisms to identify a set of candidate tables $c$ for further verification (line: 3). As a result, it can reduce the number of expensive verification operations Verify $( S , T )$ . This is realized by the function findCandidates (Section 4.2). Then for all the candidate tables, we further come up with a pruning mechanism to estimate the lower bound $\mathbf { L B } ( S , T )$ and upper bound $\mathbf { U B } ( S , T )$ of $U ( S , T )$ . If the lower bound is larger than the current lowest score, we can directly replace it with the top element without further verification (line: 10). Similarly, if the upper bound is no larger than the current lowest score, we can directly discard it (line: 12). This pruning mechanism is effective since LB and UB are much more efficient to estimate than the exact verification Verify $( S , T )$ (Section 4.3).

Algorithm 3: Online Query Processing   
Input: $S$ : the query table; $\mathcal{T}$ : the set of data lake tables; Variables :k: the number of desired results; $\tau$ : threshold of column unionable score; Output: $\mathcal{H}$ : The top-k unionable tables Initialize $\mathcal{H}$ and $C$ as 0;   
for all columns $s\in S$ do $C = C\cup$ findCandidates(s, $\tau ,T)$ .   
for all tables $T\in C$ do if $|\mathcal{H}| <   k$ then Compute Verify $(S,T)$ and add $T$ into $\mathcal{H}$ else   
if $X\gets$ the score of top element of $\mathcal{H}$ if $LB(S,T) > X$ then Replace the top element of $\mathcal{H}$ with $T$ else if $UB(S,T)\leq X$ then Discard $T$ else if Verify $(S,T) > X$ then Replace the top element of $\mathcal{H}$ with $T$ ..   
return $\mathcal{H}$

# 4.2 Reducing the Number of Candidates

Given a column with its embedding vector, we need to quickly identify tables from the data lake that contain unionable columns, which is realized by the findCandidates function in Algorithm 3. This is a problem of similarity search over high-dimensional vectors. Locality Sensitivity Hashing (LSH) [19] has been used in previous studies of table search to find joinable [59], unionable [40], and related columns [2] in sub-linear time. The basic idea is to use a family of hash functions to map high-dimensional vectors into a number of buckets, where the probability that two vectors are hashed into the same bucket is correlated to the value of a certain similarity metric between them. Following this work, we build a simHash [8] LSH index to estimate the cosine similarity between column embedding vectors. Then for each query column vector ??, we can quickly find a set of similar column vectors via an index lookup. Then the candidate set $c$ can be obtained by the union of candidates returned by utilizing each column vector ?? to query the index. In addition to LSH, we also explore the more recent HNSW [34]. HNSW is a proximity graph with multiple layers where two vertices are linked based on their proximity. It supports fast nearest neighbor search with high recall. We find that HNSW improves the query time by orders of magnitude and thus allows Starmie to support querying over the WDC corpus with 50M tables, which is much larger than the previously supported datasets for table union search.

Since such index structures return approximate instead of exact results, there might be some false negatives in the top-k results. Nevertheless, we find in the experiments that the effectiveness loss caused by the false negatives is within a reasonable range. Meanwhile, the query time can be reduced by one to three orders of magnitude (details in Section 5.3).

# 4.3 Pruning Mechanism for Verification

Once a candidate table is found, we can reduce the expensive verification cost by quickly computing lower and upper bounds on

the unionability score. We first look at how to estimate the upper bound $\mathbf { U B } ( S , T )$ between two tables $S$ and ?? . Recall that in maximum weighted bipartite matching, each column/node in both ?? and $T$ can be covered by at most 1 edge in the edges of the maximum matching. If we remove this constraint, since nodes can appear in multiple edges, the new optimal matching is easy to compute. Moreover, as it allows edges with greater weights, the total score forms an upper bound of the true table unionability score $U ( S , T )$ . For the upper bound $\mathbf { U B } ( S , T )$ , we first sort the edges by their weights in descending order. Then we add edges with the largest weights into the matching in a greedy manner. This process is repeated until all columns in $S$ or $T$ are covered or all edges are used. The time complexity of the above process for calculating $\mathbf { U B } ( S , T )$ is $O ( | E | \log | E | + n )$ , where $| E |$ is the number of edges in ??. It is much cheaper to compute than the real table unionability score.

Next, we introduce how to quickly estimate a meaningful lower bound $\mathbf { L B } ( S , T )$ . For lower bounds, we would like to find a set of edges that do not violate the constraint of bipartite matching, i.e., each column in the two tables is covered by one edge. We can also achieve this goal via a greedy algorithm. Similar to computing the upper bound, we sort the edges by weight in descending order and pick edges with the largest weights. After that, we remove edges that are associated with the columns in the selected edges so as to avoid violations. The termination condition of this process is also the same as that of calculating the upper bound. Since the resulting matching does not necessarily cover all nodes in ?? or $T$ , the total weight $\mathbf { L B } ( S , T )$ is a lower bound of the maximum matching. The time complexity of calculating $\mathbf { L B } ( S , T )$ is also $O ( | E | \log | E | + n )$ .

Example 4.2. We use the example in Figure 7 to illustrate the upper bound computation. Note this example is designed to illustrate the algorithm, not to model the actual distribution of weights in a data lake. We fetch edges in the descending order of weight: $\left. s _ { 1 } , t _ { 2 } \right.$ , $\left. s _ { 1 } , t _ { 1 } \right.$ , $\left. s _ { 2 } , t _ { 2 } \right.$ , and $\langle s _ { 4 } , t _ { 3 } \rangle$ . At this point, since all nodes $\{ t _ { 1 } , t _ { 2 } , t _ { 3 } \}$ in $T$ are covered, we stop here. The upper bound is $0 . 8 5 + 0 . 8 + 0 . 7 + 0 . 6 5 = 3$ , larger than the exact value 2.15.

To compute the lower bound, we start from edge $\langle s _ { 1 } , t _ { 2 } \rangle$ and then remove all edges associated with $s _ { 1 }$ and $t _ { 2 }$ . The remaining edge with maximum weight is $\left. s _ { 4 } , t _ { 3 } \right.$ . After involving this edge into the matching, there is no remaining one and the algorithm stops here. Hence, the lower bound is $0 . 8 5 + 0 . 6 5 = 1 . 5$ , which is smaller than the exact value 2.15.

# 5 EXPERIMENTS

We now present an evaluation of Starmie on real-world data lake corpora. First, we show that Starmie achieves new state-of-theart results on table union search by outperforming the previous best methods by $6 . 8 \%$ in MAP and Recall. Next, our scalability experiments show that Starmie (especially with the HNSW index) achieves significant performance gain (up to 3,000x) while preserving reasonable effectiveness performance. Lastly, we conduct case studies to show how Starmie can generalize to another two dataset discovery applications: column clustering and table discovery for downstream machine learning tasks. We include additional results and discussions in the appendix that is available in the full technical report [14].

# 5.1 Experiment Setup

5.1.1 Environment. We implement Starmie in Python using Pytorch and the Hugging Face Transformers library [50]. For contrastive learning, we use RoBERTa [33] as the base language model. We set the hyper-parameters batch size to 64, learning rate to 5e-5, and max sequence length to 256 across all the experiments. All experiments are run on a server with configurations similar to those of a p4d.24xlarge AWS EC2 machine with 8 A100 GPUs. The server has 2 AMD EPYC 7702 64-Core processors and 1TB RAM.

5.1.2 Datasets. We use five benchmark datasets with statistics detailed in Table 2. Firstly, we evaluate the effectiveness on the first three benchmark datasets, which are subsets of real Open Data. Since accuracy requires manually labeled ground truth, such datasets are not very large. We only use them to conduct the experiments of effectiveness reported in Section 5.2. The SANTOS Small benchmark [23] consists of 550 real data lake tables drawn from 296 Canada, UK, US, and Australian open datasets, and 50 query tables. From Table Union Search [40], there are two available benchmarks: TUS Small and TUS Large. TUS Small benchmark consists of 1,530 data lake tables that are derived from 10 base tables from Canada open data. We also use the larger benchmark, TUS Large, which consists of $\sim 5 , 0 0 0$ data lake tables derived from 32 base tables from Canada open data. For these two benchmarks, we randomly select 150 and 100 query tables, respectively, following previous studies [23, 40]. The SANTOS1 and ${ \bar { \mathsf { T } } } \mathsf { U } S ^ { 2 }$ benchmarks, along with their ground truth of unionable tables, are publicly available.

The last two benchmark datasets are utilized in the experiments for efficiency and scalability. Compared with the previous three datasets, these two datasets do not have ground truth labels but have much larger cardinalities. The SANTOS Large benchmark contains ∼11K raw data lake tables from Canada and UK open data, and 80 query tables. We also run experiments on the WDC web tables corpus [26] which contains 50.8 million relational web tables extracted from the Common Crawl. We randomly select 30 tables as the query.

Table 2: Effectiveness (top) and scalability (bottom) benchmarks.   

<table><tr><td>Benchmark</td><td># Tables</td><td># Cols</td><td>Avg # Rows</td><td>Size (GB)</td></tr><tr><td>SANTOS Small</td><td>550</td><td>6,322</td><td>6,921</td><td>0.45</td></tr><tr><td>TUS Small</td><td>1,530</td><td>14,810</td><td>4,466</td><td>1</td></tr><tr><td>TUS Large</td><td>5,043</td><td>54,923</td><td>1,915</td><td>1.5</td></tr><tr><td>SANTOS Large</td><td>11,090</td><td>123,477</td><td>7,675</td><td>11</td></tr><tr><td>WDC</td><td>50M</td><td>250M</td><td>14</td><td>500</td></tr></table>

5.1.3 Metrics. For effectiveness, we perform evaluation based on the ground truth from the first three benchmarks. For the TUS benchmarks, the tables are synthetically-partitioned from tables of distinct domains, so the ground truth is created in a generative manner. As for the SANTOS Small benchmark, the tables have been manually-annotated to create a ground truth listing expected unionable tables to each query table. Then we follow previous studies [2, 23, 35, 40] and use the Mean Average Precision at k (MAP@k), Precision at k (P@k) and Recall at k $( \mathrm { R } @ \mathrm { k } )$ to evaluate

the effectiveness in returning the top-k results. We compute each score by averaging 5 repeated runs. For efficiency, we measure the average time per query.

5.1.4 Baselines. For effectiveness experiments, we compare our approach, Starmie, with the following existing approaches.

• $D ^ { 3 } \mathsf { L }$ [2] extends Table Union Search [40] for the problem of finding related tables by using table features such as column names, value overlap, and formatting. To compare fairly with Starmie, we omit the column name feature.   
• SANTOS [23] proposes an approach that leverages both columns and relationships between columns by using external and selfcurated knowledge bases.   
• Sherlock [21] is a representation learning method that leverages several column features such as table statistics and word embeddings to learn the embedding vector of a column.   
• SATO [54] extends Sherlock by capturing the table context using LDA, and thus performing a form of multi-column prediction.   
• SingleCol is our column encoder proposed in Section 3.2 that only uses a single column as the input of the encoder in the training process. This is Starmie without the use of contextual information from Section 3.3.

For efficiency experiments, we aim at exploring the benefits brought by different design choices in the Starmie framework. Thus we compare the performance of 4 methods: basic linear search (Linear), pruning based on estimated bounds (Pruning), search with an LSH index (LSH), and search with an HNSW index (HNSW).

5.1.5 Column encoder settings. We empirically choose the most suitable sampling method (Section 3.4) and augmentation operator (introduced in Section 3.3 and more details in Appendix A). For sampling methods, we find that Starmie achieves the best performance when pre-trained with the cell-level TF-IDF scoring function on the SANTOS Small and TUS Large benchmarks, and with a column-ordered sampling method, alphaHead, that sorts tokens in alphabetical order performs the best, on TUS Small. For augmentation operators, we find that the drop_col operator performs the best on SANTOS Small while drop_cell achieves the best performance on the two TUS benchmarks.

# 5.2 Results for Effectiveness

Table 3 reports the results of MAP@k and R@k on the three benchmarks for all methods. Note that the results for SANTOS are unavailable for TUS Large because SANTOS, which requires the labeled query table intent columns [23], have not been evaluated on this benchmark due to the absence of annotated intent columns. We run the experiments up to $_ { \mathrm { k } = 1 0 }$ on SANTOS Small following [23], and up to $\scriptstyle \mathbf { k } = 6 0$ on the TUS benchmarks, which is consistent with [40]. Note the recall cannot reach $1 0 0 \%$ when $k$ is smaller than the number of correct unionable tables from the labeled ground truth as reported in previous studies [23, 40]. For example, for SANTOS Small where $k$ is 10, the ground truth includes on average around 13 tables for different queries, so even the best technique can return (recall) at most $7 5 \%$ or k of 10 of these. Table 3 indicates the maximum recall as IDEAL for each setting.

We can observe that Starmie outperforms the baselines across all three benchmarks. On the SANTOS Small benchmark, Starmie

Table 3: MAP@k and R@k results on all benchmarks with ground truth, where $\mathbf { k } \mathbf { = } \mathbf { 1 0 }$ for SANTOS Small benchmark and $\mathbf { k } { = } 6 \mathbf { 0 }$ for the TUS benchmarks. The IDEAL $\mathbf { R } @ \mathbf { k }$ for SANTOS Small is 0.75, IDEAL $\mathbf { R } @ \mathbf { k }$ for TUS Small is 0.341, and IDEAL $\mathbf { R } @ \mathbf { k }$ for TUS Large is 0.277.   

<table><tr><td rowspan="2">Method</td><td colspan="2">SANTOS Small</td><td colspan="2">TUS Small</td><td colspan="2">TUS Large</td></tr><tr><td>MAP@k</td><td>R@k</td><td>MAP@k</td><td>R@k</td><td>MAP@k</td><td>R@k</td></tr><tr><td>SingleCol</td><td>0.891</td><td>0.588</td><td>0.954</td><td>0.255</td><td>0.902</td><td>0.208</td></tr><tr><td>SATO</td><td>0.878</td><td>0.594</td><td>0.966</td><td>0.271</td><td>0.930</td><td>0.223</td></tr><tr><td>Sherlock</td><td>0.782</td><td>0.493</td><td>0.984</td><td>0.265</td><td>0.744</td><td>0.119</td></tr><tr><td>SANTOS</td><td>0.930</td><td>0.690</td><td>0.885</td><td>0.230</td><td>-</td><td>-</td></tr><tr><td>D3L</td><td>0.523</td><td>0.422</td><td>0.794</td><td>0.215</td><td>0.484</td><td>0.124</td></tr><tr><td>Starmie</td><td>0.993</td><td>0.737</td><td>0.991</td><td>0.277</td><td>0.965</td><td>0.238</td></tr></table>

achieves the highest MAP@10 of $9 9 . 3 \%$ and highest $\mathrm { R @ 1 0 }$ of $7 3 . 7 \%$ (which is close to the IDEAL), outperforming SATO, Sherlock, SANTOS, $D ^ { 3 } \mathsf { L }$ baselines by large margins of $1 3 \%$ , $2 7 \%$ , $6 . 8 \%$ , and $9 0 \%$ respectively. Also, Starmie outperforms its SingleCol variation by $1 1 \%$ , showing that a multi-column approach is necessary. Similarly, on the TUS Small benchmark, Starmie outperforms the highestachieving baseline, Sherlock, by $0 . 7 \%$ and SingleCol variation by $4 \%$ in MAP@k. On the TUS Large benchmark, Starmie outperforms SATO by $4 \%$ and SingleCol by $7 \%$ in MAP@k. Thus, the Starmie approach, by capturing column context and leveraging contrastive learning in pre-training, is very effective in solving the table union search problem.

Figure 8 shows the P@k and R@k of Starmie and the baselines as k increases on all benchmarks. Throughout all values of k, Starmie outperforms all baselines for both $\mathrm { P } @ \mathbf { k }$ and R@k. In Figures 8(b), (d), and (f), Starmie is closest to IDEAL, with $\mathrm { R @ 1 0 }$ only $1 . 8 \%$ below IDEAL on SANTOS Small, $\mathrm { R } @ 6 0 \ 1 8 . 8 \%$ below IDEAL on TUS Small, and $\mathrm { R @ 6 0 ~ 1 4 . 1 \% }$ below IDEAL on TUS Large.

To better understand the influence of datasets on the performance of Starmie, we conducted an in-depth analysis to look at its performance for different settings of arity, cardinality, and percentage of numerical columns in query tables. We evenly split the query tables into five groups for each setting. We compare Starmie with alternative representation methods SATO, Sherlock, and SingleCol that also use deep learning to encode columns into high-dimensional vectors. As shown in Figure 9(a)/(c), Starmie consistently outperforms the baselines as the number of columns are varied and as the percentage of numeric columns varies. As the number of rows increases (Figure 9(b)), the results of Starmie remain consistently high while the performances of SATO, Sherlock, and SingleCol generally decrease. We believe this is due to our efforts table preprocessing techniques (Section 3.4). Meanwhile, the performance of SingleCol is much worse than Starmie under all settings, which illustrates the importance of contextual information in training the column encoders. The methods have similar trends on the TUS Small and TUS Large (Appendix B).

5.2.1 Micro Benchmarking Experiment. To evaluate the effect of self-supervision, specifically randomly drawing two tables to create negative examples on the effectiveness of Starmie, we create a microbenchmark consisting of eight data lakes drawn from TUS Small benchmark. In each data lake, there are 470 tables, of which $2 5 \%$ of tables have the same class as the query table while the remaining $7 5 \%$ of tables are evenly divided among 2-9 negative

![](images/bf29585f8891fcaf7613621c14662e126682d553ed542dcbc60e59a968c39683.jpg)

![](images/08cdee615690621aae0355a01715aa94ba8b6905d9cb3854f4b64552809351f3.jpg)  
(a) $P { \ @ { k } }$ on SANTOS Small

![](images/28a69288e13c56bfac9ccd6f03b54d271518cb6584eb02758cc0aeb542a02818.jpg)  
(b) $R ^ { \odot k }$ on SANTOS Small

![](images/ca3960b55ca0645c7801f8496d57f19ffb47ae1bc4e53703d7cfb69cff314d1e.jpg)  
(c) $P { \ @ { k } }$ on TUS Small

![](images/b4a57c5c9e8f157226db0a574930ebf17087ff34d534027fb2daa559f6e9c97c.jpg)  
(d) ??@?? on TUS Small   
(e) $P { \ @ { k } }$ on TUS Large   
(f) $R ^ { © k }$ on TUS Large   
Figure $\mathbf { 8 } \colon P @ k$ and $R @ k$ results on different benchmarks.

classes of tables. As shown in Table 4, for data lakes with fewer classes, it is less likely that two random tables are not unionable (counterexamples), thus resulting in lower MAP scores compared to those with more classes. Still, the MAP remains high, showing that effect of assuming random tables are not unionable is negligible – even in the extreme case of a data lake with only 3 classes of tables. We report MAP for K of 60, which is consistent with other experiments on TUS Small benchmark, and K of 120, which shows a clearer trend of MAP increasing and stabilizing as the number of classes increases. Note that this benchmark contains up to only 10 class labels, so the fluctuating trend is only among a limited set of classes. Real lakes would contain many magnitudes more classes.

Table 4: Effectiveness of Starmie on data lakes with different numbers of classes of tables.   

<table><tr><td></td><td colspan="8"># of Negative Classes</td></tr><tr><td></td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td></tr><tr><td>MAP@60</td><td>0.99</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>MAP@120</td><td>0.89</td><td>0.93</td><td>0.94</td><td>0.95</td><td>0.93</td><td>0.94</td><td>0.92</td><td>0.92</td></tr></table>

![](images/d73128270c13271f544a54310420995973ee3df1d3ce00075cb744408ce26f5a.jpg)  
(a) ??????@?? of different # Cols

![](images/4213b6b2cc9709ea69df6e83ee27846dd403763cc137313a3a034a9147a0417c.jpg)  
(b) ??????@?? of different # Rows

![](images/c5365db7c9057005d5c03222bda570f3403120dbf308b2b94a21ef92c1006bef.jpg)  
(c) ??????@?? of different % Num. Cols   
3-7 8-9 9-11 11-19 19-410.0 Figure 9: In-depth analysis of Starmie, SATO, Sherlock, and SingleCol as we vary the number of columns, number of rows, and Number of Colupercentage of numerical columns on the SANTOS Small benchmark.

# 5.3 Scalability

Table 5: Effectiveness of different design choices. The first four methods are for Starmie.   

<table><tr><td>Method</td><td>MAP@10</td><td>P@10</td><td>R@10</td><td>Query Time (s)</td></tr><tr><td>Linear</td><td>0.993</td><td>0.984</td><td>0.737</td><td>96</td></tr><tr><td>Pruning</td><td>0.993</td><td>0.984</td><td>0.737</td><td>61</td></tr><tr><td>LSH Index</td><td>0.932</td><td>0.780</td><td>0.580</td><td>12</td></tr><tr><td>HNSW Index</td><td>0.945</td><td>0.810</td><td>0.606</td><td>4</td></tr><tr><td>SATO</td><td>0.878</td><td>0.806</td><td>0.594</td><td>252</td></tr><tr><td>Sherlock</td><td>0.782</td><td>0.672</td><td>0.493</td><td>264</td></tr><tr><td>SingleCol</td><td>0.891</td><td>0.798</td><td>0.588</td><td>108</td></tr></table>

Impacts on effectiveness. Since some design choices might result in effectiveness loss, we report their results of three evaluation metrics on the SANTOS Small benchmark. As shown in Table 5, we compare Starmie with a basic linear scan with three other design choices (above the horizontal line), as well as baselines SATO, Sherlock, and SingleCol (full experiment results are shown in Appendix C). The main takeaway is that HSNW preserves the effectiveness as much if not better than the LSH index that is widely used in previous studies, while having tremendous speed improvement. This suggests HSNW is a very promising direction for providing real-time search over massive data lakes.

Preprocessing time. Since Starmie requires model pre-training and model inference, in addition to possibly indexing, we provide some insights of such overhead by comparing its preprocessing time with existing systems $D ^ { 3 } \mathsf { L }$ and SANTOS that are not based on pre-trained LMs. The preprocessing time of Starmie consists of the following parts: pre-training taking 3.1 hours, model inference taking $4 . 4 ~ \mathrm { m i n }$ , and indexing taking 10-30 sec. Meanwhile, $D ^ { 3 } \mathsf { L }$ takes 7.6 hours to create four indexes for each column feature and SANTOS takes 17 hours to create indexes using a knowledge base and the data lake. Thus, pre-training a language model in Starmie does not incur too much overhead compared to existing systems.

Time efficiency. We have observed that the employed design choices can speed up the online query time while sufficiently preserving the effectiveness scores. Next we evaluate the scalability of different design choices. In Figure 10(a), we first evaluate the four variations of Starmie on the SANTOS Large benchmark, as we increase the number of returned unionable tables k from 10 to 60. We then evaluate their query times as the data lake size grows to

its full size of ${ \sim } 1 1 \mathrm { K }$ tables / ${ \sim } 1 2 0 \mathrm { K }$ columns. We also experiment on the WDC benchmark, specifically when the data lake grows to 1M tables / 5M columns (Figure 10(b)) to show the trend of each method , and when the data lake grows to 50M tables / 250M columns (Figure 10(c)). For each method, if a data point’s query time does not finish within 24 hours, then we consider it as timeout and omit the result from the corresponding figures. To show the effectiveness of the LB/UB pruning mechanisms proposed in Section 4.3, we compare the number of verification steps needed per query with and without pruning. We found that on the SANTOS Small benchmark, the average number of verifications for Linear (without LB/UB pruning) is 550, while that of Pruning is 342 $3 8 \%$ reduction). This result shows that the pruning heuristic indeed helps significantly reduce the unnecessary verification and thus improves the overall performance.

Throughout all these experiments, we see that the design choice with the HNSW index leads to the best performance. On the SAN-TOS Large benchmark in Figure 10(a), the k-scalability experiment shows that Pruning is 2X faster than Linear, while LSH index is 20X faster than Linear. Meanwhile, HNSW index, which leads to an average query time of around 300 ms, is 220X faster than Linear and 11X faster than the popular LSH index. As the data lake grows to its full size, there is a steady increase in query time of Linear and Pruning; while that of LSH index and HNSW index remain stable, with the query time of HNSW index remaining around 400 ms. On the WDC benchmark in Figure 10(b), there is a similar trend as the data lake grows to 1M tables. On the full WDC benchmark in Figure 10(c), Linear and Pruning time out after 1M tables, while LSH index times out after having an average query time of 2,520 sec on 10M tables. Meanwhile, the query time for HNSW index stays consistent at around $6 0 ~ \mathrm { m s }$ as the data lake grows to its full size of 50M tables / 250M columns. The reason is that the hierarchical graph-based structure of HNSW allows it to locate to the nearest neighbors much faster than hash-based indexes [34]. Overall, the design choices explored in this paper, especially HNSW index, show a great improvement in the average query time, even when the data lake grows to an immense size of 50M tables. Meanwhile, to the best of our knowledge, the largest dataset that are evaluated by existing solutions of table union search is with only 5,000 tables / 1M columns [40], which has 250 times smaller number of columns. Memory overhead. Lastly, we examine the relative memory overhead of Starmie with different design choices (No index denotes

![](images/62404d1050703d78caa3bc5ecb0a48ea99edf3a8853619985aa448a01f3da7ee.jpg)  
Linear

![](images/7083d803f4487efda3e3793102c858e71394a7cd074a4a9dcda9827521f63b5a.jpg)  
Pruning

![](images/7e4155448792838fe149e34c4db1f35ed787bcdf36f63733aaa6e9c2945662e7.jpg)  
LSH Index

![](images/354dbd6a976b359503bcd8427259e5d0eeec53988afd231c89fbdec80194cde0.jpg)  
HNSW Index

![](images/3ee9491c2c4fb8b1a30157442c371e79796e127d386e358f0b7a2eaddec59ea5.jpg)

![](images/bf27f7f722832851bd8adb72ac56a59c7867e0343931f6a14067e34fec38c90d.jpg)  
(a) SANTOS Large benchmark

![](images/4e48ab76a27c622547c378e65122ea0f56c2c83cd5f095b46a47276cd5fe9f3e.jpg)  
30 (b) WDC (Sample) benchmark

![](images/5c1104ce444ceb7838f1935aca8ba2966007b0d75e3303d8bb142ea278045685.jpg)  
(c) WDC (Full) benchmark   
Figure 10: Scalability on the SANTOS Large benchmark, a sample of 1M WDC tables, and the full WDC benchmark

linear scan and pruning methods from Table 5). In Table 6, we report the memory usage of Starmie relative to the total data lake size (11 GB) of SANTOS Large. The results show that Starmie is not only scalable but also memory efficient: its variations take up around $3 - 7 \%$ space overhead. The memory saving is mainly due to the condensed vector column representations of Starmie which take up only $3 \%$ of the original data lake size.

Table 6: Relative Memory Overhead on the SANTOS Large benchmark for Starmie with the data lake of 11 GB.   

<table><tr><td>Method</td><td>Memory Usage</td><td>Space Overhead</td></tr><tr><td>No Index</td><td>359 MB</td><td>3.26%</td></tr><tr><td>LSH Index</td><td>733 MB</td><td>6.66%</td></tr><tr><td>HNSW Index</td><td>749 MB</td><td>6.81%</td></tr></table>

# 5.4 Data discovery for ML tasks

Next, we conduct a case study to show that Starmie can be applied to another application scenario of dataset discovery, i.e., retrieving relevant tables to improve the performance of downstream ML tasks. For this case study, we consider a subset of 78k WDC tables used in the evaluation of SATO [54], from which we collect all the 4,130 tables of at least 50 rows as the data lake tables. Among these tables, we find that 25 tables of at least 200 rows contain a numeric column called “Rating”. These 25 tables contain various types of ratings including those for sportsmen, TV shows, US congress members, etc. From these tables, we construct 25 regression tasks with the goal of training an ML model that predicts “Rating” as the target column. Since the ratings are from different domains, we normalize their values to the range [0, 1]. More details about the setting can be found in Appendix D.

For each task, we train a Gradient-Boosted Tree model [9] with all non-target columns as features. We featurize the textual columns using Sentence Transformers [42]. We split each dataset into training and test sets at a ratio of 4:1. Note that the original dataset may not contain informative features. Figure 11 shows such a dataset of US congress members.

To improve the model’s performance on these downstream tasks, we leverage Starmie to retrieve relevant tables from the data lake to join with the datasets (i.e., the query tables) to provide additional features. To showcase the effectiveness of Starmie, we use Starmie’s

10 contextualized column embeddings to retrieve from the data lake table that contains a column having the highest cosine similarity 0 with a non-target column of the query table. Finally, we augment the query table by performing a left-join with the retrieved table to ensure that the size of the augmented table stays unchanged. We also consider two popular similarity methods for this task, Jaccard and Overlap [13, 58], as baselines by replacing the cosine similarity scores with the corresponding similarity functions.

Data Lake Size (# tables / # columTable 7 summarizes the results of the 3 evaluated methods. While all 3 methods result in performance improvement (i.e., reduction of MSE), Starmie achieves significantly better overall improvements with a $1 4 . 7 5 \%$ MSE reduction, on 15/25 tasks improved, and by an average of $2 0 . 6 4 \%$ . By inspecting the retrieved tables, we find that Starmie indeed retrieves qualitatively better candidate tables. As Figure 11 shows, for the same US congress members table, Jaccard similarity retrieves an irrelevant table of dog competitions that also contains a similar “State” column, but the two tables are not semantically relevant. On the other hand, Starmie retrieves a table consisting of the amount of money raised from different interest groups, which is a potentially relevant feature to “Rating”. Indeed, by joining with the retrieved table by Starmie, the MSE of the model drops from 0.1598 to 0.1198 (by ${ > } 2 5 \%$ ).

Table 7: Performance gain of data discovery methods on 25 rating prediction tasks from WDC.   

<table><tr><td></td><td>NoJoin</td><td>Jaccard</td><td>Overlap</td><td>Starmie</td></tr><tr><td>Avg. MSE</td><td>0.0820</td><td>0.0753</td><td>0.0748</td><td>0.0699</td></tr><tr><td>Improvement</td><td>-</td><td>8.23%</td><td>8.82%</td><td>14.75%</td></tr><tr><td>#improved</td><td>-</td><td>13</td><td>12</td><td>15</td></tr><tr><td>avg. Improve</td><td>-</td><td>14.74%</td><td>14.05%</td><td>20.64%</td></tr></table>

# 5.5 Case study: Column clustering

Finally we show another application scenario of Starmie in dataset discovery: column clustering. Specifically, we apply Starmie as a column encoder to provide embeddings for clustering all the 119,360 columns from the 78k WDC tables used in the experiments of Sherlock, SATO, and others [21, 45, 54]. These columns are annotated with 78 ground truth semantic types such as population, city, name, etc. The goal of column clustering is to discover clusters of columns that are semantically relevant. The task of semantic

Query Table:

<table><tr><td>State</td><td>Office</td><td>District</td><td>Name</td><td>Party</td><td>Rating</td></tr><tr><td>AZ</td><td>U.S. House</td><td>8</td><td>Trent Franks</td><td>Republican</td><td>95.0</td></tr><tr><td>TX</td><td>U.S. House</td><td>3</td><td>Sam Johnson</td><td>Republican</td><td>95.0</td></tr><tr><td>OH</td><td>U.S. House</td><td>4</td><td>Jim Jordan</td><td>Republican</td><td>95.0</td></tr></table>

Retrieved by Jaccard:

<table><tr><td>SHOW</td><td>State</td><td>CITY</td><td>DATE</td><td>BREED</td><td>ENTRY</td><td>DOG PTS</td><td>BITCH PTS</td></tr><tr><td>Tucson Kennel Club</td><td>AZ</td><td>Tucson</td><td>03/28/99</td><td>Chinese Cresteds</td><td>NaN</td><td>NaN</td><td>NaN</td></tr><tr><td>Fort Bend Kc</td><td>TX</td><td>Richmond</td><td>11/17/96</td><td>Chinese Cresteds</td><td>NaN</td><td>NaN</td><td>NaN</td></tr><tr><td>Marion Oh Kc</td><td>OH</td><td>Marion</td><td>07/26/98</td><td>Chinese Cresteds</td><td>12.0</td><td>1.0</td><td>2.0</td></tr></table>

Ours:

<table><tr><td>Name</td><td>Party</td><td>State</td><td>$ From Interest Groups That Supported</td><td>$ From Interest Groups That Opposed</td><td>Vote</td></tr><tr><td>Trent Franks</td><td>R</td><td>AZ-2</td><td>$12,000</td><td>$0</td><td>Yes</td></tr><tr><td>Sam Johnson</td><td>R</td><td>TX-3</td><td>$4,000</td><td>$0</td><td>Yes</td></tr><tr><td>Jim Jordan</td><td>R</td><td>OH-4</td><td>$22,000</td><td>$0</td><td>Yes</td></tr></table>

Figure 11: Example tables retrieved by Jaccard vs. Starmie. By joining the query table with the DL table retrieved by Starmie, the MSE for predicting the “Rating” attribute drops from 0.1598 to 0.1195 (vs. 0.1544 when joining with the table retrieved by Jaccard).

type detection has traditionally been solved as a supervised multiclass classification problem which requires significant annotated training data [45]. Starmie provides an unsupervised solution. From the contextualized column embeddings, we can construct a similarity graph over all data lake columns as nodes. We can then add undirected edges between all pairs of columns having cosine similarities above a threshold $\theta$ (e.g., 0.6). Next the column clusters can be generated via any graph clustering algorithm. We choose the connected component algorithm for efficiency and simplicity.

With Starmie, the clustering algorithm generates 2,297 clusters with an average cluster size of 51.96. We measure the quality of the clusters by the purity score, which is the percentage of columns assigned with the same semantic type as the majority ground truth type of each cluster. The discovered clusters are generally of high quality as they achieve a purity score of 51.19 while using baselines such as Sherlock and SATO only achieves 30.5 or 37.36 purity scores when generating a similar number of clusters. A more detailed example of discovered clusters is shown in Appendix E.

# 6 RELATED WORK

# 6.1 Dataset Discovery

Dataset Discovery has been a hot topic in the data management community. Earlier studies [1, 5, 47] relied on keyword search over web tables to identify essential information. Octopus [4] and InfoGather [52] focused on the problem of schema complement, an important topic in exploring web tables. Aurum [16], S3D [18] and Tableminer $^ +$ [36, 56] utilized knowledge bases to identify relationship between datasets. SemProp [17] followed this route by leveraging ontologies and word embeddings, and Leva [57] solved a similar problem with graph neural networks. $D ^ { 4 }$ [41] addressed the problem of column clustering in data lake tables. Valentine [24] provided resources for evaluating column matching tasks. Domain-Net [27] studied the problem of disambiguation in data lakes.

Finding related tables from data lakes is an essential task in dataset discovery. There are two sub-tasks in this application, namely

finding joinable tables and table union search [44]. To support finding joinable tables, earlier studies utilized syntactic similarity metrics that are widely used in the applications of string similarity search and join [20, 28, 51]. LSH Ensemble used containment (overlap) [59] as the similarity metric and provided a high-dimensional similarity search based solution. Josie [58] employed overlap over tokens and developed an exact data-optimized solution. PEXESO [13] relied on cosine similarity over word embeddings and proposed indexing techniques to improve performance. The table union search problem has been well explored recently. Ling et al. [32] and Lehmberg et al. [25] illustrated the importance of finding unionable Web tables. Nargesian et al. [40] proposed the first definition and comprehensive solution for the table union search problem in data lakes. Bogatu et al. [2] proposed the $D ^ { 3 } \mathsf { L }$ system by dividing columns into different categories. The SANTOS [23] system uses a knowledge base along with binary relationships in the data lake to identify tables that share unionable columns and relationships, and it is the state-of-the-art approach in this field. To the best of our knowledge, our work is the first solution to utilize contrastive learning techniques in table union search.

# 6.2 Representation Learning for Tables

Recently many efforts use representation learning techniques to address problems related to tabular data. Sherlock [21] and Sato [54] used a supervised feature based approach to learn vector representations for tables and columns. TURL [11] proposed to use a pre-trained language model for web table related tasks and to come up with benchmark datasets for several tasks. And pre-trained language models have been widely applied to different table-related applications, including entity matching [6, 29, 30], column type detection [45, 48], and question answering [22, 53]. Our work follows this line of study and proposes the first solution that employs a pre-trained language model in a fully unsupervised way for the problem of table union search.

# 7 CONCLUSION AND FUTURE WORK

In this paper, we mainly focused on the problem of table union search, an essential application in dataset discovery from data lakes. We argued that it is crucial to utilize contextual information to determine whether two columns are unionable and proposed Starmie, an end-to-end framework based on contrastive representation learning as the solution. We also developed a multi-column table Transformer encoder that can capture the contextual information from a table so as to learn contextualized column embeddings. Experimental results on popular benchmark datasets demonstrated that Starmie significantly outperformed existing solutions for table union search.

Our results show the promise of self-supervised contrastive learning in improving the accuracy of table union search, as well as joinable table search, and column clustering – the latter areas we are exploring further. We believe the improved accuracy justifies the use of learning over previous heuristic approaches and the self-supervision will be important to data lakes where labeled training data is expensive to collect and generalize. Our results using the relatively new HNSW index are exciting and important in the development of real-time data lake search solutions.

