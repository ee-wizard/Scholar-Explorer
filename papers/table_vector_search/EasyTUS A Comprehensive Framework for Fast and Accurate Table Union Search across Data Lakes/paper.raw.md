# EasyTUS: A Comprehensive Framework for Fast and Accurate Table Union Search across Data Lakes

Tim Otto

Heterogeneous Information Systems

RPTU Kaiserslautern-Landau

Kaiserslautern, Germany

tim.otto@cs.rptu.de

Abstract—Data lakes enable easy maintenance of heterogeneous data in its native form. While this flexibility can accelerate data ingestion, it shifts the complexity of data preparation and query processing to data discovery tasks. One such task is Table Union Search (TUS), which identifies tables that can be unioned with a given input table. In this work, we present EasyTUS, a comprehensive framework that leverages Large Language Models (LLMs) to perform efficient and scalable Table Union Search across data lakes. EasyTUS implements the search pipeline as three modular steps: Table Serialization for consistent formatting and sampling, Table Representation that utilizes LLMs to generate embeddings, and Vector Search that leverages approximate nearest neighbor indexing for semantic matching. To enable reproducible and systematic evaluation, in this paper, we also introduce TUSBench, a novel standardized benchmarking environment within the EasyTUS framework. TUSBench supports unified comparisons across approaches and data lakes, promoting transparency and progress in the field. Our experiments using TUSBench show that EasyTUS consistently outperforms most of the state-of-the-art approaches, achieving improvements in average of up to $3 4 . 3 \%$ in Mean Average Precision (MAP), up to $7 9 . 2 \times$ speedup in data preparation, and up to $7 . 7 \times$ faster query processing performance. Furthermore, EasyTUS maintains strong performance even in metadata-absent settings, highlighting its robustness and adaptability across data lakes.

Index Terms—Data Lake, Table Union Search, Benchmark, Large Language Models

# I. INTRODUCTION

The exponential growth of heterogeneous data generation across the domains calls for innovative strategies for storing and leveraging information at scale. The data may appear in structured, semi-structured, or unstructured formats, and require minimal processing and quick gateway access [1]. Data lakes have emerged as a practical solution aligned with big data principles, enabling easy storage of vast volumes of raw, uncurated data, where metadata may be lacking or ambiguous. While this flexibility accelerates data ingestion by several folds, it shifts the burden of data discovery and preparation to query execution time. The primary challenge in querying from data lakes is the difficulty of post-hoc identifying relevant or valuable data [2]. Among the key techniques for querying structured data in such environments is Table Union Search (TUS) [3]. TUS aims to identify other tables within and across multiple data lakes that can be merged with a given query table via a union operation. This capability is valuable not only for

![](images/e940092e6dd2948d61a2842b1e3d236f69e8f7313cbe08dedfea7782c2ee0725.jpg)  
Fig. 1: Illustration of Table Union Search on a single data lake. The query table (left) and potential unionable tables from the data lake (right) are highlighted in green. Tables that are not unionable are marked in red.

data analysts seeking to enrich their information pool, but also for machine learning pipelines, where augmenting datasets with unionable tables can improve model generalization by increasing the number of data points.

A simple example of a TUS query executed within a single data lake is illustrated in Figure 1. In this scenario, the query table contains ZIP codes and city names from the German state of Hesse. The goal is to find other tables containing ZIP codes and city names from different regions of Germany that can be unioned with the query table. Tables such as Thuringia and Saxony are valid unionable results, as their schemas match and their tuples can be seamlessly appended to the query table. However, table Bavaria contains ZIP codes and street names instead of city names. Although it may be joinable based on the shared attribute ZIP code, it is not unionable because of differing column semantics. Performing a union operation in this case would result in an ill-defined outer union, where non-matching columns would be padded with null values, therefore, defeating the purpose of a clean union operation.

Earlier Table Union Search approaches [4], [5] have explored techniques such as metadata analysis, column-level set similarity, and ontology-based annotation. However, their accuracy is constrained by the limitations of data lakes: metadata is often sparse or inaccurate, exact value matches are rare, and domain-specific ontologies, particularly for numerical or non-English data, are frequently unavailable. In addition, querying

across multiple data lakes is not supported. Beyond the challenge of capturing semantic relationships between queries and targets (e.g. ensuring that a query about German cities ranks tables containing other German cities higher than those with cities from different countries), computationally expensive operations such as discovering functional dependencies become non-trivial as data volumes grow.

Recent approaches [6]–[8] rely on extensively fine-tuning the models and manually configuring the hyperparameters for embedding-based comparisons, aiming to improve semantic matching. However, these approaches also have challenges: smaller models often fail to generalize across domains or languages, and fine-tuning models on specific data lakes restricts applicability, making searches across multiple data lakes impossible. Given that data lakes can expand rapidly, generalization across data lakes and domains, and runtime efficiency, spanning data preparation (offline time) and query execution (online time), are critical. In such environments, it is impractical to re-train and fine-tune TUS approaches for each set of new data lakes.

This paper introduces EasyTUS, a novel Table Union Search framework that leverages Large Language Models (LLMs) without relying on metadata, ontologies, or finetuning. Trained on broad and diverse corpora, EasyTUS offers strong generalizability across domains and languages, eliminating the need for time-consuming, task-specific adaptation. Its framework is built on LLMs and operates directly on native table content, requiring only data lakes at offline time and a query table as input during online time, with no additional parameter configuration. This makes it especially effective for poorly curated data lakes. Importantly, removing fine-tuning steps unlocks the capability to query across multiple data lakes.

Alongside EasyTUS, this paper introduces TUSBench, a novel benchmarking environment for a robust comparison method with reproducible results, evaluating both accuracy and runtime efficiency of table union approaches across different data lakes. TUSBench is integrated within the EasyTUS framework and specifically designed to assess the effectiveness and efficiency of Table Union Search approaches and data lakes in a standardized and streamlined manner while underlining the importance of reproducible benchmarking results. It supports one-command execution, enforces consistent input/output formats, and facilitates result sharing and publication, ultimately promoting transparency, reproducibility, and accelerating progress in the field. Our contributions include:

1) We propose EasyTUS, a comprehensive framework that leverages LLMs for fast and accurate Table Union Search across multiple data lakes. EasyTUS operates independently of metadata, ontologies, or fine-tuning, relying solely on the underlying data in the data lakes.   
2) EasyTUS also includes TUSBench, a novel benchmarking environment that enables the reproducible comparison of the state-of-the-art and future Table Union Search approaches across various data lakes. TUSBench also supports varying the configurations of individual approaches, e.g. evaluating different models for any spe-

cific LLM-based TUS approach. It unifies input/output formats, streamlines evaluation, and facilitates the publication of reproducible results.

3) We provide a detailed analysis of the existing state-of-art Table Union Search techniques.   
4) In the experiments, we demonstrate that EasyTUS outperforms the state-of-the-art in most cases in terms of accuracy and runtime efficiency.

In the next section, we formally define the Table Union Search problem. Section III introduces the architecture of EasyTUS, followed by Section IV, which discusses related work and contrasts existing approaches with EasyTUS, motivating the need for a standardized benchmarking environment. Section V presents TUSBench, our proposed benchmarking environment, after which we report experimental results in Section VI and conclude.

The source code for EasyTUS and TUSBench is publicly available at: https://sci-git.cs.rptu.de/t otto18/easytus

# II. BACKGROUND

Formally, Table Union Search [3] can be defined as a top- $k$ retrieval problem.

Definition 1: Given a set of tables in a data lake $\tau$ and a query table $q$ , the Table Union Search $@ k$ task retrieves a subset $\mathcal { T } _ { q } \subseteq \mathcal { T }$ with $| \mathcal { T } _ { q } | = k$ , such that each union $q \cup t _ { i }$ , with $t _ { i } \in \mathcal { T } _ { q }$ , is well-defined and

$$
\operatorname {s c o r e} (q \cup t _ {i}) \geq \operatorname {s c o r e} (q \cup t _ {i + 1}),
$$

where $s c o r e ( \cdot )$ measures the union compatibility between a target table in the data lake and the query table.

Among the well-defined union operations between two tables $t _ { 1 }$ und $t _ { 2 }$ , with respective columns $t _ { 1 } = ( c _ { 1 } , \ldots , c _ { n } )$ and $t _ { 2 } = ( c _ { 1 } , \ldots , c _ { m } )$ , two special cases can be highlighted that frequently occur in real-world data lakes:

• One-to-One: $| t _ { 1 } | = | t _ { 2 } |$ . The union operation is a bijective function that maps each unique column in $t _ { 1 }$ to a unique column in $t _ { 2 }$ .   
• Onto: $| t _ { 1 } | > | t _ { 2 } |$ (resp. $| t _ { 2 } | > | t _ { 1 } | )$ . The union operation is an injective function that maps all columns of $t _ { 2 }$ (resp. $t _ { 1 , }$ ) to unique columns in $t _ { 1 }$ (resp. $t _ { 2 }$ ).

It is important to note that an onto mapping may semantically correspond to a one-to-one mapping. This can occur, for instance, when a single column in $t _ { 1 }$ is split into two separate columns in $t _ { 2 }$ , preserving the overall information content despite the structural mismatch. To complete the classification, an ill-defined union operation is a partial injective function that maps a subset of columns $t _ { 1 } ^ { \prime } \subset t _ { 1 }$ to a subset $t _ { 2 } ^ { \prime } \subset t _ { 2 }$ .

# III. EASYTUS ARCHITECTURE

EasyTUS is designed on the principle of architectural simplicity, incorporating only the minimal necessary functionality while improving both accuracy and time efficiency. The framework is structured into multiple steps, namely Table Serialization, Table Representation, Storage, and Search, as illustrated in Figure 2. A distinction is made between offline and online time. Offline time refers to executing all EasyTUS

![](images/a7968f503e203feed5a79a19a75fc11d80d7ce9fa01fcf2fad0357642e750a5f.jpg)  
Fig. 2: Overview of the EasyTUS architecture for Table Union Search across multiple data lakes. The core steps, Table Serialization and Table Representation, are executed in both the offline and online phases. In the offline phase, embedding vectors for all tables from individual data lakes are generated and persisted, while in the online phase, an approximate nearest neighbor search is performed using the query table vector and the persisted embeddings.

pipeline steps, with the generated embedding vectors for tables across data lakes persisted in the vector database. Online time corresponds to executing only the first two pipeline steps followed by the search, applied when a query table is received. In this case, embedding vectors for the query table are generated identically but are never stored.

# A. Table Serialization

Using Large Language Models for Table Union Search requires representing tables from data lakes as serialized inputs, i.e., as serialized tokens of table values, to the LLMs. In general, two strategies can be distinguished: zero-shot and few-shot [9]. In the few-shot strategy, metadata such as column names or table context is included alongside table values, while zero-shot relies solely on the table values itself. Since metadata in data lakes can be ambiguous, missing, or unreliable, EasyTUS employs the zero-shot approach to ensure robustness and general applicability. Regardless of the specific LLM used, there is typically a maximum input token limit, which gives rise to two fundamental challenges: determining how many table values can be included in a single serialization without exceeding the model’s token limit, and deciding which parts of the table (i.e., which columns) should be selected to ensure an informative yet complete representation. Addressing these challenges requires effective strategies.

1) Batching Strategy: In general, each LLM has a maximum token limit $N$ that constrains the length of a single table serialization. Additionally, usually their respective APIs support batch processing, where multiple serializations can be ingested in parallel, subject to a global batch limit $M$ . In cases where batch processing is not supported, we set $N \ = \ M$ without loss of generality. The core constraint for batching is defined as:

$$
\sum_ {0 \dots b, 0 \dots i} \operatorname {t o k e n s} [ b ] [ i ] \leq b \cdot M \wedge \forall_ {0 \dots b} \sum_ {0 \dots i} \operatorname {t o k e n s} [ b ] [ i ] \leq N \tag {1}
$$

flwhere tokens $[ b ] [ i ]$ denotes the number of tokens in the $i \cdot$ -th serialization of batch $b$ . The goal is to maximize the number

of serializations in each batch to reduce the number of API calls and overall execution time. EasyTUS applies a greedy batching algorithm, as shown in Algorithm 1, that selects just enough rows per table to fit within the table serialization limit $N$ , and then packs as many such serializations into a batch until the batch-wide maximum token limit $M$ is reached.

Algorithm 1 EasyTUS Greedy Batching Algorithm   
Require: Table set $\mathcal{T}$ , serialization token limit $N$ , batch token limit $M$ procedure Batchinging(T, N, M)   
Batches $\leftarrow \left[\right]$ , Batch $\leftarrow \left[\right]$ , Tokens $\leftarrow 0$ for all table $T\in \mathcal{T}$ do   
Rows $\leftarrow$ ROWSELECTION(T, x) $S\gets$ SERIALIZE Rows) $D\gets$ MIN(0, $|S| - N$ $S\gets$ Remove last $D$ tokens from $S$ if Tokens $+|S| > M$ then   
Append Batch to Batches   
Batch $\leftarrow \left[\right]$ , Tokens $\leftarrow 0$ end if   
Append $S$ to Batch   
Tokens $\leftarrow$ Tokens $+|S|$ end for   
Append Batch to Batches   
end procedure

2) Selection Strategy: The maximum token limit $N$ also poses the challenge of determining which columns to prioritize for the serialization in order to make the most effective use of the available token limit space. Prior work [10] on finding most valuable columns commonly applies TF-IDF scoring across tables during offline time to identify the most informative columns. However, such global relevance scoring is computationally expensive and not easily maintainable in dynamic or frequently expanded data lakes.

Therefore, EasyTUS employs a pairwise-independent, lightweight row and column selection strategy. From each

table, $x$ random rows are sampled, ignoring any metadata such as table or column names. The use of randomly selected rows reduces the risk of using approach-tailored data lakes that fail to reflect real-world scenarios, thereby improving generalizability. Within each row, column values are converted to strings and concatenated using a separator token [SEP] (e.g., a comma). Rows themselves are separated by a [NEW] token (e.g., $\backslash \mathtt { n } \mathtt { \Gamma } $ . If the resulting prompt exceeds the token limit $N$ , tokens are removed from the end until the constraint is satisfied. If an entire table contains fewer data than the maximum token limit, every row from the table is included exactly once. In the case of a single row exceeding the token limit, its values are first shuffled before the algorithm proceeds with removing tokens from the end. The final serialized format looks like:

$$
\left\{\text {I D}: \quad^ {\prime \prime} v _ {i, 0} [ \text {S E P} ] v _ {i, 1} [ \text {S E P} ] \dots [ \text {N E W} ] v _ {i + 1, 0} \dots^ {\prime \prime} \right\}
$$

where ID represents the composite key of data lake and table name, and $v _ { i , j }$ denotes the value of the $j$ -th column in the $i$ -th randomly selected row.

By reducing the number of rows, EasyTUS preserves the maximal number of columns within each serialization. This trade-off favors breadth over depth, increasing the likelihood that relevant column semantics are captured in the embedding despite tight token constraints. Importantly, this method generates a semantic representation for the entire table rather than for individual columns. This reduces the risk of overestimating unionability due to superficial similarities in isolated columns, such as primary keys with common integer patterns, by considering the table’s content as a whole. It also reduces computational costs by requiring only one representation per table, instead of computing and matching separate representations for each column. Furthermore, the LLM can capture structural relationships between columns based on their cooccurrence within the same serialization, thereby enhancing the overall semantic representation.

# B. Table Representation

To generate vector representations of the table serializations across all batches, EasyTUS employs LLM-based embedders where each serialization in the batch is transformed into one embedding, resulting in a batch of corresponding vectors. This enables querying based on semantic similarity, thus allowing for more flexible, nuanced, and efficient comparison across diverse data lakes. EasyTUS is compatible with any embedding model: for example, models by OpenAI [11] or published on HuggingFace [12], demonstrating the flexibility and balance between cloud-based, high-performance models and on-premise solutions that prioritize data privacy. Within the EasyTUS framework, users only need to specify the model name and its token limit for the batching algorithm at offline time. It is essential to select a sufficiently large model to ensure robust cross-domain generalization. EasyTUS makes no use of fine-tuning or post-processing of embeddings, prioritizing maximum throughput, scalability, and minimal configuration.

# C. Storage

The embedding vectors are stored in a vector database. To support efficient retrieval, two types of indexes are created: a Hierarchical Navigable Small World (HNSW) [13] index for fast approximate nearest neighbor (ANN) search over the embedding space, and an inverted index that maps embedding vectors back to their corresponding tables and data lakes.

# D. Table Union Search

To ensure consistent and effective table retrieval across data lakes, the same Table Serialization and Representation methods that are applied to the data lake tables during offline time must also be applied to query tables during online time. Upon receiving a query table, the identical pipeline steps are executed, but the resulting embedding vector is never persisted. Instead, the online phase additionally performs an approximate nearest neighbor search using the query vector against the embedding vectors in the vector database. EasyTUS supports any distance function, with cosine similarity as the default metric for similarity search. Alternative metrics, such as dot product or Euclidean distance, can also be used interchangeably.

Given a query table $q$ , the system returns a ranked list of $k$ embedding vectors with the highest similarity scores to $q$ , enabling efficient retrieval of unionable tables across data lakes. Using the inverted index, these embedding vectors are then mapped back to the corresponding tables, producing a uniformly scored, ranked list of unionable tables spanning multiple data lakes.

# E. Incremental Table Additions

Since data lakes are typically append-only and not actively maintained, challenges related to updates and deletions are largely avoided, i.e. new data gets added, but existing entries remain unchanged. As a result, the key operational concern is efficient table insertions, which must be fast and independent of the existing tables of the data lakes, thus making the total end-to-end execution time, comprising both offline and online time, a crucial metric. It also implies that all algorithms used for data selection or sampling must be stateless, meaning they cannot rely on tables that have already been processed. EasyTUS addresses this by using uniform embedding vectors: when a new table is added, its embedding can be computed independently and inserted directly into the index, without requiring re-processing of tables across the existing data lakes. Due to the use of random sampling, when rows in a table are added, updated, or deleted, it still have little to no effect on the generated embedding. Our sampling strategy offsets the table updates effectively. Only extensive modifications would noticeably affect the sampling; an assumption contrary to the intended use of data lakes.

# IV. RELATED WORK

Existing Table Union Search approaches can be classified along several dimensions. An overview of how each method aligns with these criteria is summarized in Table I.

TABLE I: Classification of Table Union Search Approaches   

<table><tr><td></td><td>EasyTUS</td><td>D3L [4]</td><td>Santos [5]</td><td>Starmie [6]</td><td>AutoTUS [7]</td><td>TabSketchFM [8]</td></tr><tr><td>Search Composition</td><td>Individual</td><td>Composite (5 Indexes)</td><td>Composite (KB + synth. KB)</td><td>Individual</td><td>Individual</td><td>Pipelined</td></tr><tr><td>Representation Method</td><td>Transformer-based</td><td>Feature-based</td><td>Graph-based</td><td>Transformer-based</td><td>Transformer-based</td><td>Transformer-based</td></tr><tr><td>Learning Strategy</td><td>No Learning</td><td>-</td><td>-</td><td>Self-Supervised</td><td>Self-Supervised</td><td>(Self-) + Supervised</td></tr><tr><td>Cross-Comparitibility</td><td>Yes</td><td>Partial</td><td>Partial</td><td>No</td><td>No</td><td>No</td></tr><tr><td>Representation Level</td><td>Table</td><td>Column</td><td>(Inter-) + Column</td><td>Column</td><td>Column</td><td>Table</td></tr><tr><td>Representation</td><td>Embedding</td><td>Feature</td><td>Annotation</td><td>Embedding</td><td>Embedding</td><td>Embedding</td></tr><tr><td>Data Type Dependency</td><td>No</td><td>Yes</td><td>Yes</td><td>Yes</td><td>No</td><td>Yes</td></tr><tr><td>Knowledge Integration</td><td>No</td><td>No</td><td>External KB</td><td>No</td><td>No</td><td>No</td></tr><tr><td>Indexing Strategy</td><td>HNSW</td><td>LSH-Forest</td><td>Inverted Index</td><td>HNSW</td><td>-</td><td>-</td></tr></table>

Search Composition. Approaches can either use a single technique (individual) or combine multiple techniques. For instance, D3L [4] uses a composite strategy with five different indexes, improving robustness by aggregating information from multiple perspectives. In contrast, TabSketchFM [8] applies a pipelined method where the output of one step forms the input of the next step.

Representation Method. This refers to how table content is encoded for comparison. While many approaches, including EasyTUS and Starmie [6], use transformer-based embeddings for semantic understanding, Santos [5] stands out by using graph-based representations, relying on structural and relational information derived from knowledge bases.

Learning Strategy. Learning strategies vary in supervision level. Starmie, for example, applies a self-supervised approach, learning from internal table relationships without labeled data. In contrast, TabSketchFM includes both self-supervised and supervised training to boost in-domain accuracy.

Cross-Compatibility. Some systems are designed to operate across multiple data lakes only in part, meaning individual components may function across lakes, but the complete system does not work in composition without reconfiguration. EasyTUS is the only approach supporting full cross-data lake compatibility, using a universal scoring function without requiring fine-tuning or knowledge base generation.

Representation Level. Table content can be represented at different granularities. Santos uses a mix of inter-column and column-level representations to capture both individual column semantics and their relational structure, enhancing expressive power for complex tables.

Representation. Information may be encoded as learned embeddings, extracted features, or external annotations.

Data Type Dependency. Some approaches factor in column data types to improve matching precision. D3L is typedependent, leveraging type-aware indexes to better align numeric and textual columns. In contrast, EasyTUS remains type-agnostic, treating all columns uniformly.

Knowledge Integration. Only Santos includes external domain knowledge, enriching its representation with semantic annotations from sources like YAGO [14].

Indexing Strategy. Efficient search at scale requires in-

dexing. EasyTUS and Starmie both use HNSW, a scalable vector-based method, while Santos relies on an inverted index, suitable for textual matches derived from knowledge bases.

EasyTUS stands out by being the only training-free, fully cross-compatible method with minimal setup and strong scalability. Its transformer-based embeddings, table-level abstraction, and HNSW indexing enable fast, accurate search even across diverse and growing data lakes. By contrast, each competing approach outlines their respective unique strengths. D3L emphasizes robustness via composite indexing and typeawareness. Santos enriches semantic matching with external knowledge. Starmie and AutoTUS [7] provide a lightweight self-supervised alternative, and TabSketchFM improves expressiveness with task-specific fine-tuning.

These inherent differences across the state-of-the-art highlight the importance of a standardized benchmarking environment, which can enable consistent, reproducible evaluation across such diverse approaches.

# V. NOVEL BENCHMARKING ENVIRONMENT

The diversity of data lakes and Table Union Search approaches leads to inconsistent methods for setting hyperparameters and measuring accuracy, typically through precision and recall. To ensure verifiable and reproducible results, there is a pressing need for a standardized framework that enables consistent evaluation of both approaches and data lakes, using common metrics for effectiveness and efficiency. Such standardization is essential for reproducible research and to foster progress in the scientific community. While a few benchmarking tools have been proposed [15], [16], they often suffer from issues such as non-executability, faulty code, nonstandardization, and irreproducible results.

TUSBench addresses these challenges by structuring the execution into modular components and intermediate interfaces, ensuring a consistent and extensible workflow. The environment is structured into a pipeline consisting of three main stages: Data Corpus Preparation, Framework Execution, and Empirical Validation, which are connected by two welldefined intermediate interfaces (Intermediates I and Intermediates II). These standardized interfaces ensure consistency,

![](images/571d1c60e5f703d752aa23fd807eb7f8ed54c451b8b021f41da1d7eb545ecb94.jpg)  
Fig. 3: Overview of the TUSBench environment setup, illustrated with two example data lakes, namely TUS-SANTOS and ECB Union, and two example approaches, Starmie and EasyTUS. Blue boxes represent provided components, orange boxes indicate generated intermediates, green boxes denote customizable metrics, and red boxes highlight shareable results.

extensibility, and reproducibly testing across all stages. An example TUSBench environment is shown in Figure 3.

1) Data Corpus Preparation: TUSBench is compatible with any data lake, provided the data can be transformed into a tabular representation. This process is enforced through two sequential steps, executed once per data lake: download and prepare. The download step ensures that the executable code for respective approaches and the tables from the data lakes are stored separately, for example, in a GitHub repository and a storage server respectively. Since data lakes can occupy substantial storage space, this separation enables a dynamic and space-efficient environment where the data stays within the data lakes. It also allows new approaches to be shared without the need to transfer the data. The prepare step translates known file formats, structures, and locations into the standardized format required by the Intermediates I interface, enabling seamless integration with any supported approach. If these steps have already been completed, they are skipped by default unless explicitly forced to re-run.

2) Intermediates I: The Intermediates $I$ interface defines a structure that enables consistent handoff between the Data Corpus Preparation and Framework Execution stage, ensuring interoperability across varying interpretations of data lake structures and approach inputs.

After completing the Data Corpus Preparation stage, the tables of each data lake are consolidated into a representational directory structure:

• /query – Contains query tables   
• /datalake – Contains (non-)unionable tables   
• groundtruth.pickle – Maps query tables to corresponding unionable tables

This standardized layout enables any approach to interact with any data lake avoiding the need for custom adaptations.

3) Framework Execution: Each TUS approach is executed via a dedicated module that serves as its entry point and defines a function that returns a tuple consisting of three components:

• a dictionary of mapping steps to their corresponding execution and undo logic,   
• a dictionary for tracking the execution status of each step,   
• a file path for persistently storing the status information

The mapping steps are flexible and can be linked to any custom implementation, including functions, shell scripts, or other executable routines tailored to individual approaches. These steps must be manually defined by the admin of the data lake or the approach.

At benchmark runtime, the defined steps are executed sequentially, typically including embedding, indexing, and querying. If a specific step fails or needs to be rerun, TUSBench first invokes the corresponding undo function to revert its effects before re-execution, therefore, ensuring its consistent state. Upon successful completion of a step, its status is marked in the status dictionary, which is persisted to disk. During subsequent runs, this status file is used to identify completed steps, which are skipped unless explicitly re-executed, thus minimizing redundant computation. This mechanism also allows shared tasks, such as downloading model weights from HuggingFace or auxiliary resources, to persist across multiple executions of the same approach. Since input data adheres to a standardized format defined by the Intermediates I interface, all approaches can be applied to any data lake without modifications on the approach side.

For each query table, the approach performs a top- $k$ search over the data lake and stores the resulting ranked list of predicted unionable tables for evaluation.

4) Intermediates II: The Intermediates II interface standardizes the structure of prediction outputs so that all approaches can be evaluated in a consistent manner. Top- $k$ ranked lists are saved as a dictionary in a prediction.pickle file, mapping each query table to a ranked list of predicted unionable tables.   
5) Empirical Validation: In the final evaluation phase, TUSBench computes performance metrics by comparing the top- $k$ results from the Intermediates $\boldsymbol { { I I } }$ interface with the ground truth from the Intermediates $I$ interface. This is achieved using TUSBench’s functionality to evaluate performance across varying values of $k$ . Core metrics include, but are not limited to, Mean Average Precision and Average Recall, evaluated over a predefined list of $k$ values. These results are recorded in a resulting experiments.csv file. In addition, the execution times for each processing step defined in the

TABLE II: Specifications of Benchmark Data Lakes used in Evaluation.   

<table><tr><td>Data Lake</td><td>#Queries</td><td>#Tables</td><td>Avg. Hits</td><td>Avg. Cols</td><td>Avg. Rows</td><td>Size (MB)</td><td>String (%)</td><td>Float (%)</td><td>Int (%)</td><td>Bool (%)</td></tr><tr><td>tus_santos [17]</td><td>40</td><td>2040</td><td>13.69</td><td>10.16</td><td>4728.73</td><td>1500</td><td>85.46</td><td>6.45</td><td>8.09</td><td>0</td></tr><tr><td>wiki_union [17]</td><td>7158</td><td>33594</td><td>11.78</td><td>2.59</td><td>51.13</td><td>142</td><td>60.06</td><td>25.22</td><td>14.72</td><td>0</td></tr><tr><td>ecb_union [17]</td><td>508</td><td>3718</td><td>51.81</td><td>35.95</td><td>307.38</td><td>435</td><td>49.17</td><td>36.67</td><td>14.16</td><td>0</td></tr><tr><td>ugen_v1 [18]</td><td>49</td><td>960</td><td>9.73</td><td>10.37</td><td>7.66</td><td>3.8</td><td>92.72</td><td>3.91</td><td>3.37</td><td>0</td></tr><tr><td>ugen_v2 [18]</td><td>50</td><td>1000</td><td>10</td><td>13.36</td><td>18.74</td><td>8.1</td><td>82.83</td><td>5.50</td><td>11.67</td><td>0</td></tr><tr><td>webstable [15]</td><td>5476</td><td>34796</td><td>16.33</td><td>12.61</td><td>39.41</td><td>198</td><td>58.61</td><td>13.27</td><td>28.12</td><td>0</td></tr><tr><td>opendata [15]</td><td>3062</td><td>7925</td><td>18.95</td><td>19.07</td><td>73948.43</td><td>107000</td><td>52.40</td><td>24.30</td><td>23.04</td><td>0.26</td></tr></table>

Framework Execution stage is tracked. By structuring the approach into finer-grained steps, users can obtain a more detailed breakdown of execution performance. All output files can optionally be saved to a shared repository.

# VI. EXPERIMENTS

We evaluate EasyTUS against state-of-the-art, publicly available baseline approaches, therefore excluding AutoTUS and TabSketchFM, using established benchmark data lakes within the TUSBench environment. To assess both time efficiency and accuracy, we measure execution time along with Mean Average Precision (MAP) and Average Recall (AR), two standard metrics commonly used in prior work to assess the ranking quality of top- $k$ Table Union Search results across multiple queries. To further highlight EasyTUS’s generalization capabilities and avoid any risk of data contamination, the evaluation includes a synthetic, previously unseen data lake.

# A. Setup

The data lakes in these experiments are sourced from previous studies and are listed in Table II. They serve as the basis for benchmarking the current state-of-the-art. Both webtable and opendata are light-weight versions of the originally published data lakes [15], excluding the noise tables that are not unionable with any given query table.

To further assess robustness, each data lake is accompanied by an adapted variant in which all metadata has been removed to particularly demonstrate that EasyTUS not only generalizes across structured data lakes but also performs well in minimalcontext scenarios, with no metadata. In these adapted variants, table names are replaced with unique hexadecimal identifiers, and column names are anonymized as $c o l _ { i }$ , where $0 \leq i < n$ and $n$ is the number of columns. Mean Average Precision (MAP) is computed uniformly using Equation 2, with a key modification: instead of a fixed $k$ , we use $m i n ( k , k ^ { * } )$ , where $k ^ { * }$ is the number of unionable tables in the ground truth for a given query. This adjustment ensures that setting $k > k ^ { * }$ does not unfairly penalize approaches and eliminates the need to tailor $k$ per query. For $k$ , values of $2 ^ { i }$ with $0 \leq i \leq 6$ are selected, which on average cover the full range of target tables per query (Avg. Hits) reported in Table II.

$$
M A P @ k = \frac {1}{| Q |} \cdot \sum_ {q \in Q} A P _ {q} @ k \tag {2}
$$

where

$$
A P _ {q} @ k = \frac {1}{\operatorname* {m i n} \left(k , k ^ {*}\right)} \cdot \sum_ {i = 1} ^ {k} P _ {q, i} \cdot \operatorname {r e l} _ {q, i} \tag {3}
$$

![](images/a89fe3fbbd6e4a8d158145907b5ff61c5e7da477a69e18489d07508e10b351ae.jpg)  
(a) Different embedding models

![](images/24fb0b5f95b7b34def998d1c60908e31b1de35e097552b8117b9435374ed96b1.jpg)  
(b) Different table-row sizes   
Fig. 4: Mean Average Precision $( \mathbf { M A P } @ k )$ on the tus santos data lake benchmark. Values of k are plotted on the $\mathbf { X }$ -axis. Similar trends are observed for other data lakes as well.

with $Q$ denoting the set of queries, $P _ { q , i }$ the precision at position $i$ for query $q \in Q$ , $k ^ { * }$ the number of unionable tables in the ground truth for query $q$ , and $r e l _ { q , i } = 1$ if the table at rank $i$ is relevant, and 0 otherwise.

Additionally, we evaluate the standard metric Recall using the Average Recall (AR), as defined in Equation 4.

$$
A R @ k = \frac {1}{| Q |} \cdot \sum_ {q \in Q} \sum_ {i = 1} ^ {k} r e l _ {q, i} \tag {4}
$$

All experiments are conducted on a high-performance server equipped with 996GB RAM, an AMD EPYC 7662 64-core processor, Ubuntu $2 2 . 0 4 . 5 ~ \mathrm { L T S }$ , and a single NVIDIA A100 80GB GPU. Each approach is run using its default configuration, as specified in the original publications.

# B. LLM Selection

We maximize usability by minimizing manual parameter tuning while ensuring robust generalization. This also applies to selecting models for generating the embeddings. Since EasyTUS can support a wide range of LLM-based embedders, exhaustively evaluating all possible models is impractical. However, factors such as data privacy, resource availability, and model size play a crucial role in selecting the appropriate model. Therefore, we compare OpenAI’s cloud-based models [19], which offer large-scale capabilities, with JinaAI’s onpremise models [20], [21], which prioritize data privacy. Figure 4a shows $\mathbf { M A P @ } k$ scores for four embedding models. Due to space constraints, the tus santos data lake is shown as representative results, however, similar results were observed across other data lakes. Our findings show that OpenAI models yield slightly higher precision, though the difference is modest: JinaAI’s best model lags OpenAI’s lowest by only

one percentage point and differences among OpenAI’s models are negligible. These findings suggest JinaAI is a practical alternative when data privacy outweighs slightly higher accuracy gains. For this reason, in the following experiments, we use both openai-text-embedding-3-large, hereafter referred to as ET-O, and jina-embeddings-v3, referred to as ET-J. This dual setup offers users flexibility in balancing precision and privacy, without compromising overall performance.

# C. Row Count

The EasyTUS batching algorithm samples a set of random rows per table. To optimally balance diversity of data points against the risk of exceeding token limits, experiments were conducted on the tus santos data lake, again selected as a representative case, with similar results observed for the other data lakes. We evaluated system performance across row counts of $2 ^ { i }$ with $0 \leq i \leq 7$ , as shown in Figure 4b, using $\mathbf { M A P @ } k$ as the primary metric. Embeddings were generated using the openai-text-embedding-3-large model as a representative example, with $n$ randomly selected rows (without repetition) per table; due to space constraints, only results for this model are shown, though all other tested models yielded comparable performance. The overall results indicate that row counts of 1–8 yield significantly lower $\mathbf { M A P @ } k$ scores, while performance stabilizes above 0.9 for row counts of 16 and higher. However, precision is not the sole performance criterion. Embedding latency and the frequency of table truncation due to token limitations are equally critical, as they influence the scalability and applicability of the approach to large-scale or high-complexity tabular data. For 128 rows, embedding initialization took 396 seconds, with $2 2 . 6 \%$ of total 2,040 tables exceeding the 8,192-token context limit, thus, requiring row reduction. With 64 rows, the time dropped to 83 seconds (175 reductions), and at 32 rows, to just 66 seconds (52 reductions). These findings reveal that 16 to 32 rows strike the optimal balance: maintaining high accuracy (comparable to 64 and 128), while reducing computation time and minimizing token-related table reductions. Comparable results were obtained across the other data lakes and LLMs. As a result, we adopt 32 rows as the default.

# D. Accuracy

We evaluate the accuracy of EasyTUS against the leading baseline Table Union Search systems Starmie, Santos, and D3L across all given data lakes. All experiments use a consistent configuration of 32 rows and the cosine similarity as distance metric. Figure 5 presents the results across data lakes (a–g) with metadata included. To evaluate robustness under limited metadata conditions, each data lake was also prepared in a version with metadata removed, as shown in Figure 6. Due to space constraints, only a subset of those results is shown here for illustration. For each benchmark data lake, both $\mathbf { M A P @ } k$ and $\mathbf { A } \mathbf { R } \ @ k$ are evaluated.

When metadata is available, the majority of approaches initially achieve high MAP scores. In many cases, across $k$ values and data lakes, the EasyTUS variants ET-J and

![](images/ffb285bb1e0466638809fe60b310ba5d72842c5936dceadf4b4c6fe73f8927cb.jpg)  
Fig. 5: Comparison of EasyTUS (ET-J and ET-O) against the state-of-the-art in terms of Mean Average Precision $( \mathbf { M A P } @ k )$ and Average Recall $( \mathbf { A } \mathbf { R } @ k )$ across selected benchmark data lakes with metadata. Values of $k$ are plotted on the x-axis.

ET-O achieve the highest MAP scores, generally performing on par with each other. A notable exception appears in the wiki union data lake (sub-figure 1b), where ET-O performs better than ET-J. For Average Recall, ET-J and ET-O typically identify relevant tables more quickly than competing approaches. The improvement is computed as the average over all values of $k$ for each approach within a data lake, and then averaged across all data lakes. ET-O achieves a $3 4 . 3 3 \%$ improvement and ET-J a $2 8 . 8 9 \%$ improvement over

TABLE III: Execution Times in Seconds for ET-O, ET-J, D3L, Santos, and Starmie across Data Lakes.   

<table><tr><td rowspan="2">Data Lake</td><td colspan="5">Offline Time (s)</td><td colspan="5">Online Time (s)</td></tr><tr><td>ET-O</td><td>ET-J</td><td>D3L</td><td>Santos</td><td>Starmie</td><td>ET-O</td><td>ET-J</td><td>D3L</td><td>Santos</td><td>Starmie</td></tr><tr><td>tus_santos</td><td>106.87</td><td>411.2</td><td>3726.82</td><td>39294.51</td><td>10484.51</td><td>4345.68</td><td>7.04</td><td>683.95</td><td>1220.81</td><td>3.57</td></tr><tr><td>wiki_union</td><td>469.99</td><td>1761.99</td><td>1049.48</td><td>21033.82</td><td>4587.16</td><td>693.77</td><td>106.3</td><td>410799.26</td><td>446.99</td><td>16.49</td></tr><tr><td>ecb_union</td><td>156.2</td><td>1738.23</td><td>1177.43</td><td>15419.13</td><td>4467.51</td><td>7155.68</td><td>13.51</td><td>248823.55</td><td>30477.76</td><td>479.52</td></tr><tr><td>ugen_v1</td><td>29.55</td><td>104.24</td><td>186.04</td><td>3764.89</td><td>330.02</td><td>2177.16</td><td>13.65</td><td>611.48</td><td>86.28</td><td>5.4</td></tr><tr><td>ugen_v2</td><td>33.43</td><td>164.64</td><td>190.79</td><td>3629.16</td><td>287.45</td><td>1870.99</td><td>6.56</td><td>161.44</td><td>72.85</td><td>3,02</td></tr><tr><td>opendata</td><td>6279.12</td><td>4138.58</td><td>603670.9</td><td>-</td><td>573017.87</td><td>78.08</td><td>46.62</td><td>99238.98</td><td>-</td><td>1723,64</td></tr><tr><td>webtable</td><td>472.03</td><td>1305.76</td><td>1783.4</td><td>33432.14</td><td>4875.06</td><td>34447.39</td><td>103.04</td><td>300455.58</td><td>28564.22</td><td>60.11</td></tr><tr><td>Average</td><td>1078.17</td><td>1374.95</td><td>87397.84</td><td>19428.94</td><td>85435.65</td><td>7252.68</td><td>42.39</td><td>182364.27</td><td>10144.82</td><td>327.39</td></tr></table>

the next-highest precision, obtained by D3L. In the ecb union data lake (Figure 5c), EasyTUS is outperformed by all other approaches for $k \lesssim 1 6$ , with D3L achieving the highest MAP and AR before and beyond this point. This behavior is likely due to the composition of the data lake, which, as shown in Table II, contains predominantly numerical values that are difficult to interpret without fine-tuning. Even with fine-tuning, composite search approaches such as D3L, which maintain a dedicated index for numerical values, have a clear advantage. In summary, EasyTUS outperforms all baseline systems across most benchmark data lakes, with the exception of cases where numerical values dominate over strings.

![](images/9dd83a07635f8b163d084663b2553abc31a7b7ae58d651145d280bc592d4210a.jpg)

![](images/68b9eb15ca08a8ddde6810641b77e61bb4577392e4cbbe9ff186654180946207.jpg)  
(a) tus_santos   
(b) wiki_union

![](images/045158bb4dd3767300d9d61d209b9a3b1a64f86a1b1c9ba6f324a6a25f875faa.jpg)  
(c) ugen_v2   
Fig. 6: Comparison of EasyTUS (ET-J and ET-O) against the state-of-the-art in terms of Mean Average Precision $( \mathbf { M A P } @ k )$ and Average Recall $( \mathbf { A } \mathbf { R } @ k )$ across selected benchmark data lakes without metadata. Values of $k$ are plotted on the $\mathbf { X }$ -axis.

When metadata is removed, some approaches experience a significant drop in accuracy, e.g. D3L on tus santos and Starmie on wiki union. Interestingly, ugen v2 even shows improved accuracy for Starmie and Santos. Only EasyTUS demonstrates consistently robust performance across all data lakes including ecb union, regardless of the absence of metadata. This makes EasyTUS particularly well-suited for data

lakes, as its performance is entirely independent of metadata.

# E. Time Efficiency

Table III reports offline and online execution times, whose sum gives the total end-to-end runtime for each approach across all data lakes. We terminated the execution of Santos on opendata after 527 hours, as it was impossible to surpass the best state-of-the-art results. The results show that EasyTUS, particularly with the JinaAI model (ET-J), on average, outperforms most competing approaches. Speedup is computed by averaging execution times per approach within each data lake and then averaging across all lakes. ET-O achieves an offline speedup of $7 9 . 2 4 \times$ and ET-J $6 2 . 1 4 \times$ compared to the next-fastest approach, Starmie. For online execution, ET-J is $7 . 7 2 \times$ faster than Starmie. In contrast, ET-O suffers from network latency, making it $2 2 . 1 5 \times$ slower online, underscoring the trade-off between local LLMs and cloud-based models in terms of both accuracy and time efficiency. Overall, EasyTUS delivers the fastest average execution times across all evaluated approaches.

# F. Data Contamination

Data contamination refers to the risk of evaluating LLMs on benchmark data lakes that may have been included in their training data, thereby undermining the validity of experimental results [22]. To assess EasyTUS’s generalization capabilities and its robustness to truly unseen data, we constructed a synthetic data lake using only tables from open data sources that were published after the release dates of the LLMs employed. This ensures the exclusion of any overlap with the pretraining data of respective LLMs. To support well-defined union operations, we created two distinct data lakes following one-to-one and onto mapping strategies. For both settings, each source table was randomly partitioned horizontally: the initial partition served as the query table, while the remaining partitions were assigned to the data lake. We enforced strict domain separation to prevent semantic overlap between unrelated tables. To better reflect real-world conditions, we injected noise tables, i.e. tables that have no unionable counterparts. For the onto dataset variant, we further randomly removed columns from the target tables. As illustrated in Figure 7, EasyTUS achieves performance on par with or better than state-of-theart in the (a) one-to-one setting, and significantly outperforms them in the more challenging (b) onto scenario, as measured

![](images/ba332e37c47ba6c8fb242450c46b2f9776580bfe6316c0a730ee0cc26448386a.jpg)  
ET-J ET-O

![](images/70ada9531ca6336350df6fb35a273de36d94d54d77e5eaa0756ccd9f06840554.jpg)  
D3L Santos

![](images/187d855da708918183705c690069ac71693e0f465a8c35142228c02a5884ad6d.jpg)  
rmie

![](images/9177d2050b2f4381dc81cdee930fa7e9e0ca85c8dfe0c1efbfd471970303f9ec.jpg)  
(a) One-to-one

![](images/31d6fca8352ff29881ebc0567c758ab2eac7ab0a0c5a2991c4d805a0e2d27961.jpg)

![](images/c2f587046de82b5287c46f8ab947b8abc0dcc6220de0868d17223c1ed2a5798b.jpg)  
Fig. 7: Comparison of Table Union Search approaches on an unseen synthetic data lake with (a) one-to-one and (b) onto mappings. Values of $k$ are plotted on the x-axis.

by Mean Average Precision. These results underscore the robustness and domain-agnostic generalization of EasyTUS.

# VII. CONCLUSION

In this paper, we introduced EasyTUS, a faster and more accurate LLM-based comprehensive framework to the Table Union Search problem. By leveraging off-the-shelf models pre-trained on broad and diverse corpora, EasyTUS demonstrates superior performance without the need for fine-tuning and offers broad applicability across different types of data lakes. Importantly, it operates solely on raw data, while ignoring the metadata to avoid the pitfalls of ambiguity, inaccuracy, or incompleteness. The framework also includes TUSBench, a standardized and modular testbed for evaluating the existing and future Table Union Search approaches. TUSBench enables independent and reproducible benchmarking across different approaches, data lakes, and evaluation metrics. Our experiments show that EasyTUS achieves, on average, up to $3 4 . 3 \%$ higher Mean Average Precision, while being up to $7 9 . 2 \times$ faster in data preparation and up to $7 . 7 \times$ faster in query execution compared to the best prior approaches. Thanks to its modular architecture and support for seamlessly integrating newer, more efficient, and effective models, EasyTUS is wellpositioned to maintain its performance advantage over other approaches without the need for expensive fine-tuning.

# REFERENCES

[1] R. Hai, C. Koutras, C. Quix, and M. Jarke, “Data lakes: A survey of functions and systems,” IEEE Trans. Knowl. Data Eng., vol. 35, no. 12, pp. 12 571–12 590, 2023. [Online]. Available: https://doi.org/10.1109/TKDE.2023.3270101   
[2] F. Nargesian, E. Zhu, R. J. Miller, K. Q. Pu, and P. C. Arocena, “Data lake management: Challenges and opportunities,” Proc. VLDB Endow., vol. 12, no. 12, pp. 1986–1989, 2019.   
[3] F. Nargesian, E. Zhu, K. Q. Pu, and R. J. Miller, “Table union search on open data,” Proc. VLDB Endow., vol. 11, no. 7, pp. 813–825, 2018.   
[4] A. Bogatu, A. A. A. Fernandes, N. W. Paton, and N. Konstantinou, “Dataset discovery in data lakes,” in 36th IEEE International Conference on Data Engineering, ICDE 2020, Dallas, TX, USA, April 20-24, 2020. IEEE, 2020, pp. 709–720. [Online]. Available: https://doi.org/10.1109/ICDE48307.2020.00067   
[5] A. Khatiwada, G. Fan, R. Shraga, Z. Chen, W. Gatterbauer, R. J. Miller, and M. Riedewald, “SANTOS: relationship-based semantic table union search,” Proc. ACM Manag. Data, vol. 1, no. 1, pp. 9:1–9:25, 2023. [Online]. Available: https://doi.org/10.1145/3588689

[6] G. Fan, J. Wang, Y. Li, D. Zhang, and R. J. Miller, “Semantics-aware dataset discovery from data lakes with contextualized column-based representation learning,” Proc. VLDB Endow., vol. 16, no. 7, pp. 1726– 1739, 2023.   
[7] X. Hu, S. Wang, X. Qin, C. Lei, Z. Shen, C. Faloutsos, A. Katsifodimos, G. Karypis, L. Wen, and P. S. Yu, “Automatic table union search with tabular representation learning,” in Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023, A. Rogers, J. L. Boyd-Graber, and N. Okazaki, Eds. Association for Computational Linguistics, 2023, pp. 3786–3800. [Online]. Available: https://doi.org/10.18653/v1/2023.findings-acl.233   
[8] A. Khatiwada, H. Kokel, I. Abdelaziz, S. Chaudhury, J. Dolby, O. Hassanzadeh, Z. Huang, T. Pedapati, H. Samulowitz, and K. Srinivas, “Tabsketchfm: Sketch-based tabular representation learning for data discovery over data lakes,” CoRR, vol. abs/2407.01619, 2024. [Online]. Available: https://doi.org/10.48550/arXiv.2407.01619   
[9] P. Sahoo, A. K. Singh, S. Saha, V. Jain, S. Mondal, and A. Chadha, “A systematic survey of prompt engineering in large language models: Techniques and applications,” CoRR, vol. abs/2402.07927, 2024. [Online]. Available: https://doi.org/10.48550/arXiv.2402.07927   
[10] D. Paulsen, Y. Govind, and A. Doan, “Sparkly: A simple yet surprisingly strong TF/IDF blocker for entity matching,” Proc. VLDB Endow., vol. 16, no. 6, pp. 1507–1519, 2023.   
[11] N. B. Korade, M. B. Salunke, A. A. Bhosle, P. B. Kumbharkar, G. G. Asalkar, and R. G. Khedkar, “Strengthening sentence similarity identification through openai embeddings and deep learning.” International Journal of Advanced Computer Science & Applications, 2024.   
[12] Hugging Face, “Hugging face: The ai community building the future,” https://huggingface.co, 2023, accessed on August 22, 2025.   
[13] Y. A. Malkov and D. A. Yashunin, “Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 42, no. 4, pp. 824–836, 2020. [Online]. Available: https://doi.org/10.1109/TPAMI. 2018.2889473   
[14] F. M. Suchanek, G. Kasneci, and G. Weikum, “Yago: a core of semantic knowledge,” in Proceedings of the 16th International Conference on World Wide Web, WWW 2007, Banff, Alberta, Canada, May 8-12, 2007, C. L. Williamson, M. E. Zurko, P. F. Patel-Schneider, and P. J. Shenoy, Eds. ACM, 2007, pp. 697–706. [Online]. Available: https://doi.org/10.1145/1242572.1242667   
[15] Y. Deng, C. Chai, L. Cao, Q. Yuan, S. Chen, Y. Yu, Z. Sun, J. Wang, J. Li, Z. Cao, K. Jin, C. Zhang, Y. Jiang, Y. Zhang, Y. Wang, Y. Yuan, G. Wang, and N. Tang, “Lakebench: A benchmark for discovering joinable and unionable tables in data lakes,” Proc. VLDB Endow., vol. 17, no. 8, pp. 1925–1938, 2024.   
[16] A. Boutaleb, B. Amann, H. Naacke, and R. Angarita, “Something’s fishy in the data lake: A critical re-evaluation of table union search benchmarks,” CoRR, vol. abs/2505.21329, 2025. [Online]. Available: https://doi.org/10.48550/arXiv.2505.21329   
[17] K. Srinivas, J. Dolby, I. Abdelaziz, O. Hassanzadeh, H. Kokel, A. Khatiwada, T. Pedapati, S. Chaudhury, and H. Samulowitz, “Lakebench: Benchmarks for data discovery over data lakes,” CoRR, vol. abs/2307.04217, 2023. [Online]. Available: https://doi.org/10. 48550/arXiv.2307.04217   
[18] K. Pal, A. Khatiwada, R. Shraga, and R. J. Miller, “Generative benchmark creation for table union search,” CoRR, vol. abs/2308.03883, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2308.03883   
[19] OpenAI, “New embedding models and api updates,” https://openai.com/ index/new-embedding-models-and-api-updates/, Jan. 25 2024.   
[20] S. Sturua, I. Mohr, M. K. Akram, M. Gunther, B. Wang, ¨ M. Krimmel, F. Wang, G. Mastrapas, A. Koukounas, N. Wang, and H. Xiao, “jina-embeddings-v3: Multilingual embeddings with task lora,” CoRR, vol. abs/2409.10173, 2024. [Online]. Available: https://doi.org/10.48550/arXiv.2409.10173   
[21] A. Koukounas, G. Mastrapas, B. Wang, M. K. Akram, S. Eslami, M. Gunther, I. Mohr, S. Sturua, S. Martens, N. Wang, and ¨ H. Xiao, “jina-clip-v2: Multilingual multimodal embeddings for text and images,” CoRR, vol. abs/2412.08802, 2024. [Online]. Available: https://doi.org/10.48550/arXiv.2412.08802   
[22] C. Xu, S. Guan, D. Greene, and M. T. Kechadi, “Benchmark data contamination of large language models: A survey,” CoRR, vol. abs/2406.04244, 2024. [Online]. Available: https://doi.org/10.48550/ arXiv.2406.04244