# jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval

Michael Günther∗, Saba Sturua∗, Mohammad Kalim Akram∗, Isabelle Mohr∗, Andrei Ungureanu∗, Bo Wang∗, Sedigheh Eslami, Scott Martens, Maximilian Werk, Nan Wang and Han Xiao Jina AI GmbH, Prinzessinnenstraße 19, 10969, Berlin, Germany research@jina.ai

# Abstract

We introduce jina-embeddings-v4, a 3.8 billion parameter multimodal embedding model that unifies text and image representations through a novel architecture supporting both single-vector and multi-vector embeddings in the late interaction style. The model incorporates task-specific Low-Rank Adaptation (LoRA) adapters to optimize performance across diverse retrieval scenarios, including query-document retrieval, semantic text similarity, and code search. Comprehensive evaluations demonstrate that jina-embeddings-v4 achieves state-of-the-art performance on both single-modal and cross-modal retrieval tasks, with particular strength in processing visually rich content such as tables, charts, diagrams, and mixed-media formats. To facilitate evaluation of this capability, we also introduce Jina-VDR, a novel benchmark specifically designed for visually rich image retrieval.

# 1 Introduction

We present jina-embeddings-v4, a multimodal embedding model capable of processing text and image data to produce semantic embedding vectors of varying lengths, optimized for a broad array of applications. It incorporates optimized LoRA adapters [Hu et al., 2022] for information retrieval and semantic text similarity. An adapter is also provided for programming language embeddings, technical question-answering, and natural language code retrieval. It also brings new functionality to processing visually rich images (also called visual documents), i.e., materials mixing texts and images, containing tables, charts, diagrams, and other kinds of common mixed media [Ding et al., 2024]. We have also developed Jina-VDR, a new multilingual, multi-domain benchmark suite for a broad range of visual retrieval tasks, to evaluate the capabilities of jina-embeddings-v4.

We discuss the challenges of developing a multimodal, multi-functional, state-of-the-art embedding model capable of handling texts in a variety of languages, including computer coding languages, images, and “visually rich” data. The resulting model, jina-embeddings-v4, projects inputs from all modalities into a unified semantic space, minimizing or eliminating the “modality gap” that has troubled similar projects [Liang et al., 2022a]. In addition, we introduce Jina-VDR, an advanced benchmark for images like screenshots and scans of visually complex documents.

The major contributions of this work are as follows:

• We introduce a unified multi-task learning paradigm that jointly optimizes embedding models to represent texts and images as singleand multi-vector embeddings. • Building on work done for jina-embeddings-v3, we train LoRA extensions to enhance support for specific domains and task types, achieving results comparable to specialized models. • We have made particularly strong progress in handling visually rich images, especially for tasks outside of the existing ViDoRe benchmark [Faysse et al., 2025], which is limited to question-answering. jina-embeddings-v4 outperforms other multimodal models by a significant margin on this type of material and supports a much more diverse set of use scenarios. • We construct a multilingual, multi-domain benchmark for screenshot retrieval. In contrast to other retrieval benchmarks (i.e., [Faysse et al., 2025, Xiao et al., 2025]) that focus on question answering and OCR-related tasks, we expand the scope of visual document benchmarking to multilingual retrieval, more query types, and a much more diverse array of materials, like maps, diagrams, advertisements, and other mixed media.

# 2 Background

The underlying principles behind embedding models are indifferent to data modality. An embedding model transforms digitally encoded objects into vectors in a high-dimensional embedding space such that some of the semantic features of the objects, depending on the model’s training regimen, correspond to subspaces in that embedding space. Objects with more such features in common will have corresponding vectors that are closer to each other by some metric (typically cosine similarity) than objects with fewer common features.

Individual models, however, only support the modalities for which they are designed and trained. Embedding models initially developed primarily to support natural language texts, like jina-embeddings-v3 [Sturua et al., 2024], but there are many image embedding models, and more recently, audio and video models. The semantic embedding paradigm can also encompass models that support more than one modality, like bimodal image-text models, including OpenAI’s CLIP [Radford et al., 2021] and subsequent developments including jina-clip [Koukounas et al., 2024]. The principal purpose of multimodal embedding models is to project objects from multiple modalities into the same semantic embedding space, so that, for example, a picture of a cat and a text discussing or describing a cat will correspond to relatively close embedding vectors.

Embedding models can also specialize in specific types of input within a single modality. There are text embedding models designed for programming code [Liu et al., 2024], legal texts [Voyage AI, 2024], and other special domains. There is also recent work in specialized image embedding models designed to support “visually rich” data, such as screenshots, charts, and printed pages that combine text and imagery and have internal visual structure [Faysse et al., 2025, Ma et al., 2024].

There are other dimensions of embedding model specialization as well. Models can be optimized for specific tasks, such as information retrieval, clustering, and classification [Sturua et al., 2024]. They can also vary based on the nature of the embeddings they produce. Most are single-/dense vector models, generating one embedding vector for whatever input they are given. There are also multi-vector/late interaction models, such as ColBERT [Khattab and Zaharia, 2020] and ColPali [Faysse et al., 2025]. Late interaction is generally a more precise measure of semantic similarity for retrieval, but has significantly greater storage and computing costs.

Instead of specializing, jina-embeddings-v4 builds on a single base model to provide competitive performance as a text, image, and cross-modal embedding model with strong performance handling visually rich documents. It supports both singlevector and multi-vector and is optimized to provide embeddings of varying lengths. Furthermore, the model includes LoRA extensions that optimize it for specific application classes: information retrieval, multimodal semantic similarity, and computer code retrieval.

This single-model approach entails significant savings in practical use cases when compared to deploying multiple AI models for different tasks and modalities.

# 3 Related Work

Transformer-based neural network architectures that generate semantic embeddings are wellestablished [Reimers and Gurevych, 2019], and there is a sizable literature on training techniques for them. Multi-stage contrastive training [Wang et al., 2022], and techniques for supporting longer texts [Günther et al., 2023] are particularly relevant to this work.

Compact embedding vectors bring valuable performance benefits to AI applications, and this motivates work in Matryoshka Representational Learning (MRL) [Kusupati et al., 2022] as a way to train models for truncatable embedding vectors.

Contrastive text-image training has led to groundbreaking results in zero-shot image classification and cross-modal retrieval in conjunction with dual encoder architectures like CLIP [Radford et al., 2021]. However, recent work shows better performance from vision-language models (VLMs) like Qwen2.5-VL-3B-Instruct [Bai et al., 2025]. Jiang et al. [2024] show that VLMs suffer less from a modality gap than dual encoder architectures.

In contrast to [Zhang et al., 2024], jina-embeddings-v4 is trained on multilingual data, supports single as well as multi-vector retrieval, and does not require task-specific instructions. Other VLM models are trained exclusively on data for text-to-image [Faysse et al., 2025, Ma et al., 2024] or text-to-text retrieval [Jiang et al., 2024].

Similarity scoring in late interaction models does not use simple cosine similarity. [Khattab and Zaharia, 2020] Instead, similarity is calculated asymmetrically over two sequences of token embeddings — a query and a document — by summing up the maximum cosine similarity values of each query token embedding to any of the token embeddings from the document. Thus, for query embedding $q$ and document embedding $p$ , their late interaction similarity score $s _ { \mathrm { l a t e } } ( q , p )$ is determined by:

$$
s _ { \mathrm { l a t e } } ( \boldsymbol { q } , \boldsymbol { p } ) = \sum _ { i = 1 } ^ { n } \operatorname* { m a x } _ { \boldsymbol { j } \in \{ 1 , \dots , m \} } \pmb { q } _ { i } \cdot \pmb { p } _ { j } ^ { T }
$$

Faysse et al. [2025] train a late-interaction embedding model to search document screenshots using text queries, performing significantly better than traditional approaches involving OCR and CLIPstyle models trained on image captions. To show this, they introduce the ViDoRe (Vision Document Retrieval) benchmark. However, this benchmark is limited to question-answering tasks in English and French involving only charts, tables, and pages from PDF documents. Xiao et al. [2025] extend this benchmark to create MIEB (Massive Image Embedding Benchmark) by adding semantic textual similarity (STS) tasks for visually rich documents like screenshots.

# 4 Model Architecture

The architecture of jina-embeddings-v4, schematized in Figure 1, employs a unified multimodal language model built on the Qwen2.5-VL-3B-Instruct1 backbone [Bai et al., 2025]. Text and image inputs are processed through a shared pathway: Images are first converted to token sequences via a vision encoder, then both modalities are jointly processed by the language model decoder with contextual attention layers. This unified design eliminates the modality gap present in dual-encoder architectures while maintaining competitive performance across text, image, and cross-modal tasks.

As shown in Figure 1, this architecture supports dual output modes, as outlined in Section 4.2. Furthermore, three task-specific LoRA adapters, each with 60M parameters, provide specialized task optimization without modifying the frozen backbone weights. These are described in Section 4.3.

Table 1: Basic specifications of jina-embeddings-v4   

<table><tr><td>Model Parameters Text Input Size Image Input</td><td>3.8 billion (3.8×10) plus 60M perLoRA Up to 32,768 tokens All images resized to 20</td></tr><tr><td>Single-vector Embedding Size</td><td>megapixels 2048 dimensions, ，truncatable</td></tr><tr><td>Multi-vector Embedding Size</td><td>down to 128 128 dimensions per token</td></tr></table>

The core specifications of jina-embeddings-v4 are summarized in Table 1.

# 4.1 True Multimodal Processing

The Qwen2.5-VL-3B-Instruct paradigm differs from CLIP-style dual-encoder models in offering a single processing path that’s truly multimodal.

For text input, jina-embeddings-v4 and Qwen2.5-VL-3B-Instruct initially behave like other transformer-based embedding models: The text is tokenized, each token is replaced with a vector representation from a lookup table, and then these vectors are stacked and presented to a large language model (LLM).

In CLIP-style models, images are processed by a separate embedding model, typically a transformer-based model that divides them into patches and then processes them much like a text model. The text model and image model are aligned during training to produce similar embeddings for similar semantic content in the different media.

The Qwen2.5-VL-3B-Instruct paradigm used in jina-embeddings-v4 also includes a discrete image model but uses it in a different way. It produces a multi-vector result, comparable to late interaction models, and then passes this output to the LLM, as if it were a sequence of vectorized text tokens. The image embedding model acts as a preprocessor for the LLM, converting the image into what amounts to a sequence of vectorized “image tokens.”

This approach, which is the core of jina-embeddings-v4, beyond having performance advantages, makes it possible to pass a text prompt into the LLM along with an image. The VLM is truly multimodal, since it is one model supporting multiple data types in a single input field.

# 4.2 Dual Mode Output

In contrast to Qwen2.5-VL-3B-Instruct and other embedding models in general, users can choose between two output options: Traditional single (dense)

![](images/cd2f0dcb0b53b333345c31be788a2c0a71842789ddb3d529b13f372c144b8366.jpg)  
Figure 1: Architecture of jina-embeddings-v4. The model employs a unified LM built on the Qwen2.5-VL-3B-Instruct backbone (3.8B parameters). Text and image inputs are processed through a shared pathway: images are first converted to token sequences via a vision encoder, then both modalities are jointly processed by the language model decoder with contextual attention layers. Three task-specific LoRA adapters (60M parameters each) provide specialized optimization for retrieval, text-matching, and code search tasks without modifying the frozen backbone weights. The architecture supports dual output modes: (1) single-vector embeddings (2048 dimensions, truncatable to 128) generated via mean pooling for efficient similarity search, and (2) multi-vector embeddings (128 dimensions per token) via projection layers for the late interaction style retrieval.

vector embeddings and ColBERT-style multi-vector embeddings for late interaction strategies.

Single-vector embeddings are 2048 dimensions, but can be truncated to as little as 128 with minimal loss of precision. jina-embeddings-v4 has been trained with Matryoshka Representation Learning [Kusupati et al., 2022], so that the scalar values of single-vector embeddings are roughly ordered by semantic significance. Eliminating the least significant dimensions reduces precision very little.

Multi-vector embeddings are the unpooled result of processing tokens through a transformer model. They correspond to tokens as the model analyses them, given their context. The length of the output vector is proportionate to the number of input tokens (including “image tokens”), with each token corresponding to a 128-dimensional output vector. This output is directly comparable to the unpooled embeddings produced by ColBERT [Khattab and Zaharia, 2020] and ColPali [Faysse et al., 2025] and is intended for use in late interaction comparison strategies.

For single-vector embeddings, mean pooling is applied to the final layer of the base model to produce the output. The model incorporates an additional layer to project the output of the base model into multi-vector outputs.

# 4.3 Task Specialization with LoRA

Following the methods used for jina-embeddings-v3 [Sturua et al., 2024], we have implemented three task-specific LoRA adapters for different information retrieval use cases:

• Asymmetric Query-Document Retrieval • Semantic Similarity and Symmetric Retrieval • Code (i.e., computer programming language) Retrieval

Asymmetric retrieval means encoding queries and documents differently in order to improve retrieval for queries that are not structured like documents, i.e., short queries, questions, etc. This is in contrast to symmetric retrieval, which assumes a symmetry between query and document, and is used to find comparable content.

Each LoRA adapter set has only 60M parameters, so maintaining all three adds less than $2 \%$ to the memory footprint of jina-embeddings-v4. Users can select among them at inference time, and all three support image and text encoding.

See Section 7 for performance information about these adapters.

# 5 Training Method

Before training, model weights are initialized to match Qwen/Qwen2.5-VL-3B-Instruct. The multi-vector projection layer and LoRA adapters are randomly initialized. The weights of the backbone model are not modified during the training process. The LoRA adapters modify the effect of the backbone model layers and the projection layer. Only the adapters are trained.

Training proceeds in two phases:

1. A single LoRA adapter is trained using contrasting text pairs and text-image pairs. We use the contrastive InfoNCE [van den Oord et al., 2018] loss function to co-train for both single-vector and multi-vector similarity, as detailed in the section below. No task-specific training is performed at this stage. 2. The resulting LoRA adapter is duplicated to create the three task-specific adapters, which are then trained individually with task-specific text triplets and text-image triplets.

In both phases of training, we apply Matryoshka loss [Kusupati et al., 2022] to the base loss so that single-vector embeddings from jina-embeddings-v4 are truncatable.

# 5.1 Pair Training

Initially, training is performed with a contrastive objective. Pairs of inputs are classed as related or unrelated, and the model learns to embed related items closely together and unrelated items further apart.

In each training step, we sample two different batches of training data:

• A batch $\boldsymbol { B } _ { t e x t }$ of text pairs.   
• A batch $B _ { m u l t i }$ of multimodal pairs containing a text and a related image.

We generate normalized single-vector and multivector embeddings for all texts and images in the selected pairs. We then construct a matrix of similarity values $\mathbf { S } _ { \mathrm { d e n s e } } ( B )$ by calculating the cosine similarity of all combinations of single-vector embeddings $\pmb q _ { i }$ and $p _ { j }$ in $\boldsymbol { B }$ . We construct an analogous matrix $\mathbf { S } _ { \mathrm { l a t e } }$ for each $\boldsymbol { B }$ for the multi-vector embeddings using a slightly modified version of Equation (1) to calculate their similarity. Our choice of loss function requires a normalized score, so we divide the late interaction score by the number of tokens in the query:

$$
s _ { \mathrm { l a t e } } ^ { \prime } ( q _ { i } , p _ { j } ) = \frac { s _ { \mathrm { l a t e } } ( q _ { i } , p _ { j } ) } { t }
$$

where $t$ is the number of tokens in $q _ { i }$ and $q _ { i } , p _ { j } \in B$

This modification is only for training. For retrieval applications, normalization is not necessary since the query is invariant.

Then, we apply the contrastive InfoNCE loss function $\mathcal { L } _ { \mathrm { N C E } }$ [van den Oord et al., 2018] on each of the four resulting matrices of similarity scores $s _ { i , j } \in \mathbf { S }$

$$
\begin{array} { r } { \mathrm { s o f t m a x } ( \mathbf { S } , \tau , i , j ) { : = } \mathrm { l n } \frac { e ^ { s _ { i , j } / \tau } } { \displaystyle \sum _ { k = 0 } ^ { n } e ^ { s _ { i , k } / \tau } } } \end{array}
$$

$$
\mathcal { L } _ { \mathrm { N C E } } ( \mathbf { S } ( \mathcal { B } ) , \tau ) : = - \sum _ { i , j = 0 } ^ { n } \mathrm { s o f t m a x } ( \mathbf { S } ( \mathcal { B } ) , \tau , i , i )
$$

where $\tau$ is the temperature parameter, $n$ is the batch size, which increases the weight of small differences in similarity scores in calculating the loss.

Following Hinton et al. [2015], we compensate for differences in error distributions between the single-vector and multi-vector by adding the weighted Kullback–Leibler divergence $( D _ { K L } )$ of the two sets of softmax-normalized similarity scores. This enables us to train for the single-vector and multi-vector outputs simultaneously, even though the multi-vector/late interaction scores have much less error.

$$
\begin{array} { r l } & { \mathcal { L } _ { D } ( B , \tau ) : = D _ { \mathrm { K L } } ( \mathbf { S } _ { \mathrm { d e n s e } } ^ { \prime } ( \mathcal { B } ) \| \mathbf { S } _ { \mathrm { l a t e } } ^ { \prime } ( \mathcal { B } ) ) } \\ & { \quad \mathrm { w h e r e } ~ \mathbf { S } _ { i , j } ^ { \prime } = \mathrm { s o f t m a x } ( \mathbf { S } , \tau , i , j ) } \end{array}
$$

The resulting joint loss function, which we use in training, is defined as:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { j o i n t } } ( \mathcal { B } _ { \mathrm { t x t } } , \mathcal { B } _ { \mathrm { m u l t i } } , \tau ) : = } \\ & { \quad \quad \quad w _ { 1 } \mathcal { L } _ { \mathrm { N C E } } ( \mathbf { S } _ { \mathrm { d e n s e } } ( \mathcal { B } _ { \mathrm { t x t } } ) , \tau ) } \\ & { \quad \quad \quad + w _ { 2 } \mathcal { L } _ { \mathrm { N C E } } ( \mathbf { S } _ { \mathrm { l a t e } } ( \mathcal { B } _ { \mathrm { t x t } } ) , \tau ) + w _ { 3 } \mathcal { L } _ { D } ( \mathcal { B } _ { \mathrm { t x t } } ) } \\ & { \quad \quad \quad + w _ { 4 } \mathcal { L } _ { \mathrm { N C E } } ( \mathbf { S } _ { \mathrm { d e n s e } } ( \mathcal { B } _ { \mathrm { m u l t i } } ) , \tau ) } \\ & { \quad \quad \quad + w _ { 5 } \mathcal { L } _ { \mathrm { N C E } } ( \mathbf { S } _ { \mathrm { l a t e } } ( \mathcal { B } _ { \mathrm { m u l t i } } ) , \tau ) + w _ { 6 } \mathcal { L } _ { D } ( \mathcal { B } _ { \mathrm { m u l t i } } ) } \end{array}
$$

Table 2: Supported tasks of jina-embeddings-v4, each corresponding to a LoRA adapter and trained independently   

<table><tr><td>Task Name</td><td colspan="3">Description</td></tr><tr><td>retrieval</td><td>Asymmetric6 queriesand documents</td><td>embedding</td><td rowspan="3">of for</td></tr><tr><td rowspan="2">text-matching</td><td>retrieval</td><td></td></tr><tr><td>metric retrieval</td><td>Semantic text similarity and sym-</td></tr><tr><td>code</td><td></td><td>Retrieving code snippets</td><td></td></tr></table>

The weights $w _ { 1 } , \ldots , w _ { 6 }$ and temperature $\tau$ are training hyperparameters.

Our training data contains hard negatives, i.e., triplets of a query, a document that matches the query, and a document that is closely semantically related but not a correct match [Wang et al., 2022, Li et al., 2023, Günther et al., 2023]. For every pair $( q _ { i } , p _ { i } ) \in B$ in a batch, $p _ { i }$ is intended to be a good match for $q _ { i }$ , and we presume that for all $( q _ { j } , p _ { j } ) \in B$ where $j \neq i , p _ { j }$ is a bad match for $q _ { i }$ .

We incorporate those additional negatives into the training process via an extended version of the $\mathcal { L } _ { \mathrm { N C E } }$ loss described in Günther et al. [2023], denoted as $\mathcal { L } _ { \mathrm { N C E + } }$ , in our joint loss function $\mathcal { L } _ { \mathrm { j o i n t } }$

# 5.1.1 Pair Training Data

The training data consists of text-text and textimage pairs from more than 300 sources. Text-text pairs are selected and filtered as described in Sturua et al. [2024]. Text-image pairs have been curated from a variety of sources following a more eclectic strategy than previous work on training text-image embedding models. In contrast to relying on image-caption pairs or pairs of queries and images derived from documents, we have also created images from other document types. Our training data includes website screenshots, rendered Markdown files, charts, tables, and other kinds of materials "found in the wild." The queries consist primarily of questions, keywords and key phrases, long descriptions, and statements of fact.

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { N C E + } } ( \mathbf { S } ( \mathcal { B } ) , \tau ) : = } \\ { \displaystyle \sum _ { r \in \mathcal { B } } \left[ - \ln \frac { e ^ { s ( q , p ) / \tau } } { \displaystyle \sum _ { i = 1 } ^ { k } \left[ e ^ { s ( q , p _ { i } ) / \tau } + \sum _ { j = 1 } ^ { m } e ^ { s ( q , n _ { j , i } ) / \tau } \right] } \right] } \end{array}
$$

with $r = ( q , p , n _ { 1 } , . . . , n _ { m } )$ , where $( q , p )$ is a pair in batch $\boldsymbol { B }$ and $n _ { 1 } , . . . , n _ { m }$ and the other $p \in B$ .

Our dataset of text hard negatives is similar to the data used to train jina-embeddings-v3 [Sturua et al., 2024]. We rely on existing datasets to create multimodal hard negatives for training, including Wiki-SS [Ma et al., 2024] and VDR multilingual2, but also mined hard negatives from curated multimodal datasets.

# 5.2 Task-Specific Training

We instantiate three copies of the pair-trained LoRA adapter and give each specific training for its intended task. Training data and loss functions differ for the three tasks.

# 5.2.1 Asymmetric Retrieval Adapter

Asymmetric retrieval assigns substantially and qualitatively different embeddings to documents and queries, even if they happen to have the very same text. Having distinct encoding mechanisms for the two often significantly benefits embeddingsbased retrieval performance. Sturua et al. [2024] shows that this can be achieved either by training two separate adapters or by employing two distinct prefixes as proposed in Wang et al. [2022], so that embedding models can readily distinguish them when they generate embeddings.

We have used the prefix method for jina-embeddings-v4. Previous work shows little benefit from combining both methods.

# 5.2.2 Text Matching Adapter

Symmetric semantic similarity tasks require different training from asymmetric retrieval. We find that training data with ground truth similarity values works best for this kind of task. As discussed in Sturua et al. [2024], we use the $\mathrm { C o S E N T } ^ { 3 }$ loss function $\mathcal { L } _ { \mathrm { c o } }$ from Li and Li [2024]:

$$
\begin{array} { r l } {  { \mathcal { L } _ { \mathrm { c o } } ( \mathbf { S } ( \boldsymbol { B } ) , \tau ) : = \ln \Big [ 1 + \sum _ { \begin{array} { c } { ( q _ { 1 } , p _ { 1 } ) , } \\ { ( q _ { 2 } , p _ { 2 } ) } \end{array} } \frac { e ^ { s ( q _ { 2 } , p _ { 2 } ) } - e ^ { s ( q _ { 1 } , p _ { 1 } ) } } { \tau } \Big ] } \quad } & { { } } \\ { \in \mathbf { S } ( \boldsymbol { B } ) } \end{array}
$$

where $\zeta ( q , p )$ is the ground truth semantic similarity of $q$ with $p$ , $\zeta ( q _ { 1 } , p _ { 1 } ) > \zeta ( q _ { 2 } , p _ { 2 } )$ , and $\tau$ is the temperature parameter.

The loss function operates on two pairs of text values, $( q _ { 1 } , p _ { 1 } )$ and $( q _ { 2 } , p _ { 2 } )$ , with known ground truth similarity.

To train the model with this objective, we use data from semantic textual similarity (STS) training datasets such as STS12 [Agirre et al., 2012] and SICK [Marelli et al., 2014]. The amount of data in this format is limited, so we enhance our ground truth training data with pairs that do not have known similarity scores. For these pairs, we proceed the same way as we did for pair training in Section 5.1 and use the standard InfoNCE loss from Equation (4). The joint loss function is calculated as in Equation (7) except the CoSENT loss is used where pairs with known ground truth values exist.

# 5.2.3 Code Adapter

Code embeddings in jina-embeddings-v4 are designed for natural language-to-code retrieval, code-to-code similarity search, and technical question answering. Code is a very specialized kind of text and requires distinct data sources. Because code embeddings do not involve image processing, the vision portion of jina-embeddings-v4 is not affected by training the code retrieval LoRA adapter.

The backbone LLM Qwen2.5-VL-3B-Instruct was pre-trained on data including the StackExchangeQA4 and the CodeSearchNet [Husain et al., 2020] datasets, giving it some capacity to support code embeddings before further adaptation. Our LoRA training used the same triplet-based method described in Section 5.2.1. Training triplets are derived from a variety of sources, including CodeSearchNet, CodeFeedback [Zheng et al., 2024], APPS [Hendrycks et al., 2021], and the CornStack dataset [Suresh et al., 2025].

We maintained a consistent training configuration by using the same input prefix tokens (e.g., query, passage) and temperature hyperparameter (set to 0.02) during the triplet-based training.

# 6 Jina-VDR: Visually Rich Document Retrieval Benchmark

To evaluate the performance o f jina-embeddings-v4 across a broad range of visually rich document retrieval tasks, we have produced a new benchmark collection and released

it to the public.5

This new collection tests an embedding model’s ability to integrate textual and visual understanding of documents that consist of in the form of rendered images of visual elements like charts, tables, and running text. It extends the ViDoRe benchmark [Faysse et al., 2025] by adding a diverse collection of datasets spanning a broad range of domains (e.g. legal texts, historic documents, marketing materials), covering a variety of material types (e.g. charts, tables, manuals, printed text, maps) and query types (e.g. questions, facts, descriptions), as well as multiple languages.

The benchmark suite encompasses ViDoRe and adds 30 additional tests. These tests include re-purposed existing datasets, new manuallyannotated datasets, and generated synthetic data

For a comprehensive overview of the individual benchmarks, see Appendix A.1.

# 6.1 Re-purposed Datasets

We have adapted a number of existing VQA and OCR datasets, modifying and restructuring them into appropriate query-document pairs.

For example, for DonutVQA6, TableVQA [Tom Agonnoude, 2024], MP-MQA [Zhang et al., 2023], CharXiv [Wang et al., 2024], and PlotQA [Methani et al., 2020], we used structured templates and generative language models to formulate text queries to match their contents.

JDocQAJP7 and HungarianDocQA8 already contain documents and queries in forms that require minimal processing to adapt as benchmarks.

We also created datasets from available data that extend beyond conventional question formats. The OurWorldInData and WikimediaMaps datasets use encyclopedia article fragments and image descriptions as queries to match with charts and maps. The GitHubREADMERetrieval dataset contains rendered Markdown pages drawn from GitHub README files, paired with generated natural language descriptions in 17 languages. The WikimediaCommonsDocuments benchmark pairs multilingual document pages with paragraph-level references extracted from Wikipedia.

# 6.2 Manually Annotated Datasets

We have curated a number of human-annotated resources to better reflect real-world use cases. These include academic slides from Stanford lectures [Mitchener, 2021], educational figures in the TQA dataset [Kembhavi et al., 2017], and marketing and institutional documents such as the Jina AI 2024 Yearbook [Jina AI, 2024], Japanese Ramen [Niigatashi Kanko Kokusaik ¯ ory ¯ ubu Kank ¯ o Suishinka ¯ , 2024], and the Shanghai Master Plan [Shanghai Municipal People’s Government Urban Planning and Land Resource Administration Bureau, 2018]. Documents in these datasets were paired with carefully written human queries without template-based phrasing, capturing genuine information-seeking intent. Some of these datasets target specific languages and regions to provide broader coverage.

We also incorporated pre-existing humanannotated datasets like ChartQA9 and its Arabic counterpart, ArabicChartQA [Ghaboura et al., 2024], which focus on charts and infographics.

# 6.3 Synthetic Data Generation

We have been attentive, in constructing Jina-VDR, to the lack of diversity that often plagues information retrieval benchmarks. We cannot commission human-annotated datasets for everything and have had recourse to generative AI to fill in the gaps.

We obtained a number of datasets from primarily European sources containing scans of historical, legal, and journalistic documents in German, French, Spanish, Italian, and Dutch. We used Qwen2 to generate queries for these documents. We handled the HindiGovernmentVQA and RussianBeverages datasets in the same way, adding not only often underrepresented languages, but also public service documents and commercial catalogs to this benchmark set.

TweetStockRetrieval10 is a collection of chartbased financial data, which we have paired with multilingual template-based generated queries. AirBn-BRetrieval11 is a collection of rendered tables that we have paired with queries in 10 languages generated from a template.

In several cases, such as TableVQA [Delestre, 2024], we introduced bilingual examples (e.g., French/English) to better assess cross-lingual retrieval performance, with questions and answers synthesized using advanced multilingual LLMs such as Gemini 1.5 Pro and Claude 3.5 Sonnet.

# 6.4 Jina-VDR in a Nutshell

Jina-VDR extends the ViDoRe benchmark with:

• 30 new tasks, using both real-world and synthetic data   
• All datasets adapted for retrieval and designed to be compatible with ViDoRe   
• LLM-based filtering to ensure all queries are relevant and reflective of real-world querying   
• Non-question queries, such as GitHub descriptions matched to rendered markdown images, and map images from Wikimedia Commons with accompanying textual descriptions   
• Multilingual coverage, with some datasets spanning up to 20 languages

# 7 Evaluation

We have evaluated jina-embeddings-v4 on a diverse set of benchmarks to reflect its multiple functions. Table 3 provides an overview of benchmark averages for jina-embeddings-v4 and other embedding models.

# 7.1 Multilingual Text Retrieval

MTEB and MMTEB [Enevoldsen et al., 2025] are the most widely used text retrieval benchmarks. For most tasks, we have used the asymmetric retrieval adapter, but for some symmetric retrieval tasks like ArguAna12, we have used the text matching adapter instead. For evaluation, we prepend the query with the prefix “Given a claim, find documents that refute the claim” to reflect the task’s focus on retrieving passages that contradict, rather than support, the input claim, similar to Wang et al. [2023]. The results are tabulated in Appendix A.4.

For the MTEB benchmarks, which are all in English, see Table A11, and for the multilingual MMTEB, see Table A12. The performance of this new model is generally better than our previous model jina-embeddings-v3 and broadly comparable with the state-of-the-art.

Table 3: Average Retrieval Scores of Embedding Models on Various Benchmarks.   

<table><tr><td>Model</td><td>J-VDR</td><td>ViDoRe</td><td>CLIPB</td><td>MMTEB</td><td>MTEB-en</td><td>COIR</td><td>LEMB</td><td> STS-m</td><td> STS-en</td></tr><tr><td>jina-embeddings-v4 (dense) jina-embeddings-v4 (late)</td><td>73.98 80.55</td><td>84.11 90.17</td><td>84.11</td><td>66.49</td><td>55.97</td><td>71.59</td><td>67.11</td><td>72.70</td><td>85.89</td></tr><tr><td>text-embedding-3-large</td><td>1</td><td>1</td><td></td><td>59.27</td><td>57.98</td><td>62.36</td><td>52.42</td><td>70.17</td><td>81.44</td></tr><tr><td>bge-m3 multilingual-e5-large-instruct</td><td>1</td><td>1</td><td></td><td>55.36 57.12</td><td></td><td></td><td>58.73 41.76</td><td></td><td></td></tr><tr><td>jina-embeddings-v3</td><td>47.82</td><td>26.02</td><td></td><td>58.58</td><td>53.47</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>54.33</td><td>55.07</td><td>55.66</td><td>75.77</td><td>85.82</td></tr><tr><td>voyage-3</td><td>1</td><td>1</td><td></td><td>66.13</td><td>53.46</td><td>67.23</td><td>74.06</td><td>68.33</td><td>78.59</td></tr><tr><td>gemini-embedding-001</td><td></td><td></td><td></td><td>67.71</td><td>64.35</td><td>73.11</td><td></td><td>78.35</td><td>85.29</td></tr><tr><td>jina-embedings-v2-code</td><td></td><td></td><td></td><td></td><td></td><td>52.24</td><td></td><td></td><td></td></tr><tr><td>voyage-code</td><td></td><td></td><td>1</td><td></td><td></td><td>77.33</td><td></td><td></td><td></td></tr><tr><td>nllb-clip-large-siglip</td><td></td><td></td><td>83.19</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> jina-clip-v2</td><td>40.52</td><td>53.61</td><td>81.12</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>colpali-v1.2 (late)</td><td>63.80</td><td>83.90</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>dse-qwen2-2b-mrl-v1(dense)</td><td>67.25</td><td>85.80</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>voyage-multimodal-v3 (dense)</td><td></td><td>84.24</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Task Acronyms: $\scriptstyle \mathbf { J - V D R } =$ Jina VDR, VidoRE $=$ ViDoRe, CLIPB $=$ CLIP Benchmark, MMTEB $=$ MTEB(Multilingual, v2)Retrieval Tasks, MTEB-EN $=$ MTEB(eng, v2) Retrieval Tasks, $\mathbf { C O I R } = \mathbf { C o I R }$ Code Retrieval, LEMB $=$ LongEmbed, STS- $\mathbf { m } =$ MTEB(Multilingual, v2) Semantic Textual Similarity Tasks, $\mathrm { S T S - e n } = \mathrm { M T E B } ( \mathrm { e n g } , \mathrm { v 2 } )$ Semantic Textual Similarity Tasks

Average Calculation: For J-VDR and ViDoRE, we calculate the average for the multilingual tasks first and consider this as a single score before calculating the average across all tasks. Scores are $\mathrm { n D C G } @ 5$ for J-VDR, ViDoRe, and CLIPB, and nDCG $@ 1 0$ for MMTEB, MTEB-en, COIR, and LEMB, and Spearman coefficient for STS-m and STS-en.

Evaluation of Text Retrieval Models on J-VDR: For evaluating text retrieval models on J-VDR, we used EasyOCR (https: //github.com/JaidedAI/EasyOCR) and the provided extracted texts from the original ViDoRe datasets.

We have also evaluated the performance of our model on retrieval tasks that involve long text documents using the LongEmbed benchmark [Zhu et al., 2024]. The results are tabulated in Table A13 of Appendix A.4. Long document performance for jina-embeddings-v4 significantly outpaces competing models except the voyage-3 series and improves dramatically on jina-embeddings-v3’s performance.

# 7.2 Textual Semantic Similarity

We evaluated jina-embeddings-v4 with textbased semantic similarity (STS) benchmarks. The results for MTEB STS and MMTEB STS benchmarks are tabulated in Tables A14 and A15 of Appendix A.4 respectively. Our results are competitive with the state-of-the-art and are best-in-class for English similarity tasks.

# 7.3 Multimodal Retrieval

To evaluate the model’s performance on typical text-to-image search tasks, we used the common English and non-English tasks of the CLIP Benchmark13. The results are tabulated in Tables A8 to A10 of Appendix A.3. jina-embeddings-v4 has a higher average score than jina-clip-v2 and nllb-siglip-large, but the latter performs somewhat higher on the Crossmodal3600 benchmark [Thapliyal et al., 2022] (see Table A10) because it includes content from low-resource languages not supported in jina-embeddings-v4’s Qwen2.5-VL-3B-Instruct backbone.

We further tested jina-embeddings-v4 on the ViDoRe and Jina-VDR benchmarks, to evaluate its performance on visually rich documents. The results are compiled in Appendix A.2. ViDoRe scores are tabulated in Table A3. Table A2 provides an overview of jina-embeddings-v4 compared to other models, with Tables A4 to A7 providing details results for some individual Jina-VDR benchmarks.

This suggests that other models are primarily trained to perform well on document retrieval tasks that are similar to the ViDoRe tasks but underperform on other tasks, e.g., that do not involve queries that resemble questions.

jina-embeddings-v4 excels at this benchmark, providing the current state-of-the-art, in both singleand multi-vector mode. Multi-vector/late interaction matching is generally recognized as more precise than single-vector matching in other applications, and this remains true for Jina-VDR.

# 7.4 Code Retrieval

To assess performance on code retrieval, we evaluate the model on the MTEB-CoIR benchmark [Li et al., 2024], which consists of 10 tasks spanning text-to-code, code-to-text, code-to-code, and hybrid code retrieval types. The results are reported in Table A16 of Appendix A.4. jina-embeddings-v4 is competitive with the state-of-the-art in general-purpose embedding models, but the specialized voyage-code model has somewhat better benchmark performance.

# 8 Analysis of the Embedding Space

The large difference in architecture between jina-embeddings-v4 and CLIP-style models like OpenAI CLIP [Radford et al., 2021] and jina-clip-v2 implies a large difference in the structure of the embedding spaces those models generate. We look here at a few of these issues.

# 8.1 Modality Gap

Previous work has shed light on the so-called modality gap in multimodal models trained with contrastive learning [Liang et al., 2022b, Schrodi et al., 2024, Eslami and de Melo, 2025]. Good semantic matches across modalities tend to lie considerably further apart in the embedding space than comparable or even worse matches of the same modality, i.e., texts in CLIP-style models are more similar to semantically unrelated texts than to semantically similar images.

We can see the modality gap directly by examining the distribution of pairwise cosine similarities of matching image-text pairs versus matching text-text pairs. In Figure 2, we see the distribution of similarity values for the two pair types in OpenAI CLIP, jina-clip-v2, and jina-embeddings-v4.

The gap is dramatically reduced with jina-embeddings-v4 because of its crossmodel encoder, illustrated in Figure 1. Eslami and de Melo [2025] has shown that sharing an encoder between modalities introduces an inductive bias towards using a shared region of the embedding space, while Liang et al. [2022b] shows the opposite is true for CLIP-style architectures with separate encoders.

# 8.2 Cross-Modal Alignment

Eslami and de Melo [2025] has defined the crossmodal alignment score of a multimodal embedding model as the average of cosine similarities of matching pairs of image and text embeddings. Table 4 calculates this score for jina-embeddings-v4 and OpenAI CLIP with data sampled from the Flickr $3 0 \mathrm { K } ^ { 1 5 }$ , MSCOCO [Lin et al., 2014], and CIFAR- $1 0 0 ^ { 1 6 }$ datasets.

![](images/64e1a2a86fd10e434985b139d5cb795de1b0cb555de6575443f759c845db8924.jpg)  
Figure 2: Distribution of the cosine similarities of the paired image-text embeddings versus paired texttext embeddings from the Flick $\mathrm { \Omega } \cdot 8 \mathrm { K } ^ { 1 4 }$ dataset. Top: OpenAI CLIP, Middle: jina-clip-v2, Bottom: jina-embeddings-v4

These results confirm that jina-embeddings-v4 generates a far better aligned cross-modal embedding space than CLIP-style models.

It is worth noting that jina-embeddings-v4 shows much poorer alignment for CIFAR-100 data than MSCOCO and Flickr30K. This is because

Table 4: Comparison of cross-modal alignment scores on 1K of random samples from each dataset.   

<table><tr><td>Model</td><td>Flickr30K</td><td>MSCOCO</td><td>CIFAR-100</td></tr><tr><td>OpenAI-CLIP</td><td>0.15</td><td>0.14</td><td>0.2</td></tr><tr><td> jina-clip-v2</td><td>0.38</td><td>0.37</td><td>0.32</td></tr><tr><td> jina-embeddings-v4</td><td>0.71</td><td>0.72</td><td>0.56</td></tr></table>

CIFAR-100 is a classification dataset and its labels are far less informative than the more descriptive texts in MSCOCO and Flickr30K.

# 8.3 Cone Effect

Liang et al. [2022b] demonstrate that multimodal models trained with contrastive loss suffer from an inductive bias known as the cone effect. Each modality tends to cluster together in randomized embedding spaces before training, and contrastive loss tends to make the cross-modal matching pairs form a kind of high-dimensional cone, linking one part of the embedding space to another rather than distributing embeddings evenly.

The impact of the cone effect can be seen in Figure 3. The difference in cosine similarity between correct and incorrect text-image matches is quite small for OpenAI CLIP (top), significantly greater in jina-clip-v2 (middle), but jina-embeddings-v4 (bottom) shows a much greater spread of cosine similarity ranges with very distinctly separate peaks for positive and negative pairs. This shows that jina-embeddings-v4 uses much more of the embedding space and image and text embeddings have a much greater overlap in distribution.

# 9 Conclusion

We present jina-embeddings-v4, a state-of-theart multimodal and multilingual embedding model designed for a wide range of tasks, including semantic text retrieval, text-to-image retrieval, text-to-visually-rich document retrieval, and code search. The model achieves strong performance using single-vector representations and demonstrates even greater effectiveness with multi-vector representations, particularly in visually rich document retrieval. jina-embeddings-v4 aligns representations across modalities into a single, shared semantic space, sharply reducing structural gaps between modalities compared to CLIP-style dual-tower models, enabling more effective cross-modal retrieval.

In future work, we plan to further enhance this model’s multilingual capabilities and explore techniques to create smaller, more efficient variants.

![](images/5471744f9a5a8a7ff50d50bbdb17d92b5780933075df76ce34b000adcc6a950e.jpg)  
Figure 3: Distribution of the cosine similarities of positive (correct matches) versus negative (incorrect matches) image-text samples. (top) OpenAI CLIP, (middle) jina-clip-v2, (bottom) jina-embeddings-v4.

