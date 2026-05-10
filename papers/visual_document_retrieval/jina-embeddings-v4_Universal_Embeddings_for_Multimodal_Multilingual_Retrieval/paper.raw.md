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

# References

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR, 2022.

Yihao Ding, Soyeon Caren Han, Jean Lee, and Eduard Hovy. Deep learning based visually rich document content understanding: A survey. arXiv preprint arXiv:2408.01287, 2024.

Victor Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Y Zou. Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning. Advances in Neural Information Processing Systems, 35:17612–17625, 2022a.

Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. ColPali: Efficient Document Retrieval with Vision Language Models. In The Thirteenth International Conference on Learning Representations, 2025.

Chenghao Xiao, Isaac Chung, Imene Kerboua, Jamie Stirling, Xin Zhang, Márton Kardos, Roman Solomatin, Noura Al Moubayed, Kenneth Enevoldsen, and Niklas Muennighoff. MIEB: Massive Image Embedding Benchmark. arXiv preprint arXiv:2504.10471, 2025.

Saba Sturua, Isabelle Mohr, Mohammad Kalim Akram, Michael Günther, Bo Wang, Markus Krimmel, Feng Wang, Georgios Mastrapas, Andreas Koukounas, Nan Wang, et al. jina-embeddings-v3: Multilingual Embeddings With Task LoRA. arXiv preprint arXiv:2409.10173, 2024.

Alec Radford, Jong Wook Kim, et al. Learning Transferable Visual Models from Natural Language Supervision. In International Conference on Machine Learning, pages 8748–8763, 2021.

Andreas Koukounas, Georgios Mastrapas, Michael Günther, Bo Wang, Scott Martens, Isabelle Mohr, Saba Sturua, Mohammad Kalim Akram, Joan Fontanals Martínez, Saahil Ognawala, et al. Jina CLIP: Your CLIP Model Is Also Your Text Retriever. arXiv preprint arXiv:2405.20204, 2024.

Ye Liu, Rui Meng, Shafiq Joty, Silvio Savarese, Caiming Xiong, Yingbo Zhou, and Semih Yavuz. CodeX-Embed: A Generalist Embedding Model Family for Multilingual and Multi-task Code Retrieval. arXiv preprint arXiv:2411.12644, 2024.

Voyage AI. Domain-Specific Embeddings and Retrieval: Legal Edition (voyage-law-2), 2024. https://blog.voyageai.com/2024/04/15/domainspecific-embeddings-and-retrieval-legal-editionvoyage-law-2/.

Xueguang Ma, Sheng-Chieh Lin, Minghan Li, Wenhu Chen, and Jimmy Lin. Unifying Multimodal Retrieval via Document Screenshot Embedding. In

Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 6492–6505, 2024.

Omar Khattab and Matei Zaharia. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, pages 39–48, 2020.

Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, 2019.

Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. Text Embeddings by Weakly-Supervised Contrastive Pre-training. arXiv preprint arXiv:2212.03533, 2022.

Michael Günther, Jackmin Ong, Isabelle Mohr, Alaeddine Abdessalem, Tanguy Abel, Mohammad Kalim Akram, Susana Guzman, Georgios Mastrapas, Saba Sturua, Bo Wang, et al. Jina Embeddings 2: 8192- Token General-Purpose Text Embeddings for Long Documents. arXiv preprint arXiv:2310.19923, 2023.

Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, et al. Matryoshka Representation Learning. Advances in Neural Information Processing Systems, 35:30233–30249, 2022.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923, 2025.

Ting Jiang, Minghui Song, Zihan Zhang, Haizhen Huang, Weiwei Deng, Feng Sun, Qi Zhang, Deqing Wang, and Fuzhen Zhuang. E5-V: Universal Embeddings with Multimodal Large Language Models. arXiv preprint arXiv:2407.12580, 2024.

Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. GME: Improving Universal Multimodal Retrieval by Multimodal LLMs. arXiv preprint arXiv:2412.16855, 2024.

Aäron van den Oord, Yazhe Li, and Oriol Vinyals. Representation Learning with Contrastive Predictive Coding. CoRR, abs/1807.03748, 2018. URL http://arxiv.org/abs/1807.03748.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the Knowledge in a Neural Network. arXiv preprint arXiv:1503.02531, 2015.

Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards General Text Embeddings with Multi-stage Contrastive Learning. arXiv preprint arXiv:2308.03281, 2023.

Xianming Li and Jing Li. Aoe: Angle-optimized embeddings for semantic textual similarity. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1825–1839, 2024.

Eneko Agirre, Daniel Cer, Mona Diab, and Aitor Gonzalez-Agirre. Semeval-2012 task 6: A pilot on semantic textual similarity. In SEM 2012: The First Joint Conference on Lexical and Computational Semantics–Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation (SemEval 2012), pages 385–393, 2012.

Marco Marelli, Stefano Menini, Marco Baroni, Luisa Bentivogli, Raffaella Bernardi, and Roberto Zamparelli. A SICK cure for the evaluation of compositional distributional semantic models. In Nicoletta Calzolari, Khalid Choukri, Thierry Declerck, Hrafn Loftsson, Bente Maegaard, Joseph Mariani, Asuncion Moreno, Jan Odijk, and Stelios Piperidis, editors, Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14), pages 216–223, Reykjavik, Iceland, May 2014. European Language Resources Association (ELRA). URL http://www.lrec-conf.org/proceedings/ lrec2014/pdf/363_Paper.pdf.

Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. CodeSearchNet Challenge: Evaluating the State of Semantic Code Search. arXiv preprint arXiv:1909.09436, 2020.

Tianyu Zheng, Ge Zhang, Tianhao Shen, Xueling Liu, Bill Yuchen Lin, Jie Fu, Wenhu Chen, and Xiang Yue. OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024, pages 12834–12859. Association for Computational Linguistics, August 2024.

Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, and Jacob Steinhardt. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 2021.

Tarun Suresh, Revanth Gangi Reddy, Yifei Xu, Zach Nussbaum, Andriy Mulyar, Brandon Duderstadt, and Heng Ji. Cornstack: High-quality contrastive data for better code retrieval and reranking. arXiv preprint arXiv:2412.01007, 2025.

Cyrile Delestre Tom Agonnoude, 2024. URL https:// huggingface.co/datasets/cmarkea/table-vqa.

Liang Zhang, Anwen Hu, Jing Zhang, Shuo Hu, and Qin Jin. Mpmqa: Multimodal question answering on product manuals. 2023. URL https://arxiv.org/abs/2304.09660.

Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, Alexis Chevalier, Sanjeev Arora, and Danqi Chen. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. arXiv preprint arXiv:2406.18521, 2024.

Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra, and Pratyush Kumar. Plotqa: Reasoning over scientific plots. In The IEEE Winter Conference on Applications of Computer Vision (WACV), March 2020.

Kris James Mitchener. National banking statistics, 1867–1896. https://purl.stanford.edu/ mv327tb8364, 2021. Dataset published via Stanford Digital Repository.

Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. pages 5376–5384, 07 2017. doi:10.1109/CVPR.2017.571.

Jina AI. Re·Search: Order 2024 Yearbook of Search Foundation Advances. Jina AI, Sunnyvale, CA & Berlin, Germany, December 16 2024. Hardcover with spot UV coating; includes complimentary digital copy. Available at: https://jina.ai/news/re-searchorder-2024-yearbook-of-search-foundationadvances/.

Niigata-shi Kanko Kokusaik ¯ ory ¯ ubu Kank ¯ o Su- ¯ ishinka. Niigata City Ramen Guidebook. City of Niigata, Niigata, Japan, July 2024. URL https://www.city.niigata.lg.jp/kanko/kanko/ oshirase/ramen.files/guidebook.pdf. PDF, approximately 28 MB.

Shanghai Municipal People’s Government Urban Planning and Land Resource Administration Bureau. Shanghai Master Plan 2017–2035: Striving for the Excellent Global City. Shanghai Municipal People’s Government, Shanghai, China, January 2018. URL https://www.shanghai.gov.cn/newshanghai/ xxgkfj/2035004.pdf. Public Reading edition; government-issued planning document.

Sara Ghaboura, Ahmed Heakl, Omkar Thawakar, Ali Alharthi, Ines Riahi, Abduljalil Saif, Jorma Laaksonen, Fahad S. Khan, Salman Khan, and Rao M. Anwer. Camel-bench: A comprehensive arabic lmm benchmark. 2024. URL https://arxiv.org/abs/2410.18976.

Cyrile Delestre, 2024. URL https:// huggingface.co/datasets/cmarkea/aftdb.

Kenneth Enevoldsen, Isaac Chung, et al. MMTEB: Massive Multilingual Text Embedding Benchmark. 2025.

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. Improving text embeddings with large language models. arXiv preprint arXiv:2401.00368, 2023.

Dawei Zhu, Liang Wang, Nan Yang, Yifan Song, Wenhao Wu, Furu Wei, and Sujian Li. LongEmbed: Extending Embedding Models for Long Context Retrieval. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 802–816, 2024.

Ashish V. Thapliyal, Jordi Pont Tuset, Xi Chen, and Radu Soricut. Crossmodal-3600: A massively multilingual multimodal evaluation dataset. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 715–729, 2022.

Xiangyang Li, Kuicai Dong, Yi Quan Lee, Wei Xia, Yichun Yin, Hao Zhang, Yong Liu, Yasheng Wang, and Ruiming Tang. Coir: A comprehensive benchmark for code information retrieval models. CoRR, 2024.

Victor Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Y Zou. Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. Advances in Neural Information Processing Systems, 35:17612–17625, 2022b.

Simon Schrodi, David T Hoffmann, Max Argus, Volker Fischer, and Thomas Brox. Two effects, one trigger: on the modality gap, object bias, and information imbalance in contrastive vision-language representation learning. arXiv preprint arXiv:2404.07983, 2024.

Sedigheh Eslami and Gerard de Melo. Mitigate the gap: Improving cross-modal alignment in clip. In The Thirteenth International Conference on Learning Representations, 2025.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft coco: Common objects in context. In David Fleet, Tomas Pajdla, Bernt Schiele, and Tinne Tuytelaars, editors, Computer Vision – ECCV 2014, pages 740–755, Cham, 2014. Springer International Publishing. ISBN 978-3-319-10602-1.

# A Appendix

# A.1 Datasets in the Jina-VDR Benchmark

Table A1: Overview of the Dataset Collection   

<table><tr><td>Dataset Name</td><td>Domain</td><td>Document Format</td><td>Query Format</td><td>Number of Queries / Documents</td><td>Languages</td></tr><tr><td> jinaai/airbnb-synthetic-retrievalt</td><td>Housing</td><td>Tables</td><td>Instruction</td><td>4953/10000</td><td>ar, de, en, es, fr, hi, hu,ja</td></tr><tr><td>jinaai/arabic_chartqa_ar</td><td>Mixed</td><td>Charts</td><td>Question</td><td>745 /342</td><td>ru, zh ar</td></tr><tr><td>jinaai/arabic_infographicsvqa_ar</td><td>Mixed</td><td>Illustrations</td><td>Question</td><td>120 /40</td><td>ar</td></tr><tr><td> jinaai/automobile_catalogue_jp</td><td>Marketing</td><td>Catalog</td><td>Question</td><td>45/15</td><td>ja</td></tr><tr><td>jinaai/arxivqa</td><td>Science</td><td>Mixed</td><td>Question</td><td>30/499</td><td>en</td></tr><tr><td> jinaai/beverages_catalogue_ru</td><td>Marketing</td><td>Digital Docs</td><td>Question</td><td>100/34</td><td>ru</td></tr><tr><td> jinaai/ChartQA</td><td>Mixed</td><td>Charts</td><td>Question</td><td>996 /834</td><td>en</td></tr><tr><td> jinaai/CharXiv-en</td><td>Science</td><td>Charts</td><td>Question</td><td>999 /1000</td><td>en</td></tr><tr><td> jinaai/docvqa</td><td>Mixed</td><td>Scans</td><td>Question</td><td>39 /499</td><td>en</td></tr><tr><td> jinaai/donut_vqa</td><td>Medical</td><td>Scans / Handwriting</td><td>Question</td><td>704/800</td><td>en</td></tr><tr><td>jinaai/docqa_artificial_intelligence</td><td>Software /IT</td><td>Digital Docs</td><td>Question</td><td>70/962</td><td>en</td></tr><tr><td>jinaai/docqa_energy</td><td>Energy</td><td>Digital Docs</td><td>Question</td><td>69/971</td><td>en</td></tr><tr><td>jinaai/docqa_gov_report</td><td>Government</td><td>Digital Docs</td><td>Question</td><td>77 /970</td><td>en</td></tr><tr><td> jinaai/docqa_healthcare_industry</td><td>Medial</td><td>Digital Docs</td><td>Question</td><td>90/961</td><td>en</td></tr><tr><td>jinaai/europeana-de-news</td><td>Historic</td><td>Scans /News Articles</td><td>Question</td><td>379/137</td><td>de</td></tr><tr><td>jinaai/europeana-es-news</td><td>Historic</td><td>Scans /News Articles</td><td>Question</td><td>474/179</td><td>es</td></tr><tr><td>jinaai/europeana-fr-news</td><td>Historic</td><td>Scans /News</td><td>Question</td><td>237/145</td><td>fr</td></tr><tr><td>jinaai/europeana-it-scans</td><td>Historic</td><td>Articles Scans</td><td>Question</td><td>618 /265</td><td>it</td></tr><tr><td> jinaai/europeana-nl-legal</td><td>Legal</td><td>Scans</td><td>Question</td><td>199 /244</td><td>nl</td></tr><tr><td>jinaai/github-readme-retrieval- multilingualt</td><td>Software /IT</td><td>Markdown Docs</td><td>Description</td><td>16953 /16998</td><td>ar, bn,de,en, es,fr, hi,id,it, ja, ko,nl pt,</td></tr><tr><td>jinaai/hindi-gov-vqa</td><td>Governmental</td><td>Digital Docs</td><td>Question</td><td>454/337</td><td>ru,th,vi,zh hi</td></tr><tr><td>jinaai/hungarian_doc_qa_hu</td><td>Mixed</td><td>Digital Docs</td><td>Question</td><td>54/51</td><td>hu</td></tr><tr><td> jinaai/infovqa</td><td>Mixed</td><td>Illustrations</td><td>Question</td><td>363 /500</td><td>en</td></tr><tr><td> jinaai/jdocqa</td><td>News</td><td>Digital Docs Digital Docs</td><td>Question</td><td>744/758 75/33</td><td>ja</td></tr><tr><td>jinaai/jina_2024_yearly_book</td><td>Software /IT</td><td>Digital Docs</td><td>Question</td><td></td><td>en</td></tr><tr><td> jinaai/medical-prescriptions</td><td>Medical</td><td>Digital Docs</td><td>Question</td><td>100/100</td><td>en</td></tr><tr><td> jinaai/mpmqa-small</td><td>Manuals</td><td>Tables</td><td>Question</td><td>155/782</td><td>en</td></tr><tr><td> jinaai/MMTab</td><td>Mixed</td><td></td><td>Fact</td><td>987/906</td><td>en</td></tr><tr><td>jinaai/openai-news</td><td>Software /IT</td><td>Digital Docs</td><td>Question</td><td>31/30</td><td>en</td></tr><tr><td>jinaai/owid_charts_en</td><td>Mixed</td><td>Charts</td><td>Question</td><td>132/937</td><td>en</td></tr><tr><td> jinaai/plotqa</td><td>Mixed</td><td>Charts</td><td>Question</td><td>610/986</td><td>en</td></tr><tr><td> jinaai/ramen_benchmark_jp</td><td>Marketing</td><td>Catalog</td><td>Question</td><td>29/10</td><td>ja</td></tr><tr><td> jinaai/shanghai_master_plan</td><td>Governmental</td><td>Digital Docs</td><td>Question /</td><td>57/23</td><td>zh,en</td></tr><tr><td>jinaai/wikimedia-commons-</td><td>Mixed</td><td>Mixed</td><td>Key Phrase Description</td><td>15593/15217</td><td>ar, bn,de, en, es,fr, hi, hu,</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>id, it, ja, ko, my, nl, pt, ru, th,ur,vi, zh</td></tr><tr><td></td><td>Environmental Documents</td><td>Digital Docs</td><td>Question</td><td>89/998</td><td>fr</td></tr><tr><td>jinaai/stanford_slide jinaai/student-enrollment</td><td>Education</td><td>Slides</td><td>Question</td><td>14 /994</td><td>en</td></tr><tr><td>jinaai/tabfquad</td><td>Demographics Mixed</td><td>Charts Tables</td><td>Question</td><td>1000 /489</td><td>en fr,en</td></tr><tr><td> jinaai/table-vqa</td><td>Science</td><td>Tables</td><td>Question Question</td><td>126/70 992/385</td><td>en</td></tr><tr><td> jinaai/tatqa</td><td>Finance</td><td>Digital Docs</td><td>Question</td><td>121/270</td><td>en</td></tr><tr><td> jinaai/tqa</td><td>Education</td><td>Illustrations</td><td>Question</td><td>981/393</td><td>en</td></tr><tr><td>jinaai/tweet-stock-synthetic- retrievalt</td><td>Finance</td><td>Charts</td><td>Question</td><td>6278/10000</td><td>ar,de, en, es, hi, hu, ja, ru, zh</td></tr><tr><td> jinaai/wikimedia-commons-maps</td><td>Mixed</td><td>Maps</td><td>Description</td><td>443 /451</td><td>en</td></tr></table>

†For multilingual datasets, the total number of queries and documents is the sum across all language-specific splits.

# A.2 JinaVDR (Visual Document Retrieval) Benchmark Results

Table A2: Overview of JinaVDR Results for Various Models   

<table><tr><td>Task</td><td>bm25 +</td><td>jev3 + OCR</td><td>j-clip- v2</td><td>colpali- v1.2</td><td>dse-qwen2- 2b-mrl-v1</td><td>jev4- single</td><td>jev4- multi</td></tr><tr><td>Average</td><td>OCR 46.88</td><td>48.97</td><td>40.96</td><td>65.39</td><td>68.89</td><td>75.47</td><td>81.52</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>medical-prescriptions stanford_slide</td><td>38.18 81.78</td><td>38.12 95.28</td><td>15.68 91.48</td><td>66.22 100.0</td><td>38.86 100.0</td><td>80.95 100.0</td><td>97.69 97.16</td></tr><tr><td>donut_vqa</td><td>19.39</td><td>2.59</td><td>1.46</td><td>34.12</td><td>25.31</td><td>78.60</td><td>74.08</td></tr><tr><td>table-vqa</td><td>55.22</td><td>63.04</td><td>36.34</td><td>80.98</td><td>85.70</td><td>86.57</td><td>89.21</td></tr><tr><td>ChartQA</td><td>28.39</td><td>31.47</td><td>39.73</td><td>54.45</td><td>58.38</td><td>70.88</td><td>71.80</td></tr><tr><td>tqa</td><td>50.11</td><td>24.40</td><td>27.80</td><td>63.03</td><td>65.35</td><td>65.44</td><td>68.46</td></tr><tr><td>openai-news</td><td>76.63</td><td>87.30</td><td>70.05</td><td>94.81</td><td>93.75</td><td>93.97</td><td>96.43</td></tr><tr><td>europeana-de-news</td><td>11.26</td><td>12.02</td><td>11.18</td><td>35.20</td><td>44.32</td><td>48.89</td><td>63.76</td></tr><tr><td>europeana-es-news</td><td>51.99</td><td>43.82</td><td>12.95</td><td>45.70</td><td>60.66</td><td>60.81</td><td>80.70</td></tr><tr><td>europeana-it-scans</td><td>39.11</td><td>38.77</td><td>16.54</td><td>58.70</td><td>54.28</td><td>58.01</td><td>73.29</td></tr><tr><td>europeana-nl-legal</td><td>39.38</td><td>34.24</td><td>11.30</td><td>39.13</td><td>33.12</td><td>42.77</td><td>59.82</td></tr><tr><td>hindi-gov-vqa</td><td>1.83</td><td>7.51</td><td>5.21</td><td>11.43</td><td>10.19</td><td>15.32</td><td>22.49</td></tr><tr><td>automobile_catalogue_jp</td><td>20.92</td><td>50.39</td><td>32.54</td><td>35.72</td><td>66.44</td><td>72.22</td><td>81.32</td></tr><tr><td>beverages_catalogue_ru</td><td>11.05</td><td>14.09</td><td>39.66</td><td>68.47</td><td>80.32</td><td>85.68</td><td>87.73</td></tr><tr><td>ramen_benchmark_jp</td><td>28.02</td><td>63.37</td><td>41.28</td><td>52.03</td><td>51.66</td><td>90.77</td><td>94.65</td></tr><tr><td> jdocqa</td><td>1.64</td><td>7.85</td><td>19.94</td><td>35.68</td><td>67.00</td><td>75.63</td><td>82.42</td></tr><tr><td>hungarian_doc_qa</td><td>34.28</td><td>57.84</td><td>50.44</td><td>68.83</td><td>55.25</td><td>74.64</td><td>75.56</td></tr><tr><td>arabic_chartqa_ar</td><td>9.32</td><td>8.63</td><td>6.62</td><td>26.92</td><td>49.35</td><td>62.16</td><td>66.64</td></tr><tr><td>arabic_infographicsvqa_ar</td><td>13.26</td><td>13.43</td><td>50.36</td><td>34.76</td><td>71.72</td><td>85.38</td><td>93.21</td></tr><tr><td>owid_charts_en</td><td>66.19</td><td>62.10</td><td>57.71</td><td>78.17</td><td>84.26</td><td>92.06</td><td>92.29</td></tr><tr><td>arxivqa</td><td>56.73</td><td>54.41</td><td>83.41</td><td>92.54</td><td>93.33</td><td>95.44</td><td>95.44</td></tr><tr><td>docvqa</td><td>81.11</td><td>50.81</td><td>45.29</td><td>90.38</td><td>86.28</td><td>83.06</td><td>92.98</td></tr><tr><td>shiftproject</td><td>62.42</td><td>70.25</td><td>31.85</td><td>75.18</td><td>78.54</td><td>82.55</td><td>91.13</td></tr><tr><td>docqa_artificial_intelligence</td><td>91.68</td><td>82.98</td><td>66.52</td><td>96.09</td><td>97.52</td><td>96.43</td><td>98.04</td></tr><tr><td>docqa_energy</td><td>89.97</td><td>76.97</td><td>65.56</td><td>96.03</td><td>90.08</td><td>88.66</td><td>96.28</td></tr><tr><td>docqa_gov_report</td><td>87.20</td><td>82.72</td><td>68.84</td><td>92.92</td><td>94.19</td><td>92.03</td><td>95.97</td></tr><tr><td>docqa_healthcare_industry</td><td>86.44</td><td>86.88</td><td>68.13</td><td>93.14</td><td>96.14</td><td>94.62</td><td>97.51</td></tr><tr><td>tabfquad</td><td>45.67</td><td>80.49</td><td>47.04</td><td>89.18</td><td>92.38</td><td>95.57</td><td>95.38</td></tr><tr><td>mpmqa_small</td><td>85.54</td><td>67.39</td><td>59.72</td><td>88.88</td><td>81.62</td><td>80.44</td><td>91.28</td></tr><tr><td>jina_2024_yearly_book</td><td>87.67</td><td>85.98</td><td>77.12</td><td>95.77</td><td>93.39</td><td>94.29</td><td>98.17</td></tr><tr><td>wikimedia-commons-maps</td><td>5.37</td><td>5.06</td><td>20.67</td><td>27.46</td><td>33.06</td><td>40.23</td><td>53.45</td></tr><tr><td>plotqa</td><td>61.13</td><td>51.44</td><td>24.05</td><td>70.58</td><td>75.99</td><td>77.48</td><td>78.75</td></tr><tr><td>MMTab</td><td>74.82</td><td>74.06</td><td>44.54</td><td>84.66</td><td>86.04</td><td>86.08</td><td>90.03</td></tr><tr><td>CharXiv-en</td><td>46.85</td><td>41.47</td><td>56.28</td><td>79.64</td><td>83.86</td><td>83.00</td><td>87.66</td></tr><tr><td>student-enrollment</td><td>1.05</td><td>1.30</td><td>0.70</td><td>3.95</td><td>4.09</td><td>8.04</td><td>11.55</td></tr><tr><td>tatqa</td><td>75.62</td><td>49.88</td><td>44.23</td><td>82.57</td><td>80.97</td><td>80.14</td><td>92.76</td></tr><tr><td>shanghai_master_plan</td><td>12.69</td><td>92.67</td><td>75.28</td><td>88.87</td><td>92.56</td><td>95.53</td><td>97.41</td></tr><tr><td>europeana-fr-news</td><td>24.55</td><td>23.69</td><td>16.43</td><td>30.33</td><td>38.23</td><td>36.66</td><td>50.16</td></tr><tr><td>infovqa</td><td>73.61</td><td>75.09</td><td>63.38</td><td>87.53</td><td>92.64</td><td>92.16</td><td>96.69</td></tr></table>

Models: $ { \mathrm { b m } } 2 5 { \mathrm { + O C R } }$ : BM25 with EasyOCR, jev3+OCR: jina-embeddings-v3 with EasyOCR, j-clip-v2: jina-clip-v2, colpali-v1.2: ColPALI-v1.2, dse-qwen2- 2b-mrl-v1: DSE-QWen2-2b-MRL-V1, je4-single: jina-embeddings-v4 single-vector, jev4-multi: jina-embeddings-v4 multi-vector

Table A3: Retrieval performance on ViDoRe $( \mathrm { n D C G } @ 1 0 \% )$ .   

<table><tr><td>Model</td><td>Avg</td><td>AQA</td><td>DVQA</td><td>InfoVQA</td><td>Shift</td><td>AI</td><td>Energy</td><td>Gov</td><td></td><td>Health TabFQ</td><td>TQA</td></tr><tr><td>OCR + jina-embeddings-v3</td><td></td><td>26.02 26.31</td><td>12.62</td><td>32.79</td><td>14.18</td><td>22.84</td><td>27.47</td><td>31.16</td><td>45.78</td><td>44.54</td><td>2.53</td></tr><tr><td>jina-clip-v2</td><td>53.61</td><td>68.33</td><td>27.62</td><td>60.6</td><td>34.12</td><td>66.55</td><td>64.69</td><td>67.47</td><td>68.38</td><td>46.89</td><td>31.43</td></tr><tr><td>voyage-multimodal-3</td><td></td><td>84.20 84.90</td><td>55.60</td><td>85.40</td><td>78.70</td><td>94.50</td><td>89.50</td><td>96.00</td><td>95.10</td><td>92.80</td><td>69.90</td></tr><tr><td>colpali-v1.2</td><td>83.90</td><td>78.00</td><td>57.20</td><td>82.80</td><td>79.10</td><td>98.10</td><td>95.20</td><td>94.80</td><td>96.70</td><td>89.70</td><td>68.10</td></tr><tr><td>dse-qwen2-2b-mrl-v1</td><td>85.80</td><td>85.60</td><td>57.10</td><td>88.10</td><td>82.00</td><td>97.50</td><td>92.90</td><td>96.00</td><td>96.40</td><td>93.10</td><td>69.40</td></tr><tr><td>OCR +bm25</td><td></td><td>65.50 31.60</td><td>36.80</td><td>62.90</td><td>64.30</td><td>92.80</td><td>85.90</td><td>83.90</td><td>87.20</td><td>46.50</td><td>62.70</td></tr><tr><td>siglip-so400m-patch14-384 51.40 43.20</td><td></td><td></td><td>30.30</td><td>64.10</td><td>18.70</td><td>62.50</td><td>65.70</td><td>66.10</td><td>79.10</td><td>58.10</td><td>26.20</td></tr><tr><td>jina-embeddings-v4 (dense)</td><td></td><td>84.11 83.57</td><td>50.54</td><td>87.85</td><td>84.07</td><td>97.16</td><td>91.66</td><td>91.48</td><td>94.92</td><td>94.48</td><td>65.35</td></tr><tr><td>jina-embeddings-v4 (late)</td><td></td><td>90.17 88.95</td><td>59.98</td><td>93.57</td><td>92.35</td><td>99.26</td><td>96.76</td><td>96.95</td><td>98.39</td><td>95.13</td><td>80.34</td></tr></table>

Tasks: Avg: Mean nDCG $@ 1 0 \%$ over all tasks, AQA: ArxivQA, Shift: Shift Project, DVQA: DocVQA, InfoVQA: InfographicVQA, AI: Artificial Intelligence, Gov: Government Reports, Health: Healthcare Industry, TabFQ: TabFQuad, TQA: TAT-DQA

Table A4: Wikimedia Commons Retreival Benchmark Results   

<table><tr><td>Language</td><td>bm25 +OCR</td><td>jev3 + OCR</td><td>j-clip- v2</td><td>colpali- v1.2</td><td>dse- qwen2- 2b-mrl- v1</td><td>jev4- single</td><td>jev4- multi</td></tr><tr><td>Average</td><td>21.99</td><td>37.43</td><td>48.63</td><td>33.60</td><td>58.67</td><td>66.04</td><td>75.63</td></tr><tr><td>Arabic (ar)</td><td>19.62</td><td>38.40</td><td>45.85</td><td>28.40</td><td>63.06</td><td>71.41</td><td>81.81</td></tr><tr><td>Bengali (bn)</td><td>22.93</td><td>44.55</td><td>49.37</td><td>26.63</td><td>52.89</td><td>66.98</td><td>76.41</td></tr><tr><td>German (de)</td><td>12.74</td><td>39.58</td><td>52.87</td><td>40.36</td><td>62.99</td><td>70.21</td><td>80.86</td></tr><tr><td>English (en)</td><td>36.45</td><td>45.24</td><td>56.58</td><td>64.98</td><td>70.23</td><td>73.55</td><td>81.66</td></tr><tr><td>Spanish (es)</td><td>12.75</td><td>46.10</td><td>54.85</td><td>41.34</td><td>66.43</td><td>71.68</td><td>80.82</td></tr><tr><td>French (fr)</td><td>15.59</td><td>36.06</td><td>35.73</td><td>43.93</td><td>41.32</td><td>53.58</td><td>59.42</td></tr><tr><td>Hindi (hi)</td><td>16.73</td><td>36.94</td><td>48.42</td><td>18.02</td><td>50.94</td><td>62.64</td><td>71.77</td></tr><tr><td>Hungarian (hu)</td><td>25.38</td><td>33.88</td><td>44.42</td><td>12.67</td><td>52.35</td><td>65.86</td><td>76.00</td></tr><tr><td>Indonesian (id)</td><td>28.79</td><td>39.48</td><td>50.85</td><td>40.46</td><td>62.03</td><td>66.02</td><td>73.72</td></tr><tr><td>Italian (it)</td><td>19.63</td><td>37.98</td><td>49.77</td><td>34.76</td><td>60.05</td><td>63.96</td><td>73.68</td></tr><tr><td>Japanese (jp)</td><td>21.41</td><td>30.43</td><td>44.03</td><td>28.83</td><td>63.71</td><td>66.50</td><td>77.13</td></tr><tr><td>Korean (ko)</td><td>34.98</td><td>35.24</td><td>47.61</td><td>29.82</td><td>68.37</td><td>71.45</td><td>81.77</td></tr><tr><td>Burmese (my)</td><td>22.84</td><td>29.45</td><td>54.36</td><td>10.28</td><td>37.61</td><td>56.58</td><td>65.01</td></tr><tr><td>Dutch (nl)</td><td>14.90</td><td>39.89</td><td>50.40</td><td>52.29 51.30</td><td>65.09 67.53</td><td>68.58</td><td>78.94</td></tr><tr><td>Portuguese (pt)</td><td>23.32 16.82</td><td>45.85</td><td>54.28</td><td>31.88</td><td></td><td>69.04</td><td>78.85</td></tr><tr><td>Russian (ru)</td><td></td><td>38.95 29.64</td><td>49.34 46.25</td><td>39.13</td><td>64.44 56.41</td><td>68.86</td><td>80.70</td></tr><tr><td>Thai (th)</td><td>30.00 13.64</td><td>32.73</td><td>36.52</td><td>9.45</td><td>38.76</td><td>61.68 49.76</td><td>71.02</td></tr><tr><td>Urdu (ur)</td><td>32.40</td><td>39.80</td><td>54.59</td><td>43.72</td><td>64.62</td><td>73.30</td><td>62.17 80.24</td></tr><tr><td>Vietnamese (vi) Chinese (zh)</td><td>18.82</td><td>28.41</td><td>46.45</td><td>23.82</td><td>64.51</td><td>69.23</td><td>80.58</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table A5: GitHub Readme Retrieval Benchmark Results   

<table><tr><td>Language</td><td>bm25 +OCR</td><td>jev3 + OCR</td><td>j-clip- v2</td><td>colpali- v1.2</td><td>dse- qwen2- 2b-mrl- v1</td><td>jev4- single</td><td>jev4- multi</td></tr><tr><td>Average</td><td>50.11</td><td>65.14</td><td>39.06</td><td>72.91</td><td>72.24</td><td>85.57</td><td>85.69</td></tr><tr><td>Arabic (ar)</td><td>27.49</td><td>27.98</td><td>31.02</td><td>55.19</td><td>55.95</td><td>75.02</td><td>75.26</td></tr><tr><td>Bengali (bn)</td><td>1.29</td><td>28.27</td><td>26.96</td><td>49.25</td><td>47.30</td><td>65.70</td><td>66.08</td></tr><tr><td>German (de)</td><td>60.11</td><td>84.58</td><td>45.46</td><td>84.15</td><td>80.62</td><td>91.09</td><td>91.35</td></tr><tr><td>English (en)</td><td>87.43</td><td>91.67</td><td>48.69</td><td>91.10</td><td>90.69</td><td>96.94</td><td>97.34</td></tr><tr><td>Spanish (es)</td><td>78.57</td><td>83.31</td><td>43.35</td><td>84.02</td><td>78.70</td><td>89.60</td><td>90.19</td></tr><tr><td>French (fr)</td><td>77.55</td><td>83.54</td><td>42.42</td><td>83.73</td><td>79.11</td><td>90.25</td><td>90.45</td></tr><tr><td>Hindi (hi)</td><td>2.72</td><td>48.08</td><td>28.55</td><td>51.22</td><td>46.49</td><td>69.31</td><td>70.98</td></tr><tr><td>Indonesian (id)</td><td>78.05</td><td>82.46</td><td>38.59</td><td>79.67</td><td>74.57</td><td>88.42</td><td>88.62</td></tr><tr><td>Italian (it)</td><td>78.83</td><td>86.54</td><td>44.26</td><td>85.31</td><td>80.81</td><td>91.76</td><td>91.41</td></tr><tr><td>Japanese (jp)</td><td>14.46</td><td>63.20</td><td>42.02</td><td>69.02</td><td>75.42</td><td>89.74</td><td>90.80</td></tr><tr><td>Korean (ko)</td><td>40.01</td><td>35.23</td><td>37.87</td><td>64.16</td><td>68.83</td><td>87.04</td><td>86.89</td></tr><tr><td>Dutch (nl)</td><td>76.52</td><td>86.36</td><td>43.25</td><td>84.10</td><td>82.85</td><td>92.83</td><td>91.37</td></tr><tr><td>Portuguese (pt)</td><td>80.33</td><td>84.46</td><td>43.88</td><td>85.00</td><td>80.09</td><td>91.43</td><td>91.47</td></tr><tr><td>Russian (ru)</td><td>39.78</td><td>50.86</td><td>37.04</td><td>78.16</td><td>78.92</td><td>89.51</td><td>88.61</td></tr><tr><td>Thai (th)</td><td>1.47</td><td>36.67</td><td>37.62</td><td>65.29</td><td>65.45</td><td>77.61</td><td>76.67</td></tr><tr><td>Vietnamese (vi)</td><td>66.70</td><td>79.67</td><td>37.14</td><td>70.05</td><td>68.20</td><td>86.90</td><td>86.94</td></tr><tr><td>Chinese (zh)</td><td>40.52</td><td>54.53</td><td>35.89</td><td>60.05</td><td>74.05</td><td>81.44</td><td>82.26</td></tr></table>

Table A6: Tweet Stock Retrieval Benchmark Results   

<table><tr><td>Language</td><td>bm25 +OCR</td><td>jev3 + OCR</td><td>j-clip- v2</td><td>colpali- v1.2</td><td>dse- qwen2- 2b-mrl-</td><td>jev4- single</td><td>jev4- multi</td></tr><tr><td>Average</td><td>22.30</td><td>42.77</td><td>55.36</td><td>76.36</td><td>62.76</td><td>78.10</td><td>85.34</td></tr><tr><td>Arabic (ar)</td><td>0.38</td><td>1.67</td><td>49.36</td><td>77.31</td><td>52.73</td><td>66.15</td><td>77.66</td></tr><tr><td>German (de)</td><td>48.27</td><td>66.86</td><td>52.49</td><td>73.53</td><td>57.35</td><td>79.38</td><td>85.63</td></tr><tr><td>English (en)</td><td>51.38</td><td>63.66</td><td>48.35</td><td>77.13</td><td>63.47</td><td>77.92</td><td>85.36</td></tr><tr><td>Spanish (es)</td><td>54.28</td><td>63.44</td><td>53.44</td><td>79.02</td><td>62.57</td><td>78.68</td><td>84.62</td></tr><tr><td>French (fr)</td><td>51.69</td><td>64.76</td><td>54.94</td><td>76.91</td><td>62.17</td><td>78.65</td><td>85.27</td></tr><tr><td>Hindi (hi)</td><td>0.08</td><td>0.08</td><td>88.55</td><td>93.39</td><td>97.00</td><td>97.46</td><td>96.50</td></tr><tr><td>Hungarian (hu)</td><td>15.55</td><td>62.31</td><td>52.30</td><td>71.06</td><td>58.17</td><td>80.09</td><td>85.01</td></tr><tr><td>Japanese (jp)</td><td>0.40</td><td>47.80</td><td>54.74</td><td>70.00</td><td>57.76</td><td>77.04</td><td>85.67</td></tr><tr><td>Russian (ru)</td><td>0.47</td><td>3.07</td><td>47.08</td><td>70.72</td><td>57.43</td><td>76.33</td><td>83.11</td></tr><tr><td>Chinese (zh)</td><td>0.45</td><td>54.04</td><td>52.30</td><td>74.54</td><td>58.94</td><td>69.33</td><td>84.55</td></tr></table>

Table A7: AirBnB Retrieval Benchmark Results   

<table><tr><td>Language</td><td>bm25 +OCR</td><td>jev3 + OCR</td><td>j-clip- v2</td><td>colpali- v1.2</td><td>dse- qwen2- 2b-mrl-</td><td>jev4- single</td><td>jev4- multi</td></tr><tr><td>Average</td><td>7.20</td><td>1.13</td><td>2.13</td><td>10.42</td><td>11.10</td><td>8.18</td><td>37.51</td></tr><tr><td>Arabic (ar)</td><td>1.10</td><td>0.40</td><td>0.47</td><td>3.06</td><td>3.64</td><td>2.20</td><td>6.20</td></tr><tr><td>German (de)</td><td>4.03</td><td>0.71</td><td>5.54</td><td>20.17</td><td>15.09</td><td>9.27</td><td>41.94</td></tr><tr><td>English (en)</td><td>48.39</td><td>1.70</td><td>4.83</td><td>23.26</td><td>12.94</td><td>13.33</td><td>64.17</td></tr><tr><td>Spanish (es)</td><td>6.25</td><td>0.18</td><td>2.10</td><td>18.06</td><td>8.61</td><td>9.11</td><td>39.84</td></tr><tr><td>French (fr)</td><td>3.86</td><td>2.00</td><td>2.05</td><td>10.86</td><td>11.87</td><td>8.70</td><td>30.55</td></tr><tr><td>Hindi (hi)</td><td>0.16</td><td>0.86</td><td>0.82</td><td>3.19</td><td>4.93</td><td>4.05</td><td>17.44</td></tr><tr><td>Hungarian (hu)</td><td>5.58</td><td>0.69</td><td>3.01</td><td>7.34</td><td>11.10</td><td>6.69</td><td>27.30</td></tr><tr><td>Japanese (jp)</td><td>0.36</td><td>1.53</td><td>0.54</td><td>3.44</td><td>14.91</td><td>7.63</td><td>45.65</td></tr><tr><td>Russian (ru)</td><td>1.67</td><td>1.39</td><td>0.88</td><td>13.16</td><td>13.61</td><td>8.66</td><td>40.80</td></tr><tr><td>Chinese (zh)</td><td>0.58</td><td>1.84</td><td>1.04</td><td>1.62</td><td>14.28</td><td>12.14</td><td>61.19</td></tr></table>

Table A8: Cross-modal (Text-to-image) retrieval performance (Recall $60 5 \%$ ) on the CLIP benchmark.   

<table><tr><td>Model</td><td>Avg</td><td>flickr30k</td><td>mscoco_captions</td><td>crossmodal3600</td><td> xtd10</td></tr><tr><td>nllb-clip-large-siglip</td><td>83.19</td><td>92.24</td><td>70.84</td><td>82.07</td><td>87.60</td></tr><tr><td> jina-clip-v2</td><td>81.12</td><td>89.84</td><td>68.35</td><td>81.43</td><td>84.87</td></tr><tr><td> jina-embeddings-v4</td><td>84.11</td><td>91.36</td><td>76.18</td><td>79.42</td><td>89.46</td></tr></table>

Avg: Mean Recall $60 \%$ over all 4 tasks.

Table A9: Text-to-image retrieval performance (Recall $@ 5 \%$ ) on xtd10 for all supported languages.   

<table><tr><td>Language</td><td>jina-embeddings-v4</td><td> jina-clip-v2</td><td>nllb-clip-large-siglip</td></tr><tr><td>average</td><td>89.46</td><td>84.87</td><td>87.60</td></tr><tr><td>de</td><td>92.10</td><td>85.70</td><td>88.30</td></tr><tr><td>en</td><td>93.10</td><td>89.40</td><td>89.40</td></tr><tr><td>es</td><td>91.50</td><td>85.90</td><td>88.20</td></tr><tr><td>fr</td><td>91.30</td><td>85.10</td><td>87.70</td></tr><tr><td>it</td><td>92.20</td><td>85.80</td><td>89.30</td></tr><tr><td>ko</td><td>86.30</td><td>82.10</td><td>85.20</td></tr><tr><td>pl</td><td>89.10</td><td>86.50</td><td>89.40</td></tr><tr><td>ru</td><td>91.50</td><td>81.10</td><td>83.40</td></tr><tr><td>tr</td><td>84.70</td><td>83.70</td><td>88.30</td></tr><tr><td>zh</td><td>82.80</td><td>83.40</td><td>86.80</td></tr></table>

Table A10: Text-to-image retrieval performance (Recall $@5 \%$ ) on crossmodal3600 for all supported languages.   

<table><tr><td>Language</td><td>jina-embeddings-v4</td><td> jina-clip-v2</td><td>nllb-clip-large-siglip</td></tr><tr><td>average</td><td>79.42</td><td>81.43</td><td>82.07</td></tr><tr><td>ar</td><td>75.75</td><td>73.56</td><td>78.92</td></tr><tr><td>bn</td><td>57.97</td><td>63.78</td><td>75.19</td></tr><tr><td>da</td><td>80.47</td><td>85.39</td><td>87.14</td></tr><tr><td>de</td><td>91.75</td><td>91.25</td><td>89.56</td></tr><tr><td>el</td><td>66.50</td><td>75.03</td><td>77.83</td></tr><tr><td>en</td><td>76.47</td><td>75.83</td><td>73.11</td></tr><tr><td>es</td><td>83.64</td><td>83.64</td><td>82.64</td></tr><tr><td>f</td><td>66.67</td><td>82.83</td><td>86.42</td></tr><tr><td>fr</td><td>88.69</td><td>88.78</td><td>87.86</td></tr><tr><td>hi</td><td>47.81</td><td>55.25</td><td>60.31</td></tr><tr><td>id</td><td>87.41</td><td>84.22</td><td>86.31</td></tr><tr><td>it</td><td>87.97</td><td>88.33</td><td>85.94</td></tr><tr><td>ja</td><td>91.22</td><td>87.03</td><td>86.06</td></tr><tr><td>ko</td><td>82.19</td><td>78.81</td><td>78.75</td></tr><tr><td>nl</td><td>81.00</td><td>82.56</td><td>81.69</td></tr><tr><td>no</td><td>71.94</td><td>81.08</td><td>82.69</td></tr><tr><td>pl</td><td>80.86</td><td>84.00</td><td>82.72</td></tr><tr><td>pt</td><td>81.42</td><td>82.42</td><td>82.69</td></tr><tr><td>ro</td><td>84.33</td><td>89.36</td><td>90.03</td></tr><tr><td>ru</td><td>90.28</td><td>88.97</td><td>86.44</td></tr><tr><td>SV</td><td>72.58</td><td>78.06</td><td>79.33</td></tr><tr><td>th</td><td>83.36</td><td>81.61</td><td>81.14</td></tr><tr><td>tr</td><td>73.08</td><td>81.31</td><td>83.47</td></tr><tr><td>uk</td><td>86.28</td><td>88.56</td><td>85.44</td></tr><tr><td>vi</td><td>88.81</td><td>86.64</td><td>85.56</td></tr><tr><td>zh</td><td>86.67</td><td>78.97</td><td>76.56</td></tr></table>

# A.4 MTEB and MMTEB

Table A11: Evaluation Results for Various Models on MTEB Retrieval Tasks $( \mathrm { n D C G } @ 1 0 \% )$   

<table><tr><td>Model</td><td>Arg</td><td>CQG</td><td>CQU</td><td>CFHN</td><td>FEV</td><td>FiQA</td><td>HPQA</td><td>SCI</td><td>TREC</td><td>TOU</td><td>AVG</td></tr><tr><td>multilingual-e5-large</td><td>54.36</td><td>58.70</td><td>39.89</td><td>26.00</td><td>83.79</td><td>43.82</td><td>70.55</td><td>17.45</td><td>71.15</td><td>49.59</td><td>51.53</td></tr><tr><td>e5-mistral-7b-instruct</td><td>61.65</td><td>63.52</td><td>46.75</td><td>28.50</td><td>86.99</td><td>56.81</td><td>73.21</td><td>16.32</td><td>87.03</td><td>55.44</td><td>57.62</td></tr><tr><td>text-embedding-3-large</td><td>57.99</td><td>65.40</td><td>50.02</td><td>30.10</td><td>88.53</td><td>55.00</td><td>71.66</td><td>23.07</td><td>79.56</td><td>58.42</td><td>57.98</td></tr><tr><td>gemini-embedding-001</td><td>86.44</td><td>70.68</td><td>53.69</td><td>31.06</td><td>88.98</td><td>61.78</td><td>87.01</td><td>25.15</td><td>86.32</td><td>52.39</td><td>64.35</td></tr><tr><td> jina-embedding-l-en-v1</td><td>48.3</td><td>51.68</td><td>38.66</td><td>25.93</td><td>71.16</td><td>41.02</td><td>57.26</td><td>18.54</td><td>60.34</td><td>62.34</td><td>47.52</td></tr><tr><td>jina-embeddings-v2-base-en</td><td>44.18</td><td>56.52</td><td>38.66</td><td>23.77</td><td>73.41</td><td>41.58</td><td>63.24</td><td>19.86</td><td>65.91</td><td>63.35</td><td>49.05</td></tr><tr><td>jina-embeddings-v3+</td><td>54.33</td><td>58.02</td><td>43.52</td><td>43.14</td><td>89.90</td><td>47.35</td><td>64.70</td><td>19.92</td><td>77.74</td><td>55.28</td><td>55.39</td></tr><tr><td> jina-embeddings-v4+</td><td>67.07</td><td>57.59</td><td>42.95</td><td>34.57</td><td>87.16</td><td>46.51</td><td>69.01</td><td>21.47</td><td>80.36</td><td>52.41</td><td>55.91</td></tr></table>

†using the text-matching adapter

Table A12: Evaluation Results for Various Models on MMTEB Retrieval Tasks (nDCG $@ 1 0 \%$ )  

<table><tr><td>Model</td><td>Avg</td><td>AI</td><td></td><td>Arg Bel</td><td>Cov Hag PK</td><td></td><td>LB MIR ML</td><td></td><td></td><td>SD SQA</td><td></td><td>so</td><td>TC</td><td>STC TR TW Wiki WG</td><td></td><td></td><td></td></tr><tr><td> jina-embeddings-v3</td><td></td><td></td><td></td><td></td><td>58.6 32.8 54.3 73.4 78.6 98.7 38.0 93.4</td><td></td><td></td><td>：62.6 73.4 19.8</td><td></td><td>0.7</td><td></td><td>90.8 77.7</td><td>39.2</td><td>0.6</td><td>73.0</td><td>89.1</td><td>18.6</td></tr><tr><td> jina-embeddings-v4</td><td></td><td></td><td>66.5 50.2 67.1</td><td></td><td>174.380.298.8</td><td>69.894.8</td><td>61.2</td><td>74.9 21.5</td><td></td><td>30.2</td><td>91.9</td><td>80.4</td><td></td><td>59.5</td><td>1.3 84.4</td><td>88.5</td><td>67.3</td></tr><tr><td>bge-m3</td><td></td><td></td><td></td><td>55.4 29.0 54.0 78.21</td><td>77.5 98.8</td><td>59.0 90.3</td><td>69.6</td><td>74.8</td><td>16.3</td><td>7.5</td><td>80.6</td><td>54.9</td><td>21.9</td><td>1.0</td><td>37.8</td><td>89.9</td><td>41.7</td></tr><tr><td>Cohere-embed-mult.-v3 59.2 29.7 55.1 81.1 77.1</td><td></td><td></td><td></td><td></td><td></td><td>98.8 38.2 93.8</td><td>68.0</td><td>76.1</td><td>19.3</td><td>4.7</td><td></td><td>89.4 83.4</td><td>24.2</td><td></td><td>0.9 75.8</td><td>90.9</td><td>58.4</td></tr><tr><td>gemini-embedding-001</td><td></td><td></td><td></td><td>68.1 48.8 86.4 90.7 79.1</td><td></td><td>99.3 38.5 96.0</td><td>70.4</td><td>84.2 25.2</td><td></td><td>10.3</td><td>96.7</td><td>86.3</td><td>51.1</td><td></td><td>3.0 98.0</td><td>94.2</td><td>60.5</td></tr><tr><td>text-embedding-3-large</td><td></td><td></td><td></td><td>61.1 42.0 58.0 68.8（</td><td>68.4 99.1</td><td>69.8 95.2</td><td>56.9</td><td>73.2 23.1</td><td></td><td>7.4</td><td>92.4</td><td>79.6</td><td>31.1</td><td>2.1</td><td>81.4</td><td>89.2</td><td>29.1</td></tr><tr><td>voyage-3</td><td></td><td></td><td></td><td></td><td>66.0 42.5 61.0 76.5 88.5 98.6 94.8 94.5 57.7</td><td></td><td></td><td>75.7 21.4 10.7</td><td></td><td></td><td></td><td>94.3 80.5 49.2</td><td></td><td></td><td>1.2 85.7</td><td>89.7</td><td>67.7</td></tr><tr><td>voyage-multilingual-2</td><td>1</td><td></td><td>45.0 61.8</td><td>1</td><td>1</td><td>98.9 97.0 95.9</td><td>1</td><td>1</td><td>22.510.2</td><td></td><td>1</td><td>80.1</td><td>1</td><td></td><td>1.4 87.3</td><td>1</td><td>39.1</td></tr></table>

Tasks: Arg: ArguAna, CQG: CQADupstackGamingRetrieval, CQU: CQADupstackUnixRetrieval, CFHN: ClimateFEVERHardNegatives, FEV: FEVERHardNegatives, FiQA: FiQA2018, HPQA: HotpotQAHardNegatives, SCI: SCIDOCS, TREC: TRECCOVID, TOU: Touche2020Retrieval.v3   
Tasks: Avg: Mean nDCG $@ 1 0 \%$ for all tasks, AI: AILAStatutes, Arg: ArguAna, Bel: BelebeleRetrieval, Cov: CovidRetrieval, Hag: HagridRetrieval, PK: LEMBPasskeyRetrieval, LB: LegalBenchCorporateLobbying, MIR: MIRACLRetrievalHardNegatives, ML: MLQARetrieval, SD: SCIDOCS, SQA: SpartQA, SO: StackOverflowQA, TC: TREC-COVID, STC: StatcanDialogueDatasetRetrieval, TR: TempReasonL1, TW: TwitterHjerneRetrieval, Wiki: WikipediaRetrievalMultilingual, WG: WinoGrande

Table A13: Retrieval performance on MTEB LongEmbed $( \mathrm { n D C G } @ 1 0 \% )$   

<table><tr><td>Model</td><td>Avg</td><td>NaQA</td><td>Needle</td><td>Passkey</td><td>QMSum</td><td>SummScreen</td><td>Wikim</td></tr><tr><td> jina-embeddings-v3</td><td>55.66</td><td>34.30</td><td>64.00</td><td>38.00</td><td>39.34</td><td>92.33</td><td>66.02</td></tr><tr><td>jina-embeddings-v4</td><td>67.11</td><td>57.52</td><td>51.75</td><td>65.50</td><td>46.49</td><td>96.30</td><td>85.08</td></tr><tr><td>voyage-multilingual-2</td><td>79.17</td><td>64.69</td><td>75.25</td><td>97.00</td><td>51.50</td><td>99.11</td><td>87.49</td></tr><tr><td>voyage-3</td><td>74.07</td><td>54.12</td><td>57.75</td><td>94.75</td><td>51.05</td><td>97.82</td><td>88.90</td></tr><tr><td>voyage-3-lite</td><td>71.41</td><td>51.67</td><td>54.00</td><td>84.75</td><td>53.01</td><td>96.71</td><td>88.34</td></tr><tr><td>bge-m3</td><td>58.73</td><td>45.76</td><td>40.25</td><td>59.00</td><td>35.54</td><td>94.09</td><td>77.73</td></tr><tr><td>text-embedding-3-large</td><td>52.42</td><td>44.09</td><td>29.25</td><td>69.75</td><td>32.49</td><td>84.80</td><td>54.16</td></tr><tr><td>Cohere-embed-english-v3</td><td>42.11</td><td>25.04</td><td>30.50</td><td>38.50</td><td>23.82</td><td>75.77</td><td>59.03</td></tr><tr><td>multilingual-e5-large-instruct</td><td>41.76</td><td>26.71</td><td>29.50</td><td>37.75</td><td>26.08</td><td>72.75</td><td>57.79</td></tr><tr><td>multilingual-e5-large</td><td>40.44</td><td>24.22</td><td>28.00</td><td>38.25</td><td>24.26</td><td>71.12</td><td>56.80</td></tr></table>

Tasks: Avg: Mean nDCG $@ 1 0 \%$ for all tasks, NaQA: LEMBNarrativeQARetrieval, Needle: LEMBNeedleRetrieval, Passkey: LEMBPasskeyRetrieval, QMSum: LEMBQMSumRetrieval, SummScreen: LEMBSummScreenFDRetrieval, Wikim: LEMBWikimQARetrieval

Table A14: STS performance on MTEB v2 (Spearman correlation $\%$ ).   

<table><tr><td>Model</td><td>Avg</td><td>BIO</td><td>SICK-R</td><td>STS12</td><td>STS13</td><td>STS14</td><td>STS15</td><td>STS17</td><td>STS22</td><td>STSB</td></tr><tr><td>jina-embeddings-v3</td><td>85.82</td><td>88.69</td><td>89.62</td><td>82.44</td><td>89.49</td><td>84.95</td><td>89.32</td><td>90.01</td><td>68.45</td><td>89.43</td></tr><tr><td> jina-embeddings-v4</td><td>85.89</td><td>89.21</td><td>89.23</td><td>83.50</td><td>88.61</td><td>84.77</td><td>89.69</td><td>88.71</td><td>70.71</td><td>88.58</td></tr><tr><td>BAAI/bge-m3</td><td>80.61</td><td>1</td><td>79.72</td><td>78.73</td><td>79.60</td><td>79.00</td><td>87.81</td><td>87.13</td><td>67.99</td><td>84.87</td></tr><tr><td>Cohere-embed-English-3</td><td>82.40</td><td>83.50</td><td>81.27</td><td>74.37</td><td>85.20</td><td>80.98</td><td>89.23</td><td>90.34</td><td>68.18</td><td>88.55</td></tr><tr><td>Cohere-embed-multilingual-v3</td><td>83.05</td><td>85.01</td><td>82.18</td><td>77.62</td><td>85.16</td><td>80.02</td><td>88.92</td><td>90.09</td><td>69.63</td><td>88.79</td></tr><tr><td>gemini-embedding-001</td><td>85.29</td><td>88.97</td><td>82.75</td><td>81.55</td><td>89.89</td><td>85.41</td><td>90.44</td><td>91.61</td><td>67.97</td><td>89.08</td></tr><tr><td>multilingual-e5-large</td><td>81.39</td><td>84.57</td><td>80.23</td><td>80.02</td><td>81.55</td><td>77.72</td><td>89.31</td><td>88.12</td><td>63.66</td><td>87.29</td></tr><tr><td>text-embedding-3-large</td><td>81.44</td><td>84.68</td><td>79.00</td><td>72.84</td><td>86.10</td><td>81.15</td><td>88.49</td><td>90.22</td><td>66.89</td><td>83.56</td></tr><tr><td>voyage-3</td><td>78.59</td><td>87.92</td><td>79.63</td><td>69.52</td><td>80.56</td><td>73.33</td><td>80.39</td><td>86.81</td><td>69.60</td><td>79.53</td></tr><tr><td> voyage-large-2</td><td>82.63</td><td>89.13</td><td>79.78</td><td>72.94</td><td>83.11</td><td>77.21</td><td>85.30</td><td>88.77</td><td>1</td><td>84.78</td></tr><tr><td>voyage-multilingual-v2</td><td>76.98</td><td>87.11</td><td>78.97</td><td>67.30</td><td>80.09</td><td>71.98</td><td>78.07</td><td>86.52</td><td>67.02</td><td>75.79</td></tr></table>

Tasks: Avg: Mean Spearman Correlation $\%$ for all tasks, BIO: BIOSSES, STS22: STS22v2, STSB: STSBenchmark

Table A15: STS performance on MMTEB v2 (Spearman correlation $\%$ ).   

<table><tr><td>Model</td><td></td><td>Avg Faro FinPara Indic JSICK SICK-R STS12 STS13 STS14 STS15 STS17 STS22 STSB STSES SemRel</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> jina-embeddings-v4</td><td>72.70 72.28</td><td>14.43</td><td>35.22</td><td>80.33</td><td>89.23</td><td>83.50</td><td>88.61</td><td>84.77</td><td>89.69</td><td>88.71</td><td>70.71</td><td>88.58 75.31</td><td>56.46</td></tr><tr><td> jina-embeddings-v3</td><td>75.77 80.82</td><td>22.38</td><td>54.66</td><td>78.16</td><td>89.62</td><td>82.44</td><td>89.49</td><td>84.94 89.31</td><td>85.94</td><td>71.14</td><td>89.44</td><td>77.87</td><td>64.58</td></tr><tr><td>bge-m3</td><td>72.99 77.80</td><td>30.43</td><td>52.13</td><td>79.21</td><td>79.72</td><td>78.73</td><td>79.60</td><td>79.00 87.81</td><td>79.65</td><td>70.03</td><td>84.87</td><td>77.50</td><td>65.38</td></tr><tr><td>Cohere-embed-mult.-v3 73.77 75.95</td><td></td><td>28.24</td><td>46.73</td><td>77.19</td><td>82.18</td><td>77.62</td><td>85.16</td><td>80.02 88.92</td><td>90.09</td><td>69.36</td><td>88.79</td><td>78.76</td><td>63.84</td></tr><tr><td>gemini-embedding-001</td><td>78.35 86.12</td><td>28.60</td><td>62.87</td><td>84.99</td><td>82.75</td><td>81.55</td><td>89.89</td><td>85.41 90.44</td><td>88.58</td><td>71.69</td><td>89.08</td><td>81.75</td><td>73.14</td></tr><tr><td>text-embedding-3-large</td><td>70.17 74.96</td><td>23.51</td><td>12.59</td><td>81.24</td><td>79.00</td><td>72.84</td><td>86.10</td><td>81.15 88.49</td><td>90.22</td><td>69.29</td><td>83.56</td><td>74.20</td><td>65.25</td></tr><tr><td>voyage-3</td><td>68.33 72.51</td><td>22.51</td><td></td><td>41.6371.76</td><td>79.63</td><td>69.52</td><td>80.56</td><td>73.33 80.39</td><td>76.24</td><td>71.88</td><td>379.53</td><td>72.51</td><td>64.66</td></tr><tr><td>voyage-multilingual-2</td><td>68.02 74.42</td><td>27.07</td><td></td><td>35.03 75.94</td><td>78.97</td><td>67.30</td><td>80.09</td><td>71.98 78.07</td><td>77.06</td><td>69.03</td><td>75.79</td><td>76.69</td><td>64.88</td></tr></table>

Tasks: Avg: Mean Spearman Correlation $\%$ for all tasks, Faro: FaroeseSTS, FinPara: FinParaSTS, Indic: IndicCrosslingualSTS, STS22: STS22v2, STSB: STSBenchmark, SemRel: SemRel24STS

Table A16: Performance on MTEB Code Information Retrieval (MTEB-CoIR) (nDCG@10%).  

<table><tr><td>Model</td><td>Avg</td><td>AppsR CCSN CodeMT CodeST CodeSN CodeTO CodeTD CosQA StackO SynSQL</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> jina-embeddings-v2-code 52.24</td><td></td><td>16.37</td><td>83.97</td><td>44.40</td><td>68.66</td><td>59.62</td><td>75.68</td><td>27.25</td><td>41.92</td><td>89.26</td><td>46.99</td></tr><tr><td>jina-embeddings-v3</td><td>55.07</td><td>29.01</td><td>1</td><td>59.67</td><td>78.14</td><td>53.18</td><td>77.37</td><td>30.91</td><td>35.34</td><td>90.79</td><td>41.27</td></tr><tr><td> jina-embeddings-v4</td><td>71.59</td><td>76.08</td><td>84.05</td><td>70.60</td><td>85.06</td><td>83.69</td><td>89.34</td><td>44.19</td><td>31.48</td><td>93.45</td><td>70.45</td></tr><tr><td>Cohere-embed-English-3 51.36</td><td></td><td>13.72</td><td>1</td><td>47.02</td><td>74.82</td><td>52.81</td><td>65.28</td><td>31.38</td><td>30.65</td><td>89.35</td><td>57.20</td></tr><tr><td>Cohere-embed-mult.-v3</td><td>54.31</td><td>31.91</td><td></td><td>42.91</td><td>74.19</td><td>57.57</td><td>70.25</td><td>30.14</td><td>32.58</td><td>89.42</td><td>59.79</td></tr><tr><td>gemini-embedding-001</td><td>73.11</td><td>93.75</td><td>81.06</td><td>56.28</td><td>85.33</td><td>84.69</td><td>89.53</td><td>31.47</td><td>50.24</td><td>96.71</td><td>69.96</td></tr><tr><td>text-embedding-3-large</td><td>62.36</td><td>28.37</td><td>1</td><td>68.92</td><td>80.42</td><td>73.18</td><td>84.25</td><td>34.23</td><td>31.00</td><td>92.44</td><td>68.45</td></tr><tr><td>voyage-3</td><td>67.23</td><td>73.03</td><td></td><td>66.69</td><td>83.02</td><td>77.87</td><td>89.92</td><td>33.92</td><td>28.70</td><td>94.34</td><td>57.56</td></tr><tr><td>voyage-code-3</td><td>77.33</td><td>93.62</td><td>89.35</td><td>93.58</td><td>90.67</td><td>90.09</td><td>94.96</td><td>38.57</td><td>34.45</td><td>97.17</td><td>62.87</td></tr></table>

Tasks: Avg: Mean nDCG $@ 1 0 \%$ for all tasks, AppsR: AppsRetrieval, COIR: COIRCodeSearchNetRetrieval, CodeMT: CodeFeedbackMT, CodeST: CodeFeedbackST, CodeSN: CodeSearchNetCCRetrieval, CodeTO: CodeTransOceanContest, CodeTD: CodeTransOceanDL, StackO: StackOverflowQA, SynSQL: SyntheticText2SQL