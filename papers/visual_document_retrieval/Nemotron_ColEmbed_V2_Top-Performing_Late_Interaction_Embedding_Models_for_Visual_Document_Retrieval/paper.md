# Nemotron ColEmbed V2: Top-Performing Late Interaction Embedding Models for Visual Document Retrieval

Gabriel de Souza P. Moreira1,†, Ronay $\mathrm { A k } ^ { 1 , \dag }$ , Mengyao ${ \mathrm { X u } } ^ { 1 }$ , Oliver Holworthy1, Benedikt Schifferer\*, Zhiding $\mathrm { Y u } ^ { 1 }$ , Yauhen Babakhin1, Radek Osmulski $^ 1$ , Jiarui Cai1, Ryan Chesler1, Bo Liu1 and Even Oldridge1

1NVIDIA

# Abstract

Retrieval-Augmented Generation (RAG) systems have been popular for generative applications, powering language models by injecting external knowledge. Companies have been trying to leverage their large catalog of documents (e.g. PDFs, presentation slides) in such RAG pipelines, whose first step is the retrieval component. Dense retrieval has been a popular approach, where embedding models are used to generate a dense representation of the user query that is closer to relevant content embeddings. More recently, VLM-based embedding models have become popular for visual document retrieval, as they preserve visual information and simplify the indexing pipeline compared to OCR text extraction.

Motivated by the growing demand for visual document retrieval, we introduce Nemotron ColEmbed V2, a family of open-weights models that achieve state-of-the-art performance on the ViDoRe benchmarks. We release three model sizes—with 3B, 4B, and 8B parameters—based on pre-trained VLMs: NVIDIA Eagle 2 with Llama 3.2 3B backbone, Qwen3-VL-4B-Instruct and Qwen3-VL-8B-Instruct, respectively. The 8B model ranks first on the ViDoRe V3 leaderboard as of February 03, 2026, achieving an average $\mathrm { N D C G } @ 1 0$ of 63.42.

We describe the main techniques used across data processing, training, and post-training—such as clusterbased sampling, hard-negative mining, bidirectional attention, late interaction, and model merging—that helped us build our top-performing models. We also discuss compute and storage engineering challenges posed by the late interaction mechanism and present experiments on how to balance accuracy and storage with lower dimensional embeddings.

# Keywords

Visual Document Retrieval, Late interaction, Visual Language Model, RAG, Dense Retrieval, ViDoRe

# 1. Introduction

Retrieval-Augmented Generation (RAG) has become a widely adopted paradigm for enhancing language models generation with external knowledge, enabling them to retrieve and reason on relevant content from large-scale corpora. Numerous high-performing text retrieval models including NV-Embed [1], NV-Retriever [2], Qwen3-Embedding [3], and e5-mistral [4] have been proposed, and evaluated on benchmarks such as MTEB [5, 6] text benchmarks, that assume clean and well-formatted textual inputs. In contrast, real-world use cases typically involve documents stored in formats like PDFs, PowerPoint slides, or Word documents, requiring preprocessing pipelines to extract textual content (e.g. text parsing, OCR). This process often results in the loss of critical visual information for modalities like tables, charts, and infographics. To address those limitations, Visual Document Retrieval (VDR) [7] has been proposed to retrieve document pages directly from their image, with no need for text extraction, further preserving visual information and simplifying the indexing and search pipelines from document retrieval systems. If page text is easily available, both page image and text can be used for better page multimodal representation and retrieval.

Recent Vision-Language models (VLMs) aim to bridge the gap between text and image understanding by learning joint representations across modalities. Models such as Qwen-VL [8], LLaMA-3.1-Nemotron-Nano-VL [9], PaliGemma 2 [10], NVIDIA’s Eagle 2 [11, 12] and Nemotron Nano V2 VL [13] have demonstrated strong performance across a range of vision-language tasks. VLMs use image encoders like CLIP [14], SigLIP [15] and C-RADIO [16] to extract image features and project them to the LLM space. VLM decoder models have been adapted as contrastive embedding models for visual document retrieval, like Jina CLIP [17] and Nomic Embed Vision [18].

ColBERT [19] demonstrated that a multi-vector late interaction approach could boost retrieval accuracy, for allowing deeper interaction between query and textual context tokens, compared to the pooling approaches (e.g., average, last) that compress the representation to a single-vector embedding. ColPali [7] VLM embedding model leveraged late interaction between text-image tokens for improved retrieval accuracy.

In order to evaluate visual document retrieval models, several benchmarks were introduced. The most popular ones belong to the ViDoRe, which was released in three versions: V1 [7], V2 [20] and V3 [21]. The latest Vidore V3 [21] is an important expansion for evaluating VDR on complex real-world scenarios, including multi-type and multi-language queries across ten professional domains.

In this paper, we introduce the Nemotron ColEmbed V2, a family of state-of-the-art embedding models for visual document retrieval. Our best-performing model nemotron-colembed-vl-8b-v2 achieves an $\mathrm { N D C G } @ 1 0$ of 63.42 on the Vidore V3 benchmark $( + 3 \%$ to the second place), ranking first on that benchmark as of Feb. 03, 2026.

We initialized our 3B model from NVIDIA’s Eagle 2 vision-language model [11, 12] with Llama 3.2 3B LLM backbone and initialize our 4B and 8B models from Qwen3-VL models [22]. We also replaced the original causal attention with bidirectional attention, and fine-tuned the models through contrastive training with late interaction mechanism on curated datasets for visual document retrieval. Our training datasets contain both text-only and text-image examples for contrastive learning, and we apply hard negative mining following the methods proposed in NV-Retriever [2] to improve retrieval accuracy. Finally, we use model merging to provide ensemble-level accuracy in a single model.

The main contributions of this paper are:

• We release three state-of-the-art models for visual document retrieval: llama-nemotron-colembed-$\nu l - 3 b \ – \nu 2 ^ { 1 }$ , nemotron-colembed-vl-4b- $\cdot \nu 2 ^ { 2 }$ , and nemotron-colembed-vl-8b- ${ \cdot \nu } 2 ^ { 3 }$ . The 8B model achieves top-1 performance in the MTEB ViDoRe V3 leaderboard, while the 4B and 3B models are among top-6, outperforming the models of the same size in the leaderboard on NDCG@10. • We describe the techniques that helped boost our model accuracy to the top of ViDoRe leaderboards regarding data preprocessing (clustering-based sampling, hard-negative mining, crosslingual translation), two-stage training, late interaction, and model merging. • Finally, we discuss some performance trade-offs of using late interaction mechanism, as its higher accuracy comes at the price of increasing indexing embeddings storage and higher compute at serving compared to simpler vector pooling. We provide an ablation on the reduced embedding sizes for the late interaction.

# 2. Background

# 2.1. Visual Document Retrieval

Visual document retrieval has transitioned from rigid pipelines to integrated multimodal frameworks. Early systems [23, 24] relied on OCR-centric pipelines, where text was first extracted and then processed by layout-aware encoders that fused content with 2D positions. Although effective for document understanding, these early designs were computationally prohibitive for first-stage retrieval and were largely restricted to reranking small candidate sets. In parallel, early CLIP-style bi-encoders [25] offered global multimodal representations, but frequently struggled with cluttered pages where relevant information was sparse or layout-dependent.

The rapid advancement of Large VLMs has blurred the boundaries between OCR, layout detection, and visual understanding. Modern VLMs, trained on web-scale data, implicitly recognize rendered text and layout structures directly from pixels, eliminating the need for explicit OCR or handcrafted features. This shift has catalyzed a new generation of visual document retrieval models that combine VLM backbones with ColBERT-style late interaction [19]. Notable implementations include ColPali [7] (PaliGemma-based [26]), ColQwen [7] (Qwen2-based [27]), Jina Embedding Models [28] (Qwen2.5- based [29]), and NVIDIA’s Nemoretriever Colembed [30] (Eagle2-based [11]). These models project document images into a sequence of dense patch-level embeddings, enabling fine-grained, multi-vector matching against text queries. Additionally, current research in this space also actively exploring various optimization strategies to refine performance and efficiency. These include knowledge distillation from high-capacity teachers [31], quantization for reduced memory footprints [31], and model merging [3, 32, 33, 34] to ensemble the weights of diverse pre-trained models.

Our work builds upon this emerging paradigm of VLM-based late interaction. We leverage pretrained VLM models as backbones for representation extraction and employ a late interaction mechanism specifically tailored for visually rich documents.

# 2.2. Dense Retrieval

Dense retrieval methods for both text and visual documents are often categorized by how and when the query and document interact. Existing approaches follow three primary paradigms:

(1) bi-encoders: These models independently encode queries and documents into single global vectors. Relevance is determined by a simple similarity function, usually cosine similarity, as shown in Figure 1a. While highly efficient and common in early text-only retrieval [35] or CLIP/SigLIP-style VLMs [14, 15], the single-vector representation bottleneck is often limited to capture fine-grained lexical, visual, or layout-dependent cues.

(2) Cross-encoders: These architectures jointly encode queries and candidate documents, allowing for dense early interaction via self- and cross-attention over all tokens or patches. They can model intricate token–level alignments and thus excel as strong rerankers. However, it does not support pre-computing/indexing document embeddings, and the quadratic computational cost renders them impractical for first-stage retrievers across large-scale collections.

(3) Late interaction models: Bridging the gap between the two, late interaction retains some expressiveness from cross-encoders, but still allows pre-computing documents’ multi-vector embeddings, as further discussed in the next section.

# 2.3. Late Interaction

The late interaction mechanism introduced by ColBERT [19] enables fine-grained interactions between query and document tokens. As shown in Figure 1b, for a query, each token embedding interacts with all document token embeddings using a MaxSim operator, which selects the maximum similarity per query token and sums these scores to produce the final relevance score.

This requires storing all token embeddings of the document corpus (text or images). At inference time, query token embeddings are computed and interact with the stored document embeddings through MaxSim op. We adopt this mechanism in our models to enable fine-grained retrieval.

While this approach offers the expressiveness of token-level matching, compared to simpler pooling methods such as average or last-token pooling, as shown in Figure 1a, the late-interaction method introduces latency and storage overhead that may need to be assessed, as these overheads become a concern for real-world applications. To ensure scalability, we further investigate dimensionality reduction techniques, refining the trade-off between dense representation quality and storage overhead, as discussed in Section 5.

![](images/2711432170d132073512068a2518bffe5fb35a4703ee70a705e51ab6cd9ea658.jpg)  
Figure 1: Illustration of the bi-encoder and late-interaction architectures.

# 2.4. Contrastive Learning

Dense retrieval models are typically trained with contrastive learning to maximize the embedding similarity between the query and positive passage, while minimizing the similarity between the query and negative corpus. The InfoNCE contrastive loss [36] is a popular choice to train the model to distinguish between positive and negative pairs in a shared embedding space,

$$
\mathcal { L } ( q , d ^ { + } , D _ { N } ) = - \log \frac { \exp ( \sin ( q , d ^ { + } ) / \tau ) } { \sum _ { d _ { i } \in \{ d ^ { + } \} \cup D _ { N } } \exp ( \sin ( q , d _ { i } ) / \tau ) } ,
$$

where $q$ is the embedding of a query, and $d ^ { + }$ are embeddings of positive documents. $D _ { N }$ denotes the set of negative passages. $\tau$ is the temperature parameter. $s i m ( \cdot )$ represents a similarity function like cosine similarity, dot product, or late interaction similarity based on the MaxSim operation.

# 3. Nemotron ColEmbed V2: a Family of Multimodal Late Interaction Models for Visual Document Retrieval

In this section, we describe the Nemotron ColEmbed V2 models’ architectures and the main methods we used for those top performing models, listed in Table 1.

Table 1 The Nemotron ColEmbed V2 family   

<table><tr><td>Model (Huggingface ID)</td><td># Parameters (without embeddings)</td><td>Embedding Dimension</td></tr><tr><td>nvidia/llama-nemotron-colembed-3b-v2</td><td>4.40 (3.99) B</td><td>3072</td></tr><tr><td>nvidia/nemotron-colembed-vl-4b-v2</td><td>4.82 (4.43) B</td><td>2560</td></tr><tr><td>nvidia/nemotron-colembed-vl-8b-v2</td><td>8.76 (8.14) B</td><td>4096</td></tr></table>

# 3.1. llama-nemotron-colembed-vl-3b-v2 Architecture

Our llama-nemotron-colembed-vl-3b- $\cdot \nu 2$ late-interaction VLM embedding model is based on NVIDIA Eagle 2 vision-language model [11, 12]. It uses the same architecture as its first version llama-nemoretrievercolembed- $3 b \ – \nu 1 ^ { 4 }$ [30]. These models adopt dynamic image tiling to support inputs of varying resolutions, and employ a carefully curated data strategy that improves multimodal learning. These design choices enable Eagle 2 models to achieve state-of-the-art results on several multimodal benchmarks, providing a solid foundation for retrieval tasks. We initialize our model from an internal pre-trained Eagle 2 VLM that uses SigLip 2 [15] as the image encoder and Llama 3.2 3B [37] as the LLM backbone.

Regarding the dynamic tiling mechanism, the max_input_tiles parameter is used to control the number of tiles produced from each image. Each image tile generates 256 visual tokens. For training, we set max_input_tiles $\qquad = \quad 2$ (including an additional thumbnail tile from the full page) to maintain memory efficiency, as increasing it to 4 did not yield performance gains. During inference, we set max_input_tiles $\qquad = \ 8$ to allow for finer visual granularity.

![](images/1c64a11ee858010502319578aa257939ecbcb49b14fcce05c84a3614dc22b464.jpg)  
Figure 2 illustrates the dynamic image tiling, image encoding into visual tokens, and the late interaction mechanism.   
Figure 2: llama-nemotron-colembed-vl-3b-v2 architecture with dynamic image tiling and late interaction scoring mechanisms[30].

# 3.2. nemotron-colembed-vl-4/8b-v2 Architectures

Our nemotron-colembed-vl-4b-v2 and nemotron-colembed-vl-8b-v2 late-interaction embedding models are based on Qwen3-VL 4B and 8B VLMs[22], which support multimodal inputs and long context. Similarly to Eagle 2 used by llama-nemotron-colembed-vl-3b-v2, Qwen3-VL adopts a three-module architecture, comprising SigLIP-2 vision encoder[15], a two-layer MLP-based vision–language merger, and Qwen3 LLMs [38].

Their vision encoder is designed to handle dynamic, native-resolution images, mapping them to visual token sequences of variable length. To enhance perceptual capability and preserve rich visual details, Qwen3-VL extends the DeepStack mechanism by injecting visual tokens from intermediate layers of the vision encoder into multiple layers of the LLM [22].

# 3.3. Training data and Hyperparameters

The ColEmbed v2 models were trained on 500,000 samples sampled from publicly available datasets, including Vidore-ColPali-Training [7], Wiki-SS-NQ and DocMatix-IR [25], $\mathrm { \Delta V D R } ^ { 5 }$ , and VisRAG [39]. We employed some data-related techniques (hard-negative mining, cluster-based data sampling, crosslingual query translation, and two-stage training) that are described in Section 3.4.

ColEmbed v2 models were trained for one epoch, with a learning rate of 2e-6, AdamW optimizer, and weight decay of 0.1, batch size and gradient accumulation of 1, and two negatives per sample. The ColEmbed 8B v2 training takes $3 \mathrm { h } 4 0 \mathrm { m } ^ { 6 }$ using 8x A100 80GB GPU cards.

# 3.4. Key Methods for Better Performance

# 3.4.1. Modifying LLM decoder causal attention to bi-direction attention for encoders

LLMs and VLMs are decoder models and use causal (uni-directional) attention. It means during training, when predicting a token, the model is prevented from accessing the following tokens (on the right) to avoid leaking information that would not be available during inference.

When adapting decoder LLMs as embedding models (encoders), a common practice involves transitioning the uni-directional attention to bi-directional attention. This modification enables Transformer layers to attend to the full context of a sequence, allowing each token to integrate information from both preceding and succeeding tokens. Such global representation has been shown to significantly enhance retrieval accuracy for LLM-based embedding models [40, 2, 41]. Consistent with these findings, we implement this technique in Nemotron ColEmbed V2 models, where we observed substantial performance improvement when adapting Eagle 2 and Qwen3-VL architectures.

# 3.4.2. Hard-Negative Mining

Embedding models for retrieval are primarily trained with contrastive learning, a paradigm requiring triplets composed of a query, positive examples, and negative examples. Extensive research on information retrieval indicates the efficiency of contrastive learning hinges on the availability of hard-negatives, i.e., false examples that exhibit high semantic or lexical similarity to the query. The hard-negatives can be mined from the corpus using external sparse or dense "teacher" retrieval models.

For Nemotron ColEmbed V2 models, we used an internal Llama-Eagle 3B VLM embedding model for mining from the corpus the top-k most similar page images to the queries.

We leverage the top- $k$ with percentage to positive threshold method from NV-Retriever [2]. It filters the potential hard-negatives by limiting their maximum similarity scores to a percentage of the positive sample similarity score, creating a margin that reduces the number of false negatives (e.g. the ones that should be actually positives). We set the threshold as 0.95, meaning we select the $K$ most relevant negative samples whose similarity to the query is less than $9 5 \%$ of the query–positive similarity score.

This encourages the model to learn from challenging negatives, while removing potential false negatives that have high similarity scores.

# 3.4.3. Cluster-based Data Sampling

Public training datasets typically exhibit imbalance, with disproportionate numbers of samples across different domains. Training on such skewed data could lead to overfitting to specific domains, which compromises the model’s ability to generalize to others, especially the underrepresented ones.

To mitigate this, Nemotron-CLIMB[42] proposed a framework for optimizing the blend of LLM training data by partitioning the corpus into distinct clusters and sampling a curated percentage of training examples from each.

For Nemotron ColEmbed V2 models, we adapted their method, clustering the positive contexts, then sampling some positive samples from each cluster together with the associated queries and negatives.

For clustering, we generate embeddings from document page images using an internal Llama-Eagle VLM embedding model, resulting in 3072-dim vectors. We then apply PCA to reduce embeddings dimension to 50, followed by a K-Means clustering approach utilizing gap statistics [43] to choose the most representative $k$ clusters. To ensure diversity and balance the domains of our training blend, we perform uniform sampling from the 14 discovered clusters.

# 3.4.4. Cross-lingual Translation

There has been growing interest from the community in multi-lingual and cross-lingual retrieval, the latter for cases where the query and corpus languages are different. This is particularly challenging for visual document retrieval, as document pages are represented as images to the model. Vidore V3

exemplifies this interest, as the corpus of PDF documents is in English or French, and queries are translated into six languages: English, French, Spanish, German, Italian, and Portuguese.

To enable Nemotron ColEmbed V2 models to support cross-lingual retrieval, we have augmented our training data using Qwen3-235B-A22 model to translate sampled queries from each discovered cluster into other languages.

# 3.4.5. Two-stage Training

The llama-nemotron-colembed-vl-3b-v2 model is trained in two stages [30]. In the first stage, the model is trained on a textual corpus consisting of query-positives-negatives triplets. This stage is designed to establish a robust foundation for semantic similarity within the textual embedding space.

In the second stage, we fine-tune the model on an image-retrieval corpus. This stage facilitates cross-modal alignment by grounding visual features in the textual representation space.

For nemotron-colembed-vl-4b- $\cdot \nu 2$ and nemotron-colembed-vl-8b-v2, due to Qwen3-VL’s strong crossmodal pre-training, we perform a single-stage contrastive learning training with image corpus.

# 3.4.6. Model Merging

Model merging or model souping is a technique that combines weights of multiple models, typically those sharing the same architecture but trained with different data or hyperparameters [44, 45]. It is observed that the efficacy of this approach increases with the diversity of the constituent models’ weights. Recently, this approach has been popularized for improving the robustness and generalization of embedding models, such as Qwen3-Embedding [3], Gemini Embedding [32], EmbeddingGemma [33], and Llama-Embed-Nemotron-8B [34].

For Nemotron ColEmbed V2 models, we employed a simple weighted average of model weights for merging. The 3B model is an ensemble of 8 individual models, and the 4B and 8B are merged from 4 individual models each. The individual models are trained with variations in the training blend (datasets and data size) and in the random seed that is used by the dataloader for sampling.

By comparing the ensembled model with the best individual model used in the ensemble, we have noticed that the accuracy gains might scale with the model size. We observed an improvement of $0 . 8 \%$ for llama-nemotron-colembed-vl-3b-v2, $1 . 0 \%$ for the nemotron-colembed-vl-4b-v2, and $1 . 5 \%$ for the nemotron-colembed-vl-8b-v2 model.

# 4. Results

In this section we demonstrate the effectiveness of Nemotron ColEmbed V2 models on different visual document retrieval benchmarks, with nemotron-colembed-vl-8b- $\cdot \nu 2$ being the top-performer in all of them.

# 4.1. Vidore V3 Leaderboard

The Vidore V3 [21] is a recent benchmark that emphasizes evaluation of vision document retrieval for complex enterprise and real-world scenarios, including multi-type and multi-language queries across ten professional domains.

To ensure submitted models do not overfit to the leaderboard, Vidore V3 has created two sets of tasks/datasets: eight public and two private, for which test datasets were not released publicly. The MTEB maintainers have established a process in which they run themselves the submitted models evaluation on the private tasks and report in the leaderboard.

Table 2 presents the retrieval accuracy $( \mathrm { N D C G } @ 1 0 )$ of the top models in MTEB Vidore V3 leaderboard, as of Feb. 03, 2026. We can see that our nemotron-colembed-vl-8b- $\cdot \nu 2$ model places 1st in the leaderboard, with NDCG $@ 1 0$ of 63.42, with $+ 3 \%$ improvement to the second place. The leaderboard also presents results for nemotron-colembed-vl-4b-v2 and llama-nemotron-colembed-vl-3b-v2, which demonstrate the highest Avg. NDCG $@ 1 0$ over the other models with the same size.

Table 2 Vidore V3 leaderboard as of Feb. 03, 2026. Our Nemotron ColEmbed V2 models are highlighted in gray. Retrieval accuracy scores are NDCG@10.   

<table><tr><td rowspan="2">Rank Model</td><td rowspan="2"></td><td rowspan="2">Avg</td><td colspan="8">Public tasks</td><td colspan="2">Private tasks</td></tr><tr><td>CompSci</td><td>Energy</td><td>FinanceEn</td><td>FinanceFr</td><td>HR</td><td>Industrial</td><td>Pharma</td><td>Physics</td><td>NuclearTelecom</td><td></td></tr><tr><td>1</td><td>nemotron-colembed-vl-8b-v2</td><td>63.42</td><td>79.30</td><td>69.82</td><td>67.29</td><td>51.54</td><td>66.32</td><td>56.03</td><td>67.19</td><td>50.84</td><td>53.84</td><td>72.00</td></tr><tr><td>2</td><td>tomoro-colqwen3-embed-8b</td><td>61.59</td><td>75.35</td><td>68.41</td><td>65.08</td><td>49.10</td><td>63.98</td><td>54.41</td><td>66.36</td><td>50.13</td><td>52.65</td><td>70.46</td></tr><tr><td>3</td><td>nemotron-colembed-vl-4b-v2</td><td>61.54</td><td>78.56</td><td>67.48</td><td>65.02</td><td>49.01</td><td>62.39</td><td>53.91</td><td>66.10</td><td>48.86</td><td>52.78</td><td>71.30</td></tr><tr><td>4</td><td>Ops-Colqwen3-4B</td><td>61.17</td><td>77.74</td><td>66.49</td><td>65.71</td><td>48.81</td><td>61.81</td><td>53.99</td><td>66.42</td><td>49.14</td><td>52.23</td><td>69.33</td></tr><tr><td>5</td><td>tomoro-colqwen3-embed-4b</td><td>60.20</td><td>75.44</td><td>66.43</td><td>63.84</td><td>46.83</td><td>60.09</td><td>53.58</td><td>65.74</td><td>49.32</td><td>51.23</td><td>69.44</td></tr><tr><td>6</td><td>Ilama-nemotron-colembed-vl-3b-v2</td><td>59.79</td><td>77.09</td><td>64.88</td><td>64.23</td><td>44.41</td><td>62.28</td><td>51.71</td><td>66.04</td><td>46.93</td><td>50.65</td><td>69.68</td></tr><tr><td>7</td><td> jina-embeddings-v4</td><td>57.52</td><td>71.81</td><td>63.50</td><td>59.30</td><td>46.10</td><td>59.53</td><td>50.38</td><td>63.09</td><td>46.63</td><td>50.02</td><td>64.81</td></tr><tr><td>8</td><td>colnomic-embed-multimodal-7b</td><td>57.33</td><td>76.20</td><td>63.58</td><td>56.57</td><td>45.46</td><td>58.67</td><td>50.13</td><td>62.26</td><td>48.25</td><td>45.02</td><td>67.16</td></tr><tr><td>9</td><td>Illama-nemoretriever-colembed-3b-v1</td><td>57.26</td><td>75.16</td><td>62.07</td><td>60.88</td><td>43.77</td><td>58.69</td><td>47.09</td><td>63.74</td><td>45.13</td><td>49.15</td><td>64.74</td></tr></table>

# 4.2. Vidore V1&V2 Leaderboard

Table 3 presents results for the MTEB Vidore V1&V2 leaderboard as of Feb. 03, 2026.

The leaderboard integrates the two benchmarks because Vidore V1 provided public in-domain training data, and many models overfit to it. For Vidore V2, no in-domain training data was provided, as well for Vidore V3.

Our nemotron-colembed-vl- $\cdot 8 b$ -v2 model places second, close to the leading model. The llama-nemotroncolembed-vl-3b- $\cdot \nu 2$ and nemotron-colembed-vl-4b- $\cdot \nu 2$ are also among the top-4.

Table 3 Vidore V1&V2 leaderboard as of Feb. 03, 2026. Our Nemotron ColEmbed V2 models are highlighted in gray. Retrieval accuracy scores are ${ \mathsf { N D C G } } ( \varpi 5 $ .   

<table><tr><td rowspan="2">Rank</td><td rowspan="2">Model</td><td rowspan="2">Avg.</td><td colspan="10">VidoreV1 ArxivQADocVQA InfoVQA Shift Project Energy</td><td colspan="4">Vidore V2</td></tr><tr><td></td><td></td><td></td><td></td><td>Al</td><td></td><td></td><td>Gov. ReportsHealthcare</td><td></td><td>TabFQuadTAT-DQA</td><td>MIT Biomed.</td><td></td><td>ESG Restau. EnESG Restau. MultiEcon. Macro</td><td></td></tr><tr><td>1</td><td>Ops-Colqwen3-4B</td><td>84.87</td><td></td><td>66.45</td><td>94.02</td><td>90.84</td><td>99.63</td><td>97.26</td><td>98.02</td><td>99.63</td><td>93.55</td><td>82.38</td><td>65.53</td><td>78.61</td><td>66.05</td><td>64.45</td></tr><tr><td>2</td><td>nemotron-colembed-vl-8b-v2</td><td>84.80</td><td>91.78 93.08</td><td></td><td>94.56</td><td>93.30</td><td>100.00</td><td>97.89</td><td>98.89</td><td>99.63</td><td>97.74</td><td>83.37</td><td>66.16</td><td>73.15</td><td>60.56</td><td>60.76</td></tr><tr><td>3</td><td>nemotron-colembed-vl-4b-v2</td><td>83.87</td><td>92.03</td><td>68.05 67.39</td><td>93.31</td><td>92.26</td><td>99.26</td><td>96.19</td><td>98.02</td><td>98.52</td><td>98.05</td><td>81.19</td><td>64.32</td><td>71.43</td><td>61.48</td><td>60.75</td></tr><tr><td>4</td><td>llama-nemotron-colembed-vl-3b-v2</td><td>83.64</td><td>90.40</td><td>67.17</td><td>94.68</td><td>92.00</td><td>100.00</td><td>98.02</td><td>97.95</td><td>98.89</td><td>97.25</td><td>81.04</td><td>63.19</td><td>73.11</td><td>58.64</td><td>58.59</td></tr><tr><td>5</td><td>tomoro-colqwen3-embed-8b</td><td>83.52</td><td>91.15</td><td>66.37</td><td>94.48</td><td>87.89</td><td>99.26</td><td>96.71</td><td>97.58</td><td>99.06</td><td>94.23</td><td>80.92</td><td>65.47</td><td>75.98</td><td>60.71</td><td>59.46</td></tr><tr><td>6</td><td>EvoQwen2.5-VL-Retriever-7B-v1</td><td>83.41</td><td>91.49</td><td>65.07</td><td>94.11</td><td>88.80</td><td>99.63</td><td>96.63</td><td>96.29</td><td>98.89</td><td>93.63</td><td>82.26</td><td>65.20</td><td>76.98</td><td>59.67</td><td>59.13</td></tr><tr><td>7</td><td>tomoro-colqwen3-embed-4b</td><td>83.18</td><td>90.58</td><td>66.30</td><td>94.31</td><td>87.39</td><td>99.26</td><td>96.91</td><td>97.17</td><td>99.63</td><td>94.33</td><td>79.87</td><td>65.38</td><td>74.65</td><td>62.44</td><td>56.30</td></tr><tr><td>8</td><td>llama-nemoretriever-colembed-3b-v1</td><td>83.10</td><td>88.35</td><td>66.21</td><td>94.92</td><td>90.70</td><td>99.63</td><td>96.63</td><td>97.82</td><td>99.26</td><td>95.94</td><td>80.57</td><td>62.70</td><td>75.38</td><td>57.38</td><td>57.84</td></tr><tr><td>9</td><td>SauerkrautLM-ColQwen3-8b-v0.1</td><td>82.91</td><td>93.80</td><td>64.69</td><td>94.51</td><td>90.41</td><td>98.65</td><td>96.52</td><td>96.79</td><td>99.26</td><td>92.18</td><td>84.04</td><td>63.26</td><td>70.77</td><td>57.85</td><td>57.98</td></tr><tr><td>10</td><td>EvoQwen2.5-VL-Retriever-3B-v1</td><td>82.76</td><td>90.46</td><td>63.67</td><td>92.22</td><td>88.60</td><td>100.00</td><td>97.63</td><td>98.89</td><td>99.26</td><td>93.99</td><td>82.00</td><td>63.63</td><td>67.11</td><td>59.05</td><td>62.19</td></tr></table>

# 4.3. MIRACL-Vision benchmark

MIRACL-Vision is a large multi-lingual VDR benchmark [46], covering many popular and underresourced languages. It is based on MIRACL[47] text multilingual benchmark and the corpus is composed by Wikipedia passages. MIRACL-VISION corpus, instead, is composed of screenshot images extracted from Wikipedia pages.

Table 4 shows retrieval accuracy $( \mathrm { N D C G } @ 1 0 )$ on MIRACL-Vision multilingual benchmark. It can be observed how models’ accuracy on visual document retrieval vary for popular vs. under-resourced languages (e.g. Telugu, Yoruba).

Nemotron ColEmbed V2 models perform better among the analyzed models, because of the pretraining of the backbone LLMs and also our augmented training blend containing cross-lingual examples. The nemotron-colembed-vl-8b-v2 provides the highest score for most of the languages.

Table 4 MIRACL-Vision results $( \mathsf { N D C G } @ 1 0 )$ on multi-lingual visual document retrieval. Nemotron ColEmbed V2 models are in the last three columns, results from other models obtained from MIRACL-Vision paper[46].   

<table><tr><td>Language</td><td>dse- qwen2- 2b-mrl-v1</td><td>gme- Qwen2- VL-2B- Instruct</td><td>vdr-2b- multi- v1</td><td>colqwen2- v1.0</td><td>llama- nemoretriever- colembed- 3b-v1</td><td>colembed- 3b-v2</td><td>nemoretriever-nemotron- colembed- vl-4b-v2</td><td>nemotron- colembed- vl-8b-v2</td></tr><tr><td>Arabic</td><td>0.3893</td><td>0.4888</td><td>0.4379</td><td>0.4129</td><td>0.4247</td><td>0.5250</td><td>0.6028</td><td>0.7863</td></tr><tr><td>Bengali</td><td>0.2352</td><td>0.3755</td><td>0.2473</td><td>0.2888</td><td>0.4878</td><td>0.5391</td><td>0.5156</td><td>0.6160</td></tr><tr><td>Chinese</td><td>0.5962</td><td>0.6314</td><td>0.5963</td><td>0.4926</td><td>0.4355</td><td>0.4878</td><td>0.6697</td><td>0.7204</td></tr><tr><td>English</td><td>0.6605</td><td>0.6784</td><td>0.6784</td><td>0.6417</td><td>0.7363</td><td>0.7397</td><td>0.7246</td><td>0.7480</td></tr><tr><td>Farsi</td><td>0.2250</td><td>0.3085</td><td>0.2398</td><td>0.2616</td><td>0.3109</td><td>0.3570</td><td>0.4266</td><td>0.5289</td></tr><tr><td>Finnish</td><td>0.4162</td><td>0.6863</td><td>0.5283</td><td>0.6604</td><td>0.8513</td><td>0.8541</td><td>0.8398</td><td>0.8726</td></tr><tr><td>French</td><td>0.7160</td><td>0.6851</td><td>0.7194</td><td>0.6876</td><td>0.7988</td><td>0.7943</td><td>0.7943</td><td>0.8171</td></tr><tr><td>German</td><td>0.6267</td><td>0.6345</td><td>0.6205</td><td>0.5995</td><td>0.6831</td><td>0.6924</td><td>0.7100</td><td>0.7233</td></tr><tr><td>Hindi</td><td>0.1740</td><td>0.3127</td><td>0.2058</td><td>0.2209</td><td>0.4867</td><td>0.5319</td><td>0.5338</td><td>0.5902</td></tr><tr><td>Indonesian</td><td>0.4866</td><td>0.5416</td><td>0.5254</td><td>0.5320</td><td>0.6428</td><td>0.6550</td><td>0.6480</td><td>0.6680</td></tr><tr><td>Japanese</td><td>0.6232</td><td>0.7305</td><td>0.6553</td><td>0.6970</td><td>0.7260</td><td>0.7493</td><td>0.8326</td><td>0.8690</td></tr><tr><td>Korean</td><td>0.4446</td><td>0.6202</td><td>0.4952</td><td>0.4419</td><td>0.5158</td><td>0.5394</td><td>0.6136</td><td>0.7316</td></tr><tr><td>Russian</td><td>0.6505</td><td>0.7202</td><td>0.6995</td><td>0.6811</td><td>0.7670</td><td>0.7920</td><td>0.7879</td><td>0.8399</td></tr><tr><td>Spanish</td><td>0.5927</td><td>0.6277</td><td>0.6274</td><td>0.6224</td><td>0.7109</td><td>0.7236</td><td>0.7033</td><td>0.7089</td></tr><tr><td>Swahili</td><td>0.4156</td><td>0.5348</td><td>0.4509</td><td>0.4931</td><td>0.7767</td><td>0.7495</td><td>0.6886</td><td>0.7422</td></tr><tr><td>Telugu</td><td>0.0274</td><td>0.0893</td><td>0.0318</td><td>0.0264</td><td>0.1669</td><td>0.2325</td><td>0.1579</td><td>0.1899</td></tr><tr><td>Thai</td><td>0.2692</td><td>0.3563</td><td>0.3177</td><td>0.2389</td><td>0.4035</td><td>0.4727</td><td>0.5928</td><td>0.6699</td></tr><tr><td>Yoruba</td><td>0.4178</td><td>0.4884</td><td>0.4577</td><td>0.5120</td><td>0.5888</td><td>0.5943</td><td>0.4469</td><td>0.5252</td></tr><tr><td>Average</td><td>0.4426</td><td>0.5283</td><td>0.4741</td><td>0.4728</td><td>0.5841</td><td>0.6127</td><td>0.6272</td><td>0.6860</td></tr></table>

# 5. Late-interaction Deployment Challenges

Leaderboards and benchmarks typically evaluate performance based on accuracy metrics. Some also include proxy indicators of computational efficiency, such as model size or embedding dimensionality. Ultimately, rankings are determined by accuracy, which may not reflect the broader needs of real-world applications. No solution fits all use-cases. In this section, we discuss trade-offs of late interaction in the context of production deployment.

# 5.1. Accuracy Considerations

To help understand the retrieval accuracy difference obtained with single vector (average pooling) vs late interaction, we have run experiments comparing both methods, using the same model architecture, training data and hyperparameters. The results are reported in Table 5. We can see that the accuracy obtained with late interaction is about $1 2 \%$ and $9 \%$ higher for ColEmbed 4B and ColEmbed 8B, respectively. Such effectiveness is the reason why late interaction embedding models have dominated Visual Document Retrieval benchmarks like ViDoRe.

Table 5 Comparison retrieval accuracy (Vidore V3 NDCG@10) of late-interaction vs single vector (avg. pooling), using same training data and hyperparameters.   

<table><tr><td>Model</td><td>Single vector</td><td>Late interaction</td><td>% improvement</td></tr><tr><td>nemotron-colembed-vl-4b-v2</td><td>54.07</td><td>60.52</td><td>+ 11.92%</td></tr><tr><td>nemotron-colembed-vl-8b-v2</td><td>56.91</td><td>62.24</td><td>+ 9.36%</td></tr></table>

# 5.2. Retrieval Systems Considerations

Deploying a production system involves balancing accuracy, latency/throughput, and cost. Typical retrieval system requirements involve the following aspects:

• Model size: All embeddings of documents are generated by retrieval model. This step can be performed in batches, with support for continuous updates as new documents arrive. Throughput and cost are key considerations, and the overall retrieval performance is primarily influenced by the size of the model.   
• Storage: The embeddings storage requirements are primarily determined by the embedding dimension, numeric precision and number of vectors per document.   
• Serving: Latency measures how quickly documents can be retrieved in response to a user query. Since queries are typically short (around 50–100 tokens), the size of the embedding model plays a smaller role in this stage. Incorporating a reranker in the retrieval pipeline, such as a cross-encoder, can improve accuracy, but at the cost of increasing the latency to serve another model.

We discuss the trade-offs between those aspects in the next section.

# 5.3. Retrieval Pipelines Trade-offs

The late-interaction paradigm [19] has demonstrated significant performance improvements in retrieval tasks by preserving fine-grained token-level interactions between queries and documents. Unlike traditional pooling strategies that compress entire sequences into single vectors, late-interaction models leverage all token-level representations. However, this approach introduces a fundamental trade-off between accuracy and storage cost, as each document requires multiple token embeddings, leading to significantly increased storage requirements.

Table 6 summarizes these trade-offs aspects for different retrieval approaches, reporting requirements in GigaBytes (GB) for storing the embeddings for one million page images. The storage footprint of lateinteraction models depends on three key factors: token/embedding count (sequence length), embedding dimension, and numerical precision (e.g., float32, float16, int8). As can be observed in the table, late interaction models require much more storage for indexing document multi-vector embeddings.

Table 6 Comparison of different VLM models in terms of model size, embedding dimension, storage requirements and retrieval accuracy (Vidore V3 NDCG@10). Last line is a retrieval pipeline in which an embedding model retrieves the top 50 page images, subsequently reranked by a cross-encoder.   

<table><tr><td>Model</td><td>Params (B)</td><td>Embed. dim</td><td>Avg. tokens/ embeddings per image</td><td># floating points perimage</td><td>Storage for 1M fp16 embed.(GB)</td><td>Vidore V3 NDCG@10</td></tr><tr><td>nemotron-colembed-vl-8b-v2</td><td>8.14</td><td>4096</td><td>773</td><td>3166208</td><td>5897.5</td><td>63.42</td></tr><tr><td>nemotron-colembed-vl-4b-v2</td><td>4.43</td><td>2560</td><td>773</td><td>1978880</td><td>3686.0</td><td>61.42</td></tr><tr><td>llama-nemotron-colembed-vl-3b-v2</td><td>3.99</td><td>3072</td><td>2304</td><td>7077888</td><td>13183.6</td><td>59.70</td></tr><tr><td>llama-nemoretriever-colembed-1b-v1</td><td>2.15</td><td>2048</td><td>2304</td><td>4718592</td><td>8789.1</td><td>55.48</td></tr><tr><td>llama-nemotron-embed-vl-1b-v2 llama-nemotron-embed-vl-1b-v2 w/</td><td>1.41</td><td>2048</td><td>1</td><td>2048</td><td>3.8</td><td>48.69</td></tr><tr><td>llama-nemotron-rerank-vl-1b-v2</td><td>1.41 + 1.41</td><td>2048</td><td>1</td><td>2048</td><td>3.8</td><td>54.41</td></tr></table>

The number of tokens per image (sequence length) is determined by the VLM image tiling/resizing logic and its image encoder. For example, Qwen-3 VL (backbone models from our 8B and 4B embedding models) generates on average 773 visual embeddings for Vidore V3 pages, while Eagle 2 generates 2304 visual embeddings for Vidore V3 page images (with resolution of about $1 6 5 4 \mathrm { x } 2 3 3 9$ pixels).

Late interaction models require orders of magnitude more storage. For instance, the llamanemoretriever-colembed- $1 b$ -v1 late interaction model and the llama-nemotron-embed-vl-1b-v2 singlevector model. Both use Llama 3.2 1B as LLM backbone. While the single-vector model requires 3.8 GB for embedding storage, the late-interaction model requires 8,789.1 GB (2312x). If we add the llamanemotron-rerank-vl-1b- $\cdot \nu 2$ cross-encoder model to the retrieval pipeline, reranking the top-50 page images retrieved by llama-nemotron-embed-vl-1b-v2, the NDCG@10 boosts from 48.69 to 54.40, which is close to the 55.48 accuracy from the late interaction llama-nemoretriever-colembed-1b-v1 for a small fraction of storage requirements7.

Latency is another important challenge for Late-interaction models. During inference, it requires calculation between a query and the multi-vectors of all page images in the corpus. Late interaction requires specialized vector database support to the MaxSim operation, and might introduce latency overhead. The alternative pipeline composed of a single-vector embedding followed by a cross-encoder provides much lower latency, because the cross-encoder performs early-interaction between query and document tokens only for the top-k documents retrieved by the embedding model, and not for the whole corpus as in the late interaction approach.

Retrieval pipeline design that should be carefully aligned with the specific use case. For example, [48] results demonstrate that in scenarios where corpus is large and number of queries is moderate, a smaller less accurate embedding model combined with a reranker (to improve accuracy) can be more cost-efficient than a larger embedding model.

Ultimately, the choice between late-interaction and bi-encoder paradigms depends on specific use case requirements and system constraints.

# 5.4. Ablation on Embedding-size Reduction

Several techniques can minimize storage requirements for both paradigms, like reducing embedding dimensions. Linear projection layers can be used to downsize the embeddings output by the LLM backbone. Matryoshka Representation Learning [49] allows having a single model that outputs embeddings that can be sliced/pruned to multiple smaller dimensions.

Following the approach used in vidore/colqwen2-v1.0 models [7], we applied a linear projection layer to reduce the output dimension to 512 and 128. To minimize accuracy variation in the ablation, for each architecture and dim size, we train four models with different seeds (that sample different portions of our data blend for training), and report the average $\mathrm { N D C G } @ 1 0$ across these four models.

We can observe the ablation results in Table 7. For the nemotron-colembed- $\cdot \nu l - 8 b - \nu 2 ,$ projecting embeddings to 512-dim reduces storage requirements by $8 7 . 5 \%$ , while keeping $9 6 . 0 2 \%$ of retrieval accuracy. With 128-dim embeddings, it requires only $3 \%$ of storage, keeping $9 5 . 3 6 \%$ of the accuracy. We see a similar trend for nemotron-colembed-vl-4b-v2 model. However, even with 128-dim, the storage requirement of 184.3 GB for 1M pages may still be too high for production environments handling large document corpora.

Table 7 Ablation study on reducing the embedding sizes of late interaction models. For these models, we don’t apply model merging; instead, we report the average ${ \mathsf { N D C G } } { \widehat { \varrho } } 1 0$ across four models trained with different seeds.   

<table><tr><td rowspan="2">Model</td><td rowspan="2">Params (B)</td><td rowspan="2">Embed. dim</td><td rowspan="2">Avg. tokens/ embeddings</td><td rowspan="2">#floating points per image</td><td rowspan="2">Storage for 1M fp16 embed.(GB)</td><td rowspan="2">% storage</td><td rowspan="2">Vidore V3 NDCG@10</td><td rowspan="2">% NDCG@10</td></tr><tr><td>perimage</td></tr><tr><td rowspan="4">nemotron-colembed-vl-8b-v2</td><td rowspan="4">8.14</td><td>4096</td><td>773</td><td>3166208</td><td>5897.5</td><td>100%</td><td>62.29</td><td>100.00%</td></tr><tr><td>512</td><td>773</td><td>395776</td><td>737.2</td><td>13%</td><td>59.81</td><td>96.02%</td></tr><tr><td>128</td><td>773</td><td>98944</td><td>184.3</td><td>3%</td><td>59.40</td><td>95.36%</td></tr><tr><td>2560</td><td>773</td><td>1978880</td><td>3686.0</td><td>100%</td><td>60.42</td><td>100.00%</td></tr><tr><td rowspan="2">nemotron-colembed-vl-4b-v2</td><td>4.43</td><td>512</td><td>773 773</td><td>395776 98944</td><td>737.2 184.3</td><td>20% 5%</td><td>59.29 58.47</td><td>98.13%</td></tr><tr><td></td><td>128</td><td></td><td></td><td></td><td></td><td></td><td>96.77%</td></tr></table>

Decreasing embeddings numerical precision, i.e., to float16 or int8, is an alternative to reducing storage footprint. Many vector databases already support storage and retrieval of lower-precision embeddings, some of them offering post-training quantization for precision reduction.

Additionally, binary quantization reduces precision to 1-bit per element, potentially reducing storage by 16x. However, our experience with bi-encoders indicates that binary quantization performs poorly when the embedding dimensionality is too small, and these techniques require further testing with late-interaction embedding size of 128. AnswerAI’s late-pooling approach [50] can reduce token vectors by factors of 3-5, while MUVERA [51] proposes converting multi-vector embeddings into single Fixed Dimensional Encodings (FDEs) whose inner product approximates multi-vector similarity, enabling the use of standard single-vector retrieval with smaller total embedding size.

# 6. Conclusion

In this paper, we introduce the Nemotron ColEmbed V2 family of late-interaction models for visual document retrieval. We demonstrate their top-performance on ViDoRe benchmarks and multi-lingual capabilities on MIRACL-Vision benchmark. We describe the main methods that boosted the accuracy of our late-interaction models, like changing VLMs backbones to use bi-directional attention, using positive-aware hard-negative mining, cluster-based data sampling, cross-lingual translation, and model merging. Finally, we discuss the deployment challenges of late-interactions models and highlight key considerations for real-world deployment. We present numbers that illustrate the trade-offs between accuracy and storage requirements and provide an ablation on reducing embedding sizes, thus storage requirements. Our release of Nemotron ColEmbed V2 late-interaction models provides a strong foundation for future research and practical applications in visual document retrieval.

We recommend further research on reducing the storage requirements of late interaction models, e.g., by using smaller embedding dimensions or lower numerical precision, or fewer embeddings per sample, without significantly penalizing retrieval accuracy.

