# VIDORE BENCHMARK V2: Raising the Bar for Visual Retrieval

Quentin Mace´1 Antonio Loison ´ 1 Manuel Faysse 1,2 1Illuin Technology 2CentraleSupelec ´

The ViDoRe Benchmark V1 was approaching saturation with top models exceeding $9 0 \%$ nDCG $\ @ 5$ , limiting its ability to discern improvements. ViDoRe Benchmark V2 introduces realistic, challenging retrieval scenarios via blind contextual querying, long and cross-document queries, and a hybrid synthetic and human-in-the-loop query generation process. It comprises four diverse, multilingual datasets and provides clear evaluation instructions. Initial results demonstrate substantial room for advancement and highlight insights on model generalization and multilingual capability. This benchmark is designed as a living resource, inviting community contributions to maintain relevance through future evaluations.

Contact: quentin.mace@illuin.tech   
Web Blog Version: https://huggingface.co/blog/manu/vidore-v2   
Leaderboard: https://huggingface.co/spaces/vidore/vidore-leaderboard   
Date: March 18, 2025

# Why a new benchmark?

Since the release of the original ViDoRe Benchmark (Faysse et al., 2025), evaluating visual models on document retrieval tasks, visual retrieval models have significantly advanced! While the original ColPali model reported an average score of $8 1 . 3 \mathrm { n D C G } @ 5 $ , current SOTA models on the leaderboard surpass a nDCG $\ @ 5$ of 90, with some tasks becoming “too easy” to yield a meaningful signal! With the benchmark approaching saturation for SOTA models, there is limited room to truly measure improvements and understand model capabilities in realistic scenarios. To continue pushing the boundaries of visual retrieval, it became essential to introduce a new benchmark designed specifically to challenge these advanced models: VIDORE BENCHMARK V2.

# 1 Motivating the Creation of ViDoRe Benchmark V2

In developing ViDoRe Benchmark V2, our main goal was to create a benchmark reflective of real-world retrieval challenges—difficult, diverse, and meaningful. Current benchmarks exhibit limitations that prevent them from accurately reflecting real user behavior and complex retrieval scenarios (Thakur et al., 2025). We identified three critical issues in existing benchmarks:

1. Extractive Nature of Queries: Current benchmarks typically rely on extractive queries, providing unrealistic retrieval contexts since real users rarely formulate queries from exact phrases in documents.   
2. Single-Page Query Bias: Many benchmarks overly emphasize retrieval from singlepage contexts, neglecting complex, multi-document or cross-document queries common in real-world applications.   
3. Challenges in Synthetic Query Generation: Purely synthetic benchmarks, while appealing in theory, are difficult to implement effectively without extensive manual oversight. They often produce outliers, irrelevant or trivial queries, making human filtering essential yet costly.

# 2 Design Decisions and Techniques Used

To address these challenges and create a robust, realistic benchmark, ViDoRe Benchmark V2 includes several innovative features:

• Blind Contextual Querying: In practice, users don’t often know the content of the corpus they are querying. To reduce the widespread extractive bias in most synthetic query-document datasets (datasets are often created with knowledge of the document content), we only provided query annotator models with limited information about the document (summaries, metadata, etc) and filtered out the many irrelevant queries that resulted, better reproducing real-world user interactions with the corpus.   
Long and Cross-Document Queries: Unlike traditional benchmarks, ViDoRe Benchmark V2 emphasizes long-form and cross-document queries, closely mirroring real-world retrieval situations. Multiple datasets specifically focus on scenarios involving comprehensive documents or multi-document retrieval tasks.   
Hybrid Synthetic and Human-in-the-Loop Creation: Recognizing the limitations of synthetic query generation alone, we adopted a hybrid approach—generating queries synthetically and extensively refining them through human review. This process, though intensive, ensured significantly higher query quality and dataset reliability.

# 3 Dataset Selection for ViDoRe Benchmark V2

The selected datasets (Table 1) for ViDoRe Benchmark V2 are diverse, publicly available, and challenging. Each dataset presents distinct visual complexity and is suitable for realistic retrieval tasks, including multilingual versions with queries translated into French, English, Spanish, and German. This multilingual approach further extends the applicability and challenge level of the benchmark. Each dataset is associated to a multilingual version with translated queries.

<table><tr><td></td><td>Suen Sue</td><td>20</td><td>S ebu </td><td>efes#</td><td>20</td><td>sn #</td><td></td><td>Aen/aSyV h </td><td>Comments</td></tr><tr><td>Insurance Terms of Service1 Biomedical</td><td>En</td><td>Fr En</td><td>4 27</td><td>260 1,016</td><td>-</td><td>18 160</td><td>86 515</td><td>4.8</td><td>Small but challenging, multi-document Largest dataset, most extractive</td></tr><tr><td>Economics</td><td>En</td><td>En</td><td>5</td><td></td><td>-</td><td>58</td><td>907</td><td>3.2 15.6</td><td>Cross-document queries, high complexity</td></tr><tr><td></td><td></td><td></td><td></td><td>452</td><td>Synthetic</td><td>57</td><td>222</td><td>3.9</td><td></td></tr><tr><td>ESG Reports</td><td>En</td><td>En</td><td>30</td><td>1,538</td><td>Human</td><td>52</td><td>128</td><td>2.5</td><td>Natively cross-lingual, industry-specific</td></tr></table>

Table 1: Summary of dataset statistics. Feel free to explore datasets on HuggingFace.

# 4 Evaluating Models

To evaluate models on ViDoRe Benchmark 2, we follow these steps:

# Option 1: Using the CLI

Here is a CLI example for using a colpali type retriever on vidore benchmark 2. For other retrievers, please refer to this repo.

idore-benchmark evaluate-retriever \\--model-class colpali \\--model-name vidore/colpali-v1.3 \\--collection-name vidore/vidore-benchmark-v2-dev-67ae03e3924e85b36e7f53b0 \\--dataset-format beir \\--split test

# Option 2: Creating a custom retriever

Detailed instructions on how to create a custom retriever are available at https:// github.com/illuin-tech/vidore-benchmark. We will soon transition to using the MTEB (Muennighoff et al., 2022) library to evaluate all models.

# 5 Results

Here are for example some nDCG $\ @ 5$ results of visual retrieval models on ViDoRe Benchmark 2 (Faysse et al., 2025; Ma et al., 2024; Zhang et al., 2024; Yu et al., 2024; Team, 2025).2

<table><tr><td>Model voyageai</td><td>(ru son nnn</td><td>iunnee</td><td>Innuin wnnu</td><td>SotcouS</td><td>20</td><td>C!in o</td><td>SEF</td><td></td><td>Er Hod n</td><td>Cnu on</td><td>AAee</td></tr><tr><td>metrics-AI/colqwen2.5-3B</td><td>0.561 0.641</td><td>0.579</td><td>0.595</td><td>0.588</td><td>0.564</td><td>0.515</td><td>0.472</td><td></td><td>0.462</td><td>0.550</td><td>0.550</td></tr><tr><td></td><td>0.645</td><td></td><td>0.557</td><td>0.566</td><td>0.639</td><td>0.569</td><td>0.496</td><td></td><td>0.492</td><td>0.535</td><td>0.564</td></tr><tr><td>colsmolvlm-v0.1</td><td>0.624</td><td>0.555</td><td>0.432</td><td>0.609</td><td>0.581</td><td>0.505</td><td>0.511</td><td></td><td>0.476</td><td>0.474</td><td>0.530</td></tr><tr><td>colqwen2-v1.0</td><td>0.622</td><td>0.651</td><td>0.572</td><td>0.615</td><td>0.618</td><td>0.565</td><td>0.534</td><td></td><td>0.542</td><td>0.532</td><td>0.583</td></tr><tr><td>colpali-v1.2</td><td>0.321</td><td>0.560</td><td>0.458</td><td>0.531</td><td>0.585</td><td>0.557</td><td>0.519</td><td>0.540</td><td></td><td>0.479</td><td>0.505</td></tr><tr><td>dse-qwen2-2b-mrl-v1</td><td>0.614</td><td>0.655</td><td>0.563</td><td>0.615</td><td>0.592 0.532</td><td>0.551</td><td>0.549</td><td>0.557</td><td></td><td>0.528</td><td>0.580</td></tr><tr><td>colSmol-256M</td><td>0.460</td><td>0.504</td><td>0.341</td><td>0.534</td><td>0.597</td><td>0.340</td><td>0.272</td><td>0.313</td><td></td><td>0.273</td><td>0.397</td></tr><tr><td>colpali-v1.3</td><td>0.511</td><td>0.598</td><td>0.501</td><td>0.516</td><td></td><td>0.565</td><td>0.570</td><td>0.557</td><td></td><td>0.499</td><td>0.546</td></tr><tr><td>colqwen2.5-v0.2</td><td>0.684</td><td>0.603</td><td>0.532</td><td>0.598</td><td>0.636</td><td>0.611</td><td>0.574</td><td>0.574</td><td></td><td>0.565</td><td>0.597</td></tr><tr><td>dse-llamaindex</td><td>0.631</td><td>0.688</td><td>0.610 0.600</td><td>0.612 0.548</td><td>0.606 0.653</td><td>0.569</td><td>0.503 0.517</td><td>0.512</td><td></td><td>0.528</td><td>0.584</td></tr><tr><td>tsystems/colqwen2.5-3b-multi-v1.0</td><td>0.721</td><td>0.693 0.607</td><td>0.554</td><td>0.629</td><td>0.640</td><td>0.617 0.551</td><td>0.543</td><td>0.533 0.567</td><td></td><td>0.512 0.562</td><td>0.599</td></tr><tr><td>gme-qwen2-VL-7B</td><td>0.658 0.537</td><td>0.505</td><td>0.452</td><td>0.596</td><td>0.548</td><td>0.477</td><td>0.459</td><td>0.464</td><td></td><td>0.487</td><td>0.590 0.503</td></tr><tr><td>visrag-ret</td><td>0.522</td><td>0.587</td><td>0.377</td><td>0.503</td><td>0.543</td><td>0.421</td><td>0.392</td><td>0.391</td><td></td><td>0.361</td><td>0.455</td></tr><tr><td>colSmol-500M colpali-v1.1</td><td>0.465</td><td>0.547</td><td>0.484</td><td>0.567</td><td>0.564</td><td>0.507</td><td>0.461</td><td>0.481</td><td></td><td>0.438</td><td>0.502</td></tr></table>

Table 2: Model performance across datasets (nDCG@5). Highest per column in bold.

Analyzing the results, we can extract a few general takeaways:

• ViDoRe v2 maintains strong correlation with v1, with consistent model rankings across versions (Figure 1).   
• ViDoRe v2 leaves substantial room for future improvements, contrasting with ViDoRe v1, which was approaching performance saturation (scores exceeding $9 0 \%$ as seen in Figure 2).   
Certain models exhibit signs of slightly overfitting to the training distribution, resulting in reduced generalization to novel data (e.g., vidore/colSmol-256M, vidore/colSmol-500M, Metric-AI/ColQwen2.5-3b-multilingual-v1.0). These models perform worst on the V2 than what their performance on the V1 would lead to believe (Figure 1).

![](images/f03a3ef342b9a850e97854e97b4fb659481ebff256deb088b3d799d5d0cdc177.jpg)  
Figure 1: Performance results across models for V1 and V2. We observe strong correlations, although a clear saturation on V1 for top models. Results are in nDCG@5.

• The multilingual splits in ViDoRe v2 provide a more accurate assessment of nonenglish capabilities in visual retriever models. We observe a significant performance gap between models trained exclusively in English using an English-only VLM and those that are not.(Figure 3)

• Larger model scale is beneficial; notably, the gme-qwen7B model achieves strong overall performance but incurs significant computational cost and inference latency. Inversely, while impressive for their sizes, models under 1B parameters tend to lag behind, especially on previously unseen data distributions.

• We tend to see better separation between model performances with the human labeled dataset (ESG human), indicating it is of slightly higher quality than the synthetic datasets and is a more discriminating signal (Figure 2).

![](images/85cbff96dc09dedac538e5037c6ada40c2674cc77f43220b8ee1083c2508b98e.jpg)  
Figure 2: Performance results across monolingual tasks. ViDoRe v2 leaves substantial room for future improvements, contrasting with ViDoRe v1, which was approaching performance saturation.

# 6 Moving Forward

Our goal is for ViDoRe V2 to become a dynamic, ”living benchmark” that regularly grows with new tasks and datasets. To achieve this, we welcome and encourage the community to contribute datasets and evaluation tasks. This collaborative approach helps ensure that the benchmark stays relevant, useful, and reflective of real-world challenges.

![](images/8565eaee52c14ca2ff535fbe0f687f75fc8cbb6f5c3eabf149db93b3101ca87b.jpg)  
Figure 3: Performance results across crosslingual tasks.We observe a significant performance gap between models trained exclusively in English using an English-only VLM and those that are not.

We are also open on integrating new retrieval metrics such as confidence estimation measures (Gisserot-Boukhlef et al., 2024), increasing multilingual coverage allowed by ever better base models (Yang et al., 2025; Boizard et al., 2025), and extending the leaderboard to new modalities (audio, image querying, etc...)

# Acknowledgements

Training compute for running evaluations is obtained on the Jean Zay supercomputer operated by GENCI IDRIS through compute grant AD011016393.

For deeper discussions and projects around Visual RAG, ColPali, or agentic systems, please contact contact@illuin.tech or visit https://www.illuin.tech. We welcome community contributions of document-query sets to enhance this living benchmark.

