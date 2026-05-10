# DRAGIN: Dynamic Retrieval Augmented Generation based on the Information Needs of Large Language Models

Weihang Su\*1, Yichen Tang\*1, Qingyao Ai\*1, Zhijing Wu2, Yiqun Liu1

$^{1}$ Department of Computer Science and Technology, Tsinghua University $^{2}$ School of Computer Science and Technology, Beijing Institute of Technology

# Abstract

Dynamic retrieval augmented generation (RAG) paradigm actively decides when and what to retrieve during the text generation process of Large Language Models (LLMs). There are two key elements of this paradigm: identifying the optimal moment to activate the retrieval module (deciding when to retrieve) and crafting the appropriate query once retrieval is triggered (determining what to retrieve). However, current dynamic RAG methods fall short in both aspects. Firstly, the strategies for deciding when to retrieve often rely on static rules. Moreover, the strategies for deciding what to retrieve typically limit themselves to the LLM's most recent sentence or the last few tokens, while the LLM's information needs may span across the entire context. To overcome these limitations, we introduce a new framework, DRAGIN, i.e., Dynamic Retrieval Augmented Generation based on the Information Needs of LLMs. Our framework is specifically designed to make decisions on when and what to retrieve based on the LLM's information needs during the text generation process. We evaluate DRAGIN along with existing methods comprehensively over 4 knowledge-intensive generation datasets. Experimental results show that DRAGIN achieves superior performance on all tasks, demonstrating the effectiveness of our method<sup>1</sup>.

# 1 Introduction

In recent years, large language models (LLMs) have made significant advancements across various natural language processing (NLP) tasks, quickly becoming a critical element in numerous AI applications (Brown et al., 2020; Chowdhery et al., 2022; Touvron et al., 2023a; Scao et al., 2022; Zhang

et al., 2022). Despite their impressive capabilities, these models often produce text that seems coherent and plausible but factually incorrect, a problem commonly known as hallucination (Maynez et al., 2020; Zhou et al., 2020; Liu et al., 2021; Ji et al., 2023; Su et al., 2024).

To mitigate this issue, Retrieval-Augmented Generation (RAG) has emerged as a prominent solution. RAG enhances LLMs by retrieving and incorporating relevant information from external databases into the LLMs' inputs. It has demonstrated superior effectiveness across numerous NLP challenges (Khandelwal et al., 2019; Borgeaud et al., 2022; Lewis et al., 2020; Guu et al., 2020; Izacard and Grave, 2020; Jiang et al., 2022; Shi et al., 2023). Traditional methods of RAG typically rely on single-round retrieval, using the LLM's initial input to retrieve relevant information from external corpora. While this method is effective for straightforward tasks, it tends to fall short for complex multi-step tasks and long-form generation tasks (Jiang et al., 2023). In contrast, dynamic RAG (Trivedi et al., 2022; Borgeaud et al., 2022; Ram et al., 2023; Jiang et al., 2023) performs multiple times of retrieval during the generation process of LLMs. It includes two steps: identifying the optimal moment to activate the retrieval module (deciding when to retrieve), and crafting the appropriate query once retrieval is triggered (determining what to retrieve). Depending on when and what to retrieve, a variety of methods have been proposed in this direction. For example, IR-CoT (Trivedi et al., 2022) adopts a global augmentation method where retrieval is conducted for each generated sentence, with the latest generated sentence used as the query. RETRO (Borgeaud et al., 2022) and IC-RALM (Ram et al., 2023) define a sliding window and trigger the retrieval module based on a preset number of processed tokens, and the last $n$ tokens are used as the query.

However, existing dynamic RAG methods face

several critical challenges, primarily in determining the optimal timing for retrieval and formulating effective queries when retrieval is triggered. First of all, existing approaches often rely on static rules to decide when to retrieve, neglecting the assessment of necessity and potential risks involved. On the one hand, depending on the quality of the input query and retrieval models, unnecessary retrieval augmentation may introduce irrelevant or noisy data to LLMs which could jeopardize the quality of the outputs. On the other hand, conducting retrieval augmentation will inevitably increase the time and computation cost of LLM inference, such cost is unworthy if LLMs can generate correct outputs by themselves. Additionally, the strategies of existing studies in determining what to retrieve often restrict themselves to the LLM's most recent sentence or the last few tokens. This approach may not capture the model's real-time information needs since the LLM's information needs may actually be related to terms that span the entire context. Retrieving documents in this manner is thus suboptimal in many cases.

To overcome these limitations, we introduce a new framework, DRAGIN, i.e., Dynamic Retrieval Augmented Generation based on the Information Needs of LLMs. Our framework is specifically designed to make decisions on when and what to retrieve, based on the LLM's information needs during the text generation process. For the timing of retrieval, we propose RIND: Real-time Information Needs Detection, which considers the LLM's uncertainty about its own generated content, the influence of each token on subsequent tokens, and the semantic significance of each token. For the formulation of retrieval queries, we propose QFS: Query Formulation based on Self-attention, which innovates query formulation by leveraging the LLM's self-attention across the entire context. DRAGIN is a lightweight RAG framework that can be incorporated into any Transformer-based LLMs without further training, fine-tuning, or prompt engineering.

We comprehensively evaluate DRAGIN along with existing dynamic RAG frameworks over four knowledge-intensive generation benchmarks. Experimental results show that DRAGIN achieves superior performance on all datasets, demonstrating the effectiveness of our method. Moreover, the results of the ablation study indicate that our proposed new strategies for "when to retrieval" (i.e., RIND) and "what to retrieval" (i.e., QFS) perform

uniformly better than other strategies in existing RAG methods despite retrieval models and LLMs.

In summary, the contributions of our paper are as follows:

- We propose a novel dynamic RAG framework: DRAGIN. In contrast to previous works, our framework optimizes when and what to retrieve based on the real-time information needs of the LLM.   
- We evaluate existing dynamic RAG methods and DRAGIN on four knowledge-intensive datasets using three different LLMs. Experimental results indicate that DRAGIN achieves state-of-the-art (SOTA) performance.

# 2 Related Work

# 2.1 Single-round Retrieval-augmented LLM

LLMs have demonstrated significant effectiveness across a wide range of tasks. However, their built-in knowledge can sometimes fall short when dealing with knowledge-intensive tasks. To address this limitation, Retrieval-Augmented Generation (RAG) strategies are widely employed to enhance the performance of LLMs. One of the most direct methods is single-round retrieval augmentation (Khandelwal et al., 2019; Borgeaud et al., 2022; Lewis et al., 2020; Guu et al., 2020; Izacard and Grave, 2020; Jiang et al., 2022; Shi et al., 2023), which involves using the initial input as a query to retrieve information from an external corpus. The retrieved external knowledge is then incorporated as part of the input for the model. Previous research has explored single-round retrieval augmentation extensively. For instance, RE-PLUG (Shi et al., 2023) treats LLMs as a black box and leverages them to generate training data for the retrieval model. From a different perspective, UniWeb (Li et al., 2023d) proposes an adaptive search engine-assisted learning method that can self-assess whether the LLM requires retrieval augmentation.

# 2.2 Multi-round Retrieval-augmented LLM

Single-round retrieval can be relatively effective for simple tasks or cases where user information needs are clear-cut. However, for complex tasks or tasks involving the generation of lengthy text, such as long-form question answering, multi-hop reasoning, chain-of-thought reasoning, etc., relying solely on the user's initial input for retrieval

![](images/e356e0195b016a8c58eb38270fb9c3b74ee2b87d6bcf5ba5d22e20b273c06353.jpg)  
Figure 1: An illustration of our DRAGIN framework.

may not adequately cover all the external knowledge that the model requires (Jiang et al., 2023). Therefore, some researchers have begun to explore multi-round retrieval augmentation. For example, RETRO (Borgeaud et al., 2022) and IC-ralm (Ram et al., 2023) trigger retrieval every 4 to 32 tokens, and IRCot (Trivedi et al., 2022) triggers retrieval every sentence. However, solely relying on fixed interval-based retrieval without considering the information needs of the LLM itself could produce suboptimal results. Inspired by this, FLARE (Jiang et al., 2023) triggers retrieval when encountering an uncertain token. Specifically, if any token in the generated text has a probability lower than a certain threshold, the retrieval module is triggered.

# 3 Methodology

In this section, we introduce the DRAGIN framework in detail. DRAGIN consists of two components: Real-time Information Needs Detection (RIND) and Query Formulation based on Self-attention (QFS), as illustrated in Figure 1. We introduce RIND in section 3.1, and QFS in sec

tion 3.2.

# 3.1 Real-time Information Need Detection

As discussed above, most existing dynamic RAG frameworks trigger the retrieval module based on static, predefined rules. To the best of our knowledge, the only notable exception is FLARE (Jiang et al., 2023) which triggers retrieval dynamically when the LLM's confidence (i.e., the generation probability) on the next token is lower than certain thresholds.

However, the necessity of retrieval augmentation not only depends on the generation confidence, but also depends on the importance of the token, the semantic of the token, and the influence of each token on subsequent tokens. To address the limitations of the existing approaches, we propose an enhanced approach for triggering retrieval within dynamic RAG frameworks, named Real-time Information Needs Detection (RIND). This method refines the retrieval activation process by evaluating not only the uncertainty of each token, but also its semantic contribution and the impact on the following context.

RIND begins by quantifying the uncertainty of each token generated during the LLM's inference process. This is accomplished by recording the entropy of the token's probability distribution across the vocabulary. Consider an output sequence generated by an LLM, denoted as $T = \{t_1, t_2, \dots, t_n\}$ , with each $t_i$ representing an individual token within the sequence at position $i$ . For any token $t_i$ , the entropy $\mathcal{H}_i$ is computed as follows:

$$
\mathcal {H} _ {i} = - \sum_ {v \in \mathcal {V}} p _ {i} (v) \log p _ {i} (v), \tag {1}
$$

where $p_i(v)$ denotes the probability of generating the token $v$ over all tokens in the vocabulary $\mathcal{V}$ at position $i$ . This measurement of uncertainty serves as the first dimension in our multi-faceted evaluation of tokens.

In addition, RIND leverages the self-attention mechanism inherent in Transformer-based LLMs to allocate weights to tokens, which represent the tokens' impact on the subsequent context. Specifically, for any given token $t_i$ , we quantify its influence by recording the maximum attention value $a_{\max}(i)$ , which records the maximum attention from all following tokens<sup>2</sup>. The attention value

$A_{i,j}$ between two tokens $t_i$ and $t_j$ ( $i < j$ ) is computed as follows:

$$
A _ {i, j} = \operatorname {s o f t m a x} \left(\frac {Q _ {i} K _ {j} ^ {T}}{\sqrt {d _ {k}}}\right), \tag {2}
$$

where $Q_{i}$ represents the query vector of token $t_i$ , $K_j$ is the key vector of token $t_j$ , and $d_k$ denotes the dimensionality of the key vector. The softmax function is applied to the dot product of $Q_{i}$ and $K_j$ , normalized by the square root of $d_k$ . Following this, the maximum attention value $a_{\max}(i)$ for token $t_i$ is identified by locating the highest $A_{i,j}$ for all $j > i$ :

$$
a _ {\max } (i) = \max  _ {j > i} A _ {i, j} \tag {3}
$$

Consider the semantic contribution of each token, RIND employs a binary semantic indicator to filter out stopwords, thus concentrating on tokens with significant semantic value:

$$
s _ {i} = \left\{ \begin{array}{l l} 0, & \text {i f} t _ {i} \in S \\ 1, & \text {o t h e r w i s e} \end{array} , \right. \tag {4}
$$

where $S$ is the stopwords set, $s_i$ is the semantic contribution score of the token $t_i$ . This process ensures that only semantically potent tokens contribute to the retrieval decision-making process.

Combining uncertainty, significance, and semantics, RIND computes a comprehensive score for each token $t_i$ :

$$
\mathcal {S} _ {R I N D} \left(t _ {i}\right) = \mathcal {H} _ {i} \cdot a _ {\max } (i) \cdot s _ {i} \tag {5}
$$

Let $T = \{t_1, t_2, \ldots, t_n\}$ represent the set of tokens already generated by the LLM. The retrieval module activates when the score $S_{RIND}(t_i)$ for any token exceeds a predefined threshold, $\theta$ .

# 3.2 Query Formulation based on Self-attention

Once the position to conduct retrieval augmentation is determined, the next step in the RAG framework is to formulate a query to retrieve necessary information from external databases for the continued generation of LLMs. In the existing dynamic RAG frameworks, all the query formulation methods limit their focus to the LLM's most recent sentence or the last few tokens. This narrow scope fails to adequately cater to the model's real-time information needs, which may span across the entire context. To overcome the shortcomings of these approaches, we propose a novel strategy that utilizes the self-attention mechanisms inherent in Transformer-based LLMs. Our method, termed "Query Formulation based on Self-Attention" (QFS), seeks to

ascertain the LLM's information needs more precisely by examining its understanding of the full context.

Consider an output sequence generated by an LLM, denoted as $T = \{t_1, t_2, \dots, t_n\}$ , with each $t_i$ representing an individual token within the sequence. Suppose the RIND module identifies the token at position $i$ , which requires external knowledge and triggers the retrieval module. The QFS approach then focuses on this specific position to formulate a query. For the token at position $i$ , the QFS method evaluates the attention weights across the preceding token sequence $\{t_{i-1}, t_{i-2}, \dots, t_1\}$ . Since the generation of $t_i$ by the LLM is based on its interpretation of the entire preceding context, the attention weights reflect the model's self-assessed importance of each token in generating $t_i$ . The QFS method prioritizes these tokens based on their attention scores, selecting the top $n$ tokens to construct the query. The query formulation process includes the following steps: (1) Extract the attention scores of the last Transformer layer $A_i = \{a_{i,1}, a_{i,2}, \dots, a_{i,i-1}\}$ for each token $t_i$ in $T$ , where $a_{i,j}$ represents the attention score assigned by $t_i$ to $t_j$ ; (2) Sort $A_i$ in descending order to identify the top $n$ tokens with the highest attention scores; (3) Find the words corresponding to these tokens from the vocabulary and arrange them according to their original order in the text; (4) Construct the query $Q_i$ using the words from these top $n$ tokens, ensuring the query reflects the most relevant aspects of the context as determined by the LLM's self-attention mechanism.

# 3.3 Continue Generation after Retrieval

Once the RIND module detects the position $i$ that needs external knowledge, the QFS module creates the query and utilizes an off-the-shelf retrieval model (e.g. BM25) to retrieve relevant information from external knowledge bases. Suppose the retrieved documents are denoted as $D i_{1}$ , $D i_{2}$ , and $D i_{3}$ . Upon successful retrieval, the next step of the dynamic RAG framework is to integrate this external knowledge into the LLM's generation process. This integration begins with truncating the LLM's output at the identified position $i$ for retrieval augmentation:

$$
T ^ {\prime} = \operatorname {t r u n c a t e} \left(T, t _ {i}\right), \tag {6}
$$

where $T^{\prime}$ represents the truncated output, $T$ is the original sequence generated by the LLM, and $t_i$ is the token at which the need for external knowledge was identified by RIND. To integrate the re

trieved knowledge $D i_{1}$ , $D i_{2}$ , and $D i_{3}$ , we adopt a meticulously designed prompt template<sup>3</sup>, which is structured as follows:

# The entire input for LLM:

Below are the external knowledge references:

[1] $D i_{1}$   
[2] $D i_{2}$   
[3] $D i_{3}$

Please answer the question based on the external knowledge:

Question: xxx

Answer: T'

At this point, the LLM continues generating content based on the retrieved external knowledge and the truncated output $T'$ . Following the integration of the retrieved knowledge, the LLM resumes generating content from the truncation point, enhanced with additional information. This procedure allows the LLM to bridge the previously identified knowledge gap, facilitating a more informed and precise continuation of its output.

Suppose at a subsequent position $j$ where RIND detects again that the LLM requires external knowledge. In that case, the QFS module is triggered again at position $j$ to generate a new query, retrieving a new set of documents $Dj_{1}$ , $Dj_{2}$ , and $Dj_{3}$ to replace $Di_{1}$ , $Di_{2}$ , and $Di_{3}$ . The LLM will then continue generating from position $j$ based on the newly retrieved documents, following the same process.

# 4 Experimental Setup

# 4.1 Datasets

We choose two MultihopQA datasets 2WikiMultihopQA (Ho et al., 2020) and HotpotQA (Yang et al., 2018) to evaluate the RAG framework's ability to answer complex questions that require multihop reasoning. We choose the IIRC (Ferguson et al., 2020) dataset to evaluate the RAG framework's ability in reading comprehension tasks. Furthermore, we utilize the StrategyQA (Geva et al., 2021) dataset to evaluate the commonsense reasoning capabilities of DRAGIN and other baselines.

# 4.2 Settings for each Dataset

- 2WikiMultihopQA (Ho et al., 2020). We follow the setting of (Wang et al., 2022) to generate both chain-of-thought (CoT) reasoning process as well as the final answer. We follow the prompt template of (Trivedi et al., 2022) and (Jiang et al., 2023). For the evaluation metrics, we extract the final answer from the generated output using pattern matching techniques. The extracted answer is then compared with the reference answer, utilizing methods such as exact match at the answer level, along with token-level measurements of F1 score and precision.   
- HotpotQA (Yang et al., 2018). We follow the setting and the prompt template of (Trivedi et al., 2022) to generate both chain-of-thought (CoT) reasoning process as well as the final answer. Our evaluation metric on this dataset is the same as 2WikiMultihopQA.   
- StrategyQA (Geva et al., 2021). We follow the setting of (Wei et al., 2022) to generate both the CoT reasoning process as well as the final answer. We follow the prompt template of (Wei et al., 2022) and (Jiang et al., 2023). For the evaluation metrics, the obtained yes/no response is extracted and compared with the standard correct answer using an exact match approach.   
- IIRC (Ferguson et al., 2020). We follow the setting and the prompt template of (Trivedi et al., 2022) to generate the final answer. Our evaluation metric on this dataset is the same as 2Wiki-MultihopQA.

Besides the settings introduced in this section, the specific prompt templates corresponding to each dataset are presented in Appendix F. Appendix A provides more detailed descriptions of each dataset's settings.

# 4.3 Baselines

We choose the following Text Generation baselines for comparison. Following the setting of FLARE (Jiang et al., 2023), we implemented the existing multi-round RAG frameworks using the same settings, with the only variation being the timing of triggering retrieval (when to retrieve) and the query formulation method when the retrieval is triggered (what to retrieve).

- wo-RAG. LLM provides direct answers to questions without RAG.

Table 1: A comparative overview of our selected Retrieval-Augmented Generation baselines.   

<table><tr><td></td><td>Timing for Retrieval</td><td>Query Formulation</td></tr><tr><td>SR-RAG</td><td>Before Generation</td><td>Initial Input</td></tr><tr><td>FL-RAG</td><td>Per n Tokens</td><td>Last Generated Tokens</td></tr><tr><td>FS-RAG</td><td>Per Sentence</td><td>Last Generated Sentence</td></tr><tr><td>FLARE</td><td>Any token&#x27;s probability below the threshold</td><td>Last generated Sentence exclude low-probability tokens</td></tr><tr><td>DRAGIN</td><td>Generated token&#x27;s importance and uncertainty</td><td>LLM&#x27;s attention over the entire context</td></tr></table>

- SR-RAG (Single-round RAG). Relevant passages are retrieved from an external corpus based on the initial question. The retrieved passages are then added into the LLM's input.   
- FL-RAG (Fix Length RAG) (Khandelwal et al., 2019; Borgeaud et al., 2022; Ram et al., 2023). A multi-round retrieval augmentation method that triggers the retrieval module every $n$ tokens. The tokens generated in the previous token window are utilized as the query.   
- FS-RAG (Fix Sentence RAG) (Trivedi et al., 2022). A multi-round retrieval augmentation method that triggers the retrieval module every sentence. The last generated sentence are utilized as the query.   
- FLARE (Jiang et al., 2023). A multi-round retrieval augmentation method that triggers retrieval each time it encounters an uncertain token. When the retrieval module is triggered, the last generated sentence without the uncertain tokens are defines as the query.

To illustrate the differences between DRAGIN and other dynamic RAG baselines directly, we present a comparison of retrieval timing and query formation methods for each dynamic RAG frameworks in Table 1.

# 4.4 Selected LLMs

To validate the effectiveness of DRAGIN and other RAG baselines, we conducted experiments with the following LLMs:

- LLaMA-2-Chat (Touvron et al., 2023b). LLaMA2 is a collection of pre-trained and finetuned LLMs. This series includes fine-tuned LLMs, known as Llama2-Chat, specifically designed for optimal performance in dialogue

based applications. We choose LLaMA-2-Chat7B and LLaMA-2-Chat-13B.

- Vicuna-13B-v1.5 (Chiang et al., 2023) is a collection of open-source chatbots fine-tuned from LLaMA using user-shared conversations gathered from ShareGPT. We have selected the latest versions of Vicuna, namely Vicuna-13B-v1.5.

# 4.5 Implementation Details

Hyperparameter: The hyperparameters are all presented in Appendix E.

Retriever: We adopt BM25 as our retrieval model based on findings from (Ram et al., 2023), which demonstrated its superior performance in Retrieval-Augmented Generation, even outperforming some dense retrieval models. We also explored the impact of replacing BM25 with a SOTA dense retrieval method SGPT (Muennighoff, 2022), which is detailed in Section 5.5.

Stopwords: For the identification of stopwords within the RIND module, we utilized the en_core_web_sm language model from the Spacy library, a tool recognized for its effectiveness and efficiency in Natural Language Processing tasks as evidenced by previous research (Shelar et al., 2020).

External Knowledge Corpus: We adopt Wikipedia as our external knowledge corpus. Each article are segmented into 100-token passages.

LLM Configuration: For the selected LLMs, we directly download model parameters from the official Hugging Face repositories for each model, and use the code provided by Hugging Face to conduct text generation. For the generation configuration, we have chosen greedy decoding as the decoding strategy for LLM inference to ensure the reproducibility of our experimental results. However, for practical applications, we recommend using the official default generation configuration provided by each model, as this will yield better performance.

# 5 Experimental Results

# 5.1 Overall Results of DRAGIN and Baselines

Our experiments comprehensively evaluate the performance of DRAGIN against various baselines across four benchmark datasets: 2WikiMultihopQA, HotpotQA, StrategyQA, and IIRC. The results, summarized in Table 2, underscore several critical insights: (1) The integration of single-round retrieval augmentation consistently boosts LLMs' performance across all datasets when compared

Table 2: The overall experimental results of DRAGIN and other baselines on four benchmarks. The best results are in bold.   

<table><tr><td colspan="2"></td><td colspan="2">2WikiMultihopQA</td><td colspan="2">HotpotQA</td><td>StrategyQA</td><td colspan="2">IIRC</td></tr><tr><td>LLM</td><td>RAG Method</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>Accuracy</td><td>EM</td><td>F1</td></tr><tr><td rowspan="6">Llama2-13b-chat</td><td>wo-RAG</td><td>0.187</td><td>0.2721</td><td>0.223</td><td>0.3097</td><td>0.650</td><td>0.168</td><td>0.2039</td></tr><tr><td>SR-RAG</td><td>0.245</td><td>0.3364</td><td>0.263</td><td>0.3706</td><td>0.654</td><td>0.196</td><td>0.2303</td></tr><tr><td>FL-RAG</td><td>0.217</td><td>0.3054</td><td>0.177</td><td>0.2682</td><td>0.648</td><td>0.155</td><td>0.1875</td></tr><tr><td>FS-RAG</td><td>0.270</td><td>0.3610</td><td>0.267</td><td>0.3715</td><td>0.655</td><td>0.171</td><td>0.2061</td></tr><tr><td>FLARE</td><td>0.224</td><td>0.3076</td><td>0.180</td><td>0.2756</td><td>0.655</td><td>0.138</td><td>0.1667</td></tr><tr><td>DRAGIN (Ours)</td><td>0.304</td><td>0.3931</td><td>0.314</td><td>0.4238</td><td>0.689</td><td>0.185</td><td>0.2221</td></tr><tr><td rowspan="6">Llama2-7b-chat</td><td>wo-RAG</td><td>0.146</td><td>0.2232</td><td>0.184</td><td>0.2745</td><td>0.659</td><td>0.139</td><td>0.1731</td></tr><tr><td>SR-RAG</td><td>0.169</td><td>0.2549</td><td>0.164</td><td>0.2499</td><td>0.645</td><td>0.187</td><td>0.2258</td></tr><tr><td>FL-RAG</td><td>0.112</td><td>0.1922</td><td>0.146</td><td>0.2107</td><td>0.635</td><td>0.172</td><td>0.2023</td></tr><tr><td>FS-RAG</td><td>0.189</td><td>0.2652</td><td>0.214</td><td>0.3035</td><td>0.629</td><td>0.178</td><td>0.2157</td></tr><tr><td>FLARE</td><td>0.143</td><td>0.2134</td><td>0.149</td><td>0.2208</td><td>0.627</td><td>0.136</td><td>0.1644</td></tr><tr><td>DRAGIN (Ours)</td><td>0.220</td><td>0.2926</td><td>0.232</td><td>0.3344</td><td>0.641</td><td>0.192</td><td>0.2336</td></tr><tr><td rowspan="6">Vicuna-13b-v1.5</td><td>wo-RAG</td><td>0.146</td><td>0.2232</td><td>0.228</td><td>0.3256</td><td>0.682</td><td>0.175</td><td>0.2149</td></tr><tr><td>SR-RAG</td><td>0.170</td><td>0.2564</td><td>0.254</td><td>0.3531</td><td>0.686</td><td>0.217</td><td>0.2564</td></tr><tr><td>FL-RAG</td><td>0.135</td><td>0.2133</td><td>0.187</td><td>0.3039</td><td>0.645</td><td>0.0985</td><td>0.1285</td></tr><tr><td>FS-RAG</td><td>0.188</td><td>0.2625</td><td>0.185</td><td>0.3216</td><td>0.622</td><td>0.1027</td><td>0.1344</td></tr><tr><td>FLARE</td><td>0.157</td><td>0.2257</td><td>0.092</td><td>0.1808</td><td>0.599</td><td>0.1174</td><td>0.1469</td></tr><tr><td>DRAGIN (Ours)</td><td>0.252</td><td>0.3516</td><td>0.288</td><td>0.4164</td><td>0.687</td><td>0.2233</td><td>0.2652</td></tr></table>

to direct question answering, confirming the effectiveness of RAG. (2) Despite the overall positive impact of retrieval augmentation, we observe that fixed rules-based retrieval methods, e.g. FL-RAG and FS-RAG, do not always outperform single-round retrieval. This observation validates our hypothesis that retrieval augmentation, if conducted at a wrong position, may not be helpful in improving the quality of LLM's outputs. This underscores the significance of timing in the activation of retrieval processes, which should be tailored to the information needs of Large Language Models (LLMs), activating retrieval only when LLMs necessitate external knowledge. (3) The performance of FLARE varies significantly among different datasets. Interestingly, as shown in our ablation study ( $\S$ 5.4), the query formulation strategies are significantly better than those used by other baselines, but its overall performance is not. This indicates that the timing of retrieval augmentation in FLARE is far from perfect. (4) Our proposed DRAGIN method demonstrates superior performance across most LLMs and datasets. This indicates the robustness and effectiveness of DRAGIN in enhancing LLMs' capabilities. (5) DRAGIN demonstrates more substantial performance improvements in MultihopQA tasks, such as 2WikiMultihopQA and HotpotQA, than in tasks requiring common sense reasoning, like those in the StrategyQA dataset. This difference highlights DRAGIN's specialized capability in managing complex, multi-step reasoning tasks.

Table 3: Comparison of the frequency of retrieval module activation in dynamic RAG frameworks across all datasets. 2WMQA, HQA, SQA indicates 2WikiMulti-hopQA, HotpotQA, StrategyQA respectively.   

<table><tr><td colspan="2"></td><td>2WMQA</td><td>HQA</td><td>SQA</td><td>IIRC</td></tr><tr><td colspan="2"></td><td>#Num</td><td>#Num</td><td>#Num</td><td>#Num</td></tr><tr><td rowspan="4">L13B</td><td>FL-RAG</td><td>3.770</td><td>3.194</td><td>3.626</td><td>3.426</td></tr><tr><td>FS-RAG</td><td>3.131</td><td>4.583</td><td>4.885</td><td>4.305</td></tr><tr><td>FLARE</td><td>1.592</td><td>3.378</td><td>0.625</td><td>5.521</td></tr><tr><td>DRAGIN</td><td>2.631</td><td>3.505</td><td>4.786</td><td>2.829</td></tr><tr><td rowspan="4">L7B</td><td>FL-RAG</td><td>3.342</td><td>3.809</td><td>3.757</td><td>2.839</td></tr><tr><td>FS-RAG</td><td>3.833</td><td>4.152</td><td>4.546</td><td>4.210</td></tr><tr><td>FLARE</td><td>0.941</td><td>1.064</td><td>1.271</td><td>1.095</td></tr><tr><td>DRAGIN</td><td>2.836</td><td>3.013</td><td>4.629</td><td>2.927</td></tr><tr><td rowspan="4">V13B</td><td>FL-RAG</td><td>4.199</td><td>3.564</td><td>3.591</td><td>3.189</td></tr><tr><td>FS-RAG</td><td>3.720</td><td>5.701</td><td>6.820</td><td>6.032</td></tr><tr><td>FLARE</td><td>1.093</td><td>1.078</td><td>1.118</td><td>0.335</td></tr><tr><td>DRAGIN</td><td>2.542</td><td>3.184</td><td>3.744</td><td>3.120</td></tr></table>

# 5.2 Efficiency

In this section, we investigate the efficiency of various dynamic RAG frameworks across multiple datasets. We measure efficiency based on the number of retrieval calls made, as outlined in Table 3. Due to the special design of FS-RAG, the #NUM for FS-RAG also indicates the average number of sentences produced by the LLM in response to queries on this dataset. Among the evaluated frameworks, FLARE stood out for its efficiency, requiring the fewest retrieval calls across all datasets. DRAGIN followed closely, with fewer retrieval

Table 4: The influence of the 'When to Retrieve' decision on various dynamic RAG frameworks, with the IIRC dataset as the evaluation benchmark. The best results are in bold. L13B indicates LLaMA2-13B-Chat, V13B indicates Vicuna-13b-v1.5. We fix the query formulation method, the last complete sentence generated by the LLM is selected as the query for all the baselines.   

<table><tr><td></td><td></td><td>EM</td><td>F1</td><td>Prec.</td></tr><tr><td rowspan="4">L13B</td><td>FLARE</td><td>0.128</td><td>0.1599</td><td>0.1677</td></tr><tr><td>FL-RAG</td><td>0.155</td><td>0.1875</td><td>0.1986</td></tr><tr><td>FS-RAG</td><td>0.171</td><td>0.2061</td><td>0.2185</td></tr><tr><td>DRAGIN</td><td>0.187</td><td>0.2242</td><td>0.2319</td></tr><tr><td rowspan="4">V13B</td><td>FLARE</td><td>0.097</td><td>0.1277</td><td>0.1324</td></tr><tr><td>FL-RAG</td><td>0.099</td><td>0.1285</td><td>0.1324</td></tr><tr><td>FS-RAG</td><td>0.103</td><td>0.1344</td><td>0.1358</td></tr><tr><td>DRAGIN</td><td>0.196</td><td>0.2367</td><td>0.2476</td></tr></table>

calls than FS-RAG and FL-RAG.

# 5.3 Timing of Retrieval

In this subsection, we investigate the impact of the timing of retrieval on the performance of dynamic RAG frameworks. Specifically, we fixed the method of query formulation to use the last complete sentence generated by the LLM as the query, and varied the timing of retrieval as the only variable. We examined DRAGIN alongside three existing frameworks: FLARE, FL-RAG, and FS-RAG on the IIRC dataset. As shown in Table 4, DRAGIN consistently outperforms all other dynamic RAG methods. This highlights the effectiveness of our novel approach to determining the optimal moment for retrieval. DRAGIN's superior performance suggests that its method for detecting the real-time information needs of LLMs and triggering retrieval accordingly is particularly adept at enhancing the quality of the generated text.

We also evaluate the impact of varying threshold values within the RIND module on the performance of the DRAGIN framework. We present the experimental results on the HotpotQA dataset in Table 5. Our experimental results show that DRAGIN's performance remains stable across threshold settings, indicating a low sensitivity to changes in this hyperparameter. The threshold value is pivotal in determining the retrieval module's activation frequency. As the threshold increases, the frequency of the retrieval module's activation decreases, suggesting that adjusting the threshold can strike a balance between the system's efficiency and the accuracy of its outputs in practical applications.

Table 5: Comparison between different threshold of RIND for LLaMA-13B-Chat model on the HotpotQA dataset. The best results are in bold.   

<table><tr><td>threshold</td><td>EM</td><td>F1</td><td>Prec.</td></tr><tr><td>0.3</td><td>0.295</td><td>0.3856</td><td>0.3873</td></tr><tr><td>0.4</td><td>0.297</td><td>0.387</td><td>0.389</td></tr><tr><td>0.5</td><td>0.299</td><td>0.3897</td><td>0.3915</td></tr><tr><td>0.6</td><td>0.304</td><td>0.3931</td><td>0.3946</td></tr><tr><td>0.7</td><td>0.304</td><td>0.3927</td><td>0.3937</td></tr><tr><td>0.8</td><td>0.301</td><td>0.392</td><td>0.3927</td></tr><tr><td>0.9</td><td>0.301</td><td>0.3944</td><td>0.3947</td></tr><tr><td>1</td><td>0.293</td><td>0.3869</td><td>0.3875</td></tr></table>

Table 6: The influence of the query formulation methods on various dynamic RAG frameworks, with the HotpotQA dataset as the evaluation benchmark. The best results are in bold. L13B indicates LLaMA2-13B-Chat, V13B indicates Vicuna-13b-v1.5.   

<table><tr><td></td><td></td><td>EM</td><td>F1</td><td>Prec.</td></tr><tr><td rowspan="5">L13B</td><td>FLARE</td><td>0.262</td><td>0.3674</td><td>0.3792</td></tr><tr><td>Full Context</td><td>0.252</td><td>0.3584</td><td>0.3711</td></tr><tr><td>FS-RAG</td><td>0.255</td><td>0.3574</td><td>0.3685</td></tr><tr><td>FL-RAG</td><td>0.241</td><td>0.3394</td><td>0.3495</td></tr><tr><td>DRAGIN</td><td>0.314</td><td>0.4238</td><td>0.4401</td></tr><tr><td rowspan="5">V13B</td><td>FLARE</td><td>0.225</td><td>0.3366</td><td>0.3420</td></tr><tr><td>Full Context</td><td>0.221</td><td>0.3402</td><td>0.3457</td></tr><tr><td>FS-RAG</td><td>0.216</td><td>0.3432</td><td>0.3507</td></tr><tr><td>FL-RAG</td><td>0.214</td><td>0.3268</td><td>0.3264</td></tr><tr><td>DRAGIN</td><td>0.288</td><td>0.4164</td><td>0.4226</td></tr></table>

# 5.4 Query Formulation

This subsection delves into the impact of query formulation techniques on the performance of dynamic RAG frameworks. We standardize the timing of trigger retrieval to RIND, which is proven to be the most effective timing based on the experimental results detailed in section 5.3. We focus on the comparison between DRAGIN and three existing frameworks: FLARE, FL-RAG, and FS-RAG. The query formulation method of FLARE is the last generated sentence excludes low-probability tokens. FL-RAG selects the closest 25 tokens to this position as the query. FS-RAG selects the sentence before this position as the query. We also evaluate the effectiveness of using the full context as the query. As shown in Table 6, DRAGIN's query formulation method performs best among all the dynamic RAG frameworks. FLARE emerged as the second most effective query formulation method, outperforming the FS-RAG and FL-RAG methods. Moreover, leveraging the entire context as a query did not yield optimal results, indicating potential redundancy within the full context. This finding

Table 7: Comparison of performance between BM25 and SGPT using the LLaMA2-13B-Chat model. The method with better performance is highlighted.   

<table><tr><td></td><td>retriever</td><td>EM</td><td>F1</td><td>Prec.</td></tr><tr><td rowspan="2">2WMQA</td><td>BM25</td><td>0.304</td><td>0.393</td><td>0.395</td></tr><tr><td>SGPT</td><td>0.273</td><td>0.356</td><td>0.357</td></tr><tr><td rowspan="2">HQA</td><td>BM25</td><td>0.314</td><td>0.424</td><td>0.437</td></tr><tr><td>SGPT</td><td>0.264</td><td>0.371</td><td>0.388</td></tr><tr><td rowspan="2">IIRC</td><td>BM25</td><td>0.185</td><td>0.222</td><td>0.235</td></tr><tr><td>SGPT</td><td>0.169</td><td>0.201</td><td>0.207</td></tr></table>

validates the effectiveness of our proposed QFS method, which aims to select tokens from the context that can represent the real-time information needs of the LLM as the query.

# 5.5 Impact of Retriever

In the dynamic RAG paradigm, the choice of the retriever plays an important role in retrieving relevant passages from a corpus based on a given query. In the field of information retrieval, the two popular types of retrieval methods are lexical matching (Zhai, 2008; Robertson et al., 2009) and dense retrieval (Su et al., 2023b; Gao and Callan, 2021; Su et al., 2023a; Muennighoff, 2022; Li et al., 2023b; Ma et al., 2023; Ye et al., 2024; Su et al., 2023c; Li et al., 2023a; Chen et al., 2023, 2022; Li et al., 2023c; Fang et al., 2024). Among lexical matching techniques, BM25 stands out for its widespread adoption and effectiveness (Robertson et al., 2009). Conversely, among existing dense retrieval methods, none has achieved the widespread popularity of BM25. We have opted for SGPT, which has recently attained state-of-the-art performance across a variety of datasets. (Muennighoff, 2022).

In our experimental analysis presented in Table 7, we found that BM25 consistently surpasses SGPT in performance across various datasets within the dynamic RAG framework, despite SGPT's generally better performance in numerous information retrieval tasks. This outcome aligns with the findings of prior research, such as the study by (Ram et al., 2023), which underscored BM25's effectiveness in RAG tasks. These results indicate that despite progress in dense retrieval technologies like SGPT, the simpler, lexicon-based BM25 algorithm is still a strong baseline for enhancing the performance of LLM in RAG tasks.

# 6 Conclusions and Future Works

In this work, we propose DRAGIN, a dynamic RAG framework tailored to address the real-time information needs of LLMs during text generation. By integrating RIND for timely retrieval activation and QFS for precise query formulation, DRAGIN significantly outperforms existing dynamic RAG methods across various knowledge-intensive benchmarks.

# 7 Limitations

We acknowledge the limitations of this paper. One of the primary limitations is the reliance on the self-attention mechanism of Transformer-based LLMs for both Real-time Information Needs Detection (RIND) and Query Formulation based on Self-attention (QFS). While self-attention scores are accessible for all open-source LLMs, it's important to note that our method is not applicable to certain APIs that do not provide access to the self-attention scores. Thus, our future work aims to develop more methods to overcome this constraint.

# 8 Ethics Statement

In conducting this research, we have prioritized ethical considerations at every stage to ensure the responsible development and application of AI technologies. Our research does not rely on personally identifiable information or require manually annotated datasets. We firmly believe in the principles of open research and the scientific value of reproducibility. To this end, we have made all models, data, and code associated with our paper publicly available on GitHub. This transparency not only facilitates the verification of our findings by the community but also encourages the application of our methods in other contexts.

