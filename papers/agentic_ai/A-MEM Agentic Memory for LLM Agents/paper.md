# A-Mem: Agentic Memory for LLM Agents

Wujiang Xu1, Zujie Liang2, Kai Mei1, Hang Gao1, Juntao Tan1, Yongfeng Zhang1,3

1Rutgers University 2Independent Researcher 3AIOS Foundation

wujiang.xu@rutgers.edu

# Abstract

While large language model (LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current memory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems’ fixed operations and structures limit their adaptability across diverse tasks. To address this limitation, this paper proposes a novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way. Following the basic principles of the Zettelkasten method, we designed our memory system to create interconnected knowledge networks through dynamic indexing and linking. When a new memory is added, we generate a comprehensive note containing multiple structured attributes, including contextual descriptions, keywords, and tags. The system then analyzes historical memories to identify relevant connections, establishing links where meaningful similarities exist. Additionally, this process enables memory evolution – as new memories are integrated, they can trigger updates to the contextual representations and attributes of existing historical memories, allowing the memory network to continuously refine its understanding. Our approach combines the structured organization principles of Zettelkasten with the flexibility of agent-driven decision making, allowing for more adaptive and context-aware memory management. Empirical experiments on six foundation models show superior improvement against existing SOTA baselines.

$\ Q$ Code for Benchmark Evaluation:

https://github.com/WujiangXu/AgenticMemory

$\ Q$ Code for Production-ready Agentic Memory:

https://github.com/WujiangXu/A-mem-sys

# 1 Introduction

Large Language Model (LLM) agents have demonstrated remarkable capabilities in various tasks, with recent advances enabling them to interact with environments, execute tasks, and make decisions autonomously [23, 33, 7]. They integrate LLMs with external tools and delicate workflows to improve reasoning and planning abilities. Though LLM agent has strong reasoning performance, it still needs a memory system to provide long-term interaction ability with the external environment [35].

Existing memory systems [25, 39, 28, 21] for LLM agents provide basic memory storage functionality. These systems require agent developers to predefine memory storage structures, specify storage points within the workflow, and establish retrieval timing. Meanwhile, to improve structured memory organization, Mem0 [8], following the principles of RAG [9, 18, 30], incorporates graph databases for storage and retrieval processes. While graph databases provide structured organization for memory systems, their reliance on predefined schemas and relationships fundamentally limits their adaptability. This limitation manifests clearly in practical scenarios - when an agent learns a novel mathematical solution, current systems can only categorize and link this information within their preset framework,

![](images/6c4ed1f823679c367219639e211e1c7d1c5efd47f18f7a360cf6dc525f8df1d8.jpg)  
(a) Traditional memory system.

![](images/fcd3164730d1b1660f85e5d91f9c14e1fd5d5775224f3c9b6f85d26303fed5fa.jpg)  
(b) Our proposed agentic memory.   
Figure 1: Traditional memory systems require predefined memory access patterns specified in the workflow, limiting their adaptability to diverse scenarios. Contrastly, our A-MEM enhances the flexibility of LLM agents by enabling dynamic memory operations.

unable to forge innovative connections or develop new organizational patterns as knowledge evolves. Such rigid structures, coupled with fixed agent workflows, severely restrict these systems’ ability to generalize across new environments and maintain effectiveness in long-term interactions. The challenge becomes increasingly critical as LLM agents tackle more complex, open-ended tasks, where flexible knowledge organization and continuous adaptation are essential. Therefore, how to design a flexible and universal memory system that supports LLM agents’ long-term interactions remains a crucial challenge.

In this paper, we introduce a novel agentic memory system, named as A-MEM, for LLM agents that enables dynamic memory structuring without relying on static, predetermined memory operations. Our approach draws inspiration from the Zettelkasten method [15, 1], a sophisticated knowledge management system that creates interconnected information networks through atomic notes and flexible linking mechanisms. Our system introduces an agentic memory architecture that enables autonomous and flexible memory management for LLM agents. For each new memory, we construct comprehensive notes, which integrates multiple representations: structured textual attributes including several attributes and embedding vectors for similarity matching. Then A-MEM analyzes the historical memory repository to establish meaningful connections based on semantic similarities and shared attributes. This integration process not only creates new links but also enables dynamic evolution when new memories are incorporated, they can trigger updates to the contextual representations of existing memories, allowing the entire memories to continuously refine and deepen its understanding over time. The contributions are summarized as:

• We present A-MEM, an agentic memory system for LLM agents that enables autonomous generation of contextual descriptions, dynamic establishment of memory connections, and intelligent evolution of existing memories based on new experiences. This system equips LLM agents with long-term interaction capabilities without requiring predetermined memory operations.   
• We design an agentic memory update mechanism where new memories automatically trigger two key operations: link generation and memory evolution. Link generation automatically establishes connections between memories by identifying shared attributes and similar contextual descriptions. Memory evolution enables existing memories to dynamically adapt as new experiences are analyzed, leading to the emergence of higher-order patterns and attributes.   
• We conduct comprehensive evaluations of our system using a long-term conversational dataset, comparing performance across six foundation models using six distinct evaluation metrics, demonstrating significant improvements. Moreover, we provide T-SNE visualizations to illustrate the structured organization of our agentic memory system.

# 2 Related Work

# 2.1 Memory for LLM Agents

Prior works on LLM agent memory systems have explored various mechanisms for memory management and utilization [23, 21, 8, 39]. Some approaches complete interaction storage, which maintains comprehensive historical records through dense retrieval models [39] or read-write memory structures [24]. Moreover, MemGPT [25] leverages cache-like architectures to prioritize recent information. Similarly, SCM [32] proposes a Self-Controlled Memory framework that enhances LLMs’ capability to maintain long-term memory through a memory stream and controller mechanism. However, these approaches face significant limitations in handling diverse real-world tasks. While they can provide basic memory functionality, their operations are typically constrained by predefined structures and fixed workflows. These constraints stem from their reliance on rigid operational

![](images/cfc9947912b7e5b1256f564e6e14fe17d92d8888449b27653ea15289c76343fc.jpg)  
Figure 2: Our A-MEM architecture comprises three integral parts in memory storage. During note construction, the system processes new interaction memories and stores them as notes with multiple attributes. The link generation process first retrieves the most relevant historical memories and then employs an LLM to determine whether connections should be established between them. The concept of a ’box’ describes that related memories become interconnected through their similar contextual descriptions, analogous to the Zettelkasten method. However, our approach allows individual memories to exist simultaneously within multiple different boxes. During the memory retrieval stage, we extract query embeddings using a text encoding model and search the memory database for relevant matches. When related memory is retrieved, similar memories that are linked within the same box are also automatically accessed.

patterns, particularly in memory writing and retrieval processes. Such inflexibility leads to poor generalization in new environments and limited effectiveness in long-term interactions. Therefore, designing a flexible and universal memory system that supports agents’ long-term interactions remains a crucial challenge.

# 2.2 Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) has emerged as a powerful approach to enhance LLMs by incorporating external knowledge sources [18, 6, 10]. The standard RAG [37, 34] process involves indexing documents into chunks, retrieving relevant chunks based on semantic similarity, and augmenting the LLM’s prompt with this retrieved context for generation. Advanced RAG systems [20, 12] have evolved to include sophisticated pre-retrieval and post-retrieval optimizations. Building upon these foundations, recent researches has introduced agentic RAG systems that demonstrate more autonomous and adaptive behaviors in the retrieval process. These systems can dynamically determine when and what to retrieve [4, 14], generate hypothetical responses to guide retrieval, and iteratively refine their search strategies based on intermediate results [31, 29].

However, while agentic RAG approaches demonstrate agency in the retrieval phase by autonomously deciding when and what to retrieve [4, 14, 38], our agentic memory system exhibits agency at a more fundamental level through the autonomous evolution of its memory structure. Inspired by the Zettelkasten method, our system allows memories to actively generate their own contextual descriptions, form meaningful connections with related memories, and evolve both their content and relationships as new experiences emerge. This fundamental distinction in agency between retrieval versus storage and evolution distinguishes our approach from agentic RAG systems, which maintain static knowledge bases despite their sophisticated retrieval mechanisms.

# 3 Methodolodgy

Our proposed agentic memory system draws inspiration from the Zettelkasten method, implementing a dynamic and self-evolving memory system that enables LLM agents to maintain long-term memory without predetermined operations. The system’s design emphasizes atomic note-taking, flexible linking mechanisms, and continuous evolution of knowledge structures.

# 3.1 Note Construction

Building upon the Zettelkasten method’s principles of atomic note-taking and flexible organization, we introduce an LLM-driven approach to memory note construction. When an agent interacts with its environment, we construct structured memory notes that capture both explicit information and LLMgenerated contextual understanding. Each memory note $m _ { i }$ in our collection $\mathcal { M } = \{ m _ { 1 } , m _ { 2 } , . . . , m _ { N } \}$ is represented as:

$$
m _ {i} = \left\{c _ {i}, t _ {i}, K _ {i}, G _ {i}, X _ {i}, e _ {i}, L _ {i} \right\} \tag {1}
$$

where $c _ { i }$ represents the original interaction content, $t _ { i }$ is the timestamp of the interaction, $K _ { i }$ denotes LLM-generated keywords that capture key concepts, $G _ { i }$ contains LLM-generated tags for categorization, $X _ { i }$ represents the LLM-generated contextual description that provides rich semantic understanding, and $L _ { i }$ maintains the set of linked memories that share semantic relationships. To enrich each memory note with meaningful context beyond its basic content and timestamp, we leverage an LLM to analyze the interaction and generate these semantic components. The note construction process involves prompting the LLM with carefully designed templates $P _ { s 1 }$ :

$$
K _ {i}, G _ {i}, X _ {i} \leftarrow \operatorname {L L M} \left(c _ {i} \| t _ {i} \| P _ {s 1}\right) \tag {2}
$$

Following the Zettelkasten principle of atomicity, each note captures a single, self-contained unit of knowledge. To enable efficient retrieval and linking, we compute a dense vector representation via a text encoder [27] that encapsulates all textual components of the note:

$$
e _ {i} = f _ {\mathrm {e n c}} \left[ \operatorname {c o n c a t} \left(c _ {i}, K _ {i}, G _ {i}, X _ {i}\right) \right] \tag {3}
$$

By using LLMs to generate enriched components, we enable autonomous extraction of implicit knowledge from raw interactions. The multi-faceted note structure $( K _ { i } , G _ { i } , X _ { i } )$ creates rich representations that capture different aspects of the memory, facilitating nuanced organization and retrieval. Additionally, the combination of LLM-generated semantic components with dense vector representations provides both context and computationally efficient similarity matching.

# 3.2 Link Generation

Our system implements an autonomous link generation mechanism that enables new memory notes to form meaningful connections without predefined rules. When the constrctd memory note $m _ { n }$ is added to the system, we first leverage its semantic embedding for similarity-based retrieval. For each existing memory note $m _ { j } \in { \mathcal { M } }$ , we compute a similarity score:

$$
s _ {n, j} = \frac {e _ {n} \cdot e _ {j}}{\left| e _ {n} \right| \left| e _ {j} \right|} \tag {4}
$$

The system then identifies the top- $k$ most relevant memories:

$$
\mathcal {M} _ {\text {n e a r}} ^ {n} = \left\{m _ {j} \mid \operatorname {r a n k} \left(s _ {n, j}\right) \leq k, m _ {j} \in \mathcal {M} \right\} \tag {5}
$$

Based on these candidate nearest memories, we prompt the LLM to analyze potential connections based on their potential common attributes. Formally, the link set of memory $m _ { n }$ update like:

$$
L _ {i} \leftarrow \operatorname {L L M} \left(m _ {n} \| \mathcal {M} _ {\text {n e a r}} ^ {n} \| P _ {s 2}\right) \tag {6}
$$

Each generated link $l _ { i }$ is structured as: $L _ { i } = \{ m _ { i } , . . . , m _ { k } \}$ . By using embedding-based retrieval as an initial filter, we enable efficient scalability while maintaining semantic relevance. A-MEM can quickly identify potential connections even in large memory collections without exhaustive comparison. More importantly, the LLM-driven analysis allows for nuanced understanding of relationships that goes beyond simple similarity metrics. The language model can identify subtle patterns, causal relationships, and conceptual connections that might not be apparent from embedding similarity alone. We implements the Zettelkasten principle of flexible linking while leveraging modern language models. The resulting network emerges organically from memory content and context, enabling natural knowledge organization.

# 3.3 Memory Evolution

After creating links for the new memory, A-MEM evolves the retrieved memories based on their textual information and relationships with the new memory. For each memory $m _ { j }$ in the nearest

neighbor set $\mathcal { M } _ { \mathrm { n e a r } } ^ { n }$ , the system determines whether to update its context, keywords, and tags. This evolution process can be formally expressed as:

$$
m _ {j} ^ {*} \leftarrow \operatorname {L L M} \left(m _ {n} \| \mathcal {M} _ {\text {n e a r}} ^ {n} \backslash m _ {j} \| m _ {j} \| P _ {s 3}\right) \tag {7}
$$

The evolved memory $m _ { j } ^ { * }$ then replaces the original memory $m _ { j }$ in the memory set $\mathcal { M }$ . This evolutionary approach enables continuous updates and new connections, mimicking human learning processes. As the system processes more memories over time, it develops increasingly sophisticated knowledge structures, discovering higher-order patterns and concepts across multiple memories. This creates a foundation for autonomous memory learning where knowledge organization becomes progressively richer through the ongoing interaction between new experiences and existing memories.

# 3.4 Retrieve Relative Memory

In each interaction, our A-MEM performs context-aware memory retrieval to provide the agent with relevant historical information. Given a query text $q$ from the current interaction, we first compute its dense vector representation using the same text encoder used for memory notes:

$$
e _ {q} = f _ {\text {e n c}} (q) \tag {8}
$$

The system then computes similarity scores between the query embedding and all existing memory notes in $\mathcal { M }$ using cosine similarity:

$$
s _ {q, i} = \frac {e _ {q} \cdot e _ {i}}{\left| e _ {q} \right| \left| e _ {i} \right|}, \text {w h e r e} e _ {i} \in m _ {i}, \forall m _ {i} \in \mathcal {M} \tag {9}
$$

Then we retrieve the k most relevant memories from the historical memory storage to construct a contextually appropriate prompt.

$$
\mathcal {M} _ {\text {r e t r i e v e d}} = \left\{m _ {i} \mid \operatorname {r a n k} \left(s _ {q, i}\right) \leq k, m _ {i} \in \mathcal {M} \right\} \tag {10}
$$

These retrieved memories provide relevant historical context that helps the agent better understand and respond to the current interaction. The retrieved context enriches the agent’s reasoning process by connecting the current interaction with related past experiences stored in the memory system.

# 4 Experiment

# 4.1 Dataset and Evaluation

To evaluate the effectiveness of instruction-aware recommendation in long-term conversations, we utilize the LoCoMo dataset [22], which contains significantly longer dialogues compared to existing conversational datasets [36, 13]. While previous datasets contain dialogues with around 1K tokens over 4-5 sessions, LoCoMo features much longer conversations averaging 9K tokens spanning up to 35 sessions, making it particularly suitable for evaluating models’ ability to handle long-range dependencies and maintain consistency over extended conversations. The LoCoMo dataset comprises diverse question types designed to comprehensively evaluate different aspects of model understanding: (1) single-hop questions answerable from a single session; (2) multihop questions requiring information synthesis across sessions; (3) temporal reasoning questions testing understanding of time-related information; (4) open-domain knowledge questions requiring integration of conversation context with external knowledge; and (5) adversarial questions assessing models’ ability to identify unanswerable queries. In total, LoCoMo contains 7,512 question-answer pairs across these categories. Besides, we use a new dataset, named DialSim [16], to evaluate the effectiveness of our memory system. It is question-answering dataset derived from long-term multi-party dialogues. The dataset is derived from popular TV shows (Friends, The Big Bang Theory, and The Office), covering 1,300 sessions spanning five years, containing approximately 350,000 tokens, and including more than 1,000 questions per session from refined fan quiz website questions and complex questions generated from temporal knowledge graphs.

For comparison baselines, we compare to LoCoMo [22], ReadAgent [17], MemoryBank [39] and MemGPT [25]. The detailed introduction of baselines can be found in Appendix A.1 For evaluation, we employ two primary metrics: the F1 score to assess answer accuracy by balancing precision and recall, and BLEU-1 [26] to evaluate generated response quality by measuring word overlap

Table 1: Experimental results on LoCoMo dataset of QA tasks across five categories (Multi Hop, Temporal, Open Domain, Single Hop, and Adversial) using different methods. Results are reported in F1 and BLEU-1 $( \% )$ scores. The best performance is marked in bold, and our proposed method A-MEM (highlighted in gray) demonstrates competitive performance across six foundation language models.   

<table><tr><td rowspan="3">Model</td><td rowspan="3">Method</td><td colspan="10">Category</td><td colspan="3">Average</td></tr><tr><td colspan="2">Multi Hop</td><td colspan="2">Temporal</td><td colspan="2">Open Domain</td><td colspan="2">Single Hop</td><td colspan="2">Adversal</td><td colspan="2">Ranking</td><td rowspan="2">Token Length</td></tr><tr><td>F1</td><td>BLEU</td><td>F1</td><td>BLEU</td><td>F1</td><td>BLEU</td><td>F1</td><td>BLEU</td><td>F1</td><td>BLEU</td><td>F1</td><td>BLEU</td></tr><tr><td rowspan="10">GPT</td><td rowspan="5">40-mini</td><td>LOCOMO</td><td>25.02</td><td>19.75</td><td>18.41</td><td>14.77</td><td>12.04</td><td>11.16</td><td>40.36</td><td>29.05</td><td>69.23</td><td>68.75</td><td>2.4</td><td>2.4</td></tr><tr><td>READAGENT</td><td>9.15</td><td>6.48</td><td>12.60</td><td>8.87</td><td>5.31</td><td>5.12</td><td>9.67</td><td>7.66</td><td>9.81</td><td>9.02</td><td>4.2</td><td>4.2</td></tr><tr><td>MEMORYBANK</td><td>5.00</td><td>4.77</td><td>9.68</td><td>6.99</td><td>5.56</td><td>5.94</td><td>6.61</td><td>5.16</td><td>7.36</td><td>6.48</td><td>4.8</td><td>4.8</td></tr><tr><td>MEMGPT</td><td>26.65</td><td>17.72</td><td>25.52</td><td>19.44</td><td>9.15</td><td>7.44</td><td>41.04</td><td>34.34</td><td>43.29</td><td>42.73</td><td>2.4</td><td>2.4</td></tr><tr><td>A-MEM</td><td>27.02</td><td>20.09</td><td>45.85</td><td>36.67</td><td>12.14</td><td>12.00</td><td>44.65</td><td>37.06</td><td>50.03</td><td>49.47</td><td>1.2</td><td>1.2</td></tr><tr><td rowspan="5">40</td><td>LOCOMO</td><td>28.00</td><td>18.47</td><td>9.09</td><td>5.78</td><td>16.47</td><td>14.80</td><td>61.56</td><td>54.19</td><td>52.61</td><td>51.13</td><td>2.0</td><td>2.0</td></tr><tr><td>READAGENT</td><td>14.61</td><td>9.95</td><td>4.16</td><td>3.19</td><td>8.84</td><td>8.37</td><td>12.46</td><td>10.29</td><td>6.81</td><td>6.13</td><td>4.0</td><td>4.0</td></tr><tr><td>MEMORYBANK</td><td>6.49</td><td>4.69</td><td>2.47</td><td>2.43</td><td>6.43</td><td>5.30</td><td>8.28</td><td>7.10</td><td>4.42</td><td>3.67</td><td>5.0</td><td>5.0</td></tr><tr><td>MEMGPT</td><td>30.36</td><td>22.83</td><td>17.29</td><td>13.18</td><td>12.24</td><td>11.87</td><td>60.16</td><td>53.35</td><td>34.96</td><td>34.25</td><td>2.4</td><td>2.4</td></tr><tr><td>A-MEM</td><td>32.86</td><td>23.76</td><td>39.41</td><td>31.23</td><td>17.10</td><td>15.84</td><td>48.43</td><td>42.97</td><td>36.35</td><td>35.53</td><td>1.6</td><td>1.6</td></tr><tr><td rowspan="10">Qwen2.5</td><td rowspan="5">1.5b</td><td>LOCOMO</td><td>9.05</td><td>6.55</td><td>4.25</td><td>4.04</td><td>9.91</td><td>8.50</td><td>11.15</td><td>8.67</td><td>40.38</td><td>40.23</td><td>3.4</td><td>3.4</td></tr><tr><td>READAGENT</td><td>6.61</td><td>4.93</td><td>2.55</td><td>2.51</td><td>5.31</td><td>12.24</td><td>10.13</td><td>7.54</td><td>5.42</td><td>27.32</td><td>4.6</td><td>4.6</td></tr><tr><td>MEMORYBANK</td><td>11.14</td><td>8.25</td><td>4.46</td><td>2.87</td><td>8.05</td><td>6.21</td><td>13.42</td><td>11.01</td><td>36.76</td><td>34.00</td><td>2.6</td><td>2.6</td></tr><tr><td>MEMGPT</td><td>10.44</td><td>7.61</td><td>4.21</td><td>3.89</td><td>13.42</td><td>11.64</td><td>9.56</td><td>7.34</td><td>31.51</td><td>28.90</td><td>3.4</td><td>3.4</td></tr><tr><td>A-MEM</td><td>18.23</td><td>11.94</td><td>24.32</td><td>19.74</td><td>16.48</td><td>14.31</td><td>23.63</td><td>19.23</td><td>46.00</td><td>43.26</td><td>1.0</td><td>1.0</td></tr><tr><td rowspan="5">3b</td><td>LOCOMO</td><td>4.61</td><td>4.29</td><td>3.11</td><td>2.71</td><td>4.55</td><td>5.97</td><td>7.03</td><td>5.69</td><td>16.95</td><td>14.81</td><td>3.2</td><td>3.2</td></tr><tr><td>READAGENT</td><td>2.47</td><td>1.78</td><td>3.01</td><td>3.01</td><td>5.57</td><td>5.22</td><td>3.25</td><td>2.51</td><td>15.78</td><td>14.01</td><td>4.2</td><td>4.2</td></tr><tr><td>MEMORYBANK</td><td>3.60</td><td>3.39</td><td>1.72</td><td>1.97</td><td>6.63</td><td>6.58</td><td>4.11</td><td>3.32</td><td>13.07</td><td>10.30</td><td>4.2</td><td>4.2</td></tr><tr><td>MEMGPT</td><td>5.07</td><td>4.31</td><td>2.94</td><td>2.95</td><td>7.04</td><td>7.10</td><td>7.26</td><td>5.52</td><td>14.47</td><td>12.39</td><td>2.4</td><td>2.4</td></tr><tr><td>A-MEM</td><td>12.57</td><td>9.01</td><td>27.59</td><td>25.07</td><td>7.12</td><td>7.28</td><td>17.23</td><td>13.12</td><td>27.91</td><td>25.15</td><td>1.0</td><td>1.0</td></tr><tr><td rowspan="10">Llama 3.2</td><td rowspan="5">1b</td><td>LOCOMO</td><td>11.25</td><td>9.18</td><td>7.38</td><td>6.82</td><td>11.90</td><td>10.38</td><td>12.86</td><td>10.50</td><td>51.89</td><td>48.27</td><td>3.4</td><td>3.4</td></tr><tr><td>READAGENT</td><td>5.96</td><td>5.12</td><td>1.93</td><td>2.30</td><td>12.46</td><td>11.17</td><td>7.75</td><td>6.03</td><td>44.64</td><td>40.15</td><td>4.6</td><td>4.6</td></tr><tr><td>MEMORYBANK</td><td>13.18</td><td>10.03</td><td>7.61</td><td>6.27</td><td>15.78</td><td>12.94</td><td>17.30</td><td>14.03</td><td>52.61</td><td>47.53</td><td>2.0</td><td>2.0</td></tr><tr><td>MEMGPT</td><td>9.19</td><td>6.96</td><td>4.02</td><td>4.79</td><td>11.14</td><td>8.24</td><td>10.16</td><td>7.68</td><td>49.75</td><td>45.11</td><td>4.0</td><td>4.0</td></tr><tr><td>A-MEM</td><td>19.06</td><td>11.71</td><td>17.80</td><td>10.28</td><td>17.55</td><td>14.67</td><td>28.51</td><td>24.13</td><td>58.81</td><td>54.28</td><td>1.0</td><td>1.0</td></tr><tr><td rowspan="5">3b</td><td>LOCOMO</td><td>6.88</td><td>5.77</td><td>4.37</td><td>4.40</td><td>10.65</td><td>9.29</td><td>8.37</td><td>6.93</td><td>30.25</td><td>28.46</td><td>2.8</td><td>2.8</td></tr><tr><td>READAGENT</td><td>2.47</td><td>1.78</td><td>3.01</td><td>3.01</td><td>5.57</td><td>5.22</td><td>3.25</td><td>2.51</td><td>15.78</td><td>14.01</td><td>4.2</td><td>4.2</td></tr><tr><td>MEMORYBANK</td><td>6.19</td><td>4.47</td><td>3.49</td><td>3.13</td><td>4.07</td><td>4.57</td><td>7.61</td><td>6.03</td><td>18.65</td><td>17.05</td><td>3.2</td><td>3.2</td></tr><tr><td>MEMGPT</td><td>5.32</td><td>3.99</td><td>2.68</td><td>2.72</td><td>5.64</td><td>5.54</td><td>4.32</td><td>3.51</td><td>21.45</td><td>19.37</td><td>3.8</td><td>3.8</td></tr><tr><td>A-MEM</td><td>17.44</td><td>11.74</td><td>26.38</td><td>19.50</td><td>12.53</td><td>11.83</td><td>28.14</td><td>23.87</td><td>42.04</td><td>40.60</td><td>1.0</td><td>1.0</td></tr></table>

with ground truth responses. Also, we report the average token length for answering one question. Besides reporting experiment results with four additional metrics (ROUGE-L, ROUGE-2, METEOR, and SBERT Similarity), we also present experimental outcomes using different foundation models including DeepSeek-R1-32B [11], Claude 3.0 Haiku [2], and Claude 3.5 Haiku [3] in Appendix A.3.

# 4.2 Implementation Details

For all baselines and our proposed method, we maintain consistency by employing identical system prompts as detailed in Appendix B. The deployment of Qwen-1.5B/3B and Llama 3.2 1B/3B models is accomplished through local instantiation using Ollama 1, with LiteLLM 2 managing structured output generation. For GPT models, we utilize the official structured output API. In our memory retrieval process, we primarily employ $k { = } 1 0$ for top- $k$ memory selection to maintain computational efficiency, while adjusting this parameter for specific categories to optimize performance. The detailed configurations of $k$ can be found in Appendix A.5. For text embedding, we implement the all-minilm-l6-v2 model across all experiments.

# 4.3 Empricial Results

Performance Analysis. In our empirical evaluation, we compared A-MEM with four competitive baselines including LoCoMo [22], ReadAgent [17], MemoryBank [39], and MemGPT [25] on the LoCoMo dataset. For non-GPT foundation models, our A-MEM consistently outperforms all baselines across different categories, demonstrating the effectiveness of our agentic memory approach. For GPT-based models, while LoCoMo and MemGPT show strong performance in certain categories like Open Domain and Adversial tasks due to their robust pre-trained knowledge in simple fact retrieval, our A-MEM demonstrates superior performance in Multi-Hop tasks achieves at least two times better performance that require complex reasoning chains. In addition to experiments on the LoCoMo dataset, we also compare our method on the DialSim dataset against LoCoMo and MemGPT. A-MEM consistently outperforms all baselines across evaluation metrics, achieving an F1

Table 2: Comparison of different memory mechanisms across multiple evaluation metrics on DialSim [16]. Higher scores indicate better performance, with A-MEM showing superior results across all metrics.   
Table 3: An ablation study was conducted to evaluate our proposed method against the GPT-4o-mini base model. The notation ’w/o’ indicates experiments where specific modules were removed. The abbreviations LG and ME denote the link generation module and memory evolution module, respectively.   

<table><tr><td>Method</td><td>F1</td><td>BLEU-1</td><td>ROUGE-L</td><td>ROUGE-2</td><td>METEOR</td><td>SBERT Similarity</td></tr><tr><td>LoCoMo</td><td>2.55</td><td>3.13</td><td>2.75</td><td>0.90</td><td>1.64</td><td>15.76</td></tr><tr><td>MemGPT</td><td>1.18</td><td>1.07</td><td>0.96</td><td>0.42</td><td>0.95</td><td>8.54</td></tr><tr><td>A-MEM</td><td>3.45</td><td>3.37</td><td>3.54</td><td>3.60</td><td>2.05</td><td>19.51</td></tr></table>

<table><tr><td colspan="11">Category</td></tr><tr><td rowspan="2">Method</td><td colspan="2">Multi Hop</td><td colspan="2">Temporal</td><td colspan="2">Open Domain</td><td colspan="2">Single Hop</td><td colspan="2">Adversal</td></tr><tr><td>F1</td><td>BLEU-1</td><td>F1</td><td>BLEU-1</td><td>F1</td><td>BLEU-1</td><td>F1</td><td>BLEU-1</td><td>F1</td><td>BLEU-1</td></tr><tr><td>w/o LG &amp; ME</td><td>9.65</td><td>7.09</td><td>24.55</td><td>19.48</td><td>7.77</td><td>6.70</td><td>13.28</td><td>10.30</td><td>15.32</td><td>18.02</td></tr><tr><td>w/o ME</td><td>21.35</td><td>15.13</td><td>31.24</td><td>27.31</td><td>10.13</td><td>10.85</td><td>39.17</td><td>34.70</td><td>44.16</td><td>45.33</td></tr><tr><td>A-MEM</td><td>27.02</td><td>20.09</td><td>45.85</td><td>36.67</td><td>12.14</td><td>12.00</td><td>44.65</td><td>37.06</td><td>50.03</td><td>49.47</td></tr></table>

score of 3.45 (a $3 5 \%$ improvement over LoCoMo’s 2.55 and $192 \%$ higher than MemGPT’s 1.18). The effectiveness of A-MEM stems from its novel agentic memory architecture that enables dynamic and structured memory management. Unlike traditional approaches that use static memory operations, our system creates interconnected memory networks through atomic notes with rich contextual descriptions, enabling more effective multi-hop reasoning. The system’s ability to dynamically establish connections between memories based on shared attributes and continuously update existing memory descriptions with new contextual information allows it to better capture and utilize the relationships between different pieces of information.

Cost-Efficiency Analysis. A-MEM demonstrates significant computational and cost efficiency alongside strong performance. The system requires approximately 1,200 tokens per memory operation, achieving an $8 5 - 9 3 \%$ reduction in token usage compared to baseline methods (LoCoMo and MemGPT with 16,900 tokens) through our selective top-k retrieval mechanism. This substantial token reduction directly translates to lower operational costs, with each memory operation costing less than $\$ 0.0003$ when using commercial API services—making large-scale deployments economically viable. Processing times average 5.4 seconds using GPT-4o-mini and only 1.1 seconds with locally-hosted Llama 3.2 1B on a single GPU. Despite requiring multiple LLM calls during memory processing, A-MEM maintains this cost-effective resource utilization while consistently outperforming baseline approaches across all foundation models tested, particularly doubling performance on complex multi-hop reasoning tasks. This balance of low computational cost and superior reasoning capability highlights A-MEM’s practical advantage for deployment in the real world.

# 4.4 Ablation Study

To evaluate the effectiveness of the Link Generation (LG) and Memory Evolution (ME) modules, we conduct the ablation study by systematically removing key components of our model. When both LG and ME modules are removed, the system exhibits substantial performance degradation, particularly in Multi Hop reasoning and Open Domain tasks. The system with only LG active (w/o ME) shows intermediate performance levels, maintaining significantly better results than the version without both modules, which demonstrates the fundamental importance of link generation in establishing memory connections. Our full model, A-MEM, consistently achieves the best performance across all evaluation categories, with particularly strong results in complex reasoning tasks. These results reveal that while the link generation module serves as a critical foundation for memory organization, the memory evolution module provides essential refinements to the memory structure. The ablation study validates our architectural design choices and highlights the complementary nature of these two modules in creating an effective memory system.

# 4.5 Hyperparameter Analysis

We conducted extensive experiments to analyze the impact of the memory retrieval parameter k, which controls the number of relevant memories retrieved for each interaction. As shown in Figure 3, we evaluated performance across different k values (10, 20, 30, 40, 50) on five categories of tasks using GPT-4o-mini as our base model. The results reveal an interesting pattern: while increasing k generally leads to improved performance, this improvement gradually plateaus and sometimes slightly decreases at higher values. This trend is particularly evident in Multi Hop and Open Domain

![](images/b29b8490509fbf13940718fe24a30a2987752e1ccedd9c7b55250e25437bf5a1.jpg)  
(a) Multi Hop

![](images/95c522a9ac24ecb542dda9c9f8a6216b06df3c4782b8aa14b35baaabc24bca43.jpg)  
(b) Temporal

![](images/a9f2ffb62a87bfff6b84830fddbb3e63ce8f205a2a19e59c1c5abbfc9549de1e.jpg)  
(c) Open Domain

![](images/4687d0173e5b518c5877b3720215cdfd970d9523e3032c296594700aa830373d.jpg)  
(d) Single Hop

![](images/e293505fb21200afbcef8d0fc783ee7d0f071a5452ab809fbc7dec004cb8e0a1.jpg)  
(e) Adversarial   
Figure 3: Impact of memory retrieval parameter k across different task categories with GPT-4o-mini as the base model. While larger k values generally improve performance by providing richer historical context, the gains diminish beyond certain thresholds, suggesting a trade-off between context richness and effective information processing. This pattern is consistent across all evaluation categories, indicating the importance of balanced context retrieval for optimal performance.

Table 4: Comparison of memory usage and retrieval time across different memory methods and scales.   

<table><tr><td>Memory Size</td><td>Method</td><td>Memory Usage (MB)</td><td>Retrieval Time (μs)</td></tr><tr><td rowspan="3">1,000</td><td>A-MEM</td><td>1.46</td><td>0.31 ± 0.30</td></tr><tr><td>MemoryBank [39]</td><td>1.46</td><td>0.24 ± 0.20</td></tr><tr><td>ReadAgent [17]</td><td>1.46</td><td>43.62 ± 8.47</td></tr><tr><td rowspan="3">10,000</td><td>A-MEM</td><td>14.65</td><td>0.38 ± 0.25</td></tr><tr><td>MemoryBank [39]</td><td>14.65</td><td>0.26 ± 0.13</td></tr><tr><td>ReadAgent [17]</td><td>14.65</td><td>484.45 ± 93.86</td></tr><tr><td rowspan="3">100,000</td><td>A-MEM</td><td>146.48</td><td>1.40 ± 0.49</td></tr><tr><td>MemoryBank [39]</td><td>146.48</td><td>0.78 ± 0.26</td></tr><tr><td>ReadAgent [17]</td><td>146.48</td><td>6,682.22 ± 111.63</td></tr><tr><td rowspan="3">1,000,000</td><td>A-MEM</td><td>1464.84</td><td>3.70 ± 0.74</td></tr><tr><td>MemoryBank [39]</td><td>1464.84</td><td>1.91 ± 0.31</td></tr><tr><td>ReadAgent [17]</td><td>1464.84</td><td>120,069.68 ± 1,673.39</td></tr></table>

tasks. The observation suggests a delicate balance in memory retrieval - while larger k values provide richer historical context for reasoning, they may also introduce noise and challenge the model’s capacity to process longer sequences effectively. Our analysis indicates that moderate k values strike an optimal balance between context richness and information processing efficiency.

# 4.6 Scaling Analysis

To evaluate storage costs with accumulating memory, we examined the relationship between storage size and retrieval time across our A-MEM system and two baseline approaches: MemoryBank [39] and ReadAgent [17]. We evaluated these three memory systems with identical memory content across four scale points, increasing the number of entries by a factor of 10 at each step (from 1,000 to 10,000, 100,000, and finally 1,000,000 entries). The experimental results reveal key insights about our A-MEM system’s scaling properties: In terms of space complexity, all three systems exhibit identical linear memory usage scaling $( O ( N ) )$ , as expected for vector-based retrieval systems. This confirms that A-MEM introduces no additional storage overhead compared to baseline approaches. For retrieval time, A-MEM demonstrates excellent efficiency with minimal increases as memory size grows. Even when scaling to 1 million memories, A-MEM’s retrieval time increases only from $0 . 3 1 \mu \mathrm { s }$ to $3 . 7 0 \mu \mathrm { s }$ , representing exceptional performance. While MemoryBank shows slightly faster retrieval times, A-MEM maintains comparable performance while providing richer memory representations and functionality. Based on our space complexity and retrieval time analysis, we conclude that A-MEM’s retrieval mechanisms maintain excellent efficiency even at large scales. The minimal growth in retrieval time across memory sizes addresses concerns about efficiency in large-scale memory systems, demonstrating that A-MEM provides a highly scalable solution for long-term conversation management. This unique combination of efficiency, scalability, and enhanced memory capabilities positions A-MEM as a significant advancement in building powerful and long-term memory mechanism for LLM Agents.

![](images/0836f3c9c350f97c3238fb0a589827722dd728fab9b71303cc4e9c787183260e.jpg)  
(a) Dialogue 1

![](images/f897a0e33c71ea228769901bec687d04782970307186c857f0515a92a9880bb0.jpg)  
(b) Dialogue 2   
Figure 4: T-SNE Visualization of Memory Embeddings Showing More Organized Distribution with A-MEM (blue) Compared to Base Memory (red) Across Different Dialogues. Base Memory represents A-MEM without link generation and memory evolution.

# 4.7 Memory Analysis

We present the t-SNE visualization in Figure 4 of memory embeddings to demonstrate the structural advantages of our agentic memory system. Analyzing two dialogues sampled from long-term conversations in LoCoMo [22], we observe that A-MEM (shown in blue) consistently exhibits more coherent clustering patterns compared to the baseline system (shown in red). This structural organization is particularly evident in Dialogue 2, where well-defined clusters emerge in the central region, providing empirical evidence for the effectiveness of our memory evolution mechanism and contextual description generation. In contrast, the baseline memory embeddings display a more dispersed distribution, demonstrating that memories lack structural organization without our link generation and memory evolution components. These visualization results validate that A-MEM can autonomously maintain meaningful memory structures through dynamic evolution and linking mechanisms. More results can be seen in Appendix A.4.

# 5 Conclusions

In this work, we introduced A-MEM, a novel agentic memory system that enables LLM agents to dynamically organize and evolve their memories without relying on predefined structures. Drawing inspiration from the Zettelkasten method, our system creates an interconnected knowledge network through dynamic indexing and linking mechanisms that adapt to diverse real-world tasks. The system’s core architecture features autonomous generation of contextual descriptions for new memories and intelligent establishment of connections with existing memories based on shared attributes. Furthermore, our approach enables continuous evolution of historical memories by incorporating new experiences and developing higher-order attributes through ongoing interactions. Through extensive empirical evaluation across six foundation models, we demonstrated that A-MEM achieves superior performance compared to existing state-of-the-art baselines in long-term conversational tasks. Visualization analysis further validates the effectiveness of our memory organization approach. These results suggest that agentic memory systems can significantly enhance LLM agents’ ability to utilize long-term knowledge in complex environments.

# 6 Limitations

While our agentic memory system achieves promising results, we acknowledge several areas for potential future exploration. First, although our system dynamically organizes memories, the quality of these organizations may still be influenced by the inherent capabilities of the underlying language models. Different LLMs might generate slightly different contextual descriptions or establish varying connections between memories. Additionally, while our current implementation focuses on text-based interactions, future work could explore extending the system to handle multimodal information, such as images or audio, which could provide richer contextual representations.