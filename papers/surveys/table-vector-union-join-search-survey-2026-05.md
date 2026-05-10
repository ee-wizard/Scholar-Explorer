# 表格可联合/可连接搜索中的向量化表示：方法综述与写作模板（2026-05）

## 1. 综述目标

本文聚焦数据湖中的两类任务：
- Table Union Search (TUS): 找可做纵向拼接（增行）的表。
- Joinable Table Search (JTS): 找可做横向连接（增列）的表。

本综述重点不是“是否用了深度模型”，而是回答一个更细粒度问题：
- 作者如何在论文中把“表格表示为向量”写清楚？
- 向量表示如何与检索/重排/匹配目标函数衔接？

## 2. 样本论文（已本地拉取并解析）

1. SANTOS: Relationship-based Semantic Table Union Search (SIGMOD/PACMMOD 2023)
- 本地目录: papers/table_vector_search/SANTOS Relationship-based Semantic Table Union Search
- 核心: 关系语义图（semantic graph）驱动的 union search，非典型 embedding 主线。

2. EasyTUS: A Comprehensive Framework for Fast and Accurate Table Union Search across Data Lakes (IEEE BigData 2025 / arXiv 2025)
- 本地目录: papers/table_vector_search/EasyTUS A Comprehensive Framework for Fast and Accurate Table Union Search across Data Lakes
- 核心: LLM embedding + ANN 索引，单表单向量检索。

3. An Efficient Proximity Graph-based Approach to Table Union Search (PGTUS, arXiv 2025)
- 本地目录: papers/table_vector_search/An Efficient Proximity Graph-based Approach to Table Union Search
- 核心: 每列表向量（multi-vector）+ 最大二分匹配评分 + 图索引加速。

4. Efficient and Effective Table-Centric Table Union Search in Data Lakes (TACTUS, arXiv 2026)
- 本地目录: papers/table_vector_search/Efficient and Effective Table-Centric Table Union Search in Data Lakes
- 核心: 从列中心转向表中心，先表向量检索后列级重排。

## 3. 任务定义与向量化对象

### 3.1 传统定义
- 输入: 查询表 q，数据湖中的候选表集 T。
- 输出: top-k unionable/joinable tables。
- 关键分数: unionability/joinability score。

### 3.2 向量化对象的三种主流写法

1. 列向量（column embeddings）
- 典型写法: 把每列编码为一个向量，表表示为向量集合。
- 优点: 细粒度对齐强，支持列匹配解释。
- 缺点: 评分通常依赖二分图匹配，在线代价高。

2. 表向量（table embedding）
- 典型写法: 直接把整表编码为单向量，用 ANN 先召回候选。
- 优点: 检索快，召回阶段成本低。
- 缺点: 细粒度列对齐信息被压缩，常需二阶段重排补救。

3. 图/关系增强表示（semantic graph + vector/hybrid score）
- 典型写法: 关系语义图刻画列间关系，再与列语义或表语义联合打分。
- 优点: 避免仅凭列值近似导致误召回。
- 缺点: 依赖关系抽取质量与覆盖率。

## 4. 四篇论文逐篇分析：它们如何写“表格向量化”

### 4.1 SANTOS（关系语义主线）

写法特征:
- 先批判“只看列语义”会误判 unionability。
- 提出 column semantics + relationship semantics 的联合定义。
- 用语义图匹配代替纯向量最近邻。

可借鉴表述:
- 先定义“语义对象”（列语义、关系语义），再定义“结构化中间表示”（semantic graph）。
- 即使不走纯 embedding，也可以把“表示学习问题”写成“可比较的语义结构表示问题”。

对“表格向量化写作”的启发:
- 这篇论文提供了一个反例：当你要论证向量表示必要性时，必须先交代“非向量方案为何不足”。

### 4.2 EasyTUS（单表向量主线）

写法特征:
- 三段式管线写法非常标准：Table Serialization -> Table Representation -> Vector Search。
- 明确给出“为何使用整表向量而非列向量”的工程与语义理由（token budget、鲁棒性、多湖检索、无微调）。
- 把离线/在线过程拆开，强调成本归属。

可借鉴表述:
- “We generate one embedding per table...” 这类句式清晰绑定了表示粒度。
- “semantic representation for the entire table rather than for individual columns” 直接说明设计取舍。

对写作最有价值的一点:
- 它把“向量化”写成了一个系统接口问题，而不只是模型问题。你写 survey 时应保留这种工程可落地的叙述框架。

### 4.3 PGTUS（多向量 + 匹配主线）

写法特征:
- 开篇即形式化“表 -> 列向量集合”。
- 用 t-matching / maximum bipartite matching 定义 unionability 目标函数。
- 在算法段强调“为何要做近似过滤”：many-to-one matching 上下界 + refinement/pruning。

可借鉴表述:
- 从定义开始就把“向量表示”和“检索目标”绑死：
  - 表示: each column -> one vector
  - 目标: maximize unionability under matching constraints
  - 系统: 先粗筛再精配

对写作最有价值的一点:
- 这是“向量化 + 可解释目标函数”范式，适合写给数据库/算法审稿人。

### 4.4 TACTUS（表中心两阶段主线）

写法特征:
- 先通过反例展示“列中心方法的误排”。
- 提出 table-centric 设计：先表向量筛候选，再做列级证据融合重排。
- 把训练数据构造（正样本/难负样本）写得很实，直接服务于 unionability 语义。

可借鉴表述:
- “table-to-column rather than column-to-table” 是非常强的问题重述句式。
- “dual-evidence reranking” 把表向量分数与列对齐分数融合，叙事完整。

对写作最有价值的一点:
- 它解决了单向量语义粗糙与多向量计算开销之间的折中，是 survey 中“方法演进主线”的关键节点。

## 5. 跨论文比较：表格向量化写作范式

### 5.1 常见段落骨架

1. 问题重述
- 传统 union/join 判定在无元数据、跨域、多语言场景失效。

2. 表示粒度声明
- 我们采用列级多向量/表级单向量/混合表示。

3. 表示与目标绑定
- 向量相似度如何映射到 unionability/joinability 分数。

4. 检索流程分层
- 离线建库（表示学习、索引构建）与在线检索（召回、重排）分离。

5. 效率-效果权衡
- 为什么要近似筛选、为何引入匹配约束或二阶段重排。

### 5.2 你可直接复用的写作模板

模板 A（表向量）:
- “We encode each table into a single dense vector that captures holistic table semantics, and perform ANN-based retrieval to obtain a compact candidate set.”
- “To mitigate information loss of single-vector representations, we apply a column-aware reranking stage over retrieved candidates.”

模板 B（列向量）:
- “We represent a table as a set of column vectors, and define unionability as a constrained maximum matching objective over cross-table column pairs.”
- “To improve scalability, we introduce a filter-and-verify pipeline with lightweight upper bounds before exact matching.”

模板 C（混合）:
- “We decouple table discovery and schema alignment: table-level embeddings drive candidate retrieval, while column-level evidence refines final ranking.”

## 6. 对 joinable search 的外推

虽然本轮拉取论文主要是 union search，但其向量化写法可直接迁移到 joinable search:
- 把“unionability score”替换为“joinability score”；
- 把“列语义对齐”替换为“连接键兼容性 + 连接后信息增益”；
- 保持“先召回、后精排”两阶段结构不变。

## 7. 八篇论文如何引出向量表示的必要性

所有分析论文（4篇本地论文 + 4篇新论文）都采用向量表示作为核心方案，但它们的**引出方式和动机不同**。本节整理它们如何说明"为什么需要向量表示"。

### 7.1 SANTOS（关系语义主线）——**批判列语义不足**

**Abstract 引出方式**：
> *"Existing techniques for unionable table search define unionability using metadata or column-based metrics... In this work, we introduce the use of **semantic relationships between pairs of columns** in a table to improve the accuracy of union search."*

**关键动机**：
- 列语义方法（包括列值、列名、word embeddings）**不够充分**。
- 例子驱动：仅看列语义会产生假正例（表c的3列与查询表都相似，但关系语义完全不同）。
- 结论：必须把**关系语义**纳入打分，而不是仅用列向量最近邻。

**启示**：这篇论文实际上不走"纯向量化"路线，而是提出"向量表示不足需加关系约束"——是一个反例/对照组。

---

### 7.2 EasyTUS（单表向量主线）——**批判传统方法元数据依赖与跨域泛化难**

**Abstract 引出方式**：
> *"Earlier Table Union Search approaches explored techniques such as **metadata analysis, column-level set similarity, and ontology-based annotation**. However, their accuracy is constrained by the limitations of data lakes: **metadata is often sparse or inaccurate, exact value matches are rare, and domain-specific ontologies... are frequently unavailable**."*

**Introduction 深层动机**：
> *"Recent approaches rely on extensively fine-tuning the models and manually configuring hyperparameters... smaller models often fail to generalize across domains or languages, and fine-tuning models on specific data lakes restricts applicability, making searches across multiple data lakes **impossible**."*

**EasyTUS 的引出结论**：
> *"This paper introduces EasyTUS, a novel Table Union Search framework that leverages Large Language Models (LLMs) **without relying on metadata, ontologies, or finetuning**. Trained on broad and diverse corpora, EasyTUS offers strong generalizability across domains and languages..."*

**启示**：用"三个无"（no metadata, no ontology, no finetuning）对比"三个难"（sparse metadata, limited ontology, poor generalization），动机清晰。整表单向量 + LLM embedding 的正当性来自于**泛化性与跨域可用性**。

---

### 7.3 PGTUS（多向量匹配主线）——**承认向量质量好，针对效率瓶颈**

**Abstract 引出方式**：
> *"**Multi-vector models, which represent a table as a vector set (typically one vector per column), have been demonstrated to achieve superior retrieval quality** by capturing fine-grained semantic alignments. However, this problem faces **more severe efficiency challenges than the single-vector problem** due to the inherent dependency on **bipartite graph maximum matching** to compute unionability scores."*

**关键动机**：
- 不是说向量表示有问题，而是说**多向量的精度已被验证更优**。
- 瓶颈在于：二分图匹配的计算复杂度与索引成本（Starmie 的低效率）。

**PGTUS 的引出结论**：
> *"Therefore, this paper proposes an efficient Proximity Graph-based Table Union Search (PGTUS) approach... reduces the latency of vector-set based table union search."*

**启示**：这是"承认向量化正确性，优化其执行"的写法——是工程优化论文的标准范式，而非方法创新的论文。

---

### 7.4 TACTUS（表中心两阶段主线）——**批判列中心粒度细但整体语义被忽略**

**Abstract 引出方式**：
> *"Existing methods are mainly **column-centric**: they focus on modeling column unionability scores using column embeddings, which are then used throughout the search process for column matching, filtering, and aggregation. However, this **overlooks holistic table-level semantics, which may result in suboptimal rankings and inefficiencies**."*

**TACTUS 的引出结论**：
> *"We introduce TACTUS, a novel **table-centric** method for table union search. Unlike prior work that searches from columns to tables, **we search in a table-first way** and examine columns only in the final step... we directly generate **table embeddings** for holistic, table-level unionability scoring..."*

**启示**：这是"粒度迁移"论文——从列粒度向上升到表粒度。动机是表级特征（如表风格、列间关联、行数特征）在列级分解表示中被丢失。

---

### 7.5 Starmie（列向量对比学习主线）——**提出多列上下文对比学习**

**Abstract 引出方式**：
> *"Our proposed framework features **a contrastive learning method to train column encoders from pre-trained language models in a fully unsupervised manner**. The column encoder of Starmie **captures the rich contextual semantic information within tables by leveraging a contrastive multi-column pre-training strategy**."*

**关键动机**：
- 不是"为什么用向量"，而是"**如何训练列向量才能捕获表内上下文**"。
- 多列协同对比学习（multi-column contrastive pre-training）是为了利用**列间关系进行自监督**。

**启示**：Starmie 的贡献点不在"为什么向量化"，而在"**如何让列向量编码器学到多列上下文**"——是一个训练目标与数据利用策略的论文。

---

### 7.6 Pylon（列向量索引感知对比学习主线）——**把表示学习与索引结构联合优化**

**Abstract 引出方式**：
> *"The challenge is to recognize unionable columns even if they are represented differently... Our key idea is to exploit **self-supervised contrastive learning to learn an embedding model that takes into account the indexing/search data structure**..."*

**Introduction 深层论证**：
> *"...we argue that these models in training **do not take into account the indexing/search data structure**, which is a core component of any efficient solution to table union search... the embeddings they use and recent more advanced tabular models are **not optimized in training for approximate cosine similarity search** so the performance of indexing/search data structure is **suboptimal**."*

**Pylon 的引出结论**：
> *"Instead of relying on low-coverage ontologies or pre-trained embeddings that are unaware of required data structure in table union search, we propose a data-driven learning approach to capture data semantics **and model characteristics of the essential indexing/search data structure**."*

**启示**：Pylon 批判的不是"向量表示本身"，而是"现有向量表示没有与索引结构协同优化"。这是一个**表示学习与检索系统协设计**的论文。

---

### 7.7 DeepJoin（列向量PLM微调主线）——**用PLM处理语义相似，用固定长向量解决可扩展性**

**Abstract 引出方式**：
> *"Existing approaches target equi-joins or semantic joins. They are either exact solutions whose running time is **linear in the sizes of query column and target table repository**, or approximate solutions lacking precision... Our solution is an **embedding-based retrieval**, which employs a **pre-trained language model (PLM)**..."*

**关键解决的两个问题**：
1. 精度问题：用 PLM 捕获语义相似（word embedding similarity）。
2. 扩展性问题：*"Since the output of the PLM is **fixed in length**, the subsequent search procedure becomes **independent of the column size**. With a state-of-the-art approximate nearest neighbor search algorithm, the search time is **sublinear in the repository size**."*

**启示**：DeepJoin 的向量化正当性来自两方面：**语义建模能力**（PLM） + **计算代价解耦**（fixed-length vectors）。

---

### 7.8 TabSketchFM（Sketch向量主线）——**用Sketch突破Token长度限制与数值语义丢失**

**Abstract 引出方式**：
> *"Tabular neural models can be helpful for such data discovery tasks... we present TabSketchFM, a neural tabular model for data discovery over data lakes. First, we propose novel pre-training: **a sketch-based approach** to enhance the effectiveness of data discovery..."*

**Introduction 批判现有方案的两个局限**：
> *"Existing pre-trained tabular models... have two limitations: (a) only a small number of values can be passed in as input because of **severe limits on token length**... (b) **treating numerical values in the table as text tends to lose their semantics**; it may be better to pass numerical information directly as vectors into the model."*

**TabSketchFM 的引出结论**：
> *"To address these limitations on input length and numerical representation, we introduce a novel transformer-based tabular model—TabSketchFM—... that **creates different sketches over them instead of linearizing cell values**... Such sketches capture tabular features, and we input them into the neural models."*

**启示**：TabSketchFM 是"突破表示能力与计算约束"的方案——用 sketch（MinHash、statistical summary）作为**压缩表示**，同时**保留数值语义**。

---

### 7.9 跨论文对比：引出向量表示的八种方式

| 论文 | 主要批判对象 | 向量化的正当性来源 | 核心创新维度 |
|------|-----------|-----------------|----------|
| SANTOS | 列语义不足 | 向量表示需加关系约束 | 关系语义图 vs 纯向量 |
| EasyTUS | 元数据稀疏、模型泛化差 | 用LLM无需元数据/微调，强泛化 | 统一框架 + LLM通用性 |
| PGTUS | 多向量匹配效率 | 向量质量优，但需加速 | 二分图匹配加速 |
| TACTUS | 列中心遗漏表级特征 | 表向量捕获整体语义 | 粒度迁移：列→表 |
| Starmie | （不批判向量本身） | 列向量通过多列协同对比学习 | 多列上下文对比预训练 |
| Pylon | 表示未与索引协同 | 表示学习+索引结构联合优化 | 索引感知的对比学习 |
| DeepJoin | 列大小与仓库大小导致线性复杂度 | 固定长向量解耦代价；PLM捕语义 | PLM + ANNS 的两层架构 |
| TabSketchFM | Token长度限制、数值语义丢失 | Sketch压缩表示，保留语义 | Sketch预训练 + 混合嵌入 |

**写作启示**：
- 论文如果要**引出向量表示**，必须先说明"**不用向量或用非向量方案的具体问题**"。
- 问题可以来自：精度不足（SANTOS）、泛化性差（EasyTUS）、效率瓶颈（PGTUS、DeepJoin）、粒度设计（TACTUS）、编码方式（Starmie、Pylon、TabSketchFM）。
- **最有说服力的动机结构**是：(1) 诊断现有方案的具体失效情景，(2) 说明为什么向量化能解决该问题，(3) 说明你的向量化方案相比其他向量化的优势。

---

## 8. 四篇新论文的原文写法陈列（含章节位置）

本节引用 Starmie、Pylon、DeepJoin、TabSketchFM 四篇论文中关于"表格表示为向量"的原始英文表述，标注对应章节，供写作参照。

---

### 8.1 Starmie（PVLDB 2021）——列向量 + 多列上下文 Transformer

**粒度**：列级（column-level），每列一个向量；表 = 向量集合。

**Abstract（第一段）**：

> *"Our proposed framework features a contrastive learning method to train column encoders from pre-trained language models in a fully unsupervised manner. The column encoder of Starmie captures the rich contextual semantic information within tables by leveraging a contrastive multi-column pre-training strategy. We utilize the cosine similarity between column embedding vectors as the column unionability score..."*

**§2.1 Problem Definition**（形式化绑定）：

> *"A column encoder M takes a column t as input and outputs M(t) as the representation. Given two columns ti and tj, the column unionability score is computed as F(M(ti), M(tj)), where F is a scoring function between two column representations."*
>
> *"Given two tables S and T, we define a table unionability scoring mechanism as U = {F, M, A}, where M and F are the column encoder and scoring function for two column representations, respectively. Here A is a mechanism to aggregate the column unionability scores between all pairs of columns from the two tables."*

**§2.2 System Architecture（Offline Stage）**：

> *"During the offline stage, Starmie pre-trains a column representation model that encodes columns of data lake tables into dense high-dimensional vectors (i.e., column embeddings). Then, we apply the trained model to all data lake tables to obtain the column embeddings via model inference. We store the embedding vectors in efficient vector indices for online retrieval."*
>
> *"...then we apply the column encoder to all tables to convert each table into a collection of embedding vectors."*

**Figure 2 Caption**：

> *"During the offline phase, Starmie pre-trains a multicolumn table encoder using contrastive learning and stores the embeddings of data lake columns in vector indices like HNSW. During online processing, Starmie retrieves candidate tables with similar contextualized column embeddings then verifies their table-level unionability scores using column alignment algorithms."*

**写法要点**：先定义列编码器 M，再定义表分数 U = {F, M, A}，三要素明确分离。"collection of embedding vectors"而非"table embedding"——刻意强调粒度是集合而非单向量。离线/在线两段式叙事，职责明确。

---

### 8.2 Pylon（IEEE 论文）——列向量 + 索引感知对比学习

**粒度**：列级（column-level），一列一向量；表 = 列向量集合。

**Abstract**：

> *"Our key idea is to exploit self-supervised contrastive learning to learn an embedding model that takes into account the indexing/search data structure and produces embeddings close by for columns with semantically similar values while pushing apart columns with semantically dissimilar values. We then find union-able tables based on similarities between their constituent columns in embedding space."*

**§II-A Table Union Search（问题形式化）**：

> *"U_attr(A, B) = M(τ(A), τ(B)) where τ(·) is a feature extraction technique that transforms raw columns (attribute names, attribute values, or both) to a feature space and M(·,·) is a similarity measure between two instances in the feature space."*

**§II-C General Challenges**：

> *"Efficient solutions to table union search involve two stages: profiling (e.g., embed columns into a feature space) and index-based search. Taking off-the-shelf embedding models or training a new model without considering the indexing/search data structure indispensable in table union search is suboptimal. We argue that aligning representation learning with indexing/search data structure can further improve effectiveness and efficiency of a solution to table union search."*

**§III-A Framework（核心表述）**：

> *"Pylon is designed to generate a vector representation for each column of input tables where columns containing semantically similar values have embeddings closer to one another."*

**§III-A Base Encoder & Projection Head**：

> *"We pass column instances {xk} through a base encoder f(·) to get initial column embeddings {ek}. The encoder can give embeddings at token/cell/column level, and if necessary, we can apply aggregation (e.g, average) to obtain column-level embeddings."*
>
> *"Following the encoder, a small multi-layer neural network g(·), called projection head, maps the representations from the encoder to a latent space through linear transformations and non-linear activation in between. ... we preserve projection head and use projected embeddings for table union search."*

**写法要点**：明确声明"取决于索引结构"——把表示学习和检索结构视为一体设计，是区别于 Starmie 的核心叙事。base encoder f(·) → projection head g(·) 两层明确命名，提高写作精度。

---

### 8.3 DeepJoin（VLDB 2023）——列向量 + PLM 微调 + Prompt 工程

**粒度**：列级（column-level），每列经序列化后输入 PLM 得到固定长向量。

**Abstract**：

> *"Our solution is an embedding-based retrieval, which employs a pre-trained language model (PLM) and is designed as one framework serving both equi- and semantic joins... We propose a set of contextualization options to transform column contents to a text sequence. The PLM reads the sequence and is fine-tuned to embed columns to vectors such that columns are expected to be joinable if they are close to each other in the vector space."*

**§3 Method（管线概述）**：

> *"In DeepJoin, we use a fine-tuned PLM to embed columns to a vector space such that columns with high joinability are close to each other in the vector space. Since PLMs take as input raw text, we transform (i.e., contextualize) the contents in each column to a text sequence, and then feed the sequence to the fine-tuned PLM to produce a column embedding."*

**§3.1 Column-to-Text Transformation（序列化步骤）**：

> *"The column-to-text transformation belongs to prompt engineering, which works by including the description of the task in the input... DeepJoin takes advantage of metadata and considers seven options [e.g., title-colname-stat-col]."*

示例（Figure 2 running example）：列 "Company" 被序列化为：*"Company information. Company contains 5 values (9, 2, 5.6): Apple, GE, Microsoft, Yahoo!, Amazon."*

**§3.2 Column Embedding**：

> *"Since PLMs have an input length limit max_seq_length (e.g., 512 tokens for BERT), in the case of a tall input column, we choose a frequency-based approach, e.g., taking a sample of the most frequent cell values from the column, whose number of tokens is just within max_seq_length."*

**§3 Method（工程价值的说明）**：

> *"By embedding original data objects (i.e., columns) to a fixed-length vector, the subsequent procedure can be independent of the size of the data object, thereby achieving [sublinear search time]."*

**写法要点**："contextualize"/"column-to-text transformation"是独立的子步骤，值得单独命名并给出七种选项的设计空间。"fixed-length vector"直接点明 PLM 的工程价值——解耦列大小与搜索代价。

---

### 8.4 TabSketchFM（IEEE 2024）——Sketch 向量 + 修改 BERT 输入架构

**粒度**：混合——列级 sketch 向量（数值/MinHash）+ 表级 content snapshot，最终 [CLS] token 输出为单表向量，用于搜索。

**§I Introduction（批判现有方案，引出 sketch 动机）**：

> *"Existing pre-trained tabular models... have two limitations for understanding relevant tables: (a) only a small number of values can be passed in as input because of severe limits on token length in encoder-only models that can generate embeddings for search (e.g. 512 tokens in BERT); (b) treating numerical values in the table as text tends to lose their semantics; it may be better to pass numerical information directly as vectors into the model."*
>
> *"To address these limitations on input length and numerical representation, we introduce a novel transformer-based tabular model—TabSketchFM—... that creates different sketches over them instead of linearizing cell values. Specifically, we abstract column cell values using data sketches... Such sketches capture tabular features, and we input them into the neural models."*

**§III-A Sketches Preparation（三类 sketch 的定义）**：

Content Snapshot（表级）：
> *"We create a sketch from the first 10000 rows. We convert each row into a string and generate a MinHash signature from the set of rows. This is the only table-level sketch that we create."*

Numerical Sketch（列级统计向量）：
> *"For each column, we extract a set of statistical features... a numerical sketch vector consists of the following elements: [unique count, NaN count, cell width, 10th percentile, ..., 90th percentile, mean, standard deviation, min value, max value]."*

MinHash Sketch（列级字符串集合）：
> *"For each column, we treat cell-value as a string and compute a MinHash signature from the set of cell-values. For string columns, we also compute a MinHash signature for set of words within the column."*

**§III-B Input Representation（六种嵌入叠加的设计）**：

> *"The input embedding is constructed from this input string by summing the following different embeddings: 1. Token Embeddings. 2. Token Position Embeddings. 3. Column Position Embeddings. 4. Column Type Embeddings. 5. MinHash Sketch Embeddings. 6. Numerical Sketch Embedding. The sum of these embeddings is used as the input embedding for the 12-layered BERT encoder model with self-attention."*

**§III-E Overview（搜索时表向量的提取方式）**：

> *"To do so, we extract the table embeddings from the finetuned TabSketchFM, and use that to create nearest neighbor indexes for search tasks."*（即取 BERT [CLS] pooler 输出作为表向量）

**写法要点**：先用"两个局限性"驱动 sketch 设计，动机-方案结构感强。六种嵌入逐一命名，是"输入表示设计空间"的完整展示写法。区分 pre-training sketch 和 search embedding 的用途，叙事不混淆。

---

### 8.5 跨四篇论文对比表

| 论文 | 向量粒度 | 序列化方式 | 模型类型 | 表级分数计算 | 关键章节 |
|------|---------|----------|---------|------------|---------|
| Starmie | 列向量集合 | 列值直接送入 BERT（含多列上下文） | 对比学习预训练 | 二分图加权匹配 | §2.1, §2.2 |
| Pylon | 列向量集合 | 列值 → base encoder f(·) → projection head g(·) | 对比学习（索引感知） | 贪心最大匹配 | §II-A, §III-A |
| DeepJoin | 列向量（单向量/列） | 列 → Prompt 文本序列 → fine-tuned PLM → 固定长向量 | PLM 微调（metric learning） | 单列向量 kNN（无表级聚合） | §3.1, §3.2, §3.3 |
| TabSketchFM | 表向量（[CLS]） | 列 → 数值 sketch / MinHash sketch → 修改 BERT 输入 → [CLS] 输出 | Sketch 预训练 + 微调 | 单表向量 kNN | §III-A, §III-B, §III-E |

---

## 9. 结论：如何写好"表格表示为向量"

最稳妥的写法不是一句“我们把表编码成向量”，而是把下面三件事写全：
1. 表示对象是什么（整表/列集/关系图）；
2. 分数函数是什么（cosine、匹配和、融合分数）；
3. 检索系统怎么跑（离线建模与在线召回重排）。

在当前文献里：
- EasyTUS 代表“系统工程化的单向量写法”；
- PGTUS 代表“算法可解释的多向量写法”；
- TACTUS 代表“表中心两阶段折中写法”；
- SANTOS 代表“关系语义约束写法（向量化必要性的对照组）”。

因此，若你要写新论文，建议直接采用“TACTUS式两阶段叙事骨架 + PGTUS式目标函数清晰度 + EasyTUS式系统接口表达”。