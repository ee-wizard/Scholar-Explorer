# 学术论文分析报告

> **论文标题**：CITADEL: Conditional Token Interaction via Dynamic Lexical Routing for Efficient and Effective Multi-Vector Retrieval
> **论文 ID**：arXiv:2211.10411 / ACL 2023（Meta AI + University of Waterloo）
> **分析日期**：2026-05-07
> **主标签**：multi_vector_retrieval
> **论文标签**：multi_vector_retrieval
> **知识库关联论文**：PLAID（17× 更快的对比基线）；ColBERTv2（质量对标）；COIL（静态路由的前驱）

---

## 1. 问题定义

**问题背景**：
Multi-vector retrieval（ColBERT）在质量上领先但存在两个根本缺陷：(1) 全对全（all-to-all）token 交互导致延迟极高（GPU 122ms/query）；(2) 索引规模巨大（ColBERTv2 29GB，COIL-full 78.5GB）。COIL 通过精确词汇匹配约束减少交互，但引入词汇不匹配问题且负载不均衡（常见词 bucket 过大）。

**问题背景中的关键挑战（Challenges）**：
1. **交互冗余**：ColBERT 的 all-to-all routing 中大量 token 对与最终得分无关，产生 4213M 次点积/query
2. **词汇不匹配**：COIL 的精确匹配路由在 query 与 document 用词不同时完全失效（$\mathcal{T}_i = \emptyset$）
3. **负载不均衡**：COIL 的静态路由导致频繁词（"the"、"a"）对应的 bucket 极大，成为检索瓶颈
4. **质量-效率权衡困难**：现有方法（ColBERT/COIL）在质量和效率之间难以同时兼顾

**形式化定义**：
给定 query token 向量集合 $\{v_{q_i}\}$ 和 document token 向量集合 $\{v_{d_j}\}$，通过学习的路由函数 $\phi: \mathbb{R}^c \to \mathbb{R}^{|\mathcal{V}|}$ 将每个 token 映射到词汇键集合，使得：
$$s(q,d) = \sum_{i=1}^N \sum_{k=1}^K \max_{j,l \in \mathcal{E}_i^{(k)}} (w_{q_i}^{(k)} \cdot v_{q_i})^T (w_{d_j}^{(l)} \cdot v_{d_j})$$
其中 $\mathcal{E}_i^{(k)} = \{j,l \mid E_{d_j}^{(l)} = E_{q_i}^{(k)}\}$ 是同键的 document token 集合。

**问题的重要性**：
CITADEL 是将 ColBERT 级质量与近 DPR 级延迟（GPU 3.21ms vs DPR-128 的 0.63ms）结合的关键突破，对实时检索系统具有直接部署意义。

---

## 2. 前人工作的方法缺陷

| 缺陷类型 | 具体表现 | 代表工作 |
|----------|----------|----------|
| 效率问题 | all-to-all routing 需要 4213M 次点积/query，GPU 延迟 122ms | ColBERT, ColBERTv2 |
| 效率问题 | 即使 PLAID 优化引擎，GPU 延迟仍 55ms | PLAID |
| 精度/性能 | COIL 精确匹配路由在词汇不匹配时 $\mathcal{T}_i = \emptyset$，退化为单向量评分 | COIL |
| 效率问题 | COIL 索引负载严重不均衡（高频词 bucket 8× 大于 CITADEL），成为延迟瓶颈 | COIL-full: 47.9ms GPU |
| 泛化能力 | 单向量 DPR 在 entity-heavy 问题和 out-of-domain 上表现差 | DPR-768 |
| 理论局限 | COIL 的路由策略（token ID 匹配）是预定义启发式，无法通过训练优化 | COIL |

---

## 3. 研究动机与提出方案

**研究动机**：
ColBERT 的 all-to-all 路由冗余，COIL 的 static routing 引入词汇不匹配。作者认为路由策略应该**可学习**：既能处理词汇不匹配（通过语义相近 token 路由到同一"语义键"），又能避免冗余交互（通过 L1 正则化使大量 token 路由权重归零）。SPLADE 的 MLM 层激活函数恰好提供了稀疏、可学习的词汇维度权重，可作为路由器基础。

**方法本质（一句话）**：
> 本质上，这是一种通过**将 SPLADE 风格的稀疏词汇激活用作 token 路由键**，把 ColBERT 的 all-to-all 矩阵交互转化为词汇倒排索引查找的方法，在不损失质量的前提下将点积数量减少 400×。

**【批判性剥壳】方法还原**：
> 剥离 CITADEL 的包装后，核心操作等价于：
> 1. **Router = SPLADE 头**：在 ColBERT 的 token embedding 上接 BERT MLM 层，用 $\log(1+\text{ReLU}(Wv+b))$ 产生词汇维度稀疏权重（与 SPLADE 完全相同）
> 2. **Routing = top-K 词汇键选择**：每个 token 选 top-K 激活词汇键（query K=1，document K=5）
> 3. **检索 = 向量倒排索引查找**：等同于 COIL 的倒排索引机制，但"键"是学习得到的词汇而非 token ID
> 4. **评分 = 加权 MaxSim**：路由权重乘以 token embedding 后做点积
>
> 本质是：**ColBERT 双 encoder + SPLADE 路由头 + COIL 倒排索引检索机制**。创新在于将"静态词汇路由"替换为"端到端学习的词汇路由"，并引入 load balancing 损失控制 bucket 均衡性。

**方案类型与适配说明**：
端到端模型训练 + 倒排索引检索。训练目标为对比损失（相似度优化）+ 路由器损失（键重叠最大化）+ L1 正则化（稀疏性）+ 负载均衡损失（bucket 均匀性）。

**核心假设及其边界**：
1. **词汇键覆盖假设**：BERT 的词汇表（约 30K）足够丰富，能够覆盖语义上相关但词法不同的 query-document token 对（通过学习路由实现）
2. **稀疏性假设**：L1 正则化能引导大量 token 路由权重归零，不损失质量（消融显示约 50 tokens/passage 被去激活）
3. **边界**：越激进的 post-hoc pruning（$\tau > 1.1$）越会损害 out-of-domain 泛化（BEIR 实验表明正则化会牺牲域外 BEIR 平均分）；CLS token 去除导致精度下降

**核心创新点**：
1. **统一 token routing 视角**：将 single-vector / COIL / ColBERT 统一为路由机制的不同特例（all-to-none / static lexical / all-to-all）
2. **动态词汇路由（Dynamic Lexical Routing）**：用端到端学习的路由器替代静态词汇匹配，解决 COIL 的词汇不匹配问题
3. **路由器损失（$\mathcal{L}_r$）**：专门优化 query-document 键重叠，是独立于检索损失的正交训练信号
4. **负载均衡损失（$\mathcal{L}_b$）**：借鉴 Switch Transformer 的 expert 均衡思想，确保倒排 bucket 大小均匀，消除 COIL 的负载不均问题
5. **Post-hoc pruning**：训练后用阈值 $\tau$ 进一步裁剪低权重 token，提供灵活的延迟-精度-存储三方权衡

**论文核心贡献（Contributions）**：
1. 统一 token routing 视角，为理解多向量检索提供新框架
2. CITADEL 方法：动态词汇路由 + 四部分联合损失
3. GPU 延迟 3.21ms（vs ColBERTv2 122ms，40× 加速），质量与 ColBERTv2 持平
4. 与 PQ 结合后 0.9ms/query，13.3GB 索引

**方法框架概述**：
CITADEL 保留 ColBERT 双 encoder 结构，增加 SPLADE 风格路由头，通过四部分损失函数联合优化检索质量和路由效率，检索时使用词汇键倒排索引（类 COIL）做向量搜索。

**整体流程拆解（按阶段）**：
1. **编码（Offline Indexing）**：对每个 document token，计算 token embedding + 路由权重；top-L 路由键（L=5）对应的加权 embedding 存入词汇倒排索引 $\mathcal{T}_E$；post-hoc pruning 移除路由权重 < $\tau$ 的 token
2. **编码（Query Time）**：对 query tokens 计算 embedding + 路由权重；每个 token 取 top-K 路由键（K=1）
3. **检索**：对每个 query token 的路由键 $E$，在倒排索引 $\mathcal{T}_E$ 中做向量搜索（内积），得到候选 document 及 MaxSim 得分
4. **合并排序**：对各 query token 的搜索结果按式(5)合并得到最终排名

**核心模块与职责分工**：
- `CITADEL Encoder`：ColBERT 风格 BERT（token embeddings）
- `LexicalRouter`（$\phi$）：BERT MLM 层，输出 $|\mathcal{V}|$ 维稀疏路由权重（SPLADE 激活）
- `InvertedVectorIndex`（$\mathcal{T}$）：词汇键 → 加权 token embedding 列表（类 COIL 结构）
- `LoadBalancer`：训练时 Switch Transformer 风格的 bucket 均衡器

**输入、输出与中间表示**：
- 输入：query 文本 / document 文本
- 中间：token embedding 矩阵、路由权重矩阵（$N \times |\mathcal{V}|$ 稀疏）、词汇键集合
- 输出：top-k 相关文档及得分

**训练阶段/索引构建阶段细节**：
四部分损失函数：
- $\mathcal{L}_e$：检索对比损失（BM25 hard negatives 或 cross-encoder 蒸馏）
- $\mathcal{L}_r$：路由器对比损失（最大化正例 query-document 路由键重叠）
- $\mathcal{L}_b$：负载均衡损失（Switch Transformer 风格，最小化 $\sum_k f_k \cdot p_k$）
- $\mathcal{L}_s$：L1 稀疏正则化（推动路由权重归零）

训练数据：MS MARCO + BM25 hard negatives（CITADEL）；+ cross-encoder 蒸馏 + 一轮 hard negative mining（CITADEL+）。

**推理阶段/检索阶段细节**：
每个 query token（K=1 路由键）在对应词汇倒排 bucket 中进行向量搜索，合并各 query token 的结果得到 document 级 MaxSim 得分，排序返回 top-k。实验超参：query K=1，document L=5，pruning $\tau=0.9$。

**目标函数、优化目标或评分机制**：
$$\mathcal{L} = \mathcal{L}_e + \mathcal{L}_r + \alpha \cdot \mathcal{L}_b + \beta \cdot \mathcal{L}_s$$
检索评分：路由权重加权的 token embedding 点积 MaxSim（式5）。

**相对已有方法的关键改动点**：
| 改动点 | ColBERT | COIL | CITADEL |
|--------|---------|------|---------|
| 路由类型 | All-to-all（全交互） | Static lexical（精确匹配） | Dynamic lexical（学习路由）|
| 路由键来源 | 无（穷举） | Token ID（预定义） | SPLADE 激活（学习得到）|
| Bucket 均衡 | N/A | 不均衡（高频词 8× 倾斜） | 负载均衡损失控制 |
| 点积数/query | 4213M | 45.6M | 10.5M |
| GPU 延迟 | 122ms | 47.9ms | 3.21ms |

**为什么这个方案有效（机制解释）**：
- 动态路由消除词汇不匹配：SPLADE 激活将语义相关词（如"light, theory, relativity"→"Einstein"键）映射到同一词汇维度，实现语义层面的"exact match"
- L1 稀疏化：约 50/passage token 被完全去激活，减少索引规模和检索 FLOPs
- 负载均衡：即使 COIL-full 的最大 bucket 是 CITADEL 的 8×，CITADEL 通过 $\mathcal{L}_b$ 使各键 bucket 大小接近均匀，消除长尾瓶颈

**关键技术细节**：
- 路由激活函数：$\phi(v_j) = \log(1+\text{ReLU}(W^T v_j + b))$（与 SPLADE 完全相同）
- Max pooling 序列级路由表示（优于 sum/mean pooling）
- Post-hoc pruning $\tau \in [0.5, 1.5]$，sweet spot $\tau=0.9$（MRR@10=0.362, 78.5GB, 3.95ms）
- PQ (nbits=2)：MRR@10=0.361, 13.3GB, 0.90ms（6× 索引压缩，4× 延迟降低）
- CLS token 路由到额外语义键（可选，提升精度约 0.002 MRR@10）

---

## 4. 实验对比

**数据集**：MS MARCO v1（8.8M），TREC DL 2019/2020（43/54 queries），BEIR（13 子集）

**评估指标**：MRR@10，nDCG@10，R@1000，GPU/CPU Search Latency (ms/query)，Index Storage (GB)

**对比基线**：

| 基线方法 | 类型 | 发表年份 |
|----------|------|----------|
| BM25 | 稀疏词袋检索 | 经典 |
| DPR-128/768 | 单向量密集检索 | 2020 |
| SPLADE/SPLADEv2 | 稀疏神经检索 | 2021 |
| COIL-tok/COIL-full | 精确匹配多向量 | 2021 |
| ColBERT/ColBERTv2 | All-to-all late interaction | 2020/2022 |
| ColBERT-PLAID | ColBERTv2 优化引擎 | 2022 |

**关键结果表格**：

MS MARCO Dev (trained with distillation/HN mining 组）：

| 方法 | MRR@10 | R@1k | GPU Latency (ms) | Index (GB) |
|------|--------|------|-----------------|------------|
| ColBERTv2 | 0.397 | 0.985 | 122 | 29.0 |
| ColBERT-PLAID | 0.397 | 0.984 | 55.0 | 22.1 |
| **CITADEL+** | **0.399** | **0.981** | **3.21** | 81.3 |
| CITADEL+ + PQ (nbits=2) | 0.396 (估) | - | **0.90** | **13.3** |

BEIR 平均 nDCG@10（distilled 组）：ColBERTv2=0.500，CITADEL+=0.493（-0.007，约 1.4% 差距）

---

## 5. 性能提升

**总体提升**：
CITADEL+ 以与 ColBERTv2 持平的 MS MARCO 质量（MRR@10=0.399 vs 0.397），实现 **40× GPU 加速**（3.21ms vs 122ms）；结合 PQ 可进一步达到 0.90ms/query，索引从 29GB 降至 13.3GB。

**最显著提升场景**：
- GPU 延迟：40× vs ColBERTv2，17× vs PLAID
- CPU 延迟：635ms vs ColBERTv2 3275ms（5×），但不如 PLAID 的 370ms
- 点积计算量减少：4213M → 10.5M（~400×），与 DPR-128 的 8.84M 相当

**提升较弱的场景**：
- TREC DL 2019/2020 nDCG@10：CITADEL+ 在 DL19/DL20 上低于 ColBERTv2（0.703/0.702 vs 0.744/0.750），但 t-test 无统计显著性（样本量小）
- BEIR 平均 nDCG@10：CITADEL+ 0.493 vs ColBERTv2 0.500（-1.4%）；去掉正则化后 CITADEL+(w/o reg) 可达 0.501，超越 ColBERTv2，但延迟更高

**消融实验分析**：
1. **[CLS] token 路由**：去除后 MRR@10 从 0.362→0.360，延迟从 3.95→1.64ms，精度-延迟权衡合理
2. **Document 路由键数量**：从 1→5 键，MRR@10 从 0.347→0.364，索引从 53.6→185GB；5 键是质量-效率 sweet spot
3. **Post-hoc pruning**：$\tau=0.5$→$1.1$，MRR@10 几乎不变（0.362→0.359），延迟从 18→0.61ms；$\tau>1.1$ 后质量开始明显下降
4. **负载均衡效果**：CITADEL 最大 bucket 为 COIL-full 的 1/8，是 47.9ms → 3.95ms 的核心原因

---

## 6. 方法局限与缺陷

**论文自述局限**：
1. TREC DL 2019/2020 上 CITADEL 未超越 ColBERTv2（但 t-test 无显著差异）
2. 正则化（$\mathcal{L}_s, \mathcal{L}_b$）影响 out-of-domain 泛化（BEIR-0.007 差距）

**独立分析发现的缺陷**：
1. **索引大小偏大（无 PQ）**：CITADEL+ 81.3GB > ColBERTv2 29GB，原因是 document 每 token 路由到 5 个键，实际存储 5 倍 embedding；与声明的"高效"存在矛盾，仅 PQ 后才达到更小（13.3GB）
2. **CPU 延迟未超 PLAID**：CITADEL+ CPU 635ms vs PLAID CPU 370ms，CPU 场景下 PLAID 仍更优（原因：向量倒排索引在 CPU 多线程上的局部性不如 PLAID 的 centroid 方案）
3. **正则化与域外泛化的内在矛盾**：L1 + 负载均衡使模型倾向于使用高频/泛化词汇键，可能削减低频但域外重要的 token 交互，BEIR 实验已验证这一矛盾
4. **训练开销**：四部分损失函数的调参（$\alpha, \beta$）增加训练复杂度，论文未提供超参敏感性分析
5. **路由键分析不充分**：论文定性分析了几个例子（图5），但未系统验证"路由是否真正基于语义而非词形"的核心声明

**【批判性审查】实验设计与声明一致性**：

| 审查维度 | 问题 | 结论 |
|----------|------|------|
| 基线完整性 | 对比了 PLAID（2022），但未对比 DESSERT（NeurIPS 2022 同期） | 轻度问题：DESSERT 与 CITADEL 思路不同但目标相同，缺少此对比 |
| 消融充分性 | 消融了 CLS、routing key 数、pruning 阈值，但未消融负载均衡损失的单独贡献 | 不完整：$\mathcal{L}_b$ 对延迟的独立贡献未量化，图3 仅定性对比 CITADEL vs COIL 分布 |
| 数据集偏差 | 训练在 MS MARCO，测试 in-domain 和 BEIR out-of-domain | 无偏差，BEIR 结果已显示域外性能略有损失 |
| 声明-数字一致 | 摘要"nearly 40× faster than ColBERTv2"：3.21ms vs 122ms ≈ 38× ✓ | 基本一致；"17× faster than PLAID"：3.21ms vs 55ms ≈ 17.1× ✓ |
| 适用范围泛化 | 声明"same or slightly better than ColBERTv2"：BEIR 上低 1.4%，TREC DL 上低约 0.04 nDCG@10 | 轻度过度泛化：in-domain 成立，out-of-domain 有差距，尤其去掉正则化才真正超越 ColBERTv2 |

**潜在的改进空间**：
1. **正则化强度自适应**：根据训练分布动态调整 $\alpha, \beta$，使域内和域外性能达到更好权衡
2. **路由器分离 encoder**：将 SPLADE 路由头与 ColBERT token embedding encoder 分离训练，避免多任务干扰
3. **多粒度路由键**：除词汇键外引入语义聚类键（类 PLAID centroid），实现词汇路由（精确）+ 语义路由（模糊）混合
4. **GPU 向量倒排索引优化**：当前 GPU 加速来自 batch 向量搜索，可借鉴 PLAID 的 padding-free kernel 思想进一步优化

---

## 7. 科研启发（面向博士研究生）

### 7.1 新问题定义方向
1. **视觉 token 的路由检索**：ColPali 风格的 patch-level late interaction 中，能否为视觉 patch embedding 学习"视觉词汇路由"？图像 patch 的语义聚类可作为视觉词汇键
2. **跨语言路由对齐**：对多语言检索，路由键是否可以跨语言对齐，使英文 query token 与中文 document token 通过同一语义键交互？
3. **动态路由在流式检索中的一致性**：文档实时入库时，已有倒排索引的路由分布会随新文档偏移，如何保持负载均衡的动态性？

### 7.2 新方法/技术迁移
1. **CITADEL 路由 → VDR 图像检索加速**：将 SPLADE 路由头接在视觉 encoder 后，为 patch embedding 学习视觉词汇路由，实现 patch-level 倒排索引检索（不同于 ColBERT 密集倒排）
2. **负载均衡损失 → 向量量化码本设计**：CITADEL 的 $\mathcal{L}_b$ 使 bucket 均匀；这与 VQ-VAE/FAISS 量化中的均匀码本使用问题高度相关，可用 Switch Transformer 的 balancing 方法改善 PQ/RQ 的码本利用率
3. **Post-hoc pruning → 渐进式检索系统**：将 $\tau$ 动态化为 per-query 阈值，根据查询难度（路由权重分布熵）调整精度-效率权衡

### 7.3 现有缺陷的改进思路
1. **正则化-泛化矛盾的解决**：引入 domain-adaptive 路由，对稀有域的 query 使用较低 L1 正则化强度（通过 domain ID 或 query 难度预测）
2. **CPU 场景改进**：对 CPU 部署，将词汇倒排索引与 PLAID 的 passage-level centroid IVF 混合——先用 CITADEL 路由筛选候选键，再用 PLAID 的 centroid pruning 精排，可能在 CPU 延迟上超过 PLAID
3. **索引大小降低**：文档端路由键数从 5 降至 3，PQ 提前应用到训练阶段（对比 post-hoc PQ），可使索引接近 ColBERTv2 大小同时保持延迟优势

### 7.4 跨领域联系与灵感
1. **Mixture-of-Experts（MoE）路由**：CITADEL 的 load balancing loss 完全等同于 Switch Transformer/GLaM 的 expert 路由均衡机制，可直接引用 MoE 领域的路由优化研究（entropy 正则化、噪声路由等）改进 CITADEL 的路由稳定性
2. **信息论视角**：token 路由到词汇键等同于对 token embedding 的有损压缩；最优路由问题等价于在信道容量约束下最大化 query-document 互信息，可借鉴率失真理论分析路由键数量-质量权衡

### 7.5 综合建议
CITADEL 在 GPU 推理场景是 2022 年最优解（3.21ms vs ColBERTv2 的 122ms），但 CPU 场景被 PLAID 超越，BEIR out-of-domain 有 1.4% 劣势。理解 CITADEL 对分析后续工作至关重要：WARP/IGP 等 2025 年工作解决的是 ColBERT 架构不变时的引擎加速，而 CITADEL 探索的是"改变 token 交互结构"的根本性方法，两者路线正交。

---

## 8. 参考文献图谱

### 文献分类表

| 文献名 | 作者/年份 | 角色 | 知识库状态 |
|--------|----------|------|-----------|
| ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction | Santhanam et al., 2022 | 质量对标基线 | 待分析 |
| PLAID: An Efficient Engine for Late Interaction Retrieval | Santhanam et al., 2022 | 延迟对比基线 | ✅ 已分析（序号31）|
| COIL: Revisit Exact Lexical Match in Information Retrieval | Gao et al., 2021 | 被批判文献（静态路由前驱）| 未收录 |
| SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval | Formal et al., 2021 | 路由器激活函数来源 | 未收录 |
| Switch Transformers: Scaling to Trillion Parameter Models | Fedus et al., 2022 | 负载均衡损失灵感来源 | 未收录 |
| BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation | Thakur et al., 2021 | 域外评测基准 | 未收录 |

---

## 推荐阅读列表

### P0 必读（方法基础，知识库未收录）
- ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction (Santhanam et al., NAACL 2022)
- COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List (Gao et al., NAACL 2021) — CITADEL 的直接前驱，静态路由的对比

### P1 重要（关键路线对比）
- SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval (Formal et al., 2021) — CITADEL 路由器激活函数的直接来源
- Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Fedus et al., 2022) — 负载均衡损失的技术来源

### P2 建议（同期演进工作）
- WARP: An Efficient Engine for Multi-Vector Retrieval (Bruch et al., SIGIR 2025) — 引擎加速路线，与 CITADEL 结构改变路线正交对比
- SLIM: Sparsified Late Interaction for Multi-Vector Retrieval with Inverted Indexes (2023) — 同样探索稀疏化 late interaction 的不同方案

### P3 参考（背景综述，选读）
- BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models (Thakur et al., NeurIPS 2021) — 域外评测标准

---

## mem0 知识库记录

- **问题域**：multi-vector late interaction retrieval, conditional token interaction, dynamic lexical routing, efficient MaxSim
- **方法关键词**：CITADEL, dynamic lexical routing, SPLADE router, inverted vector index, load balancing loss, L1 sparsity, post-hoc pruning, token routing view, $A \circ \sigma$ MaxSim
- **数据集**：MS MARCO v1, TREC DL 2019/2020, BEIR (13 subsets)
- **性能基准**：CITADEL+ GPU 3.21ms (40× vs ColBERTv2 122ms, 17× vs PLAID 55ms)；MRR@10=0.399 (≈ColBERTv2 0.397)；BEIR avg nDCG@10=0.493 vs ColBERTv2 0.500。With PQ: 0.90ms, 13.3GB
- **关联论文 ID**：ColBERTv2 (arXiv:2112.01488), PLAID (arXiv:2205.09707), COIL (arXiv:2104.07186), DESSERT (arXiv:2210.15748)
- **核心方法机制摘要**：CITADEL 用 SPLADE 风格的 MLM 头作为 token 路由器，将 token embedding 路由到学习得到的词汇键（每 document token 路由到 5 键），构建词汇键倒排索引；训练时用负载均衡损失（Switch Transformer 风格）确保 bucket 均匀；检索时 query token 的 top-1 路由键在对应 bucket 做向量搜索，点积数从 4213M 降至 10.5M，GPU 延迟 3.21ms
- **推荐下一轮阅读线索**：SLIM（稀疏 late interaction 另一路线）；WARP（引擎加速，与 CITADEL 路线正交对比，同为 SIGIR 2025）

---

## 跨论文联想补充

---
> **[关联更新]** 读《PLAID》（paper_id: arXiv:2205.09707，日期：2026-05-07）后的补充思考：
> PLAID 是本文的关键延迟对比基线。CITADEL GPU 3.21ms vs PLAID GPU 55ms（17×），但 PLAID CPU 370ms < CITADEL CPU 635ms。这揭示了两种加速路线的根本区别：PLAID 的 centroid 方案在 CPU 大批量并行化上更有优势，而 CITADEL 的词汇倒排索引在 GPU batch 向量搜索上更高效。对本文 §7 的影响：混合两种方案（CITADEL 路由 + PLAID centroid 精排）可能是 CPU/GPU 均衡的最优解。
