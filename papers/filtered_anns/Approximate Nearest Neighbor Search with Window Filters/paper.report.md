---
title: "Approximate Nearest Neighbor Search with Window Filters"
paper_id: "arXiv:2402.00943"
analysis_date: "2026-05-08 13:00:00 +08:00"
main_tag: "filtered_anns"
tags:
  - "filtered_anns"
  - "range_filtering_anns"
  - "graph_index"
  - "segment_tree"
  - "approximate_nearest_neighbor"
  - "window_search"
  - "tree_index"
related_papers: "iRangeGraph, Filtered-DiskANN, ACORN, DiskANN/Vamana, VBASE, UNIFY"
---

# Approximate Nearest Neighbor Search with Window Filters

> 学术分析报告 | 主标签：filtered_anns | 论文标签：filtered_anns;range_filtering_anns;graph_index;segment_tree;approximate_nearest_neighbor;window_search;tree_index

**作者**：Joshua Engels, Benjamin Landrum, Shangdi Yu, Laxman Dhulipala, Julian Shun（MIT CSAIL + University of Maryland）
**发表**：ICML 2024（Vienna）
**arXiv**：2402.00943

---

## 1. 问题定义

**背景**：Approximate Nearest Neighbor Search (ANNS) 在文本、图像、视频等向量嵌入检索中已被广泛研究。然而，真实系统中的搜索请求往往附带元数据过滤条件，例如"找最相似的图片，但仅限去年夏天拍摄的"、"找相关论坛帖子，但仅限某版本发布后的几天内"。这类过滤条件是对数值标签的区间约束，本文将其形式化为 **Window Search**。

**形式化定义**：
- **带标签数据集（Labeled Dataset）**：$(D, \ell)$，其中 $D \subset V$（度量空间），$\ell: V \to \mathbb{R}$ 为标签函数（如时间戳、价格等数值属性）。
- **窗口过滤数据集（Window Filtered Dataset）**：$D_{(a,b)} = \{x \in D \mid \ell(x) \in (a,b)\}$，窗口过滤器为开区间 $(a,b)$。
- **窗口搜索（Window Search）**：给定查询 $q \in V$ 和窗口过滤器 $(a,b)$，找 $q^* = \arg\min_{x \in D_{(a,b)}} \text{dist}_V(x, q)$，即标签在 $(a,b)$ 内且向量最近的点。
- **$c$-近似窗口搜索**：返回 $y \in D_{(a,b)}$ 满足 $\text{dist}_V(q, y) \leq c \cdot \text{dist}_V(q, q^*)$。推广到 top-10 ANN（实验标准）。

**问题特殊性**：与 label-filtering（标签取有限离散值）不同，window search 中的标签连续且区间任意，无法预设固定的标签集合；与全量 ANNS 不同，在范围之外的点即使向量更近也不是有效答案。这两点使得现有的两种朴素策略（预过滤和后过滤）在不同的过滤粒度下各有致命弱点，且已有的 Filtered-DiskANN、CAPS 等 label-filtering 方法不直接适用。

**过滤粒度（Filter Fraction）**：$2^i$（$i \leq 0$）表示窗口覆盖数据集中 $2^i$ 比例的点。大区间（如 $2^{-2}$）、中区间（如 $2^{-8}$）、小区间（如 $2^{-15}$）对应完全不同的挑战性。

---

## 2. 前人工作的方法缺陷

**预过滤（Prefiltering）**：对数据集按标签排序，查询时二分搜索找到满足 $(a,b)$ 的区间子集，然后做线性扫描找最近邻。当区间大（覆盖比例高）时，对大数量子集进行线性扫描代价极高；无法利用向量空间结构加速搜索。

**后过滤（Postfiltering）**：先在全量数据上做 ANNS，然后对结果过滤不满足窗口的点，逐步扩大 top-k 直到找到满足条件的结果。当区间小（选择率低）时，满足条件的点在向量空间中可能距离查询向量很远，后过滤需要极大的 k 才能找到；Adverse 数据集实验（Figure 3）清晰展示了当目标点所在的 cluster 与查询点所在的 cluster 完全不同时，所有后过滤方法召回率崩溃。

**VBASE（Zhang et al., 2023）**：通用 filtered ANNS 系统，在图搜索过程中用遍历顺序近似相对距离，再后过滤满足谓词的结果。实验（Figure 2）表明 VBASE 在中等过滤粒度下被本文最朴素的基线（Prefiltering/Postfiltering）完全支配，无竞争力。

**Milvus（IVF/HNSW + bitset 过滤）**：对 HNSW 图搜索中不满足窗口过滤的点用 bitset 标记跳过（In-filtering 策略）。当小区间导致图上几乎没有满足条件的邻居时，搜索路径退化，效率极差；实验同样被本文方法完全支配。

**Filtered-DiskANN / CAPS**：设计针对布尔标签的过滤 ANNS，依赖有限标签集合构建专用图。不支持连续数值区间过滤；将 range 近似为离散桶分配标签时，粒度损失严重。

**本文核心诊断**：现有方法的根本问题在于缺乏利用数值标签连续性和区间特性的专用数据结构——所有方法都把窗口过滤当作通用谓词处理，忽视了"区间可以由若干预建子集的并集精确覆盖"这一结构性质。

---

## 3. 研究动机与提出方案

### 研究动机

如果将数据集按标签值排序，任意窗口区间 $(a,b)$ 都对应排序数组的一个子区间 $[L, R]$。标准线段树能将任意 $[L, R]$ 分解为 $O(\log N)$ 个预建的不相交节点区间的并集。**β-WST 的核心思想**：在每个线段树节点存储一个完整的 ANNS 索引（而非仅存储基础信息），查询时递归找到刚好覆盖窗口区间的若干节点，在这些节点的索引上查询并合并结果——此时在每个节点上的搜索都是标准 ANNS 问题（无需过滤），从而规避了预过滤/后过滤的固有缺陷。

### 方法本质

β-WST 的本质是**将 window search 归约为若干个标准 ANNS 子问题的组合**：通过在线段树每个节点预存完整 ANNS 索引，使得任意窗口查询都能找到一组节点，每个节点的索引数据集刚好被查询窗口完全覆盖（即查询时不需要任何过滤），最后合并各节点的搜索结果。整体空间复杂度为 $O(N \log_\beta N)$（每层 $N$ 个点，共 $\log_\beta N$ 层），查询时间为 $O(\beta \log_\beta N \cdot A_q)$（每层查询 $\beta$ 个节点，每个节点索引大小递减）。

### 方法还原

**索引构建（Algorithm 1，BuildTree）**：
- 按标签值对数据集 $D$ 排序，递归构建 $\beta$ 叉线段树（$\beta=2$ 为二叉树）。
- 在每个内部节点，对该节点对应子集构建完整的 ANNS 索引（本文使用 Vamana 图索引，$\alpha=1$，degree=64，建图 beam width=500）。
- 递归地将子集均分为 $\beta$ 份，对每份子集建子树。叶节点（子集大小 $< \beta$，实现中 base case=1000）直接存储点，不建索引。
- 一个关键的工程优化：整个数据集的向量只存储一份（而非在每个节点复制），节约内存。

**查询（Algorithm 2，Tree Query）**：
- 输入：β-WST $T$，查询点 $q$，窗口 $(a,b)$。
- 若当前节点为叶，暴力扫描满足 $(a,b)$ 的点，找最近邻。
- 若窗口 $(a,b)$ 完全覆盖当前节点所有点的标签范围，则直接调用该节点的 ANNS 索引查 $q$——**此时无需过滤**。
- 否则，遍历 $\beta$ 个子节点，对每个子节点判断其标签范围与 $(a,b)$ 是否有交集，有则递归查询，合并候选集，返回最近邻。

**查询时间上界（Theorem 5.2）**：若 ANNS 算法 $A$ 对大小为 $m$ 的子集的查询时间为 $O(A_q(D,m))$，则 Algorithm 2 的运行时间为
$$O\!\left(\beta \log_\beta(N) d + \beta \sum_{j=0}^{\log_\beta N} A_q(D, N \cdot \beta^{-j})\right)$$
其中 $d$ 为一次距离计算的时间。当 $A_q(D,m) = O(Cdm^\rho)$（$\rho \in (0,1)$）时，运行时间为 $O\!\left(\frac{C\beta dN^\rho}{1-\beta^{-\rho}}\right)$，与单次 ANNS 同阶。

**SuperPostfiltering（Algorithm 3, OptimizedPostfiltering）**：不用 tree query，而是在整棵树中找**最小的能完全包含** $(a,b)$ 的节点索引，然后在该索引上做后过滤（逐渐扩大 top-k）。其关键特点是后过滤的 blowup factor（目标索引大小 vs 实际过滤子集大小的比值）有界——但最坏情况下 blowup 为 $O(N)$（Lemma 5.6）。实际实验中 SuperPostfiltering 在中等过滤精度下表现最好（约 0.9–0.99 Recall），但在 Adverse（对抗数据集）上失败。

**ThreeSplit（Algorithm 4）**：混合策略。先找窗口中间段完全被节点覆盖的最大子集，对该子集做 Tree Query；然后对左右两侧剩余区间各做一次 OptimizedPostfiltering。这保证了任何情况下不会出现"大 blowup"的 OptimizedPostfiltering 调用，是 SuperPostfiltering 和 Tree Query 的平衡。

**$\beta$ 值影响**：更大的 $\beta$ 减少树的层数（$\log_\beta N$），降低内存和构建时间，但每次查询时需要递归遍历更多子节点（每层 $\beta$ 个），且每个内部节点索引更小导致 ANNS 质量更低，因此 recall 和延迟变差（Figure 4 实验验证）。实验中 $\beta=2$ 为最优配置。

### 核心创新

1. **首次形式化 Window Search 问题**：明确区分 window search 与 label-filtering ANNS 的差异，填补文献空白。
2. **β-WST：模块化树形框架**：利用线段树节点分解将任意 window query 归约为若干无需过滤的标准 ANNS 子查询，理论上适用于任何 ANNS 算法。
3. **理论运行时间上界（Theorem 5.2, Lemma 5.3, Lemma 5.4）**：结合 Vamana 的近期分析，给出完整的复杂度分析，包括倍增维度（doubling dimension）参数化的结果。
4. **Blowup 分析框架（Definition 5.5, Theorem 5.7）**：量化"最优后过滤"的代价——最坏情况 blowup factor $2\gamma$ 可在空间 $O(N \log_\gamma N)$ 下实现，优于 β-WST 的 $O(N)$ blowup 最坏情况。
5. **对抗数据集（Adverse）**：主动构造了使后过滤方法失败的对抗场景，验证了 β-WST 的鲁棒性来源于无需过滤的搜索设计。

### 机制有效性解释

**Figure 2**（主要对比图，Deep 数据集不同过滤粒度）：在中等过滤粒度（$2^{-7}$ 至 $2^{-9}$）下，β-WST 系列方法比 Prefiltering/Postfiltering/Milvus/VBASE 高出数倍到数十倍 QPS @ 同等 recall。大过滤粒度（接近全量）时 Postfiltering 有竞争力；小过滤粒度（极小区间）时 Prefiltering 有竞争力。本文方法在中间地带的优势最显著。

**Figure 3**（Adverse 对抗数据集）：VamanaWST 和 ThreeSplit 能维持良好 recall，而所有依赖后过滤的方法（包括 Postfiltering、OptimizedPostfiltering、SuperPostfiltering）召回率崩溃。这直接验证了 tree query "无需过滤"设计的鲁棒性来源。

**Table 3**（各数据集最高加速比，0.95 Recall）：Deep 最高 75×，SIFT 最高 16×，GloVe 最高 9×，Redcaps 最高 17×，均发生在中等过滤粒度。

---

## 4. 实验对比与性能提升

**数据集（5 个，含真实标签和对抗场景）**：
| 数据集 | 类型 | 维度 | 大小 | 标签来源 |
|--------|------|------|------|---------|
| SIFT | 图像特征 | 128 | 1M | 均匀随机 |
| GloVe | 词嵌入 | 100 | 1.18M | 均匀随机 |
| Deep | GoogLeNet | 96 | 9.9M | 均匀随机 |
| Redcaps | CLIP 图像 | 512 | 11.6M | 真实时间戳 |
| Adverse | 混合高斯 | 100 | 1M | 对抗构造 |

**基线**：Prefiltering、Postfiltering、VBASE、Milvus（HNSW/IVF-PQ/IVF-SQ8/SCANN/IVF-FLAT）。

**本文方法**：VamanaWST（即 β=2 的 β-WST，Algorithm 2）、OptimizedPostfiltering（Algorithm 3）、ThreeSplit（Algorithm 4）、SuperPostfiltering（Theorem 5.7 分析的方案，最坏 blowup=4γ）。

**关键作者报告数据**（@ 0.95 Recall，16 线程）：
- **Deep（9.9M，96维）**：最高 75× 加速（过滤粒度 $2^{-7}$）；
- **SIFT（1M，128维）**：最高 16× 加速（$2^{-6}$）；
- **GloVe（1.18M，100维）**：最高 9× 加速（$2^{-5}$）；
- **Redcaps（11.6M，512维，真实时间戳）**：最高 17× 加速（$2^{-6}$）。
- **Adverse（对抗）**：VamanaWST/ThreeSplit 正常工作；Postfiltering 类方法召回率崩溃。

**内存（Table 5）**：
- 2-WST 内存：单 Vamana 索引的 3–8×；
- SuperPostfiltering 内存：单 Vamana 索引的 5–14×；
- 以 Deep 为例：Raw 3.6GB，Vamana 6.8GB，2-WST 53.2GB，Super 94.6GB。

**构建时间（Table 4，80 线程）**：
- 2-WST vs 单 Vamana：5–10× 更慢（Deep: 2h vs 17min，Redcaps: 7h vs 2h）；
- SuperPostfiltering vs 单 Vamana：10–20× 更慢。

**性能边界条件**：
- 大过滤粒度（$2^{-0}$–$2^{-1}$）时，Postfiltering 竞争力相当，本文方法无显著优势（甚至略慢）；
- 极小过滤粒度（$2^{-11}$）时，Prefiltering 仍有竞争力（线性扫描代价低），本文方法加速比下降到 1–10×；
- $\beta$ 增大时内存显著降低（$\beta=8$ vs $\beta=2$ 内存约降 2×），但 recall 和延迟变差（Figure 4）。

---

## 5. 方法局限与缺陷

**作者已知局限**：
- **大 blowup（Lemma 5.6）**：β-WST（$\beta=2$）的 OptimizedPostfiltering 最坏 blowup 为 $O(N)$，导致 OptimizedPostfiltering 在某些情况下表现极差（Figure 2 中 OptimizedPostfiltering 是本文方法中最差的）。
- **内存消耗大**：Deep 数据集下 2-WST 需要 53.2GB（Raw 数据 3.6GB 的 14.8×），高维大数据集下索引内存极大；Redcaps（11.6M，512维）需 79.2GB。
- **构建时间长**：Redcaps 上 2-WST 需 7 小时（80 线程），SuperPostfiltering 需 19 小时，不适合频繁重建的场景。
- **未支持动态更新**：线段树结构不支持增量插入/删除，需要完全重建。

**分析者判断的独立局限**：
- **仅支持单属性 range filter**：多属性联合过滤（如 range AND category）不在本文讨论范围内；扩展思路未给出。
- **理论查询上界的实用性有限**：Theorem 5.2 中的 $A_q(D, N \cdot \beta^{-j})$ 项在 Vamana 的理论分析（Indyk & Xu, 2023 的"slow-preprocessing"模型）下仍有 $O(N^3)$ 的构建时间假设，与实际 Vamana 的线性构建时间不符；理论结果与实验结果之间有模型假设差距。
- **标签分布假设**：SIFT/GloVe/Deep 使用均匀随机标签，不代表真实场景中标签分布的偏斜（如时序数据中查询区间集中在最近时间段）；只有 Redcaps 使用真实时间戳，但也仅 11.6M 数据。
- **对 Milvus 比较不完全公平**：Milvus 测试采用 Python multiprocessing（不是原生并发），且测试了 HNSW、IVF-PQ、IVF-SQ8、SCANN、IVF-FLAT 五种索引（结果图中显示最好的），未做完全等价的系统优化对比。
- **索引节点互相独立，无共享利用**：β-WST 在每个节点完全独立建 ANNS 索引，无法复用父节点的图结构信息（与 iRangeGraph 的 bottom-up 复用相比，这是 iRangeGraph 构建效率更高的原因之一）。

---

## 6. 科研启发

1. **Window Search 的在线自适应树结构**：β-WST 的树结构在构建期确定，无法根据历史查询分布动态调整分支点。研究问题：若知道查询区间的分布 $f(a,b)$，能否在线构建一棵非均匀线段树，使得期望查询时间最小？这需要最优树结构的理论分析，并探索与 BST 最优化（Knuth 1971 最优 BST）的联系。

2. **多属性 Window Search 的积结构理论**：当查询条件为多个数值属性的联合区间约束（$a_1 \leq x_1 \leq b_1 \wedge a_2 \leq x_2 \leq b_2$）时，β-WST 的单属性线段树结构无法直接推广。研究问题：能否在 2D 或 $k$D 属性空间中构建类似的"积线段树"，保证对任意矩形查询区间的分解复杂度为 $O(\log^k N)$？这涉及正交区间搜索与 ANNS 的深度融合，目前尚无系统研究。

3. **blowup-cost 权衡的最优设计**：Theorem 5.7 证明了存在 blowup factor $2\gamma$、cost $O(N \log_\gamma N)$ 的范围集合，但未给出构造算法的实用实现与实验对比。研究问题：在给定内存预算 $B$ 的条件下，能否设计最优的范围集合构造算法，同时最小化最坏情况 blowup 和平均查询延迟？需要结合查询分布建模和随机化优化技术。

4. **动态 β-WST**：现有实现不支持增量插入/删除，而实际向量数据库对动态更新有强需求。研究问题：能否设计一种延迟重建（lazy rebuild）或局部更新策略，使得对于一个插入或删除操作，只需重建 $O(\log N)$ 个受影响树节点的 ANNS 索引（类似 B+ tree 的分裂/合并）？关键挑战在于保证动态过程中查询的正确性（在节点部分更新时不遗漏满足条件的点）。

---

## 7. 参考文献图谱

### 文献分类表

| 文献名 | 作者/年份 | 角色 | 知识库状态 |
|--------|----------|------|-----------|
| Filtered-DiskANN (Gollapudi et al., WWW 2023) | Gollapudi et al., 2023 | 实验对比基础 / 相关工作 | 未收录 |
| VBASE (Zhang et al., 2023) | Zhang et al., 2023 | 实验基线 | 未收录 |
| Milvus | Wang et al., 2021 | 实验基线 | 未收录（系统） |
| Vamana / DiskANN | Subramanya et al., NeurIPS 2019 | 方法基础（内部 ANNS 索引） | 已收录（seq#40） |
| HNSW | Malkov & Yashunin, 2020 | 相关基础索引 | 已收录（seq#39） |
| ParlayANN | Manohar et al., 2024 | 工程基础（代码库） | 未收录 |
| Indyk & Xu, 2023 | Vamana 理论分析 | 理论分析基础 | 未收录 |
| iRangeGraph (Xu et al., SIGMOD 2025) | Xu et al., 2025 | 后续对比方法 / 知识库 | 已收录（seq#68） |
| 2DSegmentGraph | Zhang et al., 2024 | 知识库未收录；被 iRangeGraph 引用为基线 | 未收录 |

---

## 推荐阅读列表

### P0 必读（方法基础与核心对比）
- **iRangeGraph: Improvising Range-dedicated Graphs for Range-filtering Nearest Neighbor Search** (Xu et al., SIGMOD 2025) — 已收录（seq#68）；iRangeGraph 将本文的 SuperPostFiltering 作为主要对比基线，空间上相比 β-WST 更高效，构建效率更高，建议对比阅读。
- **Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters** (Gollapudi et al., WWW 2023) — 最直接的 label-filtering ANNS 对比基线，理解其局限有助于明确 window search 的独特挑战。

### P1 重要（延伸理解）
- **UNIFY: Unified Index for Range Filtered Approximate Nearest Neighbors Search** (arXiv:2412.02448) — 最新统一 range-filtering 方法，当前 pipeline 待分析。
- **ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data** (SIGMOD 2024) — 已收录（seq#67）；通用谓词过滤 ANNS，与 window search 思路互补。

### P2 扩展阅读（已收录）
- **DiskANN/Vamana** (seq#40) — β-WST 内部 ANNS 索引的实现基础。

---

## mem0 记录块

```
paper_id: arXiv:2402.00943
title: Approximate Nearest Neighbor Search with Window Filters
venue: ICML 2024
main_tag: filtered_anns
core_method: β-WST (β-Window Search Tree) — stores full Vamana ANNS index at every segment tree node; query recurses tree, queries node index where window fully covers node's label range; tree query is filter-free; also proposes OptimizedPostfiltering, ThreeSplit, SuperPostfiltering as alternative query methods
key_result: up to 75x speedup over baselines @ 0.95 Recall; medium filter fractions see largest gains; Adverse dataset proves robustness of tree query vs. postfiltering methods; 2-WST memory 3-8x single Vamana index
limitations: high memory (Deep: 53GB for 9.9M 96-dim); long build time (Redcaps: 7h, 80 threads); no dynamic update; single attribute only; OptimizedPostfiltering has worst-case O(N) blowup
related: iRangeGraph (competitor, cites SuperPostfiltering from this paper as main baseline), Filtered-DiskANN (label-filter baseline), ACORN (general predicate), UNIFY (follow-up), DiskANN/Vamana (inner index)
note: "SuperPostFiltering" that iRangeGraph cites as [29] is one of the query algorithms (Algorithm 3/Theorem 5.7) from this paper, not a standalone paper
```
