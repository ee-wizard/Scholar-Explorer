---
title: "Attribute Filtering in Approximate Nearest Neighbor Search: An In-depth Experimental Study"
paper_id: "filtering-ann-survey-2026"
analysis_date: "2026-05-07 09:30:00 +08:00"
main_tag: "filtered_anns"
tags:
  - "filtered_anns"
  - "approximate_nearest_neighbor"
  - "attribute_filtering"
  - "vector_search"
  - "experimental_study"
  - "benchmark"
  - "index_structure"
  - "pruning"
related_papers: "SeRF, DSG, iRangeGraph, UNIFY, β-WST, FDiskANN, NHQ, ACORN, Milvus, Faiss, DiskANN, HNSW"
---

# Attribute Filtering in Approximate Nearest Neighbor Search: An In-depth Experimental Study

> 学术分析报告

> **主标签**：filtered_anns  
> **论文标签**：filtered_anns; approximate_nearest_neighbor; attribute_filtering; vector_search; experimental_study; benchmark; index_structure; pruning

---

## 1. 问题定义

**问题背景**：

向量数据库中的 ANN 搜索通常忽略结构化属性约束，而实际场景往往需要二者融合，例如"找视觉上相似的商品，但价格必须在 500–800 元之间"或"找相似图片，但只在特定用户拍摄的范围内"。这类融合了属性约束的 ANN 搜索被称为 **Filtering ANN（过滤近似最近邻搜索）**。

近年来 Filtering ANN 算法大量涌现，分别针对不同属性类型（数值/分类/任意）和不同过滤策略（预过滤/后过滤/联合过滤），但彼此之间缺乏统一比较。本文是该方向的首篇全面实验性综述。

**形式化定义**：

给定含属性的数据集 $\mathcal{D} = \{O_i\}$，查询 $Q = (q, r)$，目标是高效返回：

$$NN(q|r) = \arg\min_{o \in \mathcal{D},\, r(o.a)} \|q - o.v\|$$

属性约束类型：
- **范围过滤**：$l \leq O_i.a \leq u$（数值属性）
- **标签过滤**：$f \in O_i.A$（分类属性）
- **任意过滤**：用户自定义过滤函数

**问题重要性**：

Filtering ANN 是向量数据库的核心能力（Milvus、Qdrant、Pinecone 均支持），也是 RAG 系统、推荐系统、多模态搜索的关键路径。但目前没有统一的评测框架，各算法在不同数据/选择率下的真实表现缺乏客观对比。

---

## 2. 前人工作的方法缺陷

| 缺陷类型 | 具体表现 | 代表工作 |
|----------|----------|----------|
| 朴素后过滤 | ANN 先不管属性，最后过滤；低选择率时大量候选无效 | Faiss-HNSW |
| 朴素预过滤 | 先对属性过滤缩小数据集，再在子集上 ANN；打破图结构完整性 | Milvus 基础模式 |
| 单一属性类型设计 | 每类方法只针对数值或分类，不能处理另一类，更无法任意过滤 | SeRF (range only)、FDiskANN (label only) |
| 评测不统一 | 各论文使用不同数据集、超参、实现，无法横向比较 | SeRF vs iRangeGraph vs β-WST |
| 图连通性问题 | 图索引方法在低选择率（<1%）下因 RNG 单调搜索性，无法找到有效邻居 | SeRF、DSG、Faiss-HNSW |
| 层次结构收益夸大 | 分层 HNSW 结构在 Filtering ANN 中实际仅在高选择率（>50%）才有显著收益 | UNIFY（层次设计） |

---

## 3. 研究动机与评测框架设计

### 研究动机

本文的出发点是一个被大量算法论文所忽视的元问题：**Filtering ANN 算法之间的对比实验从未在统一条件下做过**。现有论文各自对比不同 baseline、使用不同数据集和参数，导致 SeRF vs iRangeGraph vs UNIFY 这类问题无法通过阅读原始论文得到可信答案。与此同时，每篇算法论文都声称自己在某一场景最优，但没有人系统地问：**是什么组件（剪枝/入口点/属性索引）决定了性能差异？**

本文的动机不是提出新算法，而是建立一张精确的"方法地图"——搞清楚现有方法在哪里有效、为什么有效、在哪里失效、为什么失效。

### 本文本质贡献

> *"In this work, we present a unified Filtering ANN search interface that encompasses the latest algorithms and evaluate them extensively from multiple perspectives."* —— Abstract

从分析者视角看，本文的贡献不在于"综述了这些方法"，而在于**揭示了三个设计决策维度**（剪枝策略、入口点选择、边过滤开销），并用受控实验证明了它们各自对性能的影响量级。这三个维度在各个原始算法论文中从未被独立评测。

### 统一分类框架（§3，Table 3）

本文沿两个正交轴建立分类：

**轴1：过滤属性类型**
- **范围过滤（Range）**：$r = (l, u)$，数值属性区间约束。代表：SeRF, DSG, β-WST, UNIFY, iRangeGraph
- **标签过滤（Label）**：$r = f$，分类属性匹配。代表：FDiskANN, NHQ
- **任意过滤（Arbitrary）**：用户自定义谓词函数。代表：Faiss, ACORN, Milvus

**轴2：过滤策略**
- **预过滤（Pre-filtering）**：先按属性缩小搜索空间，再做 ANN。缺点是破坏图连通性。
- **后过滤（Post-filtering）**：先做完整 ANN，最后过滤不合格候选。高选择率效率高，低选择率浪费大。
- **联合过滤（Joint-filtering）**：在图遍历中直接跳过不满足约束的节点。中等选择率最有效。

多数"高级方法"（UNIFY、iRangeGraph、ACORN）实际上是**根据查询的选择率自适应选择策略**，而非单一策略。

### 统一评测平台设计（§5.1）

评测平台的关键决策：
- 所有图方法统一 M=40、ef\_construction=1000（保证公平性）
- 单线程查询（消除并行化带来的噪声）
- 评测指标：QPS（高优）+ Comparisons per query（图方法专用）；固定 90% recall@10 下比较
- 数据集：SIFT(10M, 128d)、SpaceV(10M, 100d)、Redcaps(1M, 512d, 真实标签)、Youtube-RGB(1M, 1024d, 真实标签)；选择率从 0.1% 到 100%
- 合成属性用均匀分布；真实属性来自数据集本身的分类标签

### 三大关键组件实验设计（§4，§5.6–5.8）

这三个组件实验是本文最有方法论价值的部分：

**（1）剪枝策略**（§5.6）：同一图结构（SeRF）分别用 RNG 剪枝 vs KGraph 风格（保留所有近邻），比较在不同选择率下的 recall/QPS 曲线；同时对 ACORN 比较两跳剪枝 vs 标准 RNG。

**（2）入口点选择**（§5.7）：固定其他参数，对 UNIFY 比较"有层次结构 vs 无层次结构"；对 SeRF/DSG 比较"3个入口 vs 30个入口 vs 300个入口"；测量 comparisons 减少量。

**（3）边过滤开销**（§5.8）：将 DSG 和 SeRF 的查询时间分解为"向量比较时间"和"边过滤时间"，证明 DSG 更低的 comparisons 数量被更高的边检查开销所抵消。

---

## 4. 实验对比与性能提升

### 4.1 范围过滤（§5.2）

以下结论均为 author-reported，在 SIFT(10M)/SpaceV(10M) 上 90% recall@10 基准：

| 选择率区间 | 最优方案 | 核心原因 |
|-----------|---------|---------|
| 0.1% | iRangeGraph、β-WST-Vamana | BST 细粒度子图；图搜索单调性不会失效 |
| 1%–10% | iRangeGraph、UNIFY-hybrid、SeRF、WST-opt 相当 | 所有专用范围过滤方法均有效 |
| >50% | Faiss-HNSW 即可竞争 | 属性感知索引的增量收益下降 |

关键负面发现（§5.2, 观察6）：

> *"Segmented edges fail at low selectivity. SeRF and DSG fail at 0.1% — graph-based monotonicity issue."* —— §5.2

SeRF 和 DSG 在 0.1% 选择率下完全失败（无法达到 90% recall），即便增大 ef\_search 也无效。根本原因是构建时图的 RNG 剪枝单调性——当过滤后的有效邻居极少时，贪心搜索无法"跨越"到其他有效节点。

**IVF 基准的反直觉表现**（§5.2, 观察2）：

Faiss-IVFPQ 在低选择率下始终可用，因为 IVF 的分区非单调搜索不受图搜索单调性限制，在稀疏有效集合下仍能找到候选。这说明**量化基方法的鲁棒性系统性地优于图基方法，代价是更低的绝对 QPS**。

### 4.2 标签过滤（§5.3）

> *"Label filtering algorithms remain underdeveloped; no method performs reliably across all settings."* —— §6.1, I4

FDiskANN-SVG 在 cardinality=1 时最鲁棒，但在 SIFT 上因底层 Vamana 图质量不足而失败（无法达到 90% recall，见 §5.3, 观察1）。NHQ-KGraph 在高选择率（>10%）有竞争力，但 <1% 时连通性崩溃（§5.3, 观察2）。

一个 analyst assessment：标签过滤领域的根本问题是"标签分布的不均匀性"与"图连通性的均匀性假设"之间的矛盾——稀有标签的节点天然孤立，任何基于图遍历的方法都需要专门处理这类"图中的孤岛"。

### 4.3 任意过滤（§5.4）

| 选择率 | 最优 | 次优 |
|-------|------|------|
| >50% | Faiss-HNSW | ACORN |
| 10%–50% | ACORN | Faiss-HNSW |
| <1% | Milvus-HNSW（分区扫描） | Faiss-IVFPQ |

Milvus 在低选择率下的稳定性来自"强制提高分区内有效节点密度"——64个分区相当于每个分区内有效节点密度提升64倍，本质上是将低选择率转化为中等选择率（§5.4, 观察2）。

### 4.4 索引构建开销（§5.5，Table 5，SIFT 10M）

| 方法 | 索引大小 (GB) | 内存 (GB) | 构建时间 (s) |
|------|-------------|---------|------------|
| SeRF | 6.96 | 10.44 | 364 |
| iRangeGraph | 24.27 | 44.72 | **69,801（≈19.4h）** |
| UNIFY | 30.25 | 36.96 | 17,721 |
| WST-opt | 40.22 | **97.69** | 8,048 |
| ACORN | 10.45 | **109.33** | 3,859 |
| DSG | 39.35 | 40.35 | 2,905 |
| Faiss-HNSW | 7.90 | 7.99 | 1,733 |

iRangeGraph 的19小时构建时间是单线程顺序建 2n-1 个子图所致——论文指出这部分**易于并行化**，但未在论文中提供并行化结果（analyst assessment：这是一个系统工程缺口，不是算法缺陷）。

### 4.5 组件分析关键数据（§5.6–5.8）

**剪枝策略**（§5.6）：SeRF\_kg（KGraph剪枝）在 <5% 选择率下优于 SeRF\_rng（RNG剪枝），但在 0.1% 时仍失败，说明剪枝策略改变不足以解决低选择率的根本连通性问题。

**入口点**（§5.7）：SeRF 使用 300 个底层入口点（vs 默认 3 个），在 1% 选择率下 comparisons 减少 5.53%，在 50% 下减少 3.85%；300个底层入口的效果**优于 UNIFY 的层次结构**。

**边过滤开销**（§5.8）：DSG 的向量比较次数少于 SeRF，但总查询时间更长——边过滤时间（检查每条边是否满足区间约束）随 ef\_search 增大线性增长，成为 DSG 的主要瓶颈。

---

## 5. 方法局限性

1. **合成属性均匀分布**：所有范围过滤实验使用 $U[0, 100000]$ 合成属性，真实数据（如日志时间戳、价格分布）往往是 Zipf 或长尾分布，均匀分布下的结论能否迁移未验证。

2. **单线程查询基准**：实际向量数据库在高并发场景下运行，多线程查询可能改变各方法之间的相对 QPS 排名（尤其是 iRangeGraph 等内存密集型方法的缓存行为）。

3. **静态数据集**：除 DSG 外所有方法不支持动态更新，实验也未评测增量插入场景，排除了动态数据湖场景的考量。

4. **最大 10M 规模**：工业界向量数据库常见 100M–1B 规模，本文最大 10M，无法反映超大规模下的磁盘 I/O 影响（如 DiskANN 的 SSD 优化场景）。

5. **选型指南缺少"构建预算"维度**：图15 的决策树仅按属性类型和选择率分类，未将"构建时间容忍度"纳入（iRangeGraph 构建19小时与 SeRF 构建6分钟，使用场景差异极大）。

---

## 6. 关联论文与可信推荐阅读

**基础论文（本文直接评测/引用的核心方法）**：

1. Peng et al., "SeRF: Segment Graph for Range-Filtering Approximate Nearest Neighbor Search" (SIGMOD 2024 PACMMOD) — 分段边方案，本文评测中范围过滤代表基线。
2. Peng et al., "iRangeGraph: Improvising Range-dedicated Graphs for Range-filtering Nearest Neighbor Search" (SIGMOD 2025) — BST + 合并入口，全实验中查询性能最优的范围过滤方法。
3. Wei et al., "UNIFY: A Unified Index for Range Filtered Approximate Nearest Neighbors Search" (PVLDB 2024) — 固定分区 + 自适应策略，本文发现其层次结构在低选择率下无益。
4. Gollapudi et al., "Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters" (WWW 2023) — 标签过滤分支中最鲁棒的方案（SVG变体）。
5. Zhang et al., "ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data" (SIGMOD 2024) — 任意过滤唯一专用方案；两跳邻居设计。
6. Dong et al., "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs" (TPAMI 2018) — HNSW 基础，多数被评测方法的底层图结构。
7. Jayaram Subramanya et al., "DiskANN: Fast Accurate Billion-Point Nearest Neighbor Search on a Single Node" (NeurIPS 2019) — Vamana Graph，FDiskANN 和 β-WST 的基础。

**后续扩展推荐**：
1. 属性分布对 Filtering ANN 性能影响的实验研究（目前论文未做，O5 开放问题）。
2. 动态 Filtering ANN 方向（DSG 是唯一已知方案，O4 开放问题）。
3. NHQ 原论文（NeurIPS 2024）——联合距离方法的详细理论分析。
4. β-WST 原论文（ICML 2024，arXiv:2402.00943）——BST 子图方案的理论界分析。

---

## 7. 研究启发

1. **图搜索单调性是 Filtering ANN 低选择率的系统性障碍**：本文用实验证明 RNG 剪枝 + 图遍历单调性在 <1% 选择率下几乎必然失效。研究问题：能否在图构建时显式加入"跨稀疏属性子集的长程边"（类似 DiskANN 保留远端连接的动机），在不显著增大图度数的前提下解决低选择率连通性？

2. **入口点数量是被低估的杠杆**：本文证明 300 个底层入口点优于层次 HNSW 结构。研究问题：属性感知的入口点选择策略（如：优先选择属性值均匀分布在查询区间内的向量作为入口）是否能进一步提升效果，是否存在最优入口点集合大小的理论界？

3. **标签过滤方向是真正的方法论空白**：本文确认任何现有方法都无法在所有标签过滤场景稳定达到 90% recall@10。研究问题：标签分布的不均匀性（稀有标签的孤立节点）是核心难题，能否设计一种"标签感知的图增强"策略——对稀有标签节点强制保留跨标签的近邻边，以牺牲少量图质量换取连通性保障？

4. **边过滤开销是 DSG 类"动态范围边"方案的系统性成本**：DSG 的设计选择（每条边存多个范围）使构建灵活但查询受限。研究问题：能否用"批量区间预计算"或"区间掩码位图压缩"在 O(1) 时间内完成边过滤，将 DSG 类方案的查询开销降至 SeRF 级别？

5. **统一 Range + Label + Arbitrary 过滤的可行性**：目前三类过滤分属三套方法体系，无单一方法覆盖全部。研究问题：能否设计一个通用的"属性索引层"，将范围属性、分类属性、任意谓词统一表达为对向量子集的动态选择，再对接任意 ANN 图索引？ACORN 已向这个方向迈出一步，但内存代价过大。

---

## 8. 参考文献关系图

```
HNSW (Malkov 2018) ────────────────────────┬──▶ SeRF（分段边 + HNSW）
                                            ├──▶ UNIFY（固定分区 + 跨子图边 + HNSW）
                                            ├──▶ ACORN（两跳扩展 + HNSW）
                                            └──▶ Milvus（分区 HNSW）
Vamana/DiskANN (Subramanya 2019) ──────────┬──▶ FDiskANN（标签边 + Stitched Graph）
                                            └──▶ β-WST（BST + 每节点 Vamana）
SeRF ──▶ DSG（动态插入变体，SeRF 的动态版）
β-WST ──▶ iRangeGraph（共享 BST 结构；iRangeGraph 用合并入口代替后过滤合并）
                                            
本文（综述）评测以上全部，揭示：
  - RNG 剪枝在低选择率失效（影响 SeRF/DSG/FDiskANN/ACORN）
  - 层次结构在低选择率无益（影响 UNIFY）
  - 边过滤开销是 DSG 的关键瓶颈
  - 底层多入口点是所有图方法的通用改进
```

---

## mem0 record

```json
{
  "paper": "Attribute Filtering in ANN Search: An In-depth Experimental Study",
  "venue": "SIGMOD 2026, arXiv:2508.16263",
  "year": 2026,
  "type": "experimental_survey",
  "task": "Filtering Approximate Nearest Neighbor Search (Filtering ANN)",
  "scope": "12 methods from 10 papers; 4 datasets; selectivity 0.1%-100%; range/label/arbitrary filtering",
  "key_findings": {
    "I1": "Fine-grained segmented subgraphs (iRangeGraph, β-WST) best for range filtering",
    "I2": "Segmented edge methods (SeRF, DSG) fail at <1% selectivity due to graph monotonicity",
    "I3": "Partitioning (Milvus) is stable fallback for low selectivity",
    "I4": "Label filtering has no reliable solution across all settings",
    "I6": "Hierarchy only benefits >50% selectivity",
    "I7": "More bottom-layer entry points consistently improve all methods"
  },
  "open_problems": ["arbitrary filtering robustness", "adaptive hyperparameters", "dynamic indexing", "non-uniform attribute distributions"],
  "best_methods_by_scenario": {
    "range_low_sel": "iRangeGraph or β-WST-Vamana",
    "range_medium": "iRangeGraph or UNIFY-hybrid",
    "range_high": "Faiss-HNSW",
    "label_any": "FDiskANN-SVG",
    "arbitrary_high": "Faiss-HNSW",
    "arbitrary_medium": "ACORN",
    "arbitrary_low": "Milvus-HNSW"
  },
  "related": ["SeRF", "iRangeGraph", "UNIFY", "β-WST", "FDiskANN", "NHQ", "ACORN", "Milvus", "DiskANN", "HNSW"]
}
```
