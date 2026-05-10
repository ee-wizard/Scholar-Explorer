---
title: "UNIFY: Unified Index for Range Filtered Approximate Nearest Neighbors Search"
paper_id: "arXiv:2412.02448 / PVLDB:10.14778/3717755.3717770"
analysis_date: "2026-05-08 14:00:00 +08:00"
main_tag: "filtered_anns"
tags:
  - "filtered_anns"
  - "range_filtering_anns"
  - "graph_index"
  - "HNSW"
  - "approximate_nearest_neighbor"
  - "unified_index"
  - "hybrid_filtering"
  - "incremental_update"
related_papers: "SeRF, iRangeGraph, β-WST, ACORN, NHQ, Filtered-DiskANN, HNSW"
---

# UNIFY: Unified Index for Range Filtered Approximate Nearest Neighbors Search

> 学术分析报告 | 主标签：filtered_anns | 论文标签：filtered_anns;range_filtering_anns;graph_index;HNSW;approximate_nearest_neighbor;unified_index;hybrid_filtering;incremental_update

**作者**：Anqi Liang, Zhongpu Chen, Pengcheng Zhang, Yitong Song, Bin Yao（SJTU + SWUFE + Tencent）
**发表**：PVLDB 18(4): 1118–1130, 2024（VLDB 2025）
**DOI**：10.14778/3717755.3717770

---

## 1. 问题定义

**背景**：RF-ANNS（Range Filtered Approximate Nearest Neighbors Search）是高维向量检索与数值属性区间约束的联合查询，在推荐系统、RAG、电商搜索等场景有广泛应用。

**形式化定义（Definition 1，RF-NNS）**：
- 给定数据集 $\mathcal{D} = \{v_1, \ldots, v_n\}$，每个向量 $v$ 关联一个属性值 $v[A]$（如时间、价格）；
- 查询 $Q = (q, [l,h], k)$：查询向量 $q$，属性区间 $[l,h]$，查找数量 $k$；
- 目标：在 $\mathcal{R} = \{v \in \mathcal{D} \mid l \leq v[A] \leq h\}$ 中找 $kNN'(q, \mathcal{R})$（近似结果），最大化 $\text{recall} = \frac{|kNN'(q,\mathcal{R}) \cap kNN(q,\mathcal{R})|}{\min(k, |\mathcal{R}|)}$。

**三种基本策略（全部存在内在局限）**：
- **Strategy A（预过滤）**：先按属性区间过滤，再在子集上做向量搜索。小区间高效，大区间开销随区间线性增长。
- **Strategy B（后过滤）**：先全量 ANNS 得 $k'$ 个候选，再过滤属性。大区间高效，小区间需要过多候选才能找到满足属性的结果，效率差。
- **Strategy C（混合过滤）**：同时融合向量相似度和属性区间约束，在包含区间内对象的图上做搜索。中等区间最优，但对不同区间的适应性需要额外设计。

**核心挑战**：
1. **单一策略不能覆盖全部区间**：没有任何一种策略在所有区间长度下都是最优的，但为每种策略维护独立索引会导致数据一致性问题和高维护成本。
2. **增量插入**：先驱方法 SeRF 构建压缩 HNSW，不支持在线增量插入，限制了其在动态数据场景中的应用。
3. **图inclusivity（包容性）**：要在一个统一图中同时支持三种策略，核心难题是如何保证：对于任意属性区间 $[l,h]$，在 $\mathcal{R}$ 上独立构建的 PG（Proximity Graph）是该统一图的子图——即统一图的边"包含"了所有可能子集上的 PG 边。

**问题重要性**：三策略统一索引问题此前未被解决——SeRF 仅支持 Strategy C，ADBV/Milvus 维护多个分离索引。UNIFY 是首个在单一 PG-based 索引中同时支持三种策略的方法。

---

## 2. 前人工作的方法缺陷

**ADBV（Alibaba）**：B-tree 属性索引 + PQ 向量索引，基于代价模型选择查询策略。PQ 压缩导致精度损失，中等区间表现较差（Figure 5 中被 HSIG 高出 1–2 个数量级）。

**Milvus**：按属性分区，每个分区建 HNSW，策略同 ADBV。中等区间需要在多个分区上搜索，性能下降；不支持增量插入时的整体索引更新。

**NGT / Vearch**：纯后过滤，适合大区间；小区间下需要搜索大量不满足属性约束的候选，性能崩溃。

**NHQ**：用"融合距离"（向量距离 + 属性距离）构建复合图索引，理论基础薄弱（向量距离和属性值本质上不同质，融合距离缺乏语义支撑），且在属性区间较小时精度不稳定。

**SeRF（最直接的 Strategy C 先驱）**：构建"2D Segment Graph"，对所有半界区间（$L=1$ 或 $R=n$）构建 HNSW，通过压缩扩展到任意区间。核心问题：**不支持增量插入**（构建时压缩方式不允许动态更新）；小区间和大区间性能较差；HNSW 重建时用固定边度，限制了查询性能调整的灵活性。

**ARKGraph**：为所有可能属性区间组合预构建 PG，通过压缩存储；查询时解压缩，引入额外开销，效率受损。

**"分离专用索引 + 自适应选择"方案**：为预过滤（Btree+暴力）、后过滤（HNSW）、混合过滤（SeRF）分别建专用索引，根据区间选最优策略。主要问题是：多索引间数据一致性维护成本高，增量插入需要同时维护多个索引，工程复杂度大幅增加。

**根因**：现有方法都是在单一策略或粗粒度策略切换上做优化，没有从图结构设计层面解决"单图包容所有区间子图"的根本问题。

---

## 3. 研究动机与提出方案

### 研究动机

UNIFY 的出发点是：能否设计一个单一的 PG-based 图索引，使得在任意属性区间 $[l,h]$ 上独立构建的 PG 都是该统一图的子图（graph inclusivity）？若能做到，则：
- **混合过滤**：提取出只含区间内对象的子图边，在子图上搜索，等效于在专用 PG 上做 ANNS；
- **后过滤**：在全量图上搜索，忽略属性约束，等效于标准 ANNS；
- **预过滤**：利用属性索引（skip list）快速定位区间内对象，做向量距离计算。
三种策略全部集成到同一图结构中，消除多索引的一致性问题，同时通过范围感知策略选择（range-aware selection）在不同区间长度下自动使用最优策略。

### 方法本质

UNIFY 的核心是 **HSIG（Hierarchical Segmented Inclusive Graph）**：一个类 HNSW 的层次图，其中每个对象的邻居边按属性分段（segment）存储（**分段邻接表**），每个分段槽存储该对象在对应分段上的 HNSW 边。通过 Theorem 1 证明的 inclusivity 性质，HSIG 近似包含了任意分段组合上 HNSW 的全部边，从而支持高效的混合过滤。同时，skip list 连接和全局边掩码（bitmap）分别为预过滤和后过滤提供高效支持。

### 理论基础：SIG 的 Inclusivity

**关键引理（Lemma 1）**：对 $r$ 个不相交数据集 $\mathcal{D}_1, \ldots, \mathcal{D}_r$，令 $\mathcal{U} = \cup_i \mathcal{D}_i$，则 $\forall v \in \mathcal{U}$，
$$kNN(v, \mathcal{U} \setminus \{v\}) \subseteq \bigcup_{i=1}^r kNN(v, \mathcal{D}_i \setminus \{v\})$$
直觉：并集上的 $k$ 近邻一定包含在各子集的 $k$ 近邻集合的并集中。

**推论（Theorem 1，SIG-kNNG 的 inclusivity）**：将 $\mathcal{D}$ 按属性分成 $S$ 个不相交子集，SIG-kNNG 对每个对象 $v$ 在每个子集 $\mathcal{D}_i$ 中分别搜索 $k$ 近邻（共 $S$ 个 kNN 搜索），将结果按子集分槽存储。则对任意子集组合的并集 $\mathcal{U}$，$kNNG_k(\mathcal{U}) \subseteq \mathbb{F}_k(\mathcal{D})$——即 SIG-kNNG 包含所有组合上的 kNNG 边，不需要枚举所有 $2^S - 1$ 种组合构建索引。

### 方法还原：HSIG 的三大结构

**（1）骨干图（Backbone Graph）——支持混合过滤**  
用 HNSW 替代 kNNG 作为基础图。每个对象 $v$ 的邻接表分为 $S$ 个 chunk：$\mathbb{G}[v][j]$ 存储 $v$ 在分段 $j$ 上 HNSW 邻居（最多 $M$ 条边，默认 $M=16$，$S=8$）。插入对象时，对每个分段 $j$ 独立执行 ANNSearch + Prune + 双向连接（Algorithm 4: BackboneConnectionsBuild）。混合过滤查询时（Algorithm 7），对于与查询区间 $[l,h]$ 相交的 $S'$ 个分段，从每个分段的 chunk 中各取 top-$\lceil m/S' \rceil$ 个邻居（Hybrid-S2 策略），避免额外距离计算；只将属性值在 $[l,h]$ 内的对象加入结果集（Line 15）。

**（2）Skip List 连接——支持预过滤**  
在每个对象的邻接表中额外存储一个 skip list 后继指针（$\mathbb{G}[v].\text{next}$），共同形成按属性值排序的 skip list 结构。预过滤时，通过层次化 skip list 连接快速定位区间内第一个对象，再线性扫描 $[l,h]$ 内所有对象的向量距离。

**（3）全局边掩码（Global Edge Masking）——支持后过滤**  
HSIG 的全局 HNSW 边（对整个数据集 $\mathcal{D}$ 构建的 HNSW）近似是 HSIG 骨干图边的子图（由 inclusivity 保证）。Algorithm 5（GlobalEdgeMasking）用 bitmap 标记哪些 HSIG 边属于全局 HNSW——后过滤时只沿着 bitmap=1 的边做 ANNS，等效于在全量 HNSW 上搜索，无需额外内存存储全量 HNSW。

**构建**（Algorithm 6, HSIGInsert）：对新对象 $v$，先确定所属分段 $i$ 和随机层级，然后对所有 $S$ 个分段依次建骨干图连接，再插入 skip list，最后做全局边掩码。支持增量插入——每次插入是独立的，不依赖预先完整的数据集。

**范围感知策略选择**：根据查询区间内对象数量 $Y$，使用阈值 $\tau_A$（默认 1% 数据集大小）和 $\tau_B$（默认 50% 数据集大小）选择策略：$Y \leq \tau_A$ 用预过滤，$Y \geq \tau_B$ 用后过滤，$\tau_A < Y < \tau_B$ 用混合过滤。阈值通过对历史样本查询的统计分析确定。

**时间复杂度分析**：
- **预过滤**：skip list 搜索 $O(\log n)$ + 区间扫描 $O(Y)$；范围感知保证 $Y < \tau_A$ 时为 $O(\log n)$。
- **后过滤**：全局 HNSW 搜索 $O(\log n)$。
- **混合过滤**：在 $n'$ 个对象（相交分段内）的 HNSW 上搜索 $O(\log n')$，$n' \leq n$，故为 $O(\log n)$。

### 核心创新

1. **SIG Inclusivity 定理（Theorem 1）**：从理论上证明 SIG-kNNG 满足 inclusivity，无需枚举 $2^S - 1$ 个组合——这是整个框架的理论基础。
2. **HSIG = 骨干图 + Skip List + 全局边掩码**：三种辅助结构融合到单一图中，首次实现预/后/混合三种策略在同一索引上的统一支持。
3. **增量插入支持**：基于 HNSW 的增量构建范式，无需批量重建，实验验证构建时间稳定（Figure 9）。
4. **近似 inclusivity 的实用验证**：HSIG 的实际 inclusiveness 约 80%（Figure 7），实验（Figure 8）证明 80% inclusiveness 的性能接近 Optimal HNSW，远优于无 inclusivity 的方案（MS-HNSW，线性搜索多个独立 HNSW）。

### 机制有效性解释

**Figure 5**（六数据集综合对比，$k=10$ 和 $k=100$）：HSIG 在所有数据集上的 QPS-Recall Pareto 曲线均优于所有基线。GIST1M 和 GloVe 上相比 ADBV 高出 2 个数量级；SIFT1M 上比 NHQ 高出一个数量级；所有数据集上比 SeRF 最高高出 2.29×。

**Figure 6**（GloVe，不同区间尺度）：HSIG-range-aware 在小、中、大三类区间均表现最优；甚至超过为每种策略分别优化的"Dedicated"方案（最高 1.1×），证明统一图结构本身比分离索引更高效（消除了冗余数据扫描）。

**Figure 9**（增量插入）：HSIG 每轮插入时间稳定（每次 200K 对象），而 SeRF 因不支持增量构建而时间线性增长；插入后 HSIG 的查询性能持续优于 SeRF @ 0.95 Recall。

**Figure 15**（可扩展性，10M–100M 对象）：索引大小和构建时间线性增长，查询延迟对数增长，@ 0.99 Recall 稳定——验证了 $O(n)$ 空间和 $O(\log n)$ 查询的理论分析。

---

## 4. 实验对比与性能提升

**数据集（6 个，含真实属性和随机属性）**：
| 数据集 | 维度 | 大小 | 属性类型 |
|--------|------|------|---------|
| SIFT1M | 128 | 1M | 图像特征，随机属性 |
| GIST1M | 960 | 1M | 图像，随机属性 |
| GloVe | 100 | 1.18M | 词嵌入，随机属性 |
| Msong | 420 | 0.99M | 音频，随机属性 |
| WIT-Image | 2048 | 1M | 图像，真实图像大小属性 |
| Paper | 200 | 2M | 文本，发表时间/主题等转换属性 |

**基线**：ADBV、Milvus、NHQ、NGT、Vearch、SeRF。

**关键作者报告结果**：
- **HSIG vs SeRF（最强 Strategy C 基线）**：全部数据集上 QPS 最高提升 2.29× @ 同等 Recall；SeRF 在小/大区间下性能退化，UNIFY 通过策略切换弥补。
- **HSIG-range-aware vs Dedicated（理论最强对比）**：超越分离专用索引方案最高 1.1×——证明统一图优于分离索引。
- **HSIG-pre vs Btree-pre**：小区间下 HSIG-pre 超越 Btree-pre 20.3% @ 0.9 Recall（skip list 层次化搜索优于 B-tree）。
- **HSIG-hybrid vs SeRF**：中等区间超越 SeRF 1.21× @ 0.95 Recall；SeRF 的固定边度限制了性能调整灵活性。
- **HSIG-post vs HNSW-post**：大区间超越 HNSW-post 37.5% @ 0.99 Recall（全局边掩码的 HNSW 路径质量略好于独立 HNSW，因为 HSIG 包含更多跨分段连接信息）。

**内存（Table 2）**：HSIG 内存比 SeRF 高约 1.5–2×（因为需要同时存储分段连接 + skip list + bitmap）；比 NHQ 高约 20–40×（NHQ 内存极小但精度差）；比 ADBV/Milvus 高约 50–100×（PQ 压缩方法内存极小）。

**构建时间（Table 2）**：HSIG 与 SeRF 相当（均比 ADBV/Milvus 慢 5–10×）；增量插入场景下 HSIG 有显著优势。

---

## 5. 方法局限与缺陷

**作者已知局限**：
- **近似 inclusivity**：HSIG 的 inclusiveness 约 80%（Figure 7），不完全满足 Theorem 1 的精确条件（因为 HNSW 本身是近似图，不是精确 kNNG）。论文通过实验论证 80% inclusiveness 足够，但没有给出近似误差的量化上界。
- **多属性支持尚不完善**：论文仅讨论单属性 RF-ANNS，多属性扩展（z-order 映射或多索引方案）被留作未来工作，没有实验验证。
- **阈值 τ_A, τ_B 依赖历史数据**：策略选择阈值来自历史查询统计，若数据分布大幅变化则需要重新标定（论文提出了基于阈值检测的自适应方法，但未给出实验）。

**分析者判断的独立局限**：
- **内存消耗较大**：HSIG 需要为每个对象存储 $S = 8$ 个分段的 HNSW 边（$M \times S = 128$ 条潜在边），加上 bitmap 和 skip list 指针，内存是单个 HNSW 的 4–8×（Table 2 中 HSIG 是 SeRF 的 ~1.8×，而 SeRF 本身已经比单 HNSW 大）。
- **实验数据集较小（最大 2M 对象）**：可扩展性实验（Section 5.8）测试到 100M，但是仅限 SIFT 数据集（128 维），对于高维数据集（如 WIT-Image 2048 维）在大规模场景下的内存和时间没有实验数据。
- **混合过滤的边选择是启发式的**：Hybrid-S2（从每个分段取 top-$\lceil m/S' \rceil$ 邻居）没有理论保证其优于 Hybrid-S1，只有经验验证（Figure 10）；参数 $m$ 的最优设置对不同数据集和区间长度有不同最优值。
- **对 SeRF 作为主要对比的依赖**：论文将 SeRF 作为最主要竞争对手（Strategy C 最强方法），但 iRangeGraph（SIGMOD 2025，比 UNIFY 更先进）未被纳入对比——UNIFY 发表时 iRangeGraph 可能尚未公开，但两者在同一问题上方法差异显著，互相比较有价值。
- **数据集属性大多是随机生成的**：6 个数据集中 4 个使用随机生成的数值属性（SIFT/GloVe/GloVe/Msong），只有 WIT-Image 和 Paper 使用真实属性。随机属性场景下三种策略的切换阈值和性能边界可能与真实场景有差异。

---

## 6. 科研启发

1. **近似 Graph Inclusivity 的形式化与量化**：UNIFY 通过实验验证了 ~80% inclusiveness 足够，但缺乏理论分析：inclusiveness 与搜索质量损失之间的关系是否有界？给定构建图的 HNSW 边误差率（approximate kNNG vs exact kNNG），HSIG 的 inclusiveness 有怎样的概率下界？这需要将 HNSW 的近似误差分析（如 Indyk & Xu 2023 的 Vamana 分析）与 SIG Lemma 1 的传递性结合，建立近似 inclusivity 的理论体系。

2. **动态分段与自适应 HSIG 构建**：UNIFY 使用等深直方图（equi-depth histogram）划分 $S$ 个固定分段，分段数 $S$ 是全局常量。研究问题：对于分布高度倾斜的属性（如时序数据中的长尾分布），自适应分段（如历史查询区间密度高的地方多分段，稀疏处少分段）能否在相同内存预算下获得更高的 inclusiveness？这需要将自适应数据结构（如 B-tree 分裂策略）与 SIG 的理论框架结合。

3. **HSIG 与 on-the-fly 图构造的结合**：iRangeGraph 在查询时按需构造专用图（on-the-fly），而 UNIFY/HSIG 在索引阶段预存完整边。两种范式在内存、构建时间、查询性能上有不同取舍。研究问题：能否设计一种"分层 on-the-fly + 预存 backbone"的混合方案——仅在精细粒度层做 on-the-fly 边选择，粗粒度层用 HSIG 的预存边导航？这可能在保持 iRangeGraph on-the-fly 的空间效率的同时，获得 UNIFY 混合过滤的搜索质量。

4. **RF-ANNS 的索引大小 vs 区间覆盖度 Pareto 最优性**：UNIFY（$O(nS)$ 内存）、iRangeGraph（$O(nm\log n)$ 内存）、β-WST（$O(N\log_\beta N)$ 内存）在空间开销与查询性能之间有不同取舍，且都是针对 RF-ANNS 问题的系列近似解法。研究问题：对于 RF-ANNS 查询的特定区间长度分布 $f(s)$，这三类方法的 Pareto 最优性边界在哪里？即在给定内存预算和查询延迟目标的条件下，哪种图构建范式（统一图/线段树图/完整独立索引树）理论上最优？这是一个开放的理论问题，需要建立统一的分析框架。

---

## 7. 参考文献图谱

### 文献分类表

| 文献名 | 作者/年份 | 角色 | 知识库状态 |
|--------|----------|------|-----------|
| SeRF (ARKGraph/2D segment graph) | [52] | 主要对比基线（Strategy C）| 未收录 |
| NHQ | [40] | 实验基线（融合距离图） | 未收录 |
| ADBV (Alibaba) | [43] | 实验基线（PQ + B-tree） | 未收录 |
| Milvus | [39] | 实验基线 | 未收录（系统） |
| HNSW | [23] Malkov & Yashunin, 2020 | 方法基础（骨干图基础结构）| 已收录（seq#39） |
| Filtered-DiskANN | [10] | 相关工作（categorical filtering） | 未收录 |
| iRangeGraph (Xu et al., SIGMOD 2025) | Xu et al. | 同领域最新方法（同时期） | 已收录（seq#68） |
| β-WST / Window Filters | Engels et al., ICML 2024 | 同领域方法（segment tree 路线） | 已收录（seq#69） |
| ACORN (Patel et al., SIGMOD 2024) | Patel et al. | 通用谓词过滤 ANNS | 已收录（seq#67） |

---

## 推荐阅读列表

### P0 必读（核心对比方法，知识库未收录）
- **SeRF: Segment Graph for Range-Filtering Approximate Nearest Neighbor Search** (SIGMOD 2024 / PACMMOD 2024) — UNIFY 最主要的对比基线，也是 iRangeGraph 的对比基线；2D segment graph + 半界区间压缩的设计思路与 UNIFY/iRangeGraph 均有直接关联，是理解本领域技术演进的关键论文。
- **NHQ: An Efficient and Robust Framework for Approximate Nearest Neighbor Search with Attribute Constraint** (NeurIPS 2024) — 实验基线，融合距离图方法的代表，提供与 UNIFY 的重要对比视角。

### P1 重要（延伸理解）
- **iRangeGraph** (seq#68) — 同期 SIGMOD 2025 工作，on-the-fly 图构造路线，与 UNIFY 的预存统一图路线形成最直接对比。
- **β-WST / Window Filters** (seq#69) — β-WST 路线，β-WST 查询算法（SuperPostfiltering）是 iRangeGraph 的主要基线，也与 UNIFY 的 Strategy C 方向有关联。

### P2 扩展阅读（已收录）
- **HNSW** (seq#39) — HSIG 的骨干图实现基础。

---

## mem0 记录块

```
paper_id: PVLDB:10.14778/3717755.3717770
title: UNIFY: Unified Index for Range Filtered Approximate Nearest Neighbors Search
venue: PVLDB 18(4), 2024 (VLDB 2025)
main_tag: filtered_anns
core_method: HSIG (Hierarchical Segmented Inclusive Graph) — single unified PG-based index supporting pre/post/hybrid filtering; backbone graph with segmented adjacency list (S=8 segments, M=16 edges/chunk) for hybrid; skip list connections for pre-filtering; global edge masking bitmap for post-filtering; range-aware strategy selection with thresholds τ_A, τ_B; incremental insertion supported
key_theorem: SIG inclusivity (Theorem 1) — for any segment combination union U, kNNG(U) ⊆ SIG; empirical inclusiveness ~80% in HSIG
key_result: up to 2.29x over SeRF on all 6 datasets; outperforms Dedicated (separate best indexes per strategy); logarithmic search complexity at 10M-100M scale
limitations: approximate inclusivity (~80%); high memory (4-8x single HNSW); single attribute only (multi-attribute as future work); τ_A/τ_B calibration depends on historical data
related: SeRF (main competitor), iRangeGraph (concurrent SIGMOD 2025, on-the-fly construction alternative), β-WST/Window Filters (segment tree route), ACORN (general predicate), HNSW (foundation)
```
