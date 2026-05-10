---
title: "iRangeGraph: Improvising Range-dedicated Graphs for Range-filtering Nearest Neighbor Search"
paper_id: "arXiv:2409.02571"
analysis_date: "2026-05-08 12:30:00 +08:00"
main_tag: "filtered_anns"
tags:
  - "filtered_anns"
  - "range_filtering_anns"
  - "graph_index"
  - "segment_tree"
  - "approximate_nearest_neighbor"
  - "on_the_fly_construction"
related_papers: "2DSegmentGraph, Filtered-DiskANN, SuperPostFiltering, HNSW, ACORN, UNIFY"
---

# iRangeGraph: Improvising Range-dedicated Graphs for Range-filtering Nearest Neighbor Search

> 学术分析报告 | 主标签：filtered_anns | 论文标签：filtered_anns;range_filtering_anns;graph_index;segment_tree;approximate_nearest_neighbor;on_the_fly_construction

**作者**：Yuexuan Xu*, Jianyang Gao*, Yutong Gou, Cheng Long†（NTU），Christian S. Jensen（Aalborg University）
**发表**：SIGMOD 2025（Berlin）
**arXiv**：2409.02571

---

## 1. 问题定义

**背景**：Range-filtering Approximate Nearest Neighbor (RFANN) 查询是一类重要的混合查询，在工业场景（Apple、Milvus/Zilliz、Alibaba ADBV）和学术界均受到关注。给定数据对象集合，每个对象包含一个高维向量和一个数值属性值，RFANN 查询接受一个查询向量和一个数值区间作为参数，返回属性值在区间内、且向量与查询向量最近的对象。

**形式化定义**：  
- 数据对象：$O = (v, a)$，$v$ 为高维向量，$a$ 为数值属性值。
- 数据集：$\mathcal{D} = \{O_1, \ldots, O_n\}$，按 $a$ 升序排列后，对象索引与其属性排名一一对应（通过二分搜索可将原始区间 $[a_l, a_r]$ 转换为排名区间 $[L, R]$）。
- 查询：$Q = (q, a_l, a_r)$，$q$ 为查询向量，$[a_l, a_r]$ 为数值区间。
- 目标：$\arg\min_{O \in \mathcal{D},\, a_l \leq O.a \leq a_r} \delta(q, O.v)$，其中 $\delta$ 为距离函数。推广到 top-$k$ 版本即找 $k$ 个最近邻。
- 准确率度量：$\text{Recall@}k = |G \cap S| / k$，$G$ 为真实近邻集，$S$ 为算法结果集。

**关键挑战**：
1. **查询区间长度可变性**：不同查询的区间长度可能跨越多个数量级（从极小区间到全量数据），没有一种单一搜索策略在所有区间长度下都表现最优。
2. **空间效率与搜索质量的权衡**：为每个可能的查询区间 $[L,R]$（共 $O(n^2)$ 种）构建专用图索引，空间复杂度为 $O(n^3 m)$，完全不可行。有损压缩后虽然降低了空间，但导致短区间查询召回率不足（现有方法 2DSegmentGraph 在小区间下无法达到 0.8 Recall）。
3. **动态区间适应性**：混合查询工作负载（即 query workload 中同时存在大、中、小区间的查询）要求单一索引能处理所有情况，而基于预设策略（如 post-filtering）的方法对某些区间范围表现极差。

**问题重要性**：RFANN 查询是 range-filtering 版本的 Filtered ANN，与 label-filtering（如 Filtered-DiskANN）相比，数值区间的连续性和可变性使得 label-filtering 的标签分区思路难以直接适用；需要专门设计针对区间特性的索引结构。

---

## 2. 前人工作的方法缺陷

**三种基础策略的内在缺陷**均是已知的，但尚无方法彻底解决：

- **预过滤（Pre-filtering）**：先过滤出在范围内的对象，然后线性扫描找最近邻。当区间选择率高（区间大）时退化为对大数据集的顺序扫描，时间复杂度 $O(|[L,R]| \cdot d)$，效率极差。
- **后过滤（Post-filtering）**：先做全量 ANN 搜索，再过滤不满足区间的结果。当区间很小（选择率低）时，需要访问大量不在范围内的对象才能找到满足条件的近邻；实际复杂度远高于 $O(\log n)$。
- **内过滤（In-filtering）**：将区间过滤嵌入图搜索，只遍历在范围内的邻居。问题是固定图中某节点的在范围内邻居数量取决于查询区间，区间很小时图连通性严重退化，近邻可能不可达，召回率大幅下降。

**SuperPostFiltering（并发工作，arXiv:2402.01373）**：基于线段树预设多个重叠区间并为每个区间建一个 graph-based 索引，查询时找覆盖查询区间的最小预设区间，然后用 Post-filtering 在该区间上搜索。虽然缩小了后过滤的范围（最多 $4 \cdot |[L,R]|$ 个对象），但仍有最多 $3 \cdot |[L,R]|$ 个不在范围内的对象被访问，Post-filtering 的固有缺陷未被完全消除。

**2DSegmentGraph**：构想是为所有 $O(n^2)$ 个可能的查询区间都建专用图，有损压缩后空间可接受，但压缩引入的质量损失使得小区间（高选择率）查询无法达到 0.8 Recall——压缩保证只对半界区间（$L=1$ 或 $R=n$）成立，一般区间性能不稳定。

**Filtered-DiskANN（标签过滤）**：将全量范围 $[1,n]$ 均匀划分为 10 个桶并分配标签，然后做标签过滤 ANN。这是一种粗糙近似，与真实 RFANN 查询场景不匹配，小区间时性能尤其差。

**根因**：上述所有方法的根本问题在于无法为任意查询区间提供高质量的专用图索引，同时又控制空间开销。缺少一种能在查询时按需高效构造专用图的机制。

---

## 3. 研究动机与提出方案

### 研究动机

iRangeGraph 的出发点是：如果能为任意查询区间 $[L,R]$ 在查询期间高效构造一个专用 RNG-based 图索引，则 RFANN 查询就等价于在该专用图上的标准 ANN 查询，从而避免三种基础策略的固有缺陷。关键约束是：(1) 索引空间必须远小于 $O(n^3m)$；(2) 查询期间构造专用图的开销必须远小于在图上搜索的代价。

**Oracle Partition（理想基准）**：若为每个查询的具体区间预先构建一个 HNSW，则 RFANN 即为标准 ANN 查询。实验（Section 5.2.4）表明 iRangeGraph 的 QPS 与此 Oracle-HNSW 的差距不超过 2× @ 0.9 Recall，而 Oracle 的空间复杂度为 $O(n^3m)$，完全不可行。

### 方法本质

iRangeGraph 的本质是：**利用线段树的递归区间分解特性，预先构建 $O(n\log n)$ 个基础（elemental）图，使得任意查询区间可以由 $O(\log n)$ 个线段树分段覆盖，并在查询时通过高效的边选择算法（分摊 $O(m + \log n)$）按需拼装出对应该区间的专用 RNG-based 图索引**。

本质上是将"为所有可能区间预建索引"转换为"为 $O(n\log n)$ 个标准分段建基础图，查询时按需合成"，以亚线性空间开销换取接近 Oracle 的搜索质量。

### 方法还原

**核心数据结构：线段树基础图**  
构建一棵线段树（$O(\log n)$ 层），每层将 $[1,n]$ 划分为 $2^i$ 个等长分段。对每个分段，在该分段包含的对象上构建一个近似 RNG-based 图（称为 elemental graph），最大出度为 $m$。线段树的 RNG 递推性质（Theorem 3.1）使得整体构建复杂度仅比构建单个 HNSW 高一个次对数因子：对于分段 $[l,r]$，其子分段 $[l,mid]$ 的 elemental graph 中已有对象 $u$ 到其他分段内对象的边——这些边在 $[l,r]$ 的 RNG 中不会被 $[l,mid]$ 内的对象剪枝（单调性），因此可以直接复用；只有来自另一个子分段 $[mid+1,r]$ 的候选边需要重新搜索。

**空间复杂度**：每个数据对象在每层恰好出现一次，每层每对象最多 $m$ 条边，共 $O(\log n)$ 层，总空间为 $O(nm\log n)$——远小于为所有区间显式存储专用图的 $O(n^3m)$。

**查询期间专用图构造（Algorithm 1，Edge Selection）**：  
给定查询区间 $[L,R]$，对某个待访问对象 $u$，从线段树根节点开始自顶向下选择其边：
1. 若当前节点的子分段 $[l_c,r_c]$ 与查询区间的交集等于其父分段 $[l,r]$ 与查询区间的交集（说明两层对 $u$ 的约束程度相同），则跳过该层（skip）；
2. 否则，从当前层的 elemental graph 中选取属于 $[L,R]$ 的边，加入 $u$ 的当前选边集 $S$；
3. 若 $|S| \geq m$ 或当前分段已被查询区间完全覆盖，则停止。

"跳过层"机制是降低开销的关键，使得分摊时间复杂度从朴素的 $O(m\log n)$ 降至 $O(m + \log n)$（Theorem 3.2）。

**搜索**：在专用图上做标准 Greedy Beam Search，边在访问邻居时按需构造（on-the-fly）。整体搜索流程与标准 graph-based ANN 相同，只是邻居列表在访问时才通过 Edge Selection 动态生成。

### 核心假设

- 数值属性排序后，对象索引与属性排名一一对应；RFANN 查询等价于在排名区间内做 ANN。
- 构建近似 RNG 而非精确 RNG——基于现有 graph-based ANN 方法的通用假设。
- 专用图不保证完全等价于从头构建的 RNG（因为 elemental graph 中存在来自区间外对象的错误剪枝），但实验上与 Oracle-HNSW 的差距 $< 2\times$。

### 核心创新

1. **线段树 + On-the-fly 专用图构造**：将"为所有区间预建"转换为"为标准分段预建+查询时拼合"，打破了空间与性能之间的原始取舍。
2. **Layer-skipping Edge Selection（Algorithm 1）**：利用线段树的递归交集关系，证明分摊复杂度为 $O(m + \log n)$，使得构图开销相对搜索开销可忽略不计。
3. **Bottom-up 构建利用 RNG 单调性**：证明父分段 RNG 中来自同侧子分段的边可以从子分段 elemental graph 中直接复用，避免重复搜索。
4. **多属性扩展 + 随机跳过策略**：对第二个属性引入以 $p = \exp(-t)$ 衰减的随机访问概率，在 In-filtering 和 Post-filtering 之间平滑插值，实验上带来 70% QPS 提升。

### 机制有效性解释

**图 2**（主要性能对比）是最关键的图表：iRangeGraph 是唯一在所有数据集和所有区间长度（large/moderate/small/mixed）下均能稳定达到 0.9 Recall 的方法。这直接证明了 on-the-fly 专用图构造消除了三种基础策略在不同区间长度下的性能瓶颈。在混合 workload 下，iRangeGraph vs 最具竞争力的基线 SuperPostFiltering 的 QPS 差距为 2×–5×（大多数数据集），vs Milvus 为 2×–300×。

**图 4**（Oracle-HNSW 对比）验证了方法的理论上界：iRangeGraph 与不可行的 Oracle-HNSW 的性能差距 $< 2\times$，证明 edge selection 算法引入的图质量损失极小。

**图 3**（消融实验）：(1) 使用专用图 vs 分别在 $O(\log n)$ 个分段上独立搜索再合并（BasicSearch 基线），QPS 提升 2×–4×；(2) 使用 layer-skipping edge selection vs 朴素按层遍历，性能一致提升——证明跳层机制不以搜索质量为代价，只降低开销。

---

## 4. 实验对比与性能提升

**数据集（5 个真实数据集，各 100 万对象，1000 查询）**：
| 数据集 | 向量类型 | 维度 | 属性类型 |
|--------|---------|------|---------|
| WIT-Image | 图像 | 2,048 | 图像大小 |
| TripClick | 文本 | 768 | 发布日期 |
| Redcaps | 多模态 | 512 | 时间戳 |
| YT-Rgb | 视频 | 1,024 | 点赞数、评论数 |
| YT-Audio | 音频 | 128 | 发布时间、播放量 |

**查询 workload**：按区间覆盖比例分为大（$2^{-0}$–$2^{-3}$）、中（$2^{-4}$–$2^{-6}$）、小（$2^{-7}$–$2^{-9}$）三挡及混合 workload。

**基线**：2DSegmentGraph、Filtered-DiskANN（FilteredVamana+StitchedVamana，改造为 range filtering）、Milvus（HNSW）、SuperPostFiltering、Pre-filtering。

**关键作者报告结果**：
- **iRangeGraph vs SuperPostFiltering**（最具竞争力基线）：@0.9 Recall，WIT/TripClick/Redcaps/YT-Rgb 上 QPS 2×–5× 提升；YT-Audio（128维）性能相当（低维时构图开销相对搜索开销比例更高）。
- **iRangeGraph vs Milvus**：@0.9 Recall，大多数数据集 2×–300× QPS 提升。
- **iRangeGraph vs Oracle-HNSW**：差距 $< 2\times$ @ 0.9 Recall（大多数数据集）。
- **内存（Table 2）**：iRangeGraph 内存小于 SuperPostFiltering，大于 Milvus 和 Pre-filtering。以 WIT 为例：iRangeGraph 12.87GB vs SuperPostFiltering 28.98GB vs Milvus 8.38GB（原始向量 7.7GB）。
- **索引时间（Table 3）**：iRangeGraph 比 SuperPostFiltering 更快（WIT: 3,776s vs 11,603s），比 Milvus 慢（WIT: 3,776s vs 784s）。2DSegmentGraph 和 FilteredVamana/StitchedVamana 虽然索引更快，但小区间查询失败。
- **多属性结果（Figure 5）**：iRangeGraph 比 2DSegmentGraph 快 $> 2\times$ @ 0.9 Recall；iRangeGraph+ 在 YT-Rgb 上比 Pre-filtering 快 5×，在 YT-Audio 上快 35× @ 0.9 Recall。

**性能边界条件**：
- YT-Audio（128 维）是 iRangeGraph 相对表现最弱的数据集，低维下构图开销不可忽略。
- 极小区间（$2^{-9}$ 以下），Pre-filtering 是最优策略（线性扫描对象极少），iRangeGraph 仍优于 Pre-filtering @ 0.9 Recall（未要求完美 Recall 时）。
- 2DSegmentGraph 的压缩策略在小区间下完全失效（这是其根本局限）。

---

## 5. 方法局限与缺陷

**作者明确限制**：
- **YT-Audio 表现相对一般**：128 维低维数据集下，on-the-fly 构图的固定开销（$O(m + \log n)$）相对于搜索开销（$O(md)$）比例更高，导致优势缩小。
- **多属性扩展不支持完美 Recall**：iRangeGraph 多属性版本无法在所有情况下达到 1.0 Recall（由 elemental graph 剪枝引入的图质量损失，以及 Post-filtering 的固有不完美性）。
- **动态更新**：论文明确指出"动态插入/删除"是未来方向，当前方法不支持增量更新。
- **空间开销不可忽略**：WIT 数据集（2048 维）下内存为原始向量的 1.67×；低维数据集（YT-Audio 128 维）下内存为原始向量的 4.15×——低维数据集的内存开销相对较大。

**分析者判断的独立局限**：
- **构建时间较长**：WIT 数据集上 3,776 秒（约 1 小时），对于需要频繁重建的场景不友好。
- **仅支持单数值属性的 RFANN 优化**：对等值过滤（categorical labels）类型谓词没有优化；扩展到多属性时性能保证减弱。
- **近似 RNG 引入的质量损失没有界**：Theorem 3.2 只保证 $O(m + \log n)$ 的分摊构图时间，但构造出的专用图与真实 RNG 的差距没有量化分析（只有经验验证）。
- **评测协议局限**：仅使用 5 个 1M 对象的数据集，未测试 10M/100M 规模；scalability 实验（Section 5.2.3）放在技术报告而非主论文，主论文缺少完整展示。
- **区间长度分布假设**：实验中查询区间位置随机生成，但真实场景中区间位置分布可能有偏斜（如时序数据中"最近 N 天"的查询集中在区间右端），这种分布偏差可能影响性能。

---

## 6. 科研启发

1. **线段树 on-the-fly 图合成的理论质量保证**：当前 iRangeGraph 构造的专用图与真实 RNG 的差距仅有经验支撑（差距 $< 2\times$），缺乏理论界。研究问题：给定数据集分布和区间长度 $s$，elemental graph 的 out-of-range 对象引起的错误剪枝率是否有与 $s$ 相关的上界？能否借助 RNG 的几何性质（高维空间中 RNG 边的"稳定性"）给出图质量损失的概率保证？这将把 iRangeGraph 从纯经验方法提升为有理论背书的近似方法。

2. **可自适应区间长度的多分支线段树**：iRangeGraph 使用标准二叉线段树，空间复杂度 $O(nm\log n)$。论文指出未来可用多分支树（$b$-ary segment tree）减少层数，降低内存。研究问题：在给定空间预算 $B$ 和查询区间长度分布 $f(s)$ 的条件下，如何确定最优树分支数 $b$（或非均匀分支结构），使得 edge selection 的期望开销最小？这需要新的优化框架和对 $f(s)$ 的建模。

3. **跨谓词类型的统一 On-the-fly 图索引框架**：iRangeGraph 的线段树机制依赖数值区间的全序性（可以按排名划分线段）。对于集合包含谓词（如 ACORN 处理的 keyword list）、等值谓词（标签过滤）或复合谓词（range AND label），能否设计一个统一的 on-the-fly 图构造框架？关键研究问题是：什么样的谓词结构允许"基础索引单元"的有效合成，使得合成结果图的质量有界？这是将 iRangeGraph 的思路推广到通用属性过滤 ANN 的关键理论问题。

4. **增量/流式更新的 Elemental Graph 维护**：iRangeGraph 当前不支持动态更新。RFANN 的动态更新问题比标准 ANN 更复杂——插入一个对象 $O_{\text{new}} = (v, a)$ 不仅改变包含该对象的分段的 elemental graph，还可能改变其他分段中已有对象的 RNG 边（因为 $O_{\text{new}}$ 可能成为新的剪枝依据）。研究问题：能否设计局部更新算法，使得插入/删除一个对象时只需更新 $O(\log n)$ 个受影响分段中的 $O(m)$ 条边，而不需要完全重建？这需要分析 RNG 边的局部性（locality）在动态场景中的保持条件。

---

## 7. 参考文献图谱

### 文献分类表

| 文献名 | 作者/年份 | 角色 | 知识库状态 |
|--------|----------|------|-----------|
| 2DSegmentGraph | Zhang et al., 2024 | 实验基线 / 被批判文献 | 未收录 |
| SuperPostFiltering (arXiv:2402.01373) | Engels et al., 2024 | 实验基线 / 并发工作 | 未收录（即 β-WST/Window Filters） |
| Filtered-DiskANN | Gollapudi et al., WWW 2023 | 实验基线 | 未收录 |
| Milvus | Wang et al., 2021 | 实验基线 | 未收录（系统） |
| HNSW | Malkov & Yashunin, 2020 | 方法基础（构建基础图） | 已收录（seq#39） |
| NSG | Fu et al., 2019 | 方法基础（RNG 近似） | 未收录 |
| DiskANN | Subramanya et al., 2019 | 方法基础 | 已收录（seq#40） |
| Segment Tree | Bentley, 1977 | 方法基础（核心数据结构） | 不适用（算法基础） |
| Apple HQI | Douze et al., 2024 | 背景综述 | 未收录 |
| ADBV (Alibaba) | Zhang et al., 2019 | 背景综述 | 未收录 |

---

## 推荐阅读列表

### P0 必读（方法基础，知识库未收录）
- **2DSegmentGraph: Dedicated Graph-based Indexes for Range Filtering Nearest Neighbor Search** (Zhang et al., 2024) — iRangeGraph 最直接对比的 range-filtering 前驱，理解其有损压缩导致小区间性能差的机制是读懂 iRangeGraph 的前提。
- **Approximate Nearest Neighbor Search with Window Filters** (Engels et al., 2024, arXiv:2402.00943) — 即 SuperPostFiltering/β-WST，iRangeGraph 最具竞争力的基线，基于线段树的后过滤方法。

### P1 重要（延伸理解）
- **UNIFY: Unified Index for Range Filtered Approximate Nearest Neighbors Search** (arXiv:2412.02448) — 最新的统一 range-filtering 方法，与 iRangeGraph 同领域的后续进展。
- **ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data** (Patel et al., SIGMOD 2024) — 已收录（seq#67）；通用谓词过滤 ANN，与 iRangeGraph 专注 range 谓词的路线互补。

### P2 扩展阅读（已收录）
- **HNSW** (seq#39) — elemental graph 构建基础结构。
- **DiskANN** (seq#40) — RNG-based 图构建的另一基础参考。

---

## mem0 记录块

```
paper_id: arXiv:2409.02571
title: iRangeGraph: Improvising Range-dedicated Graphs for Range-filtering Nearest Neighbor Search
venue: SIGMOD 2025
main_tag: filtered_anns
core_method: segment tree elemental graphs + on-the-fly edge selection (Algorithm 1, amortized O(m+logn)); bottom-up construction reusing RNG edges from child segments; layer-skipping for efficiency; space O(nmlognl)
key_result: 2x-5x QPS vs SuperPostFiltering @ 0.9 Recall; stable performance across all range lengths (large/moderate/small/mixed); within 2x of Oracle-HNSW; index memory 1.6x-4x raw vectors
limitations: no dynamic update; YT-Audio (128-dim) performance weaker; multi-attribute extension loses perfect recall; no theoretical quality bound for constructed graph
related: 2DSegmentGraph (predecessor/baseline), SuperPostFiltering/β-WST (concurrent work/competitor), ACORN (general predicate), UNIFY (follow-up), HNSW/DiskANN (graph foundations)
```
