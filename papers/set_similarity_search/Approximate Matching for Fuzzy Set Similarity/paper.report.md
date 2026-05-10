---
title: "Approximate Matching for Fuzzy Set Similarity"
paper_id: "GrAPL-2026-APPROXJOIN"
analysis_date: "2026-05-04 14:00:00 +08:00"
main_tag: "set_similarity_search"
tags:
  - "set_similarity_search"
  - "fuzzy_set_join"
  - "approximate_matching"
  - "bipartite_matching"
related_papers: "TokenJoin: Efficient Filtering for Set Similarity Join with Maximum Weighted Bipartite Matching (Zeakis et al., PVLDB 2022); Silkmoth: An Efficient Method for Finding Related Sets with Maximum Matching Constraints (Deng et al., PVLDB 2017)"
---

# Approximate Matching for Fuzzy Set Similarity

> **主标签**：set_similarity_search  
> **论文标签**：set_similarity_search;fuzzy_set_join;approximate_matching;bipartite_matching  
> **发表信息**：GrAPL 2026（Workshop on Graphs in the Programming Languages and Systems，IPDPS 2026 附属工作坊）  
> **作者**：Michael Mandulak, S. M. Ferdous, Sayan Ghosh, Mahantesh Halappanavar, George M. Slota  
> **机构**：Rensselaer Polytechnic Institute (RPI) + Pacific Northwest National Laboratory (PNNL)  
> **代码**：https://github.com/mmandulak1/sasimi

---

## 1. 问题定义

### 背景与挑战

集合相似性计算是数据管理与分析中的基础操作，广泛用于数据清洗（entity resolution）、Web 数据发现、数据湖中的可连接表检索等场景。两类核心任务是集合相似性搜索（set similarity search）与集合相似性连接（set similarity join）：搜索关注给定查询集合的相似近邻，连接则关注所有集合对之间的相似配对。

传统集合相似性通常基于精确 token 重叠（如 Jaccard 相似度），但真实数据普遍含有噪声——同一实体在不同来源中的 token 表示可能存在拼写变体、同义词替换或截断差异。**模糊集合相似性（Fuzzy Set Similarity）**正是为此而生：它不要求 token 完全匹配，而是把两个集合的元素构建为带权二部图（weighted bipartite graph），边权由 token 之间的相似度（如子 token 粒度的 Jaccard 或归一化编辑距离）决定，再通过最大权重匹配（Maximum Weight Matching, MWM）获得集合间的相似分数。

### 形式化定义

给定两个集合族 $\mathcal{R}$ 与 $\mathcal{S}$、集合相似函数 $sim_{\phi}(R, S)$ 和用户定义阈值 $\delta$，**模糊集合相似性连接**的目标是计算所有满足相似性的配对集合：

$$\mathcal{R} \bowtie_{\delta} \mathcal{S} = \{(R, S) \in \mathcal{R} \times \mathcal{S} \mid sim_{\phi}(R, S) \geq \delta\}$$

其中集合 $R \in \mathcal{R}$ 由元素 $r \in R$ 构成，每个元素可以是若干 $q$-gram token 的集合。对于两个集合 $R$ 和 $S$，构建带权二部图 $G(V, E, w)$，$V = R \cup S$，边权 $\phi(r, s) \in [0, 1]$ 为 token 间相似度（主要使用 Jaccard）。在 $G$ 上求**最大权重二部匹配（MWM）** $M_*$，匹配权重之和 $w(M_*) = \sum_{e \in M_*} w(e)$ 即构成集合间的模糊相似分数。

对 $0 < \alpha < 1$，$\alpha$-近似匹配 $M_a$ 满足 $w(M_a) \geq \alpha \cdot w(M_*)$。

### 核心难点

最优 MWM 的时间复杂度为 $O(n^3)$、空间复杂度为 $O(n^2)$（$n = |R| + |S|$）。对于大规模 Web 数据（如包含数千元素/集合的 KOSARAK 数据集），这意味着验证阶段可以占据总执行时间的 94%，运行时间长达数十小时。高计算与内存开销是制约模糊集合相似性连接实际可扩展性的根本瓶颈。

---

## 2. 前人工作的方法缺陷

**以 TOKENJOIN 为代表的最优 MWM 方案**（Zeakis et al., PVLDB 2022）是当前模糊集合相似性连接的 SOTA：它采用 filter-verify 框架，在过滤阶段用 Positional 和 Joint 过滤器减少候选对，验证阶段用 Hungarian 算法（Kuhn-Munkres, 1955）求最优 MWM。Hungarian 方法（及其变体 HGE，包含提前终止条件）提供理论最优结果，但代价是验证阶段的 $O(n^3)$ 时间与 $O(n^2)$ 存储，在密集数据集（如 KOSARAK、ENRON）上成为严重瓶颈。

**Silkmoth**（Deng et al., PVLDB 2017）同样依赖 MWM 约束下的集合检索，也使用 primal-dual 最优匹配，未能突破计算复杂度壁垒。其他方法（Wang et al. TODS 2014, MF-Join ICDE 2019）则基于精确 token 匹配，不处理噪声情形。

核心矛盾在于：**精确的 MWM 计算本身是验证阶段的语义需求（相似分数就等于最优匹配权重），但最优算法的计算代价在大规模场景不可接受**。此前无工作尝试用近似匹配替换验证阶段的最优 MWM，同时维持足够高的 recall。

---

## 3. 研究动机与提出方案

### 研究动机与方法本质

APPROXJOIN 的核心洞察是：验证阶段的最优 MWM 对于最终的集合相似性连接结果而言**存在精度余量可以利用**。模糊集合相似性连接本质上是阈值判断——只要连接结果集（满足 $\delta$ 的配对集合）的 recall 在高概率下接近 1，允许匹配权重有适度低估是合理的工程取舍。作者由此提出：用三种已知 $\alpha$-近似 MWM 算法替换 Hungarian 方法，在保持高 recall（$\geq 0.99$）的同时大幅降低计算代价。

这一方向并非"把近似算法接入现有系统"那么简单。关键设计判断有两点：（1）仅替换验证阶段，候选生成与过滤阶段沿用 TOKENJOIN 的 Positional & Joint 过滤器，不引入额外误差；（2）通过对近似匹配权重的 upper bound 估计（GD/LD 的双倍估计、PS 的对偶值之和），同时维护精确 recall 与 precision 上界，使 APPROXJOIN 可以在连接结果集上提供双向保证，而不仅仅是模糊的"约 99% 准确"。

### 核心假设

完整二部图上的最优 MWM 权重与近似权重之间的差距，在统计意义上足够小，使得大多数集合对的阈值判断不因近似而翻转。ENRON 数据集的 recall 偏低（AJ–GD 为 0.9293）支持这一假设在高密度、高匹配使用率数据集上存在边界条件。

### 三种近似匹配算法的机制

这三种算法均来自近似匹配领域的已有工作，APPROXJOIN 的贡献在于首次将它们集成到模糊集合相似性连接的验证阶段。三者均作用于完整带权二部图 $G(V,E,w)$，$V = R \cup S$，目标是在 $O(n^3)$ 的 Hungarian 代价之外找到满足近似比保证的匹配。

**Greedy (GD)**（来自 Avis, Networks 1983）：$\frac{1}{2}$-近似，确定性算法。

```
初始化空匹配 M = ∅
将所有边 E 按权重 w(e) 降序排列
for each edge e = (r, s) in sorted order:
    if r 未被匹配 AND s 未被匹配:
        M ← M ∪ {e}
return M
```

时间复杂度 $O(m \log m)$（边排序主导），空间 $O(m)$（需存所有边）。近似比保证来自：若边 $(r,s)$ 未被 GD 选入，其至少一端已被权重 $\geq w(r,s)$ 的贪心边占用；对任意最优匹配边 $e^*$，GD 匹配中占用其端点的边权重之和构成 $2w(e^*)$ 的上界，因此整体近似比为 $\frac{1}{2}$。实现极简，在大多数实例（16/25）中执行时间最优。

**精度上界估计**：由于 GD 是 $\frac{1}{2}$-近似，将近似匹配权重 $w(M_a)$ 直接乘以 2 即得最优权重的上界估计，用于 upper bound 模式的连接结果 precision 计算。

---

**Locally Dominant (LD)**（来自 Preis, STACS 1999，Pointer Chasing 方法）：$\frac{1}{2}$-近似，线性时间。

```
for each vertex v in V:
    ptr[v] ← 指向 v 的最重相邻边的另一端顶点（即 v 的"局部首选"）

for each edge e = (r, s):
    if ptr[r] == s AND ptr[s] == r:   // r 与 s 互相指向对方
        M ← M ∪ {e}                   // e 为局部主导边
        标记 r、s 已匹配，删除其所有相邻边

// 对剩余未匹配顶点重复迭代（或递归处理剩余子图）
return M
```

时间复杂度 $O(m)$（线性，无需全局排序）。**核心概念**：当两个顶点互相将对方的连接边视为本地最重边时，该边称为"局部主导边"——任何比它权重更高的替代方案必然使至少一端的局部最优被破坏，因此它是 $\frac{1}{2}$-近似保证的结构基础。实验结果与 GD 高度相近，因此论文精度分析中将 LD 排除（以 GD 和 PS 的对比为主），仅在执行时间实验中保留。

**精度上界估计**：与 GD 相同，将近似权重乘以 2 得到 $w(M_*)$ 的上界，理论依据相同。

---

**Paz–Schwartzman (PS)**（来自 Paz & Schwartzman, SODA 2017）：$\frac{1}{2+\varepsilon}$-近似（$\varepsilon > 0$），semi-streaming 算法，内存 $O(n)$。

```
初始化空栈 S，空匹配 M = ∅，参数 ε > 0

// 单次流式扫描所有边（Pass 1）
for each edge e = (r, s, w) in stream order:
    if S 为空:
        push e onto S
    elif w > (1 + ε) × w(栈顶边):
        push e onto S
    elif e 与当前栈中边无顶点冲突:
        push e onto S

// 后处理栈（Pass 2，反向扫描）
从栈底到栈顶:
    if e 的两端顶点均未被匹配:
        M ← M ∪ {e}

return M
```

时间复杂度 $O(m)$（单次边遍历），空间 $O(n)$（栈深度以顶点数为界，不依赖边数）。这是三种算法中**唯一无需将完整二部图加载入内存**的方法——对于完整二部图，边数 $m = |R| \times |S|$ 在大集合下远超顶点数 $n = |R| + |S|$，这直接决定了 PS 在内存上的优势。验证阶段的完整图存储是 TOKENJOIN 内存峰值的主要来源，PS 的流式特性绕过了这一瓶颈，这是 APPROXJOIN 实现内存节省（平均 23%，KOSARAK 上 45%）的根本机制。

近似比 $\frac{1}{2+\varepsilon}$ 的证明基于 primal-dual 分析：栈式比较策略保证被丢弃边与被保留边之间有 $(1+\varepsilon)$ 倍权重差距，通过对偶变量的约束可证整体匹配权重的下界。$\varepsilon$ 越小，近似比越接近 $\frac{1}{2}$（与 GD/LD 持平），但栈操作开销可能略有增加。

**精度上界估计（最为精确）**：PS 的 primal-dual 结构提供了天然的对偶值上界——对偶变量之和构成 $w(M_*)$ 的严格上界，比 GD/LD 的简单"乘以 2"更紧，这使得 PS 在 upper bound 模式下的 precision 估计（几何均值 0.9980）优于 GD（0.9978）。

---

**三者横向对比**：

| | GD | LD | PS |
|---|---|---|---|
| 近似比 | $\frac{1}{2}$ | $\frac{1}{2}$ | $\frac{1}{2+\varepsilon}$ |
| 时间复杂度 | $O(m \log m)$ | $O(m)$ | $O(m)$ |
| 内存需求 | $O(m)$ | $O(m)$ | **$O(n)$**（流式）|
| 上界估计 | 权重 ×2 | 权重 ×2 | 对偶值之和（更紧）|
| recall（几何均值）| 0.9978（上界模式）| ≈GD | 0.9974（下界）/ 0.9980（上界）|
| 最优场景 | 大多数数据集（16/25）| 与 GD 接近 | 高密度大集合（KOSARAK）|
| 主要计算代价 | 全局边排序 | 指针追逐迭代 | 栈操作 + 两次边扫描 |

### 整体流程

APPROXJOIN 完全沿用 TOKENJOIN 的过滤阶段（TJPJ，Positional + Joint 过滤器，$\delta = 0.7$），仅在验证阶段将 Hungarian 方法替换为 GD/LD/PS。对每个候选对 $(R, S)$，构建带权完整二部图，调用所选近似匹配算法，将匹配权重代入集合相似性公式，判断是否满足阈值。

**精度界估计机制**：  
- 在 lower bound 模式下，直接使用近似权重 $w(M_a)$（低估了 $w(M_*)$），连接结果是最优结果的子集，precision 恒为 1.0。  
- 在 upper bound 模式下，对 GD/LD 将权重乘以 2（利用 $\frac{1}{2}$-近似保证），对 PS 使用 primal-dual 对偶值之和作为 $w(M_*)$ 的上界估计，得到连接结果的超集，recall 恒为 1.0。

**图 1** 直观展示了 fuzzy set similarity join 的匹配工作流：Web 文本数据（如发表标签）被分词为 $q$-gram token，token 间相似度构成二部图边权，MWM 即为最终相似分数的核心计算步骤。

### 相对已有工作的核心改动

与 TOKENJOIN 相比，APPROXJOIN 的唯一变化是验证阶段的 MWM 算法替换。与 Silkmoth 等方法相比，APPROXJOIN 在保留模糊语义（非精确 token 匹配）的前提下引入近似计算，是该问题域内的**首个近似 MWM 方法**（作者明确声明）。

### 机制有效性解释

KOSARAK 上的极端加速（验证阶段 $19.1\times$，总执行时间 $6.7\times$）直接来自该数据集验证阶段占总时间比例极高（94%），而高密度大集合（平均每集合 12 元素，最大 2500 元素）使匹配计算成本集中爆发。AJ–PS 在 ENRON 上 recall 偏低（0.9757），分析者判断原因是 ENRON 的数据密度导致匹配使用率高，近似权重的低估更易跨越相似性阈值，造成更多漏报，与作者报告一致。

**图 2**（性能分布图）显示所有近似方法在每个测试实例上均优于最优方法，直接支撑 APPROXJOIN 的 speedup 主张。**图 5**（内存使用进度图）展示 KOSARAK 验证阶段 AJ–PS 的内存峰值（512MB）显著低于 TJPJ–HG（944MB）与 TJPJ–HGE（591MB），且 AJ–PS 的内存曲线在验证阶段几乎不出现峰值突刺，直接证实了流式图处理减少图存储开销的假设。

---

## 4. 实验对比与性能提升

**数据集**：10 个 Web 文本数据集，规模从 123K（MIND）到 3.1M（LIVEJ），集合大小从均值 3（AOL）到均值 134（ENRON），最大集合大小从 47（YELP）到 2500（KOSARAK）。以 $q=3$ 的 $q$-gram 切分 token。所有实验均为自连接（self join），相似性阈值 $\delta = 0.7$。

**指标**：总执行时间（Total Execution Time）、验证阶段执行时间（Verification Time）、峰值内存使用（通过 valgrind Massif），以及基于 Hungarian ground truth 的 Recall 与 Precision。

**基线**：  
- **TJPJ–HG**（TOKENJOIN + Hungarian，NetworkX 实现）：最优 MWM，$O(n^3)$，无近似。  
- **TJPJ–HGE**（TOKENJOIN + Hungarian + 提前终止 UBLB 变体）：TOKENJOIN 原文的内置优化，快于纯 HG 但仍为精确解。

**核心结果**：

| 方法 | 相对 TJPJ–HG 总时间提升 | 相对 TJPJ–HGE 总时间提升 | 几何均值 Recall |
|------|------------------------|------------------------|----------------|
| AJ–GD | 3.78× (平均) | 2.18× (平均) | ~0.9978（精度上界模式） |
| AJ–LD | 相近于 AJ–GD | 相近于 AJ–GD | 与 GD 类似 |
| AJ–PS | 3.78× (平均) | 2.18× (平均) | 0.9974（默认下界模式） |

验证阶段单独提升最高：KOSARAK 上 AJ–PS 达到 $19.1\times$，FLICKR 上 AJ–GD 达到 $4.8\times$。内存方面，KOSARAK 上 AJ–PS 峰值 512MB，比 TJPJ–HG（944MB）低 45%。

**最佳场景**：KOSARAK——验证阶段占总执行时间比例高（94%），集合密度大，MWM 计算代价集中，总执行时间从 35 小时降至 5 小时（AJ–PS）。

**弱场景**：ENRON——AJ–GD recall 降至 0.9293（Δ#Sets = -4515），低于作者设定的 0.99 基准线，分析者判断原因是 ENRON 的高密度集合使近似权重低估更频繁翻转阈值判断，数据集特性决定 PS 方法在此场景下显著优于 GD（recall 0.9757 vs 0.9293）。

**消融分析**：论文本质上以三种近似算法变体作为消融对比：GD 与 LD 均为 $\frac{1}{2}$-近似，实验结果相近，GD 在多数实例（16/25）中最优；PS 在 recall 上系统性高于 GD/LD，且内存优势最显著，但两者的总时间差距较小（约 14% 平均差异）。这说明近似质量（$\frac{1}{2+\varepsilon}$ vs $\frac{1}{2}$）对精度有实质影响，对执行时间的影响则更多取决于数据集特性。

---

## 5. 方法局限与缺陷

**算法近似比的理论保障与实践差距**：GD 与 LD 的 $\frac{1}{2}$-近似保证在最坏情况下是紧的（tight），意味着理论上最优匹配权重的 50% 足以满足近似保证。ENRON 数据集已经显现出 recall 低于 0.93 的情况，表明在高密度、高匹配依赖的数据上，近似比的保守性会实质性地影响连接结果质量。**外部有效性威胁（分析者判断）**：所有数据集均为 Web 文本数据，且均为自连接实验；在实体分辨等非对称场景（$\mathcal{R} \neq \mathcal{S}$）或领域专有数据（如生物序列、代码库）中，近似匹配对 recall 的影响尚未验证。

**与过滤阶段的解耦假设**：APPROXJOIN 直接复用 TOKENJOIN 的过滤器，未分析过滤阶段的候选生成方式是否与近似验证兼容——例如，若过滤器依赖 MWM 的某些精确性质来判定候选对（TOKENJOIN 的 Positional/Joint 过滤器实际上不依赖验证结果，所以这里风险较低，但作者未明确讨论）。

**单线程基准测试**：所有实验在单线程下进行，无并行版本对比。作为 GrAPL（图算法并行化工作坊）论文，缺乏多线程或多核并行实现，弱化了对并行图计算社区的贡献感。

**实现依赖 Python + NetworkX**：TJPJ–HG 使用 NetworkX（Python 图库）进行 Hungarian 匹配，这本身引入了额外的解释器开销和图表示内存开销。与用 C/C++ 实现的 TOKENJOIN 原始代码相比，基准测试的公平性存在实现层面的不一致（作者使用 TOKENJOIN 的 Python 开源代码，因此同一套 Python 实现下的对比是合理的，但相对纯 C 实现的"SOTA"比较需谨慎解读）。

**研究体量属工作坊论文范畴**：本文为 GrAPL 2026 工作坊短文，缺乏更细粒度的理论分析（如不同数据分布下 recall 下降的概率界），且不包含对自身上界估计机制的精度分析（如 precision 的可靠性边界）。论文本身已承认这是初步探索，更完整的理论与实验验证见 arXiv 长文（arXiv:2507.18891）。

---

## 6. 科研启发

1. **近似匹配质量对集合连接结果的统计影响建模**：当前工作仅凭实验展示高 recall，未给出"给定 $\alpha$-近似匹配，连接结果 recall 下降的概率界"的理论分析。一个有价值的研究方向是：**形式化近似匹配误差在集合相似性阈值判断中的传播模型**，利用集合元素数量分布、边权分布等统计特性，推导 recall 的分布上/下界。这为工程部署时的近似算法选型提供理论基础（例如：在什么数据分布条件下 GD 的 $\frac{1}{2}$-近似已足够，何时必须使用 PS 的更高近似比），可能需要建立连接到随机图上 MWM 近似理论的新假设框架。

2. **自适应近似策略：按候选对动态选择匹配精度**：当前 APPROXJOIN 对所有候选对使用统一的近似算法。直观上，当近似匹配权重远离阈值 $\delta$ 时，精确程度对最终判断影响小；当权重恰好在 $\delta$ 附近时，近似误差可能翻转判断。一个新问题定义是：**设计一个两阶段候选验证协议**——以低精度近似匹配做首次估计，将"阈值附近"的不确定对（定义为 $|w(M_a) - \delta| < \varepsilon'$）路由到高精度算法，其他确定对直接输出。核心假设是此类不确定对在真实数据中占比较小，整体开销可以控制在小常数倍以内，且 $\varepsilon'$ 可以被自动估计。

3. **模糊集合相似性在多向量检索中的应用**：ColBERT 等 late interaction 模型的 MaxSim 评分本质上是 document token 集合与 query token 集合之间的最大相似性聚合，是模糊集合相似性的一种特例（非对称、每个 query token 仅取最相似 document token）。DESSERT（arXiv:2304.01598）已将向量集相似性问题形式化为 MaxSim 搜索。一个值得探索的方向是：**将 APPROXJOIN 的近似匹配思想移植到多向量检索的验证阶段**，替换全精度 MaxSim 计算，以牺牲微小精度换取验证开销降低，尤其对长文档（token 数量大）或文档预期差异大时的候选集合验证有潜在价值。理论假设是：检索相关性分数分布的稀疏性使得阈值附近的不确定对比例低，近似匹配引入的排序扰动对最终检索 recall 影响可控。

---

## 7. 参考文献图谱

### 文献分类表

| 文献名 | 作者/年份 | 角色 | 知识库状态 |
|--------|----------|------|-----------|
| TokenJoin: Efficient Filtering for Set Similarity Join with Maximum Weighted Bipartite Matching | Zeakis et al., PVLDB 2022 | 实验基线 / 被批判文献 | 未收录 |
| Silkmoth: An Efficient Method for Finding Related Sets with Maximum Matching Constraints | Deng et al., PVLDB 2017 | 被批判文献 | 未收录 |
| The Hungarian Method for the Assignment Problem | H. W. Kuhn, Naval Res. Logist. Q. 1955 | 方法基础（最优 MWM） | 未收录 |
| A Survey of Heuristics for the Weighted Matching Problem | Avis, Networks 1983 | 方法基础（GD 算法来源） | 未收录 |
| Linear Time 1/2-Approximation Algorithm for Maximum Weighted Matching in General Graphs | Preis, STACS 1999 | 方法基础（LD 算法来源） | 未收录 |
| A (2+ε)-Approximation for Maximum Weight Matching in the Semi-Streaming Model | Paz & Schwartzman, SODA 2017 | 方法基础（PS 算法来源） | 未收录 |
| Streaming Matching and Edge Cover in Practice | Ferdous et al., SEA 2024 | 方法基础（PS 实践参考） | 未收录 |
| An Empirical Evaluation of Set Similarity Join Techniques | Mann et al., PVLDB 2016 | 背景综述 | 未收录 |
| JOSIE: Overlap Set Similarity Search for Finding Joinable Tables in Data Lakes | Zhu et al., SIGMOD 2019 | 背景综述 | 未收录 |
| A Primitive Operator for Similarity Joins in Data Cleaning | Chaudhuri et al., ICDE 2006 | 背景综述 | 未收录 |
| Extending String Similarity Join to Tolerant Fuzzy Token Matching | Wang et al., TODS 2014 | 背景综述 | 未收录 |
| MF-Join: Efficient Fuzzy String Similarity Join with Multi-Level Filtering | Wang et al., ICDE 2019 | 被批判文献 | 未收录 |
| DESSERT: An Efficient Algorithm for Vector Set Search with Vector Set Queries | Davidson et al., NeurIPS 2023 | 相关工作（多向量集合检索）| 已收录（multi_vector_retrieval）|

> 角色说明：**实验基线** = 实验对比方法来源；**被批判文献** = 论文指出其不足作为动机依据；**方法基础** = 本文方案的技术依赖；**背景综述** = 划定问题域的综述引用

---

## 推荐阅读列表

> 基于文献图谱，按优先级排序。

### P0 必读（方法基础，知识库未收录）
- **TokenJoin: Efficient Filtering for Set Similarity Join with Maximum Weighted Bipartite Matching**（Zeakis, Skoutas, Sacharidis, Papapetrou, Koubarakis, PVLDB 2022）——APPROXJOIN 直接基于此框架扩展，理解本文方法的关键前置。
- **A (2+ε)-Approximation for Maximum Weight Matching in the Semi-Streaming Model**（Paz & Schwartzman, SODA 2017）——AJ–PS 方法的理论来源，semi-streaming 算法的原始证明。

### P1 重要（被批判文献，理解动机必读）
- **Silkmoth: An Efficient Method for Finding Related Sets with Maximum Matching Constraints**（Deng, Kim, Madden, Stonebraker, PVLDB 2017）——与 TOKENJOIN 并列的 SOTA 前驱，帮助理解模糊集合连接的已有瓶颈。
- **An Empirical Evaluation of Set Similarity Join Techniques**（Mann, Augsten, Bouros, PVLDB 2016）——评测集合相似性连接方法的综合性实证研究，建立问题领域基准认知。

### P2 建议（近似匹配理论基础）
- **Linear Time 1/2-Approximation Algorithm for Maximum Weighted Matching in General Graphs**（Preis, STACS 1999）——AJ–LD 方法的理论来源，最早提出 pointer chasing 近似匹配。
- **Streaming Matching and Edge Cover in Practice**（Ferdous, Pothen, Halappanavar, SEA 2024）——本文作者群体前期对 streaming 匹配实践性能的研究，直接支撑 AJ–PS 的实现细节。

### P3 参考（扩展视角）
- **DESSERT: An Efficient Algorithm for Vector Set Search with Vector Set Queries**（Davidson et al., NeurIPS 2023）——将向量集相似性搜索形式化，与本文的集合相似性连接问题在结构上具有内在联系，是理解"近似匹配思想迁移至多向量检索"的关键桥梁。

---

## mem0 知识库记录

- **问题域**：fuzzy set similarity join, set similarity search, bipartite matching, approximate algorithms
- **方法关键词**：approximate MWM, APPROXJOIN, greedy matching, locally dominant matching, Paz-Schwartzman semi-streaming, filter-verify framework, TokenJoin
- **数据集**：LIVEJ (3.1M), AOL (1.8M), KOSARAK (610K), ENRON (518K), DBLP (500K), FLICKR (500K), GDELT (500K), BMS-POS (320K), YELP (160K), MIND (123K)；均为 Web 文本自连接，$\delta = 0.7$
- **性能基准**：2-19× 总执行时间提升，验证阶段最高 19.1× (KOSARAK/AJ–PS)，平均 recall 0.9974 (AJ–PS)，内存降低 23% (平均)
- **关联论文 ID**：TokenJoin (Zeakis et al. PVLDB 2022), DESSERT (Davidson et al. NeurIPS 2023, 本库已收录), Paz-Schwartzman (SODA 2017)
- **核心方法机制摘要**：在 filter-verify 框架的验证阶段，以 GD/LD/PS 三种近似 MWM 替换 Hungarian 算法；PS 的 semi-streaming 特性额外降低图存储内存；upper/lower bound 估计提供双向精度保证
- **推荐下一轮阅读线索**：TokenJoin 原文（PVLDB 2022）、Silkmoth（PVLDB 2017）；以及 DESSERT（已收录）中的向量集搜索与本文集合相似性的联系

---

## 跨论文联想补充

本文的 APPROXJOIN 与仓库中已收录的 **DESSERT**（multi_vector_retrieval 领域）在问题结构上存在深层关联：DESSERT 处理的是"向量集到向量集的 MaxSim 检索"，即用近似算法（LSH + sketches）加速集合间最大权重匹配的候选检索阶段；APPROXJOIN 则在更传统的 string/token 集合相似性连接的验证阶段引入近似 MWM。两者共同指向一个研究方向：**集合相似性计算中 MWM/MaxSim 精度与效率的权衡空间**，但一个在候选检索层面操作，另一个在验证层面操作。这一对比有助于在 set_similarity_search 与 multi_vector_retrieval 两个领域之间建立方法桥梁。
