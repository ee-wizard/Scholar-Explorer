---
title: "Multi-Field Tool Retrieval"
paper_id: "arXiv:2602.05366"
analysis_date: "2026-05-03 12:00:00 +08:00"
main_tag: "agentic_ai"
tags:
  - "agentic_ai;tool_retrieval;multi_field;agent_tools"
related_papers: "DRAGIN (2024), PLUTo (2024), EasyTool (2024), OnlineRAG (2024)"
---

# Multi-Field Tool Retrieval

> 学术分析报告

---

## 1. 问题定义

### 背景与核心挑战

大语言模型与外部工具的集成是扩展 LLM 能力的主要方向，使其能够与真实环境交互、适应动态演变的任务生态。然而，随着工具库规模的扩大（数千至数百万个工具），如何从庞大的工具仓库中高效检索适配用户需求的工具成为了关键瓶颈。

论文识别了工具检索的三个根本性挑战，这些挑战将其与传统文档检索区分开来：

**第一，工具文档的不完整性与结构不一致。** 工具文档来源多元且无统一规范，存在显著的格式差异、粒度不同、术语不统一等问题。例如 Gorilla 数据集中仅提供参数名称而缺少详细描述；不同数据集中对概念（如"parameters"vs"api_arguments"）的命名差异导致工具无法统一语义空间。某些文档过于简略，某些包含过多技术细节和实现噪声。

**第二，用户查询与技术文档间的语义与粒度错配。** 用户查询通常是高层次的、模糊的、多意图的任务表述（如"分析销售趋势并生成汇总报告"），涉及多个工具的协作使用。而工具文档采用技术性、原子化的操作规范，为单一功能设计。这种根本的表述模式不匹配导致语义匹配算法不适配。

**第三，工具效用的多方面本质（核心创新点）。** 工具的有用性不是简单的语义相似度度量，而是受多个独立维度制约的复合概念：
- **功能性**（description 字段）：工具是否实现了用户所需的功能
- **输入可行性**（parameters 字段）：用户是否能提供所需的输入参数，特别是必需参数的缺失会导致工具不可执行
- **输出适配性**（response 字段）：工具输出是否满足下游任务的需求
- **使用示例契合度**（examples 字段）：工具的典型应用场景与用户意图的对齐程度

论文通过实证展示（Figure 1），对不同字段进行掩蔽时，性能变化幅度在不同数据集间差异显著（在 Gorilla、APIBank、APIGen 上分别高达 ±10% 以上），说明仅用单一检索模型处理整体文档是根本性缺陷，而非工程优化问题。

### 问题形式化

给定工具仓库 $\mathcal{T} = \{(t_i, d_i)\}_{i=1}^N$，其中 $t_i$ 为工具，$d_i$ 为其文档，用户查询 $q$ 对应的检索目标是找到功能适配的工具子集 $\mathcal{T}_q \subseteq \mathcal{T}$。

传统范式将检索目标形式化为单一相似度计算：
$$S(q, t_i) = \phi(q, d_i)$$

必要原文引用（论文 §3.1, Eq.(1)）：
> "Conventional tool retrieval methods typically treat $d_i$ as a single unstructured text and compute a semantic similarity score between $q$ and $d_i$: $S(q, t_i) = \phi(q, d_i)$, where $\phi(\cdot, \cdot)$ denotes a relevance scoring function."

引用说明：该单一相似度公式是传统工具检索范式的核心形式化（论文原式 Eq.(1)），与 MFTR 的多字段对齐方案形成直接对比，是本文改进动机的出发点。

MFTR 改进为多维度对齐：
- 标准化文档 $d_i \rightarrow d_i' = \{d_i^{\text{desc}}, d_i^{\text{param}}, d_i^{\text{resp}}, d_i^{\text{ex}}\}$
- 重写查询 $q \rightarrow q' = \{a_1, a_2, \ldots, a_k\}$（工具需求分解）
- 计算多字段相似度 $S_f(q, t_i)$，并通过参数缺失惩罚与自适应权重融合

约束条件：工具必须满足所有必需参数的可执行性，不仅需要语义匹配，更需要功能正确性。

### 问题重要性

工具检索是 LLM Agent 架构中的关键模块。检索质量直接影响：（1）Agent 决策质量——错误的工具选择导致任务失败；（2）系统效率——在有限的上下文窗口内，高精度检索减少冗余工具加载；（3）可部署性——在百万级工具库中进行实时、准确的检索需要具有强泛化性的方案。

---

## 2. 前人工作的方法缺陷

### 传统 ad-hoc 检索范式的不适配

现有方法通常将工具检索视为传统信息检索任务，采用 BM25（稀疏检索）或密集嵌入模型（如 Contriever、DPR）对整体工具文档进行语义匹配。

**缺陷**：
- 无法处理文档不完整性，简单的 LLM 增强（如 EasyTool 的文档补全）虽然增加了信息量，但引入的语义噪声导致性能不稳定甚至下降（Table 3 表明 Stand. Only 或 Rewrite Only 单独应用时性能波动甚至衰退）
- 对多意图查询的处理不足，无法将用户高层次的复合任务分解为原子级工具需求
- 忽视了工具的功能可执行性约束，即使文档在语义上匹配，若缺少必需参数，工具也无法执行

### 工具文档的结构化尝试的局限

PLUTo 和 OnlineRAG 等最新工作提出了多步骤工作流与动态反馈机制，但：
- PLUTo 的 LLM 选择模块增加了多次 LLM 调用（平均 4.72 次/查询），明显高于 MFTR（1.02 次）
- OnlineRAG 的在线学习机制在小样本数据集上失效（如 APIBank 上性能显著下降），泛化性不足
- 这些方法仍然在 ad-hoc 范式内优化，未从根本上重新定义工具检索问题

### 函数级工具的检索特殊性被忽视

与网页检索、学术文献检索不同，工具检索面临独特约束：
1. **功能可执行性是硬约束**：文献检索中相关性是梯度的（partially relevant documents 仍有价值），而工具检索中，功能需求不匹配的工具完全无用
2. **参数约束的二元性**：不区分必需参数与可选参数的处理方式无法准确评估工具的真实可用性
3. **文档质量的剧烈差异**：工具文档来自异构源（REST API、Python 函数、LLM 生成代码），质量与完整度范围极大；不像学术文献有相对稳定的写作风格

---

## 3. 研究动机与提出方案

### 核心观察

MFTR 的核心观察是：**工具检索中，用户意图对工具的需求不是均匀分布在"功能相关性"这一个维度上，而是分布在多个独立的功能维度上，每个维度的权重因任务、数据集、检索器而异**。

论文通过 Figure 1 的字段掩蔽实验直观展示了这一点。对 Gorilla 数据集掩蔽不同字段时，NDCG 变化范围 -5% 到 +10%，说明字段的贡献不均等且存在负面贡献（某些字段本身是噪声）。同时不同数据集对字段的依赖差异大（APIGen 掩蔽某字段有收益，而 Gorilla 上同样掩蔽却有损失），体现了泛化性挑战。

### 方法本质

MFTR 将工具检索重新定义为**多字段对齐问题**，而非单一的文本相似度问题。核心创新包括三个阶段：

**第一阶段：工具文档标准化（Tool Documentation Standardization）**

通过 LLM 将异构文档转换为统一的四字段 schema：
- **description**：工具功能的高层概括（对应工具目的）
- **parameters**：输入参数的形式化规范（参数名、类型、语义、必需性）
- **response**：输出内容与行为的自然语言描述
- **examples**：工具适用的具体使用场景

设计原则：
- **功能覆盖**：四字段应捕获理解和使用工具所需的全部关键方面
- **泛化与鲁棒性**：字段数量应最小化，以避免 LLM 在信息缺失时的幻觉

与其他 LLM 文档增强方法（EasyTool）的差异在于，MFTR 的标准化是结构化的、有约束的，建立了与查询重写的显式对齐契约。

**第二阶段：查询重写与对齐（Query Rewriting and Alignment）**

采用工具感知的伪相关反馈（Pseudo-Relevance Feedback）方法：

1. 查询分解：将单一用户查询 $q$ 分解为多个工具需求 $\{a_1, a_2, \ldots, a_k\}$，每个需求对应三个子字段：
   - 用户意图（对应 examples 字段）
   - 工具描述（对应 description 字段）
   - 期望响应（对应 response 字段）

2. 工具感知重写：在重写前，用 BM25 检索仓库中与原查询最相关的 20 个工具描述作为伪正样本，连同这些工具的完整文档一起输入 LLM，指导其生成对工具仓库语义空间的一致认知。这一设计避免了盲目重写导致的与实际工具术语不一致问题。

该阶段的直觉：将查询从高层次的自然语言转换为与标准化文档对齐的结构化表示，使后续的多字段匹配能够独立且精准地进行。

**第三阶段：自适应多字段权重融合（Adaptive Weighting）**

（1）**语义字段匹配**（description、response、examples）：对每个字段独立计算相似度。当查询被分解为 $k$ 个工具需求时，工具与整体查询的相似度定义为与任意工具需求匹配的最大值：
$$S_f(q, t) = \max_{i \in \{1, \ldots, k\}} \phi(q'_f^i, d'_f)$$

必要原文引用（论文 §3.4, 多字段相似度公式）：
> "When the query is decomposed into $k$ tool requirements, the similarity between a tool and the whole query for field $f$ is defined as: $S_f(q, t) = \max_{i \in \{1, \ldots, k\}} \phi(q'^i_f, d'_f)$"

引用说明：对每个字段取跨多工具需求的最大相似度，实现"最佳子需求匹配优先"策略——只要用户的任一子需求与该字段对齐，即视为字段相关。

（2）**参数对齐与缺失惩罚**（parameters 字段核心）：
- 基础匹配：对工具的每个参数 $p_j$ 找到查询中最匹配的参数 $a$，计算平均相似度 $S_{\text{param}}(q, t) = \frac{1}{|\mathcal{P}_t|}\sum_j \max_a \phi(p_j, a)$
- **自适应缺失惩罚**（Eq. 5-6）：使用 sigmoid 函数和可学习的阈值 $\tau$，对缺失参数应用惩罚。关键创新是区分必需参数与可选参数，对必需参数缺失的惩罚 $w_{\text{req}}$ 显著高于可选参数缺失的惩罚 $w_{\text{opt}}$。这是论文对传统检索的重要突破——将功能可执行性从后处理约束变为排序目标中的硬约束。

（3）**最终融合与学习**（Eq. 7-8）：
$$S(q, t) = \left(\sum_f w_f \cdot S_f(q, t)\right) + b - P(q, t)$$

其中 $w_f$ 是可学习的字段权重，$b$ 是偏置，$P(q, t)$ 是参数惩罚。采用 pairwise ranking loss 进行端到端训练，自动学习最优的字段权重组合。

### 关键假设与创新点

**假设 1（多字段独立性）**：工具的四个功能维度可以独立建模，通过加权融合能够捕获比全文档方法更丰富的信号。消融实验（Table 3）通过对比单字段、两字段组合、完整方案，验证了这一假设。

**假设 2（工具感知查询重写的必要性）**：查询重写若不考虑工具仓库的实际术语与概念用法，会产生与现实工具库不对齐的重写结果。通过伪相关反馈将仓库知识注入重写过程，确保生成的查询在语义空间中"说工具的语言"。

**核心创新**：
- 第一个明确指出工具检索中参数缺失的功能约束性质，并将其编码为可学习的、区分必需性的惩罚机制
- 第一个系统化的工具文档标准化 schema，通过结构化字段设计而非简单补全
- 自适应字段权重学习，能够动态捕获不同任务/数据集间字段重要性的差异

---

## 4. 实验对比与性能提升

### 数据集与基准

评测在五个代表性工具检索数据集上进行：
- **ToolBench**（10,992 查询，14,059 REST API）：LLM 合成指令与真实 API
- **APIGen**（1,000 查询，3,605 API/Python 函数）：高质量清洗的 API 文档
- **APIBank**（101 查询，101 Python 函数）：小规模手工编制，最小数据集
- **Gorilla/HuggingFace**（500 查询，907 HF 模型 API）：文档质量参差，参数信息不完整
- **Toolink**（497 查询，1,804 LLM 生成函数）：评测 LLM 生成代码的可检索性

此外构建**Mixed 混合基准**（3,197 查询，20,476 工具）模拟真实异构部署场景。

评测指标：NDCG@K 和 Recall@K。

### 基线方法

- **Full-Doc**：原始文档的直接语义检索
- **EasyTool**：LLM 补全与增强文档的检索
- **PLUTo**：查询分解 + 迭代工具描述优化 + LLM 选择
- **OnlineRAG**：反馈驱动的在线嵌入更新

### 实验设置

- 使用 9 种检索器：BM25 + 8 种密集模型（MiniLM-L6、Contriever、E5-Base/Large、GTE、BGE、COLT、API Retriever）
- 5 折交叉验证，负样本为 BM25 检索的前 64 个错误工具
- 5 epoch 训练，Adam 优化器学习率 0.1，batch size 256

### 主要结果

**整体性能（Table 2）**：
- MFTR 在五个数据集与混合基准上均显著超越所有基线，多数情况下改进具有统计显著性（Fisher Randomization Test, $p < 0.05$）
- 相对强基线 EasyTool 的相对改进幅度：ToolBench 上 NDCG@10 +6.2%（BM25, 0.5573 vs 0.5245），APIGen 上 +6.3%（BM25），Gorilla 上 +37.8%（MiniLM-L6，0.3775 vs 0.2336）
- 在 Mixed 混合基准上相对次优方案提升 28.6%（NDCG@10）和 18.98%（Recall@10），体现强泛化性

**跨检索器鲁棒性**：
- MFTR 对检索器选择不敏感，MiniLM-L6（小模型）配 MFTR 可匹敌或超过 10× 参数量的基线
- 相比 OnlineRAG 在特定数据集上的大幅衰退（如 APIBank 上 Recall@10 下降 47%），MFTR 表现稳定

**数据集适配性**：
- Gorilla 数据集（文档质量最差）上改进最显著：MiniLM-L6 从 0.3960 (runner-up) 提升至 0.5860 (+47.98%)，说明方法对不完整文档的容错能力
- APIGen（高质量数据集）上也保持 11.42% 相对提升，证明即使在高质量场景下多字段建模也能捕获额外信号

---

### 性能提升来源分析

### 标准化与查询重写的互补作用

消融实验（Table 3）揭示了关键发现：

**单独应用 LLM 文档增强无效**。Stand. Only 或 Rewrite Only 相比 Full-Doc 性能不稳定甚至恶化（如 APIGen Stand. Only 从 0.7126 降至 0.6484），说明单纯的 LLM 文本增强引入的噪声超过了信息增益。这与直觉相反但合理——LLM 补全缺失信息时可能产生幻觉，这些幻觉文本与真实查询的匹配反而是噪声。

**标准化是多字段匹配的前提**。Weighting + Stand. 变体（应用自适应权重到标准化字段，保留原查询）稳定优于 Full-Doc，而 Weighting + Rewrite 变体（应用权重到重写查询与原文档）失效，说明**对齐契约**的重要性——查询与文档必须都标准化到同一 schema，否则多字段策略失效。

**自适应权重的关键作用**。"w/o Adaptive Weighting"变体采用字段均值聚合，性能相比 MFTR 下降 1-3%。更重要的是，learned field weights（Table 5）展示了显著的跨数据集差异：
- APIGen 上 description 权重 26.9%，各字段相对均衡
- APIBank 上 description 权重 26.4%，但 response 权重仅 24.3%
- Toolink 上 response 权重激增至 53%，说明该数据集的用户意图更多由输出格式约束，而非输入参数

这种权重的动态学习能力是 MFTR 超越固定策略的关键。

### 参数缺失惩罚的功能正确性保证

**w/o Missing Penalty** 变体表明，去除参数惩罚导致性能衰退（Toolink 上从 0.5796 降至 0.5714），且在 APIBank 这类参数约束关键的小数据集上影响更显著。关键洞察是：

传统检索中相关性是连续梯度（部分相关文献仍有价值），工具检索中参数缺失是**离散硬约束**（无法提供必需参数的工具完全无用）。参数惩罚通过将功能可执行性从后处理约束变为排序目标中的显式信号，强制模型学习到这一二元约束。

案例分析（Table 4）中，Full-Doc 将目标工具（speech denoising）排在第 24 位，主要因为原文档包含过多技术细节（"SepFormer model"、"WHAM! dataset"、"16kHz"），这些细节虽然增加了文本相似度噪声，但也是 MFTR 通过标准化与字段隔离能够压制的噪声。同时，参数字段中的 "audio_file_path" 与查询中对应的参数明确匹配，参数权重（13.3%）足以支持功能正确性约束。

### 鲁棒性与泛化性的来源

**跨数据集的权重一致性**。虽然 learned weights 跨数据集存在差异（Table 5），但 MFTR 在 Mixed 混合基准上反而表现最优（相对提升 28.6%），而 EasyTool 与 OnlineRAG 在混合设置下接近于 Full-Doc。这说明多字段框架的结构本身具有较强的跨源泛化能力，即使最优权重在个别数据集上波动，整体设计能够容纳这种波动。

**小样本数据集的有效性**。APIBank 仅 101 条查询，OnlineRAG 的在线学习在此失效（Recall@10 从 runner-up 的 0.6972 下降至 0.3094，-55.7%），而 MFTR 保持稳定（0.7459）。原因是 OnlineRAG 的反馈机制需要大量训练数据来稳定学习，而 MFTR 的多字段结构本身是强归纳偏好，能在小数据下有效泛化。

---

## 5. 方法局限与缺陷

### 四字段 Schema 的完备性问题

选择四个字段（description、parameters、response、examples）的理由是**功能覆盖与泛化平衡**，但这种选择可能在特定应用中不够：
- 对于需要认证信息的 API（如 OAuth token、API key），原 schema 中无 credentials/authentication 字段，文档标准化时此类信息可能被 LLM 忽略或混入 parameters
- 对于有复杂 side effects 的工具（如修改外部系统状态的操作），缺乏显式字段表示其风险性，可能导致检索到功能正确但不安全的工具

论文未讨论 schema 的可扩展性或对其他字段的探索。

### LLM 依赖的两次关键步骤中的不确定性

（1）**标准化的 LLM 幻觉风险**。使用 gpt-4o-mini 对所有工具文档进行标准化时，对于信息不完整的原文档，LLM 被指示将缺失参数全部标记为"必需"。这一设计选择可能过度约束：某些可选但未明确说明的参数会被标记为必需，导致不必要的参数缺失惩罚。

（2）**查询重写的工具感知偏差**。伪相关反馈（PRF）方法虽然有效，但存在循环依赖风险：BM25 检索的前 20 个工具不一定能代表整个工具库的真实语言空间。若 BM25 的初始检索质量本身不佳（某些小众工具库可能如此），PRF 提供的伪正样本也将是偏颇的，可能将查询重写推向错误的语义空间。论文未提供 PRF 样本大小的敏感性分析。

### 参数匹配的语义空间问题

参数对齐采用的相似度计算（Eq. 3）仍然是语义匹配，对参数名称的同义词识别依赖检索模型的能力：
- 语义相似的参数名称可能被标记为"缺失"。例如"input_file" vs "source_file" vs "filepath"，如果检索模型的相似度空间中这些名称的距离较大，即使语义本质相同，也会导致参数匹配失败
- 论文未展示参数匹配的 precision/recall 分析，即实际有多少参数被误认为缺失或虚假匹配

### 自适应权重学习的数据效率与外部有效性

（1）**权重学习对训练数据大小的敏感性未知**。Table 3 消融实验表明 w/o Adaptive Weighting 时性能下降 1-3%，但这一差异的统计显著性与对新数据集泛化性未被评估。特别是在 APIBank（101 查询）这类极小数据集上，learned weights 是否真的优于固定加权策略不确定。

（2）**超参数选择的人工干预**。sigmoid 曲线形状参数 $\alpha=15$ 是固定值，未进行敏感性分析。不同数据集、不同检索器下 $\alpha$ 的最优值可能差异大，但论文未讨论。

### 查询分解的多意图假设的限制

查询分解假设一个用户查询可分解为独立的、无序的多个工具需求。但实际场景中：
- 工具需求间可能存在**顺序依赖**（第一个工具的输出是第二个工具的输入），单纯的并行多字段匹配无法捕获这种依赖
- 某些查询是**单工具任务**，但分解时仍被拆分为多个需求，可能引入歧义

论文未量化有多少查询被不当分解，也未讨论分解错误对检索性能的影响。

### 计算效率的隐含成本

论文指出 MFTR 的平均延迟为 1.02s（LLM 查询重写）+ 0.07s（多字段聚合），而 PLUTo 为 15.35s。但这一比较在两个方面存在潜在偏差：

（1）**离线成本未计入**。标准化所有工具文档需要的 LLM 调用总数未报告。若工具库持续更新（新工具不断加入），持续的标准化成本可能显著。

（2）**检索数量的隐含成本**。MFTR 对四个字段独立进行检索并聚合，实际的检索调用数是 Full-Doc 的 ~4 倍（虽然每个字段的检索量可能较小）。对于超大规模工具库（百万级），这一成本可能不可忽视。

### 外部有效性的边界

（1）**工具类型的代表性**。五个数据集中工具主要是 REST API 与 Python 函数。其他类型工具（SQL 查询、GraphQL API、CLI 命令、自定义 DSL）的检索需求可能显著不同。例如 CLI 命令的参数通常是位置性的与标志性并存，与 REST API 的键值参数结构完全不同。

（2）**查询类型的代表性**。数据集中的查询是 LLM 合成或手工编制的，未包含真实生产环境中的 Agent 系统查询（可能更加多样化、包含更多澄清与重试）。

（3）**LLM 评估工具的性能**。实验只使用 gpt-4o-mini 进行标准化与重写。不同 LLM 的效果差异未知，特别是开源 LLM（如 Llama、Mistral）的表现不确定。

---

## 6. 科研启发

1. **功能约束与语义相关性的联合建模范式**。MFTR 通过参数缺失惩罚将功能正确性编码为排序目标，这一思路可迁移到其他需要硬约束的检索场景。例如，在知识库检索中可以编码数据类型兼容性约束；在代码搜索中可以编码 API 兼容性约束。未来工作可以形式化这一范式，为不同约束类型设计通用的编码机制。

2. **Schema-driven 的文档标准化与检索**。传统检索优化关注检索模型或排序函数，MFTR 通过显式的 schema 设计改变了检索范式。这启发了一个研究方向：**数据结构化对检索泛化性的影响**。具体可探索：（1）最优 schema 设计的原则（现有选择是启发式的）；（2）Schema 对不同检索器的适配性差异；（3）多个竞争 schema 的自动选择机制。

3. **查询-文档对齐的可学习指导**。工具感知的伪相关反馈虽然有效，但本质上是一次性的初始化。新研究方向可以设计**迭代的双向对齐**：查询重写时考虑已检索的工具反馈，同时优化文档标准化使其更贴近查询分布。这涉及查询-文档对齐的联合学习框架。

4. **字段权重学习的跨任务转移**。Table 5 展示了不同数据集的 learned weights 差异显著，但缺乏转移学习分析。新方向可以研究：给定一个新工具库（未见过），如何从已有数据集的权重分布预测合理的初始权重？这可能涉及元学习（meta-learning）或权重分布的聚类。

5. **多阶段检索的决策优化**。当前 MFTR 是单阶段的多字段融合，但实际 Agent 系统中检索往往是多阶段的（初步筛选 → 精排 → 最终选择）。未来工作可以研究**多阶段工具检索的联合优化**：如何在初期阶段用轻量级多字段策略快速筛选，在精排阶段用更复杂的约束与反馈？

6. **工具间的依赖与组合性建模**。现有工作假设工具相互独立，但实际工具库中工具间存在调用依赖与组合关系。新方向可以构建**工具依赖图**，在检索时考虑：用户需求能否通过单个工具满足，若需要多个工具协作，这些工具间的兼容性如何保证？这涉及工具检索与工具链规划的联合问题。

7. **检索后的工具可执行性验证**。MFTR 的参数缺失惩罚是基于查询中显式提及的参数，但实际执行时可能出现：检索到的工具在语法上要求的参数用户可以提供，但实际执行会失败（如权限不足、外部服务不可用）。新方向可以研究**轻量级的可执行性预测**，在检索排序后添加预测层，预估检索到的工具在真实环境中的实际可执行率。

---

## 7. 参考文献图谱

| 文献名 | 作者/年份 | 角色 | 知识库状态 |
|--------|----------|------|-----------|
| ToolBench: A Large-Scale Python Function Calling Benchmark for LLMs | Qin et al., 2024 | 主要评测基准 | 已涵盖 |
| APIBank: A Benchmark for Tool-Augmented LLMs | Li et al., 2023 | 评测基准 | 已涵盖 |
| Gorilla: Large Language Model Connected with Massive APIs | Patil et al., 2023 | 评测基准 | 已涵盖 |
| EasyTool: Enhancing Tool Use in Language Models with Ease | Wang et al., 2024 | 主要竞争基线 | 已涵盖 |
| PLUTo: A Plug-and-Learn Unprompted Tool-Use Framework | Sun et al., 2024 | 竞争基线 | 未收录 |
| OnlineRAG: Online Retrieval Augmented Generation for Tool-use | Liu et al., 2024 | 竞争基线 | 未收录 |
| COLT: Towards Optimized Language Model Powered Tool-use | Zhang et al., 2024 | 专用检索器基线 | 未收录 |
| Attention is All You Need | Vaswani et al., 2017 | 方法基础（Transformer） | 已涵盖 |
| BM25: A Probabilistic Retrieval Model | Robertson et al., 2009 | 稀疏检索基础 | 已涵盖 |

---

## 推荐阅读列表

### P0 必读（多字段建模与工具检索的基础）
- ToolBench: A Large-Scale Python Function Calling Benchmark for LLMs (Qin et al., 2024) — 理解工具检索数据集的构造与评测协议
- EasyTool: Enhancing Tool Use in Language Models with Ease (Wang et al., 2024) — 理解 MFTR 相对改进的主要竞争方案

### P1 重要（被批判工作，理解工具检索的现状与缺陷）
- PLUTo: A Plug-and-Learn Unprompted Tool-Use Framework (Sun et al., 2024) — 理解多步骤工具检索工作流的局限
- OnlineRAG: Online Retrieval Augmented Generation for Tool-use (Liu et al., 2024) — 理解在线学习方法在工具检索中的不稳定性
- Gorilla: Large Language Model Connected with Massive APIs (Patil et al., 2023) — 理解工具文档的异构性挑战

### P2 建议（相关的多字段检索与对齐工作）
- ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT (Khattab & Zaharia, 2020) — 理解多粒度检索与后期交互的思想
- Query2doc: Query Expansion with Large Language Models (Wang et al., 2023) — 理解 LLM 驱动的查询重写与文档增强
- COLT: Towards Optimized Language Model Powered Tool-use (Zhang et al., 2024) — 理解专用工具检索器的训练范式

### P3 参考（工具集成与 Agent 系统背景）
- Tool Learning with Large Language Models (Zeng et al., 2023) — 工具使用的综述性背景
- Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023) — 理解 LLM 工具学习的基础

---

## mem0 知识库记录

- **问题域**：工具检索（Tool Retrieval）、Agent 工具管理、多字段相关性建模
- **方法关键词**：多字段对齐、文档标准化、查询重写、参数约束、自适应权重、伪相关反馈
- **数据集**：ToolBench（10.9k 查询）、APIGen（1k 查询）、APIBank（101 查询）、Gorilla（500 查询）、Toolink（497 查询）、Mixed（3.2k 查询，20.5k 工具）
- **性能基准**：ToolBench NDCG@10 0.5573（BM25, vs EasyTool 0.5245）；Gorilla NDCG@10 0.3775（MiniLM-L6, vs runner-up 0.2336）；Mixed NDCG@10 0.5807（相对次优 +28.6%）
- **关键消融**：标准化+重写单独应用无效甚至有害；自适应权重相比固定加权提升 1-3%；参数缺失惩罚关键（尤其小数据集）
- **核心机制**：四字段 schema（description/parameters/response/examples）;工具感知伪相关反馈的查询重写；sigmoid 参数缺失惩罚（区分必需/可选参数）；端到端学习字段权重 $S(q,t) = \sum w_f S_f - P(q,t)$
- **局限**：参数匹配仍基于语义空间，同义词可能失配；query 分解假设工具需求独立无序；小样本数据集上权重学习有效性未验证
- **推荐下一轮阅读线索**：DRAGIN（动态 RAG 中的信息需求检测，可迁移到工具选择时机决策）、Skill Retrieval Augmentation（技能库的多维度效用建模，与工具多字段概念相近）、Parameter-Efficient Fine-tuning（LLM 在工具文档标准化中的成本控制）

