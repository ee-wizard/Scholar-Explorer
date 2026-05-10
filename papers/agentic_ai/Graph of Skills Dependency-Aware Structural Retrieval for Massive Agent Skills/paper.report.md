---
title: "Graph of Skills: Dependency-Aware Structural Retrieval for Massive Agent Skills"
paper_id: "arXiv:2604.05333"
analysis_date: "2026-05-07 09:00:00 +08:00"
main_tag: "agentic_ai"
tags:
  - "agentic_ai"
  - "agent_skill"
  - "skill_retrieval"
  - "graph_retrieval"
  - "personalized_pagerank"
  - "dependency_aware"
  - "skill_library"
  - "tool_retrieval"
related_papers: "SkillsBench (arXiv:2602.12670), SRA (arXiv:2604.24594), Voyager (arXiv:2305.16291), HippoRAG (arXiv:2405.14831), ToolNet (arXiv:2403.00839), ToolLLM (Qin et al., 2024)"
---

# Graph of Skills: Dependency-Aware Structural Retrieval for Massive Agent Skills

> 主标签：agentic_ai　　论文标签：agentic_ai;agent_skill;skill_retrieval;graph_retrieval;personalized_pagerank;dependency_aware;skill_library;tool_retrieval

> 学术分析报告 | arXiv:2604.05333 | Liu et al., 2026 | UPenn / UMD / Brown / CMU / Lehigh

---

## 1. 问题定义

**问题背景与核心挑战**

当 LLM Agent 从"几十个工具"扩展到"上千个可重用技能"时，检索方式的选择直接决定了任务成功率与系统成本之间的权衡。论文把这个问题定义为：在一个包含数千技能的本地技能库上，如何在每次推理时找到一个小的、执行上自洽的技能子集，使得 Agent 能在 token 预算内完成任务。这个问题有两条关键约束：第一，结果集必须"执行完整（execution-complete）"，即不能遗漏目标技能所依赖的前置工具；第二，结果集必须紧凑，不能把大量无关技能暴露给 Agent 导致上下文拥挤。

**形式化定义**

设本地技能库为 $\mathcal{C} = \{d_1, \ldots, d_m\}$，GoS 将其转化为带类型的有向图 $G = (V, E, w, \phi)$，其中节点 $v \in V$ 为可执行技能记录，边类型 $\phi(e) \in \mathcal{R} = \{\text{dep, wf, sem, alt}\}$ 分别表示依赖、工作流、语义和替代关系。给定任务查询 $q$ 和上下文预算 $\tau$，检索目标是：

$$\max_{B \subseteq V} \sum_{v \in B} \text{rel}(v, q) + \beta \sum_{(u,v) \in E_{\text{dep}}} \mathbb{I}[u \in B \wedge v \in B] \quad \text{s.t.} \quad \text{cost}(B) \leq \tau$$

第一项最大化相关性，第二项奖励依赖完整性（即同时检索到依赖对），整体约束在 token 预算内。该优化问题不精确求解，GoS 通过三阶段近似：混合种子检索 → 反向图扩散 → 预算化重排序与水化（hydration）。

**问题重要性**

Shi et al. (2025) 已实证"工具检索本身是真实工具生态中的主要瓶颈"。此前主流方案或把全部技能塞入上下文（Vanilla Skills），或做语义最近邻检索（Vector Skills）——前者 token 成本随库规模线性增长，后者只检索语义相似项，会系统性遗漏"语义弱但执行必需"的前置技能（如 parser、preprocessor、setup utility），导致 Agent 拿到不完整的执行束（prerequisite gap）。GoS 的贡献是把技能检索从"相关性匹配"升级为"依赖完整性感知的图扩散"。

---

## 2. 前人工作的方法缺陷

**两类主流方案的共同根因**

**Vanilla Skills（全量暴露）** 把整个技能库拼入 prompt，在小规模时有效，但 token 成本随库规模线性增长（500→2000 技能时从 1.93M 升至 5.84M token，约 3×），同时把无关技能淹没相关约束，造成 context pollution 和幻觉风险（Liu et al., 2024a，"Lost in the Middle"已实证长上下文中间信息被忽略的现象）。

**Vector Skills（语义向量检索）** 用 text-embedding-3-large 做 top-k 语义检索，大幅压缩 token 成本，但语义相似度不等于执行充分性——许多工程任务的入口函数语义显著，但其前置解析器、数据转换器、环境初始化脚本等在任务描述中几乎不出现，因此向量检索系统性错失这些功能必需但语义弱的技能。实验中 Vector Skills 在 SkillsBench 上全面低于 Vanilla Skills（Claude: 19.3 vs. 25.0；GPT: 21.5 vs. 27.4），直接说明检索不完整的代价高于 token 浪费的代价。

**根本原因**：两类方案都没有对技能间的结构关系建模。技能库不是一组独立文档，而是有依赖拓扑的可执行生态系统。不利用这一结构，检索结果的"可执行完整性"无法保证。

**相关图结构工作的局限**

GraphRAG（Edge et al., 2024）和 HippoRAG（Jiménez Gutiérrez et al., 2024）把图用于知识合成或记忆检索，目标是关系型 QA 或长期记忆；ToolNet（Liu et al., 2024b）对大规模工具接入引入图结构，但面向工具发现与导航，不解决"检索一个依赖完整的本地执行束"问题。GoS 针对的上游检索目标（在生成开始前选出紧凑的可执行束）与上述工作有本质差异。

---

## 3. 研究动机与提出方案

### 研究动机与方法本质

GoS 的核心洞察是：**技能库是一个有依赖拓扑的图，而不是一组独立文档**。一个高层求解器（solver）的语义与查询高度匹配，但它依赖的底层解析器、数据转换器在查询中几乎没有词汇信号。传统检索把"相关性信号"与"依赖结构"分开处理，前者仅用于召回，后者完全忽略。GoS 的设计决策是：先用语义+词汇混合信号定位"种子节点"（即高层入口技能），再沿依赖反向边向图内扩散，把前置技能传播性地拉入检索结果。这样，依赖恢复不需要显式推理每个前置条件，而是通过图扩散隐式完成。

这一思路与 HippoRAG 的 PPR 扩散高度同源——两者都用 Personalized PageRank 在图上做知识传播——但 HippoRAG 的图是从文本抽取的 KG（知识实体-关系），GoS 的图是从可执行 I/O schema 归纳的依赖图，目标从"多跳知识推理"变为"依赖完整的执行束选择"。

### 核心假设与创新点

**核心假设**：技能间的依赖关系（谁的输出是谁的输入）可以通过 I/O 字段兼容性在离线阶段大规模归纳，并在推理时通过图扩散高效利用，无需为每个查询重新推理依赖。

**核心创新**：
1. **带类型有向图的离线构建**：四类边（dep/wf/sem/alt）中，dep 边由 I/O 兼容性确定性归纳，其余通过稀疏验证（lexical + semantic 候选池 + 关系验证）添加，避免全对归纳的 $O(m^2)$ 开销。
2. **反向感知的 PPR 扩散**：标准 PPR 沿正向边传播，但 GoS 需要"从已检索的高层技能反向找其前置依赖"。因此 GoS 同时使用正向算子 $T_r^\rightarrow$ 和反向算子 $T_r^\leftarrow$，通过 $\gamma_r$ 参数控制每类边的反向穿越强度。
3. **预算化水化（Hydration）**：检索结果不是元数据摘要，而是包含稳定源路径 + 能力描述 + 执行注记的"可消费 payload"，在全局 token 预算 $\tau$ 下按分数降序装配。

### 方法流程详解

**离线阶段：图构建**

每个技能包被解析为规范化记录，包含：能力摘要、I/O 字段、domain 标签、工具依赖、脚本入口点、兼容性注记。解析优先确定性提取，仅在文档不完整时调用轻量 LLM 补全语义字段。

依赖边归纳：若技能 $v_i$ 的输出类型与 $v_j$ 的输入类型兼容，则添加边 $v_i \to v_j$（即 $v_i$ 是 $v_j$ 的前置依赖）。非依赖边（wf/sem/alt）通过稀疏验证：先用词汇相似度 + 语义邻居 + I/O 扩展构建候选池，再在池内做关系验证，避免全对推理。

**在线阶段：结构化检索**

*Step 1 — 混合种子检索*：查询 $q$ 映射为轻量检索 schema（任务目标、关键操作、引用 artifact、标准化关键词）。计算语义分 $s_i^{\text{sem}}(q)$ 和词汇分 $s_i^{\text{lex}}(q)$，合并为：
$$z_i(q) = \eta s_i^{\text{sem}}(q) + (1-\eta) s_i^{\text{lex}}(q)$$
归一化后得初始种子分布 $\mathbf{p}$。词汇信号对 artifact 名称、文件名等具体引用鲁棒，语义信号覆盖同义描述，两者互补。

*Step 2 — 反向感知 PPR 扩散*：构建统一转移算子：
$$T = \text{RowNorm}\left(\sum_{r \in \mathcal{R}} \lambda_r (T_r^\rightarrow + \gamma_r T_r^\leftarrow)\right)$$
迭代扩散：
$$\mathbf{s}^{(\ell+1)} = \alpha \mathbf{p} + (1-\alpha) T^\top \mathbf{s}^{(\ell)}$$
重启参数 $\alpha$ 保持对种子节点的锚定。一个高层求解器作为种子后，其上游解析器、转换器通过反向 dep 边积累分数，即使这些前置技能对查询语义贡献极低。

*Step 3 — 预算化重排序与水化*：合并图扩散分与字段级直接匹配分：
$$\rho_i(q) = \mathbf{s}_i^\star + \mu m_i(q)$$
按 $\rho_i$ 降序，在 per-skill 和全局预算 $\tau$ 下装配执行束。水化输出为 agent 可直接消费的 payload（稳定源路径 + 能力摘要 + 执行注记），而非裸元数据。

### 机制有效性的解释

消融实验（Table 2，1000 技能，GPT-5.2 Codex）给出清晰分离：
- 去掉图传播：reward 34.4 → 29.3（↓5.1），token 1.38M → 0.89M；
- 去掉词汇+重排序：reward 34.4 → 26.7（↓7.7），token 1.38M → 1.01M。

词汇+重排序的降幅更大，说明"种子质量"是瓶颈——种子差时图扩散无法从弱入口点恢复有效的依赖结构。这证明两个组件有非对称的依赖关系：词汇种子质量决定了图扩散的起点，图传播在好种子上才能充分发挥依赖恢复能力。

**关键图表解读**：Figure 2 展示了三阶段流程的完整架构——左侧是离线归纳的有向多关系图（dep/wf/sem/alt 四类边），中心是图结构本身，右侧是在线检索路径（混合种子 → PPR → 预算水化）。Figure 3 和 Table（灵敏度研究）展示规模扩展行为：Vanilla Skills token 成本 3× 增长时，GoS 与 Vector Skills 均保持稳定（约 1.1–1.4M），而 GoS 在奖励上持续超越两者，advantage 在 1000 技能时最大。

---

## 4. 实验对比与性能提升

**数据集**
- **SkillsBench**（Li et al., 2026b，arXiv:2602.12670）：多领域真实技术任务，1000 技能配置；涵盖宏观经济去趋势、电网可行性分析、三维扫描分析、金融建模、地震相位拾取等 11 个领域。
- **ALFWorld**（Shridhar et al., 2020b）：文本具身交互仿真器，140 个 episode，任务为家庭多步操作（导航/寻找/操作物体）；平均奖励等价于成功率。

**基线**
- **Vanilla Skills**：全量技能库塞入 prompt，最大召回但无压缩，ALFWorld 沿用官方 Agent Skills 格式。
- **Vector Skills**：text-embedding-3-large（3072 维）做 top-k 语义检索，隔离图结构贡献——与 GoS 共享同一 embedding 模型，差异仅在是否使用依赖图。

**模型**：Claude Sonnet 4.5、MiniMax M2.7、GPT-5.2 Codex，每设置跑两次取均值。

**主要结果（Table 1）**

| 模型 | 方法 | SkillsBench R↑ | SkillsBench T↓ | ALFWorld R↑ | ALFWorld T↓ |
|------|------|---------------|---------------|------------|------------|
| Claude | Vanilla | 25.0 | 967,791 | 89.3 | 1,524,401 |
| Claude | Vector | 19.3 | 894,640 | 93.6 | 28,407 |
| Claude | **GoS** | **31.0** | **860,315** | **97.9** | **27,215** |
| MiniMax | Vanilla | 17.2 | 942,113 | 47.1 | 2,184,823 |
| MiniMax | Vector | 10.4 | 852,881 | 50.7 | 66,109 |
| MiniMax | **GoS** | **18.7** | 867,452 | **54.3** | **65,227** |
| GPT-5.2 | Vanilla | 27.4 | 3,187,749 | 89.3 | 1,435,614 |
| GPT-5.2 | Vector | 21.5 | 1,243,648 | 92.9 | 34,436 |
| GPT-5.2 | **GoS** | **34.4** | 1,379,773 | **93.6** | **46,462** |

（数据来源：论文 Table 1）

**综合结论**：GoS 在全部六个模型×基准组合中均取得最高奖励；相对 Vanilla Skills 在五个组合中减少 runtime；token 成本远低于 Vanilla Skills，与 Vector Skills 接近。论文报告的整体提升为：相对 Vanilla Skills，平均奖励提升 43.6%，input token 减少 37.8%（跨两个基准和三种模型族）。

**最强改善场景**：SkillsBench 的长时域任务（需要多步骤组合技能+前置工具），Vector Skills 显著低于 Vanilla Skills（说明依赖缺失是真实痛点），GoS 大幅反超。Claude 在 SkillsBench 上：GoS 31.0 vs. Vector 19.3 vs. Vanilla 25.0。

**较弱改善场景**：GPT-5.2 在 ALFWorld 上 GoS(93.6) 与 Vector(92.9) 差距不大，说明短时域且语义直接的任务对图结构依赖较小；Vector Skills 已足够。MiniMax 整体绝对分值较低，GoS 仍有相对优势但幅度较小，可能与该模型对工具调用的基础能力有关。

**规模灵敏度（Table/Figure 3，GPT-5.2，200→2000 技能）**：
- Vanilla: 1.85→5.84M token（3×），GoS: 1.36→1.14M（稳定）
- 200 技能时 Vanilla(32.5) 略超 GoS(32.1)；500 技能起 GoS 持续领先，advantage 在 1000 技能时最大（34.4 vs. 27.4/21.5）

**消融（Table 2）**：
- Full GoS: R=34.4, T=1.38M
- w/o 图传播: R=29.3（↓5.1），T=0.89M（更省但效果更差）
- w/o 词汇+重排序: R=26.7（↓7.7），T=1.01M

词汇+重排序去掉后降幅更大，确认种子质量是系统的首要约束。

---

## 5. 方法局限与缺陷

**作者明示的局限**

1. **图质量依赖文档质量**：文档不完整、I/O schema 模糊、缺少执行元数据的技能会导致 dep 边质量下降，进而损害检索。这是一个不可忽视的现实约束——工程实践中许多工具文档并不规范。
2. **静态图**：当前实现不从执行轨迹、验证器结果或用户反馈在线更新图结构。随技能库动态增长或任务分布偏移，图可能过时。
3. **未评测多模态和交互式场景**：实验范围限于 SkillsBench（结构化技术任务）和 ALFWorld（文本具身任务），对多模态技能库或实时交互 Agent 的泛化性未验证。

**分析者补充的批判**

4. **实验规模**：两次运行均值作为主要结论，未报告标准差或统计显著性检验。论文声明"解释为一致经验模式而非正式显著性主张"，诚实但也削弱了结论强度。
5. **图构建成本未量化**：论文 Reproducibility Statement 明确代码未随提交公开，图构建时间、内存消耗、各技能库规模下的 offline indexing 耗时均未报告，难以评估实际部署成本。
6. **基线公平性**：Vector Skills 使用与 GoS 相同的 embedding 模型（openai/text-embedding-3-large），这使"图结构 vs. 纯语义"的对比清晰，但忽略了更强的 dense retriever（如领域微调模型）或混合检索（BM25 + dense）作为 baseline，可能低估了无图方案的上限。
7. **SkillsBench 任务分布**：SkillsBench 涵盖的高度专业化任务（地震相位、电网可行性）天然更依赖前置工具链，对 GoS 有利。是否存在任务类型使 GoS 系统性弱于简单向量检索，论文未分析负向案例。
8. **runtime 结论的模型依赖性**：GPT-5.2 下 GoS 在某些配置 runtime 高于 Vanilla（论文解释为 GPT 有 fixed library 缓存机制），而 Claude 和 MiniMax 下 GoS 反而更快。这说明 runtime 优势依赖模型内部实现，不是 GoS 的通用保证。

---

## 6. 科研启发

1. **在线自适应图更新**：GoS 当前图是静态的，无法从执行反馈自动修正错误边。一个可验证的研究问题是：能否设计一个"执行-回流"机制，用 Agent 的成功/失败轨迹在线更新 dep 边权重，并证明在一定条件下边权收敛到与真实执行依赖一致的估计？核心假设：执行成功时共同出现的技能对提供了边正向证据，失败时出现但未被调用的技能对提供了噪声边负向证据。可验证性：在受控技能库上注入已知错误边，观察自适应更新能否在有限轮次内纠正。

2. **依赖感知检索的理论界**：GoS 的预算化选择问题（公式 1）是一个带奖励子模式项的背包问题变种（覆盖 dep 完整对时有额外奖励），其近似算法是否有可分析的逼近比？当前 PPR 扩散是该问题的无保证启发式近似。若能证明在技能依赖图满足某类结构（如有界树宽或有界最大入度）时，PPR 逼近原始整数规划的误差有上界，则为检索设计提供了理论基础，也解释了 GoS 在不同任务类型上的一致性。

3. **文档质量到图质量的传播模型**：GoS 图质量的上限取决于技能文档质量（I/O schema 完整性）。当前实现在文档缺失时用 LLM 轻量补全，但没有对补全误差的传播分析。一个新问题是：在技能文档质量分布已知的前提下（比如通过自动化文档质量评分），能否预测 dep 边的期望精度，并据此决定哪些技能需要人工完善文档才能被安全纳入图？这需要建立"文档不完整性 → 边归纳误差 → 检索覆盖率下降"的完整传播模型。

4. **图结构与 Agent 决策深度整合**：GoS 将图检索作为推理前的"上游选择层"，Agent 接收的是已经装配好的执行束，不感知图结构。一个更激进的设计是：让 Agent 在推理过程中迭代地查询图（例如在执行到某步发现需要额外依赖时，触发局部图扩散），形成"检索-执行-再检索"的闭环。可验证假设：动态多轮图查询在任务复杂度高（依赖链长）时优于单轮预加载，但在简单任务上增加无效延迟。这与 DRAGIN 的"按需检索"思路在问题层面同构，但作用于技能图而非文档语料。

---

## 7. 参考文献图谱

### 文献分类表

| 文献名 | 作者/年份 | 角色 | 知识库状态 |
|--------|----------|------|-----------|
| SkillsBench: Benchmarking how well agent skills work across diverse tasks | Li et al., 2026 (arXiv:2602.12670) | 实验基线·评测基准 | 已分析 |
| Voyager: An Open-Ended Embodied Agent with Large Language Models | Wang et al., 2023 (arXiv:2305.16291) | 背景·扁平技能库代表工作 | 已分析 |
| HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models | Jiménez Gutiérrez et al., 2024 (arXiv:2405.14831) | 方法基础·PPR 扩散在记忆检索的应用 | 已分析 |
| Reflexion: Language Agents with Verbal Reinforcement Learning | Shinn et al., 2023 (arXiv:2303.11366) | 实验基线 | 已分析 |
| Lost in the Middle: How Language Models Use Long Contexts | Liu et al., 2024 | 被批判文献·Vanilla Skills 问题的理论依据 | 未收录 |
| ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs | Qin et al., 2024 | 被批判文献·大规模工具检索挑战来源 | 未收录 |
| Gorilla: Large Language Model Connected with Massive APIs | Patil et al., 2023 | 被批判文献·大规模 API 检索代表工作 | 未收录 |
| Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for LLMs | Shi et al., 2025 (arXiv:2503.01763) | 方法基础·工具检索作为独立难题的实证 | 未收录 |
| ToolNet: Connecting LLMs with Massive Tools via Tool Graph | Liu et al., 2024 (arXiv:2403.00839) | 被批判文献·图结构工具访问但目标不同 | 未收录 |
| From Local to Global: A Graph RAG Approach to Query-Focused Summarization | Edge et al., 2024 (arXiv:2404.16130) | 方法基础·图结构检索用于知识合成 | 未收录 |
| SkillNet: Create, Evaluate, and Connect AI Skills | Liang et al., 2026 (arXiv:2603.04448) | 背景·技能生态系统管理（互补而非竞争） | 未收录 |
| Topic-Sensitive PageRank | Haveliwala, 2002 | 方法基础·PPR 算法基础 | 未收录 |
| Dense Passage Retrieval for Open-Domain Question Answering | Karpukhin et al., 2020 | 方法基础·dense retrieval 基础 | 未收录 |
| Dynamic Tool Dependency Retrieval for Efficient Function Calling | Patel et al., 2025 (arXiv:2512.17052) | 方法基础·工具依赖检索的近期工作 | 未收录 |
| ALFWorld: Aligning Text and Embodied Environments for Interactive Learning | Shridhar et al., 2020 | 实验基准 | 未收录 |

---

## 推荐阅读列表

### P0 必读（方法基础，知识库未收录）

- **Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models** (Shi et al., 2025, arXiv:2503.01763) — GoS 直接引用的理论动机来源，实证了通用稠密检索器对工具检索的不适配性，是理解 GoS 问题设定必读的前置工作。
- **Dynamic Tool Dependency Retrieval for Efficient Function Calling** (Patel et al., 2025, arXiv:2512.17052) — 近期工作，与 GoS 问题设定（工具依赖检索）最直接相关，需要了解两者的异同。
- **Lost in the Middle: How Language Models Use Long Contexts** (Liu et al., 2024, TACL) — Vanilla Skills 方案之所以失败的理论解释；上下文中间位置信息被遗忘的实证。
- **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs** (Qin et al., 2024, ICLR) — GoS 批判的核心背景工作，16000+ API 规模下的工具调用与检索挑战。

### P1 重要（被批判文献 / 方法基础，理解 GoS 机制必读）

- **SkillNet: Create, Evaluate, and Connect AI Skills** (Liang et al., 2026, arXiv:2603.04448) — GoS 的互补系统，聚焦技能生态系统的创建与管理（对应 GoS 的检索层），了解两者协作关系有助于定位 GoS 的系统边界。
- **ToolNet: Connecting Large Language Models with Massive Tools via Tool Graph** (Liu et al., 2024, arXiv:2403.00839) — 图结构用于工具访问的先前工作，GoS 明确将自身与其区分：ToolNet 面向工具导航，GoS 面向依赖完整的执行束检索。

### P2 建议（同方向近期工作，扩展视野）

- **Graph of Skills: Dependency-Aware Structural Retrieval for Massive Agent Skills** 项目页: https://github.com/davidliuk/graph-of-skills — 代码开源后可直接复现。
- **Skill Retrieval Augmentation for Agentic AI** (Su et al., 2026, arXiv:2604.24594) — 本库已分析，GoS 直接定位的同期并列工作，两者在 skill retrieval 问题框架上高度重叠但方法不同（SRA 三阶段框架 vs. GoS 图扩散）。
- **GraSP: Graph-Structured Skill Compositions for LLM Agents** (Xia et al., 2026, arXiv:2604.17870) — 与 GoS 同期的图结构技能工作，但切入点不同：GraSP 在技能执行层引入 DAG 编译，GoS 在检索层引入依赖图扩散；两篇合读形成"检索-执行"完整链路的图结构视角。

### P3 参考（背景综述，选读）

- **Augmented Language Models: A Survey** (Mialon et al., 2023, TMLR) — 工具增强语言模型的综述，GoS 的问题域背景。
- **Efficient Algorithms for Personalized PageRank Computation: A Survey** (Yang et al., 2024, IEEE TKDE) — GoS PPR 扩散的算法基础综述。

---

## mem0 知识库记录

- **问题域**：大规模 Agent 技能库的依赖感知结构化检索
- **方法关键词**：graph of skills, offline dependency graph, typed edge induction, hybrid seeding, reverse-aware PPR, budgeted hydration, prerequisite gap
- **数据集**：SkillsBench（1000 技能，11 领域技术任务）、ALFWorld（140 episode 文本具身任务）
- **性能基准**：GoS vs. Vanilla: +43.6% avg reward, -37.8% input tokens（两基准三模型均值）；Claude/SkillsBench: GoS 31.0 > Vanilla 25.0 > Vector 19.3；Claude/ALFWorld: GoS 97.9% > Vector 93.6% > Vanilla 89.3%
- **关联论文 ID**：arXiv:2602.12670 (SkillsBench), arXiv:2604.24594 (SRA), arXiv:2305.16291 (Voyager), arXiv:2405.14831 (HippoRAG), arXiv:2503.01763 (Shi et al. tool retrieval benchmark), arXiv:2604.17870 (GraSP)
- **核心方法机制摘要**：离线归纳 I/O 兼容性 dep 边构建有向多关系图（dep/wf/sem/alt）；推理时混合语义+词汇种子，通过反向感知 PPR 扩散恢复前置依赖，预算化水化输出可执行束；关键参数 α（PPR 重启）、η（语义词汇权重）、γ_r（反向穿越强度）、μ（字段匹配权重）。
- **推荐下一轮阅读线索**：Retrieval Models Aren't Tool-Savvy (arXiv:2503.01763)；GraSP (arXiv:2604.17870)；SkillNet (arXiv:2603.04448)；Dynamic Tool Dependency Retrieval (arXiv:2512.17052)
