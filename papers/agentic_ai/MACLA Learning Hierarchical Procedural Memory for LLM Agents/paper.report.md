---
title: "MACLA: Learning Hierarchical Procedural Memory for LLM Agents through Bayesian Selection and Contrastive Refinement"
paper_id: "AAMAS:2026"
analysis_date: "2026-05-06 10:30:00 +08:00"
main_tag: "agentic_ai"
tags:
  - "agentic_ai"
  - "procedural_memory"
  - "hierarchical_memory"
  - "bayesian_selection"
  - "contrastive_refinement"
  - "LLM_agents"
related_papers: "MemGPT:arXiv:2310.08560; A-MEM:arXiv:2502.12110; IPR:2025"
---

# MACLA: Learning Hierarchical Procedural Memory for LLM Agents through Bayesian Selection and Contrastive Refinement

> 学术分析报告 | 主标签：agentic_ai | 论文标签：agentic_ai;procedural_memory;hierarchical_memory;bayesian_selection;contrastive_refinement;LLM_agents

---

## 1. 问题定义

**问题背景**：LLM Agent 在执行复杂交互任务时（如家务机器人、网上购物、旅行规划、代码生成），需要从历史轨迹中学习"如何完成"（procedural knowledge / know-how），而不仅仅是记住"发生了什么"（episodic knowledge）。现有方法要么对整条轨迹做微调（参数更新型），要么仅存储自由文本摘要（记忆型），两者都无法高效提取、组织和可靠复用结构化的过程性知识。

**关键挑战**：

1. **推理与学习的耦合**：微调方法将推理能力与任务经验混合编码在同一参数中，导致学习成本高（需 GPU 小时级训练）、灾难性遗忘风险大，且参数更新后无法解释"学了什么"；
2. **整体轨迹优化的盲点**：传统 SFT/RLHF 将整条轨迹作为单元按终结奖励加权，忽略了失败轨迹中存在的正确子步骤，以及成功轨迹中偶然取消的错误动作；
3. **探索与利用失衡**：简单相似度检索无法量化过程的可靠性，无法在不确定性高的场景中选择性探索，也无法避免在已知高风险场景中复用失败过程；
4. **长任务的组合性**：单个原子过程不足以捕捉长期任务的策略，需要一种机制将常用的过程序列抽象为高层"剧本"。

**操作性形式化描述**：

给定一批历史轨迹 $\tau = \{(o_t, a_t, r_t)\}_{t=0}^T$（观测、动作、奖励），目标是从中提取结构化的过程记忆库 $\mathbb{M}_{\text{proc}} = \{(\text{Proc}_i, \mathbf{e}_i, \alpha_i, \beta_i)\}$，其中每条过程 $\text{Proc}_i = \langle \mathcal{G}_i, \Psi_i, \pi_i, \Phi_i \rangle$（目标、前置条件、动作序列、后置条件）。给定新任务观测 $o_t$，系统选择使期望效用最大化的过程：

$$\text{Proc}_t^* = \arg\max_{\text{Proc}_i} \text{EU}(\text{Proc}_i | o_t)$$

其中 $\text{EU}$ 综合考量语义相关性、Beta 后验成功概率、失败风险和探索增益。约束：LLM 参数全程冻结，所有学习发生在外部记忆的显式操作中。

**问题重要性**：本文定位在"推理与学习解耦"这个核心架构问题上，试图证明：一个冻结的 7B LLM + 结构化外部记忆，在 $2800\times$ 更低的训练成本下，可以超越需要数十 GPU 小时的迭代微调方法。若成立，将从根本上改变 Agent 持续学习的范式。

---

## 2. 前人工作的方法缺陷

**共同瓶颈**：现有工作在"如何让 Agent 从经验中学习"这个问题上面临两条均有根本缺陷的路径：

**路径 A：参数更新型（SFT/RLHF/PPO）**：以 IPR（Iterative Process Refinement）和 Step-PPO 为代表。IPR（被批判的主要对比方法）通过步骤级奖励进行迭代训练，需要 44.8 GPU 小时（8×A100）。核心问题有三：(1) 训练与推理耦合，无法在推理时持续适应；(2) 失败轨迹中的正确子步骤信息被完全丢弃（整体奖励掩盖了局部信号）；(3) 新任务域的适应需要重新训练，无法增量更新。ETO（Exploratory Trajectory Optimization）和 RFT-CR（Reject-based Fine-Tuning）也面临相同的计算成本和静态适应问题。

**路径 B：记忆型（缓冲区+检索）**：以 MemGPT（Packer et al., 2023）、MemoryBank（Zhong et al., 2024）、A-MEM（Xu et al., 2025）为代表。这类方法将自由文本（对话片段、摘要、关键词）存入外部记忆，但存在两个根本限制：(1) 自由文本缺乏可执行语义——"成功找到鸡蛋并放置"和"找到鸡蛋但放错位置"在摘要层面的差异极小，无法用于可靠的过程复用；(2) 简单相似度检索无法量化记忆的可靠性，无法区分"曾经成功过一次"和"成功率 90%"的过程。

**具体被批判工作**：Memp（代表性过程记忆工作，被 MACLA 指名批判）将过程记忆视为一等公民，但表示仍以整体脚本或完整轨迹为主（monolithic text），缺乏结构化前/后置条件；检索依赖启发式规则，无不确定性感知；没有从成功/失败对比中精化过程的机制。

这两条路径的缺陷共同引出 MACLA 的核心设计原则：将 LLM 固定为语义解析器，所有学习发生在外部结构化记忆的显式编辑中。

---

## 3. 研究动机与提出方案

### 研究动机

本文的核心张力是：**参数更新代价高昂、无法增量；自由文本记忆可执行性不足、可靠性不可量化**。出路是第三条路：以 LLM 的语义能力提取结构化过程，并用贝叶斯后验对过程的可靠性进行在线估计，在不修改 LLM 参数的前提下持续学习。

这一思路与操作系统中"进程与数据分离"的哲学类似——LLM 是不变的计算引擎，记忆是随经验增长的持久化数据存储，两者通过结构化接口解耦。

### 方法本质

MACLA 是一个**从轨迹提取结构化可执行过程、通过贝叶斯期望效用选取最优过程、通过成功/失败对比精化过程质量、并将频繁共现过程组合为层次化"剧本"**的记忆增强 Agent 框架。其核心假设是：冻结 LLM 具有足够的语义能力进行轨迹分段、抽象和对比分析，而结构化过程（含前/后置条件）提供了自由文本无法提供的可执行约束。

### 四个核心机制

**4.1 LLM 驱动的过程抽象**

给定轨迹 $\tau = \{(o_t, a_t, r_t)\}$，冻结 LLM 将其分段为语义连贯的子任务：

$$\text{Seg} = \mathcal{L}_\theta(\text{Prompt}_{\text{segment}}(\tau)) = \{(t_k^{\text{start}}, t_k^{\text{end}}, d_k)\}_{k=1}^K$$

对每个段提取结构化过程 $\text{Proc}_k = \langle \mathcal{G}_k, \Psi_k, \pi_k, \Phi_k \rangle$（目标、前置条件、动作序列、后置条件）。嵌入向量 $\mathbf{e}_k = \phi([\mathcal{G}_k; \Psi_k; \Phi_k])$ 用于检索和去重（余弦相似度超过 $\theta_{\text{dup}} = 0.85$ 时合并）。

关键设计意图：前/后置条件的显式化使"过程何时适用"有了可验证的语义约束，而不是依赖模糊的句子相似度。2,851 条 ALFWorld 轨迹被压缩为 187 条唯一过程（15:1 压缩），说明任务空间的内在复杂度远低于轨迹数量。

**4.2 贝叶斯可靠性与期望效用选择**

每条过程维护 Beta 后验：

$$p(\rho_i | \mathcal{D}_i) = \text{Beta}(\rho_i; \alpha_i, \beta_i)$$

其中 $\alpha_i$ 和 $\beta_i$ 分别累计历史成功和失败次数。期望效用：

$$\text{EU}(\text{Proc}_i | o_t) = \underbrace{\text{Rel}_i(o_t) \cdot \frac{\alpha_i}{\alpha_i + \beta_i} \cdot R_{\max}}_{\text{期望奖励}} - \underbrace{\text{Risk}_i(o_t) \cdot \frac{\beta_i}{\alpha_i + \beta_i} \cdot C_{\text{fail}}}_{\text{失败成本}} + \underbrace{\lambda_{\text{info}} H[\text{Beta}(\alpha_i, \beta_i)]}_{\text{探索增益}}$$

其中 $\text{Rel}_i(o_t) = \cos(\phi(o_t), \mathbf{e}_i)$ 为上下文相似度，$\text{Risk}_i(o_t)$ 为历史失败上下文中的相似比例，$H[\cdot]$ 为 Beta 分布的微分熵（鼓励探索）。

这一设计的核心价值在于：探索-利用的权衡是自动的——早期熵高，探索主导；证据积累后，期望奖励主导。不需要人工调节探索率。若 $\max_i \text{EU} < \theta_{\text{conf}}$，回退到零样本 LLM 推理。

**4.3 对比式过程精化**

当某过程同时积累了 $\geq 3$ 次成功和 $\geq 3$ 次失败时，触发 LLM 对比分析：

$$\Delta_i = \text{ContrastiveExtract}(S_i, \mathcal{F}_i)$$

提取三维辨别信息：(1) $\Delta\Psi_i^+$（成功特有的前置条件特征）和 $\Delta\Psi_i^-$（失败特有，取反后加入）；(2) $\Delta\pi_i$（动作序列的差异，缺失步骤或顺序错误）；(3) $\Delta\Phi_i$（后置条件的不匹配模式）。精化操作：

$$\Psi_i \gets \Psi_i \cup \Delta\Psi_i^+ \cup \{\neg\psi | \psi \in \Delta\Psi_i^-\}$$
$$\pi_i \leftarrow \text{Merge}(\pi_i, \Delta\pi_i); \quad \Phi_i \leftarrow \Phi_i \cup \Delta\Phi_i$$

当检测到截然不同的执行模式时，将过程特化为多个变体并继承先验计数。这一机制使过程质量随经验增长而提升，且完全通过记忆编辑实现，无需梯度更新。

**4.4 元过程层次化组合**

当序列 $\langle \text{Proc}_{i_1}, \ldots, \text{Proc}_{i_m} \rangle$ 在成功 episode 中重复出现时，将其抽象为元过程：

$$\text{MP}_j = \langle \mathcal{G}_j^{\text{meta}}, \Psi_j^{\text{meta}}, \{\text{Proc}_{i_1}, \ldots, \text{Proc}_{i_m}\}, \Theta_j \rangle$$

其中 $\Theta_j(o_t, \text{index}) \in \{\text{continue, skip, repeat, abort}\}$ 是一个轻量级控制策略，由 LLM 从成功轨迹的分歧点蒸馏得到。元过程也维护独立的 Beta 后验，并周期性精化（添加新分支、重排子过程、裁剪冗余）。

**4.5 本体语义接地**

为实现跨上下文泛化（如"mug"的过程应能适用于"cup"），MACLA 在记忆构建阶段提取高频词并用 Sentence-Transformer 聚类形成隐式领域本体：

$$C_w = \{w' | \text{sim}(\phi(w), \phi(w')) > \theta_{\text{sim}}\}$$

检索时观测被映射到本体类别，使前置/后置条件的词汇匹配跨越词形差异。

### 核心图表解释

**Figure 1**（架构对比）：左侧（传统微调）将轨迹 $(T, A, O, R)$ 通过后训练编码进 LLM 参数，推理时参数不变；右侧（MACLA）LLM 参数完全冻结，学习发生在外部记忆的 Bayesian Selection + Contrastive Refinement + Meta-procedural Composition 三个操作中。核心创新在于：记忆的学习发生在推理时（inference-time learning），而非离线后训练。

**Figure 2**（记忆容量消融）：过程记忆容量在 100→200 时性能饱和（187 条过程，3.6 MB 内存），300 容量时性能略有下降（检索噪声增加）。这说明 ALFWorld 任务空间的内在复杂度有限，自动发现了边界。

**Figure 4**（剪枝分析）：被剪枝过程平均成功率 0.38（vs 保留过程的 0.73），且无高质量过程（>0.7 成功率、>10 次使用）被错误剪枝，验证了 Utility 评分的有效性。

### 机制有效性

消融实验（Table 2，ALFWorld）：

| 消融项 | Seen | Unseen |
|--------|------|--------|
| 全 MACLA | 87.1 | 90.3 |
| w/o Bayesian | 79.4 | 81.2 |
| w/o Contrastive | 83.6 | 85.7 |
| w/o Meta | 81.2 | 78.4 |
| w/o Ontology | 82.8 | 84.1 |

贝叶斯选择的移除导致最大降幅（-7.7/-9.1），说明可靠性感知选择是最关键的组件。元过程对 Unseen 任务的贡献（-11.9）显著大于 Seen（-5.9），说明其主要贡献是组合泛化而非记忆复用。

---

## 4. 实验对比与性能提升

### 数据集

- **ALFWorld** (Shridhar et al., 2021)：文本交互家务任务，6 类（取物、放置等），2,851 训练轨迹，274 测试用例，分 Seen/Unseen 两个测试集；
- **WebShop** (Yao et al., 2022)：电商搜索购物，12,087 商品，1,624 训练，200 测试，评估多步导航和过滤；
- **TravelPlanner** (Xie et al., 2024)：多日行程规划，1,000 训练，45 测试，含硬约束（预算、日期）和软偏好；
- **InterCodeSQL** (Yang et al., 2023)：交互式文本到 SQL 生成，需处理跨 schema 的关系查询。

### 基线

| 基线 | 类型 | 参数规模 |
|------|------|--------|
| GPT-4 (零样本) | Prompt-based | 闭源 |
| Llama-2-7B + SFT | Outcome Refinement | 7B |
| Llama-2-7B + Step-PPO | Process Refinement | 7B |
| Llama-2-7B + IPR | Process Refinement | 7B |
| Claude-3.5-Sonnet + Memp | Memory-based | 闭源 |
| Qwen2.5-72B + Memp | Memory-based | 72B |

### 关键结果（Table 1）

| 方法 | WebShop | InterCodeSQL | TravelPlanner | ALFWorld-Seen | ALFWorld-Unseen | 平均 |
|------|---------|-------------|--------------|--------------|-----------------|------|
| GPT-4 (zero-shot) | 63.2 | 38.5 | 71.9 | 42.9 | 38.1 | 50.9 |
| IPR (7B, 44.8 GPU-hrs) | 71.3 | 61.3 | — | 70.3 | 74.7 | 69.4 |
| Memp (Claude-3.5) | — | — | 65.5 | 82.5 | 74.7 | 74.2 |
| Memp (Qwen2.5-72B) | — | — | 63.8 | 85.7 | 77.2 | 75.6 |
| **MACLA (7B, 0.016 GPU-hrs)** | **70.2** | **59.3** | **83.3** | **87.2** | **90.3** | **78.1** |

**作者报告核心结论**：MACLA（7B）平均 78.1%，超越所有基线，包括使用 10× 更大模型的 Memp（72B 版本 75.6%）。ALFWorld-Unseen 达到 90.3%，正泛化差距 +3.1%（Unseen > Seen），说明过程组合而非记忆过拟合。训练效率：IPR 需 44.8 GPU-hrs，MACLA 仅需 0.016 GPU-hrs（$2800\times$ 加速）。

### 弱改进场景

InterCodeSQL：MACLA 59.3%，低于 IPR 61.3%（-2 points）。作者分析（Figure 5）揭示原因：SQL 查询的 schema 特定性导致过程复用率仅 51%（ALFWorld 约 80%+），可靠性仅 64%，且 SQL 步骤过短（2-3 步）无法形成元过程。这一结果清晰界定了 MACLA 的适用边界：当任务具备**可复用动作**、**层次结构**和**语义一致性**时效果最好，SQL 类原子化任务不满足前两个条件。

---

## 5. 方法局限与缺陷

**自述局限**：论文在 §5.2 中明确指出 InterCodeSQL 的低复用率问题，并分析了 MACLA 的三个适用性先决条件（可复用动作、层次结构、语义一致性），这是罕见的诚实边界界定。

**分析者推断的局限**：

**依赖 LLM 语义能力的隐性成本**：过程抽象（分段、提取前/后置条件）和对比精化均依赖冻结 LLM 的输出质量。论文使用 Llama-2-7B（4 位量化），在语义细粒度场景下 7B 模型的判断可能不可靠，错误的前置条件提取会导致过程被错误触发。论文未报告过程提取的精确率/召回率，这是一个重要的证据缺口。

**评测数据集的任务结构偏向**（validity threat，分析者推断）：ALFWorld 是文本型家务任务，其动作空间相对封闭（"take", "put", "heat", "cool" 等有限动词），天然适合过程抽象。这种结构性优势可能使 MACLA 在 ALFWorld 上的优势无法直接外推到动作空间开放的现实场景（如代码调试、复杂 API 调用）。

**对比精化的样本效率**：要触发对比精化需要同一过程积累 $\geq 3$ 次成功和 $\geq 3$ 次失败。对于稀有任务或高成功率过程，可能长期无法触发精化，记忆质量无法自动提升。论文未分析训练集中有多少比例的过程达到了精化触发条件。

**元过程控制策略的蒸馏质量**：$\Theta_j$ 是从成功轨迹中蒸馏的轻量级控制策略，但"continue/skip/repeat/abort"的判断本质上是序列决策，使用 LLM 的模式匹配可能无法捕捉所有分支条件，尤其在状态空间复杂的环境中。

---

## 6. 科研启发

1. **过程记忆的可靠性衰减问题**：MACLA 的 Beta 后验仅追踪历史成功/失败的总数，但过程的适用性可能随环境分布漂移而变化（如平台更新后旧的 WebShop 过程失效）。一个未解决的问题是：如何设计时间感知的可靠性估计，在环境分布发生跳跃时快速识别过程失效并降低其后验置信度？形式化假设是：环境跳跃点可以通过过程成功率的突然下降来检测，设计出能在有限次失败后快速重置后验的贝叶斯更新规则，并在受控环境漂移实验中验证。

2. **跨任务域的过程迁移学习**：MACLA 的过程库是任务特定的，不同任务域的过程库之间无法共享。然而，"取物品"这类动作模式在家务任务（ALFWorld）和工厂搬运等任务中是相同的。设计一个任务无关的过程原语库（domain-agnostic procedure primitives），使过程能以组合方式在新域快速适应，是一个有明确问题定义和可验证假设的方向。

3. **从轨迹对比中学习负样本的形式化**：MACLA 的对比精化机制依赖 LLM 识别成功/失败的差异，但"为什么失败"本身是一个复杂推理问题。一个更严格的方向是：将失败归因形式化为结构化因果图（将失败节点定位到特定动作步骤），替代当前的自由文本对比，从而使过程精化的粒度更精确。验证实验可通过人工标注失败步骤作为 ground truth，比较两种精化方式的过程改进速度。

4. **过程记忆的隐私与安全边界**：在多用户 Agent 场景中，不同用户的交互轨迹被抽象为共享过程记忆，可能导致用户特定信息（如购物偏好、密码格式）泄漏到其他用户的过程上下文中。过程记忆的隐私感知设计（区分可共享的通用操作过程和用户特定过程）是一个在隐私计算与 Agent 记忆交叉处的新问题。

---

## 7. 参考文献图谱

### 文献分类表

| 文献名 | 作者/年份 | 角色 | 知识库状态 |
|--------|----------|------|-----------|
| ALFWorld: Aligning Text and Embodied Environments for Interactive Learning | Shridhar et al., 2021 | 数据集来源 | 未收录 |
| WebShop: Towards Scalable Real-World Web Interaction | Yao et al., 2022 | 数据集来源 | 未收录 |
| IPR: Iterative Process Refinement | (MACLA 内引，2025) | 被批判文献 + 主要基线 | 未收录 |
| Memp: Procedural Memory for LLM Agents | (MACLA 内引，2025) | 被批判文献 | 未收录 |
| ReAct: Synergizing Reasoning and Acting in Language Models | Yao et al., 2023 | 方法背景 | 未收录 |
| Reflexion: Language Agents with Verbal Reinforcement Learning | Shinn et al., 2023 | 方法背景 | 未收录 |
| MemGPT: Towards LLMs as Operating Systems | Packer et al., 2023 | 记忆型基线对比 | 已收录（本批次）|
| A-MEM: Agentic Memory for LLM Agents | Xu et al., 2025 | 记忆型相关工作 | 已收录（本批次）|
| Llama 2: Open Foundation and Fine-Tuned Chat Models | Touvron et al., 2023 | 基础模型 | 未收录 |

---

## 推荐阅读列表

### P0 必读（被批判文献，理解动机必读）
- **Memp: Procedural Memory for LLM Agents** — MACLA 直接批判的工作，是本文"有结构化过程记忆但缺乏可靠性量化和对比精化"的批判对象。理解 Memp 的不足是理解 MACLA 三项核心改进的前提。
- **IPR: Iterative Process Refinement** — MACLA 的主要对比基线（$2800\times$ 加速的对比参照），理解其步骤级奖励训练方式有助于评估 MACLA 的效率声明。

### P1 重要（实验基准与方法背景）
- **ALFWorld: Aligning Text and Embodied Environments for Interactive Learning** (Shridhar et al., ICLR 2021) — MACLA 的主要评测环境，理解任务设计是判断外部有效性的基础。
- **ReAct: Synergizing Reasoning and Acting in Language Models** (Yao et al., ICLR 2023, CCF-A) — Prompt-based Agent 的代表性工作，构成 MACLA 的 zero-shot 对比背景。

### P2 建议（相关记忆型工作）
- **HiAgent: Hierarchical Working Memory for Solving Long-Horizon Tasks** — 同类层次化记忆架构，与 MACLA 的元过程层次有直接可比性。
- **SAGE: Reflective Memory Management for LLM Agents** — 多 Agent 反射式记忆管理，与 MACLA 在记忆维护策略上的对比值得关注。

---

## mem0 知识库记录

- **问题域**：LLM Agent 过程记忆；推理学习解耦；贝叶斯过程选择；对比式记忆精化
- **方法关键词**：Procedural Memory；Beta 后验；期望效用选择；对比精化；元过程；本体语义接地；冻结 LLM
- **数据集**：ALFWorld（2851/274 train/test，6 类家务任务）；WebShop（1624/200）；TravelPlanner（1000/45）；InterCodeSQL
- **性能基准**：平均 78.1%（4 基准）；ALFWorld-Unseen 90.3%（+3.1% 正泛化间隙）；TravelPlanner CS 83.3%；2800× 训练加速 vs IPR；187 过程/2851 轨迹（15:1 压缩）
- **关联论文 ID**：MemGPT（arXiv:2310.08560）；A-MEM（arXiv:2502.12110）
- **核心方法机制摘要**：四阶段：(1) 冻结 LLM 将轨迹分段抽象为结构化过程 $\langle \mathcal{G}, \Psi, \pi, \Phi \rangle$；(2) Beta 后验期望效用选择（平衡相关性、成功率、失败风险、探索熵）；(3) 成功/失败对比精化过程质量；(4) 频繁共现过程抽象为元过程"剧本"
- **推荐下一轮阅读线索**：Memp（过程记忆先驱）；IPR（对比基线）；HiAgent（层次记忆）；ReAct（Prompt-based 基线）
