---
title: "Cognitive Architectures for Language Agents"
paper_id: "arXiv:2309.02427 / TMLR 2024"
analysis_date: "2026-05-06 00:35:00 +08:00"
main_tag: "agentic_ai"
tags:
  - "agentic_ai"
  - "cognitive_architecture"
  - "agent_memory"
  - "agent_framework"
  - "action_space"
  - "decision_making"
  - "survey"
related_papers: "Voyager, Reflexion, Generative Agents, ReAct, SayCan, Tree of Thoughts, MemGPT"
---

# Cognitive Architectures for Language Agents

> 学术分析报告 | 主标签：agentic_ai | 论文标签：agentic_ai;cognitive_architecture;agent_memory;agent_framework;action_space;decision_making;survey

---

## 1. 问题定义

**背景与挑战**

随着 LLM 能力快速提升，研究者开始将其嵌入复杂交互系统——从机器人控制（SayCan）到社会仿真（Generative Agents），从网络导航（ReAct）到游戏 agent（Voyager）。然而，各系统在**术语、内部结构、设计决策**上高度异构：有的系统有长期记忆，有的没有；有的有多类型内部动作，有的仅有外部动作；有的采用多步规划，有的仅做单次动作生成。这种异构性导致：

1. **概念比较困难**：SayCan 和 Reflexion 都被称为"语言 agent"，但其内部机制几乎没有共同术语；
2. **设计缺乏指导原则**：开发新 agent 时，研究者不清楚应包含哪些模块，遗漏哪些关键组件；
3. **经验难以积累**：不同 agent 的成功/失败难以归因到具体设计选择。

**形式化问题描述**：

论文定义**语言 agent（language agent）**为：一个以 LLM 为核心计算引擎，通过感知外部输入、维护内部状态、选择并执行动作来实现目标的智能系统。目标是给出一个**统一框架（CoALA）**，能够：
- 刻画任意语言 agent 的内部组件及其交互关系（$\mathcal{M}, \mathcal{A}, \mathcal{D}$）；
- 指导新 agent 的模块化设计；
- 揭示当前研究的空白与未来方向。

这是一个**框架性/综述性工作**，核心贡献不在于提出新算法，而在于提供**概念统一**与**设计导航**。

**问题的重要性**：缺乏统一框架导致语言 agent 领域的技术积累分散，不同团队重复发明相似机制，开源实现碎片化。CoALA 的出现类似于强化学习领域 MDP 框架的作用——提供标准化术语（state/action/reward/transition）供所有工作对齐。

---

## 2. 前人工作的方法缺陷

**现有 agent 设计的共同缺陷**：

早期语言 agent（如 ReAct, SayCan）仅将 LLM 与外部环境直接对接，缺乏长期记忆和系统性内部动作设计。这导致 agent 在多步任务中无法利用历史经验，每次决策仅基于当前上下文窗口。Reflexion 引入了语义记忆（将反思结果写入 LTM），Voyager 引入了程序性记忆（技能库），Generative Agents 引入了情景记忆+语义反思——但这些设计都是针对特定任务特设的，没有统一视角。

**决策机制的缺位**：绝大多数现有 agent 固定于"单步提议（propose）"的决策模式——LLM 直接输出下一步动作，没有显式的评估（evaluate）或选择（select）阶段。Tree of Thoughts 和 RAP 虽然引入了更复杂的树搜索，但仅用于特定推理任务，未被统一到更广泛的 agent 设计框架中。

**经验教训难以迁移**：由于缺乏统一描述语言，ReAct 中"推理支持行动"的成功经验无法被系统性地移植到其他 agent 架构中。每篇 agent 论文都从头构建自己的概念体系，造成领域知识碎片化。

---

## 3. 研究动机与提出方案

**研究动机**

CoALA 的出发点是**认知科学与符号 AI 研究已经发展出成熟的 agent 认知架构理论**（如 ACT-R、Soar），而 LLM 时代的 agent 设计却完全脱离了这一积累。论文提出：LLM 可以被视为一种**概率产生式系统（probabilistic production system）**——产生式系统（production system）定义了字符串的可能转换规则集，LLM 则以概率 $P(Y|X)$ 的形式实现了相同功能。这一类比将 Soar 等认知架构的设计原则（工作记忆、长期记忆、决策循环）直接对应到 LLM agent 的设计中。

**方法本质**

CoALA（Cognitive Architectures for Language Agents）是一个**三维设计框架**，描述任意语言 agent 为：

$$\text{Agent} = (\mathcal{M}, \mathcal{A}, \mathcal{D})$$

- $\mathcal{M}$（Memory）：记忆模块集合，分层存储不同类型的信息
- $\mathcal{A}$（Action Space）：动作空间，分为内部动作与外部动作
- $\mathcal{D}$（Decision Procedure）：决策程序，驱动 agent 的"主循环"

**去包装后的框架还原**

CoALA 的核心主张可以还原为以下设计约束：
1. LLM 是无状态（stateless）组件，agent 的"持久状态"必须由外部记忆模块维护；
2. LLM 调用产生的中间结果（推理链、检索结果）应被显式存入工作记忆（working memory），而非依赖 LLM 上下文隐式传递；
3. agent 的行为类型不止有"外部执行"（grounding），还有三类内部行为（推理/检索/学习），必须显式设计；
4. 决策不仅是动作生成（propose），还包括评估（evaluate）和选择（select）——大多数现有 agent 缺失后两者。

**记忆模块（§4.1）**

| 记忆类型 | 内容 | 对应示例 |
|----------|------|----------|
| 工作记忆（Working Memory） | 当前决策周期的活跃状态：感知输入、检索到的知识、当前目标 | LLM 上下文窗口（扩展版） |
| 情景记忆（Episodic Memory） | 历史经历轨迹：过去的状态-动作-反馈序列 | Generative Agents 的事件流 |
| 语义记忆（Semantic Memory） | 关于世界和自身的知识：事实、规则、概念 | RAG 的外部知识库；Reflexion 反思结果 |
| 程序性记忆（Procedural Memory） | 关于"如何行动"的知识：LLM 权重（隐式）+ agent 代码（显式） | Voyager 技能库（代码级程序） |

工作记忆是四类记忆的**中央枢纽**：LLM 从工作记忆中选取相关内容构建 prompt，LLM 输出解析后写回工作记忆，再触发后续动作执行。

**动作空间（§4.2-4.5）**

外部动作（External/Grounding）：
- 物理环境交互（机器人指令）
- 人类/多 agent 对话
- 数字环境操作（网页、API、代码执行）

内部动作（Internal）：
- **推理（Reasoning）**：读写工作记忆，LLM 生成推理链、摘要、计划（写入工作记忆但不写入 LTM）
- **检索（Retrieval）**：从 LTM 读取内容到工作记忆（规则/稀疏/稠密三种实现）
- **学习（Learning）**：写入 LTM，包括：更新情景记忆（保存轨迹）、更新语义记忆（保存反思结论）、更新 LLM 参数（finetuning）、更新 agent 代码（生成新技能）

**决策程序（§4.6）**

CoALA 将 agent 的决策分解为**重复决策周期（decision cycle）**，每个周期包含：

1. **规划阶段（Planning Stage）**：
   - *提议（Propose）*：通过推理/检索生成动作候选集
   - *评估（Evaluate）*：为候选动作赋值（启发式、LLM 推理、学习值函数）
   - *选择（Select）*：argmax/softmax/投票选出执行动作
2. **执行阶段（Execution Stage）**：执行选出的外部或学习动作，获取环境观察，进入下一周期

**与已有工作的关键映射**（Table 2）：

| Agent | 长期记忆类型 | 外部基础 | 内部动作 | 决策方式 |
|-------|------------|---------|---------|---------|
| SayCan | 无 | 物理 | 无 | 评估 |
| ReAct | 无 | 数字 | 推理 | 提议 |
| Voyager | 程序性 | 数字 | 推理/检索/学习 | 提议 |
| Generative Agents | 情景+语义 | 数字/agent | 推理/检索/学习 | 提议 |
| Tree of Thoughts | 无 | 数字 | 推理 | 提议+评估+选择 |

**核心贡献**

1. 揭示了 LLM 与产生式系统的类比，为认知架构指导 LLM agent 设计提供理论依据；
2. 建立了 Memory×Action×Decision 三维统一描述框架，支持跨 agent 系统比较；
3. 识别当前研究空白：大多数 agent 缺乏明确的 evaluate+select 阶段，检索与推理的深度整合（adaptive retrieval-reasoning interleaving）未被充分研究，学习动作（尤其是程序性记忆学习）高度欠缺；
4. 提供了应用实例：以"个性化零售 assistant"为案例，说明如何用 CoALA 三步骤（确定记忆模块→设计内部动作空间→定义决策程序）进行应用导向的 agent 设计。

**机制有效性**：CoALA 是概念框架，不做量化验证；其有效性体现在两方面：(a) 能无损描述多种已有 agent（Table 2 的案例映射无遗漏）；(b) 通过 §6 的设计指引识别了多个此前未被明确描述的研究空白，部分后续工作（如 MemGPT, A-MEM, MACLA）已沿着 §6 指出的方向展开。

---

## 4. 实验对比与性能提升

本文是**综述/框架类论文**，不包含量化实验对比。作者用 Table 2 的结构化案例映射代替性能表格，展示 CoALA 对五种代表性 agent 的描述能力。

**案例覆盖**：SayCan（物理+单步评估）、ReAct（数字+推理+提议）、Voyager（程序性记忆+全类型内部动作+提议）、Generative Agents（情景+语义记忆+全类型动作+提议）、Tree of Thoughts（推理+提议/评估/选择）

**analyst assessment**：Table 2 的映射完整、无例外，说明 CoALA 对五个主要 agent 有足够描述力。但框架的**预测能力**（能否预判新设计的有效性）未被验证，这是概念框架类工作的固有局限。

---

## 5. 方法局限与缺陷

**框架性局限（分析者推断）**：

CoALA 是**描述性框架**而非处方性理论——它能描述现有 agent 是什么，但无法从理论上导出"在给定任务场景下，哪种记忆类型+动作空间组合最优"。换言之，框架指出了设计空间，但没有给出设计空间中的寻优方法。

**覆盖完整性的有限性**：框架的案例主要来自 2022-2023 年初的工作，多 agent 协作（multi-agent collaboration）、工具调用（tool use）等方向虽有提及，但未被深入框架化；外部效度依赖后续工作的反哺验证。

**"学习动作"风险被低估**：论文指出程序性记忆学习（agent 代码自我修改）"风险最高"，但缺乏对风险边界的形式化分析——何种条件下自我修改会造成功能破坏或对齐偏移，没有给出可操作的判断准则。

**评估缺失是有意为之**：论文明确定位为综述/框架类贡献，量化验证不在目标范围内，但这意味着框架本身的**设计指导能力**（如"使用程序性记忆比不使用好多少"）仍需大量实证工作支撑。

---

## 6. 科研启发

1. **自适应记忆访问策略（Adaptive Memory Access）**：CoALA 定义了四类记忆，但当前 agent 多使用静态访问策略（固定何时检索、固定检索哪类记忆）。一个有新问题定义的研究方向是：**给定当前工作记忆状态和任务进展，如何决策触发哪类 LTM 访问（检索情景 vs 检索语义 vs 推理生成新信息）？** 可形式化为记忆访问决策的强化学习或元学习问题，与 DRAGIN 的动态检索触发思路类似，但扩展到多记忆类型选择。

2. **程序性记忆的安全学习（Safe Procedural Memory Learning）**：CoALA 指出 agent 代码自修改风险最高，但缺乏可操作的安全保障机制。一个可实验验证的假设是：引入**形式化约束（precondition-postcondition specification）**的程序性记忆写入——agent 只能在新技能通过形式化验证（如类型检查、有限模拟）后才将其写入技能库。这与 Voyager 的 self-verification 思路有区别：Voyager 用 GPT-4 判断任务完成，而此方向用程序形式化验证约束技能质量。

3. **决策程序的元推理（Metareasoning for Decision Procedure）**：CoALA §6 明确指出"metareasoning 以改善效率"是重要方向——agent 应能估计"继续规划（propose+evaluate+select 更多轮）的边际价值"并决定是否停止。这是可形式化为"计算价值估计"的研究问题（Computational Rationality 框架），实验假设是：具备 metareasoning 的 agent 在给定 LLM 调用预算下，能达到比固定规划步数 agent 更高的任务完成率。

4. **多 agent 协作的 CoALA 扩展**：当前 CoALA 框架为单 agent 设计，多 agent 场景（如 CAMEL, ChatDev, MetaGPT）中 agent 间通信本质上是 grounding 到"其他 agent 环境"，但这种 grounding 的内部状态同步（共享记忆 vs 私有记忆，一致性协议）未被 CoALA 覆盖。定义 multi-agent CoALA 扩展（M-CoALA），形式化 agent 间记忆共享协议与一致性约束，是具有新问题定义的方向。

---

## 7. 参考文献图谱

### 文献分类表

| 文献名 | 作者/年份 | 角色 | 知识库状态 |
|--------|----------|------|-----------|
| Voyager: An Open-Ended Embodied Agent with Large Language Models | Wang et al., 2023 | 方法案例 / 实验基线 | 已收录（#63） |
| Reflexion: Language Agents with Verbal Reinforcement Learning | Shinn et al., 2023 | 方法案例 | 已收录（#58） |
| Generative Agents: Interactive Simulacra of Human Behavior | Park et al., 2023 | 方法案例 | 未收录 |
| ReAct: Synergizing Reasoning and Acting in Language Models | Yao et al., 2022 | 方法案例 | 未收录 |
| SayCan: Do As I Can, Not As I Say | Ahn et al., 2022 | 方法案例 | 未收录 |
| Tree of Thoughts: Deliberate Problem Solving with Large Language Models | Yao et al., 2023 | 方法案例 | 未收录 |
| Soar: A Unified Theory of Cognition | Laird et al., 1987 | 方法基础（认知架构） | 未收录 |
| Production Systems and Human Cognition | Newell et al. | 方法基础（产生式系统） | 未收录 |

---

## 推荐阅读列表

### P0 必读（方法案例，知识库未收录）
- Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023) — 最完整实现 CoALA 多类记忆的案例（情景+语义+学习），理解 CoALA 情景/语义记忆的核心参考
- ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022) — 最简单完整 agent，推理+grounding 的基础范式

### P1 重要（框架动机必读）
- Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Yao et al., 2023) — 唯一在 Table 2 中实现 propose+evaluate+select 完整决策程序的方法
- Cognitive Architecture: A Universal Scale of Description for Cognitive Systems (Anderson, 2007) — CoALA 框架的认知科学理论根基

### P2 建议（主要竞争视角）
- AgentBench: Evaluating LLMs as Agents (Liu et al., 2023) — 量化不同 agent 设计在多任务下的对比，是 CoALA 框架的实证补充
- MetaGPT: Meta Programming for Multi-Agent Collaborative Framework (Hong et al., 2023) — 扩展至多 agent 协作的实践，是 CoALA 多 agent 扩展方向的重要参考

### P3 参考（背景综述，选读）
- A Survey on Large Language Model based Autonomous Agents (Wang et al., 2023) — 同期综述，与 CoALA 互补

---

## mem0 知识库记录

- **问题域**：语言 agent 设计框架，认知架构，agent 记忆/动作/决策统一描述
- **方法关键词**：CoALA, memory (working/episodic/semantic/procedural), action space (grounding/reasoning/retrieval/learning), decision procedure (propose/evaluate/select), decision cycle, cognitive architecture
- **数据集**：无（框架类论文）
- **性能基准**：无量化评测；通过 Table 2 展示五类 agent 的描述覆盖能力
- **关联论文 ID**：arXiv:2303.11366 (Reflexion/#58), arXiv:2305.16291 (Voyager/#63), arXiv:2305.10250 (MemoryBank/#59), arXiv:2310.08560 (MemGPT/#57)
- **核心方法机制摘要**：Memory(工作+情景+语义+程序) × Action(grounding+reasoning+retrieval+learning) × Decision(propose+evaluate+select决策周期)，提供完整语言 agent 设计空间描述。
- **推荐下一轮阅读线索**：Generative Agents（完整 CoALA 实现）、Tree of Thoughts（完整决策程序）、AgentBench（量化对比）
