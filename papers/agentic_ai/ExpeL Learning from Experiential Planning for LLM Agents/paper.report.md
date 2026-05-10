---
title: "ExpeL: LLM Agents Are Experiential Learners"
paper_id: "arXiv:2308.10144 / AAAI 2024"
analysis_date: "2026-05-06 13:00:00 +08:00"
main_tag: "agentic_ai"
tags:
  - "agentic_ai"
  - "experiential_learning"
  - "agent_knowledge_extraction"
  - "in-context_learning"
  - "transfer_learning"
  - "agent_memory"
  - "ReAct"
related_papers: "Reflexion, ReAct, HotPotQA, ALFWorld, WebShop"
---

# ExpeL: LLM Agents Are Experiential Learners

> 学术分析报告  
> **主标签**：agentic_ai  
> **论文标签**：agentic_ai;experiential_learning;agent_knowledge_extraction;in-context_learning;transfer_learning;agent_memory;ReAct

---

## 1. 问题定义

### 背景与核心张力

将 LLM 用于决策任务时，存在两种极端方案：（1）在大量人工标注轨迹上微调——成本高、可能损害泛化、无法访问闭源模型；（2）纯 few-shot prompting——受 context 窗口限制，Agent 无跨任务记忆，从本质上不能"学习"。现有的 Reflexion（Shinn et al. 2023）等自我改进方法局限于同一任务的多轮重试（intra-task 学习），无法将一个任务上的经验迁移到后续不同任务中。

### 形式化定义

设训练任务集合 $\mathcal{T}_{\text{train}} = \{t_1, \ldots, t_N\}$，评估任务集合 $\mathcal{T}_{\text{eval}} = \{t_1', \ldots, t_M'\}$，两者来自同一分布但内容不同。每个任务为序列决策问题：在时间步 $i$，Agent 观测 $o_i \in \mathcal{O}$，执行动作 $a_i \in \mathcal{A}$，目标是达成目标 $g$。

ExpeL 的目标：**在不更新 LLM 参数的前提下**，从 $\mathcal{T}_{\text{train}}$ 的经验中提取可复用知识，使 Agent 在 $\mathcal{T}_{\text{eval}}$ 上的单次（one-shot）执行成功率最大化。关键约束是"单次尝试"——不依赖部署时的重试能力。

形式化地，ExpeL 输出两个知识结构：
- 经验池 $\mathcal{B}$：存储训练阶段的成功轨迹集合（用于推理时 kNN 检索）。
- 洞察集合 $\hat{\iota}$：从成败对比中提取的自然语言规则列表（用于推理时 prompt 增强）。

### 问题重要性

现实 Agent 部署场景中"单次成功"极为重要：工具调用有副作用（如执行删除操作），多轮重试可能导致不可逆损失。ExpeL 的研究定位正是在这一约束下最大化性能，同时保持与闭源 API 的兼容性。

---

## 2. 前人工作的方法缺陷

**微调方法（IL, RL with finetune）**：需要大量标注数据和参数访问权限；微调会限制模型泛化能力（Du et al. 2022）；无法在 GPT-4/Claude 等闭源模型上使用。

**纯 prompting / ReAct**：依赖手工设计的固定 few-shot 示例，Agent 无法从自身历史交互中学习；context 窗口限制导致"无记忆"——每个任务从零开始，不能利用之前任务的成功策略。

**Reflexion（自我反思）**：intra-task 改进有效，但反思仅在当前任务的多次重试间积累；跨任务泛化缺失——"我在 Task A 上学到不要去 stoveburner 找水杯"这种洞察不能被 Task B 利用。

这些缺陷形成了 ExpeL 的设计空间：**off-policy 式的经验离线提炼**（类比人类考前复习：练习题 → 归纳规律 → 考试时单次应用）。

---

## 3. 研究动机与提出方案

### 研究动机：学生考试类比

ExpeL 的直接灵感来自 Tom Mitchell 对"机器学习"的经典定义，以及"学生备考"场景：学生在练习题上多次尝试，从中归纳出通用解题洞察；考试时仅凭这些洞察和相似题目的记忆一次性作答。这一认知范式与 Reflexion（只在考试中重试）和微调（大量考题记忆化）都有本质区别。

### 方法本质

ExpeL 的本质是：**将试错经验"蒸馏"为跨任务可复用的 prompt 知识**，分为三阶段：

**阶段一：经验收集（Experience Gathering）**

在每个训练任务 $t_n$ 上，用 Reflexion（ReAct + 自我反思）反复尝试，最多 $Z$ 次。将所有轨迹（成功和失败）存入经验池 $\mathcal{B}$。关键是同时收集**成败对**：对同一任务的成功轨迹 $\tau_n^{\text{success}}$ 和各次失败轨迹 $\tau_{n,z}^{\text{fail}}$，为后续对比分析提供数据基础。

**阶段二：洞察提取（Insight Extraction）**

从 $\mathcal{B}$ 中提取自然语言规则集合 $\hat{\iota}$，通过两种比较方式迭代维护：

1. **成败对比**（Cross-task failure analysis）：输入同一任务的成功/失败轨迹对，LLM 诊断失败原因，提取可避免此类错误的通用规则。
2. **成功模式归纳**（Success pattern extraction）：输入 $L$ 条不同任务的成功轨迹，LLM 归纳共同"最佳实践"。

规则操作为四类算子：**ADD**（新规则）、**EDIT**（修改现有规则）、**UPVOTE**（支持：计数+1）、**DOWNVOTE**（反对：计数-1）。新规则初始计数=2；计数归零则删除。此设计使规则集在迭代中自我修剪，抑制噪声规则积累。

**阶段三：任务推理（Task Inference）**

对评估任务 $t_m$：
- 将全量洞察 $\hat{\iota}$ 拼接到 prompt 的 task specification 部分。
- 通过 FAISS + all-mpnet-base-v2 嵌入器检索经验池中 top-k 成功轨迹（按任务相似度），替换固定 few-shot 示例。
- Agent 执行单次尝试。

### 核心算法流程

```
【训练阶段】
初始化经验池 B = 固定few-shot，洞察集 ι_hat = ∅
for 每个训练任务 t_n:
    for 重试 z = 0 to Z:
        执行 ReAct 轨迹 τ_{n,z}
        B ← B ∪ τ_{n,z}
        if 成功 or z == Z: break
        ν ← LLM_reflect(τ_{n,z})  # Reflexion 反思
for 每对 (成功轨迹, 失败轨迹):
    ι_hat ← LLM_insights(比较成败对, ι_hat)  # ADD/EDIT/VOTE
for 每批 L 条成功轨迹:
    ι_hat ← LLM_insights(成功批次, ι_hat)

【评估阶段】
for 每个评估任务 t_m:
    F_similar ← FAISS 检索(t_m, B, k)  # 基于任务相似度
    执行单次 ReAct，context = [ι_hat] + [F_similar]
```

### 迁移学习能力

ExpeL 支持"知识微调迁移"：将源任务（HotPotQA）提取的洞察 $\hat{\iota}$，通过 GPT-4 以目标任务（FEVER）的少量 few-shot 示例进行"迁移微调"，产生适配目标任务的洞察段落。此过程无参数更新，仅为 prompt 重写。

### 与 Reflexion 的关键差异

| 维度 | Reflexion | ExpeL |
|-----|----------|------|
| 学习范围 | 单任务 intra-task | 跨任务 inter-task |
| 知识形态 | 任务特定的反思文本 | 通用洞察规则列表 |
| 部署时重试 | 依赖（核心机制） | 不依赖（单次推理） |
| 知识迁移 | 不支持 | 支持（prompt 迁移） |
| 经验来源 | 当前任务的失败 | 训练任务集的成败对 |

### 为什么这样设计有效

Figure 5 的消融对比（ExpeL vs. ExpeL retrieve-only vs. ExpeL insights-only）揭示了一个非平凡发现：洞察（insights）和轨迹检索（retrieval）在不同任务上的贡献比例相反——HotPotQA 以 insights 为主（36% vs. 31% retrieve-only），ALFWorld 以 retrieval 为主（55% vs. 50% insights-only）。这说明两个学习模式互为补充：知识密集型推理任务受益于"通用策略规则"，动作密集型任务受益于"具体成功轨迹示范"。两者联合的 ExpeL 在所有任务上均优于单独任何一种。

---

## 4. 实验对比与性能提升

### 数据集

| 数据集 | 任务类型 | 评估指标 |
|-------|---------|---------|
| HotPotQA（Yang et al. 2018） | Wikipedia 多跳 QA | 精确匹配成功率（%） |
| ALFWorld（Shridhar et al. 2021） | 家务环境顺序决策 | 任务完成成功率（%） |
| WebShop（Yao et al. 2022） | 在线购物决策 | 成功率 + 平均奖励（0–1） |
| FEVER（Thorne et al. 2018） | 事实核查（迁移目标域） | 成功率（%） |

所有实验采用四折交叉验证，报告均值和标准误差。

### 基线

- **ReAct**（Yao et al. 2023）：think-act-observe 框架，固定 few-shot，无学习
- **Act**：ReAct 去掉 reasoning 步骤的简化版
- **Reflexion（ReAct+Reflexion）**：intra-task 多轮重试改进

### 关键结果（作者报告数据）

| 数据集 | ReAct | Reflexion R3 | ExpeL（单次） | 提升（vs ReAct） |
|-------|-------|--------------|-------------|----------------|
| HotPotQA | 28.0±1.4% | 39.0% | **39.0±1.7%** | +11% |
| ALFWorld | 40.0±0.3% | 54.4% | **59.0±0.3%** | +19% |
| WebShop（SR） | — | — | 优于 ReAct | — |

**ExpeL 在不依赖重试的前提下，HotPotQA 与 Reflexion R3 持平，ALFWorld 超越 Reflexion R3（59% vs 54.4%）**。

**迁移学习（HotPotQA → FEVER）**：

| 方法 | FEVER 成功率 |
|-----|------------|
| Act | 58 ± 0.0% |
| ReAct | 63 ± 0.4% |
| ExpeL Transfer w/o Task Demos | 65 ± 1.7% |
| **ExpeL Transfer** | **70 ± 0.7%** |

### 消融分析

**经验多样性实验**（Figure 6，HotPotQA）：
- 仅有固定 few-shot 提取洞察 → 与 ReAct 无显著差异
- ReAct 收集经验（无重试）→ 中等提升
- **Reflexion 收集经验（有成败对）→ 最优**

结论：成败对比是洞察提取质量的关键来源；经验多样性（包含失败轨迹）比经验数量更重要。

**洞察提取消融**（Table 3，HotPotQA）：
- 手工洞察：32%（仍优于 ReAct 28%，但低于自动提取 39%）
- 加入反思 $\nu$ 到提取：29%（低于 ReAct！），说明反思信号引入噪声
- GPT-3.5 提取：32%，GPT-4 提取：39%——洞察质量与底层模型能力正相关

**检索策略消融**（Table 3，ALFWorld）：
- 任务相似度检索（ExpeL）：59%
- 推理步相似度检索：48.5%（动态切换 few-shot 引入不稳定）
- 随机采样：42.5%

---

## 5. 方法局限与缺陷

**洞察数量的 context 窗口约束**：当前实现要求全量洞察 $\hat{\iota}$ 塞入 context，随训练任务数增多洞察也会增多，最终将超出 context 限制。作者提出洞察检索作为未来方向，但当前版本未解决。

**文本观测局限**（作者明示）：实验均为文本环境（AlfWorld、HotPotQA、WebShop），不支持图像/多模态观测。现实 Agent 大量需要视觉感知（GUI、机器人）。

**闭源 API 依赖**：GPT-4 被用于洞察提取（$\text{LLM}_{\text{insights}}$），消融显示 GPT-3.5 提取的洞察显著变差（32% vs 39%）——方法性能强绑定于高质量 LLM 的可用性。

**缺乏理论保证**（作者明示）：prompting 技术缺乏 RL 理论框架下的最优性或收敛性证据，策略的学习效率边界未知。

**有效性边界**（分析者判断）：四折交叉验证在 HotPotQA/ALFWorld/WebShop 三个基准上的结果高度互相关——这些均是 ReAct 论文中的原始测试集，ExpeL 对 ReAct 的改进是否能推广到更多样的 Agent 任务（如代码执行、数据库操作）仍有待验证。

---

## 6. 实用复现信息

### 关键超参

- 经验收集最大重试数 $Z$：任务相关
- 成功批次大小 $L$：消融中未明确，默认较小值
- 检索 top-k：默认 k（未明确具体数值，任务相关）
- 洞察提取 LLM：GPT-4（gpt-4-0613）
- 执行 LLM：gpt-3.5-turbo-0613
- 嵌入器：all-mpnet-base-v2
- 向量库：FAISS

### 代码

- 项目主页：https://andrewzh112.github.io/expel
- 含完整轨迹 demo 展示

---

## 7. 研究启发

1. **成败对比作为知识提炼的核心信号**：ExpeL 最重要的非平凡发现之一是：在没有失败轨迹的经验集上提取的洞察与 ReAct 无显著差别（Figure 6）。这提示一个形式化假设：**有效洞察的质量与成败对的多样性（而非总量）成正比**——这可以在不同失败覆盖率下设计受控实验严格验证，形成"经验最小化洞察提取"的理论边界。

2. **洞察老化与实例化问题**：ExpeL 的洞察集是静态的（训练后固定），部署时如果新任务分布偏移，洞察可能失效甚至产生误导。一个新研究方向是"在线洞察更新"：Agent 在部署时遇到新失败后，能否安全地将新洞察 ADD 到现有集合，同时避免知识污染或遗忘？

3. **从 inter-task 到 continual agent learning**：ExpeL 是批量离线训练范式，训练集与评估集明确分离。一个自然扩展是"持续学习 Agent"：Agent 在长期部署中持续从新任务中累积经验，同时维持对旧任务的洞察有效性——这构成"Agent 持续学习"的正式问题定义，与 MACLA 的分层程序性记忆、A-MEM 的动态 Zettelkasten 节点在机制层面存在深刻联系。

---

## 8. 参考文献关系图

```
ExpeL (AAAI 2024)
├── 直接使用/扩展
│   ├── ReAct (Yao et al., 2023) — 基础执行框架（Actor）
│   ├── Reflexion (Shinn et al., 2023) — 经验收集机制（借用自我反思）
│   ├── FAISS (Johnson et al., 2019) — 经验池向量检索
│   └── all-mpnet-base-v2 (Song et al., 2020) — 嵌入器
├── 批判对象
│   ├── IL Finetuning (Lin et al., Shaw et al.) — 需参数访问，成本高
│   └── 纯 prompting (ReAct, Act) — 无跨任务记忆
├── 实验数据集
│   ├── HotPotQA (Yang et al., 2018)
│   ├── ALFWorld (Shridhar et al., 2021)
│   ├── WebShop (Yao et al., 2022)
│   └── FEVER (Thorne et al., 2018)
└── 后续关联
    ├── MACLA (2026) — 层级程序性记忆，同样关注跨任务知识复用
    ├── A-MEM (arXiv:2502.12110) — 动态知识图谱记忆，可对比洞察提取与节点链接的知识表示
    └── MemGPT (Packer et al., 2024) — 分层记忆，可对比持久化设计
```

### 推荐扩展阅读

**论文内提及（方法基础）**：
- *ReAct: Synergizing Reasoning and Acting in Language Models*（Yao et al., 2023）— ExpeL 的执行框架基础
- *Reflexion: Language Agents with Verbal Reinforcement Learning*（Shinn et al., NeurIPS 2023, arXiv:2303.11366）— ExpeL 的经验收集机制来源，两者互为对比参照

**后续扩展（文献检索补充）**：
- *MACLA: Learning Hierarchical Procedural Memory for LLM Agents*（AAMAS 2026）— 层级程序性记忆 + Bayesian 选择，可与 ExpeL 的 insight 列表对比跨任务知识组织方式
- *A-MEM: Agentic Memory for LLM Agents*（arXiv:2502.12110, NeurIPS 2025）— Zettelkasten 式动态知识图谱，从"洞察提取"切换为"知识节点链接"的范式差异值得研究
- *Cognitive Architectures for Language Agents (CoALA)*（Sumers et al., TMLR 2024, arXiv:2309.02427）— 对 LLM Agent 的记忆/行动/决策框架进行系统综述，可为 ExpeL 定位提供元层级分析框架

---

## mem0 记录

```
entity_type: paper
entity_id: expel_2308.10144
summary: >
  ExpeL 提出"经验性学习 LLM Agent"，三阶段：
  (1) 用 Reflexion 在训练任务上收集成败轨迹；
  (2) 用 GPT-4 从成败对中提取/维护通用洞察（ADD/EDIT/UPVOTE/DOWNVOTE 算子）；
  (3) 评估时以全量洞察+top-k 相似成功轨迹单次执行。
  核心优势：跨任务 inter-task 学习，无参数更新，兼容闭源 LLM。
  结果：HotPotQA 39%=Reflexion R3，ALFWorld 59%>Reflexion R3(54.4%)。
  迁移：HotPotQA→FEVER 70%（vs ReAct 63%）。
  关键发现：成败对比（失败轨迹）对洞察质量至关重要；洞察与检索在不同任务上互补。
venue: AAAI 2024
main_tag: agentic_ai
tags: agentic_ai;experiential_learning;agent_knowledge_extraction;in-context_learning;transfer_learning;agent_memory;ReAct
```
