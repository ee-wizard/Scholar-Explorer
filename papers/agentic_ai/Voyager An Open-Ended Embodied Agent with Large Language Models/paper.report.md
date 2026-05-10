---
title: "Voyager: An Open-Ended Embodied Agent with Large Language Models"
paper_id: "arXiv:2305.16291"
analysis_date: "2026-05-06 00:40:00 +08:00"
main_tag: "agentic_ai"
tags:
  - "agentic_ai"
  - "agent_skill"
  - "lifelong_learning"
  - "skill_library"
  - "automatic_curriculum"
  - "embodied_agent"
  - "code_generation"
related_papers: "ReAct, Reflexion, AutoGPT, Generative Agents, CoALA"
---

# Voyager: An Open-Ended Embodied Agent with Large Language Models

> 学术分析报告 | 主标签：agentic_ai | 论文标签：agentic_ai;agent_skill;lifelong_learning;skill_library;automatic_curriculum;embodied_agent;code_generation

---

## 1. 问题定义

**背景与挑战**

在开放世界环境（Minecraft）中实现 **无需人工干预的终身学习（lifelong learning）**：agent 必须在没有预定义任务列表、没有人工课程设计、没有梯度更新的条件下，持续探索环境、习得越来越复杂的技能，并将已有技能泛化到新出现的任务上。

**形式化定义**：

设 Minecraft 世界 $\mathcal{W}$，agent 在 $t$ 时刻拥有状态 $s_t$（背包、位置、生命值等），可执行动作序列 $a_{1:T}$（基于 Mineflayer JS API）。目标是实现：

$$\text{Lifelong Learning} \triangleq \max_{\text{agent}} \mathbb{E}\left[\sum_{t=1}^{\infty} \text{ItemsDiscovered}(s_t)\right] \text{ s.t. no gradient updates, no human intervention}$$

三个子任务互相关联：
1. **探索效率**：在给定 prompting 迭代次数内发现尽可能多的新道具（unique items）
2. **技术树掌握**：按照 Minecraft 资源采集依赖关系（木器→石器→铁器→钻石器）依次解锁更高阶能力
3. **零样本泛化**：在全新世界（不同地图）中，利用先前学习的技能库直接完成未见任务

**挑战的本质**：Minecraft 的任务空间本质上是开放的——任务边界、难度梯度、解锁条件都不是固定的；agent 必须自主发现探索边界并主动学习，同时避免在已掌握技能上浪费资源。与此同时，"技能"需要可复用且可组合，否则越复杂的任务越难从零生成。

---

## 2. 前人工作的方法缺陷

**强化学习方法（VPT, DreamerV3, hierarchical RL）**：依赖像素级输入和梯度更新，无法利用 LLM 的语言先验；技能必须从大量人类视频或环境交互中训练，样本效率极低；抽象层次低（低层动作控制），难以实现高阶技能组合。

**先前 LLM Agent 方法（ReAct, Reflexion, AutoGPT）**：
- **ReAct/Reflexion**：在 Minecraft 探索任务中，ReAct 和 Reflexion 完全无法取得实质进展——两者在最大 prompting 迭代（160 次）内发现道具数接近 0，原因是开放探索目标太抽象，缺乏渐进课程引导，LLM 无法独立决定下一个有意义的子目标；
- **AutoGPT**：能发现道具，但依赖静态人工任务分解思路（非自适应），且无技能记忆——每次任务均从零开始生成代码，导致发现道具数量（约 19）远低于 Voyager（63）且无法解锁钻石级技术树；
- **所有先前方法均无技能库**：每轮任务完成后，生成的代码不被保留，无法在后续任务中复用，使得复杂技能的组合（如"crafting iron sword"调用"mine iron"和"smelt iron"子技能）永远无法实现。

共同瓶颈：**缺乏技能积累机制**（无程序性记忆）+ **缺乏自适应课程**（探索目标随机或静态）+ **缺乏闭环验证**（生成代码是否真正完成任务没有机制确认）。

---

## 3. 研究动机与提出方案

**研究动机**

Voyager 的核心假设是：LLM（GPT-4）已经具备足够强的**代码生成和推理能力**，因此可以绕开梯度更新，通过"生成可执行 JS 代码"的方式直接表达技能，并通过迭代提示反馈闭环验证技能正确性。技能一旦验证通过就写入可检索的**技能库（skill library）**，供未来任务直接调用。这本质上是将"学习"从参数空间搬到了**代码空间**（程序性记忆），使技能获取变成一个无需梯度的、可检索可组合的过程。

**方法本质**

Voyager = 自动课程（自适应任务生成）+ 技能库（程序性记忆）+ 迭代提示机制（代码验证-修正闭环），三者协同实现开放世界终身学习：
- 课程保证学习方向始终处于"合适难度"的探索边界；
- 技能库保证成功经验可复用，使复杂任务分解为已有技能的组合；
- 迭代提示保证每个技能生成过程的正确性。

**核心假设**

1. **可执行代码是技能的最优表示**：JS 代码提供精确语义、可执行验证、可组合调用，比自然语言技能描述更可靠；
2. **自底向上好奇心课程优于固定课程**：在开放世界中，基于当前状态的"下一个可完成的新颖任务"比预设任务表更适应 agent 动态能力边界；
3. **GPT-4 的代码生成质量是关键**：GPT-3.5 替换 GPT-4 后发现道具数量下降 5.7×，说明 GPT-4 的代码能力是整个方法的底层必要条件。

**核心创新点**

1. **自动课程（Automatic Curriculum）**：GPT-4 根据当前游戏状态（背包、位置、生命值、已解锁技能、已知事实）生成下一个探索目标，以"难度适中"和"能推动探索边界"为双重标准，自底向上，课程无需人工设计；
2. **可执行技能库（Executable Skill Library）**：成功验证的 JS 函数被保存到技能库，同时用 GPT-3.5 对函数生成自然语言描述，描述文本用 text-embedding-ada-002 编码存储；检索时计算查询 embedding 与所有技能描述 embedding 的余弦相似度，返回 top-5 技能代码作为新任务的上下文参考；技能库是分层的（复杂技能可调用简单技能作为子过程）；
3. **迭代提示机制（Iterative Prompting Mechanism）**：每轮代码生成后，agent 执行代码并收集三类反馈信号：(a) 环境反馈（执行后的游戏状态变化）；(b) 执行错误（代码报错信息）；(c) 自我验证（GPT-4 Critic 判断任务是否真正完成）；三类反馈拼接回 prompt，LLM 修改代码，最多迭代若干次直至验证通过。

**方法流程**

```
初始化：空技能库 L, 初始游戏状态 s_0

决策周期 t:
1. [自动课程] GPT-4(s_t, L, 已知事实) → 新任务目标 g_t
2. [技能检索] query=g_t → embed → top-5 skills from L
3. [代码生成] GPT-4(g_t, top-5 skills, s_t) → JS code c_0
4. [迭代提示循环]:
   while not verified:
       执行 c_i → (env_feedback, exec_errors)
       GPT-4 Critic(g_t, s_after) → 完成/失败+原因
       if 完成: break
       GPT-4(c_i, env_feedback, exec_errors, critic_feedback) → c_{i+1}
5. [技能学习] if 成功验证:
       GPT-3.5(c_i) → 自然语言描述 d
       L ← L ∪ {(c_i, embed(d))}
6. s_{t+1} ← 执行后状态
```

**模块职责**

| 模块 | 输入 | 输出 | 底层模型 |
|------|------|------|---------|
| 自动课程 | 当前状态 + 已知技能 + 事实 | 新任务目标（自然语言） | GPT-4 |
| 技能检索 | 任务描述 embedding | top-5 技能代码 | text-embedding-ada-002 + 余弦相似度 |
| 代码生成 | 任务 + 技能上下文 + 状态 | JS 可执行技能代码 | GPT-4 |
| 环境执行 | JS 代码 | 状态变化 + 执行错误 | Mineflayer API |
| 自我验证（Critic） | 任务目标 + 执行后状态 | 完成/失败 + 原因 | GPT-4 |
| 描述生成 | JS 代码 | 自然语言描述 | GPT-3.5-turbo |

**机制有效性解释与证据锚点**

- **Figure 1（探索曲线）**：Voyager 在 160 次 prompting 迭代内发现 63 种道具，而 AutoGPT 约 19 种，ReAct/Reflexion 接近 0——差距来自技能库积累效应：后期任务建立在前期技能之上，使探索边界持续推进；
- **Table 1（技术树）**：Voyager 解锁木器快 15.3×（6±2 vs AutoGPT 92±72 次迭代），且唯一解锁钻石级（1/3 试验）；无技能库变体无法解锁钻石，确认技能积累对高阶任务的必要性；
- **消融（Figure 9 左）**：移除自动课程（改为随机课程）后发现道具数下降 93%——随机课程生成的任务顺序错误会导致连续失败；移除技能库后 agent 后期出现性能平台（plateau），说明技能复用在 agent 发展后期尤为关键；
- **消融（Figure 9 右）**：自我验证（Critic）最重要，移除后发现道具数下降 73%（-33 道具）——Critic 是判断"何时任务真正完成、何时需要重试"的关键机制；GPT-4 vs GPT-3.5 代码生成：GPT-4 获得 5.7× 更多道具；
- **Table 2（零样本泛化）**：所有基线在 50 次迭代内无法完成任何 4 项测试任务，Voyager 全部成功（Diamond Pickaxe 19±3 次，Golden Sword 18±7 次等）；将 Voyager 技能库给 AutoGPT 使用，AutoGPT 也能完成部分任务（3/4 种），证明技能库具备跨 agent 迁移价值。

---

## 4. 实验对比与性能提升

**数据集/评测环境**：Minecraft（通过 MineDojo + Mineflayer JS API 进行高层控制），开放世界设置，不固定地图种子

**评估指标**：
- 探索：160 次 prompting 迭代内发现的 unique items 数量
- 技术树：解锁各级别所需 prompting 迭代数（越少越好），成功率（3 次试验）
- 地图覆盖：agent 从出生点出发的最大移动距离（map coverage）
- 零样本泛化：在新世界中完成 4 项指定未见任务（Diamond Pickaxe, Golden Sword, Lava Bucket, Compass）所需迭代数和成功率

**基线**：
| 基线 | 类型 | 特点 |
|------|------|------|
| ReAct | LLM Agent | 推理+动作循环，无技能库 |
| Reflexion | LLM Agent | ReAct + 自我反思，无技能库 |
| AutoGPT | LLM Agent | 任务分解自动化，无技能库 |
| Voyager w/o Skill Library | 消融 | 移除技能库，其他不变 |

**主要结果（author-reported）**：

| 指标 | ReAct | Reflexion | AutoGPT | **Voyager** |
|------|-------|-----------|---------|-------------|
| 探索道具数 | ~0 | ~0 | ~19 | **63** (3.3×) |
| 木器解锁（迭代数） | 失败 | 失败 | 92±72 | **6±2** (15.3×) |
| 石器解锁 | 失败 | 失败 | 94±72 | **11±2** (8.5×) |
| 铁器解锁 | 失败 | 失败 | 135±103 | **21±7** (6.4×) |
| 钻石解锁（成功率） | 0/3 | 0/3 | 0/3 | **1/3** (唯一) |
| 地图覆盖距离 | — | — | 基准 | **2.3×** |
| 零样本任务（4任务成功率） | 0/4 | 0/4 | 0/4 | **4/4** |

**消融关键结论**：
- 自动课程：移除后道具数下降 93%（最大单一影响）
- 技能库：移除后中后期性能平台，无法解锁钻石
- Critic（自我验证）：最重要的反馈类型，移除后 -73% 道具
- GPT-4 vs GPT-3.5：GPT-4 比 GPT-3.5 多 5.7× 道具

---

## 5. 方法局限与缺陷

**作者明示局限**：
- **成本高昂**：GPT-4 API 比 GPT-3.5 贵 15×，大规模长时间运行成本显著；
- **幻觉问题**：自动课程偶尔生成 Minecraft 中不存在的物品（如"copper sword"）；代码生成时 GPT-4 会调用不存在的 API 函数或使用无效材料（如 cobblestone 作为燃料）；
- **无视觉感知**：Voyager 仅使用文本状态描述（背包、位置等），不能感知视觉信息；3D 空间建造任务需要人工反馈才能完成（Figure 10）。

**分析者推断的局限**：
- **技能库的语义漂移**：随着技能库规模增长，不同技能之间可能产生语义重叠或冲突，top-5 检索未必总能返回最相关技能；当技能库超过百个函数时，检索噪声如何影响代码生成质量未被评测；
- **任务域的封闭性**：实验完全在 Minecraft 内完成，迁移到其他数字环境（网页操作、API 调用、机器人控制）的可行性仅被定性讨论，未经验证；技能库的代码 Minecraft-specific（调用 Mineflayer API），无法直接迁移；
- **零样本泛化评测局限**：4 项测试任务相对简单（Diamond Pickaxe, Golden Sword 等），均在技能库已有基础技能的合理延伸范围内，"真正未见场景"（如完全不同的游戏机制）的泛化能力未被检验；
- **评测规模**：每个实验仅 3 次独立试验（variance 较大），例如钻石解锁 1/3 的结论在统计上说服力有限。

---

## 6. 科研启发

1. **技能库的有效组织与遗忘机制**：当前 Voyager 技能库只增不减（append-only），不存在技能淘汰或更新机制。一个有新问题定义的研究方向是：**技能库的持续性能跟踪与主动遗忘**——当某技能在多次尝试中失败率超过阈值，或被更新技能完全覆盖时，如何安全移除或合并？可形式化为"技能效用评估 + 图依赖分析下的安全删除"问题，对应 CoALA §6 提出的"unlearning"研究空白。

2. **可验证的技能生成（Formally-Verified Skill Generation）**：Voyager 的 self-verification 依赖 LLM（GPT-4 Critic）判断任务完成，这本身容易产生幻觉（论文中已报告 spider string 未被正确识别）。一个可实验验证的假设是：引入基于**游戏状态形式化规约（precondition + postcondition）**的技能验证，替代或辅助 LLM Critic，能否在保持技能质量的同时减少错误技能写入技能库的概率？这是程序验证与 LLM agent 的交叉研究问题。

3. **技能检索策略的优化**：当前技能检索使用固定 top-5 余弦相似度，未考虑技能依赖关系（如新任务需要技能 A 和 B，而 B 依赖 A）。一个新的问题定义是：**依赖感知的技能检索（dependency-aware skill retrieval）**——在检索时显式建模技能的调用依赖图，返回一个"自洽的技能子集"而非 top-5 个独立技能，从而减少代码生成中的 API 缺失错误。这与知识图谱 RAG（如 HippoRAG）的图传播思路有潜在结合点。

4. **终身学习中的课程与遗忘的权衡**：Voyager 的课程生成假设 agent 状态空间单调扩大，但在真实场景中 agent 可能因失误导致资源损失（如道具丢失），使状态空间出现退化。研究问题是：当 agent 状态发生负向变化时，自动课程应如何自适应地回退到更简单目标，同时保持对已掌握技能的正向记忆？这需要定义"课程韧性"的形式化度量，是终身学习中已有 catastrophic forgetting 研究的开放世界扩展。

---

## 7. 参考文献图谱

### 文献分类表

| 文献名 | 作者/年份 | 角色 | 知识库状态 |
|--------|----------|------|-----------|
| ReAct: Synergizing Reasoning and Acting in Language Models | Yao et al., 2022 | 实验基线 | 未收录 |
| Reflexion: Language Agents with Verbal Reinforcement Learning | Shinn et al., 2023 | 实验基线 | 已收录（#58） |
| AutoGPT | Significant Gravitas, 2023 | 实验基线 | 未收录 |
| Generative Agents: Interactive Simulacra of Human Behavior | Park et al., 2023 | 被对比文献（代理设计参考） | 未收录 |
| MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge | Fan et al., 2022 | 方法基础（环境/数据集） | 未收录 |
| Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos | Baker et al., 2022 | 被批判文献（梯度式先验方法） | 未收录 |
| Cognitive Architectures for Language Agents (CoALA) | Sumers et al., 2023 | 框架参考（程序性记忆分析） | 已收录（#62） |

---

## 推荐阅读列表

### P0 必读（方法基础，知识库未收录）
- ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022) — Voyager 最直接的基线，理解 ReAct 的失败原因是理解 Voyager 贡献的前提
- Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023) — 同期最完整的 agent 记忆设计，与 Voyager 技能库（程序性记忆）互补的情景+语义记忆实现

### P1 重要（被批判文献）
- Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos (Baker et al., 2022) — Voyager 明确说明不对比此类低层控制方法，理解设计边界的必要参考
- MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge (Fan et al., 2022) — Voyager 的实验环境与任务空间基础

### P2 建议（相关方法，扩展理解）
- DEPS: Describe, Explain, Plan and Select (Wang et al., 2023) — Minecraft 中另一类 LLM 规划方法，与 Voyager 自动课程的对比
- Steve-1: A Generative Model for Text-to-Behavior in Minecraft (Lifshitz et al., 2023) — 基于条件视频模型的 Minecraft agent

### P3 参考（背景综述，选读）
- Cognitive Architectures for Language Agents (Sumers et al., 2023) — 已收录（#62），Voyager 是其程序性记忆学习的最佳实证案例

---

## mem0 知识库记录

- **问题域**：开放世界终身学习，程序性技能积累，embodied agent，LLM-driven code generation
- **方法关键词**：automatic curriculum (novelty-driven, bottom-up), skill library (executable JS code + embedding index), iterative prompting (env feedback + execution errors + self-verification Critic), GPT-4 code generation
- **数据集**：Minecraft（MineDojo + Mineflayer API）
- **性能基准**：63 unique items (3.3× vs AutoGPT); 木器解锁快 15.3×; 唯一解锁钻石级 (1/3); 零样本 4/4 任务 vs 基线 0/4
- **关联论文 ID**：arXiv:2303.11366 (Reflexion/#58), arXiv:2309.02427 (CoALA/#62)
- **核心方法机制摘要**：三组件：(1) GPT-4 自动课程（状态感知、自底向上、新颖性驱动）；(2) JS 技能库（text-embedding-ada-002 索引，top-5 检索，分层可组合）；(3) 迭代提示机制（环境反馈+执行错误+GPT-4 Critic 自我验证闭环）。无梯度更新，纯 prompting 实现终身学习。
- **推荐下一轮阅读线索**：ReAct（基线理解）、Generative Agents（互补记忆设计）、依赖感知技能检索的后续工作
