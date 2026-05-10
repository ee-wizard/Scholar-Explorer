---
title: "MemGPT: Towards LLMs as Operating Systems"
paper_id: "arXiv:2310.08560"
analysis_date: "2026-05-06 11:00:00 +08:00"
main_tag: "agentic_ai"
tags:
  - "agentic_ai"
  - "agent_memory"
  - "operating_system"
  - "tiered_memory"
  - "virtual_context_management"
  - "LLM_agents"
related_papers: "A-MEM:arXiv:2502.12110; MACLA:AAMAS2026; ChatDB:2023"
---

# MemGPT: Towards LLMs as Operating Systems

> 学术分析报告 | 主标签：agentic_ai | 论文标签：agentic_ai;agent_memory;operating_system;tiered_memory;virtual_context_management;LLM_agents

---

## 1. 问题定义

**问题背景**：LLM 的固定上下文窗口（4K–128K tokens）构成 Agent 长期记忆的根本障碍。对话、文档、推理过程不能完整地同时放入上下文，必须在某种截断或压缩策略下运行。问题的本质是：如何在有限 RAM（上下文窗口）和无限外存（向量数据库、消息历史）之间高效管理 Agent 的"运行内存"，使 LLM 在保持推理连贯性的同时访问任意长度的历史信息。

**关键挑战**：

1. **上下文容量上限硬约束**：超过上下文窗口的内容会被截断，在对话轮次多或文档集大的场景下，重要信息不可避免地丢失；
2. **用户不可见的内存调度**：现有方案（如固定摘要、滑动窗口）在设计时确定内存调度策略，LLM 无法根据当前推理需求主动决策调用哪段历史；
3. **外存访问语义缺失**：外部数据库（向量库、消息历史）提供了存储，但 LLM 无法以结构化的函数调用方式自主读写，只能依赖外部框架注入内容。

**操作性形式化描述**：

设 LLM 上下文窗口大小为 $C$（tokens）。主上下文（Main Context）$M$ 是当前被 LLM 处理的 token 序列，$|M| \leq C$。外部上下文（External Context）$E$ 是规模不受限的存储池。目标是设计一个控制器 $\pi$，使得：

$$a_t = \text{LLM}(M_t), \quad M_{t+1} = \pi(M_t, E, a_t, q_t)$$

其中 $q_t$ 是当前查询/事件，$\pi$ 是内存调度策略，$a_t$ 是 LLM 的下一个动作（包括正常回复或内存操作指令）。关键约束是：$\pi$ 由 LLM 自身发起（通过函数调用），而不是由外部规则触发——这是与现有方案的根本差异。

**问题重要性**：上下文窗口瓶颈是影响 Agent 长期任务能力的关键系统问题。MemGPT 在 2023 年提出了第一个将 OS 内存管理哲学引入 LLM 上下文管理的完整框架，其 OS 类比成为后续大量 Agent 记忆工作的参考框架。

---

## 2. 前人工作的方法缺陷

**共同瓶颈**：现有方案无一例外是"被动截断"范式——当上下文满时由外部规则决定丢弃什么，LLM 对此不知情也无法干预：

**固定摘要（Static Summarization）**：每隔 K 轮生成摘要替换原始对话。问题是摘要时机固定，无法响应突发的跨段推理需求；摘要压缩不可逆，细节信息永久丢失；多轮摘要会导致信息累积衰减。ChatGPT 和早期对话 Agent 均使用此方案。

**滑动窗口（Sliding Window）**：保留最近 K 条消息，丢弃更早的历史。问题是只有时序相关性，完全无法处理"用户两周前提到的偏好"这类长程依赖；窗口大小的选择是超参数，无法自适应任务需求。

**全上下文注入（Full Context Injection，如 RAG）**：将文档检索结果完整注入上下文。问题是文档规模超过窗口时仍然失效；多文档场景下注入的内容本身超出上下文；且与对话历史同时存在时会进一步压缩有效窗口。

**ReadAgent**（被批判的直接前驱）：将长文档分页，对每页生成 gist（摘要），用分层摘要替代原始内容，支持按需重载。改进在于分层组织，但仍然依赖预定义的分页规则，LLM 无法主动触发细节重载；且 gist 生成在推理开始前完成，不支持动态按需调页。

**关键差异**：MemGPT 的核心突破是将**内存调度决策的主体**从外部规则迁移到 LLM 本身——LLM 通过函数调用主动发起"从外存取数据"或"将内容写入外存"，这与 OS 中进程发起 syscall 请求内存页的机制完全类似。

---

## 3. 研究动机与提出方案

### 研究动机

本文的核心类比是：**LLM 上下文窗口 ≈ 操作系统物理 RAM，外部存储 ≈ 磁盘，内存调度 ≈ 操作系统虚拟内存管理**。但与 OS 不同的是，LLM 的"RAM"不仅是数据，也是计算的输入——在上下文中的内容才能被 LLM 推理使用，上下文外的内容需要先调入。

这一类比直接引出设计目标：(1) 像 OS 一样维护分层的内存层次（Fast-access: in-context, Slow-access: external）；(2) 像进程发 syscall 一样让 LLM 通过函数调用主动控制内存调度；(3) 像虚拟内存一样让 LLM 处理"无限长"的逻辑对话，而不受物理上下文窗口的限制。

### 方法本质

MemGPT 是一个**以 LLM 为核心控制器、通过函数调用（function calling）接口实现自主内存分页的虚拟上下文管理框架**。其核心机制是：LLM 接收来自各类事件的输入，处理时通过内置的内存函数（读取外存、写入外存、更新工作内存等）主动调度内存内容，所有记忆操作都由 LLM 自身发起，而非外部规则注入。

去包装后的真实机制：MemGPT 在 LLM 的 system prompt 层植入"内存管理工具箱"——一组描述内存状态和操作的 function schema。当 LLM 响应时，除了回复用户外，还可以生成函数调用（如 `archival_memory_search`, `conversation_search`, `core_memory_append`），框架执行这些调用并将结果作为新消息注入上下文。通过 `request_heartbeat=true` 标志，LLM 可以在一次用户交互中连续调用多个内存函数，再回复用户，从而实现多步内存操作。

### 主上下文结构

Main Context（即 LLM prompt）由三部分组成，总大小受 $C$ 约束：

$$M = \underbrace{I}_{\text{System Instr.}} \oplus \underbrace{W}_{\text{Working Context}} \oplus \underbrace{Q}_{\text{FIFO Queue}}$$

- **System Instructions $I$**（只读）：系统 prompt，包含 Agent 角色、内存工具函数描述、内存状态描述，Agent 的身份和能力由此定义；
- **Working Context $W$**（读写，固定大小）：Agent 的"工作台"，存放当前任务相关的临时信息（如用户偏好、当前任务摘要），LLM 可通过 `core_memory_append/replace` 函数主动修改；
- **FIFO Queue $Q$**（读写，滚动，固定大小）：最近消息历史（用户消息、Agent 回复、函数调用/结果），按时间顺序排列，满时触发队列管理器的溢出处理。

### 外部上下文结构

$$E = \underbrace{A}_{\text{Archival Storage}} \cup \underbrace{R}_{\text{Recall Storage}}$$

- **Archival Storage $A$**：任意长度对象（文档、知识条目），向量化存储，支持语义搜索（pgvector + HNSW）；
- **Recall Storage $R$**：完整消息历史数据库，支持关键词全文搜索和语义查询。

### 内存事件流与触发机制

**队列溢出处理**（由队列管理器 Queue Manager 自动触发，非 LLM 主动）：

- 警告触发：$|Q| \geq 70\% \cdot C_Q$，向 LLM 注入内存压力警告，提示应总结保存重要信息；
- 溢出处理：$|Q| = C_Q$ 时，弹出 50% 最旧消息，生成递归摘要并注入，替换被弹出的内容。

**LLM 主动调度**（LLM 发起，通过 function calling）：

$$\text{search}: q \to \text{pgvector.search}(q) \to \text{results injected to } Q$$

$$\text{write}: m \to \text{archival.insert}(m), \text{recall.insert}(m)$$

$$\text{modify\_working\_context}: \delta W \to W' = f(W, \delta W)$$

**函数链机制**（request_heartbeat）：LLM 返回带有 `request_heartbeat=true` 的函数调用时，框架执行函数并立即触发新一轮 LLM 推理（不等待用户输入），允许 LLM 在一次用户交互内顺序执行多步内存操作。

**事件驱动控制流**：除用户消息外，MemGPT 还接收系统警告（内存压力）、定时事件（heartbeat）、外部触发（tool completion），形成完整的事件响应模型。

### 关键图表解释

**Figure 2**（内存层次图）：清晰对应 OS 层次——Main Context 对应 RAM（L1/L2 cache + physical RAM），External Storage 对应磁盘。Working Context 类比为寄存器/缓存（频繁读写，小而快），FIFO Queue 类比为 RAM（较大，访问稍慢），Archival/Recall Storage 类比为 SSD/HDD（大而慢，需调页）。

**Figure 5**（嵌套 KV 检索）：展示 MemGPT 在 4 层嵌套 KV 任务中成功（GPT-4 在 3 层失败），直观验证了多步主动内存检索的能力上界。正确率曲线显示：MemGPT 在 3-4 层时稳定在 80%+，GPT-4 基线在 3 层跌至 0%。

### 机制有效性

实验通过三类独立任务验证不同机制：
- **Deep Memory Retrieval (DMR)**：验证跨会话长期记忆检索（GPT-4+MemGPT 92.5% vs GPT-4 32.1%），主要测试 Archival Storage 的长期检索机制；
- **Nested KV Retrieval**：验证多步主动内存调度（函数链机制）的深度能力；
- **Document QA**：验证 Archival Storage 在大文档集（$>$ 上下文窗口）下的稳定性，MemGPT 性能几乎不随文档数增加而下降，而基线在文档数超窗口后迅速下降。

---

## 4. 实验对比与性能提升

### 数据集

- **Multi-Session Chat**（Xu et al., 2022）：跨会话多轮对话，包含持久用户信息，测试 Deep Memory Retrieval（DMR）；
- **Nested KV**（自构，论文中描述）：$k$ 层嵌套键值对，值仅在展开所有层后可知，测试多步内存调度深度；
- **Document QA**（多文档问答，引用 SCROLLS benchmark 中的数据）：多文档集（3–32 个文档），测试超上下文场景下的 QA。

### 基线

| 基线 | 类型 |
|------|------|
| GPT-4 (zero-shot, full context) | 固定上下文基线 |
| GPT-3.5 (zero-shot, full context) | 固定上下文基线 |
| ReadAgent (gist memory) | 层次摘要基线 |
| RecurSum (recursive summarization) | 递归摘要基线 |
| Human Annotators | 人类基准（对话类） |

### 关键结果

**Deep Memory Retrieval（表 1，作者报告）**：

| 方法 | Accuracy |
|------|---------|
| GPT-3.5 (zero-shot) | 38.7% |
| GPT-3.5 + MemGPT | 66.9% |
| GPT-4 (zero-shot) | 32.1% |
| GPT-4 + MemGPT | **92.5%** |
| Human | ~100% |

GPT-4+MemGPT 相对 GPT-4 零样本基线提升 +60.4 points（作者报告），说明 MemGPT 的跨会话记忆检索能力质的突破。值得注意的是 GPT-4 零样本（32.1%）低于 GPT-3.5 零样本（38.7%），作者未在论文中分析这一异常。

**Document QA（Figure 4，作者报告）**：MemGPT 在 3 到 32 个文档的全范围内保持稳定性能（约 80%），而固定上下文基线在文档数超过上下文容量后（约 8 个文档）急剧下降。RecurSum 表现类似 MemGPT，但在 32 文档时性能略低。

**Nested KV（Figure 5，作者报告）**：MemGPT 在 1-4 层均保持高准确率（约 80-95%），GPT-4 在 3 层时跌至 0%。这是最强的功能性演示，证明了函数链机制的多步记忆调度能力。

**对话开场（Figure 3）**：MemGPT 生成的对话开场（"上次你提到..."类型的个性化开场）的用户参与度，使用人工评分，接近/超过人类基准（作者报告 MemGPT≈Human），优于未使用记忆的基线（低于 Human）。

### 弱改进场景

- **计算开销**：每次内存函数调用都消耗额外 LLM tokens，在对话密集型任务中函数调用频率高时，实际延迟和 API 成本会显著增加。论文未报告实际延迟或 cost per query。
- **长文档检索精度上限**：MemGPT 依赖 pgvector+HNSW 的近似最近邻搜索，在高密度事实型文档（如法律文书，多个条款语义相似）场景下，ANN 精度不足可能导致召回失败。这在论文的 Document QA 实验中未被分析（仅报告准确率，未分析召回率）。

---

## 5. 方法局限与缺陷

**作者自述局限**（§7）：(1) 在 LLM 无函数调用能力（如纯 chat-only API）时框架不适用；(2) 工作内存（Working Context）的大小需要手动配置，不同任务的适宜大小不同；(3) 递归摘要的信息损失不可避免，只是被延迟。

**分析者推断的局限**：

**队列溢出的递归摘要链问题**：随着对话时长增加，早期对话的摘要会成为更早期对话的摘要的摘要，形成"摘要的摘要链"，信息压缩的累积损耗可能随轮次增长而非线性放大。MemGPT 未报告在超长对话（100+ 轮）下的性能曲线，这是一个未验证的性能边界。

**LLM 自主调度质量的不可控性**：当 LLM 误判为不需要检索历史（即不生成 `archival_memory_search` 调用）时，系统无法自动纠正，即使用户的问题明显需要历史信息。这种"无声失败"比截断失败更难检测，因为 LLM 仍然会生成一个看起来合理的回复（用幻觉填充记忆缺失）。

**函数调用格式的模型依赖性**：MemGPT 的整个内存调度机制依赖 LLM 可靠地生成结构化函数调用。2023 年时，仅 GPT-4 和 GPT-3.5-turbo 具有原生函数调用能力，其他开源模型需要大量 prompt engineering 才能可靠生成结构化输出。这限制了方法的通用性。

**与 A-MEM 的关键对比**：A-MEM 被本文指出的"预定义操作时机"问题所批判——MemGPT 的内存触发时机（70% 警告、100% 溢出）虽然由 LLM 感知，但触发阈值本身是硬编码的。A-MEM 将这一问题彻底解决：任意新交互都触发记忆更新，且链接建立完全由语义判断，没有预定义阈值。

---

## 6. 科研启发

1. **自适应上下文窗口分配问题**：MemGPT 中 Working Context $W$ 的大小是固定的，但不同任务阶段需要不同大小的"工作台"（复杂多文档任务需要更大的 $W$，简单对话需要更小）。研究可以形式化一个在线优化问题：在总上下文约束 $C$ 下，动态分配 $|I|, |W|, |Q|$ 三部分的大小以最大化下游任务准确率。理论假设是：存在针对不同任务类型的最优上下文分配比例，可以通过强化学习或自适应启发式方法在线估计。

2. **虚拟上下文与结构化记忆的混合架构**：MemGPT 的外存（pgvector）和 A-MEM 的记忆图（链接网络）代表了两种外存组织路径：前者是无结构向量库，后者是显式语义图。一个未被探索的问题是：对于不同类型的 Agent 任务，哪种外存结构更有利于 LLM 的主动内存调度？是否可以设计一个混合外存架构，为向量相似性和显式链接路径分别保留最优场景？

3. **内存操作的可解释性与审计问题**：MemGPT 的 LLM 自主决定什么时候调用 `archival_memory_search` 和 `core_memory_replace`，这些决策对外部观察者（用户、监管者）是不透明的。当 Agent 在敏感场景（如医疗助手、法律顾问）下运行时，记忆调度决策的可审计性至关重要。定义"内存操作审计日志"的形式化标准，并研究如何在不影响推理性能的前提下提供可解释的内存调度记录，是一个具有实际价值的新问题。

4. **压缩率与信息保真度的显式权衡**：MemGPT 的递归摘要在溢出时丢弃部分内容，但信息保真度从未被显式优化——摘要质量完全取决于 LLM 的摘要能力，没有可量化的信息保留指标。研究信息论框架下的最优上下文压缩问题（在固定压缩率下最大化关键信息的保留概率）构成了一个清晰的理论问题，可用于评估和改进递归摘要策略。

---

## 7. 参考文献图谱

### 文献分类表

| 文献名 | 作者/年份 | 角色 | 知识库状态 |
|--------|----------|------|-----------|
| ReadAgent: A Human-Inspired Reading Agent with Gist Memory | Lee et al., 2024 | 被批判文献 + 基线 | 未收录 |
| RecurSum: Recursively Summarizing Books | Wu et al., 2021 | 实验基线 | 未收录 |
| ChatDB: Augmenting LLMs with Databases as Their Symbolic Memory | Hu et al., 2023 | 相关工作（结构化外存） | 未收录 |
| Multi-Session Chat (MSC) dataset | Xu et al., 2022 | 数据集来源 | 未收录 |
| pgvector: PostgreSQL vector data type | PostgreSQL extension | 工具依赖（向量检索） | 不适用 |
| Efficient and Robust Approximate Nearest Neighbor Search Using HNSW | Malkov & Yashunin, 2018 | 工具依赖（检索索引） | 已收录（multi_vector_retrieval） |
| A-MEM: Agentic Memory for LLM Agents | Xu et al., 2025 | 后续相关工作（批判 MemGPT） | 已收录（本批次）|
| MACLA: Hierarchical Procedural Memory | Forouzandeh et al., 2026 | 后续相关工作（过程记忆） | 已收录（本批次）|
| Self-RAG | Asai et al., 2023 | 背景综述（RAG 路线） | 已收录（序号54）|
| MemoryBank: Enhancing LLMs with Long-term Memory | Zhong et al., 2024 | 同类记忆方法 | 未收录 |
| AIOS: LLM Agent Operating System | Mei et al., 2024 | 同类 OS 类比工作 | 未收录 |

---

## 推荐阅读列表

### P0 必读（被批判文献，理解动机必读）
- **ReadAgent: A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts** (Lee et al., 2024) — MemGPT 中被直接对比的分层摘要方法，理解其"预定义分页"的局限是理解 MemGPT 动机的前提，且同时是 A-MEM 论文的基线方法。
- **ChatDB: Augmenting LLMs with Databases as Their Symbolic Memory** (Hu et al., 2023) — 将结构化数据库作为 LLM 符号记忆，与 MemGPT 的向量外存路线形成对比，代表了外存组织的另一条路径。

### P1 重要
- **AIOS: LLM Agent Operating System** (Mei et al., 2024, arXiv) — 与 MemGPT 的 OS 类比一脉相承，将整个 LLM 运行时设计为操作系统，是 MemGPT 思路的系统化扩展。
- **MemoryBank: Enhancing Large Language Models with Long-term Memory** (Zhong et al., 2024, AAAI CCF-A) — 基于遗忘曲线的记忆系统，代表了 MemGPT 之后在记忆组织上的一次系统性改进，可对比 MemGPT 的差异和进展。

### P2 建议
- **Multi-Session Chat** (Xu et al., 2022) — DMR 任务的数据集来源，理解数据集设计有助于评估 MemGPT DMR 实验的外部有效性。

---

## mem0 知识库记录

- **问题域**：LLM Agent 长期记忆；虚拟上下文管理；OS 类比记忆；上下文窗口管理
- **方法关键词**：Virtual Context Management；Main Context；External Context；FIFO Queue；Archival Storage；Recall Storage；Function Chaining；request_heartbeat
- **数据集**：Multi-Session Chat（DMR）；Nested KV（自构，4 层）；Document QA（SCROLLS 衍生，3-32 文档）
- **性能基准**：DMR：GPT-4+MemGPT 92.5%（vs GPT-4 32.1%）；Nested KV 4 层保持 80%+（GPT-4 在 3 层失效）；Document QA 在 3-32 文档范围内性能平稳
- **关联论文 ID**：A-MEM（arXiv:2502.12110）；HNSW（multi_vector_retrieval）；Self-RAG（序号54）
- **核心方法机制摘要**：OS 类比分层记忆：主上下文（系统指令 + 工作内存 + FIFO 队列）+ 外部上下文（档案存储 + 召回存储）。LLM 通过函数调用主动发起内存读写，函数链（request_heartbeat）允许多步连续内存操作。队列溢出触发递归摘要压缩。
- **推荐下一轮阅读线索**：ReadAgent（被批判基线）；AIOS（OS 类比系统化扩展）；MemoryBank（AAAI 2024，记忆系统演进）；ChatDB（符号记忆路线）
