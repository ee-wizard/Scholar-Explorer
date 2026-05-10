# 大规模 Agent Skill 检索 — 近两年文献全景（2024–2026）

> 生成时间：2026-05-07  
> 检索关键词：agent skill retrieval / skill library / skill routing / skill bank / tool retrieval agent  
> 数据来源：arXiv（135 篇候选，筛选后保留 12 篇强相关）  
> 本库已分析基础：SRA (2604.24594)、SkillsBench (2602.12670)、Multi-field Tool Retrieval (2602.05366)、Voyager (2305.16291)

---

## 一、技术路线演进

| 时期 | 主流方向 | 特征 |
|---|---|---|
| 2024 年 | 扁平向量技能库（Voyager 式）、工具 API 选择（multi-field） | 技能存储为代码片段，text-embedding 索引，top-k 检索 |
| 2025 年初 | 评测基准崛起（SRA-Bench、SkillsBench） | 拆解三阶段（检索→加载→执行），暴露 loading 为主要瓶颈 |
| 2026 年初 | 爆发式多元化：图结构检索、自动化挖掘、协同进化 | 从"能检索"转向"检索质量"与"结构化组合" |

---

## 二、核心论文分类汇总

### 2.1 检索架构（直接研究"大规模 Skill 检索"）

#### Graph of Skills: Dependency-Aware Structural Retrieval for Massive Agent Skills
- **arXiv**: [2604.05333](https://arxiv.org/abs/2604.05333)  
- **时间**: 2026-04  
- **核心贡献**: 直接面向海量 Skill 的依赖感知结构化检索。引入技能依赖图，检索时利用技能间的前提-结果边进行图上传播，而非孤立 top-k 向量匹配。  
- **项目**: https://github.com/davidliuk/graph-of-skills  
- **关联**: 与 GraSP 形成互补——GoS 聚焦检索阶段，GraSP 聚焦执行阶段。

#### SkillRouter: Skill Routing for LLM Agents at Scale
- **arXiv**: [2603.22455](https://arxiv.org/abs/2603.22455)  
- **时间**: 2026-03（v2: 2026-04-01）  
- **核心贡献**: 大规模 Skill 路由框架，解决随 Skill 数量增长时检索退化（context pollution）问题；提出 delayed appraisal + epistemic vigilance 机制控制路由误判。

#### GraSP: Graph-Structured Skill Compositions for LLM Agents
- **arXiv**: [2604.17870](https://arxiv.org/abs/2604.17870)  
- **时间**: 2026-04  
- **核心贡献**: 在技能检索与执行之间引入 DAG 编译层。将扁平技能集转为带前提-效果边的有向无环图，节点级验证，五类算子局部修复，将重规划从 O(N) 降为 O(d^h)。ALFWorld/ScienceWorld/WebShop/InterCode 上对比 ReAct/Reflexion/ExpeL 全赢，最大 +19 points reward，环境步数减少最多 41%。advantage 随任务复杂度增强，对 over-retrieval 和技能质量下降鲁棒。

#### Ask Only When Needed: Proactive Retrieval from Memory and Skills for Experience-Driven Lifelong Agents
- **arXiv**: [2604.20572](https://arxiv.org/abs/2604.20572)  
- **时间**: 2026-04  
- **核心贡献**: 解决"何时触发 Skill 检索"问题。在长期在线学习场景下，通过 proactive 策略主动判断当前任务状态是否需要检索 Skill 或 Memory，避免过度检索带来的上下文污染。

---

### 2.2 技能库自动构建

#### SkillX: Automatically Constructing Skill Knowledge Bases for Agents
- **arXiv**: [2604.04804](https://arxiv.org/abs/2604.04804)  
- **时间**: 2026-04  
- **核心贡献**: 三层层次技能库自动构建框架。从原始轨迹提炼为：(1) 战略计划 → (2) 功能技能 → (3) 原子技能；迭代精炼基于执行反馈自动修订；探索扩展主动生成和验证超出训练数据覆盖的新技能。AppWorld/BFCL-v3/τ²-Bench 评测；plug-and-play，可嫁接至弱 base agent。  
- **代码**: https://github.com/zjunlp/SkillX

#### SkillForge: Forging Domain-Specific, Self-Evolving Agent Skills in Cloud Technical Support
- **arXiv**: [2604.08618](https://arxiv.org/abs/2604.08618)  
- **发表**: **SIGIR 2026 Industry Track**  
- **时间**: 2026-04  
- **核心贡献**: 企业云技术支持场景的域特定技能自进化框架。工业界验证，覆盖发布监控、告警响应、根因分析场景。

#### Automating Skill Acquisition through Large-Scale Mining of Open-Source Agentic Repositories
- **arXiv**: [2603.11808](https://arxiv.org/abs/2603.11808)  
- **时间**: 2026-03  
- **核心贡献**: 从 GitHub 大规模挖掘开源 Agent 仓库（TheoremExplainAgent、Code2Video 等）自动提取程序性技能。流程：仓库结构分析 → dense retrieval 语义技能识别 → 标准化 skill.md 格式。教育内容代理生成质量提升 40% 知识迁移效率。

#### Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills
- **arXiv**: [2603.25158](https://arxiv.org/abs/2603.25158)  
- **时间**: 2026-03（v2: 2026-04-27）  
- **核心贡献**: 从轨迹中提炼局部经验成为可迁移 Skill，解决技能的跨域泛化问题。

---

### 2.3 Skill Bank 协同进化

#### Co-Evolving LLM Decision and Skill Bank Agents for Long-Horizon Tasks (COSPLAY)
- **arXiv**: [2604.20987](https://arxiv.org/abs/2604.20987)  
- **时间**: 2026-04  
- **核心贡献**: 决策 Agent 和 Skill Bank Agent 双路协同进化。决策 Agent 检索 Skill Bank 指导行动；Skill Bank Agent 从无标注 rollout 中自动发现、精炼、更新技能及其 contracts。六个游戏环境，8B 模型对比四个 frontier LLM 基线平均 +25.1% reward。

#### AEL: Agent Evolving Learning for Open-Ended Environments
- **arXiv**: [2604.21725](https://arxiv.org/abs/2604.21725)  
- **时间**: 2026-04  
- **核心贡献**: 跨数百 episode 的开放环境中，解决"如何使用已记忆内容"的核心问题，将历史经验转化为更好未来行为。

#### Dynamic Dual-Granularity Skill Bank for Agentic RL
- **arXiv**: [2603.28716](https://arxiv.org/abs/2603.28716)  
- **时间**: 2026-03  
- **核心贡献**: 面向强化学习 Agent 的双粒度（粗粒度策略 + 细粒度原子动作）动态技能库，提供可复用经验。

---

### 2.4 评测基准

#### How Well Do Agentic Skills Work in the Wild: Benchmarking LLM Skill Usage in Realistic Settings
- **arXiv**: [2604.04323](https://arxiv.org/abs/2604.04323)  
- **时间**: 2026-04  
- **核心贡献**: 真实场景（非实验室设置）下 Agent Skill 使用效果的系统性评测基准。

#### TARSE: Test-Time Adaptation via Retrieval of Skills and Experience for Reasoning Agents
- **DBLP**: CoRR 2026  
- **核心贡献**: 测试时通过检索技能与经验进行自适应，专注推理 Agent 场景。

---

## 三、关键研究问题与进展对照

| 研究问题 | 代表工作 | 当前状态 |
|---|---|---|
| 扁平向量检索随规模退化 | SkillRouter、Graph of Skills | 提出图结构和路由机制，但大规模实验仍有限 |
| 技能依赖与组合 | GraSP、Graph of Skills | DAG 编译层有效，但构建成本未充分分析 |
| 技能库自动构建 | SkillX、Automating Mining | 层次化设计已成共识，探索扩展（novelty-driven）是新趋势 |
| 检索时机（何时触发） | Ask Only When Needed、SRA | 需求感知（need-aware）触发仍是开放问题 |
| 技能质量评估 | SkillsBench、SRA | 技能效果受 harness 影响大，跨 harness 评测是必要维度 |
| 技能进化与协同 | COSPLAY、AEL | 双路协同进化显示出对弱基础模型的放大效果 |

---

## 四、已在本库分析的相关论文

| 论文 | 路径 | 与本方向关联 |
|---|---|---|
| SRA (Skill Retrieval Augmentation for Agentic AI) | papers/agentic_ai/Skill Retrieval Augmentation for Agentic AI/ | 三阶段框架的直接基础 |
| SkillsBench | papers/agentic_ai/SkillsBench Benchmarking how well agent skills work across diverse tasks/ | 评测基准，揭示 loading 瓶颈 |
| Multi-field Tool Retrieval | papers/agentic_ai/Multi-field tool retrieval/ | 工具文档多字段检索，与 Skill 检索互补 |
| Voyager | papers/agentic_ai/Voyager An Open-Ended Embodied Agent with Large Language Models/ | 扁平向量 Skill 库的代表工作，text-embedding-ada-002 索引 |
| MACLA | papers/agentic_ai/MACLA Learning Hierarchical Procedural Memory for LLM Agents/ | 层次化程序性记忆，与 SkillX 三层设计思路相近 |

---

## 五、推荐优先精读（按研究价值排序）

1. **Graph of Skills** (2604.05333) — 大规模依赖感知检索，与研究方向最吻合
2. **GraSP** (2604.17870) — 编译层架构解决 over-retrieval，实验最扎实
3. **SkillRouter** (2603.22455) — 规模化路由，需读全文确认具体机制
4. **SkillX** (2604.04804) — 层次化自动构建，代码开源
5. **COSPLAY** (2604.20987) — 协同进化视角，小模型放大效果显著
