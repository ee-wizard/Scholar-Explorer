# SkillNet: Create, Evaluate, and Connect AI Skills

## 1. 核心结论
- SkillNet 提出了一套面向大规模技能生态的开放基础设施，侧重技能创建、评价、关联与持续演化。该论文比 AgentSkillOS 更偏向“生态系统治理”和“技能市场管理”，是 Skill 数据市场 related work 的重要首选。
- 核心贡献包括：统一技能本体、技能关联网络、五维评估指标（Safety、Completeness、Executability、Maintainability、Cost-awareness）以及 200K 级技能库与 Python 工具包。
- 实验显示 SkillNet 能将可复用技能转化为“耐用资产”，在 ALFWorld / WebShop / ScienceWorld 上平均提升 40% reward 并减少 30% 执行步数，表明系统性技能积累优于孤立经验。

## 2. 研究问题
### 2.1 形式化问题定义
在一个由分散贡献者产生的大规模技能生态中，设计一个基础设施 F，使其能够：
1. 从异构来源创建可重用技能；
2. 对技能进行多维评价与筛选；
3. 建立技能间的关系网络，以支持发现、组合和迁移；
4. 使技能从“临时接口”演化为“持久可复用资产”。

### 2.2 设计目标
- 技能可创建性：支持从执行轨迹、文档、GitHub 项目等多源自动生成技能。
- 技能可靠性：建立 Safety / Completeness / Executability / Maintainability / Cost-awareness 五维评估。
- 技能互联性：构建技能本体与关系图，支持技能相似性、依赖、组合等语义关系。
- 可扩展性：管理 200K+ 技能规模，支持工具包与平台接口。

## 3. 方法概述
### 3.1 技能创建
- SkillNet 从执行轨迹、开源仓库、文档和用户提示中抽取技能，先经过结构化提取，再通过规范化为 `SKILL.md + 资源包`。
- 该创建流程强调“经验到技能”的转化，与传统工具增强不同，它把技能视为可复用、可验证的产品。

### 3.2 技能评价
- 设计了五维评估框架：Safety、Completeness、Executability、Maintainability、Cost-awareness。
- 使用 LLM evaluator + 沙箱执行验证混合评估，尤其在 Executability 维度引入实际运行检查，缓解仅靠文本评价的空洞性。
- 评估结果分为 Good / Average / Poor，帮助技能库筛除低质量或危险技能。

### 3.3 技能关联与本体
- 构建三层 Skill Ontology：Taxonomy、Relation Graph、Package Library。
- Relation Graph 表示技能之间的相似、组合、依赖、归属等多类型关系，支持检索与组合推理。
- 该结构可与 AgentSkillOS 的能力树互补：SkillNet 的本体侧重“谁是什么”，而 AgentSkillOS 侧重“如何用它们”。

## 4. 实验与结果
### 4.1 主要定量结果
- SkillNet 在 ALFWorld / WebShop / ScienceWorld 上平均提升 reward 40%，并减少 30% 执行步数。
- 这说明统一技能网络不仅提升了执行成功率，还改善了效率，符合“技能资产化提高长期收益”的命题。

### 4.2 评估发现
- Safety 与 Maintainability 的 Good 比例较高，说明 SkillNet 的评估框架对规范性保障效果明显。
- Executability 仍是最大瓶颈，说明“技能可执行性”依然需要结合实际运行验证，而非仅靠文本评分。
- Cost-awareness 作为评价维度首次提出，对工程化部署尤为关键。

## 5. 关键证据与局限
### 5.1 证据锚点
- **五维评估框架**：Safety、Completeness、Executability、Maintainability、Cost-awareness 的设计是该论文的核心贡献之一。
- **关系图模型**：通过多类型边构造技能网络，SkillNet 可以支持查询、替代、组合建议与依赖解析。
- **规模实验**：200K+ 技能库展示了该系统在真实市场级规模下的可扩展性。

### 5.2 局限与风险
- **评价偏见风险**：评价框架仍依赖 LLM 评分。即使加入沙箱验证，主观维度（Safety、Completeness）难以完全标准化。
- **技能创建可信度**：自动从文档和仓库生成技能存在“伪装正确但不可执行”的风险，论文强调可执行性验证但未给出全面工程化失败率分析。
- **互操作性缺口**：SkillNet 的技能关系图有助发现关联，但若技能接口不标准，组合阶段仍会出现实际执行失败。AgentSkillOS 的 DAG 计划机制可以作为补充。
- **落地门槛**：从 200K 技能规模到社区级生态，实际仍需面对治理、贡献激励和版本更新问题，论文侧重基础设施而非社区治理机制本身。

## 6. 论文定位与比较
### 6.1 最适合“生态系统管理/市场治理”的候选
- SkillNet 是最接近“市场治理”定位的参考，它不只是检索消费端，而是在技能创建、评价、连接与资产化层面搭建基础设施。
- 相比 AgentSkillOS，更强调“技能库的健康度与可复用性”，而非单次任务的检索/编排性能。

### 6.2 与 SkillFlow 的关系
- SkillFlow 关注技能生态的组织与执行层；SkillNet 提供了生态层面的创建与评价基础。
- 可以把 SkillFlow 看作 SkillNet 提供的技能资产在任务执行端的调度器。

### 6.3 与 SkillsBench 的互补
- SkillsBench 提供技能有效性衡量标准，SkillNet 提供技能质量保障框架。前者测“技能是否有用”，后者测“技能是否值得纳入生态”。

## 7. 科研启发与跨领域类比
### 7.1 Mem0 检索词比对
| Mem0 检索词 | 命中条目摘要 | 关联类型 | 可迁移洞察 |
|---|---|---|---|
| `skills asset evaluation` | 技能评分与可复用性相关 | 直接补充 | SkillNet 的多维评估可作为技能市场的资产评级标准 |
| `graph-based skill relation` | Graph of Skills / SkillNet 等 | 方法互补 | 将 SkillNet 的关系图与 GoS 的依赖图融合，可支持更强的检索 + 组合 |
| `safety-aware skill repository` | 现有 agent skill papers 对安全关注较少 | 约束补充 | SkillNet 的 Safety 维度是市场治理中必须的质量控制机制 |

### 7.2 跨领域类比
- **供应链管理中的质量控制体系**：SkillNet 的五维评估类似于制造业的质量检查表，将技能视为“零部件”，只有通过多维质量检测的零部件才能进入供应链。
- **软件工程中的包管理器与审查流程**：SkillNet 的技能创建与评价类似于 npm/Maven 包审核流程，说明技能市场需要兼具“开放贡献”和“中央验收”的双重机制。
- **金融资产评级**：SkillNet 的多维评价框架可类比信用评级体系，技能不仅需要“可用”，还要“安全、易维护、成本可控”。

### 7.3 潜在改进方向
1. 将 SkillNet 的关系图与 AgentSkillOS 的能力树联合，形成“语义 + 结构”混合检索层。  
2. 在评估框架中引入技能生命周期指标，如“过时风险”“修订频率”“社区活跃度”。  
3. 建设技能市场治理规则：贡献者信誉、技能审计、自动淘汰机制。

## 8. 推荐阅读
- **SkillFlow: Organizing, Orchestrating, and Benchmarking Agent Skills at Ecosystem Scale** (2026, arXiv:2603.02176) — 任务执行侧市场治理视角。  
- **Graph of Skills: Dependency-Aware Structural Retrieval for Massive Agent Skills** (2026, arXiv:2604.05333) — 依赖图视角的结构化检索。  
- **SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks** (2026, arXiv:2602.12670) — 技能评估与异质性证据。  
- **Skill Retrieval Augmentation for Agentic AI** (2026, arXiv:2604.24594) — 大规模技能检索与需求匹配。

## 9. 推荐写入队列
- `python3 papers/.db/paperdb_cli.py queue add-reading-list '[{"title":"SkillFlow: Organizing, Orchestrating, and Benchmarking Agent Skills at Ecosystem Scale","source":"arXiv","priority":"中"},{"title":"Graph of Skills: Dependency-Aware Structural Retrieval for Massive Agent Skills","source":"arXiv","priority":"中"},{"title":"SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks","source":"arXiv","priority":"中"}]'`

## 10. mem0 知识库记录
- SkillNet 将技能视为可评估资产，提出了“技能评估 + 技能关系图 + 技能创建”的组合，用于构建可治理的技能市场基础设施。  
- 该论文适合作为研究“技能库质量控制”“技能链路可视化”“技能长期演化”问题的基础性引用。  
- 若结合 SkillFlow 的 DAG 计划层，可形成“生态建设 + 任务编排”的闭环市场基础设施。
