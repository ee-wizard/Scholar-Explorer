# SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks

## 1. 核心结论
- SkillsBench 是首个以“技能作为评估对象”的大规模基准，明确衡量技能是否真正提高代理表现，而不仅仅是提供附加上下文。
- 现实实验表明：筛选技能能提高平均通过率 16.2pp，但收益高度异质，16/84 任务甚至出现负增益，说明技能并非总是有益。
- 自生成技能平均无效或负效应，表明有效技能需要专业设计与人类策划，不可简单由模型自发产生。

## 2. 研究问题
### 2.1 形式化问题定义
定义一个标准化基准 B，使其能够回答以下问题：
1. 给定代理 A、技能集 S、任务集合 T，技能是否提高任务解决率？
2. 不同技能来源（人工策划的 Skills vs. 模型自生成 Skills）的效果差异如何？
3. 技能效果在不同领域和模型配置间的稳定性如何？

### 2.2 设计目标
- 任务覆盖广泛：86 个任务、11 个领域。  
- 条件对比明确：no Skills / curated Skills / self-generated Skills。  
- 验证严格：确定性 verifier、7,308 条轨迹评估。  
- 量化异质性：按领域、模型配置、技能类型细粒度拆分。

## 3. 方法概述
### 3.1 基准结构
- 每个任务包含 instruction、环境、技能、验证器和参考解。
- Skills 以目录化包形式注入环境，供代理在推理时读取。
- 自生成技能条件要求模型在不使用外部技能库的前提下自行编写过程性知识。

### 3.2 评估流程
- 7 个代理模型配置、7,308 条轨迹，覆盖 Claude Code、Gemini CLI、Codex CLI 等主流 agent 平台。
- 验证采用 deterministic verifiers，避免基于 LLM judge 的随机性，面向实际任务成功率。
- 指标包括绝对通过率、归一化增益、领域差异和负增益比例。

### 3.3 关键设计要点
- **集中式技能注入**：Skills 以系统上下文方式注入，保持与现实代理集成路径一致。
- **自生成技能实验**：这是本基准最具洞察力之处，它直接检验了“技能是否可由模型自行发明”的假设。
- **负增益分析**：16/84 任务出现负效应，强调技能设计不当的潜在风险。

## 4. 结果要点
### 4.1 技能总体收益
- Curated Skills 平均提高 16.2pp，证明技能在大多数任务中具有实证价值。
- 不同领域差异巨大：Software Engineering +4.5pp，Healthcare +51.9pp，说明技能效果强依赖领域与任务类型。

### 4.2 自生成技能结论
- 自生成技能平均无提升甚至负效应，说明当前模型尚不具备稳定自行构建高质量技能的能力。
- 该结果与 Skill Retrieval Augmentation 的经验型消费端结论一致：技能需求匹配是必要而非充分条件。[1]

### 4.3 小模型+Skills 现象
- 小模型在 Skills 辅助下可与大模型匹配，说明技能市场可以成为一种“参数节省的能力放大器”。
- 这也支持“市场消费端”方向：不只是技能本身，而是技能如何补偿模型能力的差异。

## 5. 关键证据与局限
### 5.1 证据锚点
- 86 任务、11 领域、7,308 条轨迹是该基准最强的证据基础。
- 通过“无技能 / curated / self-generated”三条件对比，明确回答了技能是否有益以及模型是否能自发生成技能。

### 5.2 风险与假设
- **任务分布偏差**：86 任务虽大，但仍属于特定领域；结果对其它行业场景（如金融合规、机器人操作）是否成立仍需验证。
- **自生成技能实现门槛**：论文将当前模型自生成技能的失败归因于模型能力，但部分失败也可能来自“评估规则过于严格”或“自生成形式未与技能规范完全一致”。
- **技能注入方式限制**：Skills 以系统上下文注入，未考察所谓“技能 market API / catalog service”中更复杂的发现与授权场景。
- **负增益解读**：16/84 任务负增强说明技能并非全局正效应，但论文并未充分区分“技能无用”与“错误技能误导”两种失败模式。

## 6. 论文定位与比较
### 6.1 市场质量评测标准证据
- SkillsBench 最适合作为“市场质量评测标准”的引用，尤其适合论述技能有效性、技能异质性和技能设计风险。

### 6.2 与 SkillFlow 的关系
- SkillsBench 提供了技能效益与异质性的实证背景；SkillFlow 则在此基础上研究如何通过结构化检索与编排最大化这些技能的实际价值。

### 6.3 与 Skill Retrieval Augmentation 的对比
- SRA 关注如何从大规模技能库中检索和合并技能；SkillsBench 关注技能是否真的有用。二者结合可以回答“检索到的技能是否真正带来正收益”。

## 7. 科研启发与跨领域类比
### 7.1 Mem0 检索词比对
| Mem0 检索词 | 命中条目摘要 | 关联类型 | 可迁移洞察 |
|---|---|---|---|
| `skill benchmark heterogeneity` | SkillsBench 证明技能效用高度依赖领域 | 质量评测 | 说明技能市场需要细粒度评价指标，而非整体增益 |
| `skill negative utility` | 16/84 任务负增益 | 风险分析 | 提醒研究者技能市场必须包含“淘汰机制”与“误导检测” |
| `small-model skill amplification` | 小模型在技能辅助下能匹配大模型 | 市场机会 | 技能市场可以成为“低成本能力扩展”机制 |

### 7.2 跨领域类比
- **医学临床试验中的副作用检测**：技能市场中的负增益类似于药物试验中的不良反应，说明仅有“平均有效”不足，必须评估“个体风险”。
- **教育测评中的教学辅助工具**：SkillsBench 类似教育评估工具的 A/B 测试，既要看平均提升，也要看“哪些学生/任务受益、哪些受损”。
- **金融风控中的策略鲁棒性**：技能的异质性与负收益相当于投资组合中的尾部风险，市场治理应包含“止损机制”。

### 7.3 潜在改进方向
1. 将 SkillsBench 的负增益分析与 AgentSkillOS 的结构化组合结合，设计“仅在模型具有正向可组合性时才启用技能”的决策机制。  
2. 将自生成技能失败案例作为技能库扩展信号，构建“模型建议 + 人工审查”的技能采纳流程。  
3. 增加“技能误导性”评估指标，区分误导型技能与无效型技能。

## 8. 推荐阅读
- **Skill Retrieval Augmentation for Agentic AI** (2026, arXiv:2604.24594) — 需求匹配与大规模技能检索。  
- **SkillNet: Create, Evaluate, and Connect AI Skills** (2026, arXiv:2603.04448) — 技能市场治理与评价。  
- **SkillFlow: Organizing, Orchestrating, and Benchmarking Agent Skills at Ecosystem Scale** (2026, arXiv:2603.02176) — 生态执行层的检索与 DAG 编排。  
- **Graph of Skills: Dependency-Aware Structural Retrieval for Massive Agent Skills** (2026, arXiv:2604.05333) — 技能依赖图检索。

## 9. 推荐写入队列
- `python3 papers/.db/paperdb_cli.py queue add-reading-list '[{"title":"Skill Retrieval Augmentation for Agentic AI","source":"arXiv","priority":"高"},{"title":"SkillNet: Create, Evaluate, and Connect AI Skills","source":"arXiv","priority":"中"},{"title":"SkillFlow: Organizing, Orchestrating, and Benchmarking Agent Skills at Ecosystem Scale","source":"arXiv","priority":"中"}]'`

## 10. mem0 知识库记录
- SkillsBench 证明技能市场的一个关键事实：技能不是天然正效应体，而是“有条件的资产”。  
- 该基准为技能市场质量治理提供了第一手数据，特别是在“异质任务下的有效性波动”和“自生成技能失败”方面。  
- 如果将该结论应用于市场治理，应优先构建“技能验证 + 技能淘汰”机制，而不是盲目扩充技能库。
