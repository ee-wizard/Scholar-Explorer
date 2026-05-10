---
title: "SkillFlow: Scalable and Efficient Agent Skill Retrieval System"
paper_id: "arXiv:2504.06188"
analysis_date: "2026-05-08 10:30:00 +08:00"
main_tag: "agentic_ai"
tags:
  - "agentic_ai"
  - "skill_retrieval"
  - "multi_stage_retrieval"
  - "agent_skills"
  - "reranking"
  - "dense_retrieval"
  - "LLM_agents"
related_papers: "SkillsBench (arXiv:2602.12670), SRA (arXiv:2604.24594), GoS (arXiv:2604.05333), Voyager (arXiv:2305.16291)"
---

# SkillFlow: Scalable and Efficient Agent Skill Retrieval System

> 学术分析报告  
> **主标签**：agentic_ai  
> **论文标签**：agentic_ai; skill_retrieval; multi_stage_retrieval; agent_skills; reranking; dense_retrieval; LLM_agents

---

## 1. 问题定义

### 问题背景

AI agent 通过在推理时将可复用"技能"加载到上下文来扩展自身能力。Anthropic 于 2025 年提出 SKILL.md 标准化格式，定义程序性知识包，包含自然语言描述、YAML 元数据、可执行代码及脚本束。该格式被开源社区广泛采用，GitHub 上现有数万个 community 贡献的 SKILL.md 文件，由 SkillsMP 这一开放技能市场聚合。

然而，大规模技能库带来了**选择性负担**问题：实验与理论均已证明，将不相关技能加载进 agent 上下文会同时损害精度与效率（Li et al., 2026；Gao, 2026）。暴力加载全库不可行；人工筛选无法跟上库的增速。

### 关键挑战

1. **规模问题**：~36K 技能文件，任意任务需在毫秒至秒级完成初筛。
2. **语义异构性**：SKILL.md 混合自然语言描述、YAML 元数据与代码片段，传统词汇检索（BM25）无法有效跨越任务描述与技能内容之间的语义鸿沟。
3. **质量异构性**：Community 贡献质量良莠不齐，绝大多数社区技能的可执行代码密度和脚本捆绑率远低于经过专家审核的 oracle 技能，导致检索到高相关技能不等于获得有效技能。
4. **精度-召回权衡**：检索阶段需高召回（保证 oracle 技能不被漏掉），但最终加载到 agent 的技能数量必须极小（通常 ≤5），以避免上下文负担。

### 形式化定义

给定任务描述 $t$ 和技能库 $\boldsymbol{S} = \{s_1, s_2, \ldots, s_N\}$（$N \approx 36000$），目标是在交互式 agent 延迟预算内返回小子集 $S^* \subset S$，$|S^*| \leq 5$，使得 $S^*$ 与任务最相关、且包含可实际提升 agent 完成率的 oracle 技能。优化目标可分解为两个层次：
- **检索层**：最大化 oracle 技能的 recall@k（$k = 1000 \to 100 \to 10$），通过逐级缩减候选集实现；
- **端到端层**：最大化 Pass@1（agent 在任务上的成功率），这受限于技能库覆盖率上界（oracle ceiling）。

### 问题重要性

SkillFlow 是**第一个将 agent 技能获取形式化为信息检索问题**的工作。之前的工作或假设小而固定的手工技能集，或依赖暴力遍历；本文对 large-scale、community-driven、heterogeneous 技能库的检索挑战进行了专门的系统化研究，并通过两个基准上的对比实验揭示了"检索质量"与"库覆盖率/技能质量"对 agent 性能的独立贡献。

---

## 2. 前人工作的方法缺陷

### 工具检索系统的适配性缺陷

现有工具与 API 检索系统（Toolformer、Gorilla、ToolLLM）的检索单元是**原子 API 签名**——名称、参数类型、简短说明——文档小而结构一致，适合扁平检索。SKILL.md 技能定义是**程序性知识包**，内容混合了多段自然语言描述、YAML 元数据、嵌套代码块与脚本文件，文档长度分布高度不均（中位数约 600-700 词，但分布长尾），且存在严重的语义与结构异构性。直接将 API 检索架构迁移至技能检索会面临内容截断（重要信息在末尾）和语义覆盖不足的双重问题。

### 自生成技能库的局限

Voyager（Wang et al., 2023）在 Minecraft 开放世界中通过 LLM 自主合成技能并逐步扩充技能库，检索依赖 text-embedding-ada-002 的向量相似度。这一方案的根本局限在于：技能库是**任务特定、自生成**的，只覆盖已探索过的动作空间，不具备对社区贡献的多源、多领域异构技能库的泛化能力。此外，Voyager 未解决技能库依赖关系（GoS 的核心问题）也未解决规模检索。

### 简单检索方案的不足

- **BM25 词汇检索**：任务描述与技能内容之间存在显著词汇鸿沟（任务用高层意图词，技能用工具/框架名词），BM25 MRR 仅 0.266，远低于密集检索的 0.553（两者均在 SkillsBench 上，源自论文 Table 4）。
- **单阶段密集检索**：高召回但精度低——recall@1000=0.905，但 P@10 仅 0.121；最终向 agent 暴露大量低质候选，上下文负担与噪声同步增加。
- **商业 API（Vercel）**：单一检索，返回 1 个技能/任务，Use Rate 仅 40.6%（SkillsBench）和 22.3%（Terminal-Bench），说明被返回的技能对 agent 来说往往不被判为有用。

### 缺乏库质量感知

现有检索方案均未系统分析和验证"什么样的技能在端到端上有效"。SkillFlow 通过结构代理指标（代码分数、脚本捆绑率）的统计分析揭示了 oracle 技能与 community 技能之间的关键质量差异，这一分析本身是新的贡献。

---

## 3. 研究动机与提出方案

### 研究动机

SkillFlow 的核心驱动是一个"已存在规模问题但无人正式建模"的空白：GitHub 上有数万个社区技能定义，任何 agent 理论上都可使用，但没有可扩展的机制在运行时从中精准召回最相关的技能。作者观察到：（1）技能太少限制 agent 能力，（2）技能太多特别是不相关技能损害性能，（3）现有 tool retrieval 方案针对 API 而非 procedural skill bundle，（4）Vercel 等商业 API 质量低下（use rate < 25%）。

这一张力指向清晰的问题定义：将技能获取重构为信息检索问题，设计一条从大规模候选（36K）逐步精筛到极小返回集（≤5）的检索流水线，同时满足交互式 agent 的延迟约束。

### 方法本质

SkillFlow 是一条**四阶段渐进式精筛检索流水线**（dense retrieval → shallow reranking → deep reranking → LLM selection），每阶段以不同的计算预算对候选集做不同粒度的质量判断，并通过参数化的 top-k 缩减逐步平衡召回与精度。其设计思路直接借鉴 IR 社区的 multi-stage retrieval 范式（Nogueira & Cho, 2019），但适配了 SKILL.md 异构内容结构（短 vs. 长，自然语言 vs. 代码）。

### 去包装后的方法还原

SkillFlow 的核心设计决策可以还原为三个工程约束的交叉解：
1. **召回约束**：oracle 技能必须在 k=1000 内找到（才有机会传入后续精排），因此第一阶段用密集向量检索 + 多查询并集（M=5），recall@1000=0.905。
2. **精度约束**：最终返回 ≤5 个技能，需要高 P@5，因此需要两轮 cross-encoder 精排，分别在截断内容（512 token）和完整内容（4096 token）上计分。
3. **延迟约束**：完整 SKILL.md 评分代价高（bge-reranker-v2-m3 × 4096 tokens × 100 候选），因此浅层 cross-encoder（MiniLM）先从 1000 → 100，再用深层 cross-encoder 从 100 → 10，避免对全量候选做完整内容评分。

这一设计在根本上是"用级联计算换质量"，而非提出新的检索模型或新的评分函数。

### 核心假设

1. 单个 dense bi-encoder 能以较高召回率从 36K 技能中筛出 oracle 技能（recall@1000 > 0.9）。
2. Cross-encoder 在截断内容上的粗排（512 token）足以将候选缩减 10×，并保持足够的 oracle 覆盖率；在完整内容上的精排（4096 token）可进一步提升排名质量。
3. LLM-based filter 能有效过滤剩余 10 个候选中的不相关技能，且计算成本（GPT-4o-mini × 10 次）可接受。
4. **多查询在检索阶段有益（覆盖率），在重排阶段有害（引入噪声）**——这是论文中最重要的反直觉发现之一，并通过消融实验系统验证。

### 核心创新点

1. **第一个将 agent 技能获取形式化为 IR 问题的系统**，在 ~36K community 技能库上建立可扩展的多阶段检索基线。
2. **多查询与多阶段的交叉效应分析**：实验揭示 M=5 在检索阶段提升召回（+3.0%），但在重排阶段显著损害 MRR（-18.7% vs M=1 at deep reranker）。这一结论有明确的可验证解释：候选集越小，噪声查询的干扰效应越强；任务语义在小集合中已足够区分，额外查询反而引入歧义。
3. **技能质量的结构代理指标实证分析**：oracle 技能 vs. community 技能在"代码块分数"（中位数 0.39 vs. 0.24，p=2.83×10⁻¹²）和"脚本捆绑率"（33.2% vs. 13.4%，p=2.83×10⁻¹⁷）上差异显著，而长度和文档量无显著差异——直接转化为可操作的技能编写指导。
4. **库覆盖率是下游瓶颈的实证证据**：Terminal-Bench 上 70.1% use rate 但 0% 显著性能提升，与 SkillsBench 上的+78.3% Pass@1 形成锐利对比，系统性确认了"检索质量 ≠ 端到端收益"的边界条件。

### 方法总览与流程

**离线阶段**：
- 爬取 GitHub 上所有包含 SKILL.md 的仓库（通过 SkillsMP API 枚举），保留 35,866 个有效技能
- 用 BAAI/bge-base-en-v1.5 对每个技能的描述和内容计算 d 维 dense embedding，存入 FAISS 索引

**在线阶段（每个任务 t）**：

**Stage 1 — Dense Retrieval**（目标：高召回，候选 1K）：
- 用 GPT-4o-mini 将任务 t 分解为 M₁=5 条查询（按技术域、编程语言、框架、工具、核心问题不同层次各生成一条）
- 每条查询用 bge-base-en-v1.5 编码，计算与全库 embedding 的余弦相似度
- 5 条查询结果取并集并去重，得到候选集 $\mathcal{C}_1$（$k_1=1000$）
- 聚合策略：union（最优，相比 RRF 和 max 策略）

代码片段（仓库实现，对应 Stage 1：multi-query + dense retrieval + union/max 聚合）：

```python
# skill_flow/pipeline/stage_helpers.py
def retrieve_task(searcher, query, top_k, query_gen=None, aggregation="max", top_k_per_query=0, task_id=""):
  if query_gen:
    queries = query_gen.generate_multi(task_id, query)
    if len(queries) > 1:
      collected = collect_multi_scores(searcher, queries)
      if aggregation == "union":
        return union_from_collected(collected, top_k_per_query)
      return rank_from_scores(collected, aggregation, top_k)
    return search_multi(searcher, queries, top_k, aggregation)
  return searcher.search(query, top_k=top_k)

# skill_flow/retriever/retriever.py
class IndexSearcher:
  def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
    effective_k = top_k if top_k is not None else self._config.top_k
    query_vec = self._encoder.encode_query(query)
    k = min(effective_k, self._index.ntotal)
    scores, indices = self._index.search(query_vec, k)
    ...
```

**Stage 2 — Shallow Reranking**（目标：候选从 1K → 100）：
- 用 cross-encoder/ms-marco-MiniLM-L-6-v2（轻量 cross-encoder）
- 内容截断至 $L_{shallow}=512$ tokens
- 单查询（M₂=1），保留 top $k_2=100$
- 公式：$r_{shallow}(q, s_i) = \text{CE}_{shallow}([q; \text{trunc}(s_i, 512)])$

代码片段（仓库实现，对应 Stage 2：cross-encoder shallow rerank）：

```python
# skill_flow/reranker/reranker.py
class Reranker:
  def rerank(self, query: str, candidates: list[SearchResult], top_k: int | None = None) -> list[SearchResult]:
    if not candidates:
      return []
    input_k = self._config.input_top_k
    if input_k > 0:
      candidates = candidates[:input_k]

    max_chars = self._config.max_content_chars
    scorable = [c for c in candidates if c.content]
    pairs = [[query, self._truncate(c.content or "", max_chars)] for c in scorable]
    scores = self._model.predict(pairs, batch_size=self._config.batch_size).tolist()
    ...
```

**Stage 3 — Deep Reranking**（目标：候选从 100 → 10）：
- 用 BAAI/bge-reranker-v2-m3（强 cross-encoder，支持 4096 token 输入）
- 内容截断至 $L_{deep}=4096$ tokens（接近完整 SKILL.md）
- 单查询（M₃=1），保留 top $k_3=10$
- 公式：$r_{deep}(q, s_i) = \text{CE}_{deep}([q; \text{trunc}(s_i, 4096)])$

代码片段（仓库实现，对应 Stage 3：deep reranker 作为第二次 rerank）：

```python
# skill_flow/pipeline/stages.py
def run_stage3(config, tasks, task_candidates, output_dir, *, gt_context=None, tasks_dir=None) -> None:
  _run_rerank_stage(
    config.models.deep_reranker,
    3,
    "deep_reranker",
    config,
    tasks,
    task_candidates,
    output_dir,
    gt_context,
    tasks_dir,
  )
```

**Stage 4 — LLM Selection**（目标：候选从 10 → ≤5）：
- 用 GPT-4o-mini 对 $\mathcal{C}_3$ 中每个技能做二值相关性判断（relevancy filter）
- 可选链式 specificity filter（SkillFlow-specific 变体）：在相关性过滤后再判断是否提供任务特定可操作指导
- 输出 $S^*$，$|S^*| \leq 5$

代码片段（仓库实现，对应 Stage 4：LLM selector + specificity filter）：

```python
# skill_flow/selector/selector.py
class Selector:
  def select(self, query: str, candidates: list[SearchResult], task_id: str = "") -> list[SearchResult]:
    ...
    system_prompt = self._render_system_prompt()
    user_prompt = self._render_user_prompt(query, candidates)
    content = self._call_llm(system_prompt, user_prompt)
    selected_keys = self._parse_response(content, index_to_key)
    ...
    relevant = [c for c in candidates if c.key in selected_set]
    if self._specificity_template and relevant:
      return self._run_specificity(query, relevant, cache_key)
    return relevant
```

### 关键超参数

| 参数 | 值 | 来源 |
|------|-----|------|
| 查询生成数（Dense）M₁ | 5 | 消融实验（A.4）|
| dense 检索候选 k₁ | 1000 | 消融实验 |
| 浅排 token 截断 $L_{shallow}$ | 512 | 模型限制 |
| 浅排候选 k₂ | 100 | 消融实验（A.5）|
| 深排 token 截断 $L_{deep}$ | 4096 | 消融实验 |
| 深排候选 k₃ | 10 | 消融实验 |
| 最终输出 $|S^*|$ | ≤5 | 消融实验（A.6）|

### 输入输出与中间表示

- **输入**：任务描述字符串 $t$，技能库 $\boldsymbol{S}$（FAISS 向量索引 + 原始文本）
- **Stage 1 输出**：$\mathcal{C}_1$ — 至多 1000 个技能（ID + 向量分数）
- **Stage 2 输出**：$\mathcal{C}_2$ — 至多 100 个技能（ID + cross-encoder 分数）
- **Stage 3 输出**：$\mathcal{C}_3$ — 至多 10 个技能（ID + deep cross-encoder 分数）
- **最终输出**：$S^*$ — 至多 5 个技能的完整 SKILL.md 文本，注入 agent 上下文

### 训练/索引构建细节

- 所有模型均为预训练权重，无任何针对技能数据的微调（zero-shot 使用）
- 离线 embedding 建库：bge-base-en-v1.5 对技能描述计算嵌入，FAISS 索引存储（具体索引类型论文未明确说明，Not clearly stated in parsed content.）
- 图片压缩与脚本捆绑检测：爬取时保留仓库元数据，统计 SKILL.md 所在目录的文件列表以判断是否捆绑可执行脚本

### 机制有效性解释

多阶段精筛之所以奏效，根本原因是**不同计算预算对应不同的区分粒度**：
- Dense retrieval 用全局语义相似度做大范围初筛，对 36K 候选是唯一可行的复杂度；
- Shallow cross-encoder 利用 query-document 交互注意力做更精准但局部（截断）的判断，从 1000 缩至 100；
- Deep cross-encoder 在完整 SKILL.md 内容上做精判，从 100 缩至 10，Stage ablation 中 MRR 从 0.587 → 0.634（+8.0%），R@10 从 0.520 → 0.595（+14.4%，源自 Table 4）；
- LLM filter 提供语义层面的最终人工类似判断，但消融显示其贡献有限（relevant skills 经过前三阶段已被良好过滤）。

**多查询悖论**（图3是最能支撑核心结论的图表）：M=5 在检索阶段因查询多样性覆盖不同语义维度而提升 recall@1000（+3.0%），但在重排阶段，小候选集中多查询对同一技能的分数聚合引入了额外的得分方差，损害了基于排序的 MRR 度量。图3(b) 清晰显示，在每个 pipeline 阶段，M=1 的 MRR 均高于 M=3 和 M=5，而图3(c) 显示 M=5 的 R@k 曲线在 k≈224 时才超过 M=1，这解释了"相同架构在不同阶段选择不同 M 值"的反直觉设计。

---

## 4. 实验对比与性能提升

### 数据集

**SkillsBench**（Li et al., 2026，arXiv:2602.12670）：87 个任务，229 个 oracle 技能（每任务配 1-3 个经专家审核的技能），是唯一有检索 ground truth 的技能增强 benchmark。实际评测后去除问题任务，保留 65 个任务（Pass 评测）和 87 个任务（检索评测）。

**Terminal-Bench**（Merrill et al., 2026）：89 个终端编程任务，无技能标注，用于评测技能检索对无 oracle 覆盖领域的泛化能力。保留 88 个任务。

**技能库**：SkillsMP 社区技能，35,866 个有效 SKILL.md（SkillsBench 的 229 个 oracle 技能注入其中）。

### 指标

- **端到端**：Pass@1（组合无偏估计器；3次运行），Pass@3，Pass˜@3（三次均成功概率）
- **技能使用**：Skill Use Rate（agent 在任务中使用 ≥1 个检索技能的比例）
- **检索质量**：Recall@k，Precision@k，MRR，Hit@k；95% Bootstrap CI，Holm-Bonferroni 校正 p 值

### 基线

| 基线 | 类型 | 描述 |
|------|------|------|
| No Skills | 消融基线 | agent 不加载任何技能 |
| Oracle Skills | 上界 | 注入经专家审核的完全相关技能 |
| Vercel | 商业 API | 第三方 skill retrieval API，每任务返回 1 个技能 |
| BM25 | 词汇检索 | 稀疏第一阶段替代方案 |
| Hybrid (Dense + BM25 via RRF) | 混合检索 | 稠密与稀疏融合 |

Agent 模型：Codex CLI + GPT-5-mini（medium reasoning effort），通过 Harbor 框架 + Daytona Sandboxes 云服务运行。所有非 OpenAI 模型在 Nvidia RTX A5000 上运行。

代码片段（仓库实现，对应测试入口与执行流程）：

```python
# benchmark/scripts/cli.py
def cmd_run(args: argparse.Namespace) -> int:
  config_path = Path(args.config) if args.config else None
  config = load_config(config_path)
  runner = EvaluationRunner(config, config_path=config_path)
  if config.num_runs > 1:
    return runner.run_multiple()
  return runner.run()

# benchmark/core/runner.py
class EvaluationRunner:
  def _run_new(self, job_name: str) -> int:
    cmd = build_harbor_run_command(self.config, job_name)
    return self._execute_command(cmd)

# benchmark/core/commands.py
def build_harbor_run_command(config: EvalConfig, job_name: str) -> list[str]:
  cmd = [
    *_harbor_base_command(),
    "run",
    "--job-name", job_name,
    "--jobs-dir", str(config.jobs_dir),
    "--n-concurrent", str(config.environment.n_concurrent),
  ]
  cmd.extend(build_mode_args(config))
  cmd.extend(build_task_args(config))
  if config.tasks_path:
    cmd.extend(["-p", str(config.tasks_path)])
  elif config.dataset:
    cmd.extend(["--dataset", config.dataset])
  if config.environment.use_daytona:
    cmd.extend(["--env", "daytona"])
  return cmd
```

### 端到端结果（Table 1）

**SkillsBench**（65 个任务）：

| 条件 | Pass@1 | Pass@3 | Pass˜@3 |
|------|--------|--------|---------|
| Oracle | 19.5% [12.3, 27.2] | 35.4% | 6.2% |
| **SkillFlow** | **16.4%** [9.7, 23.6]* | **29.2%** | **4.6%** |
| Vercel | 9.7% [4.6, 15.4] | 20.0% | 1.5% |
| No Skills | 9.2% [4.1, 14.9] | 16.9% | 0.0% |

*$p_{adj} = 3.64 \times 10^{-2}$，Cohen's $h=0.22$（小效应）；达到 oracle ceiling 的 84.1%。

**Terminal-Bench**（88 个任务）：No Skills 34.8%，SkillFlow 31.8%（$p_{adj} \geq 0.05$，无显著差异），SkillFlow-specific 34.9%（无显著差异）。Cost 和 Steps/Task 在所有条件下 CI 重叠，说明性能差异来自技能质量而非计算量。

### 检索与使用率结果（Table 2）

| 条件 | SkillsBench Use Rate | Terminal-Bench Use Rate |
|------|---------------------|------------------------|
| Oracle | 69.2% | — |
| SkillFlow | 68.8%** | 70.1%** |
| SkillFlow top-1 | 68.8%** | 72.7%** |
| Vercel | 40.6% | 22.3% |

**SkillFlow（全量）在 SkillsBench 从 36K 库中召回 61.8% oracle 技能**。Top-1 变体也召回 37.6%，表明排名质量而非数量是 use rate 优势的来源。

### 阶段级检索消融（Table 3 & 4）

| 方案 | MRR | R@10 | P@10 | Hit@10 |
|------|-----|------|------|--------|
| BM25 only | 0.266 | 0.238 | 0.055 | 0.391 |
| Hybrid (Dense+BM25) | 0.522 | 0.480 | 0.114 | 0.713 |
| Dense only | 0.553 | 0.477 | 0.121 | 0.713 |
| + Shallow Reranker | 0.587 | 0.520 | 0.130 | 0.724 |
| + Deep Reranker | **0.634** | **0.595** | **0.147** | **0.793** |

逐级精筛有效：BM25 词汇检索极差（语义鸿沟大），Hybrid 未提升（BM25 弱信号稀释了 dense），Dense 是强基线，两轮 cross-encoder 各自贡献稳定增益。

### 强/弱场景

- **最强场景**：SkillsBench（库中存在 oracle 技能），Pass@1 从 9.2% → 16.4%（+78.3%），检索到 84.1% oracle ceiling。
- **最弱场景**：Terminal-Bench（库缺乏领域覆盖），技能 use rate 高（70.1%）但端到端无显著增益，说明检索质量无法弥补库覆盖率的缺失。

---

## 5. 方法局限与缺陷

### 作者明示局限

1. **评测规模受限**：仅用单一 agent 模型（Codex CLI + GPT-5-mini），仅两个 benchmark，无法充分评测 SkillFlow 在不同 agent 架构和任务域的泛化性。
2. **质量代理指标而非直接测量**："代码分数"和"脚本捆绑率"是结构性代理，并非人类对技能可用性的直接评分。Future work 应纳入专家评分或下游效用分数。
3. **技能库时效性**：库构建于固定时间点，GitHub 上 SKILL.md 数量持续增长，更新后的库可能改善覆盖率和结果。

### 分析者补充局限

**Oracle 上界本身较低（19.5% Pass@1）**：即便精确检索到所有 oracle 技能，agent 的成功率也仅 19.5%，说明技能增强 agent 的瓶颈已从检索下移到了"技能内容质量 → agent 推理能力"这一链条。论文的 Section A.10 对此做了初步探查，但未提供系统性解释。

**Terminal-Bench 的 use rate 悖论**：agent 使用了 70.1% 的 SkillFlow 技能，但端到端无提升。这既可能是库覆盖率问题，也可能是 agent 在使用低质量技能时产生了"有害增益"（按照 SkillsBench 论文的先验，负收益的技能注入是已知现象）。论文未区分这两种解释。

**评测协议偏置**：SkillsBench 的 oracle 技能由 task 作者与 skill 作者协同定义，相关性是主观定义的；oracle 技能与任务描述的语义一致性高于真实 community 技能，可能使检索任务比真实场景更简单。

**深度精排延迟问题**：~26s 的深度精排延迟占总流程 74%，对交互式 agent（预期响应时间秒级）是显著负担。论文虽提出可去掉此阶段（~8s 延迟），但未评测缩短流水线后端到端的实际影响。

**BM25+Dense Hybrid 无提升**：论文解释为"BM25 弱信号稀释 dense"，但未尝试词汇增强的 dense 检索（如 SPLADE 或 ColBERT）等混合检索范式的更优实现，使得"混合检索不适用于技能库"的结论欠缺充分对照。

---

## 6. 科研启发

1. **技能库动态扩充与质量感知爬取**：当前 SkillFlow 库是在固定时间点静态爬取的，且对 community 技能的质量无预筛选。一个可验证的研究方向是：设计基于结构质量代理指标（代码分数、脚本捆绑率、执行可验证性）的**在线过滤爬取框架**，仅将满足质量阈值的技能纳入库，建立"有界技能质量假设"（库中所有技能的代码分数 ≥ θ 时，SkillFlow 的 Pass@1 增益是否与 oracle ceiling 线性相关？）。新颖性在于将 IR 系统的文档质量问题从后处理（检索后过滤）前移到语料库构建阶段，并给出可验证的质量-性能曲线。

2. **技能粒度与可分解性的检索建模**：SKILL.md 技能是程序性知识包，可能本身具有依赖结构（如 GoS 所揭示），而 SkillFlow 将每个 SKILL.md 视为原子检索单元。研究问题：**当技能存在依赖关系时，如何在多阶段检索中建模依赖完整性约束**？形式化定义为：给定依赖图 G，检索子集 $S^*$ 不仅需最大化相关性，还需满足 $S^*$ 的闭包（在 G 中）包含所有前置依赖。可验证假设：在依赖密集的任务域（如系统编程），添加依赖约束的检索相比纯相关性检索显著提升端到端成功率。

3. **交错式动态技能检索**：SkillFlow 当前在任务执行前一次性检索。对于"需求随执行进展而涌现"的复杂多步任务，可研究**基于 agent 中间状态触发的动态二次检索**（类比 DRAGIN 的动态检索触发机制，但适配技能粒度而非 passage 粒度）。核心问题是：agent 遭遇什么类型的执行障碍时应触发再检索？障碍信号（工具调用失败、能力缺口、错误类型）与技能相关性之间存在什么可测量的对应关系？

4. **可执行性感知的技能重排序**：SkillFlow 的重排阶段以语义相关性为唯一信号，但 Terminal-Bench 结果表明高相关性技能在无可执行代码时仍无用。可研究**将可执行性代理指标（代码分数、脚本捆绑检测、语法有效性检查）融入重排评分**的方法，建立"相关性 × 可执行性"联合评分模型。可验证假设：相同相关性排名条件下，代码分数高的技能在端到端指标上显著优于代码分数低的技能（可在 SkillsBench oracle 内的子集分析中验证）。

---

## 7. 参考文献图谱

### 文献分类表

| 文献名 | 作者/年份 | 角色 | 知识库状态 |
|--------|----------|------|-----------|
| SkillsBench: Benchmarking how well agent skills work across diverse tasks | Li et al., 2026 (arXiv:2602.12670) | 实验基线（benchmark） | ✅ 已收录 |
| Skill Retrieval Augmentation for Agentic AI | Li et al., 2026 (arXiv:2604.24594) | 方法基础 | ✅ 已收录 |
| Graph of Skills: Dependency-Aware Structural Retrieval for Massive Agent Skills | 2026 (arXiv:2604.05333) | 方法基础 | ✅ 已收录 |
| Voyager: An Open-Ended Embodied Agent with Large Language Models | Wang et al., 2023 | 被批判文献（自生成技能库） | ✅ 已收录 |
| Improving Text Embeddings with Large Language Models (HyDE) | Gao et al., 2023 | 方法基础（查询生成相关） | ❌ 未收录 |
| Passage Re-ranking with BERT | Nogueira & Cho, 2019 | 方法基础（cross-encoder 重排） | ❌ 未收录 |
| Dense Passage Retrieval for Open-Domain Question Answering | Karpukhin et al., 2020 | 方法基础（dense retrieval） | ❌ 未收录 |
| Toolformer: Language Models Can Teach Themselves to Use Tools | Schick et al., 2023 | 被批判文献（工具调用范式） | ❌ 未收录 |
| Gorilla: Large Language Model Connected with Massive APIs | Patil et al., 2024 | 被批判文献（API 检索） | ❌ 未收录 |
| ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs | Qin et al., 2024 | 被批判文献（工具检索） | ❌ 未收录 |
| RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval | Sarthi et al., 2024 | 背景方法（长文档处理） | ❌ 未收录 |
| RepLLaMA & RankLLaMA: Language Model-based Retrieval and Reranking | Ma et al., 2024 | 被批判文献（LLM-based 重排） | ❌ 未收录 |
| SWE-bench: Can Language Models Resolve Real-World GitHub Issues? | Jimenez et al., 2024 | 背景 benchmark | ❌ 未收录 |

---

## 推荐阅读列表

### P0 必读（方法基础，知识库未收录）

- **Improving Text Embeddings with Large Language Models** (Gao et al., 2023 / HyDE)：SkillFlow 的查询生成步骤（任务 → 多维度 LLM 查询）在思路上与 HyDE 的假设文档生成相关，但 SkillFlow 生成的是结构化搜索查询而非假设技能文本；理解其区别对评估 SkillFlow 查询生成设计有参考价值。
- **Passage Re-ranking with BERT** (Nogueira & Cho, 2019)：SkillFlow 的两轮 cross-encoder 重排直接继承这一范式；阅读此论文可以理解 SkillFlow 设计选型的历史依据和性能预期。
- **Dense Passage Retrieval for Open-Domain Question Answering** (Karpukhin et al., 2020)：dense bi-encoder 检索的标准参考，理解 SkillFlow Stage 1 的理论基础。

### P1 重要（被批判文献，理解动机必读）

- **Gorilla: Large Language Model Connected with Massive APIs** (Patil et al., 2024)：与 SkillFlow 最直接可比的 tool retrieval 工作，但针对原子 API 签名而非 procedural skill bundle；对比两者定位可清晰理解 SkillFlow 的问题边界。
- **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs** (Qin et al., 2024)：16K API 规模的工具选择，与 SkillFlow 的 36K 技能库规模相近，但文档类型和检索挑战不同；是理解 SkillFlow 创新定位的重要对照。

### P2 建议（主要竞争基线 / 邻近方法）

- **RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval** (Sarthi et al., 2024)：论文提及此方法作为长文档处理方案的对照，SkillFlow 认为 skill 文档足够短而不需要递归摘要；验证此边界假设是否成立是值得探索的实验问题。
- **RepLLaMA & RankLLaMA** (Ma et al., 2024)：用大 LLM 初始化的重排器，精度更高但推理代价显著更大；SkillFlow 选择轻量 cross-encoder 是工程取舍的结果，此论文给出了精度上界参考。

### P3 参考（背景综述，选读）

- **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?** (Jimenez et al., 2024)：agent 编程 benchmark 的重要参考，理解 SkillsBench 和 Terminal-Bench 与主流 agent 评测的定位差异。
- **Toolformer: Language Models Can Teach Themselves to Use Tools** (Schick et al., 2023)：工具调用范式的代表作，理解 SkillFlow 技能增强与 function calling 范式的本质差异。

---

## mem0 知识库记录

- **问题域**：agent skill retrieval；large-scale community skill library；multi-stage IR for agent augmentation
- **方法关键词**：dense retrieval; cross-encoder reranking; LLM selection; multi-query generation; progressive candidate narrowing; SKILL.md; SkillsMP
- **数据集**：SkillsBench (87 tasks, 229 oracle skills); Terminal-Bench (89 tasks); 技能库 35,866 SKILL.md files from GitHub
- **性能基准**：SkillsBench Pass@1: 9.2% (no skills) → 16.4% (SkillFlow) → 19.5% (oracle); Terminal-Bench: 无显著增益（library coverage bottleneck）; 检索 R@1000=0.905, MRR: dense=0.553→+shallow=0.587→+deep=0.634
- **关联论文 ID**：arXiv:2602.12670 (SkillsBench), arXiv:2604.24594 (SRA), arXiv:2604.05333 (GoS), arXiv:2305.16291 (Voyager)
- **核心方法机制摘要**：四阶段流水线（dense→shallow cross-encoder→deep cross-encoder→LLM filter），逐级候选缩减（1000→100→10→≤5），多查询在检索阶段有益（M=5）但在重排阶段有害（M=1），BM25 对技能检索失效
- **推荐下一轮阅读线索**：(1) 可执行性感知重排（技能质量融入 scoring）；(2) 依赖感知技能检索（GoS 路线 + SkillFlow 规模）；(3) 动态交错检索（DRAGIN 触发机制 + skill 粒度）；(4) 库质量自动过滤爬取框架

---

## 跨论文联想补充

**与 GoS（arXiv:2604.05333）的互补关系**：GoS 解决 SkillsBench 1000 技能规模下的依赖感知检索（图传播扩散），SkillFlow 解决 36K community 库规模下的多阶段精筛。两者在检索层的设计思路互补：GoS 的 PPR 扩散假设有图结构先验，SkillFlow 的级联 cross-encoder 不需要图，只需文档内容。结合方向是：在 SkillFlow Stage 1（dense recall）之后，用 GoS 的依赖图扩散做补充候选恢复，再送入 SkillFlow Stage 2-4 精排。

**与 SRA（arXiv:2604.24594）的关系**：SRA 把技能增强拆为 retrieval → incorporation → application，并在 SRA-Bench 上诊断瓶颈在 skill loading（relevance-aware/need-aware）。SkillFlow 主要贡献对应 SRA 框架的第一阶段（retrieval），而 Terminal-Bench 的 null result 提示 skill quality 是比 retrieval 更基础的瓶颈，与 SRA 的 loading 瓶颈诊断形成互补视角。

---

## 附录：本地功能性复现结果

### 复现环境

- 操作系统：Windows
- Python：3.13.11（项目本地解释器 `.venv\Scripts\python.exe`）
- Dense Retriever：BAAI/bge-base-en-v1.5
- Shallow Reranker：cross-encoder/ms-marco-MiniLM-L-6-v2
- Deep Reranker：已下载但本次搜索未启用（与默认配置一致）
- 运行设备：CPU

### 本地语料与索引状态

- 本次索引构建使用本地 `data/skills/_metadata/index.json` 中可加载的 21,544 条技能记录。
- 索引产物已成功生成在 `projects/skill-flow/outputs/indices/bge-base/`，包括 `faiss.index`、`embeddings.npy`、`skill_ids.json`、`skill_descriptions.json`、`skill_contents.json`。
- 终端进度条显示 build-index 共处理 85 个 batch，总耗时约 4 分 07 秒。

### 搜索功能复现

以下测试使用默认检索管线：Dense Retriever + Shallow Cross-Encoder Reranker，返回 top-5 结果。

#### Query 1: write unit tests for FastAPI endpoints

- 耗时：1778.1 ms
- Top-5:
  1. `[1.8340] skillsmp/python-testing-3`
  2. `[1.4834] skillsmp/testing-patterns-22`
  3. `[1.4434] skillsmp/bknd-testing`
  4. `[1.1616] skillsmp/python-1`
  5. `[1.1149] skillsmp/unit-test-controller-layer`

#### Query 2: set up a Docker container for a Python web application

- 耗时：1346.3 ms
- Top-5:
  1. `[6.5890] skillsmp/docker-rocker`
  2. `[5.7030] skillsmp/docker-hub-toolkit-1`
  3. `[5.7030] skillsmp/docker-hub-toolkit`
  4. `[5.1637] skillsmp/docker-dev-env`
  5. `[4.0595] skillsmp/docker-10`

#### Query 3: run database migrations with SQLAlchemy

- 耗时：1339.8 ms
- Top-5:
  1. `[6.3202] skillsmp/fastapi-alembic-setup`
  2. `[6.3202] skillsmp/fastapi-alembic-setup-1`
  3. `[4.4740] skillsmp/backend-dev-1`
  4. `[4.2506] skillsmp/sqlalchemy-orm`
  5. `[3.9351] skillsmp/database-migration-runner`

### 复现结论

- 从功能角度看，本地 SkillFlow 流水线已经跑通：语料加载、FAISS 建索引、Dense 检索、Shallow Reranker 精排均可执行。
- 3 个示例查询均返回了语义上合理的技能候选，且 top-5 结果中未观察到“因缺少 content 导致 shallow reranker 全面退化”的现象。
- 查询延迟约 1.3 到 1.8 秒，说明在 CPU 条件下，至少前两阶段检索已经具备可用的工程验证价值。

### 与论文表格结果的差异说明

- 论文 Table 3 / Table 4 中的 MRR、R@10、P@10、Hit@10 等定量结果依赖 `integration/skillsbench/tasks` 的官方评测任务。
- 当前开源仓库未包含该目录中的任务数据，因此无法直接复现论文报告的 `Dense=0.553 -> +Shallow=0.587 -> +Deep=0.634` 等指标。
- 因而本次复现应视为“功能性复现”而非“官方 benchmark 数值复现”：证明了 SkillFlow 的核心检索流水线可以在本地成功运行，但尚不能对论文中的最终 benchmark 数值做逐项对齐。

---

## 8. 外部 skillflow.zip 数据核验与可执行实验

### 外部数据核验结果

- 外部数据目录：`D:\LocalEnvironments\Datasets\skillflow`
- 顶层 `skills/skillsmp` 目录数：35,865
- 顶层 `skills/skills-no-letta` 目录数：35,865
- zip 清单中 `skills/skillsmp` 的顶层唯一条目数：35,865
- 结论：这份外部数据包与此前的核验一致，真实可用 skill 数是 35,865，不是 35,866

### 外部索引构建结果

- 构建命令：`skill_flow.cli build-index --corpus-path D:\LocalEnvironments\Datasets\skillflow\skills --output-dir projects/skill-flow/outputs/indices/external-skillflow`
- 构建耗时：约 24 分 37 秒
- 最终产物：`35866 vectors, dim=768`
- 可加载内容数：35,487
- 说明：有一部分顶层目录仅含元数据或索引占位，没有可读的 `SKILL.md`，因此在内容导出阶段被跳过；这与构建日志中的 `SKILL.md not found` 告警一致

### 外部语料检索实验

本次在外部索引上跑的是“可执行的 retrieval-only 实验”，配置为 `default_eval.json`，关闭 selector，仅保留 dense retrieval + cross-encoder rerank。这样可以避免 selector 阶段把大候选集发给 LLM 时触发 8K token 上限。
---

## 附录9：完整 SkillsBench 实验结果（94任务，外部 skillflow 数据集）

### 实验配置

| 参数 | 值 |
|------|-----|
| 配置文件 | `skill_flow/config/default_eval_external.json` |
| 检索模型（Retriever） | BAAI/bge-base-en-v1.5 |
| 重排模型（Reranker） | BAAI/bge-reranker-v2-m3 |
| 检索候选数（top_k） | 300 |
| 重排输入数（input_top_k） | 200 |
| 重排 batch_size | 8 |
| max_content_chars | 384 |
| Deep Reranker | 禁用 |
| Selector | 禁用 |
| 外部索引 | `outputs/indices/external-skillflow`（35,866 vectors，dim=768） |
| 评估任务数 | 94（可评估） |
| 注入 Oracle Skills 数 | 249 |
| 输出目录 | `outputs/pipeline/skillsbench_external_full/` |
| 运行时间 | 2026-05-09 13:16 – 13:31（约15分钟） |
| GPU | NVIDIA RTX 4060 Ti（CUDA 12.4，PyTorch 2.6.0+cu124） |

### 检索层评估指标

> 与论文 Table 4 对比（论文使用内部 skillflow 数据集，本实验使用外部公开 skillflow.zip 数据集）

#### Stage 1：Dense Only（FAISS 密集检索，top_k=300）

| 指标 | 本实验结果 | 论文 Table 4（Dense only） | 差异 |
|------|-----------|--------------------------|------|
| MRR | **0.583** | 0.553 | +0.030 |
| R@10 | **0.517** | 0.477 | +0.040 |
| P@10 | **0.134** | 0.121 | +0.013 |
| Hit@10 | **0.734** | 0.713 | +0.021 |

其他 Recall@k 数据：

| k | R@k |
|---|-----|
| 1 | 0.238 |
| 2 | 0.328 |
| 5 | 0.455 |
| 10 | 0.517 |
| 50 | 0.634 |
| 100 | 0.667 |

检索延迟：均值 **19.2 ms/task**，中位数 19.1 ms/task，94任务总计 1.8 s

#### Stage 2：+Shallow Reranker（bge-reranker-v2-m3 对 top-200 重排）

| 指标 | 本实验结果 | 论文 Table 4（+Shallow） | 差异 |
|------|-----------|------------------------|------|
| MRR | **0.484** | 0.587 | -0.103 |
| R@10 | **0.446** | 0.520 | -0.074 |
| P@10 | **0.112** | 0.130 | -0.018 |
| Hit@10 | **0.660** | 0.724 | -0.064 |

其他 Recall@k 数据：

| k | R@k |
|---|-----|
| 1 | 0.174 |
| 2 | 0.268 |
| 5 | 0.337 |
| 10 | 0.446 |
| 50 | 0.646 |
| 100 | 0.719 |

重排延迟：均值 **9,458 ms/task**（中位数 7,649 ms），94任务重排总计约 889 s（≈14.8 min）

### 实验结论与分析

1. **Dense 检索超过论文基线**：本实验 Stage 1（Dense only）在 MRR、R@10、P@10、Hit@10 上均优于论文报告值，说明 BAAI/bge-base-en-v1.5 + FAISS 的组合在外部 skillflow 数据集上具有良好的检索性能。

2. **Shallow Reranker 未改善性能**：Stage 2 所有指标均低于 Stage 1，与论文中 Reranker 提升效果相反。分析可能原因：
   - **数据分布差异**：外部 skillflow.zip 与论文作者使用的内部数据集可能存在 skill 描述风格差异，reranker（bge-reranker-v2-m3）在此分布上泛化性不如 dense retriever；
   - **内容截断影响**：`max_content_chars=384` 对于较长 skill 描述存在截断，可能损失对 reranker 判断有利的关键信息；
   - **召回集变化**：Stage 1 候选 300 → Reranker 输入 200，部分 oracle skill 在进入重排前已被筛除，而 R@50/R@100 在 Stage 2 反而更高（0.646 vs 0.634，0.719 vs 0.667），说明 reranker 在更宽泛的范围内效果更好，但 top-10 排序能力有所下降。

3. **端到端 Pass@k 未评估**：SkillsBench 的 Pass@1/Pass@3 需要实际执行 agent + 工具链，当前环境不可行，仅报告检索层指标。

4. **实验完整性**：全部 94 个可评估任务均完成，249 个 oracle skill 全部注入检索索引，0 个任务跳过，结果具有完整性。

### 输出文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `eval-stage1-retriever.json` | 15.2 MB | Stage 1 逐任务评估结果 |
| `pipeline-stage1-retriever.json` | 3.8 MB | Stage 1 原始检索结果 |
| `eval-stage2-reranker.json` | 10.3 MB | Stage 2 逐任务评估结果 |
| `pipeline-stage2-reranker.json` | 2.6 MB | Stage 2 重排后结果 |
| `result_cache.json` | 0.7 MB | 结果缓存 |
| `latency_summary.json` | 10 KB | 各阶段延迟统计 |