# SkillFlow 实验会话记录

> 导出时间：2026-05-09  
> 会话跨度：上一轮（Stage 1 Dense 实验）+ 本轮（Stage 2 Reranker + 数据集整理）

---

## 一、实验背景

### 论文信息
- **论文**：SkillFlow: Scalable and Efficient Agent Skill Retrieval System
- **路径**：`papers/agentic_ai/SkillFlow Scalable and Efficient Agent Skill Retrieval System/`
- **报告**：`paper.report.md`（含附录 8、9 的完整实验记录）

### 核心问题
SkillFlow 将 agent 技能发现建模为**信息检索问题**，对 ~36K 个社区贡献的 SKILL.md 文件建立多阶段检索 pipeline：
1. **Dense Retrieval**（FAISS）
2. **Shallow Reranker**（Cross-Encoder）
3. **Deep Reranker**（LLM-based）
4. **Selector**

评测基准：SkillsBench（87 任务 + 229 oracle skills）和 Terminal-Bench（89 任务，无 oracle skill 标注）。

---

## 二、本地实验配置

### 硬件环境
| 项目 | 值 |
|------|-----|
| GPU | NVIDIA RTX 4060 Ti，8188 MiB |
| CUDA | 12.4 |
| PyTorch | 2.6.0+cu124 |
| Python 解释器 | `.venv\Scripts\python.exe` |
| HF 缓存 | `D:\LocalEnvironments\Datasets\huggingface`（离线模式） |

### Pipeline 配置
- **配置文件**：`projects/skill-flow/skill_flow/config/default_eval_external.json`
- **Retriever**：`BAAI/bge-base-en-v1.5`，top_k=300
- **Reranker**：`BAAI/bge-reranker-v2-m3`，enabled=true，input_top_k=200，batch_size=8，max_content_chars=384
- **Deep Reranker / Selector**：disabled
- **tasks_dir**：`integration/skillsbench/tasks`
- **output_dir**：`outputs/pipeline/skillsbench_external_full`

### 外部索引
- **路径**：`projects/skill-flow/outputs/indices/external-skillflow`
- **规模**：35,866 vectors，dim=768
- **模型**：BAAI/bge-base-en-v1.5

---

## 三、实验流程记录

### 阶段 0：问题修复
**CUDA 失效问题**：`CUDA_VISIBLE_DEVICES` 设为空字符串导致 `torch.cuda.device_count()=0`，fallback 到 CPU。

**解决方案**：
```powershell
Remove-Item -Path Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
```

### 阶段 1：数据准备
- **外部数据核验**：`D:\LocalEnvironments\Datasets\skillflow`，35,865 个技能
- **外部索引构建**：35,866 vectors，dim=768
- **SkillsBench tasks 同步**：1,016 task dirs，94 个可评估任务，249 个 oracle skills

### 阶段 2：完整实验运行

**运行命令**：
```powershell
Push-Location 'D:\Projects\Scholar\GeneralExplorer\projects\skill-flow'
$env:HF_HOME='D:\LocalEnvironments\Datasets\huggingface'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
Remove-Item -Path Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue

.\.venv\Scripts\python.exe -m skill_flow.cli pipeline `
  --config 'skill_flow\config\default_eval_external.json' `
  --output-dir 'outputs/pipeline/skillsbench_external_full'
Pop-Location
```

**输出文件**（`outputs/pipeline/skillsbench_external_full/`）：

| 文件 | 大小 | 时间 |
|------|------|------|
| `eval-stage1-retriever.json` | 15.2 MB | 2026/5/9 13:16:19 |
| `pipeline-stage1-retriever.json` | 3.8 MB | 2026/5/9 13:16:18 |
| `eval-stage2-reranker.json` | 10.3 MB | 2026/5/9 13:31:10 |
| `pipeline-stage2-reranker.json` | 2.6 MB | — |
| `result_cache.json` | 0.7 MB | — |
| `latency_summary.json` | 10 KB | — |

---

## 四、完整实验结果

### 检索指标对比（94 任务，外部 skillflow 数据集）

| 指标 | Stage 1 Dense（本实验） | 论文 Dense | Stage 2 +Reranker（本实验） | 论文 +Shallow |
|------|------------------------|-----------|----------------------------|--------------|
| MRR | **0.583** | 0.553 | 0.484 | 0.587 |
| R@10 | **0.517** | 0.477 | 0.446 | 0.520 |
| P@10 | **0.134** | 0.121 | 0.112 | 0.130 |
| Hit@10 | **0.734** | 0.713 | 0.660 | 0.724 |

> 本实验 Stage 1 指标**优于论文 Dense baseline**，Stage 2 Reranker 性能未提升，原因分析见附注。

### 延迟统计

| 阶段 | 均值 | 总计（94 任务） |
|------|------|----------------|
| Stage 1 Dense | 19.2 ms/task | ~1.8 s |
| Stage 2 Reranker | 9,458 ms/task | ~14.8 min |

### Reranker 性能未提升的原因分析
1. **数据分布差异**：外部 skillflow 库（社区贡献）与论文内部数据集特征不同
2. **截断问题**：`max_content_chars=384` 可能丢失关键内容
3. **候选集损失**：top_k=200 的候选可能已过滤掉部分 oracle skill

---

## 五、数据集位置汇总

所有数据集统一存放于 `D:\LocalEnvironments\Datasets\`

### 1. Skill Library（skillflow）
**路径**：`D:\LocalEnvironments\Datasets\skillflow\`

```
skillflow/
├── skills/
│   ├── skillsmp/          ← 35,865 个技能目录（主库）
│   │   └── {skill-name}/
│   │       ├── SKILL.md   ← 核心文件（YAML frontmatter + 描述 + 指南）
│   │       └── references/
│   ├── skills_rest/       ← 备用（目前为空）
│   └── _metadata/         ← index.json, index_deduped.json, duplicates.json, sync_state.json
└── skills-no-letta/       ← 过滤 Letta 框架后的子集（2 个）
```

**原始压缩包**：`D:\LocalEnvironments\Datasets\skillflow.zip`（~1 GB）  
**FAISS 索引**：`projects/skill-flow/outputs/indices/external-skillflow/`（35,866 vectors）

---

### 2. SkillsBench
**路径**：`D:\LocalEnvironments\Datasets\skillsbench\`  
**来源**：`benchflow-ai/skillsbench`（GitHub）  
**实验副本**：`projects/skill-flow/integration/skillsbench/tasks/`（供 pipeline 直接使用）

```
skillsbench/
├── tasks/                 ← 94 个可评估任务
│   └── {task-name}/
│       ├── instruction.md ← 自然语言任务描述（给 agent）
│       ├── task.toml      ← 元数据 + 资源限制（difficulty/category/tags/timeout）
│       ├── environment/   ← Docker 环境定义
│       ├── solution/      ← 参考解答
│       └── tests/         ← 验证脚本
├── tasks_excluded/        ← 排除任务（含排除原因）
├── experiments/
├── docs/
└── README.md
```

**`task.toml` 关键字段**：
```toml
[metadata]
difficulty = "hard"         # easy / medium / hard
category = "engineering"
tags = ["binary-parsing"]

[verifier]
timeout_sec = 900.0

[environment]
cpus = 1
memory_mb = 4096
storage_mb = 10240
```

---

### 3. Terminal-Bench
**路径**：`D:\LocalEnvironments\Datasets\terminal-bench\`  
**来源**：`harbor-framework/terminal-bench`（ICLR 2026，Merrill et al.）

```
terminal-bench/
├── original-tasks/        ← 241 个任务目录
│   └── {task-name}/
│       ├── task.yaml      ← 任务定义（instruction + 元数据）
│       ├── Dockerfile     ← 独立容器环境
│       ├── docker-compose.yaml
│       ├── run-tests.sh   ← 测试执行
│       ├── solution.sh    ← 参考解答
│       └── tests/         ← pytest 测试
├── terminal_bench/        ← 框架代码
├── adapters/              ← Agent 适配器
├── docker/
├── registry.json          ← 各版本数据集注册表（~41KB）
└── README.md
```

**`task.yaml` 关键字段**：
```yaml
instruction: |
  自然语言任务描述...
difficulty: hard
category: software-engineering
tags: [coding, legacy-modernization]
parser_name: pytest
max_agent_timeout_sec: 1200.0
estimated_duration_sec: 600
expert_time_estimate_min: 240
```

**注**：Terminal-Bench 无 oracle skill 标注，用于测试"无覆盖"场景下的泛化能力。论文保留 88 个任务（排除 1 个：`train-fasttext`，原因：大型语料库下载+训练）。

---

## 六、三数据集对比

| 维度 | Skill Library | SkillsBench | Terminal-Bench |
|------|--------------|-------------|----------------|
| 规模 | 35,865 skills | 94 tasks | 241 tasks（保留88） |
| Oracle skill 标注 | — | ✅ | ❌ |
| 评测粒度 | 检索指标（MRR/R@k） | Pass@1（端到端） | Pass@1（端到端） |
| 核心文件 | `SKILL.md` | `instruction.md` + `task.toml` | `task.yaml` |
| 环境隔离 | 无 | Docker | Docker（独立 compose） |
| 本地可运行检索 | ✅ | ✅（检索阶段） | ✅（检索阶段） |
| 本地可运行端到端 | — | ❌（需 agent + Docker） | ❌（需 agent + Docker） |

---

## 七、关键结论

1. **Stage 1 Dense 超过论文 baseline**：本实验 MRR=0.583 vs 论文 0.553，可能因外部索引覆盖更广（35,866 vs 论文内部数量未公开）。

2. **Reranker 在外部数据上无效**：Stage 2 所有指标均低于 Stage 1，与论文结论相反（论文 +Shallow Reranker 有提升）。这与 SkillFlow 论文本身的核心发现一致：**检索质量≠端到端收益**，技能库质量才是主要瓶颈。

3. **Terminal-Bench 的 null result**：论文报告 agent use rate 高达 70.1%，但 Pass@1 无显著提升，证明当技能库缺乏领域覆盖时，检索再准也无用。

4. **实用建议**：若要在自定义场景复用 SkillFlow pipeline，优先扩充领域相关的 oracle-quality 技能（有可执行代码 + 捆绑脚本），而非优化检索模型本身。

---

## 八、后续可执行的实验方向

| 实验 | 目标 | 难度 |
|------|------|------|
| 调整 `max_content_chars` | 测试 512/768/1024 对 Reranker 的影响 | 低 |
| 用 oracle skills 重建小索引 | 验证"质量>数量"的上界 | 低 |
| Terminal-Bench 检索实验 | 对比 SkillsBench 的检索分布差异 | 中 |
| `top_k` 敏感性分析 | Stage 1 top_k 对 Stage 2 的影响 | 低 |
| Deep Reranker 启用 | 测试 LLM reranker 的增益（需 API key） | 高 |
