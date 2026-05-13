---
name: research-code-validation
description: Given a paper or idea, implement/reproduce with a user-specified programming language and validate on real datasets. Produces runnable code, experiment configs, metrics, and reproducible reports.
---

# Research Code Validation

## Overview

Use this skill when the user asks for:
- 论文复现（reproduction）
- 想法实现（idea-to-prototype）
- 指定编程语言实现（Python / C++ / Rust / Java / JS / 其他）
- 在**真实数据集**上进行实验验证

This skill is intended to be orchestrated by:
- `.github/agents/research-code-validation.agent.md`

## Hard Constraints

1. **必须使用用户指定编程语言**。如果现有代码不是该语言，需明确改造范围与兼容边界。
2. **必须使用真实数据集**（公开可下载或用户本地真实数据），禁止仅用 toy data 冒充验证。
3. **必须可复现**：固定随机种子、记录依赖版本、保存运行配置与结果。
4. **必须可执行验收**：至少完成一次端到端训练/推理/评测流程并给出指标。
5. 环境变量统一从 `.github/agents/assets/.env` 读取，不硬编码凭证。
6. **所有生成代码必须遵守 `CODE_CONVENTIONS.md`**（同目录），包含简洁优先、外科手术式改动、内存/I-O 管控、异常处理规约。

## Workspace Environment Memo（当前会话确认）

详细步骤与排障手册见：
- `ENV_STARMIE_GPU_KMEANS.md`

以下约定来自当前 GeneralExplorer / `projects/starmie` 的实际运行环境；后续复现与验证优先沿用这些设置，避免重复踩坑：

- **工作区根目录**：`/home/wizard/projects/GeneralExplorer`
- **当前项目目录**：`/home/wizard/projects/GeneralExplorer/projects/starmie`
- **推荐虚拟环境**：`/home/wizard/projects/GeneralExplorer/.venv-unix`
- **虚拟环境来源**：由 `uv` 管理，`pyvenv.cfg` 显示 Python 3.12.13；某些情况下 venv 内部不自带 `pip`
- **pip 安装策略**：
   - 若 `.venv-unix` 内缺少 `pip`，可先激活 venv，再用 `python3 -m venv /home/wizard/projects/GeneralExplorer/.venv-unix` 重新初始化，随后升级 `pip`
   - 若必须调用 uv 管理的解释器自带 `pip`，需要显式使用 `--break-system-packages`；否则会触发 PEP 668 的 `externally-managed-environment`
- **GPU KMeans 后端**：当前 `starmie_rsq_utils.py` 已从 FAISS 切换为 **CuPy 优先 + NumPy 兜底** 的 KMeans 实现；不要为 KMeans 重新引入 FAISS
- **CuPy 相关包**：GPU 路径通常需要 `cupy-cuda12x` 以及配套的 `nvidia-*` wheel（例如 `nvidia-curand-cu12`、`nvidia-cuda-nvrtc-cu12`、`nvidia-cublas-cu12`）
- **CUDA 路径**：若 CuPy 不能自动识别 CUDA，可通过 `CUDA_PATH` 与 `LD_LIBRARY_PATH` 指向 venv 的 `site-packages/nvidia/.../lib` 目录
- **失败回退原则**：GPU 库不全或自动发现失败时，KMeans 必须回退到 NumPy 实现，保证流程可继续
- **依赖可用性检查**：在运行实验前，优先确认 `numpy`、`cupy`、`mlflow`、`hnswlib`、`munkres`、`tqdm` 可导入
- **数据集根目录**：真实数据集统一使用 `/mnt/f/Datasets/TableDiscovery/`；不要再写 `/f/Datasets/TableDiscovery/`
- **路径适配**：若原始脚本仍硬编码 `data/` 或其他旧目录，优先修改代码的数据加载层，而不是改动真实数据文件结构

## Input Contract

Minimum required input:
- 任务来源：`paper`（标题/DOI/arXiv/路径）或 `idea`（文字说明）
- 编程语言：如 `python` / `rust` / `cpp` / `java` / `typescript`
- 目标任务：分类/检索/生成/检测/回归等

Optional but preferred input:
- 待复现代码仓库（Git URL / 本地仓库路径 / 指定分支或commit）

Recommended input:
- 目标数据集（优先用户指定；否则由 agent 提供候选并确认）
- 评测指标（accuracy/F1/AUC/mAP/R@K/BLEU/ROUGE/latency/memory 等）
- 资源预算（CPU/GPU、最大时长、内存）

## Workflow

### 阶段 0：环境与约束加载（必做）

1. 加载 `.env`：
   - `set -a && source .github/agents/assets/.env && set +a`
2. 确认语言工具链是否可用（如 python/rustc/g++/javac/node）。
3. 确认实验预算（时间、硬件、数据规模）。
4. 确认输出根目录 `{project_path}/eval_results/` 可写；不存在则创建。

### 阶段 1：任务定义与可复现计划

1. 将论文/想法转为可执行实验问题定义：
   - 输入、输出、损失/目标、约束、评估指标。
2. 形成最小可行实验（MVE）计划：
   - baseline、核心改动、消融项（如适用）。

### 阶段 1.5：指标确认（运行前必做，任何代码执行之前）

1. 从论文原文或 `paper.report.md` 提取所有已汇报的实验指标，列出清单，例如：
   - 质量：Recall@K、Precision@K、F1、AUC、NDCG、mAP、BLEU、ROUGE
   - 效率：Latency（p50/p90/p99 ms）、Throughput（QPS）
   - 资源：Memory Usage（MB）、GPU Utilization
2. 向用户确认：
   - 论文原文汇报了哪些指标（确认避免遗漏）
   - 本次验证的**主指标**与**次指标**分别是什么
   - 评测粒度（全集 / top-K / 分桶 / 特定切片）
3. 用户确认（或合理推断）后**锁定指标列表**，写入实验配置，才进入实现阶段。

> **阻塞规则**：指标列表未确认前，禁止启动任何代码执行或实验命令。

### 阶段 2：仓库优先复现路径（用户提供代码仓库时必做）

1. 拉取或定位用户提供的原始仓库代码。
2. 优先尝试“最小改动跑通”：
   - 依赖安装与版本对齐
   - 本地路径修正（数据路径、输出路径、缓存路径）
   - 配置修正（设备、batch size、端口、文件系统差异）
3. 仅在必要时改动原逻辑；改动需记录为“本地适配补丁”。
4. 若用户给定 commit/tag，必须固定到该版本执行。

### 阶段 2：真实数据集准备（必做）

1. 优先使用用户指定数据集。
2. 若未指定，给出 2-3 个真实数据集候选并按任务匹配选择。
3. 完成数据下载/校验/切分（train/val/test）并记录：
   - 数据来源、版本、许可、样本数、预处理。

### 阶段 4：按指定语言实现

1. 使用指定语言实现数据管道、模型/算法、训练或推理逻辑。
2. 生成可运行入口（CLI/script）与配置文件。
3. 固定随机种子并记录依赖版本。

> 若已提供仓库且语言一致，本阶段默认以"仓库适配 + 运行验证"为主，而非从零重写。

> **代码质量**：所有生成代码必须遵守同目录 `CODE_CONVENTIONS.md` 全部规约（简洁优先、外科手术式改动、单次读取、流式加载、禁止空 catch）。

### 阶段 5：实验执行与验证（必做）

1. 在真实数据集上运行至少 1 次完整实验。
2. 按锁定的指标列表采集结果（含单位与测量粒度）：
   - 质量指标、时延、吞吐、显存/内存。
3. 将所有输出写入 `eval_results/{dataset_name}/`（结构见 `CODE_CONVENTIONS.md` 第 6 条）：
   - `metrics.json`、`run_config.yaml`、`checkpoints/`（索引/权重/缓存）、`logs/`
4. 若失败，必须提供可复现错误与修复路径，再重试。

### 阶段 6：结果分析与交付

1. 输出结果报告（含配置、指标、日志、失败案例）。
2. 给出“是否验证成功”的结论与置信度。
3. 标注局限与下一步改进方向。

## Required Artifacts

At minimum, produce:

1. 可运行代码入口（与指定语言一致）
2. `eval_results/{dataset_name}/run_config.yaml`（含 seed / 超参 / 数据路径）
3. 数据准备说明（真实数据来源与处理）
4. `eval_results/{dataset_name}/metrics.json` 与 `logs/`
5. `eval_results/{dataset_name}/checkpoints/`（索引、权重、缓存等可复用产物）
6. 复现实验报告（markdown）
7. 若使用用户仓库：本地适配记录（修改文件、修改原因、与原仓库差异）

## Acceptance Gate (Blocking)

Only pass when all checks succeed:

1. 语言一致性：实现语言 == 用户指定语言
2. 真实数据集：非 toy，且来源可追溯
3. 可执行性：命令可运行，退出码成功
4. 指标确认：运行前已锁定指标列表（来自论文或用户确认）
5. 指标产出：锁定的所有主指标均有数值输出（含单位）
6. 可复现性：seed、依赖、配置已记录
7. 代码质量：无空 catch、无重复文件读取、内存管控符合规范；资源不足时停止并交还用户而非自行降参
8. 输出结构：`eval_results/{dataset_name}/` 下存在 `metrics.json`、`run_config.yaml`、`checkpoints/`、`logs/`
9. 若用户提供仓库：仓库已拉取并在本地环境完成路径/配置适配后可跑通

If any check fails, block delivery and return exact failure reasons + next actions.

## Output Contract

Return a concise execution report with:

- 任务来源（paper/idea）与目标
- 指定语言与实现范围
- 数据集信息（名称/版本/样本数/来源）
- 运行命令与环境
- 原始仓库与版本（若提供）：`repo url/path + branch/commit`
- 本地适配改动摘要（若有）
- 确认的指标列表（来源：论文 / 用户指定，含单位与粒度）
- 指标结果（主指标 + 次指标，数值与单位）
- 代码质量自查：无冗余 catch / 无重复读取 / 内存管控（pass/fail per item）
- 验收结论（pass/fail）
- 失败项与修复建议（若有）
