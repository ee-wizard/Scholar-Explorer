---
name: PDF 转换
description: '负责将论文 PDF 转换为结构化 Markdown 制品（paper.raw.md / paper.md / paper.references.md），并进行图像后处理。当前优先支持 PDF -> Markdown。'
argument-hint: 'papers/.raw/<filename>.pdf 路径，或论文标题（在 .raw/ 中查找对应 PDF）'
---

# PDF 转换技能

负责将 `paper.pdf` 转换为可分析的 Markdown 制品，为 `paper-analyst` 提供标准输入。

## 前置条件

- `papers/.raw/<filename>.pdf` 存在且大小 > 0（来自 `paper-fetcher` 输出）。
- 对应的 `papers/.raw/<filename>.meta.json` 存在，可读取 `main_tag` 和 `paper_name`。
- 目标目录 `papers/<main_tag>/<paper_name>/` 将由本技能创建。
- 全部环境变量已从 `.github/agents/assets/.env` 加载（由 Agent 第零阶段 Step B 完成，或本技能单独调用时自行执行 `set -a && source .github/agents/assets/.env && set +a`）。

## Definition Of Done (DoD)

仅当以下所有检查均通过时，本 Skill 的工作才算完成：

- `paper.raw.md` 存在且非空（MinerU 原始输出）。
- `paper.md` 存在且非空（清洗后的主体内容）。
- `paper.references.md` 存在且非空（参考文献列表）。
- 若 `images/` 目录存在，则图像已压缩，或明确记录跳过原因。
- `index.csv` 中该论文状态已更新为 **已转换**。
- `papers/.raw/<filename>.pdf` 及对应 `.meta.json` 已从 `.raw/` 目录删除。

## 目标制品

```text
papers/<main_tag>/<paper_name>/
  paper.raw.md
  paper.md
  paper.references.md
  images/           ← 若转换过程提取出图片
```

## Workflow

### 步骤 1：输入校验与路径决策
- **输入**：`papers/.raw/<filename>.pdf` 路径或论文标题。
- **操作**：
  1. 读取 `papers/.raw/<filename>.meta.json` 获取 main_tag 和 paper_name。
  2. 确认目标目录 `papers/<main_tag>/<paper_name>/` 并创建。
  3. 将 PDF 复制到目标目录（命名为 `paper.pdf`）。
- **输出**：有效目标目录 + `paper.pdf` 就绪。

### 步骤 2：PDF -> Markdown 转换
- **输入**：`paper.pdf` + 环境变量 `MINER_API_TOKEN`（已由前置条件加载）。
- **操作**：运行 [`./scripts/minerU_convert.py`](./scripts/minerU_convert.py)。
- **输出**：`paper.raw.md`、`paper.md`、`paper.references.md`。

### 步骤 3：图像后处理
- **输入**：`images/`（若存在）。
- **操作**：压缩图像（Pillow 或等效方式），控制体积并保留可读性。
- **输出**：优化后图像或跳过说明。

### 步骤 4：状态更新与 .raw 清理
- **输入**：转换成功的制品路径。
- **操作**：
  1. 校验 `paper.raw.md`、`paper.md`、`paper.references.md` 存在且非空。
  2. 调用 CLI 更新状态为"已转换"，**同时自动删除** `papers/.raw/<filename>.pdf` 和对应的 `.meta.json`：
     ```bash
     cd papers/.db
     python paperdb_cli.py index mark-converted "<title>" "papers/<main_tag>/<paper_name>"
     ```
- **输出**：`index.csv` 中该论文状态更新为 **已转换**；`.raw/` 中对应 PDF 已删除；可进入 `paper-analyst`。

## 错误处理

| 失败场景 | 行为 |
|---|---|
| `.raw/<filename>.pdf` 不存在/为空 | 立即失败，提示先调用 `paper-fetcher` |
| `.meta.json` 缺失 | 尝试从 `index.csv` 反查元数据；若无则要求补写 meta 后重试 |
| MinerU 转换失败 | 记录错误，**不删除** `.raw/` 中的 PDF（保留重试来源） |
| `paper.references.md` 为空 | 标记部分失败，不更新状态，要求重试或人工补全 |
| 图像压缩失败 | 记录跳过原因，不阻塞 Markdown 主流程 |
