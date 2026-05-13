---
name: 论文PDF获取
description: '负责论文的去重检查、目录路径决策与 PDF 下载（支持 arXiv / DBLP / Google Scholar / DOI-SciHub）。仅处理物料获取阶段，不进行格式转换或内容分析。'
argument-hint: '论文标题、arXiv ID、DOI、doi.org 链接或下载 URL'
---

# 论文PDF获取技能

负责将一篇论文从"标识符"推进到"PDF 就绪"状态，为后续 `pdf-conversion` 技能提供输入。

## Definition Of Done (DoD)

仅当以下所有检查均通过时，本 Skill 的工作才算完成：

- `papers/.raw/<filename>.pdf` 存在且大小 > 0。
- `papers/.raw/<filename>.meta.json` 已创建，包含 title、main_tag、tags 字段。
- `index.csv` 中已新增一行，状态为 **已拉取**，路径为 `papers/.raw/<filename>.pdf`。
- 若论文已存在（去重命中），则输出 "existing" 并停止，不重复下载。

## 使用场景

- 通过标题、arXiv ID 或 DOI 下载一篇新论文。
- 通过 `https://doi.org/...` 或裸 DOI 解析并尝试 Sci-Hub 下载。
- 检查仓库中是否已存在某篇论文（去重查询）。

## 目标制品

```text
papers/.raw/
  <filename>.pdf          ← 下载的 PDF（文件名为标题的下划线化形式）
  <filename>.meta.json    ← 元数据（title, main_tag, tags, doi/arxiv_id）
```

> 注意：PDF **不直接放入** `papers/<main_tag>/<paper_name>/`，统一在 `.raw/` 中暂存。转换完成后由 `pdf-conversion` 技能将 PDF 移走并删除。

## 目录规范

遵循 [papers-paths.instructions.md](../../instructions/papers-paths.instructions.md) 中的路径规则。
- `main_tag` 优先按**问题类型**选取，其次按方法族。
- `paper_name` 使用论文标题的精简版本（去除冠词、缩短至合理长度）。

## Workflow

### 步骤 1：去重检查
- **输入**：论文标题 / DOI / arXiv ID。
- **操作**：在 `papers/` 目录搜索已有文件夹或 PDF 元数据匹配。
- **输出**：`"existing"` 或 `"new"` 决策。若为 `"existing"`，停止流程并报告路径。

### 步骤 2：路径决策
- **输入**：论文问题定义与方法族。
- **操作**：按 `papers-paths.instructions.md` 选择 `main_tag`，确定 `paper_name`（标题精简化、下划线化）。
- **输出**：确定的 `main_tag` 与 `paper_name`（**不创建**目标目录，目录由 `pdf-conversion` 负责创建）。

### 步骤 3：PDF 获取
- **输入**：来源 URL / arXiv ID / DOI。
- **操作**：依次尝试以下来源，取第一个成功结果：
  1. arXiv：[`./scripts/arxiv_fetch_paper.py`](./scripts/arxiv_fetch_paper.py)
  2. DBLP：[`./scripts/dblp_fetch_paper.py`](./scripts/dblp_fetch_paper.py)
  3. Google Scholar：[`./scripts/gc_fetch_paper.py`](./scripts/gc_fetch_paper.py)
  4. DOI / Sci-Hub：使用 DOI（支持 `doi.org` 前缀）构造 `https://www.sci-hub.su/<DOI>` 并下载 PDF（若可访问）
- **输出**：`papers/.raw/<filename>.pdf` 存在且大小 > 0。若所有来源失败，报告失败原因并终止。

### 步骤 4：登记索引并移交
- **输入**：下载成功的 PDF 文件名与元数据。
- **操作**：
  1. 在 `papers/.raw/` 写入 `<filename>.meta.json`（title、main_tag、tags）。
  2. 调用 CLI 将论文登记为"已拉取"：
     ```bash
     cd papers/.db
     python paperdb_cli.py index add-fetched "<title>" "<main_tag>" "<filename>.pdf" "<tags>"
     ```
- **输出**：`index.csv` 中新增状态为 **已拉取** 的条目；移交 `pdf-conversion` 技能继续处理。

## 错误处理

| 失败场景 | 行为 |
|---|---|
| PDF 获取失败（所有来源均失败） | 报告失败来源，终止流程，不写入 `.raw/` |
| 去重命中（index.csv 中已存在） | 输出现有路径和状态，停止流程（不覆盖） |
| Sci-Hub 未命中或不可访问 | 记录该来源失败，继续其他来源或给出手动下载建议 |
| meta.json 写入失败 | 记录警告，不阻塞主流程，但需手动补写 |
