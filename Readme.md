

<div align="center">
  <img src="./asserts/banner.png" alt="GeneralExplorer Banner" style="max-width: 100%; margin: 24px 0; border-radius: 12px; box-shadow: 0 2px 16px #0002;" />
  <h1>GeneralExplorer</h1>
  <p> <b>智能科研论文自动拉取与分析 Agent</b> </p>
  <p> <em>一站式学术研究自动化管道</em> </p>
  <p>
    <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python" />
    <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%28partial%29-lightgrey" />
    <img src="https://img.shields.io/badge/License-Apache%202.0-green" />
    <img src="https://img.shields.io/badge/Agent-Auto-blueviolet" />
  </p>
  <!-- Banner 已上移为主 LOGO -->
</div>

---



<div align="center" style="margin: 18px 0;">
  <div style="display:flex; justify-content:center; align-items:center; gap:0; margin:0 auto; width:100%;">
    <span style="display:inline-flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; width:86px; height:86px; border:1px solid #bfdbfe; border-radius:14px; background:#eff6ff; padding:4px; vertical-align:middle; box-sizing:border-box;">
      <div style="font-size:18px; line-height:1.1;">🔍</div>
      <div style="font-size:12px; font-weight:700; color:#1e3a8a; line-height:1.2;">Discover</div>
    </span>
    <span style="display:inline-block; padding:0 10px; color:#94a3b8; font-size:16px; font-weight:700; vertical-align:middle;">→</span>
    <span style="display:inline-flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; width:86px; height:86px; border:1px solid #bbf7d0; border-radius:14px; background:#f0fdf4; padding:4px; vertical-align:middle; box-sizing:border-box;">
      <div style="font-size:18px; line-height:1.1;">⬇</div>
      <div style="font-size:12px; font-weight:700; color:#166534; line-height:1.2;">Download</div>
    </span>
    <span style="display:inline-block; padding:0 10px; color:#94a3b8; font-size:16px; font-weight:700; vertical-align:middle;">→</span>
    <span style="display:inline-flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; width:86px; height:86px; border:1px solid #fed7aa; border-radius:14px; background:#fff7ed; padding:4px; vertical-align:middle; box-sizing:border-box;">
      <div style="font-size:18px; line-height:1.1;">📝</div>
      <div style="font-size:12px; font-weight:700; color:#9a3412; line-height:1.2;">Convert</div>
    </span>
    <span style="display:inline-block; padding:0 10px; color:#94a3b8; font-size:16px; font-weight:700; vertical-align:middle;">→</span>
    <span style="display:inline-flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; width:86px; height:86px; border:1px solid #ddd6fe; border-radius:14px; background:#f5f3ff; padding:4px; vertical-align:middle; box-sizing:border-box;">
      <div style="font-size:18px; line-height:1.1;">📊</div>
      <div style="font-size:12px; font-weight:700; color:#5b21b6; line-height:1.2;">Analyze</div>
    </span>
  </div>
</div>

GeneralExplorer 是面向科研人员与学术开发者的智能论文检索与分析 Agent，自动完成从多源发现、筛选、下载、格式转换到结构化分析报告生成的全流程。项目聚焦于“最小交互、强证据、可复现”的学术工作流，助力高效文献综述、领域前沿追踪与创新启发。

它不是一个“只会搜论文”的脚本集合，而是一个可持续演进的研究引擎：
- 把一次性的读论文行为，升级为可积累、可复盘、可复用的研究资产。
- 把分散在 PDF、笔记、表格中的信息，统一为标准化目录与结构化报告。
- 把“今天读完就结束”的流程，变成“发现问题 -> 形成假设 -> 继续检索 -> 持续迭代”的研究闭环。

如果你正在做长期研究，这个项目能直接提升三个关键指标：
- 速度：减少重复性的下载、转换、整理和格式化工作。
- 质量：每条结论可追溯，避免无证据陈述与信息漂移。
- 复现：环境、产物、路径、标签和索引统一，便于团队协作和历史回溯。

---


## ✨ 核心特性

- **多源检索**：集成 Google Scholar、arXiv、DBLP 等主流学术源，支持关键词扩展、CCF 分级筛选、标题去歧义。
- **自动下载与格式转换**：一键下载 PDF，自动调用 MinerU 本地引擎高质量转换为 Markdown，图片与引用自动迁移。
- **结构化报告生成**：严格遵循科研分析模板，自动生成分章节、可追溯证据的 `paper.report.md`，支持增量修订与 in-place 合并。
- **全流程自动化**：默认“零交互”链路，自动完成检索、筛选、下载、转换、分析、索引与队列维护。
- **科研环境隔离**：所有脚本强制运行于本地虚拟环境 `.venv`，依赖一致、复现友好。
- **记忆系统与索引**：自动维护全局论文索引、分析队列、分类标签与历史分析记忆，避免重复劳动。

---

## 📂 目录结构

```
papers.csv                           # 全局论文台账（每分析一篇即追加）
papers/
  INDEX.md                           # 全局论文索引表
  QUEUE.md                           # 全局下载/分析队列
  raw/                               # 原始 PDF 暂存区
  {主标签}/
    {论文名}/
      paper.pdf
      paper.raw.md                   # MinerU 原始输出，禁止手动编辑
      paper.md                       # 清洗正文（无附录/参考文献）
      paper.references.md            # 仅参考文献
      paper.report.md                # 结构化分析报告（含主标签/论文标签）
      images/                        # 图片资源
scripts/

```
---

## 🚀 快速上手

这是一个 **SKILL/Agent 工作流项目**，推荐使用 GitHub Copilot 或 Claude 的 coding plan 模式驱动执行，而不是把它当作单一脚本工程直接运行。

1. **准备 Agent 运行环境**
   - 在 VS Code 中启用 Copilot Chat（或使用支持计划执行的 Claude 工作流）。
   - 推荐 Windows PowerShell，使用项目本地虚拟环境：
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - 依赖安装统一使用：
     ```powershell
     .\.venv\Scripts\pip.exe install -r requirements.txt
     ```

2. **用 Coding Plan 启动 SKILL（主入口）**
   - 在 Copilot Chat 中使用 Research Paper Pipeline Agent：
     ```text
     /research-paper-pipeline <研究主题 | 种子论文 | PDF路径>
     ```
   - 推荐一次给出完整目标，例如：
     ```text
     /research-paper-pipeline 主题: multi-vector retrieval; 先检索近5年CCF-A/B论文, 再下载、转换、清洗并生成报告
     ```
   - 在 Claude/其他 Agent 中，同样建议以“计划 -> 执行 -> 回写”三段式提示驱动，确保流程完整闭环。

3. **按 SKILL 规范自动产出**
   - Agent 会按仓库规范自动完成 discover/download/convert/analyze。
   - 结果会回写到 `papers/` 目录，并同步维护 `papers.csv`、`papers/INDEX.md`、`papers/QUEUE.md`。

4. **脚本直跑仅用于调试（非推荐主路径）**
   - 若需排障或单步验证，可手动调用脚本；但默认工作方式应是通过 SKILL 的 coding plan 驱动端到端执行。

---


## 🎯 适用场景

无论你是个人研究者、研究生团队，还是要维护长期技术路线图的工程研究组，GeneralExplorer 都适合作为“文献基础设施层”：

- **快速搭建领域全景图**：围绕一个主题自动扩展关键词、拉取候选论文并形成可追踪队列，显著降低从“问题定义”到“文献版图”之间的启动成本。
- **持续追踪前沿演进**：对新文献进行批量拉取与统一分析，把零散更新转成可比较、可回溯的连续观察，适合做周报、组会和阶段性综述。
- **高质量报告与复现准备**：将 PDF 内容转为结构化文本并输出标准化分析报告，明确已知条件与缺失条件，减少“看懂但复现不了”的断层。
- **团队协作知识沉淀**：通过统一目录、标签、索引和台账机制，让不同成员的阅读产物可以无缝汇总，避免重复读同一批论文。
- **选题与技术路线评估**：结合已分析论文的证据链，快速识别值得继续投入的方向、潜在风险点与可验证假设，提升决策效率。

---


## 🏆 技术亮点

- **严格证据链与结论约束**：所有关键结论都要求可回溯到源文本，默认抑制“无证据推断”，让报告更接近可审计的研究记录而非主观摘要。
- **自动化闭环而非单点工具**：从检索、下载、转换、清洗到报告与索引维护形成完整链路，减少人工在多工具间切换造成的上下文损耗。
- **最小交互设计**：仅在标题歧义高、候选冲突大等高风险节点请求确认，日常流程尽可能自动推进，保证效率同时控制误判成本。
- **可复现工程实践**：统一虚拟环境、统一产物命名、统一目录结构与台账字段，确保“今天能跑通”的流程在未来仍可稳定重放。
- **结构化产物体系**：`paper.raw.md`、`paper.md`、`paper.references.md`、`paper.report.md` 分工明确，既便于人读，也便于后续自动化处理。
- **可扩展架构能力**：支持按研究需求替换检索源、调整模板、追加后处理脚本与记忆策略，便于从个人项目平滑演进到团队平台。

---



## 📑 数据与命名规范

- `papers.csv` 列定义：`序号, 论文标题, 分析时间, 科研领域, tags`
- 论文目录与路径命名规则：见 `.github/instructions/papers-paths.instructions.md`
- `paper.raw.md` 为转换原始输出，不应直接编辑。
- `paper.md` 为清洗后的正文（不含附录和参考文献）。
- `paper.references.md` 仅包含参考文献内容（不含标题行）。




---

## 📚 已分析论文与领域

GeneralExplorer 已自动拉取与分析 50+ 篇高质量论文，覆盖如下研究方向：

- [Visual Document Retrieval](papers/visual_document_retrieval)
  - [Sculpting_the_Vector_Space_Towards_Efficient_Multi-Vector_Visual_Document_Retrieval_via_Prune-then-Merge_Framework](papers/visual_document_retrieval/Sculpting_the_Vector_Space_Towards_Efficient_Multi-Vector_Visual_Document_Retrieval_via_Prune-then-Merge_Framework)
  - [ColPali_Efficient_Document_Retrieval_with_Vision_Language_Models](papers/visual_document_retrieval/ColPali_Efficient_Document_Retrieval_with_Vision_Language_Models)
  - [DocPruner_A_Storage-Efficient_Framework_for_Multi-Vector_Visual_Document_Retrieval_via_Adaptive_Patch-Level_Embedding_Pruning](papers/visual_document_retrieval/DocPruner_A_Storage-Efficient_Framework_for_Multi-Vector_Visual_Document_Retrieval_via_Adaptive_Patch-Level_Embedding_Pruning)
  - [Visual_RAG_Toolkit_Scaling_Multi-Vector_Visual_Retrieval_with_Training-Free_Pooling_and_Multi-Stage_Search](papers/visual_document_retrieval/Visual_RAG_Toolkit_Scaling_Multi-Vector_Visual_Retrieval_with_Training-Free_Pooling_and_Multi-Stage_Search)

- [Multi-Vector Retrieval](papers/multi_vector_retrieval)
  - [XTR Rethinking the Role of Token Retrieval in Multi-Vector Retrieval](papers/multi_vector_retrieval/XTR%20Rethinking%20the%20Role%20of%20Token%20Retrieval%20in%20Multi-Vector%20Retrieval)
  - [MUVERA Multi-Vector Retrieval via Fixed Dimensional Encodings](papers/multi_vector_retrieval/MUVERA%20Multi-Vector%20Retrieval%20via%20Fixed%20Dimensional%20Encodings)
  - [PLAID An Efficient Engine for Late Interaction Retrieval](papers/multi_vector_retrieval/PLAID%20An%20Efficient%20Engine%20for%20Late%20Interaction%20Retrieval)
  - [DiskANN Fast Accurate Billion-Point Nearest Neighbor Search on a Single Node](papers/multi_vector_retrieval/DiskANN%20Fast%20Accurate%20Billion-Point%20Nearest%20Neighbor%20Search%20on%20a%20Single%20Node)

- [Agentic AI](papers/agentic_ai)
  - [Skill Retrieval Augmentation for Agentic AI](papers/agentic_ai/Skill%20Retrieval%20Augmentation%20for%20Agentic%20AI)
  - [SkillsBench Benchmarking how well agent skills work across diverse tasks](papers/agentic_ai/SkillsBench%20Benchmarking%20how%20well%20agent%20skills%20work%20across%20diverse%20tasks)
  - [Parametric Retrieval Augmented Generation](papers/agentic_ai/Parametric%20Retrieval%20Augmented%20Generation)
  - [DRAGIN Dynamic Retrieval Augmented Generation based on the real-time information needs of large language models](papers/agentic_ai/DRAGIN%20Dynamic%20Retrieval%20Augmented%20Generation%20based%20on%20the%20real-time%20information%20needs%20of%20large%20language%20models)

- [Set Similarity Search](papers/set_similarity_search)
  - [Approximate Matching for Fuzzy Set Similarity](papers/set_similarity_search/Approximate%20Matching%20for%20Fuzzy%20Set%20Similarity)

更多论文与详细分析见 [papers/INDEX.md](papers/INDEX.md)。

本项目受益于 MinerU、CCF 会议分级、Google Scholar、arXiv、DBLP 等开源资源。

---


<div align="center" style="margin-top: 32px;">
  <div style="display:inline-block; max-width: 760px; padding: 18px 20px; border-radius: 12px; background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%); border: 1px solid #bfdbfe; box-shadow: 0 8px 24px rgba(30, 64, 175, 0.10);">
    <div style="font-size: 18px; font-weight: 800; color: #1e3a8a; letter-spacing: 0.4px;">GeneralExplorer</div>
    <div style="margin-top: 6px; font-size: 14px; color: #334155; line-height: 1.7;">
      为长期科研而生的自动化文献分析 Agent：从发现到报告，从单篇到知识网络。
    </div>
    <div style="margin-top: 10px; font-size: 13px; color: #1d4ed8; font-weight: 700;">
      Discover • Download • Convert • Analyze • Rediscover
    </div>
  </div>
  <br/>
  <b>让科研分析更高效、更严谨、更自动化</b>
</div>
