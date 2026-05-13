---
name: research-paper-writing
description: Improve academic paper writing quality for ML/CV/NLP-style papers with clear section structure, paragraph flow, and reviewer-facing presentation. Use when drafting or revising Abstract, Introduction, Related Work, Method, Experiments, or Conclusion; polishing figures/tables; checking claim-support alignment; or performing self-review before submission.
---
# Research Paper Writing

## Overview

Use this skill to produce a reviewer-friendly, high-clarity draft **inside a user-specified LaTeX template project**.
This skill is designed to be orchestrated by a dedicated Copilot agent: `.github/agents/research-paper-writing.agent.md`.

Priorities:
- evidence-grounded writing (no citation hallucination),
- strong section-level narrative flow,
- output as a complete LaTeX project cloned from template assets.

## Copilot-Oriented Structure (Agent + Skills)

1. **Writing Agent Orchestration (required)**
   - A dedicated writing agent is responsible for coordinating this skill and upstream literature-grounding skills.
   - Do not run this skill as a stand-alone freeform writer when references are missing.

2. **Literature Grounding Before Writing (required)**
   - First, check whether relevant papers already have local analysis reports (`paper.report.md`).
   - If a reference is already analyzed locally, read and reuse that report as primary evidence.
   - If not analyzed locally, the agent must invoke literature-search/analysis skills (e.g., `paper-fetcher` -> `pdf-conversion` -> `paper-analyst`) to obtain grounded summaries before writing.

3. **Template-Driven Final Output (required)**
   - Final deliverable must be a LaTeX project generated from user-selected template under:
     - `.github/skills/research-paper-writing/assets/latex_templates/`
   - Even if the template directory is currently empty, the workflow and output contract remain fixed to this location.

4. **Compile Environment From .env (required)**
    - The skill must read LaTeX compile settings from:
       - `.github/agents/assets/.env`
    - Required keys: `LATEX_ENGINE`, `LATEX_BIB_ENGINE`, `LATEX_MAIN_FILE`, `LATEX_OUT_DIR`, `LATEX_COMPILE_TIMEOUT_SEC`, `LATEX_REQUIRED`.

## Core Workflow

1. Confirm user-selected LaTeX template name/path under `assets/latex_templates/`.
2. Build evidence pool from local reports first (`papers/**/paper.report.md`).
3. For missing evidence, invoke literature search + analysis skills to produce grounded notes.
4. Clarify paper story before sentence-level edits.
5. Use section-specific guidance in `references/`.
6. Rewrite paragraph-by-paragraph with one message per paragraph.
7. Run reverse outlining after writing each section.
8. Check every major claim in Abstract/Introduction against grounded evidence.
9. Run final-paper adversarial review with `references/paper-review.md`.
10. Populate the selected LaTeX template project and output the full project tree.
11. Load LaTeX env from `.github/agents/assets/.env` and run compile acceptance.

## Citation Grounding Rules (Anti-Hallucination)

1. No fabricated references, venues, years, or DOI.
2. Every citation used in writing must come from one of:
   - local analyzed report (`paper.report.md`), or
   - newly grounded literature-analysis output from search/conversion/analysis skills.
3. If evidence is insufficient, mark as `needs verification` rather than inventing references.
4. Related Work and Introduction must include a claim-evidence-citation mapping.
5. Prefer local report reuse when available; do not re-search the same paper unnecessarily.

## Global Principles

1. Keep one paragraph for one message only.
2. State the paragraph message in the first sentence.
3. Make nouns self-contained; define new terms before reusing them.
4. Maintain sentence-to-sentence flow (cause, contrast, consequence, or refinement).
5. Iterate with adversarial self-review: read as a skeptical reviewer.
6. Treat visual quality as core content, not decoration.
7. Use a clean teaser and pipeline figure.
8. Use readable, minimal-ink tables.
9. Keep formatting consistent and tidy.

## Paragraph Clarity Check (Important)

Use this quick test whenever the user asks whether a paragraph "flows" or is clear.

1. Read as an external reader:
   - Does this paragraph have one explicit message?
   - Does the first sentence state what this paragraph will do?
   - Are all key nouns/terms readable without hidden context?
   - Does each sentence connect to the previous one with a clear relation (cause, contrast, consequence, refinement, example)?
2. Run reverse outlining for the current section:
   - Write down thesis/main claim.
   - Write down each paragraph topic sentence.
   - Write down the evidence/explanation points under each paragraph.
   - Check mapping: topic sentence -> thesis, and evidence -> topic sentence.
   - Revise or remove any paragraph that cannot be mapped cleanly.
3. If flow is still weak, add temporary section headers and explicit transition phrases during revision, then remove unnecessary headers before finalizing.

Source reference for this check:

- `references/does-my-writing-flow-source.md`

## Section Guides

Load only the needed section file:

- Introduction: `references/introduction.md`
- Abstract: `references/abstract.md`
- Related Work: `references/related-work.md`
- Method: `references/method.md`
- Experiments: `references/experiments.md`
- Conclusion: `references/conclusion.md`
- Paper review (Paper Rview): `references/paper-review.md`
- Paragraph clarity source: `references/does-my-writing-flow-source.md`
- Example bank index: `references/examples/index.md`

## Local-First Literature Reuse Protocol

Before calling external literature search:

1. Query local analyzed papers and reports:
   - `papers/.db/index.csv` / `papers/.db/papers.csv`
   - `papers/**/paper.report.md`
2. For each needed citation topic, match nearest local report and extract:
   - problem setting,
   - method core idea,
   - key empirical finding,
   - limitation/boundary.
3. Only unresolved topics go to literature-search skills.

If unresolved:

4. Invoke literature-search + analysis path (recommended):
   - `paper-fetcher` (obtain PDF)
   - `pdf-conversion` (build markdown artifacts)
   - `paper-analyst` (grounded report with evidence anchors)
5. Feed returned grounded notes back into writing.

## Paper Review Core Points

Use `references/paper-review.md` for the full checklist and workflow.

1. Add an end-of-draft self-review question list in five dimensions:
   - contribution,
   - writing clarity,
   - experimental strength,
   - evaluation completeness,
   - method design soundness.
2. Treat claim-evidence alignment as a hard constraint, especially for Abstract and Introduction.
3. Perform adversarial writing: review as a skeptical reviewer and resolve every high-risk question.
4. Revise until major rejection risks are explicitly addressed.

## Execution Rules

1. Build a mini-outline before drafting prose.
2. For each subsection, explicitly include motivation, design, and technical advantage when applicable.
3. Avoid writing style that looks like incremental patching of a naive baseline.
4. Keep terminology stable across the full paper.
5. If a claim cannot be supported by results, weaken or remove the claim.
6. Before finalizing, append and answer a five-dimension self-review question list, then revise the paper based on unresolved items.
7. Do not load all section references (Introduction/Abstract/Related Work/Method/Experiments/Conclusion) at once; load only the specific section guide needed for the current edit target.
8. Always perform local-first report reuse before external literature search.
9. Keep a citation provenance log: `citation -> source report/search result`.
10. Final output must be a LaTeX template project, not plain markdown-only prose.

## LaTeX Template Output Contract (Required)

1. Template source root is fixed:
   - `.github/skills/research-paper-writing/assets/latex_templates/`
2. User must specify a template folder name (or exact path under this root).
3. If no template exists, stop and return a blocking message:
   - `latex_templates is empty; please add/select a template project first.`
4. Generate output by cloning selected template project to target paper workspace, then fill/replace sections.
5. Final output must include:
   - compilable `.tex` main file,
   - citation database (`.bib`) with grounded references only,
   - updated section files (Abstract/Intro/Related Work/Method/Experiments/Conclusion as requested),
   - figure/table assets referenced by LaTeX.

## Compile Acceptance (Required)

1. Before final delivery, load `.env`:
   - `set -a && source .github/agents/assets/.env && set +a`
2. Compile using env-driven settings (`LATEX_ENGINE`, `LATEX_BIB_ENGINE`, `LATEX_MAIN_FILE`, `LATEX_OUT_DIR`).
   - If `LATEX_ENGINE` is `tectonic`, treat bibliography handling as internal when `LATEX_BIB_ENGINE=internal`.
   - Ensure output directory exists before compile (e.g., create `LATEX_OUT_DIR` if missing).
3. If `LATEX_REQUIRED=1`, delivery is blocked unless compile exits successfully.
4. Acceptance criteria:
   - compile command exit code is 0,
   - expected PDF exists in output directory,
   - no unresolved citation markers in final PDF/log (e.g., `??`, undefined references).
5. On failure, return blocking diagnostics with actionable fixes (missing package, missing bib entry, missing file path).

## Output Contract

When asked to rewrite or draft sections, return:

1. A compact section outline (3-7 bullets).
2. Revised paragraphs with explicit paragraph roles (opening/challenge/method/advantage/evidence/limitation).
3. A short self-review checklist covering clarity, flow, terminology consistency, unsupported claims, and missing evidence.
4. A claim-evidence map for each major claim using:
   - `Claim: ... | Evidence: ... | Citation Source: local-report/search-analysis | Status: supported/needs evidence`.
5. A LaTeX project delivery summary:
   - selected template path,
   - generated target path,
   - updated `.tex` / `.bib` / figures / tables files list,
   - compile result (`pass`/`fail`) with command, output PDF path, and key log diagnostics.
