---
description: "Use when creating, moving, or naming files under papers/. Covers main_tag selection, paper folder naming, and required artifacts for GeneralExplorer."
applyTo: "papers/**"
---

# Papers Path Rules

## Main Tag Selection

- Choose the main_tag from the research problem first.
- Reuse an existing top-level tag whenever a close problem bucket already exists.
- Prefer stable categories such as agentic_ai, multi_vector_retrieval, visual_document_retrieval, and set_similarity_search.

## Paper Directory Naming

- Use the paper title as the directory name.
- Preserve readable spacing unless the existing category clearly uses another convention.
- Avoid ad hoc abbreviations unless the original paper title is commonly abbreviated in the repository.

## Required Artifacts

Each paper directory should converge toward:

- paper.pdf
- paper.raw.md
- paper.md
- paper.references.md

Optional artifacts:

- images/
- paper.report.md
- mineru_full/ or other temporary directories only during conversion; clean them if they are not needed as final outputs.

## General Rules

- Final outputs must be written to the paper directory root, not a nested auto folder.
- Keep filenames stable so later indexing scripts can rely on them.
- When conversion produces images, store them under images/ in the paper directory.