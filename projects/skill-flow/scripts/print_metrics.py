#!/usr/bin/env python3
"""
Stage metrics summary for SkillFlow pipeline ablation.
Run after each stage completes to see updated ablation table.
"""
import json
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR = Path("outputs/pipeline/skillsbench_no_qgen")
BM25_DIR = Path("outputs/pipeline/ablation_bm25")
LOG_FILE = Path("logs/pipeline-no-qgen.log")


def fmt_metrics(summary):
    ra = summary['mean_recall_at']
    pa = summary['mean_precision_at']
    ha = summary['mean_hit_at']
    mrr = summary['mrr']
    ks = ['1', '5', '10', '50', '100']
    print(f"  MRR: {mrr:.4f}")
    print(f"  {'Metric':<12} " + "  ".join(f"@{k:>4}" for k in ks))
    print(f"  {'Recall':<12} " + "  ".join(f"{ra[k]:>6.4f}" for k in ks))
    print(f"  {'Precision':<12} " + "  ".join(f"{pa[k]:>6.4f}" for k in ks))
    print(f"  {'Hit':<12} " + "  ".join(f"{ha[k]:>6.4f}" for k in ks))


def get_timing(log_path, stage_num):
    from datetime import datetime
    import re
    pattern = rf'(\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}}),\d+ INFO skill_flow.pipeline.stages: Stage {stage_num} \['
    times = re.findall(pattern, log_path.read_text())
    if len(times) >= 2:
        t0 = datetime.strptime(times[0], '%Y-%m-%d %H:%M:%S')
        t1 = datetime.strptime(times[-1], '%Y-%m-%d %H:%M:%S')
        return (t1 - t0).seconds
    return None


def compute_hybrid_rrf(dense_path, bm25_path, tasks_dir, k=60):
    """Compute Hybrid Dense+BM25 RRF metrics."""
    dense_d = json.load(open(dense_path))
    bm25_d = json.load(open(bm25_path))
    dense_map = {t['task_id']: t['retrieved_skills'] for t in dense_d['task_results']}
    bm25_map = {t['task_id']: t['retrieved_skills'] for t in bm25_d['task_results']}

    tasks_path = Path(tasks_dir)
    gt_map = {}
    for task_dir in sorted(tasks_path.iterdir()):
        if not task_dir.is_dir():
            continue
        skills_dir = task_dir / 'environment' / 'skills'
        if not skills_dir.exists():
            continue
        gt_keys = [f"skillsbench/{task_dir.name}/{e.name}"
                   for e in skills_dir.iterdir()
                   if e.is_dir() and (e / 'SKILL.md').exists()]
        if gt_keys:
            gt_map[task_dir.name] = gt_keys

    ks = [1, 5, 10, 50, 100, 1000]
    r_sums = defaultdict(float)
    h_sums = defaultdict(float)
    mrr_sum = 0.0
    n = 0

    for tid, gt in gt_map.items():
        if tid not in dense_map or tid not in bm25_map:
            continue
        scores = defaultdict(float)
        for rank, s in enumerate(dense_map[tid]):
            scores[s['key']] += 1.0 / (k + rank + 1)
        for rank, s in enumerate(bm25_map[tid]):
            scores[s['key']] += 1.0 / (k + rank + 1)
        merged = sorted(scores.keys(), key=lambda x: -scores[x])
        gt_set = set(gt)
        for kk in ks:
            top = set(merged[:kk])
            r_sums[kk] += sum(1 for g in gt if g in top) / len(gt)
            h_sums[kk] += 1.0 if any(g in top for g in gt) else 0.0
        for i, key in enumerate(merged):
            if key in gt_set:
                mrr_sum += 1.0 / (i + 1)
                break
        n += 1

    if n == 0:
        return None
    return {
        'mrr': mrr_sum / n,
        'recall': {str(k): r_sums[k] / n for k in ks},
        'hit': {str(k): h_sums[k] / n for k in ks},
        'n': n,
    }


print("=" * 60)
print("SkillFlow Pipeline Metrics Summary")
print("=" * 60)

files = {
    1: ("eval-stage1-retriever.json", "Stage 1 — Dense Retrieval (bge-base-en-v1.5)"),
    2: ("eval-stage2-reranker.json", "Stage 2 — Shallow Reranker (ms-marco-MiniLM-L-6-v2)"),
    3: ("eval-stage3-deep_reranker.json", "Stage 3 — Deep Reranker (bge-reranker-v2-m3)"),
}

stage_metrics = {}
for stage_num, (fname, label) in files.items():
    fpath = OUTPUT_DIR / fname
    if fpath.exists():
        d = json.load(open(fpath))
        elapsed = get_timing(LOG_FILE, stage_num) if LOG_FILE.exists() else None
        print(f"\n{label}")
        if elapsed:
            print(f"  Total time: {elapsed}s ({elapsed/94:.1f}s/task)")
        fmt_metrics(d['summary'])
        stage_metrics[stage_num] = d['summary']
    else:
        print(f"\n{label}")
        print(f"  ❌ Not yet available: {fname}")

# BM25
bm25_path = BM25_DIR / "eval-stage1-retriever.json"
if bm25_path.exists():
    d = json.load(open(bm25_path))
    print(f"\nBM25 Only")
    fmt_metrics(d['summary'])

print("\n" + "=" * 60)
print("\nAblation Table (vs Paper Table 3 & 4):")
header = f"{'Method':<40} {'MRR':>6} {'R@10':>6} {'R@1000':>7} {'Hit@10':>7}"
print(header)
print("-" * len(header))

paper = [
    ("Paper: BM25 only", 0.266, 0.238, None, 0.391),
    ("Paper: Hybrid (Dense+BM25, RRF)", 0.522, 0.480, None, 0.713),
    ("Paper: Dense only", 0.553, 0.477, None, 0.713),
    ("Paper: + Shallow Reranker", 0.587, 0.520, None, 0.724),
    ("Paper: + Deep Reranker", 0.634, 0.595, None, 0.793),
]
for row in paper:
    r1000 = f"{row[3]:>7.3f}" if row[3] else "      —"
    print(f"{row[0]:<40} {row[1]:>6.3f} {row[2]:>6.3f} {r1000} {row[4]:>7.3f}")

print()
if bm25_path.exists():
    s = json.load(open(bm25_path))['summary']
    ra = s['mean_recall_at']
    ha = s['mean_hit_at']
    print(f"{'Local: BM25 only':<40} {s['mrr']:>6.3f} {ra['10']:>6.3f} {ra['1000']:>7.3f} {ha['10']:>7.3f}")

# Hybrid
dense_path = OUTPUT_DIR / "eval-stage1-retriever.json"
if dense_path.exists() and bm25_path.exists():
    h = compute_hybrid_rrf(dense_path, bm25_path, '/mnt/d/LocalEnvironments/Datasets/skillsbench/tasks')
    if h:
        print(f"{'Local: Hybrid (Dense+BM25, RRF)':<40} {h['mrr']:>6.3f} {h['recall']['10']:>6.3f} {h['recall']['1000']:>7.3f} {h['hit']['10']:>7.3f}")

for stage_num, (fname, label) in files.items():
    fpath = OUTPUT_DIR / fname
    if fpath.exists():
        s = json.load(open(fpath))['summary']
        ra = s['mean_recall_at']
        ha = s['mean_hit_at']
        short = f"Local: Stage {stage_num} M=1 ({'Dense' if stage_num==1 else 'Shallow' if stage_num==2 else 'Deep'})"
        r1000 = f"{ra.get('1000', 0):>7.3f}" if '1000' in ra else "      —"
        print(f"{short:<40} {s['mrr']:>6.3f} {ra['10']:>6.3f} {r1000} {ha['10']:>7.3f}")
    else:
        short = f"Local: Stage {stage_num} M=1 — pending"
        print(f"{short:<40} {'—':>6} {'—':>6} {'—':>7} {'—':>7}")

# M=5 pipeline results
M5_DIR = Path("outputs/pipeline/skillsbench_m5")
M5_LOG = Path("logs/pipeline-m5.log")
m5_stage1 = M5_DIR / "eval-stage1-retriever.json"
m5_stage2 = M5_DIR / "eval-stage2-reranker.json"
m5_stage3 = M5_DIR / "eval-stage3-deep_reranker.json"
print()
for fpath, label, snum in [
    (m5_stage1, "Local: Stage 1 M=5 (Dense, union)", 1),
    (m5_stage2, "Local: Stage 1+2 M=5 (Shallow)", None),
    (m5_stage3, "Local: Stage 1+2+3 M=5 (Deep)", None),
]:
    if fpath.exists():
        s = json.load(open(fpath))['summary']
        ra = s['mean_recall_at']
        ha = s['mean_hit_at']
        r1000 = f"{ra.get('1000', 0):>7.3f}" if '1000' in ra else "      —"
        print(f"{label:<40} {s['mrr']:>6.3f} {ra['10']:>6.3f} {r1000} {ha['10']:>7.3f}")
    else:
        print(f"{label:<40} {'—':>6} {'—':>6} {'—':>7} {'—':>7}")


