#!/usr/bin/env python3
"""
重建论文全状态索引（三态版）
- 已拉取: papers/.raw/<name>.pdf（已下载，未转换）
- 已转换: papers/<tag>/<name>/paper.raw.md 存在，paper.report.md 不存在
- 已分析: papers/<tag>/<name>/paper.report.md 存在

运行方式: python3 rebuild_index.py
"""
import csv
import json
from pathlib import Path
from datetime import datetime


def extract_title(md_path: Path):
    """从 Markdown 的第一个 # 标题行提取论文标题"""
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# "):
                    return line[2:].strip()
    except Exception:
        pass
    return None


def main():
    script_dir = Path(__file__).resolve().parent   # papers/.db/
    papers_root = script_dir.parent                # papers/
    raw_dir = papers_root / ".raw"
    db_dir = script_dir

    print(f"Scanning: {papers_root}")

    index_data = []
    seq = 0

    # ── State 1: 已拉取 ── PDFs in papers/.raw/
    if raw_dir.exists():
        for pdf in sorted(raw_dir.glob("*.pdf")):
            meta_file = raw_dir / (pdf.stem + ".meta.json")
            mtime = datetime.fromtimestamp(pdf.stat().st_mtime)
            ts = mtime.strftime("%Y-%m-%d %H:%M:%S")

            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    title    = meta.get("title", pdf.stem.replace("_", " "))
                    main_tag = meta.get("main_tag", "")
                    tags     = meta.get("tags", main_tag)
                except Exception:
                    title    = pdf.stem.replace("_", " ")
                    main_tag = ""
                    tags     = ""
            else:
                title    = pdf.stem.replace("_", " ")
                main_tag = ""
                tags     = ""

            seq += 1
            index_data.append({
                "序号":     str(seq),
                "论文标题": title,
                "主标签":   main_tag,
                "tags":     tags,
                "状态":     "已拉取",
                "路径":     f"papers/.raw/{pdf.name}",
                "更新时间": ts,
            })

    # ── States 2 & 3: scan all main_tag/paper directories ──
    for tag_dir in sorted(papers_root.iterdir()):
        if not tag_dir.is_dir() or tag_dir.name.startswith("."):
            continue

        for paper_dir in sorted(tag_dir.iterdir()):
            if not paper_dir.is_dir() or paper_dir.name.startswith("."):
                continue

            raw_md    = paper_dir / "paper.raw.md"
            report_md = paper_dir / "paper.report.md"

            if report_md.exists():
                # 已分析: 优先从 paper.raw.md 取标题（更准确）
                title = extract_title(raw_md) or extract_title(report_md)
                if not title:
                    continue
                mtime = datetime.fromtimestamp(report_md.stat().st_mtime)
                state = "已分析"

            elif raw_md.exists():
                # 已转换: 有 Markdown，无 report
                title = extract_title(raw_md)
                if not title:
                    continue
                mtime = datetime.fromtimestamp(raw_md.stat().st_mtime)
                state = "已转换"

            else:
                # 有 paper.pdf 但在主目录中（不规范，记录但不报错）
                pdf = paper_dir / "paper.pdf"
                if pdf.exists():
                    title = paper_dir.name
                    mtime = datetime.fromtimestamp(pdf.stat().st_mtime)
                    state = "已拉取"
                else:
                    continue

            seq += 1
            rel_path = f"papers/{tag_dir.name}/{paper_dir.name}"
            index_data.append({
                "序号":     str(seq),
                "论文标题": title,
                "主标签":   tag_dir.name,
                "tags":     tag_dir.name,
                "状态":     state,
                "路径":     rel_path,
                "更新时间": mtime.strftime("%Y-%m-%d %H:%M:%S"),
            })

    # ── Write index.csv (all states) ──
    INDEX_FIELDS = ["序号", "论文标题", "主标签", "tags", "状态", "路径", "更新时间"]
    with open(db_dir / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(index_data)

    # ── Write papers.csv (已分析 only, for backward compat) ──
    PAPERS_FIELDS = ["序号", "论文标题", "主标签", "路径", "分析时间", "状态"]
    analyzed = [r for r in index_data if r["状态"] == "已分析"]
    papers_rows = [{
        "序号":     r["序号"],
        "论文标题": r["论文标题"],
        "主标签":   r["主标签"],
        "路径":     r["路径"],
        "分析时间": r["更新时间"],
        "状态":     r["状态"],
    } for r in analyzed]

    with open(db_dir / "papers.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PAPERS_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(papers_rows)

    # ── Summary ──
    cnt = {s: sum(1 for r in index_data if r["状态"] == s)
           for s in ("已拉取", "已转换", "已分析")}
    print(f"\n✓ index.csv: {len(index_data)} total")
    print(f"  已拉取: {cnt['已拉取']}")
    print(f"  已转换: {cnt['已转换']}")
    print(f"  已分析: {cnt['已分析']}")
    print(f"✓ papers.csv: {len(analyzed)}")


if __name__ == "__main__":
    main()
