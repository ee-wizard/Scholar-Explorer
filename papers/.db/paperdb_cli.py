#!/usr/bin/env python3
"""
CSV-based Paper Database CLI Tool（三态版）
管理 papers/.db 下的 CSV 索引，支持三种状态：已拉取 / 已转换 / 已分析

新增命令:
  index add-fetched <title> <main_tag> <pdf_name> [tags]   # 登记已拉取
  index mark-converted <title> <paper_path>                 # 更新为已转换（并删除 .raw PDF）
  index mark-analyzed <title> <paper_path>                  # 更新为已分析（写入 papers.csv）
  index update-state <title> <state>                        # 手动更新状态
  raw list                                                  # 列出 .raw/ 中的 PDF
  raw clean <pdf_name>                                      # 删除 .raw/ 中的 PDF
"""

import csv
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


INDEX_FIELDS   = ["序号", "论文标题", "主标签", "tags", "状态", "路径", "更新时间"]
PAPERS_FIELDS  = ["序号", "论文标题", "主标签", "路径", "分析时间", "状态"]
QUEUE_FIELDS   = ["序号", "论文标题", "来源", "arXiv/DOI", "优先级", "状态", "添加时间"]


class PaperDB:
    def __init__(self, db_dir: str = "."):
        self.db_dir      = Path(db_dir).resolve()
        self.papers_root = self.db_dir.parent          # papers/
        self.raw_dir     = self.papers_root / ".raw"
        self.queue_file  = self.db_dir / "queue.csv"
        self.papers_file = self.db_dir / "papers.csv"
        self.index_file  = self.db_dir / "index.csv"

    # ── internal ──────────────────────────────────────────────

    def _read(self, filepath: Path) -> List[Dict]:
        if not filepath.exists():
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _write(self, filepath: Path, rows: List[Dict], fieldnames: List[str]) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(rows)

    def _now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _next_seq(self) -> int:
        rows = self._read(self.index_file)
        if not rows:
            return 1
        seqs = [int(r["序号"]) for r in rows if r.get("序号", "").isdigit()]
        return max(seqs) + 1 if seqs else 1

    # ── Queue ─────────────────────────────────────────────────

    def queue_add(self, title: str, source="", doi="", priority="中") -> bool:
        rows = self._read(self.queue_file)
        rows.append({"序号": str(len(rows)+1), "论文标题": title, "来源": source,
                     "arXiv/DOI": doi, "优先级": priority, "状态": "待处理",
                     "添加时间": self._now()})
        self._write(self.queue_file, rows, QUEUE_FIELDS)
        return True

    def queue_remove(self, title: str) -> bool:
        rows = self._read(self.queue_file)
        new = [r for r in rows if r.get("论文标题") != title]
        if len(new) < len(rows):
            self._write(self.queue_file, new, QUEUE_FIELDS)
            return True
        return False

    def queue_list(self, status: Optional[str] = None) -> List[Dict]:
        rows = self._read(self.queue_file)
        return [r for r in rows if not status or r.get("状态") == status]

    def queue_add_reading_list(self, items: List[Dict]) -> Dict:
        """批量将推荐阅读条目写入 queue.csv（跳过已存在的标题）"""
        rows = self._read(self.queue_file)
        existing = {r.get("论文标题", "") for r in rows}
        added = []
        for item in items:
            title = item.get("title", "").strip()
            if not title or title in existing:
                continue
            rows.append({
                "序号": str(len(rows) + 1),
                "论文标题": title,
                "来源": item.get("source", "推荐阅读"),
                "arXiv/DOI": item.get("doi", ""),
                "优先级": item.get("priority", "中"),
                "状态": "待处理",
                "添加时间": self._now(),
            })
            existing.add(title)
            added.append(title)
        self._write(self.queue_file, rows, QUEUE_FIELDS)
        return {"added": len(added), "skipped": len(items) - len(added), "titles": added}

    # ── Index / State management ───────────────────────────────

    def index_add_fetched(self, title: str, main_tag: str, pdf_name: str, tags: str = "") -> Dict:
        """登记一个已拉取到 .raw/ 的 PDF"""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        seq = str(self._next_seq())
        row = {
            "序号": seq, "论文标题": title, "主标签": main_tag,
            "tags": tags or main_tag, "状态": "已拉取",
            "路径": f"papers/.raw/{pdf_name}", "更新时间": self._now(),
        }
        # 写 meta.json 到 .raw/
        stem = Path(pdf_name).stem
        meta_path = self.raw_dir / f"{stem}.meta.json"
        meta_path.write_text(json.dumps(
            {"title": title, "main_tag": main_tag, "tags": tags or main_tag, "seq": seq},
            ensure_ascii=False, indent=2
        ), encoding="utf-8")

        rows = self._read(self.index_file)
        rows.append(row)
        self._write(self.index_file, rows, INDEX_FIELDS)
        return {"status": "success", "seq": seq, "state": "已拉取", "path": row["路径"]}

    def index_mark_converted(self, title: str, paper_path: str, delete_raw_pdf: bool = True) -> Dict:
        """PDF 已转换为 Markdown，更新状态并删除 .raw/ 中的 PDF"""
        rows = self._read(self.index_file)
        found = False
        for r in rows:
            if r.get("论文标题") == title:
                old_path = r.get("路径", "")
                r["状态"]     = "已转换"
                r["路径"]     = paper_path
                r["更新时间"] = self._now()
                found = True
                # 删除 .raw/ 中的 PDF
                if delete_raw_pdf and old_path.startswith("papers/.raw/"):
                    pdf_file = self.papers_root.parent / old_path
                    if pdf_file.exists():
                        pdf_file.unlink()
                        meta = pdf_file.with_suffix(".meta.json")
                        if meta.exists():
                            meta.unlink()
                break
        if not found:
            return {"status": "error", "msg": f"Title not found: {title}"}
        self._write(self.index_file, rows, INDEX_FIELDS)
        return {"status": "success", "state": "已转换", "path": paper_path}

    def index_mark_analyzed(self, title: str, paper_path: str, tags: str = "") -> Dict:
        """论文分析完成：更新 index.csv 状态，并写入 papers.csv"""
        rows = self._read(self.index_file)
        seq = None
        main_tag = ""
        found = False
        for r in rows:
            if r.get("论文标题") == title:
                r["状态"]     = "已分析"
                r["路径"]     = paper_path
                r["更新时间"] = self._now()
                if tags:
                    r["tags"] = tags
                seq = r["序号"]
                main_tag = r["主标签"]
                found = True
                break
        if not found:
            # 不在 index 中（直接分析的论文），新增条目
            seq = str(self._next_seq())
            rows.append({
                "序号": seq, "论文标题": title, "主标签": main_tag,
                "tags": tags or main_tag, "状态": "已分析",
                "路径": paper_path, "更新时间": self._now(),
            })
        self._write(self.index_file, rows, INDEX_FIELDS)

        # 同步写入 papers.csv
        p_rows = self._read(self.papers_file)
        if not any(r.get("论文标题") == title for r in p_rows):
            p_rows.append({
                "序号": seq, "论文标题": title, "主标签": main_tag,
                "路径": paper_path, "分析时间": self._now(), "状态": "已分析",
            })
            self._write(self.papers_file, p_rows, PAPERS_FIELDS)

        # 从 queue 移除
        self.queue_remove(title)

        return {"status": "success", "seq": seq, "state": "已分析", "path": paper_path}

    def index_update_state(self, title: str, new_state: str) -> Dict:
        """手动更新论文状态（不删 raw PDF，不写 papers.csv）"""
        rows = self._read(self.index_file)
        for r in rows:
            if r.get("论文标题") == title:
                r["状态"]     = new_state
                r["更新时间"] = self._now()
                self._write(self.index_file, rows, INDEX_FIELDS)
                return {"status": "success", "state": new_state}
        return {"status": "error", "msg": f"Not found: {title}"}

    def index_search(self, title: str) -> Optional[Dict]:
        rows = self._read(self.index_file)
        for r in rows:
            if r.get("论文标题") == title:
                return r
        return None

    def index_list_by_state(self, state: str) -> List[Dict]:
        return [r for r in self._read(self.index_file) if r.get("状态") == state]

    # ── .raw/ management ──────────────────────────────────────

    def raw_list(self) -> List[Dict]:
        """列出 .raw/ 中的所有 PDF 及其元数据"""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        result = []
        for pdf in sorted(self.raw_dir.glob("*.pdf")):
            meta_file = pdf.with_suffix(".meta.json")
            title = pdf.stem.replace("_", " ")
            main_tag = ""
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    title    = meta.get("title", title)
                    main_tag = meta.get("main_tag", "")
                except Exception:
                    pass
            result.append({"pdf": pdf.name, "title": title, "main_tag": main_tag,
                           "size_kb": round(pdf.stat().st_size / 1024, 1)})
        return result

    def raw_clean(self, pdf_name: str) -> bool:
        """删除 .raw/ 中的 PDF 及其 meta.json"""
        pdf = self.raw_dir / pdf_name
        if pdf.exists():
            pdf.unlink()
        meta = self.raw_dir / (Path(pdf_name).stem + ".meta.json")
        if meta.exists():
            meta.unlink()
        return True

    # ── Legacy: add-completed (backward compat) ───────────────

    def add_completed_paper(self, title: str, main_tag: str, path: str, tags: str) -> Dict:
        return self.index_mark_analyzed(title, path, tags)

    # ── papers.csv helpers (backward compat) ─────────────────

    def papers_get_next_seq(self) -> int:
        return self._next_seq()

    def papers_search_by_title(self, title: str) -> Optional[Dict]:
        rows = self._read(self.papers_file)
        for r in rows:
            if r.get("论文标题") == title:
                return r
        return None

    def papers_list_by_tag(self, main_tag: str) -> List[Dict]:
        return [r for r in self._read(self.papers_file) if r.get("主标签") == main_tag]


# ── CLI entrypoint ────────────────────────────────────────────

def usage():
    print("""Usage: python paperdb_cli.py <command> [args]

State management:
  index add-fetched <title> <main_tag> <pdf_name> [tags]
  index mark-converted <title> <paper_path>
  index mark-analyzed <title> <paper_path> [tags]
  index update-state <title> <state>
  index search <title>
  index list-by-state <状态>           # 已拉取|已转换|已分析

Raw PDF management:
  raw list
  raw clean <pdf_name>

Queue:
  queue add <title> [source] [doi] [priority]
  queue remove <title>
  queue add-reading-list <json_array>           # 批量写入推荐阅读
  queue list [status]

Legacy (backward compat):
  add-completed <title> <main_tag> <path> <tags>
  papers next-seq
  papers search <title>
  papers list-by-tag <main_tag>
""")


def main():
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    import json as _json

    db = PaperDB(".")
    cmd = sys.argv[1]
    args = sys.argv[2:]

    try:
        if cmd == "index":
            sub = args[0] if args else ""
            if sub == "add-fetched" and len(args) >= 4:
                r = db.index_add_fetched(args[1], args[2], args[3], args[4] if len(args) > 4 else "")
                print(_json.dumps(r, ensure_ascii=False))
            elif sub == "mark-converted" and len(args) >= 3:
                r = db.index_mark_converted(args[1], args[2])
                print(_json.dumps(r, ensure_ascii=False))
            elif sub == "mark-analyzed" and len(args) >= 3:
                r = db.index_mark_analyzed(args[1], args[2], args[3] if len(args) > 3 else "")
                print(_json.dumps(r, ensure_ascii=False))
            elif sub == "update-state" and len(args) >= 3:
                r = db.index_update_state(args[1], args[2])
                print(_json.dumps(r, ensure_ascii=False))
            elif sub == "search" and len(args) >= 2:
                r = db.index_search(args[1])
                print(_json.dumps(r, ensure_ascii=False) if r else "Not found")
            elif sub == "list-by-state" and len(args) >= 2:
                rows = db.index_list_by_state(args[1])
                for row in rows:
                    print(_json.dumps(row, ensure_ascii=False))
            else:
                usage(); sys.exit(1)

        elif cmd == "raw":
            sub = args[0] if args else ""
            if sub == "list":
                items = db.raw_list()
                for item in items:
                    print(_json.dumps(item, ensure_ascii=False))
            elif sub == "clean" and len(args) >= 2:
                db.raw_clean(args[1])
                print(f"Cleaned: {args[1]}")
            else:
                usage(); sys.exit(1)

        elif cmd == "queue":
            sub = args[0] if args else ""
            if sub == "add" and len(args) >= 2:
                db.queue_add(args[1], args[2] if len(args)>2 else "",
                             args[3] if len(args)>3 else "", args[4] if len(args)>4 else "中")
                print("Added to queue")
            elif sub == "remove" and len(args) >= 2:
                print(db.queue_remove(args[1]))
            elif sub == "add-reading-list" and len(args) >= 2:
                items = _json.loads(args[1])
                r = db.queue_add_reading_list(items)
                print(_json.dumps(r, ensure_ascii=False))
            elif sub == "list":
                for row in db.queue_list(args[1] if len(args)>1 else None):
                    print(_json.dumps(row, ensure_ascii=False))
            else:
                usage(); sys.exit(1)

        elif cmd == "add-completed" and len(args) >= 4:
            r = db.add_completed_paper(args[0], args[1], args[2], args[3])
            print(_json.dumps(r, ensure_ascii=False))

        elif cmd == "papers":
            sub = args[0] if args else ""
            if sub == "next-seq":
                print(db.papers_get_next_seq())
            elif sub == "search" and len(args) >= 2:
                r = db.papers_search_by_title(args[1])
                print(r if r else "Not found")
            elif sub == "list-by-tag" and len(args) >= 2:
                for row in db.papers_list_by_tag(args[1]):
                    print(row)
            else:
                usage(); sys.exit(1)
        else:
            usage(); sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
