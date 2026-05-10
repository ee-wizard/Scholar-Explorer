"""
Build a synthetic _metadata/index.json from existing SKILL.md files.
This allows build-index to run even if the official _metadata hasn't been extracted.
"""

import json
import sys
from pathlib import Path

import yaml


def parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from a SKILL.md file."""
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    fm_text = content[3:end].strip()
    try:
        meta = yaml.safe_load(fm_text)
        return meta if isinstance(meta, dict) else {}
    except yaml.YAMLError:
        return {}


def build_index(corpus_path: Path) -> dict:
    """Scan corpus_path for SKILL.md files and build index.json structure."""
    skills = {}
    count = 0
    skipped = 0

    # Scan skillsmp first (main corpus)
    for source_dir in ["skillsmp", "skills_rest"]:
        src = corpus_path / source_dir
        if not src.exists():
            print(f"  [skip] {source_dir} not found")
            continue
        skill_dirs = list(src.iterdir())
        print(f"  Scanning {source_dir}: {len(skill_dirs)} directories...")
        for skill_dir in skill_dirs:
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                skipped += 1
                continue
            try:
                content = skill_md.read_text(encoding="utf-8", errors="replace")
                meta = parse_frontmatter(content)
                name = meta.get("name", skill_dir.name)
                description = (meta.get("description") or "").strip()
                if not description:
                    skipped += 1
                    continue
                key = f"{source_dir}/{skill_dir.name}"
                skills[key] = {
                    "name": name or skill_dir.name,
                    "description": description,
                    "source": source_dir,
                    "local_path": key,
                }
                count += 1
                if count % 5000 == 0:
                    print(f"    Processed {count} skills...")
            except Exception as e:
                skipped += 1
                continue

    print(f"\nTotal valid skills: {count}, skipped: {skipped}")
    return {"skills": skills}


def main():
    corpus_path = Path(__file__).parent.parent / "data" / "skills"
    if not corpus_path.exists():
        print(f"ERROR: corpus path not found: {corpus_path}")
        sys.exit(1)

    metadata_dir = corpus_path / "_metadata"
    index_path = metadata_dir / "index.json"

    if index_path.exists():
        # Count existing entries
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        n = len(existing.get("skills", {}))
        print(f"Official _metadata/index.json already exists with {n} skills.")
        print("No need to rebuild. Exiting.")
        sys.exit(0)

    print(f"Building synthetic index from: {corpus_path}")
    index = build_index(corpus_path)

    metadata_dir.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(index['skills'])} skills to {index_path}")


if __name__ == "__main__":
    main()
