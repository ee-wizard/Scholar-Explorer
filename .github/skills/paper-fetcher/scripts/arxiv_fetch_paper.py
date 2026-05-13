#!/usr/bin/env python3
"""Entrypoint wrapper for arXiv retrieval script."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "arxiv" / "arxiv_fetch_paper.py"
    runpy.run_path(str(target), run_name="__main__")