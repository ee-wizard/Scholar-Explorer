#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Prefer the workspace-root Unix venv created by the user, then fallback.
PYTHON_BIN="/mnt/d/Projects/scholar/GeneralExplorer/.venv-unix/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: Python not found at $PYTHON_BIN"
  echo "Please create a venv first (recommended: /mnt/d/Projects/scholar/GeneralExplorer/.venv-unix)."
  exit 1
fi

# Force local HuggingFace cache usage to avoid network pulls.
export HF_HOME="${HF_HOME:-/mnt/d/LocalEnvironments/Datasets/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$HF_HUB_CACHE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ ! -d "$HF_HUB_CACHE" ]]; then
  echo "Error: HuggingFace cache directory not found: $HF_HUB_CACHE"
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Warning: OPENAI_API_KEY is not set."
  echo "Query generation / selector stages may fail if enabled."
fi

LOCAL_CFG="skill_flow/config/local-skillflow.json"
BENCH_CFG="benchmark/config/skillsbench/skillflow-inject-local.json"

if [[ ! -f "$LOCAL_CFG" ]]; then
  echo "Error: missing config $LOCAL_CFG"
  exit 1
fi

if [[ ! -f "$BENCH_CFG" ]]; then
  echo "Error: missing config $BENCH_CFG"
  exit 1
fi

if [[ "${SKIP_BUILD_INDEX:-0}" != "1" ]]; then
  echo "[1/3] Build index"
  "$PYTHON_BIN" -m skill_flow.cli build-index --config "$LOCAL_CFG"
else
  echo "[1/3] Skip index build (SKIP_BUILD_INDEX=1)"
fi

if [[ "${SKIP_PIPELINE:-0}" != "1" ]]; then
  echo "[2/3] Run SkillFlow pipeline on SkillsBench tasks"
  "$PYTHON_BIN" -m skill_flow.cli pipeline --config "$LOCAL_CFG"
else
  echo "[2/3] Skip pipeline (SKIP_PIPELINE=1)"
fi

echo "[3/3] Run Harbor benchmark with SkillFlow injection"
uv run --group eval python -m benchmark.scripts.cli run --config "$BENCH_CFG"

echo "Done."
