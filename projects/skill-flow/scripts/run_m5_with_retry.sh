#!/usr/bin/env bash
# 带重试的 M=5 pipeline 运行脚本
# LLM 调用异常时自动重启，缓存保证断点续跑

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

source /home/wizard/projects/GeneralExplorer/.venv-unix/bin/activate

# Load .env: prefer local project .env, fall back to workspace-root .env
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
elif [ -f "$SCRIPT_DIR/../../.env" ]; then
    set -a
    source "$SCRIPT_DIR/../../.env"
    set +a
fi

CONFIG="${1:-skill_flow/config/eval_external_m5_stage1only.json}"
OUTPUT_DIR="${2:-outputs/pipeline/skillsbench_m5}"
TASKS_DIR="/mnt/d/LocalEnvironments/Datasets/skillsbench/tasks"

mkdir -p "$OUTPUT_DIR"

# This experiment script defaults to Copilot.
# To override intentionally, set SKILL_FLOW_LLM_BACKEND_OVERRIDE.
export SKILL_FLOW_LLM_BACKEND="${SKILL_FLOW_LLM_BACKEND_OVERRIDE:-copilot}"
export SKILL_FLOW_QGEN_STRICT_MULTI=1
export SKILL_FLOW_DEVICE=cuda:0
export PYTHONUNBUFFERED=1

MAX_RETRIES=20
attempt=0

while [ $attempt -lt $MAX_RETRIES ]; do
    attempt=$((attempt + 1))

    # 已完成任务数（通过缓存）
    CACHE="$OUTPUT_DIR/retriever_query_gen_cache.json"
    if [ -f "$CACHE" ]; then
        DONE=$(python3 -c "import json; d=json.load(open('$CACHE')); print(len(d))")
    else
        DONE=0
    fi

    # 检查 Stage 1 输出是否已完整（eval 文件存在）
    if [ -f "$OUTPUT_DIR/eval-stage1-retriever.json" ]; then
        echo "[$(date '+%H:%M:%S')] Stage 1 complete. Exiting."
        exit 0
    fi

    echo "[$(date '+%H:%M:%S')] Attempt $attempt/$MAX_RETRIES — cache: $DONE/94 tasks"

    python -u -m skill_flow.cli pipeline \
        --config "$CONFIG" \
        --tasks-dir "$TASKS_DIR" \
        --output-dir "$OUTPUT_DIR" 2>&1 | awk -v done="$DONE" '
            {
                print
                if ($0 ~ /openai\.AuthenticationError/ || $0 ~ /Incorrect API key provided/ || $0 ~ /invalid_api_key/) {
                    auth_failed = 1
                }
                if ($0 ~ /Stage 1 \[/) {
                    done += 1
                    printf("[%s] Progress: %d/94 tasks\n", strftime("%H:%M:%S"), done)
                    fflush()
                }
            }

            END {
                if (auth_failed) {
                    exit 42
                }
            }
        ' && {
        echo "[$(date '+%H:%M:%S')] Pipeline completed successfully."
        exit 0
    } || {
        EXIT=$?
        if [ "$EXIT" -eq 42 ]; then
            echo "[$(date '+%H:%M:%S')] Detected invalid OpenAI API key. Aborting retries."
            exit 1
        fi
        echo "[$(date '+%H:%M:%S')] Exit code $EXIT — waiting 10s before retry..."
        sleep 10
    }
done

echo "Max retries ($MAX_RETRIES) reached."
exit 1
