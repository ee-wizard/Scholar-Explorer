#!/bin/bash
# Monitor and execute SkillFlow reproduction steps

cd /home/wizard/projects/GeneralExplorer/projects/skill-flow
source /home/wizard/projects/GeneralExplorer/.venv-unix/bin/activate

echo "=== Waiting for build-index to complete ==="
while pgrep -f "python -m skill_flow.cli build-index" > /dev/null; do
    sleep 5
    ps aux | grep build-index | grep -v grep | awk '{print "Process still running:", $11, $12, $13, "CPU:", $3"%", "MEM:", $4"%"}'
done

echo ""
echo "=== Build-index completed! ==="
echo "=== Verifying index output ==="
ls -lh outputs/indices/external-skillflow/ | tail -10
echo ""

# Count index files
if [ -f outputs/indices/external-skillflow/faiss.index ]; then
    echo "✓ FAISS index created"
    du -h outputs/indices/external-skillflow/faiss.index
fi

if [ -f outputs/indices/external-skillflow/embeddings.npy ]; then
    echo "✓ Embeddings file created"
    du -h outputs/indices/external-skillflow/embeddings.npy
fi

echo ""
echo "=== Ready for next steps ==="
echo "Next: Run simple search test"
echo "Command:"
echo "python -m skill_flow.cli search --config skill_flow/config/default.json --index-dir outputs/indices/external-skillflow --query 'write unit tests for FastAPI endpoints' --top-k 5 --rerank false"
