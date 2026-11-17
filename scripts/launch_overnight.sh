#!/usr/bin/env bash
set -eo pipefail

REPO_DIR="/home/stackops/whale/spade-speculative-decoding"
cd "$REPO_DIR"

echo "=== [SPADE] Starting Overnight Benchmark Matrix at $(date) ==="

# Trap to guarantee docker container is restored on exit/failure
cleanup() {
    echo "=== [SPADE] Restoring resident docker container spades-qwen3-8b-planner at $(date) ==="
    docker start spades-qwen3-8b-planner || true
    echo "=== [SPADE] Run finished at $(date) ==="
}
trap cleanup EXIT INT TERM

echo "=== [SPADE] Freeing GPU VRAM ==="
docker stop spades-qwen3-8b-planner || true
sleep 3
nvidia-smi

echo "=== [SPADE] Executing run_overnight_matrix.py ==="
/home/stackops/miniconda3/bin/python scripts/run_overnight_matrix.py \
    --benchmark outputs/benchmarks/dev-oracle-pilot100.jsonl \
    --limit 100 \
    --max-new-tokens 384 \
    --cache-strategy kv_cache

echo "=== [SPADE] Overnight Matrix Completed Successfully ==="
