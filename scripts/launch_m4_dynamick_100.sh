#!/usr/bin/env bash
set -eo pipefail

REPO_DIR="/home/stackops/whale/spade-speculative-decoding"
cd "$REPO_DIR"

echo "=== [SPADE] Starting M4 Dynamic K Benchmark (100 Claims) at $(date) ==="

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

TARGET="/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DRAFT="/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
BENCHMARK="outputs/benchmarks/dev-oracle-pilot100.jsonl"
RUN_ID="full-m4-dynamic-k-lim100"

echo "=== [SPADE] Executing run_baselines.py for $RUN_ID ==="
/home/stackops/miniconda3/bin/python scripts/run_baselines.py \
    --mode M4 \
    --dynamic-k \
    --input "$BENCHMARK" \
    --run-id "$RUN_ID" \
    --target "$TARGET" \
    --draft "$DRAFT" \
    --draft-length 4 \
    --max-new-tokens 384 \
    --temperature 0.7 \
    --top-p 0.9 \
    --limit 100 \
    --cache-strategy kv_cache \
    --seed 42

echo "=== [SPADE] M4 Dynamic K Run Completed Successfully ==="
