#!/usr/bin/env bash
set -euo pipefail
cd /home/stackops/whale/spade-speculative-decoding

PY=/home/stackops/miniconda3/bin/python
TARGET=/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
DRAFT=/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
INPUT=outputs/benchmarks/retrieved-heldout300.jsonl
export PYTHONPATH=src

echo "=== b1-4-heldout300-bm25-sent-b10k at $(date -u +%FT%TZ) ==="
$PY scripts/run_baselines.py \
  --mode B2 --input "$INPUT" --run-id b1-4-heldout300-bm25-sent-b10k \
  --target "$TARGET" --draft "$DRAFT" --dtype bfloat16 --draft-length 4 --max-new-tokens 384 \
  --temperature 0.7 --top-p 0.9 --seed 42 --global-warmup 1 \
  --cache-strategy kv_cache --draft-runtime flex_compile_reduce_overhead --distribution-mode fp32_fast \
  --draft-context-policy sentence_budget --draft-budget-chars 10000 | tail -n 8

echo "=== COMPLETE at $(date -u +%FT%TZ) ==="
