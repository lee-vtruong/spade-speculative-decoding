#!/usr/bin/env bash
set -euo pipefail
cd /home/stackops/whale/spade-speculative-decoding

PY=/home/stackops/miniconda3/bin/python
TARGET=/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
DRAFT=/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
INPUT=outputs/benchmarks/retrieved-heldout300.jsonl
export PYTHONPATH=src

LIMIT="${1:-100}"
SUFFIX="${2:-heldout100}"

echo "================================================================="
echo "SPADE P2 Generalization: Qwen2.5-7B (Target) + Qwen2.5-0.5B (Draft)"
echo "Dataset: $INPUT (Limit: $LIMIT claims, Suffix: $SUFFIX)"
echo "Started at: $(date -u +%FT%TZ)"
echo "================================================================="

# Clean any existing partial run directories with this suffix
rm -rf outputs/runs/p2-qwen25-${SUFFIX}-*

echo "=== [1/5] Arm: Target AR (B0-S) ==="
$PY scripts/run_baselines.py \
  --mode B0-S --input "$INPUT" --run-id "p2-qwen25-${SUFFIX}-b0s" \
  --target "$TARGET" --dtype bfloat16 --max-new-tokens 384 \
  --temperature 0.7 --top-p 0.9 --seed 42 --limit "$LIMIT" --global-warmup 1 \
  --cache-strategy kv_cache | tail -n 8

echo "=== [2/5] Arm: Baseline Full Speculative (Full -> Full) ==="
$PY scripts/run_baselines.py \
  --mode B1 --input "$INPUT" --run-id "p2-qwen25-${SUFFIX}-full" \
  --target "$TARGET" --draft "$DRAFT" --dtype bfloat16 --draft-length 4 --max-new-tokens 384 \
  --temperature 0.7 --top-p 0.9 --seed 42 --limit "$LIMIT" --global-warmup 1 \
  --cache-strategy kv_cache --draft-runtime flex_compile_reduce_overhead --distribution-mode fp32_fast | tail -n 8

echo "=== [3/5] Arm: BM25 Top-2 Passages (Full -> Top-2) ==="
$PY scripts/run_baselines.py \
  --mode B2 --input "$INPUT" --run-id "p2-qwen25-${SUFFIX}-topk2" \
  --target "$TARGET" --draft "$DRAFT" --dtype bfloat16 --draft-length 4 --max-new-tokens 384 \
  --temperature 0.7 --top-p 0.9 --seed 42 --limit "$LIMIT" --global-warmup 1 \
  --cache-strategy kv_cache --draft-runtime flex_compile_reduce_overhead --distribution-mode fp32_fast \
  --draft-context-policy selective_topk --draft-top-k 2 | tail -n 8

echo "=== [4/5] Arm: BM25 Sent-B10k (Full -> Sent B=10k) ==="
$PY scripts/run_baselines.py \
  --mode B2 --input "$INPUT" --run-id "p2-qwen25-${SUFFIX}-bm25-sent-b10k" \
  --target "$TARGET" --draft "$DRAFT" --dtype bfloat16 --draft-length 4 --max-new-tokens 384 \
  --temperature 0.7 --top-p 0.9 --seed 42 --limit "$LIMIT" --global-warmup 1 \
  --cache-strategy kv_cache --draft-runtime flex_compile_reduce_overhead --distribution-mode fp32_fast \
  --draft-context-policy sentence_budget --draft-budget-chars 10000 \
  --draft-min-sentences-per-passage 1 | tail -n 8

echo "=== [5/5] Arm: SPADE (Full -> Attn B=10k) ==="
$PY scripts/run_baselines.py \
  --mode B2 --input "$INPUT" --run-id "p2-qwen25-${SUFFIX}-spade" \
  --target "$TARGET" --draft "$DRAFT" --dtype bfloat16 --draft-length 4 --max-new-tokens 384 \
  --temperature 0.7 --top-p 0.9 --seed 42 --limit "$LIMIT" --global-warmup 1 \
  --cache-strategy kv_cache --draft-runtime flex_compile_reduce_overhead --distribution-mode fp32_fast \
  --draft-context-policy target_attn_sentence_budget --draft-budget-chars 10000 \
  --draft-min-sentences-per-passage 1 --draft-attn-tail-window 16 | tail -n 8

echo "================================================================="
echo "P2 GENERALIZATION EVALUATION COMPLETE at $(date -u +%FT%TZ)"
echo "================================================================="
