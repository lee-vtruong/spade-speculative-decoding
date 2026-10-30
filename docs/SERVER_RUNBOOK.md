# M1–M3 server runbook (hvtham-server)

Run all commands from the SPADE repository. Inputs in `ERRORFlow` and `training-free-fact-verification` are read-only. Every generated artifact goes under this repository's `outputs/` directory. Do not extract, rename, or rewrite files in the shared projects.

## 0. Environment

```bash
cd ~/whale/spade-speculative-decoding
git pull
source .venv/bin/activate
export DATA="$HOME/whale/ERRORFlow/data/raw/averitec"
export STORE="$HOME/whale/training-free-fact-verification/data_store/knowledge_store/train/uncompressed"
export TARGET="$HOME/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
export DRAFT="$HOME/whale/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
python scripts/inspect_environment.py
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")'
```

## M1 — Build benchmark and inspect evidence coverage

First audit the existing train store without modifying it. The adapter accepts only exact `type=claim` query matches to the claim text; QA, gold, answer, GPT, provenance, and other rows are excluded to avoid label/evidence leakage.

```bash
python scripts/inspect_knowledge_store.py "$DATA/train.json" "$STORE" --top-k 8
```

Create a full oracle QA benchmark for diagnostic comparison and a retrieved benchmark from the safe exact-claim subset. Outputs are immutable; use a new name if rerunning.

```bash
mkdir -p outputs/benchmarks
python scripts/prepare_benchmark.py "$DATA/train.json" outputs/benchmarks/train-oracle.jsonl --split train --mode oracle_qa --size full --seed 42 --tokenizer "$TARGET"
python scripts/prepare_benchmark.py "$DATA/train.json" outputs/benchmarks/train-retrieved-claim-only.jsonl --split train --mode retrieved --knowledge-store-dir "$STORE" --size full --top-k 8 --seed 42 --tokenizer "$TARGET"
```

Use `--size 3` for a smoke run, `100` or `200` for a pilot. If the claim-only audit reports poor coverage, do not silently include other query types. Oracle QA is a separate diagnostic condition and is not retrieval evidence. Dev has no store in the shared workspace; use oracle QA for dev, or download the official dev archive only into a new SPADE-owned directory (never into the shared project):

```bash
mkdir -p data/external/averitec-dev-knowledge-store
curl -L 'https://huggingface.co/chenxwh/AVeriTeC/resolve/74b0b0fa7357e5bbddf83e087a9feed082d8d8f4/data_store/knowledge_store/dev_knowledge_store.zip' -o data/external/averitec-dev-knowledge-store/dev_knowledge_store.zip
unzip -q data/external/averitec-dev-knowledge-store/dev_knowledge_store.zip -d data/external/averitec-dev-knowledge-store/unpacked
```

Inspect the archive layout before passing the appropriate per-claim directory to `--knowledge-store-dir`. The external directory is git-ignored.

## M2 — Run decoding baselines

Smoke the target-only baseline first, then compare greedy target-only B0 against speculative B1 under the same prompt, target, dtype, token limit, and claims. `run_baselines.py` defaults to full-prefix recomputation (no KV cache): these runs validate behavior and collect traces, but their latency is not suitable for speedup claims.

```bash
python scripts/run_baselines.py --mode B0 --input outputs/benchmarks/train-oracle.jsonl --target "$TARGET" --dtype bfloat16 --max-new-tokens 64 --limit 3 --warmup 1 --trials 3 --run-id m2-b0-greedy --output-root outputs/runs
python scripts/run_baselines.py --mode B1 --input outputs/benchmarks/train-oracle.jsonl --target "$TARGET" --draft "$DRAFT" --dtype bfloat16 --draft-length 4 --max-new-tokens 64 --limit 3 --warmup 1 --trials 3 --run-id m2-b1-greedy --output-root outputs/runs
python scripts/compare_greedy.py outputs/runs/m2-b0-greedy outputs/runs/m2-b1-greedy --output outputs/runs/m2-greedy-comparison.json
```

After the smoke run passes, execute the complete target-only/greedy sweep with unique run IDs (change the input path to the selected M1 benchmark):

```bash
for K in 2 4 6 8; do
  python scripts/run_baselines.py --mode B1 --input outputs/benchmarks/train-retrieved-claim-only.jsonl --target "$TARGET" --draft "$DRAFT" --dtype bfloat16 --draft-length "$K" --max-new-tokens 256 --warmup 1 --trials 3 --run-id "m2-b1-k${K}" --output-root outputs/runs
done
```

For exact speculative sampling B2, pair it with target sampling B0-S. Fix sampling policy and seed; repeat with several seeds to study statistical behavior. A single B2 run exercising rejection is not a distributional validation.

```bash
python scripts/run_baselines.py --mode B0-S --input outputs/benchmarks/train-oracle.jsonl --target "$TARGET" --dtype bfloat16 --temperature 0.7 --top-p 0.9 --seed 42 --max-new-tokens 128 --limit 3 --run-id m2-b0s-sampling --output-root outputs/runs
python scripts/run_baselines.py --mode B2 --input outputs/benchmarks/train-oracle.jsonl --target "$TARGET" --draft "$DRAFT" --dtype bfloat16 --draft-length 4 --temperature 0.7 --top-p 0.9 --seed 42 --max-new-tokens 128 --limit 3 --run-id m2-b2-exact-sampling --output-root outputs/runs
```

For the full B2 draft-length sweep, use independent run IDs and repeat with multiple seeds:

```bash
for SEED in 42 123 2026; do
  for K in 2 4 6 8; do
    python scripts/run_baselines.py --mode B2 --input outputs/benchmarks/train-retrieved-claim-only.jsonl --target "$TARGET" --draft "$DRAFT" --dtype bfloat16 --draft-length "$K" --temperature 0.7 --top-p 0.9 --seed "$SEED" --max-new-tokens 256 --run-id "m2-b2-s${SEED}-k${K}" --output-root outputs/runs
  done
done
```

Each run directory contains resolved config, environment/GPU metadata, predictions, timing trial rows, and round/token traces. B0-S/B2 token-by-token equality is not expected; test distributional agreement over a sufficiently large fixed prompt set and independent seeds. Factuality requires a separate evaluation protocol.

## M3 — Analyze real traces and evaluate predictions

Pass only actual run directories. This writes JSON summaries, CSV tables, and available PNG figures under the chosen SPADE output path.

```bash
python scripts/analyze_traces.py outputs/runs/m2-b1-greedy outputs/runs/m2-b2-exact-sampling --output-dir outputs/analysis/m2-m3
python scripts/evaluate_outputs.py outputs/runs/m2-b1-greedy/predictions.jsonl --output-dir outputs/evaluation/m2-b1-greedy
python scripts/evaluate_outputs.py outputs/runs/m2-b2-exact-sampling/predictions.jsonl --output-dir outputs/evaluation/m2-b2-exact-sampling
```

M3 reports acceptance/rejection, proposal probability diagnostics, latency decomposition, context length, and example token traces. Any speedup derived from this recompute adapter is explicitly diagnostic, not authoritative.

## Completion checks

```bash
python -m pytest -q
python -c 'import json,pathlib; p=pathlib.Path("outputs/benchmarks/train-oracle.jsonl.manifest.json"); print(json.dumps(json.loads(p.read_text()),indent=2))'
```

Archive the commit hash, manifests, resolved configs, environment files, metrics, and run IDs with experiment notes. Never place model outputs or benchmark artifacts in either shared dataset project.
