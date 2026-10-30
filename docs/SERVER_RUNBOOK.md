# Linux GPU server runbook

**Readiness:** benchmark preparation and CPU sampling primitives are available. A diagnostic B0–B2 Hugging Face adapter now uses full-prefix recomputation without KV cache. It is intended only for short correctness/integration smoke runs; timing is explicitly non-authoritative. Optimized cache handling, full M3 trace export, and real GPU validation remain pending.

## A. Repository synchronization

```bash
git clone <REPOSITORY_URL>
cd spade-speculative-decoding
git pull
git rev-parse HEAD | tee experiment-commit.txt
```

## B. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/inspect_environment.py
nvidia-smi
```

PyTorch is deliberately excluded from `requirements.txt` so pip will not replace the server's CUDA-specific build with a large generic wheel. Check the installed PyTorch, CUDA, and BF16 values with `inspect_environment.py`; install PyTorch separately only if it is absent, using the official wheel matching the server Python/CUDA setup.

## C. Dataset download and inspection

Set `DATA_ROOT` to a server-owned location. Obtain official AVeriTeC files from the project release/repository or the Hugging Face dataset repository using its documented download process; this repository deliberately does not download data. Place split JSON in `$DATA_ROOT/raw/`, retrieved passages in a JSON mapping from claim ID to ranked passage arrays, and inspect:

```bash
python scripts/inspect_dataset.py "$DATA_ROOT/raw/averitec_dev.json"
```

Retrieved JSON example shape: `{"claim-id": [{"text":"...", "url":"...", "rank":1, "score":12.3}]}`. Rankings must be generated without gold labels, justifications, or QA. Oracle QA is a diagnostic fallback only.

## D. Model checkpoints

Download `Qwen/Qwen3-8B` and `Qwen/Qwen3-0.6B` to `MODEL_ROOT/Qwen3-8B` and `MODEL_ROOT/Qwen3-0.6B` using Hugging Face Hub tooling on the server. Pin a Hub revision/commit for both and record it with the experiment. `local_files_only` defaults true in the model helper; no official run should silently fetch missing weights.

## E. M1 benchmark preparation

```bash
export DATA_ROOT=/srv/spade/data
python scripts/inspect_dataset.py "$DATA_ROOT/raw/averitec_dev.json"
python scripts/prepare_benchmark.py "$DATA_ROOT/raw/averitec_dev.json" "$DATA_ROOT/processed/smoke.jsonl" --mode oracle_qa --size 3 --seed 42
```

Retrieved mode requires `--retrieved "$DATA_ROOT/processed/retrieved_dev.json" --top-k 8`. `--size 100` and `--size 200` select pilot sizes. The prepared JSONL preserves evaluation metadata separately from the prompt field. Leakage tests: `python -m pytest tests/test_core.py -k gold`.

## F. M2 baseline execution

Example diagnostic run (not a performance benchmark):

```bash
python scripts/run_baselines.py --mode B1 --input benchmark.jsonl --target /path/to/Qwen3-8B --draft /path/to/Qwen3-0.6B --max-new-tokens 16 --draft-length 2 --limit 3 --output-root outputs/runs
```

The adapter writes an immutable run directory with raw/parsed generations, `round_traces.jsonl`, and `token_traces.jsonl` and reports `timings_authoritative: false`, because it recomputes full prefixes and does not use KV cache. Do not use these timings for official latency/speedup claims. B0–B2 GPU behavior still needs server validation.

## G. M3 analysis

`scripts/analyze_traces.py` accepts schema-compatible JSONL and writes aggregate JSON:

```bash
python scripts/analyze_traces.py outputs/run/round_traces.jsonl outputs/run/token_traces.jsonl --output outputs/run/metrics.json
```

No model-generated trace producer or plot exporter is available yet. Do not generate benchmark plots from synthetic or absent runs.

## H. Troubleshooting

- CUDA OOM: not yet addressed by benchmark orchestration; do not silently quantize/offload/change model. Record GPU memory and stop.
- BF16 unsupported: stop official run; do not silently substitute precision.
- Tokenizer mismatch: model helper fails closed on vocab IDs/special tokens; inspect revisions.
- Dataset fields: inspect actual keys first; adapt the loader only with fixtures.
- Malformed JSON: evaluation stores parse failure; no repair generation is implemented.
- KV cache mismatch / output differences: inference is pending; no cache fallback is claimed.
- Poor acceptance / negative speedup: expected research outcomes; measure only after adapter validation.
- Interrupted execution: immutable run IDs/resume are not implemented yet; preserve run files and avoid rerunning into the same output directory.
