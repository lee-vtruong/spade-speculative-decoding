# SPADE Remote Server Execution Runbook (Milestones M1–M3)

This runbook provides step-by-step, copy-pasteable instructions for executing SPADE experiments on a remote Linux GPU server with NVIDIA CUDA support.

The local development workflow strictly forbids downloading weights or running GPU inference locally:
```text
LOCAL DEVELOPMENT (CPU unit tests & plumbing)
       ↓
GIT COMMIT & USER PUSH
       ↓
REMOTE SERVER (Linux + CUDA GPU)
       ↓
DATASET & MODEL DOWNLOADS → BENCHMARK BUILD → BASELINE RUNS → M3 ANALYSIS
```

---

## Stage A — Repository Synchronization

### 1. Clone the repository on the remote server
```bash
git clone <REPOSITORY_URL>
cd spade-speculative-decoding
```

### 2. For updates / pulling latest commits
```bash
git pull origin main
```

### 3. Record the exact Git commit used for experiment tracking
```bash
git rev-parse HEAD > outputs/git_commit.txt
```

---

## Stage B — Python Environment Setup

### 1. Create and activate a Python virtual environment (Python >= 3.10)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Upgrade pip and install pinned dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
```

### 3. Validate CUDA, BF16 support, and GPU memory
```bash
python scripts/inspect_environment.py
```
Expected output confirms:
- PyTorch version and CUDA version
- `cuda_available: true`
- `bf16_supported: true`
- GPU name(s) and available VRAM

---

## Stage C — Dataset Download (AVeriTeC)

Datasets must reside under `$DATA_ROOT` (default `data/raw/averitec` or a dedicated local directory).

### 1. Download official AVeriTeC split JSON files
```bash
mkdir -p data/raw/averitec
# Download train and dev splits
curl -L "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/train.json" -o data/raw/averitec/train.json
curl -L "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/dev.json" -o data/raw/averitec/dev.json
```

### 2. Optional: Download AVeriTeC knowledge store (per-claim JSONL files)
```bash
mkdir -p data/knowledge_store/dev
curl -L "https://huggingface.co/chenxwh/AVeriTeC/resolve/74b0b0fa7357e5bbddf83e087a9feed082d8d8f4/data_store/knowledge_store/dev_knowledge_store.zip" -o data/knowledge_store/dev.zip
unzip -q data/knowledge_store/dev.zip -d data/knowledge_store/dev
```

### 3. Inspect dataset schema and count
```bash
python scripts/inspect_dataset.py data/raw/averitec/train.json
python scripts/inspect_dataset.py data/raw/averitec/dev.json
```

If using an existing knowledge store, audit safe exact-claim query coverage before building:
```bash
python scripts/inspect_knowledge_store.py data/raw/averitec/train.json data/knowledge_store/train --top-k 8
```

---

## Stage D — Model Download

Target model: `Qwen/Qwen3-8B`  
Draft model: `Qwen/Qwen3-0.6B`  

Store weights locally under `models/` (or use Hugging Face cache):
```bash
mkdir -p models
huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B --local-dir-use-symlinks False
huggingface-cli download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B --local-dir-use-symlinks False
```

Export environment paths for convenience:
```bash
export TARGET_MODEL="models/Qwen3-8B"
export DRAFT_MODEL="models/Qwen3-0.6B"
```

---

## Stage E — M1 Benchmark Preparation

Build deterministic, leakage-separated benchmark JSONL files and cryptographic manifests.

### 1. Diagnostic Mode A: Oracle QA Evidence (Smoke, Pilot, Full)
```bash
mkdir -p outputs/benchmarks

# 3-claim smoke test benchmark
python scripts/prepare_benchmark.py data/raw/averitec/dev.json outputs/benchmarks/dev-oracle-smoke.jsonl \
    --split dev --mode oracle_qa --size 3 --seed 42 --tokenizer "$TARGET_MODEL"

# 100-claim pilot benchmark
python scripts/prepare_benchmark.py data/raw/averitec/dev.json outputs/benchmarks/dev-oracle-pilot.jsonl \
    --split dev --mode oracle_qa --size 100 --seed 42 --tokenizer "$TARGET_MODEL"

# Full dev benchmark
python scripts/prepare_benchmark.py data/raw/averitec/dev.json outputs/benchmarks/dev-oracle-full.jsonl \
    --split dev --mode oracle_qa --size full --seed 42 --tokenizer "$TARGET_MODEL"
```

### 2. Mode B: Retrieved Evidence (Realistic Fact-Checking Benchmark)
```bash
# Using audited knowledge store
python scripts/prepare_benchmark.py data/raw/averitec/train.json outputs/benchmarks/train-retrieved-pilot.jsonl \
    --split train --mode retrieved --knowledge-store-dir data/knowledge_store/train \
    --size 100 --top-k 8 --seed 42 --tokenizer "$TARGET_MODEL"
```

Each run generates:
- `<output>.jsonl`: Clean model-visible prompts with separate evaluation metadata.
- `<output>.jsonl.manifest.json`: Checksum, claim IDs, token statistics, and retrieval audit.

---

## Stage F — M2 Baseline Execution

Run baselines using either the exact recompute path (`--cache-strategy recompute`) or fast KV-cache decoding (`--cache-strategy kv_cache`).

### 1. 3-Claim GPU Smoke Test
```bash
# B0: Target-only Greedy
python scripts/run_baselines.py --mode B0 \
    --input outputs/benchmarks/dev-oracle-smoke.jsonl \
    --target "$TARGET_MODEL" --dtype bfloat16 \
    --max-new-tokens 64 --limit 3 --warmup 1 --trials 1 \
    --cache-strategy kv_cache \
    --run-id smoke-b0-greedy --output-root outputs/runs

# B1: Speculative Greedy (Draft Length = 4)
python scripts/run_baselines.py --mode B1 \
    --input outputs/benchmarks/dev-oracle-smoke.jsonl \
    --target "$TARGET_MODEL" --draft "$DRAFT_MODEL" --dtype bfloat16 \
    --draft-length 4 --max-new-tokens 64 --limit 3 --warmup 1 --trials 1 \
    --cache-strategy kv_cache \
    --run-id smoke-b1-greedy --output-root outputs/runs

# Compare token-level equivalence between B0 and B1
python scripts/compare_greedy.py outputs/runs/smoke-b0-greedy outputs/runs/smoke-b1-greedy \
    --output outputs/runs/smoke-greedy-comparison.json
```

### 2. Full Experiment Matrix (100-claim Pilot)

#### Target Baselines:
```bash
# E0: Target-only greedy (B0)
python scripts/run_baselines.py --mode B0 \
    --input outputs/benchmarks/dev-oracle-pilot.jsonl \
    --target "$TARGET_MODEL" --dtype bfloat16 \
    --max-new-tokens 256 --warmup 1 --trials 3 \
    --cache-strategy kv_cache \
    --run-id m2-e0-b0-greedy --output-root outputs/runs

# E2: Target-only sampling (B0-S)
python scripts/run_baselines.py --mode B0-S \
    --input outputs/benchmarks/dev-oracle-pilot.jsonl \
    --target "$TARGET_MODEL" --dtype bfloat16 \
    --temperature 0.7 --top-p 0.9 --seed 42 \
    --max-new-tokens 256 --warmup 1 --trials 3 \
    --cache-strategy kv_cache \
    --run-id m2-e2-b0s-sampling --output-root outputs/runs
```

#### Greedy Speculative Sweep:
```bash
# E1: Greedy speculative decoding (B1, K=4)
python scripts/run_baselines.py --mode B1 \
    --input outputs/benchmarks/dev-oracle-pilot.jsonl \
    --target "$TARGET_MODEL" --draft "$DRAFT_MODEL" --dtype bfloat16 \
    --draft-length 4 --max-new-tokens 256 --warmup 1 --trials 3 \
    --cache-strategy kv_cache \
    --run-id m2-e1-b1-k4 --output-root outputs/runs
```

#### Exact Speculative Sampling Sweeps (E3–E6: Draft lengths 2, 4, 6, 8):
```bash
for K in 2 4 6 8; do
    python scripts/run_baselines.py --mode B2 \
        --input outputs/benchmarks/dev-oracle-pilot.jsonl \
        --target "$TARGET_MODEL" --draft "$DRAFT_MODEL" --dtype bfloat16 \
        --draft-length "$K" --temperature 0.7 --top-p 0.9 --seed 42 \
        --max-new-tokens 256 --warmup 1 --trials 3 \
        --cache-strategy kv_cache \
        --run-id "m2-exact-b2-k${K}" --output-root outputs/runs
done
```

---

## Stage G — M3 Analysis and Visualizations

Generate acceptance diagnostics, probability breakdowns, latency charts, and CSV summaries:

```bash
python scripts/analyze_traces.py \
    outputs/runs/m2-e0-b0-greedy \
    outputs/runs/m2-e1-b1-k4 \
    outputs/runs/m2-e2-b0s-sampling \
    outputs/runs/m2-exact-b2-k2 \
    outputs/runs/m2-exact-b2-k4 \
    outputs/runs/m2-exact-b2-k6 \
    outputs/runs/m2-exact-b2-k8 \
    --output-dir outputs/analysis/pilot_m3
```

Evaluate factual summarization JSON parsing and citation accuracy:
```bash
python scripts/evaluate_outputs.py outputs/runs/m2-e1-b1-k4/predictions.jsonl \
    --output-dir outputs/evaluation/m2-e1-b1-k4

python scripts/evaluate_outputs.py outputs/runs/m2-exact-b2-k4/predictions.jsonl \
    --output-dir outputs/evaluation/m2-exact-b2-k4
```

Launch the Jupyter research notebook for interactive inspection:
```bash
jupyter notebook notebooks/m3_rejection_analysis.ipynb
```

---

## Stage H — Experiment Troubleshooting

### 1. CUDA Out-of-Memory (OOM)
- Qwen3-8B in BF16 requires ~16 GB VRAM. When paired with Qwen3-0.6B (~1.5 GB), total memory is ~18 GB.
- If running on a GPU with <= 24 GB VRAM under long contexts (8k+ tokens), pass `--max-evidence-tokens 1024` during `prepare_benchmark.py` to bound prompt length.
- Enable `logits_to_keep=1` (supported natively by `hf_assisted.py`) to prevent materializing large LM-head logits.

### 2. BF16 Unsupported
- Older architectures (e.g. NVIDIA Turing T4 / V100) do not have native BF16 tensor cores.
- Set `--dtype float16` instead of `bfloat16`.

### 3. Model/Tokenizer Incompatibility
- `validate_tokenizers()` verifies that target and draft models share identical vocabulary sizes, special token IDs (`eos_token_id`, `pad_token_id`), and token mappings.
- If an incompatible draft model is supplied, the script aborts with an explicit error rather than silently generating invalid token IDs.

### 4. Malformed JSON Generation
- The model prompt instructs deterministic JSON generation with keys `summary`, `citations`, and `verdict`.
- If generation stops before completing JSON (e.g. truncated by `max_new_tokens`), `parse_generation()` flags `parse_success: false` and preserves the raw text.
- Never perform post-hoc repair calls during timing runs to avoid distorting latency benchmarks.

### 5. Negative Speculative Speedup
- Speculative decoding speedup requires:
  $$\alpha > \frac{1 + c}{K}$$
  where $\alpha$ is acceptance rate and $c$ is the relative draft-to-target forward pass cost.
- If draft length $K$ is too large or draft quality is low on complex fact-checking texts, speedup may be $< 1.0$.
- Check `latency_breakdown.csv` and `acceptance_by_category.csv` to diagnose where rejections are concentrated.

### 6. Interrupted Execution & Recovery
- Run directories are immutable. If a run was interrupted:
  - Check `predictions.jsonl` and `trial_metrics.jsonl` for already completed claim IDs.
  - Choose a new run ID (e.g. `m2-run-resumed`) and pass `--limit` or filtered benchmark input.
