# SPADE: Speculative Prediction for Adaptive Decoding with Evidence

Research foundation for **Milestones M1, M2, and M3**:
- **M1**: Fact-checking summarization benchmark construction on AVeriTeC (oracle QA & retrieved evidence modes, strict leakage prevention).
- **M2**: Autoregressive and standard speculative decoding baselines (B0 greedy, B0-S sampling, B1 greedy speculative, B2 exact speculative sampling via Leviathan et al. ICML 2023, KV-cache rollback engine).
- **M3**: Token-level rejection analysis, profiling, probability diagnostics, and visualization suite.

---

## Architecture Overview

```text
SPADE/
├── configs/               # Versioned experiment configs (benchmark, decoding, logging)
├── src/
│   ├── benchmark/         # AVeriTeC loader, evidence adapters, prompt builder, evaluation
│   ├── decoding/          # Autoregressive, speculative, exact sampling, KV-cache rollback
│   ├── instrumentation/   # Trace schemas, append-only JSONL logger, profiling
│   └── analysis/          # Metrics aggregation, M3 plotting suite, research reporting
├── scripts/               # CLIs for benchmark prep, baseline runs, evaluation, analysis
├── tests/                 # 100% CPU-executable unit tests (74 tests)
│   ├── fixtures/          # Synthetic AVeriTeC and knowledge-store fixtures
│   ├── test_dataset.py
│   ├── test_prompt_leakage.py
│   ├── test_sampling_math.py
│   ├── test_speculative_correctness.py
│   ├── test_cache_handling.py
│   └── test_logging.py
├── notebooks/             # m3_rejection_analysis.ipynb
├── docs/                  # SERVER_RUNBOOK.md, M1_M3_REPORT.md, EXPERIMENTS.md
├── data/                  # Remote server only
├── models/                # Remote server only
└── outputs/               # Remote server only
```

---

## Local Development & Validation (CPU Only)

Local development does **not** download models or datasets. All tests run on CPU using synthetic fixtures:

```powershell
# Create virtual environment and install dev requirements
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps

# Execute complete test suite (74 tests)
python -m pytest
```

---

## Remote Server Execution (GPU Inference)

Real model checkpoints (`Qwen/Qwen3-8B`, `Qwen/Qwen3-0.6B`) and full AVeriTeC datasets are executed on the remote Linux GPU server.

See **[docs/SERVER_RUNBOOK.md](docs/SERVER_RUNBOOK.md)** for complete step-by-step commands covering:
- Stage A: Repository synchronization
- Stage B: Environment setup
- Stage C: AVeriTeC download & audit
- Stage D: Qwen model download
- Stage E: M1 benchmark preparation
- Stage F: M2 baseline execution (B0, B0-S, B1, B2 sweeps)
- Stage G: M3 rejection analysis & visualization
- Stage H: Troubleshooting (CUDA OOM, BF16, tokenizer compatibility, etc.)

---

## Milestone Status & Verification Report

See **[docs/M1_M3_REPORT.md](docs/M1_M3_REPORT.md)** for the complete scientific status report, mathematical verification, and research questions for Milestone M4.
