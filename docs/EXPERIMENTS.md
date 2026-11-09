# SPADE M1–M3 Experiment Matrix

This document defines the baseline experimental matrix for comparing autoregressive and speculative decoding policies under exact match conditions on the AVeriTeC fact-checking benchmark.

---

## 1. Official Experiment Matrix (E0–E6)

| ID | Configuration | Decoding Method | Speculation Length ($K$) | Target Model | Draft Model | Sampling Settings | Cache Strategy |
|---|---|---|---|---|---|---|---|
| **E0** | B0 Reference | Target-only greedy | N/A | Qwen3-8B | None | Deterministic ($\arg\max$) | `kv_cache` / `recompute` |
| **E1** | B1 Speculative | Greedy speculative | 4 | Qwen3-8B | Qwen3-0.6B | Deterministic ($\arg\max$) | `kv_cache` / `recompute` |
| **E2** | B0-S Reference | Target-only sampling | N/A | Qwen3-8B | None | $T=0.7$, Top-P=0.9 | `kv_cache` / `recompute` |
| **E3** | B2 Speculative | Exact speculative sampling | 2 | Qwen3-8B | Qwen3-0.6B | $T=0.7$, Top-P=0.9 | `kv_cache` / `recompute` |
| **E4** | B2 Speculative | Exact speculative sampling | 4 | Qwen3-8B | Qwen3-0.6B | $T=0.7$, Top-P=0.9 | `kv_cache` / `recompute` |
| **E5** | B2 Speculative | Exact speculative sampling | 6 | Qwen3-8B | Qwen3-0.6B | $T=0.7$, Top-P=0.9 | `kv_cache` / `recompute` |
| **E6** | B2 Speculative | Exact speculative sampling | 8 | Qwen3-8B | Qwen3-0.6B | $T=0.7$, Top-P=0.9 | `kv_cache` / `recompute` |

---

## 2. Experimental Controls & Guidelines

1. **Greedy vs. Stochastic Separation**:
   - Greedy runs (E0, E1) must only be compared against each other. Output token IDs must match exactly under equivalent numerical settings.
   - Stochastic runs (E2–E6) must only be compared against each other with identical sampling distributions ($T=0.7$, $p=0.9$).
2. **Output Length Normalization**:
   - Both target and speculative runs record total output tokens to prevent distortions in throughput and speedup metrics.
3. **Hardware & Environment Metadata**:
   - Every run stores an immutable `environment.json` containing GPU model, VRAM, PyTorch/CUDA versions, Transformers version, and Git commit hash.
4. **Reproducibility**:
   - Random seeds are fixed and explicitly recorded in `config_resolved.json`.
