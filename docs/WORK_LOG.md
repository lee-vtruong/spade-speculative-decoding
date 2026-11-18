# SPADE Research Work Log

## 2026-11-18 — Post-`a6a3613` Audit, Sensitivity Study Design, Disjoint Split Discipline, and 5-Way Ablation Pipeline

### 1. Audit of `unified-v2` (Commit `a6a3613`, Read-Only)
- Audited `outputs/unified_v2_report.json` locally via `scripts/audit_unified_runs.py` without modifying any existing data.
- **Result:** `PASS` across all 5 methods (`B0-S`, `HF-Assisted K=4`, `B2 K=4`, `SPADE Fixed K=4`, `SPADE Dynamic K`):
  - Exact token conservation verified: $N_{\text{acc}} + N_{\text{res}} + N_{\text{bon}} = N_{\text{output}}$ (`B2`: `6752+1432+1264=9448`; `SPADE Fixed`: `6767+1412+1267=9446`; `SPADE Dynamic`: `6923+1444+817=9184`).
  - No `residual_replacement` double-counting: $N_{\text{eval}} = N_{\text{acc}} + N_{\text{res}}$ (`8184`, `8179`, `8367`) and $\alpha_{\text{eval}}$ lies strictly within $[\alpha_{\text{ev}}, \alpha_{\text{non-ev}}]$.
  - Environment uniform across all 5 runs: commit `a6a3613`, `git_dirty: false`, PyTorch `2.13.0+cu130`, Transformers `5.14.1`.
  - Server-side raw trace reconciliation (`round_traces.jsonl`, `token_traces.jsonl`, `trial_metrics.jsonl`, `predictions.jsonl`) added to `scripts/audit_unified_runs.py --runs-root outputs/runs` for execution on the GPU server.

### 2. Evidence Logit Biasing Sensitivity Study (Exploratory Tuning Only)
- **Tuning Split:** Fixed exploratory tuning set `outputs/benchmarks/dev-oracle-pilot100.jsonl` (100 claims). Because `pilot100` has already been used for prior exploratory runs, it is strictly designated as the **exploratory tuning split** and excluded from final evaluation.
- **Grid:** $\beta_0 \in \{0.0, 0.25, 0.5, 1.0, 1.5, 2.0\}$ at fixed speculation horizon $K=4$, keeping all other decoding parameters fixed (`temp=0.7`, `top_p=0.9`, `max_new_tokens=384`, `seed=42`, `global_warmup=1`, `order_decay=0.5`, `entropy_threshold=2.0`, `max_boost_cap=3.0`).
- **Two Separate Variants:**
  1. `static`: Static evidence biasing without entropy gating (`--no-entropy-gating`).
  2. `entropy`: Entropy-gated evidence biasing (`adaptive_entropy=True`).
- **Measured Metrics:** `acceptance_eval`, `acceptance_prop`, `evidence_overlap_acceptance`, `non_overlap_acceptance`, `target_rounds`, `proposals/committed`, `draft_latency` (ms/round & % decode), `verification_latency` (ms/round & % decode), `throughput_e2e` (tok/s), `throughput_decode` (tok/s), `latency_e2e_ms_per_tok`, JSON validity, citation validity, verdict accuracy, and macro-F1.
- **Selection Rule:** Select hyperparameters strictly by **highest measured end-to-end throughput** (equivalently lowest end-to-end `ms/tok`), subject to quality guardrails (`json_valid >= 99%`, `citation_valid >= 99%`, `audit_passed == True`), NOT solely by highest acceptance rate. Both the throughput-selected config and the highest-acceptance config are reported side-by-side.

### 3. Strictly Disjoint Final Evaluation Split
- Added `scripts/prepare_eval_splits.py` to generate a held-out final evaluation split (e.g., `outputs/benchmarks/dev-oracle-heldout400.jsonl` from `dev-oracle-500.jsonl` minus all `claim_id`s in `dev-oracle-pilot100.jsonl`) with a cryptographic manifest asserting `overlap_count == 0` and `strictly_disjoint: true`.
- `scripts/run_ablation_suite.py` programmatically verifies zero claim overlap against the tuning split before launching any final evaluation run.

### 4. Locked 5-Way Ablation & Honest Statistical Protocol
- Added `--beta0`, `--order-decay`, `--entropy-threshold`, `--max-boost-cap`, and `--no-biasing` to `scripts/run_baselines.py` (and `enable_biasing` to `QwenPair.generate`) so `Dynamic-only` runs cleanly with zero biasing overhead.
- **5-Way Ablation Arms** (all using hyperparameters locked from the sensitivity study via `configs/locked_ablation_hparams.json`):
  1. `B2` ($K=4$, exact speculative sampling baseline)
  2. `Bias-only` ($K=4$, static biasing, locked `bias_only_beta0`)
  3. `Bias+Entropy` ($K=4$, entropy-gated biasing, locked `bias_entropy_beta0`)
  4. `Dynamic-only` ($K(t) \in \{2,4,6\}$, `--no-biasing`)
  5. `Full-SPADE` ($K(t) \in \{2,4,6\}$, biasing + entropy gating, locked `full_spade_beta0`)
  *(Optional `--include-references` adds `B0-S` and `HF-Assisted K=4` on the same held-out split.)*
- **Statistical Reporting (`scripts/analyze_ablation.py`):** Paired claim-cluster bootstrap ($B=10{,}000$) reporting actual effect sizes, 95% percentile CIs, two-sided p-values, claim win/loss/tie counts, and explicit interpretation (`positive_ci_excludes_zero`, `negative_ci_excludes_zero`, `inconclusive_ci_crosses_zero`). No p-hacking; negative/null effects are reported transparently.

### 5. Manuscript Policy
- `docs/paper/` was **NOT modified**. No new speedup claims or paper edits will be made until real GPU results from the sensitivity study and held-out ablation are collected and audited.

---

## 2026-11-18 (Phase 2) — Sensitivity Study Results & M4.1 Distribution-Level Evidence Biasing Audit

### 1. Summary of Sensitivity Study Findings (`sens-k4-tune100`, Commit `2fa8cdd`, 12/12 `PASS`)
- Raw trace reconciliation of `unified-v2` passed 5/5 runs (`outputs/analysis/unified_v2_raw_audit.json`).
- Across $\beta_0 \in \{0.0, 0.25, 0.5, 1.0, 1.5, 2.0\}$ on `pilot100` ($K=4$):
  - **Static biasing (`--no-entropy-gating`):** monotonically degrades $\alpha_{\text{ev}}$ at $\beta_0 \ge 1.0$ (`79.65% -> 78.96% -> 78.54% -> 78.07%`, $-1.58\text{ pp}$ at $\beta_0=2.0$, $p=0.073$).
  - **Entropy-gated biasing:** avoids high-$\beta_0$ degradation ($\alpha_{\text{ev}}$ peaks at `80.11%` at $\beta_0=0.5$, $\alpha_{\text{eval}}$ peaks at `82.74%` at $\beta_0=1.5$), but every non-zero $\beta_0$ yields a small effect whose claim-clustered 95% CI crosses zero, and E2E throughput remains within GPU clock jitter (`25.94–26.37 tok/s`).
- **Decision:** Hold off on running the 393-claim held-out ablation. Execute **M4.1 — Distribution-Level Evidence Biasing Audit** first to diagnose *why* biasing lift is tiny at the $(p, q_{\text{base}}, q_{\text{biased}})$ level.

### 2. M4.1 Implementation (Read-Only Replay + Claim-Clustered Offline Audit)
- `src/spade/analysis/distribution_audit.py`: Computes exact `TV(p, q_base)`, `TV(p, q_biased)`, `JS(p, q_base)`, `JS(p, q_biased)`, exact per-prefix expected acceptance $1 - \operatorname{TV}(p, q) = \sum_v \min(p(v), q(v))$, target/draft probability mass on evidence candidates $p(S_E)$ vs $q(S_E)$, mass deficit $p(S_E) - q_{\text{base}}(S_E)$, draft entropy, gating value, n-gram match order, candidate count, and microsecond biasing overhead.
- `scripts/run_distribution_audit.py`: Replays fixed teacher-forced prefixes from read-only `outputs/runs/unified-v2-b0s-lim100/predictions.jsonl` on `pilot100` in one batched forward pass per model per claim (never overwrites existing `unified-v2` or sensitivity outputs).
- `scripts/analyze_distribution_audit.py`: CPU-only analysis computing claim-clustered paired bootstrap ($B=10{,}000$, strictly by `claim_id`, never independent token samples), answering all 5 core audit questions, strictly separating per-prefix expected acceptance ($1-\operatorname{TV}$) from observed stochastic acceptance, and rendering 5 real-data diagnostic PNG figures.

### 3. Pre-Registered Decision Criteria (Keep vs. Modify vs. Drop Evidence Biasing)
- **KEEP (`KEEP_CURRENT_BIASING`):** All-prefix $\Delta(1 - \operatorname{TV}) \ge +1.0\text{ pp}$ with claim-clustered 95% CI strictly $> 0$ AND expected round-time savings exceed measured biasing overhead.
- **MODIFY (`TARGETED_MODIFICATION_CANDIDATE`, max 1–2 changes):** Overall lift $< +1.0\text{ pp}$, BUT audit identifies a concrete failure regime (e.g., `order=1` unigram matches or large $|S_E|$ with $p(S_E) \approx 0$) AND a clean positive regime achieving $\ge +0.50\text{ pp}$ lift (95% CI $> 0$).
- **DROP / REFRAME (`DROP_OR_REFRAME_AS_NEGATIVE_RESULT`):** Neither overall nor any clean subgroup achieves $\ge +0.50\text{ pp}$ with 95% CI $> 0$, or biasing overhead exceeds theoretical maximum round savings. No new algorithm implemented prior to running M4.1 on GPU.

