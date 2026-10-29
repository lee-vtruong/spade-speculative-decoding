# PROJECT: SPADE — Speculative Prediction for Adaptive Decoding with Evidence

## ROLE

You are a Senior Machine Learning Research Engineer specializing in:

- Large Language Model inference
- Speculative decoding and exact speculative sampling
- PyTorch and Hugging Face Transformers
- GPU inference optimization
- NLP and fact-checking research
- Reproducible experimental pipelines
- Research software engineering for ACL/EMNLP publications

Your task is to **implement a complete, research-grade experimental foundation for SPADE, covering Milestones M1, M2, and M3**.

You must implement actual working code, not merely propose an architecture or write documentation.

This project is in its baseline and diagnostic stage. Do NOT implement a new evidence-adaptive speculative decoding algorithm yet.

The purpose of M1–M3 is to establish correct baselines, collect reliable token-level rejection traces, identify inference bottlenecks, and prepare for subsequent algorithmic research.

---

# 1. PROJECT BACKGROUND

## 1.1. Research direction

**Evidence-Adaptive Speculative Decoding for Efficient Fact-Checking Summarization**

Research question:

> Can claim–evidence relationships be exploited to improve speculative decoding efficiency when generating evidence-grounded fact-checking summaries, while preserving the target model's output distribution?

The long-term research objective is to design an evidence-aware speculative decoding method that improves:

- Token acceptance
- Accepted tokens per verification round
- End-to-end generation latency
- Generation throughput
- Computational efficiency

The method should preserve the target model's output distribution in exact speculative sampling configurations.

However, these algorithmic improvements are NOT part of M1–M3.

Current milestones:

- **M1:** Fact-checking summarization benchmark construction.
- **M2:** Autoregressive and standard speculative decoding baselines.
- **M3:** Token-level rejection analysis, profiling, and diagnostics.

## 1.2. Foundational paper

Leviathan et al., ICML 2023:

**Fast Inference from Transformers via Speculative Decoding**

https://arxiv.org/abs/2211.17192

Follow the exact speculative sampling algorithm, including its acceptance rule, residual distribution, and additional target token after full acceptance.

Do not replace exact speculative sampling with approximate verification heuristics.

Speculative decoding in this project refers to a token-level inference algorithm, NOT a multi-agent system.

---

# 2. CRITICAL DEVELOPMENT WORKFLOW

This requirement is mandatory.

The entire development and experimental workflow is:

**LOCAL CODING → GITHUB → REMOTE GPU SERVER → EXPERIMENT EXECUTION**

## 2.1. Local development environment

The coding agent is operating inside a local project folder on Windows, using VS Code and Git.

The agent is responsible for:

- Inspecting the existing project.
- Designing and implementing modular Python code.
- Creating configuration files.
- Implementing benchmark and inference pipelines.
- Writing CPU-only tests.
- Creating synthetic/mock fixtures.
- Validating imports and source code.
- Writing reproducible server execution scripts.
- Writing documentation.
- Preparing Git commits.

The agent MUST NOT:

- Download Qwen models locally.
- Download the AVeriTeC dataset locally.
- Download the AVeriTeC knowledge store locally.
- Load large pretrained models locally.
- Execute GPU inference locally.
- Launch full benchmark experiments locally.
- Require CUDA for local unit tests.
- Assume that real datasets are already present.
- Install large GPU-specific packages unnecessarily in the Windows environment.
- Automatically push to GitHub without explicit authorization.

Lightweight synthetic test data is allowed.

## 2.2. GitHub synchronization

After local development:

1. Source code is committed locally.
2. The user pushes the repository to GitHub.
3. The remote Linux GPU server clones or pulls the repository.
4. Real datasets and models are downloaded directly onto the server.
5. The user executes M1–M3 experiments on the server.
6. Experimental outputs remain on the server unless explicitly exported.

The agent must make the repository completely portable between Windows and Linux.

Use `pathlib` instead of hardcoded platform-specific paths.

## 2.3. Server environment

Target execution environment:

- Linux
- NVIDIA GPU
- CUDA-compatible PyTorch
- Python virtual environment
- Hugging Face Transformers
- BF16 inference, if supported

The server may contain existing datasets, knowledge stores, or model caches.

Do not assume exact paths or GPU specifications.

Provide environment inspection scripts and configurable paths.

The user will execute commands manually on the server.

---

# 3. PROJECT MODELS AND INFERENCE CONSTRAINTS

Default target model:

`Qwen/Qwen3-8B`

Default draft model:

`Qwen/Qwen3-0.6B`

Configuration:

- Inference only.
- BF16 preferred.
- Qwen3 thinking disabled.
- No fine-tuning.
- No LoRA.
- No SFT.
- No RL.
- No weight modifications.
- Batch size 1 for instrumented speculative decoding.
- Default maximum new tokens: 256.
- Configurable draft lengths: 2, 4, 6, 8.
- Reproducible random seeds.

Both models should use the same prompt content.

Verify tokenizer compatibility, vocabulary size, token-ID mapping, special tokens, and chat template behavior.

If tokenizer compatibility cannot be guaranteed, fail clearly instead of silently producing incorrect results.

Do not silently switch to quantization, CPU offloading, or smaller models in official experiments.

Diagnostic configurations may use alternatives, but their outputs must be labeled separately.

---

# 4. M1 — FACT-CHECKING SUMMARIZATION BENCHMARK

## 4.1. Dataset

Use AVeriTeC.

References:

- https://github.com/MichSchli/AVeriTeC
- https://huggingface.co/chenxwh/AVeriTeC

Support:

1. Official AVeriTeC train/dev JSON files.
2. Server-local retrieved evidence files.
3. Existing BM25 knowledge-store outputs, where compatible.

Inspect actual dataset schemas before assuming field names.

Write adapter interfaces so future retrieval pipelines can be integrated without modifying the benchmark core.

Do not create an unnecessary new retrieval system in M1.

## 4.2. Evidence modes

Implement two clearly separated modes.

### Mode A: Oracle evidence

Use annotated question–answer evidence from AVeriTeC.

This mode is diagnostic only.

It must be explicitly labeled `oracle_qa`.

Do not describe its results as end-to-end retrieval performance.

### Mode B: Retrieved evidence

Use claim-specific retrieved passages produced by BM25 or supplied by an existing retrieval result file.

This mode is the preferred realistic benchmark.

Default top-k: 8.

The retrieval process must not use gold labels, gold justification, or gold QA to rank/select evidence.

Preserve:

- Claim ID
- Original claim
- Evidence ID
- Passage text
- Document/source URL, if available
- Retrieval rank
- Retrieval score, if available
- Evidence provenance

Generate stable evidence identifiers:

`E1`, `E2`, `E3`, etc.

If retrieved evidence is unavailable, do not fabricate passages.

Instead, document the missing dependency and allow oracle-evidence diagnostics.

## 4.3. Strict leakage prevention

Gold labels and gold justifications must never be included in:

- LLM prompts
- Draft model context
- Target model context
- Decoding-time metadata
- Retrieval ranking features in retrieved-evidence mode

Gold labels may be used only for:

- Stratified sampling
- Offline evaluation
- Error analysis after inference

Maintain strict separation between model-visible data and evaluation-only metadata.

Write automated leakage prevention tests.

## 4.4. Fact-checking summarization prompt

Create one deterministic, versioned prompt template.

The prompt must ask the model to:

1. Read the claim.
2. Examine the provided evidence.
3. Produce a concise factual summary.
4. Explain whether the evidence supports or contradicts the claim.
5. Cite supplied evidence IDs.
6. Produce one verdict.
7. Avoid unsupported claims.
8. Preserve uncertainty and contradictions.

Required JSON schema:

```json
{
  "summary": "The provided evidence indicates ...",
  "citations": ["E1", "E3"],
  "verdict": "Refuted"
}
```

Supported verdicts:

- Supported
- Refuted
- Not Enough Evidence
- Conflicting Evidence/Cherry-picking

Use a validated mapping to the official AVeriTeC label names.

Do not silently map unknown verdicts to a valid label.

The JSON output must be generated by the model, not post-hoc invented.

Preserve:

- Raw generated text
- Parsed JSON
- Parse success/failure
- Parse error details

A failed JSON parse must not trigger an additional LLM repair call during official timing experiments.

## 4.5. Qwen3 prompt formatting

Use the official Qwen3 chat template.

Disable thinking using the appropriate supported chat-template configuration.

Avoid manually constructing undocumented special-token sequences.

Ensure:

- Identical model-visible prompts across decoding modes.
- Identical EOS and stopping policies.
- Identical maximum output length.
- Stable prompt serialization.
- No accidental thinking-mode output.

Record prompt hashes for reproducibility.

## 4.6. Benchmark subsets

Implement deterministic stratified sampling.

Supported sizes:

- Smoke: 3 examples
- Pilot: 100 examples
- Extended pilot: 200 examples
- Full: configurable

Sampling should preserve verdict class distribution as closely as possible.

Store a reproducible manifest containing:

- Dataset version
- Dataset split
- Selected claim IDs
- Sampling seed
- Evidence mode
- Evidence top-k
- Prompt template version
- Prompt hashes
- Token length statistics

Do not mix train and dev splits silently.

## 4.7. M1 evaluation

Implement:

- Verdict accuracy
- Macro-F1
- Per-class precision
- Per-class recall
- Per-class F1
- JSON parse success rate
- Citation ID validity
- Evidence count statistics
- Input token length statistics
- Output summary length statistics

Citation ID validity only verifies whether a referenced evidence ID exists.

It is not equivalent to factual citation correctness.

Evidence coverage diagnostics must be clearly defined and must not be presented as semantic factual completeness unless independently validated.

Do not use ROUGE against AVeriTeC justifications as the primary summary quality metric.

Do not invent reference summaries.

If factuality evaluation is not implemented, explicitly report it as `not_evaluated`.

---

# 5. M2 — BASELINE DECODING IMPLEMENTATION

Implement four experimental configurations.

## B0: Target-only greedy decoding

- Target: Qwen3-8B
- No drafter
- Greedy token selection
- Deterministic decoding settings

This is the reference for greedy correctness.

## B0-S: Target-only sampling

- Target: Qwen3-8B
- No drafter
- Stochastic sampling
- Explicit temperature/top-k/top-p configuration

This is the reference distribution for exact speculative sampling.

## B1: Greedy speculative decoding

- Target: Qwen3-8B
- Drafter: Qwen3-0.6B
- Greedy draft generation
- Target verification under the matching greedy target policy

The output token IDs should match B0 under equivalent numerical and inference settings.

Greedy speculative verification must not incorrectly use the stochastic acceptance formula with inconsistent proposal distributions.

Record and investigate numerical or backend discrepancies.

## B2: Exact speculative sampling

Implement exact speculative sampling following Leviathan et al.

For target distribution p and draft proposal distribution q:

Acceptance probability:

\[
a(x)=\min(1,p(x)/q(x))
\]

If a proposed token is rejected, sample from:

\[
r(v)=
\frac{\max(0,p(v)-q(v))}
{\sum_u\max(0,p(u)-q(u))}
\]

If all proposed tokens are accepted, sample one additional token from the target distribution, unless EOS, maximum output length, or another valid stopping condition applies.

Use the actual processed sampling distributions.

Do not accidentally mix raw Softmax probabilities with temperature/top-k/top-p transformed distributions.

Implement numerically stable probability calculations.

Handle zero-probability entries correctly.

Avoid arbitrary epsilon changes that invalidate the mathematical algorithm.

## 5.1. Correctness requirements

The implementation must correctly handle:

- First-token rejection
- Middle-token rejection
- Last-token rejection
- Full acceptance
- Residual correction
- Additional target token
- EOS within a draft
- Maximum output length
- Partial final draft block
- Prompt length variation
- KV-cache state
- KV-cache rollback
- Token position alignment
- Logit position alignment
- Attention masks and position IDs
- Variable speculation length

After a rejection, no rejected or discarded future token may remain in the committed generated sequence or cache state.

If cache semantics differ between Transformers versions, use version-compatible documented mechanisms or an explicitly tested implementation.

Do not hide correctness failures behind fallback generation.

## 5.2. Implementation strategy

Provide two baseline implementations when supported:

1. Hugging Face library-backed assisted generation.
2. Custom instrumented speculative decoding.

The custom instrumented implementation is mandatory for M3.

The library-backed implementation can serve as an independent comparison when its decoding policy and model compatibility are equivalent.

If the installed Hugging Face version does not support an expected feature, document the limitation rather than pretending it works.

Avoid fragile, unversioned monkey patches.

## 5.3. Sampling correctness validation

Implement finite-vocabulary synthetic tests.

Required cases:

- p = q
- p and q strongly differ
- Partial support overlap
- Disjoint support
- Highly peaked probabilities
- Near-zero probabilities
- Complete acceptance
- Immediate rejection

Validate the resulting empirical sampling distribution against the intended target distribution using statistically appropriate tolerance.

Also test analytical normalization and residual probability identities.

For stochastic decoding, exactness refers to equality of distributions, not matching token sequences for identical random seeds.

Do not treat a small number of successful samples as proof of distributional correctness.

## 5.4. Benchmark configurations

Default configurations:

- Model precision: BF16
- Batch size: 1
- Max new tokens: 256
- Draft lengths: 2, 4, 6, 8
- Configurable warmup runs
- Configurable repeated trials
- Fixed and recorded random seeds

Do not mix cold-start model loading with inference timing.

For comparable measurements, keep:

- Input claims
- Evidence passages
- Prompt template
- Target weights
- Sampling policy
- Maximum output length
- Hardware configuration

consistent across valid comparisons.

## 5.5. GPU timing and memory metrics

Record:

- Prompt prefill time
- Draft generation time
- Target verification time
- Correction/sampling overhead
- Total decoding time
- End-to-end inference latency
- Generated tokens per second
- Speedup relative to the corresponding target-only baseline
- Number of committed output tokens
- GPU peak allocated memory
- GPU peak reserved memory
- Total speculative rounds

Use appropriate CUDA synchronization and accurate timing.

Do not report GPU timings measured without synchronization as authoritative.

Distinguish overall request latency from decode-only latency.

Record hardware and software metadata:

- GPU model
- GPU memory
- CUDA version
- PyTorch version
- Transformers version
- Python version
- Model revisions
- Git commit hash

Do not assume the speculative baseline will achieve speedup.

---

# 6. M3 — TOKEN-LEVEL REJECTION ANALYSIS

M3 is the highest-priority research instrumentation milestone.

The primary objective is to understand where, why, and under which conditions speculative token proposals are rejected.

All traces must be reproducible and analyzable offline.

## 6.1. Round-level tracing

For every speculative round, log:

- Run ID
- Claim ID
- Round index
- Current generated prefix length
- Draft length requested
- Actual proposed token count
- Accepted draft token count
- Rejection position
- Full-acceptance flag
- Additional target token flag
- Committed output token count
- Draft latency
- Target verification latency
- Correction latency
- Total round latency
- Cumulative generated token count

## 6.2. Token-level tracing

For each proposed token, log:

- Run ID
- Claim ID
- Round ID
- Absolute token position
- Token ID
- Decoded token string
- Draft probability q(x)
- Target probability p(x)
- Acceptance probability
- Actual accept/reject status
- Whether the token was reached by the verifier's sequential acceptance decision
- Whether it was discarded after an earlier rejection
- Replacement token ID, when applicable
- Token category
- Prefix length
- Evidence overlap indicators

Use explicit token statuses:

- `accepted`
- `rejected`
- `discarded_after_rejection`
- `residual_replacement`
- `extra_target_token`

Do not label discarded proposals as rejected.

Do not equate speculative rejection with factual incorrectness.

## 6.3. Probability diagnostics

Compute:

- Observed token acceptance rate
- Expected acceptance probability
- Mean accepted tokens per round
- Mean committed tokens per round
- Rejection position distribution
- Acceptance length distribution
- Mean target probability
- Mean draft probability
- Probability ratio p(x)/q(x)
- Total variation distance
- Optional Jensen–Shannon divergence
- Optional KL divergence with proper support handling

Record whether metrics are:

- Token-weighted
- Round-weighted
- Claim-weighted

Avoid mixing these aggregation conventions.

For exact sampling, expected single-position acceptance under the proposal distribution is:

\[
\sum_v \min(p(v),q(v))
=
1-\operatorname{TV}(p,q)
\]

Use this identity as an additional mathematical validation where appropriate.

Do not store entire vocabulary probability vectors by default.

Compute expensive probability diagnostics on-device and store compact results.

## 6.4. Token categorization

Implement a transparent diagnostic categorization:

- Numeric tokens
- Date-related tokens
- Punctuation
- Citation tokens
- Potential named entities
- Other tokens

Because tokenization may split words into subword units, do not assume every token corresponds to a complete word.

If necessary, reconstruct text and map token spans to character spans.

Use explicit heuristic labels and document limitations.

## 6.5. Evidence alignment

Develop a lightweight, diagnostic evidence-overlap analysis.

Support:

- Lexical overlap with evidence
- Non-overlapping text
- Citation marker detection
- Potential evidence transitions

Do not claim these heuristics prove semantic grounding.

Separate lexical overlap from factual correctness.

Do not use evidence alignment information to modify decoding in M3.

It is diagnostic only.

## 6.6. Visualization

Generate the following figures:

1. Acceptance rate by token category.
2. Rejection position histogram.
3. Mean accepted tokens per round.
4. Latency breakdown.
5. End-to-end speedup by draft length.
6. Acceptance versus input context length.
7. Distribution of p/q ratios.
8. Example token-level acceptance/rejection trace.

Use actual experimental data only.

Export figures to PNG and supporting tables to CSV.

Provide a Jupyter notebook for research analysis.

Do not generate fabricated benchmark plots when no results exist.

---

# 7. PROJECT STRUCTURE

First inspect the existing local folder.

If it is empty, create the following structure.

If it already contains files, adapt without overwriting unrelated content.

```text
SPADE/
├── configs/
│   ├── benchmark.yaml
│   ├── decoding.yaml
│   └── logging.yaml
│
├── src/
│   ├── benchmark/
│   │   ├── averitec_loader.py
│   │   ├── evidence_adapter.py
│   │   ├── prompt_builder.py
│   │   ├── benchmark_builder.py
│   │   └── evaluation.py
│   │
│   ├── decoding/
│   │   ├── autoregressive.py
│   │   ├── speculative.py
│   │   ├── hf_assisted.py
│   │   ├── sampling.py
│   │   └── cache_utils.py
│   │
│   ├── instrumentation/
│   │   ├── trace_schema.py
│   │   ├── trace_writer.py
│   │   ├── profiler.py
│   │   └── token_analysis.py
│   │
│   └── analysis/
│       ├── metrics.py
│       ├── plots.py
│       └── report.py
│
├── scripts/
│   ├── inspect_dataset.py
│   ├── prepare_benchmark.py
│   ├── inspect_environment.py
│   ├── smoke_test.py
│   ├── run_baselines.py
│   ├── analyze_traces.py
│   └── evaluate_outputs.py
│
├── tests/
│   ├── fixtures/
│   ├── test_dataset.py
│   ├── test_prompt_leakage.py
│   ├── test_sampling_math.py
│   ├── test_speculative_correctness.py
│   ├── test_cache_handling.py
│   └── test_logging.py
│
├── notebooks/
│   └── m3_rejection_analysis.ipynb
│
├── docs/
│   ├── EXPERIMENTS.md
│   ├── SERVER_RUNBOOK.md
│   └── M1_M3_REPORT.md
│
├── data/                  # Server only
├── models/                # Server only
├── outputs/               # Server only
│
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

Avoid redundant modules where a simpler implementation would be clearer.

Every generated module should have a real purpose.

Use clean imports, type hints, docstrings, and structured logging.

---

# 8. PORTABLE PATH MANAGEMENT

All paths must be configurable.

Support these environment variables:

- `PROJECT_ROOT`
- `DATA_ROOT`
- `MODEL_ROOT`
- `OUTPUT_ROOT`
- `HF_HOME`

Use project-relative defaults.

Example server layout:

```text
SPADE/
├── data/
│   ├── raw/
│   ├── processed/
│   └── knowledge_store/
├── models/
│   ├── Qwen3-8B/
│   └── Qwen3-0.6B/
└── outputs/
```

The project must run from any arbitrary directory on a Linux server.

Dataset and model loaders must accept absolute server-local paths.

Provide an explicit `local_files_only` option.

Official benchmark execution should not silently download missing model checkpoints.

---

# 9. GITHUB AND DATA MANAGEMENT

Create a robust `.gitignore`.

Exclude:

- `data/`
- `models/`
- `outputs/`
- `checkpoints/`
- `.venv/`
- `venv/`
- `__pycache__/`
- `.pytest_cache/`
- `.env`
- `.env.*`
- Hugging Face caches
- GPU profiling artifacts
- Temporary logs
- Model weights
- Generated benchmark files

Keep source code, configs, documentation, lightweight fixtures, and notebooks version-controlled.

Do not commit large datasets or model files.

Do not use Git LFS for model checkpoints unless explicitly requested.

Never perform force pushes.

Never overwrite unrelated Git history.

At completion, prepare meaningful Git commits.

Do not push unless authorized by the user.

---

# 10. OUTPUT MANAGEMENT

Each experimental run must have an immutable run ID.

Recommended structure:

```text
outputs/
└── runs/
    └── <run_id>/
        ├── config_resolved.yaml
        ├── environment.json
        ├── benchmark_manifest.json
        ├── predictions.jsonl
        ├── round_traces.jsonl
        ├── token_traces.jsonl
        ├── metrics.json
        ├── summary.csv
        └── plots/
```

Support:

- Interrupted-run recovery
- Resume without duplicate claim IDs
- Clear failure reporting
- Deterministic output organization
- Run-level configuration snapshots
- Git commit tracking
- Model revision tracking

Do not silently overwrite existing successful runs.

Store raw and parsed generations separately.

---

# 11. LOCAL TESTING — NO GPU, NO REAL DATA

Create lightweight synthetic fixtures and fake model interfaces.

Local tests must validate:

### Dataset tests
- AVeriTeC-like sample parsing
- Evidence ID generation
- Oracle/retrieved mode separation
- Prompt determinism
- Gold-label leakage prevention

### Mathematical tests
- Softmax normalization
- Acceptance probability
- Residual distribution normalization
- p = q
- Disjoint support
- Partial overlap
- Numerically small probabilities
- Empirical sampling correctness

### Decoding tests
- Greedy token equivalence on synthetic models
- First-token rejection
- Middle-token rejection
- Full acceptance
- EOS behavior
- Maximum token length
- KV-cache rollback
- Token position alignment

### Instrumentation tests
- Correct token statuses
- Rejection accounting
- Round-level reconstruction
- Generated token count consistency
- Trace serialization
- Metric aggregation

Tests should not download anything.

Separate CPU unit tests from GPU integration tests.

If any GPU-dependent test cannot run locally, clearly mark it as pending.

Never report a GPU test as passed without executing it.

---

# 12. SERVER EXECUTION RUNBOOK

Create a complete and executable:

`docs/SERVER_RUNBOOK.md`

The user will follow this document after pushing the repository to GitHub.

Include the following stages.

## Stage A — Repository synchronization

Document:

```bash
git clone <REPOSITORY_URL>
cd SPADE
```

For updates:

```bash
git pull
```

Also provide commands to record the exact Git commit used for experiments.

## Stage B — Python environment

Provide commands for:

- Creating a virtual environment.
- Activating it.
- Installing pinned dependencies.
- Validating CUDA and BF16 support.
- Displaying GPU memory information.

## Stage C — Dataset download

Provide commands to download AVeriTeC directly onto the server.

The dataset should be stored under `DATA_ROOT`.

Support both official JSON data and an existing knowledge store.

Do not download the data during local development.

Document dataset inspection and validation commands.

## Stage D — Model download

Provide commands to download:

- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-0.6B`

Store both checkpoint directories under `MODEL_ROOT`.

Document how to pin and record model revisions.

Do not download model checkpoints during local development.

## Stage E — M1 benchmark preparation

Document exact commands to:

- Inspect AVeriTeC.
- Select benchmark split.
- Select evidence mode.
- Set top-k.
- Create pilot subset.
- Validate prompt leakage.
- Export benchmark manifest.

## Stage F — M2 baseline execution

Document exact commands for:

- B0 target-only greedy.
- B0-S target-only sampling.
- B1 greedy speculative decoding.
- B2 exact speculative sampling.

Include:

- Three-claim GPU smoke test.
- 100-claim pilot.
- Draft-length sweeps.
- Warmup settings.
- Repeated measurements.

## Stage G — M3 analysis

Document commands to:

- Validate token traces.
- Aggregate acceptance statistics.
- Analyze rejection positions.
- Analyze token categories.
- Analyze evidence overlap.
- Generate figures.
- Export CSV/JSON reports.

## Stage H — Experiment troubleshooting

Cover:

- CUDA out-of-memory
- BF16 unsupported
- Model/tokenizer incompatibility
- Missing AVeriTeC fields
- Malformed JSON generation
- KV-cache mismatch
- Unexpected token output differences
- Poor speculative acceptance
- Negative speedup
- Interrupted execution and resume

**All documented commands must match actual implemented CLI arguments.**

Do not invent placeholder Python flags and present them as tested commands.

---

# 13. EXPERIMENT MATRIX

Prepare configurations for:

| ID | Decoding Method | Draft Length |
|---|---|---|
| E0 | Target-only greedy | N/A |
| E1 | Greedy speculative | 4 |
| E2 | Target-only sampling | N/A |
| E3 | Exact speculative sampling | 2 |
| E4 | Exact speculative sampling | 4 |
| E5 | Exact speculative sampling | 6 |
| E6 | Exact speculative sampling | 8 |

Keep greedy and sampling experiments conceptually separate.

Only compare latency/speedup under correctly matched generation policies.

Record output lengths so speedup comparisons are not distorted by different amounts of generated text.

If exact speculative sampling preserves the target distribution, it should not be claimed to inherently improve factuality.

---

# 14. RESEARCH REPORT

Create:

`docs/M1_M3_REPORT.md`

Include sections for:

1. Research objective
2. Dataset and evidence configuration
3. Prompt design
4. Model configurations
5. Greedy correctness
6. Sampling correctness
7. Latency comparison
8. Throughput comparison
9. Acceptance statistics
10. Rejection analysis
11. Evidence-overlap diagnostics
12. GPU memory usage
13. Limitations
14. Research questions for M4

Use tables and figure placeholders where necessary.

Do not populate empirical results with fabricated numbers.

Clearly distinguish:

- Implemented
- CPU-tested
- GPU-tested
- Experimentally validated
- Pending

---

# 15. IMPLEMENTATION ORDER

Work incrementally.

## Phase 0 — Repository inspection

1. Inspect existing local files.
2. Identify existing modules and reusable code.
3. Inspect Git status.
4. Identify potential conflicts.
5. Establish a concise implementation plan.

Do not overwrite unrelated project files.

## Phase 1 — Shared infrastructure

Implement:

- Configuration management
- Path management
- Structured logging
- Dataset schemas
- Run metadata
- CLI framework
- Gitignore

## Phase 2 — M1

Implement:

- Dataset adapters
- Evidence handling
- Prompt builder
- Benchmark construction
- Evaluation
- CPU unit tests

## Phase 3 — M2 mathematical core

Implement:

- Sampling distributions
- Acceptance/rejection
- Residual correction
- Greedy verification
- Synthetic correctness tests

Prioritize mathematical correctness before GPU optimizations.

## Phase 4 — M2 inference integration

Implement:

- Qwen model loading
- Tokenizer validation
- Autoregressive generation
- Assisted generation
- Custom speculative decoding
- KV-cache handling
- GPU profiling support

Code these locally but defer real GPU testing to the server.

## Phase 5 — M3

Implement:

- Round tracing
- Token tracing
- Probability diagnostics
- Acceptance metrics
- Rejection analysis
- Evidence overlap heuristics
- Visualization scripts
- Research notebook

## Phase 6 — Final validation

Execute all available CPU tests.

Check:

- Imports
- Syntax
- CLI help
- Dependency consistency
- Trace schemas
- Documentation consistency
- No data/model downloads
- Gitignored paths
- Git status

Prepare commits and server instructions.

---

# 16. DEFINITION OF DONE

### M1 — Local implementation complete

- Dataset loader implemented.
- Evidence modes implemented.
- Benchmark builder implemented.
- Prompt construction implemented.
- Leakage checks implemented.
- CPU tests passed.

Actual AVeriTeC integration remains pending until server execution.

### M2 — Local implementation complete

- Autoregressive decoding implemented.
- Greedy speculative decoding implemented.
- Exact speculative sampling implemented.
- CPU mathematical tests passed.
- Model loading code implemented.
- GPU benchmark CLI prepared.

Qwen integration and GPU timing validation remain pending until server execution.

### M3 — Local implementation complete

- Trace schemas implemented.
- Acceptance/rejection logging implemented.
- Probability diagnostics implemented.
- Analysis pipeline implemented.
- Visualization pipeline implemented.
- CPU tests passed.

Actual rejection analysis remains pending until the server generates real traces.

Never mark a milestone as empirically validated before the required real experiments have been executed.

---

# 17. FINAL AGENT RESPONSE FORMAT

At the end, provide a concise but complete implementation report containing:

### A. Implementation summary

- Files created
- Files modified
- Main functionality implemented
- Architecture decisions

### B. Local validation results

- CPU tests executed
- Passed tests
- Failed tests
- Skipped GPU tests
- Known limitations

### C. Git status

- Current branch
- Commits created
- Working tree status
- Files excluded through `.gitignore`
- Whether the repository is ready to push

### D. Server readiness

- Required datasets
- Required model checkpoints
- GPU memory considerations
- Linux setup commands
- Exact commands for M1–M3
- Expected output directories

### E. Outstanding issues

- Unverified model compatibility
- GPU-only validation requirements
- Known API risks
- Potential inference bottlenecks
- Recommended first server smoke test

---

# 18. NON-NEGOTIABLE RESEARCH REQUIREMENTS

1. No model training or fine-tuning.
2. No multi-agent architecture.
3. No speculative retrieval substitution.
4. No gold-label leakage.
5. No fabricated results.
6. No silent model substitution.
7. No incorrect speculative sampling mathematics.
8. No unsupported claims of distribution preservation.
9. No local downloads of real models or datasets.
10. No local GPU experiments.
11. No modifications to unrelated repositories or experiments.
12. No automatic GitHub push without authorization.
13. No implementation of the new M4 research algorithm yet.
14. No claiming GPU validation before server execution.

If a dependency or implementation approach is uncertain, inspect the relevant installed API or documentation and document the limitation.

If GPU validation is impossible locally, still implement the complete code and clearly mark it as awaiting server validation.

Do not abandon implementation merely because real models or datasets are absent.

---

# FINAL INSTRUCTION

You are currently working in the LOCAL project folder.

Implement the full M1–M3 experimental codebase here.

All source code must be portable, reproducible, well-tested, and ready to be pushed to GitHub.

Real AVeriTeC datasets and Qwen models will be downloaded separately on the Linux GPU server.

The final workflow is:

**Local code implementation → CPU validation → Git commit → User GitHub push → Server Git pull → Dataset/model download → GPU smoke tests → M1–M3 experiments → Rejection analysis.**

Do not stop at a plan or partial scaffold.

Begin by inspecting the current project directory, then implement all milestones systematically.