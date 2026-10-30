# SPADE M1–M3 implementation and validation report

This report separates implemented code, local CPU tests, and user-reported server runs. Server validation below was performed by the user on `hvtham-server`; no results have been fabricated. No M4 evidence-adaptive decoding algorithm is implemented.

Local validation: 16 CPU tests pass; Python source compiles. No local GPU inference or real-data download was performed.

## Research objective

Establish reproducible baselines and token-level diagnostics for evidence-grounded fact-checking summaries. Exact speculative sampling should preserve the selected target sampling distribution; this is a mathematical property to validate, not a factuality improvement claim.

## M1 — Benchmark construction

### Implemented

- JSON and JSONL record loading, including the observed AVeriTeC records that omit explicit IDs. Stable IDs are derived from non-gold claim metadata and claim text.
- Oracle QA adaptation supports nested AVeriTeC `questions[].answers[]` records.
- Retrieved-evidence adapter accepts externally prepared claim-keyed passages. It does not implement BM25 or parse the downloaded knowledge-store format.
- Versioned deterministic prompt, prompt hash, evidence IDs, deterministic stratified sampling, strict verdict parsing, and classification/citation metrics.
- Gold label and justification are kept in evaluation metadata, not the model prompt.

### User-reported server validation

- AVeriTeC dev JSON loaded as 500 examples; train JSON loaded as 3,068 examples.
- Three-example `oracle_qa` preparation completed. Evidence counts were 7, 2, and 1; the sampled gold justifications were absent from those prompts.
- Qwen3-8B and Qwen3-0.6B tokenizer vocabularies both had 151,669 entries and passed the strict token-ID/special-token compatibility check.

### Pending

- No claim-specific retrieved-evidence run has been prepared or evaluated.
- The existing train knowledge store has `gold`, `question`, and other query types. Sampled `question` queries matched AVeriTeC annotated questions (33/33, 48/48, 37/37); provenance of other query types is unresolved. It is therefore not approved as retrieved evidence.
- No dev knowledge store was downloaded. No full benchmark manifest, evidence coverage analysis, or full M1 metrics report has been produced.

## M2 — Decoding baselines

### Implemented

- B0 target greedy, B0-S target sampling, B1 greedy speculative, and B2 exact speculative-sampling code paths.
- Numerically stable processed distributions, acceptance/rejection, residual correction, extra target token, EOS and output-length handling.
- The current Hugging Face diagnostic adapter recomputes full prefixes and does not use KV cache. B1/B2 target verification is sequential by prefix to match B0 call shape. This is a correctness/debug path, not an optimized performance baseline.
- CPU finite-vocabulary empirical sampling tests and analytical normalization/TV checks.

### User-reported server validation

- B0 and B1 token IDs matched exactly on all 3 smoke claims for 64 generated tokens per claim after sequential verification was added.
- B2 generated 64 tokens with 64 committed tokens. Trace statuses included 48 accepted proposals, 13 rejected proposals, 13 residual replacements, 61 discarded proposals, and 3 extra target tokens across 13 rejection rounds.
- Models loaded in BF16 on an RTX 5090; PyTorch reported `2.14.1+cu130` with CUDA available.

### Pending

- B0-S and B2 distribution behavior have not been statistically validated on GPU. One B2 run exercising rejection is not proof of distribution equality.
- EOS, maximum-length, partial-block, and numerical/backend behavior have only limited synthetic/integration coverage; no KV-cache rollback implementation exists.
- No cache-based optimized implementation, warmup/repeated-trial benchmark, authoritative latency/throughput, or speedup result. Current timing values must not be used for performance claims.

## M3 — Rejection instrumentation

### Implemented and exercised

- Round and token JSONL trace schemas/writer, token categories, lexical evidence-overlap heuristic, and basic acceptance/rejection aggregation.
- User-reported B2 run verified committed-token accounting and exercised rejection/residual trace statuses on the real model pair.

### Pending

- No comprehensive M3 report or analysis of real runs yet.
- Requested plots and CSV exports are not implemented; the notebook is only a trace-loading starting point.
- Aggregation does not yet provide all requested token-, round-, and claim-weighted probability diagnostics or distributions.
- Evidence overlap is lexical only and does not establish semantic grounding or factual correctness.

## Current readiness

| Area | Status |
|---|---|
| M1 oracle smoke preparation | Implemented and server-smoke-tested |
| M1 retrieved evidence | Adapter interface exists; source provenance and real retrieval remain pending |
| M2 B0/B1 greedy correctness | Passed on 3 claims × 64 tokens on server |
| M2 B2 rejection path | Exercised on server; distribution preservation not GPU-validated |
| M2 performance comparison | Pending; no-cache timings are non-authoritative |
| M3 trace production | Implemented and exercised for B2 smoke |
| M3 plots/full analysis | Pending |

## Limitations and next work

1. Audit the knowledge-store generator and query provenance before selecting any record types for retrieved mode; preserve the shared source data as read-only.
2. Add an independent claim-only retrieval path or compatible retrieved-evidence file for dev.
3. Expand synthetic correctness cases for EOS, middle/last rejection, partial blocks, and stopping conditions.
4. Implement/validate cache rollback or retain the current adapter only for correctness diagnostics.
5. Complete M3 aggregation, plots, CSV exports, and analysis notebook using real traces only.

No factuality or semantic-grounding metric is currently evaluated (`not_evaluated`).
