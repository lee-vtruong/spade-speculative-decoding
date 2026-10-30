# M1–M3 experiment matrix

| Milestone | Experiment | Input/evidence | Main artifacts | Status |
|---|---|---|---|---|
| M1 | Oracle QA diagnostic | AVeriTeC questions and answers | JSONL benchmark + manifest | Implemented; user smoke run passed |
| M1 | Retrieved evidence | Exact claim-query passages from read-only knowledge store | JSONL benchmark + retrieval audit/manifest | Implemented; audit coverage before full run |
| M2 | B0 target greedy | Same prepared prompts as B1 | Predictions, timings, traces | Implemented; user-reported 3-claim comparison passed |
| M2 | B1 greedy speculative, k=2/4/6/8 | Target + draft models | Predictions, timings, round/token traces | Implemented; no-cache correctness path |
| M2 | B0-S target sampling | Fixed temperature/top-k/top-p and seed | Predictions, traces | Implemented |
| M2 | B2 exact speculative sampling, k=2/4/6/8 | Same target sampling policy | Predictions, acceptance/rejection traces | Implemented; GPU distribution calibration outstanding |
| M3 | Rejection analysis | Real M2 run directories | JSON summary, CSV tables, PNG plots | Implemented; run after experiments |
| M3 | Output evaluation | Prediction JSONL + held-out gold metadata | Parse/citation/verdict metrics | Implemented; factuality is not evaluated |

The current Hugging Face model adapter recomputes full prefixes without KV cache. Its latency and any derived speedup are diagnostic only; authoritative performance results require a validated cache-based adapter. See [SERVER_RUNBOOK.md](SERVER_RUNBOOK.md) for server commands and safe data paths.
