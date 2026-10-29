# Experiment matrix

| ID | Method | Draft length | Status |
|---|---|---:|---|
| E0 | Target-only greedy | N/A | CLI pending model adapter |
| E1 | Greedy speculative | 4 | CLI pending model adapter |
| E2 | Target-only sampling | N/A | CLI pending model adapter |
| E3–E6 | Exact speculative sampling | 2, 4, 6, 8 | Distribution primitives available; model/cache adapter pending |

Compare latency only between matched policies and report output lengths. No factuality improvement follows from distribution preservation. Record actual hardware, software and revisions before interpreting timings.
