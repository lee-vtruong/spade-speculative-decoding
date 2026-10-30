# SPADE M1–M3 implementation report

## Research objective
Test whether evidence relationships can later inform speculative inference while preserving the target distribution. This implementation does not implement the M4 algorithm.

## Dataset and evidence configuration
AVeriTeC JSON/JSONL loading and normalized oracle QA/retrieved passages are implemented. Oracle mode is diagnostic. Retrieved mode accepts external claim keyed results and never ranks with labels or justifications. Real dataset integration is pending server data.

## Prompt design
Version `spade_factcheck_v1`; deterministic prompt hash, four explicit verdict values, and structural separation of prompt text and evaluation-only metadata. Qwen chat-template helper fails closed if `enable_thinking=False` is unsupported.

## Model configurations
Defaults are Qwen3-8B and Qwen3-0.6B, BF16, local-only loading. Tokenizer checks compare token IDs and special tokens. Model loading helper exists; inference adapter remains pending.

## Greedy correctness
Greedy token verifier primitive implemented. Synthetic tests are CPU checks only. Qwen equality is pending.

## Sampling correctness
Processed distribution, acceptance, residual, TV, and expected acceptance primitives implemented. Small deterministic CPU identity tests exist; large empirical validation remains pending. Exact round primitive accepts precomputed distributions; model/cache orchestration remains pending.

## Latency, throughput, acceptance, rejection, evidence overlap, memory
No empirical measurements exist. No plots are fabricated. Trace fields and lexical overlap helpers exist; metrics are pending real traces. GPU memory/timing, factuality and semantic grounding are not evaluated.

## Limitations
The first Hugging Face diagnostic adapter now supports B0/B0-S/B1/B2 with full-prefix recomputation and no KV cache. It exports round and token JSONL traces for speculative modes, but timings are explicitly non-authoritative and unsuitable for official performance claims. Cache optimization/rollback, run recovery, complete plotting, and GPU validation remain pending.

## M4 research questions
Which claim/evidence features predict proposal mismatch? Can evidence conditioning change draft proposals while retaining exact correction? What trade-off appears across context length, draft length, and latency?
