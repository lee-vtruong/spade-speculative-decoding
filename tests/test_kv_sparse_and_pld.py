"""CPU unit tests for KV-Sparse StaticDraftRunner, Prompt-Lookup Decoding (PLD), selector aggs, and LongBench QA prompt builder."""
from __future__ import annotations

import pytest
import torch

from spade.adaptive.context_selection import build_draft_context
from spade.adaptive.target_attention_selector import (
    find_evidence_and_sentence_token_spans,
    resolve_layer_indices,
    score_evidence_from_token_importance,
)
from spade.benchmark.prompt_builder import build_prompt
from spade.decoding.hf_assisted import QwenPair, _PromptNgramMatcher
from spade.decoding.speculative import exact_pld_verify_round


class _RepeatTok:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return messages[-1]["content"]

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False, add_special_tokens=False):
        import re
        matches = list(re.finditer(r"\S+", text))
        # Deterministic repeating token pattern so _PromptNgramMatcher finds repeated trigrams/bigrams
        ids = [5 + (i % 7) for i in range(max(4, len(matches)))]
        offsets = [(m.start(), m.end()) for m in matches]
        if return_tensors == "pt":
            out = {"input_ids": torch.tensor([ids], dtype=torch.long)}
        else:
            out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def _tiny_qwen_pair():
    transformers = pytest.importorskip("transformers")
    cfg_cls = getattr(transformers, "Qwen3Config", getattr(transformers, "Qwen2Config", None))
    mdl_cls = getattr(transformers, "Qwen3ForCausalLM", getattr(transformers, "Qwen2ForCausalLM", None))
    if cfg_cls is None or mdl_cls is None:
        pytest.skip("Qwen2/Qwen3 classes unavailable")

    torch.manual_seed(101)
    cfg = cfg_cls(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    target = mdl_cls(cfg).eval()
    torch.manual_seed(202)
    draft = mdl_cls(cfg).eval()

    pair = object.__new__(QwenPair)
    pair.target = target
    pair.draft = draft
    pair.target_tokenizer = _RepeatTok()
    pair.draft_tokenizer = _RepeatTok()
    pair.draft_runner = None
    pair.draft_setup_timings = None
    pair.common_vocab_size = None
    return pair


def test_kv_sparse_prefill_pos_offset_and_rollback():
    pair = _tiny_qwen_pair()
    pair.prepare_draft_runner(mode="eager_static", max_cache_len=64, warmup_steps=1)
    runner = pair.draft_runner
    assert runner is not None

    prompt_ids = [5 + (i % 15) for i in range(32)]
    logits, dyn_s, ev_s, kept = runner.prefill_kv_sparse(
        prompt_ids, kv_budget=16, sink_tokens=4, recent_tokens=4, obs_window=4, pool_kernel=3
    )
    assert kept == 16
    assert runner.cur_len == 16
    assert runner.pos_offset == 16  # 32 - 16
    assert dyn_s >= 0.0 and ev_s >= 0.0
    assert logits.shape == (64,)

    # Advance 3 steps -> cur_len=19, position_id reached 32+2=34
    _ = runner.step_many([7, 8, 9])
    assert runner.cur_len == 19
    assert int(runner._static_position_ids.item()) == 34

    # Rollback to 17 -> next step must use cache_position=17 and RoPE position_id=33
    runner.rollback(17)
    assert runner.cur_len == 17
    _ = runner.step(11)
    assert runner.cur_len == 18
    assert int(runner._static_cache_position.item()) == 17
    assert int(runner._static_position_ids.item()) == 33

    # End-to-end B2 generation using kv_sparse_cfg
    long_prompt = " ".join(f"w{i}" for i in range(30))
    gen = pair.generate(
        long_prompt,
        mode="B2",
        max_new_tokens=10,
        draft_length=4,
        temperature=0.7,
        top_p=0.9,
        seed=42,
        cache_strategy="kv_cache",
        distribution_mode="fp32_fast",
        kv_sparse_cfg={"kv_budget": 16, "sink_tokens": 4, "recent_tokens": 4, "obs_window": 4, "pool_kernel": 3},
    )
    assert len(gen.token_ids) == 10
    assert gen.timings["draft_prompt_tokens"] == 16.0
    assert gen.timings["context_selection_s"] >= 0.0
    assert gen.timings["draft_kv_prefill_bytes"] > 0.0


def test_pld_exact_residual_monte_carlo_and_e2e():
    # 1. Analytical/Monte Carlo verification that exact_pld_verify_round preserves p exactly
    p = torch.tensor([0.10, 0.25, 0.35, 0.20, 0.10], dtype=torch.float64)
    p = p / p.sum()
    gen = torch.Generator().manual_seed(77)
    counts = torch.zeros(5, dtype=torch.float64)
    n_draws = 30000
    for i in range(n_draws):
        prop_tok = i % 5
        res = exact_pld_verify_round([prop_tok], [p], generator=gen, remaining=1)
        counts[res.committed[0]] += 1.0
    emp = counts / float(n_draws)
    tv = float(0.5 * torch.abs(emp - p).sum().item())
    assert tv < 0.015, f"PLD exact sampling TV too large: {tv}"

    # 2. Matcher logic check
    matcher = _PromptNgramMatcher([10, 11, 12, 13, 14, 10, 11], max_n=3)
    prop = matcher.propose(4)
    assert prop == [12, 13, 14, 10]

    # 3. E2E QwenPair PLD generation (with and without draft_model)
    pair = _tiny_qwen_pair()
    pair.draft = None
    prompt = " ".join(f"tok{i}" for i in range(28))
    gen_pld = pair.generate(
        prompt,
        mode="PLD",
        max_new_tokens=14,
        draft_length=4,
        temperature=0.7,
        top_p=0.9,
        seed=42,
        cache_strategy="kv_cache",
        distribution_mode="fp32_fast",
    )
    assert len(gen_pld.token_ids) == 14
    assert gen_pld.timings["draft_prompt_tokens"] == 0.0
    assert gen_pld.timings["draft_kv_peak_bytes"] == 0.0


def test_resolve_layer_indices_sentence_aggs_and_longbench_prompt():
    assert resolve_layer_indices(36, "upper50") == list(range(18, 36))
    assert resolve_layer_indices(36, "upper25") == list(range(27, 36))
    assert resolve_layer_indices(36, "middle50") == list(range(9, 27))
    assert resolve_layer_indices(36, "all") == list(range(36))
    assert resolve_layer_indices(36, "last1") == [35]

    ev = [
        {"evidence_id": "P1", "text": "Alpha beta gamma. Delta epsilon."},
        {"evidence_id": "P2", "text": "Zeta eta theta iota kappa lambda."},
    ]
    prompt, phash = build_prompt("What is Alpha?", ev, mode="longbench_qa")
    assert len(phash) == 64
    assert "Final Answer:" in prompt
    assert '"evidence_id":"P1"' in prompt

    row = {"claim_id": "lb-1", "claim": "What is Alpha?", "evidence": ev, "evidence_mode": "longbench_qa", "prompt": prompt}
    dctx = build_draft_context(row, policy="sentence_budget", budget_chars=60)
    assert "evidence_id" in dctx.draft_prompt and "Final Answer:" in dctx.draft_prompt

    offsets = [(i, i + 1) for i in range(len(prompt))]
    spans = find_evidence_and_sentence_token_spans(prompt, ev, offsets)
    imp = torch.ones(len(prompt), dtype=torch.float32) * 0.5
    for agg in ("sum", "mean", "sqrt", "max"):
        _, s_sc = score_evidence_from_token_importance(imp, spans, sentence_agg=agg)
        assert len(s_sc) == 2 and all(v > 0 for v in s_sc[0])


def test_format_qwen_prompt_supports_llama_fallback():
    from spade.decoding.autoregressive import format_qwen_prompt

    class _LlamaTok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert tokenize is False and add_generation_prompt is True
            return f"<|begin_of_text|>{messages[0]['content']}<|eot_id|>"

    out = format_qwen_prompt(_LlamaTok(), "hello world")
    assert out == "<|begin_of_text|>hello world<|eot_id|>"


def test_streaming_llm_prefill_free_policy():
    pair = _tiny_qwen_pair()
    pair.prepare_draft_runner(mode="eager_static", max_cache_len=64, warmup_steps=1)

    long_prompt = " ".join(f"tok{i}" for i in range(25))
    gen_long = pair.generate(
        long_prompt,
        mode="B2",
        max_new_tokens=6,
        draft_length=3,
        temperature=1.0,
        top_p=1.0,
        seed=42,
        cache_strategy="kv_cache",
        distribution_mode="fp32_fast",
        streaming_llm_cfg={"kv_budget": 10, "sink_tokens": 4},
    )
    assert gen_long.timings["draft_prompt_tokens"] == 10.0
    assert gen_long.timings["context_selection_s"] >= 0.0
    assert pair.draft_runner is not None
    assert pair.draft_runner.pos_offset == 0

    short_prompt = " ".join(f"s{i}" for i in range(8))
    gen_short = pair.generate(
        short_prompt,
        mode="B2",
        max_new_tokens=4,
        draft_length=3,
        temperature=1.0,
        top_p=1.0,
        seed=42,
        cache_strategy="kv_cache",
        distribution_mode="fp32_fast",
        streaming_llm_cfg={"kv_budget": 20, "sink_tokens": 4},
    )
    assert gen_short.timings["draft_prompt_tokens"] == 8.0
