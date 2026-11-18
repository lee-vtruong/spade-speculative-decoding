"""CPU unit tests for Step B1.2 optimized draft runtime + fp32_fast integration in QwenPair."""
from __future__ import annotations

import pytest
import torch

from spade.decoding.hf_assisted import QwenPair


class _TinyTok:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return messages[-1]["content"]

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = [5 + (i % 40) for i in range(max(4, len(text.split())))]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def test_qwen_pair_static_draft_runner_and_fp32_fast_asymmetric_b2():
    transformers = pytest.importorskip("transformers")
    cfg_cls = getattr(transformers, "Qwen3Config", getattr(transformers, "Qwen2Config", None))
    mdl_cls = getattr(transformers, "Qwen3ForCausalLM", getattr(transformers, "Qwen2ForCausalLM", None))
    if cfg_cls is None or mdl_cls is None:
        pytest.skip("Qwen2/Qwen3 classes unavailable")

    torch.manual_seed(11)
    cfg = cfg_cls(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    target = mdl_cls(cfg).eval()
    torch.manual_seed(29)
    draft = mdl_cls(cfg).eval()

    pair = object.__new__(QwenPair)
    pair.target = target
    pair.draft = draft
    pair.target_tokenizer = _TinyTok()
    pair.draft_tokenizer = _TinyTok()
    pair.draft_runner = None
    pair.draft_setup_timings = None

    setup = pair.prepare_draft_runner(mode="eager_static", max_cache_len=128, warmup_steps=2)
    assert setup.max_cache_len == 128
    assert pair.draft_runner is not None

    long_prompt = " ".join(f"tok{i}" for i in range(24))
    short_draft_prompt = " ".join(f"tok{i}" for i in range(10))

    gen = pair.generate(
        long_prompt,
        draft_prompt=short_draft_prompt,
        mode="B2",
        max_new_tokens=12,
        draft_length=4,
        temperature=0.7,
        top_p=0.9,
        seed=42,
        cache_strategy="kv_cache",
        distribution_mode="fp32_fast",
    )
    assert len(gen.token_ids) == 12
    assert gen.prompt_tokens == 24
    assert gen.timings["draft_prompt_tokens"] == 10.0
    assert gen.timings["draft_kv_prefill_bytes"] > 0
    assert gen.timings["draft_kv_peak_bytes"] >= gen.timings["draft_kv_prefill_bytes"]
    assert gen.timings["draft_kv_static_capacity_bytes"] > gen.timings["draft_kv_peak_bytes"]

    # Exact token accounting check across accepted / residual_replacement / extra_target_token
    n_acc = sum(1 for t in gen.token_traces if t["status"] == "accepted")
    n_res = sum(1 for t in gen.token_traces if t["status"] == "residual_replacement")
    n_bon = sum(1 for t in gen.token_traces if t["status"] == "extra_target_token")
    assert n_acc + n_res + n_bon == len(gen.token_ids)

    # Also verify B0-S with fp32_fast works cleanly when draft_runner is attached
    gen_b0s = pair.generate(
        long_prompt,
        mode="B0-S",
        max_new_tokens=8,
        temperature=0.7,
        top_p=0.9,
        seed=42,
        cache_strategy="kv_cache",
        distribution_mode="fp32_fast",
    )
    assert len(gen_b0s.token_ids) == 8
    assert len(gen_b0s.rounds) == 8
