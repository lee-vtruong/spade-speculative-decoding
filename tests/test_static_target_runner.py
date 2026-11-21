"""CPU regression tests for StaticTargetRunner, StaticCache hook slicing, and suppress_eos."""
from __future__ import annotations

import pytest
import torch

from spade.adaptive.target_attention_selector import (
    TargetPrefillAttentionHook,
    resolve_layer_indices,
)
from spade.decoding.cache_utils import crop_past_key_values
from spade.decoding.hf_assisted import QwenPair
from spade.decoding.runtime_opt import StaticTargetRunner


class _DeterministicTinyTok:
    eos_token_id = 2
    pad_token_id = 0
    bos_token_id = 1
    unk_token_id = 3

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return messages[-1]["content"]

    def get_vocab(self):
        return {f"<t{i}>": i for i in range(64)}

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = [5 + (i % 40) for i in range(max(4, len(text.split())))]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def _make_tiny_qwen(seed: int = 7):
    transformers = pytest.importorskip("transformers")
    cfg_cls = getattr(transformers, "Qwen3Config", getattr(transformers, "Qwen2Config", None))
    mdl_cls = getattr(transformers, "Qwen3ForCausalLM", getattr(transformers, "Qwen2ForCausalLM", None))
    if cfg_cls is None or mdl_cls is None:
        pytest.skip("Qwen2/Qwen3 classes unavailable")

    torch.manual_seed(seed)
    cfg = cfg_cls(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    return mdl_cls(cfg).eval()


def test_static_target_runner_step_verify_rollback_matches_dynamic_cache():
    from transformers import DynamicCache

    model = _make_tiny_qwen(seed=13)
    prompt_ids = [1, 5, 12, 7, 3, 9, 2, 14, 21, 18]

    runner = StaticTargetRunner(model, max_cache_len=64, max_verify_len=5, mode="eager_static", device=torch.device("cpu"))
    runner.prepare(warmup_steps=2)

    dyn = DynamicCache()
    with torch.inference_mode():
        d_pre = model(input_ids=torch.tensor([prompt_ids], dtype=torch.long), past_key_values=dyn, use_cache=True)
        ref = d_pre.logits[0, -1].float()
    got = runner.prefill(prompt_ids)
    assert torch.max(torch.abs(got - ref)).item() < 1e-4

    # Interleave 1-token steps, variable-length verify(m) for every m in 1..5, and rollbacks
    ops = [
        ("step", 4),
        ("verify", [8, 15, 16, 23, 42]),  # m = 5
        ("rollback", 2),
        ("verify", [9, 10, 11]),           # m = 3 (< max_verify_len)
        ("step", 7),
        ("rollback", 1),
        ("verify", [12, 13, 14, 15]),      # m = 4
        ("verify", [19, 20]),              # m = 2
        ("verify", [25]),                  # m = 1
        ("rollback", 2),
        ("step", 31),
    ]
    for kind, val in ops:
        if kind == "step":
            with torch.inference_mode():
                dout = model(input_ids=torch.tensor([[val]], dtype=torch.long), past_key_values=dyn, use_cache=True)
                ref = dout.logits[0, -1].float()
            got = runner.step(val)
            assert torch.max(torch.abs(got - ref)).item() < 1e-4, f"step mismatch at val={val}"
        elif kind == "verify":
            with torch.inference_mode():
                dout = model(input_ids=torch.tensor([val], dtype=torch.long), past_key_values=dyn, use_cache=True)
                ref = dout.logits[0].float()
            got = runner.verify(val)
            assert got.shape == ref.shape
            assert torch.max(torch.abs(got - ref)).item() < 1e-4, f"verify({len(val)}) mismatch"
        else:
            keep = int(dyn.get_seq_length()) - int(val)
            dyn = crop_past_key_values(dyn, keep)
            runner.rollback(runner.cur_len - int(val))
            assert runner.cur_len == keep


def test_target_prefill_hook_seq_len_slicing_on_static_cache():
    model = _make_tiny_qwen(seed=19)
    prompt_ids = [4, 11, 22, 9, 17, 6, 30, 8, 13, 27, 5, 19, 24, 31, 12]
    layers = resolve_layer_indices(4, "upper50")

    with TargetPrefillAttentionHook(model, layer_indices=layers, tail_window=6) as hk_dyn:
        with torch.inference_mode():
            out_dyn = model(input_ids=torch.tensor([prompt_ids], dtype=torch.long), use_cache=True)
        imp_dyn, _ = hk_dyn.compute_token_importance(out_dyn.past_key_values, head_agg="mean")

    runner = StaticTargetRunner(model, max_cache_len=64, max_verify_len=5, mode="eager_static", device=torch.device("cpu"))
    runner.prepare(warmup_steps=2)
    with TargetPrefillAttentionHook(model, layer_indices=layers, tail_window=6) as hk_stat:
        _ = runner.prefill(prompt_ids)
        imp_stat, _ = hk_stat.compute_token_importance(runner.cache, head_agg="mean", seq_len=len(prompt_ids))

    assert imp_stat.shape == (len(prompt_ids),)
    assert torch.max(torch.abs(imp_stat - imp_dyn)).item() < 1e-5


def _make_pair() -> QwenPair:
    target = _make_tiny_qwen(seed=101)
    draft = _make_tiny_qwen(seed=202)
    pair = object.__new__(QwenPair)
    pair.target = target
    pair.draft = draft
    pair.target_tokenizer = _DeterministicTinyTok()
    pair.draft_tokenizer = _DeterministicTinyTok()
    pair.target_runner = None
    pair.target_setup_timings = None
    pair.draft_runner = None
    pair.draft_setup_timings = None
    pair.common_vocab_size = None
    return pair


def test_qwen_pair_static_target_runner_equivalence_and_suppress_eos():
    prompt = " ".join(f"w{i}" for i in range(18))

    eager_pair = _make_pair()
    runner_pair = _make_pair()
    runner_pair.prepare_target_runner(mode="eager_static", max_cache_len=128, max_verify_len=5, warmup_steps=2)
    runner_pair.prepare_draft_runner(mode="eager_static", max_cache_len=128, warmup_steps=2)

    # Greedy AR (B0) & Greedy Speculative (B1) equivalence
    e_b0 = eager_pair.generate(prompt, mode="B0", max_new_tokens=14, cache_strategy="kv_cache", distribution_mode="fp32_fast", suppress_eos=True)
    r_b0 = runner_pair.generate(prompt, mode="B0", max_new_tokens=14, cache_strategy="kv_cache", distribution_mode="fp32_fast", suppress_eos=True)
    assert r_b0.token_ids == e_b0.token_ids
    assert len(r_b0.token_ids) == 14

    e_b1 = eager_pair.generate(prompt, mode="B1", draft_length=4, max_new_tokens=14, cache_strategy="kv_cache", distribution_mode="fp32_fast", suppress_eos=True)
    r_b1 = runner_pair.generate(prompt, mode="B1", draft_length=4, max_new_tokens=14, cache_strategy="kv_cache", distribution_mode="fp32_fast", suppress_eos=True)
    assert r_b1.token_ids == e_b1.token_ids == r_b0.token_ids

    # Stochastic AR (B0-S) & Stochastic Speculative (B2) equivalence with fixed RNG seed
    e_b0s = eager_pair.generate(prompt, mode="B0-S", max_new_tokens=14, temperature=0.7, top_p=0.9, seed=42, cache_strategy="kv_cache", distribution_mode="fp32_fast", suppress_eos=True)
    r_b0s = runner_pair.generate(prompt, mode="B0-S", max_new_tokens=14, temperature=0.7, top_p=0.9, seed=42, cache_strategy="kv_cache", distribution_mode="fp32_fast", suppress_eos=True)
    assert r_b0s.token_ids == e_b0s.token_ids
    assert 2 not in r_b0s.token_ids

    e_b2 = eager_pair.generate(prompt, mode="B2", draft_length=4, max_new_tokens=14, temperature=0.7, top_p=0.9, seed=42, cache_strategy="kv_cache", distribution_mode="fp32_fast", suppress_eos=True)
    r_b2 = runner_pair.generate(prompt, mode="B2", draft_length=4, max_new_tokens=14, temperature=0.7, top_p=0.9, seed=42, cache_strategy="kv_cache", distribution_mode="fp32_fast", suppress_eos=True)
    assert r_b2.token_ids == e_b2.token_ids
    assert len(r_b2.token_ids) == 14
    assert 2 not in r_b2.token_ids
    assert r_b2.timings["verify_forward_s"] > 0.0
    assert r_b2.timings["target_kv_static_capacity_bytes"] > 0.0
    assert r_b2.timings["target_kv_peak_bytes"] > 0.0
