"""CPU-only unit tests for Milestone M4.2: Evidence-Selective Context Speculative Decoding."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from spade.adaptive.context_selection import build_draft_context
from spade.benchmark.prompt_builder import build_prompt
from spade.decoding.cache_utils import CacheManager, get_cache_memory_bytes
import spade.decoding.hf_assisted as hf_mod

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_m4_2_pilot import build_live_m4_2_experiments, replay_asymmetric_claim_prefixes
from analyze_m4_2 import analyze_m4_2, render_markdown


def _sample_row() -> dict:
    claim = "Georgia reported 120 cases in October 2024."
    evidence = [
        {"evidence_id": "E1", "text": "Unrelated weather forecast for Paris mentions rain and mild winds all week. Another filler sentence about umbrella sales in Europe."},
        {"evidence_id": "E2", "text": "Background note on tourism. Official Georgia health department reported 120 confirmed cases in October 2024. Additional administrative footnote."},
        {"evidence_id": "E3", "text": "Sports update about football championships in Spain and Italy during summer tournaments."},
        {"evidence_id": "E4", "text": "State officials in Georgia clarified the October 2024 case tally reached 120 after audit revisions."},
    ]
    prompt, phash = build_prompt(claim, evidence, mode="retrieved")
    return {
        "claim_id": "avc-test-001",
        "claim": claim,
        "evidence_mode": "retrieved",
        "evidence": evidence,
        "prompt": prompt,
        "prompt_hash": phash,
        "evaluation": {"label": "Supported", "justification": "SECRET_GOLD_JUSTIFICATION_MUST_NOT_LEAK"},
    }


def test_context_selection_policies_zero_leakage_and_ranking():
    row = _sample_row()

    res_full = build_draft_context(row, policy="full")
    assert res_full.draft_prompt == row["prompt"]
    assert res_full.selected_evidence_ids == ["E1", "E2", "E3", "E4"]
    assert res_full.context_selection_s == 0.0

    res_topk = build_draft_context(row, policy="selective_topk", top_k=2)
    assert res_topk.selected_evidence_ids == ["E2", "E4"]
    assert res_topk.draft_evidence_chars < res_topk.full_evidence_chars
    assert "SECRET_GOLD_JUSTIFICATION" not in res_topk.draft_prompt
    assert "E2" in res_topk.draft_prompt and "E4" in res_topk.draft_prompt

    res_compact = build_draft_context(row, policy="compact", compact_sentences=1, compact_chars=120)
    assert res_compact.selected_evidence_ids == ["E1", "E2", "E3", "E4"]
    assert res_compact.draft_evidence_chars < res_compact.full_evidence_chars
    assert "120 confirmed cases" in res_compact.draft_prompt
    assert "SECRET_GOLD_JUSTIFICATION" not in res_compact.draft_prompt


def test_get_cache_memory_bytes_handles_all_cache_formats():
    k = torch.zeros((1, 2, 10, 8), dtype=torch.bfloat16)  # 160 * 2 = 320 bytes
    v = torch.zeros((1, 2, 10, 8), dtype=torch.bfloat16)  # 320 bytes

    # Legacy tuple-of-tuples
    assert get_cache_memory_bytes(((k, v), (k, v))) == 1280

    # DynamicCache key_cache / value_cache
    dc = type("DC", (), {"key_cache": [k, k], "value_cache": [v, v]})()
    assert get_cache_memory_bytes(dc) == 1280

    # Transformers 5.x DynamicCache.layers
    layer = type("Layer", (), {"keys": k, "values": v})()
    dc5 = type("DC5", (), {"layers": [layer, layer]})()
    assert get_cache_memory_bytes(dc5) == 1280
    assert CacheManager(dc5).memory_bytes == 1280


def test_asymmetric_kv_cache_generation_and_rollback():
    """Verify QwenPair.generate handles shorter draft_prompt cleanly with KV-cache rollback."""

    class _Out:
        def __init__(self, logits, pkv):
            self.logits = logits
            self.past_key_values = pkv

    class _MockKVLM:
        def __init__(self, prefer_token: int, vocab_size: int = 32):
            self.prefer_token = prefer_token
            self.vocab_size = vocab_size
            self.config = type("Cfg", (), {"max_position_embeddings": 128})()

        def get_input_embeddings(self):
            return type("Emb", (), {"weight": torch.zeros(1)})()

        def __call__(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
            seq_len = input_ids.shape[1]
            prev_len = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
            new_len = prev_len + seq_len
            k = torch.zeros((1, 2, new_len, 4), dtype=torch.float32)
            v = torch.zeros((1, 2, new_len, 4), dtype=torch.float32)
            logits = torch.full((1, seq_len, self.vocab_size), -2.0, dtype=torch.float32)
            logits[:, :, self.prefer_token] = 5.0
            return _Out(logits, ((k, v),))

    class _MockTok:
        eos_token_id = 2
        pad_token_id = 0

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
            return messages[-1]["content"]

        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            ids = [10 + (i % 15) for i in range(max(2, len(text.split())))]
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(i) for i in ids)

    pair = object.__new__(hf_mod.QwenPair)
    pair.target = _MockKVLM(prefer_token=7)
    pair.draft = _MockKVLM(prefer_token=9)  # disagrees -> triggers rollback every round!
    pair.target_tokenizer = _MockTok()
    pair.draft_tokenizer = _MockTok()

    long_prompt = " ".join(f"w{i}" for i in range(20))
    short_draft_prompt = " ".join(f"w{i}" for i in range(6))

    gen = pair.generate(
        long_prompt,
        draft_prompt=short_draft_prompt,
        mode="B2",
        max_new_tokens=6,
        draft_length=4,
        temperature=0.7,
        top_p=0.9,
        seed=42,
        cache_strategy="kv_cache",
    )
    assert len(gen.token_ids) == 6
    assert gen.prompt_tokens == 20
    assert gen.timings["draft_prompt_tokens"] == 6.0
    assert gen.timings["draft_kv_prefill_bytes"] < gen.timings["target_kv_prefill_bytes"]
    assert gen.timings["draft_kv_peak_bytes"] < gen.timings["target_kv_peak_bytes"]
    # Exact verification must commit target's strongly preferred token 7 despite draft preferring 9
    assert all(t == 7 for t in gen.token_ids)


def test_m4_2_replay_and_analysis_pipeline(tmp_path: Path):
    row = _sample_row()

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    class _SimpleLM:
        def __init__(self, bias_tok: int, vocab_size: int = 32):
            self.bias_tok = bias_tok
            self.vocab_size = vocab_size

        def get_input_embeddings(self):
            return type("Emb", (), {"weight": torch.zeros(1)})()

        def __call__(self, input_ids, **kwargs):
            seq_len = input_ids.shape[1]
            logits = torch.zeros((1, seq_len, self.vocab_size), dtype=torch.float32)
            logits[:, :, self.bias_tok] = 3.0
            return _Out(logits)

    class _SimpleTok:
        eos_token_id = 2

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
            return messages[-1]["content"]

        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            ids = [5 + (i % 10) for i in range(max(3, len(text.split())))]
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    recs = replay_asymmetric_claim_prefixes(
        row=row,
        token_ids=[6, 7, 8],
        target_model=_SimpleLM(6),
        draft_model=_SimpleLM(6),
        tokenizer=_SimpleTok(),
        draft_top_k=2,
        compact_sentences=1,
        compact_chars=120,
    )
    assert len(recs) == 3
    assert recs[0]["policies"]["selective_topk"]["draft_prompt_tokens"] < recs[0]["target_prompt_tokens"]
    assert recs[0]["policies"]["compact"]["draft_prompt_tokens"] < recs[0]["target_prompt_tokens"]
    assert recs[0]["policies"]["full"]["expected_acceptance"] == pytest.approx(1.0, abs=1e-6)

    replay_path = tmp_path / "m4_2_prefix_divergence.jsonl"
    replay_path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")

    # Verify live orchestrator command builder generates exact 3 configs
    exps = build_live_m4_2_experiments(
        python_bin="python",
        benchmark_input=Path("outputs/benchmarks/dev-oracle-pilot100.jsonl"),
        target="Qwen/Qwen3-8B",
        draft="Qwen/Qwen3-0.6B",
        output_root=tmp_path,
        run_prefix="m4-2-test",
        limit=100,
        draft_length=4,
        max_new_tokens=384,
        temperature=0.7,
        top_p=0.9,
        seed=42,
        global_warmup=1,
        draft_top_k=2,
        compact_sentences=1,
        compact_chars=160,
    )
    assert [e["policy"] for e in exps] == ["full", "selective_topk", "compact"]

    # Reuse synthetic run helper from test_sensitivity_and_ablation to test analyze_m4_2 end-to-end
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_sensitivity_and_ablation import _write_synthetic_run

    _write_synthetic_run(tmp_path / "m4-2-test-baseline-full", run_id="m4-2-test-baseline-full", mode="B2", base_decode_s=1.00, base_acc_per_round=3)
    _write_synthetic_run(tmp_path / "m4-2-test-selective-topk2", run_id="m4-2-test-selective-topk2", mode="B2", base_decode_s=0.90, base_acc_per_round=3)
    _write_synthetic_run(tmp_path / "m4-2-test-compact", run_id="m4-2-test-compact", mode="B2", base_decode_s=1.10, base_acc_per_round=2)

    report = analyze_m4_2(
        runs_root=tmp_path,
        run_prefix="m4-2-test",
        draft_top_k=2,
        replay_path=replay_path,
        n_bootstrap=300,
        seed=42,
    )
    assert report["all_audits_passed"] is True
    assert len(report["live_rows"]) == 3
    assert len(report["break_even_evaluation"]) == 2
    be_by_pol = {b["policy"]: b for b in report["break_even_evaluation"]}
    assert be_by_pol["selective_topk"]["break_even_verdict"] == "PASS_BREAK_EVEN"
    assert be_by_pol["compact"]["break_even_verdict"] == "FAIL_BREAK_EVEN"
    md = render_markdown(report)
    assert "## 3. Break-Even Decision Check" in md
