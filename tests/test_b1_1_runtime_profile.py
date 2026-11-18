"""CPU unit tests for Step B1.1 runtime primitives and M4.2 100% additive E2E accounting."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_m4_2 import analyze_m4_2
from spade.decoding.runtime_opt import (
    StaticDraftRunner,
    next_power_of_two_bucket,
    processed_distribution_fast,
)
from spade.decoding.sampling import processed_distribution, total_variation
from test_sensitivity_and_ablation import _write_synthetic_run


def test_processed_distribution_fast_matches_original_exactness():
    torch.manual_seed(123)
    vocab = 4096
    test_vectors = [
        torch.randn(vocab),
        torch.randn(vocab) * 5.0,
        torch.zeros(vocab),
        torch.linspace(-10.0, 10.0, vocab),
    ]
    for temp, tk, tp in [(1.0, 0, 1.0), (0.7, 0, 0.9), (0.7, 50, 0.9), (0.3, 20, 0.95)]:
        for lg in test_vectors:
            p_orig = processed_distribution(lg, temperature=temp, top_k=tk, top_p=tp)
            p_fast = processed_distribution_fast(lg, temperature=temp, top_k=tk, top_p=tp, dtype=torch.float32)
            assert p_fast.shape == p_orig.shape
            tv = total_variation(p_orig, p_fast)
            assert tv < 1e-5, f"TV mismatch {tv} for temp={temp}, top_k={tk}, top_p={tp}"

    # Verify batched [4, V] matches sequential 1D
    stacked = torch.stack(test_vectors, dim=0)
    batched = processed_distribution_fast(stacked, temperature=0.7, top_k=0, top_p=0.9, dtype=torch.float32)
    assert batched.shape == (4, vocab)
    for i, lg in enumerate(test_vectors):
        p_orig = processed_distribution(lg, temperature=0.7, top_k=0, top_p=0.9)
        assert total_variation(p_orig, batched[i]) < 1e-5


def test_next_power_of_two_bucket():
    assert next_power_of_two_bucket(320, min_bucket=512, headroom=256) == 1024
    assert next_power_of_two_bucket(2560, min_bucket=512, headroom=256) == 4096
    assert next_power_of_two_bucket(6150, min_bucket=512, headroom=256) == 8192
    assert next_power_of_two_bucket(12288, min_bucket=512, headroom=256) == 16384


def test_static_draft_runner_matches_dynamic_cache_across_rollbacks():
    transformers = pytest.importorskip("transformers")
    cfg_cls = getattr(transformers, "Qwen3Config", getattr(transformers, "Qwen2Config", None))
    mdl_cls = getattr(transformers, "Qwen3ForCausalLM", getattr(transformers, "Qwen2ForCausalLM", None))
    if cfg_cls is None or mdl_cls is None:
        pytest.skip("Qwen2/Qwen3 classes unavailable")

    torch.manual_seed(7)
    cfg = cfg_cls(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = mdl_cls(cfg).eval()
    prompt_ids = [1, 5, 12, 7, 3, 9, 2, 14]
    step_tokens = [4, 8, 15, 16, 23, 42, 6, 11, 19, 27]
    rollback_plan = {4: 2, 7: 3}

    # Reference DynamicCache run
    from profile_b1_1_draft_runtime import _run_eager_dynamic_sequence

    ref_logits = _run_eager_dynamic_sequence(model, prompt_ids, step_tokens, rollback_plan=rollback_plan)

    # StaticDraftRunner run (eager_static on CPU)
    runner = StaticDraftRunner(model, max_cache_len=64, mode="eager_static", device=torch.device("cpu"))
    runner.prepare(warmup_steps=2)
    got_logits = [runner.prefill(prompt_ids).clone()]
    for idx, tok in enumerate(step_tokens):
        if idx in rollback_plan:
            runner.rollback(runner.cur_len - rollback_plan[idx])
        got_logits.append(runner.step(tok).clone())

    assert len(got_logits) == len(ref_logits)
    for i, (r, g) in enumerate(zip(ref_logits, got_logits)):
        max_diff = float(torch.max(torch.abs(r - g)).item())
        assert max_diff < 1e-4, f"Step {i} logit mismatch after rollback: max_diff={max_diff}"


def test_analyze_m4_2_additive_e2e_accounting_zero_residual(tmp_path: Path):
    runs_root = tmp_path / "runs"
    base_dir = runs_root / "m4-2-acc-baseline-full"
    sel_dir = runs_root / "m4-2-acc-selective-topk2"
    _write_synthetic_run(base_dir, run_id="m4-2-acc-baseline-full", mode="B2", base_decode_s=1.00, base_acc_per_round=3)
    _write_synthetic_run(sel_dir, run_id="m4-2-acc-selective-topk2", mode="B2", base_decode_s=1.10, base_acc_per_round=2)
    # Inject realistic unattributed decode overhead (~16.6 ms/rnd) + correction_s into trial_metrics
    for rdir, sel_s, d_pref_s, extra_unattr in [
        (base_dir, 0.0, 0.097, 0.52),
        (sel_dir, 0.0015, 0.035, 0.58),
    ]:
        tpath = rdir / "trial_metrics.jsonl"
        rows = [json.loads(l) for l in tpath.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            r["context_selection_s"] = sel_s
            r["draft_prefill_s"] = d_pref_s
            r["correction_s"] = 0.019
            r["decode_s"] = r["draft_generation_s"] + r["target_verification_s"] + r["correction_s"] + extra_unattr
        tpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    rep = analyze_m4_2(runs_root=runs_root, run_prefix="m4-2-acc", draft_top_k=2, n_bootstrap=200, seed=42)
    assert len(rep["break_even_evaluation"]) == 1
    be = rep["break_even_evaluation"][0]
    assert abs(be["accounting_residual_ms_per_claim"]) <= 0.02
    assert abs(be["unattributed_decode_delta_ms_per_claim"] - 60.0) < 0.1
