"""CPU-only unit tests for unified-v2 audit, sensitivity study, disjoint splits, and 5-way ablation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from spade.adaptive.biasing import EvidenceBiasingConfig, apply_evidence_bias
from spade.adaptive.evidence_indexer import EvidenceIndexer
from spade.analysis.experiment_audit import (
    audit_report_json,
    extract_claim_level_metrics,
    paired_cluster_bootstrap,
    select_best_by_throughput,
    summarize_and_audit_run,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_eval_splits import build_disjoint_heldout_split
from run_sensitivity_study import DEFAULT_BETAS, build_sensitivity_experiments
from analyze_sensitivity import analyze_sensitivity_runs, render_markdown as render_sens_md
from run_ablation_suite import build_ablation_experiments, verify_disjoint_splits
from analyze_ablation import analyze_ablation_runs, render_markdown as render_abl_md


class _MockTokenizer:
    def __init__(self):
        self.vocab = {"The": 10, "state": 11, "of": 12, "Georgia": 13, "reported": 14, "cases": 15}
        self.eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.vocab.get(t, 99) for t in text.split()]


def _write_synthetic_run(
    run_dir: Path,
    *,
    run_id: str,
    mode: str = "M4",
    beta0: float = 1.0,
    no_entropy_gating: bool = False,
    dynamic_k: bool = False,
    enable_biasing: bool = True,
    n_claims: int = 8,
    base_decode_s: float = 1.0,
    base_acc_per_round: int = 3,
    ev_accept: bool = True,
) -> None:
    """Create a mathematically self-consistent synthetic raw run directory on CPU."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "environment.json").write_text(
        json.dumps({
            "git_commit": "a6a3613f8d34cb54eb3a03ae8770df576ed939db",
            "git_dirty": False,
            "torch": "2.13.0+cu130",
            "transformers": "5.14.1",
        }),
        encoding="utf-8",
    )
    (run_dir / "config_resolved.json").write_text(
        json.dumps({
            "run_id": run_id,
            "mode": mode,
            "draft_length": 4,
            "dynamic_k": dynamic_k,
            "enable_biasing": enable_biasing,
            "no_entropy_gating": no_entropy_gating,
            "beta0": beta0,
            "trials": 1,
        }),
        encoding="utf-8",
    )

    preds, trials, rounds, tokens = [], [], [], []
    labels = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherry-picking"]
    total_tokens_all = 0
    total_decode_all = 0.0

    for i in range(n_claims):
        cid = f"claim-{i:03d}"
        gold = labels[i % len(labels)]
        # 2 rounds per claim: round 0 has a rejection at pos=base_acc_per_round, round 1 full accept + bonus
        r0_acc = base_acc_per_round  # e.g. 2 or 3 out of 4
        r0_prop = 4
        r0_comm = r0_acc + 1  # accepted + residual replacement
        r1_acc = 4
        r1_prop = 4
        r1_comm = 5  # 4 accepted + 1 bonus
        claim_tokens = r0_comm + r1_comm
        claim_draft_s = base_decode_s * 0.65 * (1.0 + 0.02 * (i % 3))
        claim_verify_s = base_decode_s * 0.25
        claim_decode_s = base_decode_s * (1.0 + 0.02 * (i % 3))
        claim_prefill_s = 0.03

        total_tokens_all += claim_tokens
        total_decode_all += claim_decode_s

        preds.append({
            "claim_id": cid,
            "output_tokens": claim_tokens,
            "parse_success": True,
            "citation_id_validity": True,
            "parsed": {"summary": "test", "citations": ["E1"], "verdict": gold},
            "evaluation": {"label": gold},
        })
        trials.append({
            "run_id": run_id,
            "trial_index": 0,
            "claim_id": cid,
            "prompt_tokens": 120,
            "output_tokens": claim_tokens,
            "prompt_prefill_s": claim_prefill_s,
            "draft_prefill_s": 0.01,
            "draft_generation_s": claim_draft_s,
            "target_verification_s": claim_verify_s,
            "correction_s": 0.01,
            "decode_s": claim_decode_s,
            "parse_success": True,
        })
        rounds.append({
            "claim_id": cid,
            "round_index": 0,
            "proposed_token_count": r0_prop,
            "accepted_draft_tokens": r0_acc,
            "rejection_position": r0_acc,
            "full_acceptance": False,
            "additional_target_token": False,
            "committed_output_tokens": r0_comm,
            "draft_latency_s": claim_draft_s * 0.5,
            "verification_latency_s": claim_verify_s * 0.5,
            "total_round_latency_s": claim_decode_s * 0.5,
        })
        rounds.append({
            "claim_id": cid,
            "round_index": 1,
            "proposed_token_count": r1_prop,
            "accepted_draft_tokens": r1_acc,
            "rejection_position": None,
            "full_acceptance": True,
            "additional_target_token": True,
            "committed_output_tokens": r1_comm,
            "draft_latency_s": claim_draft_s * 0.5,
            "verification_latency_s": claim_verify_s * 0.5,
            "total_round_latency_s": claim_decode_s * 0.5,
        })

        # Token traces for round 0: r0_acc accepted, 1 rejected, 1 residual_replacement (must NOT be double-counted!)
        for pos in range(r0_prop):
            if pos < r0_acc:
                st = "accepted"
                reached = True
            elif pos == r0_acc:
                st = "rejected"
                reached = True
            else:
                st = "discarded_after_rejection"
                reached = False
            tokens.append({
                "claim_id": cid,
                "round_id": 0,
                "status": st,
                "reached_by_verifier": reached,
                "evidence_overlap": {"lexical_overlap": 1.0 if (pos == 0 and ev_accept) or (pos == r0_acc and not ev_accept) else 0.0},
            })
        # Deliberately include residual_replacement with reached_by_verifier=True to verify no double-counting
        tokens.append({
            "claim_id": cid,
            "round_id": 0,
            "status": "residual_replacement",
            "reached_by_verifier": True,
            "evidence_overlap": {"lexical_overlap": 1.0},
        })
        # Token traces for round 1: 4 accepted + 1 extra_target_token
        for pos in range(r1_prop):
            tokens.append({
                "claim_id": cid,
                "round_id": 1,
                "status": "accepted",
                "reached_by_verifier": True,
                "evidence_overlap": {"lexical_overlap": 1.0 if pos == 0 else 0.0},
            })
        tokens.append({
            "claim_id": cid,
            "round_id": 1,
            "status": "extra_target_token",
            "reached_by_verifier": False,
            "evidence_overlap": {"lexical_overlap": 0.0},
        })

    (run_dir / "predictions.jsonl").write_text("\n".join(json.dumps(x) for x in preds) + "\n", encoding="utf-8")
    (run_dir / "trial_metrics.jsonl").write_text("\n".join(json.dumps(x) for x in trials) + "\n", encoding="utf-8")
    (run_dir / "round_traces.jsonl").write_text("\n".join(json.dumps(x) for x in rounds) + "\n", encoding="utf-8")
    (run_dir / "token_traces.jsonl").write_text("\n".join(json.dumps(x) for x in tokens) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({
            "run_id": run_id,
            "examples": n_claims,
            "json_parse_success_rate": 1.0,
            "citation_id_validity_rate": 1.0,
            "mean_output_tokens": total_tokens_all / n_claims,
            "mean_decode_s_per_trial": total_decode_all / n_claims,
        }),
        encoding="utf-8",
    )


def test_audit_local_unified_v2_report_passes():
    report_path = Path(__file__).resolve().parents[1] / "outputs" / "unified_v2_report.json"
    if not report_path.exists():
        pytest.skip("outputs/unified_v2_report.json not present")
    res = audit_report_json(report_path)
    assert res["all_passed"] is True, res
    assert res["methods_checked"] == 5
    assert res["git_commit"] == "a6a3613"


def test_raw_trace_audit_passes_and_avoids_residual_double_counting(tmp_path: Path):
    run_dir = tmp_path / "clean-run"
    _write_synthetic_run(run_dir, run_id="clean-run", base_acc_per_round=3)
    summary = summarize_and_audit_run(run_dir)
    assert summary["audit_passed"] is True, summary["audit_errors"]
    assert summary["accounting_exact"] is True
    # 8 claims * (3+4 accepted = 7 acc, 1 rej) -> N_eval = 8*8 = 64, NOT 72!
    assert summary["n_acc"] == 56
    assert summary["n_res"] == 8
    assert summary["n_bon"] == 8
    assert summary["n_eval"] == 64
    assert summary["acceptance_eval_pct"] == 87.5
    assert summary["verdict_accuracy_pct"] == 100.0


def test_biasing_beta0_grid_monotonic_and_zero_is_identity():
    tok = _MockTokenizer()
    indexer = EvidenceIndexer(tok, ["state of Georgia reported"], max_ngram=3)
    context = [11, 12]  # matches "Georgia" -> token 13

    # 1. beta_0 = 0 is exact identity for both static and entropy-gated
    logits = torch.zeros(100)
    for adaptive in (False, True):
        cfg0 = EvidenceBiasingConfig(boost_weight=0.0, adaptive_entropy=adaptive)
        out0 = apply_evidence_bias(logits, indexer, context, cfg0)
        assert torch.equal(out0, logits)

    # 2. Static monotonic across beta_0 grid
    static_vals = []
    for b in DEFAULT_BETAS:
        out = apply_evidence_bias(logits, indexer, context, EvidenceBiasingConfig(boost_weight=b, adaptive_entropy=False))
        static_vals.append(float(out[13].item()))
    assert static_vals[0] == 0.0
    assert all(static_vals[i] < static_vals[i + 1] for i in range(len(static_vals) - 1))

    # 3. Entropy gating reduces intervention on peaked (low-entropy) logits vs static
    peaked = torch.zeros(100)
    peaked[10] = 20.0
    static_peaked = apply_evidence_bias(peaked, indexer, context, EvidenceBiasingConfig(boost_weight=1.0, adaptive_entropy=False))
    gated_peaked = apply_evidence_bias(peaked, indexer, context, EvidenceBiasingConfig(boost_weight=1.0, adaptive_entropy=True))
    assert float(gated_peaked[13].item()) < float(static_peaked[13].item())


def test_disjoint_split_builder_and_leakage_guard(tmp_path: Path):
    full_path = tmp_path / "full20.jsonl"
    tune_path = tmp_path / "tune5.jsonl"
    heldout_path = tmp_path / "heldout15.jsonl"

    labels = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherry-picking"]
    full_rows = [{"claim_id": f"c-{i}", "prompt": "p", "evaluation": {"label": labels[i % 4]}} for i in range(20)]
    tune_rows = full_rows[:5]

    full_path.write_text("\n".join(json.dumps(r) for r in full_rows) + "\n", encoding="utf-8")
    tune_path.write_text("\n".join(json.dumps(r) for r in tune_rows) + "\n", encoding="utf-8")

    manifest = build_disjoint_heldout_split(full_path, tune_path, heldout_path, heldout_size=12, seed=42)
    assert manifest["strictly_disjoint"] is True
    assert manifest["overlap_count"] == 0
    assert manifest["heldout_claim_count"] == 12
    verify_disjoint_splits(heldout_path, tune_path)

    # Leakage must raise ValueError
    with pytest.raises(ValueError, match="Split leakage"):
        verify_disjoint_splits(full_path, tune_path)


def test_select_best_by_throughput_not_acceptance():
    rows = [
        {
            "run_id": "high-acc-slow",
            "variant": "entropy",
            "beta0": 2.0,
            "adaptive_entropy": True,
            "throughput_e2e_tok_s": 24.5,
            "latency_e2e_ms_per_tok": 40.82,
            "acceptance_eval_pct": 85.0,
            "evidence_overlap_acceptance_pct": 82.0,
            "json_valid_pct": 100.0,
            "citation_valid_pct": 100.0,
            "audit_passed": True,
        },
        {
            "run_id": "fast-e2e-moderate-acc",
            "variant": "static",
            "beta0": 0.5,
            "adaptive_entropy": False,
            "throughput_e2e_tok_s": 27.8,
            "latency_e2e_ms_per_tok": 35.97,
            "acceptance_eval_pct": 82.6,
            "evidence_overlap_acceptance_pct": 80.1,
            "json_valid_pct": 100.0,
            "citation_valid_pct": 100.0,
            "audit_passed": True,
        },
    ]
    decision = select_best_by_throughput(rows)
    assert decision["selected"]["run_id"] == "fast-e2e-moderate-acc"
    assert decision["highest_acceptance_for_comparison"]["run_id"] == "high-acc-slow"
    assert decision["selected_differs_from_highest_acceptance"] is True


def test_sensitivity_and_ablation_plumbing_and_bootstrap(tmp_path: Path):
    # Verify sensitivity command builder builds 12 exact configs (2 variants x 6 betas, fixed K=4)
    sens_exps = build_sensitivity_experiments(
        python_bin="python",
        benchmark_input=Path("outputs/benchmarks/dev-oracle-pilot100.jsonl"),
        target="Qwen/Qwen3-8B",
        draft="Qwen/Qwen3-0.6B",
        output_root=tmp_path,
        run_prefix="sens-test",
        limit=100,
        draft_length=4,
        max_new_tokens=384,
        temperature=0.7,
        top_p=0.9,
        seed=42,
        global_warmup=1,
        betas=DEFAULT_BETAS,
        variants=["static", "entropy"],
        order_decay=0.5,
        entropy_threshold=2.0,
        max_boost_cap=3.0,
    )
    assert len(sens_exps) == 12
    assert sum("--no-entropy-gating" in e["cmd"] for e in sens_exps) == 6
    assert all("--dynamic-k" not in e["cmd"] for e in sens_exps)

    # Create synthetic sensitivity runs (2 betas x 2 variants) and run analyzer
    _write_synthetic_run(tmp_path / "sens-test-static-b0p00", run_id="sens-test-static-b0p00", beta0=0.0, no_entropy_gating=True, base_decode_s=1.00, base_acc_per_round=2)
    _write_synthetic_run(tmp_path / "sens-test-static-b0p50", run_id="sens-test-static-b0p50", beta0=0.5, no_entropy_gating=True, base_decode_s=0.90, base_acc_per_round=3)
    _write_synthetic_run(tmp_path / "sens-test-entropy-b0p00", run_id="sens-test-entropy-b0p00", beta0=0.0, no_entropy_gating=False, base_decode_s=1.00, base_acc_per_round=2)
    _write_synthetic_run(tmp_path / "sens-test-entropy-b0p50", run_id="sens-test-entropy-b0p50", beta0=0.5, no_entropy_gating=False, base_decode_s=1.10, base_acc_per_round=3)

    sens_report = analyze_sensitivity_runs(
        runs_root=tmp_path,
        run_prefix="sens-test",
        betas=[0.0, 0.5],
        variants=["static", "entropy"],
        n_bootstrap=500,
        seed=42,
    )
    assert sens_report["all_audits_passed"] is True
    assert sens_report["overall_selection"]["selected"]["run_id"] == "sens-test-static-b0p50"
    assert "## Selection Decision" in render_sens_md(sens_report)

    # Verify 5-way ablation command builder includes B2, Bias-only, Bias+Entropy, Dynamic-only, Full SPADE
    abl_exps = build_ablation_experiments(
        python_bin="python",
        benchmark_input=Path("outputs/benchmarks/dev-oracle-heldout400.jsonl"),
        target="Qwen/Qwen3-8B",
        draft="Qwen/Qwen3-0.6B",
        output_root=tmp_path,
        run_prefix="abl-test",
        limit=None,
        draft_length=4,
        max_new_tokens=384,
        temperature=0.7,
        top_p=0.9,
        seed=42,
        global_warmup=1,
        bias_only_beta0=0.5,
        bias_entropy_beta0=0.5,
        full_spade_beta0=0.5,
        full_spade_adaptive_entropy=True,
        order_decay=0.5,
        entropy_threshold=2.0,
        max_boost_cap=3.0,
        include_references=False,
    )
    keys = [e["key"] for e in abl_exps]
    assert keys == ["b2", "bias_only", "bias_entropy", "dynamic_only", "full_spade"]
    dyn_only = [e for e in abl_exps if e["key"] == "dynamic_only"][0]
    assert "--dynamic-k" in dyn_only["cmd"] and "--no-biasing" in dyn_only["cmd"]

    # Create synthetic ablation runs and verify paired cluster bootstrap detects negative effect honestly
    _write_synthetic_run(tmp_path / "abl-test-b2-k4", run_id="abl-test-b2-k4", mode="B2", beta0=0.0, base_decode_s=1.00, base_acc_per_round=2)
    _write_synthetic_run(tmp_path / "abl-test-bias-only", run_id="abl-test-bias-only", mode="M4", beta0=0.5, no_entropy_gating=True, base_decode_s=0.92, base_acc_per_round=3)
    _write_synthetic_run(tmp_path / "abl-test-bias-entropy", run_id="abl-test-bias-entropy", mode="M4", beta0=0.5, no_entropy_gating=False, base_decode_s=1.00, base_acc_per_round=2)
    _write_synthetic_run(tmp_path / "abl-test-dynamic-only", run_id="abl-test-dynamic-only", mode="M4", beta0=0.0, dynamic_k=True, enable_biasing=False, base_decode_s=1.20, base_acc_per_round=2)
    _write_synthetic_run(tmp_path / "abl-test-full-spade", run_id="abl-test-full-spade", mode="M4", beta0=0.5, dynamic_k=True, base_decode_s=1.15, base_acc_per_round=3)

    abl_report = analyze_ablation_runs(runs_root=tmp_path, run_prefix="abl-test", n_bootstrap=500, seed=42)
    assert abl_report["all_audits_passed"] is True
    dyn_vs_b2 = [c for c in abl_report["paired_comparisons"] if c["treatment_key"] == "dynamic_only" and c["baseline_key"] == "b2"][0]
    # Dynamic-only was slower (1.20s vs 1.00s) -> must honestly flag negative_ci_excludes_zero!
    assert dyn_vs_b2["metrics"]["throughput_e2e_tok_s"]["interpretation"] == "negative_ci_excludes_zero"
    assert dyn_vs_b2["metrics"]["throughput_e2e_tok_s"]["delta"] < 0.0
    assert "## Paired Cluster Bootstrap Comparisons" in render_abl_md(abl_report)


def test_raw_trace_audit_catches_token_conservation_violation(tmp_path: Path):
    run_dir = tmp_path / "corrupt-run"
    _write_synthetic_run(run_dir, run_id="corrupt-run", base_acc_per_round=3)
    # Corrupt trial_metrics output_tokens by +1 to violate conservation
    trials_path = run_dir / "trial_metrics.jsonl"
    rows = [json.loads(line) for line in trials_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows[0]["output_tokens"] += 1
    trials_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    summary = summarize_and_audit_run(run_dir)
    assert summary["audit_passed"] is False
    assert any("Token conservation failed" in err for err in summary["audit_errors"])


def test_dynamic_only_disables_biasing_in_qwen_pair(monkeypatch):
    """Verify mode='M4' with dynamic_k=True and enable_biasing=False skips apply_evidence_bias."""
    import spade.decoding.hf_assisted as hf_mod

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    class _TinyLM:
        def __init__(self, vocab_size=32):
            self.vocab_size = vocab_size
            self.device = torch.device("cpu")
            self.config = type("Cfg", (), {"max_position_embeddings": 128})()

        def parameters(self):
            yield torch.zeros(1)

        def get_input_embeddings(self):
            return type("Emb", (), {"weight": torch.zeros(1)})()

        def __call__(self, input_ids, **kwargs):
            seq_len = input_ids.shape[1]
            logits = torch.zeros((1, seq_len, self.vocab_size), dtype=torch.float32)
            logits[:, :, 5] = 10.0  # deterministic peaked token 5
            return _Out(logits)

    class _TinyTok:
        eos_token_id = 2
        pad_token_id = 0
        chat_template = None

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
            return messages[-1]["content"]

        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            return {"input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long)}

        def encode(self, text, add_special_tokens=False):
            return [10, 11, 12, 13]

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(i) for i in ids)

    pair = object.__new__(hf_mod.QwenPair)
    pair.target = _TinyLM()
    pair.draft = _TinyLM()
    pair.target_tokenizer = _TinyTok()
    pair.draft_tokenizer = _TinyTok()

    bias_called = {"count": 0}
    orig_bias = hf_mod.apply_evidence_bias

    def _spy_bias(*args, **kwargs):
        bias_called["count"] += 1
        return orig_bias(*args, **kwargs)

    monkeypatch.setattr(hf_mod, "apply_evidence_bias", _spy_bias)

    gen = pair.generate(
        "test prompt",
        mode="M4",
        max_new_tokens=6,
        draft_length=4,
        temperature=0.7,
        top_p=0.9,
        seed=42,
        evidence_texts=["The state of Georgia"],
        cache_strategy="recompute",
        dynamic_k=True,
        enable_biasing=False,
    )
    assert len(gen.token_ids) == 6
    assert bias_called["count"] == 0
    assert any(r.get("dynamic_k_reason") is not None for r in gen.rounds)

