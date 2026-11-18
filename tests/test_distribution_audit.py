"""CPU-only unit tests for M4.1 Distribution-Level Evidence Biasing Audit."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

from spade.adaptive.evidence_indexer import EvidenceIndexer
from spade.analysis.distribution_audit import (
    build_default_audit_specs,
    build_distribution_audit_report,
    claim_clustered_bootstrap_mean,
    evaluate_prefix_biasing,
    jensen_shannon_divergence,
)
from spade.decoding.sampling import expected_acceptance, processed_distribution, total_variation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_distribution_audit import replay_claim_prefixes
from analyze_distribution_audit import generate_figures, render_markdown


class _MockTokenizer:
    def __init__(self):
        self.vocab = {
            "The": 10, "state": 11, "of": 12, "Georgia": 13, "reported": 14,
            "120": 15, "cases": 16, "[E1]": 17, "other": 18, "wrong": 19,
        }
        self.inv = {v: k for k, v in self.vocab.items()}
        self.eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return messages[-1]["content"]

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.vocab.get(t, 18) for t in text.split()]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(self.inv.get(i, "other") for i in ids)


def test_js_divergence_properties_and_expected_acc_identity():
    p = torch.tensor([0.7, 0.2, 0.1, 0.0], dtype=torch.float64)
    q = torch.tensor([0.2, 0.5, 0.2, 0.1], dtype=torch.float64)

    # Zero iff p == q
    assert jensen_shannon_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    # Symmetry
    assert jensen_shannon_divergence(p, q) == pytest.approx(jensen_shannon_divergence(q, p), abs=1e-12)
    # Upper bound ln(2) for disjoint distributions
    d1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    d2 = torch.tensor([0.0, 1.0], dtype=torch.float64)
    assert jensen_shannon_divergence(d1, d2) == pytest.approx(math.log(2.0), abs=1e-12)

    # Exact speculative identity: sum min(p, q) == 1 - TV(p, q)
    assert expected_acceptance(p, q) == pytest.approx(1.0 - total_variation(p, q), abs=1e-12)


def test_evaluate_prefix_biasing_helpful_harmful_and_unmatched():
    tok = _MockTokenizer()
    evidence = ["The state of Georgia reported 120 cases"]
    indexer = EvidenceIndexer(tok, evidence, max_ngram=4)
    specs = build_default_audit_specs(betas=[1.0, 1.5], variants=["static", "entropy"])

    # 1. Helpful case: context ["state", "of"] -> candidate is "Georgia" (13).
    # Target strongly prefers Georgia (13), draft under-weights it -> biasing MUST reduce TV!
    t_logits = torch.full((32,), -2.0)
    t_logits[13] = 4.0
    d_logits = torch.full((32,), -1.0)
    d_logits[18] = 1.5
    d_logits[13] = 0.2

    rec_help = evaluate_prefix_biasing(
        claim_id="c1",
        position=0,
        realized_token_id=13,
        realized_token_text="Georgia",
        target_logits=t_logits,
        draft_logits=d_logits,
        evidence_indexer=indexer,
        evidence_texts=evidence,
        context_tokens=[10, 11, 12],
        specs=specs,
        temperature=0.7,
        top_p=0.9,
    )
    assert rec_help["matched_order"] == 3
    assert rec_help["candidate_count"] == 1
    assert rec_help["target_mass_on_candidates"] > 0.9
    assert rec_help["configs"]["static-b1p50"]["is_biased"] is True
    assert rec_help["configs"]["static-b1p50"]["delta_tv"] < 0.0
    assert rec_help["configs"]["static-b1p50"]["delta_expected_acceptance"] > 0.0

    # 2. Harmful case: spurious match where target puts ~0 mass on candidate (13) and prefers 19
    t_harm = torch.full((32,), -2.0)
    t_harm[19] = 5.0
    d_harm = torch.full((32,), -1.0)
    d_harm[19] = 1.5
    d_harm[13] = 0.5

    rec_harm = evaluate_prefix_biasing(
        claim_id="c1",
        position=1,
        realized_token_id=19,
        realized_token_text="wrong",
        target_logits=t_harm,
        draft_logits=d_harm,
        evidence_indexer=indexer,
        evidence_texts=evidence,
        context_tokens=[10, 11, 12],
        specs=specs,
        temperature=0.7,
        top_p=0.9,
    )
    assert rec_harm["target_mass_on_candidates"] < 0.05
    assert rec_harm["configs"]["static-b1p50"]["delta_tv"] > 0.0
    assert rec_harm["configs"]["static-b1p50"]["delta_expected_acceptance"] < 0.0

    # 3. Unmatched prefix: context [19, 19] has no evidence candidates -> exact zero delta
    rec_none = evaluate_prefix_biasing(
        claim_id="c1",
        position=2,
        realized_token_id=18,
        realized_token_text="other",
        target_logits=t_logits,
        draft_logits=d_logits,
        evidence_indexer=indexer,
        evidence_texts=evidence,
        context_tokens=[19, 19],
        specs=specs,
    )
    assert rec_none["candidate_count"] == 0
    assert rec_none["configs"]["entropy-b1p50"]["is_biased"] is False
    assert rec_none["configs"]["entropy-b1p50"]["delta_tv"] == 0.0
    assert rec_none["configs"]["entropy-b1p50"]["delta_expected_acceptance"] == 0.0


def test_replay_claim_prefixes_exact_teacher_forced_alignment():
    """Verify target_logits[P-1+t] aligns with context prompt_ids + token_ids[:t]."""
    tok = _MockTokenizer()

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    class _DeterministicLM:
        def __init__(self, boost_next_from_prev: dict[int, int], vocab_size: int = 32):
            self.boost_map = boost_next_from_prev
            self.vocab_size = vocab_size

        def get_input_embeddings(self):
            return type("Emb", (), {"weight": torch.zeros(1)})()

        def __call__(self, input_ids, **kwargs):
            seq_len = input_ids.shape[1]
            logits = torch.zeros((1, seq_len, self.vocab_size), dtype=torch.float32)
            for pos in range(seq_len):
                prev_tok = int(input_ids[0, pos].item())
                nxt = self.boost_map.get(prev_tok, 18)
                logits[0, pos, nxt] = 4.0
            return _Out(logits)

    # Prompt: "The state of" -> [10, 11, 12] (P=3)
    # Generated: [13, 14, 15] ("Georgia reported 120") (G=3)
    # After token 12 ("of"), target predicts 13 ("Georgia"), draft predicts 18 ("other")
    target_lm = _DeterministicLM({12: 13, 13: 14, 14: 15})
    draft_lm = _DeterministicLM({12: 18, 13: 14, 14: 18})
    specs = build_default_audit_specs(betas=[1.0], variants=["static", "entropy"])

    recs = replay_claim_prefixes(
        claim_id="c-align",
        prompt="The state of",
        evidence_texts=["The state of Georgia reported 120 cases"],
        token_ids=[13, 14, 15],
        target_model=target_lm,
        draft_model=draft_lm,
        tokenizer=tok,
        specs=specs,
        temperature=0.7,
        top_p=0.9,
    )
    assert len(recs) == 3
    # Position 0 predicts "Georgia" (13) after [10, 11, 12] -> matched_order=3, target puts high mass on 13
    assert recs[0]["realized_token_id"] == 13
    assert recs[0]["matched_order"] == 3
    assert recs[0]["target_mass_on_candidates"] > 0.9
    assert recs[0]["configs"]["static-b1p00"]["delta_tv"] < 0.0
    # Position 1 predicts "reported" (14) after [..., 13] -> target and draft agree closely
    assert recs[1]["realized_token_id"] == 14
    # Position 2 predicts "120" (15) -> numeric semantic category
    assert recs[2]["semantic_category"] == "numeric"


def test_claim_clustered_bootstrap_and_full_report_with_figures(tmp_path: Path):
    # Verify claim-clustered bootstrap detects degraded effect honestly
    degraded_sums = {f"c{i}": (-0.05 * (i + 1), 10) for i in range(10)}
    res_deg = claim_clustered_bootstrap_mean(
        degraded_sums, all_claim_ids=list(degraded_sums.keys()), n_bootstrap=500, seed=42, higher_is_better=True
    )
    assert res_deg["cluster_unit"] == "claim_id"
    assert res_deg["interpretation"] == "degraded_ci_excludes_zero"
    assert res_deg["ci_95_high"] < 0.0

    # End-to-end report + 5 real PNG figures on synthetic multi-claim records
    tok = _MockTokenizer()
    evidence = ["The state of Georgia reported 120 cases [E1]"]
    indexer = EvidenceIndexer(tok, evidence, max_ngram=4)
    specs = build_default_audit_specs(betas=[0.5, 1.5], variants=["static", "entropy"])

    all_records = []
    for c_idx in range(6):
        cid = f"claim-{c_idx}"
        # 1 helpful prefix (Georgia), 1 harmful prefix (wrong), 1 unmatched prefix (other)
        t_help = torch.full((32,), -1.5); t_help[13] = 4.0
        d_help = torch.full((32,), -0.5); d_help[18] = 1.2; d_help[13] = 0.2
        all_records.append(evaluate_prefix_biasing(
            claim_id=cid, position=0, realized_token_id=13, realized_token_text="Georgia",
            target_logits=t_help, draft_logits=d_help, evidence_indexer=indexer,
            evidence_texts=evidence, context_tokens=[10, 11, 12], specs=specs,
        ))

        t_harm = torch.full((32,), -1.5); t_harm[19] = 4.5
        d_harm = torch.full((32,), -0.5); d_harm[19] = 1.5; d_harm[13] = 0.4
        all_records.append(evaluate_prefix_biasing(
            claim_id=cid, position=1, realized_token_id=19, realized_token_text="wrong",
            target_logits=t_harm, draft_logits=d_harm, evidence_indexer=indexer,
            evidence_texts=evidence, context_tokens=[12], specs=specs,
        ))

        all_records.append(evaluate_prefix_biasing(
            claim_id=cid, position=2, realized_token_id=15, realized_token_text="120",
            target_logits=t_help, draft_logits=d_help, evidence_indexer=indexer,
            evidence_texts=evidence, context_tokens=[19, 19], specs=specs,
        ))

    fake_sens = {
        "rows": [
            {"variant": "static", "beta0": 0.0, "acceptance_eval_pct": 82.5, "throughput_e2e_tok_s": 26.0},
            {"variant": "static", "beta0": 0.5, "acceptance_eval_pct": 82.45, "throughput_e2e_tok_s": 26.37},
            {"variant": "static", "beta0": 1.5, "acceptance_eval_pct": 82.36, "throughput_e2e_tok_s": 26.35},
            {"variant": "entropy", "beta0": 0.0, "acceptance_eval_pct": 82.5, "throughput_e2e_tok_s": 26.27},
            {"variant": "entropy", "beta0": 0.5, "acceptance_eval_pct": 82.66, "throughput_e2e_tok_s": 26.27},
            {"variant": "entropy", "beta0": 1.5, "acceptance_eval_pct": 82.74, "throughput_e2e_tok_s": 26.19},
        ]
    }
    report = build_distribution_audit_report(
        all_records,
        primary_config="entropy-b1p50",
        sensitivity_report=fake_sens,
        n_bootstrap=300,
        seed=42,
    )
    assert report["schema_version"] == "m4_1_distribution_audit_v1"
    assert report["cluster_unit"] == "claim_id"
    assert report["n_claims"] == 6
    assert report["n_prefixes"] == 18
    assert len(report["expected_vs_observed_comparison"]) == 4
    assert report["decision_gate"]["verdict"] in {
        "KEEP_CURRENT_BIASING",
        "TARGETED_MODIFICATION_CANDIDATE",
        "DROP_OR_REFRAME_AS_NEGATIVE_RESULT",
    }

    figs = generate_figures(report, all_records, tmp_path / "figures")
    assert len(figs) == 5
    for fn in figs:
        fp = tmp_path / "figures" / fn
        assert fp.exists() and fp.stat().st_size > 1000

    md = render_markdown(report, {"git_commit": "2fa8cdd", "git_dirty": False}, figs)
    for required_heading in [
        "### Q1. What percentage of prefixes are biased?",
        "### Q2. What percentage of prefixes have TV reduced vs. increased?",
        "### Q3. In which cases does biasing increase TV?",
        "### Q4. Does draft entropy actually predict biasing benefit?",
        "### Q5. Is biasing overhead worth the expected benefit?",
        "## 4. Per-Prefix Expected Acceptance vs. Observed Stochastic Acceptance",
        "## 5. Pre-Registered Decision Gate Evaluation",
    ]:
        assert required_heading in md
