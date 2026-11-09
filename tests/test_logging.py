"""Tests for trace schemas, append-only logging, metric aggregations, and profiler."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import pytest
import torch
from spade.instrumentation.trace_schema import ROUND_FIELDS, TOKEN_FIELDS, TOKEN_STATUSES, validate_record
from spade.instrumentation.trace_writer import TraceWriter
from spade.instrumentation.profiler import InferenceProfiler
from spade.analysis.metrics import acceptance_summary
from spade.benchmark.evaluation import parse_generation, classification_metrics

def test_trace_schema_validation_success():
    valid_round = {
        "run_id": "r1", "claim_id": "c1", "round_index": 0, "prefix_length": 10,
        "draft_length_requested": 4, "proposed_token_count": 4, "accepted_draft_tokens": 3,
        "rejection_position": 3, "full_acceptance": False, "additional_target_token": False,
        "committed_output_tokens": 4, "draft_latency_s": 0.01, "verification_latency_s": 0.02,
        "correction_latency_s": 0.001, "total_round_latency_s": 0.031, "cumulative_generated_tokens": 4
    }
    validate_record(valid_round, ROUND_FIELDS)

    valid_token = {
        "run_id": "r1", "claim_id": "c1", "round_id": 0, "absolute_token_position": 10,
        "token_id": 42, "token_text": "hello", "draft_probability": 0.4, "target_probability": 0.8,
        "acceptance_probability": 1.0, "total_variation": 0.1, "expected_acceptance_probability": 0.9,
        "status": "accepted", "reached_by_verifier": True, "discarded_after_rejection": False,
        "replacement_token_id": None, "token_category": "other", "prefix_length": 10,
        "evidence_overlap": {"lexical_overlap": 0.0, "has_citation_marker": False, "overlapping_terms": []}
    }
    validate_record(valid_token, TOKEN_FIELDS)

def test_trace_schema_validation_unknown_status():
    invalid_token = {
        "run_id": "r1", "claim_id": "c1", "round_id": 0, "absolute_token_position": 10,
        "token_id": 42, "token_text": "hello", "draft_probability": 0.4, "target_probability": 0.8,
        "acceptance_probability": 1.0, "total_variation": 0.1, "expected_acceptance_probability": 0.9,
        "status": "unknown_heuristic_status", "reached_by_verifier": True, "discarded_after_rejection": False,
        "replacement_token_id": None, "token_category": "other", "prefix_length": 10,
        "evidence_overlap": {}
    }
    with pytest.raises(ValueError, match="Unknown token status"):
        validate_record(invalid_token, TOKEN_FIELDS)

def test_trace_writer_append_only():
    with tempfile.TemporaryDirectory() as temp:
        tmp = Path(temp)
        writer = TraceWriter(tmp / "round.jsonl", tmp / "token.jsonl")

        sample_round = {f: 0 for f in ROUND_FIELDS}
        sample_round.update(run_id="run_1", claim_id="claim_1", full_acceptance=False, additional_target_token=False)

        writer.write_round(sample_round)
        writer.write_round(sample_round)

        lines = (tmp / "round.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

def test_acceptance_summary_weighting_modes():
    token_rows = [
        # Claim 1: 2 accepted, 0 rejected (rate 1.0)
        {"claim_id": "c1", "status": "accepted", "reached_by_verifier": True, "target_probability": 0.8, "draft_probability": 0.5, "expected_acceptance_probability": 0.8, "total_variation": 0.2, "token_category": "numeric"},
        {"claim_id": "c1", "status": "accepted", "reached_by_verifier": True, "target_probability": 0.7, "draft_probability": 0.6, "expected_acceptance_probability": 0.7, "total_variation": 0.3, "token_category": "numeric"},
        # Claim 2: 0 accepted, 1 rejected (rate 0.0)
        {"claim_id": "c2", "status": "rejected", "reached_by_verifier": True, "target_probability": 0.1, "draft_probability": 0.9, "expected_acceptance_probability": 0.2, "total_variation": 0.8, "token_category": "punctuation"},
    ]
    round_rows = [
        {"claim_id": "c1", "proposed_token_count": 2, "accepted_draft_tokens": 2, "committed_output_tokens": 3},
        {"claim_id": "c2", "proposed_token_count": 1, "accepted_draft_tokens": 0, "committed_output_tokens": 1},
    ]

    summary = acceptance_summary(token_rows, round_rows)

    # Token-weighted: 2 accepted / 3 total reached = 2/3
    assert summary["observed_acceptance"]["token_weighted"] == pytest.approx(2/3)
    # Round-weighted: mean([2/2, 0/1]) = mean([1.0, 0.0]) = 0.5
    assert summary["observed_acceptance"]["round_weighted"] == pytest.approx(0.5)
    # Claim-weighted: mean([1.0, 0.0]) = 0.5
    assert summary["observed_acceptance"]["claim_weighted"] == pytest.approx(0.5)

    assert summary["token_category_acceptance"]["numeric"]["acceptance_rate"] == 1.0
    assert summary["token_category_acceptance"]["punctuation"]["acceptance_rate"] == 0.0

def test_inference_profiler():
    profiler = InferenceProfiler(torch.device("cpu"))
    profiler.start_request()

    with profiler.time_stage("prompt_prefill_s"):
        pass

    with profiler.time_stage("draft_generation_s"):
        pass

    with profiler.time_stage("target_verification_s"):
        pass

    with profiler.time_stage("decode_s"):
        pass

    report = profiler.end_request(prompt_tokens=100, output_tokens=20)
    assert report.prompt_tokens == 100
    assert report.output_tokens == 20
    assert report.end_to_end_s >= 0.0

def test_classification_metrics_and_factuality_not_evaluated():
    gold = ["Supported", "Refuted", "Supported", "Not Enough Evidence"]
    pred = ["Supported", "Refuted", "Refuted", "Not Enough Evidence"]

    metrics = classification_metrics(gold, pred)
    assert metrics["accuracy"] == 0.75
    assert "Supported" in metrics["per_class"]
    assert metrics["factuality"] == "not_evaluated"
