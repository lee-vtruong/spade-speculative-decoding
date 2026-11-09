"""Tests for AVeriTeC dataset loading, evidence adapters, and benchmark construction."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import pytest
from spade.benchmark.averitec_loader import load_records, claim_id, claim_text
from spade.benchmark.evidence_adapter import oracle_evidence, retrieved_evidence, load_claim_only_knowledge_store
from spade.benchmark.benchmark_builder import build_examples, build_manifest, stratified_sample
from scripts.run_baselines import read_jsonl

FIXTURES = Path(__file__).parent / "fixtures"

def test_load_records_from_json_and_jsonl():
    json_path = FIXTURES / "averitec_sample.json"
    rows = load_records(json_path)
    assert len(rows) == 4
    assert rows[0]["label"] == "Supported"

    # Test loading from JSONL
    with tempfile.TemporaryDirectory() as temp:
        jsonl_path = Path(temp) / "test.jsonl"
        jsonl_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        loaded = load_records(jsonl_path)
        assert len(loaded) == len(rows)
        assert claim_id(loaded[0]) == "0"

def test_claim_id_derivation_without_explicit_id():
    row_no_id = {
        "claim": "The moon is made of green cheese.",
        "claim_date": "2020",
        "speaker": "Anonymous",
        "label": "Refuted",
        "justification": "Astronauts collected rock samples."
    }
    cid1 = claim_id(row_no_id)
    assert cid1.startswith("avc-")

    # If label or justification changes, claim ID must remain identical
    row_modified = {
        **row_no_id,
        "label": "Supported",
        "justification": "Completely different text"
    }
    assert claim_id(row_modified) == cid1

def test_oracle_evidence_extraction():
    rows = load_records(FIXTURES / "averitec_sample.json")
    ev = oracle_evidence(rows[0])
    assert len(ev) == 1
    assert ev[0]["evidence_id"] == "E1"
    assert "Can mRNA vaccines alter DNA?" in ev[0]["text"]
    assert ev[0]["source_url"] == "https://who.int/vaccines-dna"
    assert ev[0]["provenance"] == "oracle_qa"

def test_oracle_evidence_top_k():
    row = {
        "claim": "Test claim",
        "questions": [{"question": f"Q{i}?", "answer": f"A{i}."} for i in range(10)]
    }
    examples = build_examples([row], mode="oracle_qa", top_k=3)
    assert len(examples[0]["evidence"]) == 3
    assert [e["evidence_id"] for e in examples[0]["evidence"]] == ["E1", "E2", "E3"]

def test_retrieved_evidence_adapter_from_knowledge_store():
    rows = load_records(FIXTURES / "averitec_sample.json")
    store_dir = FIXTURES / "knowledge_store_sample"
    mapped, audit = load_claim_only_knowledge_store(rows, store_dir, top_k=8)

    assert "0" in mapped
    assert len(mapped["0"]) == 2  # 2 exact claim matches in 0.json
    assert all("COVID-19" in p["text"] or "CDC" in p["text"] or "WHO" in p["text"] for p in mapped["0"])

    # Verify gold passages are excluded
    assert not any("Gold justification" in p["text"] for p in mapped["0"])
    # Verify question query passages are excluded
    assert not any("QA text" in p["text"] for p in mapped["0"])
    # Verify audit counts
    assert audit["query_types_seen"]["gold"] >= 1
    assert audit["query_types_seen"]["question"] >= 1
    assert audit["query_type_allowlist"] == ["claim"]

def test_stratified_sample_distribution():
    rows = [
        {"id": f"s_{i}", "claim": f"claim {i}", "label": "Supported"} for i in range(60)
    ] + [
        {"id": f"r_{i}", "claim": f"claim {i}", "label": "Refuted"} for i in range(40)
    ]
    sample = stratified_sample(rows, size=10, seed=42)
    assert len(sample) == 10
    supported = sum(r["label"] == "Supported" for r in sample)
    refuted = sum(r["label"] == "Refuted" for r in sample)
    assert supported == 6
    assert refuted == 4

    # Determinism check across runs
    sample2 = stratified_sample(rows, size=10, seed=42)
    assert [r["id"] for r in sample] == [r["id"] for r in sample2]

def test_manifest_metadata_and_hashes():
    rows = load_records(FIXTURES / "averitec_sample.json")
    examples = build_examples(rows, mode="oracle_qa", top_k=8)
    manifest = build_manifest(
        dataset_path="test_path.json",
        dataset_hash="deadbeef1234",
        split="dev",
        seed=42,
        mode="oracle_qa",
        top_k=8,
        examples=examples,
        token_lengths=[150, 200, 180, 220]
    )
    assert manifest["manifest_version"] == 1
    assert manifest["split"] == "dev"
    assert manifest["evidence_mode"] == "oracle_qa"
    assert len(manifest["selected_claim_ids"]) == 4
    assert manifest["statistics"]["example_count"] == 4
    assert manifest["statistics"]["input_token_count"]["mean"] == pytest.approx(187.5)

def test_read_jsonl_unicode_line_separator():
    with tempfile.TemporaryDirectory() as temp:
        p = Path(temp) / "test_unicode.jsonl"
        record = {"text": "Hello\u2028World", "claim_id": "test_1"}
        p.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        loaded = read_jsonl(p)
        assert len(loaded) == 1
        assert loaded[0]["text"] == "Hello\u2028World"
