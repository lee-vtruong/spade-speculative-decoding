"""Automated tests ensuring zero leakage of gold labels or justifications into LLM prompts."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from spade.benchmark.benchmark_builder import build_examples
from spade.benchmark.prompt_builder import build_prompt, VERDICTS

FIXTURES = Path(__file__).parent / "fixtures"

def test_gold_justification_never_in_prompt():
    row = {
        "id": "leakage_test_1",
        "claim": "Water boils at 100 degrees Celsius under standard atmospheric pressure.",
        "label": "Supported",
        "justification": "CLASSIFIED_GOLD_JUSTIFICATION_DO_NOT_LEAK_XYZ123",
        "questions": [
            {
                "question": "At what temperature does water boil at 1 atm?",
                "answers": [{"answer": "100 C", "source_url": "https://physics.org"}]
            }
        ]
    }
    examples = build_examples([row], mode="oracle_qa")
    prompt_text = examples[0]["prompt"]

    assert "CLASSIFIED_GOLD_JUSTIFICATION_DO_NOT_LEAK_XYZ123" not in prompt_text
    assert "CLASSIFIED_GOLD_JUSTIFICATION_DO_NOT_LEAK_XYZ123" not in examples[0]["prompt_hash"]

    # Evaluation-only field preserves gold justification safely
    assert examples[0]["evaluation"]["gold_justification"] == "CLASSIFIED_GOLD_JUSTIFICATION_DO_NOT_LEAK_XYZ123"

def test_prompt_visible_verdict_vocabulary_vs_instance_gold_label():
    """Verify that while the schema lists allowed verdict categories, the specific gold

    label for this instance is not indicated as the answer.
    """
    row = {
        "id": "leakage_test_2",
        "claim": "A unique synthetic claim.",
        "label": "Refuted",
        "justification": "Unique secret justification 999888.",
        "questions": [{"question": "Q?", "answer": "A."}]
    }
    examples = build_examples([row], mode="oracle_qa")
    prompt = examples[0]["prompt"]

    # The prompt lists candidate verdicts:
    for verdict in VERDICTS:
        assert verdict in prompt

    # But the instance prompt does not contain 'verdict: Refuted' or mention the gold label
    assert '"verdict": "Refuted"' not in prompt
    assert '"verdict":"Refuted"' not in prompt
    assert "Unique secret justification 999888" not in prompt

def test_retrieved_evidence_mode_leakage_prevention():
    row = {
        "id": "leakage_test_3",
        "claim": "Earth is flat.",
        "label": "Refuted",
        "justification": "SECRET_GOLD_GROUND_TRUTH",
    }
    retrieved_data = {
        "leakage_test_3": [
            {"text": "Earth is an oblate spheroid.", "rank": 1, "score": 0.95, "source_url": "https://nasa.gov"}
        ]
    }
    examples = build_examples([row], mode="retrieved", retrieved=retrieved_data)
    prompt = examples[0]["prompt"]

    assert "SECRET_GOLD_GROUND_TRUTH" not in prompt
    assert "nasa.gov" not in prompt  # Visible prompt excludes URLs to keep prompts minimal and clean
    assert "Earth is an oblate spheroid." in prompt
    assert "E1" in prompt
    assert examples[0]["evaluation"]["gold_label"] == "Refuted"

def test_prompt_hash_reproducibility():
    claim = "Testing determinism of prompt construction"
    evidence = [{"evidence_id": "E1", "text": "Evidence snippet"}]
    prompt1, hash1 = build_prompt(claim, evidence, "retrieved")
    prompt2, hash2 = build_prompt(claim, evidence, "retrieved")

    assert prompt1 == prompt2
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex length
