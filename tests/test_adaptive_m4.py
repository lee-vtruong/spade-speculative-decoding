"""Unit tests for Milestone M4: Evidence-Adaptive Speculative Decoding."""
from __future__ import annotations
import math
import pytest
import torch
from spade.adaptive.evidence_indexer import EvidenceIndexer
from spade.adaptive.biasing import apply_evidence_bias, EvidenceBiasingConfig, compute_entropy
from spade.decoding.speculative import exact_speculative_round

class MockTokenizer:
    """Lightweight mock tokenizer mapping words to predictable token IDs."""
    def __init__(self):
        self.vocab = {"The": 10, "state": 11, "of": 12, "Georgia": 13, "reported": 14, "cases": 15}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = text.split()
        return [self.vocab.get(t, 99) for t in tokens]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join([self.inv_vocab.get(t, "<unk>") for t in token_ids])

def test_evidence_indexer_ngram_matching():
    tok = MockTokenizer()
    evidence = [
        "The state of Georgia reported",
        "Georgia reported cases"
    ]
    indexer = EvidenceIndexer(tok, evidence, max_ngram=4)

    # Context ends in ["state", "of"] -> [11, 12]
    context = [100, 11, 12]
    candidates, order = indexer.get_candidate_tokens(context)
    assert order == 2
    assert 13 in candidates  # "Georgia" token ID is 13

    # Context ends in ["Georgia", "reported"] -> [13, 14]
    candidates2, order2 = indexer.get_candidate_tokens([13, 14])
    assert order2 == 2
    assert 15 in candidates2  # "cases" token ID is 15

    # Context not matching evidence
    candidates_none, order_none = indexer.get_candidate_tokens([999, 888])
    assert order_none == 0
    assert len(candidates_none) == 0

def test_apply_evidence_bias_modulates_logits():
    tok = MockTokenizer()
    indexer = EvidenceIndexer(tok, ["state of Georgia"], max_ngram=3)
    vocab_size = 100
    logits = torch.zeros(vocab_size)

    # Context: ["state", "of"] -> [11, 12]. Next should boost token 13 ("Georgia")
    context = [11, 12]
    config = EvidenceBiasingConfig(boost_weight=2.0, adaptive_entropy=False)
    biased = apply_evidence_bias(logits, indexer, context, config)

    assert biased[13] > logits[13]
    assert float(biased[13].item()) > 1.5
    # Non-candidate tokens should remain 0.0
    assert float(biased[0].item()) == 0.0
    assert float(biased[99].item()) == 0.0

def test_entropy_gating_behavior():
    # Low entropy distribution (peaked on token 0)
    peaked_logits = torch.tensor([50.0, 0.0, 0.0, 0.0])
    h_low = compute_entropy(peaked_logits)
    assert h_low < 0.1

    # High entropy distribution (uniform)
    uniform_logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
    h_high = compute_entropy(uniform_logits)
    assert h_high > 1.3

def test_m4_exact_speculative_sampling_distribution_invariance():
    """Verify that exact speculative sampling with biased proposals preserves target distribution."""
    # Target distribution p
    p = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)
    # Draft proposal distribution q (e.g. heavily biased towards token 2)
    q_biased = torch.tensor([0.1, 0.1, 0.8], dtype=torch.float64)

    # Draw many samples from speculative round and compare empirical distribution to p
    N = 10000
    committed_counts = torch.zeros(3, dtype=torch.float64)
    gen = torch.Generator().manual_seed(42)

    for _ in range(N):
        # Sample proposal from q
        prop = int(torch.multinomial(q_biased, num_samples=1, generator=gen).item())
        res = exact_speculative_round(
            proposals=[prop],
            q_distributions=[q_biased],
            p_distributions=[p],
            generator=gen,
            remaining=1  # Only 1 token needed
        )
        first_token = res.committed[0]
        committed_counts[first_token] += 1

    empirical_p = committed_counts / N
    # Theorem 1: empirical distribution of committed first token converges to target p
    assert torch.allclose(empirical_p, p, atol=0.03)
