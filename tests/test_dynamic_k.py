"""Unit tests for Dynamic Draft Length Adaptation K(t) in SPADE (M4)."""
import math
import torch
import pytest

from spade.adaptive.evidence_indexer import EvidenceIndexer
from spade.adaptive.dynamic_k import DynamicKConfig, DynamicKController
from spade.decoding.speculative import exact_speculative_round


class DummyTokenizer:
    """Mock tokenizer mapping words to token ids."""
    def __init__(self):
        self.vocab = {"the": 1, "claim": 2, "is": 3, "supported": 4, "by": 5, "evidence": 6, "1994": 7, "October": 8}
        self.inv = {v: k for k, v in self.vocab.items()}
        self.eos_token_id = 999

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) + 1
                self.inv[self.vocab[word]] = word
            tokens.append(self.vocab[word])
        return tokens

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return " ".join(self.inv.get(i, f"tok_{i}") for i in token_ids)


def test_entropy_computation():
    controller = DynamicKController()
    vocab_size = 100
    
    # Peaked distribution (one-hot): entropy = 0
    peaked = torch.zeros(vocab_size)
    peaked[42] = 100.0
    h_peaked = controller.compute_entropy(peaked)
    assert h_peaked < 0.01

    # Uniform distribution: entropy = ln(vocab_size)
    uniform = torch.zeros(vocab_size)
    h_uniform = controller.compute_entropy(uniform)
    expected_h = math.log(vocab_size)
    assert abs(h_uniform - expected_h) < 1e-4


def test_low_entropy_triggers_expansion():
    cfg = DynamicKConfig(min_k=2, default_k=4, max_k=6, entropy_low=1.0, entropy_high=2.5)
    controller = DynamicKController(cfg)
    vocab_size = 50

    # Sharp distribution with top prob > 0.99
    sharp_logits = torch.zeros(vocab_size)
    sharp_logits[10] = 10.0
    
    k, reason = controller.determine_k(sharp_logits, evidence_indexer=None, context_tokens=[])
    assert k == 6
    assert "low_entropy" in reason


def test_high_entropy_triggers_contraction():
    cfg = DynamicKConfig(min_k=2, default_k=4, max_k=6, entropy_low=1.0, entropy_high=2.0)
    controller = DynamicKController(cfg)
    vocab_size = 50

    # Flat, uncertain distribution
    flat_logits = torch.randn(vocab_size) * 0.1
    
    k, reason = controller.determine_k(flat_logits, evidence_indexer=None, context_tokens=[])
    assert k == 2
    assert "high_entropy" in reason


def test_evidence_continuation_expansion():
    tok = DummyTokenizer()
    evidence_indexer = EvidenceIndexer(tok, ["the claim is supported by evidence"])
    cfg = DynamicKConfig(min_k=2, default_k=4, max_k=6, entropy_low=0.5, entropy_high=3.0, min_evidence_order=2)
    controller = DynamicKController(cfg)

    # Context: "the claim is" -> next token in evidence is "supported" (id=4)
    ctx = tok.encode("the claim is")
    
    # Logits has moderate entropy (e.g. 1.8), but predicts "supported" with high probability
    logits = torch.randn(len(tok.vocab) + 10) * 0.5
    supp_id = tok.vocab["supported"]
    logits[supp_id] = 4.0  # high mass on evidence continuation

    k, reason = controller.determine_k(logits, evidence_indexer=evidence_indexer, context_tokens=ctx)
    assert k == 6
    assert "evidence_continuation" in reason


def test_history_rejection_throttles_k():
    cfg = DynamicKConfig(min_k=2, default_k=4, max_k=6, entropy_low=0.5, entropy_high=4.0, use_history=True)
    controller = DynamicKController(cfg)
    vocab_size = 50

    # Previous round was proposed 4, accepted 0 (immediate rejection at Pos 0)
    controller.update_history(accepted_tokens=0, proposed_tokens=4)

    # Moderate logits
    logits = torch.randn(vocab_size)
    k, reason = controller.determine_k(logits, evidence_indexer=None, context_tokens=[])
    assert k == 2
    assert "recent_immediate_rejection" in reason


def test_dynamic_k_preserves_target_distribution():
    """Verify that varying K dynamically across rounds strictly preserves the target distribution.
    
    Under Theorem 1 of Leviathan et al. (2023), exact speculative sampling produces identical
    marginal distribution to the target model regardless of K, because each accepted or replacement
    token is drawn according to the exact target distribution p(x).
    """
    vocab_size = 5
    torch.manual_seed(123)
    p_dist = torch.tensor([0.4, 0.3, 0.15, 0.1, 0.05], dtype=torch.float64)
    q_dist = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], dtype=torch.float64)

    # Simulate 5,000 sampling events with alternating K in {2, 4, 6}
    n_samples = 5000
    committed_counts = torch.zeros(vocab_size, dtype=torch.float64)
    k_choices = [2, 4, 6]

    for round_idx in range(n_samples):
        k = k_choices[round_idx % len(k_choices)]
        proposed = torch.multinomial(q_dist, num_samples=k, replacement=True).tolist()
        q_dists = [q_dist.clone() for _ in range(k)]
        p_dists = [p_dist.clone() for _ in range(k)]
        target_extra = p_dist.clone()

        res = exact_speculative_round(
            proposed, q_dists, p_dists,
            generator=torch.Generator().manual_seed(round_idx + 100),
            remaining=10,
            target_extra=target_extra
        )
        # The first committed token must marginally follow p_dist
        first_token = res.committed[0]
        committed_counts[first_token] += 1

    emp_probs = committed_counts / n_samples
    tv = float(0.5 * torch.abs(emp_probs - p_dist).sum().item())
    
    # TV distance between empirical sampling with alternating K and target distribution must be tiny (< 0.03)
    assert tv < 0.03, f"Total variation distance {tv} is too large; target distribution not preserved!"
