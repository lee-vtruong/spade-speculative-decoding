"""Tests for sampling mathematics, Leviathan acceptance rule, and residual distributions."""
from __future__ import annotations
import pytest
import torch
from spade.decoding.sampling import (
    processed_distribution,
    acceptance_probability,
    residual_distribution,
    sample_distribution,
    total_variation,
    expected_acceptance,
)
from spade.decoding.speculative import exact_speculative_round

def test_softmax_temperature_scaling():
    logits = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    # High temperature -> more uniform
    high_t = processed_distribution(logits, temperature=100.0)
    assert high_t.tolist() == pytest.approx([1/3, 1/3, 1/3], abs=0.01)

    # Low temperature -> more peaked at argmax (index 2)
    low_t = processed_distribution(logits, temperature=0.1)
    assert low_t[2].item() > 0.99

def test_top_k_filtering():
    logits = torch.tensor([1.0, 5.0, 2.0, 4.0, 3.0], dtype=torch.float64)
    # top_k=2 should keep indices 1 (val 5.0) and 3 (val 4.0)
    probs = processed_distribution(logits, top_k=2)
    assert probs[0].item() == 0.0
    assert probs[2].item() == 0.0
    assert probs[4].item() == 0.0
    assert probs[1].item() > probs[3].item()
    assert probs.sum().item() == pytest.approx(1.0)

def test_top_p_nucleus_filtering():
    # Mass concentrated on first elements
    probs = torch.tensor([0.6, 0.3, 0.08, 0.02], dtype=torch.float64)
    logits = torch.log(probs)
    # top_p=0.8 should keep [0.6, 0.3] which sums to 0.9 >= 0.8
    filtered = processed_distribution(logits, top_p=0.8)
    assert filtered[2].item() == 0.0
    assert filtered[3].item() == 0.0
    assert filtered[0].item() == pytest.approx(0.6 / 0.9, rel=1e-5)
    assert filtered[1].item() == pytest.approx(0.3 / 0.9, rel=1e-5)

def test_total_variation_and_expected_acceptance_identity():
    # \sum_v \min(p(v), q(v)) = 1 - TV(p, q)
    test_pairs = [
        ([0.7, 0.3], [0.4, 0.6]),
        ([0.5, 0.5], [0.5, 0.5]),
        ([1.0, 0.0, 0.0], [0.0, 0.5, 0.5]),
        ([0.2, 0.3, 0.5], [0.5, 0.2, 0.3]),
        ([0.01, 0.99], [0.99, 0.01]),
    ]
    for p_vals, q_vals in test_pairs:
        p = torch.tensor(p_vals, dtype=torch.float64)
        q = torch.tensor(q_vals, dtype=torch.float64)
        tv = total_variation(p, q)
        ea = expected_acceptance(p, q)
        assert ea == pytest.approx(1.0 - tv, abs=1e-12)

def test_acceptance_probability_bounds():
    p = torch.tensor([0.8, 0.2], dtype=torch.float64)
    q = torch.tensor([0.4, 0.6], dtype=torch.float64)

    # For token 0: p(0)=0.8, q(0)=0.4 -> min(1, 0.8/0.4) = 1.0
    assert acceptance_probability(p, q, 0).item() == 1.0

    # For token 1: p(1)=0.2, q(1)=0.6 -> min(1, 0.2/0.6) = 1/3
    assert acceptance_probability(p, q, 1).item() == pytest.approx(1/3, rel=1e-6)

def test_residual_distribution_normalization_and_disjoint_support():
    p = torch.tensor([0.7, 0.3, 0.0], dtype=torch.float64)
    q = torch.tensor([0.2, 0.0, 0.8], dtype=torch.float64)

    # p - q = [0.5, 0.3, -0.8]
    # max(0, p - q) = [0.5, 0.3, 0.0], sum = 0.8
    # residual = [0.5/0.8, 0.3/0.8, 0.0] = [0.625, 0.375, 0.0]
    r = residual_distribution(p, q)
    assert r.tolist() == pytest.approx([0.625, 0.375, 0.0], rel=1e-6)
    assert r.sum().item() == pytest.approx(1.0)

    # Completely disjoint support: p=[1, 0], q=[0, 1]
    p_disjoint = torch.tensor([1.0, 0.0], dtype=torch.float64)
    q_disjoint = torch.tensor([0.0, 1.0], dtype=torch.float64)
    r_disjoint = residual_distribution(p_disjoint, q_disjoint)
    assert r_disjoint.tolist() == [1.0, 0.0]

def test_residual_distribution_undefined_when_p_equals_q():
    p = torch.tensor([0.5, 0.5], dtype=torch.float64)
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    with pytest.raises(ValueError, match="Residual distribution is undefined"):
        residual_distribution(p, q)

@pytest.mark.parametrize("p_values, q_values", [
    ([0.5, 0.5], [0.5, 0.5]),
    ([0.2, 0.8], [0.8, 0.2]),
    ([0.0, 0.3, 0.7], [0.6, 0.4, 0.0]),
    ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
    ([1.0 - 1e-10, 1e-10], [1e-10, 1.0 - 1e-10]),
])
def test_empirical_exact_sampling_distribution(p_values, q_values):
    """Statistically validate that exact speculative sampling preserves the target distribution p."""
    p = torch.tensor(p_values, dtype=torch.float64)
    q = torch.tensor(q_values, dtype=torch.float64)
    gen = torch.Generator().manual_seed(4242)
    n_samples = 6000
    counts = torch.zeros_like(p, dtype=torch.int64)

    for _ in range(n_samples):
        proposal = sample_distribution(q, gen)
        res = exact_speculative_round([proposal], [q], [p], generator=gen, remaining=1)
        committed_token = res.committed[0]
        counts[committed_token] += 1

    empirical_p = counts.double() / n_samples
    # 5 standard deviations threshold for binomial proportion: 5 * sqrt(p*(1-p)/N)
    std_error = torch.sqrt(p * (1.0 - p) / n_samples)
    tolerance = torch.maximum(torch.full_like(p, 0.03), 5.0 * std_error)
    assert torch.all(torch.abs(empirical_p - p) <= tolerance), f"Empirical {empirical_p.tolist()} deviates from target {p.tolist()}"
