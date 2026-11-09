"""Tests for speculative decoding verification logic, greedy matching, and exact sampling rounds."""
from __future__ import annotations
import pytest
import torch
from spade.decoding.speculative import exact_speculative_round, greedy_verify

def test_greedy_verify_perfect_match_with_extra():
    proposals = [10, 20, 30]
    target_greedy = [10, 20, 30]
    out = greedy_verify(proposals, target_greedy)
    assert out == [10, 20, 30]

def test_greedy_verify_first_token_mismatch():
    proposals = [10, 20, 30]
    target_greedy = [99, 20, 30]
    out = greedy_verify(proposals, target_greedy)
    # At position 0, proposed is 10 but target expects 99 -> commits 99 and stops immediately
    assert out == [99]

def test_greedy_verify_middle_token_mismatch():
    proposals = [10, 20, 30]
    target_greedy = [10, 99, 30]
    out = greedy_verify(proposals, target_greedy)
    # Position 0 matches (10), position 1 mismatches (99) -> commits [10, 99]
    assert out == [10, 99]

def test_greedy_verify_eos_stopping():
    proposals = [10, 2, 30]
    target_greedy = [10, 2, 30]
    # eos_token_id=2
    out = greedy_verify(proposals, target_greedy, eos_token_id=2)
    assert out == [10, 2]

def test_exact_speculative_full_acceptance_with_extra_target_token():
    # p and q agree perfectly
    p = torch.tensor([1.0, 0.0], dtype=torch.float64)
    q = torch.tensor([1.0, 0.0], dtype=torch.float64)
    target_extra = torch.tensor([0.0, 1.0], dtype=torch.float64)  # will sample token 1

    gen = torch.Generator().manual_seed(101)
    res = exact_speculative_round(
        proposals=[0, 0],
        q_distributions=[q, q],
        p_distributions=[p, p],
        generator=gen,
        target_extra=target_extra,
        remaining=5
    )

    assert res.full_acceptance is True
    assert res.accepted == 2
    assert res.rejected_position is None
    # Committed should be proposed [0, 0] plus extra [1]
    assert res.committed == [0, 0, 1]
    assert res.extra_target == 1
    assert res.statuses == ["accepted", "accepted"]

@pytest.mark.parametrize("rejection_pos", [0, 1, 2])
def test_exact_speculative_rejection_at_any_position(rejection_pos):
    # Construct distributions where position `rejection_pos` is forced to reject
    accept_dist = torch.tensor([1.0, 0.0], dtype=torch.float64)
    reject_target = torch.tensor([0.0, 1.0], dtype=torch.float64)  # target wants 1
    reject_draft = torch.tensor([1.0, 0.0], dtype=torch.float64)   # draft proposed 0

    proposals = [0, 0, 0]
    q_dists = [reject_draft, reject_draft, reject_draft]
    p_dists = [accept_dist if i != rejection_pos else reject_target for i in range(3)]

    gen = torch.Generator().manual_seed(202)
    res = exact_speculative_round(proposals, q_dists, p_dists, generator=gen)

    assert res.rejected_position == rejection_pos
    assert res.accepted == rejection_pos
    # Committed should have accepted tokens up to rejection_pos, then residual replacement (1)
    expected_committed = [0] * rejection_pos + [1]
    assert res.committed == expected_committed

    expected_statuses = ["accepted"] * rejection_pos + ["rejected"] + ["discarded_after_rejection"] * (2 - rejection_pos)
    assert res.statuses == expected_statuses

def test_exact_speculative_eos_stops_round_without_extra_token():
    p = torch.tensor([0.5, 0.5], dtype=torch.float64)
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    target_extra = torch.tensor([1.0, 0.0], dtype=torch.float64)

    # Token 1 is EOS
    gen = torch.Generator().manual_seed(303)
    res = exact_speculative_round(
        proposals=[0, 1, 0],
        q_distributions=[p, p, p],
        p_distributions=[p, p, p],
        generator=gen,
        eos_token_id=1,
        target_extra=target_extra
    )
    # Must stop at EOS (token 1) and not append extra token
    assert res.committed == [0, 1]
    assert res.extra_target is None

def test_exact_speculative_remaining_budget_clips_round():
    p = torch.tensor([1.0, 0.0], dtype=torch.float64)
    gen = torch.Generator().manual_seed(404)
    # Draft requested 4, but budget remaining is 2
    res = exact_speculative_round(
        proposals=[0, 0, 0, 0],
        q_distributions=[p, p, p, p],
        p_distributions=[p, p, p, p],
        generator=gen,
        remaining=2,
        target_extra=p
    )
    assert len(res.committed) == 2
    assert res.extra_target is None  # no extra token when hitting remaining limit
