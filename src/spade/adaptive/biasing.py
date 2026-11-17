"""Evidence-Guided Draft Logit Biasing for SPADE (M4).

Modulates draft model logits towards evidence-supported tokens based on n-gram
continuity and dynamic entropy gating, without altering the exact target verification rule.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from spade.adaptive.evidence_indexer import EvidenceIndexer

@dataclass
class EvidenceBiasingConfig:
    """Hyperparameters for evidence-adaptive draft biasing."""
    boost_weight: float = 1.5
    order_decay: float = 0.5
    single_token_boost: float = 0.2
    adaptive_entropy: bool = True
    entropy_threshold: float = 2.0
    max_boost_cap: float = 3.0

def compute_entropy(logits: torch.Tensor) -> float:
    """Compute Shannon entropy (in nats) of the softmax distribution."""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs).item()
    return float(entropy)

def apply_evidence_bias(
    draft_logits: torch.Tensor,
    evidence_indexer: EvidenceIndexer,
    context_tokens: list[int],
    config: EvidenceBiasingConfig | None = None
) -> torch.Tensor:
    """Apply evidence-guided boost to draft logits in-place or on a clone.

    Args:
        draft_logits: 1D tensor [vocab_size] of unnormalized logits from draft model.
        evidence_indexer: Pre-indexed evidence token continuations.
        context_tokens: Sequence of tokens generated so far.
        config: Biasing configuration.

    Returns:
        Biased logits tensor [vocab_size].
    """
    if config is None:
        config = EvidenceBiasingConfig()

    candidates, matched_order = evidence_indexer.get_candidate_tokens(context_tokens)
    if not candidates and not evidence_indexer.evidence_token_set:
        return draft_logits

    biased_logits = draft_logits.clone()

    # Dynamic gating factor based on draft entropy
    gating = 1.0
    if config.adaptive_entropy:
        h = compute_entropy(draft_logits)
        # When drafter is uncertain (high entropy), boost evidence grounding
        # When drafter is confident (low entropy), reduce intervention
        gating = 1.0 / (1.0 + math.exp(- (h - config.entropy_threshold)))

    # 1. Boost exact n-gram continuation tokens
    if candidates:
        order_factor = 1.0 + config.order_decay * (matched_order - 1)
        boost_val = min(config.max_boost_cap, config.boost_weight * order_factor * gating)
        cand_tensor = torch.tensor(list(candidates), dtype=torch.long, device=draft_logits.device)
        biased_logits[cand_tensor] += boost_val

    return biased_logits
