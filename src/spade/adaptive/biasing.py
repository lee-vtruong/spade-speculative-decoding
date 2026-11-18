"""Evidence-Guided Draft Logit Biasing for SPADE (M4).

Modulates draft model logits towards evidence-supported tokens based on n-gram
continuity and entropy/support gating, without altering the exact target verification rule.
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
    # M4.1-motivated precision & band-pass gating parameters (backward-compatible defaults)
    min_match_order: int = 1          # Minimum matched n-gram order (1 = legacy; >=2 filters unigram noise)
    max_candidates: int = 0           # Maximum candidate set size (0 = unlimited; e.g. 3 filters high-fanout noise)
    gating_mode: str = "sigmoid"      # "sigmoid" (legacy when adaptive_entropy=True), "none", or "bandpass"
    entropy_low: float = 0.4          # Lower draft entropy bound (nats) for "bandpass" gating
    entropy_high: float = 2.5         # Upper draft entropy bound (nats) for "bandpass" gating
    min_draft_mass: float = 0.0       # Minimum draft softmax probability mass on candidate set (0.0 = disabled)


def compute_entropy(logits: torch.Tensor) -> float:
    """Compute Shannon entropy (in nats) of the softmax distribution via single-pass log_softmax."""
    log_p = torch.log_softmax(logits, dim=-1)
    return float((-torch.sum(torch.exp(log_p) * log_p)).item())


def compute_gating_and_boost(
    draft_logits: torch.Tensor,
    candidates: set[int],
    matched_order: int,
    config: EvidenceBiasingConfig,
) -> tuple[float, float, torch.Tensor | None]:
    """Evaluate precision filters + gating and return (gating_value, applied_boost, cand_tensor).

    Returns (0.0, 0.0, None) when any filter rejects the prefix, avoiding tensor cloning.
    """
    if config.boost_weight <= 0.0 or not candidates:
        return 0.0, 0.0, None
    if matched_order < config.min_match_order:
        return 0.0, 0.0, None
    if config.max_candidates > 0 and len(candidates) > config.max_candidates:
        return 0.0, 0.0, None

    mode = config.gating_mode
    cand_tensor: torch.Tensor | None = None

    if mode == "bandpass":
        log_p = torch.log_softmax(draft_logits, dim=-1)
        probs = torch.exp(log_p)
        h = float((-torch.sum(probs * log_p)).item())
        if h < config.entropy_low or h > config.entropy_high:
            return 0.0, 0.0, None
        cand_tensor = torch.as_tensor(tuple(candidates), dtype=torch.long, device=draft_logits.device)
        if config.min_draft_mass > 0.0:
            cand_mass = float(probs[cand_tensor].sum().item())
            if cand_mass < config.min_draft_mass:
                return 0.0, 0.0, None
        gating = 1.0
    elif mode == "sigmoid" and config.adaptive_entropy:
        h = compute_entropy(draft_logits)
        gating = 1.0 / (1.0 + math.exp(-(h - config.entropy_threshold)))
        if config.min_draft_mass > 0.0:
            probs = torch.softmax(draft_logits, dim=-1)
            cand_tensor = torch.as_tensor(tuple(candidates), dtype=torch.long, device=draft_logits.device)
            if float(probs[cand_tensor].sum().item()) < config.min_draft_mass:
                return 0.0, 0.0, None
    elif mode == "none" or not config.adaptive_entropy:
        gating = 1.0
        if config.min_draft_mass > 0.0:
            probs = torch.softmax(draft_logits, dim=-1)
            cand_tensor = torch.as_tensor(tuple(candidates), dtype=torch.long, device=draft_logits.device)
            if float(probs[cand_tensor].sum().item()) < config.min_draft_mass:
                return 0.0, 0.0, None
    else:
        raise ValueError(f"Unsupported gating_mode: {mode!r}")

    order_factor = 1.0 + config.order_decay * (matched_order - 1)
    boost_val = min(config.max_boost_cap, config.boost_weight * order_factor * gating)
    if boost_val <= 0.0:
        return 0.0, 0.0, None

    if cand_tensor is None:
        cand_tensor = torch.as_tensor(tuple(candidates), dtype=torch.long, device=draft_logits.device)
    return gating, boost_val, cand_tensor


def apply_evidence_bias(
    draft_logits: torch.Tensor,
    evidence_indexer: EvidenceIndexer,
    context_tokens: list[int],
    config: EvidenceBiasingConfig | None = None,
) -> torch.Tensor:
    """Apply evidence-guided boost to draft logits on a clone when filters and gates pass.

    Args:
        draft_logits: 1D tensor [vocab_size] of unnormalized logits from draft model.
        evidence_indexer: Pre-indexed evidence token continuations.
        context_tokens: Sequence of tokens generated so far.
        config: Biasing configuration.

    Returns:
        Biased logits tensor [vocab_size], or original draft_logits if filtered out.
    """
    if config is None:
        config = EvidenceBiasingConfig()

    if config.boost_weight <= 0.0:
        return draft_logits

    candidates, matched_order = evidence_indexer.get_candidate_tokens(context_tokens)
    if not candidates:
        return draft_logits

    _, boost_val, cand_tensor = compute_gating_and_boost(draft_logits, candidates, matched_order, config)
    if boost_val <= 0.0 or cand_tensor is None:
        return draft_logits

    biased_logits = draft_logits.clone()
    biased_logits[cand_tensor] += boost_val
    return biased_logits
