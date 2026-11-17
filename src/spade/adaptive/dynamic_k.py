"""Dynamic Draft Length Adaptation for SPADE (M4).

Dynamically schedules speculative draft length K(t) in {K_min, K_mid, K_max}
(e.g., {2, 4, 6}) based on draft entropy concentration, evidence n-gram continuity,
and preceding speculative acceptance history.

Mathematical Note:
Leviathan et al. (ICML 2023, Theorem 1) guarantees exact distribution preservation
for any round-specific draft length K(t) >= 1, because exact residual verification
independently ensures that every accepted and replacement token matches the target distribution.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import torch
from spade.adaptive.evidence_indexer import EvidenceIndexer


@dataclass
class DynamicKConfig:
    """Configuration for entropy and evidence-gated draft length adaptation."""
    min_k: int = 2
    default_k: int = 4
    max_k: int = 6
    entropy_low: float = 1.2       # Entropy threshold below which generation is considered high-certainty -> K=6
    entropy_high: float = 2.8      # Entropy threshold above which generation is high-divergence risk -> K=2
    min_evidence_order: int = 2     # Minimum matching n-gram order to indicate active evidence passage continuation
    evidence_mass_threshold: float = 0.25 # Probability mass on evidence candidate set to trigger expansion
    use_history: bool = True       # Whether to throttle K after an immediate rejection in the previous round


class DynamicKController:
    """Determines the optimal draft length K(t) for each speculative round."""

    def __init__(self, config: DynamicKConfig | None = None):
        self.config = config or DynamicKConfig()
        self.last_accepted: int | None = None
        self.last_proposed: int | None = None
        self.decision_history: list[dict] = []

    def compute_entropy(self, logits: torch.Tensor) -> float:
        """Compute Shannon entropy in nats from unnormalized logits."""
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        return float(-torch.sum(probs * log_probs).item())

    def update_history(self, accepted_tokens: int, proposed_tokens: int) -> None:
        """Record the outcome of the completed round to inform the next round."""
        self.last_accepted = accepted_tokens
        self.last_proposed = proposed_tokens

    def determine_k(
        self,
        draft_logits: torch.Tensor,
        evidence_indexer: EvidenceIndexer | None,
        context_tokens: list[int],
    ) -> tuple[int, str]:
        """Select draft length K(t) in {min_k, default_k, max_k} and state reason.

        Args:
            draft_logits: 1D tensor [vocab_size] of next-token logits from draft model.
            evidence_indexer: Pre-indexed evidence token continuation dictionary.
            context_tokens: Tokens generated up to the start of this round.

        Returns:
            Tuple of (chosen_k, reason_string).
        """
        # 1. Compute entropy of the draft model's immediate proposal
        entropy = self.compute_entropy(draft_logits)

        # 2. Check evidence n-gram match and mass
        has_evidence_continuation = False
        evidence_matched_order = 0
        cand_mass = 0.0

        if evidence_indexer is not None and context_tokens:
            candidates, matched_order = evidence_indexer.get_candidate_tokens(context_tokens)
            evidence_matched_order = matched_order
            if candidates and matched_order >= self.config.min_evidence_order:
                probs = torch.softmax(draft_logits, dim=-1)
                cand_tensor = torch.tensor(list(candidates), dtype=torch.long, device=draft_logits.device)
                cand_mass = float(probs[cand_tensor].sum().item())
                if cand_mass >= self.config.evidence_mass_threshold:
                    has_evidence_continuation = True

        # 3. Check historical acceptance from previous round
        had_immediate_rejection = False
        had_full_acceptance = False
        if self.config.use_history and self.last_accepted is not None and self.last_proposed is not None:
            if self.last_proposed >= 2 and self.last_accepted <= 1:
                had_immediate_rejection = True
            elif self.last_accepted == self.last_proposed and self.last_proposed > 0:
                had_full_acceptance = True

        # Policy decision:
        # A. Expansion criteria -> max_k
        if entropy < self.config.entropy_low:
            chosen_k = self.config.max_k
            reason = f"low_entropy ({entropy:.2f} < {self.config.entropy_low})"
        elif has_evidence_continuation:
            chosen_k = self.config.max_k
            reason = f"evidence_continuation (order={evidence_matched_order}, mass={cand_mass:.2f})"
        elif had_full_acceptance and entropy <= self.config.entropy_high:
            chosen_k = self.config.max_k
            reason = "consecutive_full_acceptance"

        # B. Contraction criteria -> min_k
        elif entropy > self.config.entropy_high:
            chosen_k = self.config.min_k
            reason = f"high_entropy ({entropy:.2f} > {self.config.entropy_high})"
        elif had_immediate_rejection and not has_evidence_continuation:
            chosen_k = self.config.min_k
            reason = "recent_immediate_rejection"

        # C. Default moderate -> default_k
        else:
            chosen_k = self.config.default_k
            reason = "moderate_uncertainty"

        self.decision_history.append({
            "chosen_k": chosen_k,
            "entropy": entropy,
            "reason": reason,
            "has_evidence": has_evidence_continuation,
            "evidence_order": evidence_matched_order,
            "evidence_mass": cand_mass,
        })
        return chosen_k, reason
