"""Exact speculative-sampling round over explicit processed distributions.

The model/cache adapter is intentionally isolated: callers supply q and p distributions
for each sequential proposal position, and commit only the accepted prefix plus correction.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .sampling import acceptance_probability, residual_distribution, sample_distribution

@dataclass
class RoundResult:
    committed: list[int]
    proposed: list[int]
    accepted: int
    rejected_position: int | None
    statuses: list[str]
    replacement: int | None
    extra_target: int | None
    full_acceptance: bool

def exact_speculative_round(proposals: list[int], q_distributions: list[torch.Tensor], p_distributions: list[torch.Tensor], *, generator: torch.Generator | None = None, eos_token_id: int | None = None, remaining: int | None = None, target_extra: torch.Tensor | None = None) -> RoundResult:
    """Apply Leviathan acceptance and residual correction to a precomputed round.

    p_distributions[i] is target distribution at proposal i; target_extra is the
    next target distribution used only after full acceptance.
    """
    if not (len(proposals)==len(q_distributions)==len(p_distributions)): raise ValueError("Proposal and distribution lengths differ")
    statuses=["discarded_after_rejection"]*len(proposals); committed=[]; rejected=None; replacement=None
    limit=len(proposals) if remaining is None else max(0,remaining)
    for i, token in enumerate(proposals):
        if i>=limit: break
        acc=acceptance_probability(p_distributions[i],q_distributions[i],token)
        draw=float(torch.rand((),device=q_distributions[i].device,generator=generator).item())
        if draw <= float(acc.item()):
            statuses[i]="accepted"; committed.append(token)
            if token==eos_token_id: return RoundResult(committed,proposals,i+1,None,statuses,None,None,False)
            continue
        statuses[i]="rejected"; rejected=i
        replacement=sample_distribution(residual_distribution(p_distributions[i],q_distributions[i]),generator)
        statuses[i]="rejected"; committed.append(replacement)
        # Any token after i is discarded, not rejected.
        return RoundResult(committed,proposals,i,rejected,statuses,replacement,None,False)
    full=(len(committed)==min(len(proposals),limit) and limit>0)
    extra=None
    if full and len(proposals)<=limit and target_extra is not None and (eos_token_id is None or not committed or committed[-1]!=eos_token_id) and (remaining is None or len(committed)<remaining):
        extra=sample_distribution(target_extra,generator); committed.append(extra)
        if len(statuses)<len(proposals): statuses.extend(["discarded_after_rejection"]*(len(proposals)-len(statuses)))
    return RoundResult(committed,proposals,len([s for s in statuses if s=="accepted"]),None,statuses,None,extra,full)

def greedy_verify(proposals: list[int], target_greedy: list[int], eos_token_id: int | None = None, max_new_tokens: int | None = None) -> list[int]:
    """Greedy verification: commit matching prefix, then target argmax at divergence."""
    out=[]
    for i, proposed in enumerate(proposals):
        if max_new_tokens is not None and len(out)>=max_new_tokens: break
        expected=target_greedy[i]
        token=proposed if proposed==expected else expected; out.append(token)
        if token==eos_token_id: break
        if proposed!=expected: break
    return out
