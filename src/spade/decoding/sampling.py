"""Numerically stable exact speculative-sampling primitives (PyTorch tensors)."""
from __future__ import annotations
import torch

def processed_distribution(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Apply temperature/top-k/nucleus transforms, then normalize in float64."""
    if temperature <= 0: raise ValueError("temperature must be > 0 for sampling")
    if not 0 < top_p <= 1: raise ValueError("top_p must be in (0, 1]")
    x=logits.detach().to(torch.float64)/temperature
    if top_k > 0:
        k=min(top_k, x.numel()); threshold=torch.topk(x,k).values[-1]; x=x.masked_fill(x < threshold, -torch.inf)
    if top_p < 1:
        sorted_logits, indices=torch.sort(x, descending=True); probs=torch.softmax(sorted_logits,dim=-1)
        # Retain the smallest descending prefix whose cumulative mass reaches p.
        remove=torch.cumsum(probs,dim=-1)-probs >= top_p
        remove[0]=False
        x=x.clone(); x[indices[remove]]=-torch.inf
    probs=torch.softmax(x,dim=-1)
    if not torch.isfinite(probs).all() or probs.sum() <= 0: raise ValueError("Could not form a valid sampling distribution")
    return probs/probs.sum()

def acceptance_probability(p: torch.Tensor, q: torch.Tensor, token: int) -> torch.Tensor:
    """Return min(1,p(token)/q(token)); q(token)==0 is an invalid proposal."""
    qx=q[token]; px=p[token]
    if qx <= 0: raise ValueError("Proposed token has zero probability under q")
    return torch.clamp(px/qx, max=1.0)

def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Normalized positive residual max(p-q,0), with no epsilon perturbation."""
    residual=torch.clamp(p.to(torch.float64)-q.to(torch.float64), min=0)
    total=residual.sum()
    if total <= 0:
        raise ValueError("Residual distribution is undefined when p=q; rejection has zero probability")
    return residual/total

def sample_distribution(probs: torch.Tensor, generator: torch.Generator | None = None) -> int:
    return int(torch.multinomial(probs, 1, generator=generator).item())

def total_variation(p: torch.Tensor, q: torch.Tensor) -> float:
    return float(0.5*torch.abs(p.to(torch.float64)-q.to(torch.float64)).sum().item())

def expected_acceptance(p: torch.Tensor, q: torch.Tensor) -> float:
    """Sum min(p,q), equal to 1-TV(p,q)."""
    return float(torch.minimum(p.to(torch.float64),q.to(torch.float64)).sum().item())
