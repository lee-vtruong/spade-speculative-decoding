"""CPU-only plumbing smoke test; GPU inference is performed on the server."""
import torch
from spade.decoding.sampling import expected_acceptance, residual_distribution
p=torch.tensor([.7,.3],dtype=torch.float64); q=torch.tensor([.4,.6],dtype=torch.float64)
assert abs(expected_acceptance(p,q)-.7)<1e-9
assert abs(float(residual_distribution(p,q).sum())-1)<1e-12
print("CPU sampling primitives OK. This does not validate model or GPU integration.")
