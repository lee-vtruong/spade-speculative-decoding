import time
import torch

def benchmark():
    device = torch.device("cuda")
    x = torch.randn(151643, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize()

    # Original double softmax
    t0 = time.perf_counter()
    for _ in range(100):
        probs = torch.softmax(x, dim=-1)
        log_probs = torch.log_softmax(x, dim=-1)
        h1 = float(-torch.sum(probs * log_probs).item())
    torch.cuda.synchronize()
    ms_orig = (time.perf_counter() - t0) * 10

    # Single pass log_softmax
    t0 = time.perf_counter()
    for _ in range(100):
        log_probs = torch.log_softmax(x, dim=-1)
        h2 = float(-torch.sum(torch.exp(log_probs) * log_probs).item())
    torch.cuda.synchronize()
    ms_single = (time.perf_counter() - t0) * 10

    # Without .item() synchronization in loop
    t0 = time.perf_counter()
    for _ in range(100):
        log_probs = torch.log_softmax(x, dim=-1)
        h_t = -torch.sum(torch.exp(log_probs) * log_probs)
    torch.cuda.synchronize()
    ms_no_sync = (time.perf_counter() - t0) * 10

    print(f"Original double softmax: {ms_orig:.3f} ms")
    print(f"Single pass log_softmax: {ms_single:.3f} ms")
    print(f"Async on GPU (no .item()): {ms_no_sync:.3f} ms")
    print(f"Values match: {abs(h1 - h2) < 1e-3}")

if __name__ == "__main__":
    benchmark()
