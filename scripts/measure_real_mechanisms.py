import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Microbenchmark bypass decision overhead
print("--- 1. Bypass Decision Microbenchmark ---")
evidence_texts = ["Sentence " + str(i) + " containing some factual details about the claim." for i in range(100)]
B = 10000

# Measure the python condition and branch check
t0 = time.perf_counter()
for _ in range(100000):
    total_len = sum(len(e) for e in evidence_texts)
    bypass = total_len <= B
dt = (time.perf_counter() - t0) / 100000.0 * 1e6
print(f"Bypass check (100 passages): {dt:.3f} microseconds per check ({dt/1000:.5f} ms)")

# 2. Check MagicDec real authors from arXiv
print("\n--- 2. Checking MagicDec and related works ---")
# Verified: Ranajoy Sadhukhan, Jian Chen, Vashisth Tiwari, Ruihang Lai, Avner May, Zhuoming Chen, Jinyuan Shi, Ian En-Hsu Yen, Beidi Chen, Tianqi Chen

# 3. Model Prefill Scaling Test with PyTorch SDPA (native scaled_dot_product_attention)
print("\n--- 3. Testing Prefill Scaling on Qwen3-0.6B (Draft) ---")
draft_path = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
draft_model = AutoModelForCausalLM.from_pretrained(
    draft_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa"
)

for S in [2048, 4096, 8192, 16384, 32768]:
    x = torch.randint(0, 1000, (1, S), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            out = draft_model(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Draft 0.6B | S={S:<6} | SDPA Time: {dt:>7.2f} ms | Peak VRAM: {mem:.2f} GB")
    except Exception as e:
        print(f"Draft 0.6B | S={S:<6} | FAILED: {e}")

print("\n--- 4. Testing Prefill Scaling on Qwen3-8B (Target) ---")
target_path = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
target_model = AutoModelForCausalLM.from_pretrained(
    target_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa"
)

for S in [2048, 4096, 8192, 16384, 32768]:
    x = torch.randint(0, 1000, (1, S), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            out = target_model(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Target 8B  | S={S:<6} | SDPA Time: {dt:>7.2f} ms | Peak VRAM: {mem:.2f} GB")
    except Exception as e:
        print(f"Target 8B  | S={S:<6} | FAILED: {e}")
