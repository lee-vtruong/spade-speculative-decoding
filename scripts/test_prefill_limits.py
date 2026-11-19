import torch
import time
from transformers import AutoModelForCausalLM

model_id = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")

for S in [2048, 4096, 8192, 16384, 32768]:
    x = torch.randint(0, 1000, (1, S), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            out = model(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"S={S:<6} | Time: {dt:>7.2f} ms | Peak VRAM: {mem:.2f} GB")
    except Exception as e:
        print(f"S={S:<6} | FAILED with error: {e}")
        break
