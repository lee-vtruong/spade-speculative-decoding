"""Benchmark Long-Context Scaling (2k -> 32k tokens) on NVIDIA RTX 5090.
Demonstrates the expanding latency / throughput gap between Full Speculative Decoding and SPADE
as context length expands.
"""
import argparse
import json
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_PATH = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DRAFT_PATH = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

def main():
    device = torch.device("cuda:0")
    print("Loading models for long-context scaling benchmark...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH)
    target = AutoModelForCausalLM.from_pretrained(TARGET_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    draft = AutoModelForCausalLM.from_pretrained(DRAFT_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()

    context_lengths = [2048, 4096, 8192, 16384, 32768]
    spade_budget_tokens = 1800  # corresponds to B=10k chars

    results = []
    print(f"\n{'Context Len S':<15} | {'Full Prefill (ms)':<18} | {'SPADE Prefill (ms)':<18} | {'Draft Full Step (ms)':<20} | {'Draft SPADE Step (ms)':<22} | {'Projected Speedup Gap':<22}")
    print("-" * 125)

    with torch.inference_mode():
        for S in context_lengths:
            # Create synthetic prompt tokens
            dummy_input = torch.randint(100, 10000, (1, S), dtype=torch.long, device=device)
            dummy_spade = torch.randint(100, 10000, (1, spade_budget_tokens), dtype=torch.long, device=device)

            # Warmup
            _ = draft(dummy_spade, use_cache=True)
            torch.cuda.synchronize()

            # Benchmark Full Draft Prefill
            t0 = time.perf_counter()
            for _ in range(3):
                out_full = draft(dummy_input, use_cache=True)
                torch.cuda.synchronize()
            draft_pref_full_ms = (time.perf_counter() - t0) / 3.0 * 1000.0

            # Benchmark SPADE Draft Prefill (clamped at B=1.8k)
            t0 = time.perf_counter()
            for _ in range(3):
                out_spade = draft(dummy_spade, use_cache=True)
                torch.cuda.synchronize()
            draft_pref_spade_ms = (time.perf_counter() - t0) / 3.0 * 1000.0

            # Benchmark Draft Decode Step (K=4 candidates)
            tok = torch.tensor([[500]], device=device)
            # Full step with large past_key_values
            t0 = time.perf_counter()
            for _ in range(10):
                _ = draft(tok, past_key_values=out_full.past_key_values, use_cache=True)
                torch.cuda.synchronize()
            step_full_ms = (time.perf_counter() - t0) / 10.0 * 1000.0

            # SPADE step with clamped past_key_values
            t0 = time.perf_counter()
            for _ in range(10):
                _ = draft(tok, past_key_values=out_spade.past_key_values, use_cache=True)
                torch.cuda.synchronize()
            step_spade_ms = (time.perf_counter() - t0) / 10.0 * 1000.0

            # Net draft latency for 100 generated tokens (25 rounds of K=4)
            # Full draft total = draft_prefill + 100 * step_full
            # SPADE draft total = draft_spade_prefill + 100 * step_spade + 0.163*100 (hook)
            full_tot_ms = draft_pref_full_ms + 100 * step_full_ms
            spade_tot_ms = draft_pref_spade_ms + 100 * step_spade_ms + 16.3
            speedup_ratio = full_tot_ms / spade_tot_ms

            row = {
                "context_tokens": S,
                "draft_prefill_full_ms": round(draft_pref_full_ms, 2),
                "draft_prefill_spade_ms": round(draft_pref_spade_ms, 2),
                "draft_step_full_ms": round(step_full_ms, 3),
                "draft_step_spade_ms": round(step_spade_ms, 3),
                "draft_overhead_ratio": round(speedup_ratio, 2),
            }
            results.append(row)
            print(f"{S:<15} | {draft_pref_full_ms:<18.2f} | {draft_pref_spade_ms:<18.2f} | {step_full_ms:<20.3f} | {step_spade_ms:<22.3f} | {speedup_ratio:<21.2f}x")

            del out_full, out_spade
            torch.cuda.empty_cache()

    out_file = Path("outputs/runs/long_context_scaling_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved long-context scaling report to {out_file}")

if __name__ == "__main__":
    main()
