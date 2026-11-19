"""Measure bypass noise floor and inter-run variance on NVIDIA RTX 5090.
Runs multiple seeds (seeds 42, 43, 44) with interleaved method execution and warm-up
to quantify hardware throughput noise floor and evaluate bypass equivalence.
"""
import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

PYTHON_BIN = "/home/stackops/miniconda3/bin/python"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="outputs/benchmarks/dev-oracle-pilot100.jsonl")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    parser.add_argument("--output-dir", type=str, default="outputs/runs/noise_floor_study")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_path = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    draft_path = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

    methods = [
        {"name": "Baseline Full", "mode": "B2", "policy": "full"},
        {"name": "SPADE (Bypass)", "mode": "B2", "policy": "target_attn_sentence_budget"},
    ]

    # Interleave runs across seeds
    run_plan = []
    for s in args.seeds:
        for m in methods:
            run_plan.append((s, m))
    random.seed(42)
    random.shuffle(run_plan)

    print(f"Total planned runs: {len(run_plan)} across seeds {args.seeds}")
    results = []

    for idx, (seed, m) in enumerate(run_plan):
        run_id = f"noise-{m['policy']}-s{seed}-run{idx}"
        run_path = out_dir / run_id
        cmd = [
            PYTHON_BIN, "scripts/run_baselines.py",
            "--mode", m["mode"],
            "--input", args.benchmark,
            "--target", target_path,
            "--draft", draft_path,
            "--draft-context-policy", m["policy"],
            "--draft-budget-chars", "10000",
            "--draft-length", "4",
            "--cache-strategy", "kv_cache",
            "--draft-runtime", "flex_compile_reduce_overhead",
            "--distribution-mode", "fp32_fast",
            "--temperature", "0.7",
            "--top-p", "0.9",
            "--max-new-tokens", "128",
            "--seed", str(seed),
            "--limit", str(args.limit),
            "--global-warmup", "3",
            "--run-id", run_id,
            "--output-root", str(out_dir),
        ]
        print(f"[{idx+1}/{len(run_plan)}] Running {m['name']} (seed={seed})...")
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True)
        dur = time.time() - t0
        print(f"Finished in {dur:.1f}s (exit {res.returncode})")
        
        metrics_file = run_path / "metrics.json"
        if metrics_file.exists():
            data = json.loads(metrics_file.read_text(encoding="utf-8"))
            thpt = data.get("end_to_end_tokens_per_second", 0.0)
            acc = data.get("acceptance_rate", 0.0)
            results.append({
                "method": m["name"],
                "seed": seed,
                "throughput": thpt,
                "acceptance": acc,
                "run_id": run_id
            })
            print(f"  -> Throughput: {thpt:.2f} tok/s, Acceptance: {acc*100:.2f}%")
        else:
            print(f"  -> ERROR: Missing metrics.json, stderr snippet:\n{res.stderr[-500:]}")

    # Summary analysis
    print("\n================ SUMMARY NOISE FLOOR ANALYSIS ================")
    full_thpts = [r["throughput"] for r in results if r["method"] == "Baseline Full"]
    bypass_thpts = [r["throughput"] for r in results if r["method"] == "SPADE (Bypass)"]
    full_accs = [r["acceptance"] for r in results if r["method"] == "Baseline Full"]
    bypass_accs = [r["acceptance"] for r in results if r["method"] == "SPADE (Bypass)"]

    if full_thpts and bypass_thpts:
        print(f"Baseline Full:   Thpt = {statistics.mean(full_thpts):.2f} +/- {statistics.stdev(full_thpts) if len(full_thpts)>1 else 0:.2f} tok/s (Min: {min(full_thpts):.2f}, Max: {max(full_thpts):.2f})")
        print(f"SPADE (Bypass):   Thpt = {statistics.mean(bypass_thpts):.2f} +/- {statistics.stdev(bypass_thpts) if len(bypass_thpts)>1 else 0:.2f} tok/s (Min: {min(bypass_thpts):.2f}, Max: {max(bypass_thpts):.2f})")
        diffs = [b - f for b, f in zip(bypass_thpts, full_thpts)]
        print(f"Throughput Delta (Bypass - Full): Mean = {statistics.mean(diffs):.2f} tok/s, Range = [{min(diffs):.2f}, {max(diffs):.2f}]")
        print(f"Baseline Full Acc: {statistics.mean(full_accs)*100:.2f}%")
        print(f"SPADE Bypass Acc:  {statistics.mean(bypass_accs)*100:.2f}%")
        
        noise_floor = statistics.stdev(full_thpts + bypass_thpts) if len(full_thpts + bypass_thpts) > 1 else 0.0
        print(f"Empirical GPU Jitter Noise Floor: +/- {noise_floor:.2f} tok/s")

    report_path = out_dir / "noise_floor_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
