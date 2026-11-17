"""Unified Verification Benchmark Orchestrator for SPADE and Baselines.

Runs all 5 decoding methods sequentially under identical hardware, software,
warm-up, and decoding hyperparameters:
1. B0-S: Target-Only Stochastic Autoregressive (KV Cache)
2. HF-Assisted (K=4): Hugging Face Official Native Assisted Generation (K=4, constant schedule)
3. B2 (K=4): Baseline Exact Speculative Sampling (KV Cache with Rollback)
4. SPADE Fixed (K=4): Evidence-Adaptive Draft Biasing + Entropy Gating (Fixed K=4)
5. SPADE Dynamic (K in {2,4,6}): Complete SPADE (Biasing + Gating + Dynamic Horizon)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

PYTHON_BIN = "/home/stackops/miniconda3/bin/python"
TARGET_PATH = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DRAFT_PATH = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"


def build_experiments(
    python_bin: str,
    benchmark_input: str,
    limit: int,
    run_prefix: str,
    max_tokens: int,
    temp: float,
    top_p: float,
    seed: int,
    global_warmup: int,
) -> list[dict]:
    suffix = f"lim{limit}"
    common = [
        "--input", benchmark_input,
        "--target", TARGET_PATH,
        "--limit", str(limit),
        "--temperature", str(temp),
        "--top-p", str(top_p),
        "--seed", str(seed),
        "--max-new-tokens", str(max_tokens),
        "--global-warmup", str(global_warmup),
    ]
    return [
        {
            "name": "B0-S (Target-Only)",
            "run_id": f"{run_prefix}-b0s-{suffix}",
            "cmd": [
                python_bin, "scripts/run_baselines.py",
                "--mode", "B0-S",
                "--cache-strategy", "kv_cache",
                "--run-id", f"{run_prefix}-b0s-{suffix}",
                *common,
            ],
        },
        {
            "name": "HF-Assisted (K=4)",
            "run_id": f"{run_prefix}-hf-assisted-k4-{suffix}",
            "cmd": [
                python_bin, "scripts/run_baselines.py",
                "--mode", "HF-Assisted",
                "--draft", DRAFT_PATH,
                "--draft-length", "4",
                "--cache-strategy", "kv_cache",
                "--run-id", f"{run_prefix}-hf-assisted-k4-{suffix}",
                *common,
            ],
        },
        {
            "name": "B2 (Speculative K=4)",
            "run_id": f"{run_prefix}-b2-k4-{suffix}",
            "cmd": [
                python_bin, "scripts/run_baselines.py",
                "--mode", "B2",
                "--draft", DRAFT_PATH,
                "--draft-length", "4",
                "--cache-strategy", "kv_cache",
                "--run-id", f"{run_prefix}-b2-k4-{suffix}",
                *common,
            ],
        },
        {
            "name": "SPADE Fixed (K=4)",
            "run_id": f"{run_prefix}-spade-fixed-k4-{suffix}",
            "cmd": [
                python_bin, "scripts/run_baselines.py",
                "--mode", "M4",
                "--draft", DRAFT_PATH,
                "--draft-length", "4",
                "--cache-strategy", "kv_cache",
                "--run-id", f"{run_prefix}-spade-fixed-k4-{suffix}",
                *common,
            ],
        },
        {
            "name": "SPADE Dynamic (K in {2,4,6})",
            "run_id": f"{run_prefix}-spade-dynamic-k-{suffix}",
            "cmd": [
                python_bin, "scripts/run_baselines.py",
                "--mode", "M4",
                "--draft", DRAFT_PATH,
                "--draft-length", "4",
                "--dynamic-k",
                "--cache-strategy", "kv_cache",
                "--run-id", f"{run_prefix}-spade-dynamic-k-{suffix}",
                *common,
            ],
        },
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="outputs/benchmarks/dev-oracle-pilot100.jsonl")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--run-prefix", default="unified-v2")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--global-warmup", type=int, default=1)
    p.add_argument("--python", default=PYTHON_BIN)
    args = p.parse_args()

    experiments = build_experiments(
        python_bin=args.python,
        benchmark_input=args.input,
        limit=args.limit,
        run_prefix=args.run_prefix,
        max_tokens=args.max_new_tokens,
        temp=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        global_warmup=args.global_warmup,
    )

    print(f"=== Starting Unified Verification Benchmark ({args.run_prefix}, limit={args.limit}, warmup={args.global_warmup}) ===")
    total_start = time.perf_counter()

    for i, exp in enumerate(experiments, start=1):
        print(f"\n[{i}/{len(experiments)}] Running {exp['name']} ({exp['run_id']})...", flush=True)
        t0 = time.perf_counter()
        run_dir = Path("outputs/runs") / exp["run_id"]
        if run_dir.exists():
            print(f"  Removing existing {run_dir} for clean run...", flush=True)
            shutil.rmtree(run_dir)

        ret = subprocess.run(exp["cmd"])
        elapsed = time.perf_counter() - t0
        print(f"  Finished {exp['name']} in {elapsed:.1f}s (Exit code: {ret.returncode})", flush=True)
        if ret.returncode != 0:
            print(f"  ERROR: {exp['name']} failed!", flush=True)
            sys.exit(ret.returncode)

    total_elapsed = time.perf_counter() - total_start
    print(f"\n=== All {len(experiments)} benchmark runs finished in {total_elapsed/60:.2f} minutes ===", flush=True)


if __name__ == "__main__":
    main()
