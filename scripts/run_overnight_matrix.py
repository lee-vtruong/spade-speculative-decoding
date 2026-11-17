"""Overnight Comprehensive Experimental Runner for SPADE (Milestones M1-M4).

Runs the full 100-example benchmark across:
- E1: B0 (Greedy Autoregressive) & B0-S (Stochastic Autoregressive, T=0.7, p=0.9)
- E2: B1 (Speculative Greedy, K=4)
- E3/E4: B2 (Exact Speculative Sampling across K in {2, 4, 6}, T=0.7, p=0.9)
- M4: SPADE Evidence-Adaptive Speculative Sampling across K in {4, 6} (T=0.7, p=0.9)

Features:
- Checkpointing & automatic resumption (skips already completed runs)
- Automated evaluation (evaluate_outputs.py) per run
- Aggregated M3/M4 trace and rejection diagnostics (analyze_traces.py)
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

def run_cmd(cmd: list[str], log_file: Path | None = None) -> None:
    print(f"\n[RUNNING] {' '.join(cmd)}")
    t0 = time.perf_counter()
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n=== CMD: {' '.join(cmd)} at {time.ctime()} ===\n")
            f.flush()
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True)
    dt = time.perf_counter() - t0
    print(f"[COMPLETED in {dt:.2f}s]")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=Path, default=Path("outputs/benchmarks/dev-oracle-pilot100.jsonl"))
    p.add_argument("--target", default="/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218")
    p.add_argument("--draft", default="/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
    p.add_argument("--limit", type=int, default=100, help="Number of claims to evaluate (default 100)")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--cache-strategy", default="kv_cache", choices=["kv_cache", "recompute"])
    p.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/overnight_m1_m4"))
    a = p.parse_args()

    python = sys.executable
    runs_dir = Path("outputs/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = a.output_dir / "overnight_execution.log"

    matrix = [
        {"run_id": f"full-b0-lim{a.limit}", "mode": "B0", "k": 4, "temp": 1.0, "top_p": 1.0},
        {"run_id": f"full-b0s-lim{a.limit}", "mode": "B0-S", "k": 4, "temp": 0.7, "top_p": 0.9},
        {"run_id": f"full-b1-k4-lim{a.limit}", "mode": "B1", "k": 4, "temp": 1.0, "top_p": 1.0},
        {"run_id": f"full-b2-k2-lim{a.limit}", "mode": "B2", "k": 2, "temp": 0.7, "top_p": 0.9},
        {"run_id": f"full-b2-k4-lim{a.limit}", "mode": "B2", "k": 4, "temp": 0.7, "top_p": 0.9},
        {"run_id": f"full-b2-k6-lim{a.limit}", "mode": "B2", "k": 6, "temp": 0.7, "top_p": 0.9},
        {"run_id": f"full-m4-k4-lim{a.limit}", "mode": "M4", "k": 4, "temp": 0.7, "top_p": 0.9},
        {"run_id": f"full-m4-k6-lim{a.limit}", "mode": "M4", "k": 6, "temp": 0.7, "top_p": 0.9},
    ]

    completed_runs = []

    for cfg in matrix:
        rid = cfg["run_id"]
        run_path = runs_dir / rid
        if (run_path / "metrics.json").exists() and (run_path / "predictions.jsonl").exists():
            print(f"[SKIP] Run {rid} already completed at {run_path}")
            completed_runs.append(run_path)
            continue

        cmd = [
            python, "scripts/run_baselines.py",
            "--mode", cfg["mode"],
            "--input", str(a.benchmark),
            "--run-id", rid,
            "--target", a.target,
            "--draft", a.draft,
            "--draft-length", str(cfg["k"]),
            "--max-new-tokens", str(a.max_new_tokens),
            "--temperature", str(cfg["temp"]),
            "--top-p", str(cfg["top_p"]),
            "--limit", str(a.limit),
            "--cache-strategy", a.cache_strategy,
            "--seed", "42",
        ]
        try:
            run_cmd(cmd, log_file)
            completed_runs.append(run_path)

            # Evaluate output metrics
            eval_cmd = [
                python, "scripts/evaluate_outputs.py",
                str(run_path / "predictions.jsonl"),
                "--output-dir", str(run_path / "evaluation"),
            ]
            run_cmd(eval_cmd, log_file)
        except Exception as exc:
            print(f"[ERROR in {rid}]: {exc}")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n[FAILED] {rid}: {exc}\n")

    # Run aggregated M3/M4 trace analysis over all completed runs
    if completed_runs:
        print("\n[RUNNING] Aggregated M3/M4 Trace Diagnostics...")
        analysis_cmd = [
            python, "scripts/analyze_traces.py",
            *[str(r) for r in completed_runs],
            "--output-dir", str(a.output_dir),
        ]
        try:
            run_cmd(analysis_cmd, log_file)
        except Exception as exc:
            print(f"[ERROR in analysis]: {exc}")

    print(f"\n[OVERNIGHT FINISHED] All results and figures stored in {a.output_dir}")

if __name__ == "__main__":
    main()
