"""Orchestrator for SPADE Pilot Baseline & Diagnostic Matrix (E1-E6).

Runs:
- B0: Autoregressive target greedy
- B0-S: Autoregressive target stochastic (T=0.7, p=0.9)
- B1: Speculative greedy (K=4)
- B2: Exact speculative sampling across K in [2, 4, 6] (T=0.7, p=0.9)

Followed by:
- evaluate_outputs.py for JSON parse, verdict metrics, and citation validity
- analyze_traces.py for M3 rejection breakdown, latency, and speedup plots
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

def run_cmd(cmd: list[str]) -> None:
    print(f"\n[RUNNING] {' '.join(cmd)}")
    t0 = time.perf_counter()
    res = subprocess.run(cmd, check=True)
    dt = time.perf_counter() - t0
    print(f"[COMPLETED in {dt:.2f}s]")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=Path, default=Path("outputs/benchmarks/dev-oracle-pilot100.jsonl"))
    p.add_argument("--target", default="/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218")
    p.add_argument("--draft", default="/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
    p.add_argument("--limit", type=int, default=15, help="Number of claims to evaluate (default 15)")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--cache-strategy", default="kv_cache", choices=["kv_cache", "recompute"])
    p.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/pilot_m1_m3"))
    a = p.parse_args()

    python = sys.executable
    runs_dir = Path("outputs/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    matrix = [
        {"run_id": f"pilot-b0-lim{a.limit}", "mode": "B0", "k": 4, "temp": 1.0, "top_p": 1.0},
        {"run_id": f"pilot-b0s-lim{a.limit}", "mode": "B0-S", "k": 4, "temp": 0.7, "top_p": 0.9},
        {"run_id": f"pilot-b1-k4-lim{a.limit}", "mode": "B1", "k": 4, "temp": 1.0, "top_p": 1.0},
        {"run_id": f"pilot-b2-k2-lim{a.limit}", "mode": "B2", "k": 2, "temp": 0.7, "top_p": 0.9},
        {"run_id": f"pilot-b2-k4-lim{a.limit}", "mode": "B2", "k": 4, "temp": 0.7, "top_p": 0.9},
        {"run_id": f"pilot-b2-k6-lim{a.limit}", "mode": "B2", "k": 6, "temp": 0.7, "top_p": 0.9},
    ]

    completed_runs = []

    for cfg in matrix:
        rid = cfg["run_id"]
        run_path = runs_dir / rid
        if (run_path / "metrics.json").exists():
            print(f"[SKIP] Run {rid} already exists at {run_path}")
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
        run_cmd(cmd)
        completed_runs.append(run_path)

        # Run offline evaluation on outputs
        eval_cmd = [
            python, "scripts/evaluate_outputs.py",
            str(run_path / "predictions.jsonl"),
            "--output-dir", str(run_path / "evaluation"),
        ]
        run_cmd(eval_cmd)

    # Run M3 comprehensive trace analysis
    print("\n[RUNNING] M3 Analysis over completed runs...")
    analysis_cmd = [
        python, "scripts/analyze_traces.py",
        "--runs-dir", str(runs_dir),
        "--output-dir", str(a.output_dir),
    ]
    for r in completed_runs:
        analysis_cmd.extend(["--run-id", r.name])
    run_cmd(analysis_cmd)

    print(f"\n[DONE] Pilot suite finished successfully! Results saved to {a.output_dir}")

if __name__ == "__main__":
    main()
