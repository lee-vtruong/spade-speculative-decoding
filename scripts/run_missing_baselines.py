"""Run missing baselines (No Evidence and Sliding Window) on 20 claims."""
import subprocess
import sys
from pathlib import Path

PYTHON_BIN = "/home/stackops/miniconda3/bin/python"
TARGET_PATH = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DRAFT_PATH = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

def run(policy, run_id, budget=0):
    cmd = [
        PYTHON_BIN, "scripts/run_baselines.py",
        "--mode", "B2",
        "--input", "outputs/benchmarks/retrieved-heldout300.jsonl",
        "--target", TARGET_PATH,
        "--draft", DRAFT_PATH,
        "--draft-context-policy", policy,
        "--draft-budget-chars", str(budget),
        "--draft-length", "4",
        "--cache-strategy", "kv_cache",
        "--draft-runtime", "flex_compile_reduce_overhead",
        "--distribution-mode", "fp32_fast",
        "--temperature", "0.7",
        "--top-p", "0.9",
        "--max-new-tokens", "128",
        "--seed", "42",
        "--limit", "20",
        "--global-warmup", "3",
        "--run-id", run_id,
        "--output-root", "outputs/runs/decisive_experiments",
    ]
    print(f"Executing {run_id} ({policy})...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Finished {run_id}: exit {res.returncode}")
    if res.returncode != 0:
        print(f"Error:\n{res.stderr[-500:]}")

if __name__ == "__main__":
    # Clean previous empty folders if any
    import shutil
    p1 = Path("outputs/runs/decisive_experiments/base-no-evidence")
    p2 = Path("outputs/runs/decisive_experiments/base-sliding-window")
    if p1.exists(): shutil.rmtree(p1)
    if p2.exists(): shutil.rmtree(p2)
    
    run("no_evidence", "base-no-evidence", 0)
    run("sliding_window", "base-sliding-window", 8000)
