"""Decisive Experiments Orchestrator for SPADE (Giai doan 2).

Runs:
1. New Baselines on dev-oracle-pilot100 (and heldout):
   - No Evidence (B=0): draft context has instruction + claim only.
   - Sliding Window (StreamingLLM style): initial sink + local tail up to 8k chars.
2. Budget Sweep:
   - B in {1k, 2k, 4k, 8k, 10k, 16k} under target_attn_sentence_budget.
3. Long-Context Scaling:
   - Measures prompt prefill and decode latency as context length scales (2k, 5k, 10k, 16k, 32k).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

PYTHON_BIN = "/home/stackops/miniconda3/bin/python"
TARGET_PATH = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DRAFT_PATH = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

def run_experiment(run_id: str, policy: str, budget_chars: int, input_benchmark: str, limit: int, output_root: Path, extra_args=None):
    run_dir = output_root / run_id
    if run_dir.exists() and (run_dir / "trial_metrics.jsonl").exists():
        print(f"Skipping {run_id} (already completed)")
        return parse_run_results(run_dir)

    cmd = [
        PYTHON_BIN, "scripts/run_baselines.py",
        "--mode", "B2",
        "--input", input_benchmark,
        "--target", TARGET_PATH,
        "--draft", DRAFT_PATH,
        "--draft-context-policy", policy,
        "--draft-budget-chars", str(budget_chars),
        "--draft-length", "4",
        "--cache-strategy", "kv_cache",
        "--draft-runtime", "flex_compile_reduce_overhead",
        "--distribution-mode", "fp32_fast",
        "--temperature", "0.7",
        "--top-p", "0.9",
        "--max-new-tokens", "128",
        "--seed", "42",
        "--limit", str(limit),
        "--global-warmup", "3",
        "--run-id", run_id,
        "--output-root", str(output_root),
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"Running {run_id} (policy={policy}, budget={budget_chars}, limit={limit})...")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    dur = time.time() - t0
    print(f"Finished in {dur:.1f}s (exit {res.returncode})")
    if res.returncode != 0:
        print(f"Error output snippet:\n{res.stderr[-500:]}")
        return None
    return parse_run_results(run_dir)

def parse_run_results(run_dir: Path):
    trials_path = run_dir / "trial_metrics.jsonl"
    rounds_path = run_dir / "round_traces.jsonl"
    if not trials_path.exists():
        return None
    t_rows = [json.loads(l) for l in trials_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    tot_toks = sum(x["output_tokens"] for x in t_rows)
    tot_e2e = sum(x["prompt_prefill_s"] + x.get("draft_prefill_s", 0) + x["decode_s"] + x.get("context_selection_s", 0) for x in t_rows)
    tot_dec = sum(x["decode_s"] for x in t_rows)
    mean_draft_tok = statistics.mean(x.get("draft_prompt_tokens", 0) for x in t_rows) if t_rows else 0
    mean_kv_peak_mb = statistics.mean(x.get("draft_kv_peak_bytes", 0) for x in t_rows) / (1024 * 1024) if t_rows else 0

    acc = 0.0
    if rounds_path.exists():
        r_rows = [json.loads(l) for l in rounds_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        eval_acc = sum(1 for x in r_rows if x.get("status") == "accepted")
        eval_tot = sum(1 for x in r_rows if x.get("status") in {"accepted", "rejected"})
        acc = (eval_acc / eval_tot) if eval_tot > 0 else 0.0

    return {
        "run_id": run_dir.name,
        "claims": len(t_rows),
        "e2e_thpt": tot_toks / tot_e2e if tot_e2e > 0 else 0.0,
        "dec_thpt": tot_toks / tot_dec if tot_dec > 0 else 0.0,
        "acceptance": acc,
        "mean_draft_tokens": mean_draft_tok,
        "draft_kv_peak_mb": mean_kv_peak_mb,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="outputs/benchmarks/retrieved-heldout300.jsonl")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="outputs/runs/decisive_experiments")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {}

    # 1. New Baselines
    print("\n=== STEP 1: NEW BASELINES ===")
    res_noev = run_experiment("base-no-evidence", "no_evidence", 0, args.benchmark, args.limit, out_root)
    summary["No Evidence (B=0)"] = res_noev

    res_stream = run_experiment("base-sliding-window", "sliding_window", 8000, args.benchmark, args.limit, out_root)
    summary["Sliding Window / StreamSink"] = res_stream

    # 2. Budget Sweep
    print("\n=== STEP 2: BUDGET SWEEP ===")
    budgets = [1000, 2000, 4000, 8000, 10000, 16000]
    for b in budgets:
        r_id = f"sweep-budget-{b}chars"
        res_b = run_experiment(r_id, "target_attn_sentence_budget", b, args.benchmark, args.limit, out_root)
        summary[f"SPADE B={b}"] = res_b

    # Save summary report
    report_file = out_root / "decisive_experiments_report.json"
    report_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved report to {report_file}")

    print("\n================ DECISIVE EXPERIMENT RESULTS ================")
    print(f"{'Method / Config':<30} | {'E2E tok/s':<10} | {'Dec tok/s':<10} | {'Accept %':<10} | {'Draft Tok':<10} | {'KV Peak MB':<10}")
    print("-" * 90)
    for k, v in summary.items():
        if v:
            print(f"{k:<30} | {v['e2e_thpt']:<10.2f} | {v['dec_thpt']:<10.2f} | {v['acceptance']*100:<10.2f} | {v['mean_draft_tokens']:<10.1f} | {v['draft_kv_peak_mb']:<10.1f}")

if __name__ == "__main__":
    main()
