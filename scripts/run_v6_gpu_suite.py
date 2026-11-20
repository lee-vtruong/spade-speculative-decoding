"""Resumable sequential GPU orchestrator for SPADE v6 ACL revision experiments on RTX 5090."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

PYTHON_BIN = "/home/stackops/miniconda3/bin/python"
TARGET_PATH = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DRAFT_PATH = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
PROGRESS_PATH = Path("outputs/analysis/v6_revisions/v6_gpu_suite_progress.json")


def _build_cmd(
    *,
    mode: str,
    input_path: str,
    output_root: str,
    run_id: str,
    max_new_tokens: int,
    seed: int,
    global_warmup: int,
    policy: str = "full",
    budget_chars: int = 10000,
    kv_budget: int = 1776,
) -> list[str]:
    cmd = [
        PYTHON_BIN, "scripts/run_baselines.py",
        "--mode", mode,
        "--input", input_path,
        "--output-root", output_root,
        "--run-id", run_id,
        "--target", TARGET_PATH,
        "--draft-length", "4",
        "--cache-strategy", "kv_cache",
        "--distribution-mode", "fp32_fast",
        "--temperature", "0.7",
        "--top-p", "0.9",
        "--max-new-tokens", str(max_new_tokens),
        "--seed", str(seed),
        "--global-warmup", str(global_warmup),
    ]
    if mode not in {"B0", "B0-S", "PLD"}:
        cmd.extend([
            "--draft", DRAFT_PATH,
            "--draft-runtime", "flex_compile_reduce_overhead",
            "--draft-context-policy", policy,
            "--draft-budget-chars", str(budget_chars),
            "--draft-kv-budget", str(kv_budget),
        ])
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blocks", nargs="+", default=["seed43", "long173", "longbench"], choices=["seed43", "long173", "longbench"])
    args = p.parse_args()

    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []

    if "seed43" in args.blocks:
        jobs.append({
            "block": "seed43",
            "output_root": "outputs/runs",
            "run_id": "b1-4-heldout300-b0s-seed43",
            "cmd": _build_cmd(
                mode="B0-S",
                input_path="outputs/benchmarks/retrieved-heldout300.jsonl",
                output_root="outputs/runs",
                run_id="b1-4-heldout300-b0s-seed43",
                max_new_tokens=384,
                seed=43,
                global_warmup=1,
            ),
        })

    if "long173" in args.blocks:
        l_in = "outputs/benchmarks/retrieved-heldout-long173.jsonl"
        l_root = "outputs/runs/v6_long173"
        l_specs = [
            ("long173-baseline-full", "B2", "full", 10000, 1776),
            ("long173-spade-b10k", "B2", "target_attn_sentence_budget", 10000, 1776),
            ("long173-spade-b2k", "B2", "target_attn_sentence_budget", 2000, 1776),
            ("long173-bm25-b10k", "B2", "sentence_budget", 10000, 1776),
            ("long173-kvsparse-1776", "B2", "kv_sparse", 10000, 1776),
            ("long173-kvsparse-512", "B2", "kv_sparse", 10000, 512),
            ("long173-pld", "PLD", "full", 10000, 1776),
            ("long173-sliding-window-b8k", "B2", "sliding_window", 8000, 1776),
            ("long173-no-evidence", "B2", "no_evidence", 0, 1776),
        ]
        for rid, m, pol, bud, kvb in l_specs:
            jobs.append({
                "block": "long173",
                "output_root": l_root,
                "run_id": rid,
                "cmd": _build_cmd(
                    mode=m, input_path=l_in, output_root=l_root, run_id=rid,
                    max_new_tokens=384, seed=42, global_warmup=3,
                    policy=pol, budget_chars=bud, kv_budget=kvb,
                ),
            })

    if "longbench" in args.blocks:
        lb_in = "outputs/benchmarks/longbench-hotpotqa-100.jsonl"
        lb_root = "outputs/runs/v6_longbench"
        lb_specs = [
            ("lb-hotpot-b0s", "B0-S", "full", 10000, 1776),
            ("lb-hotpot-baseline-full", "B2", "full", 10000, 1776),
            ("lb-hotpot-bm25-b10k", "B2", "sentence_budget", 10000, 1776),
            ("lb-hotpot-kvsparse-1776", "B2", "kv_sparse", 10000, 1776),
            ("lb-hotpot-pld", "PLD", "full", 10000, 1776),
            ("lb-hotpot-spade-b10k", "B2", "target_attn_sentence_budget", 10000, 1776),
            ("lb-hotpot-spade-b2k", "B2", "target_attn_sentence_budget", 2000, 1776),
        ]
        for rid, m, pol, bud, kvb in lb_specs:
            jobs.append({
                "block": "longbench",
                "output_root": lb_root,
                "run_id": rid,
                "cmd": _build_cmd(
                    mode=m, input_path=lb_in, output_root=lb_root, run_id=rid,
                    max_new_tokens=128, seed=42, global_warmup=3,
                    policy=pol, budget_chars=bud, kv_budget=kvb,
                ),
            })

    progress = []
    for idx, job in enumerate(jobs, start=1):
        run_dir = Path(job["output_root"]) / job["run_id"]
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            print(f"[{idx}/{len(jobs)}] SKIP {job['run_id']} (metrics.json already exists)", flush=True)
            progress.append({"run_id": job["run_id"], "status": "skipped_existing"})
            PROGRESS_PATH.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
            continue
        if run_dir.exists():
            print(f"[{idx}/{len(jobs)}] Cleaning incomplete dir {run_dir}...", flush=True)
            shutil.rmtree(run_dir)

        print(f"[{idx}/{len(jobs)}] START {job['run_id']} ({job['block']})...", flush=True)
        t0 = time.perf_counter()
        subprocess.run(job["cmd"], check=True)
        dt = time.perf_counter() - t0
        print(f"[{idx}/{len(jobs)}] DONE {job['run_id']} in {dt/60:.2f} min", flush=True)
        progress.append({"run_id": job["run_id"], "block": job["block"], "status": "done", "minutes": round(dt / 60.0, 2)})
        PROGRESS_PATH.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
