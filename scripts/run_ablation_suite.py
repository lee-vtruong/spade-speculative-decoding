"""Orchestrate 5-Way SPADE Ablation Suite on the strictly disjoint held-out final evaluation split.

Ablation matrix (all with locked hyperparameters selected on the separate tuning split):
  1. B2           : Exact speculative sampling baseline (K=4, no biasing, no dynamic K)
  2. Bias-only    : Static evidence logit biasing without entropy gating (K=4, locked bias_only_beta0)
  3. Bias+Entropy : Entropy-gated evidence logit biasing (K=4, locked bias_entropy_beta0)
  4. Dynamic-only : Adaptive speculative horizon K(t) in {2,4,6} without evidence logit biasing
  5. Full-SPADE   : Biasing + Entropy Gating + Dynamic K(t) (locked full_spade_beta0)

Optional reference anchors (--include-references):
  - B0-S          : Target-only stochastic autoregressive baseline
  - HF-Assisted   : Hugging Face native assisted generation (K=4, constant schedule)

Enforces strict tuning/evaluation split disjointness before launching GPU runs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def read_claim_ids(jsonl_path: Path) -> set[str]:
    return {
        str(json.loads(line)["claim_id"])
        for line in jsonl_path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    }


def verify_disjoint_splits(eval_input: Path, tuning_input: Path) -> None:
    eval_ids = read_claim_ids(eval_input)
    tuning_ids = read_claim_ids(tuning_input)
    overlap = eval_ids & tuning_ids
    if overlap:
        raise ValueError(
            f"Split leakage error: {len(overlap)} claim_ids overlap between final eval ({eval_input}) "
            f"and tuning split ({tuning_input})! Refusing to run."
        )


def build_ablation_experiments(
    *,
    python_bin: str,
    benchmark_input: Path,
    target: str,
    draft: str,
    output_root: Path,
    run_prefix: str,
    limit: int | None,
    draft_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    global_warmup: int,
    bias_only_beta0: float,
    bias_entropy_beta0: float,
    full_spade_beta0: float,
    full_spade_adaptive_entropy: bool,
    order_decay: float,
    entropy_threshold: float,
    max_boost_cap: float,
    include_references: bool,
) -> list[dict[str, Any]]:
    common = [
        "--input", str(benchmark_input),
        "--output-root", str(output_root),
        "--target", target,
        "--cache-strategy", "kv_cache",
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--seed", str(seed),
        "--max-new-tokens", str(max_new_tokens),
        "--global-warmup", str(global_warmup),
    ]
    if limit is not None:
        common.extend(["--limit", str(limit)])

    exps: list[dict[str, Any]] = []

    if include_references:
        exps.extend([
            {
                "label": "B0-S (Target-Only)",
                "key": "b0s",
                "run_id": f"{run_prefix}-b0s",
                "cmd": [
                    python_bin, "scripts/run_baselines.py",
                    "--mode", "B0-S",
                    "--run-id", f"{run_prefix}-b0s",
                    *common,
                ],
            },
            {
                "label": f"HF-Assisted (K={draft_length})",
                "key": "hf_assisted",
                "run_id": f"{run_prefix}-hf-assisted-k{draft_length}",
                "cmd": [
                    python_bin, "scripts/run_baselines.py",
                    "--mode", "HF-Assisted",
                    "--draft", draft,
                    "--draft-length", str(draft_length),
                    "--run-id", f"{run_prefix}-hf-assisted-k{draft_length}",
                    *common,
                ],
            },
        ])

    bias_common = [
        "--order-decay", str(order_decay),
        "--entropy-threshold", str(entropy_threshold),
        "--max-boost-cap", str(max_boost_cap),
    ]

    full_cmd = [
        python_bin, "scripts/run_baselines.py",
        "--mode", "M4",
        "--draft", draft,
        "--draft-length", str(draft_length),
        "--dynamic-k",
        "--beta0", str(full_spade_beta0),
        *bias_common,
        "--run-id", f"{run_prefix}-full-spade",
        *common,
    ]
    if not full_spade_adaptive_entropy:
        full_cmd.append("--no-entropy-gating")

    exps.extend([
        {
            "label": f"B2 (K={draft_length})",
            "key": "b2",
            "run_id": f"{run_prefix}-b2-k{draft_length}",
            "cmd": [
                python_bin, "scripts/run_baselines.py",
                "--mode", "B2",
                "--draft", draft,
                "--draft-length", str(draft_length),
                "--run-id", f"{run_prefix}-b2-k{draft_length}",
                *common,
            ],
        },
        {
            "label": f"Bias-only (K={draft_length}, beta0={bias_only_beta0})",
            "key": "bias_only",
            "run_id": f"{run_prefix}-bias-only",
            "cmd": [
                python_bin, "scripts/run_baselines.py",
                "--mode", "M4",
                "--draft", draft,
                "--draft-length", str(draft_length),
                "--no-entropy-gating",
                "--beta0", str(bias_only_beta0),
                *bias_common,
                "--run-id", f"{run_prefix}-bias-only",
                *common,
            ],
        },
        {
            "label": f"Bias+Entropy (K={draft_length}, beta0={bias_entropy_beta0})",
            "key": "bias_entropy",
            "run_id": f"{run_prefix}-bias-entropy",
            "cmd": [
                python_bin, "scripts/run_baselines.py",
                "--mode", "M4",
                "--draft", draft,
                "--draft-length", str(draft_length),
                "--beta0", str(bias_entropy_beta0),
                *bias_common,
                "--run-id", f"{run_prefix}-bias-entropy",
                *common,
            ],
        },
        {
            "label": "Dynamic-only (K in {2,4,6}, no bias)",
            "key": "dynamic_only",
            "run_id": f"{run_prefix}-dynamic-only",
            "cmd": [
                python_bin, "scripts/run_baselines.py",
                "--mode", "M4",
                "--draft", draft,
                "--draft-length", str(draft_length),
                "--dynamic-k",
                "--no-biasing",
                "--run-id", f"{run_prefix}-dynamic-only",
                *common,
            ],
        },
        {
            "label": f"Full SPADE (Dyn K, beta0={full_spade_beta0})",
            "key": "full_spade",
            "run_id": f"{run_prefix}-full-spade",
            "cmd": full_cmd,
        },
    ])
    return exps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True,
                   help="Disjoint held-out final evaluation JSONL (e.g. dev-oracle-heldout400.jsonl)")
    p.add_argument("--tuning-benchmark", type=Path, default=None,
                   help="Fixed tuning JSONL to verify zero claim overlap before running")
    p.add_argument("--locked-config", type=Path, default=None,
                   help="Locked hyperparameters JSON exported by scripts/analyze_sensitivity.py")
    p.add_argument("--bias-only-beta0", type=float, default=None)
    p.add_argument("--bias-entropy-beta0", type=float, default=None)
    p.add_argument("--full-spade-beta0", type=float, default=None)
    p.add_argument("--full-spade-static", action="store_true", default=False,
                   help="Use static biasing inside Full SPADE (overrides locked adaptive_entropy)")
    p.add_argument("--order-decay", type=float, default=0.5)
    p.add_argument("--entropy-threshold", type=float, default=2.0)
    p.add_argument("--max-boost-cap", type=float, default=3.0)
    p.add_argument("--target", default=os.environ.get("TARGET_MODEL", "Qwen/Qwen3-8B"))
    p.add_argument("--draft", default=os.environ.get("DRAFT_MODEL", "Qwen/Qwen3-0.6B"))
    p.add_argument("--output-root", type=Path, default=Path(os.environ.get("OUTPUT_ROOT", "outputs/runs")))
    p.add_argument("--run-prefix", default="ablation-heldout400")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--draft-length", type=int, default=4, choices=[2, 4, 6, 8])
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--global-warmup", type=int, default=1)
    p.add_argument("--include-references", action="store_true", default=False,
                   help="Also run B0-S and HF-Assisted (K=4) reference baselines on the same held-out split")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    args = p.parse_args()

    if args.resume and args.overwrite:
        p.error("Cannot combine --resume and --overwrite")

    locked: dict[str, Any] = {}
    if args.locked_config is not None:
        locked = json.loads(args.locked_config.read_text(encoding="utf-8"))

    bias_only_beta0 = args.bias_only_beta0 if args.bias_only_beta0 is not None else locked.get("bias_only_beta0")
    bias_entropy_beta0 = args.bias_entropy_beta0 if args.bias_entropy_beta0 is not None else locked.get("bias_entropy_beta0")
    full_spade_beta0 = args.full_spade_beta0 if args.full_spade_beta0 is not None else locked.get("full_spade_beta0")

    if bias_only_beta0 is None or bias_entropy_beta0 is None or full_spade_beta0 is None:
        p.error(
            "Hyperparameters must be locked before final evaluation: provide --locked-config "
            "or all three of --bias-only-beta0, --bias-entropy-beta0, --full-spade-beta0."
        )

    full_spade_adaptive_entropy = (
        False if args.full_spade_static else bool(locked.get("full_spade_adaptive_entropy", True))
    )
    order_decay = float(locked.get("order_decay", args.order_decay))
    entropy_threshold = float(locked.get("entropy_threshold", args.entropy_threshold))
    max_boost_cap = float(locked.get("max_boost_cap", args.max_boost_cap))

    tuning_path = args.tuning_benchmark
    if tuning_path is None and locked.get("tuning_benchmark"):
        candidate = Path(locked["tuning_benchmark"])
        if candidate.exists():
            tuning_path = candidate

    if tuning_path is not None and args.input.exists() and tuning_path.exists():
        verify_disjoint_splits(args.input, tuning_path)
        print(f"Split disjointness check PASSED: 0 overlapping claims between {args.input} and {tuning_path}")

    experiments = build_ablation_experiments(
        python_bin=args.python,
        benchmark_input=args.input,
        target=args.target,
        draft=args.draft,
        output_root=args.output_root,
        run_prefix=args.run_prefix,
        limit=args.limit,
        draft_length=args.draft_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        global_warmup=args.global_warmup,
        bias_only_beta0=float(bias_only_beta0),
        bias_entropy_beta0=float(bias_entropy_beta0),
        full_spade_beta0=float(full_spade_beta0),
        full_spade_adaptive_entropy=full_spade_adaptive_entropy,
        order_decay=order_decay,
        entropy_threshold=entropy_threshold,
        max_boost_cap=max_boost_cap,
        include_references=args.include_references,
    )

    print(
        f"=== 5-Way Ablation Suite ({args.run_prefix}): {len(experiments)} runs | "
        f"bias_only_beta0={bias_only_beta0}, bias_entropy_beta0={bias_entropy_beta0}, "
        f"full_spade_beta0={full_spade_beta0} (adaptive_entropy={full_spade_adaptive_entropy}) ==="
    )

    if args.dry_run:
        for i, exp in enumerate(experiments, start=1):
            print(f"[{i}/{len(experiments)}] {exp['label']} ({exp['run_id']}): {' '.join(exp['cmd'])}")
        return

    total_start = time.perf_counter()
    for i, exp in enumerate(experiments, start=1):
        run_dir = args.output_root / exp["run_id"]
        if args.resume and (run_dir / "metrics.json").exists():
            print(f"[{i}/{len(experiments)}] Skipping completed run {exp['run_id']}", flush=True)
            continue
        if run_dir.exists():
            if args.overwrite:
                shutil.rmtree(run_dir)
            else:
                raise SystemExit(
                    f"Run directory already exists: {run_dir}. Pass --resume or --overwrite."
                )
        print(f"\n[{i}/{len(experiments)}] Running {exp['label']} ({exp['run_id']})...", flush=True)
        t0 = time.perf_counter()
        ret = subprocess.run(exp["cmd"])
        elapsed = time.perf_counter() - t0
        print(f"  Finished {exp['run_id']} in {elapsed:.1f}s (exit={ret.returncode})", flush=True)
        if ret.returncode != 0:
            sys.exit(ret.returncode)

    print(f"\n=== Ablation suite finished in {(time.perf_counter() - total_start)/60.0:.2f} minutes ===", flush=True)


if __name__ == "__main__":
    main()
