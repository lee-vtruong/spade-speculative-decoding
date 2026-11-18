"""Orchestrate Evidence Logit Biasing Sensitivity Study on the fixed tuning split.

Sweeps beta_0 in {0.0, 0.25, 0.5, 1.0, 1.5, 2.0} at fixed K=4 across two variants:
  1. static:  Static evidence biasing without entropy gating (--no-entropy-gating)
  2. entropy: Entropy-gated evidence biasing (adaptive_entropy=True)

Keep all other decoding hyperparameters identical across every run.
Intended to run on the fixed exploratory tuning split (e.g. dev-oracle-pilot100.jsonl),
NEVER on the held-out final evaluation split.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_BETAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
DEFAULT_VARIANTS = ["static", "entropy"]


def format_beta_tag(beta0: float) -> str:
    return f"b{beta0:.2f}".replace(".", "p")


def build_sensitivity_experiments(
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
    betas: list[float],
    variants: list[str],
    order_decay: float,
    entropy_threshold: float,
    max_boost_cap: float,
) -> list[dict]:
    experiments: list[dict] = []
    for variant in variants:
        if variant not in {"static", "entropy"}:
            raise ValueError(f"Unknown sensitivity variant: {variant}")
        for beta0 in betas:
            if beta0 < 0.0:
                raise ValueError(f"beta0 must be non-negative, got {beta0}")
            btag = format_beta_tag(beta0)
            run_id = f"{run_prefix}-{variant}-{btag}"
            cmd = [
                python_bin,
                "scripts/run_baselines.py",
                "--mode", "M4",
                "--input", str(benchmark_input),
                "--output-root", str(output_root),
                "--run-id", run_id,
                "--target", target,
                "--draft", draft,
                "--draft-length", str(draft_length),
                "--cache-strategy", "kv_cache",
                "--temperature", str(temperature),
                "--top-p", str(top_p),
                "--seed", str(seed),
                "--max-new-tokens", str(max_new_tokens),
                "--global-warmup", str(global_warmup),
                "--beta0", str(beta0),
                "--order-decay", str(order_decay),
                "--entropy-threshold", str(entropy_threshold),
                "--max-boost-cap", str(max_boost_cap),
            ]
            if limit is not None:
                cmd.extend(["--limit", str(limit)])
            if variant == "static":
                cmd.append("--no-entropy-gating")

            experiments.append({
                "variant": variant,
                "beta0": beta0,
                "run_id": run_id,
                "cmd": cmd,
            })
    return experiments


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("outputs/benchmarks/dev-oracle-pilot100.jsonl"),
                   help="Fixed tuning benchmark JSONL (exploratory tuning split)")
    p.add_argument("--target", default=os.environ.get("TARGET_MODEL", "Qwen/Qwen3-8B"))
    p.add_argument("--draft", default=os.environ.get("DRAFT_MODEL", "Qwen/Qwen3-0.6B"))
    p.add_argument("--output-root", type=Path, default=Path(os.environ.get("OUTPUT_ROOT", "outputs/runs")))
    p.add_argument("--run-prefix", default="sens-k4-tune100")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--draft-length", type=int, default=4, choices=[2, 4, 6, 8])
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--global-warmup", type=int, default=1)
    p.add_argument("--betas", type=float, nargs="+", default=DEFAULT_BETAS)
    p.add_argument("--variants", nargs="+", choices=["static", "entropy"], default=DEFAULT_VARIANTS)
    p.add_argument("--order-decay", type=float, default=0.5)
    p.add_argument("--entropy-threshold", type=float, default=2.0)
    p.add_argument("--max-boost-cap", type=float, default=3.0)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--resume", action="store_true", default=False,
                   help="Skip runs whose metrics.json already exists")
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="Remove existing run directories before launching")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Print planned commands without executing")
    args = p.parse_args()

    if args.resume and args.overwrite:
        p.error("Cannot combine --resume and --overwrite")

    experiments = build_sensitivity_experiments(
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
        betas=args.betas,
        variants=args.variants,
        order_decay=args.order_decay,
        entropy_threshold=args.entropy_threshold,
        max_boost_cap=args.max_boost_cap,
    )

    print(
        f"=== Sensitivity Study ({args.run_prefix}): {len(experiments)} runs "
        f"| K={args.draft_length} | betas={args.betas} | variants={args.variants} ==="
    )

    if args.dry_run:
        for i, exp in enumerate(experiments, start=1):
            print(f"[{i}/{len(experiments)}] {exp['run_id']}: {' '.join(exp['cmd'])}")
        return

    total_start = time.perf_counter()
    for i, exp in enumerate(experiments, start=1):
        run_dir = args.output_root / exp["run_id"]
        if args.resume and (run_dir / "metrics.json").exists():
            print(f"[{i}/{len(experiments)}] Skipping completed run {exp['run_id']}", flush=True)
            continue
        if run_dir.exists():
            if args.overwrite:
                print(f"  Removing existing {run_dir}...", flush=True)
                shutil.rmtree(run_dir)
            else:
                raise SystemExit(
                    f"Run directory already exists: {run_dir}. Pass --resume or --overwrite."
                )

        print(
            f"\n[{i}/{len(experiments)}] Running {exp['run_id']} "
            f"(variant={exp['variant']}, beta0={exp['beta0']})...",
            flush=True,
        )
        t0 = time.perf_counter()
        ret = subprocess.run(exp["cmd"])
        elapsed = time.perf_counter() - t0
        print(f"  Finished {exp['run_id']} in {elapsed:.1f}s (exit={ret.returncode})", flush=True)
        if ret.returncode != 0:
            sys.exit(ret.returncode)

    total_min = (time.perf_counter() - total_start) / 60.0
    print(f"\n=== Sensitivity study finished in {total_min:.2f} minutes ===", flush=True)


if __name__ == "__main__":
    main()
