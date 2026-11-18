"""Milestone M4.2 Orchestrator: Evidence-Selective Context Speculative Decoding Pilot.

Tests the foundational asymmetric-context hypothesis with 3 clean configurations while
keeping target context (Full Evidence), target sampling policy, and claim set identical:
  1. Baseline       : Target=Full Evidence | Draft=Full Evidence
  2. Selective-TopK : Target=Full Evidence | Draft=Top-k claim-relevant evidence passages
  3. Compact        : Target=Full Evidence | Draft=Claim + compact extractive evidence (all E-IDs)

Executes BOTH:
  - Part A: Fixed-prefix teacher-forced replay measuring exact TV(p, q), JS(p, q), and 1 - TV(p, q)
  - Part B: Live KV-cache exact speculative sampling (B2, K=4) measuring draft prefill/decode latency,
            draft KV-cache memory, context selection cost, observed acceptance, committed/round, and E2E throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from spade.adaptive.context_selection import build_draft_context
from spade.analysis.distribution_audit import jensen_shannon_divergence
from spade.decoding.autoregressive import format_qwen_prompt, load_causal_lm, validate_tokenizers
from spade.decoding.sampling import expected_acceptance, processed_distribution, total_variation
from spade.paths import project_root


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _model_device(model: Any) -> torch.device:
    if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
        return model.get_input_embeddings().weight.device
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def _supports_logits_to_keep(model: Any) -> bool:
    try:
        forward = getattr(model, "forward", model.__call__)
        return "logits_to_keep" in inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False


def _tail_logits(model: Any, input_ids: torch.Tensor, g_len: int) -> torch.Tensor:
    kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": False}
    if _supports_logits_to_keep(model):
        kwargs["logits_to_keep"] = g_len
    out = model(**kwargs)
    logits = out.logits[0, -g_len:].detach().float()
    del out
    return logits


def replay_asymmetric_claim_prefixes(
    *,
    row: dict[str, Any],
    token_ids: list[int],
    target_model: Any,
    draft_model: Any,
    tokenizer: Any,
    draft_top_k: int = 2,
    compact_sentences: int = 1,
    compact_chars: int = 160,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 0.9,
) -> list[dict[str, Any]]:
    """Compute exact per-prefix TV(p, q), JS(p, q), and 1-TV(p, q) across full/selective_topk/compact draft contexts."""
    if not token_ids:
        return []

    cid = str(row["claim_id"])
    g_len = len(token_ids)
    t_dev = _model_device(target_model)
    d_dev = _model_device(draft_model)

    # Target full prompt
    ser_full = format_qwen_prompt(tokenizer, row["prompt"])
    t_prompt_ids = tokenizer(ser_full, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    p_t = len(t_prompt_ids)
    t_full_ids = t_prompt_ids + token_ids[:-1]

    with torch.inference_mode():
        t_inp = torch.tensor([t_full_ids], dtype=torch.long, device=t_dev)
        t_logits_all = _tail_logits(target_model, t_inp, g_len)
        del t_inp

    # Build draft prompts and run 1 teacher-forced draft forward pass per policy
    policy_logits: dict[str, torch.Tensor] = {}
    policy_meta: dict[str, dict[str, Any]] = {}
    for policy in ("full", "selective_topk", "compact"):
        dctx = build_draft_context(
            row,
            policy=policy,
            top_k=draft_top_k,
            compact_sentences=compact_sentences,
            compact_chars=compact_chars,
        )
        ser_d = format_qwen_prompt(tokenizer, dctx.draft_prompt)
        d_prompt_ids = tokenizer(ser_d, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        p_d = len(d_prompt_ids)
        d_full_ids = d_prompt_ids + token_ids[:-1]
        with torch.inference_mode():
            d_inp = torch.tensor([d_full_ids], dtype=torch.long, device=d_dev)
            d_logits = _tail_logits(draft_model, d_inp, g_len)
            del d_inp
        if d_logits.device != t_logits_all.device:
            d_logits = d_logits.to(t_logits_all.device)
        policy_logits[policy] = d_logits
        policy_meta[policy] = {
            "draft_prompt_tokens": p_d,
            "target_prompt_tokens": p_t,
            "selected_evidence_count": dctx.selected_evidence_count,
            "total_evidence_count": dctx.total_evidence_count,
            "context_selection_s": dctx.context_selection_s,
        }

    records: list[dict[str, Any]] = []
    for pos in range(g_len):
        p_dist = processed_distribution(t_logits_all[pos], temperature=temperature, top_k=top_k, top_p=top_p)
        pol_stats: dict[str, dict[str, Any]] = {}
        for policy in ("full", "selective_topk", "compact"):
            q_dist = processed_distribution(policy_logits[policy][pos], temperature=temperature, top_k=top_k, top_p=top_p)
            tv = total_variation(p_dist, q_dist)
            exp_acc = expected_acceptance(p_dist, q_dist)
            js = jensen_shannon_divergence(p_dist, q_dist)
            pol_stats[policy] = {
                "tv": tv,
                "js": js,
                "expected_acceptance": exp_acc,
                "draft_prompt_tokens": policy_meta[policy]["draft_prompt_tokens"],
                "context_selection_s": policy_meta[policy]["context_selection_s"],
            }
        records.append({
            "claim_id": cid,
            "position": pos,
            "realized_token_id": int(token_ids[pos]),
            "target_prompt_tokens": p_t,
            "policies": pol_stats,
        })
    del t_logits_all, policy_logits
    if t_dev.type == "cuda" and p_t >= 4096:
        torch.cuda.empty_cache()
    return records


def build_live_m4_2_experiments(
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
    draft_top_k: int,
    compact_sentences: int,
    compact_chars: int,
) -> list[dict[str, Any]]:
    common = [
        "--mode", "B2",
        "--input", str(benchmark_input),
        "--output-root", str(output_root),
        "--target", target,
        "--draft", draft,
        "--draft-length", str(draft_length),
        "--cache-strategy", "kv_cache",
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--seed", str(seed),
        "--max-new-tokens", str(max_new_tokens),
        "--global-warmup", str(global_warmup),
    ]
    if limit is not None:
        common.extend(["--limit", str(limit)])

    return [
        {
            "label": "Baseline (Full -> Full)",
            "policy": "full",
            "run_id": f"{run_prefix}-baseline-full",
            "cmd": [
                python_bin, "scripts/run_baselines.py",
                "--run-id", f"{run_prefix}-baseline-full",
                "--draft-context-policy", "full",
                *common,
            ],
        },
        {
            "label": f"Selective-TopK (Full -> Top-{draft_top_k})",
            "policy": "selective_topk",
            "run_id": f"{run_prefix}-selective-topk{draft_top_k}",
            "cmd": [
                python_bin, "scripts/run_baselines.py",
                "--run-id", f"{run_prefix}-selective-topk{draft_top_k}",
                "--draft-context-policy", "selective_topk",
                "--draft-top-k", str(draft_top_k),
                *common,
            ],
        },
        {
            "label": f"Compact (Full -> Compact {compact_sentences}s/{compact_chars}c)",
            "policy": "compact",
            "run_id": f"{run_prefix}-compact",
            "cmd": [
                python_bin, "scripts/run_baselines.py",
                "--run-id", f"{run_prefix}-compact",
                "--draft-context-policy", "compact",
                "--draft-compact-sentences", str(compact_sentences),
                "--draft-compact-chars", str(compact_chars),
                *common,
            ],
        },
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("outputs/benchmarks/dev-oracle-pilot100.jsonl"))
    p.add_argument("--source-predictions", type=Path, default=None,
                   help="Optional read-only predictions.jsonl for fixed-prefix replay (defaults to baseline live run)")
    p.add_argument("--target", default=os.environ.get("TARGET_MODEL", "Qwen/Qwen3-8B"))
    p.add_argument("--draft", default=os.environ.get("DRAFT_MODEL", "Qwen/Qwen3-0.6B"))
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--output-root", type=Path, default=Path(os.environ.get("OUTPUT_ROOT", "outputs/runs")))
    p.add_argument("--analysis-dir", type=Path, default=Path("outputs/analysis/m4_2_pilot"))
    p.add_argument("--run-prefix", default="m4-2-pilot100")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--draft-length", type=int, default=4, choices=[2, 4, 6, 8])
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--global-warmup", type=int, default=1)
    p.add_argument("--draft-top-k", type=int, default=2)
    p.add_argument("--compact-sentences", type=int, default=1)
    p.add_argument("--compact-chars", type=int, default=160)
    p.add_argument("--skip-live", action="store_true", default=False)
    p.add_argument("--skip-replay", action="store_true", default=False)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    args = p.parse_args()

    if args.resume and args.overwrite:
        p.error("Cannot combine --resume and --overwrite")

    exps = build_live_m4_2_experiments(
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
        draft_top_k=args.draft_top_k,
        compact_sentences=args.compact_sentences,
        compact_chars=args.compact_chars,
    )

    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "run_prefix": args.run_prefix,
            "input": str(args.input),
            "analysis_dir": str(args.analysis_dir),
            "live_experiments": [{"label": e["label"], "run_id": e["run_id"], "cmd": " ".join(e["cmd"])} for e in exps],
            "run_replay": not args.skip_replay,
        }, indent=2))
        return

    # Part B: Live speculative KV-cache benchmark (3 configs)
    if not args.skip_live:
        print(f"=== M4.2 Part B: Live Speculative KV-Cache Benchmark ({args.run_prefix}) ===", flush=True)
        for i, exp in enumerate(exps, start=1):
            run_dir = args.output_root / exp["run_id"]
            if args.resume and (run_dir / "metrics.json").exists():
                print(f"[{i}/{len(exps)}] Skipping completed run {exp['run_id']}", flush=True)
                continue
            if run_dir.exists():
                if args.overwrite:
                    shutil.rmtree(run_dir)
                else:
                    raise SystemExit(f"Run dir exists: {run_dir}. Pass --resume or --overwrite.")
            print(f"\n[{i}/{len(exps)}] Running {exp['label']} ({exp['run_id']})...", flush=True)
            t0 = time.perf_counter()
            ret = subprocess.run(exp["cmd"])
            print(f"  Finished {exp['run_id']} in {time.perf_counter() - t0:.1f}s (exit={ret.returncode})", flush=True)
            if ret.returncode != 0:
                sys.exit(ret.returncode)

    # Part A: Fixed-prefix asymmetric distribution replay
    if not args.skip_replay:
        replay_path = args.analysis_dir / "m4_2_prefix_divergence.jsonl"
        meta_path = args.analysis_dir / "m4_2_replay_metadata.json"
        if args.resume and replay_path.exists() and meta_path.exists():
            print(f"Skipping completed fixed-prefix replay at {replay_path}", flush=True)
            return

        source_preds = args.source_predictions or (args.output_root / exps[0]["run_id"] / "predictions.jsonl")
        if not source_preds.exists():
            raise SystemExit(f"Source predictions for replay not found: {source_preds}")

        print(f"\n=== M4.2 Part A: Fixed-Prefix Asymmetric Replay (source={source_preds}) ===", flush=True)
        bench_rows = read_jsonl(args.input)
        bench_by_id = {str(r["claim_id"]): r for r in bench_rows}
        pred_rows = read_jsonl(source_preds)
        if args.limit is not None:
            pred_rows = pred_rows[: args.limit]

        from transformers import AutoTokenizer
        target_tok = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
        draft_tok = AutoTokenizer.from_pretrained(args.draft, local_files_only=True)
        validate_tokenizers(target_tok, draft_tok)
        target_model = load_causal_lm(args.target, local_files_only=True, dtype=args.dtype)
        draft_model = load_causal_lm(args.draft, local_files_only=True, dtype=args.dtype)

        args.analysis_dir.mkdir(parents=True, exist_ok=True)
        t0_rep = time.perf_counter()
        total_prefixes = 0
        with replay_path.open("w", encoding="utf-8") as fout:
            for idx, pred in enumerate(pred_rows, start=1):
                cid = str(pred["claim_id"])
                brow = bench_by_id[cid]
                tok_ids = [int(x) for x in pred.get("token_ids", [])]
                recs = replay_asymmetric_claim_prefixes(
                    row=brow,
                    token_ids=tok_ids,
                    target_model=target_model,
                    draft_model=draft_model,
                    tokenizer=target_tok,
                    draft_top_k=args.draft_top_k,
                    compact_sentences=args.compact_sentences,
                    compact_chars=args.compact_chars,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                for rec in recs:
                    fout.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")
                total_prefixes += len(recs)
                if idx % 10 == 0 or idx == len(pred_rows):
                    print(f"  [{idx}/{len(pred_rows)}] Replayed {cid}: {len(recs)} prefixes (cum={total_prefixes}, {time.perf_counter() - t0_rep:.1f}s)", flush=True)

        git_commit = None
        git_dirty = None
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root(), text=True, stderr=subprocess.DEVNULL).strip()
            git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=project_root(), text=True, stderr=subprocess.DEVNULL).strip())
        except Exception:
            pass

        meta_path.write_text(json.dumps({
            "run_prefix": args.run_prefix,
            "benchmark": str(args.input.resolve()),
            "benchmark_sha256": sha256_file(args.input),
            "source_predictions": str(source_preds.resolve()),
            "source_predictions_sha256": sha256_file(source_preds),
            "draft_top_k": args.draft_top_k,
            "compact_sentences": args.compact_sentences,
            "compact_chars": args.compact_chars,
            "claims_replayed": len(pred_rows),
            "total_prefixes": total_prefixes,
            "elapsed_s": round(time.perf_counter() - t0_rep, 2),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Replay saved to {replay_path}", flush=True)


if __name__ == "__main__":
    main()
