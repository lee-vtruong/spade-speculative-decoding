"""M4.1 GPU Replay Script: Distribution-Level Evidence Biasing Audit on Fixed Prefixes.

Replays fixed teacher-forced prefixes from an existing predictions.jsonl (read-only,
e.g. unified-v2-b0s-lim100/predictions.jsonl on pilot100) through the target and draft
models in a single batched forward pass per claim, recording exact per-prefix
distributions p, q_base, q_biased, TV, JS, 1-TV, candidate mass, entropy, gating,
match length, candidate count, and micro-overhead.

Never modifies or overwrites existing unified-v2 or sensitivity study raw outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from spade.adaptive.evidence_indexer import EvidenceIndexer
from spade.analysis.distribution_audit import (
    AuditBiasSpec,
    build_default_audit_specs,
    evaluate_prefix_biasing,
)
from spade.decoding.autoregressive import format_qwen_prompt, load_causal_lm, validate_tokenizers
from spade.paths import project_root

DEFAULT_BETAS = [0.25, 0.5, 1.0, 1.5, 2.0]
DEFAULT_VARIANTS = ["static", "entropy"]


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


def replay_claim_prefixes(
    *,
    claim_id: str,
    prompt: str,
    evidence_texts: list[str],
    token_ids: list[int],
    target_model: Any,
    draft_model: Any,
    tokenizer: Any,
    specs: list[AuditBiasSpec],
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 0.9,
) -> list[dict[str, Any]]:
    """Run one teacher-forced forward pass on target and draft and audit every fixed prefix."""
    if not token_ids:
        return []

    serialized = format_qwen_prompt(tokenizer, prompt)
    prompt_ids = tokenizer(serialized, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    p_len = len(prompt_ids)
    g_len = len(token_ids)

    # full_ids[0 : p_len + g_len - 1] produces next-token logits at indices [p_len - 1 : p_len - 1 + g_len]
    full_ids = prompt_ids + token_ids[:-1]
    t_dev = _model_device(target_model)
    d_dev = _model_device(draft_model)

    with torch.inference_mode():
        t_inp = torch.tensor([full_ids], dtype=torch.long, device=t_dev)
        t_logits_all = target_model(input_ids=t_inp, use_cache=False).logits[0, p_len - 1 : p_len - 1 + g_len].detach().float()

        d_inp = torch.tensor([full_ids], dtype=torch.long, device=d_dev)
        d_logits_all = draft_model(input_ids=d_inp, use_cache=False).logits[0, p_len - 1 : p_len - 1 + g_len].detach().float()

    if d_logits_all.device != t_logits_all.device:
        d_logits_all = d_logits_all.to(t_logits_all.device)

    indexer = EvidenceIndexer(tokenizer, evidence_texts)
    records: list[dict[str, Any]] = []
    for pos in range(g_len):
        context_tokens = prompt_ids + token_ids[:pos]
        tok_id = int(token_ids[pos])
        tok_text = tokenizer.decode([tok_id])
        rec = evaluate_prefix_biasing(
            claim_id=claim_id,
            position=pos,
            realized_token_id=tok_id,
            realized_token_text=tok_text,
            target_logits=t_logits_all[pos],
            draft_logits=d_logits_all[pos],
            evidence_indexer=indexer,
            evidence_texts=evidence_texts,
            context_tokens=context_tokens,
            specs=specs,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        records.append(rec)
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=Path, default=Path("outputs/benchmarks/dev-oracle-pilot100.jsonl"),
                   help="Fixed tuning benchmark JSONL (read-only)")
    p.add_argument("--source-predictions", type=Path, default=Path("outputs/runs/unified-v2-b0s-lim100/predictions.jsonl"),
                   help="Read-only predictions.jsonl supplying fixed teacher-forced token_ids")
    p.add_argument("--target", default=os.environ.get("TARGET_MODEL", "Qwen/Qwen3-8B"))
    p.add_argument("--draft", default=os.environ.get("DRAFT_MODEL", "Qwen/Qwen3-0.6B"))
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--betas", type=float, nargs="+", default=DEFAULT_BETAS)
    p.add_argument("--variants", nargs="+", choices=["static", "entropy"], default=DEFAULT_VARIANTS)
    p.add_argument("--order-decay", type=float, default=0.5)
    p.add_argument("--entropy-threshold", type=float, default=2.0)
    p.add_argument("--max-boost-cap", type=float, default=3.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/m4_1_distribution_audit"))
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    args = p.parse_args()

    specs = build_default_audit_specs(
        betas=args.betas,
        variants=args.variants,
        order_decay=args.order_decay,
        entropy_threshold=args.entropy_threshold,
        max_boost_cap=args.max_boost_cap,
    )
    if not specs:
        p.error("No positive beta_0 values specified for biasing audit")

    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "benchmark": str(args.benchmark),
            "source_predictions": str(args.source_predictions),
            "target": args.target,
            "draft": args.draft,
            "output_dir": str(args.output_dir),
            "specs": [{"name": s.name, "variant": s.variant, "beta0": s.beta0} for s in specs],
        }, indent=2))
        return

    if not args.benchmark.exists():
        raise SystemExit(f"Benchmark file not found: {args.benchmark}")
    if not args.source_predictions.exists():
        raise SystemExit(f"Source predictions not found: {args.source_predictions}")

    records_path = args.output_dir / "prefix_audit_records.jsonl"
    meta_path = args.output_dir / "audit_metadata.json"
    if not args.overwrite and (records_path.exists() or meta_path.exists()):
        raise SystemExit(
            f"Output already exists in {args.output_dir}; refusing to overwrite (pass --overwrite)."
        )

    bench_rows = read_jsonl(args.benchmark)
    bench_by_id = {str(r["claim_id"]): r for r in bench_rows}
    pred_rows = read_jsonl(args.source_predictions)
    if args.limit is not None:
        pred_rows = pred_rows[: args.limit]
    if not pred_rows:
        raise SystemExit("No prediction records found to replay")

    from transformers import AutoTokenizer
    import transformers

    target_tok = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    draft_tok = AutoTokenizer.from_pretrained(args.draft, local_files_only=True)
    validate_tokenizers(target_tok, draft_tok)

    target_model = load_causal_lm(args.target, local_files_only=True, dtype=args.dtype)
    draft_model = load_causal_lm(args.draft, local_files_only=True, dtype=args.dtype)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    git_commit = None
    git_dirty = None
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root(), text=True, stderr=subprocess.DEVNULL).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=project_root(), text=True, stderr=subprocess.DEVNULL).strip())
    except Exception:
        pass

    t0_all = time.perf_counter()
    total_prefixes = 0
    claims_replayed = 0

    with records_path.open("w", encoding="utf-8") as fout:
        for idx, pred in enumerate(pred_rows, start=1):
            cid = str(pred["claim_id"])
            brow = bench_by_id.get(cid)
            if brow is None:
                raise KeyError(f"claim_id {cid!r} in predictions not found in benchmark {args.benchmark}")
            ev_texts = [e.get("text", "") for e in brow.get("evidence", [])]
            tok_ids = [int(x) for x in pred.get("token_ids", [])]

            claim_recs = replay_claim_prefixes(
                claim_id=cid,
                prompt=brow["prompt"],
                evidence_texts=ev_texts,
                token_ids=tok_ids,
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=target_tok,
                specs=specs,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            for rec in claim_recs:
                fout.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")
            total_prefixes += len(claim_recs)
            claims_replayed += 1
            if idx % 10 == 0 or idx == len(pred_rows):
                print(
                    f"[{idx}/{len(pred_rows)}] Replayed {cid}: {len(claim_recs)} prefixes "
                    f"(cumulative={total_prefixes}, elapsed={time.perf_counter() - t0_all:.1f}s)",
                    flush=True,
                )

    elapsed_s = time.perf_counter() - t0_all
    meta = {
        "schema_version": "m4_1_distribution_audit_v1",
        "benchmark_path": str(args.benchmark.resolve()),
        "benchmark_sha256": sha256_file(args.benchmark),
        "source_predictions_path": str(args.source_predictions.resolve()),
        "source_predictions_sha256": sha256_file(args.source_predictions),
        "target_model": args.target,
        "draft_model": args.draft,
        "dtype": args.dtype,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "order_decay": args.order_decay,
        "entropy_threshold": args.entropy_threshold,
        "max_boost_cap": args.max_boost_cap,
        "specs": [{"name": s.name, "variant": s.variant, "beta0": s.beta0} for s in specs],
        "claims_replayed": claims_replayed,
        "total_prefixes": total_prefixes,
        "elapsed_s": round(elapsed_s, 2),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Replay complete: {claims_replayed} claims, {total_prefixes} prefixes in {elapsed_s:.1f}s -> {args.output_dir}")


if __name__ == "__main__":
    main()
