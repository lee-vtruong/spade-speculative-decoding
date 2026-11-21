"""Resumable GPU orchestrator for SPADE v7 ACL revision experiments on RTX 5090.

Blocks:
  - kv2529:    KV-Sparse budget-matched to SPADE B=10k (~2,529 slots) on Long-173 (N=173).
  - scaling20: Interleaved E2E scaling across 2k-32k (N=20 at 2k-16k, N=10 at 32k) with CIs.
  - llama3:    Cross-family generalization on Llama-3.1-8B-Instruct / Llama-3.2-1B-Instruct
               (300 held-out AVeriTeC claims across 5 arms + eager hook exactness check).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

PYTHON_BIN = "/home/stackops/miniconda3/bin/python"
CACHE_DIR = "/home/stackops/whale/cache/huggingface/hub"
PROGRESS_PATH = Path("outputs/analysis/v6_revisions/v7_gpu_suite_progress.json")
LLAMA3_HOOK_OUT = Path("outputs/analysis/v6_revisions/hook_exactness_llama3.json")

LLAMA_TARGET_REPO = "unsloth/Meta-Llama-3.1-8B-Instruct"
LLAMA_DRAFT_REPO = "unsloth/Llama-3.2-1B-Instruct"


def _ensure_llama_snapshots() -> tuple[str, str]:
    from huggingface_hub import snapshot_download

    t_dir = snapshot_download(repo_id=LLAMA_TARGET_REPO, cache_dir=CACHE_DIR)
    d_dir = snapshot_download(repo_id=LLAMA_DRAFT_REPO, cache_dir=CACHE_DIR)
    print(f"Llama-3 snapshots ready:\n  target={t_dir}\n  draft={d_dir}", flush=True)
    return t_dir, d_dir


def _verify_llama3_hook_exactness(target_snap: str) -> None:
    if LLAMA3_HOOK_OUT.exists():
        print(f"SKIP Llama-3 hook exactness check ({LLAMA3_HOOK_OUT} already exists)", flush=True)
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from spade.adaptive.target_attention_selector import (
        TargetPrefillAttentionHook,
        resolve_layer_indices,
    )

    tok = AutoTokenizer.from_pretrained(target_snap, local_files_only=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        target_snap, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager", local_files_only=True
    ).eval()
    layers = resolve_layer_indices(len(mdl.model.layers), "upper50")
    W = 16
    samples = []
    for i, seq_len in enumerate([512, 768, 1024, 1536, 2048]):
        ids = torch.randint(100, min(30000, tok.vocab_size), (1, seq_len), device="cuda", dtype=torch.long)
        with TargetPrefillAttentionHook(mdl, layer_indices=layers, tail_window=W) as hk:
            with torch.inference_mode():
                out = mdl(input_ids=ids, use_cache=True, output_attentions=True)
            hook_imp, hk_s = hk.compute_token_importance(out.past_key_values)
        ref_layers = torch.stack(
            [out.attentions[l][0, :, -W:, :].float().mean(dim=(0, 1)) for l in layers], dim=0
        ).mean(dim=0).detach().cpu()
        diff = (hook_imp.float() - ref_layers).abs()
        x = hook_imp.float()
        y = ref_layers
        xc = x - x.mean()
        yc = y - y.mean()
        pearson = float((xc * yc).sum() / (xc.norm() * yc.norm() + 1e-12))
        cos = float(torch.nn.functional.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)).item())
        rx = torch.argsort(torch.argsort(x)).float()
        ry = torch.argsort(torch.argsort(y)).float()
        rxc = rx - rx.mean()
        ryc = ry - ry.mean()
        spearman = float((rxc * ryc).sum() / (rxc.norm() * ryc.norm() + 1e-12))
        top_h = set(torch.topk(x, min(64, seq_len)).indices.tolist())
        top_r = set(torch.topk(y, min(64, seq_len)).indices.tolist())
        samples.append({
            "sample_idx": i,
            "seq_len": seq_len,
            "max_abs_error": float(diff.max().item()),
            "mean_abs_error": float(diff.mean().item()),
            "cosine_similarity": cos,
            "pearson_r": pearson,
            "spearman_rho": spearman,
            "top64_token_overlap": len(top_h & top_r) / float(min(64, seq_len)),
            "hook_ms": hk_s * 1000.0,
        })
        del out, ids
        torch.cuda.empty_cache()

    rep = {
        "model": LLAMA_TARGET_REPO,
        "dtype": "torch.bfloat16",
        "n_layers": len(mdl.model.layers),
        "hooked_layers": len(layers),
        "tail_window": W,
        "samples": samples,
        "mean_max_abs_error": sum(s["max_abs_error"] for s in samples) / len(samples),
        "worst_max_abs_error": max(s["max_abs_error"] for s in samples),
        "mean_abs_error": sum(s["mean_abs_error"] for s in samples) / len(samples),
        "mean_cosine_similarity": sum(s["cosine_similarity"] for s in samples) / len(samples),
        "mean_pearson_r": sum(s["pearson_r"] for s in samples) / len(samples),
        "min_spearman_rho": min(s["spearman_rho"] for s in samples),
        "mean_top64_overlap": sum(s["top64_token_overlap"] for s in samples) / len(samples),
    }
    LLAMA3_HOOK_OUT.parent.mkdir(parents=True, exist_ok=True)
    LLAMA3_HOOK_OUT.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=2), flush=True)
    del mdl
    gc.collect()
    torch.cuda.empty_cache()


def _verify_llama3_smoke(target_snap: str, draft_snap: str) -> None:
    if (Path("outputs/runs/v7_llama3/llama3-heldout300-b0s") / "metrics.json").exists():
        print("SKIP Llama-3 smoke test (v7_llama3/llama3-heldout300-b0s already exists)", flush=True)
        return
    shutil.rmtree("outputs/runs/v7_smoke/llama3-smoke2", ignore_errors=True)
    smoke_cmd = [
        PYTHON_BIN, "scripts/run_baselines.py",
        "--mode", "B2",
        "--input", "outputs/benchmarks/retrieved-heldout300.jsonl",
        "--output-root", "outputs/runs/v7_smoke",
        "--run-id", "llama3-smoke2",
        "--target", target_snap,
        "--draft", draft_snap,
        "--draft-length", "4",
        "--cache-strategy", "kv_cache",
        "--distribution-mode", "fp32_fast",
        "--temperature", "0.7",
        "--top-p", "0.9",
        "--max-new-tokens", "64",
        "--seed", "42",
        "--limit", "2",
        "--global-warmup", "1",
        "--draft-runtime", "flex_compile_reduce_overhead",
        "--draft-context-policy", "target_attn_sentence_budget",
        "--draft-budget-chars", "10000",
    ]
    subprocess.run(smoke_cmd, check=True)
    shutil.rmtree("outputs/runs/v7_smoke/llama3-smoke2", ignore_errors=True)
    print("Llama-3 2-claim smoke test PASSED.", flush=True)


def _build_cmd(
    *,
    mode: str,
    input_path: str,
    output_root: str,
    run_id: str,
    target_path: str,
    draft_path: str | None,
    max_new_tokens: int,
    seed: int,
    global_warmup: int,
    policy: str = "full",
    budget_chars: int = 10000,
    kv_budget: int = 1776,
    top_k: int = 2,
) -> list[str]:
    cmd = [
        PYTHON_BIN, "scripts/run_baselines.py",
        "--mode", mode,
        "--input", input_path,
        "--output-root", output_root,
        "--run-id", run_id,
        "--target", target_path,
        "--draft-length", "4",
        "--cache-strategy", "kv_cache",
        "--distribution-mode", "fp32_fast",
        "--temperature", "0.7",
        "--top-p", "0.9",
        "--max-new-tokens", str(max_new_tokens),
        "--seed", str(seed),
        "--global-warmup", str(global_warmup),
    ]
    if mode not in {"B0", "B0-S", "PLD"} and draft_path is not None:
        cmd.extend([
            "--draft", draft_path,
            "--draft-runtime", "flex_compile_reduce_overhead",
            "--draft-context-policy", policy,
            "--draft-budget-chars", str(budget_chars),
            "--draft-kv-budget", str(kv_budget),
            "--draft-top-k", str(top_k),
        ])
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blocks", nargs="+", default=["kv2529", "scaling20", "llama3"], choices=["kv2529", "streaming2529", "scaling20", "llama3"])
    args = p.parse_args()

    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    progress: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []

    qwen3_t = f"{CACHE_DIR}/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    qwen3_d = f"{CACHE_DIR}/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

    if "kv2529" in args.blocks:
        jobs.append({
            "kind": "run",
            "block": "kv2529",
            "output_root": "outputs/runs/v6_long173",
            "run_id": "long173-kvsparse-2529",
            "cmd": _build_cmd(
                mode="B2",
                input_path="outputs/benchmarks/retrieved-heldout-long173.jsonl",
                output_root="outputs/runs/v6_long173",
                run_id="long173-kvsparse-2529",
                target_path=qwen3_t,
                draft_path=qwen3_d,
                max_new_tokens=384,
                seed=42,
                global_warmup=3,
                policy="kv_sparse",
                kv_budget=2529,
            ),
        })

    if "streaming2529" in args.blocks:
        jobs.append({
            "kind": "run",
            "block": "streaming2529",
            "output_root": "outputs/runs/v6_long173",
            "run_id": "long173-streamingllm-2529",
            "cmd": _build_cmd(
                mode="B2",
                input_path="outputs/benchmarks/retrieved-heldout-long173.jsonl",
                output_root="outputs/runs/v6_long173",
                run_id="long173-streamingllm-2529",
                target_path=qwen3_t,
                draft_path=qwen3_d,
                max_new_tokens=384,
                seed=42,
                global_warmup=3,
                policy="streaming_llm",
                kv_budget=2529,
            ),
        })

    if "scaling20" in args.blocks:
        jobs.append({
            "kind": "script",
            "block": "scaling20",
            "run_id": "benchmark_e2e_scaling_20",
            "cmd": [PYTHON_BIN, "scripts/benchmark_e2e_scaling.py"],
        })

    if "llama3" in args.blocks:
        llama_t, llama_d = _ensure_llama_snapshots()
        _verify_llama3_hook_exactness(llama_t)
        _verify_llama3_smoke(llama_t, llama_d)

        l_root = "outputs/runs/v7_llama3"
        l_in = "outputs/benchmarks/retrieved-heldout300.jsonl"
        specs = [
            ("llama3-heldout300-b0s", "B0-S", "full", 10000, 2),
            ("llama3-heldout300-baseline-full", "B2", "full", 10000, 2),
            ("llama3-heldout300-bm25-top2", "B2", "selective_topk", 10000, 2),
            ("llama3-heldout300-bm25-b10k", "B2", "sentence_budget", 10000, 2),
            ("llama3-heldout300-spade-b10k", "B2", "target_attn_sentence_budget", 10000, 2),
        ]
        for rid, mode, pol, bud, tk in specs:
            jobs.append({
                "kind": "run",
                "block": "llama3",
                "output_root": l_root,
                "run_id": rid,
                "cmd": _build_cmd(
                    mode=mode,
                    input_path=l_in,
                    output_root=l_root,
                    run_id=rid,
                    target_path=llama_t,
                    draft_path=llama_d,
                    max_new_tokens=384,
                    seed=42,
                    global_warmup=3,
                    policy=pol,
                    budget_chars=bud,
                    top_k=tk,
                ),
            })

    for idx, job in enumerate(jobs, start=1):
        if job["kind"] == "run":
            run_dir = Path(job["output_root"]) / job["run_id"]
            if (run_dir / "metrics.json").exists():
                print(f"[{idx}/{len(jobs)}] SKIP {job['run_id']}", flush=True)
                progress.append({"run_id": job["run_id"], "block": job["block"], "status": "skipped_existing"})
                PROGRESS_PATH.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
                continue
            if run_dir.exists():
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
