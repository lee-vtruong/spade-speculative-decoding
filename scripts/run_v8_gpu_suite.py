"""Resumable RTX 5090 GPU experiment suite for SPADE v8 ACL revisions (points a-f).

Blocks:
  - batch:              Batched draft KV memory, max co-located concurrency on 32 GB, and
                        batched draft decode throughput (batch = 1..64 at 8k and 16k).
  - fixed_bm25:         Deterministic teacher-forced fixed-prefix 1 - TV(p, q) comparison of
                        SPADE vs BM25 Sent-B10k, Sink+Recent, and Random across Long-173 and
                        HotpotQA-100, stratified by query-evidence lexical overlap & distractors
                        (plus live-rollout lexical stratification across saved runs).
  - sliding_fix:        Re-run repaired sliding_window (B=8k chars) on Long-173 via run_baselines.py.
  - oracle_interleaved: Per-claim randomized-order interleaving of Baseline Full vs SPADE Bypass
                        across all N=393 oracle claims to eliminate sequential drift by design.
  - qwen3_ratio:        Same-family draft/target capacity ratio sweep pairing Qwen3-8B with
                        Qwen3-0.6B (1:13.3), Qwen3-1.7B (1:4.7), and Qwen3-4B (1:2.0) on N=100
                        stratified held-out claims, reporting cost ratio c and Leviathan bounds.
  - long173_interleaved: R=3 independent per-claim randomized-order passes over all N=173 long
                        claims comparing Target AR, Full-SD, Sink+Recent (B=2,529), BM25 Sent-B10k,
                        and SPADE B=10k, reporting mean +/- SD across passes and paired CIs.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

from spade.adaptive.context_selection import build_draft_context, build_draft_context_from_scores
from spade.adaptive.target_attention_selector import (
    TargetPrefillAttentionHook,
    find_evidence_and_sentence_token_spans,
    resolve_layer_indices,
    score_evidence_from_token_importance,
)
from spade.decoding.autoregressive import format_qwen_prompt
from spade.decoding.hf_assisted import QwenPair
from spade.decoding.runtime_opt import StaticDraftRunner, processed_distribution_fast

PYTHON_BIN = "/home/stackops/miniconda3/bin/python"
CACHE_DIR = "/home/stackops/whale/cache/huggingface/hub"
QWEN3_8B_SNAP = f"{CACHE_DIR}/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
QWEN3_06B_SNAP = f"{CACHE_DIR}/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

REV8_DIR = Path("outputs/analysis/v8_revisions")
HELD300_PATH = Path("outputs/benchmarks/retrieved-heldout300.jsonl")
LONG173_PATH = Path("outputs/benchmarks/retrieved-heldout-long173.jsonl")
HOTPOT100_PATH = Path("outputs/benchmarks/longbench-hotpotqa-100.jsonl")
ORACLE393_PATH = Path("outputs/benchmarks/dev-oracle-heldout400.jsonl")

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "these", "those",
    "and", "or", "but", "not", "it", "its", "he", "she", "they", "we", "you", "his", "her",
    "their", "our", "has", "have", "had", "do", "does", "did", "will", "would", "can", "could",
    "may", "might", "must", "about", "into", "over", "after", "before", "between", "under",
    "during", "through", "what", "which", "who", "whom", "whose", "where", "when", "why",
    "how", "than", "also", "more", "most", "such", "some", "any", "all", "other", "only",
    "very", "just", "there", "here", "then",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _content_tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if len(w) > 2 and w not in STOPWORDS}


def _extract_gen_record(claim_id: str, gen: Any, *, include_sel: bool = True) -> dict[str, Any]:
    rds = gen.rounds or []
    t = gen.timings or {}
    sel_s = float(t.get("context_selection_s", 0.0)) if include_sel else 0.0
    e2e_s = sel_s + float(t.get("prompt_prefill_s", 0.0)) + float(t.get("draft_prefill_s", 0.0)) + float(t.get("decode_s", 0.0))
    n_eval = sum(
        min(int(rd["proposed_token_count"]), int(rd["accepted_draft_tokens"]) + (0 if rd["full_acceptance"] else 1))
        for rd in rds
    )
    n_acc = sum(int(rd["accepted_draft_tokens"]) for rd in rds)
    n_prop = sum(int(rd["proposed_token_count"]) for rd in rds)
    d_lat_s = sum(float(rd.get("draft_latency_s", 0.0)) for rd in rds)
    v_lat_s = sum(float(rd.get("verification_latency_s", 0.0)) for rd in rds)
    return {
        "claim_id": claim_id,
        "output_tokens": len(gen.token_ids),
        "e2e_s": e2e_s,
        "decode_s": float(t.get("decode_s", 0.0)),
        "context_selection_s": sel_s,
        "draft_prefill_s": float(t.get("draft_prefill_s", 0.0)),
        "prompt_prefill_s": float(t.get("prompt_prefill_s", 0.0)),
        "draft_prompt_tokens": float(t.get("draft_prompt_tokens", 0.0)),
        "draft_kv_peak_mb": float(t.get("draft_kv_peak_bytes", 0.0)) / (1024.0 ** 2),
        "target_rounds": len(rds),
        "n_eval": n_eval,
        "n_eval_acc": n_acc,
        "n_prop": n_prop,
        "n_acc": n_acc,
        "draft_latency_s": d_lat_s,
        "verification_latency_s": v_lat_s,
    }


def _aggregate_metrics(recs: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(recs))
    tot_tok = sum(r["output_tokens"] for r in recs)
    tot_e2e = sum(r["e2e_s"] for r in recs)
    tot_dec = sum(r["decode_s"] for r in recs)
    tot_eval = sum(r["n_eval"] for r in recs)
    tot_eval_acc = sum(r["n_eval_acc"] for r in recs)
    tot_prop = sum(r["n_prop"] for r in recs)
    tot_acc = sum(r["n_acc"] for r in recs)
    tot_rnd = sum(r["target_rounds"] for r in recs)
    tot_dlat = sum(r["draft_latency_s"] for r in recs)
    tot_vlat = sum(r["verification_latency_s"] for r in recs)
    per_claim = [r["output_tokens"] / r["e2e_s"] for r in recs if r["e2e_s"] > 0]
    return {
        "n_claims": len(recs),
        "e2e_tok_s": round(tot_tok / tot_e2e, 4) if tot_e2e > 0 else 0.0,
        "per_claim_e2e_tok_s": round(sum(per_claim) / max(1, len(per_claim)), 4),
        "dec_tok_s": round(tot_tok / tot_dec, 4) if tot_dec > 0 else 0.0,
        "alpha_eval_pct": round((tot_eval_acc / tot_eval) * 100.0, 4) if tot_eval > 0 else 0.0,
        "alpha_prop_pct": round((tot_acc / tot_prop) * 100.0, 4) if tot_prop > 0 else 0.0,
        "tok_per_round": round(tot_tok / tot_rnd, 4) if tot_rnd > 0 else 0.0,
        "draft_tok": round(sum(r["draft_prompt_tokens"] for r in recs) / n, 1),
        "draft_kv_mb": round(sum(r["draft_kv_peak_mb"] for r in recs) / n, 1),
        "sel_ms_claim": round((sum(r["context_selection_s"] for r in recs) / n) * 1000.0, 2),
        "draft_step_ms": round((tot_dlat / tot_prop) * 1000.0, 4) if tot_prop > 0 else 0.0,
        "verify_rnd_ms": round((tot_vlat / tot_rnd) * 1000.0, 4) if tot_rnd > 0 else 0.0,
        "total_output_tokens": tot_tok,
        "total_e2e_s": round(tot_e2e, 4),
    }


def _pooled_val(recs: list[dict[str, Any]], metric: str) -> float:
    if metric == "e2e_tok_s":
        den = sum(r["e2e_s"] for r in recs)
        return sum(r["output_tokens"] for r in recs) / den if den > 0 else 0.0
    if metric == "dec_tok_s":
        den = sum(r["decode_s"] for r in recs)
        return sum(r["output_tokens"] for r in recs) / den if den > 0 else 0.0
    if metric == "per_claim_e2e_tok_s":
        vals = [r["output_tokens"] / r["e2e_s"] for r in recs if r["e2e_s"] > 0]
        return sum(vals) / len(vals) if vals else 0.0
    if metric == "alpha_eval_pct":
        den = sum(r["n_eval"] for r in recs)
        return (sum(r["n_eval_acc"] for r in recs) / den) * 100.0 if den > 0 else 0.0
    raise ValueError(metric)


def _paired_boot(
    base_map: dict[str, dict[str, Any]],
    treat_map: dict[str, dict[str, Any]],
    metric: str,
    *,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    common = sorted(set(base_map) & set(treat_map))
    bl = [base_map[c] for c in common]
    tl = [treat_map[c] for c in common]
    b_val = _pooled_val(bl, metric)
    t_val = _pooled_val(tl, metric)
    delta = t_val - b_val
    rng = random.Random(seed)
    n = len(common)
    boots: list[float] = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        boots.append(_pooled_val([tl[i] for i in idxs], metric) - _pooled_val([bl[i] for i in idxs], metric))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(n_boot - 1, int(0.975 * n_boot))]
    p2 = min(1.0, 2.0 * min(sum(1 for x in boots if x <= 0.0) / n_boot, sum(1 for x in boots if x >= 0.0) / n_boot))
    return {
        "n_claims": n,
        "base": round(b_val, 4),
        "treat": round(t_val, 4),
        "delta": round(delta, 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
        "p_raw": round(p2, 6),
    }


def _holm_adjust(items: list[dict[str, Any]], key: str = "p_raw") -> None:
    m = len(items)
    ordered = sorted(range(m), key=lambda i: items[i][key])
    prev = 0.0
    for rank, idx in enumerate(ordered):
        adj = min(1.0, max(prev, items[idx][key] * (m - rank)))
        items[idx]["p_holm"] = round(adj, 6)
        prev = adj


def _cache_kv_bytes(cache: StaticCache) -> int:
    total = 0
    for layer in cache.layers:
        k = getattr(layer, "keys", None)
        v = getattr(layer, "values", None)
        if k is not None:
            total += k.numel() * k.element_size()
        if v is not None:
            total += v.numel() * v.element_size()
    return total


# ---------------------------------------------------------------------------
# Block 1: Batched KV Headroom & Multi-Request Serving Throughput (Point b)
# ---------------------------------------------------------------------------

def run_batch_block() -> None:
    print("\n=== [BLOCK: batch] Multi-request KV & batched draft decode on RTX 5090 ===", flush=True)
    draft = AutoModelForCausalLM.from_pretrained(QWEN3_06B_SNAP, dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()
    cfg = draft.config
    n_kvh = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
    hd = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    supports_l2k = "logits_to_keep" in draft.forward.__code__.co_varnames if hasattr(draft.forward, "__code__") else True
    s_zero = torch.tensor(0.0, dtype=torch.bfloat16, device="cuda")
    s_neg = torch.tensor(torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16, device="cuda")

    def _probe_standalone(bs: int, kv_tok: int, max_cache_len: int) -> tuple[float, float, float, float]:
        cache = StaticCache(config=cfg, max_cache_len=max_cache_len)
        cache.early_initialization(batch_size=bs, num_heads=n_kvh, head_dim=hd, dtype=torch.bfloat16, device=torch.device("cuda"))
        kv_bytes = _cache_kv_bytes(cache)
        inp_step = torch.ones((bs, 1), dtype=torch.long, device="cuda")
        arange_k = torch.arange(max_cache_len, dtype=torch.long, device="cuda").view(1, 1, 1, -1)
        for s_idx in range(5):
            pos = kv_tok + s_idx
            cpos = torch.tensor([pos], dtype=torch.long, device="cuda")
            pids = torch.full((bs, 1), pos, dtype=torch.long, device="cuda")
            amask = torch.where(arange_k <= cpos.view(1, 1, 1, 1), s_zero, s_neg)
            kw: dict[str, Any] = {
                "input_ids": inp_step, "attention_mask": amask, "past_key_values": cache,
                "cache_position": cpos, "position_ids": pids, "use_cache": True,
            }
            if supports_l2k:
                kw["logits_to_keep"] = 1
            with torch.inference_mode():
                _ = draft(**kw)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_steps = 32
        for s_idx in range(n_steps):
            pos = kv_tok + 5 + s_idx
            cpos = torch.tensor([pos], dtype=torch.long, device="cuda")
            pids = torch.full((bs, 1), pos, dtype=torch.long, device="cuda")
            amask = torch.where(arange_k <= cpos.view(1, 1, 1, 1), s_zero, s_neg)
            kw = {
                "input_ids": inp_step, "attention_mask": amask, "past_key_values": cache,
                "cache_position": cpos, "position_ids": pids, "use_cache": True,
            }
            if supports_l2k:
                kw["logits_to_keep"] = 1
            with torch.inference_mode():
                _ = draft(**kw)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        step_ms = (dt / n_steps) * 1000.0
        agg_tok_s = (bs * n_steps) / dt
        peak_gb = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
        return kv_bytes / (1024.0 ** 3), peak_gb, step_ms, agg_tok_s

    standalone_rows = []
    policies = [("Full Context", None), ("SPADE B=10k", 2529), ("SPADE B=2k", 722)]
    for ctx_len in (8192, 16384):
        for pol_name, fixed_tok in policies:
            kv_tok = ctx_len if fixed_tok is None else fixed_tok
            max_cache_len = kv_tok + 48
            for bs in (1, 4, 8, 16, 32, 64):
                _cleanup_cuda()
                torch.cuda.reset_peak_memory_stats()
                try:
                    kv_gb, peak_gb, step_ms, agg_tok_s = _probe_standalone(bs, kv_tok, max_cache_len)
                    standalone_rows.append({
                        "context_len": ctx_len, "policy": pol_name, "kv_slots": kv_tok,
                        "max_cache_len": max_cache_len, "batch_size": bs, "oom": False,
                        "draft_kv_gb": round(kv_gb, 3),
                        "peak_vram_gb": round(peak_gb, 2),
                        "step_ms": round(step_ms, 2),
                        "draft_decode_tok_s": round(agg_tok_s, 1),
                    })
                    print(f"  [standalone L={ctx_len} {pol_name:12s} bs={bs:2d}] KV={kv_gb:.2f}GB peak={peak_gb:.2f}GB step={step_ms:.2f}ms tok/s={agg_tok_s:.1f}", flush=True)
                except torch.OutOfMemoryError:
                    standalone_rows.append({
                        "context_len": ctx_len, "policy": pol_name, "kv_slots": kv_tok,
                        "max_cache_len": max_cache_len, "batch_size": bs, "oom": True,
                    })
                    print(f"  [standalone L={ctx_len} {pol_name:12s} bs={bs:2d}] OOM", flush=True)
                    _cleanup_cuda()
                    break
                _cleanup_cuda()

    # Co-located 8B Target + 0.6B Draft concurrency probe on 32 GB RTX 5090
    target = AutoModelForCausalLM.from_pretrained(QWEN3_8B_SNAP, dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()
    t_cfg = target.config
    t_kvh = int(getattr(t_cfg, "num_key_value_heads", t_cfg.num_attention_heads))
    t_hd = int(getattr(t_cfg, "head_dim", t_cfg.hidden_size // t_cfg.num_attention_heads))
    m_zero = torch.tensor(0.0, dtype=torch.bfloat16, device="cuda")
    m_neg = torch.tensor(torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16, device="cuda")

    def _probe_coloc(bs: int, ctx_len: int, d_tok: int, t_cap: int, d_cap: int) -> tuple[float, float, float]:
        t_cache = StaticCache(config=t_cfg, max_cache_len=t_cap)
        t_cache.early_initialization(batch_size=bs, num_heads=t_kvh, head_dim=t_hd, dtype=torch.bfloat16, device=torch.device("cuda"))
        d_cache = StaticCache(config=cfg, max_cache_len=d_cap)
        d_cache.early_initialization(batch_size=bs, num_heads=n_kvh, head_dim=hd, dtype=torch.bfloat16, device=torch.device("cuda"))
        d_in = torch.ones((bs, 1), dtype=torch.long, device="cuda")
        t_in = torch.ones((bs, 5), dtype=torch.long, device="cuda")
        dcpos = torch.tensor([d_tok], dtype=torch.long, device="cuda")
        tcpos = torch.arange(ctx_len, ctx_len + 5, dtype=torch.long, device="cuda")
        d_arange = torch.arange(d_cap, dtype=torch.long, device="cuda").view(1, 1, 1, -1)
        t_arange = torch.arange(t_cap, dtype=torch.long, device="cuda").view(1, 1, 1, -1)
        d_amask = torch.where(d_arange <= dcpos.view(1, 1, 1, 1), m_zero, m_neg)
        t_amask = torch.where(t_arange <= tcpos.view(1, 1, 5, 1), m_zero, m_neg)
        with torch.inference_mode():
            _ = draft(input_ids=d_in, attention_mask=d_amask, past_key_values=d_cache, cache_position=dcpos, use_cache=True, logits_to_keep=1)
            _ = target(input_ids=t_in, attention_mask=t_amask, past_key_values=t_cache, cache_position=tcpos, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        t_kv_gb = _cache_kv_bytes(t_cache) / (1024.0 ** 3)
        d_kv_gb = _cache_kv_bytes(d_cache) / (1024.0 ** 3)
        peak_gb = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
        return t_kv_gb, d_kv_gb, peak_gb

    coloc_rows = []
    max_concurrency: dict[str, dict[str, int]] = {}
    for ctx_len in (8192, 16384):
        max_concurrency[str(ctx_len)] = {}
        t_cap = ctx_len + 48
        for pol_name, fixed_tok in policies:
            d_tok = ctx_len if fixed_tok is None else fixed_tok
            d_cap = d_tok + 48
            best_bs = 0
            for bs in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16):
                _cleanup_cuda()
                torch.cuda.reset_peak_memory_stats()
                try:
                    t_kv_gb, d_kv_gb, peak_gb = _probe_coloc(bs, ctx_len, d_tok, t_cap, d_cap)
                    best_bs = bs
                    coloc_rows.append({
                        "context_len": ctx_len, "policy": pol_name, "batch_size": bs, "oom": False,
                        "target_kv_gb": round(t_kv_gb, 3), "draft_kv_gb": round(d_kv_gb, 3),
                        "total_kv_gb": round(t_kv_gb + d_kv_gb, 3), "peak_vram_gb": round(peak_gb, 2),
                    })
                    print(f"  [coloc L={ctx_len} {pol_name:12s} bs={bs:2d}] T_KV={t_kv_gb:.2f} D_KV={d_kv_gb:.2f} peak={peak_gb:.2f}GB", flush=True)
                except torch.OutOfMemoryError:
                    coloc_rows.append({
                        "context_len": ctx_len, "policy": pol_name, "batch_size": bs, "oom": True,
                    })
                    print(f"  [coloc L={ctx_len} {pol_name:12s} bs={bs:2d}] OOM", flush=True)
                    _cleanup_cuda()
                    break
                _cleanup_cuda()
            max_concurrency[str(ctx_len)][pol_name] = best_bs

    report = {
        "target_model": "Qwen/Qwen3-8B",
        "draft_model": "Qwen/Qwen3-0.6B",
        "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 3), 2),
        "standalone_batched_decode": standalone_rows,
        "colocated_concurrency": coloc_rows,
        "max_concurrent_batch_32gb": max_concurrency,
    }
    _write_json(REV8_DIR / "batch_serving_kv.json", report)
    del target, draft
    _cleanup_cuda()


# ---------------------------------------------------------------------------
# Block 2: Deterministic Fixed-Prefix SPADE vs BM25 Lexical Stratification
# ---------------------------------------------------------------------------

def _stratify_records(
    records: list[dict[str, Any]],
    metric_key: str,
    mode: str,  # "fixed" or "live"
) -> dict[str, Any]:
    import numpy as np

    vals = [r[metric_key] for r in records]
    med = float(statistics.median(vals))
    sorted_v = sorted(vals)
    q33 = sorted_v[max(0, int(0.333 * len(sorted_v)) - 1)]
    q67 = sorted_v[min(len(sorted_v) - 1, int(0.667 * len(sorted_v)))]
    strata_defs = [
        ("Low (<=med)", [r for r in records if r[metric_key] <= med]),
        ("High (>med)", [r for r in records if r[metric_key] > med]),
        ("Bot-33%", [r for r in records if r[metric_key] <= q33]),
        ("Mid-33%", [r for r in records if q33 < r[metric_key] < q67]),
        ("Top-33%", [r for r in records if r[metric_key] >= q67]),
    ]
    out: dict[str, Any] = {"median": round(med, 4), "q33": round(q33, 4), "q67": round(q67, 4), "strata": {}}
    rng = np.random.default_rng(42)
    for label, sub in strata_defs:
        if len(sub) < 5:
            continue
        if mode == "fixed":
            arr = np.array([[r["bm25_acc_sum"], r["spade_acc_sum"], r["n_prefixes"], r["bm25_tok"], r["spade_tok"]] for r in sub], dtype=float)
            tot_n = arr[:, 2].sum()
            bm_pct = float((arr[:, 0].sum() / tot_n) * 100.0)
            sp_pct = float((arr[:, 1].sum() / tot_n) * 100.0)
            bidx = rng.integers(0, len(sub), size=(10000, len(sub)))
            bsum = arr[bidx].sum(axis=1)
            deltas = ((bsum[:, 1] - bsum[:, 0]) / np.maximum(1.0, bsum[:, 2])) * 100.0
            p2 = float(min(1.0, 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())))
            out["strata"][label] = {
                "n_claims": len(sub),
                "n_prefixes": int(tot_n),
                "bm25_expected_acc_pct": round(bm_pct, 2),
                "spade_expected_acc_pct": round(sp_pct, 2),
                "delta_pp": round(sp_pct - bm_pct, 2),
                "ci_lo": round(float(np.percentile(deltas, 2.5)), 2),
                "ci_hi": round(float(np.percentile(deltas, 97.5)), 2),
                "p_raw": round(p2, 4),
                "bm25_draft_tok": round(float(arr[:, 3].mean()), 1),
                "spade_draft_tok": round(float(arr[:, 4].mean()), 1),
            }
        else:
            arr = np.array([[r["bm_acc"], r["bm_ev"], r["sp_acc"], r["sp_ev"], r["bm_ot"], r["bm_t"], r["sp_ot"], r["sp_t"]] for r in sub], dtype=float)
            bm_a = float((arr[:, 0].sum() / max(1.0, arr[:, 1].sum())) * 100.0)
            sp_a = float((arr[:, 2].sum() / max(1.0, arr[:, 3].sum())) * 100.0)
            bm_t = float(arr[:, 4].sum() / max(1e-9, arr[:, 5].sum()))
            sp_t = float(arr[:, 6].sum() / max(1e-9, arr[:, 7].sum()))
            bidx = rng.integers(0, len(sub), size=(10000, len(sub)))
            bsum = arr[bidx].sum(axis=1)
            da = 100.0 * (bsum[:, 2] / np.maximum(1.0, bsum[:, 3]) - bsum[:, 0] / np.maximum(1.0, bsum[:, 1]))
            dt = bsum[:, 6] / np.maximum(1e-9, bsum[:, 7]) - bsum[:, 4] / np.maximum(1e-9, bsum[:, 5])
            pa = float(min(1.0, 2.0 * min((da <= 0).mean(), (da >= 0).mean())))
            pt = float(min(1.0, 2.0 * min((dt <= 0).mean(), (dt >= 0).mean())))
            out["strata"][label] = {
                "n_claims": len(sub),
                "bm25_alpha_eval_pct": round(bm_a, 2),
                "spade_alpha_eval_pct": round(sp_a, 2),
                "delta_alpha_pp": round(sp_a - bm_a, 2),
                "alpha_ci": [round(float(np.percentile(da, 2.5)), 2), round(float(np.percentile(da, 97.5)), 2)],
                "alpha_p": round(pa, 4),
                "bm25_e2e_tok_s": round(bm_t, 2),
                "spade_e2e_tok_s": round(sp_t, 2),
                "delta_thpt": round(sp_t - bm_t, 2),
                "thpt_ci": [round(float(np.percentile(dt, 2.5)), 2), round(float(np.percentile(dt, 97.5)), 2)],
                "thpt_p": round(pt, 4),
            }
    return out


def _compute_lexical_features(row: dict[str, Any]) -> dict[str, float]:
    q = _content_tokens(row.get("claim", "") or row.get("question", ""))
    evs = row.get("evidence") or []
    ev_all = _content_tokens(" ".join(str(e.get("text", "")) for e in evs))
    cov = len(q & ev_all) / max(1, len(q))
    sents = [s for e in evs for s in re.split(r"(?<=[.!?])\s+", str(e.get("text", ""))) if len(s.strip()) > 20]
    j_scores = sorted([len(q & _content_tokens(s)) / max(1, len(q | _content_tokens(s))) for s in sents], reverse=True)
    top5_j = float(statistics.mean(j_scores[:5])) if j_scores else 0.0
    max_sc = max((len(q & _content_tokens(s)) / max(1, len(q)) for s in sents), default=0.0)
    ev_chars = float(sum(len(str(e.get("text", ""))) for e in evs))
    return {
        "claim_ev_recall": round(cov, 4),
        "top5_sent_jaccard": round(top5_j, 4),
        "max_sent_q_recall": round(max_sc, 4),
        "ev_chars": ev_chars,
    }


def run_fixed_bm25_block() -> None:
    print("\n=== [BLOCK: fixed_bm25] Deterministic fixed-prefix & live SPADE vs BM25 stratification ===", flush=True)
    tok = AutoTokenizer.from_pretrained(QWEN3_8B_SNAP, local_files_only=True)
    target = AutoModelForCausalLM.from_pretrained(QWEN3_8B_SNAP, dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()
    draft = AutoModelForCausalLM.from_pretrained(QWEN3_06B_SNAP, dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()
    n_t_layers = len(target.model.layers)
    upper50 = resolve_layer_indices(n_t_layers, "upper50")

    b0s_300 = {str(r["claim_id"]): r for r in _load_jsonl(Path("outputs/runs/b1-4-heldout300-b0s/predictions.jsonl"))}
    b0s_lb = {str(r["claim_id"]): r for r in _load_jsonl(Path("outputs/runs/v6_longbench/lb-hotpot-b0s/predictions.jsonl"))}

    suites = [
        ("Long-173", _load_jsonl(LONG173_PATH), b0s_300),
        ("HotpotQA-100", _load_jsonl(HOTPOT100_PATH), b0s_lb),
    ]

    fixed_out: dict[str, Any] = {}
    for suite_name, rows, traj_map in suites:
        print(f"  Running fixed-prefix evaluation on {suite_name} (N={len(rows)})...", flush=True)
        per_claim = []
        policy_names = ["Baseline Full", "SPADE (Attn B=10k)", "BM25 Sent-B10k", "Sink+Recent (B=2,529)", "Random Sent-B10k"]
        for idx, row in enumerate(rows):
            cid = str(row["claim_id"])
            traj = traj_map[cid]
            tok_ids = list(traj["token_ids"])
            if not tok_ids:
                continue
            g_len = len(tok_ids)
            ser_full = format_qwen_prompt(tok, row["prompt"])
            enc = tok(ser_full, return_offsets_mapping=True, add_special_tokens=False)
            p_ids = list(enc["input_ids"])
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
            spans = find_evidence_and_sentence_token_spans(ser_full, list(row.get("evidence") or []), offsets)

            # 1) Capture target attention & teacher-forced p_mat
            inp_p = torch.tensor([p_ids], dtype=torch.long, device="cuda")
            with TargetPrefillAttentionHook(target, layer_indices=upper50, tail_window=16) as hook:
                with torch.inference_mode():
                    pre = target(input_ids=inp_p, use_cache=True, logits_to_keep=1)
                imp, _ = hook.compute_token_importance(pre.past_key_values)
            del inp_p, pre
            _, s_scores = score_evidence_from_token_importance(imp, spans, sentence_agg="sum")
            spade_dctx = build_draft_context_from_scores(
                row, policy="sentence_budget", sentence_scores_by_passage=s_scores, budget_chars=10000, min_sentences_per_passage=1
            )
            bm25_dctx = build_draft_context(row, policy="sentence_budget", budget_chars=10000, min_sentences_per_passage=1)
            rand_dctx = build_draft_context(row, policy="random_sentence_budget", budget_chars=10000, min_sentences_per_passage=1, seed=42 + idx)

            t_inp = torch.tensor([p_ids + tok_ids[:-1]], dtype=torch.long, device="cuda")
            with torch.inference_mode():
                t_logits = target(input_ids=t_inp, use_cache=False, logits_to_keep=g_len).logits[0].float()
            p_mat = processed_distribution_fast(t_logits, temperature=0.7, top_p=0.9, dtype=torch.float32)
            del t_inp, t_logits

            sink_ids = p_ids[:256] + p_ids[-(2529 - 256) :] if len(p_ids) > 2529 else list(p_ids)
            draft_inputs = {
                "Baseline Full": p_ids,
                "SPADE (Attn B=10k)": tok(format_qwen_prompt(tok, spade_dctx.draft_prompt), add_special_tokens=False)["input_ids"],
                "BM25 Sent-B10k": tok(format_qwen_prompt(tok, bm25_dctx.draft_prompt), add_special_tokens=False)["input_ids"],
                "Sink+Recent (B=2,529)": sink_ids,
                "Random Sent-B10k": tok(format_qwen_prompt(tok, rand_dctx.draft_prompt), add_special_tokens=False)["input_ids"],
            }
            pol_stats = {}
            for pname in policy_names:
                d_ids = draft_inputs[pname]
                d_inp = torch.tensor([d_ids + tok_ids[:-1]], dtype=torch.long, device="cuda")
                with torch.inference_mode():
                    d_logits = draft(input_ids=d_inp, use_cache=False, logits_to_keep=g_len).logits[0].float()
                q_mat = processed_distribution_fast(d_logits, temperature=0.7, top_p=0.9, dtype=torch.float32)
                acc_vec = torch.minimum(p_mat, q_mat).sum(dim=-1)
                acc_sum = float(acc_vec.sum().item())
                pol_stats[pname] = {
                    "draft_tok": len(d_ids),
                    "exp_acc_sum": acc_sum,
                    "exp_acc_pct": round((acc_sum / g_len) * 100.0, 2),
                }
                del d_inp, d_logits, q_mat

            del p_mat
            lex = _compute_lexical_features(row)
            rec = {
                "claim_id": cid,
                "n_prefixes": g_len,
                **lex,
                "policies": pol_stats,
                "bm25_acc_sum": pol_stats["BM25 Sent-B10k"]["exp_acc_sum"],
                "spade_acc_sum": pol_stats["SPADE (Attn B=10k)"]["exp_acc_sum"],
                "bm25_tok": pol_stats["BM25 Sent-B10k"]["draft_tok"],
                "spade_tok": pol_stats["SPADE (Attn B=10k)"]["draft_tok"],
            }
            per_claim.append(rec)
            if (idx + 1) % 25 == 0 or (idx + 1) == len(rows):
                print(f"    {suite_name}: {idx+1}/{len(rows)}", flush=True)
                _cleanup_cuda()

        # Global fixed-prefix summary + contrasts vs BM25
        import numpy as np
        tot_pref = sum(r["n_prefixes"] for r in per_claim)
        summary = {}
        for pname in policy_names:
            acc = sum(r["policies"][pname]["exp_acc_sum"] for r in per_claim) / tot_pref * 100.0
            dtok = statistics.mean(r["policies"][pname]["draft_tok"] for r in per_claim)
            summary[pname] = {"mean_expected_acc_pct": round(acc, 2), "mean_draft_tok": round(dtok, 1)}

        rng = np.random.default_rng(42)
        bidx = rng.integers(0, len(per_claim), size=(10000, len(per_claim)))
        contrasts_vs_bm25 = {}
        for pname in ["SPADE (Attn B=10k)", "Baseline Full", "Sink+Recent (B=2,529)", "Random Sent-B10k"]:
            arr = np.array([[r["policies"]["BM25 Sent-B10k"]["exp_acc_sum"], r["policies"][pname]["exp_acc_sum"], r["n_prefixes"]] for r in per_claim], float)
            bsum = arr[bidx].sum(axis=1)
            deltas = ((bsum[:, 1] - bsum[:, 0]) / np.maximum(1.0, bsum[:, 2])) * 100.0
            d_obs = summary[pname]["mean_expected_acc_pct"] - summary["BM25 Sent-B10k"]["mean_expected_acc_pct"]
            p2 = float(min(1.0, 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())))
            contrasts_vs_bm25[f"{pname} vs BM25 Sent-B10k"] = {
                "delta_pp": round(d_obs, 2),
                "ci_lo": round(float(np.percentile(deltas, 2.5)), 2),
                "ci_hi": round(float(np.percentile(deltas, 97.5)), 2),
                "p_raw": round(p2, 4),
            }

        strat = {
            mkey: _stratify_records(per_claim, mkey, mode="fixed")
            for mkey in ("claim_ev_recall", "top5_sent_jaccard", "max_sent_q_recall", "ev_chars")
        }
        fixed_out[suite_name] = {
            "n_claims": len(per_claim),
            "n_prefixes": tot_pref,
            "summary": summary,
            "contrasts_vs_bm25": contrasts_vs_bm25,
            "stratified_spade_vs_bm25": strat,
        }
        print(f"  {suite_name} Summary: SPADE={summary['SPADE (Attn B=10k)']['mean_expected_acc_pct']}% vs BM25={summary['BM25 Sent-B10k']['mean_expected_acc_pct']}% ({contrasts_vs_bm25['SPADE (Attn B=10k) vs BM25 Sent-B10k']})", flush=True)

    # Also compute live-rollout lexical stratification from saved predictions
    live_out = {}
    for lname, dpath, sp_p, bm_p, only_long in [
        ("AVeriTeC-Sel173 (b1-4)", HELD300_PATH, Path("outputs/runs/b1-4-heldout300-attn-sent-b10k/predictions.jsonl"), Path("outputs/runs/b1-4-heldout300-bm25-sent-b10k/predictions.jsonl"), True),
        ("Long-173 (v6)", LONG173_PATH, Path("outputs/runs/v6_long173/long173-spade-b10k/predictions.jsonl"), Path("outputs/runs/v6_long173/long173-bm25-b10k/predictions.jsonl"), False),
        ("HotpotQA-100 (v6)", HOTPOT100_PATH, Path("outputs/runs/v6_longbench/lb-hotpot-spade-b10k/predictions.jsonl"), Path("outputs/runs/v6_longbench/lb-hotpot-bm25-b10k/predictions.jsonl"), False),
    ]:
        if not (sp_p.exists() and bm_p.exists() and dpath.exists()):
            continue
        ds = {str(r["claim_id"]): r for r in _load_jsonl(dpath)}
        sp_m = {str(r["claim_id"]): r for r in _load_jsonl(sp_p)}
        bm_m = {str(r["claim_id"]): r for r in _load_jsonl(bm_p)}
        lrecs = []
        for c in sorted(set(ds) & set(sp_m) & set(bm_m)):
            lex = _compute_lexical_features(ds[c])
            if only_long and lex["ev_chars"] <= 10000:
                continue
            sr = _extract_gen_record_from_dict(sp_m[c])
            br = _extract_gen_record_from_dict(bm_m[c])
            lrecs.append({
                "claim_id": c, **lex,
                "bm_acc": br["n_eval_acc"], "bm_ev": br["n_eval"], "sp_acc": sr["n_eval_acc"], "sp_ev": sr["n_eval"],
                "bm_ot": br["output_tokens"], "bm_t": br["e2e_s"], "sp_ot": sr["output_tokens"], "sp_t": sr["e2e_s"],
            })
        live_out[lname] = {
            mkey: _stratify_records(lrecs, mkey, mode="live")
            for mkey in ("claim_ev_recall", "top5_sent_jaccard", "max_sent_q_recall", "ev_chars")
        }

    _write_json(REV8_DIR / "fixed_prefix_spade_vs_bm25.json", {"fixed_prefix": fixed_out, "live_rollouts": live_out})
    del target, draft
    _cleanup_cuda()


def _extract_gen_record_from_dict(rec: dict[str, Any]) -> dict[str, Any]:
    rds = rec.get("rounds") or []
    t = rec.get("timings") or {}
    e2e_s = float(t.get("context_selection_s", 0.0)) + float(t.get("prompt_prefill_s", 0.0)) + float(t.get("draft_prefill_s", 0.0)) + float(t.get("decode_s", 0.0))
    ev = sum(min(int(rd["proposed_token_count"]), int(rd["accepted_draft_tokens"]) + (0 if rd["full_acceptance"] else 1)) for rd in rds)
    ac = sum(int(rd["accepted_draft_tokens"]) for rd in rds)
    return {
        "n_eval_acc": ac, "n_eval": ev, "output_tokens": int(rec.get("output_tokens", 0)), "e2e_s": e2e_s,
    }


# ---------------------------------------------------------------------------
# Block 3: Repaired Sliding Window Baseline on Long-173 (Point e)
# ---------------------------------------------------------------------------

def run_sliding_fix_block() -> None:
    print("\n=== [BLOCK: sliding_fix] Re-running repaired sliding_window (B=8k) on Long-173 ===", flush=True)
    run_dir = Path("outputs/runs/v6_long173/long173-sliding-window-b8k")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cmd = [
        PYTHON_BIN, "scripts/run_baselines.py",
        "--mode", "B2",
        "--input", str(LONG173_PATH),
        "--output-root", "outputs/runs/v6_long173",
        "--run-id", "long173-sliding-window-b8k",
        "--target", QWEN3_8B_SNAP,
        "--draft", QWEN3_06B_SNAP,
        "--draft-length", "4",
        "--cache-strategy", "kv_cache",
        "--distribution-mode", "fp32_fast",
        "--temperature", "0.7",
        "--top-p", "0.9",
        "--max-new-tokens", "384",
        "--seed", "42",
        "--global-warmup", "3",
        "--draft-runtime", "flex_compile_reduce_overhead",
        "--draft-context-policy", "sliding_window",
        "--draft-budget-chars", "8000",
    ]
    subprocess.run(cmd, check=True)
    # Verify non-degenerate draft token counts
    preds = _load_jsonl(run_dir / "predictions.jsonl")
    mean_dt = statistics.mean(p["timings"]["draft_prompt_tokens"] for p in preds)
    print(f"  Repaired Sliding Window mean_draft_tok = {mean_dt:.1f} across {len(preds)} claims", flush=True)
    assert mean_dt > 1000.0, f"Unexpectedly small sliding window tokens: {mean_dt}"


# ---------------------------------------------------------------------------
# Block 4: Per-Claim Interleaved Oracle N=393 Null Control (Point f)
# ---------------------------------------------------------------------------

def run_oracle_interleaved_block() -> None:
    print("\n=== [BLOCK: oracle_interleaved] Randomized per-claim Oracle N=393 Full vs SPADE Bypass ===", flush=True)
    rows = _load_jsonl(ORACLE393_PATH)
    assert len(rows) == 393
    pair = QwenPair(QWEN3_8B_SNAP, QWEN3_06B_SNAP, local_files_only=True, dtype="bfloat16", draft_attn_implementation="flex_attention")
    max_ptok = max(len(pair.draft_tokenizer(format_qwen_prompt(pair.draft_tokenizer, r["prompt"]), add_special_tokens=False)["input_ids"]) for r in rows)
    max_cache_len = ((max_ptok + 384 + 128 + 127) // 128) * 128
    pair.prepare_draft_runner(mode="compile_reduce_overhead", max_cache_len=max_cache_len, warmup_steps=3)

    # Warmup both paths outside timing
    for gw in range(3):
        pair.generate(rows[0]["prompt"], mode="B2", max_new_tokens=32, draft_length=4, temperature=0.7, top_p=0.9, seed=900 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast")
        pair.generate(rows[0]["prompt"], mode="B2", max_new_tokens=32, draft_length=4, temperature=0.7, top_p=0.9, seed=950 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": rows[0], "budget_chars": 10000, "tail_window": 16})
    _cleanup_cuda()

    # Exact 50/50 counterbalanced arm order across the 393 claims (shuffled cleanly)
    orders = [("full", "spade")] * (len(rows) // 2) + [("spade", "full")] * (len(rows) - len(rows) // 2)
    random.Random(42).shuffle(orders)

    rec_full: dict[str, dict[str, Any]] = {}
    rec_spade: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        cid = str(row["claim_id"])
        for arm in orders[idx]:
            cfg = {"row": row, "budget_chars": 10000, "tail_window": 16} if arm == "spade" else None
            g = pair.generate(
                row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9,
                seed=42 + idx, run_id=f"v8-oracle-int-{arm}", claim_id=cid,
                cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg=cfg,
            )
            rec = _extract_gen_record(cid, g)
            if arm == "full":
                rec_full[cid] = rec
            else:
                rec_spade[cid] = rec
        if (idx + 1) % 50 == 0 or (idx + 1) == len(rows):
            print(f"  Oracle interleaved progress: {idx+1}/{len(rows)}", flush=True)

    full_sum = _aggregate_metrics(list(rec_full.values()))
    spade_sum = _aggregate_metrics(list(rec_spade.values()))
    contrasts = {
        m: _paired_boot(rec_full, rec_spade, m)
        for m in ("e2e_tok_s", "per_claim_e2e_tok_s", "dec_tok_s", "alpha_eval_pct")
    }
    report = {
        "n_claims": len(rows),
        "design": "per_claim_50_50_counterbalanced_interleaved_shared_runner",
        "baseline_full": full_sum,
        "spade_bypass": spade_sum,
        "contrasts_spade_vs_full": contrasts,
        "claims_full": list(rec_full.values()),
        "claims_spade": list(rec_spade.values()),
    }
    _write_json(REV8_DIR / "oracle393_interleaved.json", report)
    print(f"  Oracle Interleaved: Full={full_sum['e2e_tok_s']} vs SPADE={spade_sum['e2e_tok_s']} | E2E CI={contrasts['e2e_tok_s']}", flush=True)
    del pair
    _cleanup_cuda()


# ---------------------------------------------------------------------------
# Block 5: Same-Family Qwen3 Draft/Target Ratio Sweep (Point d)
# ---------------------------------------------------------------------------

def _stratified_heldout100() -> list[dict[str, Any]]:
    all_rows = _load_jsonl(HELD300_PATH)
    short_rows = [r for r in all_rows if sum(len(str(e.get("text", ""))) for e in (r.get("evidence") or [])) <= 10000]
    med_long = [r for r in all_rows if 10000 < sum(len(str(e.get("text", ""))) for e in (r.get("evidence") or [])) < 20000]
    dense_rows = [r for r in all_rows if sum(len(str(e.get("text", ""))) for e in (r.get("evidence") or [])) >= 20000]
    rng = random.Random(42)
    rng.shuffle(short_rows)
    rng.shuffle(med_long)
    rng.shuffle(dense_rows)
    # Match exact 300-claim proportions: 127/300 -> 42 short, 49/300 -> 17 med_long, 124/300 -> 41 dense (total = 100)
    picked = short_rows[:42] + med_long[:17] + dense_rows[:41]
    rng.shuffle(picked)
    return picked


def run_qwen3_ratio_block() -> None:
    from huggingface_hub import snapshot_download

    print("\n=== [BLOCK: qwen3_ratio] Same-family Qwen3-8B vs {0.6B, 1.7B, 4B} sweep (N=100) ===", flush=True)
    snap_17b = snapshot_download("Qwen/Qwen3-1.7B", cache_dir=CACHE_DIR)
    snap_4b = snapshot_download("Qwen/Qwen3-4B", cache_dir=CACHE_DIR)
    rows = _stratified_heldout100()
    tok = AutoTokenizer.from_pretrained(QWEN3_8B_SNAP, local_files_only=True)
    max_full_tok = max(len(tok(format_qwen_prompt(tok, r["prompt"]), add_special_tokens=False)["input_ids"]) for r in rows)
    full_cap = ((max_full_tok + 384 + 128 + 127) // 128) * 128
    spade_cap = 6144

    # Verify identical vocab across Qwen3 family
    t_vocab = int(json.loads((Path(QWEN3_8B_SNAP) / "config.json").read_text())["vocab_size"])

    draft_specs = [
        ("Qwen3-0.6B", QWEN3_06B_SNAP, "1:13.3"),
        ("Qwen3-1.7B", snap_17b, "1:4.7"),
        ("Qwen3-4B", snap_4b, "1:2.0"),
    ]
    ckpt_path = REV8_DIR / "qwen3_ratio_sweep.json"
    sweep_results: dict[str, Any] = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.exists() else {"n_claims": len(rows), "models": {}}

    for d_name, d_snap, ratio_str in draft_specs:
        if d_name in sweep_results["models"]:
            print(f"  SKIP {d_name} (already completed in {ckpt_path})", flush=True)
            continue
        d_cfg = json.loads((Path(d_snap) / "config.json").read_text())
        assert int(d_cfg["vocab_size"]) == t_vocab, f"Vocab mismatch for {d_name}"
        print(f"  Running {d_name} ({ratio_str}) across N={len(rows)} claims...", flush=True)
        pair = QwenPair(QWEN3_8B_SNAP, d_snap, local_files_only=True, dtype="bfloat16", draft_attn_implementation="flex_attention")
        t_params = sum(p.numel() for p in pair.target.parameters())
        d_params = sum(p.numel() for p in pair.draft.parameters())
        exact_ratio = t_params / d_params

        rec_by_arm: dict[str, dict[str, dict[str, Any]]] = {"AR": {}, "SPADE": {}, "Full": {}}
        full_error: str | None = None

        # Precompute StaticDraftRunner for SPADE (compact) & Full; for 4B check if both fit or swap across 10-claim blocks
        can_hold_both = True
        runner_spade = runner_full = None
        try:
            runner_spade = StaticDraftRunner(pair.draft, max_cache_len=spade_cap, mode="compile_reduce_overhead", device=pair._input_device(pair.draft))
            runner_spade.prepare(warmup_steps=3)
            runner_full = StaticDraftRunner(pair.draft, max_cache_len=full_cap, mode="compile_reduce_overhead", device=pair._input_device(pair.draft))
            runner_full.prepare(warmup_steps=3)
        except torch.OutOfMemoryError:
            can_hold_both = False
            del runner_full, runner_spade
            runner_full = runner_spade = None
            _cleanup_cuda()

        # Warmup AR outside timing
        for gw in range(2):
            pair.generate(rows[0]["prompt"], mode="B0-S", max_new_tokens=32, temperature=0.7, top_p=0.9, seed=700 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast")

        if can_hold_both:
            assert runner_spade is not None and runner_full is not None
            pair.draft_runner = runner_spade
            pair.generate(rows[0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=710, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": rows[0], "budget_chars": 10000, "tail_window": 16})
            pair.draft_runner = runner_full
            pair.generate(rows[0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=720, cache_strategy="kv_cache", distribution_mode="fp32_fast")
            _cleanup_cuda()

            rng_ord = random.Random(42)
            for idx, row in enumerate(rows):
                cid = str(row["claim_id"])
                arms = ["AR", "SPADE", "Full"]
                rng_ord.shuffle(arms)
                for arm in arms:
                    try:
                        if arm == "AR":
                            g = pair.generate(row["prompt"], mode="B0-S", max_new_tokens=384, temperature=0.7, top_p=0.9, seed=42 + idx, run_id=f"q3-{d_name}-ar", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                        elif arm == "SPADE":
                            pair.draft_runner = runner_spade
                            g = pair.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + idx, run_id=f"q3-{d_name}-spade", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": row, "budget_chars": 10000, "tail_window": 16})
                        else:
                            pair.draft_runner = runner_full
                            g = pair.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + idx, run_id=f"q3-{d_name}-full", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                        rec_by_arm[arm][cid] = _extract_gen_record(cid, g)
                    except torch.OutOfMemoryError as exc:
                        if arm == "Full":
                            full_error = str(exc)
                            _cleanup_cuda()
                        else:
                            raise
                if (idx + 1) % 25 == 0 or (idx + 1) == len(rows):
                    print(f"    {d_name}: {idx+1}/{len(rows)}", flush=True)
            del runner_full, runner_spade
            pair.draft_runner = None
        else:
            # Alternating 20-claim mini-blocks keeping one runner resident at a time (e.g. for Qwen3-4B)
            print(f"    {d_name}: Dual StaticCaches exceeded 32GB -> running alternating 20-claim blocks", flush=True)
            chunk = 20
            for blk_i, start in enumerate(range(0, len(rows), chunk)):
                sub = rows[start : start + chunk]
                # AR runs first per sub-claim alongside whichever SD arm runs cleanly
                for j, row in enumerate(sub):
                    idx = start + j
                    cid = str(row["claim_id"])
                    g = pair.generate(row["prompt"], mode="B0-S", max_new_tokens=384, temperature=0.7, top_p=0.9, seed=42 + idx, run_id=f"q3-{d_name}-ar", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                    rec_by_arm["AR"][cid] = _extract_gen_record(cid, g)
                order = ("SPADE", "Full") if (blk_i % 2 == 0) else ("Full", "SPADE")
                for arm in order:
                    if arm == "Full" and full_error is not None:
                        continue
                    cap = spade_cap if arm == "SPADE" else full_cap
                    try:
                        pair.prepare_draft_runner(mode="compile_reduce_overhead", max_cache_len=cap, warmup_steps=3)
                        wcfg = {"row": sub[0], "budget_chars": 10000, "tail_window": 16} if arm == "SPADE" else None
                        pair.generate(sub[0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=710, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg=wcfg)
                        _cleanup_cuda()
                        for j, row in enumerate(sub):
                            idx = start + j
                            cid = str(row["claim_id"])
                            cfg = {"row": row, "budget_chars": 10000, "tail_window": 16} if arm == "SPADE" else None
                            g = pair.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + idx, run_id=f"q3-{d_name}-{arm.lower()}", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg=cfg)
                            rec_by_arm[arm][cid] = _extract_gen_record(cid, g)
                    except torch.OutOfMemoryError as exc:
                        if arm == "Full":
                            full_error = str(exc)
                        else:
                            raise
                    finally:
                        pair.draft_runner = None
                        _cleanup_cuda()

        ar_sum = _aggregate_metrics(list(rec_by_arm["AR"].values()))
        sp_sum = _aggregate_metrics(list(rec_by_arm["SPADE"].values()))
        fl_sum = _aggregate_metrics(list(rec_by_arm["Full"].values())) if len(rec_by_arm["Full"]) == len(rows) else None

        # Compute Leviathan cost factors c_AR, c_ver, gamma_ver, and theoretical decode speedup vs AR step
        ar_step_ms = (ar_sum["total_e2e_s"] - sum(r["prompt_prefill_s"] for r in rec_by_arm["AR"].values())) / ar_sum["total_output_tokens"] * 1000.0
        def _leviathan_row(arm_sum: dict[str, Any]) -> dict[str, Any]:
            alpha = arm_sum["alpha_eval_pct"] / 100.0
            c_ar = arm_sum["draft_step_ms"] / ar_step_ms if ar_step_ms > 0 else 0.0
            c_ver = arm_sum["draft_step_ms"] / arm_sum["verify_rnd_ms"] if arm_sum["verify_rnd_ms"] > 0 else 0.0
            gamma = arm_sum["verify_rnd_ms"] / ar_step_ms if ar_step_ms > 0 else 1.0
            k = 4
            exp_tok_rnd = (1.0 - alpha ** (k + 1)) / max(1e-9, 1.0 - alpha)
            lev_ideal = exp_tok_rnd / (k * c_ar + 1.0)
            lev_gamma = exp_tok_rnd / (k * c_ar + gamma)
            return {
                "ar_step_ms": round(ar_step_ms, 3),
                "c_ar": round(c_ar, 4),
                "c_ver": round(c_ver, 4),
                "gamma_ver": round(gamma, 3),
                "leviathan_ideal_decode_x": round(lev_ideal, 3),
                "leviathan_gamma_adjusted_x": round(lev_gamma, 3),
                "observed_decode_x_vs_ar": round(arm_sum["dec_tok_s"] / ar_sum["dec_tok_s"], 3),
                "observed_e2e_x_vs_ar": round(arm_sum["e2e_tok_s"] / ar_sum["e2e_tok_s"], 3),
            }

        sp_vs_ar = {m: _paired_boot(rec_by_arm["AR"], rec_by_arm["SPADE"], m) for m in ("e2e_tok_s", "dec_tok_s")}
        fl_vs_ar = {m: _paired_boot(rec_by_arm["AR"], rec_by_arm["Full"], m) for m in ("e2e_tok_s", "dec_tok_s")} if fl_sum else None
        sp_vs_fl = {m: _paired_boot(rec_by_arm["Full"], rec_by_arm["SPADE"], m) for m in ("e2e_tok_s", "dec_tok_s")} if fl_sum else None

        sweep_results["models"][d_name] = {
            "ratio_label": ratio_str,
            "exact_param_ratio": round(exact_ratio, 2),
            "target_params_b": round(t_params / 1e9, 2),
            "draft_params_b": round(d_params / 1e9, 2),
            "AR": ar_sum,
            "SPADE": {**sp_sum, **_leviathan_row(sp_sum)},
            "Full": ({**fl_sum, **_leviathan_row(fl_sum)} if fl_sum else {"error": full_error}),
            "spade_vs_ar": sp_vs_ar,
            "full_vs_ar": fl_vs_ar,
            "spade_vs_full": sp_vs_fl,
        }
        _write_json(ckpt_path, sweep_results)
        print(f"  DONE {d_name}: AR={ar_sum['e2e_tok_s']} | SPADE={sp_sum['e2e_tok_s']} (Lev_ideal={sweep_results['models'][d_name]['SPADE']['leviathan_ideal_decode_x']}x, c_AR={sweep_results['models'][d_name]['SPADE']['c_ar']})", flush=True)
        del pair
        _cleanup_cuda()


# ---------------------------------------------------------------------------
# Block 6: R=3 Counterbalanced Randomized-Order Passes on Long-173 (Point f)
# ---------------------------------------------------------------------------

def run_long173_interleaved_block() -> None:
    print("\n=== [BLOCK: long173_interleaved] R=3 per-claim randomized passes on Long-173 ===", flush=True)
    rows = _load_jsonl(LONG173_PATH)
    assert len(rows) == 173
    pair = QwenPair(QWEN3_8B_SNAP, QWEN3_06B_SNAP, local_files_only=True, dtype="bfloat16", draft_attn_implementation="flex_attention")
    max_ptok = max(len(pair.draft_tokenizer(format_qwen_prompt(pair.draft_tokenizer, r["prompt"]), add_special_tokens=False)["input_ids"]) for r in rows)
    full_cap = ((max_ptok + 384 + 128 + 127) // 128) * 128
    compact_cap = 6144

    runner_full = StaticDraftRunner(pair.draft, max_cache_len=full_cap, mode="compile_reduce_overhead", device=pair._input_device(pair.draft))
    runner_full.prepare(warmup_steps=3)
    runner_compact = StaticDraftRunner(pair.draft, max_cache_len=compact_cap, mode="compile_reduce_overhead", device=pair._input_device(pair.draft))
    runner_compact.prepare(warmup_steps=3)

    # Warm up every arm outside timed runs
    for gw in range(2):
        pair.generate(rows[0]["prompt"], mode="B0-S", max_new_tokens=16, temperature=0.7, top_p=0.9, seed=800 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast")
        pair.draft_runner = runner_full
        pair.generate(rows[0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=810 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast")
        pair.draft_runner = runner_compact
        pair.generate(rows[0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=820 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast", streaming_llm_cfg={"kv_budget": 2529, "sink_tokens": 256})
        pair.generate(rows[0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=830 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": rows[0], "budget_chars": 10000, "tail_window": 16})
    _cleanup_cuda()

    arms = ["Target AR (B0-S)", "Baseline Full", "Sink+Recent Trunc. (B=2,529)", "BM25 Sent-B10k", "SPADE B=10k"]
    pass_summaries: list[dict[str, Any]] = []
    all_pass_recs: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}

    for pass_idx in range(3):
        pass_seed = 42 + pass_idx
        pass_ckpt = REV8_DIR / f"long173_interleaved_pass{pass_idx}.json"
        if pass_ckpt.exists():
            pdata = json.loads(pass_ckpt.read_text(encoding="utf-8"))
            print(f"  SKIP Pass {pass_idx} (already completed in {pass_ckpt})", flush=True)
        else:
            print(f"  START Pass {pass_idx+1}/3 (seed={pass_seed})...", flush=True)
            t_p0 = time.perf_counter()
            per_arm_recs: dict[str, dict[str, dict[str, Any]]] = {a: {} for a in arms}
            for idx, row in enumerate(rows):
                cid = str(row["claim_id"])
                perm = list(arms)
                random.Random(1000 * (pass_idx + 1) + idx).shuffle(perm)
                for arm in perm:
                    if arm == "Target AR (B0-S)":
                        g = pair.generate(row["prompt"], mode="B0-S", max_new_tokens=384, temperature=0.7, top_p=0.9, seed=pass_seed + idx, run_id=f"l173-p{pass_idx}-ar", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                    elif arm == "Baseline Full":
                        pair.draft_runner = runner_full
                        g = pair.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=pass_seed + idx, run_id=f"l173-p{pass_idx}-full", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                    elif arm == "Sink+Recent Trunc. (B=2,529)":
                        pair.draft_runner = runner_compact
                        g = pair.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=pass_seed + idx, run_id=f"l173-p{pass_idx}-sr", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast", streaming_llm_cfg={"kv_budget": 2529, "sink_tokens": 256})
                    elif arm == "BM25 Sent-B10k":
                        pair.draft_runner = runner_compact
                        dctx = build_draft_context(row, policy="sentence_budget", budget_chars=10000, min_sentences_per_passage=1)
                        g = pair.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=pass_seed + idx, run_id=f"l173-p{pass_idx}-bm25", claim_id=cid, draft_prompt=dctx.draft_prompt, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                        g.timings["context_selection_s"] = dctx.context_selection_s
                    else:  # SPADE B=10k
                        pair.draft_runner = runner_compact
                        g = pair.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=pass_seed + idx, run_id=f"l173-p{pass_idx}-spade", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": row, "budget_chars": 10000, "tail_window": 16})
                    per_arm_recs[arm][cid] = _extract_gen_record(cid, g)
                if (idx + 1) % 35 == 0 or (idx + 1) == len(rows):
                    print(f"    Pass {pass_idx+1}/3: {idx+1}/{len(rows)} ({(time.perf_counter()-t_p0)/60:.1f} min)", flush=True)
            pdata = {
                "pass_idx": pass_idx,
                "seed": pass_seed,
                "minutes": round((time.perf_counter() - t_p0) / 60.0, 2),
                "summaries": {a: _aggregate_metrics(list(per_arm_recs[a].values())) for a in arms},
                "claims_by_arm": per_arm_recs,
            }
            _write_json(pass_ckpt, pdata)
        pass_summaries.append(pdata["summaries"])
        for a in arms:
            all_pass_recs[a].append(pdata["claims_by_arm"][a])

    # Aggregate mean +/- SD across the R=3 passes AND pooled claim-clustered bootstrap over the 3 passes!
    # Pool per claim across passes cleanly so each claim contributes its mean tokens & time across R=3 passes.
    pooled_claim_maps: dict[str, dict[str, dict[str, Any]]] = {a: {} for a in arms}
    for a in arms:
        for cid in all_pass_recs[a][0]:
            rs = [all_pass_recs[a][p][cid] for p in range(3)]
            pooled_claim_maps[a][cid] = {
                "claim_id": cid,
                "output_tokens": sum(r["output_tokens"] for r in rs),
                "e2e_s": sum(r["e2e_s"] for r in rs),
                "decode_s": sum(r["decode_s"] for r in rs),
                "context_selection_s": sum(r["context_selection_s"] for r in rs) / 3.0,
                "draft_prompt_tokens": sum(r["draft_prompt_tokens"] for r in rs) / 3.0,
                "draft_kv_peak_mb": sum(r["draft_kv_peak_mb"] for r in rs) / 3.0,
                "target_rounds": sum(r["target_rounds"] for r in rs),
                "n_eval": sum(r["n_eval"] for r in rs),
                "n_eval_acc": sum(r["n_eval_acc"] for r in rs),
                "n_prop": sum(r["n_prop"] for r in rs),
                "n_acc": sum(r["n_acc"] for r in rs),
                "draft_latency_s": sum(r["draft_latency_s"] for r in rs),
                "verification_latency_s": sum(r["verification_latency_s"] for r in rs),
            }

    summary_rows = []
    for a in arms:
        e2es = [ps[a]["e2e_tok_s"] for ps in pass_summaries]
        decs = [ps[a]["dec_tok_s"] for ps in pass_summaries]
        alps = [ps[a]["alpha_eval_pct"] for ps in pass_summaries]
        pooled = _aggregate_metrics(list(pooled_claim_maps[a].values()))
        summary_rows.append({
            "label": a,
            "e2e_mean": round(statistics.mean(e2es), 2),
            "e2e_sd": round(statistics.stdev(e2es), 2),
            "dec_mean": round(statistics.mean(decs), 2),
            "dec_sd": round(statistics.stdev(decs), 2),
            "alpha_eval_mean": round(statistics.mean(alps), 2),
            "alpha_eval_sd": round(statistics.stdev(alps), 2),
            "draft_tok": pooled["draft_tok"],
            "draft_kv_mb": pooled["draft_kv_mb"],
            "sel_ms_claim": round(pooled["sel_ms_claim"], 2),
            "pass_e2es": e2es,
        })

    contrasts = []
    for base_lbl in ("Target AR (B0-S)", "Baseline Full", "Sink+Recent Trunc. (B=2,529)", "BM25 Sent-B10k"):
        b = _paired_boot(pooled_claim_maps[base_lbl], pooled_claim_maps["SPADE B=10k"], "e2e_tok_s")
        b["contrast"] = f"SPADE vs {base_lbl}"
        contrasts.append(b)
    _holm_adjust(contrasts)

    alpha_contrasts = []
    for base_lbl in ("Baseline Full", "Sink+Recent Trunc. (B=2,529)", "BM25 Sent-B10k"):
        b = _paired_boot(pooled_claim_maps[base_lbl], pooled_claim_maps["SPADE B=10k"], "alpha_eval_pct")
        b["contrast"] = f"SPADE vs {base_lbl}"
        alpha_contrasts.append(b)
    _holm_adjust(alpha_contrasts)

    report = {
        "n_claims": len(rows),
        "n_passes": 3,
        "design": "per_claim_randomized_5arm_order_R3",
        "rows": summary_rows,
        "pairwise_e2e_holm": {c["contrast"]: c for c in contrasts},
        "pairwise_alpha_holm": {c["contrast"]: c for c in alpha_contrasts},
        "pass_summaries": pass_summaries,
    }
    _write_json(REV8_DIR / "long173_interleaved_r3.json", report)
    del runner_full, runner_compact, pair
    _cleanup_cuda()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--blocks",
        nargs="+",
        default=["batch", "fixed_bm25", "sliding_fix", "oracle_interleaved", "qwen3_ratio", "long173_interleaved"],
        choices=["batch", "fixed_bm25", "sliding_fix", "oracle_interleaved", "qwen3_ratio", "long173_interleaved"],
    )
    args = p.parse_args()
    REV8_DIR.mkdir(parents=True, exist_ok=True)
    runners = {
        "batch": run_batch_block,
        "fixed_bm25": run_fixed_bm25_block,
        "sliding_fix": run_sliding_fix_block,
        "oracle_interleaved": run_oracle_interleaved_block,
        "qwen3_ratio": run_qwen3_ratio_block,
        "long173_interleaved": run_long173_interleaved_block,
    }
    for blk in args.blocks:
        t0 = time.perf_counter()
        runners[blk]()
        print(f"[COMPLETED {blk} in {(time.perf_counter()-t0)/60:.2f} min]", flush=True)


if __name__ == "__main__":
    main()
