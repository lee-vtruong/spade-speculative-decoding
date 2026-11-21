"""Resumable RTX 5090 GPU experiment suite for SPADE v9 reviewer revisions (critiques a-g).

Each block runs cleanly as a standalone top-level Python process (`--blocks <name>`),
writing resumable JSON checkpoints under `outputs/analysis/v9_revisions/`:
  1. `target_opt_profile`:      HBM roofline, GPU exactness across block boundary/rollback,
                                and Eager vs Compiled StaticTargetRunner AR/Verify + sub-ms profile.
  2. `heldout300_interleaved_r3`: R=3 counterbalanced per-claim randomized passes on ALL N=300
                                held-out AVeriTeC claims across 5 arms (compiled target+draft).
  3. `ratio_and_cross_family`:  Compiled-stack Qwen3-{0.6B, 1.7B, 4B} (N=100 stratified seed=42),
                                Qwen2.5-7B/0.5B (N=300), and Llama-3.1-8B/3.2-1B (N=300).
  4. `batched_e2e_concurrency`: Co-located batched compiled StaticCache E2E serving throughput & max
                                batch in 32 GB for Target AR vs Full-SD vs SPADE B=10k/2k (8k & 16k).
  5. `fixed_bm25_interaction`:  Per-claim deterministic fixed-prefix records (Long-173 & HotpotQA)
                                for 3-tercile, interaction, slope, & Holm statistical audits.
  6. `scaling_2k_32k`:          Interleaved 2k, 4k, 8k, 10k, 12k, 14k, 16k, 32k context scaling
                                on the optimized compiled stack (N=20 at 2k-16k, N=10 at 32k).
  7. `outlen_sweep`:            Interleaved output-length sweep M in {128, 256, 512, 1024} at 8k & 16k.
  8. `oracle_hotpot_opt`:       Compiled-stack Oracle-393 (50/50 interleaved) & HotpotQA-100 (4 arms).
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import re
import statistics
import string
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from spade.adaptive.context_selection import build_draft_context, build_draft_context_from_scores
from spade.adaptive.target_attention_selector import (
    TargetPrefillAttentionHook,
    find_evidence_and_sentence_token_spans,
    resolve_layer_indices,
    score_evidence_from_token_importance,
)
from spade.benchmark.evaluation import classification_metrics, parse_generation
from spade.benchmark.prompt_builder import build_prompt
from spade.decoding.autoregressive import format_qwen_prompt
from spade.decoding.cache_utils import crop_past_key_values
from spade.decoding.hf_assisted import QwenPair
from spade.decoding.runtime_opt import StaticDraftRunner, StaticTargetRunner, processed_distribution_fast
from spade.decoding.sampling import total_variation
from v9_batched_serving import run_single_batched_ar, run_single_batched_sd

CACHE_DIR = "/home/stackops/whale/cache/huggingface/hub"
QWEN3_8B_SNAP = f"{CACHE_DIR}/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
QWEN3_06B_SNAP = f"{CACHE_DIR}/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

REV9_DIR = Path("outputs/analysis/v9_revisions")
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


def _cleanup_cuda(reset_dynamo: bool = False) -> None:
    if reset_dynamo:
        torch._dynamo.reset()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _content_tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if len(w) > 2 and w not in STOPWORDS}


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


def _strip_json_fences(raw: str) -> str:
    s = raw.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else s


def _normalize_qa_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _token_f1(pred: str, gold: str) -> float:
    p_toks = _normalize_qa_answer(pred).split()
    g_toks = _normalize_qa_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(p_toks)
    rec = num_same / len(g_toks)
    return (2.0 * prec * rec) / (prec + rec)


def _extract_final_answer(raw_text: str) -> str:
    m = re.search(r"Final\s+Answer\s*:\s*(.+)", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    return raw_text.strip()


def _max_prompt_tokens(tok: Any, rows: list[dict[str, Any]]) -> int:
    return max(len(tok(format_qwen_prompt(tok, r["prompt"]), add_special_tokens=False)["input_ids"]) for r in rows)


def _extract_gen_record(
    claim_id: str,
    gen: Any,
    row: dict[str, Any] | None = None,
    *,
    peak_vram_gb: float = 0.0,
) -> dict[str, Any]:
    rds = gen.rounds or []
    t = gen.timings or {}
    sel_s = float(t.get("context_selection_s", 0.0))
    e2e_s = sel_s + float(t.get("prompt_prefill_s", 0.0)) + float(t.get("draft_prefill_s", 0.0)) + float(t.get("decode_s", 0.0))
    n_eval = sum(
        min(int(rd["proposed_token_count"]), int(rd["accepted_draft_tokens"]) + (0 if rd["full_acceptance"] else 1))
        for rd in rds
    )
    n_acc = sum(int(rd["accepted_draft_tokens"]) for rd in rds)
    n_prop = sum(int(rd["proposed_token_count"]) for rd in rds)
    d_lat_s = sum(float(rd.get("draft_latency_s", 0.0)) for rd in rds)
    v_lat_s = sum(float(rd.get("verification_latency_s", 0.0)) for rd in rds)

    ev_ids = {str(e.get("evidence_id", f"E{i+1}")) for i, e in enumerate((row or {}).get("evidence") or [])}
    parsed_strict = parse_generation(gen.raw_text, ev_ids)
    parsed_relaxed = parse_generation(_strip_json_fences(gen.raw_text), ev_ids)

    gold_verdict = None
    ev_chars = 0.0
    if row is not None:
        ev_chars = float(sum(len(str(e.get("text", ""))) for e in (row.get("evidence") or [])))
        gold_verdict = (row.get("evaluation") or {}).get("gold_label") or row.get("gold_label")

    qa_f1 = None
    qa_sub = None
    if row is not None and row.get("answers"):
        golds = [str(g) for g in row["answers"]]
        pred_ans = _extract_final_answer(gen.raw_text)
        qa_f1 = max(_token_f1(pred_ans, g) for g in golds)
        norm_full = _normalize_qa_answer(gen.raw_text)
        qa_sub = 1.0 if any(_normalize_qa_answer(g) in norm_full for g in golds if _normalize_qa_answer(g)) else 0.0

    return {
        "claim_id": claim_id,
        "ev_chars": ev_chars,
        "prompt_tokens": int(gen.prompt_tokens),
        "output_tokens": len(gen.token_ids),
        "e2e_s": e2e_s,
        "decode_s": float(t.get("decode_s", 0.0)),
        "context_selection_s": sel_s,
        "prompt_prefill_s": float(t.get("prompt_prefill_s", 0.0)),
        "draft_prefill_s": float(t.get("draft_prefill_s", 0.0)),
        "draft_generation_s": float(t.get("draft_generation_s", 0.0)),
        "target_verification_s": float(t.get("target_verification_s", 0.0)),
        "correction_s": float(t.get("correction_s", 0.0)),
        "verify_forward_s": float(t.get("verify_forward_s", 0.0)),
        "verify_dist_s": float(t.get("verify_dist_s", 0.0)),
        "verify_residual_s": float(t.get("verify_residual_s", 0.0)),
        "verify_rollback_s": float(t.get("verify_rollback_s", 0.0)),
        "draft_forward_s": float(t.get("draft_forward_s", 0.0)),
        "draft_dist_s": float(t.get("draft_dist_s", 0.0)),
        "draft_catchup_s": float(t.get("draft_catchup_s", 0.0)),
        "draft_prompt_tokens": float(t.get("draft_prompt_tokens", 0.0)),
        "draft_kv_peak_mb": float(t.get("draft_kv_peak_bytes", 0.0)) / (1024.0 ** 2),
        "target_kv_peak_mb": float(t.get("target_kv_peak_bytes", 0.0)) / (1024.0 ** 2),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "target_rounds": len(rds),
        "n_eval": n_eval,
        "n_eval_acc": n_acc,
        "n_prop": n_prop,
        "n_acc": n_acc,
        "draft_latency_s": d_lat_s,
        "verification_latency_s": v_lat_s,
        "json_valid_strict": bool(parsed_strict["parse_success"]),
        "json_valid_relaxed": bool(parsed_relaxed["parse_success"]),
        "citation_valid_strict": bool(parsed_strict["citation_ids_valid"]),
        "pred_verdict": parsed_strict["mapped_label"] or parsed_relaxed["mapped_label"],
        "gold_verdict": gold_verdict,
        "qa_f1": qa_f1,
        "qa_sub": qa_sub,
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
    qa_f1s = [r["qa_f1"] for r in recs if r.get("qa_f1") is not None]
    qa_subs = [r["qa_sub"] for r in recs if r.get("qa_sub") is not None]
    out: dict[str, Any] = {
        "n_claims": len(recs),
        "e2e_tok_s": round(tot_tok / tot_e2e, 4) if tot_e2e > 0 else 0.0,
        "per_claim_e2e_tok_s": round(sum(per_claim) / max(1, len(per_claim)), 4),
        "dec_tok_s": round(tot_tok / tot_dec, 4) if tot_dec > 0 else 0.0,
        "alpha_eval_pct": round((tot_eval_acc / tot_eval) * 100.0, 4) if tot_eval > 0 else 0.0,
        "alpha_prop_pct": round((tot_acc / tot_prop) * 100.0, 4) if tot_prop > 0 else 0.0,
        "tok_per_round": round(tot_tok / tot_rnd, 4) if tot_rnd > 0 else 0.0,
        "draft_tok": round(sum(r["draft_prompt_tokens"] for r in recs) / n, 1),
        "draft_kv_mb": round(sum(r["draft_kv_peak_mb"] for r in recs) / n, 1),
        "target_kv_mb": round(sum(r["target_kv_peak_mb"] for r in recs) / n, 1),
        "peak_vram_gb": round(max((r.get("peak_vram_gb", 0.0) for r in recs), default=0.0), 2),
        "sel_ms_claim": round((sum(r["context_selection_s"] for r in recs) / n) * 1000.0, 2),
        "draft_step_ms": round((tot_dlat / tot_prop) * 1000.0, 4) if tot_prop > 0 else 0.0,
        "verify_rnd_ms": round((tot_vlat / tot_rnd) * 1000.0, 4) if tot_rnd > 0 else 0.0,
        "total_output_tokens": tot_tok,
        "total_e2e_s": round(tot_e2e, 4),
        "total_decode_s": round(tot_dec, 4),
        "json_strict_pct": round(sum(1 for r in recs if r.get("json_valid_strict")) / n * 100.0, 2),
        "json_relaxed_pct": round(sum(1 for r in recs if r.get("json_valid_relaxed")) / n * 100.0, 2),
        "citation_strict_pct": round(sum(1 for r in recs if r.get("citation_valid_strict")) / n * 100.0, 2),
        "decomp_ms_tok": {
            "context_selection": round(sum(r["context_selection_s"] for r in recs) / max(1, tot_tok) * 1000.0, 3),
            "target_prefill": round(sum(r["prompt_prefill_s"] for r in recs) / max(1, tot_tok) * 1000.0, 3),
            "draft_prefill": round(sum(r["draft_prefill_s"] for r in recs) / max(1, tot_tok) * 1000.0, 3),
            "draft_decode": round(sum(r["draft_generation_s"] for r in recs) / max(1, tot_tok) * 1000.0, 3),
            "target_verify": round(sum(r["target_verification_s"] for r in recs) / max(1, tot_tok) * 1000.0, 3),
            "verify_dist": round(sum(r.get("verify_dist_s", 0.0) for r in recs) / max(1, tot_tok) * 1000.0, 3),
            "correction": round(sum(r["correction_s"] for r in recs) / max(1, tot_tok) * 1000.0, 3),
            "verify_residual": round(sum(r.get("verify_residual_s", 0.0) for r in recs) / max(1, tot_tok) * 1000.0, 3),
            "verify_rollback": round(sum(r.get("verify_rollback_s", 0.0) for r in recs) / max(1, tot_tok) * 1000.0, 3),
            "e2e_ms_tok": round(tot_e2e / max(1, tot_tok) * 1000.0, 3),
        },
        "verify_sub_ms_round": {
            "verify_forward": round(sum(r.get("verify_forward_s", 0.0) for r in recs) / max(1, tot_rnd) * 1000.0, 3),
            "verify_dist": round(sum(r.get("verify_dist_s", 0.0) for r in recs) / max(1, tot_rnd) * 1000.0, 3),
            "verify_residual": round(sum(r.get("verify_residual_s", 0.0) for r in recs) / max(1, tot_rnd) * 1000.0, 3),
            "verify_rollback": round(sum(r.get("verify_rollback_s", 0.0) for r in recs) / max(1, tot_rnd) * 1000.0, 3),
            "draft_catchup": round(sum(r.get("draft_catchup_s", 0.0) for r in recs) / max(1, tot_rnd) * 1000.0, 3),
        },
    }
    if qa_f1s:
        out["qa_f1_pct"] = round(sum(qa_f1s) / len(qa_f1s) * 100.0, 2)
        out["qa_sub_pct"] = round(sum(qa_subs) / len(qa_subs) * 100.0, 2)
    return out


def _pooled_val(recs: list[dict[str, Any]], metric: str) -> float:
    if metric == "e2e_tok_s":
        den = sum(r["e2e_s"] for r in recs)
        return sum(r["output_tokens"] for r in recs) / den if den > 0 else 0.0
    if metric == "dec_tok_s":
        den = sum(r["decode_s"] for r in recs)
        return sum(r["output_tokens"] for r in recs) / den if den > 0 else 0.0
    if metric == "alpha_eval_pct":
        den = sum(r["n_eval"] for r in recs)
        return (sum(r["n_eval_acc"] for r in recs) / den) * 100.0 if den > 0 else 0.0
    if metric == "qa_f1_pct":
        vals = [r["qa_f1"] for r in recs if r.get("qa_f1") is not None]
        return (sum(vals) / len(vals)) * 100.0 if vals else 0.0
    if metric == "qa_sub_pct":
        vals = [r["qa_sub"] for r in recs if r.get("qa_sub") is not None]
        return (sum(vals) / len(vals)) * 100.0 if vals else 0.0
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


def _stratified_heldout100() -> list[dict[str, Any]]:
    """Deterministic seed-42 stratified subsample matching exact 300-claim proportions (42/17/41)."""
    all_rows = _load_jsonl(HELD300_PATH)
    short_rows = [r for r in all_rows if sum(len(str(e.get("text", ""))) for e in (r.get("evidence") or [])) <= 10000]
    med_long = [r for r in all_rows if 10000 < sum(len(str(e.get("text", ""))) for e in (r.get("evidence") or [])) < 20000]
    dense_rows = [r for r in all_rows if sum(len(str(e.get("text", ""))) for e in (r.get("evidence") or [])) >= 20000]
    rng = random.Random(42)
    rng.shuffle(short_rows)
    rng.shuffle(med_long)
    rng.shuffle(dense_rows)
    picked = short_rows[:42] + med_long[:17] + dense_rows[:41]
    rng.shuffle(picked)
    return picked


def _pool_r3_claims(all_pass_recs: dict[str, list[dict[str, dict[str, Any]]]]) -> dict[str, dict[str, dict[str, Any]]]:
    pooled: dict[str, dict[str, dict[str, Any]]] = {a: {} for a in all_pass_recs}
    n_passes = len(next(iter(all_pass_recs.values())))
    for a, passes in all_pass_recs.items():
        for cid in passes[0]:
            rs = [passes[p][cid] for p in range(n_passes)]
            pooled[a][cid] = {
                "claim_id": cid,
                "ev_chars": rs[0]["ev_chars"],
                "prompt_tokens": int(round(sum(r["prompt_tokens"] for r in rs) / n_passes)),
                "output_tokens": sum(r["output_tokens"] for r in rs),
                "e2e_s": sum(r["e2e_s"] for r in rs),
                "decode_s": sum(r["decode_s"] for r in rs),
                "prompt_prefill_s": sum(r["prompt_prefill_s"] for r in rs),
                "draft_prefill_s": sum(r["draft_prefill_s"] for r in rs),
                "draft_generation_s": sum(r["draft_generation_s"] for r in rs),
                "target_verification_s": sum(r["target_verification_s"] for r in rs),
                "correction_s": sum(r["correction_s"] for r in rs),
                "verify_forward_s": sum(r.get("verify_forward_s", 0.0) for r in rs),
                "verify_dist_s": sum(r.get("verify_dist_s", 0.0) for r in rs),
                "verify_residual_s": sum(r.get("verify_residual_s", 0.0) for r in rs),
                "verify_rollback_s": sum(r.get("verify_rollback_s", 0.0) for r in rs),
                "draft_catchup_s": sum(r.get("draft_catchup_s", 0.0) for r in rs),
                "context_selection_s": sum(r["context_selection_s"] for r in rs) / n_passes,
                "draft_prompt_tokens": sum(r["draft_prompt_tokens"] for r in rs) / n_passes,
                "draft_kv_peak_mb": sum(r["draft_kv_peak_mb"] for r in rs) / n_passes,
                "target_kv_peak_mb": sum(r["target_kv_peak_mb"] for r in rs) / n_passes,
                "peak_vram_gb": max(r.get("peak_vram_gb", 0.0) for r in rs),
                "target_rounds": sum(r["target_rounds"] for r in rs),
                "n_eval": sum(r["n_eval"] for r in rs),
                "n_eval_acc": sum(r["n_eval_acc"] for r in rs),
                "n_prop": sum(r["n_prop"] for r in rs),
                "n_acc": sum(r["n_acc"] for r in rs),
                "draft_latency_s": sum(r["draft_latency_s"] for r in rs),
                "verification_latency_s": sum(r["verification_latency_s"] for r in rs),
            }
    return pooled


def _build_padded_prompt(
    tok: Any, target_tokens: int, base_row: dict[str, Any], pool_passages: list[str], start_offset: int
) -> dict[str, Any]:
    ev = [
        {"evidence_id": str(e.get("evidence_id", f"E{i+1}")), "text": str(e.get("text", ""))}
        for i, e in enumerate(base_row.get("evidence") or [])
    ]
    idx_p = start_offset
    while True:
        prompt_str, phash = build_prompt(base_row["claim"], ev, mode="retrieved")
        n_tok = len(tok(format_qwen_prompt(tok, prompt_str), add_special_tokens=False)["input_ids"])
        if n_tok >= target_tokens:
            break
        ev.append({"evidence_id": f"E{len(ev)+1}", "text": pool_passages[idx_p % len(pool_passages)]})
        idx_p += 1
    excess = n_tok - target_tokens
    if excess > 30 and len(ev) > 1:
        last_txt = ev[-1]["text"]
        trim_chars = int(excess * 4.0)
        if trim_chars < len(last_txt) - 100:
            ev[-1]["text"] = last_txt[: len(last_txt) - trim_chars]
        else:
            ev.pop()
    if target_tokens == 2048:
        while len(ev) > 1 and sum(len(str(e["text"])) for e in ev) > 9800:
            over = sum(len(str(e["text"])) for e in ev) - 9800
            if over < len(ev[-1]["text"]) - 100:
                ev[-1]["text"] = ev[-1]["text"][: len(ev[-1]["text"]) - over]
                break
            ev.pop()
        if len(ev) == 1 and len(str(ev[0]["text"])) > 9800:
            ev[0]["text"] = ev[0]["text"][:9800]
    prompt_str, phash = build_prompt(base_row["claim"], ev, mode="retrieved")
    n_tok = len(tok(format_qwen_prompt(tok, prompt_str), add_special_tokens=False)["input_ids"])
    return {
        "claim_id": f"{base_row['claim_id']}-scale-{target_tokens}",
        "claim": base_row["claim"],
        "evidence": ev,
        "evidence_mode": "retrieved",
        "prompt": prompt_str,
        "prompt_hash": phash,
        "actual_tokens": n_tok,
    }


def _leviathan_row(arm_sum: dict[str, Any], ar_step_ms: float, ar_dec_tok_s: float, ar_e2e_tok_s: float) -> dict[str, Any]:
    alpha = arm_sum["alpha_eval_pct"] / 100.0
    c_ar = arm_sum["draft_step_ms"] / ar_step_ms if ar_step_ms > 0 else 0.0
    c_ver = arm_sum["draft_step_ms"] / arm_sum["verify_rnd_ms"] if arm_sum["verify_rnd_ms"] > 0 else 0.0
    gamma = arm_sum["verify_rnd_ms"] / ar_step_ms if ar_step_ms > 0 else 1.0
    k = 4
    exp_tok_rnd = (1.0 - alpha ** (k + 1)) / max(1e-9, 1.0 - alpha)
    return {
        "ar_step_ms": round(ar_step_ms, 3),
        "c_ar": round(c_ar, 4),
        "c_ver": round(c_ver, 4),
        "gamma_ver": round(gamma, 3),
        "leviathan_ideal_decode_x": round(exp_tok_rnd / (k * c_ar + 1.0), 3),
        "leviathan_gamma_adjusted_x": round(exp_tok_rnd / (k * c_ar + gamma), 3),
        "observed_decode_x_vs_ar": round(arm_sum["dec_tok_s"] / ar_dec_tok_s, 3) if ar_dec_tok_s > 0 else 0.0,
        "observed_e2e_x_vs_ar": round(arm_sum["e2e_tok_s"] / ar_e2e_tok_s, 3) if ar_e2e_tok_s > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Block 1A: target_opt_profile (Roofline, GPU Exactness, Eager vs Compiled)
# ---------------------------------------------------------------------------

def _check_model_flex_exactness(model_snap: str, label: str) -> dict[str, Any]:
    """Verify StaticTargetRunner(compile_cudagraph) vs eager DynamicCache across block boundaries & rollback."""
    out_info: dict[str, Any] = {"model": label, "attn_impl": "flex_attention", "passed": False}
    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_snap, dtype=torch.bfloat16, device_map="cuda", attn_implementation="flex_attention", local_files_only=True
        ).eval()
        runner = StaticTargetRunner(mdl, max_cache_len=512, max_verify_len=5, mode="compile_cudagraph")
        runner.prepare(warmup_steps=3)

        max_diffs: list[float] = []
        tvs: list[float] = []
        for p_len in (125, 250):
            p_ids = torch.randint(100, 5000, (1, p_len), device="cuda")
            dyn = DynamicCache()
            with torch.inference_mode():
                d_pre = mdl(input_ids=p_ids, past_key_values=dyn, use_cache=True).logits[0, -1].float()
            s_pre = runner.prefill(p_ids[0].tolist())
            max_diffs.append(float((d_pre - s_pre).abs().max().item()))
            tvs.append(total_variation(processed_distribution_fast(d_pre, 0.7, top_p=0.9), processed_distribution_fast(s_pre, 0.7, top_p=0.9)))
            if p_len == 250:
                # Cross 256 boundary with step(1) -> rollback(254) -> verify(5) -> verify(3) -> step(1)
                for tid in (301, 302, 303, 304, 305, 306, 307):
                    with torch.inference_mode():
                        d_s = mdl(input_ids=torch.tensor([[tid]], device="cuda"), past_key_values=dyn, use_cache=True).logits[0, -1].float()
                    s_s = runner.step(tid)
                    max_diffs.append(float((d_s - s_s).abs().max().item()))
                    tvs.append(total_variation(processed_distribution_fast(d_s, 0.7, top_p=0.9), processed_distribution_fast(s_s, 0.7, top_p=0.9)))
                dyn = crop_past_key_values(dyn, 254)
                runner.rollback(254)
                for vtoks in ([401, 402, 403, 404, 405], [501, 502, 503], [601]):
                    with torch.inference_mode():
                        d_v = mdl(input_ids=torch.tensor([vtoks], device="cuda"), past_key_values=dyn, use_cache=True).logits[0].float()
                    s_v = runner.verify(vtoks)
                    for i in range(len(vtoks)):
                        max_diffs.append(float((d_v[i] - s_v[i]).abs().max().item()))
                        tvs.append(total_variation(processed_distribution_fast(d_v[i], 0.7, top_p=0.9), processed_distribution_fast(s_v[i], 0.7, top_p=0.9)))

        # Micro-benchmark step1 and step5 at ctx=256
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for j in range(30):
            _ = runner.step(100 + j)
        torch.cuda.synchronize()
        step1_ms = (time.perf_counter() - t0) / 30 * 1000.0
        runner.rollback(260)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for j in range(20):
            _ = runner.verify([200, 201, 202, 203, 204])
        torch.cuda.synchronize()
        step5_ms = (time.perf_counter() - t0) / 20 * 1000.0

        wt_gb = sum(p.numel() * p.element_size() for p in mdl.parameters()) / 1e9
        out_info.update({
            "passed": (max(max_diffs) < 0.08 and max(tvs) < 1e-3),
            "worst_max_abs_logit_diff": round(max(max_diffs), 6),
            "mean_max_abs_logit_diff": round(statistics.mean(max_diffs), 6),
            "worst_tv": round(max(tvs), 7),
            "mean_tv": round(statistics.mean(tvs), 7),
            "weight_gb": round(wt_gb, 3),
            "step1_ms_short": round(step1_ms, 2),
            "step1_tok_s_short": round(1000.0 / step1_ms, 1),
            "step5_ms_short": round(step5_ms, 2),
            "gamma_ver_short": round(step5_ms / step1_ms, 3),
        })
        del runner, mdl
        _cleanup_cuda(reset_dynamo=True)
    except Exception as exc:
        out_info["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        _cleanup_cuda(reset_dynamo=True)
    return out_info


def run_target_opt_profile_block() -> None:
    from huggingface_hub import snapshot_download

    print("\n=== [BLOCK: target_opt_profile] Roofline, Exactness & Eager vs Compiled Profile ===", flush=True)
    # 1) Measure RTX 5090 sustained D2D HBM copy bandwidth
    buf_a = torch.empty((1024 * 1024 * 256,), dtype=torch.bfloat16, device="cuda")
    buf_b = torch.empty_like(buf_a)
    for _ in range(5):
        buf_b.copy_(buf_a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        buf_b.copy_(buf_a)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 50
    copy_bw_gb_s = (2 * buf_a.numel() * 2) / dt / 1e9
    datasheet_bw_gb_s = 1792.0
    gpu_props = torch.cuda.get_device_properties(0)
    del buf_a, buf_b
    _cleanup_cuda()

    # 2) Numerical exactness + compatibility checks across Qwen3-8B, Qwen2.5-7B, Llama-3.1-8B
    q25_8b_snap = snapshot_download("Qwen/Qwen2.5-7B-Instruct", cache_dir=CACHE_DIR, local_files_only=True)
    lm3_8b_snap = snapshot_download("unsloth/Meta-Llama-3.1-8B-Instruct", cache_dir=CACHE_DIR, local_files_only=True)
    exactness = {
        "Qwen3-8B": _check_model_flex_exactness(QWEN3_8B_SNAP, "Qwen3-8B"),
        "Qwen2.5-7B": _check_model_flex_exactness(q25_8b_snap, "Qwen2.5-7B"),
        "Llama-3.1-8B": _check_model_flex_exactness(lm3_8b_snap, "Llama-3.1-8B"),
    }
    print("  Exactness checks:", json.dumps(exactness, indent=2), flush=True)

    # 3) Controlled N=20 AVeriTeC comparison: Eager Target vs Compiled Target (+ Qwen3-0.6B Compiled Draft)
    rows20 = _stratified_heldout100()[:20]
    tok = AutoTokenizer.from_pretrained(QWEN3_8B_SNAP, local_files_only=True)
    max_ptok = _max_prompt_tokens(tok, rows20)
    full_cap = ((max_ptok + 384 + 32 + 127) // 128) * 128
    spade_cap = 6144

    # 3a) Eager Target (DynamicCache) + Compiled Draft (v8 baseline runtime)
    pair_eager = QwenPair(
        QWEN3_8B_SNAP, QWEN3_06B_SNAP, local_files_only=True, dtype="bfloat16", draft_attn_implementation="flex_attention"
    )
    pair_eager.prepare_draft_runner(mode="compile_reduce_overhead", max_cache_len=full_cap, warmup_steps=3)
    for gw in range(2):
        pair_eager.generate(rows20[0]["prompt"], mode="B0-S", max_new_tokens=16, temperature=0.7, top_p=0.9, seed=100 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast")
        pair_eager.generate(rows20[0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=110 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast")
    eager_recs: dict[str, dict[str, dict[str, Any]]] = {"AR": {}, "Full": {}, "SPADE": {}}
    rng = random.Random(42)
    for i, row in enumerate(rows20):
        cid = str(row["claim_id"])
        arms = ["AR", "Full", "SPADE"]
        rng.shuffle(arms)
        for arm in arms:
            torch.cuda.reset_peak_memory_stats()
            if arm == "AR":
                g = pair_eager.generate(row["prompt"], mode="B0-S", max_new_tokens=384, temperature=0.7, top_p=0.9, seed=42 + i, cache_strategy="kv_cache", distribution_mode="fp32_fast")
            elif arm == "Full":
                g = pair_eager.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + i, cache_strategy="kv_cache", distribution_mode="fp32_fast")
            else:
                g = pair_eager.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + i, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": row, "budget_chars": 10000, "tail_window": 16})
            eager_recs[arm][cid] = _extract_gen_record(cid, g, row, peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3)
    del pair_eager
    _cleanup_cuda(reset_dynamo=True)

    # 3b) Compiled StaticTargetRunner + Compiled StaticDraftRunner (compile_cudagraph, shared graph pool)
    pool = torch.cuda.graph_pool_handle()
    pair_opt = QwenPair(
        QWEN3_8B_SNAP, QWEN3_06B_SNAP, local_files_only=True, dtype="bfloat16",
        draft_attn_implementation="flex_attention", target_attn_implementation="flex_attention",
    )
    pair_opt.prepare_target_runner(mode="compile_cudagraph", max_cache_len=full_cap, max_verify_len=5, warmup_steps=3, graph_pool=pool)
    r_full = StaticDraftRunner(pair_opt.draft, max_cache_len=full_cap, mode="compile_cudagraph", device=pair_opt._input_device(pair_opt.draft), graph_pool=pool)
    r_full.prepare(warmup_steps=3)
    r_spade = StaticDraftRunner(pair_opt.draft, max_cache_len=spade_cap, mode="compile_cudagraph", device=pair_opt._input_device(pair_opt.draft), graph_pool=pool)
    r_spade.prepare(warmup_steps=3)
    for gw in range(2):
        pair_opt.generate(rows20[0]["prompt"], mode="B0-S", max_new_tokens=16, temperature=0.7, top_p=0.9, seed=200 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast")
        pair_opt.draft_runner = r_full
        pair_opt.generate(rows20[0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=210 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast")
        pair_opt.draft_runner = r_spade
        pair_opt.generate(rows20[0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=220 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": rows20[0], "budget_chars": 10000, "tail_window": 16})

    opt_recs: dict[str, dict[str, dict[str, Any]]] = {"AR": {}, "Full": {}, "SPADE": {}}
    rng = random.Random(42)
    for i, row in enumerate(rows20):
        cid = str(row["claim_id"])
        arms = ["AR", "Full", "SPADE"]
        rng.shuffle(arms)
        for arm in arms:
            torch.cuda.reset_peak_memory_stats()
            if arm == "AR":
                g = pair_opt.generate(row["prompt"], mode="B0-S", max_new_tokens=384, temperature=0.7, top_p=0.9, seed=42 + i, cache_strategy="kv_cache", distribution_mode="fp32_fast")
            elif arm == "Full":
                pair_opt.draft_runner = r_full
                g = pair_opt.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + i, cache_strategy="kv_cache", distribution_mode="fp32_fast")
            else:
                pair_opt.draft_runner = r_spade
                g = pair_opt.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + i, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": row, "budget_chars": 10000, "tail_window": 16})
            opt_recs[arm][cid] = _extract_gen_record(cid, g, row, peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3)

    # 3c) Pure GPU kernel microbenchmark at ctx=5,120 (AR step1, Verify(5), Draft step1)
    assert pair_opt.target_runner is not None
    tr = pair_opt.target_runner
    tr.prefill([100] * 5120)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for j in range(50):
        _ = tr.step(200 + j)
    torch.cuda.synchronize()
    k_step1_ms = (time.perf_counter() - t0) / 50 * 1000.0
    tr.rollback(5120)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for j in range(40):
        _ = tr.verify([300, 301, 302, 303, 304])
        tr.rollback(5120)
    torch.cuda.synchronize()
    k_step5_ms = (time.perf_counter() - t0) / 40 * 1000.0
    r_spade.prefill([100] * 1776)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for j in range(50):
        _ = r_spade.step(100 + j)
    torch.cuda.synchronize()
    k_draft1_ms = (time.perf_counter() - t0) / 50 * 1000.0

    wt_8b_gb = exactness["Qwen3-8B"].get("weight_gb", 16.382)
    kv_5k_gb = 5120 * 147456 / 1e9  # 36 * 8 * 128 * 2 * 2 = 147,456 B/tok
    tot_read_gb = wt_8b_gb + kv_5k_gb
    roof_copy_ms = tot_read_gb / copy_bw_gb_s * 1000.0
    roof_spec_ms = tot_read_gb / datasheet_bw_gb_s * 1000.0

    def _summarize_stack(stack_map: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
        ar_s = _aggregate_metrics(list(stack_map["AR"].values()))
        fl_s = _aggregate_metrics(list(stack_map["Full"].values()))
        sp_s = _aggregate_metrics(list(stack_map["SPADE"].values()))
        ar_step = ar_s["total_decode_s"] / ar_s["total_output_tokens"] * 1000.0
        return {
            "AR_step_ms": round(ar_step, 3),
            "AR": ar_s,
            "Full": {**fl_s, **_leviathan_row(fl_s, ar_step, ar_s["dec_tok_s"], ar_s["e2e_tok_s"])},
            "SPADE": {**sp_s, **_leviathan_row(sp_s, ar_step, ar_s["dec_tok_s"], ar_s["e2e_tok_s"])},
            "spade_vs_ar_e2e": _paired_boot(stack_map["AR"], stack_map["SPADE"], "e2e_tok_s"),
            "spade_vs_full_e2e": _paired_boot(stack_map["Full"], stack_map["SPADE"], "e2e_tok_s"),
        }

    eager_sum = _summarize_stack(eager_recs)
    opt_sum = _summarize_stack(opt_recs)

    report = {
        "hardware": {
            "gpu_name": gpu_props.name,
            "vram_gib": round(gpu_props.total_memory / 1024**3, 2),
            "measured_hbm_copy_bw_gb_s": round(copy_bw_gb_s, 1),
            "datasheet_peak_bw_gb_s": datasheet_bw_gb_s,
            "qwen3_8b_weight_gb": round(wt_8b_gb, 3),
            "qwen3_8b_5k_kv_gb": round(kv_5k_gb, 3),
            "roofline_min_step_ms_measured_copy": round(roof_copy_ms, 2),
            "roofline_min_step_ms_datasheet": round(roof_spec_ms, 2),
        },
        "exactness": exactness,
        "kernel_microbench_5k": {
            "qwen3_8b_step1_ms": round(k_step1_ms, 2),
            "qwen3_8b_step1_tok_s": round(1000.0 / k_step1_ms, 1),
            "qwen3_8b_verify5_ms": round(k_step5_ms, 2),
            "qwen3_8b_gamma_ver_kernel": round(k_step5_ms / k_step1_ms, 3),
            "qwen3_06b_step1_ms": round(k_draft1_ms, 2),
            "qwen3_06b_step1_tok_s": round(1000.0 / k_draft1_ms, 1),
            "c_ar_kernel": round(k_draft1_ms / k_step1_ms, 4),
            "pct_of_measured_copy_roofline": round((roof_copy_ms / k_step1_ms) * 100.0, 1),
            "pct_of_datasheet_roofline": round((roof_spec_ms / k_step1_ms) * 100.0, 1),
        },
        "eager_target_n20": eager_sum,
        "compiled_target_n20": opt_sum,
    }
    _write_json(REV9_DIR / "target_opt_profile.json", report)
    print(f"  DONE target_opt_profile: Eager AR={eager_sum['AR']['dec_tok_s']} -> Compiled AR={opt_sum['AR']['dec_tok_s']} tok/s | gamma_ver: {eager_sum['SPADE']['gamma_ver']} -> {opt_sum['SPADE']['gamma_ver']}", flush=True)
    del r_full, r_spade, pair_opt
    _cleanup_cuda(reset_dynamo=True)


# ---------------------------------------------------------------------------
# Block 2: heldout300_interleaved_r3 (Full N=300 R=3 Counterbalanced 5-Arm)
# ---------------------------------------------------------------------------

def run_heldout300_r3_block() -> None:
    print("\n=== [BLOCK: heldout300_interleaved_r3] R=3 counterbalanced N=300 on compiled stack ===", flush=True)
    rows = _load_jsonl(HELD300_PATH)
    assert len(rows) == 300
    tok = AutoTokenizer.from_pretrained(QWEN3_8B_SNAP, local_files_only=True)
    max_ptok = _max_prompt_tokens(tok, rows)
    full_cap = ((max_ptok + 384 + 32 + 127) // 128) * 128
    compact_cap = 6144

    pool = torch.cuda.graph_pool_handle()
    pair = QwenPair(
        QWEN3_8B_SNAP, QWEN3_06B_SNAP, local_files_only=True, dtype="bfloat16",
        draft_attn_implementation="flex_attention", target_attn_implementation="flex_attention",
    )
    pair.prepare_target_runner(mode="compile_cudagraph", max_cache_len=full_cap, max_verify_len=5, warmup_steps=3, graph_pool=pool)
    runner_full = StaticDraftRunner(pair.draft, max_cache_len=full_cap, mode="compile_cudagraph", device=pair._input_device(pair.draft), graph_pool=pool)
    runner_full.prepare(warmup_steps=3)
    runner_compact = StaticDraftRunner(pair.draft, max_cache_len=compact_cap, mode="compile_cudagraph", device=pair._input_device(pair.draft), graph_pool=pool)
    runner_compact.prepare(warmup_steps=3)

    # Warmup every arm outside timed passes
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
    all_pass_recs: dict[str, list[dict[str, dict[str, Any]]]] = {a: [] for a in arms}

    for pass_idx in range(3):
        pass_seed = 42 + pass_idx
        pass_ckpt = REV9_DIR / f"heldout300_r3_pass{pass_idx}.json"
        if pass_ckpt.exists():
            pdata = json.loads(pass_ckpt.read_text(encoding="utf-8"))
            print(f"  SKIP Pass {pass_idx+1}/3 (loaded {pass_ckpt})", flush=True)
        else:
            print(f"  START Pass {pass_idx+1}/3 (seed={pass_seed})...", flush=True)
            t_p0 = time.perf_counter()
            per_arm_recs: dict[str, dict[str, dict[str, Any]]] = {a: {} for a in arms}
            for idx, row in enumerate(rows):
                cid = str(row["claim_id"])
                perm = list(arms)
                random.Random(1000 * (pass_idx + 1) + idx).shuffle(perm)
                for arm in perm:
                    torch.cuda.reset_peak_memory_stats()
                    if arm == "Target AR (B0-S)":
                        g = pair.generate(row["prompt"], mode="B0-S", max_new_tokens=384, temperature=0.7, top_p=0.9, seed=pass_seed + idx, run_id=f"h300-p{pass_idx}-ar", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                    elif arm == "Baseline Full":
                        pair.draft_runner = runner_full
                        g = pair.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=pass_seed + idx, run_id=f"h300-p{pass_idx}-full", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                    elif arm == "Sink+Recent Trunc. (B=2,529)":
                        pair.draft_runner = runner_compact
                        ev_chars_row = sum(len(str(e.get("text", ""))) for e in (row.get("evidence") or []))
                        if ev_chars_row <= 10000:
                            g = pair.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=pass_seed + idx, run_id=f"h300-p{pass_idx}-sr", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                        else:
                            g = pair.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=pass_seed + idx, run_id=f"h300-p{pass_idx}-sr", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast", streaming_llm_cfg={"kv_budget": 2529, "sink_tokens": 256})
                    elif arm == "BM25 Sent-B10k":
                        pair.draft_runner = runner_compact
                        dctx = build_draft_context(row, policy="sentence_budget", budget_chars=10000, min_sentences_per_passage=1)
                        g = pair.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=pass_seed + idx, run_id=f"h300-p{pass_idx}-bm25", claim_id=cid, draft_prompt=dctx.draft_prompt, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                        g.timings["context_selection_s"] = dctx.context_selection_s
                    else:
                        pair.draft_runner = runner_compact
                        g = pair.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=pass_seed + idx, run_id=f"h300-p{pass_idx}-spade", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": row, "budget_chars": 10000, "tail_window": 16})
                    per_arm_recs[arm][cid] = _extract_gen_record(cid, g, row, peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3)
                if (idx + 1) % 60 == 0 or (idx + 1) == len(rows):
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

    pooled_claims = _pool_r3_claims(all_pass_recs)
    summary_rows = []
    for a in arms:
        e2es = [ps[a]["e2e_tok_s"] for ps in pass_summaries]
        decs = [ps[a]["dec_tok_s"] for ps in pass_summaries]
        alps = [ps[a]["alpha_eval_pct"] for ps in pass_summaries]
        aprops = [ps[a]["alpha_prop_pct"] for ps in pass_summaries]
        pooled = _aggregate_metrics(list(pooled_claims[a].values()))
        summary_rows.append({
            "label": a,
            "e2e_mean": round(statistics.mean(e2es), 2),
            "e2e_sd": round(statistics.stdev(e2es), 2),
            "dec_mean": round(statistics.mean(decs), 2),
            "dec_sd": round(statistics.stdev(decs), 2),
            "alpha_eval_mean": round(statistics.mean(alps), 2),
            "alpha_eval_sd": round(statistics.stdev(alps), 2),
            "alpha_prop_mean": round(statistics.mean(aprops), 2),
            "tok_per_round": pooled["tok_per_round"],
            "draft_tok": pooled["draft_tok"],
            "draft_kv_mb": pooled["draft_kv_mb"],
            "target_kv_mb": pooled["target_kv_mb"],
            "peak_vram_gb": pooled["peak_vram_gb"],
            "sel_ms_claim": pooled["sel_ms_claim"],
            "decomp_ms_tok": pooled["decomp_ms_tok"],
            "verify_sub_ms_round": pooled["verify_sub_ms_round"],
            "pass_e2es": e2es,
        })

    # Family-wise Holm contrasts on E2E tok/s (SPADE vs each other arm) + Full vs AR
    e2e_vs_spade = []
    for b_lbl in ("Target AR (B0-S)", "Baseline Full", "Sink+Recent Trunc. (B=2,529)", "BM25 Sent-B10k"):
        c = _paired_boot(pooled_claims[b_lbl], pooled_claims["SPADE B=10k"], "e2e_tok_s")
        c["contrast"] = f"SPADE vs {b_lbl}"
        e2e_vs_spade.append(c)
    _holm_adjust(e2e_vs_spade)

    alpha_vs_spade = []
    for b_lbl in ("Baseline Full", "Sink+Recent Trunc. (B=2,529)", "BM25 Sent-B10k"):
        c = _paired_boot(pooled_claims[b_lbl], pooled_claims["SPADE B=10k"], "alpha_eval_pct")
        c["contrast"] = f"SPADE vs {b_lbl}"
        alpha_vs_spade.append(c)
    _holm_adjust(alpha_vs_spade)

    # Context-stratified breakdown from the SAME unified R=3 dataset
    strata_preds = {
        "bypass_le_10k": lambda x: x <= 10000,
        "selective_gt_10k": lambda x: x > 10000,
        "dense_ge_20k": lambda x: x >= 20000,
        "full_all": lambda _x: True,
    }
    strata_out: dict[str, Any] = {}
    for skey, pred in strata_preds.items():
        cids_s = [c for c, r in pooled_claims["SPADE B=10k"].items() if pred(r["ev_chars"])]
        ar_m = {c: pooled_claims["Target AR (B0-S)"][c] for c in cids_s}
        fl_m = {c: pooled_claims["Baseline Full"][c] for c in cids_s}
        sr_m = {c: pooled_claims["Sink+Recent Trunc. (B=2,529)"][c] for c in cids_s}
        bm_m = {c: pooled_claims["BM25 Sent-B10k"][c] for c in cids_s}
        sp_m = {c: pooled_claims["SPADE B=10k"][c] for c in cids_s}
        # Also per-pass SD within stratum
        sp_pass_e2e = [_aggregate_metrics([all_pass_recs["SPADE B=10k"][p][c] for c in cids_s])["e2e_tok_s"] for p in range(3)]
        ar_pass_e2e = [_aggregate_metrics([all_pass_recs["Target AR (B0-S)"][p][c] for c in cids_s])["e2e_tok_s"] for p in range(3)]
        fl_pass_e2e = [_aggregate_metrics([all_pass_recs["Baseline Full"][p][c] for c in cids_s])["e2e_tok_s"] for p in range(3)]
        strata_out[skey] = {
            "n_claims": len(cids_s),
            "AR": {**_aggregate_metrics(list(ar_m.values())), "e2e_sd": round(statistics.stdev(ar_pass_e2e), 2)},
            "Full": {**_aggregate_metrics(list(fl_m.values())), "e2e_sd": round(statistics.stdev(fl_pass_e2e), 2)},
            "SinkRecent": _aggregate_metrics(list(sr_m.values())),
            "BM25": _aggregate_metrics(list(bm_m.values())),
            "SPADE": {**_aggregate_metrics(list(sp_m.values())), "e2e_sd": round(statistics.stdev(sp_pass_e2e), 2)},
            "spade_vs_ar_e2e": _paired_boot(ar_m, sp_m, "e2e_tok_s"),
            "spade_vs_full_e2e": _paired_boot(fl_m, sp_m, "e2e_tok_s"),
            "full_vs_ar_e2e": _paired_boot(ar_m, fl_m, "e2e_tok_s"),
            "spade_vs_bm25_e2e": _paired_boot(bm_m, sp_m, "e2e_tok_s"),
            "spade_vs_sr_e2e": _paired_boot(sr_m, sp_m, "e2e_tok_s"),
            "spade_vs_bm25_alpha": _paired_boot(bm_m, sp_m, "alpha_eval_pct"),
            "spade_vs_sr_alpha": _paired_boot(sr_m, sp_m, "alpha_eval_pct"),
        }

    # Pass-0 verdict & Macro-F1 audit (plus bootstrap CI on Macro-F1 diffs vs AR)
    pass0 = json.loads((REV9_DIR / "heldout300_r3_pass0.json").read_text(encoding="utf-8"))["claims_by_arm"]
    verdict_audit: dict[str, Any] = {}
    for arm in ("Target AR (B0-S)", "Baseline Full", "SPADE B=10k"):
        recs_a = list(pass0[arm].values())
        golds = [str(r["gold_verdict"]) for r in recs_a if r.get("pred_verdict")]
        preds = [str(r["pred_verdict"]) for r in recs_a if r.get("pred_verdict")]
        cm = classification_metrics(golds, preds)
        verdict_audit[arm] = {
            "macro_f1": round(cm["macro_f1"], 4),
            "json_strict_pct": round(sum(1 for r in recs_a if r["json_valid_strict"]) / len(recs_a) * 100.0, 2),
            "counts": dict(Counter(preds)),
        }

    report = {
        "n_claims": len(rows),
        "n_passes": 3,
        "design": "compiled_target_and_draft_per_claim_randomized_5arm_R3",
        "rows": summary_rows,
        "pairwise_e2e_holm": {c["contrast"]: c for c in e2e_vs_spade},
        "pairwise_alpha_holm": {c["contrast"]: c for c in alpha_vs_spade},
        "strata": strata_out,
        "verdict_pass0": verdict_audit,
        "pass_summaries": pass_summaries,
    }
    _write_json(REV9_DIR / "heldout300_interleaved_r3.json", report)
    print("  DONE heldout300_interleaved_r3:", [(r["label"], r["e2e_mean"], r["e2e_sd"]) for r in summary_rows], flush=True)
    del runner_full, runner_compact, pair
    _cleanup_cuda(reset_dynamo=True)


# ---------------------------------------------------------------------------
# Block 1B: ratio_and_cross_family (Same-family + Cross-family on Compiled Target)
# ---------------------------------------------------------------------------

def _eval_model_pair_compiled(
    name: str,
    target_snap: str,
    draft_snap: str,
    ratio_str: str,
    rows: list[dict[str, Any]],
    *,
    run_full: bool = True,
) -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(target_snap, local_files_only=True)
    max_ptok = _max_prompt_tokens(tok, rows)
    full_cap = ((max_ptok + 384 + 32 + 127) // 128) * 128
    spade_cap = min(full_cap, 6144)

    t_cfg = json.loads((Path(target_snap) / "config.json").read_text())
    d_cfg = json.loads((Path(draft_snap) / "config.json").read_text())

    # Prefer flex_attention + compile_cudagraph; fall back to sdpa + compile_cudagraph if needed
    pair = QwenPair(
        target_snap, draft_snap, local_files_only=True, dtype="bfloat16",
        draft_attn_implementation="flex_attention", target_attn_implementation="flex_attention",
    )
    t_params = sum(p.numel() for p in pair.target.parameters())
    d_params = sum(p.numel() for p in pair.draft.parameters())
    exact_ratio = t_params / d_params
    pool = torch.cuda.graph_pool_handle()

    pair.prepare_target_runner(mode="compile_cudagraph", max_cache_len=full_cap, max_verify_len=5, warmup_steps=3, graph_pool=pool)
    rec_by_arm: dict[str, dict[str, dict[str, Any]]] = {"AR": {}, "SPADE": {}, "Full": {}}
    full_error: str | None = None

    can_hold_both = (d_params < 2.5e9) and run_full
    r_spade = r_full = None
    if can_hold_both:
        try:
            r_spade = StaticDraftRunner(pair.draft, max_cache_len=spade_cap, mode="compile_cudagraph", device=pair._input_device(pair.draft), graph_pool=pool)
            r_spade.prepare(warmup_steps=3)
            r_full = StaticDraftRunner(pair.draft, max_cache_len=full_cap, mode="compile_cudagraph", device=pair._input_device(pair.draft), graph_pool=pool)
            r_full.prepare(warmup_steps=3)
        except torch.OutOfMemoryError:
            can_hold_both = False
            del r_spade, r_full
            r_spade = r_full = None
            _cleanup_cuda()

    if can_hold_both:
        assert r_spade is not None and r_full is not None
        for gw in range(2):
            pair.generate(rows[0]["prompt"], mode="B0-S", max_new_tokens=16, temperature=0.7, top_p=0.9, seed=700 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast")
            pair.draft_runner = r_spade
            pair.generate(rows[0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=710 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": rows[0], "budget_chars": 10000, "tail_window": 16})
            pair.draft_runner = r_full
            pair.generate(rows[0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=720 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast")

        rng_ord = random.Random(42)
        for idx, row in enumerate(rows):
            cid = str(row["claim_id"])
            arms = ["AR", "SPADE", "Full"]
            rng_ord.shuffle(arms)
            for arm in arms:
                torch.cuda.reset_peak_memory_stats()
                if arm == "AR":
                    g = pair.generate(row["prompt"], mode="B0-S", max_new_tokens=384, temperature=0.7, top_p=0.9, seed=42 + idx, run_id=f"{name}-ar", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                elif arm == "SPADE":
                    pair.draft_runner = r_spade
                    g = pair.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + idx, run_id=f"{name}-spade", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": row, "budget_chars": 10000, "tail_window": 16})
                else:
                    pair.draft_runner = r_full
                    g = pair.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + idx, run_id=f"{name}-full", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                rec_by_arm[arm][cid] = _extract_gen_record(cid, g, row, peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3)
            if (idx + 1) % 50 == 0 or (idx + 1) == len(rows):
                print(f"    [{name}] {idx+1}/{len(rows)}", flush=True)
        del r_spade, r_full
    else:
        # Interleave AR + SPADE cleanly first (Target 8B + 4B compact @6k fits easily), then try Full-SD alone
        r_sp = StaticDraftRunner(pair.draft, max_cache_len=spade_cap, mode="compile_cudagraph", device=pair._input_device(pair.draft), graph_pool=pool)
        r_sp.prepare(warmup_steps=3)
        for gw in range(2):
            pair.generate(rows[0]["prompt"], mode="B0-S", max_new_tokens=16, temperature=0.7, top_p=0.9, seed=700 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast")
            pair.draft_runner = r_sp
            pair.generate(rows[0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=710 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": rows[0], "budget_chars": 10000, "tail_window": 16})

        rng_ord = random.Random(42)
        for idx, row in enumerate(rows):
            cid = str(row["claim_id"])
            arms = ["AR", "SPADE"]
            rng_ord.shuffle(arms)
            for arm in arms:
                torch.cuda.reset_peak_memory_stats()
                if arm == "AR":
                    g = pair.generate(row["prompt"], mode="B0-S", max_new_tokens=384, temperature=0.7, top_p=0.9, seed=42 + idx, run_id=f"{name}-ar", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                else:
                    pair.draft_runner = r_sp
                    g = pair.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + idx, run_id=f"{name}-spade", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": row, "budget_chars": 10000, "tail_window": 16})
                rec_by_arm[arm][cid] = _extract_gen_record(cid, g, row, peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3)
            if (idx + 1) % 25 == 0 or (idx + 1) == len(rows):
                print(f"    [{name} AR/SPADE] {idx+1}/{len(rows)}", flush=True)
        pair.draft_runner = None
        del r_sp
        _cleanup_cuda()

        if run_full:
            r_fl = None
            try:
                r_fl = StaticDraftRunner(pair.draft, max_cache_len=full_cap, mode="compile_cudagraph", device=pair._input_device(pair.draft), graph_pool=pool)
                r_fl.prepare(warmup_steps=3)
                pair.draft_runner = r_fl
                pair.generate(rows[0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=720, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                for idx, row in enumerate(rows):
                    cid = str(row["claim_id"])
                    torch.cuda.reset_peak_memory_stats()
                    g = pair.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + idx, run_id=f"{name}-full", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                    rec_by_arm["Full"][cid] = _extract_gen_record(cid, g, row, peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3)
                    if (idx + 1) % 25 == 0 or (idx + 1) == len(rows):
                        print(f"    [{name} Full] {idx+1}/{len(rows)}", flush=True)
            except torch.OutOfMemoryError as exc:
                full_error = str(exc)[:220]
                rec_by_arm["Full"].clear()
            finally:
                pair.draft_runner = None
                if r_fl is not None:
                    del r_fl
                _cleanup_cuda()

    ar_sum = _aggregate_metrics(list(rec_by_arm["AR"].values()))
    sp_sum = _aggregate_metrics(list(rec_by_arm["SPADE"].values()))
    fl_sum = _aggregate_metrics(list(rec_by_arm["Full"].values())) if len(rec_by_arm["Full"]) == len(rows) else None
    ar_step_ms = (ar_sum["total_decode_s"] / ar_sum["total_output_tokens"]) * 1000.0 if ar_sum["total_output_tokens"] > 0 else 0.0

    res = {
        "name": name,
        "ratio_label": ratio_str,
        "exact_param_ratio": round(exact_ratio, 2),
        "n_claims": len(rows),
        "target_layers": int(t_cfg["num_hidden_layers"]),
        "draft_layers": int(d_cfg["num_hidden_layers"]),
        "AR": ar_sum,
        "SPADE": {**sp_sum, **_leviathan_row(sp_sum, ar_step_ms, ar_sum["dec_tok_s"], ar_sum["e2e_tok_s"])},
        "Full": ({**fl_sum, **_leviathan_row(fl_sum, ar_step_ms, ar_sum["dec_tok_s"], ar_sum["e2e_tok_s"])} if fl_sum else {"error": full_error}),
        "spade_vs_ar": {m: _paired_boot(rec_by_arm["AR"], rec_by_arm["SPADE"], m) for m in ("e2e_tok_s", "dec_tok_s")},
        "full_vs_ar": ({m: _paired_boot(rec_by_arm["AR"], rec_by_arm["Full"], m) for m in ("e2e_tok_s", "dec_tok_s")} if fl_sum else None),
        "spade_vs_full": ({m: _paired_boot(rec_by_arm["Full"], rec_by_arm["SPADE"], m) for m in ("e2e_tok_s", "dec_tok_s")} if fl_sum else None),
    }
    pair.target_runner = None
    pair.draft_runner = None
    del pair
    _cleanup_cuda(reset_dynamo=True)
    return res


def run_ratio_and_cross_family_block() -> None:
    from huggingface_hub import snapshot_download

    print("\n=== [BLOCK: ratio_and_cross_family] Optimized Capacity Ratio & Cross-Family Evaluation ===", flush=True)
    rows100 = _stratified_heldout100()
    rows300 = _load_jsonl(HELD300_PATH)
    ckpt_path = REV9_DIR / "ratio_and_cross_family_opt.json"
    out_data: dict[str, Any] = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.exists() else {"pairs": {}}

    specs = [
        ("Qwen3-8B / 0.6B", QWEN3_8B_SNAP, QWEN3_06B_SNAP, "1:13.3", rows100),
        ("Qwen3-8B / 1.7B", QWEN3_8B_SNAP, snapshot_download("Qwen/Qwen3-1.7B", cache_dir=CACHE_DIR, local_files_only=True), "1:4.7", rows100),
        ("Qwen3-8B / 4B", QWEN3_8B_SNAP, snapshot_download("Qwen/Qwen3-4B", cache_dir=CACHE_DIR, local_files_only=True), "1:2.0", rows100),
        (
            "Qwen2.5-7B / 0.5B",
            snapshot_download("Qwen/Qwen2.5-7B-Instruct", cache_dir=CACHE_DIR, local_files_only=True),
            snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", cache_dir=CACHE_DIR, local_files_only=True),
            "1:14.0",
            rows300,
        ),
        (
            "Llama-3.1-8B / 3.2-1B",
            snapshot_download("unsloth/Meta-Llama-3.1-8B-Instruct", cache_dir=CACHE_DIR, local_files_only=True),
            snapshot_download("unsloth/Llama-3.2-1B-Instruct", cache_dir=CACHE_DIR, local_files_only=True),
            "1:8.0",
            rows300,
        ),
    ]
    for label, t_snap, d_snap, r_str, r_list in specs:
        if label in out_data["pairs"]:
            print(f"  SKIP {label} (already in {ckpt_path})", flush=True)
            continue
        print(f"  Evaluating {label} ({r_str}, N={len(r_list)})...", flush=True)
        out_data["pairs"][label] = _eval_model_pair_compiled(label, t_snap, d_snap, r_str, r_list)
        _write_json(ckpt_path, out_data)
        p = out_data["pairs"][label]
        print(f"  DONE {label}: AR={p['AR']['e2e_tok_s']} | SPADE={p['SPADE']['e2e_tok_s']} | gamma={p['SPADE']['gamma_ver']} | c_AR={p['SPADE']['c_ar']}", flush=True)


# ---------------------------------------------------------------------------
# Block 3: batched_e2e_concurrency (Co-Located Batched E2E Serving in 32 GB)
# ---------------------------------------------------------------------------

def run_batched_e2e_concurrency_block() -> None:
    print("\n=== [BLOCK: batched_e2e_concurrency] Co-located batched E2E throughput in 32 GB ===", flush=True)
    ckpt_path = REV9_DIR / "batched_e2e_concurrency.json"
    report: dict[str, Any] = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.exists() else {"results": {}}

    bench_rows = _load_jsonl(HELD300_PATH)
    pool_passages = [str(e.get("text", "")).strip() for r in bench_rows for e in (r.get("evidence") or []) if len(str(e.get("text", "")).strip()) >= 1500]
    base_pool = sorted(bench_rows, key=lambda r: sum(len(str(e.get("text", ""))) for e in (r.get("evidence") or [])))[:16]
    tok = AutoTokenizer.from_pretrained(QWEN3_8B_SNAP, local_files_only=True)
    prompts_by_len = {
        L: [_build_padded_prompt(tok, L, base_pool[i], pool_passages, start_offset=i * 7) for i in range(16)]
        for L in (8192, 16384)
    }

    target = AutoModelForCausalLM.from_pretrained(QWEN3_8B_SNAP, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", local_files_only=True).eval()
    draft: Any = None

    def _run_pol_sweep(ctx_len: int, pol_name: str, budget_chars: int | None, coarse_batches: list[int]) -> None:
        key = f"{ctx_len}|{pol_name}"
        if key in report["results"]:
            print(f"  SKIP {key}", flush=True)
            return
        rows_out: list[dict[str, Any]] = []
        last_ok = 0
        first_oom: int | None = None

        def _try_bs(bs: int) -> dict[str, Any] | None:
            nonlocal first_oom, last_ok
            _cleanup_cuda(reset_dynamo=True)
            try:
                if budget_chars == -1:
                    res = run_single_batched_ar(target, tok, prompts_by_len[ctx_len], bs=bs, max_new_tokens=128)
                else:
                    assert draft is not None
                    res = run_single_batched_sd(target, draft, tok, prompts_by_len[ctx_len], bs=bs, policy_label=pol_name, budget_chars=budget_chars, max_new_tokens=128)
                last_ok = max(last_ok, bs)
                print(f"  [{key} bs={bs:2d}] E2E={res['agg_e2e_tok_s']:6.1f} Dec={res['agg_dec_tok_s']:6.1f} peak={res['peak_vram_gb']:.2f}GB T_KV={res['target_kv_gb']:.2f} D_KV={res['draft_kv_gb']:.2f}", flush=True)
                return res
            except torch.OutOfMemoryError as exc:
                first_oom = bs if first_oom is None else min(first_oom, bs)
                print(f"  [{key} bs={bs:2d}] OOM ({str(exc)[:80]})", flush=True)
                _cleanup_cuda(reset_dynamo=True)
                return None

        for bs in coarse_batches:
            res = _try_bs(bs)
            if res is None:
                rows_out.append({"policy": pol_name, "batch_size": bs, "oom": True})
                break
            rows_out.append(res)

        # Refine exact integer max batch size between last_ok and first_oom
        if first_oom is not None and first_oom - last_ok > 1:
            for probe_bs in range(last_ok + 1, first_oom):
                res = _try_bs(probe_bs)
                if res is None:
                    rows_out.append({"policy": pol_name, "batch_size": probe_bs, "oom": True})
                    break
                rows_out.append(res)

        rows_out.sort(key=lambda r: r["batch_size"])
        report["results"][key] = {"context_len": ctx_len, "policy": pol_name, "max_batch": last_ok, "rows": rows_out}
        _write_json(ckpt_path, report)

    # First run Target AR Only (8B alone, before loading 0.6B weights!)
    for L in (8192, 16384):
        _run_pol_sweep(L, "Target AR (8B Only)", -1, [1, 2, 4, 6, 8, 10, 12, 14] if L == 8192 else [1, 2, 4, 5, 6, 8])

    # Co-locate Draft 0.6B weights for Full-SD and SPADE
    draft = AutoModelForCausalLM.from_pretrained(QWEN3_06B_SNAP, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", local_files_only=True).eval()
    for L in (8192, 16384):
        _run_pol_sweep(L, "Full-SD (8B+0.6B)", None, [1, 2, 4, 6, 8] if L == 8192 else [1, 2, 3, 4, 5])
        _run_pol_sweep(L, "SPADE B=10k (8B+0.6B)", 10000, [1, 2, 4, 6, 8, 10, 12] if L == 8192 else [1, 2, 4, 5, 6, 8])
        _run_pol_sweep(L, "SPADE B=2k (8B+0.6B)", 2000, [1, 2, 4, 6, 8, 10, 12] if L == 8192 else [1, 2, 4, 5, 6, 8])

    del draft, target
    _cleanup_cuda(reset_dynamo=True)


# ---------------------------------------------------------------------------
# Block 4: fixed_bm25_interaction (Save Per-Claim Records for Interaction Audits)
# ---------------------------------------------------------------------------

def run_fixed_bm25_interaction_block() -> None:
    print("\n=== [BLOCK: fixed_bm25_interaction] Per-claim fixed-prefix records ===", flush=True)
    tok = AutoTokenizer.from_pretrained(QWEN3_8B_SNAP, local_files_only=True)
    target = AutoModelForCausalLM.from_pretrained(QWEN3_8B_SNAP, dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()
    draft = AutoModelForCausalLM.from_pretrained(QWEN3_06B_SNAP, dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()
    upper50 = resolve_layer_indices(len(target.model.layers), "upper50")

    b0s_300 = {str(r["claim_id"]): r for r in _load_jsonl(Path("outputs/runs/b1-4-heldout300-b0s/predictions.jsonl"))}
    b0s_lb = {str(r["claim_id"]): r for r in _load_jsonl(Path("outputs/runs/v6_longbench/lb-hotpot-b0s/predictions.jsonl"))}
    suites = [("Long-173", _load_jsonl(LONG173_PATH), b0s_300), ("HotpotQA-100", _load_jsonl(HOTPOT100_PATH), b0s_lb)]

    out_records: dict[str, list[dict[str, Any]]] = {}
    for sname, rows, traj_map in suites:
        per_claim: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            cid = str(row["claim_id"])
            tok_ids = list(traj_map[cid]["token_ids"])
            if not tok_ids:
                continue
            g_len = len(tok_ids)
            ser_full = format_qwen_prompt(tok, row["prompt"])
            enc = tok(ser_full, return_offsets_mapping=True, add_special_tokens=False)
            p_ids = list(enc["input_ids"])
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
            spans = find_evidence_and_sentence_token_spans(ser_full, list(row.get("evidence") or []), offsets)

            inp_p = torch.tensor([p_ids], dtype=torch.long, device="cuda")
            with TargetPrefillAttentionHook(target, layer_indices=upper50, tail_window=16) as hook:
                with torch.inference_mode():
                    pre = target(input_ids=inp_p, use_cache=True, logits_to_keep=1)
                imp, _ = hook.compute_token_importance(pre.past_key_values)
            del inp_p, pre
            _, s_scores = score_evidence_from_token_importance(imp, spans, sentence_agg="sum")
            spade_dctx = build_draft_context_from_scores(row, policy="sentence_budget", sentence_scores_by_passage=s_scores, budget_chars=10000, min_sentences_per_passage=1)
            bm25_dctx = build_draft_context(row, policy="sentence_budget", budget_chars=10000, min_sentences_per_passage=1)

            t_inp = torch.tensor([p_ids + tok_ids[:-1]], dtype=torch.long, device="cuda")
            with torch.inference_mode():
                t_logits = target(input_ids=t_inp, use_cache=False, logits_to_keep=g_len).logits[0].float()
            p_mat = processed_distribution_fast(t_logits, temperature=0.7, top_p=0.9, dtype=torch.float32)
            del t_inp, t_logits

            pol_stats = {}
            for pname, d_prompt in [("BM25 Sent-B10k", bm25_dctx.draft_prompt), ("SPADE (Attn B=10k)", spade_dctx.draft_prompt)]:
                d_ids = tok(format_qwen_prompt(tok, d_prompt), add_special_tokens=False)["input_ids"]
                d_inp = torch.tensor([d_ids + tok_ids[:-1]], dtype=torch.long, device="cuda")
                with torch.inference_mode():
                    d_logits = draft(input_ids=d_inp, use_cache=False, logits_to_keep=g_len).logits[0].float()
                q_mat = processed_distribution_fast(d_logits, temperature=0.7, top_p=0.9, dtype=torch.float32)
                acc_sum = float(torch.minimum(p_mat, q_mat).sum(dim=-1).sum().item())
                pol_stats[pname] = {"draft_tok": len(d_ids), "exp_acc_sum": acc_sum}
                del d_inp, d_logits, q_mat
            del p_mat

            lex = _compute_lexical_features(row)
            bm_pct = (pol_stats["BM25 Sent-B10k"]["exp_acc_sum"] / g_len) * 100.0
            sp_pct = (pol_stats["SPADE (Attn B=10k)"]["exp_acc_sum"] / g_len) * 100.0
            per_claim.append({
                "claim_id": cid,
                "n_prefixes": g_len,
                **lex,
                "bm25_acc_sum": pol_stats["BM25 Sent-B10k"]["exp_acc_sum"],
                "spade_acc_sum": pol_stats["SPADE (Attn B=10k)"]["exp_acc_sum"],
                "bm25_acc_pct": round(bm_pct, 4),
                "spade_acc_pct": round(sp_pct, 4),
                "delta_pp": round(sp_pct - bm_pct, 4),
                "bm25_tok": pol_stats["BM25 Sent-B10k"]["draft_tok"],
                "spade_tok": pol_stats["SPADE (Attn B=10k)"]["draft_tok"],
            })
            if (idx + 1) % 35 == 0 or (idx + 1) == len(rows):
                print(f"    [{sname}] {idx+1}/{len(rows)}", flush=True)
                _cleanup_cuda()
        out_records[sname] = per_claim

    _write_json(REV9_DIR / "fixed_bm25_claim_records.json", out_records)
    del target, draft
    _cleanup_cuda()


# ---------------------------------------------------------------------------
# Block 5: scaling_2k_32k (2k, 4k, 8k, 10k, 12k, 14k, 16k, 32k on Compiled Stack)
# ---------------------------------------------------------------------------

def run_scaling_2k_32k_block() -> None:
    print("\n=== [BLOCK: scaling_2k_32k] Interleaved 2k-32k (incl 10k/12k/14k) on compiled stack ===", flush=True)
    ckpt_path = REV9_DIR / "scaling_2k_32k_opt.json"
    results: dict[str, Any] = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.exists() else {}

    bench_rows = _load_jsonl(HELD300_PATH)
    pool_passages = [str(e.get("text", "")).strip() for r in bench_rows for e in (r.get("evidence") or []) if len(str(e.get("text", "")).strip()) >= 1500]
    base_pool = sorted(bench_rows, key=lambda r: sum(len(str(e.get("text", ""))) for e in (r.get("evidence") or [])))[:20]
    tok = AutoTokenizer.from_pretrained(QWEN3_8B_SNAP, local_files_only=True)

    lengths = [2048, 4096, 8192, 10240, 12288, 14336, 16384, 32768]
    n_by_len = {L: (10 if L == 32768 else 20) for L in lengths}
    prompts_by_len = {
        L: [_build_padded_prompt(tok, L, base_pool[i], pool_passages, start_offset=i * 7) for i in range(n_by_len[L])]
        for L in lengths
    }

    # Part A: 2k..16k reuse ONE compiled target cap=17024 + ONE spade cap=6144 (+ full_cap per len or shared 17024!)
    sub_16k = [L for L in lengths if L <= 16384 and str(L) not in results]
    if sub_16k:
        cap16 = 17024  # >= max 16442 + 128 + headroom; % 128 == 0!
        pool = torch.cuda.graph_pool_handle()
        pair = QwenPair(
            QWEN3_8B_SNAP, QWEN3_06B_SNAP, local_files_only=True, dtype="bfloat16",
            draft_attn_implementation="flex_attention", target_attn_implementation="flex_attention",
        )
        pair.prepare_target_runner(mode="compile_cudagraph", max_cache_len=cap16, max_verify_len=5, warmup_steps=3, graph_pool=pool)
        r_full = StaticDraftRunner(pair.draft, max_cache_len=cap16, mode="compile_cudagraph", device=pair._input_device(pair.draft), graph_pool=pool)
        r_full.prepare(warmup_steps=3)
        r_spade = StaticDraftRunner(pair.draft, max_cache_len=6144, mode="compile_cudagraph", device=pair._input_device(pair.draft), graph_pool=pool)
        r_spade.prepare(warmup_steps=3)
        for gw in range(2):
            pair.generate(prompts_by_len[4096][0]["prompt"], mode="B0-S", max_new_tokens=16, temperature=0.7, top_p=0.9, seed=99 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast")
            pair.draft_runner = r_full
            pair.generate(prompts_by_len[4096][0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=109 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast")
            pair.draft_runner = r_spade
            pair.generate(prompts_by_len[4096][0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=119 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": prompts_by_len[4096][0], "budget_chars": 10000, "tail_window": 16})

        for L in sub_16k:
            plist = prompts_by_len[L]
            recs: dict[str, dict[str, dict[str, Any]]] = {"AR": {}, "Full": {}, "SPADE": {}}
            rng = random.Random(42 + L)
            for i, row in enumerate(plist):
                cid = str(row["claim_id"])
                arms = ["AR", "Full", "SPADE"]
                rng.shuffle(arms)
                for arm in arms:
                    torch.cuda.reset_peak_memory_stats()
                    if arm == "AR":
                        g = pair.generate(row["prompt"], mode="B0-S", max_new_tokens=128, temperature=0.7, top_p=0.9, seed=42 + i, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                    elif arm == "Full":
                        pair.draft_runner = r_full
                        g = pair.generate(row["prompt"], mode="B2", max_new_tokens=128, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + i, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                    else:
                        pair.draft_runner = r_full if L == 2048 else r_spade
                        g = pair.generate(row["prompt"], mode="B2", max_new_tokens=128, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + i, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": row, "budget_chars": 10000, "tail_window": 16})
                    recs[arm][cid] = _extract_gen_record(cid, g, row, peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3)
            ar_s = _aggregate_metrics(list(recs["AR"].values()))
            fl_s = _aggregate_metrics(list(recs["Full"].values()))
            sp_s = _aggregate_metrics(list(recs["SPADE"].values()))
            results[str(L)] = {
                "target_tokens": L,
                "actual_tokens": round(statistics.mean(r["actual_tokens"] for r in plist)),
                "n_prompts": len(plist),
                "AR": ar_s,
                "Full": fl_s,
                "SPADE": sp_s,
                "spade_vs_ar_e2e": _paired_boot(recs["AR"], recs["SPADE"], "e2e_tok_s"),
                "spade_vs_full_e2e": _paired_boot(recs["Full"], recs["SPADE"], "e2e_tok_s"),
                "full_vs_ar_e2e": _paired_boot(recs["AR"], recs["Full"], "e2e_tok_s"),
            }
            _write_json(ckpt_path, results)
            print(f"  DONE L={L}: AR={ar_s['e2e_tok_s']} | Full={fl_s['e2e_tok_s']} | SPADE={sp_s['e2e_tok_s']} | d_AR={results[str(L)]['spade_vs_ar_e2e']['delta']:+.2f}", flush=True)
        pair.target_runner = None
        pair.draft_runner = None
        del r_full, r_spade, pair
        _cleanup_cuda(reset_dynamo=True)

    # Part B: 32k (N=10, cap32 = 33792)
    if "32768" not in results:
        cap32 = 33792
        plist = prompts_by_len[32768]
        pool = torch.cuda.graph_pool_handle()
        pair = QwenPair(
            QWEN3_8B_SNAP, QWEN3_06B_SNAP, local_files_only=True, dtype="bfloat16",
            draft_attn_implementation="flex_attention", target_attn_implementation="flex_attention",
        )
        pair.prepare_target_runner(mode="compile_cudagraph", max_cache_len=cap32, max_verify_len=5, warmup_steps=3, graph_pool=pool)
        r_spade = StaticDraftRunner(pair.draft, max_cache_len=6144, mode="compile_cudagraph", device=pair._input_device(pair.draft), graph_pool=pool)
        r_spade.prepare(warmup_steps=3)
        recs32: dict[str, dict[str, dict[str, Any]]] = {"AR": {}, "SPADE": {}}
        rng = random.Random(42 + 32768)
        for i, row in enumerate(plist):
            cid = str(row["claim_id"])
            arms = ["AR", "SPADE"]
            rng.shuffle(arms)
            for arm in arms:
                torch.cuda.reset_peak_memory_stats()
                if arm == "AR":
                    g = pair.generate(row["prompt"], mode="B0-S", max_new_tokens=128, temperature=0.7, top_p=0.9, seed=42 + i, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                else:
                    pair.draft_runner = r_spade
                    g = pair.generate(row["prompt"], mode="B2", max_new_tokens=128, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + i, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": row, "budget_chars": 10000, "tail_window": 16})
                recs32[arm][cid] = _extract_gen_record(cid, g, row, peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3)

        # Check whether Full-SD 32k fits or OOMs on the compiled stack
        pair.draft_runner = None
        del r_spade
        _cleanup_cuda()
        full32_sum: Any = None
        full32_err: str | None = None
        try:
            r_full32 = StaticDraftRunner(pair.draft, max_cache_len=cap32, mode="compile_cudagraph", device=pair._input_device(pair.draft), graph_pool=pool)
            r_full32.prepare(warmup_steps=3)
            pair.draft_runner = r_full32
            fl32_recs = {}
            for i, row in enumerate(plist):
                torch.cuda.reset_peak_memory_stats()
                g = pair.generate(row["prompt"], mode="B2", max_new_tokens=128, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + i, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                cid = str(row["claim_id"])
                fl32_recs[cid] = _extract_gen_record(cid, g, row, peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3)
            full32_sum = _aggregate_metrics(list(fl32_recs.values()))
        except Exception as exc:
            full32_err = f"{type(exc).__name__}: {str(exc)[:180]}"

        ar32 = _aggregate_metrics(list(recs32["AR"].values()))
        sp32 = _aggregate_metrics(list(recs32["SPADE"].values()))
        results["32768"] = {
            "target_tokens": 32768,
            "actual_tokens": round(statistics.mean(r["actual_tokens"] for r in plist)),
            "n_prompts": len(plist),
            "AR": ar32,
            "Full": full32_sum if full32_sum else {"oom": True, "error": full32_err},
            "SPADE": sp32,
            "spade_vs_ar_e2e": _paired_boot(recs32["AR"], recs32["SPADE"], "e2e_tok_s"),
        }
        _write_json(ckpt_path, results)
        print(f"  DONE L=32768: AR={ar32['e2e_tok_s']} | SPADE={sp32['e2e_tok_s']} | Full={'OOM' if full32_err else full32_sum['e2e_tok_s']}", flush=True)
        pair.target_runner = None
        pair.draft_runner = None
        del pair
        _cleanup_cuda(reset_dynamo=True)


# ---------------------------------------------------------------------------
# Block 6: outlen_sweep (Output Lengths 128, 256, 512, 1024 at 8k and 16k)
# ---------------------------------------------------------------------------

def run_outlen_sweep_block() -> None:
    print("\n=== [BLOCK: outlen_sweep] Output-length sweep 128-1024 at L=8k and L=16k ===", flush=True)
    ckpt_path = REV9_DIR / "outlen_sweep_8k_16k.json"
    results: dict[str, Any] = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.exists() else {}

    bench_rows = _load_jsonl(HELD300_PATH)
    pool_passages = [str(e.get("text", "")).strip() for r in bench_rows for e in (r.get("evidence") or []) if len(str(e.get("text", "")).strip()) >= 1500]
    base_pool = sorted(bench_rows, key=lambda r: sum(len(str(e.get("text", ""))) for e in (r.get("evidence") or [])))[:10]
    tok = AutoTokenizer.from_pretrained(QWEN3_8B_SNAP, local_files_only=True)
    prompts_by_len = {
        L: [_build_padded_prompt(tok, L, base_pool[i], pool_passages, start_offset=i * 7) for i in range(10)]
        for L in (8192, 16384)
    }

    cap_t = 17920  # >= 16384 + 1024 + headroom
    cap_sp = 6144
    pool = torch.cuda.graph_pool_handle()
    pair = QwenPair(
        QWEN3_8B_SNAP, QWEN3_06B_SNAP, local_files_only=True, dtype="bfloat16",
        draft_attn_implementation="flex_attention", target_attn_implementation="flex_attention",
    )
    pair.prepare_target_runner(mode="compile_cudagraph", max_cache_len=cap_t, max_verify_len=5, warmup_steps=3, graph_pool=pool)
    r_full = StaticDraftRunner(pair.draft, max_cache_len=cap_t, mode="compile_cudagraph", device=pair._input_device(pair.draft), graph_pool=pool)
    r_full.prepare(warmup_steps=3)
    r_spade = StaticDraftRunner(pair.draft, max_cache_len=cap_sp, mode="compile_cudagraph", device=pair._input_device(pair.draft), graph_pool=pool)
    r_spade.prepare(warmup_steps=3)

    for gw in range(2):
        pair.generate(prompts_by_len[8192][0]["prompt"], mode="B0-S", max_new_tokens=16, temperature=0.7, top_p=0.9, seed=600 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast", suppress_eos=True)
        pair.draft_runner = r_full
        pair.generate(prompts_by_len[8192][0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=610 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast", suppress_eos=True)
        pair.draft_runner = r_spade
        pair.generate(prompts_by_len[8192][0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=620 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": prompts_by_len[8192][0], "budget_chars": 10000, "tail_window": 16}, suppress_eos=True)

    for L in (8192, 16384):
        plist = prompts_by_len[L]
        for out_len in (128, 256, 512, 1024):
            key = f"{L}|{out_len}"
            if key in results:
                print(f"  SKIP {key}", flush=True)
                continue
            recs: dict[str, dict[str, dict[str, Any]]] = {"AR": {}, "Full": {}, "SPADE": {}}
            rng = random.Random(42 + L + out_len)
            for i, row in enumerate(plist):
                cid = str(row["claim_id"])
                arms = ["AR", "Full", "SPADE"]
                rng.shuffle(arms)
                for arm in arms:
                    torch.cuda.reset_peak_memory_stats()
                    if arm == "AR":
                        g = pair.generate(row["prompt"], mode="B0-S", max_new_tokens=out_len, temperature=0.7, top_p=0.9, seed=42 + i, cache_strategy="kv_cache", distribution_mode="fp32_fast", suppress_eos=True)
                    elif arm == "Full":
                        pair.draft_runner = r_full
                        g = pair.generate(row["prompt"], mode="B2", max_new_tokens=out_len, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + i, cache_strategy="kv_cache", distribution_mode="fp32_fast", suppress_eos=True)
                    else:
                        pair.draft_runner = r_spade
                        g = pair.generate(row["prompt"], mode="B2", max_new_tokens=out_len, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + i, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": row, "budget_chars": 10000, "tail_window": 16}, suppress_eos=True)
                    recs[arm][cid] = _extract_gen_record(cid, g, row, peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3)
            ar_s = _aggregate_metrics(list(recs["AR"].values()))
            fl_s = _aggregate_metrics(list(recs["Full"].values()))
            sp_s = _aggregate_metrics(list(recs["SPADE"].values()))
            results[key] = {
                "context_len": L,
                "max_new_tokens": out_len,
                "n_prompts": len(plist),
                "AR": ar_s,
                "Full": fl_s,
                "SPADE": sp_s,
                "spade_vs_ar_e2e": _paired_boot(recs["AR"], recs["SPADE"], "e2e_tok_s"),
                "spade_vs_full_e2e": _paired_boot(recs["Full"], recs["SPADE"], "e2e_tok_s"),
            }
            _write_json(ckpt_path, results)
            print(f"  DONE {key}: AR={ar_s['e2e_tok_s']} | Full={fl_s['e2e_tok_s']} | SPADE={sp_s['e2e_tok_s']} | d_AR={results[key]['spade_vs_ar_e2e']['delta']:+.2f}", flush=True)

    pair.target_runner = None
    pair.draft_runner = None
    del r_full, r_spade, pair
    _cleanup_cuda(reset_dynamo=True)


# ---------------------------------------------------------------------------
# Block 7: oracle_hotpot_opt (Compiled-stack Oracle-393 & HotpotQA-100)
# ---------------------------------------------------------------------------

def run_oracle_hotpot_opt_block() -> None:
    print("\n=== [BLOCK: oracle_hotpot_opt] Compiled-stack Oracle-393 & HotpotQA-100 ===", flush=True)
    tok = AutoTokenizer.from_pretrained(QWEN3_8B_SNAP, local_files_only=True)

    # 7A) Oracle N=393 50/50 interleaved
    ora_ckpt = REV9_DIR / "oracle393_interleaved_opt.json"
    if not ora_ckpt.exists():
        rows = _load_jsonl(ORACLE393_PATH)
        assert len(rows) == 393
        cap_o = ((_max_prompt_tokens(tok, rows) + 384 + 32 + 127) // 128) * 128
        pool = torch.cuda.graph_pool_handle()
        pair = QwenPair(
            QWEN3_8B_SNAP, QWEN3_06B_SNAP, local_files_only=True, dtype="bfloat16",
            draft_attn_implementation="flex_attention", target_attn_implementation="flex_attention",
        )
        pair.prepare_target_runner(mode="compile_cudagraph", max_cache_len=cap_o, max_verify_len=5, warmup_steps=3, graph_pool=pool)
        pair.prepare_draft_runner(mode="compile_cudagraph", max_cache_len=cap_o, warmup_steps=3, graph_pool=pool)
        for gw in range(2):
            pair.generate(rows[0]["prompt"], mode="B0-S", max_new_tokens=16, temperature=0.7, top_p=0.9, seed=900 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast")
            pair.generate(rows[0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=910 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast")
            pair.generate(rows[0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=920 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": rows[0], "budget_chars": 10000, "tail_window": 16})

        rec_ar: dict[str, dict[str, Any]] = {}
        rec_fl: dict[str, dict[str, Any]] = {}
        rec_sp: dict[str, dict[str, Any]] = {}
        orders = [("full", "spade")] * (len(rows) // 2) + [("spade", "full")] * (len(rows) - len(rows) // 2)
        random.Random(42).shuffle(orders)
        for idx, row in enumerate(rows):
            cid = str(row["claim_id"])
            g_ar = pair.generate(row["prompt"], mode="B0-S", max_new_tokens=384, temperature=0.7, top_p=0.9, seed=42 + idx, run_id="v9-ora-ar", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast")
            rec_ar[cid] = _extract_gen_record(cid, g_ar, row)
            for arm in orders[idx]:
                cfg = {"row": row, "budget_chars": 10000, "tail_window": 16} if arm == "spade" else None
                g = pair.generate(row["prompt"], mode="B2", max_new_tokens=384, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + idx, run_id=f"v9-ora-{arm}", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg=cfg)
                rec = _extract_gen_record(cid, g, row)
                if arm == "full":
                    rec_fl[cid] = rec
                else:
                    rec_sp[cid] = rec
            if (idx + 1) % 80 == 0 or (idx + 1) == len(rows):
                print(f"    Oracle: {idx+1}/{len(rows)}", flush=True)
        _write_json(ora_ckpt, {
            "n_claims": len(rows),
            "Target_AR": _aggregate_metrics(list(rec_ar.values())),
            "Baseline_Full": _aggregate_metrics(list(rec_fl.values())),
            "SPADE_Bypass": _aggregate_metrics(list(rec_sp.values())),
            "spade_vs_full_e2e": _paired_boot(rec_fl, rec_sp, "e2e_tok_s"),
            "spade_vs_ar_e2e": _paired_boot(rec_ar, rec_sp, "e2e_tok_s"),
        })
        pair.target_runner = None
        pair.draft_runner = None
        del pair
        _cleanup_cuda(reset_dynamo=True)

    # 7B) LongBench HotpotQA-100 interleaved (Target AR, Full-SD, BM25 Sent-B10k, SPADE B=10k)
    hp_ckpt = REV9_DIR / "hotpotqa100_opt.json"
    if not hp_ckpt.exists():
        rows = _load_jsonl(HOTPOT100_PATH)
        assert len(rows) == 100
        cap_hp = ((_max_prompt_tokens(tok, rows) + 128 + 32 + 127) // 128) * 128
        pool = torch.cuda.graph_pool_handle()
        pair = QwenPair(
            QWEN3_8B_SNAP, QWEN3_06B_SNAP, local_files_only=True, dtype="bfloat16",
            draft_attn_implementation="flex_attention", target_attn_implementation="flex_attention",
        )
        pair.prepare_target_runner(mode="compile_cudagraph", max_cache_len=cap_hp, max_verify_len=5, warmup_steps=3, graph_pool=pool)
        r_full = StaticDraftRunner(pair.draft, max_cache_len=cap_hp, mode="compile_cudagraph", device=pair._input_device(pair.draft), graph_pool=pool)
        r_full.prepare(warmup_steps=3)
        r_cmp = StaticDraftRunner(pair.draft, max_cache_len=6144, mode="compile_cudagraph", device=pair._input_device(pair.draft), graph_pool=pool)
        r_cmp.prepare(warmup_steps=3)
        for gw in range(2):
            pair.generate(rows[0]["prompt"], mode="B0-S", max_new_tokens=16, temperature=0.7, top_p=0.9, seed=930 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast")
            pair.draft_runner = r_full
            pair.generate(rows[0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=940 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast")
            pair.draft_runner = r_cmp
            pair.generate(rows[0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=950 + gw, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": rows[0], "budget_chars": 10000, "tail_window": 16})

        arms = ["Target AR (B0-S)", "Baseline Full", "BM25 Sent-B10k", "SPADE B=10k"]
        hp_recs: dict[str, dict[str, dict[str, Any]]] = {a: {} for a in arms}
        rng = random.Random(42)
        for idx, row in enumerate(rows):
            cid = str(row["claim_id"])
            perm = list(arms)
            rng.shuffle(perm)
            for arm in perm:
                torch.cuda.reset_peak_memory_stats()
                if arm == "Target AR (B0-S)":
                    g = pair.generate(row["prompt"], mode="B0-S", max_new_tokens=128, temperature=0.7, top_p=0.9, seed=42 + idx, run_id="v9-hp-ar", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                elif arm == "Baseline Full":
                    pair.draft_runner = r_full
                    g = pair.generate(row["prompt"], mode="B2", max_new_tokens=128, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + idx, run_id="v9-hp-full", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                elif arm == "BM25 Sent-B10k":
                    pair.draft_runner = r_cmp
                    dctx = build_draft_context(row, policy="sentence_budget", budget_chars=10000, min_sentences_per_passage=1)
                    g = pair.generate(row["prompt"], mode="B2", max_new_tokens=128, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + idx, run_id="v9-hp-bm25", claim_id=cid, draft_prompt=dctx.draft_prompt, cache_strategy="kv_cache", distribution_mode="fp32_fast")
                    g.timings["context_selection_s"] = dctx.context_selection_s
                else:
                    pair.draft_runner = r_cmp
                    g = pair.generate(row["prompt"], mode="B2", max_new_tokens=128, draft_length=4, temperature=0.7, top_p=0.9, seed=42 + idx, run_id="v9-hp-spade", claim_id=cid, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": row, "budget_chars": 10000, "tail_window": 16})
                hp_recs[arm][cid] = _extract_gen_record(cid, g, row, peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3)
            if (idx + 1) % 25 == 0 or (idx + 1) == len(rows):
                print(f"    HotpotQA: {idx+1}/{len(rows)}", flush=True)

        e2e_c = []
        for b_lbl in ("Target AR (B0-S)", "Baseline Full", "BM25 Sent-B10k"):
            c = _paired_boot(hp_recs[b_lbl], hp_recs["SPADE B=10k"], "e2e_tok_s")
            c["contrast"] = f"SPADE vs {b_lbl}"
            e2e_c.append(c)
        _holm_adjust(e2e_c)
        f1_c = []
        for b_lbl in ("Target AR (B0-S)", "Baseline Full", "BM25 Sent-B10k"):
            c = _paired_boot(hp_recs[b_lbl], hp_recs["SPADE B=10k"], "qa_f1_pct")
            c["contrast"] = f"SPADE vs {b_lbl}"
            f1_c.append(c)
        _holm_adjust(f1_c)

        _write_json(hp_ckpt, {
            "n_claims": len(rows),
            "arms": {a: _aggregate_metrics(list(hp_recs[a].values())) for a in arms},
            "pairwise_e2e_holm": {c["contrast"]: c for c in e2e_c},
            "pairwise_f1_holm": {c["contrast"]: c for c in f1_c},
            "spade_vs_bm25_alpha": _paired_boot(hp_recs["BM25 Sent-B10k"], hp_recs["SPADE B=10k"], "alpha_eval_pct"),
            "spade_vs_full_kv": None,
        })
        pair.target_runner = None
        pair.draft_runner = None
        del r_full, r_cmp, pair
        _cleanup_cuda(reset_dynamo=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    choices = [
        "target_opt_profile",
        "heldout300_interleaved_r3",
        "ratio_and_cross_family",
        "batched_e2e_concurrency",
        "fixed_bm25_interaction",
        "scaling_2k_32k",
        "outlen_sweep",
        "oracle_hotpot_opt",
    ]
    p.add_argument("--blocks", nargs="+", default=choices, choices=choices)
    args = p.parse_args()
    REV9_DIR.mkdir(parents=True, exist_ok=True)
    runners = {
        "target_opt_profile": run_target_opt_profile_block,
        "heldout300_interleaved_r3": run_heldout300_r3_block,
        "ratio_and_cross_family": run_ratio_and_cross_family_block,
        "batched_e2e_concurrency": run_batched_e2e_concurrency_block,
        "fixed_bm25_interaction": run_fixed_bm25_interaction_block,
        "scaling_2k_32k": run_scaling_2k_32k_block,
        "outlen_sweep": run_outlen_sweep_block,
        "oracle_hotpot_opt": run_oracle_hotpot_opt_block,
    }
    for blk in args.blocks:
        t0 = time.perf_counter()
        runners[blk]()
        print(f"[COMPLETED {blk} in {(time.perf_counter() - t0)/60.0:.2f} min]", flush=True)


if __name__ == "__main__":
    main()
