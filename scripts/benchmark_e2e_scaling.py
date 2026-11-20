"""Controlled End-to-End Long-Context Scaling Benchmark (2k to 32k tokens) on RTX 5090.

Addresses reviewer requirements for statistical power and fair 2k bypass parity:
  - Uses N=20 distinct AVeriTeC-formatted prompts at L in {2048, 4096, 8192, 16384}
    and N=10 distinct prompts at L = 32768 (128 generated tokens per prompt).
  - Interleaves AR (B0-S), Full-SD, and SPADE across 5-prompt mini-blocks (alternating
    SPADE vs Full-SD execution order across odd/even blocks) to cancel thermal drift.
  - Matches StaticCache capacity identically between Full-SD and SPADE at L=2048 (bypass)
    and isolates SPADE vs Full-SD cache sizes cleanly at longer lengths.
  - Reports paired bootstrap 95% CIs (B=10,000) for Δ(SPADE - Full-SD) and Δ(SPADE - AR).
"""
from __future__ import annotations

import gc
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from spade.adaptive.target_attention_selector import (
    TargetPrefillAttentionHook,
    resolve_layer_indices,
)
from spade.benchmark.prompt_builder import build_prompt
from spade.decoding.autoregressive import format_qwen_prompt
from spade.decoding.hf_assisted import QwenPair

TARGET_SNAP = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DRAFT_SNAP = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
BENCH_PATH = Path("outputs/benchmarks/retrieved-heldout300.jsonl")
OUT_PATH = Path("outputs/analysis/v6_revisions/e2e_scaling_2k_32k.json")


def _build_prompt_at_length(
    tok: Any,
    target_tokens: int,
    base_row: dict[str, Any],
    pool_passages: list[str],
    start_offset: int,
) -> dict[str, Any]:
    """Construct a valid AVeriTeC-formatted row padded cleanly with long passages (~target_tokens)."""
    ev = [
        {"evidence_id": str(e.get("evidence_id", f"E{i+1}")), "text": str(e.get("text", ""))}
        for i, e in enumerate(base_row.get("evidence") or [])
    ]
    idx_p = start_offset
    while True:
        prompt_str, phash = build_prompt(base_row["claim"], ev, mode="retrieved")
        ser = format_qwen_prompt(tok, prompt_str)
        n_tok = len(tok(ser, add_special_tokens=False)["input_ids"])
        if n_tok >= target_tokens:
            break
        nxt = pool_passages[idx_p % len(pool_passages)]
        ev.append({"evidence_id": f"E{len(ev)+1}", "text": nxt})
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
    ser = format_qwen_prompt(tok, prompt_str)
    n_tok = len(tok(ser, add_special_tokens=False)["input_ids"])
    return {
        "claim_id": f"{base_row['claim_id']}-scale-{target_tokens}",
        "claim": base_row["claim"],
        "evidence": ev,
        "evidence_mode": "retrieved",
        "prompt": prompt_str,
        "prompt_hash": phash,
        "actual_tokens": n_tok,
    }


def _eval_alpha_pct(gen) -> float:
    ev = [t for t in gen.token_traces if t.get("status") in {"accepted", "rejected"}]
    if not ev:
        return 0.0
    acc = sum(1 for t in ev if t.get("status") == "accepted")
    return (acc / len(ev)) * 100.0


def _paired_ratio_boot(
    base_recs: list[tuple[int, float]],
    treat_recs: list[tuple[int, float]],
    *,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired cluster bootstrap on pooled aggregate tok/s = sum(tokens)/sum(time_s)."""
    n = len(base_recs)
    b_tok = sum(r[0] for r in base_recs)
    b_tim = sum(r[1] for r in base_recs)
    t_tok = sum(r[0] for r in treat_recs)
    t_tim = sum(r[1] for r in treat_recs)
    b_val = (b_tok / b_tim) if b_tim > 0 else 0.0
    t_val = (t_tok / t_tim) if t_tim > 0 else 0.0
    delta = t_val - b_val
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        bt = bm = tt = tm = 0.0
        for _k in range(n):
            idx = rng.randrange(n)
            bt += base_recs[idx][0]
            bm += base_recs[idx][1]
            tt += treat_recs[idx][0]
            tm += treat_recs[idx][1]
        boots.append((tt / tm if tm > 0 else 0.0) - (bt / bm if bm > 0 else 0.0))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(n_boot - 1, int(0.975 * n_boot))]
    p2 = min(
        1.0,
        2.0 * min(sum(1 for x in boots if x <= 0.0) / n_boot, sum(1 for x in boots if x >= 0.0) / n_boot),
    )
    return {
        "base": round(b_val, 2),
        "treat": round(t_val, 2),
        "delta": round(delta, 2),
        "ci_lo": round(lo, 2),
        "ci_hi": round(hi, 2),
        "p_raw": round(p2, 4),
    }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    bench_rows = [json.loads(l) for l in BENCH_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Pool long passages (>= 1,500 chars) for clean realistic padding
    pool_passages: list[str] = []
    for r in bench_rows:
        for e in r.get("evidence") or []:
            txt = str(e.get("text", "")).strip()
            if len(txt) >= 1500:
                pool_passages.append(txt)

    base_pool = sorted(
        bench_rows,
        key=lambda r: sum(len(str(e.get("text", ""))) for e in (r.get("evidence") or [])),
    )[:20]

    n_by_len = {2048: 20, 4096: 20, 8192: 20, 16384: 20, 32768: 10}
    lengths = [2048, 4096, 8192, 16384, 32768]

    tok = AutoTokenizer.from_pretrained(TARGET_SNAP, local_files_only=True)
    prompts_by_len: dict[int, list[dict[str, Any]]] = {
        L: [
            _build_prompt_at_length(tok, L, base_pool[i], pool_passages, start_offset=i * 7)
            for i in range(n_by_len[L])
        ]
        for L in lengths
    }

    # Part A: Standalone Draft Prefill Latency & VRAM (mean of 3 repeats after warmup)
    comp_results: dict[int, dict[str, Any]] = {}
    draft_only = AutoModelForCausalLM.from_pretrained(
        DRAFT_SNAP, dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    ).eval()
    for L in lengths:
        inp = torch.randint(10, 5000, (1, L), device="cuda", dtype=torch.long)
        with torch.inference_mode():
            _ = draft_only(input_ids=inp[:, : min(512, L)], use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        times_ms = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = draft_only(input_ids=inp, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            del out
        comp_results[L] = {
            "n_prompts": n_by_len[L],
            "full_draft_prefill_ms": round(statistics.mean(times_ms), 2),
            "full_draft_standalone_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2),
        }
        del inp
        torch.cuda.empty_cache()

    spade_inp = torch.randint(10, 5000, (1, 1776), device="cuda", dtype=torch.long)
    torch.cuda.reset_peak_memory_stats()
    spade_draft_times = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = draft_only(input_ids=spade_inp, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        spade_draft_times.append((time.perf_counter() - t0) * 1000.0)
        del out
    spade_draft_ms = round(statistics.mean(spade_draft_times), 2)
    spade_draft_vram_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
    del draft_only, spade_inp
    gc.collect()
    torch.cuda.empty_cache()

    # Part B: Target 8B + Hook + Interleaved E2E Comparisons (AR, Full-SD, SPADE)
    pair = QwenPair(
        TARGET_SNAP, DRAFT_SNAP, local_files_only=True, dtype="bfloat16", draft_attn_implementation="flex_attention"
    )
    n_t_layers = len(pair.target.model.layers)
    hook_layers = resolve_layer_indices(n_t_layers, "upper50")

    # Initial warmup on target model AR
    wrow = prompts_by_len[4096][0]
    for gw in range(2):
        _ = pair.generate(
            wrow["prompt"],
            mode="B0-S",
            max_new_tokens=16,
            temperature=0.7,
            top_p=0.9,
            seed=99 + gw,
            cache_strategy="kv_cache",
            distribution_mode="fp32_fast",
        )
    torch.cuda.empty_cache()

    for L in lengths:
        plist = prompts_by_len[L]
        n_p = len(plist)
        max_actual = max(r["actual_tokens"] for r in plist)
        full_cap = ((max_actual + 256 + 127) // 128) * 128
        spade_cap = min(full_cap, 5632)  # Identical cache capacity at 2k bypass!

        # 1. Measure Target 8B Prefill & Hook exact overhead over 5 distinct prompts (1 repeat after warmup)
        t_times, hk_times, hk_deltas_mb = [], [], []
        t_vram_gb = 0.0
        for p_idx_h in range(min(5, n_p)):
            p_ids = tok(format_qwen_prompt(tok, plist[p_idx_h]["prompt"]), add_special_tokens=False)["input_ids"]
            inp = torch.tensor([p_ids], dtype=torch.long, device="cuda")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with TargetPrefillAttentionHook(pair.target, layer_indices=hook_layers, tail_window=16) as hk:
                with torch.inference_mode():
                    out = pair.target(input_ids=inp, use_cache=True, logits_to_keep=1)
                torch.cuda.synchronize()
                t_times.append((time.perf_counter() - t0) * 1000.0)
                base_alloc = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                imp, hk_s = hk.compute_token_importance(out.past_key_values)
                hk_peak = torch.cuda.max_memory_allocated()
                hk_times.append(hk_s * 1000.0)
                hk_deltas_mb.append(max(0.0, (hk_peak - base_alloc) / (1024 ** 2)))
                del out, imp
            t_vram_gb = max(t_vram_gb, torch.cuda.max_memory_allocated() / (1024 ** 3))
            del inp
            torch.cuda.empty_cache()

        comp_results[L]["actual_tokens"] = round(statistics.mean(r["actual_tokens"] for r in plist))
        comp_results[L]["target_8b_prefill_ms"] = round(statistics.mean(t_times), 2)
        comp_results[L]["hook_ms"] = round(statistics.mean(hk_times), 2)
        comp_results[L]["hook_peak_delta_mb"] = round(max(hk_deltas_mb), 2)
        comp_results[L]["target_colocated_vram_gb"] = round(t_vram_gb, 2)
        comp_results[L]["spade_draft_prefill_ms"] = spade_draft_ms
        comp_results[L]["spade_draft_standalone_vram_gb"] = spade_draft_vram_gb
        tot_full_pre = comp_results[L]["target_8b_prefill_ms"] + comp_results[L]["full_draft_prefill_ms"]
        tot_spade_pre = comp_results[L]["target_8b_prefill_ms"] + comp_results[L]["hook_ms"] + spade_draft_ms
        comp_results[L]["total_prefill_full_ms"] = round(tot_full_pre, 2)
        comp_results[L]["total_prefill_spade_ms"] = round(tot_spade_pre, 2)
        comp_results[L]["total_prefill_saved_pct"] = round(((tot_full_pre - tot_spade_pre) / tot_full_pre) * 100.0, 1)

        # 2. Interleaved E2E measurements
        ar_recs: list[tuple[int, float]] = [None] * n_p  # type: ignore[list-item]
        sp_recs: list[tuple[int, float]] = [None] * n_p  # type: ignore[list-item]
        fl_recs: list[tuple[int, float]] = [None] * n_p  # type: ignore[list-item]
        ar_dec_s = sp_dec_s = fl_dec_s = 0.0
        ar_vram = sp_vram = fl_vram = 0.0
        sp_alphas, fl_alphas, sp_dtoks = [], [], []
        sp_sel_ms, sp_dpref_ms, fl_dpref_ms = [], [], []
        full_sd_error: str | None = None

        def _run_ar(idx_p: int) -> None:
            nonlocal ar_dec_s, ar_vram
            row = plist[idx_p]
            torch.cuda.reset_peak_memory_stats()
            g = pair.generate(
                row["prompt"], mode="B0-S", max_new_tokens=128, temperature=0.7, top_p=0.9,
                seed=42 + idx_p, cache_strategy="kv_cache", distribution_mode="fp32_fast",
            )
            e2e = g.timings["prompt_prefill_s"] + g.timings["decode_s"]
            ar_recs[idx_p] = (len(g.token_ids), e2e)
            ar_dec_s += g.timings["decode_s"]
            ar_vram = max(ar_vram, torch.cuda.max_memory_allocated() / (1024 ** 3))
            torch.cuda.empty_cache()

        def _run_spade(idx_p: int) -> None:
            nonlocal sp_dec_s, sp_vram
            row = plist[idx_p]
            torch.cuda.reset_peak_memory_stats()
            g = pair.generate(
                row["prompt"], mode="B2", max_new_tokens=128, draft_length=4, temperature=0.7, top_p=0.9,
                seed=42 + idx_p, cache_strategy="kv_cache", distribution_mode="fp32_fast",
                target_attn_selector_cfg={"row": row, "budget_chars": 10000, "tail_window": 16},
            )
            e2e = g.timings["prompt_prefill_s"] + g.timings.get("context_selection_s", 0.0) + g.timings["draft_prefill_s"] + g.timings["decode_s"]
            sp_recs[idx_p] = (len(g.token_ids), e2e)
            sp_dec_s += g.timings["decode_s"]
            sp_vram = max(sp_vram, torch.cuda.max_memory_allocated() / (1024 ** 3))
            sp_alphas.append(_eval_alpha_pct(g))
            sp_dtoks.append(g.timings["draft_prompt_tokens"])
            sp_sel_ms.append(g.timings.get("context_selection_s", 0.0) * 1000.0)
            sp_dpref_ms.append(g.timings["draft_prefill_s"] * 1000.0)
            torch.cuda.empty_cache()

        def _run_full(idx_p: int) -> None:
            nonlocal fl_dec_s, fl_vram
            row = plist[idx_p]
            torch.cuda.reset_peak_memory_stats()
            g = pair.generate(
                row["prompt"], mode="B2", max_new_tokens=128, draft_length=4, temperature=0.7, top_p=0.9,
                seed=42 + idx_p, cache_strategy="kv_cache", distribution_mode="fp32_fast",
            )
            e2e = g.timings["prompt_prefill_s"] + g.timings["draft_prefill_s"] + g.timings["decode_s"]
            fl_recs[idx_p] = (len(g.token_ids), e2e)
            fl_dec_s += g.timings["decode_s"]
            fl_vram = max(fl_vram, torch.cuda.max_memory_allocated() / (1024 ** 3))
            fl_alphas.append(_eval_alpha_pct(g))
            fl_dpref_ms.append(g.timings["draft_prefill_s"] * 1000.0)
            torch.cuda.empty_cache()

        if L == 2048:
            # Shared identical StaticCache capacity -> true per-prompt interleaving!
            pair.prepare_draft_runner(mode="compile_reduce_overhead", max_cache_len=full_cap, warmup_steps=3)
            _ = pair.generate(plist[0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=99, cache_strategy="kv_cache", distribution_mode="fp32_fast")
            _ = pair.generate(plist[0]["prompt"], mode="B2", max_new_tokens=16, draft_length=4, temperature=0.7, top_p=0.9, seed=100, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": plist[0], "budget_chars": 10000, "tail_window": 16})
            torch.cuda.empty_cache()
            for idx_p in range(n_p):
                if idx_p % 2 == 0:
                    _run_ar(idx_p); _run_full(idx_p); _run_spade(idx_p)
                else:
                    _run_ar(idx_p); _run_spade(idx_p); _run_full(idx_p)
            pair.draft_runner = None
            gc.collect()
            torch.cuda.empty_cache()
        elif L < 32768:
            # 4 interleaved mini-blocks of 5 prompts (alternating SPADE vs Full-SD order)
            chunk_size = 5
            for c_idx in range(0, n_p, chunk_size):
                idxs = list(range(c_idx, min(n_p, c_idx + chunk_size)))
                for idx_p in idxs:
                    _run_ar(idx_p)
                block_num = c_idx // chunk_size
                order = ("spade", "full") if (block_num % 2 == 0) else ("full", "spade")
                for arm in order:
                    cap_use = spade_cap if arm == "spade" else full_cap
                    pair.prepare_draft_runner(mode="compile_reduce_overhead", max_cache_len=cap_use, warmup_steps=3)
                    warm_cfg = {"row": plist[idxs[0]], "budget_chars": 10000, "tail_window": 16} if arm == "spade" else None
                    _ = pair.generate(plist[idxs[0]]["prompt"], mode="B2", max_new_tokens=8, draft_length=4, temperature=0.7, top_p=0.9, seed=99, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg=warm_cfg)
                    torch.cuda.empty_cache()
                    for idx_p in idxs:
                        if arm == "spade":
                            _run_spade(idx_p)
                        else:
                            _run_full(idx_p)
                    pair.draft_runner = None
                    gc.collect()
                    torch.cuda.empty_cache()
        else:
            # 32k (N=10): Interleave AR and SPADE directly per prompt, then probe Full-SD (catch OOM cleanly)
            pair.prepare_draft_runner(mode="compile_reduce_overhead", max_cache_len=spade_cap, warmup_steps=3)
            _ = pair.generate(plist[0]["prompt"], mode="B2", max_new_tokens=8, draft_length=4, temperature=0.7, top_p=0.9, seed=99, cache_strategy="kv_cache", distribution_mode="fp32_fast", target_attn_selector_cfg={"row": plist[0], "budget_chars": 10000, "tail_window": 16})
            torch.cuda.empty_cache()
            for idx_p in range(n_p):
                if idx_p % 2 == 0:
                    _run_ar(idx_p); _run_spade(idx_p)
                else:
                    _run_spade(idx_p); _run_ar(idx_p)
            pair.draft_runner = None
            gc.collect()
            torch.cuda.empty_cache()
            try:
                pair.prepare_draft_runner(mode="compile_reduce_overhead", max_cache_len=full_cap, warmup_steps=3)
                _run_full(0)
            except Exception as exc:
                full_sd_error = str(exc)
            finally:
                pair.draft_runner = None
                gc.collect()
                torch.cuda.empty_cache()

        ar_tot_tok = sum(r[0] for r in ar_recs)
        ar_tot_s = sum(r[1] for r in ar_recs)
        sp_tot_tok = sum(r[0] for r in sp_recs)
        sp_tot_s = sum(r[1] for r in sp_recs)

        comp_results[L]["ar_e2e_tok_s"] = round(ar_tot_tok / ar_tot_s, 2)
        comp_results[L]["ar_dec_tok_s"] = round(ar_tot_tok / ar_dec_s, 2)
        comp_results[L]["ar_peak_vram_gb"] = round(ar_vram, 2)
        comp_results[L]["spade_e2e_tok_s"] = round(sp_tot_tok / sp_tot_s, 2)
        comp_results[L]["spade_dec_tok_s"] = round(sp_tot_tok / sp_dec_s, 2)
        comp_results[L]["spade_alpha_eval_pct"] = round(statistics.mean(sp_alphas), 2)
        comp_results[L]["spade_draft_tokens"] = round(statistics.mean(sp_dtoks), 1)
        comp_results[L]["spade_peak_vram_gb"] = round(sp_vram, 2)
        comp_results[L]["spade_e2e_sel_ms"] = round(statistics.mean(sp_sel_ms), 3)
        comp_results[L]["spade_e2e_draft_prefill_ms"] = round(statistics.mean(sp_dpref_ms), 2)
        comp_results[L]["delta_spade_vs_ar"] = _paired_ratio_boot(ar_recs, sp_recs)

        if L == 2048 and statistics.mean(sp_dtoks) >= 0.95 * comp_results[L]["actual_tokens"]:
            comp_results[L]["bypass_active"] = True
            comp_results[L]["hook_ms_measured"] = comp_results[L]["hook_ms"]
            comp_results[L]["hook_ms_effective"] = round(statistics.mean(sp_sel_ms), 3)
            comp_results[L]["spade_draft_prefill_ms"] = comp_results[L]["full_draft_prefill_ms"]
            comp_results[L]["total_prefill_spade_ms"] = comp_results[L]["total_prefill_full_ms"]
            comp_results[L]["total_prefill_saved_pct"] = 0.0
        else:
            comp_results[L]["bypass_active"] = False
            comp_results[L]["hook_ms_effective"] = comp_results[L]["hook_ms"]

        if full_sd_error is None and all(r is not None for r in fl_recs):
            fl_tot_tok = sum(r[0] for r in fl_recs)
            fl_tot_s = sum(r[1] for r in fl_recs)
            comp_results[L]["full_sd_e2e_tok_s"] = round(fl_tot_tok / fl_tot_s, 2)
            comp_results[L]["full_sd_dec_tok_s"] = round(fl_tot_tok / fl_dec_s, 2)
            comp_results[L]["full_sd_alpha_eval_pct"] = round(statistics.mean(fl_alphas), 2)
            comp_results[L]["full_sd_peak_vram_gb"] = round(fl_vram, 2)
            comp_results[L]["full_e2e_draft_prefill_ms"] = round(statistics.mean(fl_dpref_ms), 2)
            comp_results[L]["delta_spade_vs_full"] = _paired_ratio_boot(fl_recs, sp_recs)
        else:
            comp_results[L]["full_sd_error"] = full_sd_error or "Incomplete Full-SD run"

        # Save progressively after each length
        OUT_PATH.write_text(json.dumps(comp_results, indent=2) + "\n", encoding="utf-8")
        print(f"DONE L={L} (N={n_p}): AR={comp_results[L]['ar_e2e_tok_s']}, SPADE={comp_results[L]['spade_e2e_tok_s']}, Full={comp_results[L].get('full_sd_e2e_tok_s', 'OOM')}", flush=True)

    print(json.dumps(comp_results, indent=2))


if __name__ == "__main__":
    main()
