"""Full End-to-End Long-Context Scaling Benchmark (2k to 32k tokens) on RTX 5090.

Measures:
  1. Target 8B prefill latency (ms) and peak VRAM (with logits_to_keep=1, SDPA)
  2. TargetPrefillAttentionHook (upper 50% layers, W=16) latency (ms) & delta VRAM (MB) across 2k-32k
  3. Full Draft 0.6B vs SPADE Draft 0.6B (capped <= 1,776 tokens) prefill latency (ms)
  4. Total Prefill (Target + Draft + Hook) & % Total Prefill Saved
  5. Real Co-Located End-to-End Throughput (tok/s) & Peak Combined VRAM (GB) for:
       - Target AR (B0-S)
       - Full-SD (Baseline Full, K=4)
       - SPADE (Attention-Selected, B=10k -> <= 1,776 tokens, K=4)
"""
from __future__ import annotations

import json
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
from spade.decoding.autoregressive import format_qwen_prompt
from spade.decoding.hf_assisted import QwenPair

TARGET_SNAP = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DRAFT_SNAP = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
BENCH_PATH = Path("outputs/benchmarks/retrieved-heldout300.jsonl")
OUT_PATH = Path("outputs/analysis/v6_revisions/e2e_scaling_2k_32k.json")


def _build_prompt_at_length(tok: Any, target_tokens: int, bench_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Construct a valid AVeriTeC-formatted row whose serialized prompt has ~`target_tokens` tokens."""
    base_row = bench_rows[0]
    pool_passages: list[str] = []
    for r in bench_rows[1:80]:
        for e in r.get("evidence") or []:
            txt = str(e.get("text", "")).strip()
            if len(txt) > 200:
                pool_passages.append(txt)

    from spade.benchmark.prompt_builder import build_prompt
    ev = list(base_row.get("evidence") or [])
    idx_p = 0
    while True:
        prompt_str, phash = build_prompt(base_row["claim"], ev, mode="retrieved")
        ser = format_qwen_prompt(tok, prompt_str)
        n_tok = len(tok(ser, add_special_tokens=False)["input_ids"])
        if n_tok >= target_tokens:
            break
        nxt = pool_passages[idx_p % len(pool_passages)]
        ev.append({"evidence_id": f"E{len(ev)+1}", "text": nxt})
        idx_p += 1

    # Trim last passage proportionally to land closely near `target_tokens`
    excess = n_tok - target_tokens
    if excess > 20 and len(ev) > 1:
        last_txt = ev[-1]["text"]
        trim_chars = int(excess * 4.0)
        if trim_chars < len(last_txt) - 60:
            ev[-1]["text"] = last_txt[: len(last_txt) - trim_chars]
        else:
            ev.pop()
    prompt_str, phash = build_prompt(base_row["claim"], ev, mode="retrieved")
    ser = format_qwen_prompt(tok, prompt_str)
    n_tok = len(tok(ser, add_special_tokens=False)["input_ids"])
    return {
        "claim_id": f"scale-{target_tokens}",
        "claim": base_row["claim"],
        "evidence": ev,
        "evidence_mode": "retrieved",
        "prompt": prompt_str,
        "prompt_hash": phash,
        "actual_tokens": n_tok,
    }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    bench_rows = [json.loads(l) for l in BENCH_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    lengths = [2048, 4096, 8192, 16384, 32768]

    tok = AutoTokenizer.from_pretrained(TARGET_SNAP, local_files_only=True)
    synthetic_rows = {L: _build_prompt_at_length(tok, L, bench_rows) for L in lengths}

    # Part A: Standalone Component Prefill & Hook Scaling (Draft alone + Target + Hook)
    comp_results: dict[int, dict[str, Any]] = {}
    draft_only = AutoModelForCausalLM.from_pretrained(DRAFT_SNAP, dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()
    for L in lengths:
        inp = torch.randint(10, 5000, (1, L), device="cuda", dtype=torch.long)
        # Warmup
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
        draft_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        comp_results[L] = {
            "full_draft_prefill_ms": round(statistics.mean(times_ms), 2),
            "full_draft_standalone_vram_gb": round(draft_mem_gb, 2),
        }
        del inp
        torch.cuda.empty_cache()

    # SPADE capped draft prefill (~1,776 tokens)
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
    torch.cuda.empty_cache()

    # Part B: Target 8B + Hook component benchmark & Co-Located E2E Generation
    pair = QwenPair(TARGET_SNAP, DRAFT_SNAP, local_files_only=True, dtype="bfloat16", draft_attn_implementation="flex_attention")
    n_t_layers = len(pair.target.model.layers)
    hook_layers = resolve_layer_indices(n_t_layers, "upper50")

    # Prepare SPADE StaticDraftRunner (capped cache length for <=1776+128+64 -> 2560)
    pair.prepare_draft_runner(mode="compile_reduce_overhead", max_cache_len=2560, warmup_steps=3)

    # First measure Target 8B prefill + Hook exact latency/VRAM across 2k..32k
    for L in lengths:
        row = synthetic_rows[L]
        ser = format_qwen_prompt(tok, row["prompt"])
        enc = tok(ser, return_offsets_mapping=True, add_special_tokens=False)
        p_ids = list(enc["input_ids"])
        inp = torch.tensor([p_ids], dtype=torch.long, device="cuda")

        t_times = []
        hk_times = []
        hk_deltas_mb = []
        t_vram_gb = 0.0
        try:
            # 1 warmup
            with TargetPrefillAttentionHook(pair.target, layer_indices=hook_layers, tail_window=16) as hk:
                with torch.inference_mode():
                    o = pair.target(input_ids=inp, use_cache=True, logits_to_keep=1)
                _ = hk.compute_token_importance(o.past_key_values)
                del o
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            for _ in range(3):
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
                torch.cuda.empty_cache()

            comp_results[L]["actual_tokens"] = len(p_ids)
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
        except Exception as exc:
            comp_results[L]["target_error"] = str(exc)
        del inp
        torch.cuda.empty_cache()

    # Now measure E2E AR and SPADE generation across 2k..32k using the compiled SPADE runner
    for L in lengths:
        row = synthetic_rows[L]
        # Target AR
        try:
            torch.cuda.reset_peak_memory_stats()
            g_ar = pair.generate(
                row["prompt"], mode="B0-S", max_new_tokens=128, temperature=0.7, top_p=0.9,
                seed=42, cache_strategy="kv_cache", distribution_mode="fp32_fast",
            )
            e2e_s = g_ar.timings["prompt_prefill_s"] + g_ar.timings["decode_s"]
            comp_results[L]["ar_e2e_tok_s"] = round(len(g_ar.token_ids) / e2e_s, 2)
            comp_results[L]["ar_out_tokens"] = len(g_ar.token_ids)
            comp_results[L]["ar_peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
        except Exception as exc:
            comp_results[L]["ar_error"] = str(exc)
        torch.cuda.empty_cache()

        # SPADE
        try:
            torch.cuda.reset_peak_memory_stats()
            g_sp = pair.generate(
                row["prompt"], mode="B2", max_new_tokens=128, draft_length=4, temperature=0.7, top_p=0.9,
                seed=42, cache_strategy="kv_cache", distribution_mode="fp32_fast",
                target_attn_selector_cfg={"row": row, "budget_chars": 10000, "tail_window": 16, "min_sentences_per_passage": 1},
            )
            e2e_s = g_sp.timings["prompt_prefill_s"] + g_sp.timings.get("context_selection_s", 0.0) + g_sp.timings["draft_prefill_s"] + g_sp.timings["decode_s"]
            comp_results[L]["spade_e2e_tok_s"] = round(len(g_sp.token_ids) / e2e_s, 2)
            comp_results[L]["spade_out_tokens"] = len(g_sp.token_ids)
            comp_results[L]["spade_sel_total_ms"] = round(g_sp.timings.get("context_selection_s", 0.0) * 1000.0, 2)
            comp_results[L]["spade_peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
        except Exception as exc:
            comp_results[L]["spade_error"] = str(exc)
        torch.cuda.empty_cache()

    # Now measure Full-SD E2E across 2k..32k using the identical compiled StaticDraftRunner backend
    pair.draft_runner = None
    torch.cuda.empty_cache()
    for L in lengths:
        row = synthetic_rows[L]
        actual_p = int(comp_results[L].get("actual_tokens", L))
        cap = ((actual_p + 128 + 128 + 127) // 128) * 128
        try:
            pair.prepare_draft_runner(mode="compile_reduce_overhead", max_cache_len=cap, warmup_steps=3)
            torch.cuda.reset_peak_memory_stats()
            g_fl = pair.generate(
                row["prompt"], mode="B2", max_new_tokens=128, draft_length=4, temperature=0.7, top_p=0.9,
                seed=42, cache_strategy="kv_cache", distribution_mode="fp32_fast",
            )
            e2e_s = g_fl.timings["prompt_prefill_s"] + g_fl.timings["draft_prefill_s"] + g_fl.timings["decode_s"]
            comp_results[L]["full_sd_e2e_tok_s"] = round(len(g_fl.token_ids) / e2e_s, 2)
            comp_results[L]["full_sd_out_tokens"] = len(g_fl.token_ids)
            comp_results[L]["full_sd_peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
        except Exception as exc:
            comp_results[L]["full_sd_error"] = str(exc)
        finally:
            pair.draft_runner = None
            torch.cuda.empty_cache()

    OUT_PATH.write_text(json.dumps(comp_results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comp_results, indent=2))


if __name__ == "__main__":
    main()
