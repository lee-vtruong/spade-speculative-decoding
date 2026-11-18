"""Step B1.1-v2 Controlled Validation Suite:
  1. Workstream 1: Exact FlexAttention + StaticCache + In-Place BlockMask verification
     (float32 mathematical exactness + bfloat16 4-context benchmark with Qwen3-8B resident).
  2. Workstream 2: Exact CPU/GPU Timeline Decomposition of the eager 19-20 ms/step gap
     (GPU active interval union, GPU idle gaps, CUDA sync/memcpy, launch API, CPU dispatch).
  3. Workstream 3: Comprehensive FP32 distribution processing validation
     (a) 1,200+ real Qwen3-8B & Qwen3-0.6B raw logit vectors (support mismatch & TV),
     (b) 200 synthetic adversarial boundary stress cases,
     (c) 6,000+ paired accept/reject & residual distribution decision checks,
     (d) 15-claim live speculative B2 K=4 rollout comparison (orig_fp64 vs fast_fp32).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from spade.decoding.autoregressive import format_qwen_prompt, load_causal_lm
from spade.decoding.cache_utils import CacheManager
from spade.decoding.runtime_opt import (
    StaticDraftRunner,
    _supports_logits_to_keep,
    next_power_of_two_bucket,
    processed_distribution_fast,
)
from spade.decoding.sampling import (
    acceptance_probability,
    processed_distribution,
    residual_distribution,
    sample_distribution,
    total_variation,
)
from spade.decoding.speculative import exact_speculative_round


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _merge_intervals(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals.sort(key=lambda x: x[0])
    total = 0.0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            if e > cur_e:
                cur_e = e
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


def analyze_eager_timeline(
    draft_model: Any,
    prompt_ids: list[int],
    *,
    n_steps: int = 10,
) -> dict[str, Any]:
    """Export chrome trace from `torch.profiler` and compute exact GPU active vs idle vs sync timeline."""
    import tempfile

    device = draft_model.get_input_embeddings().weight.device
    l2k = _supports_logits_to_keep(draft_model)
    mgr = CacheManager()
    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    kw: dict[str, Any] = {"input_ids": inp, "use_cache": True}
    if l2k:
        kw["logits_to_keep"] = 1
    with torch.inference_mode():
        out = draft_model(**kw)
    mgr.update(out.past_key_values)
    tok = int(out.logits[0, -1].argmax().item())
    del out, inp

    for _ in range(3):
        sinp = torch.tensor([[tok]], dtype=torch.long, device=device)
        with torch.inference_mode():
            sout = draft_model(input_ids=sinp, past_key_values=mgr.past_key_values, use_cache=True)
        mgr.update(sout.past_key_values)
        tok = int(sout.logits[0, -1].argmax().item())
    _sync(device)

    # Measure un-profiled steady wall latency first (profiler itself adds CPU tracing overhead!)
    step_wall_unprofiled: list[float] = []
    for _ in range(n_steps):
        _sync(device)
        t0 = time.perf_counter()
        sinp = torch.tensor([[tok]], dtype=torch.long, device=device)
        with torch.inference_mode():
            sout = draft_model(input_ids=sinp, past_key_values=mgr.past_key_values, use_cache=True)
        _sync(device)
        step_wall_unprofiled.append((time.perf_counter() - t0) * 1000.0)
        mgr.update(sout.past_key_values)
        tok = int(sout.logits[0, -1].argmax().item())

    # Now record CUDA-only + CPU-runtime timeline (minimize profiler perturbation with record_shapes=False)
    _sync(device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(n_steps):
            sinp = torch.tensor([[tok]], dtype=torch.long, device=device)
            with torch.inference_mode():
                sout = draft_model(input_ids=sinp, past_key_values=mgr.past_key_values, use_cache=True)
            mgr.update(sout.past_key_values)
            tok = int(sout.logits[0, -1].argmax().item())
        _sync(device)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        trace_path = Path(tf.name)
    prof.export_chrome_trace(str(trace_path))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_path.unlink(missing_ok=True)

    gpu_intervals: list[tuple[float, float]] = []
    launch_intervals: list[tuple[float, float]] = []
    sync_intervals: list[tuple[float, float]] = []
    memcpy_intervals: list[tuple[float, float]] = []
    launch_count = 0
    sync_count = 0
    memcpy_count = 0

    for ev in trace.get("traceEvents", []):
        if ev.get("ph") != "X":
            continue
        cat = str(ev.get("cat", "")).lower()
        name = str(ev.get("name", ""))
        ts = float(ev.get("ts", 0.0))
        dur = float(ev.get("dur", 0.0))
        if dur <= 0:
            continue
        if cat in {"kernel", "gpu_memcpy", "gpu_memset"}:
            gpu_intervals.append((ts, ts + dur))
        elif cat == "cuda_runtime":
            lname = name.lower()
            if "launch" in lname:
                launch_intervals.append((ts, ts + dur))
                launch_count += 1
            elif "sync" in lname:
                sync_intervals.append((ts, ts + dur))
                sync_count += 1
            elif "memcpy" in lname:
                memcpy_intervals.append((ts, ts + dur))
                memcpy_count += 1

    gpu_active_us = _merge_intervals(gpu_intervals)
    gpu_span_us = (max(e for _, e in gpu_intervals) - min(s for s, _ in gpu_intervals)) if gpu_intervals else 0.0
    gpu_idle_us = max(0.0, gpu_span_us - gpu_active_us)
    launch_us = _merge_intervals(launch_intervals)
    sync_us = _merge_intervals(sync_intervals)
    memcpy_us = _merge_intervals(memcpy_intervals)

    # Inter-kernel idle gaps distribution
    gpu_intervals.sort(key=lambda x: x[0])
    gaps_us: list[float] = []
    cur_end = gpu_intervals[0][1] if gpu_intervals else 0.0
    for s, e in gpu_intervals[1:]:
        if s > cur_end:
            gaps_us.append(s - cur_end)
        if e > cur_end:
            cur_end = e

    mgr.reset()
    torch.cuda.empty_cache()

    unprof_mean = statistics.mean(step_wall_unprofiled)
    gpu_active_ms = (gpu_active_us / 1000.0) / n_steps
    gpu_span_ms = (gpu_span_us / 1000.0) / n_steps
    gpu_idle_ms = (gpu_idle_us / 1000.0) / n_steps
    # Estimate true unprofiled idle gap (GPU active time is unaffected by CPU profiler overhead!)
    est_unprof_idle_ms = max(0.0, unprof_mean - gpu_active_ms)

    return {
        "prompt_tokens": len(prompt_ids),
        "n_steps": n_steps,
        "unprofiled_wall_ms_per_step": round(unprof_mean, 3),
        "profiled_gpu_span_ms_per_step": round(gpu_span_ms, 3),
        "gpu_active_union_ms_per_step": round(gpu_active_ms, 3),
        "gpu_idle_gap_profiled_ms_per_step": round(gpu_idle_ms, 3),
        "est_gpu_idle_unprofiled_ms_per_step": round(est_unprof_idle_ms, 3),
        "gpu_busy_pct_unprofiled": round((gpu_active_ms / unprof_mean) * 100.0, 2) if unprof_mean > 0 else 0.0,
        "cuda_launch_calls_per_step": round(launch_count / n_steps, 1),
        "cuda_launch_host_ms_per_step": round((launch_us / 1000.0) / n_steps, 3),
        "cuda_sync_calls_per_step": round(sync_count / n_steps, 1),
        "cuda_sync_host_ms_per_step": round((sync_us / 1000.0) / n_steps, 3),
        "cuda_memcpy_calls_per_step": round(memcpy_count / n_steps, 1),
        "cuda_memcpy_host_ms_per_step": round((memcpy_us / 1000.0) / n_steps, 3),
        "n_inter_kernel_gaps_per_step": round(len(gaps_us) / n_steps, 1),
        "median_inter_kernel_gap_us": round(statistics.median(gaps_us), 2) if gaps_us else 0.0,
        "p95_inter_kernel_gap_us": round(sorted(gaps_us)[int(0.95 * len(gaps_us))], 2) if gaps_us else 0.0,
    }


def run_workstream_1_flex_validation(
    *,
    target_path: str,
    draft_path: str,
    tokenizer: Any,
    source_jsonl: Path,
    n_steady_steps: int = 80,
) -> dict[str, Any]:
    """Workstream 1: Float32 exactness proof + Bfloat16 benchmark of FlexAttention + StaticCache + BlockMask."""
    from transformers import AutoModelForCausalLM
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from profile_b1_1_draft_runtime import _build_token_sequence, _run_eager_dynamic_sequence

    device = torch.device("cuda")
    rb = {4: 2, 8: 3}

    # 1A) Float32 mathematical exactness proof across prefill + 12 steps + 2 rollbacks
    m_sdpa_f32 = AutoModelForCausalLM.from_pretrained(draft_path, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa").to(device).eval()
    m_flex_f32 = AutoModelForCausalLM.from_pretrained(draft_path, local_files_only=True, dtype=torch.float32, attn_implementation="flex_attention").to(device).eval()
    p_short = _build_token_sequence(tokenizer, 320, source_jsonl)
    steps_short = [p_short[-(i % 16) - 1] for i in range(12)]
    ref_f32 = _run_eager_dynamic_sequence(m_sdpa_f32, p_short, steps_short, rollback_plan=rb)

    f32_results: dict[str, Any] = {}
    for mode in ("eager_static", "compile_default", "compile_reduce_overhead"):
        runner = StaticDraftRunner(m_flex_f32, max_cache_len=1024, mode=mode, device=device, dtype=torch.float32)
        runner.prepare(warmup_steps=3)
        got = [runner.prefill(p_short).clone()]
        for idx, tok in enumerate(steps_short):
            if idx in rb:
                runner.rollback(runner.cur_len - rb[idx])
            got.append(runner.step(tok).clone())
        lds = [float((a - b).abs().max()) for a, b in zip(ref_f32, got)]
        tvs = [total_variation(processed_distribution(a, 0.7, 0, 0.9), processed_distribution(b, 0.7, 0, 0.9)) for a, b in zip(ref_f32, got)]
        f32_results[mode] = {
            "prefill_max_logit_diff": round(lds[0], 8),
            "overall_max_logit_diff": round(max(lds), 8),
            "pre_rollback_max_tv": round(max(tvs[:5]), 8),
            "post_rollback_max_tv": round(max(tvs[5:]), 8),
            "fp32_exactness_pass": bool(max(lds) < 1e-3 and max(tvs) < 1e-4),
        }
        del runner
        torch.cuda.empty_cache()

    del m_sdpa_f32, m_flex_f32
    gc.collect()
    torch.cuda.empty_cache()

    # 1B) Bfloat16 benchmark across all 4 context lengths with Qwen3-8B resident
    target_model = load_causal_lm(target_path, local_files_only=True, dtype="bfloat16")
    m_sdpa = AutoModelForCausalLM.from_pretrained(draft_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    m_flex = AutoModelForCausalLM.from_pretrained(draft_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="flex_attention").to(device).eval()

    contexts = [("short", 320), ("selective_top2", 2560), ("full_long", 6150), ("very_long", 12288)]
    bf16_rows: list[dict[str, Any]] = []

    for label, p_len in contexts:
        cap = next_power_of_two_bucket(p_len, min_bucket=512, headroom=256)
        prompt_ids = _build_token_sequence(tokenizer, p_len, source_jsonl)
        step_toks = [prompt_ids[-(i % 16) - 1] for i in range(12)]
        ref_dyn = _run_eager_dynamic_sequence(m_sdpa, prompt_ids, step_toks, rollback_plan=rb)

        # Measure eager_dynamic_cache baseline speed
        mgr = CacheManager()
        inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            out = m_sdpa(input_ids=inp, use_cache=True, logits_to_keep=1)
        mgr.update(out.past_key_values)
        tok = int(out.logits[0, -1].argmax().item())
        del out, inp
        for _ in range(3):
            with torch.inference_mode():
                sout = m_sdpa(input_ids=torch.tensor([[tok]], device=device), past_key_values=mgr.past_key_values, use_cache=True)
            mgr.update(sout.past_key_values)
            tok = int(sout.logits[0, -1].argmax().item())
        _sync(device)
        e_times = []
        for _ in range(n_steady_steps):
            _sync(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                sout = m_sdpa(input_ids=torch.tensor([[tok]], device=device), past_key_values=mgr.past_key_values, use_cache=True)
            _sync(device)
            e_times.append((time.perf_counter() - t0) * 1000.0)
            mgr.update(sout.past_key_values)
            tok = int(sout.logits[0, -1].argmax().item())
        mgr.reset()
        eager_mean_ms = statistics.mean(e_times)

        # Evaluate flex modes: eager_static, compile_default, compile_reduce_overhead, cudagraph_static
        ref_ef: list[torch.Tensor] | None = None
        for mode in ("eager_static", "compile_default", "compile_reduce_overhead", "cudagraph_static"):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            row: dict[str, Any] = {
                "context_label": label,
                "prompt_tokens": p_len,
                "max_cache_len": cap,
                "mode": f"flex_{mode}",
                "eager_dynamic_ms": round(eager_mean_ms, 3),
                "status": "OK",
                "error": None,
            }
            try:
                runner = StaticDraftRunner(m_flex, max_cache_len=cap, mode=mode, device=device, dtype=torch.bfloat16)
                setup = runner.prepare(warmup_steps=3)

                got = [runner.prefill(prompt_ids).clone()]
                for idx, st_tok in enumerate(step_toks):
                    if idx in rb:
                        runner.rollback(runner.cur_len - rb[idx])
                    got.append(runner.step(st_tok).clone())
                if mode == "eager_static":
                    ref_ef = [x.clone() for x in got]

                ld_dyn = [float((a - b).abs().max()) for a, b in zip(ref_dyn, got)]
                tv_dyn = [total_variation(processed_distribution(a, 0.7, 0, 0.9), processed_distribution(b, 0.7, 0, 0.9)) for a, b in zip(ref_dyn, got)]
                tv_ef = [total_variation(processed_distribution(a, 0.7, 0, 0.9), processed_distribution(b, 0.7, 0, 0.9)) for a, b in zip(ref_ef, got)] if ref_ef else [0.0]

                _sync(device)
                t_p0 = time.perf_counter()
                lg = runner.prefill(prompt_ids)
                _sync(device)
                prefill_ms = (time.perf_counter() - t_p0) * 1000.0
                tok = int(lg.argmax().item())
                for _ in range(3):
                    tok = int(runner.step(tok).argmax().item())
                _sync(device)
                times = []
                for _ in range(n_steady_steps):
                    _sync(device)
                    t0 = time.perf_counter()
                    lg = runner.step(tok)
                    _sync(device)
                    times.append((time.perf_counter() - t0) * 1000.0)
                    tok = int(lg.argmax().item())

                mean_ms = statistics.mean(times)
                peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                # Subtract duplicate m_sdpa weights (~1140 MB) to also report single-draft realistic peak VRAM
                row.update({
                    "compile_s": round(setup.compile_s, 3),
                    "warmup_s": round(setup.warmup_s, 4),
                    "graph_capture_s": round(setup.graph_capture_s, 4),
                    "prefill_ms": round(prefill_ms, 2),
                    "steady_mean_ms": round(mean_ms, 3),
                    "steady_median_ms": round(statistics.median(times), 3),
                    "steady_p95_ms": round(sorted(times)[int(0.95 * len(times))], 3),
                    "est_4step_ms": round(4.0 * mean_ms, 2),
                    "speedup_vs_eager": round(eager_mean_ms / mean_ms, 2) if mean_ms > 0 else 0.0,
                    "peak_vram_mb_single_draft_est": round(max(0.0, peak_mb - 1140.0), 1),
                    "peak_vram_mb_raw": round(peak_mb, 1),
                    "max_logit_diff_vs_dyn": round(max(ld_dyn), 4),
                    "pre_rb_tv_vs_dyn": round(max(tv_dyn[:5]), 6),
                    "post_rb_tv_vs_dyn": round(max(tv_dyn[5:]), 6),
                    "max_tv_vs_eager_flex": round(max(tv_ef), 6),
                    "post_rb_tv_vs_eager_flex": round(max(tv_ef[5:]), 6),
                })
                del runner
            except Exception as exc:
                row["status"] = "ERROR"
                row["error"] = f"{type(exc).__name__}: {exc}"
            bf16_rows.append(row)
            torch.cuda.empty_cache()

    return {
        "fp32_exactness": f32_results,
        "bf16_benchmark": bf16_rows,
        "models": (target_model, m_sdpa, m_flex),
    }


def run_workstream_3_fp32_audit(
    *,
    target_model: Any,
    draft_model: Any,
    tokenizer: Any,
    source_jsonl: Path,
    n_claims_logits: int = 15,
    n_claims_rollout: int = 15,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict[str, Any]:
    """Workstream 3: Comprehensive 4-part FP32 distribution processing validation."""
    device = target_model.get_input_embeddings().weight.device
    rows = [json.loads(l) for l in source_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]

    # ---- (3a) & (3c): Real raw logits (target 8B + draft 0.6B) from 15 real claims (~1,200+ prefixes) ----
    # Generate reference tokens first with target+draft B2 orig_fp64 on 15 claims, AND collect exact logits!
    all_target_logits: list[torch.Tensor] = []
    all_draft_logits: list[torch.Tensor] = []

    def _run_live_b2_claim(row: dict[str, Any], use_fast_fp32: bool, max_new: int = 128, seed: int = 42, collect_logits: bool = False) -> dict[str, Any]:
        ser = format_qwen_prompt(tokenizer, row["prompt"])
        prompt_ids = tokenizer(ser, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        eos_id = tokenizer.eos_token_id
        t_mgr = CacheManager()
        d_mgr = CacheManager()
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        def _dist1(lg: torch.Tensor) -> torch.Tensor:
            if use_fast_fp32:
                return processed_distribution_fast(lg, temperature=temperature, top_k=0, top_p=top_p, dtype=torch.float32)
            return processed_distribution(lg, temperature=temperature, top_k=0, top_p=top_p)

        def _dist_batch(lgs: torch.Tensor) -> list[torch.Tensor]:
            if use_fast_fp32:
                b = processed_distribution_fast(lgs, temperature=temperature, top_k=0, top_p=top_p, dtype=torch.float32)
                return [b[i] for i in range(b.shape[0])]
            return [processed_distribution(lgs[i], temperature=temperature, top_k=0, top_p=top_p) for i in range(lgs.shape[0])]

        _sync(device)
        t_e2e0 = time.perf_counter()
        with torch.inference_mode():
            tinp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            tout = target_model(input_ids=tinp, use_cache=True, logits_to_keep=1)
            t_mgr.update(tout.past_key_values)
            next_t_lg = tout.logits[0, -1].float()
            del tout, tinp

            dinp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            dout = draft_model(input_ids=dinp, use_cache=True, logits_to_keep=1)
            d_mgr.update(dout.past_key_values)
            next_d_lg = dout.logits[0, -1].float()
            del dout, dinp

        generated: list[int] = []
        pending_tok: int | None = None
        n_rounds = 0
        n_eval = 0
        n_eval_acc = 0
        n_prop = 0
        n_acc = 0
        dist_s_total = 0.0

        while len(generated) < max_new:
            rem = max_new - len(generated)
            k = min(4, rem)
            t_len0 = t_mgr.seq_length
            d_len0 = d_mgr.seq_length
            proposed: list[int] = []
            q_dists: list[torch.Tensor] = []
            d_lgs_round: list[torch.Tensor] = []
            cur_d = next_d_lg

            for idx in range(k):
                if collect_logits:
                    d_lgs_round.append(cur_d.detach().cpu())
                td0 = time.perf_counter()
                q = _dist1(cur_d)
                dist_s_total += time.perf_counter() - td0
                tok = sample_distribution(q, gen)
                proposed.append(tok)
                q_dists.append(q)
                if tok == eos_id:
                    break
                if idx < k - 1:
                    with torch.inference_mode():
                        sdout = draft_model(input_ids=torch.tensor([[tok]], device=device), past_key_values=d_mgr.past_key_values, use_cache=True, logits_to_keep=1)
                    d_mgr.update(sdout.past_key_values)
                    cur_d = sdout.logits[0, -1].float()

            verify_toks = ([pending_tok] + proposed) if pending_tok is not None else proposed
            has_pending = pending_tok is not None
            with torch.inference_mode():
                vtout = target_model(input_ids=torch.tensor([verify_toks], device=device), past_key_values=t_mgr.past_key_values, use_cache=True)
            t_mgr.update(vtout.past_key_values)

            if has_pending:
                t_stack = vtout.logits[0, : len(proposed) + 1].float()
            else:
                t_stack = torch.cat([next_t_lg.unsqueeze(0), vtout.logits[0, : len(proposed)].float()], dim=0)

            if collect_logits:
                for i in range(len(proposed)):
                    all_target_logits.append(t_stack[i].detach().cpu())
                    all_draft_logits.append(d_lgs_round[i])

            td0 = time.perf_counter()
            all_p = _dist_batch(t_stack)
            dist_s_total += time.perf_counter() - td0
            p_dists = all_p[: len(proposed)]
            t_extra = all_p[len(proposed)]

            res = exact_speculative_round(proposed, q_dists, p_dists, generator=gen, eos_token_id=eos_id, remaining=rem, target_extra=t_extra)
            generated.extend(res.committed)
            n_rounds += 1
            n_prop += len(proposed)
            n_acc += res.accepted
            for st in res.statuses:
                if st in {"accepted", "rejected"}:
                    n_eval += 1
                    if st == "accepted":
                        n_eval_acc += 1

            if res.rejected_position is not None:
                t_mgr.rollback(t_len0 + (1 if has_pending else 0) + res.rejected_position)
                d_mgr.rollback(d_len0 + res.rejected_position)

            last_tok = generated[-1]
            if last_tok == eos_id or len(generated) >= max_new:
                break

            if res.rejected_position is not None:
                pending_tok = last_tok
                next_t_lg = None
                upd_toks = [last_tok]
            elif res.extra_target is not None:
                pending_tok = last_tok
                next_t_lg = None
                upd_toks = [proposed[-1], last_tok]
            else:
                pending_tok = None
                next_t_lg = vtout.logits[0, -1].float()
                upd_toks = [proposed[-1]]

            with torch.inference_mode():
                udout = draft_model(input_ids=torch.tensor([upd_toks], device=device), past_key_values=d_mgr.past_key_values, use_cache=True, logits_to_keep=1)
            d_mgr.update(udout.past_key_values)
            next_d_lg = udout.logits[0, -1].float()

        _sync(device)
        e2e_s = time.perf_counter() - t_e2e0
        t_mgr.reset()
        d_mgr.reset()
        if len(prompt_ids) >= 4096:
            torch.cuda.empty_cache()

        return {
            "claim_id": str(row["claim_id"]),
            "prompt_tokens": len(prompt_ids),
            "token_ids": generated,
            "n_tokens": len(generated),
            "n_rounds": n_rounds,
            "n_eval": n_eval,
            "n_eval_acc": n_eval_acc,
            "n_prop": n_prop,
            "n_acc": n_acc,
            "e2e_s": e2e_s,
            "dist_s": dist_s_total,
        }

    # Filter claims with reasonable length (<10k tokens) for fast, clean 15-claim live rollout comparison
    selected_rows = []
    for r in rows:
        ser = format_qwen_prompt(tokenizer, r["prompt"])
        if len(ser) < 35000:
            selected_rows.append(r)
        if len(selected_rows) >= max(n_claims_logits, n_claims_rollout):
            break

    # Warmup 1 claim
    _ = _run_live_b2_claim(selected_rows[0], use_fast_fp32=False, max_new=32, seed=42, collect_logits=False)
    _ = _run_live_b2_claim(selected_rows[0], use_fast_fp32=True, max_new=32, seed=42, collect_logits=False)

    orig_runs = []
    fast_runs = []
    for idx, r in enumerate(selected_rows[:n_claims_rollout]):
        o = _run_live_b2_claim(r, use_fast_fp32=False, max_new=128, seed=42, collect_logits=(idx < n_claims_logits))
        f = _run_live_b2_claim(r, use_fast_fp32=True, max_new=128, seed=42, collect_logits=False)
        orig_runs.append(o)
        fast_runs.append(f)

    # ---- (3a) Raw real logits statistical audit ----
    n_pairs = len(all_target_logits)
    abs_diffs: list[float] = []
    tv_diffs: list[float] = []
    support_mismatches = 0
    max_boundary_mass = 0.0

    # ---- (3c) Paired speculative accept/reject & residual decision audit (5 seeds per pair!) ----
    n_decision_trials = 0
    decision_flips = 0
    accept_prob_diffs: list[float] = []
    residual_tvs: list[float] = []
    rng_cpu = torch.Generator(device="cpu")
    rng_cpu.manual_seed(2026)

    for p_lg_cpu, q_lg_cpu in zip(all_target_logits, all_draft_logits):
        p_lg = p_lg_cpu.to(device)
        q_lg = q_lg_cpu.to(device)
        p64 = processed_distribution(p_lg, temperature=temperature, top_k=0, top_p=top_p)
        p32 = processed_distribution_fast(p_lg, temperature=temperature, top_k=0, top_p=top_p, dtype=torch.float32)
        q64 = processed_distribution(q_lg, temperature=temperature, top_k=0, top_p=top_p)
        q32 = processed_distribution_fast(q_lg, temperature=temperature, top_k=0, top_p=top_p, dtype=torch.float32)

        for d64, d32 in ((p64, p32), (q64, q32)):
            abs_diffs.append(float((d64 - d32.to(torch.float64)).abs().max().item()))
            tv_diffs.append(total_variation(d64, d32))
            s64 = d64 > 0
            s32 = d32 > 0
            sym_diff = s64 ^ s32
            if bool(sym_diff.any().item()):
                support_mismatches += 1
                bm = float(torch.maximum(d64[sym_diff], d32.to(torch.float64)[sym_diff]).sum().item())
                max_boundary_mass = max(max_boundary_mass, bm)

        # 5 paired speculative acceptance trials on this (p, q) pair
        q64_cpu = q64.cpu()
        p64_cpu = p64.cpu()
        q32_cpu = q32.cpu().to(torch.float64)
        p32_cpu = p32.cpu().to(torch.float64)
        toks = torch.multinomial(q64_cpu, 5, replacement=True, generator=rng_cpu).tolist()
        us = torch.rand(5, generator=rng_cpu, dtype=torch.float64).tolist()
        any_rejected = False
        for tok, u in zip(toks, us):
            ap64 = float(acceptance_probability(p64_cpu, q64_cpu, tok).item())
            if q32_cpu[tok] <= 0:
                ap32 = 1.0 if p32_cpu[tok] > 0 else 0.0
            else:
                ap32 = float(min(1.0, (p32_cpu[tok] / q32_cpu[tok]).item()))
            accept_prob_diffs.append(abs(ap64 - ap32))
            acc64 = u < ap64
            acc32 = u < ap32
            n_decision_trials += 1
            if acc64 != acc32:
                decision_flips += 1
            if not acc64:
                any_rejected = True
        if any_rejected:
            try:
                r64 = residual_distribution(p64_cpu, q64_cpu)
                r32 = residual_distribution(p32_cpu, q32_cpu)
                residual_tvs.append(total_variation(r64, r32))
            except ValueError:
                pass

    # ---- (3b) Synthetic / Adversarial Boundary Stress Suite (200 cases) ----
    adv_rng = torch.Generator(device="cpu")
    adv_rng.manual_seed(999)
    adv_categories: dict[str, dict[str, Any]] = {}

    def _check_adv_batch(name: str, logit_list: list[torch.Tensor], temp: float = 0.7, tp: float = 0.9) -> None:
        m_tv = 0.0
        sup_mis = 0
        max_bm = 0.0
        for lg in logit_list:
            g = lg.to(device)
            d64 = processed_distribution(g, temperature=temp, top_k=0, top_p=tp)
            d32 = processed_distribution_fast(g, temperature=temp, top_k=0, top_p=tp, dtype=torch.float32)
            m_tv = max(m_tv, total_variation(d64, d32))
            sd = (d64 > 0) ^ (d32 > 0)
            if bool(sd.any().item()):
                sup_mis += 1
                max_bm = max(max_bm, float(torch.maximum(d64[sd], d32.to(torch.float64)[sd]).sum().item()))
        adv_categories[name] = {
            "n_cases": len(logit_list),
            "max_tv_diff": m_tv,
            "support_mismatch_count": sup_mis,
            "support_mismatch_rate_pct": round(sup_mis * 100.0 / len(logit_list), 2),
            "max_boundary_mass": max_bm,
        }

    V = 151936
    # 1) Exact top-p boundary ties (cumsum right at 0.9 +/- 1e-7)
    tie_cases = []
    for eps in [-1e-7, -5e-8, 0.0, 5e-8, 1e-7] * 8:
        lg = torch.full((V,), -40.0, dtype=torch.float32)
        # First 9 tokens have prob 0.1 each -> cumsum=0.9 exactly at token 8; perturbation eps shifts boundary!
        probs_target = torch.tensor([0.1] * 8 + [0.1 + eps, 0.05, 0.05 - eps], dtype=torch.float64)
        lg[: len(probs_target)] = torch.log(probs_target).to(torch.float32) * 0.7
        tie_cases.append(lg)
    _check_adv_batch("adversarial_topp_tie_1e-7", tie_cases, temp=0.7, tp=0.9)

    # 2) Extreme dynamic range [-100, +100]
    dyn_cases = [torch.empty(V).uniform_(-100.0, 100.0, generator=adv_rng) for _ in range(40)]
    _check_adv_batch("extreme_dynamic_range_pm100", dyn_cases, temp=0.7, tp=0.9)

    # 3) Low temperature 0.1 (very sharp)
    low_t_cases = [torch.randn(V, generator=adv_rng) * 2.0 for _ in range(40)]
    _check_adv_batch("sharp_temp_0.1", low_t_cases, temp=0.1, tp=0.9)

    # 4) High temperature 1.5 (wide tail across 151k vocab!)
    high_t_cases = [torch.randn(V, generator=adv_rng) * 2.0 for _ in range(40)]
    _check_adv_batch("wide_tail_temp_1.5", high_t_cases, temp=1.5, tp=0.9)

    # 5) Flat uniform + near-uniform
    flat_cases = [torch.zeros(V, dtype=torch.float32) + torch.randn(V, generator=adv_rng) * (1e-5 * (i % 5)) for i in range(40)]
    _check_adv_batch("near_uniform_151k", flat_cases, temp=0.7, tp=0.9)

    # ---- (3d) Live rollout summary (orig_fp64 vs fast_fp32) ----
    identical_claims = 0
    first_div_positions: list[int] = []
    for o, f in zip(orig_runs, fast_runs):
        if o["token_ids"] == f["token_ids"]:
            identical_claims += 1
        else:
            div_pos = next((i for i, (a, b) in enumerate(zip(o["token_ids"], f["token_ids"])) if a != b), min(len(o["token_ids"]), len(f["token_ids"])))
            first_div_positions.append(div_pos)

    def _agg(runs: list[dict[str, Any]]) -> dict[str, Any]:
        tot_tok = sum(r["n_tokens"] for r in runs)
        tot_e2e = sum(r["e2e_s"] for r in runs)
        tot_rnd = sum(r["n_rounds"] for r in runs)
        tot_ev = sum(r["n_eval"] for r in runs)
        tot_eva = sum(r["n_eval_acc"] for r in runs)
        tot_pr = sum(r["n_prop"] for r in runs)
        tot_ac = sum(r["n_acc"] for r in runs)
        tot_dist = sum(r["dist_s"] for r in runs)
        return {
            "n_claims": len(runs),
            "total_tokens": tot_tok,
            "total_rounds": tot_rnd,
            "alpha_eval_pct": round(tot_eva * 100.0 / tot_ev, 3) if tot_ev else 0.0,
            "alpha_prop_pct": round(tot_ac * 100.0 / tot_pr, 3) if tot_pr else 0.0,
            "committed_per_round": round(tot_tok / tot_rnd, 3) if tot_rnd else 0.0,
            "e2e_tok_s": round(tot_tok / tot_e2e, 2) if tot_e2e > 0 else 0.0,
            "e2e_ms_per_claim": round((tot_e2e * 1000.0) / len(runs), 2),
            "dist_ms_per_round": round((tot_dist * 1000.0) / tot_rnd, 3) if tot_rnd else 0.0,
        }

    abs_diffs.sort()
    tv_diffs.sort()
    accept_prob_diffs.sort()
    return {
        "real_raw_logits_audit": {
            "n_logit_vectors": len(abs_diffs),
            "n_pq_pairs": n_pairs,
            "max_abs_prob_diff": max(abs_diffs) if abs_diffs else 0.0,
            "p99_abs_prob_diff": abs_diffs[int(0.99 * len(abs_diffs))] if abs_diffs else 0.0,
            "max_tv_diff": max(tv_diffs) if tv_diffs else 0.0,
            "p99_tv_diff": tv_diffs[int(0.99 * len(tv_diffs))] if tv_diffs else 0.0,
            "mean_tv_diff": sum(tv_diffs) / len(tv_diffs) if tv_diffs else 0.0,
            "support_mismatch_count": support_mismatches,
            "support_mismatch_rate_pct": round(support_mismatches * 100.0 / len(abs_diffs), 4) if abs_diffs else 0.0,
            "max_support_boundary_mass": max_boundary_mass,
        },
        "adversarial_stress_suite": adv_categories,
        "paired_verification_decisions": {
            "n_decision_trials": n_decision_trials,
            "decision_flips": decision_flips,
            "decision_mismatch_rate_pct": round(decision_flips * 100.0 / n_decision_trials, 5) if n_decision_trials else 0.0,
            "max_abs_accept_prob_diff": max(accept_prob_diffs) if accept_prob_diffs else 0.0,
            "p99_abs_accept_prob_diff": accept_prob_diffs[int(0.99 * len(accept_prob_diffs))] if accept_prob_diffs else 0.0,
            "n_residual_checks": len(residual_tvs),
            "residual_max_tv": max(residual_tvs) if residual_tvs else 0.0,
            "residual_mean_tv": sum(residual_tvs) / len(residual_tvs) if residual_tvs else 0.0,
        },
        "live_rollout_comparison": {
            "identical_token_trajectories": f"{identical_claims}/{len(orig_runs)}",
            "first_divergence_positions": first_div_positions,
            "orig_fp64": _agg(orig_runs),
            "fast_fp32": _agg(fast_runs),
        },
    }


def render_v2_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Step B1.1-v2 — Controlled Validation Report (FlexAttention, Timeline & FP32 Audit)",
        "",
        "## 1. Workstream 1 — Fixed FlexAttention + StaticCache + Exact In-Place BlockMask",
        "",
        "### 1a. Float32 Mathematical Exactness Proof (Prefill + 12 Steps + 2 Rollbacks `{4: 2, 8: 3}`)",
        "",
        "| Mode | Prefill Max Logit Diff | Overall Max Logit Diff | Pre-RB Max TV | Post-RB Max TV | FP32 Exactness |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for mode, v in report["workstream_1_flex"]["fp32_exactness"].items():
        lines.append(
            f"| `flex_{mode}` | `{v['prefill_max_logit_diff']:.2e}` | `{v['overall_max_logit_diff']:.2e}` | "
            f"`{v['pre_rollback_max_tv']:.2e}` | `{v['post_rollback_max_tv']:.2e}` | `{'PASS' if v['fp32_exactness_pass'] else 'FAIL'}` |"
        )

    lines.extend([
        "",
        "### 1b. Bfloat16 Benchmark Across 4 Context Lengths (with Qwen3-8B Resident on RTX 5090)",
        "",
        "| Context | Mode | Compile (s) | Warmup (s) | Capture (s) | Prefill (ms) | Steady Mean (ms) | Median (ms) | p95 (ms) | 4-Step (ms) | Speedup vs Eager | Est. Peak VRAM (MB) | Max LD vs Dyn | Post-RB TV vs Dyn | Post-RB TV vs EagerFlex |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in report["workstream_1_flex"]["bf16_benchmark"]:
        if r["status"] != "OK":
            lines.append(f"| {r['context_label']} ({r['prompt_tokens']}) | `{r['mode']}` | ERROR: `{r['error']}` | - | - | - | - | - | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {r['context_label']} ({r['prompt_tokens']}) | `{r['mode']}` | {r['compile_s']:.2f} | {r['warmup_s']:.3f} | {r['graph_capture_s']:.3f} | "
            f"{r['prefill_ms']:.1f} | **{r['steady_mean_ms']:.3f}** | {r['steady_median_ms']:.3f} | {r['steady_p95_ms']:.3f} | "
            f"**{r['est_4step_ms']:.2f}** | **{r['speedup_vs_eager']:.2f}x** | {r['peak_vram_mb_single_draft_est']:.0f} | "
            f"{r['max_logit_diff_vs_dyn']:.4f} | {r['post_rb_tv_vs_dyn']:.5f} | {r['post_rb_tv_vs_eager_flex']:.5f} |"
        )

    lines.extend([
        "",
        "## 2. Workstream 2 — Exact CPU/GPU Timeline Decomposition (Eager DynamicCache Draft Step)",
        "",
        "| Context | Unprofiled Wall (ms) | GPU Active Union (ms) | Est. GPU Idle Unprof (ms) | GPU Busy % | Launch Calls/step | Launch Host (ms) | Sync Calls/step | Sync Host (ms) | Memcpy Calls/step | Inter-Kernel Gaps/step | Median Gap (us) | p95 Gap (us) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for lbl, tl in report["workstream_2_timeline"].items():
        lines.append(
            f"| {lbl} ({tl['prompt_tokens']}) | **{tl['unprofiled_wall_ms_per_step']:.3f}** | **{tl['gpu_active_union_ms_per_step']:.3f}** | "
            f"{tl['est_gpu_idle_unprofiled_ms_per_step']:.3f} | {tl['gpu_busy_pct_unprofiled']:.1f}% | "
            f"{tl['cuda_launch_calls_per_step']} | {tl['cuda_launch_host_ms_per_step']:.3f} | "
            f"{tl['cuda_sync_calls_per_step']} | {tl['cuda_sync_host_ms_per_step']:.3f} | "
            f"{tl['cuda_memcpy_calls_per_step']} | {tl['n_inter_kernel_gaps_per_step']} | "
            f"{tl['median_inter_kernel_gap_us']:.1f} | {tl['p95_inter_kernel_gap_us']:.1f} |"
        )

    w3 = report["workstream_3_fp32_audit"]
    rl = w3["real_raw_logits_audit"]
    pv = w3["paired_verification_decisions"]
    lr = w3["live_rollout_comparison"]
    lines.extend([
        "",
        "## 3. Workstream 3 — Comprehensive FP32 Distribution Processing Validation",
        "",
        "### 3a. Real Raw Logits Audit (Qwen3-8B + Qwen3-0.6B on Real AVeriTeC Claims)",
        f"- Checked `{rl['n_logit_vectors']}` real logit vectors (`{rl['n_pq_pairs']}` `(p, q)` pairs):",
        f"  - `max_abs_prob_diff = {rl['max_abs_prob_diff']:.2e}`, `p99_abs_prob_diff = {rl['p99_abs_prob_diff']:.2e}`",
        f"  - `max_tv_diff = {rl['max_tv_diff']:.2e}`, `p99_tv_diff = {rl['p99_tv_diff']:.2e}`, `mean_tv_diff = {rl['mean_tv_diff']:.2e}`",
        f"  - `nucleus_support_mismatch = {rl['support_mismatch_count']}/{rl['n_logit_vectors']}` (`{rl['support_mismatch_rate_pct']:.4f}%`), `max_boundary_mass = {rl['max_support_boundary_mass']:.2e}`",
        "",
        "### 3b. Synthetic / Adversarial Boundary Stress Suite (200 Cases)",
        "",
        "| Stress Category | N Cases | Max TV Diff | Support Mismatches | Mismatch Rate % | Max Boundary Mass |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for name, s in w3["adversarial_stress_suite"].items():
        lines.append(
            f"| `{name}` | {s['n_cases']} | `{s['max_tv_diff']:.2e}` | {s['support_mismatch_count']} | {s['support_mismatch_rate_pct']:.2f}% | `{s['max_boundary_mass']:.2e}` |"
        )

    lines.extend([
        "",
        "### 3c. Paired Speculative Accept/Reject & Residual Decisions Audit",
        f"- Paired trials: `{pv['n_decision_trials']}` | Decision flips: **`{pv['decision_flips']}` (`{pv['decision_mismatch_rate_pct']:.5f}%`)**",
        f"- `max_abs_accept_prob_diff = {pv['max_abs_accept_prob_diff']:.2e}`, `p99_abs_accept_prob_diff = {pv['p99_abs_accept_prob_diff']:.2e}`",
        f"- Residual distribution checks (`{pv['n_residual_checks']}`): `max_tv = {pv['residual_max_tv']:.2e}`, `mean_tv = {pv['residual_mean_tv']:.2e}`",
        "",
        "### 3d. Live Multi-Claim B2 K=4 Rollout Comparison (`orig_fp64` vs `fast_fp32`, Same Seed=42)",
        f"- Identical full token trajectories: **`{lr['identical_token_trajectories']}`** (first divergence positions if any: `{lr['first_divergence_positions']}`)",
        "",
        "| Variant | Claims | Total Tok | Rounds | $\\alpha_{\\mathrm{eval}}$ % | $\\alpha_{\\mathrm{prop}}$ % | Comm/Rnd | Dist ms/Rnd | E2E ms/Claim | E2E tok/s |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for name in ("orig_fp64", "fast_fp32"):
        v = lr[name]
        lines.append(
            f"| `{name}` | {v['n_claims']} | {v['total_tokens']} | {v['total_rounds']} | "
            f"{v['alpha_eval_pct']:.3f}% | {v['alpha_prop_pct']:.3f}% | {v['committed_per_round']:.3f} | "
            f"**{v['dist_ms_per_round']:.3f}** | {v['e2e_ms_per_claim']:.2f} | **{v['e2e_tok_s']:.2f}** |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default=os.environ.get("TARGET_MODEL", "Qwen/Qwen3-8B"))
    p.add_argument("--draft", default=os.environ.get("DRAFT_MODEL", "Qwen/Qwen3-0.6B"))
    p.add_argument("--source-jsonl", type=Path, default=Path("outputs/benchmarks/train-retrieved-bounded.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/b1_1_v2_validation"))
    p.add_argument("--n-steady-steps", type=int, default=80)
    p.add_argument("--n-claims-rollout", type=int, default=15)
    args = p.parse_args()

    from transformers import AutoTokenizer
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from profile_b1_1_draft_runtime import _build_token_sequence

    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)

    print("=== [Workstream 1] FlexAttention + StaticCache + Exact BlockMask ===", flush=True)
    w1 = run_workstream_1_flex_validation(
        target_path=args.target,
        draft_path=args.draft,
        tokenizer=tokenizer,
        source_jsonl=args.source_jsonl,
        n_steady_steps=args.n_steady_steps,
    )
    target_model, m_sdpa, m_flex = w1.pop("models")
    del m_flex
    torch.cuda.empty_cache()

    print("=== [Workstream 2] Exact CPU/GPU Timeline Decomposition ===", flush=True)
    w2: dict[str, Any] = {}
    for lbl, p_len in [("short", 320), ("selective_top2", 2560), ("full_long", 6150)]:
        pids = _build_token_sequence(tokenizer, p_len, args.source_jsonl)
        w2[lbl] = analyze_eager_timeline(m_sdpa, pids, n_steps=10)
        print(f"  timeline {lbl}: {w2[lbl]}", flush=True)

    print("=== [Workstream 3] Comprehensive FP32 Distribution Processing Audit ===", flush=True)
    w3 = run_workstream_3_fp32_audit(
        target_model=target_model,
        draft_model=m_sdpa,
        tokenizer=tokenizer,
        source_jsonl=args.source_jsonl,
        n_claims_logits=args.n_claims_rollout,
        n_claims_rollout=args.n_claims_rollout,
    )
    print(f"  w3 real_logits: {w3['real_raw_logits_audit']}", flush=True)
    print(f"  w3 paired_decisions: {w3['paired_verification_decisions']}", flush=True)
    print(f"  w3 live_rollout: {w3['live_rollout_comparison']}", flush=True)

    report = {
        "schema_version": "b1_1_v2_validation_v1",
        "workstream_1_flex": w1,
        "workstream_2_timeline": w2,
        "workstream_3_fp32_audit": w3,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "b1_1_v2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "b1_1_v2_report.md").write_text(render_v2_markdown(report), encoding="utf-8")
    print(f"Saved B1.1-v2 validation report to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
