"""Step B1.1: Fine-Grained Profiling & Draft Runtime Optimization Benchmark.

Strictly implements Step B1.1 of Direction B1:
  1. Micro-profiles EVERY component of a K=4 speculative round on real Qwen3-8B + Qwen3-0.6B
     (draft forwards, draft update, fp64 vs fp32 top-p sampling, target verify, target dist
     processing, rollback/verification) AND runs `torch.profiler` to measure GPU kernel time
     vs CPU wall time directly (zero guesswork).
  2. Benchmarks draft decode runtimes across 4 context lengths (320, 2560, 6150, 12288):
       - eager_dynamic_cache (M4.2 baseline)
       - eager_static_cache
       - compile_default_static (torch.compile mode="default")
       - compile_reduce_overhead_static (torch.compile mode="reduce-overhead")
       - cudagraph_static (explicit torch.cuda.CUDAGraph over StaticCache)
     strictly separating cache_init / compile / warmup / graph_capture from steady-state latency.
  3. Verifies all 5 B1.1 criteria:
       (a) substantial draft decode latency reduction vs eager,
       (b) improvement on both Full Context (6150) and Selective Context (2560),
       (c) peak VRAM safely within RTX 5090 (32 GB) with Qwen3-8B resident,
       (d) distribution & KV-cache rollback exactness vs eager_dynamic_cache,
       (e) compile/warmup/capture times reported separately.
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

from spade.decoding.autoregressive import load_causal_lm
from spade.decoding.cache_utils import CacheManager
from spade.decoding.runtime_opt import (
    StaticDraftRunner,
    _supports_logits_to_keep,
    next_power_of_two_bucket,
    processed_distribution_fast,
)
from spade.decoding.sampling import processed_distribution, sample_distribution, total_variation
from spade.decoding.speculative import exact_speculative_round


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_device(model: Any) -> torch.device:
    if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
        return model.get_input_embeddings().weight.device
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def _build_token_sequence(tokenizer: Any, length: int, source_jsonl: Path | None = None) -> list[int]:
    """Build a realistic prompt token sequence of exact `length` from benchmark text."""
    text_parts: list[str] = []
    if source_jsonl is not None and source_jsonl.exists():
        for line in source_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text_parts.append(str(row.get("prompt", "")))
            if sum(len(p) for p in text_parts) > length * 6:
                break
    if not text_parts:
        text_parts = [
            "Claim: Global temperatures rose by 1.1 degrees Celsius since pre-industrial times. "
            "Evidence [E1]: According to the IPCC Sixth Assessment Report, global surface temperature "
            "was 1.09 C higher in 2011-2020 than 1850-1900. "
        ] * 500
    ids: list[int] = []
    for part in text_parts:
        ids.extend(tokenizer(part, add_special_tokens=False)["input_ids"])
        if len(ids) >= length + 64:
            break
    while len(ids) < length + 64:
        ids.extend(ids[: max(64, length + 64 - len(ids))])
    return ids[:length]


def _run_eager_dynamic_sequence(
    model: Any,
    prompt_ids: list[int],
    step_tokens: list[int],
    rollback_plan: dict[int, int] | None = None,
) -> list[torch.Tensor]:
    """Reference eager DynamicCache execution returning float32 logits after each step.

    `rollback_plan`: optional mapping `{step_idx: rollback_delta}` applied BEFORE feeding
    `step_tokens[step_idx]` (crops cache back by `rollback_delta` tokens).
    """
    device = _model_device(model)
    l2k = _supports_logits_to_keep(model)
    mgr = CacheManager()
    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    kw: dict[str, Any] = {"input_ids": inp, "use_cache": True}
    if l2k:
        kw["logits_to_keep"] = 1
    with torch.inference_mode():
        out = model(**kw)
    mgr.update(out.past_key_values)
    logits_list = [out.logits[0, -1].float().clone()]
    del out, inp

    for idx, tok in enumerate(step_tokens):
        if rollback_plan and idx in rollback_plan:
            delta = rollback_plan[idx]
            mgr.rollback(mgr.seq_length - delta)
        sinp = torch.tensor([[int(tok)]], dtype=torch.long, device=device)
        skw: dict[str, Any] = {"input_ids": sinp, "past_key_values": mgr.past_key_values, "use_cache": True}
        if l2k:
            skw["logits_to_keep"] = 1
        with torch.inference_mode():
            sout = model(**skw)
        mgr.update(sout.past_key_values)
        logits_list.append(sout.logits[0, -1].float().clone())
        del sout, sinp

    mgr.reset()
    return logits_list


def benchmark_sampling_primitives(
    draft_logits_batch: list[torch.Tensor],
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict[str, Any]:
    """Compare original fp64 `processed_distribution` vs fast fp32 no-sync version."""
    device = draft_logits_batch[0].device
    # Exactness check across all real logit vectors
    max_abs_diff = 0.0
    max_tv_diff = 0.0
    tv_diffs: list[float] = []
    for lg in draft_logits_batch:
        p_orig = processed_distribution(lg, temperature=temperature, top_k=0, top_p=top_p)
        p_fast = processed_distribution_fast(lg, temperature=temperature, top_k=0, top_p=top_p, dtype=torch.float32)
        diff = float(torch.max(torch.abs(p_orig - p_fast.to(torch.float64))).item())
        tv = total_variation(p_orig, p_fast)
        max_abs_diff = max(max_abs_diff, diff)
        max_tv_diff = max(max_tv_diff, tv)
        tv_diffs.append(tv)

    # Also verify batched [5, V] matches sequential 1D
    stacked5 = torch.stack(draft_logits_batch[:5], dim=0)
    batched_fast = processed_distribution_fast(stacked5, temperature=temperature, top_k=0, top_p=top_p, dtype=torch.float32)
    batched_max_tv = max(
        total_variation(
            processed_distribution(draft_logits_batch[i], temperature=temperature, top_k=0, top_p=top_p),
            batched_fast[i],
        )
        for i in range(5)
    )

    # Warmup
    for _ in range(5):
        _ = processed_distribution(draft_logits_batch[0], temperature=temperature, top_k=0, top_p=top_p)
        _ = processed_distribution_fast(draft_logits_batch[0], temperature=temperature, top_k=0, top_p=top_p)
        _ = processed_distribution_fast(stacked5, temperature=temperature, top_k=0, top_p=top_p)
    _sync(device)

    n_iters = 40
    # 1) Original fp64 single call
    _sync(device)
    t0 = time.perf_counter()
    for i in range(n_iters):
        _ = processed_distribution(draft_logits_batch[i % len(draft_logits_batch)], temperature=temperature, top_k=0, top_p=top_p)
    _sync(device)
    orig_1_ms = (time.perf_counter() - t0) * 1000.0 / n_iters

    # 2) Fast fp32 single call
    _sync(device)
    t0 = time.perf_counter()
    for i in range(n_iters):
        _ = processed_distribution_fast(draft_logits_batch[i % len(draft_logits_batch)], temperature=temperature, top_k=0, top_p=top_p)
    _sync(device)
    fast_1_ms = (time.perf_counter() - t0) * 1000.0 / n_iters

    # 3) Fast fp32 batched [5, V] (replaces 5 target processed_distribution calls per round)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = processed_distribution_fast(stacked5, temperature=temperature, top_k=0, top_p=top_p)
    _sync(device)
    fast_5_batched_ms = (time.perf_counter() - t0) * 1000.0 / n_iters

    # 4) Original fp64 + multinomial sample
    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    _sync(device)
    t0 = time.perf_counter()
    for i in range(n_iters):
        p = processed_distribution(draft_logits_batch[i % len(draft_logits_batch)], temperature=temperature, top_k=0, top_p=top_p)
        _ = sample_distribution(p, gen)
    _sync(device)
    orig_sample_ms = (time.perf_counter() - t0) * 1000.0 / n_iters

    _sync(device)
    t0 = time.perf_counter()
    for i in range(n_iters):
        p = processed_distribution_fast(draft_logits_batch[i % len(draft_logits_batch)], temperature=temperature, top_k=0, top_p=top_p)
        _ = sample_distribution(p, gen)
    _sync(device)
    fast_sample_ms = (time.perf_counter() - t0) * 1000.0 / n_iters

    return {
        "n_vectors_checked": len(draft_logits_batch),
        "max_abs_prob_diff": max_abs_diff,
        "max_tv_diff_1d": max_tv_diff,
        "mean_tv_diff_1d": sum(tv_diffs) / len(tv_diffs),
        "max_tv_diff_batched5": batched_max_tv,
        "orig_fp64_1call_ms": round(orig_1_ms, 4),
        "fast_fp32_1call_ms": round(fast_1_ms, 4),
        "orig_fp64_5calls_ms": round(orig_1_ms * 5.0, 4),
        "fast_fp32_batched5_ms": round(fast_5_batched_ms, 4),
        "orig_fp64_dist_plus_sample_ms": round(orig_sample_ms, 4),
        "fast_fp32_dist_plus_sample_ms": round(fast_sample_ms, 4),
        "speedup_1call": round(orig_1_ms / fast_1_ms, 2) if fast_1_ms > 0 else 0.0,
        "speedup_5target_batched": round((orig_1_ms * 5.0) / fast_5_batched_ms, 2) if fast_5_batched_ms > 0 else 0.0,
    }


def profile_eager_round_breakdown(
    *,
    target_model: Any,
    draft_model: Any,
    prompt_ids: list[int],
    n_rounds: int = 15,
    draft_length: int = 4,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict[str, Any]:
    """Micro-profile every sub-component of a K=4 eager speculative round."""
    t_dev = _model_device(target_model)
    d_dev = _model_device(draft_model)
    t_l2k = _supports_logits_to_keep(target_model)
    d_l2k = _supports_logits_to_keep(draft_model)

    t_mgr = CacheManager()
    d_mgr = CacheManager()
    gen = torch.Generator(device=t_dev)
    gen.manual_seed(42)

    # Prefill target & draft
    tinp = torch.tensor([prompt_ids], dtype=torch.long, device=t_dev)
    tkw: dict[str, Any] = {"input_ids": tinp, "use_cache": True}
    if t_l2k:
        tkw["logits_to_keep"] = 1
    _sync(t_dev)
    t0 = time.perf_counter()
    with torch.inference_mode():
        tout = target_model(**tkw)
    _sync(t_dev)
    t_prefill_ms = (time.perf_counter() - t0) * 1000.0
    t_mgr.update(tout.past_key_values)
    next_t_logits = tout.logits[0, -1].float()
    del tout, tinp

    dinp = torch.tensor([prompt_ids], dtype=torch.long, device=d_dev)
    dkw: dict[str, Any] = {"input_ids": dinp, "use_cache": True}
    if d_l2k:
        dkw["logits_to_keep"] = 1
    _sync(d_dev)
    d0 = time.perf_counter()
    with torch.inference_mode():
        dout = draft_model(**dkw)
    _sync(d_dev)
    d_prefill_ms = (time.perf_counter() - d0) * 1000.0
    d_mgr.update(dout.past_key_values)
    next_d_logits = dout.logits[0, -1].float()
    del dout, dinp

    # Accumulators (skip first 2 warmup rounds)
    comp = {
        "draft_3_step_fwds_ms": [],
        "draft_4_sampling_orig_ms": [],
        "draft_update_fwd_ms": [],
        "target_verify_fwd_ms": [],
        "target_5_dist_orig_ms": [],
        "exact_spec_and_rollback_ms": [],
        "total_round_ms": [],
    }
    captured_logits: list[torch.Tensor] = []
    pending_tok: int | None = None

    for r in range(n_rounds + 2):
        _sync(t_dev)
        r_start = time.perf_counter()
        t_len_before = t_mgr.seq_length
        d_len_before = d_mgr.seq_length

        proposed: list[int] = []
        q_dists: list[torch.Tensor] = []
        cur_d_logits = next_d_logits
        d_fwd_s = 0.0
        d_samp_s = 0.0

        for idx in range(draft_length):
            if len(captured_logits) < 30:
                captured_logits.append(cur_d_logits.clone())
            _sync(d_dev)
            s0 = time.perf_counter()
            q = processed_distribution(cur_d_logits, temperature=temperature, top_k=0, top_p=top_p)
            tok = sample_distribution(q, gen)
            _sync(d_dev)
            d_samp_s += time.perf_counter() - s0
            proposed.append(tok)
            q_dists.append(q)
            if idx < draft_length - 1:
                _sync(d_dev)
                f0 = time.perf_counter()
                sinp = torch.tensor([[tok]], dtype=torch.long, device=d_dev)
                with torch.inference_mode():
                    sdout = draft_model(input_ids=sinp, past_key_values=d_mgr.past_key_values, use_cache=True)
                _sync(d_dev)
                d_fwd_s += time.perf_counter() - f0
                d_mgr.update(sdout.past_key_values)
                cur_d_logits = sdout.logits[0, -1].float()

        verify_tokens = ([pending_tok] + proposed) if pending_tok is not None else proposed
        has_pending = pending_tok is not None
        _sync(t_dev)
        v0 = time.perf_counter()
        vinp = torch.tensor([verify_tokens], dtype=torch.long, device=t_dev)
        with torch.inference_mode():
            vtout = target_model(input_ids=vinp, past_key_values=t_mgr.past_key_values, use_cache=True)
        _sync(t_dev)
        t_ver_s = time.perf_counter() - v0
        t_mgr.update(vtout.past_key_values)

        _sync(t_dev)
        td0 = time.perf_counter()
        if has_pending:
            p_dists = [processed_distribution(vtout.logits[0, i].float(), temperature=temperature, top_k=0, top_p=top_p) for i in range(len(proposed))]
            t_extra = processed_distribution(vtout.logits[0, -1].float(), temperature=temperature, top_k=0, top_p=top_p)
        else:
            p_dists = [processed_distribution(next_t_logits, temperature=temperature, top_k=0, top_p=top_p)]
            for i in range(len(proposed) - 1):
                p_dists.append(processed_distribution(vtout.logits[0, i].float(), temperature=temperature, top_k=0, top_p=top_p))
            t_extra = processed_distribution(vtout.logits[0, -1].float(), temperature=temperature, top_k=0, top_p=top_p)
        _sync(t_dev)
        t_dist_s = time.perf_counter() - td0

        _sync(t_dev)
        c0 = time.perf_counter()
        res = exact_speculative_round(proposed, q_dists, p_dists, generator=gen, eos_token_id=None, remaining=64, target_extra=t_extra)
        if res.rejected_position is not None:
            t_mgr.rollback(t_len_before + (1 if has_pending else 0) + res.rejected_position)
            d_mgr.rollback(d_len_before + res.rejected_position)
        _sync(t_dev)
        corr_s = time.perf_counter() - c0

        last_tok = res.committed[-1]
        if res.rejected_position is not None:
            pending_tok = last_tok
            next_t_logits = None
            upd_tokens = [last_tok]
        elif res.extra_target is not None:
            pending_tok = last_tok
            next_t_logits = None
            upd_tokens = [proposed[-1], last_tok]
        else:
            pending_tok = None
            next_t_logits = vtout.logits[0, -1].float()
            upd_tokens = [proposed[-1]]

        _sync(d_dev)
        u0 = time.perf_counter()
        uinp = torch.tensor([upd_tokens], dtype=torch.long, device=d_dev)
        with torch.inference_mode():
            udout = draft_model(input_ids=uinp, past_key_values=d_mgr.past_key_values, use_cache=True)
        _sync(d_dev)
        d_upd_s = time.perf_counter() - u0
        d_mgr.update(udout.past_key_values)
        next_d_logits = udout.logits[0, -1].float()

        r_total_s = time.perf_counter() - r_start
        if r >= 2:
            comp["draft_3_step_fwds_ms"].append(d_fwd_s * 1000.0)
            comp["draft_4_sampling_orig_ms"].append(d_samp_s * 1000.0)
            comp["draft_update_fwd_ms"].append(d_upd_s * 1000.0)
            comp["target_verify_fwd_ms"].append(t_ver_s * 1000.0)
            comp["target_5_dist_orig_ms"].append(t_dist_s * 1000.0)
            comp["exact_spec_and_rollback_ms"].append(corr_s * 1000.0)
            comp["total_round_ms"].append(r_total_s * 1000.0)

    t_mgr.reset()
    d_mgr.reset()
    if t_dev.type == "cuda":
        torch.cuda.empty_cache()

    means = {k: round(sum(v) / len(v), 3) for k, v in comp.items()}
    accounted = (
        means["draft_3_step_fwds_ms"]
        + means["draft_4_sampling_orig_ms"]
        + means["draft_update_fwd_ms"]
        + means["target_verify_fwd_ms"]
        + means["target_5_dist_orig_ms"]
        + means["exact_spec_and_rollback_ms"]
    )
    means["other_python_ms"] = round(means["total_round_ms"] - accounted, 3)
    means["draft_prefill_ms"] = round(d_prefill_ms, 2)
    means["target_prefill_ms"] = round(t_prefill_ms, 2)
    means["prompt_tokens"] = len(prompt_ids)
    return {"breakdown": means, "captured_logits": captured_logits}


def profile_draft_torch_profiler(
    draft_model: Any,
    prompt_ids: list[int],
    *,
    n_steps: int = 10,
) -> dict[str, Any]:
    """Use `torch.profiler` to measure exact GPU kernel time vs CPU wall time per draft step."""
    device = _model_device(draft_model)
    l2k = _supports_logits_to_keep(draft_model)
    mgr = CacheManager()
    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    kw: dict[str, Any] = {"input_ids": inp, "use_cache": True}
    if l2k:
        kw["logits_to_keep"] = 1
    with torch.inference_mode():
        out = draft_model(**kw)
    mgr.update(out.past_key_values)
    tok = int(torch.argmax(out.logits[0, -1]).item())
    del out, inp

    # Warmup 3 steps
    for _ in range(3):
        sinp = torch.tensor([[tok]], dtype=torch.long, device=device)
        with torch.inference_mode():
            sout = draft_model(input_ids=sinp, past_key_values=mgr.past_key_values, use_cache=True)
        mgr.update(sout.past_key_values)
        tok = int(torch.argmax(sout.logits[0, -1]).item())
    _sync(device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    t0 = time.perf_counter()
    with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
        for _ in range(n_steps):
            sinp = torch.tensor([[tok]], dtype=torch.long, device=device)
            with torch.inference_mode():
                sout = draft_model(input_ids=sinp, past_key_values=mgr.past_key_values, use_cache=True)
            mgr.update(sout.past_key_values)
            tok = int(torch.argmax(sout.logits[0, -1]).item())
        _sync(device)
    wall_ms_per_step = (time.perf_counter() - t0) * 1000.0 / n_steps

    # Extract CUDA kernel totals & top operators
    key_avgs = prof.key_averages()
    total_cuda_us = 0.0
    total_cpu_us = 0.0
    top_ops: list[dict[str, Any]] = []
    for evt in key_avgs:
        cuda_us = float(getattr(evt, "self_device_time_total", getattr(evt, "self_cuda_time_total", 0.0)))
        cpu_us = float(getattr(evt, "self_cpu_time_total", 0.0))
        total_cuda_us += cuda_us
        total_cpu_us += cpu_us
        if cuda_us > 0:
            top_ops.append({
                "name": evt.key[:60],
                "calls_per_step": round(evt.count / n_steps, 1),
                "cuda_ms_per_step": round((cuda_us / 1000.0) / n_steps, 4),
                "cpu_ms_per_step": round((cpu_us / 1000.0) / n_steps, 4),
            })
    top_ops.sort(key=lambda x: x["cuda_ms_per_step"], reverse=True)
    cuda_kernel_ms = (total_cuda_us / 1000.0) / n_steps

    mgr.reset()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "prompt_tokens": len(prompt_ids),
        "n_steps": n_steps,
        "wall_ms_per_step": round(wall_ms_per_step, 3),
        "sum_cuda_kernel_ms_per_step": round(cuda_kernel_ms, 3),
        "non_kernel_gap_ms_per_step": round(max(0.0, wall_ms_per_step - cuda_kernel_ms), 3),
        "cuda_kernel_share_pct": round((cuda_kernel_ms / wall_ms_per_step) * 100.0, 2) if wall_ms_per_step > 0 else 0.0,
        "top_cuda_ops": top_ops[:10],
    }


def benchmark_draft_runtime_modes(
    *,
    draft_model: Any,
    prompt_ids_by_label: dict[str, list[int]],
    modes: list[str],
    n_steady_steps: int = 80,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[dict[str, Any]]:
    """Benchmark each draft runtime mode across context lengths with exactness & VRAM checks."""
    device = _model_device(draft_model)
    l2k = _supports_logits_to_keep(draft_model)
    results: list[dict[str, Any]] = []

    # Deterministic 12-token exactness sequence with 2 rollbacks:
    # draft 4 -> rollback 2 -> draft 4 -> rollback 3 -> draft 4
    rollback_plan = {4: 2, 8: 3}

    # Pre-compute eager_dynamic_cache reference logits & baseline step timings per context
    ref_logits_by_label: dict[str, list[torch.Tensor]] = {}
    ref_step_tokens_by_label: dict[str, list[int]] = {}
    eager_baseline_ms: dict[str, float] = {}

    for label, prompt_ids in prompt_ids_by_label.items():
        # Pick 12 valid token ids from prompt tail for deterministic exactness testing
        step_toks = [prompt_ids[-(i % 16) - 1] for i in range(12)]
        ref_step_tokens_by_label[label] = step_toks
        ref_logits_by_label[label] = _run_eager_dynamic_sequence(
            draft_model, prompt_ids, step_toks, rollback_plan=rollback_plan
        )

    for label, prompt_ids in prompt_ids_by_label.items():
        p_len = len(prompt_ids)
        step_toks = ref_step_tokens_by_label[label]
        ref_logits = ref_logits_by_label[label]
        max_cache_len = next_power_of_two_bucket(p_len, min_bucket=512, headroom=256)

        for mode in modes:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            entry: dict[str, Any] = {
                "context_label": label,
                "prompt_tokens": p_len,
                "max_cache_len": max_cache_len if mode != "eager_dynamic_cache" else None,
                "mode": mode,
                "status": "OK",
                "error": None,
            }
            try:
                if mode == "eager_dynamic_cache":
                    # Warmup
                    mgr = CacheManager()
                    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                    kw: dict[str, Any] = {"input_ids": inp, "use_cache": True}
                    if l2k:
                        kw["logits_to_keep"] = 1
                    _sync(device)
                    t_pref0 = time.perf_counter()
                    with torch.inference_mode():
                        out = draft_model(**kw)
                    _sync(device)
                    prefill_ms = (time.perf_counter() - t_pref0) * 1000.0
                    mgr.update(out.past_key_values)
                    tok = int(torch.argmax(out.logits[0, -1]).item())
                    del out, inp

                    t_w0 = time.perf_counter()
                    for _ in range(5):
                        sinp = torch.tensor([[tok]], dtype=torch.long, device=device)
                        with torch.inference_mode():
                            sout = draft_model(input_ids=sinp, past_key_values=mgr.past_key_values, use_cache=True)
                        mgr.update(sout.past_key_values)
                        tok = int(torch.argmax(sout.logits[0, -1]).item())
                    _sync(device)
                    warmup_s = time.perf_counter() - t_w0

                    step_ms_list: list[float] = []
                    for _ in range(n_steady_steps):
                        _sync(device)
                        t0 = time.perf_counter()
                        sinp = torch.tensor([[tok]], dtype=torch.long, device=device)
                        with torch.inference_mode():
                            sout = draft_model(input_ids=sinp, past_key_values=mgr.past_key_values, use_cache=True)
                        _sync(device)
                        step_ms_list.append((time.perf_counter() - t0) * 1000.0)
                        mgr.update(sout.past_key_values)
                        tok = int(torch.argmax(sout.logits[0, -1]).item())
                    mgr.reset()

                    mean_ms = statistics.mean(step_ms_list)
                    eager_baseline_ms[label] = mean_ms
                    peak_vram_mb = (
                        torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                        if device.type == "cuda"
                        else 0.0
                    )
                    entry.update({
                        "cache_init_s": 0.0,
                        "compile_s": 0.0,
                        "warmup_s": round(warmup_s, 4),
                        "graph_capture_s": 0.0,
                        "prefill_ms": round(prefill_ms, 2),
                        "steady_step_mean_ms": round(mean_ms, 3),
                        "steady_step_median_ms": round(statistics.median(step_ms_list), 3),
                        "steady_step_p95_ms": round(sorted(step_ms_list)[int(0.95 * len(step_ms_list))], 3),
                        "est_4step_draft_ms": round(4.0 * mean_ms, 2),
                        "speedup_vs_eager_dynamic": 1.0,
                        "peak_vram_mb": round(peak_vram_mb, 1),
                        "vram_within_32gb": peak_vram_mb < 32000.0,
                        "max_abs_logit_diff": 0.0,
                        "mean_abs_logit_diff": 0.0,
                        "max_tv_diff": 0.0,
                        "post_rollback_max_tv_diff": 0.0,
                        "exactness_pass": True,
                    })
                else:
                    runner = StaticDraftRunner(
                        draft_model,
                        max_cache_len=max_cache_len,
                        mode=mode,
                        device=device,
                    )
                    setup = runner.prepare(warmup_steps=3)

                    # 1) Exactness + rollback check against eager_dynamic_cache
                    got_logits = [runner.prefill(prompt_ids).clone()]
                    for idx, st_tok in enumerate(step_toks):
                        if idx in rollback_plan:
                            runner.rollback(runner.cur_len - rollback_plan[idx])
                        got_logits.append(runner.step(st_tok).clone())

                    abs_diffs = [
                        float(torch.max(torch.abs(a - b)).item())
                        for a, b in zip(ref_logits, got_logits)
                    ]
                    mean_diffs = [
                        float(torch.mean(torch.abs(a - b)).item())
                        for a, b in zip(ref_logits, got_logits)
                    ]
                    tv_diffs = [
                        total_variation(
                            processed_distribution(a, temperature=temperature, top_k=0, top_p=top_p),
                            processed_distribution(b, temperature=temperature, top_k=0, top_p=top_p),
                        )
                        for a, b in zip(ref_logits, got_logits)
                    ]
                    post_rb_tv = max(tv_diffs[5:]) if len(tv_diffs) > 5 else max(tv_diffs)

                    # 2) Steady-state prefill & step latency benchmark (80 steps, strictly post-warmup)
                    _sync(device)
                    t_pref0 = time.perf_counter()
                    lg = runner.prefill(prompt_ids)
                    _sync(device)
                    prefill_ms = (time.perf_counter() - t_pref0) * 1000.0
                    tok = int(torch.argmax(lg).item())

                    # Short post-prefill steady warmup (3 steps)
                    for _ in range(3):
                        lg = runner.step(tok)
                        tok = int(torch.argmax(lg).item())
                    _sync(device)

                    step_ms_list = []
                    for _ in range(n_steady_steps):
                        _sync(device)
                        t0 = time.perf_counter()
                        lg = runner.step(tok)
                        _sync(device)
                        step_ms_list.append((time.perf_counter() - t0) * 1000.0)
                        tok = int(torch.argmax(lg).item())

                    mean_ms = statistics.mean(step_ms_list)
                    base_ms = eager_baseline_ms.get(label, mean_ms)
                    peak_vram_mb = (
                        torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                        if device.type == "cuda"
                        else 0.0
                    )
                    max_tv = max(tv_diffs)
                    entry.update({
                        "cache_init_s": round(setup.cache_init_s, 4),
                        "compile_s": round(setup.compile_s, 3),
                        "warmup_s": round(setup.warmup_s, 4),
                        "graph_capture_s": round(setup.graph_capture_s, 4),
                        "prefill_ms": round(prefill_ms, 2),
                        "steady_step_mean_ms": round(mean_ms, 3),
                        "steady_step_median_ms": round(statistics.median(step_ms_list), 3),
                        "steady_step_p95_ms": round(sorted(step_ms_list)[int(0.95 * len(step_ms_list))], 3),
                        "est_4step_draft_ms": round(4.0 * mean_ms, 2),
                        "speedup_vs_eager_dynamic": round(base_ms / mean_ms, 2) if mean_ms > 0 else 0.0,
                        "peak_vram_mb": round(peak_vram_mb, 1),
                        "vram_within_32gb": peak_vram_mb < 32000.0,
                        "max_abs_logit_diff": round(max(abs_diffs), 6),
                        "mean_abs_logit_diff": round(statistics.mean(mean_diffs), 6),
                        "max_tv_diff": round(max_tv, 6),
                        "post_rollback_max_tv_diff": round(post_rb_tv, 6),
                        "exactness_pass": bool(max_tv < 0.01 and post_rb_tv < 0.01),
                    })
                    del runner
            except Exception as exc:
                entry["status"] = "ERROR"
                entry["error"] = f"{type(exc).__name__}: {exc}"

            results.append(entry)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return results


def render_b1_1_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Step B1.1 — Draft Runtime Profiling & Optimization Report",
        "",
        f"- Hardware: `{report['hardware'].get('gpu_name', 'CPU')}` | PyTorch `{report['hardware']['pytorch']}` | Transformers `{report['hardware']['transformers']}`",
        "- All compilation, warmup, and CUDA graph capture times are reported strictly separate from steady-state step latencies.",
        "",
        "## 1. Eager Speculative Round Micro-Profiling (K=4, Qwen3-8B + Qwen3-0.6B)",
        "",
        "| Context | Prompt Tok | Draft Prefill | Draft 3-Step Fwd | Draft 4x Samp (fp64) | Draft Update Fwd | Target Verify Fwd | Target 5x Dist (fp64) | Spec+Rollback | Other Py | Total Round (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for label, bd in report["eager_round_breakdowns"].items():
        lines.append(
            f"| {label} | {bd['prompt_tokens']} | {bd['draft_prefill_ms']:.2f} | "
            f"{bd['draft_3_step_fwds_ms']:.2f} | {bd['draft_4_sampling_orig_ms']:.2f} | "
            f"{bd['draft_update_fwd_ms']:.2f} | {bd['target_verify_fwd_ms']:.2f} | "
            f"{bd['target_5_dist_orig_ms']:.2f} | {bd['exact_spec_and_rollback_ms']:.2f} | "
            f"{bd['other_python_ms']:.2f} | **{bd['total_round_ms']:.2f}** |"
        )

    sp = report["sampling_benchmark"]
    lines.extend([
        "",
        "## 2. Sampling / Top-p Distribution Micro-Benchmark (Exactness + Latency)",
        "",
        f"- Original fp64 `processed_distribution` (1 call): `{sp['orig_fp64_1call_ms']:.3f} ms` | 5 calls (target/rnd): `{sp['orig_fp64_5calls_ms']:.3f} ms`",
        f"- Fast fp32 no-sync `processed_distribution_fast` (1 call): `{sp['fast_fp32_1call_ms']:.3f} ms` (**{sp['speedup_1call']:.2f}x**) | Batched `[5, V]`: `{sp['fast_fp32_batched5_ms']:.3f} ms` (**{sp['speedup_5target_batched']:.2f}x**)",
        f"- Exactness over {sp['n_vectors_checked']} real logits: `max_abs_prob_diff={sp['max_abs_prob_diff']:.2e}`, `max_tv_diff_1d={sp['max_tv_diff_1d']:.2e}`, `max_tv_diff_batched5={sp['max_tv_diff_batched5']:.2e}`",
        "",
        "## 3. `torch.profiler` GPU Kernel Time vs Wall Time (Single Draft Decode Step, Eager)",
        "",
        "| Context | Prompt Tok | Wall ms/step | Sum CUDA Kernel ms/step | Non-Kernel Gap ms/step | Kernel Share % |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for label, tp in report["torch_profiler_draft_step"].items():
        lines.append(
            f"| {label} | {tp['prompt_tokens']} | {tp['wall_ms_per_step']:.3f} | "
            f"{tp['sum_cuda_kernel_ms_per_step']:.3f} | {tp['non_kernel_gap_ms_per_step']:.3f} | "
            f"{tp['cuda_kernel_share_pct']:.1f}% |"
        )

    lines.extend([
        "",
        "## 4. Draft Runtime Modes Benchmark (Steady-State vs Setup, Exactness & Peak VRAM)",
        "",
        "| Context | Mode | MaxCache | Compile (s) | Warmup (s) | Capture (s) | Steady Mean (ms) | Median (ms) | p95 (ms) | 4-Step Draft (ms) | Speedup | Peak VRAM (MB) | Max TV Diff | Post-RB TV | Exact |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in report["runtime_modes"]:
        if r["status"] != "OK":
            lines.append(f"| {r['context_label']} | `{r['mode']}` | {r['max_cache_len']} | ERROR: `{r['error']}` | - | - | - | - | - | - | - | - | - | - | FAIL |")
            continue
        lines.append(
            f"| {r['context_label']} ({r['prompt_tokens']}) | `{r['mode']}` | {r['max_cache_len']} | "
            f"{r['compile_s']:.2f} | {r['warmup_s']:.3f} | {r['graph_capture_s']:.3f} | "
            f"**{r['steady_step_mean_ms']:.3f}** | {r['steady_step_median_ms']:.3f} | {r['steady_step_p95_ms']:.3f} | "
            f"**{r['est_4step_draft_ms']:.2f}** | **{r['speedup_vs_eager_dynamic']:.2f}x** | "
            f"{r['peak_vram_mb']:.0f} | {r['max_tv_diff']:.6f} | {r['post_rollback_max_tv_diff']:.6f} | "
            f"{'PASS' if r['exactness_pass'] else 'FAIL'} |"
        )

    if report.get("context_sensitivity_summary"):
        lines.extend([
            "",
            "## 5. Context-Length Sensitivity Under Each Optimized Runtime (`full_long=6150` vs `selective_top2=2560` & `short=320`)",
            "",
            "| Mode | Short 320 (4-step ms) | Selective 2560 (4-step ms) | Full 6150 (4-step ms) | VeryLong 12288 (4-step ms) | Saved Full->Sel (ms/rnd) | Saved Full->Short (ms/rnd) | All 5 B1.1 Criteria |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for row in report["context_sensitivity_summary"]:
            lines.append(
                f"| `{row['mode']}` | {row['short_4step_ms']} | {row['selective_4step_ms']} | "
                f"{row['full_long_4step_ms']} | {row['very_long_4step_ms']} | "
                f"**{row['saved_full_to_sel_ms_per_rnd']}** | {row['saved_full_to_short_ms_per_rnd']} | "
                f"`{row['b1_1_verdict']}` |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default=os.environ.get("TARGET_MODEL", "Qwen/Qwen3-8B"))
    p.add_argument("--draft", default=os.environ.get("DRAFT_MODEL", "Qwen/Qwen3-0.6B"))
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--source-jsonl", type=Path, default=Path("outputs/benchmarks/train-retrieved-bounded.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/b1_1_runtime_profile"))
    p.add_argument("--n-steady-steps", type=int, default=80)
    p.add_argument("--n-profile-rounds", type=int, default=12)
    args = p.parse_args()

    from transformers import AutoTokenizer
    import transformers

    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    target_model = load_causal_lm(args.target, local_files_only=True, dtype=args.dtype)
    draft_model = load_causal_lm(args.draft, local_files_only=True, dtype=args.dtype)

    context_lengths = {
        "short": 320,
        "selective_top2": 2560,
        "full_long": 6150,
        "very_long": 12288,
    }
    prompt_ids_by_label = {
        lbl: _build_token_sequence(tokenizer, length, args.source_jsonl)
        for lbl, length in context_lengths.items()
    }

    print("=== [1/3] Micro-Profiling Eager K=4 Speculative Round ===", flush=True)
    eager_breakdowns: dict[str, Any] = {}
    captured_logits: list[torch.Tensor] = []
    for lbl in ("short", "selective_top2", "full_long"):
        res = profile_eager_round_breakdown(
            target_model=target_model,
            draft_model=draft_model,
            prompt_ids=prompt_ids_by_label[lbl],
            n_rounds=args.n_profile_rounds,
        )
        eager_breakdowns[lbl] = res["breakdown"]
        if not captured_logits:
            captured_logits = res["captured_logits"]
        print(f"  {lbl}: {res['breakdown']}", flush=True)

    print("=== [1b/3] Benchmarking Sampling / Top-p Primitives ===", flush=True)
    sampling_bench = benchmark_sampling_primitives(captured_logits)
    print(f"  sampling: {sampling_bench}", flush=True)

    print("=== [2/3] torch.profiler Single Draft Step (GPU Kernel vs Wall Time) ===", flush=True)
    profiler_res: dict[str, Any] = {}
    for lbl in ("short", "selective_top2", "full_long"):
        profiler_res[lbl] = profile_draft_torch_profiler(draft_model, prompt_ids_by_label[lbl], n_steps=10)
        print(
            f"  {lbl}: wall={profiler_res[lbl]['wall_ms_per_step']}ms, "
            f"cuda_kernels={profiler_res[lbl]['sum_cuda_kernel_ms_per_step']}ms "
            f"({profiler_res[lbl]['cuda_kernel_share_pct']}%)",
            flush=True,
        )

    print("=== [3/3] Benchmarking Draft Runtime Modes (with Qwen3-8B Resident on GPU) ===", flush=True)
    modes = [
        "eager_dynamic_cache",
        "eager_static_cache",
        "compile_default_static",
        "compile_reduce_overhead_static",
        "cudagraph_static",
    ]
    runtime_rows = benchmark_draft_runtime_modes(
        draft_model=draft_model,
        prompt_ids_by_label=prompt_ids_by_label,
        modes=modes,
        n_steady_steps=args.n_steady_steps,
    )
    for r in runtime_rows:
        print(f"  {r['context_label']:14s} | {r['mode']:30s} | {r['status']} | step={r.get('steady_step_mean_ms')}ms | speedup={r.get('speedup_vs_eager_dynamic')}x | max_tv={r.get('max_tv_diff')} | err={r.get('error')}", flush=True)

    # Summarize context-length sensitivity & B1.1 verdict per mode
    by_mode: dict[str, dict[str, dict[str, Any]]] = {}
    for r in runtime_rows:
        by_mode.setdefault(r["mode"], {})[r["context_label"]] = r

    sensitivity_rows: list[dict[str, Any]] = []
    for mode in modes:
        m_map = by_mode.get(mode, {})
        ok_all = all(m_map.get(c, {}).get("status") == "OK" for c in ("short", "selective_top2", "full_long"))
        s_ms = m_map.get("short", {}).get("est_4step_draft_ms")
        sel_ms = m_map.get("selective_top2", {}).get("est_4step_draft_ms")
        full_ms = m_map.get("full_long", {}).get("est_4step_draft_ms")
        vl_ms = m_map.get("very_long", {}).get("est_4step_draft_ms")
        saved_sel = round(full_ms - sel_ms, 2) if (full_ms is not None and sel_ms is not None) else None
        saved_short = round(full_ms - s_ms, 2) if (full_ms is not None and s_ms is not None) else None
        if mode == "eager_dynamic_cache":
            verdict = "BASELINE"
        elif not ok_all:
            verdict = "FAIL_RUNTIME_ERROR"
        else:
            sp_full = m_map["full_long"]["speedup_vs_eager_dynamic"]
            sp_sel = m_map["selective_top2"]["speedup_vs_eager_dynamic"]
            vram_ok = all(m_map[c]["vram_within_32gb"] for c in ("short", "selective_top2", "full_long"))
            exact_ok = all(m_map[c]["exactness_pass"] for c in ("short", "selective_top2", "full_long"))
            verdict = "PASS_B1_1" if (sp_full >= 1.5 and sp_sel >= 1.5 and vram_ok and exact_ok) else "FAIL_CRITERIA"
        sensitivity_rows.append({
            "mode": mode,
            "short_4step_ms": s_ms,
            "selective_4step_ms": sel_ms,
            "full_long_4step_ms": full_ms,
            "very_long_4step_ms": vl_ms,
            "saved_full_to_sel_ms_per_rnd": saved_sel,
            "saved_full_to_short_ms_per_rnd": saved_short,
            "b1_1_verdict": verdict,
        })

    hw = {
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    report = {
        "schema_version": "b1_1_runtime_profile_v1",
        "hardware": hw,
        "eager_round_breakdowns": eager_breakdowns,
        "sampling_benchmark": sampling_bench,
        "torch_profiler_draft_step": profiler_res,
        "runtime_modes": runtime_rows,
        "context_sensitivity_summary": sensitivity_rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "b1_1_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "b1_1_report.md").write_text(render_b1_1_markdown(report), encoding="utf-8")
    print(f"Saved B1.1 report to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
