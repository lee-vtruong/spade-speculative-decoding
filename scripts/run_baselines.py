"""Run B0/B0-S/B1/B2 diagnostics and preserve predictions plus traces.

Inference currently uses full-prefix recomputation (no KV cache). The outputs
are valid for correctness/rejection diagnostics, while latency is explicitly
non-authoritative for performance claims.
"""
from __future__ import annotations
import argparse, csv, json, os, platform, subprocess, sys, uuid, statistics
from datetime import datetime, timezone
from pathlib import Path

def read_jsonl(path: Path):
    # splitlines() treats Unicode separators such as U+2028 as record breaks,
    # even though they are valid characters inside JSON strings.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]

def write_json(path: Path, value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["B0","B0-S","B1","B2","M4","HF","HF-Assisted"],required=True)
    p.add_argument("--input",type=Path,required=True,help="Prepared benchmark JSONL")
    p.add_argument("--output-root",type=Path,default=Path(os.environ.get("OUTPUT_ROOT","outputs/runs")))
    p.add_argument("--run-id",default=None)
    p.add_argument("--target",default="Qwen/Qwen3-8B"); p.add_argument("--draft",default="Qwen/Qwen3-0.6B")
    p.add_argument("--dtype",choices=["bfloat16","float16","float32"],default="bfloat16")
    p.add_argument("--draft-length",type=int,choices=[2,4,6,8],default=4)
    p.add_argument("--max-new-tokens",type=int,default=256)
    p.add_argument("--temperature",type=float,default=1.0); p.add_argument("--top-k",type=int,default=0); p.add_argument("--top-p",type=float,default=1.0)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--limit",type=int,default=None)
    p.add_argument("--warmup",type=int,default=0); p.add_argument("--global-warmup",type=int,default=0); p.add_argument("--trials",type=int,default=1)
    p.add_argument("--cache-strategy",choices=["recompute","kv_cache"],default="recompute",help="Inference cache strategy: recompute (default) or kv_cache")
    p.add_argument("--dynamic-k",action="store_true",default=False,help="Enable dynamic draft length adaptation K(t) in {2, 4, 6} for M4")
    p.add_argument("--no-entropy-gating",action="store_true",default=False,help="Disable entropy gating in M4 draft biasing (ablation)")
    p.add_argument("--no-biasing",action="store_true",default=False,help="Disable evidence logit biasing in M4 (e.g. Dynamic-only ablation)")
    p.add_argument("--beta0",type=float,default=1.5,help="Base evidence logit biasing weight beta_0 (>= 0.0) for M4")
    p.add_argument("--order-decay",type=float,default=0.5,help="Higher-order n-gram scaling factor for M4 biasing")
    p.add_argument("--entropy-threshold",type=float,default=2.0,help="Entropy threshold (nats) for sigmoid gating in M4")
    p.add_argument("--max-boost-cap",type=float,default=3.0,help="Upper bound on applied logit boost in M4")
    p.add_argument("--draft-context-policy",choices=["full","selective_topk","compact","hybrid_topk_compact","sentence_budget","target_attn_sentence_budget"],default="full",help="Asymmetric draft context policy (target always uses full evidence)")
    p.add_argument("--draft-top-k",type=int,default=2,help="Evidence passages retained for draft under selective_topk / hybrid_topk_compact")
    p.add_argument("--draft-compact-sentences",type=int,default=1,help="Max claim-salient sentences per evidence passage under compact / hybrid_topk_compact")
    p.add_argument("--draft-compact-chars",type=int,default=160,help="Max characters per evidence passage under compact / hybrid_topk_compact")
    p.add_argument("--draft-budget-chars",type=int,default=8000,help="Total evidence character budget for sentence_budget / target_attn_sentence_budget")
    p.add_argument("--draft-attn-tail-window",type=int,default=16,help="Target prefill tail query window for target_attn_sentence_budget")
    p.add_argument(
        "--draft-runtime",
        choices=["eager_dynamic", "flex_compile_reduce_overhead", "flex_cudagraph_static", "flex_compile_default", "flex_eager_static"],
        default="eager_dynamic",
        help="Draft decode runtime backend (B1.1/B1.2). Defaults to eager_dynamic.",
    )
    p.add_argument(
        "--draft-max-cache-len",
        type=int,
        default=None,
        help="Optional StaticCache capacity for flex_* draft runtimes (auto-sized to multiple of 128 if omitted).",
    )
    p.add_argument(
        "--distribution-mode",
        choices=["fp64_orig", "fp32_fast"],
        default="fp64_orig",
        help="Sampling/top-p distribution precision mode (default fp64_orig; fp32_fast uses zero-sync fp32).",
    )
    a=p.parse_args()
    if a.max_new_tokens<1 or a.trials<1 or a.warmup<0 or a.global_warmup<0: p.error("Invalid token/warmup/trial count")
    if a.beta0 < 0.0: p.error("--beta0 must be non-negative")
    if a.max_boost_cap < 0.0: p.error("--max-boost-cap must be non-negative")
    if a.draft_top_k < 0 or a.draft_compact_sentences < 1 or a.draft_compact_chars < 16 or a.draft_budget_chars < 256 or a.draft_attn_tail_window < 1: p.error("Invalid draft context policy hyperparameters")
    rows=read_jsonl(a.input)
    if a.limit is not None: rows=rows[:a.limit]
    if not rows: p.error("Input JSONL has no examples")
    if any("prompt" not in row or "claim_id" not in row for row in rows): p.error("Each row needs prompt and claim_id")
    run_id=a.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8]
    out=a.output_root/run_id
    try: out.mkdir(parents=True,exist_ok=False)
    except FileExistsError: p.error(f"Run directory already exists; refusing to overwrite: {out}")
    try:
        import torch, transformers
        from spade.decoding.hf_assisted import QwenPair
        from spade.decoding.autoregressive import format_qwen_prompt
        from spade.adaptive.biasing import EvidenceBiasingConfig
        from spade.adaptive.context_selection import build_draft_context
        from spade.benchmark.evaluation import parse_generation
        from spade.paths import project_root
        from spade.instrumentation.trace_writer import TraceWriter
    except ImportError as exc: raise SystemExit(f"Required runtime dependency missing: {exc}") from exc
    use_flex_draft = (a.draft_runtime != "eager_dynamic") and (a.mode not in {"B0", "B0-S"})
    pair=QwenPair(
        a.target,
        None if a.mode in {"B0","B0-S"} else a.draft,
        local_files_only=True,
        dtype=a.dtype,
        draft_attn_implementation="flex_attention" if use_flex_draft else None,
    )
    draft_setup_dict = None
    draft_max_cache_len = a.draft_max_cache_len
    if use_flex_draft:
        if draft_max_cache_len is None:
            dtok = pair.draft_tokenizer or pair.target_tokenizer
            max_p_len = 0
            if a.draft_context_policy == "target_attn_sentence_budget":
                for row in rows:
                    ser_d = format_qwen_prompt(dtok, row["prompt"])
                    max_p_len = max(max_p_len, len(dtok(ser_d, add_special_tokens=False)["input_ids"]))
                est_budget_p_len = int(a.draft_budget_chars / 2.2) + 512
                max_p_len = min(max_p_len, est_budget_p_len)
            else:
                for row in rows:
                    dctx_tmp = build_draft_context(
                        row,
                        policy=a.draft_context_policy,
                        top_k=a.draft_top_k,
                        compact_sentences=a.draft_compact_sentences,
                        compact_chars=a.draft_compact_chars,
                        budget_chars=a.draft_budget_chars,
                    )
                    ser_d = format_qwen_prompt(dtok, dctx_tmp.draft_prompt)
                    max_p_len = max(max_p_len, len(dtok(ser_d, add_special_tokens=False)["input_ids"]))
            needed = max(512, max_p_len + a.max_new_tokens + 64)
            draft_max_cache_len = ((needed + 127) // 128) * 128
        runner_mode = a.draft_runtime.removeprefix("flex_")
        setup_timings = pair.prepare_draft_runner(mode=runner_mode, max_cache_len=draft_max_cache_len, warmup_steps=3)
        draft_setup_dict = {
            "mode": a.draft_runtime,
            "max_cache_len": setup_timings.max_cache_len,
            "cache_init_s": round(setup_timings.cache_init_s, 4),
            "compile_s": round(setup_timings.compile_s, 4),
            "warmup_s": round(setup_timings.warmup_s, 4),
            "graph_capture_s": round(setup_timings.graph_capture_s, 4),
        }
    is_kv = (a.cache_strategy == "kv_cache") or (a.mode in {"HF", "HF-Assisted"})
    enable_biasing = (a.mode == "M4") and (not a.no_biasing) and (a.beta0 > 0.0)
    biasing_cfg = EvidenceBiasingConfig(
        boost_weight=0.0 if a.no_biasing else a.beta0,
        order_decay=a.order_decay,
        adaptive_entropy=not a.no_entropy_gating,
        entropy_threshold=a.entropy_threshold,
        max_boost_cap=a.max_boost_cap,
    ) if a.mode == "M4" else None
    resolved={"run_id":run_id,"mode":a.mode,"input":str(a.input.resolve()),"target":a.target,"draft":None if a.mode in {"B0","B0-S"} else a.draft,"dtype":a.dtype,"draft_length":a.draft_length,"max_new_tokens":a.max_new_tokens,"temperature":a.temperature,"top_k":a.top_k,"top_p":a.top_p,"seed":a.seed,"limit":a.limit,"warmup":a.warmup,"global_warmup":a.global_warmup,"trials":a.trials,"cache_strategy":a.cache_strategy,"dynamic_k":a.dynamic_k,"enable_biasing":enable_biasing,"no_biasing":a.no_biasing,"no_entropy_gating":a.no_entropy_gating,"beta0":0.0 if a.no_biasing else a.beta0,"order_decay":a.order_decay,"entropy_threshold":a.entropy_threshold,"max_boost_cap":a.max_boost_cap,"draft_context_policy":a.draft_context_policy,"draft_top_k":a.draft_top_k,"draft_compact_sentences":a.draft_compact_sentences,"draft_compact_chars":a.draft_compact_chars,"draft_budget_chars":a.draft_budget_chars,"draft_attn_tail_window":a.draft_attn_tail_window,"draft_runtime":a.draft_runtime,"draft_max_cache_len":draft_max_cache_len,"distribution_mode":a.distribution_mode,"timings_authoritative":is_kv,"timing_note":"KV-cache decoding with speculative rollback." if is_kv else "No KV cache; full-prefix recompute diagnostic path."}
    write_json(out/"config_resolved.json",resolved)
    env={"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,"cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"transformers":transformers.__version__,"target_revision_or_path":a.target,"draft_revision_or_path":resolved["draft"],"draft_runtime":a.draft_runtime,"distribution_mode":a.distribution_mode,"draft_setup_timings":draft_setup_dict,"git_commit":None,"git_dirty":None}
    try:
        env["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=project_root(),text=True,stderr=subprocess.DEVNULL).strip()
        porcelain=subprocess.check_output(["git","status","--porcelain"],cwd=project_root(),text=True,stderr=subprocess.DEVNULL).strip()
        env["git_dirty"]=bool(porcelain)
    except Exception: pass
    if torch.cuda.is_available():
        env["gpu"]=[{"name":torch.cuda.get_device_name(i),"memory_bytes":torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
        torch.cuda.reset_peak_memory_stats()
    write_json(out/"environment.json",env)
    writer=TraceWriter(out/"round_traces.jsonl",out/"token_traces.jsonl")
    canonical=[]; trial_rows=[]
    common={"mode":a.mode,"max_new_tokens":a.max_new_tokens,"draft_length":a.draft_length,"temperature":a.temperature,"top_k":a.top_k,"top_p":a.top_p,"cache_strategy":a.cache_strategy,"dynamic_k":a.dynamic_k,"biasing_config":biasing_cfg,"enable_biasing":enable_biasing,"distribution_mode":a.distribution_mode}
    if a.global_warmup > 0:
        wrow = rows[0]
        wev = [e.get("text","") for e in wrow.get("evidence",[])]
        if a.draft_context_policy == "target_attn_sentence_budget":
            w_attn_cfg = {"row": wrow, "budget_chars": a.draft_budget_chars, "tail_window": a.draft_attn_tail_window}
            wdraft = None
        else:
            w_attn_cfg = None
            wdraft = build_draft_context(wrow, policy=a.draft_context_policy, top_k=a.draft_top_k, compact_sentences=a.draft_compact_sentences, compact_chars=a.draft_compact_chars, budget_chars=a.draft_budget_chars) if a.draft_context_policy != "full" else None
        for gw in range(a.global_warmup):
            pair.generate(wrow["prompt"],**{**common,"max_new_tokens":min(32,a.max_new_tokens)},seed=a.seed+9999+gw,run_id=f"{run_id}-gwarm{gw}",claim_id=str(wrow["claim_id"]),evidence_texts=wev,draft_prompt=wdraft.draft_prompt if wdraft else None,target_attn_selector_cfg=w_attn_cfg)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    with (out/"predictions.jsonl").open("x",encoding="utf-8") as prediction_file, (out/"trial_metrics.jsonl").open("x",encoding="utf-8") as trial_file:
        for index,row in enumerate(rows):
            for warm in range(a.warmup):
                pair.generate(row["prompt"],**common,seed=a.seed+index,run_id=f"{run_id}-warmup{warm}",claim_id=str(row["claim_id"]))
            evidence_texts=[e.get("text","") for e in row.get("evidence",[])]
            for trial in range(a.trials):
                trial_run_id=f"{run_id}-trial{trial}"
                if a.draft_context_policy == "target_attn_sentence_budget":
                    attn_cfg = {"row": row, "budget_chars": a.draft_budget_chars, "tail_window": a.draft_attn_tail_window}
                    generation=pair.generate(row["prompt"],**common,seed=a.seed+index+trial,run_id=trial_run_id,claim_id=str(row["claim_id"]),evidence_texts=evidence_texts,draft_prompt=None,target_attn_selector_cfg=attn_cfg)
                    dctx = generation.draft_context_result
                    if dctx is None:
                        raise RuntimeError("target_attn_sentence_budget failed to populate generation.draft_context_result")
                else:
                    dctx = build_draft_context(row, policy=a.draft_context_policy, top_k=a.draft_top_k, compact_sentences=a.draft_compact_sentences, compact_chars=a.draft_compact_chars, budget_chars=a.draft_budget_chars)
                    draft_prompt_arg = dctx.draft_prompt if a.draft_context_policy != "full" else None
                    generation=pair.generate(row["prompt"],**common,seed=a.seed+index+trial,run_id=trial_run_id,claim_id=str(row["claim_id"]),evidence_texts=evidence_texts,draft_prompt=draft_prompt_arg)
                    generation.timings["context_selection_s"] = dctx.context_selection_s
                record_run={"run_id":run_id,"claim_id":str(row["claim_id"]),"evidence_mode":row.get("evidence_mode"),"prompt_hash":row.get("prompt_hash"),"draft_context_policy":a.draft_context_policy,"draft_selected_evidence_ids":dctx.selected_evidence_ids,"evidence_ids":[e.get("evidence_id") for e in row.get("evidence",[]) if e.get("evidence_id")],"evidence_count":len(row.get("evidence",[])),"raw_text":generation.raw_text,"token_ids":generation.token_ids,"prompt_tokens":generation.prompt_tokens,"output_tokens":len(generation.token_ids),"timings":generation.timings,"rounds":generation.rounds,"evaluation":row.get("evaluation",{})}
                for trace in generation.rounds: trace["mode"]=a.mode; trace["draft_length_configured"]=a.draft_length
                for trace in generation.token_traces: trace["mode"]=a.mode; trace["draft_length_configured"]=a.draft_length
                parsed=parse_generation(generation.raw_text,{e.get("evidence_id") for e in row.get("evidence",[]) if e.get("evidence_id")})
                record_run.update({"parsed":parsed["parsed"],"parse_success":parsed["parse_success"],"parse_error":parsed["parse_error"],"citation_id_validity":parsed["citation_id_validity"],"citation_id_valid_count":parsed["citation_id_valid_count"],"citation_count":parsed["citation_count"]})
                trial_metric={"run_id":run_id,"trial_index":trial,"claim_id":str(row["claim_id"]),"prompt_hash":row.get("prompt_hash"),"draft_context_policy":a.draft_context_policy,"selected_evidence_count":dctx.selected_evidence_count,"total_evidence_count":dctx.total_evidence_count,"draft_evidence_chars":dctx.draft_evidence_chars,"full_evidence_chars":dctx.full_evidence_chars,"prompt_tokens":generation.prompt_tokens,"output_tokens":len(generation.token_ids),**generation.timings,"parse_success":parsed["parse_success"]}
                trial_file.write(json.dumps(trial_metric,ensure_ascii=False,allow_nan=False)+"\n"); trial_rows.append(trial_metric)
                if trial==0:
                    for trace in generation.rounds: writer.write_round(trace)
                    for trace in generation.token_traces: writer.write_token(trace)
                    prediction_file.write(json.dumps(record_run,ensure_ascii=False,allow_nan=False)+"\n"); canonical.append(record_run)
                    print(json.dumps({"claim_id":record_run["claim_id"],"output_tokens":record_run["output_tokens"],"parse_success":record_run["parse_success"],"decode_s":record_run["timings"]["decode_s"]}))
    if torch.cuda.is_available():
        env["gpu_peak_allocated_bytes"]=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]
        env["gpu_peak_reserved_bytes"]=[torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())]
        write_json(out/"environment.json",env)
    metric={"run_id":run_id,"examples":len(canonical),"trials_per_example":a.trials,"warmups_per_example":a.warmup,"global_warmup":a.global_warmup,
            "draft_runtime":a.draft_runtime,"distribution_mode":a.distribution_mode,"draft_setup_timings":draft_setup_dict,
            "json_parse_success_rate":sum(x["parse_success"] for x in canonical)/len(canonical),
            "citation_id_validity_rate":sum(x["citation_id_validity"] for x in canonical)/len(canonical),
            "mean_output_tokens":statistics.mean(x["output_tokens"] for x in canonical),"mean_decode_s_per_trial":statistics.mean(x["decode_s"] for x in trial_rows),
            "timings_authoritative":is_kv,"factuality":"not_evaluated"}
    write_json(out/"metrics.json",metric)
    print(f"Run saved to {out}")

if __name__=="__main__": main()
