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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def write_json(path: Path, value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["B0","B0-S","B1","B2"],required=True)
    p.add_argument("--input",type=Path,required=True,help="Prepared benchmark JSONL")
    p.add_argument("--output-root",type=Path,default=Path(os.environ.get("OUTPUT_ROOT","outputs/runs")))
    p.add_argument("--run-id",default=None)
    p.add_argument("--target",default="Qwen/Qwen3-8B"); p.add_argument("--draft",default="Qwen/Qwen3-0.6B")
    p.add_argument("--dtype",choices=["bfloat16","float16","float32"],default="bfloat16")
    p.add_argument("--draft-length",type=int,choices=[2,4,6,8],default=4)
    p.add_argument("--max-new-tokens",type=int,default=256)
    p.add_argument("--temperature",type=float,default=1.0); p.add_argument("--top-k",type=int,default=0); p.add_argument("--top-p",type=float,default=1.0)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--limit",type=int,default=None)
    p.add_argument("--warmup",type=int,default=0); p.add_argument("--trials",type=int,default=1)
    a=p.parse_args()
    if a.max_new_tokens<1 or a.trials<1 or a.warmup<0: p.error("Invalid token/warmup/trial count")
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
        from spade.benchmark.evaluation import parse_generation
        from spade.paths import project_root
        from spade.instrumentation.trace_writer import TraceWriter
    except ImportError as exc: raise SystemExit(f"Required runtime dependency missing: {exc}") from exc
    pair=QwenPair(a.target,None if a.mode in {"B0","B0-S"} else a.draft,local_files_only=True,dtype=a.dtype)
    resolved={"run_id":run_id,"mode":a.mode,"input":str(a.input.resolve()),"target":a.target,"draft":None if a.mode in {"B0","B0-S"} else a.draft,"dtype":a.dtype,"draft_length":a.draft_length,"max_new_tokens":a.max_new_tokens,"temperature":a.temperature,"top_k":a.top_k,"top_p":a.top_p,"seed":a.seed,"limit":a.limit,"warmup":a.warmup,"trials":a.trials,"cache_strategy":"full_prefix_recomputation","timings_authoritative":False,"timing_note":"No KV cache; correctness/rejection diagnostic path only."}
    write_json(out/"config_resolved.json",resolved)
    env={"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,"cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"transformers":transformers.__version__,"target_revision_or_path":a.target,"draft_revision_or_path":resolved["draft"],"git_commit":None}
    try: env["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=project_root(),text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: pass
    if torch.cuda.is_available():
        env["gpu"]=[{"name":torch.cuda.get_device_name(i),"memory_bytes":torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
        torch.cuda.reset_peak_memory_stats()
    write_json(out/"environment.json",env)
    writer=TraceWriter(out/"round_traces.jsonl",out/"token_traces.jsonl")
    canonical=[]; trial_rows=[]
    with (out/"predictions.jsonl").open("x",encoding="utf-8") as prediction_file, (out/"trial_metrics.jsonl").open("x",encoding="utf-8") as trial_file:
        for index,row in enumerate(rows):
            common={"mode":a.mode,"max_new_tokens":a.max_new_tokens,"draft_length":a.draft_length,"temperature":a.temperature,"top_k":a.top_k,"top_p":a.top_p}
            for warm in range(a.warmup):
                pair.generate(row["prompt"],**common,seed=a.seed+index,run_id=f"{run_id}-warmup{warm}",claim_id=str(row["claim_id"]))
            evidence_texts=[e.get("text","") for e in row.get("evidence",[])]
            for trial in range(a.trials):
                trial_run_id=f"{run_id}-trial{trial}"
                generation=pair.generate(row["prompt"],**common,seed=a.seed+index+trial,run_id=trial_run_id,claim_id=str(row["claim_id"]),evidence_texts=evidence_texts)
                record_run={"run_id":run_id,"claim_id":str(row["claim_id"]),"evidence_mode":row.get("evidence_mode"),"prompt_hash":row.get("prompt_hash"),"evidence_ids":[e.get("evidence_id") for e in row.get("evidence",[]) if e.get("evidence_id")],"evidence_count":len(row.get("evidence",[])),"raw_text":generation.raw_text,"token_ids":generation.token_ids,"prompt_tokens":generation.prompt_tokens,"output_tokens":len(generation.token_ids),"timings":generation.timings,"rounds":generation.rounds,"evaluation":row.get("evaluation",{})}
                for trace in generation.rounds: trace["mode"]=a.mode; trace["draft_length_configured"]=a.draft_length
                for trace in generation.token_traces: trace["mode"]=a.mode; trace["draft_length_configured"]=a.draft_length
                parsed=parse_generation(generation.raw_text,{e.get("evidence_id") for e in row.get("evidence",[]) if e.get("evidence_id")})
                record_run.update({"parsed":parsed["parsed"],"parse_success":parsed["parse_success"],"parse_error":parsed["parse_error"],"citation_id_validity":parsed["citation_id_validity"],"citation_id_valid_count":parsed["citation_id_valid_count"],"citation_count":parsed["citation_count"]})
                trial_metric={"run_id":run_id,"trial_index":trial,"claim_id":str(row["claim_id"]),"prompt_hash":row.get("prompt_hash"),"prompt_tokens":generation.prompt_tokens,"output_tokens":len(generation.token_ids),**generation.timings,"parse_success":parsed["parse_success"]}
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
    metric={"run_id":run_id,"examples":len(canonical),"trials_per_example":a.trials,"warmups_per_example":a.warmup,
            "json_parse_success_rate":sum(x["parse_success"] for x in canonical)/len(canonical),
            "citation_id_validity_rate":sum(x["citation_id_validity"] for x in canonical)/len(canonical),
            "mean_output_tokens":statistics.mean(x["output_tokens"] for x in canonical),"mean_decode_s_per_trial":statistics.mean(x["decode_s"] for x in trial_rows),
            "timings_authoritative":False,"factuality":"not_evaluated"}
    write_json(out/"metrics.json",metric)
    print(f"Run saved to {out}")

if __name__=="__main__": main()
