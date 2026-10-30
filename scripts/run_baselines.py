"""Run correctness-first B0/B0-S/B1/B2 diagnostics with full-prefix recomputation.

The current adapter deliberately avoids KV caching. Its latency is not suitable
for official performance claims; it is intended for a short GPU integration run.
"""
from __future__ import annotations
import argparse, json, os, platform, subprocess, sys, uuid
from datetime import datetime, timezone
from pathlib import Path

def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["B0","B0-S","B1","B2"],required=True)
    p.add_argument("--input",type=Path,required=True,help="Prepared benchmark JSONL")
    p.add_argument("--output-root",type=Path,default=Path(os.environ.get("OUTPUT_ROOT","outputs/runs")))
    p.add_argument("--run-id",default=None)
    p.add_argument("--target",default="Qwen/Qwen3-8B")
    p.add_argument("--draft",default="Qwen/Qwen3-0.6B")
    p.add_argument("--dtype",choices=["bfloat16","float16","float32"],default="bfloat16")
    p.add_argument("--draft-length",type=int,choices=[2,4,6,8],default=4)
    p.add_argument("--max-new-tokens",type=int,default=256)
    p.add_argument("--temperature",type=float,default=1.0)
    p.add_argument("--top-k",type=int,default=0)
    p.add_argument("--top-p",type=float,default=1.0)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--limit",type=int,default=None,help="Diagnostic subset limit")
    a=p.parse_args()
    if a.mode in {"B0","B0-S"} and a.mode=="B0" and a.draft_length<1: p.error("draft-length must be positive")
    if a.max_new_tokens<1: p.error("max-new-tokens must be positive")
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
    resolved={"run_id":run_id,"mode":a.mode,"input":str(a.input.resolve()),"target":a.target,"draft":None if a.mode in {"B0","B0-S"} else a.draft,"dtype":a.dtype,"draft_length":a.draft_length,"max_new_tokens":a.max_new_tokens,"temperature":a.temperature,"top_k":a.top_k,"top_p":a.top_p,"seed":a.seed,"limit":a.limit,"cache_strategy":"full_prefix_recomputation","timings_authoritative":False,"timing_note":"No KV cache; diagnostic correctness path only."}
    (out/"config_resolved.json").write_text(json.dumps(resolved,indent=2),encoding="utf-8")
    env={"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,"cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"transformers":transformers.__version__,"git_commit":None}
    try: env["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=project_root(),text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: pass
    if torch.cuda.is_available(): env["gpu"]=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    (out/"environment.json").write_text(json.dumps(env,indent=2),encoding="utf-8")
    predictions=[]
    trace_writer=TraceWriter(out/"round_traces.jsonl",out/"token_traces.jsonl")
    with (out/"predictions.jsonl").open("x",encoding="utf-8") as f:
        for index,row in enumerate(rows):
            generation=pair.generate(row["prompt"],mode=a.mode,max_new_tokens=a.max_new_tokens,draft_length=a.draft_length,temperature=a.temperature,top_k=a.top_k,top_p=a.top_p,seed=a.seed+index,run_id=run_id,claim_id=str(row["claim_id"]),evidence_texts=[e.get("text","") for e in row.get("evidence",[])])
            for trace in generation.rounds: trace_writer.write_round(trace)
            for trace in generation.token_traces: trace_writer.write_token(trace)
            evidence_ids=[e.get("evidence_id") for e in row.get("evidence",[]) if e.get("evidence_id")]
            parsed=parse_generation(generation.raw_text,set(evidence_ids))
            record={"run_id":run_id,"claim_id":str(row["claim_id"]),"evidence_mode":row.get("evidence_mode"),"prompt_hash":row.get("prompt_hash"),"raw_text":generation.raw_text,"token_ids":generation.token_ids,"parsed":parsed["parsed"],"parse_success":parsed["parse_success"],"parse_error":parsed["parse_error"],"citation_ids_valid":parsed["citation_ids_valid"],"prompt_tokens":generation.prompt_tokens,"output_tokens":len(generation.token_ids),"timings":generation.timings,"rounds":generation.rounds,"evaluation":row.get("evaluation",{})}
            f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n"); f.flush(); predictions.append(record)
            print(json.dumps({"claim_id":record["claim_id"],"output_tokens":record["output_tokens"],"parse_success":record["parse_success"],"decode_s":record["timings"]["decode_s"]}))
    summary={"run_id":run_id,"examples":len(predictions),"json_parse_success_rate":sum(x["parse_success"] for x in predictions)/len(predictions),"citation_id_validity_rate":sum(x["citation_ids_valid"] for x in predictions)/len(predictions),"timings_authoritative":False,"factuality":"not_evaluated"}
    (out/"metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"Run saved to {out}")

if __name__=="__main__": main()
