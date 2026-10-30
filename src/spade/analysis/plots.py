"""Build M3 plots/tables only from supplied real run directories."""
from __future__ import annotations
import csv,json,math,statistics
from collections import Counter,defaultdict
from pathlib import Path
from typing import Any

def _read_json(path: Path): return json.loads(path.read_text(encoding="utf-8"))
def _read_jsonl(path: Path): return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()] if path.exists() else []
def _csv(path: Path,rows: list[dict],fields: list[str] | None=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    cols=fields or sorted({k for row in rows for k in row})
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def _compatible_speedups(runs: list[dict]) -> list[dict]:
    results=[]
    for base in runs:
        for spec in runs:
            pair=(base["config"].get("mode"),spec["config"].get("mode"))
            if pair not in {("B0","B1"),("B0-S","B2")}: continue
            keys=("target","dtype","temperature","top_k","top_p","max_new_tokens")
            if any(base["config"].get(k)!=spec["config"].get(k) for k in keys): continue
            bpred={x["claim_id"]:x for x in base["predictions"]}; spred={x["claim_id"]:x for x in spec["predictions"]}
            if set(bpred)!=set(spred): continue
            for cid in bpred:
                b,s=bpred[cid],spred[cid]
                if b.get("prompt_hash")!=s.get("prompt_hash") or b.get("output_tokens")!=s.get("output_tokens"): continue
                bt=b.get("timings",{}).get("decode_s"); st=s.get("timings",{}).get("decode_s")
                if isinstance(bt,(int,float)) and isinstance(st,(int,float)) and st>0:
                    results.append({"baseline_run":base["config"]["run_id"],"speculative_run":spec["config"]["run_id"],"baseline_mode":pair[0],"speculative_mode":pair[1],"claim_id":cid,"matched_output_tokens":b["output_tokens"],"baseline_decode_s":bt,"speculative_decode_s":st,"diagnostic_speedup":bt/st,"timings_authoritative":False})
    return results

def analyze_runs(run_dirs: list[str | Path], output_dir: str | Path) -> dict[str,Any]:
    """Load runs and emit available PNG figures, CSV supporting tables, JSON summary."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc: raise RuntimeError("Install matplotlib to create M3 plots") from exc
    runs=[]; rounds=[]; tokens=[]
    for raw in run_dirs:
        root=Path(raw); run={"root":root,"config":_read_json(root/"config_resolved.json"),"predictions":_read_jsonl(root/"predictions.jsonl"),"rounds":_read_jsonl(root/"round_traces.jsonl"),"tokens":_read_jsonl(root/"token_traces.jsonl")}
        for row in run["rounds"]+run["tokens"]:
            row.setdefault("mode",run["config"].get("mode")); row.setdefault("draft_length_configured",run["config"].get("draft_length"))
        runs.append(run); rounds.extend(run["rounds"]); tokens.extend(run["tokens"])
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); generated=[]
    def savefig(name):
        plt.tight_layout(); plt.savefig(out/(name+".png"),dpi=160); plt.close(); generated.append(name+".png")
    reached=[x for x in tokens if x.get("mode")=="B2" and x.get("status") in {"accepted","rejected"} and x.get("reached_by_verifier")]

    # 1. Acceptance by token category.
    cat=defaultdict(list)
    for x in reached: cat[x.get("token_category","other")].append(x)
    cat_rows=[{"category":k,"accepted":sum(x["status"]=="accepted" for x in v),"proposals":len(v),"acceptance_rate":sum(x["status"]=="accepted" for x in v)/len(v)} for k,v in sorted(cat.items())]
    _csv(out/"acceptance_by_category.csv",cat_rows,["category","accepted","proposals","acceptance_rate"])
    if cat_rows:
        plt.figure(figsize=(8,4)); plt.bar([x["category"] for x in cat_rows],[x["acceptance_rate"] for x in cat_rows]); plt.ylabel("Observed acceptance rate"); plt.xticks(rotation=25,ha="right"); plt.ylim(0,1); savefig("acceptance_by_token_category")

    # 2. Rejection position histogram.
    rejected=[int(x["rejection_position"]) for x in rounds if x.get("mode")=="B2" and x.get("rejection_position") is not None]
    reject_counts=Counter(rejected); _csv(out/"rejection_positions.csv",[{"position":k,"count":v} for k,v in sorted(reject_counts.items())],["position","count"])
    if rejected:
        plt.figure(figsize=(7,4)); plt.hist(rejected,bins=range(min(rejected),max(rejected)+2),align="left",rwidth=.85); plt.xlabel("Zero-based proposal position"); plt.ylabel("Rejection rounds"); savefig("rejection_position_histogram")

    # 3. Accepted tokens per round.
    round_rows=[{"run_id":r.get("run_id"),"mode":r.get("mode"),"claim_id":r.get("claim_id"),"round_index":r.get("round_index"),"draft_length":r.get("draft_length_requested"),"proposed":r.get("proposed_token_count"),"accepted":r.get("accepted_draft_tokens"),"committed":r.get("committed_output_tokens"),"rejection_position":r.get("rejection_position"),"round_latency_s":r.get("total_round_latency_s"),"prefix_length":r.get("prefix_length")} for r in rounds]
    _csv(out/"round_metrics.csv",round_rows)
    by_draft=defaultdict(list)
    for x in rounds:
        if x.get("mode")=="B2" and x.get("draft_length_requested",0)>0: by_draft[int(x["draft_length_requested"])].append(int(x.get("accepted_draft_tokens",0)))
    if by_draft:
        xs=sorted(by_draft); ys=[statistics.mean(by_draft[x]) for x in xs]
        plt.figure(figsize=(7,4)); plt.plot(xs,ys,marker="o"); plt.xlabel("Requested draft length"); plt.ylabel("Mean accepted draft tokens / round"); savefig("mean_accepted_tokens_per_round")

    # 4. Latency breakdown from canonical prediction records.
    latency=[]
    for run in runs:
        for x in run["predictions"]:
            t=x.get("timings",{}); latency.append({"run_id":run["config"].get("run_id"),"claim_id":x["claim_id"],"mode":run["config"].get("mode"),"draft_length":run["config"].get("draft_length"),"draft_generation_s":t.get("draft_generation_s",0),"target_verification_s":t.get("target_verification_s",0),"correction_s":t.get("correction_s",0),"decode_s":t.get("decode_s",0),"output_tokens":x.get("output_tokens",0)})
    _csv(out/"latency_breakdown.csv",latency)
    if latency:
        modes=[]; draft_s=[]; target_s=[]; correction_s=[]
        for mode in sorted({x["mode"] for x in latency}):
            group=[x for x in latency if x["mode"]==mode]; modes.append(mode); draft_s.append(statistics.mean(x["draft_generation_s"] for x in group)); target_s.append(statistics.mean(x["target_verification_s"] for x in group)); correction_s.append(statistics.mean(x["correction_s"] for x in group))
        plt.figure(figsize=(8,4)); bottom=[0.0]*len(modes)
        for label,vals in (("Draft",draft_s),("Target verification",target_s),("Correction",correction_s)):
            plt.bar(modes,vals,bottom=bottom,label=label); bottom=[a+b for a,b in zip(bottom,vals)]
        plt.ylabel("Mean diagnostic decode time (s)"); plt.legend(); savefig("latency_breakdown")

    # 5. Speedup is plotted only for matched-policy runs with equal output lengths.
    speedups=_compatible_speedups(runs); _csv(out/"speedup_comparison.csv",speedups)
    if speedups:
        grouped=defaultdict(list)
        for x in speedups: grouped[(x["speculative_mode"],next((r["config"].get("draft_length") for r in runs if r["config"].get("run_id")==x["speculative_run"]),0))].append(x["diagnostic_speedup"])
        labels=[f"{m} k={k}" for m,k in sorted(grouped)]; vals=[statistics.mean(grouped[key]) for key in sorted(grouped)]
        plt.figure(figsize=(8,4)); plt.bar(labels,vals); plt.axhline(1,color="black",linewidth=.8); plt.ylabel("Diagnostic decode speedup (non-authoritative)"); plt.xticks(rotation=20); savefig("diagnostic_speedup_by_draft_length")

    # 6. Acceptance versus input/prefix context length (1024-token bins).
    context=defaultdict(list)
    for x in reached:
        length=int(x.get("prefix_length",0)); bucket=(length//1024)*1024
        context[bucket].append(x["status"]=="accepted")
    context_rows=[{"prefix_tokens_bin_start":k,"prefix_tokens_bin_end":k+1023,"proposals":len(v),"acceptance_rate":sum(v)/len(v)} for k,v in sorted(context.items())]
    _csv(out/"acceptance_by_context_length.csv",context_rows)
    if context_rows:
        plt.figure(figsize=(8,4)); plt.plot([x["prefix_tokens_bin_start"] for x in context_rows],[x["acceptance_rate"] for x in context_rows],marker="o"); plt.xlabel("Prefix tokens (1024-token bins)"); plt.ylabel("Observed acceptance rate"); plt.ylim(0,1); savefig("acceptance_vs_context_length")

    # 7. p/q ratio distribution.
    ratios=[]
    for x in reached:
        pval=x.get("target_probability"); qval=x.get("draft_probability")
        if isinstance(pval,(int,float)) and isinstance(qval,(int,float)) and qval>0: ratios.append({"run_id":x.get("run_id"),"claim_id":x.get("claim_id"),"round_id":x.get("round_id"),"token_id":x.get("token_id"),"p":pval,"q":qval,"p_over_q":pval/qval,"total_variation":x.get("total_variation"),"expected_acceptance_probability":x.get("expected_acceptance_probability")})
    _csv(out/"pq_ratio_diagnostics.csv",ratios)
    if ratios:
        plt.figure(figsize=(8,4)); plt.hist([math.log10(max(x["p_over_q"],1e-12)) for x in ratios],bins=40); plt.xlabel("log10(p(x)/q(x))"); plt.ylabel("Proposed tokens"); savefig("p_over_q_distribution")

    # 8. One real token-level trace example.
    b2_tokens=[x for x in tokens if x.get("mode")=="B2"]
    if b2_tokens:
        first_run=b2_tokens[0].get("run_id"); first_claim=b2_tokens[0].get("claim_id")
        example=sorted([x for x in b2_tokens if x.get("run_id")==first_run and x.get("claim_id")==first_claim],key=lambda x:(x.get("round_id",0),x.get("absolute_token_position",0)))
        _csv(out/"example_token_trace.csv",example)
        palette={"accepted":"#2ca02c","rejected":"#d62728","discarded_after_rejection":"#999999","residual_replacement":"#9467bd","target_replacement":"#ff7f0e","extra_target_token":"#1f77b4"}
        plt.figure(figsize=(11,4))
        for status in sorted({x["status"] for x in example}):
            group=[x for x in example if x["status"]==status]
            plt.scatter([x["absolute_token_position"] for x in group],[x["round_id"] for x in group],label=status,color=palette.get(status))
        plt.xlabel("Absolute token position"); plt.ylabel("Speculative round"); plt.legend(ncol=3,fontsize=8); savefig("example_token_trace")

    report={"run_dirs":[str(Path(x).resolve()) for x in run_dirs],"round_count":len(rounds),"token_trace_count":len(tokens),"reached_proposal_count":len(reached),"plots_generated":generated,"speedup_rows":len(speedups),"timings_authoritative":False}
    (out/"analysis_manifest.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report
