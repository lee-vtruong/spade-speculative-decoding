"""Strict comparison of target-only greedy and greedy speculative outputs."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows={}
    for line_no,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        row=json.loads(line); cid=str(row["claim_id"])
        if cid in rows: raise ValueError(f"Duplicate claim_id {cid!r} in {path}:{line_no}")
        rows[cid]=row
    return rows

def compare_greedy_runs(b0_dir: str | Path, b1_dir: str | Path) -> dict[str, Any]:
    """Compare compatible runs; fail closed on different claims/prompts/settings."""
    b0_dir,b1_dir=Path(b0_dir),Path(b1_dir)
    c0,c1=_read_json(b0_dir/"config_resolved.json"),_read_json(b1_dir/"config_resolved.json")
    if c0.get("mode")!="B0" or c1.get("mode")!="B1": raise ValueError("Expected a B0 run and a B1 run")
    for key in ("target","dtype","max_new_tokens"):
        if c0.get(key)!=c1.get(key): raise ValueError(f"Runs use different {key}: {c0.get(key)!r} vs {c1.get(key)!r}")
    r0=_read_predictions(b0_dir/"predictions.jsonl"); r1=_read_predictions(b1_dir/"predictions.jsonl")
    if set(r0)!=set(r1): raise ValueError(f"Claim IDs differ: B0-only={sorted(set(r0)-set(r1))}, B1-only={sorted(set(r1)-set(r0))}")
    comparisons=[]
    for cid in sorted(r0):
        a,b=r0[cid],r1[cid]
        if a.get("prompt_hash")!=b.get("prompt_hash"): raise ValueError(f"Prompt hash differs for claim {cid}")
        x,y=a.get("token_ids",[]),b.get("token_ids",[])
        mismatch=next((i for i,(u,v) in enumerate(zip(x,y)) if u!=v),None)
        if mismatch is None and len(x)!=len(y): mismatch=min(len(x),len(y))
        comparisons.append({"claim_id":cid,"match":mismatch is None,"b0_tokens":len(x),"b1_tokens":len(y),"first_mismatch_position":mismatch})
    return {"b0_run":str(b0_dir),"b1_run":str(b1_dir),"claims":len(comparisons),"all_match":all(x["match"] for x in comparisons),"comparisons":comparisons}
