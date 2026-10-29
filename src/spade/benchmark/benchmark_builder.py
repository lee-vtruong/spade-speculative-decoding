"""Build leakage-separated prompt records and deterministic stratified subsets."""
from __future__ import annotations
import random
from typing import Any
from .averitec_loader import claim_id, claim_text
from .evidence_adapter import oracle_evidence, retrieved_evidence
from .prompt_builder import build_prompt

def stratified_sample(rows: list[dict[str, Any]], size: int, seed: int = 42) -> list[dict[str, Any]]:
    if size < 1: raise ValueError("size must be positive")
    if size >= len(rows): return list(rows)
    groups: dict[str, list] = {}
    for r in rows: groups.setdefault(str(r.get("label", "unknown")), []).append(r)
    rng = random.Random(seed); chosen=[]
    # Largest-remainder allocation, preserving class proportions as closely as possible.
    quota = {k: size*len(v)/len(rows) for k,v in groups.items()}
    counts = {k: int(q) for k,q in quota.items()}
    for k in sorted(groups, key=lambda x: quota[x]-counts[x], reverse=True)[:size-sum(counts.values())]: counts[k]+=1
    for key, group in groups.items(): chosen.extend(rng.sample(group, min(counts[key], len(group))))
    rng.shuffle(chosen)
    return chosen[:size]

def build_examples(rows: list[dict[str, Any]], mode: str, retrieved: dict[str, list[dict]] | None = None, top_k: int = 8) -> list[dict[str, Any]]:
    result=[]
    for row in rows:
        cid=claim_id(row)
        evidence=oracle_evidence(row) if mode == "oracle_qa" else retrieved_evidence(row, (retrieved or {}).get(cid), top_k)
        prompt, digest=build_prompt(claim_text(row), evidence, mode)
        # Evaluation-only fields are structurally separate; never pass this object directly to a model.
        result.append({"claim_id":cid,"claim":claim_text(row),"evidence":evidence,"prompt":prompt,"prompt_hash":digest,"evidence_mode":mode,
                       "evaluation":{"gold_label":row.get("label"),"gold_justification":row.get("justification")}})
    return result
