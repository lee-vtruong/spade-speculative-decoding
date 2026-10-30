"""Build leakage-separated prompt records and deterministic stratified subsets."""
from __future__ import annotations
import random
import hashlib
import statistics
from datetime import datetime, timezone
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

def build_manifest(*, dataset_path: str, dataset_hash: str, split: str, seed: int, mode: str,
                   top_k: int, examples: list[dict[str, Any]], token_lengths: list[int] | None = None,
                   retrieval_audit: dict[str, Any] | None = None) -> dict[str, Any]:
    evidence_counts=[len(x["evidence"]) for x in examples]
    labels={}
    for x in examples:
        label=str(x["evaluation"].get("gold_label","unknown")); labels[label]=labels.get(label,0)+1
    stats={"example_count":len(examples),"evidence_count":{"min":min(evidence_counts,default=0),"max":max(evidence_counts,default=0),"mean":statistics.mean(evidence_counts) if evidence_counts else 0.0},
           "gold_label_counts_evaluation_only":labels,"input_token_count":None}
    if token_lengths is not None:
        stats["input_token_count"]={"min":min(token_lengths,default=0),"max":max(token_lengths,default=0),"mean":statistics.mean(token_lengths) if token_lengths else 0.0}
    return {"manifest_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),"dataset":"AVeriTeC","dataset_path":dataset_path,
            "dataset_sha256":dataset_hash,"split":split,"selected_claim_ids":[x["claim_id"] for x in examples],"sampling_seed":seed,
            "evidence_mode":mode,"evidence_top_k":top_k,"prompt_template_version":"spade_factcheck_v1",
            "prompt_hashes":{x["claim_id"]:x["prompt_hash"] for x in examples},"token_lengths_include_chat_template":token_lengths is not None,
            "statistics":stats,"retrieval_audit":retrieval_audit}
