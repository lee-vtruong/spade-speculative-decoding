"""Deterministic model-visible prompt with gold metadata kept out of scope."""
from __future__ import annotations
import hashlib, json
PROMPT_VERSION = "spade_factcheck_v1"
VERDICTS = ("Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherry-picking")

def build_prompt(claim: str, evidence: list[dict], mode: str = "retrieved") -> tuple[str, str]:
    if mode not in ("retrieved", "oracle_qa"): raise ValueError(f"Unknown evidence mode: {mode}")
    visible = [{"evidence_id": e["evidence_id"], "text": e["text"]} for e in evidence]
    body = ("You are writing an evidence-grounded fact-checking summary. Read the claim and examine the supplied evidence. "
            "Write a concise factual summary explaining whether the evidence supports or contradicts the claim. Cite only supplied evidence IDs. "
            "Avoid unsupported claims and preserve uncertainty or contradictions. Return only a JSON object with keys summary, citations, verdict. "
            f"verdict must be one of: {', '.join(VERDICTS)}. Evidence mode: {mode}.\n"
            f"Claim: {claim}\nEvidence: {json.dumps(visible, ensure_ascii=False, separators=(',', ':'))}\n"
            'JSON: {"summary":"...","citations":["E1"],"verdict":"..."}')
    return body, hashlib.sha256(body.encode("utf-8")).hexdigest()
