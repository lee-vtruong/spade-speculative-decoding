"""Deterministic model-visible prompt with gold metadata kept out of scope."""
from __future__ import annotations
import hashlib, json
PROMPT_VERSION = "spade_factcheck_v1"
VERDICTS = ("Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherry-picking")

def build_prompt(claim: str, evidence: list[dict], mode: str = "retrieved") -> tuple[str, str]:
    if mode not in ("retrieved", "oracle_qa", "longbench_qa"):
        raise ValueError(f"Unknown evidence mode: {mode}")
    visible = [{"evidence_id": e["evidence_id"], "text": e["text"]} for e in evidence]
    ev_json = json.dumps(visible, ensure_ascii=False, separators=(",", ":"))
    if mode == "longbench_qa":
        body = (
            "You are answering a multi-document question using the supplied evidence passages. "
            "Read the question and examine the evidence carefully. "
            "Write a concise 2-3 sentence explanation citing relevant passage IDs in brackets (e.g., [P1], [P2]), "
            "then end with a line formatted strictly as 'Final Answer: <short answer>'.\n"
            f"Question: {claim}\nEvidence: {ev_json}\nResponse:"
        )
    else:
        body = ("You are writing an evidence-grounded fact-checking summary. Read the claim and examine the supplied evidence. "
                "Write a concise factual summary explaining whether the evidence supports or contradicts the claim. Cite only supplied evidence IDs. "
                "Avoid unsupported claims and preserve uncertainty or contradictions. Return only a JSON object with keys summary, citations, verdict. "
                f"verdict must be one of: {', '.join(VERDICTS)}. Evidence mode: {mode}.\n"
                f"Claim: {claim}\nEvidence: {ev_json}\n"
                'JSON: {"summary":"...","citations":["E1"],"verdict":"..."}')
    return body, hashlib.sha256(body.encode("utf-8")).hexdigest()
