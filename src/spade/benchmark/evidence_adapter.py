"""Explicit oracle QA and claim-specific retrieved evidence adapters."""
from __future__ import annotations
from typing import Any

def _passage(text: str, index: int, source: str = "", rank: int | None = None, score: float | None = None, provenance: str = "") -> dict[str, Any]:
    return {"evidence_id": f"E{index}", "text": text, "source_url": source or None, "rank": rank, "score": score, "provenance": provenance}

def oracle_evidence(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert annotated QA to diagnostic oracle_qa evidence; never use as retrieval."""
    qa = row.get("questions") or row.get("qa") or row.get("question_answer_pairs") or []
    out = []
    for item in qa:
        if not isinstance(item, dict): continue
        question = item.get("question", "")
        answer = item.get("answer", "")
        text = " ".join(x.strip() for x in (question, answer) if isinstance(x, str) and x.strip())
        if text: out.append(_passage(text, len(out)+1, provenance="oracle_qa"))
    return out

def retrieved_evidence(row: dict[str, Any], retrieved: list[dict[str, Any]] | None, top_k: int = 8) -> list[dict[str, Any]]:
    """Normalize externally retrieved passages. Gold fields are deliberately ignored."""
    if top_k < 1: raise ValueError("top_k must be positive")
    out = []
    for item in (retrieved or [])[:top_k]:
        text = item.get("text") or item.get("passage") or item.get("content")
        if not isinstance(text, str) or not text.strip(): continue
        out.append(_passage(text.strip(), len(out)+1, str(item.get("source_url") or item.get("url") or ""), item.get("rank"), item.get("score"), str(item.get("provenance") or "retrieved")))
    return out
