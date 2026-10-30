"""Explicit oracle QA and claim-specific retrieved evidence adapters."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any
from .averitec_loader import claim_id, claim_text

def _passage(text: str, index: int, source: str = "", rank: int | None = None, score: float | None = None, provenance: str = "") -> dict[str, Any]:
    return {"evidence_id": f"E{index}", "text": text, "source_url": source or None, "rank": rank, "score": score, "provenance": provenance}

def oracle_evidence(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert annotated QA to diagnostic oracle_qa evidence; never use as retrieval."""
    qa = row.get("questions") or row.get("qa") or row.get("question_answer_pairs") or []
    out = []
    for item in qa:
        if not isinstance(item, dict): continue
        question = item.get("question", "")
        answers = item.get("answers", item.get("answer", []))
        if isinstance(answers, (str, dict)): answers = [answers]
        for answer_item in answers if isinstance(answers, list) else []:
            if isinstance(answer_item, dict):
                answer = answer_item.get("answer", "")
                source = answer_item.get("source_url") or answer_item.get("url") or ""
            else:
                answer, source = answer_item, ""
            text = " ".join(x.strip() for x in (question, answer) if isinstance(x, str) and x.strip())
            if text: out.append(_passage(text, len(out)+1, str(source), provenance="oracle_qa"))
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

def _norm(text: str) -> str:
    return " ".join(text.casefold().split())

def _url2text(value: Any) -> str:
    """Extract passage text from common AVeriTeC url2text list representations."""
    if isinstance(value, str): return value.strip()
    if not isinstance(value, list): return ""
    pieces=[]
    for item in value:
        if isinstance(item,str): text=item
        elif isinstance(item,dict):
            text=next((item[k] for k in ("text","content","sentence","paragraph") if isinstance(item.get(k),str)),"")
        else: text=""
        if text and text.strip(): pieces.append(text.strip())
    return "\n".join(pieces)

def load_claim_only_knowledge_store(dataset_rows: list[dict[str, Any]], store_dir: str | Path, top_k: int = 8) -> tuple[dict[str,list[dict[str,Any]]],dict[str,Any]]:
    """Read per-claim JSONL files, allowing only exact claim-text query rows.

    Files are matched by dataset row index (0.json, 1.json, ...). Any gold,
    question, answer, generated-question, or other query type is excluded.
    A row is accepted only when its stored query equals the original claim.
    """
    if top_k<1: raise ValueError("top_k must be positive")
    root=Path(store_dir); mapped={}; counts={}; missing=[]; query_mismatch=0; empty_text=0; claim_id_mismatch=0
    for index,row in enumerate(dataset_rows):
        try:
            cid=claim_id(row)
            claim=claim_text(row)
        except (ValueError, AttributeError) as exc:
            keys=sorted(row) if isinstance(row,dict) else []
            raise ValueError(f"Invalid AVeriTeC dataset record at row {index} (keys={keys}): {exc}") from exc
        path=root/f"{index}.json"
        if not path.is_file(): missing.append(index); continue
        passages=[]; seen=set()
        with path.open(encoding="utf-8") as f:
            for line_no,line in enumerate(f,1):
                if not line.strip(): continue
                item=json.loads(line); kind=str(item.get("type","missing")); counts[kind]=counts.get(kind,0)+1
                if str(item.get("claim_id",""))!=str(index):
                    claim_id_mismatch+=1; continue
                if kind!="claim": continue
                if _norm(str(item.get("query","")))!=_norm(claim):
                    query_mismatch+=1; continue
                text=_url2text(item.get("url2text"))
                if not text: empty_text+=1; continue
                key=(str(item.get("url") or ""),text)
                if key in seen: continue
                seen.add(key)
                if len(passages)<top_k:
                    passages.append({"text":text,"source_url":item.get("url"),"rank":len(passages)+1,"score":None,
                                     "provenance":"averitec_knowledge_store:exact_claim_query","store_row":line_no})
        mapped[cid]=passages
    audit={"store_dir":str(root.resolve()),"files_expected":len(dataset_rows),"files_missing":missing,"query_types_seen":counts,
           "claim_id_mismatch_rows":claim_id_mismatch,"exact_claim_query_rows_rejected":query_mismatch,"empty_passages":empty_text,"query_type_allowlist":["claim"],
           "claim_query_exact_match_required":True,"top_k":top_k}
    return mapped,audit
