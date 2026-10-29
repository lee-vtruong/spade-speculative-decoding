"""Transparent token heuristics and evidence lexical overlap."""
from __future__ import annotations
import re
_NUM=re.compile(r"\d"); _DATE=re.compile(r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|\d{4}[-/]\d{1,2})\b",re.I)
_WORD=re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
def categorize_token(text: str) -> str:
    s=text.strip()
    if re.search(r"\bE\d+\b|\[E\d+\]",s): return "citation"
    if _DATE.search(s): return "date_related"
    if _NUM.search(s): return "numeric"
    if s and all(not c.isalnum() and not c.isspace() for c in s): return "punctuation"
    # Heuristic only; subword tokens are not guaranteed word boundaries.
    if any(w[:1].isupper() and len(w)>2 for w in s.split()): return "potential_named_entity"
    return "other"
def evidence_overlap(text: str, evidence: list[str]) -> dict:
    words={w.lower() for w in _WORD.findall(text)}; ev={w.lower() for passage in evidence for w in _WORD.findall(passage)}
    return {"lexical_overlap":len(words&ev)/len(words) if words else 0.0,"has_citation_marker":bool(re.search(r"\bE\d+\b",text)),"overlapping_terms":sorted(words&ev)}
