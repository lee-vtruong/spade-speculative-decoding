"""Offline output evaluation; factuality is intentionally not inferred."""
from __future__ import annotations
import json
from collections import Counter
from typing import Any
LABEL_MAP = {"Supported":"Supported", "Refuted":"Refuted", "Not Enough Evidence":"Not Enough Evidence", "Conflicting Evidence/Cherry-picking":"Conflicting Evidence/Cherry-picking"}

def parse_generation(raw: str, evidence_ids: set[str]) -> dict[str, Any]:
    try:
        obj=json.loads(raw)
        if not isinstance(obj,dict): raise ValueError("JSON root must be an object")
        if not isinstance(obj.get("summary"),str) or not isinstance(obj.get("citations"),list): raise ValueError("summary/citations have invalid types")
        verdict=obj.get("verdict")
        if verdict not in LABEL_MAP: raise ValueError(f"Unknown verdict: {verdict!r}")
        citations=[str(x) for x in obj["citations"]]
        return {"parsed":obj,"parse_success":True,"parse_error":None,"mapped_label":LABEL_MAP[verdict],"citation_ids_valid":all(x in evidence_ids for x in citations),"summary_chars":len(obj["summary"])}
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return {"parsed":None,"parse_success":False,"parse_error":str(exc),"mapped_label":None,"citation_ids_valid":False,"summary_chars":None}

def classification_metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    labels=sorted(set(gold)|set(pred)); per={}; tp_total=0
    for label in labels:
        tp=sum(a==label and b==label for a,b in zip(gold,pred)); fp=sum(a!=label and b==label for a,b in zip(gold,pred)); fn=sum(a==label and b!=label for a,b in zip(gold,pred))
        p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.; f=2*p*r/(p+r) if p+r else 0.
        per[label]={"precision":p,"recall":r,"f1":f}; tp_total+=tp
    return {"accuracy":tp_total/len(gold) if gold else 0.,"macro_f1":sum(x["f1"] for x in per.values())/len(per) if per else 0.,"per_class":per,"factuality":"not_evaluated"}
