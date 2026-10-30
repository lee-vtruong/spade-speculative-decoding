"""Adapters for official AVeriTeC JSON exports (schema-tolerant, no downloads)."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import Any, Iterator

def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON array or JSONL file, rejecting unexpected top-level shapes."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            for key in ("data", "claims", "examples"):
                if isinstance(value.get(key), list): value = value[key]; break
        if not isinstance(value, list) or not all(isinstance(x, dict) for x in value):
            raise ValueError("Expected a list of record objects")
        return value
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.split("\n") if line.strip()]
        if not all(isinstance(x, dict) for x in rows): raise ValueError("JSONL rows must be objects")
        return rows

def claim_id(row: dict[str, Any]) -> str:
    for key in ("claim_id", "id", "claimId"):
        if row.get(key) is not None: return str(row[key])
    # Official exports may omit an explicit ID. Derive a stable identifier only
    # from claim-visible metadata; never include label or justification.
    claim = row.get("claim")
    if isinstance(claim, dict): claim = claim.get("claim") or claim.get("text")
    if isinstance(claim, str) and claim.strip():
        identity = "\0".join(str(row.get(k, "")) for k in ("original_claim_url", "claim_date", "speaker")) + "\0" + claim.strip()
        return "avc-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    raise ValueError("Record is missing an identifier and has no usable claim text")

def claim_text(row: dict[str, Any]) -> str:
    value = row.get("claim")
    if isinstance(value, dict): value = value.get("claim") or value.get("text")
    if not isinstance(value, str) or not value.strip(): raise ValueError(f"Claim {claim_id(row)} has no claim text")
    return value.strip()

def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    yield from load_records(path)
