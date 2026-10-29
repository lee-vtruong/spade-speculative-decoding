"""Adapters for official AVeriTeC JSON exports (schema-tolerant, no downloads)."""
from __future__ import annotations
import json
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
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not all(isinstance(x, dict) for x in rows): raise ValueError("JSONL rows must be objects")
        return rows

def claim_id(row: dict[str, Any]) -> str:
    for key in ("claim_id", "id", "claimId"):
        if row.get(key) is not None: return str(row[key])
    raise ValueError("Record is missing a claim identifier (claim_id/id)")

def claim_text(row: dict[str, Any]) -> str:
    value = row.get("claim")
    if isinstance(value, dict): value = value.get("claim") or value.get("text")
    if not isinstance(value, str) or not value.strip(): raise ValueError(f"Claim {claim_id(row)} has no claim text")
    return value.strip()

def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    yield from load_records(path)
