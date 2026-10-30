"""Append-only JSONL trace writer."""
from __future__ import annotations
import json
from pathlib import Path
from .trace_schema import ROUND_FIELDS, TOKEN_FIELDS, validate_record

class TraceWriter:
    def __init__(self, round_path: str | Path, token_path: str | Path):
        self.round_path=Path(round_path); self.token_path=Path(token_path)
        self.round_path.parent.mkdir(parents=True,exist_ok=True); self.token_path.parent.mkdir(parents=True,exist_ok=True)
        self.round_path.touch(exist_ok=True); self.token_path.touch(exist_ok=True)
    @staticmethod
    def _append(path: Path, row: dict):
        with path.open("a",encoding="utf-8") as f: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    def write_round(self, row: dict): validate_record(row,ROUND_FIELDS); self._append(self.round_path,row)
    def write_token(self, row: dict): validate_record(row,TOKEN_FIELDS); self._append(self.token_path,row)
