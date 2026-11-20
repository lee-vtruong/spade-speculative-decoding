"""Prepare v6 benchmarks on server:
  1. outputs/benchmarks/retrieved-heldout-long173.jsonl (173 selective claims > 10k chars)
  2. outputs/benchmarks/longbench-hotpotqa-100.jsonl (100 multi-doc QA examples from LongBench HotpotQA)
"""
from __future__ import annotations

import json
import re
import statistics
import urllib.request
import zipfile
from pathlib import Path

from transformers import AutoTokenizer

from spade.benchmark.prompt_builder import build_prompt
from spade.decoding.autoregressive import format_qwen_prompt

TARGET_SNAP = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
HELDOUT300 = Path("outputs/benchmarks/retrieved-heldout300.jsonl")
LONG173 = Path("outputs/benchmarks/retrieved-heldout-long173.jsonl")
LB_ZIP_URL = "https://huggingface.co/datasets/zai-org/LongBench/resolve/main/data.zip"
LB_ZIP_PATH = Path("outputs/benchmarks/longbench/data.zip")
LB_OUT = Path("outputs/benchmarks/longbench-hotpotqa-100.jsonl")
LB_MANIFEST = Path("outputs/benchmarks/longbench-hotpotqa-100.manifest.json")


def build_long173() -> None:
    rows = [json.loads(l) for l in HELDOUT300.read_text(encoding="utf-8").splitlines() if l.strip()]
    long_rows = [r for r in rows if sum(len(str(e.get("text", ""))) for e in (r.get("evidence") or [])) > 10000]
    assert len(long_rows) == 173, f"Expected 173 long claims, got {len(long_rows)}"
    LONG173.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in long_rows), encoding="utf-8")
    print(f"Wrote {len(long_rows)} claims to {LONG173}")


def build_longbench_hotpotqa() -> None:
    LB_ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LB_ZIP_PATH.exists():
        print(f"Downloading {LB_ZIP_URL} -> {LB_ZIP_PATH}...")
        req = urllib.request.Request(LB_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, LB_ZIP_PATH.open("wb") as out_f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out_f.write(chunk)

    tok = AutoTokenizer.from_pretrained(TARGET_SNAP, local_files_only=True)
    selected_rows = []
    tok_lens = []
    passage_counts = []
    char_lens = []

    with zipfile.ZipFile(LB_ZIP_PATH, "r") as zf:
        raw_lines = zf.read("data/hotpotqa.jsonl").decode("utf-8").splitlines()

    for idx, line in enumerate(raw_lines):
        if not line.strip():
            continue
        item = json.loads(line)
        ctx = str(item.get("context", "")).strip()
        ques = str(item.get("input", "")).strip()
        answers = list(item.get("answers") or [])
        if not ctx or not ques or not answers:
            continue

        # Split passages by 'Passage N:' markers
        parts = [p.strip() for p in re.split(r"(?:^|\n)Passage\s+\d+:\s*", ctx) if p.strip()]
        if len(parts) < 4:
            # Fallback split on double newline
            parts = [p.strip() for p in ctx.split("\n\n") if len(p.strip()) > 80]
        if len(parts) < 4:
            continue

        evidence = [{"evidence_id": f"P{i+1}", "text": p} for i, p in enumerate(parts)]
        ev_chars = sum(len(e["text"]) for e in evidence)
        if ev_chars <= 10000:
            continue

        prompt_str, phash = build_prompt(ques, evidence, mode="longbench_qa")
        ser = format_qwen_prompt(tok, prompt_str)
        n_tok = len(tok(ser, add_special_tokens=False)["input_ids"])
        if not (3000 <= n_tok <= 18000):
            continue

        selected_rows.append({
            "claim_id": f"lb-hotpot-{item.get('_id', idx)}",
            "claim": ques,
            "evidence": evidence,
            "evidence_mode": "longbench_qa",
            "prompt": prompt_str,
            "prompt_hash": phash,
            "evaluation": {
                "dataset": "LongBench/hotpotqa",
                "answers": answers,
                "prompt_tokens": n_tok,
                "evidence_chars": ev_chars,
            },
        })
        tok_lens.append(n_tok)
        passage_counts.append(len(evidence))
        char_lens.append(ev_chars)
        if len(selected_rows) >= 100:
            break

    assert len(selected_rows) == 100, f"Only found {len(selected_rows)} suitable HotpotQA examples"
    LB_OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected_rows), encoding="utf-8")
    manifest = {
        "n_examples": len(selected_rows),
        "mean_prompt_tokens": round(statistics.mean(tok_lens), 1),
        "std_prompt_tokens": round(statistics.stdev(tok_lens), 1),
        "min_prompt_tokens": min(tok_lens),
        "max_prompt_tokens": max(tok_lens),
        "mean_passages": round(statistics.mean(passage_counts), 1),
        "mean_evidence_chars": round(statistics.mean(char_lens), 1),
    }
    LB_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Built LongBench HotpotQA-100:", json.dumps(manifest))


if __name__ == "__main__":
    build_long173()
    build_longbench_hotpotqa()
