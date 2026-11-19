"""Measure evidence content recall across draft context policies on retrieved-heldout300."""
import json
import re
from collections import Counter
from pathlib import Path

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "if", "in", "into", "is", "it",
    "its", "of", "on", "or", "she", "that", "the", "their", "them", "there", "these", "they",
    "this", "to", "was", "we", "were", "what", "when", "where", "which", "who", "will", "with",
    "would", "you", "your",
})

def content_terms(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 2 and t.lower() not in _STOPWORDS}

def main():
    bench_file = Path("outputs/benchmarks/retrieved-heldout300.jsonl")
    if not bench_file.exists():
        print(f"Benchmark file not found: {bench_file}")
        return

    rows = [json.loads(l) for l in bench_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(rows)} claims from {bench_file.name}")

    runs = {
        "Baseline Full": "b1-4-heldout300-baseline-full",
        "BM25 Top-2": "b1-4-heldout300-selective-topk2",
        "BM25 Sent-B8k": "b1-4-heldout300-bm25-sent-b8k",
        "Target-Attn Sent-B10k (SPADE)": "b1-4-heldout300-attn-sent-b10k",
    }

    # Also check if bm25_sent_b10k exists
    p_b10k = Path("outputs/runs/b1-4-heldout300-bm25-sent-b10k/predictions.jsonl")
    if p_b10k.exists() and len(p_b10k.read_text(encoding="utf-8").splitlines()) >= 300:
        runs["BM25 Sent-B10k"] = "b1-4-heldout300-bm25-sent-b10k"

    for arm_label, run_id in runs.items():
        pfile = Path(f"outputs/runs/{run_id}/predictions.jsonl")
        if not pfile.exists():
            print(f"Skipping {arm_label} (not found: {pfile})")
            continue
        preds = [json.loads(l) for l in pfile.read_text(encoding="utf-8").splitlines() if l.strip()]
        p_by_cid = {p["claim_id"]: p for p in preds}

        recalls = []
        raw_char_retentions = []
        for r in rows:
            cid = r["claim_id"]
            if cid not in p_by_cid:
                continue
            pred = p_by_cid[cid]
            gold_just = r.get("evaluation", {}).get("gold_justification", "")
            gold_terms = content_terms(gold_just)
            if not gold_terms:
                continue

            # What evidence was in full prompt?
            full_evidence_text = " ".join(str(e.get("text", "")) for e in r.get("evidence", []))
            full_ev_terms = content_terms(full_evidence_text)
            target_relevant_terms = gold_terms.intersection(full_ev_terms)
            if not target_relevant_terms:
                continue

            # What evidence was retained in draft prompt?
            # Recover draft evidence text from prompt or draft_selected
            # We can check prompt tokens or parse text from prompt
            d_prompt = pred.get("prompt") or ""
            # If prompt not directly in pred, check raw_text or tokens
            # In run_baselines, dctx has draft_evidence_chars
            # Or inspect prompt from benchmark if available
            d_terms = content_terms(d_prompt)
            # If draft prompt is not saved in pred row, fallback:
            hit = len(target_relevant_terms.intersection(d_terms))
            recall = hit / len(target_relevant_terms)
            recalls.append(recall)

        if recalls:
            print(f"{arm_label}: Mean Gold Justification Content Term Recall = {sum(recalls)/len(recalls)*100.0:.2f}% (N={len(recalls)})")
        else:
            print(f"{arm_label}: No recall computed (pred format check needed)")

if __name__ == "__main__":
    main()
