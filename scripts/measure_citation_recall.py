import json
from pathlib import Path

runs_dir = Path("outputs/runs")
methods = {
    "Baseline Full": "b1-4-heldout300-baseline-full",
    "Selective Top-2": "b1-4-heldout300-selective-topk2",
    "BM25 Sent-B8k": "b1-4-heldout300-bm25-sent-b8k",
    "BM25 Sent-B10k": "b1-4-heldout300-bm25-sent-b10k",
    "SPADE": "b1-4-heldout300-attn-sent-b10k",
    "Random Sent-B10k": "b1-4-heldout300-random-sent-b10k",
    "SPADE NoCoverage": "b1-4-heldout300-attn-sent-b10k-nocoverage",
}

print("=== CITATION PASSAGE RECALL ANALYSIS ===")
for m_name, r_id in methods.items():
    p_file = runs_dir / r_id / "predictions.jsonl"
    if not p_file.exists():
        continue
    total_citations = 0
    retained_citations = 0
    claims_total = 0
    claims_full_recall = 0
    with p_file.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            claims_total += 1
            cites = r.get("parsed", {}).get("citations", []) if r.get("parsed") else []
            sel_ids = set(r.get("draft_selected_evidence_ids") or [])
            if not sel_ids and m_name == "Baseline Full":
                sel_ids = set(r.get("evidence_ids") or [])
            
            c_retained = 0
            for c in cites:
                total_citations += 1
                if c in sel_ids:
                    retained_citations += 1
                    c_retained += 1
            if cites and c_retained == len(cites):
                claims_full_recall += 1

    rec = (retained_citations / total_citations * 100) if total_citations > 0 else 100.0
    print(f"{m_name:20s} | Total Citations: {total_citations:3d} | Retained: {retained_citations:3d} | Recall: {rec:6.2f}% | Claims 100% Retained: {claims_full_recall}/{claims_total}")
