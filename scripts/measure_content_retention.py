#!/usr/bin/env python3
"""Measure evidence content retention and stratified acceptance breakdown across methods on heldout300."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

_WORD_RE = re.compile(r"\b[a-zA-Z0-9_]{3,}\b")

# Standard English stopwords to focus on content words
_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "for", "are", "was", "were", "been", "have", "has", "had",
    "will", "would", "shall", "should", "can", "could", "about", "which", "there", "their", "they", "them",
    "what", "when", "where", "who", "whom", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "than", "too", "very", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once", "here", "why", "claim", "evidence",
    "supported", "refuted", "according", "statement", "shows", "stated", "reported", "article",
}


def _extract_content_tokens(text: str) -> set[str]:
    words = {w.lower() for w in _WORD_RE.findall(text)}
    return words - _STOPWORDS


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze_retention(
    benchmark_path: Path,
    runs_dir: Path,
    method_run_ids: dict[str, str],
) -> dict[str, Any]:
    bench_rows = _load_jsonl(benchmark_path)
    bench_by_id = {str(r["claim_id"]): r for r in bench_rows}

    results: dict[str, Any] = {}

    for method_name, run_id in method_run_ids.items():
        run_path = runs_dir / run_id
        pred_path = run_path / "predictions.jsonl"
        trial_path = run_path / "trial_metrics.jsonl"
        round_path = run_path / "round_traces.jsonl"

        if not pred_path.exists():
            print(f"Skipping {method_name} ({run_id}): predictions.jsonl not found")
            continue

        pred_rows = _load_jsonl(pred_path)
        trial_rows = _load_jsonl(trial_path) if trial_path.exists() else []
        trial_by_claim = {str(t["claim_id"]): t for t in trial_rows}

        # Acceptance by claim_id
        acc_by_claim: dict[str, float] = {}
        if round_path.exists():
            round_rows = _load_jsonl(round_path)
            rounds_by_claim: dict[str, list[dict]] = {}
            for rd in round_rows:
                cid = str(rd.get("claim_id", ""))
                rounds_by_claim.setdefault(cid, []).append(rd)
            for cid, rds in rounds_by_claim.items():
                acc = sum(r.get("accepted_draft_tokens", 0) for r in rds)
                drf = sum(r.get("proposed_token_count", 0) for r in rds)
                acc_by_claim[cid] = (acc / drf) if drf > 0 else 0.0

        # Metrics accumulator
        gold_retention_ratios: list[float] = []
        passage_coverage_ratios: list[float] = []
        target_cited_total = 0
        target_cited_retained = 0
        compression_ratios: list[float] = []

        per_claim_records: list[dict[str, Any]] = []

        for p in pred_rows:
            cid = str(p["claim_id"])
            brow = bench_by_id.get(cid, {})
            ev_list = brow.get("evidence", [])
            total_ev_count = len(ev_list)

            # Selected passages
            sel_ids = set(p.get("draft_selected_evidence_ids", []))
            if not sel_ids and method_name == "Baseline Full":
                sel_ids = {str(e.get("evidence_id")) for e in ev_list if e.get("evidence_id")}

            # Passage coverage
            if total_ev_count > 0:
                coverage = len(sel_ids) / total_ev_count
                passage_coverage_ratios.append(coverage)
            else:
                coverage = 1.0

            # Target cited passages
            parsed = p.get("parsed") or {}
            citations = [str(c) for c in parsed.get("citations", [])]
            for cite in citations:
                target_cited_total += 1
                if cite in sel_ids:
                    target_cited_retained += 1

            # Exact chars & compression from trial_metrics
            t_row = trial_by_claim.get(cid, {})
            full_chars = int(t_row.get("full_evidence_chars", sum(len(str(e.get("text", ""))) for e in ev_list)))
            if method_name == "Baseline Full":
                draft_chars = full_chars
            else:
                draft_chars = int(t_row.get("draft_evidence_chars", full_chars))
            comp_ratio = (draft_chars / full_chars) if full_chars > 0 else 1.0
            compression_ratios.append(comp_ratio)
            compression_ratios.append(comp_ratio)

            per_claim_records.append({
                "claim_id": cid,
                "evidence_count": total_ev_count,
                "full_chars": full_chars,
                "draft_chars": draft_chars,
                "compression_ratio": comp_ratio,
                "passage_coverage": coverage,
                "acceptance_rate": acc_by_claim.get(cid, 0.0),
            })

        # Summary aggregates
        target_cited_recall = (target_cited_retained / target_cited_total) if target_cited_total > 0 else 1.0
        mean_gold_retention = statistics.mean(gold_retention_ratios) if gold_retention_ratios else 1.0
        mean_coverage = statistics.mean(passage_coverage_ratios) if passage_coverage_ratios else 1.0
        mean_compression = statistics.mean(compression_ratios) if compression_ratios else 1.0

        results[method_name] = {
            "mean_passage_coverage_pct": round(mean_coverage * 100, 2),
            "target_cited_passages_total": target_cited_total,
            "target_cited_passages_retained": target_cited_retained,
            "target_cited_recall_pct": round(target_cited_recall * 100, 2),
            "mean_gold_justification_recall_pct": round(mean_gold_retention * 100, 2),
            "mean_evidence_compression_pct": round(mean_compression * 100, 2),
            "claims_analyzed": len(per_claim_records),
            "per_claim": per_claim_records,
        }

    # Stratified breakdown across bins (Passage count & context length)
    stratified: dict[str, Any] = {}
    passage_bins = [
        ("m <= 2", lambda r: r["evidence_count"] <= 2),
        ("3 <= m <= 5", lambda r: 3 <= r["evidence_count"] <= 5),
        ("6 <= m <= 10", lambda r: 6 <= r["evidence_count"] <= 10),
        ("m > 10", lambda r: r["evidence_count"] > 10),
    ]

    length_bins = [
        ("L < 5k", lambda r: r["full_chars"] < 5000),
        ("5k <= L < 10k", lambda r: 5000 <= r["full_chars"] < 10000),
        ("10k <= L < 20k", lambda r: 10000 <= r["full_chars"] < 20000),
        ("L >= 20k", lambda r: r["full_chars"] >= 20000),
    ]

    for bin_group_name, bin_list in [("by_passage_count", passage_bins), ("by_evidence_chars", length_bins)]:
        stratified[bin_group_name] = {}
        for bin_label, pred_fn in bin_list:
            bin_data: dict[str, Any] = {}
            for method_name, mdata in results.items():
                matching = [c for c in mdata["per_claim"] if pred_fn(c)]
                n_claims = len(matching)
                if n_claims == 0:
                    continue
                mean_acc = statistics.mean(c["acceptance_rate"] for c in matching)
                mean_comp = statistics.mean(c["compression_ratio"] for c in matching)
                bin_data[method_name] = {
                    "count": n_claims,
                    "mean_acceptance_pct": round(mean_acc * 100, 2),
                    "mean_compression_pct": round(mean_comp * 100, 2),
                }
            stratified[bin_group_name][bin_label] = bin_data

    # Remove per-claim lists from final JSON to keep it compact
    clean_results = {k: {sub_k: sub_v for sub_k, sub_v in v.items() if sub_k != "per_claim"} for k, v in results.items()}

    out_report = {
        "benchmark": str(benchmark_path),
        "overall_retention": clean_results,
        "stratified_breakdown": stratified,
    }
    return out_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=Path, default=Path("outputs/benchmarks/retrieved-heldout300.jsonl"))
    p.add_argument("--runs-dir", type=Path, default=Path("outputs/runs"))
    p.add_argument("--output", type=Path, default=Path("outputs/analysis/b1_4_heldout300/content_retention_report.json"))
    a = p.parse_args()

    methods = {
        "Baseline Full": "b1-4-heldout300-baseline-full",
        "Selective Top-2": "b1-4-heldout300-selective-topk2",
        "BM25 Sent-B8k": "b1-4-heldout300-bm25-sent-b8k",
        "BM25 Sent-B10k": "b1-4-heldout300-bm25-sent-b10k",
        "SPADE": "b1-4-heldout300-attn-sent-b10k",
        "Random Sent-B10k": "b1-4-heldout300-random-sent-b10k",
        "SPADE NoCoverage": "b1-4-heldout300-attn-sent-b10k-nocoverage",
    }

    report = analyze_retention(a.benchmark, a.runs_dir, methods)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Content retention report saved to {a.output}")
    print("\n=== OVERALL EVIDENCE RETENTION SUMMARY ===")
    for m, d in report["overall_retention"].items():
        print(f"{m:20s} | Cov: {d['mean_passage_coverage_pct']}% | Gold Recall: {d['mean_gold_justification_recall_pct']}% | Comp: {d['mean_evidence_compression_pct']}%")


if __name__ == "__main__":
    main()
