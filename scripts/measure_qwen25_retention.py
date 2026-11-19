import sys
from pathlib import Path
sys.path.insert(0, ".")
sys.path.insert(0, "src")
import json
from scripts.measure_content_retention import analyze_retention

method_run_ids = {
    "Baseline Full": "p2-qwen25-heldout300-full",
    "Selective Top-2": "p2-qwen25-heldout300-topk2",
    "BM25 Sent-B10k": "p2-qwen25-heldout300-bm25-sent-b10k",
    "SPADE": "p2-qwen25-heldout300-spade",
}

bench_path = Path("outputs/benchmarks/retrieved-heldout300.jsonl")
runs_dir = Path("outputs/runs")

retention_report = analyze_retention(bench_path, runs_dir, method_run_ids)
out_path = Path("outputs/analysis/p2_qwen25_heldout300/content_retention_report.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(retention_report, indent=2) + "\n", encoding="utf-8")
print(f"Exported {out_path}")
print(json.dumps(retention_report.get("overall_retention", {}), indent=2))
