"""Evaluate stored generations; expects raw_text/evidence_ids/gold_label per JSONL row."""
import argparse,json
from pathlib import Path
from spade.benchmark.evaluation import parse_generation,classification_metrics
p=argparse.ArgumentParser(); p.add_argument("predictions",type=Path); a=p.parse_args(); rows=[json.loads(x) for x in a.predictions.read_text(encoding="utf-8").splitlines() if x.strip()]
parsed=[parse_generation(x["raw_text"],set(x.get("evidence_ids",[]))) for x in rows]; valid=[(x.get("gold_label"),pr["mapped_label"]) for x,pr in zip(rows,parsed) if pr["parse_success"] and x.get("gold_label")]
print(json.dumps({"json_parse_success_rate":sum(x["parse_success"] for x in parsed)/len(parsed) if parsed else 0,"citation_id_validity_rate":sum(x["citation_ids_valid"] for x in parsed)/len(parsed) if parsed else 0,"verdict_metrics":classification_metrics([x[0] for x in valid],[x[1] for x in valid]),"factuality":"not_evaluated"},indent=2))
