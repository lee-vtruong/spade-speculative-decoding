"""Offline JSON/verdict/citation-ID evaluation. Factuality is not inferred."""
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path
from spade.benchmark.evaluation import parse_generation,classification_metrics

def mean(values): return statistics.mean(values) if values else 0.0
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("predictions",type=Path); p.add_argument("--output-dir",type=Path)
a=p.parse_args()
rows=[json.loads(x) for x in a.predictions.read_text(encoding="utf-8").splitlines() if x.strip()]
parsed=[parse_generation(x.get("raw_text",""),set(x.get("evidence_ids",[]))) for x in rows]
valid_pairs=[(str(x.get("evaluation",{}).get("gold_label")),pr["mapped_label"]) for x,pr in zip(rows,parsed) if pr["parse_success"] and x.get("evaluation",{}).get("gold_label") is not None]
summary_lengths=[pr["summary_chars"] for pr in parsed if pr["summary_chars"] is not None]
parse_ok=sum(pr["parse_success"] for pr in parsed)
citation_rows=[pr for pr in parsed if pr["parse_success"]]
metrics={"examples":len(rows),"json_parse_success_rate":parse_ok/len(rows) if rows else 0.0,
 "citation_id_validity_rate_over_parsed_outputs":sum(pr["citation_id_validity"] for pr in citation_rows)/len(citation_rows) if citation_rows else 0.0,
 "citation_ids_total":sum(pr["citation_count"] for pr in citation_rows),"citation_ids_valid":sum(pr["citation_id_valid_count"] for pr in citation_rows),
 "invalid_citation_ids":sum(len(pr["invalid_citation_ids"]) for pr in citation_rows),
 "verdict_metrics":classification_metrics([x[0] for x in valid_pairs],[x[1] for x in valid_pairs]),
 "input_tokens":{"min":min((x.get("prompt_tokens",0) for x in rows),default=0),"max":max((x.get("prompt_tokens",0) for x in rows),default=0),"mean":mean([x.get("prompt_tokens",0) for x in rows])},
 "output_tokens":{"min":min((x.get("output_tokens",0) for x in rows),default=0),"max":max((x.get("output_tokens",0) for x in rows),default=0),"mean":mean([x.get("output_tokens",0) for x in rows])},
 "summary_chars":{"min":min(summary_lengths,default=0),"max":max(summary_lengths,default=0),"mean":mean(summary_lengths)},
 "evidence_count":{"min":min((x.get("evidence_count",0) for x in rows),default=0),"max":max((x.get("evidence_count",0) for x in rows),default=0),"mean":mean([x.get("evidence_count",0) for x in rows])},
 "factuality":"not_evaluated"}
print(json.dumps(metrics,ensure_ascii=False,indent=2))
if a.output_dir:
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/"evaluation_metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (a.output_dir/"per_class_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["label","precision","recall","f1"]); writer.writeheader()
        for label,values in metrics["verdict_metrics"]["per_class"].items(): writer.writerow({"label":label,**values})
