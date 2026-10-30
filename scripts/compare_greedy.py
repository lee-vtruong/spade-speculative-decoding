"""Compare B0/B1 token IDs for matched greedy runs; writes no files by default."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from spade.analysis.greedy_compare import compare_greedy_runs

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("b0_run",type=Path); p.add_argument("b1_run",type=Path)
p.add_argument("--output",type=Path,help="Optional JSON report path outside source datasets")
a=p.parse_args()
report=compare_greedy_runs(a.b0_run,a.b1_run)
rendered=json.dumps(report,indent=2)
print(rendered)
if a.output:
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(rendered+"\n",encoding="utf-8")
if not report["all_match"]: raise SystemExit(1)
