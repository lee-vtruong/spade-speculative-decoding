"""Aggregate trace diagnostics and export M3 PNG figures plus supporting CSV."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from spade.analysis.metrics import acceptance_summary
from spade.analysis.plots import analyze_runs

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("run_dirs",nargs="+",type=Path,help="One or more completed outputs/runs/<run_id> directories")
p.add_argument("--output-dir",type=Path,required=True)
a=p.parse_args()
a.output_dir.mkdir(parents=True,exist_ok=True)
rounds=[]; tokens=[]
for root in a.run_dirs:
    config=json.loads((root/"config_resolved.json").read_text(encoding="utf-8")); mode=config.get("mode")
    rr=[json.loads(x) for x in (root/"round_traces.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()]
    tt=[json.loads(x) for x in (root/"token_traces.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()]
    for row in rr+tt: row.setdefault("mode",mode)
    rounds.extend(rr); tokens.extend(tt)
summary=acceptance_summary([x for x in tokens if x.get("mode")=="B2"],[x for x in rounds if x.get("mode")=="B2"])
(a.output_dir/"acceptance_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
report=analyze_runs([str(x) for x in a.run_dirs],a.output_dir)
print(json.dumps({"acceptance_summary":summary,"analysis":report},indent=2))
