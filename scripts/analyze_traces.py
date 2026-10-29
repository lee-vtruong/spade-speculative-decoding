from __future__ import annotations
import argparse,json
from pathlib import Path
from spade.analysis.metrics import acceptance_summary
p=argparse.ArgumentParser(); p.add_argument("rounds",type=Path); p.add_argument("tokens",type=Path); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
read=lambda x:[json.loads(line) for line in x.read_text(encoding="utf-8").splitlines() if line.strip()]
result=acceptance_summary(read(a.tokens),read(a.rounds)); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
