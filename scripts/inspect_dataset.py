from __future__ import annotations
import argparse, json
from pathlib import Path
from spade.benchmark.averitec_loader import load_records, claim_id, claim_text
p=argparse.ArgumentParser(); p.add_argument("dataset",type=Path); a=p.parse_args()
rows=load_records(a.dataset); print(json.dumps({"count":len(rows),"keys":sorted(rows[0]) if rows else [],"first_claim_id":claim_id(rows[0]) if rows else None,"first_claim":claim_text(rows[0]) if rows else None},ensure_ascii=False,indent=2))
