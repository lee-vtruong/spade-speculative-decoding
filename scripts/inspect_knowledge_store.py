"""Read-only audit of the safe, exact-claim-query subset of an AVeriTeC store."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from spade.benchmark.averitec_loader import load_records
from spade.benchmark.evidence_adapter import load_claim_only_knowledge_store

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("dataset",type=Path); p.add_argument("store_dir",type=Path)
p.add_argument("--limit",type=int,default=0,help="Audit first N dataset rows; 0 audits all")
p.add_argument("--top-k",type=int,default=8)
a=p.parse_args(); rows=load_records(a.dataset)
if a.limit>0: rows=rows[:a.limit]
evidence,audit=load_claim_only_knowledge_store(rows,a.store_dir,a.top_k)
audit["eligible_claims_with_passages"]=sum(bool(v) for v in evidence.values())
audit["eligible_passage_count"]=sum(len(v) for v in evidence.values())
audit["dataset_examples_audited"]=len(rows)
print(json.dumps(audit,ensure_ascii=False,indent=2))
