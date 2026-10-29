from __future__ import annotations
import argparse, json
from pathlib import Path
from spade.benchmark.averitec_loader import load_records
from spade.benchmark.benchmark_builder import build_examples, stratified_sample
p=argparse.ArgumentParser(); p.add_argument("dataset",type=Path); p.add_argument("output",type=Path); p.add_argument("--mode",choices=["oracle_qa","retrieved"],required=True); p.add_argument("--retrieved",type=Path); p.add_argument("--size",type=int,default=3); p.add_argument("--top-k",type=int,default=8); p.add_argument("--seed",type=int,default=42); a=p.parse_args()
rows=stratified_sample(load_records(a.dataset),a.size,a.seed)
retrieved=json.loads(a.retrieved.read_text(encoding="utf-8")) if a.retrieved else None
if a.mode=="retrieved" and not isinstance(retrieved,dict): raise SystemExit("retrieved mode requires --retrieved JSON mapping claim IDs to ranked passage arrays")
examples=build_examples(rows,a.mode,retrieved,a.top_k); a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open("w",encoding="utf-8") as f:
    for x in examples: f.write(json.dumps(x,ensure_ascii=False)+"\n")
print(f"Wrote {len(examples)} examples to {a.output}; evidence_mode={a.mode}")
