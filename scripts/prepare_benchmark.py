"""Prepare a deterministic AVeriTeC JSONL benchmark and adjacent manifest."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from spade.benchmark.averitec_loader import load_records, claim_id, claim_text
from spade.benchmark.benchmark_builder import build_examples, build_manifest, stratified_sample
from spade.benchmark.evidence_adapter import load_claim_only_knowledge_store

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("dataset",type=Path); p.add_argument("output",type=Path)
p.add_argument("--split",choices=["train","dev"],required=True)
p.add_argument("--mode",choices=["oracle_qa","retrieved"],required=True)
p.add_argument("--retrieved",type=Path,help="JSON map of claim ID to ranked passage arrays")
p.add_argument("--knowledge-store-dir",type=Path,help="Read-only per-claim JSONL store directory; only exact type=claim query rows are eligible")
p.add_argument("--size",default="3",help="3, 100, 200, or full")
p.add_argument("--top-k",type=int,default=8); p.add_argument("--seed",type=int,default=42)
p.add_argument("--tokenizer",help="Local tokenizer path/ID for chat-template input token statistics")
p.add_argument("--manifest",type=Path,help="Defaults to OUTPUT.manifest.json")
a=p.parse_args()
if a.retrieved and a.knowledge_store_dir: p.error("Use only one of --retrieved and --knowledge-store-dir")
if a.mode=="retrieved" and not (a.retrieved or a.knowledge_store_dir): p.error("retrieved mode requires --retrieved or --knowledge-store-dir")
if a.mode=="oracle_qa" and (a.retrieved or a.knowledge_store_dir): p.error("oracle_qa does not accept retrieved evidence")
if a.top_k<1: p.error("--top-k must be positive")
source_rows=load_records(a.dataset)
all_rows=[]; excluded=[]
for source_index,row in enumerate(source_rows):
    try:
        claim_id(row); claim_text(row)
        all_rows.append(row)
    except (ValueError,AttributeError) as exc:
        excluded.append({"row_index":source_index,"keys":sorted(row) if isinstance(row,dict) else [],"reason":str(exc)})
size=len(all_rows) if a.size.casefold()=="full" else int(a.size)
if size<1: p.error("--size must be positive or 'full'")
rows=stratified_sample(all_rows,size,a.seed)
retrieved=None; retrieval_audit=None
if a.knowledge_store_dir:
    retrieved,retrieval_audit=load_claim_only_knowledge_store(source_rows,a.knowledge_store_dir,a.top_k)
elif a.retrieved:
    retrieved=json.loads(a.retrieved.read_text(encoding="utf-8"))
    if not isinstance(retrieved,dict): p.error("--retrieved must contain a JSON object mapping claim IDs to passage arrays")
examples=build_examples(rows,a.mode,retrieved,a.top_k)
if a.mode=="retrieved" and not any(ex["evidence"] for ex in examples):
    p.error("No eligible retrieved passages for selected examples. Check per-claim files and exact type=claim query provenance; no oracle fallback is applied.")
token_lengths=None
if a.tokenizer:
    from transformers import AutoTokenizer
    from spade.decoding.autoregressive import format_qwen_prompt
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    token_lengths=[]
    for ex in examples:
        serialized=format_qwen_prompt(tok,ex["prompt"])
        ex["serialized_prompt_hash"]=hashlib.sha256(serialized.encode()).hexdigest()
        ex["input_tokens"]=len(tok(serialized,add_special_tokens=False)["input_ids"])
        token_lengths.append(ex["input_tokens"])
manifest=build_manifest(dataset_path=str(a.dataset.resolve()),dataset_hash=sha256(a.dataset),split=a.split,seed=a.seed,mode=a.mode,top_k=a.top_k,examples=examples,token_lengths=token_lengths,retrieval_audit=retrieval_audit,excluded_dataset_rows=excluded)
manifest_path=a.manifest or a.output.with_suffix(a.output.suffix+".manifest.json")
a.output.parent.mkdir(parents=True,exist_ok=True); manifest_path.parent.mkdir(parents=True,exist_ok=True)
if a.output.exists() or manifest_path.exists(): p.error("Output/manifest exists; refusing to overwrite. Choose new paths/run name.")
with a.output.open("x",encoding="utf-8") as f:
    for ex in examples: f.write(json.dumps(ex,ensure_ascii=False,allow_nan=False)+"\n")
with manifest_path.open("x",encoding="utf-8") as f: json.dump(manifest,f,ensure_ascii=False,indent=2,allow_nan=False); f.write("\n")
print(json.dumps({"examples":len(examples),"excluded_invalid_dataset_rows":excluded,"output":str(a.output.resolve()),"manifest":str(manifest_path.resolve()),"evidence_mode":a.mode,"split":a.split,"retrieval_audit":retrieval_audit},ensure_ascii=False,indent=2))
