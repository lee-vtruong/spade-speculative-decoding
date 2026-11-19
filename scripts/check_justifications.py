import json

with open("outputs/benchmarks/retrieved-heldout300.jsonl", "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

n_with_just = sum(1 for r in rows if r.get("evaluation", {}).get("gold_justification"))
print(f"Claims with gold_justification: {n_with_just}/{len(rows)}")

lengths = [len(r["evaluation"]["gold_justification"]) for r in rows if r.get("evaluation", {}).get("gold_justification")]
print(f"Mean gold justification chars: {sum(lengths)/len(lengths):.1f}")
