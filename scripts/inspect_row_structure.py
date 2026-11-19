import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/benchmarks/retrieved-heldout300.jsonl"
with open(path, "r", encoding="utf-8") as f:
    row = json.loads(f.readline())
    print("Evaluation field:", row.get("evaluation"))
    print("Num evidence passages:", len(row.get("evidence", [])))
