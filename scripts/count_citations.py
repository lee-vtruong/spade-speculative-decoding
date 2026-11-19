import json
from pathlib import Path

run_dir = Path("outputs/runs/b1-4-heldout300-attn-sent-b10k")
n_with_citations = 0
total_citations = 0
with (run_dir / "predictions.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        cites = r.get("parsed", {}).get("citations", []) if r.get("parsed") else []
        if cites:
            n_with_citations += 1
            total_citations += len(cites)

print(f"Claims with citations: {n_with_citations}/300, total citations: {total_citations}")
