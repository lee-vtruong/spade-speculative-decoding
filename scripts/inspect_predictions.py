import json
from pathlib import Path

run_dir = Path("outputs/runs/b1-4-heldout300-attn-sent-b10k")
with (run_dir / "predictions.jsonl").open("r", encoding="utf-8") as f:
    for i in range(2):
        r = json.loads(f.readline())
        print(f"=== Example {i} ===")
        print("keys in predictions.jsonl:", list(r.keys()))
        print("claim_id:", r.get("claim_id"))
        print("draft_context_policy:", r.get("draft_context_policy"))
        print("draft_selected_evidence_ids:", r.get("draft_selected_evidence_ids"))
        print("evidence_ids:", r.get("evidence_ids"))
        print("parsed:", r.get("parsed"))
        print("citations:", r.get("parsed", {}).get("citations") if r.get("parsed") else None)
        print("raw_text:", r.get("raw_text")[:120])
