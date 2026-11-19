import json

with open("outputs/benchmarks/retrieved-heldout300.jsonl", "r", encoding="utf-8") as f:
    for i in range(3):
        r = json.loads(f.readline())
        print(f"=== Row {i} ===")
        print("claim_id:", r.get("claim_id"))
        print("claim:", r.get("claim")[:80])
        print("evidence_count:", len(r.get("evidence", [])))
        print("eval:", r.get("evaluation"))
        if r.get("evidence"):
            print("evidence[0] keys:", list(r["evidence"][0].keys()))
            print("evidence[0] url/source:", r["evidence"][0].get("url", r["evidence"][0].get("source_url", "none")))
