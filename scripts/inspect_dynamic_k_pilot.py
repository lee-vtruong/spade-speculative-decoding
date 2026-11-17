import json
from collections import Counter

k_counts = Counter()
reasons = Counter()
total_rounds = 0

with open("outputs/runs/test-dynamic-k-m4-lim5/round_traces.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        k = r.get("draft_length_requested")
        reason = r.get("dynamic_k_reason")
        k_counts[k] += 1
        total_rounds += 1
        if reason:
            reasons[reason.split()[0]] += 1

print("=" * 60)
print(f"Total Speculative Rounds Analyzed: {total_rounds}")
print("Dynamic K Distribution:")
for k, cnt in sorted(k_counts.items(), key=lambda x: str(x[0])):
    pct = cnt / total_rounds * 100
    print(f"  K = {k}: {cnt:3d} rounds ({pct:5.1f}%)")

print("\nDecision Reasons:")
for r, cnt in reasons.most_common():
    pct = cnt / total_rounds * 100
    print(f"  {r:30s}: {cnt:3d} ({pct:5.1f}%)")
print("=" * 60)
