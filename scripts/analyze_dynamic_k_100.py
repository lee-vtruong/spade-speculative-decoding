import json
import os
from collections import Counter
import pandas as pd

trace_path = os.path.join('outputs', 'runs', 'full-m4-dynamic-k-lim100', 'round_traces.jsonl')

k_counts = Counter()
reasons = Counter()
total_rounds = 0
total_proposed = 0
total_accepted = 0
total_committed = 0

with open(trace_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        k = r.get('draft_length_requested')
        reason = r.get('dynamic_k_reason')
        prop = r.get('proposed_token_count', 0)
        acc = r.get('accepted_draft_tokens', 0)
        comm = r.get('committed_output_tokens', 0)
        
        k_counts[k] += 1
        total_rounds += 1
        total_proposed += prop
        total_accepted += acc
        total_committed += comm
        
        if reason:
            reasons[reason.split()[0]] += 1

alpha = total_accepted / total_proposed if total_proposed > 0 else 0
acc_per_rnd = total_accepted / total_rounds if total_rounds > 0 else 0
comm_per_rnd = total_committed / total_rounds if total_rounds > 0 else 0

print("=" * 70)
print("SPADE M4 DYNAMIC K(t) BENCHMARK RESULTS (100 CLAIMS)")
print("=" * 70)
print(f"Total Speculative Rounds:        {total_rounds}")
print(f"Total Proposed Tokens:           {total_proposed}")
print(f"Total Accepted Tokens:           {total_accepted}")
print(f"Total Committed Tokens:          {total_committed}")
print(f"Token Acceptance Rate (Alpha):   {alpha * 100:.2f}%")
print(f"Mean Accepted Tokens / Round:    {acc_per_rnd:.3f}")
print(f"Mean Committed Tokens / Round:   {comm_per_rnd:.3f}")

print("\nDynamic K Distribution:")
for k in [2, 4, 6]:
    cnt = k_counts.get(k, 0)
    pct = cnt / total_rounds * 100 if total_rounds > 0 else 0
    print(f"  K = {k}: {cnt:4d} rounds ({pct:5.1f}%)")

print("\nDecision Reasons:")
for r, cnt in reasons.most_common():
    pct = cnt / total_rounds * 100 if total_rounds > 0 else 0
    print(f"  {r:30s}: {cnt:4d} ({pct:5.1f}%)")
print("=" * 70)
