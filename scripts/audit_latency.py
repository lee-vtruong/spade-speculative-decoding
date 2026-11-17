import json

def analyze_latency(run_name):
    path = f"outputs/runs/{run_name}/round_traces.jsonl"
    total_draft = 0
    total_verify = 0
    total_correct = 0
    total_round = 0
    count = 0
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            count += 1
            total_draft += r.get('draft_latency_s', 0)
            total_verify += r.get('verification_latency_s', 0)
            total_correct += r.get('correction_latency_s', 0)
            total_round += r.get('total_round_latency_s', 0)
    print(f"Run: {run_name} (Rounds: {count})")
    print(f"  Mean Draft Latency / Round: {total_draft/count*1000:.2f} ms ({total_draft/total_round*100:.1f}%)")
    print(f"  Mean Verify Latency / Round: {total_verify/count*1000:.2f} ms ({total_verify/total_round*100:.1f}%)")
    print(f"  Mean Correct Latency / Round: {total_correct/count*1000:.2f} ms ({total_correct/total_round*100:.1f}%)")
    print(f"  Mean Total Latency / Round: {total_round/count*1000:.2f} ms")
    print(f"  Total Round Time: {total_round:.2f} s")

for r in ['full-b2-k4-lim100', 'full-m4-k4-lim100', 'full-m4-dynamic-k-lim100', 'full-b0-lim100', 'full-b0s-lim100']:
    analyze_latency(r)
