import json
import sys

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'outputs/runs/test-opt-b2-k4-lim5/round_traces.jsonl'
    rounds = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rounds.append(json.loads(line))
                
    if not rounds:
        print("No rounds found.")
        return
        
    d_lat = sum(r['draft_latency_s'] for r in rounds) / len(rounds)
    v_lat = sum(r['verification_latency_s'] for r in rounds) / len(rounds)
    c_lat = sum(r['correction_latency_s'] for r in rounds) / len(rounds)
    tot_lat = sum(r['total_round_latency_s'] for r in rounds) / len(rounds)
    gap = tot_lat - (d_lat + v_lat + c_lat)
    
    acc = sum(r.get('accepted_draft_tokens', 0) for r in rounds) / len(rounds)
    comm = sum(r.get('committed_output_tokens', 0) for r in rounds) / len(rounds)
    
    print(f"Total rounds: {len(rounds)}")
    print(f"Draft latency: {d_lat*1000:.2f} ms")
    print(f"Verify latency: {v_lat*1000:.2f} ms")
    print(f"Correction latency: {c_lat*1000:.2f} ms")
    print(f"Accounted sum: {(d_lat + v_lat + c_lat)*1000:.2f} ms")
    print(f"Total round latency: {tot_lat*1000:.2f} ms")
    print(f"Unaccounted loop overhead: {gap*1000:.2f} ms")
    print(f"Mean accepted draft tokens/round: {acc:.2f}")
    print(f"Mean committed tokens/round: {comm:.2f}")

if __name__ == '__main__':
    main()
