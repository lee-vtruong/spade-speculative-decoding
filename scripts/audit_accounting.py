import json
import glob
import sys

def audit_run(run_dir):
    summary_path = f"{run_dir}/run_summary.json"
    trace_path = f"{run_dir}/token_traces.jsonl"
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
        
    print(f"=== AUDIT: {run_dir} ===")
    print(f"Total claims: {summary.get('total_claims')}")
    print(f"Mean output tokens: {summary.get('mean_output_tokens')}")
    print(f"Mean target rounds: {summary.get('mean_target_rounds')}")
    print(f"Total target rounds: {summary.get('total_target_rounds')}")
    print(f"Mean accepted tokens per round: {summary.get('mean_accepted_tokens_per_round')}")
    print(f"Acceptance rate: {summary.get('acceptance_rate')}")
    
    # Audit token_traces.jsonl
    status_counts = {}
    total_rows = 0
    round_ids = set()
    claim_ids = set()
    
    with open(trace_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            total_rows += 1
            rec = json.loads(line)
            st = rec.get('status')
            status_counts[st] = status_counts.get(st, 0) + 1
            round_ids.add((rec.get('claim_id'), rec.get('round_id')))
            claim_ids.add(rec.get('claim_id'))
            
    print(f"Total trace rows: {total_rows}")
    print(f"Unique (claim, round) pairs: {len(round_ids)}")
    print("Status breakdown:")
    for st, count in sorted(status_counts.items()):
        print(f"  {st}: {count}")
    
    # Check output token count:
    # Committed tokens = accepted + residual_replacement + extra_target_token (or target_replacement)
    committed = (
        status_counts.get('accepted', 0) + 
        status_counts.get('residual_replacement', 0) + 
        status_counts.get('target_replacement', 0) + 
        status_counts.get('extra_target_token', 0)
    )
    print(f"Total committed tokens in traces: {committed}")
    print(f"Committed / total claims: {committed / len(claim_ids):.2f}")
    print()

if __name__ == '__main__':
    for r in ['full-b2-k4-lim100', 'full-m4-k4-lim100', 'full-m4-dyn-lim100', 'full-b0s-lim100']:
        audit_run(f"outputs/runs/{r}")
