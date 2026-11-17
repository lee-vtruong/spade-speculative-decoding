import json
import os

runs = [
    'full-b0-lim100',
    'full-b0s-lim100',
    'full-b1-k4-lim100',
    'full-b2-k2-lim100',
    'full-b2-k4-lim100',
    'full-b2-k6-lim100',
    'full-m4-k4-lim100',
    'full-m4-k6-lim100',
    'full-m4-dynamic-k-lim100'
]

print("=== EXACT ACCOUNTING AUDIT ACROSS ALL 100-CLAIM BENCHMARKS ===")

for run in runs:
    run_dir = f"outputs/runs/{run}"
    if not os.path.exists(run_dir):
        print(f"Skipping {run}: not found")
        continue
        
    metrics_file = f"{run_dir}/metrics.json"
    with open(metrics_file) as f:
        m = json.load(f)
        
    round_traces = f"{run_dir}/round_traces.jsonl"
    
    rounds_count = 0
    total_proposed = 0
    total_accepted_draft = 0
    total_additional_target = 0
    total_rejections = 0
    total_committed = 0
    full_acceptance_rounds = 0
    
    with open(round_traces) as f:
        for line in f:
            if not line.strip():
                continue
            rounds_count += 1
            r = json.loads(line)
            
            prop = r.get('proposed_token_count', 0)
            acc = r.get('accepted_draft_tokens', 0)
            extra = 1 if r.get('additional_target_token') else 0
            rej = 1 if r.get('rejection_position') is not None else 0
            comm = r.get('committed_output_tokens', 0)
            
            total_proposed += prop
            total_accepted_draft += acc
            total_additional_target += extra
            total_rejections += rej
            total_committed += comm
            if r.get('full_acceptance'):
                full_acceptance_rounds += 1

    print(f"\nConfiguration: {run}")
    print(f"  Reported Mean Output Tokens: {m.get('mean_output_tokens')}")
    print(f"  Calculated Mean Output Tokens (committed/100): {total_committed / 100:.2f}")
    print(f"  Total Target Verification Rounds: {rounds_count}")
    print(f"  Total Proposed Tokens: {total_proposed}")
    print(f"  Total Accepted Draft Tokens: {total_accepted_draft}")
    print(f"  Total Rejections (with residual replacement): {total_rejections}")
    print(f"  Total Bonus Target Tokens: {total_additional_target}")
    print(f"  Full Acceptance Rounds: {full_acceptance_rounds}")
    print(f"  Total Committed Tokens: {total_committed}")
    print(f"  Invariant Check (accepted + rej + bonus == committed): "
          f"{total_accepted_draft + total_rejections + total_additional_target} vs {total_committed} "
          f"({'MATCH' if total_accepted_draft + total_rejections + total_additional_target == total_committed else 'MISMATCH'})")
    
    if total_proposed > 0:
        print(f"  Overall Proposal Acceptance Rate (accepted / proposed): {total_accepted_draft / total_proposed * 100:.2f}%")
    if rounds_count > 0:
        print(f"  Mean Accepted Draft Tokens / Round: {total_accepted_draft / rounds_count:.2f}")
        print(f"  Mean Committed Output Tokens / Round: {total_committed / rounds_count:.2f}")
