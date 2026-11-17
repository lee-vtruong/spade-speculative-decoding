import os
import json
import pandas as pd
import numpy as np

# Load round metrics
round_df = pd.read_csv('outputs/analysis/overnight_m1_m4/round_metrics.csv')
speedup_df = pd.read_csv('outputs/analysis/overnight_m1_m4/speedup_comparison.csv')

# Clean run_id in round_df (remove -trial0)
round_df['clean_run_id'] = round_df['run_id'].str.replace(r'-trial\d+$', '', regex=True)

runs = [
    'full-b0-lim100',
    'full-b0s-lim100',
    'full-b1-k4-lim100',
    'full-b2-k2-lim100',
    'full-b2-k4-lim100',
    'full-b2-k6-lim100',
    'full-m4-k4-lim100',
    'full-m4-k6-lim100',
]

results = []

for r in runs:
    m_path = os.path.join('outputs', 'runs', r, 'metrics.json')
    c_path = os.path.join('outputs', 'runs', r, 'config_resolved.json')
    
    with open(m_path, 'r', encoding='utf-8') as f:
        m = json.load(f)
    with open(c_path, 'r', encoding='utf-8') as f:
        c = json.load(f)
        
    mode = c.get('mode', 'N/A')
    k = c.get('draft_length', 0) if mode != 'B0' and mode != 'B0-S' else 0
    
    # Filter round_df for this run
    r_df = round_df[round_df['clean_run_id'] == r]
    
    total_proposed = r_df['proposed'].sum()
    total_accepted = r_df['accepted'].sum()
    total_committed = r_df['committed'].sum()
    n_rounds = len(r_df)
    
    alpha = total_accepted / total_proposed if total_proposed > 0 else np.nan
    acc_per_rnd = total_accepted / n_rounds if n_rounds > 0 else np.nan
    comm_per_rnd = total_committed / n_rounds if n_rounds > 0 else np.nan
    
    mean_lat = m.get('mean_decode_s_per_trial', 0)
    mean_tokens = m.get('mean_output_tokens', 0)
    tps = mean_tokens / mean_lat if mean_lat > 0 else 0
    
    # Speedup vs B0 (greedy) or B0-S (stochastic)
    # B0 is greedy, B1 is greedy. B0-S is stochastic, B2/M4 are stochastic.
    ref_time_greedy = 2.21628  # from full-b0-lim100
    ref_time_stoch = 2.4578   # from full-b0s-lim100
    
    if mode in ['B0', 'B1']:
        speedup = ref_time_greedy / mean_lat if mean_lat > 0 else 1.0
    else:
        # B0-S, B2, M4
        speedup = ref_time_stoch / mean_lat if mean_lat > 0 else 1.0
        
    results.append({
        'run_id': r,
        'mode': mode,
        'draft_k': k,
        'output_tokens': mean_tokens,
        'parse_ok': m.get('json_parse_success_rate', 0),
        'citation_ok': m.get('citation_id_validity_rate', 0),
        'decode_s': mean_lat,
        'tokens_per_s': tps,
        'alpha': alpha,
        'accepted_per_rnd': acc_per_rnd,
        'committed_per_rnd': comm_per_rnd,
        'n_rounds': n_rounds,
        'total_accepted': total_accepted,
        'total_proposed': total_proposed,
        'speedup': speedup
    })

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))

res_df.to_csv('outputs/analysis/overnight_m1_m4/final_comparative_table.csv', index=False)
