import os
import json
import pandas as pd
import numpy as np

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

results = []

for r in runs:
    m_path = os.path.join('outputs', 'runs', r, 'metrics.json')
    c_path = os.path.join('outputs', 'runs', r, 'config_resolved.json')
    t_path = os.path.join('outputs', 'runs', r, 'round_traces.jsonl')
    
    with open(m_path, 'r', encoding='utf-8') as f:
        m = json.load(f)
    with open(c_path, 'r', encoding='utf-8') as f:
        c = json.load(f)
        
    mode = c.get('mode', 'N/A')
    is_dyn = c.get('dynamic_k', False)
    k_label = "Dyn {2,4,6}" if is_dyn else str(c.get('draft_length', '-'))
    if mode in ['B0', 'B0-S']:
        k_label = "-"
        
    total_proposed = 0
    total_accepted = 0
    total_committed = 0
    n_rounds = 0
    
    if os.path.exists(t_path):
        with open(t_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                total_proposed += row.get('proposed_token_count', 0)
                total_accepted += row.get('accepted_draft_tokens', 0)
                total_committed += row.get('committed_output_tokens', 0)
                n_rounds += 1
    else:
        # read from overnight round_metrics.csv if available
        rm_path = 'outputs/analysis/overnight_m1_m4/round_metrics.csv'
        if os.path.exists(rm_path):
            rdf = pd.read_csv(rm_path)
            rdf['clean_id'] = rdf['run_id'].str.replace(r'-trial\d+$', '', regex=True)
            sub = rdf[rdf['clean_id'] == r]
            total_proposed = sub['proposed'].sum()
            total_accepted = sub['accepted'].sum()
            total_committed = sub['committed'].sum()
            n_rounds = len(sub)
            
    alpha = total_accepted / total_proposed if total_proposed > 0 else np.nan
    acc_per_rnd = total_accepted / n_rounds if n_rounds > 0 else np.nan
    comm_per_rnd = total_committed / n_rounds if n_rounds > 0 else np.nan
    
    mean_lat = m.get('mean_decode_s_per_trial', 0)
    mean_tokens = m.get('mean_output_tokens', 0)
    tps = mean_tokens / mean_lat if mean_lat > 0 else 0
    
    results.append({
        'run_id': r,
        'mode': mode,
        'draft_k': k_label,
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
    })

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
res_df.to_csv('outputs/analysis/overnight_m1_m4/final_comparative_table_with_dynamic.csv', index=False)
