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
]

header = f"{'Run ID':18s} | {'Mode':5s} | {'K':2s} | {'Alpha':7s} | {'Acc/Rnd':7s} | {'Tokens/s':8s} | {'WallTime':8s} | {'Tokens':6s}"
print(header)
print("-" * len(header))

for r_name in runs:
    p = os.path.join('outputs', 'runs', r_name, 'metrics.json')
    if not os.path.exists(p):
        print(f"Missing: {p}")
        continue
    with open(p, 'r', encoding='utf-8') as f:
        d = json.load(f)
    mode = d.get('mode', 'N/A')
    k = str(d.get('draft_length', '-'))
    alpha = d.get('observed_acceptance_rate_token_weighted')
    alpha_str = f"{alpha:.4f}" if alpha is not None else "-"
    acc_rnd = d.get('mean_accepted_tokens_per_round')
    acc_rnd_str = f"{acc_rnd:.2f}" if acc_rnd is not None else "-"
    tps = d.get('tokens_per_sec_mean')
    tps_str = f"{tps:.2f}" if tps is not None else "-"
    wall = d.get('total_wall_clock_time')
    wall_str = f"{wall:.1f}s" if wall is not None else "-"
    tot_tok = d.get('total_output_tokens', 0)
    print(f"{r_name:18s} | {mode:5s} | {k:2s} | {alpha_str:7s} | {acc_rnd_str:7s} | {tps_str:8s} | {wall_str:8s} | {tot_tok:6d}")
