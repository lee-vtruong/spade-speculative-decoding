import json

with open('outputs/analysis/p2_qwen25_heldout300/content_retention_report.json', 'r') as f:
    data = json.load(f)

print('--- Stratified by passage count ---')
for k, v in data.get('stratified_breakdown', {}).get('by_passage_count', {}).items():
    print(f"\n[{k}]")
    for m, vals in v.items():
        print(f"  {m:18s}: acc={vals['mean_acceptance_pct']:.2f}%, comp={vals['mean_compression_pct']:.2f}%")

print('\n--- Stratified by context length ---')
for k, v in data.get('stratified_breakdown', {}).get('by_context_length', {}).items():
    print(f"\n[{k}]")
    for m, vals in v.items():
        print(f"  {m:18s}: acc={vals['mean_acceptance_pct']:.2f}%, comp={vals['mean_compression_pct']:.2f}%")
