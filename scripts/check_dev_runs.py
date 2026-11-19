import json
import glob

runs = glob.glob('/home/stackops/whale/spade-speculative-decoding/outputs/runs/decisive_experiments/*/trial_metrics.jsonl')
for r in sorted(runs):
    lines = [json.loads(l) for l in open(r)]
    print(r.split('/')[-2], len(lines), 'cids:', [x['claim_id'] for x in lines[:5]])
