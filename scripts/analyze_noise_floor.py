import glob
import json

runs = sorted(glob.glob('/home/stackops/whale/spade-speculative-decoding/outputs/runs/noise_floor_study/*/trial_metrics.jsonl'))
for r in runs:
    lines = [json.loads(l) for l in open(r)]
    tot_tok = sum(x['output_tokens'] for x in lines)
    tot_time = sum(x['prompt_prefill_s'] + x.get('draft_prefill_s',0) + x['decode_s'] for x in lines)
    thpt = tot_tok / tot_time
    run_name = r.split('/')[-2]
    print(f"{run_name:<45}: {thpt:.2f} tok/s (N={len(lines)})")
