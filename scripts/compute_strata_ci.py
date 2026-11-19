import json
import statistics
import random

spade_trials = [json.loads(l) for l in open('/home/stackops/whale/spade-speculative-decoding/outputs/runs/b1-4-heldout300-attn-sent-b10k/trial_metrics.jsonl')]
full_trials = [json.loads(l) for l in open('/home/stackops/whale/spade-speculative-decoding/outputs/runs/b1-4-heldout300-baseline-full/trial_metrics.jsonl')]
spade_map = {x['claim_id']: x for x in spade_trials}
full_map = {x['claim_id']: x for x in full_trials}
common = sorted(set(spade_map.keys()) & set(full_map.keys()))

groups = {
    'Bypass Regime (Evidence <= 10k chars)': [c for c in common if spade_map[c].get('full_evidence_chars', 0) <= 10000],
    'Selective Regime (Evidence > 10k chars)': [c for c in common if spade_map[c].get('full_evidence_chars', 0) > 10000],
    'Dense Multi-Doc (Evidence >= 20k chars)': [c for c in common if spade_map[c].get('full_evidence_chars', 0) >= 20000],
    'Full Test Set': common
}

random.seed(42)
B = 10000

for gname, cids in groups.items():
    s_thpts = [spade_map[c]['output_tokens'] / (spade_map[c]['prompt_prefill_s'] + spade_map[c].get('draft_prefill_s',0) + spade_map[c]['decode_s'] + spade_map[c].get('context_selection_s',0)) for c in cids]
    f_thpts = [full_map[c]['output_tokens'] / (full_map[c]['prompt_prefill_s'] + full_map[c].get('draft_prefill_s',0) + full_map[c]['decode_s']) for c in cids]
    diffs = [s - f for s, f in zip(s_thpts, f_thpts)]

    mean_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0.0

    boot_diffs = []
    n = len(diffs)
    for _ in range(B):
        sample = random.choices(diffs, k=n)
        boot_diffs.append(statistics.mean(sample))
    boot_diffs.sort()
    ci_low = boot_diffs[int(0.025 * B)]
    ci_high = boot_diffs[int(0.975 * B)]
    p_val = sum(1 for d in boot_diffs if d <= 0) / B

    print(f"{gname}:")
    print(f"  Mean diff: {mean_diff:+.2f} tok/s (std: {std_diff:.2f}) | 95% CI: [{ci_low:+.2f}, {ci_high:+.2f}] | p={p_val:.4f}")
