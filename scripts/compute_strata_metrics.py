import json
import statistics

spade_trials = [json.loads(l) for l in open('/home/stackops/whale/spade-speculative-decoding/outputs/runs/b1-4-heldout300-attn-sent-b10k/trial_metrics.jsonl')]
full_trials = [json.loads(l) for l in open('/home/stackops/whale/spade-speculative-decoding/outputs/runs/b1-4-heldout300-baseline-full/trial_metrics.jsonl')]
spade_map = {x['claim_id']: x for x in spade_trials}
full_map = {x['claim_id']: x for x in full_trials}
common = sorted(set(spade_map.keys()) & set(full_map.keys()))

spade_rounds = [json.loads(l) for l in open('/home/stackops/whale/spade-speculative-decoding/outputs/runs/b1-4-heldout300-attn-sent-b10k/round_traces.jsonl')]
full_rounds = [json.loads(l) for l in open('/home/stackops/whale/spade-speculative-decoding/outputs/runs/b1-4-heldout300-baseline-full/round_traces.jsonl')]

spade_claim_rounds = {}
for r in spade_rounds:
    spade_claim_rounds.setdefault(r['claim_id'], []).append(r)
full_claim_rounds = {}
for r in full_rounds:
    full_claim_rounds.setdefault(r['claim_id'], []).append(r)

groups = {
    'Bypass Regime (Evidence <= 10k chars)': [c for c in common if spade_map[c].get('full_evidence_chars', 0) <= 10000],
    'Selective Regime (Evidence > 10k chars)': [c for c in common if spade_map[c].get('full_evidence_chars', 0) > 10000],
    'Dense Multi-Doc (Evidence >= 20k chars)': [c for c in common if spade_map[c].get('full_evidence_chars', 0) >= 20000],
    'Full Test Set': common
}

print(f"{'Group':<35} | {'N':<3} | {'Agg Full':<8} | {'Agg SPADE':<9} | {'Agg Gain':<8} | {'Mean Full':<9} | {'Mean SPADE':<10} | {'Mean Gain':<9} | {'alpha_eval':<16} | {'alpha_prop':<16}")
print("-" * 155)
for gname, cids in groups.items():
    s_tok = sum(spade_map[c]['output_tokens'] for c in cids)
    f_tok = sum(full_map[c]['output_tokens'] for c in cids)
    s_time = sum(spade_map[c]['prompt_prefill_s'] + spade_map[c].get('draft_prefill_s',0) + spade_map[c]['decode_s'] + spade_map[c].get('context_selection_s',0) for c in cids)
    f_time = sum(full_map[c]['prompt_prefill_s'] + full_map[c].get('draft_prefill_s',0) + full_map[c]['decode_s'] for c in cids)
    agg_s = s_tok / s_time
    agg_f = f_tok / f_time
    agg_gain = (agg_s - agg_f) / agg_f * 100.0

    s_means = [spade_map[c]['output_tokens'] / (spade_map[c]['prompt_prefill_s'] + spade_map[c].get('draft_prefill_s',0) + spade_map[c]['decode_s'] + spade_map[c].get('context_selection_s',0)) for c in cids]
    f_means = [full_map[c]['output_tokens'] / (full_map[c]['prompt_prefill_s'] + full_map[c].get('draft_prefill_s',0) + full_map[c]['decode_s']) for c in cids]
    mean_s = sum(s_means) / len(s_means)
    mean_f = sum(f_means) / len(f_means)
    mean_gain = (mean_s - mean_f) / mean_f * 100.0

    s_acc = sum(r['accepted_draft_tokens'] for c in cids for r in spade_claim_rounds.get(c, []))
    s_prop = sum(r['proposed_token_count'] for c in cids for r in spade_claim_rounds.get(c, []))
    s_eval = sum(r['accepted_draft_tokens'] + (1 if r.get('rejection_position') is not None else 0) for c in cids for r in spade_claim_rounds.get(c, []))
    s_alpha_eval = (s_acc / s_eval * 100.0) if s_eval else 0.0
    s_alpha_prop = (s_acc / s_prop * 100.0) if s_prop else 0.0

    f_acc = sum(r['accepted_draft_tokens'] for c in cids for r in full_claim_rounds.get(c, []))
    f_prop = sum(r['proposed_token_count'] for c in cids for r in full_claim_rounds.get(c, []))
    f_eval = sum(r['accepted_draft_tokens'] + (1 if r.get('rejection_position') is not None else 0) for c in cids for r in full_claim_rounds.get(c, []))
    f_alpha_eval = (f_acc / f_eval * 100.0) if f_eval else 0.0
    f_alpha_prop = (f_acc / f_prop * 100.0) if f_prop else 0.0

    print(f"{gname:<35} | {len(cids):<3} | {agg_f:<8.2f} | {agg_s:<9.2f} | {agg_gain:<+7.2f}% | {mean_f:<9.2f} | {mean_s:<10.2f} | {mean_gain:<+8.2f}% | {f_alpha_eval:.2f}%/{s_alpha_eval:.2f}% | {f_alpha_prop:.2f}%/{s_alpha_prop:.2f}%")
