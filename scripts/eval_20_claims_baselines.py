import json

cids = [x['claim_id'] for x in [json.loads(l) for l in open('/home/stackops/whale/spade-speculative-decoding/outputs/runs/decisive_experiments/sweep-budget-10000chars/trial_metrics.jsonl')]]

# Full Baseline on these 20 claims
full_trials = [json.loads(l) for l in open('/home/stackops/whale/spade-speculative-decoding/outputs/runs/b1-4-heldout300-baseline-full/trial_metrics.jsonl')]
bm25_trials = [json.loads(l) for l in open('/home/stackops/whale/spade-speculative-decoding/outputs/runs/b1-4-heldout300-bm25-sent-b10k/trial_metrics.jsonl')]
spade_trials = [json.loads(l) for l in open('/home/stackops/whale/spade-speculative-decoding/outputs/runs/b1-4-heldout300-attn-sent-b10k/trial_metrics.jsonl')]

full_map = {x['claim_id']: x for x in full_trials if x['claim_id'] in cids}
bm25_map = {x['claim_id']: x for x in bm25_trials if x['claim_id'] in cids}
spade_map = {x['claim_id']: x for x in spade_trials if x['claim_id'] in cids}

def calc(trial_map):
    s_tok = sum(trial_map[c]['output_tokens'] for c in cids)
    s_time = sum(trial_map[c]['prompt_prefill_s'] + trial_map[c].get('draft_prefill_s',0) + trial_map[c]['decode_s'] + trial_map[c].get('context_selection_s',0) for c in cids)
    dec_time = sum(trial_map[c]['decode_s'] for c in cids)
    mean_draft_tok = sum(trial_map[c].get('draft_prompt_tokens',0) for c in cids) / len(cids)
    e2e = s_tok / s_time
    dec = s_tok / dec_time
    return e2e, dec, mean_draft_tok

print("Baseline Full on 20 claims:", calc(full_map))
print("BM25 Sent B=10k on 20 claims:", calc(bm25_map))
print("SPADE B=10k on 20 claims:", calc(spade_map))
