import json
from pathlib import Path

cids = [x['claim_id'] for x in [json.loads(l) for l in open('/home/stackops/whale/spade-speculative-decoding/outputs/runs/decisive_experiments/sweep-budget-10000chars/trial_metrics.jsonl')]]

def get_stats(run_dir):
    p = Path(run_dir)
    t_rows = [x for x in [json.loads(l) for l in open(p / 'trial_metrics.jsonl')] if x['claim_id'] in cids]
    r_rows = [x for x in [json.loads(l) for l in open(p / 'round_traces.jsonl')] if x['claim_id'] in cids]
    
    tot_toks = sum(x["output_tokens"] for x in t_rows)
    tot_e2e = sum(x["prompt_prefill_s"] + x.get("draft_prefill_s", 0) + x["decode_s"] + x.get("context_selection_s", 0) for x in t_rows)
    tot_dec = sum(x["decode_s"] for x in t_rows)
    mean_draft_tok = sum(x.get("draft_prompt_tokens", 0) for x in t_rows) / len(t_rows)
    mean_kv_peak_mb = sum(x.get("draft_kv_peak_bytes", 0) for x in t_rows) / len(t_rows) / (1024 * 1024)

    tot_acc = sum(r.get("accepted_draft_tokens", 0) for r in r_rows)
    tot_prop = sum(r.get("proposed_token_count", 0) for r in r_rows)
    tot_eval = sum(r.get("accepted_draft_tokens", 0) + (1 if r.get("rejection_position") is not None else 0) for r in r_rows)

    alpha_prop = (tot_acc / tot_prop * 100.0) if tot_prop else 0.0
    alpha_eval = (tot_acc / tot_eval * 100.0) if tot_eval else 0.0

    e2e = tot_toks / tot_e2e if tot_e2e else 0.0
    dec = tot_toks / tot_dec if tot_dec else 0.0

    return {
        "e2e": round(e2e, 2),
        "dec": round(dec, 2),
        "alpha_eval": round(alpha_eval, 2),
        "alpha_prop": round(alpha_prop, 2),
        "draft_tok": round(mean_draft_tok, 1),
        "draft_kv_mb": round(mean_kv_peak_mb, 1)
    }

print("Baseline Full:", get_stats('/home/stackops/whale/spade-speculative-decoding/outputs/runs/b1-4-heldout300-baseline-full'))
print("BM25 Sent B10k:", get_stats('/home/stackops/whale/spade-speculative-decoding/outputs/runs/b1-4-heldout300-bm25-sent-b10k'))
