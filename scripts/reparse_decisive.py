import json
import glob
from pathlib import Path

runs = sorted(glob.glob('/home/stackops/whale/spade-speculative-decoding/outputs/runs/decisive_experiments/*/trial_metrics.jsonl'))

print(f"{'Run ID':<30} | {'Claims':<6} | {'E2E tok/s':<10} | {'Dec tok/s':<10} | {'alpha_eval':<12} | {'alpha_prop':<12} | {'Draft Tok':<10} | {'KV Peak MB':<10}")
print("-" * 115)

for tm_path in runs:
    run_dir = Path(tm_path).parent
    t_rows = [json.loads(l) for l in open(tm_path)]
    tot_toks = sum(x["output_tokens"] for x in t_rows)
    tot_e2e = sum(x["prompt_prefill_s"] + x.get("draft_prefill_s", 0) + x["decode_s"] + x.get("context_selection_s", 0) for x in t_rows)
    tot_dec = sum(x["decode_s"] for x in t_rows)
    mean_draft_tok = sum(x.get("draft_prompt_tokens", 0) for x in t_rows) / len(t_rows)
    mean_kv_peak_mb = sum(x.get("draft_kv_peak_bytes", 0) for x in t_rows) / len(t_rows) / (1024 * 1024)

    # Acceptance from round_traces
    r_path = run_dir / "round_traces.jsonl"
    r_rows = [json.loads(l) for l in open(r_path)] if r_path.exists() else []
    tot_acc = sum(r.get("accepted_draft_tokens", 0) for r in r_rows)
    tot_prop = sum(r.get("proposed_token_count", 0) for r in r_rows)
    tot_eval = sum(r.get("accepted_draft_tokens", 0) + (1 if r.get("rejection_position") is not None else 0) for r in r_rows)

    alpha_prop = (tot_acc / tot_prop * 100.0) if tot_prop else 0.0
    alpha_eval = (tot_acc / tot_eval * 100.0) if tot_eval else 0.0

    e2e = tot_toks / tot_e2e if tot_e2e else 0.0
    dec = tot_toks / tot_dec if tot_dec else 0.0

    print(f"{run_dir.name:<30} | {len(t_rows):<6} | {e2e:<10.2f} | {dec:<10.2f} | {alpha_eval:<11.2f}% | {alpha_prop:<11.2f}% | {mean_draft_tok:<10.1f} | {mean_kv_peak_mb:<10.1f}")
