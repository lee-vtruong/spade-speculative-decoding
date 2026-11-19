import json
import glob
import statistics
from pathlib import Path

runs = sorted(glob.glob("outputs/runs/decisive_experiments/*"))
print(f"{'Run ID':<32} | {'E2E tok/s':<10} | {'Dec tok/s':<10} | {'Accept %':<10} | {'Draft Tok':<10} | {'KV Peak MB':<10}")
print("-" * 92)

for r in runs:
    p = Path(r) / "trial_metrics.jsonl"
    if not p.exists():
        continue
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        continue
    tot_tok = sum(x["output_tokens"] for x in rows)
    tot_e2e = sum(x["prompt_prefill_s"] + x.get("draft_prefill_s", 0) + x["decode_s"] + x.get("context_selection_s", 0) for x in rows)
    tot_dec = sum(x["decode_s"] for x in rows)
    mean_draft_tok = statistics.mean(x.get("draft_prompt_tokens", 0) for x in rows)
    mean_kv = statistics.mean(x.get("draft_kv_peak_bytes", 0) for x in rows) / (1024 * 1024)
    e2e_thpt = tot_tok / tot_e2e if tot_e2e > 0 else 0
    dec_thpt = tot_tok / tot_dec if tot_dec > 0 else 0

    r_trace = Path(r) / "round_traces.jsonl"
    acc = 0.0
    if r_trace.exists():
        r_rows = [json.loads(l) for l in r_trace.read_text(encoding="utf-8").splitlines() if l.strip()]
        eval_acc = sum(x.get("accepted_draft_tokens", 0) for x in r_rows)
        eval_tot = sum(x.get("proposed_token_count", 0) for x in r_rows)
        acc = (eval_acc / eval_tot) if eval_tot > 0 else 0.0

    print(f"{Path(r).name:<32} | {e2e_thpt:<10.2f} | {dec_thpt:<10.2f} | {acc*100:<10.2f} | {mean_draft_tok:<10.1f} | {mean_kv:<10.1f}")
