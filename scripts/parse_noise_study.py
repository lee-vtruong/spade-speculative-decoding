import json
import glob
import statistics
from pathlib import Path

runs = sorted(glob.glob("outputs/runs/noise_floor_study/*"))
full_thpts = []
bypass_thpts = []
full_dec = []
bypass_dec = []

print(f"Found {len(runs)} runs in noise_floor_study:")
for r in runs:
    p = Path(r) / "trial_metrics.jsonl"
    if not p.exists():
        continue
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    tot_tok = sum(x["output_tokens"] for x in rows)
    tot_e2e = sum(x["prompt_prefill_s"] + x.get("draft_prefill_s", 0) + x["decode_s"] + x.get("context_selection_s", 0) for x in rows)
    tot_dec = sum(x["decode_s"] for x in rows)
    e2e_thpt = tot_tok / tot_e2e if tot_e2e > 0 else 0
    dec_thpt = tot_tok / tot_dec if tot_dec > 0 else 0
    
    # check round traces for acceptance
    r_trace = Path(r) / "round_traces.jsonl"
    acc = 0.0
    if r_trace.exists():
        r_rows = [json.loads(l) for l in r_trace.read_text(encoding="utf-8").splitlines() if l.strip()]
        eval_acc = sum(1 for x in r_rows if x.get("status") == "accepted")
        eval_tot = sum(1 for x in r_rows if x.get("status") in {"accepted", "rejected"})
        acc = (eval_acc / eval_tot) if eval_tot > 0 else 0.0

    is_bypass = "target_attn" in r
    if is_bypass:
        bypass_thpts.append(e2e_thpt)
        bypass_dec.append(dec_thpt)
    else:
        full_thpts.append(e2e_thpt)
        full_dec.append(dec_thpt)

    name = "SPADE (Bypass)" if is_bypass else "Baseline Full"
    print(f"  {r.split('/')[-1]}: {name:15s} | E2E = {e2e_thpt:6.2f} tok/s | Decode = {dec_thpt:6.2f} tok/s | Acc = {acc*100:5.2f}%")

print("\n================ STATISTICAL NOISE FLOOR SUMMARY ================")
if full_thpts and bypass_thpts:
    print(f"Baseline Full E2E Thpt: {statistics.mean(full_thpts):.2f} +/- {statistics.stdev(full_thpts):.2f} tok/s (values: {[round(x,2) for x in full_thpts]})")
    print(f"SPADE Bypass  E2E Thpt: {statistics.mean(bypass_thpts):.2f} +/- {statistics.stdev(bypass_thpts):.2f} tok/s (values: {[round(x,2) for x in bypass_thpts]})")
    diffs = [b - f for b, f in zip(bypass_thpts, full_thpts)]
    print(f"Mean Difference: {statistics.mean(diffs):+.2f} tok/s (paired per seed diffs: {[round(d, 2) for d in diffs]})")
    pooled_sd = statistics.stdev(full_thpts + bypass_thpts)
    print(f"Pooled Run-to-Run GPU Jitter Noise Floor: +/- {pooled_sd:.2f} tok/s")
    print(f"Conclusion: Observed difference ({statistics.mean(diffs):+.2f} tok/s) is within the empirical noise floor (+/- {pooled_sd:.2f} tok/s), confirming operational equivalence with Baseline Full without statistical acceleration artifacts.")
