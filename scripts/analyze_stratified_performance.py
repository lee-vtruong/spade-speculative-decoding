"""Analyze stratified performance on retrieved-heldout300:
Breaks down SPADE vs Baseline Full across:
- Bypass cohort (Evidence <= 10k chars, 42.3%)
- Selective cohort (Evidence > 10k chars, 57.7%)
- Ultra-long cohort (Evidence >= 20k chars, 41.3%)
"""
import json
import statistics
from pathlib import Path

def main():
    spade_path = Path("outputs/runs/b1-4-heldout300-attn-sent-b10k/trial_metrics.jsonl")
    full_path = Path("outputs/runs/b1-4-heldout300-baseline-full/trial_metrics.jsonl")
    bench_path = Path("outputs/benchmarks/retrieved-heldout300.jsonl")

    spade_trials = [json.loads(l) for l in spade_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    full_trials = [json.loads(l) for l in full_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    spade_map = {x["claim_id"]: x for x in spade_trials}
    full_map = {x["claim_id"]: x for x in full_trials}

    common = sorted(set(spade_map.keys()) & set(full_map.keys()))
    print(f"Total evaluated claims: {len(common)}")

    # Load round traces to get exact acceptance rates per group
    spade_rounds = [json.loads(l) for l in Path("outputs/runs/b1-4-heldout300-attn-sent-b10k/round_traces.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    full_rounds = [json.loads(l) for l in Path("outputs/runs/b1-4-heldout300-baseline-full/round_traces.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    spade_claim_acc = {}
    for r in spade_rounds:
        cid = r["claim_id"]
        spade_claim_acc.setdefault(cid, [0, 0])
        spade_claim_acc[cid][0] += r.get("accepted_draft_tokens", 0)
        spade_claim_acc[cid][1] += r.get("proposed_token_count", 0)

    full_claim_acc = {}
    for r in full_rounds:
        cid = r["claim_id"]
        full_claim_acc.setdefault(cid, [0, 0])
        full_claim_acc[cid][0] += r.get("accepted_draft_tokens", 0)
        full_claim_acc[cid][1] += r.get("proposed_token_count", 0)

    groups = {
        "All Claims (Full Benchmark)": common,
        "Bypass Regime (Evidence <= 10k chars)": [],
        "Selective Regime (Evidence > 10k chars)": [],
        "Dense Multi-Doc (Evidence >= 20k chars)": [],
    }

    for cid in common:
        s = spade_map[cid]
        ev_chars = s.get("full_evidence_chars", 0)
        if ev_chars <= 10000:
            groups["Bypass Regime (Evidence <= 10k chars)"].append(cid)
        else:
            groups["Selective Regime (Evidence > 10k chars)"].append(cid)
        if ev_chars >= 20000:
            groups["Dense Multi-Doc (Evidence >= 20k chars)"].append(cid)

    print(f"\n{'Cohort':<40} | {'N':<4} | {'Full Thpt':<10} | {'SPADE Thpt':<10} | {'Speedup':<10} | {'Full Tok':<10} | {'SPADE Tok':<10} | {'KV Reduc':<8} | {'SPADE Acc':<9}")
    print("-" * 125)

    summary_data = {}
    for gname, cids in groups.items():
        n = len(cids)
        s_thpts = []
        f_thpts = []
        diffs = []
        s_toks = []
        f_toks = []
        s_acc_num, s_acc_den = 0, 0
        f_acc_num, f_acc_den = 0, 0

        for cid in cids:
            s = spade_map[cid]
            f = full_map[cid]
            s_tok = s["output_tokens"]
            s_e2e = s["prompt_prefill_s"] + s.get("draft_prefill_s", 0) + s["decode_s"] + s.get("context_selection_s", 0)
            s_thpt = s_tok / s_e2e if s_e2e > 0 else 0

            f_tok = f["output_tokens"]
            f_e2e = f["prompt_prefill_s"] + f.get("draft_prefill_s", 0) + f["decode_s"]
            f_thpt = f_tok / f_e2e if f_e2e > 0 else 0

            s_thpts.append(s_thpt)
            f_thpts.append(f_thpt)
            diffs.append(s_thpt - f_thpt)
            s_toks.append(s.get("draft_prompt_tokens", 0))
            f_toks.append(f.get("draft_prompt_tokens", 0))

            if cid in spade_claim_acc:
                s_acc_num += spade_claim_acc[cid][0]
                s_acc_den += spade_claim_acc[cid][1]
            if cid in full_claim_acc:
                f_acc_num += full_claim_acc[cid][0]
                f_acc_den += full_claim_acc[cid][1]

        mean_f = statistics.mean(f_thpts)
        mean_s = statistics.mean(s_thpts)
        mean_diff = statistics.mean(diffs)
        speedup = (mean_s / mean_f - 1.0) * 100 if mean_f > 0 else 0
        mean_f_tok = statistics.mean(f_toks)
        mean_s_tok = statistics.mean(s_toks)
        kv_reduc = (1.0 - mean_s_tok / mean_f_tok) * 100 if mean_f_tok > 0 else 0
        s_acc = (s_acc_num / s_acc_den) * 100 if s_acc_den > 0 else 0
        f_acc = (f_acc_num / f_acc_den) * 100 if f_acc_den > 0 else 0

        summary_data[gname] = {
            "n": n, "full_thpt": round(mean_f, 2), "spade_thpt": round(mean_s, 2),
            "diff_thpt": round(mean_diff, 2), "speedup_pct": round(speedup, 1),
            "full_draft_tokens": round(mean_f_tok, 1), "spade_draft_tokens": round(mean_s_tok, 1),
            "kv_reduction_pct": round(kv_reduc, 1), "spade_acc": round(s_acc, 2), "full_acc": round(f_acc, 2),
        }
        print(f"{gname:<40} | {n:<4} | {mean_f:<10.2f} | {mean_s:<10.2f} | {speedup:<+9.1f}% | {mean_f_tok:<10.1f} | {mean_s_tok:<10.1f} | {kv_reduc:<7.1f}% | {s_acc:<8.2f}%")

    out_file = Path("outputs/analysis/stratified_performance_breakdown.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"\nSaved report to {out_file}")

if __name__ == "__main__":
    main()
