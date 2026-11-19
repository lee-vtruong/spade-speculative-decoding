import json

with open("outputs/analysis/b1_4_heldout300/b1_3_live_report.json", "r", encoding="utf-8") as f:
    rep = json.load(f)

print("=== COMPARISONS VS BASELINE FULL ===")
for c in rep["comparisons"]["vs_full"]:
    if c["metric"] in ["observed_alpha_eval_pct", "throughput_e2e_tok_s", "latency_e2e_ms_per_tok", "committed_per_round"]:
        arm = c["arm"]
        metric = c["metric"]
        delta = c["delta"]
        low = c["ci_95_low"]
        high = c["ci_95_high"]
        p = c["p_value_two_sided"]
        print(f"{arm:25s} | {metric:25s} | Delta: {delta:+7.4f} [{low:+7.4f}, {high:+7.4f}] | p={p:.4f}")

print("\n=== COMPARISONS VS SPADE (Target-Attn Sent-B10k) ===")
for c in rep["comparisons"].get("vs_spade", []):
    if c["metric"] in ["observed_alpha_eval_pct", "throughput_e2e_tok_s", "latency_e2e_ms_per_tok", "committed_per_round"]:
        arm = c["arm"]
        metric = c["metric"]
        delta = c["delta"]
        low = c["ci_95_low"]
        high = c["ci_95_high"]
        p = c["p_value_two_sided"]
        print(f"{arm:25s} | {metric:25s} | Delta: {delta:+7.4f} [{low:+7.4f}, {high:+7.4f}] | p={p:.4f}")
