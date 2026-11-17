"""Compile and audit unified benchmark results across B0-S, HF-Assisted, B2 K=4, SPADE Fixed, and SPADE Dynamic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-prefix", default="unified-v2")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--output", type=Path, default=Path("outputs/unified_benchmark_report.json"))
    args = p.parse_args()

    suffix = f"lim{args.limit}"
    runs = [
        ("B0-S (Target-Only)", f"{args.run_prefix}-b0s-{suffix}"),
        ("HF-Assisted (K=4)", f"{args.run_prefix}-hf-assisted-k4-{suffix}"),
        ("B2 (Speculative K=4)", f"{args.run_prefix}-b2-k4-{suffix}"),
        ("SPADE Fixed (K=4)", f"{args.run_prefix}-spade-fixed-k4-{suffix}"),
        ("SPADE Dynamic (K in {2,4,6})", f"{args.run_prefix}-spade-dynamic-k-{suffix}"),
    ]

    print("=" * 120)
    print(f"UNIFIED BENCHMARK AUDIT REPORT ({args.run_prefix}, limit={args.limit}, AVeriTeC Dev Oracle)")
    print("=" * 120)

    table_rows = []

    for label, run_id in runs:
        run_dir = Path("outputs/runs") / run_id
        if not run_dir.exists():
            print(f"Skipping {run_id} (not found)")
            continue

        metrics = load_json(run_dir / "metrics.json")
        env = load_json(run_dir / "environment.json")
        trials = load_jsonl(run_dir / "trial_metrics.jsonl")
        rounds = load_jsonl(run_dir / "round_traces.jsonl")
        tokens = load_jsonl(run_dir / "token_traces.jsonl")

        total_claims = len(trials)
        total_output_tokens = sum(t.get("output_tokens", 0) for t in trials)
        mean_tokens_per_claim = total_output_tokens / total_claims if total_claims else 0.0

        total_decode_s = sum(t.get("decode_s", 0.0) for t in trials)
        total_prefill_s = sum(t.get("prompt_prefill_s", 0.0) + t.get("draft_prefill_s", 0.0) for t in trials)
        total_e2e_s = total_decode_s + total_prefill_s

        mean_decode_s = total_decode_s / total_claims if total_claims else 0.0
        mean_e2e_s = total_e2e_s / total_claims if total_claims else 0.0
        throughput_decode = total_output_tokens / total_decode_s if total_decode_s else 0.0
        throughput_e2e = total_output_tokens / total_e2e_s if total_e2e_s else 0.0
        latency_per_token_ms = (total_decode_s / total_output_tokens) * 1000.0 if total_output_tokens else 0.0

        total_rounds = len(rounds)
        total_draft_s = sum(t.get("draft_generation_s", 0.0) for t in trials)
        total_verify_s = sum(t.get("target_verification_s", 0.0) for t in trials)
        draft_lat_ms = (total_draft_s / total_rounds) * 1000.0 if total_rounds else 0.0
        verify_lat_ms = (total_verify_s / total_rounds) * 1000.0 if total_rounds else 0.0
        draft_overhead_pct = (total_draft_s / total_decode_s) * 100.0 if total_decode_s else 0.0
        verify_overhead_pct = (total_verify_s / total_decode_s) * 100.0 if total_decode_s else 0.0
        target_fwd_reduction = (total_output_tokens / total_rounds) if total_rounds else 1.0

        # Token accounting & acceptance metrics (only for custom speculative runs with token traces)
        is_spec = label.startswith(("B2", "SPADE"))
        if is_spec and rounds:
            n_acc = sum(r.get("accepted_draft_tokens", 0) for r in rounds)
            n_prop = sum(r.get("proposed_token_count", 0) for r in rounds)
            n_res = sum(1 for r in rounds if r.get("rejection_position") is not None)
            n_bon = sum(1 for r in rounds if r.get("additional_target_token"))
            accounting_ok = (n_acc + n_res + n_bon == total_output_tokens)
            alpha_prop = (n_acc / n_prop) * 100.0 if n_prop else 0.0
            prop_per_committed = n_prop / total_output_tokens if total_output_tokens else 0.0

            # Evaluated acceptance (alpha_eval) from token_traces
            eval_tokens = [
                t for t in tokens
                if t.get("status") in {"accepted", "rejected"}
            ]
            n_eval = len(eval_tokens)
            n_eval_acc = sum(1 for t in eval_tokens if t.get("status") == "accepted")
            alpha_eval = (n_eval_acc / n_eval) * 100.0 if n_eval else 0.0

            ev_tokens = [t for t in eval_tokens if (t.get("evidence_overlap") or {}).get("lexical_overlap", 0.0) > 0]
            non_ev_tokens = [t for t in eval_tokens if (t.get("evidence_overlap") or {}).get("lexical_overlap", 0.0) == 0]
            ev_acc = sum(1 for t in ev_tokens if t.get("status") == "accepted")
            non_ev_acc = sum(1 for t in non_ev_tokens if t.get("status") == "accepted")
            alpha_ev = (ev_acc / len(ev_tokens)) * 100.0 if ev_tokens else 0.0
            alpha_non_ev = (non_ev_acc / len(non_ev_tokens)) * 100.0 if non_ev_tokens else 0.0
        else:
            n_acc = n_res = n_bon = n_prop = n_eval = 0
            accounting_ok = True
            alpha_prop = alpha_eval = alpha_ev = alpha_non_ev = None
            prop_per_committed = None

        git_commit = env.get("git_commit") or "unknown"
        git_dirty = env.get("git_dirty")

        row = {
            "Method": label,
            "Run ID": run_id,
            "Claims": total_claims,
            "Total Tokens": total_output_tokens,
            "Mean Tokens": round(mean_tokens_per_claim, 2),
            "Throughput Decode (tok/s)": round(throughput_decode, 2),
            "Throughput E2E (tok/s)": round(throughput_e2e, 2),
            "Decode/Claim (s)": round(mean_decode_s, 3),
            "E2E/Claim (s)": round(mean_e2e_s, 3),
            "Latency/Token (ms)": round(latency_per_token_ms, 2),
            "Target Rounds": total_rounds,
            "Target Fwd Reduc": round(target_fwd_reduction, 2),
            "Draft Lat/Rnd (ms)": round(draft_lat_ms, 1),
            "Draft % Decode": round(draft_overhead_pct, 1),
            "Verify Lat/Rnd (ms)": round(verify_lat_ms, 1),
            "Verify % Decode": round(verify_overhead_pct, 1),
            "N_acc": n_acc,
            "N_res": n_res,
            "N_bon": n_bon,
            "N_prop": n_prop,
            "N_eval": n_eval,
            "Accounting Exact": accounting_ok,
            "Alpha_eval (%)": round(alpha_eval, 2) if alpha_eval is not None else None,
            "Alpha_prop (%)": round(alpha_prop, 2) if alpha_prop is not None else None,
            "Alpha_ev_overlap (%)": round(alpha_ev, 2) if alpha_ev is not None else None,
            "Alpha_non_overlap (%)": round(alpha_non_ev, 2) if alpha_non_ev is not None else None,
            "Prop/Committed": round(prop_per_committed, 3) if prop_per_committed is not None else None,
            "JSON Valid (%)": round(metrics.get("json_parse_success_rate", 0.0) * 100.0, 1),
            "Citation Valid (%)": round(metrics.get("citation_id_validity_rate", 0.0) * 100.0, 1),
            "Git Commit": git_commit[:7],
            "Git Dirty": git_dirty,
            "PyTorch": env.get("torch"),
            "Transformers": env.get("transformers"),
        }
        table_rows.append(row)

    headers = [
        "Method", "Tok/s (Dec)", "Tok/s (E2E)", "Lat/Claim", "ms/Tok",
        "Tgt Rnds", "Fwd Red.", "Draft ms/Rnd", "Draft %", "Verify ms/Rnd",
        "alpha_eval", "alpha_prop", "JSON %", "Commit"
    ]
    print("\n| " + " | ".join(headers) + " |")
    print("| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |")
    for r in table_rows:
        commit_str = f"{r['Git Commit']}" + ("*" if r["Git Dirty"] else "")
        line = [
            r["Method"],
            f"{r['Throughput Decode (tok/s)']:.2f}",
            f"{r['Throughput E2E (tok/s)']:.2f}",
            f"{r['Decode/Claim (s)']:.2f}s",
            f"{r['Latency/Token (ms)']:.2f}",
            f"{r['Target Rounds']}",
            f"{r['Target Fwd Reduc']:.2f}x",
            f"{r['Draft Lat/Rnd (ms)']:.1f}",
            f"{r['Draft % Decode']:.1f}%",
            f"{r['Verify Lat/Rnd (ms)']:.1f}",
            f"{r['Alpha_eval (%)']:.2f}%" if r["Alpha_eval (%)"] is not None else "—",
            f"{r['Alpha_prop (%)']:.2f}%" if r["Alpha_prop (%)"] is not None else "—",
            f"{r['JSON Valid (%)']:.1f}%",
            commit_str,
        ]
        print("| " + " | ".join(line) + " |")

    print("\nToken Accounting Verification:")
    for r in table_rows:
        if r["N_prop"]:
            print(
                f"  {r['Method']}: N_acc={r['N_acc']} + N_res={r['N_res']} + N_bon={r['N_bon']} "
                f"= {r['N_acc'] + r['N_res'] + r['N_bon']} vs Total={r['Total Tokens']} "
                f"[EXACT={r['Accounting Exact']}] | EvOverlap={r['Alpha_ev_overlap (%)']}% NonOverlap={r['Alpha_non_overlap (%)']}%"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table_rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
