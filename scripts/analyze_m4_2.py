"""M4.2 Offline Analysis: Evidence-Selective Context Speculative Decoding Report.

Pure CPU analysis combining:
  - Fixed-prefix asymmetric distribution replay (TV(p,q), JS(p,q), expected acceptance 1-TV)
  - Live KV-cache exact speculative runs (Baseline, Selective-TopK, Compact):
      draft prefill latency, draft decode latency, draft KV-cache memory, context selection cost,
      observed acceptance (alpha_eval, alpha_prop), committed tokens/round, and end-to-end throughput
      (strictly including context_selection_s + prefill + decode).
  - Claim-clustered paired bootstrap (B=10,000, strictly by claim_id, two-sided p-value, 95% CI)
  - Explicit break-even decision criterion: draft savings vs rejection + selection cost.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from spade.analysis.distribution_audit import claim_clustered_bootstrap_mean
from spade.analysis.experiment_audit import load_jsonl, summarize_and_audit_run


def _extract_m4_2_claim_map(run_dir: Path) -> dict[str, dict[str, Any]]:
    trials = load_jsonl(run_dir / "trial_metrics.jsonl")
    rounds = load_jsonl(run_dir / "round_traces.jsonl")
    tokens = load_jsonl(run_dir / "token_traces.jsonl")

    claims: dict[str, dict[str, Any]] = {}
    for t in trials:
        cid = str(t["claim_id"])
        if t.get("trial_index", 0) != 0 and cid in claims:
            continue
        sel_s = float(t.get("context_selection_s", 0.0))
        p_pref_s = float(t.get("prompt_prefill_s", 0.0))
        d_pref_s = float(t.get("draft_prefill_s", 0.0))
        d_dec_s = float(t.get("draft_generation_s", 0.0))
        ver_s = float(t.get("target_verification_s", 0.0))
        corr_s = float(t.get("correction_s", 0.0))
        dec_s = float(t.get("decode_s", 0.0))
        unattr_dec_s = dec_s - (d_dec_s + ver_s + corr_s)
        claims[cid] = {
            "claim_id": cid,
            "output_tokens": int(t.get("output_tokens", 0)),
            "target_prompt_tokens": int(t.get("prompt_tokens", 0)),
            "draft_prompt_tokens": int(t.get("draft_prompt_tokens", t.get("prompt_tokens", 0))),
            "context_selection_s": sel_s,
            "prompt_prefill_s": p_pref_s,
            "draft_prefill_s": d_pref_s,
            "draft_decode_s": d_dec_s,
            "verify_s": ver_s,
            "correction_s": corr_s,
            "unattributed_decode_s": unattr_dec_s,
            "decode_s": dec_s,
            "e2e_s": sel_s + p_pref_s + d_pref_s + dec_s,
            "draft_kv_prefill_mb": float(t.get("draft_kv_prefill_bytes", 0.0)) / (1024.0 * 1024.0),
            "draft_kv_peak_mb": float(t.get("draft_kv_peak_bytes", 0.0)) / (1024.0 * 1024.0),
            "target_kv_peak_mb": float(t.get("target_kv_peak_bytes", 0.0)) / (1024.0 * 1024.0),
            "target_rounds": 0,
            "n_acc": 0,
            "n_prop": 0,
            "n_eval": 0,
            "n_eval_acc": 0,
        }

    for r in rounds:
        cid = str(r["claim_id"])
        if cid in claims:
            claims[cid]["target_rounds"] += 1
            claims[cid]["n_acc"] += int(r.get("accepted_draft_tokens", 0))
            claims[cid]["n_prop"] += int(r.get("proposed_token_count", 0))

    for tok in tokens:
        st = tok.get("status")
        if st not in {"accepted", "rejected"}:
            continue
        cid = str(tok["claim_id"])
        if cid in claims:
            claims[cid]["n_eval"] += 1
            if st == "accepted":
                claims[cid]["n_eval_acc"] += 1

    return claims


def _pooled_live_metric(rows: list[dict[str, Any]], metric: str) -> float:
    n = len(rows)
    if n == 0:
        return 0.0
    if metric == "throughput_e2e_tok_s":
        den = sum(r["e2e_s"] for r in rows)
        return sum(r["output_tokens"] for r in rows) / den if den > 0 else 0.0
    if metric == "latency_e2e_ms_per_tok":
        den = sum(r["output_tokens"] for r in rows)
        return (sum(r["e2e_s"] for r in rows) * 1000.0) / den if den > 0 else 0.0
    if metric == "e2e_ms_per_claim":
        return (sum(r["e2e_s"] for r in rows) * 1000.0) / n
    if metric == "draft_prefill_ms_per_claim":
        return (sum(r["draft_prefill_s"] for r in rows) * 1000.0) / n
    if metric == "draft_decode_ms_per_round":
        den = sum(r["target_rounds"] for r in rows)
        return (sum(r["draft_decode_s"] for r in rows) * 1000.0) / den if den > 0 else 0.0
    if metric == "draft_kv_peak_mb":
        return sum(r["draft_kv_peak_mb"] for r in rows) / n
    if metric == "observed_alpha_eval_pct":
        den = sum(r["n_eval"] for r in rows)
        return (sum(r["n_eval_acc"] for r in rows) / den) * 100.0 if den > 0 else 0.0
    if metric == "observed_alpha_prop_pct":
        den = sum(r["n_prop"] for r in rows)
        return (sum(r["n_acc"] for r in rows) / den) * 100.0 if den > 0 else 0.0
    if metric == "committed_per_round":
        den = sum(r["target_rounds"] for r in rows)
        return sum(r["output_tokens"] for r in rows) / den if den > 0 else 0.0
    if metric == "target_rounds_per_claim":
        return sum(r["target_rounds"] for r in rows) / n
    raise ValueError(f"Unknown live metric: {metric}")


def _paired_live_bootstrap(
    base_map: dict[str, dict[str, Any]],
    treat_map: dict[str, dict[str, Any]],
    metric: str,
    *,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    common = sorted(set(base_map) & set(treat_map))
    b_list = [base_map[c] for c in common]
    t_list = [treat_map[c] for c in common]
    b_val = _pooled_live_metric(b_list, metric)
    t_val = _pooled_live_metric(t_list, metric)
    obs_delta = t_val - b_val

    rng = random.Random(seed)
    n = len(common)
    boots: list[float] = []
    for _ in range(n_bootstrap):
        idxs = [rng.randrange(n) for _ in range(n)]
        boots.append(
            _pooled_live_metric([t_list[i] for i in idxs], metric)
            - _pooled_live_metric([b_list[i] for i in idxs], metric)
        )
    boots.sort()
    lo = boots[int(0.025 * n_bootstrap)]
    hi = boots[min(n_bootstrap - 1, int(0.975 * n_bootstrap))]
    p2 = min(1.0, 2.0 * min(sum(1 for x in boots if x <= 0.0) / n_bootstrap, sum(1 for x in boots if x >= 0.0) / n_bootstrap))
    interp = "positive_ci_excludes_zero" if lo > 0 else ("negative_ci_excludes_zero" if hi < 0 else "inconclusive_ci_crosses_zero")
    return {
        "metric": metric,
        "n_claims": n,
        "baseline": round(b_val, 4),
        "treatment": round(t_val, 4),
        "delta": round(obs_delta, 4),
        "ci_95_low": round(lo, 4),
        "ci_95_high": round(hi, 4),
        "p_value_two_sided": round(p2, 4),
        "interpretation": interp,
        "cluster_unit": "claim_id",
    }


def analyze_m4_2(
    *,
    runs_root: Path,
    run_prefix: str,
    draft_top_k: int = 2,
    replay_path: Path | None = None,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    run_specs = [
        ("Baseline (Full -> Full)", "full", f"{run_prefix}-baseline-full"),
        (f"Selective-TopK (Full -> Top-{draft_top_k})", "selective_topk", f"{run_prefix}-selective-topk{draft_top_k}"),
        ("Compact (Full -> Compact)", "compact", f"{run_prefix}-compact"),
    ]

    live_rows: list[dict[str, Any]] = []
    claim_maps: dict[str, dict[str, dict[str, Any]]] = {}

    for label, policy, run_id in run_specs:
        rdir = runs_root / run_id
        if not rdir.exists():
            continue
        base_sum = summarize_and_audit_run(rdir)
        cmap = _extract_m4_2_claim_map(rdir)
        claim_maps[policy] = cmap
        c_list = list(cmap.values())
        n_c = len(c_list)

        total_tokens = sum(c["output_tokens"] for c in c_list)
        total_e2e_s = sum(c["e2e_s"] for c in c_list)
        total_rounds = sum(c["target_rounds"] for c in c_list)
        mean_t_tok = sum(c["target_prompt_tokens"] for c in c_list) / n_c if n_c else 0.0
        mean_d_tok = sum(c["draft_prompt_tokens"] for c in c_list) / n_c if n_c else 0.0
        mean_sel_ms = (sum(c["context_selection_s"] for c in c_list) * 1000.0) / n_c if n_c else 0.0
        mean_t_pref_ms = (sum(c["prompt_prefill_s"] for c in c_list) * 1000.0) / n_c if n_c else 0.0
        mean_d_pref_ms = (sum(c["draft_prefill_s"] for c in c_list) * 1000.0) / n_c if n_c else 0.0
        mean_d_dec_ms_claim = (sum(c["draft_decode_s"] for c in c_list) * 1000.0) / n_c if n_c else 0.0
        mean_verify_ms_claim = (sum(c["verify_s"] for c in c_list) * 1000.0) / n_c if n_c else 0.0
        mean_corr_ms_claim = (sum(c["correction_s"] for c in c_list) * 1000.0) / n_c if n_c else 0.0
        mean_unattr_ms_claim = (sum(c["unattributed_decode_s"] for c in c_list) * 1000.0) / n_c if n_c else 0.0
        mean_kv_pref_mb = sum(c["draft_kv_prefill_mb"] for c in c_list) / n_c if n_c else 0.0
        mean_kv_peak_mb = sum(c["draft_kv_peak_mb"] for c in c_list) / n_c if n_c else 0.0
        mean_t_kv_peak_mb = sum(c["target_kv_peak_mb"] for c in c_list) / n_c if n_c else 0.0

        row = {
            **base_sum,
            "label": label,
            "policy": policy,
            "mean_target_prompt_tokens": round(mean_t_tok, 1),
            "mean_draft_prompt_tokens": round(mean_d_tok, 1),
            "prompt_compression_pct": round((1.0 - mean_d_tok / mean_t_tok) * 100.0, 2) if mean_t_tok > 0 else 0.0,
            "context_selection_ms_per_claim": round(mean_sel_ms, 3),
            "target_prefill_ms_per_claim": round(mean_t_pref_ms, 2),
            "draft_prefill_ms_per_claim": round(mean_d_pref_ms, 2),
            "draft_decode_ms_per_claim": round(mean_d_dec_ms_claim, 2),
            "verify_ms_per_claim": round(mean_verify_ms_claim, 2),
            "correction_ms_per_claim": round(mean_corr_ms_claim, 2),
            "unattributed_decode_ms_per_claim": round(mean_unattr_ms_claim, 2),
            "unattributed_decode_ms_per_round": round((sum(c["unattributed_decode_s"] for c in c_list) * 1000.0) / total_rounds, 2) if total_rounds else 0.0,
            "draft_kv_prefill_mb": round(mean_kv_pref_mb, 3),
            "draft_kv_peak_mb": round(mean_kv_peak_mb, 3),
            "target_kv_peak_mb": round(mean_t_kv_peak_mb, 3),
            "committed_per_round": round(total_tokens / total_rounds, 3) if total_rounds else 0.0,
            "throughput_e2e_incl_sel_tok_s": round(total_tokens / total_e2e_s, 2) if total_e2e_s > 0 else 0.0,
            "latency_e2e_incl_sel_ms_per_tok": round((total_e2e_s / total_tokens) * 1000.0, 2) if total_tokens else 0.0,
            "e2e_incl_sel_ms_per_claim": round((total_e2e_s * 1000.0) / n_c, 2) if n_c else 0.0,
        }
        live_rows.append(row)

    # Fixed-prefix asymmetric replay summary
    replay_summary: dict[str, Any] = {}
    replay_bootstrap: list[dict[str, Any]] = []
    if replay_path is not None and replay_path.exists():
        rep_recs = load_jsonl(replay_path)
        if rep_recs:
            all_cids = sorted({str(r["claim_id"]) for r in rep_recs})
            for pol in ("full", "selective_topk", "compact"):
                tvs = [r["policies"][pol]["tv"] for r in rep_recs]
                jss = [r["policies"][pol]["js"] for r in rep_recs]
                exps = [r["policies"][pol]["expected_acceptance"] for r in rep_recs]
                d_toks = [r["policies"][pol]["draft_prompt_tokens"] for r in rep_recs]
                t_toks = [r["target_prompt_tokens"] for r in rep_recs]
                replay_summary[pol] = {
                    "n_prefixes": len(rep_recs),
                    "n_claims": len(all_cids),
                    "mean_tv": round(sum(tvs) / len(tvs), 6),
                    "mean_js": round(sum(jss) / len(jss), 6),
                    "per_prefix_expected_acceptance_pct": round((sum(exps) / len(exps)) * 100.0, 4),
                    "mean_draft_prompt_tokens": round(sum(d_toks) / len(d_toks), 1),
                    "mean_target_prompt_tokens": round(sum(t_toks) / len(t_toks), 1),
                }
            for pol in ("selective_topk", "compact"):
                sums_dtv: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
                sums_dexp: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
                for r in rep_recs:
                    cid = str(r["claim_id"])
                    sums_dtv[cid][0] += r["policies"][pol]["tv"] - r["policies"]["full"]["tv"]
                    sums_dtv[cid][1] += 1
                    sums_dexp[cid][0] += (r["policies"][pol]["expected_acceptance"] - r["policies"]["full"]["expected_acceptance"]) * 100.0
                    sums_dexp[cid][1] += 1
                replay_bootstrap.append({
                    "comparison": f"{pol} vs full (fixed prefixes)",
                    "delta_tv": claim_clustered_bootstrap_mean(
                        {c: (v[0], int(v[1])) for c, v in sums_dtv.items()},
                        all_claim_ids=all_cids,
                        n_bootstrap=n_bootstrap,
                        seed=seed,
                        higher_is_better=False,
                    ),
                    "delta_expected_acceptance_pp": claim_clustered_bootstrap_mean(
                        {c: (v[0], int(v[1])) for c, v in sums_dexp.items()},
                        all_claim_ids=all_cids,
                        n_bootstrap=n_bootstrap,
                        seed=seed,
                        higher_is_better=True,
                    ),
                })

    # Live paired cluster bootstrap & Break-Even Decision Check vs Baseline
    live_comparisons: list[dict[str, Any]] = []
    break_even: list[dict[str, Any]] = []
    by_policy = {r["policy"]: r for r in live_rows}
    if "full" in claim_maps:
        base_r = by_policy["full"]
        base_kv = base_r["draft_kv_peak_mb"]
        for r in live_rows:
            r["draft_kv_reduction_pct"] = round((1.0 - r["draft_kv_peak_mb"] / base_kv) * 100.0, 2) if base_kv > 0 else 0.0

        metrics_live = [
            "throughput_e2e_tok_s",
            "latency_e2e_ms_per_tok",
            "e2e_ms_per_claim",
            "draft_prefill_ms_per_claim",
            "draft_decode_ms_per_round",
            "draft_kv_peak_mb",
            "observed_alpha_eval_pct",
            "observed_alpha_prop_pct",
            "committed_per_round",
            "target_rounds_per_claim",
        ]
        for pol in ("selective_topk", "compact"):
            if pol not in claim_maps:
                continue
            tr = by_policy[pol]
            per_m = {
                m: _paired_live_bootstrap(claim_maps["full"], claim_maps[pol], m, n_bootstrap=n_bootstrap, seed=seed)
                for m in metrics_live
            }
            live_comparisons.append({
                "comparison": f"{tr['label']} vs Baseline",
                "policy": pol,
                "metrics": per_m,
            })

            target_prefill_delta_ms = tr["target_prefill_ms_per_claim"] - base_r["target_prefill_ms_per_claim"]
            draft_prefill_delta_ms = tr["draft_prefill_ms_per_claim"] - base_r["draft_prefill_ms_per_claim"]
            draft_decode_delta_ms = tr["draft_decode_ms_per_claim"] - base_r["draft_decode_ms_per_claim"]
            draft_saved_ms = -(draft_prefill_delta_ms + draft_decode_delta_ms)
            extra_verify_cost_ms = tr["verify_ms_per_claim"] - base_r["verify_ms_per_claim"]
            correction_delta_ms = tr["correction_ms_per_claim"] - base_r["correction_ms_per_claim"]
            unattributed_decode_delta_ms = tr["unattributed_decode_ms_per_claim"] - base_r["unattributed_decode_ms_per_claim"]
            selection_cost_ms = tr["context_selection_ms_per_claim"] - base_r["context_selection_ms_per_claim"]
            net_e2e_delta_ms = tr["e2e_incl_sel_ms_per_claim"] - base_r["e2e_incl_sel_ms_per_claim"]
            net_e2e_saved_ms = -net_e2e_delta_ms
            accounting_sum_delta_ms = (
                target_prefill_delta_ms
                + draft_prefill_delta_ms
                + draft_decode_delta_ms
                + extra_verify_cost_ms
                + correction_delta_ms
                + unattributed_decode_delta_ms
                + selection_cost_ms
            )
            accounting_residual_ms = net_e2e_delta_ms - accounting_sum_delta_ms
            tp_boot = per_m["throughput_e2e_tok_s"]

            if tp_boot["interpretation"] == "positive_ci_excludes_zero" and net_e2e_saved_ms > 0:
                verdict = "PASS_BREAK_EVEN"
            elif tp_boot["interpretation"] == "negative_ci_excludes_zero" or net_e2e_saved_ms < 0:
                verdict = "FAIL_BREAK_EVEN"
            else:
                verdict = "NEUTRAL_INCONCLUSIVE"

            break_even.append({
                "policy": pol,
                "label": tr["label"],
                "target_prefill_delta_ms_per_claim": round(target_prefill_delta_ms, 2),
                "draft_prefill_delta_ms_per_claim": round(draft_prefill_delta_ms, 2),
                "draft_decode_delta_ms_per_claim": round(draft_decode_delta_ms, 2),
                "draft_compute_saved_ms_per_claim": round(draft_saved_ms, 2),
                "extra_verify_cost_ms_per_claim": round(extra_verify_cost_ms, 2),
                "correction_delta_ms_per_claim": round(correction_delta_ms, 2),
                "unattributed_decode_delta_ms_per_claim": round(unattributed_decode_delta_ms, 2),
                "context_selection_cost_ms_per_claim": round(selection_cost_ms, 3),
                "net_e2e_delta_ms_per_claim": round(net_e2e_delta_ms, 2),
                "net_e2e_saved_ms_per_claim": round(net_e2e_saved_ms, 2),
                "accounting_residual_ms_per_claim": round(accounting_residual_ms, 3),
                "delta_e2e_throughput_tok_s": tp_boot["delta"],
                "delta_e2e_throughput_ci95": [tp_boot["ci_95_low"], tp_boot["ci_95_high"]],
                "p_value_two_sided": tp_boot["p_value_two_sided"],
                "break_even_verdict": verdict,
            })

    return {
        "schema_version": "m4_2_selective_context_v1",
        "run_prefix": run_prefix,
        "all_audits_passed": all(r["audit_passed"] for r in live_rows),
        "live_rows": live_rows,
        "fixed_prefix_replay_summary": replay_summary,
        "fixed_prefix_replay_bootstrap": replay_bootstrap,
        "live_paired_comparisons": live_comparisons,
        "break_even_evaluation": break_even,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# M4.2 — Evidence-Selective Context Speculative Decoding Report (`{report['run_prefix']}`)",
        "",
        f"- All live run audits passed: `{report['all_audits_passed']}` | Cluster unit: `claim_id`",
        "- End-to-end throughput strictly includes `context_selection_s + prompt_prefill_s + draft_prefill_s + decode_s`.",
        "",
        "## 1. Live Speculative KV-Cache Results (Target = Full Evidence, B2 K=4)",
        "",
        "| Config | Draft Tok | Comp % | Draft KV Peak (MB) | KV Red % | Sel (ms) | Draft Prefill (ms) | Draft Dec ms/Rnd | Obs $\\alpha_{\\mathrm{eval}}$ | Obs $\\alpha_{\\mathrm{prop}}$ | Comm/Rnd | Tgt Rnds | E2E tok/s | E2E ms/tok | JSON / Cit % | Audit |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in report["live_rows"]:
        lines.append(
            f"| {r['label']} | {r['mean_draft_prompt_tokens']} / {r['mean_target_prompt_tokens']} | "
            f"{r['prompt_compression_pct']:.1f}% | {r['draft_kv_peak_mb']:.2f} | "
            f"{r.get('draft_kv_reduction_pct', 0.0):.1f}% | {r['context_selection_ms_per_claim']:.2f} | "
            f"{r['draft_prefill_ms_per_claim']:.2f} | {r['draft_latency_ms_per_round']:.2f} | "
            f"{r['acceptance_eval_pct']}% | {r['acceptance_prop_pct']}% | {r['committed_per_round']:.3f} | "
            f"{r['target_rounds']} | **{r['throughput_e2e_incl_sel_tok_s']:.2f}** | "
            f"{r['latency_e2e_incl_sel_ms_per_tok']:.2f} | {r['json_valid_pct']}% / {r['citation_valid_pct']}% | "
            f"{'PASS' if r['audit_passed'] else 'FAIL'} |"
        )

    if report["fixed_prefix_replay_summary"]:
        lines.extend([
            "",
            "## 2. Fixed-Prefix Distribution Audit (Strictly Separated from Observed Rollout Acceptance)",
            "",
            "| Draft Context Policy | Draft Prompt Tok | Mean $\\mathrm{TV}(p,q)$ | Mean $\\mathrm{JS}(p,q)$ | Per-Prefix Expected Acc $1-\\mathrm{TV}$ (%) |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])
        for pol in ("full", "selective_topk", "compact"):
            if pol in report["fixed_prefix_replay_summary"]:
                s = report["fixed_prefix_replay_summary"][pol]
                lines.append(
                    f"| `{pol}` | {s['mean_draft_prompt_tokens']} / {s['mean_target_prompt_tokens']} | "
                    f"{s['mean_tv']:.5f} | {s['mean_js']:.5f} | {s['per_prefix_expected_acceptance_pct']:.3f}% |"
                )
        for rb in report["fixed_prefix_replay_bootstrap"]:
            dtv = rb["delta_tv"]
            dexp = rb["delta_expected_acceptance_pp"]
            lines.append(
                f"- **{rb['comparison']}:** $\\Delta\\mathrm{{TV}} = {dtv['mean']:+.5f}$ "
                f"(95% CI `[{dtv['ci_95_low']:+.5f}, {dtv['ci_95_high']:+.5f}]`, $p={dtv['p_value_two_sided']:.4f}$); "
                f"$\\Delta(1-\\mathrm{{TV}}) = {dexp['mean']:+.4f}$ pp "
                f"(95% CI `[{dexp['ci_95_low']:+.4f}, {dexp['ci_95_high']:+.4f}]`, $p={dexp['p_value_two_sided']:.4f}$)."
            )

    if report["break_even_evaluation"]:
        lines.extend([
            "",
            "## 3. Break-Even Decision Check (100% Additive E2E Decomposition, ms/claim)",
            "",
            "| Policy | $\\Delta$ Tgt Prefill | $\\Delta$ Draft Prefill | $\\Delta$ Draft Dec | $\\Delta$ Verify | $\\Delta$ Corr | $\\Delta$ Target Dist/Loop | $\\Delta$ Sel | Net $\\Delta$ E2E | Residual | $\\Delta$ E2E tok/s (95% CI, $p$) | Verdict |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for b in report["break_even_evaluation"]:
            lines.append(
                f"| {b['label']} | {b['target_prefill_delta_ms_per_claim']:+.2f} | "
                f"{b['draft_prefill_delta_ms_per_claim']:+.2f} | {b['draft_decode_delta_ms_per_claim']:+.2f} | "
                f"{b['extra_verify_cost_ms_per_claim']:+.2f} | {b['correction_delta_ms_per_claim']:+.2f} | "
                f"{b['unattributed_decode_delta_ms_per_claim']:+.2f} | {b['context_selection_cost_ms_per_claim']:+.3f} | "
                f"**{b['net_e2e_delta_ms_per_claim']:+.2f}** | {b['accounting_residual_ms_per_claim']:+.3f} | "
                f"{b['delta_e2e_throughput_tok_s']:+.3f} "
                f"(`[{b['delta_e2e_throughput_ci95'][0]:+.3f}, {b['delta_e2e_throughput_ci95'][1]:+.3f}]`, $p={b['p_value_two_sided']:.4f}$) | "
                f"`{b['break_even_verdict']}` |"
            )

    if report["live_paired_comparisons"]:
        lines.extend([
            "",
            "## 4. Claim-Clustered Paired Bootstrap Details (Live Runs)",
            "",
            "| Comparison | Metric | Base | Treat | Delta | 95% CI | Two-Sided $p$ | Interpretation |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
        ])
        for comp in report["live_paired_comparisons"]:
            for m, s in comp["metrics"].items():
                lines.append(
                    f"| {comp['comparison']} | `{m}` | {s['baseline']:.4f} | {s['treatment']:.4f} | "
                    f"{s['delta']:+.4f} | [{s['ci_95_low']:+.4f}, {s['ci_95_high']:+.4f}] | "
                    f"{s['p_value_two_sided']:.4f} | `{s['interpretation']}` |"
                )

    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    p.add_argument("--run-prefix", default="m4-2-pilot100")
    p.add_argument("--draft-top-k", type=int, default=2)
    p.add_argument("--replay-jsonl", type=Path, default=None)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/m4_2_pilot"))
    args = p.parse_args()

    replay_path = args.replay_jsonl or (args.output_dir / "m4_2_prefix_divergence.jsonl")
    report = analyze_m4_2(
        runs_root=args.runs_root,
        run_prefix=args.run_prefix,
        draft_top_k=args.draft_top_k,
        replay_path=replay_path,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "m4_2_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = render_markdown(report)
    (args.output_dir / "m4_2_report.md").write_text(md, encoding="utf-8")

    if report["live_rows"]:
        fields = [
            "label", "policy", "run_id", "claims", "total_tokens",
            "mean_target_prompt_tokens", "mean_draft_prompt_tokens", "prompt_compression_pct",
            "draft_kv_prefill_mb", "draft_kv_peak_mb", "draft_kv_reduction_pct", "target_kv_peak_mb",
            "context_selection_ms_per_claim", "draft_prefill_ms_per_claim", "draft_latency_ms_per_round",
            "acceptance_eval_pct", "acceptance_prop_pct", "committed_per_round", "target_rounds",
            "throughput_e2e_incl_sel_tok_s", "latency_e2e_incl_sel_ms_per_tok",
            "json_valid_pct", "citation_valid_pct", "verdict_accuracy_pct", "verdict_macro_f1",
            "audit_passed", "git_commit", "git_dirty",
        ]
        with (args.output_dir / "m4_2_table.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in report["live_rows"]:
                w.writerow({k: r.get(k) for k in fields})

    print(md)
    print(f"M4.2 report written to {args.output_dir}")


if __name__ == "__main__":
    main()
