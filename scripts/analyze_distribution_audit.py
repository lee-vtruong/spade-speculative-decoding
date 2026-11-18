"""M4.1 Offline Analysis, Tables, and Real-Data Charts for Distribution-Level Evidence Biasing Audit.

Runs purely on CPU from prefix_audit_records.jsonl produced by scripts/run_distribution_audit.py.
Computes claim-clustered paired bootstrap CIs and two-sided p-values (never assuming independent
token samples), generates Markdown/JSON/CSV reports answering all 5 audit questions, and renders
5 real-data PNG diagnostic figures.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from spade.analysis.distribution_audit import build_distribution_audit_report


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def generate_figures(report: dict[str, Any], records: list[dict[str, Any]], fig_dir: Path) -> list[str]:
    """Generate 5 real-data PNG charts from actual M4.1 audit data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    primary = report["primary_config"]

    # Figure 1: Per-prefix delta expected acceptance (1 - TV) vs beta_0 across all audited variants
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=False)
    palette = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#386cb0"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    unique_variants = list(dict.fromkeys(s["variant"] for s in report["config_summaries"]))
    for idx_v, variant in enumerate(unique_variants):
        color = palette[idx_v % len(palette)]
        marker = markers[idx_v % len(markers)]
        v_rows = sorted(
            [s for s in report["config_summaries"] if s["variant"] == variant],
            key=lambda x: x["beta0"],
        )
        if not v_rows:
            continue
        xs = [r["beta0"] for r in v_rows]

        # Panel A: All prefixes
        ys_all = [r["delta_expected_acceptance_all_pp"] for r in v_rows]
        lo_all = [max(0.0, r["delta_expected_acceptance_all_pp"] - r["all_delta_expected_acceptance_bootstrap"]["ci_95_low"] * 100.0) for r in v_rows]
        hi_all = [max(0.0, r["all_delta_expected_acceptance_bootstrap"]["ci_95_high"] * 100.0 - r["delta_expected_acceptance_all_pp"]) for r in v_rows]
        axes[0].errorbar(xs, ys_all, yerr=[lo_all, hi_all], marker=marker, color=color, linewidth=1.8, capsize=3, label=variant)

        # Panel B: Biased prefixes only
        ys_b = [r["delta_expected_acceptance_biased_pp"] for r in v_rows]
        lo_b = [max(0.0, r["delta_expected_acceptance_biased_pp"] - r["biased_delta_expected_acceptance_bootstrap"]["ci_95_low"] * 100.0) for r in v_rows]
        hi_b = [max(0.0, r["biased_delta_expected_acceptance_bootstrap"]["ci_95_high"] * 100.0 - r["delta_expected_acceptance_biased_pp"]) for r in v_rows]
        axes[1].errorbar(xs, ys_b, yerr=[lo_b, hi_b], marker=marker, color=color, linewidth=1.8, capsize=3, label=variant)

    for ax, title in [(axes[0], "All Fixed Prefixes"), (axes[1], "Biased Prefixes Only")]:
        ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
        ax.set_xlabel(r"Evidence Bias Strength ($\beta_0$)")
        ax.set_ylabel(r"$\Delta$ Expected Acceptance $1 - \mathrm{TV}(p,q)$ (pp)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Per-Prefix Expected Acceptance Change vs. Bias Strength (Claim-Clustered 95% CI)", fontsize=11)
    fig.tight_layout()
    p1 = fig_dir / "fig1_delta_expected_acc_by_beta.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    saved.append(p1.name)

    # Figure 2: TV change & % TV reduced/increased across entropy buckets (biased prefixes)
    ent_rows = report["subgroups_primary"]["entropy_bucket_biased"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if ent_rows:
        labels = [r["bucket"] for r in ent_rows]
        x = list(range(len(labels)))
        ys = [r["delta_expected_acc_pp"] for r in ent_rows]
        lo = [max(0.0, r["delta_expected_acc_pp"] - r["delta_expected_acc_ci95_pp"][0]) for r in ent_rows]
        hi = [max(0.0, r["delta_expected_acc_ci95_pp"][1] - r["delta_expected_acc_pp"]) for r in ent_rows]
        colors = ["#1b9e77" if y >= 0 else "#d95f02" for y in ys]
        axes[0].bar(x, ys, yerr=[lo, hi], color=colors, capsize=4, alpha=0.85)
        axes[0].axhline(0.0, color="#444444", linestyle="--", linewidth=1)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=20, ha="right")
        axes[0].set_ylabel(r"$\Delta (1 - \mathrm{TV})$ on Biased Prefixes (pp)")
        axes[0].set_title(f"Expected Acceptance Change by Entropy ({primary})")
        axes[0].grid(axis="y", alpha=0.25)

        red = [r["pct_tv_reduced"] for r in ent_rows]
        inc = [r["pct_tv_increased"] for r in ent_rows]
        w = 0.38
        axes[1].bar([i - w / 2 for i in x], red, width=w, color="#1b9e77", label="% TV Reduced")
        axes[1].bar([i + w / 2 for i in x], inc, width=w, color="#d95f02", label="% TV Increased")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=20, ha="right")
        axes[1].set_ylabel("Share of Biased Prefixes (%)")
        axes[1].set_title("Share of Biased Prefixes Improved vs. Degraded")
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend()
    fig.tight_layout()
    p2 = fig_dir / "fig2_tv_change_by_entropy_bucket.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    saved.append(p2.name)

    # Figure 3: Match length & candidate count breakdown
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, axis_key, title in [
        (axes[0], "match_length", f"By N-Gram Match Order ({primary})"),
        (axes[1], "candidate_count", f"By Candidate Set Size ({primary})"),
    ]:
        rows = [r for r in report["subgroups_primary"][axis_key] if r["n_biased"] > 0]
        if rows:
            labels = [f"{r['bucket']}\n(n={r['n_prefixes']})" for r in rows]
            x = list(range(len(labels)))
            ys = [r["delta_expected_acc_pp"] for r in rows]
            lo = [max(0.0, r["delta_expected_acc_pp"] - r["delta_expected_acc_ci95_pp"][0]) for r in rows]
            hi = [max(0.0, r["delta_expected_acc_ci95_pp"][1] - r["delta_expected_acc_pp"]) for r in rows]
            cols = ["#1b9e77" if y >= 0 else "#d95f02" for y in ys]
            ax.bar(x, ys, yerr=[lo, hi], color=cols, capsize=4, alpha=0.85)
            ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel(r"$\Delta (1 - \mathrm{TV})$ (pp)")
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p3 = fig_dir / "fig3_tv_change_by_match_and_candidates.png"
    fig.savefig(p3, dpi=160)
    plt.close(fig)
    saved.append(p3.name)

    # Figure 4: Target probability mass on candidates vs delta(1 - TV)
    tm_rows = report["subgroups_primary"]["target_mass_biased"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if tm_rows:
        labels = [f"{r['bucket']}\n(n={r['n_prefixes']})" for r in tm_rows]
        x = list(range(len(labels)))
        ys = [r["delta_expected_acc_pp"] for r in tm_rows]
        lo = [max(0.0, r["delta_expected_acc_pp"] - r["delta_expected_acc_ci95_pp"][0]) for r in tm_rows]
        hi = [max(0.0, r["delta_expected_acc_ci95_pp"][1] - r["delta_expected_acc_pp"]) for r in tm_rows]
        cols = ["#1b9e77" if y >= 0 else "#d95f02" for y in ys]
        axes[0].bar(x, ys, yerr=[lo, hi], color=cols, capsize=4, alpha=0.85)
        axes[0].axhline(0.0, color="#444444", linestyle="--", linewidth=1)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=15, ha="right")
        axes[0].set_ylabel(r"$\Delta (1 - \mathrm{TV})$ on Biased Prefixes (pp)")
        axes[0].set_title(f"Effect by Target Mass on Candidates ({primary})")
        axes[0].grid(axis="y", alpha=0.25)

    biased_pts = [r for r in records if r["configs"][primary]["is_biased"]]
    if biased_pts:
        xs_sc = [r["candidate_mass_deficit"] for r in biased_pts]
        ys_sc = [r["configs"][primary]["delta_expected_acceptance"] * 100.0 for r in biased_pts]
        axes[1].scatter(xs_sc, ys_sc, s=8, alpha=0.25, color="#386cb0", edgecolors="none")
        axes[1].axhline(0.0, color="#444444", linestyle="--", linewidth=1)
        axes[1].axvline(0.0, color="#444444", linestyle="--", linewidth=1)
        axes[1].set_xlabel(r"Candidate Mass Deficit: $p(S_E) - q_{\mathrm{base}}(S_E)$")
        axes[1].set_ylabel(r"Per-Prefix $\Delta (1 - \mathrm{TV})$ (pp)")
        axes[1].set_title("Mass Deficit vs. Per-Prefix Expected Acceptance Gain")
        axes[1].grid(alpha=0.25)
    fig.tight_layout()
    p4 = fig_dir / "fig4_target_mass_vs_tv_delta.png"
    fig.savefig(p4, dpi=160)
    plt.close(fig)
    saved.append(p4.name)

    # Figure 5: Evidence overlap & semantic category breakdown
    combined_rows = report["subgroups_primary"]["evidence_overlap"] + report["subgroups_primary"]["semantic_category"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if combined_rows:
        labels = [f"{r['bucket']} (n={r['n_prefixes']})" for r in combined_rows]
        y_pos = list(range(len(labels)))
        xs = [r["delta_expected_acc_pp"] for r in combined_rows]
        lo = [max(0.0, r["delta_expected_acc_pp"] - r["delta_expected_acc_ci95_pp"][0]) for r in combined_rows]
        hi = [max(0.0, r["delta_expected_acc_ci95_pp"][1] - r["delta_expected_acc_pp"]) for r in combined_rows]
        cols = ["#1b9e77" if v >= 0 else "#d95f02" for v in xs]
        ax.barh(y_pos, xs, xerr=[lo, hi], color=cols, capsize=4, alpha=0.85)
        ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(r"$\Delta$ Expected Acceptance $1 - \mathrm{TV}(p,q)$ (pp, Claim-Clustered 95% CI)")
        ax.set_title(f"Subgroup Breakdown: Overlap & Semantic Categories ({primary})")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    p5 = fig_dir / "fig5_subgroup_breakdown.png"
    fig.savefig(p5, dpi=160)
    plt.close(fig)
    saved.append(p5.name)

    return saved


def render_markdown(report: dict[str, Any], metadata: dict[str, Any], figure_names: list[str]) -> str:
    q = report["audit_questions"]
    primary = report["primary_config"]
    gate = report["decision_gate"]

    lines: list[str] = [
        "# M4.1 — Distribution-Level Evidence Biasing Audit Report",
        "",
        f"- **Schema Version:** `{report['schema_version']}`",
        f"- **Claims Audited:** `{report['n_claims']}` | **Fixed Prefixes Evaluated:** `{report['n_prefixes']}` | **Bootstrap Unit:** `{report['cluster_unit']}` ($B={report['n_bootstrap']}$, two-sided 95% CI)",
        f"- **Primary Audited Config:** `{primary}` | **Git Commit:** `{(metadata.get('git_commit') or 'N/A')[:7]}` (`git_dirty={metadata.get('git_dirty')}`)",
        f"- **Source Fixed Prefixes (Read-Only):** `{metadata.get('source_predictions_path', 'N/A')}`",
        "",
        "> **Metric Separation Notice:** Per-prefix **expected acceptance** ($1 - \\mathrm{TV}(p, q) = \\sum_v \\min(p(v), q(v))$) is a deterministic functional of $(p, q)$ on fixed teacher-forced prefixes. It is strictly distinct from **observed stochastic acceptance** measured in separate speculative rollouts.",
        "",
        "## 1. Per-Configuration Summary (All Fixed Prefixes & Biased-Only)",
        "",
        "| Config | Variant | $\\beta_0$ | % Biased | % Biased TV$\\downarrow$ | % Biased TV$\\uparrow$ | Base $1-\\mathrm{TV}$ | All $\\Delta(1-\\mathrm{TV})$ (pp) | All 95% CI (pp) | $p$ (2-sided) | Biased $\\Delta(1-\\mathrm{TV})$ (pp) | All $\\Delta\\mathrm{JS}$ | Mean Overhead ($\\mu$s/prefix) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s in report["config_summaries"]:
        b_all = s["all_delta_expected_acceptance_bootstrap"]
        b_js = s["all_delta_js_bootstrap"]
        lines.append(
            f"| `{s['config_name']}` | {s['variant']} | {s['beta0']:.2f} | {s['pct_prefixes_biased']:.2f}% | "
            f"{s['pct_biased_tv_reduced']:.2f}% | {s['pct_biased_tv_increased']:.2f}% | "
            f"{s['expected_acceptance_base_pct']:.3f}% | {s['delta_expected_acceptance_all_pp']:+.4f} | "
            f"[{b_all['ci_95_low']*100:+.4f}, {b_all['ci_95_high']*100:+.4f}] | {b_all['p_value_two_sided']:.4f} | "
            f"{s['delta_expected_acceptance_biased_pp']:+.4f} | {b_js['mean']:+.6f} | {s['mean_bias_overhead_us_all']:.1f} |"
        )

    q1 = q["q1_pct_prefixes_biased"]
    q2 = q["q2_pct_prefixes_tv_reduced"]
    q3_inc = q["q3_where_biasing_increases_tv"]["tv_increased_profile"]
    q3_red = q["q3_where_biasing_increases_tv"]["tv_reduced_profile"]
    q5 = q["q5_overhead_vs_expected_benefit"]

    lines.extend([
        "",
        f"## 2. Direct Answers to the 5 Core Audit Questions (`{primary}`)",
        "",
        "### Q1. What percentage of prefixes are biased?",
        f"- **Pooled:** `{q1['pct_prefixes_biased'] if 'pct_prefixes_biased' in q1 else q1['pooled_pct']:.2f}%` ({q1['n_biased']} / {q1['n_total']} prefixes).",
        f"- **Per-claim mean ± std:** `{q1['per_claim_mean_pct']:.2f}% ± {q1['per_claim_std_pct']:.2f}%`.",
        "",
        "### Q2. What percentage of prefixes have TV reduced vs. increased?",
        f"- **Among ALL prefixes:** `{q2['pct_of_all_tv_reduced']:.2f}%` TV reduced, `{q2['pct_of_all_tv_increased']:.2f}%` TV increased; mean $\\Delta(1-\\mathrm{{TV}}) = {q2['mean_delta_expected_acc_all_pp']:+.4f}$ pp (claim-clustered 95% CI `[{q2['all_ci95_pp'][0]:+.4f}, {q2['all_ci95_pp'][1]:+.4f}]`, two-sided $p={q2['all_p_two_sided']:.4f}$).",
        f"- **Among BIASED prefixes only:** `{q2['pct_of_biased_tv_reduced']:.2f}%` TV reduced, `{q2['pct_of_biased_tv_increased']:.2f}%` TV increased, `{q2['pct_of_biased_tv_unchanged']:.2f}%` unchanged; mean $\\Delta(1-\\mathrm{{TV}}) = {q2['mean_delta_expected_acc_biased_pp']:+.4f}$ pp.",
        "",
        "### Q3. In which cases does biasing increase TV?",
        "| Profile Metric | TV Increased (Degraded) | TV Reduced (Improved) |",
        "| :--- | :---: | :---: |",
        f"| Prefix Count | {q3_inc.get('count', 0)} | {q3_red.get('count', 0)} |",
        f"| Mean $\\Delta\\mathrm{{TV}}$ | {q3_inc.get('mean_delta_tv', 0.0):+.6f} | {q3_red.get('mean_delta_tv', 0.0):+.6f} |",
        f"| Mean Target Mass on Candidates $p(S_E)$ | {q3_inc.get('mean_target_mass_cand', 0.0):.4f} | {q3_red.get('mean_target_mass_cand', 0.0):.4f} |",
        f"| Mean Draft Base Mass $q_{{base}}(S_E)$ | {q3_inc.get('mean_draft_base_mass_cand', 0.0):.4f} | {q3_red.get('mean_draft_base_mass_cand', 0.0):.4f} |",
        f"| Mean Mass Deficit $p(S_E) - q_{{base}}(S_E)$ | {q3_inc.get('mean_candidate_mass_deficit', 0.0):+.4f} | {q3_red.get('mean_candidate_mass_deficit', 0.0):+.4f} |",
        f"| % Target Argmax in Candidates | {q3_inc.get('pct_target_argmax_in_cand', 0.0):.2f}% | {q3_red.get('pct_target_argmax_in_cand', 0.0):.2f}% |",
        f"| Mean N-gram Match Order | {q3_inc.get('mean_matched_order', 0.0):.3f} | {q3_red.get('mean_matched_order', 0.0):.3f} |",
        f"| % Unigram Matches (order=1) | {q3_inc.get('pct_order_1', 0.0):.2f}% | {q3_red.get('pct_order_1', 0.0):.2f}% |",
        f"| Mean Candidate Count $|S_E|$ | {q3_inc.get('mean_candidate_count', 0.0):.2f} | {q3_red.get('mean_candidate_count', 0.0):.2f} |",
        f"| Mean Draft Entropy (nats) | {q3_inc.get('mean_entropy_nats', 0.0):.4f} | {q3_red.get('mean_entropy_nats', 0.0):.4f} |",
        "",
        "### Q4. Does draft entropy actually predict biasing benefit?",
    ])
    for label, cinfo in q["q4_does_entropy_predict_benefit"]["correlations"].items():
        corr = cinfo["correlation_entropy_vs_benefit"]
        lines.append(
            f"- **`{cinfo['config_name']}` ({label}):** Claim-clustered Spearman $r = {corr['spearman_r']:+.4f}$ "
            f"(95% CI `[{corr['spearman_ci_95'][0]:+.4f}, {corr['spearman_ci_95'][1]:+.4f}]`, $p={corr['spearman_p_two_sided']:.4f}$); "
            f"Pearson $r = {corr['pearson_r']:+.4f}$ (95% CI `[{corr['pearson_ci_95'][0]:+.4f}, {corr['pearson_ci_95'][1]:+.4f}]`, $p={corr['pearson_p_two_sided']:.4f}$)."
        )

    if report["entropy_vs_static_matched_beta"]:
        lines.extend([
            "",
            "**Direct Paired Comparison: Entropy-Gated vs. Static at Matched $\\beta_0$:**",
            "| $\\beta_0$ | All Prefixes $\\Delta(1-\\mathrm{TV})$ (pp) | All 95% CI | $p$ | Biased-Only $\\Delta(1-\\mathrm{TV})$ (pp) | Biased 95% CI | $p$ |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for row in report["entropy_vs_static_matched_beta"]:
            lines.append(
                f"| {row['beta0']:.2f} | {row['all_prefixes_delta_exp_acc_pp']:+.4f} | "
                f"[{row['all_prefixes_ci95_pp'][0]:+.4f}, {row['all_prefixes_ci95_pp'][1]:+.4f}] | {row['all_prefixes_p_two_sided']:.4f} | "
                f"{row['biased_only_delta_exp_acc_pp']:+.4f} | "
                f"[{row['biased_only_ci95_pp'][0]:+.4f}, {row['biased_only_ci95_pp'][1]:+.4f}] | {row['biased_only_p_two_sided']:.4f} |"
            )

    lines.extend([
        "",
        "### Q5. Is biasing overhead worth the expected benefit?",
        f"- **Expected acceptance shift:** `{q5['expected_acceptance_base']*100:.3f}%` $\\to$ `{q5['expected_acceptance_biased']*100:.3f}%` (`{q5['delta_expected_acceptance_pp']:+.4f}` pp).",
        f"- **Theoretical committed tokens/round ($K={q5['draft_length_k']}$):** `{q5['theoretical_committed_per_round_base']:.4f}` $\\to$ `{q5['theoretical_committed_per_round_biased']:.4f}` (`{q5['theoretical_round_reduction_pct']:+.4f}%` round reduction).",
        f"- **Measured lookup+biasing overhead:** `{q5['measured_bias_overhead_us_per_prefix']:.1f} us/prefix` (`{q5['measured_bias_overhead_ms_per_k4_round']:.4f} ms` per $K=4$ round).",
        f"- **Eager PyTorch break-even ({q5['eager_baseline_round_ms']} ms/round):** expected saved `{q5['eager_expected_saved_ms_per_round']:+.4f} ms/round`, net `{q5['eager_net_gain_ms_per_round']:+.4f} ms/round`.",
        f"- **Fused kernel break-even ({q5['fused_baseline_round_ms']} ms/round):** expected saved `{q5['fused_expected_saved_ms_per_round']:+.4f} ms/round`, net `{q5['fused_net_gain_ms_per_round']:+.4f} ms/round`.",
        "",
        f"## 3. Subgroup Breakdown (`{primary}`, Claim-Clustered 95% CI)",
        "",
        "| Axis | Bucket | n | % Biased | % TV$\\downarrow$ | % TV$\\uparrow$ | Base $1-\\mathrm{TV}$ | $\\Delta(1-\\mathrm{TV})$ (pp) | 95% CI (pp) | $p$ | Mean $p(S_E)$ | Mean $q(S_E)$ | Interpretation |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])
    for axis_name, rows in report["subgroups_primary"].items():
        for r in rows:
            lines.append(
                f"| `{axis_name}` | {r['bucket']} | {r['n_prefixes']} | {r['pct_biased']:.1f}% | "
                f"{r['pct_tv_reduced']:.1f}% | {r['pct_tv_increased']:.1f}% | {r['expected_acc_base_pct']:.2f}% | "
                f"{r['delta_expected_acc_pp']:+.4f} | [{r['delta_expected_acc_ci95_pp'][0]:+.4f}, {r['delta_expected_acc_ci95_pp'][1]:+.4f}] | "
                f"{r['p_value_two_sided']:.4f} | {r['mean_target_mass_cand']:.3f} | {r['mean_draft_base_mass_cand']:.3f} | `{r['interpretation']}` |"
            )

    if report["expected_vs_observed_comparison"]:
        lines.extend([
            "",
            "## 4. Per-Prefix Expected Acceptance vs. Observed Stochastic Acceptance (Strictly Separated)",
            "",
            "| Config | Per-Prefix Expected $\\Delta(1-\\mathrm{TV})$ (pp, Fixed Prefixes) | Observed Stochastic $\\Delta\\alpha_{\\mathrm{eval}}$ (pp, Rollouts) | Observed E2E tok/s |",
            "| :--- | :---: | :---: | :---: |",
        ])
        for row in report["expected_vs_observed_comparison"]:
            obs_d = f"{row['observed_stochastic_delta_alpha_eval_pp']:+.3f}" if row["observed_stochastic_delta_alpha_eval_pp"] is not None else "N/A"
            lines.append(
                f"| `{row['config_name']}` | {row['per_prefix_delta_expected_acc_pp']:+.4f} | {obs_d} | {row['observed_e2e_throughput_tok_s']} |"
            )

    lines.extend([
        "",
        "## 5. Pre-Registered Decision Gate Evaluation",
        "",
        f"- **Gate Verdict:** **`{gate['verdict']}`**",
        f"- **Best Config by Per-Prefix Expected Acceptance:** `{gate['best_config_by_expected_acceptance']}` (`{gate['best_delta_expected_acc_all_pp']:+.4f}` pp, 95% CI `[{gate['best_delta_expected_acc_ci95_pp'][0]:+.4f}, {gate['best_delta_expected_acc_ci95_pp'][1]:+.4f}]`, two-sided $p={gate['best_p_two_sided']:.4f}$).",
        f"- **Strong Positive Subgroups ($\\ge +0.50$ pp, 95% CI $> 0$):** `{len(gate['strong_positive_subgroups'])}`",
        "",
        "### Generated Figures",
        *[f"- `figures/{fn}`" for fn in figure_names],
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-dir", type=Path, default=Path("outputs/analysis/m4_1_distribution_audit"))
    p.add_argument("--primary-config", default=None, help="Primary config name for deep subgroup breakdown (default: entropy-b1p50)")
    p.add_argument("--sensitivity-report", type=Path, default=None,
                   help="Optional read-only sensitivity_report.json for expected-vs-observed separation table")
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=None, help="Defaults to --audit-dir")
    args = p.parse_args()

    out_dir = args.output_dir or args.audit_dir
    records_path = args.audit_dir / "prefix_audit_records.jsonl"
    meta_path = args.audit_dir / "audit_metadata.json"

    if not records_path.exists():
        raise SystemExit(f"Audit records not found: {records_path}")

    records = load_jsonl(records_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    sens_report = (
        json.loads(args.sensitivity_report.read_text(encoding="utf-8"))
        if (args.sensitivity_report is not None and args.sensitivity_report.exists())
        else None
    )

    report = build_distribution_audit_report(
        records,
        primary_config=args.primary_config,
        sensitivity_report=sens_report,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_names = generate_figures(report, records, out_dir / "figures")
    report["figures"] = fig_names
    report["metadata"] = metadata

    (out_dir / "distribution_audit_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_text = render_markdown(report, metadata, fig_names)
    (out_dir / "distribution_audit_report.md").write_text(md_text, encoding="utf-8")

    # Export CSVs
    with (out_dir / "summary_by_config.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "config_name", "variant", "beta0", "n_prefixes_total", "n_prefixes_biased",
            "pct_prefixes_biased", "pct_biased_tv_reduced", "pct_biased_tv_increased",
            "expected_acceptance_base_pct", "expected_acceptance_biased_pct",
            "delta_expected_acceptance_all_pp", "delta_expected_acceptance_biased_pp",
            "tv_base_mean", "tv_biased_mean", "js_base_mean", "js_biased_mean",
            "mean_target_mass_on_candidates_biased", "mean_draft_base_mass_on_candidates_biased",
            "mean_draft_biased_mass_on_candidates_biased", "mean_gating_value_biased",
            "mean_applied_boost_biased", "mean_bias_overhead_us_all",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in report["config_summaries"]:
            w.writerow({k: s.get(k) for k in fields})

    with (out_dir / "subgroups_primary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "group_axis", "bucket", "biased_only", "n_prefixes", "n_biased", "pct_biased",
            "pct_tv_reduced", "pct_tv_increased", "expected_acc_base_pct",
            "delta_expected_acc_pp", "p_value_two_sided", "interpretation",
            "mean_target_mass_cand", "mean_draft_base_mass_cand", "mean_entropy_nats", "n_claims_active",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rows in report["subgroups_primary"].values():
            for r in rows:
                w.writerow({k: r.get(k) for k in fields})

    worst_rows = report["audit_questions"]["q3_where_biasing_increases_tv"]["worst_20_cases"]
    if worst_rows:
        with (out_dir / "worst_tv_increase_cases.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(worst_rows[0].keys()))
            w.writeheader()
            w.writerows(worst_rows)

    print(md_text)
    print(f"Audit report, CSVs, and {len(fig_names)} figures written to {out_dir}")


if __name__ == "__main__":
    main()
