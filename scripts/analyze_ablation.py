"""Analyze 5-Way SPADE Ablation Suite with Paired Cluster Bootstrap (No p-hacking).

Evaluates:
  - B2 (K=4 baseline)
  - Bias-only (static evidence biasing, K=4)
  - Bias+Entropy (entropy-gated evidence biasing, K=4)
  - Dynamic-only (dynamic K in {2,4,6}, no biasing)
  - Full-SPADE (biasing + entropy gating + dynamic K)
  - Optional reference anchors: B0-S and HF-Assisted (K=4)

Reports exact effect sizes, 95% claim-cluster bootstrap confidence intervals, two-sided p-values,
and explicitly flags positive, negative, and inconclusive effects without hiding null results.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from spade.analysis.experiment_audit import (
    extract_claim_level_metrics,
    paired_cluster_bootstrap,
    summarize_and_audit_run,
)


def analyze_ablation_runs(
    *,
    runs_root: Path,
    run_prefix: str,
    draft_length: int = 4,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    candidates = [
        ("B0-S (Target-Only)", "b0s", f"{run_prefix}-b0s"),
        (f"HF-Assisted (K={draft_length})", "hf_assisted", f"{run_prefix}-hf-assisted-k{draft_length}"),
        (f"B2 (K={draft_length})", "b2", f"{run_prefix}-b2-k{draft_length}"),
        ("Bias-only", "bias_only", f"{run_prefix}-bias-only"),
        ("Bias+Entropy", "bias_entropy", f"{run_prefix}-bias-entropy"),
        ("Dynamic-only", "dynamic_only", f"{run_prefix}-dynamic-only"),
        ("Full SPADE", "full_spade", f"{run_prefix}-full-spade"),
    ]

    rows: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    claim_maps: dict[str, dict[str, dict[str, Any]]] = {}

    for label, key, run_id in candidates:
        run_dir = runs_root / run_id
        if not run_dir.exists():
            continue
        summary = summarize_and_audit_run(run_dir)
        summary["label"] = label
        summary["ablation_key"] = key
        rows.append(summary)
        by_key[key] = summary
        claim_maps[key] = extract_claim_level_metrics(run_dir)

    if not rows:
        raise FileNotFoundError(f"No ablation runs found under {runs_root} with prefix {run_prefix!r}")

    comparisons_spec = [
        ("b2", "bias_only", "Bias-only vs B2"),
        ("b2", "bias_entropy", "Bias+Entropy vs B2"),
        ("b2", "dynamic_only", "Dynamic-only vs B2"),
        ("b2", "full_spade", "Full SPADE vs B2"),
        ("bias_only", "bias_entropy", "Bias+Entropy vs Bias-only (Entropy Gating effect)"),
        ("bias_entropy", "full_spade", "Full SPADE vs Bias+Entropy (Dynamic K effect)"),
        ("dynamic_only", "full_spade", "Full SPADE vs Dynamic-only (Biasing effect under Dynamic K)"),
    ]
    if "b0s" in claim_maps:
        for k, name in [
            ("hf_assisted", "HF-Assisted vs B0-S"),
            ("b2", "B2 vs B0-S"),
            ("full_spade", "Full SPADE vs B0-S"),
        ]:
            if k in claim_maps:
                comparisons_spec.append(("b0s", k, name))

    spec_metrics = [
        "throughput_e2e_tok_s",
        "latency_e2e_ms_per_tok",
        "acceptance_eval_pct",
        "evidence_overlap_acceptance_pct",
        "acceptance_prop_pct",
        "proposals_per_committed",
        "target_rounds_per_claim",
        "draft_latency_ms_per_round",
    ]
    ref_metrics = [
        "throughput_e2e_tok_s",
        "latency_e2e_ms_per_tok",
        "target_rounds_per_claim",
    ]

    comparisons: list[dict[str, Any]] = []
    for base_key, treat_key, title in comparisons_spec:
        if base_key not in claim_maps or treat_key not in claim_maps:
            continue
        metrics_to_run = ref_metrics if base_key in {"b0s", "hf_assisted"} or treat_key in {"b0s", "hf_assisted"} else spec_metrics
        per_metric = {
            m: paired_cluster_bootstrap(
                claim_maps[base_key],
                claim_maps[treat_key],
                m,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            for m in metrics_to_run
        }
        comparisons.append({
            "comparison": title,
            "baseline_key": base_key,
            "treatment_key": treat_key,
            "metrics": per_metric,
        })

    return {
        "run_prefix": run_prefix,
        "runs_audited": len(rows),
        "all_audits_passed": all(r["audit_passed"] for r in rows),
        "rows": rows,
        "paired_comparisons": comparisons,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# 5-Way Ablation & Statistical Report (`{report['run_prefix']}`)",
        "",
        f"- Runs audited: `{report['runs_audited']}` | All audits passed: `{report['all_audits_passed']}`",
        "- Statistical method: Paired claim-cluster bootstrap (two-sided p-value, 95% percentile CI; null/negative results reported explicitly).",
        "",
        "## Quality & Cost Summary",
        "",
        "| Method | E2E tok/s | Dec tok/s | E2E ms/tok | Tgt Rnds | Fwd Red. | Prop/Comm | Draft ms/Rnd | Verify ms/Rnd | alpha_eval | alpha_prop | alpha_ev | JSON % | Cit % | Acc % | Macro-F1 | Audit |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in report["rows"]:
        a_eval = f"{r['acceptance_eval_pct']}%" if r["acceptance_eval_pct"] is not None else "—"
        a_prop = f"{r['acceptance_prop_pct']}%" if r["acceptance_prop_pct"] is not None else "—"
        a_ev = f"{r['evidence_overlap_acceptance_pct']}%" if r["evidence_overlap_acceptance_pct"] is not None else "—"
        p_comm = f"{r['proposals_per_committed']}" if r["proposals_per_committed"] is not None else "—"
        lines.append(
            f"| {r['label']} | {r['throughput_e2e_tok_s']:.2f} | {r['throughput_decode_tok_s']:.2f} | "
            f"{r['latency_e2e_ms_per_tok']:.2f} | {r['target_rounds']} | {r['target_fwd_reduction']:.2f}x | "
            f"{p_comm} | {r['draft_latency_ms_per_round']:.1f} | {r['verification_latency_ms_per_round']:.1f} | "
            f"{a_eval} | {a_prop} | {a_ev} | {r['json_valid_pct']}% | {r['citation_valid_pct']}% | "
            f"{r['verdict_accuracy_pct']} | {r['verdict_macro_f1']} | {'PASS' if r['audit_passed'] else 'FAIL'} |"
        )

    lines.extend([
        "",
        "## Paired Cluster Bootstrap Comparisons (Two-Sided, 95% CI)",
        "",
        "| Comparison | Metric | Base | Treat | Delta | 95% CI | Two-Sided p | W/L/T | Interpretation |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])
    for comp in report["paired_comparisons"]:
        for m_name, s in comp["metrics"].items():
            wlt = f"{s['claim_wins']}/{s['claim_losses']}/{s['claim_ties']}"
            lines.append(
                f"| {comp['comparison']} | `{m_name}` | {s['baseline']:.4f} | {s['treatment']:.4f} | "
                f"{s['delta']:+.4f} | [{s['ci_95_low']:+.4f}, {s['ci_95_high']:+.4f}] | "
                f"{s['p_value_two_sided']:.4f} | {wlt} | `{s['interpretation']}` |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    p.add_argument("--run-prefix", default="ablation-heldout400")
    p.add_argument("--draft-length", type=int, default=4)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/ablation"))
    args = p.parse_args()

    report = analyze_ablation_runs(
        runs_root=args.runs_root,
        run_prefix=args.run_prefix,
        draft_length=args.draft_length,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ablation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_text = render_markdown(report)
    (args.output_dir / "ablation_report.md").write_text(md_text, encoding="utf-8")

    csv_fields = [
        "label", "ablation_key", "run_id", "claims", "total_tokens",
        "throughput_e2e_tok_s", "throughput_decode_tok_s",
        "latency_e2e_ms_per_tok", "latency_decode_ms_per_tok",
        "target_rounds", "target_fwd_reduction", "proposals_per_committed",
        "draft_latency_ms_per_round", "draft_pct_decode",
        "verification_latency_ms_per_round", "verify_pct_decode",
        "acceptance_eval_pct", "acceptance_prop_pct",
        "evidence_overlap_acceptance_pct", "non_overlap_acceptance_pct",
        "json_valid_pct", "citation_valid_pct", "verdict_accuracy_pct", "verdict_macro_f1",
        "accounting_exact", "audit_passed", "git_commit", "git_dirty",
    ]
    with (args.output_dir / "ablation_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in report["rows"]:
            w.writerow({k: r.get(k) for k in csv_fields})

    print(md_text)
    print(f"Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
