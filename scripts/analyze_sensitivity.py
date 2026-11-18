"""Analyze Evidence Logit Biasing Sensitivity Study and lock hyperparameters by measured E2E throughput.

Reports BOTH quality (acceptance_eval, acceptance_prop, evidence_overlap_acceptance,
JSON/citation validity, verdict accuracy/macro-F1) and cost (E2E throughput, E2E latency/token,
decode throughput, target rounds, proposals/committed, draft latency, verification latency).

Selects configurations strictly by measured end-to-end throughput (or lowest E2E ms/tok),
NOT solely by highest acceptance rate. Reports paired cluster bootstrap CIs and two-sided
p-values honestly, including null or negative effects.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from spade.analysis.experiment_audit import (
    extract_claim_level_metrics,
    paired_cluster_bootstrap,
    select_best_by_throughput,
    summarize_and_audit_run,
)

DEFAULT_BETAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
DEFAULT_VARIANTS = ["static", "entropy"]


def format_beta_tag(beta0: float) -> str:
    return f"b{beta0:.2f}".replace(".", "p")


def _sha256_if_exists(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def analyze_sensitivity_runs(
    *,
    runs_root: Path,
    run_prefix: str,
    betas: list[float],
    variants: list[str],
    n_bootstrap: int = 10000,
    seed: int = 42,
    tuning_benchmark: Path | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    claim_maps: dict[str, dict[str, dict[str, Any]]] = {}

    for variant in variants:
        for beta0 in betas:
            run_id = f"{run_prefix}-{variant}-{format_beta_tag(beta0)}"
            run_dir = runs_root / run_id
            if not run_dir.exists():
                continue
            summary = summarize_and_audit_run(run_dir)
            summary["variant"] = variant
            summary["beta0"] = float(beta0)
            rows.append(summary)
            claim_maps[run_id] = extract_claim_level_metrics(run_dir)

    if not rows:
        raise FileNotFoundError(f"No sensitivity runs found under {runs_root} with prefix {run_prefix!r}")

    # Paired bootstrap vs each variant's beta0=0.0 baseline
    bootstrap_vs_zero: list[dict[str, Any]] = []
    boot_metrics = [
        "throughput_e2e_tok_s",
        "latency_e2e_ms_per_tok",
        "acceptance_eval_pct",
        "evidence_overlap_acceptance_pct",
        "acceptance_prop_pct",
        "proposals_per_committed",
        "target_rounds_per_claim",
    ]
    for variant in variants:
        zero_id = f"{run_prefix}-{variant}-{format_beta_tag(0.0)}"
        if zero_id not in claim_maps:
            continue
        base_claims = claim_maps[zero_id]
        for row in rows:
            if row["variant"] != variant or row["beta0"] == 0.0:
                continue
            treat_claims = claim_maps[row["run_id"]]
            per_metric = {
                m: paired_cluster_bootstrap(base_claims, treat_claims, m, n_bootstrap=n_bootstrap, seed=seed)
                for m in boot_metrics
            }
            bootstrap_vs_zero.append({
                "variant": variant,
                "beta0": row["beta0"],
                "baseline_run_id": zero_id,
                "treatment_run_id": row["run_id"],
                "metrics": per_metric,
            })

    overall_selection = select_best_by_throughput(rows)
    per_variant_selection: dict[str, Any] = {}
    for variant in variants:
        v_rows = [r for r in rows if r["variant"] == variant]
        if v_rows:
            per_variant_selection[variant] = select_best_by_throughput(v_rows)

    locked_hparams = {
        "protocol": "locked_before_heldout_final_evaluation",
        "selection_criterion": "max_measured_e2e_throughput_with_quality_guardrails",
        "tuning_run_prefix": run_prefix,
        "tuning_benchmark": str(tuning_benchmark.resolve()) if tuning_benchmark else None,
        "tuning_benchmark_sha256": _sha256_if_exists(tuning_benchmark),
        "draft_length": 4,
        "bias_only_beta0": (
            per_variant_selection["static"]["selected"]["beta0"]
            if "static" in per_variant_selection
            else overall_selection["selected"]["beta0"]
        ),
        "bias_entropy_beta0": (
            per_variant_selection["entropy"]["selected"]["beta0"]
            if "entropy" in per_variant_selection
            else overall_selection["selected"]["beta0"]
        ),
        "full_spade_beta0": overall_selection["selected"]["beta0"],
        "full_spade_adaptive_entropy": bool(overall_selection["selected"]["adaptive_entropy"]),
        "order_decay": 0.5,
        "entropy_threshold": 2.0,
        "max_boost_cap": 3.0,
        "selected_overall_run_id": overall_selection["selected"]["run_id"],
        "selected_differs_from_highest_acceptance": overall_selection["selected_differs_from_highest_acceptance"],
    }

    return {
        "run_prefix": run_prefix,
        "runs_audited": len(rows),
        "all_audits_passed": all(r["audit_passed"] for r in rows),
        "rows": rows,
        "overall_selection": overall_selection,
        "per_variant_selection": per_variant_selection,
        "bootstrap_vs_zero": bootstrap_vs_zero,
        "locked_hparams": locked_hparams,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        f"# Sensitivity Study Report (`{report['run_prefix']}`)",
        "",
        f"- Runs audited: `{report['runs_audited']}` | All audits passed: `{report['all_audits_passed']}`",
        f"- Selection criterion: `{report['overall_selection']['selection_criterion']}`",
        "",
        "## Quality & Cost Matrix (Fixed K=4)",
        "",
        "| Variant | beta_0 | E2E tok/s | Dec tok/s | E2E ms/tok | Tgt Rnds | Prop/Comm | Draft ms/Rnd | Verify ms/Rnd | alpha_eval | alpha_prop | alpha_ev | JSON % | Cit % | Acc % | Macro-F1 | Audit |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in report["rows"]:
        lines.append(
            f"| {r['variant']} | {r['beta0']:.2f} | {r['throughput_e2e_tok_s']:.2f} | "
            f"{r['throughput_decode_tok_s']:.2f} | {r['latency_e2e_ms_per_tok']:.2f} | "
            f"{r['target_rounds']} | {r['proposals_per_committed']} | "
            f"{r['draft_latency_ms_per_round']:.1f} | {r['verification_latency_ms_per_round']:.1f} | "
            f"{r['acceptance_eval_pct']}% | {r['acceptance_prop_pct']}% | "
            f"{r['evidence_overlap_acceptance_pct']}% | {r['json_valid_pct']}% | "
            f"{r['citation_valid_pct']}% | {r['verdict_accuracy_pct']} | {r['verdict_macro_f1']} | "
            f"{'PASS' if r['audit_passed'] else 'FAIL'} |"
        )

    sel = report["overall_selection"]["selected"]
    acc_best = report["overall_selection"]["highest_acceptance_for_comparison"]
    lines.extend([
        "",
        "## Selection Decision (Measured End-to-End Throughput)",
        "",
        f"- **Selected (Max E2E Throughput):** `{sel['run_id']}` (`variant={sel['variant']}`, `beta0={sel['beta0']}`, "
        f"`E2E={sel['throughput_e2e_tok_s']:.2f} tok/s`, `ms/tok={sel['latency_e2e_ms_per_tok']:.2f}`, "
        f"`alpha_eval={sel['acceptance_eval_pct']}%`, `alpha_ev={sel['evidence_overlap_acceptance_pct']}%`)",
    ])
    if acc_best is not None:
        lines.append(
            f"- **Highest Acceptance (Comparison Only):** `{acc_best['run_id']}` (`variant={acc_best['variant']}`, "
            f"`beta0={acc_best['beta0']}`, `alpha_eval={acc_best['acceptance_eval_pct']}%`, "
            f"`E2E={acc_best['throughput_e2e_tok_s']:.2f} tok/s`, `ms/tok={acc_best['latency_e2e_ms_per_tok']:.2f}`)"
        )
        lines.append(
            f"- Selected differs from highest-acceptance config: `{report['overall_selection']['selected_differs_from_highest_acceptance']}`"
        )

    if report["bootstrap_vs_zero"]:
        lines.extend([
            "",
            "## Paired Cluster Bootstrap vs beta_0 = 0.0 (Two-Sided, 95% CI)",
            "",
            "| Variant | beta_0 | Metric | Delta | 95% CI | Two-Sided p | Interpretation |",
            "| :--- | :---: | :--- | :---: | :---: | :---: | :--- |",
        ])
        for entry in report["bootstrap_vs_zero"]:
            for m_name, stats in entry["metrics"].items():
                lines.append(
                    f"| {entry['variant']} | {entry['beta0']:.2f} | `{m_name}` | "
                    f"{stats['delta']:+.4f} | [{stats['ci_95_low']:+.4f}, {stats['ci_95_high']:+.4f}] | "
                    f"{stats['p_value_two_sided']:.4f} | `{stats['interpretation']}` |"
                )

    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    p.add_argument("--run-prefix", default="sens-k4-tune100")
    p.add_argument("--betas", type=float, nargs="+", default=DEFAULT_BETAS)
    p.add_argument("--variants", nargs="+", choices=["static", "entropy"], default=DEFAULT_VARIANTS)
    p.add_argument("--tuning-benchmark", type=Path, default=None)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/sensitivity"))
    p.add_argument("--lock-config-out", type=Path, default=None,
                   help="Optional path to write locked hyperparameters JSON before final evaluation")
    args = p.parse_args()

    report = analyze_sensitivity_runs(
        runs_root=args.runs_root,
        run_prefix=args.run_prefix,
        betas=args.betas,
        variants=args.variants,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        tuning_benchmark=args.tuning_benchmark,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "sensitivity_report.json"
    md_path = args.output_dir / "sensitivity_report.md"
    csv_path = args.output_dir / "sensitivity_table.csv"

    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_text = render_markdown(report)
    md_path.write_text(md_text, encoding="utf-8")

    csv_fields = [
        "run_id", "variant", "beta0", "claims", "total_tokens",
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
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in report["rows"]:
            w.writerow({k: r.get(k) for k in csv_fields})

    if args.lock_config_out is not None:
        args.lock_config_out.parent.mkdir(parents=True, exist_ok=True)
        args.lock_config_out.write_text(json.dumps(report["locked_hparams"], indent=2) + "\n", encoding="utf-8")
        print(f"Locked hyperparameters written to {args.lock_config_out}")

    print(md_text)
    print(f"Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
