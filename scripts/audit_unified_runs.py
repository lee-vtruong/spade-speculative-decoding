"""Audit unified-v2 benchmark report and (optionally) raw run traces without modifying old data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spade.analysis.experiment_audit import audit_report_json, summarize_and_audit_run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report-json",
        type=Path,
        default=Path("outputs/unified_v2_report.json"),
        help="Path to unified benchmark report JSON (read-only)",
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Optional path to outputs/runs directory containing raw unified-v2-* run traces",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write audit result JSON (does NOT modify input report)",
    )
    args = p.parse_args()

    if not args.report_json.exists():
        raise SystemExit(f"Report JSON not found: {args.report_json}")

    report_audit = audit_report_json(args.report_json)
    raw_rows = json.loads(args.report_json.read_text(encoding="utf-8"))

    raw_trace_audits: list[dict] = []
    if args.runs_root is not None:
        for row in raw_rows:
            run_id = row["Run ID"]
            run_dir = args.runs_root / run_id
            if not run_dir.exists():
                raw_trace_audits.append({
                    "run_id": run_id,
                    "audit_passed": False,
                    "audit_errors": [f"Run directory not found: {run_dir}"],
                })
                continue
            raw_trace_audits.append(summarize_and_audit_run(run_dir, reference_row=row))

    all_passed = report_audit["all_passed"] and (
        all(r["audit_passed"] for r in raw_trace_audits) if raw_trace_audits else True
    )
    payload = {
        "report_audit": report_audit,
        "raw_trace_audits": raw_trace_audits,
        "all_passed": all_passed,
    }

    print("=" * 90)
    print(f"UNIFIED BENCHMARK AUDIT: {args.report_json}")
    print("=" * 90)
    print(f"Git Commit: {report_audit['git_commit']} | PyTorch: {report_audit['pytorch']} | Transformers: {report_audit['transformers']}")
    print(f"Global checks passed: {not report_audit['global_errors']}")
    if report_audit["global_errors"]:
        for err in report_audit["global_errors"]:
            print(f"  [FAIL] {err}")

    for chk in report_audit["method_checks"]:
        status = "PASS" if chk["passed"] else "FAIL"
        print(f"  [{status}] Report arithmetic: {chk['method']} ({chk['run_id']})")
        for err in chk["errors"]:
            print(f"         -> {err}")

    if raw_trace_audits:
        print("\nRaw Trace Reconciliation:")
        for r in raw_trace_audits:
            status = "PASS" if r["audit_passed"] else "FAIL"
            print(
                f"  [{status}] Raw traces: {r['run_id']} | tokens={r.get('total_tokens')} "
                f"rounds={r.get('target_rounds')} e2e_tok_s={r.get('throughput_e2e_tok_s')} "
                f"alpha_eval={r.get('acceptance_eval_pct')} alpha_prop={r.get('acceptance_prop_pct')}"
            )
            for err in r["audit_errors"]:
                print(f"         -> {err}")

    print(f"\nOVERALL AUDIT STATUS: {'PASS' if all_passed else 'FAIL'}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Audit written to {args.output}")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
