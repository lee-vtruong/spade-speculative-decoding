"""Synthesize M1-M3 experimental artifacts into structured research reports."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .metrics import acceptance_summary
from .plots import analyze_runs

def generate_experiment_report(run_dirs: list[str | Path], output_dir: str | Path) -> dict[str, Any]:
    """Generate comprehensive JSON summary and Markdown research report from run directories."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = []
    rounds = []
    tokens = []
    for raw in run_dirs:
        root = Path(raw)
        cfg_path = root / "config_resolved.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        mode = cfg.get("mode", "unknown")
        r_traces = [json.loads(s) for s in (root / "round_traces.jsonl").read_text(encoding="utf-8").split("\n") if s.strip()] if (root / "round_traces.jsonl").exists() else []
        t_traces = [json.loads(s) for s in (root / "token_traces.jsonl").read_text(encoding="utf-8").split("\n") if s.strip()] if (root / "token_traces.jsonl").exists() else []
        for r in r_traces:
            r.setdefault("mode", mode)
        for t in t_traces:
            t.setdefault("mode", mode)
        runs.append({"root": str(root.resolve()), "config": cfg, "round_count": len(r_traces), "token_count": len(t_traces)})
        rounds.extend(r_traces)
        tokens.extend(t_traces)

    b2_tokens = [x for x in tokens if x.get("mode") == "B2"]
    b2_rounds = [x for x in rounds if x.get("mode") == "B2"]
    summary = acceptance_summary(b2_tokens, b2_rounds) if b2_tokens else {}

    plot_report = {}
    try:
        plot_report = analyze_runs(run_dirs, out / "plots")
    except Exception as exc:
        plot_report = {"plots_generated": [], "error": str(exc)}

    report = {
        "report_version": "1.0",
        "run_count": len(runs),
        "runs": runs,
        "total_rounds": len(rounds),
        "total_tokens_traced": len(tokens),
        "acceptance_summary": summary,
        "plot_artifacts": plot_report.get("plots_generated", []),
    }

    (out / "m1_m3_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Generate Markdown summary
    md_lines = [
        "# SPADE M1–M3 Experimental Synthesis Report",
        "",
        "## 1. Overview",
        f"- Analyzed runs: {len(runs)}",
        f"- Total speculative rounds logged: {len(rounds)}",
        f"- Total token traces logged: {len(tokens)}",
        "",
        "## 2. Analyzed Runs",
        "| Run ID | Mode | Target | Draft | Draft Length | Output Tokens |",
        "|---|---|---|---|---|---|",
    ]
    for r in runs:
        cfg = r["config"]
        md_lines.append(f"| `{cfg.get('run_id')}` | {cfg.get('mode')} | `{cfg.get('target')}` | `{cfg.get('draft') or 'N/A'}` | {cfg.get('draft_length', 'N/A')} | {cfg.get('max_new_tokens')} |")

    if summary:
        obs = summary.get("observed_acceptance", {})
        md_lines.extend([
            "",
            "## 3. Speculative Acceptance Diagnostics (B2 Exact Sampling)",
            f"- **Token-weighted acceptance rate**: {obs.get('token_weighted', 0.0):.4f}",
            f"- **Round-weighted acceptance rate**: {obs.get('round_weighted', 0.0):.4f}",
            f"- **Claim-weighted acceptance rate**: {obs.get('claim_weighted', 0.0):.4f}",
            f"- **Expected acceptance probability (1 - TV)**: {summary.get('expected_acceptance_probability_mean', 0.0) or 0.0:.4f}",
            f"- **Mean target probability p(x)**: {summary.get('mean_target_probability', 0.0) or 0.0:.4f}",
            f"- **Mean draft proposal probability q(x)**: {summary.get('mean_draft_probability', 0.0) or 0.0:.4f}",
            f"- **Mean p(x)/q(x) ratio**: {summary.get('mean_p_over_q', 0.0) or 0.0:.4f}",
            f"- **Mean Total Variation distance TV(p, q)**: {summary.get('mean_total_variation', 0.0) or 0.0:.4f}",
            f"- **Mean accepted draft tokens per round**: {summary.get('mean_accepted_tokens_per_round', 0.0):.2f}",
            f"- **Mean committed tokens per round**: {summary.get('mean_committed_tokens_per_round', 0.0):.2f}",
            "",
            "### Token Category Acceptance",
            "| Category | Proposals | Accepted | Acceptance Rate |",
            "|---|---|---|---|",
        ])
        for cat, stats in summary.get("token_category_acceptance", {}).items():
            md_lines.append(f"| {cat} | {stats.get('proposals')} | {stats.get('accepted')} | {stats.get('acceptance_rate', 0.0):.4f} |")

    plots = plot_report.get("plots_generated", [])
    if plots:
        md_lines.extend([
            "",
            "## 4. Generated Figures",
        ])
        for p in plots:
            md_lines.append(f"- `plots/{p}`")

    md_lines.extend([
        "",
        "## 5. Methodological Notes and Limitations",
        "- Timing measurements must distinguish full-prefix recomputation from optimized KV-cache decoding.",
        "- Lexical evidence overlap indicates token surface sharing, not semantic factual correctness.",
        "- Output factuality is reported as `not_evaluated` unless verified by dedicated fact-checking protocols.",
    ])

    (out / "m1_m3_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return report
