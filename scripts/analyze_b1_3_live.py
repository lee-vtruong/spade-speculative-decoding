"""Step B1.3c / B1.4 Live Multi-Arm Evaluation & Break-Even Analysis.

Pure CPU analysis for comparing B0-S, Baseline Full, BM25 Top-2, Compact,
BM25 Sentence-Budget, and Target-Attn Sentence-Budget under the verified B1.1 runtime.

Reports:
  - Per-arm audit integrity, JSON/citation validity, one-time setup timings (separated)
  - 100% additive E2E decomposition in BOTH ms/claim AND normalized ms/tok
  - Active Draft KV, Static Draft KV Buffer, and Peak GPU VRAM
  - Claim-clustered paired bootstrap (B=10,000, strictly by claim_id) vs Full, vs Top-2, vs B0-S.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from spade.analysis.experiment_audit import load_jsonl, summarize_and_audit_run


def _extract_claim_map(run_dir: Path) -> dict[str, dict[str, Any]]:
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
            "draft_kv_static_mb": float(t.get("draft_kv_static_capacity_bytes", 0.0)) / (1024.0 * 1024.0),
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


def _pooled_metric(rows: list[dict[str, Any]], metric: str) -> float:
    n = len(rows)
    if n == 0:
        return 0.0
    if metric == "throughput_e2e_tok_s":
        den = sum(r["e2e_s"] for r in rows)
        return sum(r["output_tokens"] for r in rows) / den if den > 0 else 0.0
    if metric == "throughput_decode_tok_s":
        den = sum(r["decode_s"] for r in rows)
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
    raise ValueError(f"Unknown metric: {metric}")


def _paired_bootstrap(
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
    b_val = _pooled_metric(b_list, metric)
    t_val = _pooled_metric(t_list, metric)
    obs_delta = t_val - b_val

    rng = random.Random(seed)
    n = len(common)
    boots: list[float] = []
    for _ in range(n_bootstrap):
        idxs = [rng.randrange(n) for _ in range(n)]
        boots.append(
            _pooled_metric([t_list[i] for i in idxs], metric)
            - _pooled_metric([b_list[i] for i in idxs], metric)
        )
    boots.sort()
    lo = boots[int(0.025 * n_bootstrap)]
    hi = boots[min(n_bootstrap - 1, int(0.975 * n_bootstrap))]
    p2 = min(
        1.0,
        2.0
        * min(
            sum(1 for x in boots if x <= 0.0) / n_bootstrap,
            sum(1 for x in boots if x >= 0.0) / n_bootstrap,
        ),
    )
    return {
        "metric": metric,
        "n_claims": n,
        "baseline": round(b_val, 4),
        "treatment": round(t_val, 4),
        "delta": round(obs_delta, 4),
        "ci_95_low": round(lo, 4),
        "ci_95_high": round(hi, 4),
        "p_value_two_sided": round(p2, 4),
    }


def _summarize_arm(label: str, key: str, run_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    base_sum = summarize_and_audit_run(run_dir)
    env = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    cfg = json.loads((run_dir / "config_resolved.json").read_text(encoding="utf-8"))
    cmap = _extract_claim_map(run_dir)
    c_list = list(cmap.values())
    n_c = len(c_list)

    tot_tok = sum(c["output_tokens"] for c in c_list)
    tot_e2e_s = sum(c["e2e_s"] for c in c_list)
    tot_dec_s = sum(c["decode_s"] for c in c_list)
    tot_rounds = sum(c["target_rounds"] for c in c_list)
    tot_eval = sum(c["n_eval"] for c in c_list)
    tot_eval_acc = sum(c["n_eval_acc"] for c in c_list)
    tot_prop = sum(c["n_prop"] for c in c_list)
    tot_acc = sum(c["n_acc"] for c in c_list)

    sel_ms_c = sum(c["context_selection_s"] for c in c_list) * 1000.0 / n_c
    t_pref_ms_c = sum(c["prompt_prefill_s"] for c in c_list) * 1000.0 / n_c
    d_pref_ms_c = sum(c["draft_prefill_s"] for c in c_list) * 1000.0 / n_c
    d_dec_ms_c = sum(c["draft_decode_s"] for c in c_list) * 1000.0 / n_c
    ver_ms_c = sum(c["verify_s"] for c in c_list) * 1000.0 / n_c
    corr_ms_c = sum(c["correction_s"] for c in c_list) * 1000.0 / n_c
    unattr_ms_c = sum(c["unattributed_decode_s"] for c in c_list) * 1000.0 / n_c
    e2e_ms_c = tot_e2e_s * 1000.0 / n_c
    sum_parts_c = sel_ms_c + t_pref_ms_c + d_pref_ms_c + d_dec_ms_c + ver_ms_c + corr_ms_c + unattr_ms_c

    sel_ms_t = sum(c["context_selection_s"] for c in c_list) * 1000.0 / tot_tok
    t_pref_ms_t = sum(c["prompt_prefill_s"] for c in c_list) * 1000.0 / tot_tok
    d_pref_ms_t = sum(c["draft_prefill_s"] for c in c_list) * 1000.0 / tot_tok
    d_dec_ms_t = sum(c["draft_decode_s"] for c in c_list) * 1000.0 / tot_tok
    ver_ms_t = sum(c["verify_s"] for c in c_list) * 1000.0 / tot_tok
    corr_ms_t = sum(c["correction_s"] for c in c_list) * 1000.0 / tot_tok
    unattr_ms_t = sum(c["unattributed_decode_s"] for c in c_list) * 1000.0 / tot_tok
    e2e_ms_t = tot_e2e_s * 1000.0 / tot_tok

    peak_alloc_gb = 0.0
    if env.get("gpu_peak_allocated_bytes"):
        peak_alloc_gb = max(env["gpu_peak_allocated_bytes"]) / (1024.0 ** 3)

    mean_t_tok = sum(c["target_prompt_tokens"] for c in c_list) / n_c
    mean_d_tok = sum(c["draft_prompt_tokens"] for c in c_list) / n_c
    mean_d_kv_peak = sum(c["draft_kv_peak_mb"] for c in c_list) / n_c
    mean_d_kv_static = sum(c["draft_kv_static_mb"] for c in c_list) / n_c

    row = {
        "key": key,
        "label": label,
        "run_id": run_dir.name,
        "audit_passed": base_sum["audit_passed"],
        "git_dirty": env.get("git_dirty"),
        "git_commit": (env.get("git_commit") or "")[:7],
        "draft_runtime": cfg.get("draft_runtime"),
        "distribution_mode": cfg.get("distribution_mode"),
        "draft_max_cache_len": cfg.get("draft_max_cache_len"),
        "draft_setup_timings": env.get("draft_setup_timings"),
        "n_claims": n_c,
        "mean_output_tokens": round(tot_tok / n_c, 2),
        "total_output_tokens": tot_tok,
        "json_parse_pct": round(float(base_sum.get("json_valid_pct", 0.0)), 1),
        "citation_valid_pct": round(float(base_sum.get("citation_valid_pct", 0.0)), 1),
        "verdict_accuracy_pct": base_sum.get("verdict_accuracy_pct"),
        "verdict_macro_f1": base_sum.get("verdict_macro_f1"),
        "mean_target_prompt_tokens": round(mean_t_tok, 1),
        "mean_draft_prompt_tokens": round(mean_d_tok, 1),
        "prompt_compression_pct": round((1.0 - mean_d_tok / mean_t_tok) * 100.0, 1) if mean_t_tok > 0 else 0.0,
        "draft_kv_peak_mb": round(mean_d_kv_peak, 2),
        "draft_kv_static_mb": round(mean_d_kv_static, 2),
        "gpu_peak_allocated_gb": round(peak_alloc_gb, 3),
        "observed_alpha_eval_pct": round((tot_eval_acc / tot_eval) * 100.0, 2) if tot_eval > 0 else 0.0,
        "observed_alpha_prop_pct": round((tot_acc / tot_prop) * 100.0, 2) if tot_prop > 0 else 0.0,
        "committed_per_round": round(tot_tok / tot_rounds, 3) if tot_rounds > 0 else 0.0,
        "target_rounds_per_claim": round(tot_rounds / n_c, 2),
        "draft_decode_ms_per_round": round(sum(c["draft_decode_s"] for c in c_list) * 1000.0 / tot_rounds, 2) if tot_rounds > 0 else 0.0,
        "throughput_decode_tok_s": round(tot_tok / tot_dec_s, 2) if tot_dec_s > 0 else 0.0,
        "throughput_e2e_tok_s": round(tot_tok / tot_e2e_s, 2) if tot_e2e_s > 0 else 0.0,
        "latency_e2e_ms_per_tok": round(e2e_ms_t, 2),
        "e2e_ms_per_claim": round(e2e_ms_c, 2),
        "decomp_ms_claim": {
            "sel": round(sel_ms_c, 2),
            "target_prefill": round(t_pref_ms_c, 2),
            "draft_prefill": round(d_pref_ms_c, 2),
            "draft_decode": round(d_dec_ms_c, 2),
            "target_verify": round(ver_ms_c, 2),
            "correction": round(corr_ms_c, 2),
            "unattr_decode": round(unattr_ms_c, 2),
            "e2e_total": round(e2e_ms_c, 2),
            "residual": round(e2e_ms_c - sum_parts_c, 4),
        },
        "decomp_ms_tok": {
            "sel": round(sel_ms_t, 3),
            "target_prefill": round(t_pref_ms_t, 3),
            "draft_prefill": round(d_pref_ms_t, 3),
            "draft_decode": round(d_dec_ms_t, 3),
            "target_verify": round(ver_ms_t, 3),
            "correction": round(corr_ms_t, 3),
            "unattr_decode": round(unattr_ms_t, 3),
            "e2e_total": round(e2e_ms_t, 3),
        },
    }
    return row, cmap


def render_markdown(report: dict[str, Any]) -> str:
    arms = report["arms"]
    lines: list[str] = [
        f"# {report['title']}",
        "",
        f"- Claims: `{arms[0]['n_claims']}` | Paired cluster bootstrap: `B={report['n_bootstrap']}` by `claim_id`",
        "",
        "## 1. End-to-End Performance, Memory & Correctness Summary",
        "",
        "| Arm | Audit / Dirty | Draft Tok (Comp%) | Active Draft KV (MB) | Static Buf (MB) | GPU Peak (GB) | Out Tok | $\\alpha_{\\text{eval}}$ (%) | Comm/Rnd | Draft Dec (ms/rnd) | Decode tok/s | **E2E tok/s** | **E2E ms/tok** |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for a in arms:
        st = f"{'PASS' if a['audit_passed'] else 'FAIL'} / {str(a['git_dirty']).lower()}"
        d_str = f"{a['mean_draft_prompt_tokens']:.1f} (-{a['prompt_compression_pct']:.1f}%)" if a["key"] != "b0s" else "—"
        kv_str = f"{a['draft_kv_peak_mb']:.1f}" if a["key"] != "b0s" else "—"
        sb_str = f"{a['draft_kv_static_mb']:.1f}" if a["draft_kv_static_mb"] > 0 else "—"
        al_str = f"{a['observed_alpha_eval_pct']:.2f}%" if a["key"] != "b0s" else "—"
        dd_str = f"{a['draft_decode_ms_per_round']:.2f}" if a["key"] != "b0s" else "—"
        lines.append(
            f"| `{a['label']}` | {st} | {d_str} | {kv_str} | {sb_str} | {a['gpu_peak_allocated_gb']:.2f} | "
            f"{a['mean_output_tokens']:.2f} | {al_str} | {a['committed_per_round']:.3f} | {dd_str} | "
            f"{a['throughput_decode_tok_s']:.2f} | **{a['throughput_e2e_tok_s']:.2f}** | **{a['latency_e2e_ms_per_tok']:.2f}** |"
        )

    lines += [
        "",
        "## 2. 100% Additive E2E Decomposition — Normalized `ms/tok` (Length-Invariant)",
        "",
        "| Arm | Sel | Target Prefill | Draft Prefill | Draft Decode | Target Verify | Correction | Unattr Dec | **Total E2E ms/tok** | $\\Delta$ vs Full |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    full_t = next((a["decomp_ms_tok"]["e2e_total"] for a in arms if a["key"] == "full"), None)
    for a in arms:
        d = a["decomp_ms_tok"]
        dt = f"{d['e2e_total'] - full_t:+.3f}" if (full_t is not None and a["key"] != "full") else "—"
        lines.append(
            f"| `{a['label']}` | {d['sel']:.3f} | {d['target_prefill']:.3f} | {d['draft_prefill']:.3f} | "
            f"{d['draft_decode']:.3f} | {d['target_verify']:.3f} | {d['correction']:.3f} | {d['unattr_decode']:.3f} | "
            f"**{d['e2e_total']:.3f}** | **{dt}** |"
        )

    lines += [
        "",
        "## 3. 100% Additive E2E Decomposition — Per-Claim `ms/claim`",
        "",
        "| Arm | Sel | Target Prefill | Draft Prefill | Draft Decode | Target Verify | Correction | Unattr Dec | **Total E2E ms/claim** | Residual |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for a in arms:
        d = a["decomp_ms_claim"]
        lines.append(
            f"| `{a['label']}` | {d['sel']:.2f} | {d['target_prefill']:.2f} | {d['draft_prefill']:.2f} | "
            f"{d['draft_decode']:.2f} | {d['target_verify']:.2f} | {d['correction']:.2f} | {d['unattr_decode']:.2f} | "
            f"**{d['e2e_total']:.2f}** | {d['residual']:+.4f} |"
        )

    for grp_title, grp_key in [
        ("4. Claim-Clustered Paired Bootstrap vs `Baseline Full`", "vs_full"),
        ("5. Claim-Clustered Paired Bootstrap vs `BM25 Top-2`", "vs_topk2"),
        ("6. Claim-Clustered Paired Bootstrap vs `B0-S` (Autoregressive)", "vs_b0s"),
    ]:
        comps = report["comparisons"].get(grp_key, [])
        if not comps:
            continue
        lines += [
            "",
            f"## {grp_title}",
            "",
            "| Arm | Metric | Baseline | Treatment | $\\Delta$ | 95% CI | Two-Sided $p$ |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for c in comps:
            lines.append(
                f"| `{c['arm']}` | `{c['metric']}` | {c['baseline']:.4f} | {c['treatment']:.4f} | "
                f"**{c['delta']:+.4f}** | `[{c['ci_95_low']:+.4f}, {c['ci_95_high']:+.4f}]` | {c['p_value_two_sided']:.4f} |"
            )

    lines += [
        "",
        "## 7. One-Time Setup & Compilation Timings (Excluded from Steady-State Latency)",
        "",
        "| Arm | Runtime | `max_cache_len` | Cache Init (s) | Compile (s) | Warmup (s) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for a in arms:
        st = a.get("draft_setup_timings") or {}
        if st:
            lines.append(
                f"| `{a['label']}` | `{a['draft_runtime']}` | `{st.get('max_cache_len')}` | "
                f"{st.get('cache_init_s', 0.0):.3f} | {st.get('compile_s', 0.0):.3f} | {st.get('warmup_s', 0.0):.3f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    p.add_argument("--run-prefix", default="b1-2-retrieved100")
    p.add_argument("--title", default="Step B1.3c — Live 100-Claim Evidence-Selective Speculative Decoding Comparison")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    arm_specs = [
        ("b0s", "B0-S (Target AR)", f"{args.run_prefix}-b0s"),
        ("full", "Baseline Full", f"{args.run_prefix}-baseline-full"),
        ("topk2", "BM25 Top-2", f"{args.run_prefix}-selective-topk2"),
        ("compact", "Compact (1s/200c)", f"{args.run_prefix}-compact"),
        ("bm25_sent_b8k", "BM25 Sent-B8k", f"{args.run_prefix}-bm25-sent-b8k"),
        ("attn_sent_b10k", "Target-Attn Sent-B10k", f"{args.run_prefix}-attn-sent-b10k"),
    ]

    arms: list[dict[str, Any]] = []
    cmaps: dict[str, dict[str, dict[str, Any]]] = {}
    for key, label, run_id in arm_specs:
        rdir = args.runs_root / run_id
        if not rdir.exists():
            continue
        row, cmap = _summarize_arm(label, key, rdir)
        arms.append(row)
        cmaps[key] = cmap

    metrics_full = [
        "throughput_e2e_tok_s",
        "latency_e2e_ms_per_tok",
        "e2e_ms_per_claim",
        "throughput_decode_tok_s",
        "observed_alpha_eval_pct",
        "committed_per_round",
        "draft_decode_ms_per_round",
        "draft_kv_peak_mb",
    ]
    metrics_b0s = [
        "throughput_e2e_tok_s",
        "latency_e2e_ms_per_tok",
        "throughput_decode_tok_s",
    ]

    comparisons: dict[str, list[dict[str, Any]]] = {"vs_full": [], "vs_topk2": [], "vs_b0s": []}
    if "full" in cmaps:
        for a in arms:
            if a["key"] in {"b0s", "full"}:
                continue
            for m in metrics_full:
                res = _paired_bootstrap(cmaps["full"], cmaps[a["key"]], m, n_bootstrap=args.n_bootstrap, seed=args.seed)
                comparisons["vs_full"].append({"arm": a["label"], **res})

    if "topk2" in cmaps:
        for a in arms:
            if a["key"] not in {"bm25_sent_b8k", "attn_sent_b10k"}:
                continue
            for m in ["throughput_e2e_tok_s", "latency_e2e_ms_per_tok", "observed_alpha_eval_pct", "committed_per_round", "draft_kv_peak_mb"]:
                res = _paired_bootstrap(cmaps["topk2"], cmaps[a["key"]], m, n_bootstrap=args.n_bootstrap, seed=args.seed)
                comparisons["vs_topk2"].append({"arm": a["label"], **res})

    if "b0s" in cmaps:
        for a in arms:
            if a["key"] == "b0s":
                continue
            for m in metrics_b0s:
                res = _paired_bootstrap(cmaps["b0s"], cmaps[a["key"]], m, n_bootstrap=args.n_bootstrap, seed=args.seed)
                comparisons["vs_b0s"].append({"arm": a["label"], **res})

    report = {
        "title": args.title,
        "run_prefix": args.run_prefix,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "arms": arms,
        "comparisons": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "b1_3_live_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = render_markdown(report)
    (args.output_dir / "b1_3_live_report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
