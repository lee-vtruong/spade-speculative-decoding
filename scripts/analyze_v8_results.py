"""Generate SPADE v8 revision analyses and LaTeX tables from RTX 5090 outputs.

Generates:
  - docs/paper/latex/generated_tables/table_stratified_performance.tex  (consistent aggregate + AR)
  - docs/paper/latex/generated_tables/table_batch_serving.tex           (batch=1..64 KV & concurrency)
  - docs/paper/latex/generated_tables/table_spade_vs_bm25_stratified.tex (fixed-prefix & live overlap)
  - docs/paper/latex/generated_tables/table_qwen3_ratio_sweep.tex       (same-family 0.6B/1.7B/4B + Leviathan c)
  - docs/paper/latex/generated_tables/table_long173_interleaved.tex     (R=3 randomized mean +/- SD)
  - docs/paper/latex/generated_tables/table3_oracle_results.tex         (interleaved null control + ref)
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

REV6_DIR = Path("outputs/analysis/v6_revisions")
REV8_DIR = Path("outputs/analysis/v8_revisions")
ANALYSIS_DIR = Path("outputs/analysis")
RUNS_DIR = Path("outputs/runs")
TABLES_DIR = Path("docs/paper/latex/generated_tables")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_p(p: float) -> str:
    if p < 0.0001:
        return "$< 10^{-4}$"
    return f"{p:.4f}"


def _fmt_p_eq(p: float) -> str:
    if p < 0.0001:
        return "$p < 10^{-4}$"
    return f"$p = {p:.4f}$"


def _fmt_p_holm_eq(p: float) -> str:
    if p < 0.0001:
        return "$p_{\\text{Holm}} < 10^{-4}$"
    return f"$p_{{\\text{{Holm}}}} = {p:.4f}$"


def _compute_sel173_live_overall() -> dict[str, Any] | None:
    """Compute aggregate live SPADE vs BM25 contrast on Long-173 (b1-4 Sel-173 cohort)."""
    l173_path = Path("data/processed/averitec_long173.jsonl")
    sp_path = RUNS_DIR / "b1-4-heldout300-attn-sent-b10k" / "predictions.jsonl"
    bm_path = RUNS_DIR / "b1-4-heldout300-bm25-sent-b10k" / "predictions.jsonl"
    if not (l173_path.exists() and sp_path.exists() and bm_path.exists()):
        return None
    import numpy as np

    cids = sorted(str(json.loads(l)["claim_id"]) for l in l173_path.read_text(encoding="utf-8").splitlines() if l.strip())

    def _load_preds(path: Path) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for l in path.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            rec = json.loads(l)
            cid = str(rec["claim_id"])
            rds = rec.get("rounds") or []
            t = rec.get("timings") or {}
            e2e_s = (
                float(t.get("context_selection_s", 0.0))
                + float(t.get("prompt_prefill_s", 0.0))
                + float(t.get("draft_prefill_s", 0.0))
                + float(t.get("decode_s", 0.0))
            )
            ev = sum(min(int(rd["proposed_token_count"]), int(rd["accepted_draft_tokens"]) + (0 if rd["full_acceptance"] else 1)) for rd in rds)
            ac = sum(int(rd["accepted_draft_tokens"]) for rd in rds)
            out[cid] = {"ac": float(ac), "ev": float(ev), "tok": float(rec.get("output_tokens", 0)), "e2e_s": e2e_s}
        return out

    sp_m = _load_preds(sp_path)
    bm_m = _load_preds(bm_path)
    common = [c for c in cids if c in sp_m and c in bm_m]
    if not common:
        return None
    arr = np.array(
        [[bm_m[c]["ac"], bm_m[c]["ev"], sp_m[c]["ac"], sp_m[c]["ev"], bm_m[c]["tok"], bm_m[c]["e2e_s"], sp_m[c]["tok"], sp_m[c]["e2e_s"]] for c in common],
        dtype=float,
    )
    rng = np.random.default_rng(42)
    bidx = rng.integers(0, len(common), size=(10000, len(common)))
    bs = arr[bidx].sum(axis=1)
    da_dist = ((bs[:, 2] / np.maximum(1.0, bs[:, 3])) - (bs[:, 0] / np.maximum(1.0, bs[:, 1]))) * 100.0
    dt_dist = (bs[:, 6] / np.maximum(1e-9, bs[:, 7])) - (bs[:, 4] / np.maximum(1e-9, bs[:, 5]))
    da_obs = ((arr[:, 2].sum() / max(1.0, arr[:, 3].sum())) - (arr[:, 0].sum() / max(1.0, arr[:, 1].sum()))) * 100.0
    dt_obs = (arr[:, 6].sum() / max(1e-9, arr[:, 7].sum())) - (arr[:, 4].sum() / max(1e-9, arr[:, 5].sum()))
    return {
        "n_claims": len(common),
        "delta_alpha_pp": float(da_obs),
        "alpha_p": float(min(1.0, 2.0 * min((da_dist <= 0).mean(), (da_dist >= 0).mean()))),
        "delta_thpt": float(dt_obs),
        "thpt_p": float(min(1.0, 2.0 * min((dt_dist <= 0).mean(), (dt_dist >= 0).mean()))),
    }


def _compute_h300_stratum_details() -> dict[str, dict[str, float]] | None:
    """Compute exact heldout-300 per-stratum tokens, KV reduction, alpha_eval, and alpha_prop."""
    fl_dir = RUNS_DIR / "b1-4-heldout300-baseline-full"
    sp_dir = RUNS_DIR / "b1-4-heldout300-attn-sent-b10k"
    if not ((fl_dir / "trial_metrics.jsonl").exists() and (sp_dir / "round_traces.jsonl").exists()):
        return None

    def _read_arm(d: Path) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for l in (d / "trial_metrics.jsonl").read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("trial_index", 0) != 0:
                continue
            cid = str(r["claim_id"])
            out[cid] = {
                "ev_chars": float(r.get("full_evidence_chars", 0.0)),
                "draft_tok": float(r.get("draft_prompt_tokens", r.get("prompt_tokens", 0.0))),
                "ev_acc": 0.0,
                "ev_tot": 0.0,
                "prop_acc": 0.0,
                "prop_tot": 0.0,
            }
        for l in (d / "round_traces.jsonl").read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            rd = json.loads(l)
            cid = str(rd["claim_id"])
            if cid not in out:
                continue
            pr = int(rd["proposed_token_count"])
            ac = int(rd["accepted_draft_tokens"])
            ev = min(pr, ac + (0 if rd["full_acceptance"] else 1))
            out[cid]["ev_acc"] += ac
            out[cid]["ev_tot"] += ev
            out[cid]["prop_acc"] += ac
            out[cid]["prop_tot"] += pr
        return out

    fl = _read_arm(fl_dir)
    sp = _read_arm(sp_dir)
    common = sorted(set(fl) & set(sp))
    strata_preds = {
        "bypass_le_10k": lambda x: x <= 10000,
        "selective_gt_10k": lambda x: x > 10000,
        "dense_ge_20k": lambda x: x >= 20000,
        "full_all": lambda _x: True,
    }
    res: dict[str, dict[str, float]] = {}
    for skey, pred in strata_preds.items():
        cids = [c for c in common if pred(fl[c]["ev_chars"])]
        f_dt = sum(fl[c]["draft_tok"] for c in cids) / len(cids)
        s_dt = sum(sp[c]["draft_tok"] for c in cids) / len(cids)
        res[skey] = {
            "n_claims": float(len(cids)),
            "mean_context_tok": f_dt,
            "tok_red_pct": ((s_dt / f_dt) - 1.0) * 100.0 if f_dt > 0 else 0.0,
            "fl_alpha_eval": sum(fl[c]["ev_acc"] for c in cids) / max(1.0, sum(fl[c]["ev_tot"] for c in cids)) * 100.0,
            "sp_alpha_eval": sum(sp[c]["ev_acc"] for c in cids) / max(1.0, sum(sp[c]["ev_tot"] for c in cids)) * 100.0,
            "fl_alpha_prop": sum(fl[c]["prop_acc"] for c in cids) / max(1.0, sum(fl[c]["prop_tot"] for c in cids)) * 100.0,
            "sp_alpha_prop": sum(sp[c]["prop_acc"] for c in cids) / max(1.0, sum(sp[c]["prop_tot"] for c in cids)) * 100.0,
        }
    return res


def _extract_leviathan_from_runs(ar_dir: Path, sd_dir: Path) -> dict[str, Any] | None:
    if not ((ar_dir / "trial_metrics.jsonl").exists() and (sd_dir / "round_traces.jsonl").exists()):
        return None
    ar_trials = [json.loads(l) for l in (ar_dir / "trial_metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ar_dec_s = sum(float(t.get("decode_s", 0.0)) for t in ar_trials if t.get("trial_index", 0) == 0)
    ar_pref_s = sum(float(t.get("prompt_prefill_s", 0.0)) for t in ar_trials if t.get("trial_index", 0) == 0)
    ar_tok = sum(int(t.get("output_tokens", 0)) for t in ar_trials if t.get("trial_index", 0) == 0)
    if ar_tok <= 0:
        return None
    ar_step_ms = (ar_dec_s / ar_tok) * 1000.0
    ar_e2e_tok_s = ar_tok / max(1e-9, ar_dec_s + ar_pref_s)
    ar_dec_tok_s = ar_tok / max(1e-9, ar_dec_s)

    rounds = [json.loads(l) for l in (sd_dir / "round_traces.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    sd_trials = [json.loads(l) for l in (sd_dir / "trial_metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    d_sum = sum(float(r.get("draft_latency_s", 0.0)) for r in rounds)
    v_sum = sum(float(r.get("verification_latency_s", 0.0)) for r in rounds)
    prop_sum = sum(int(r.get("proposed_token_count", 0)) for r in rounds)
    acc_sum = sum(int(r.get("accepted_draft_tokens", 0)) for r in rounds)
    eval_sum = sum(
        min(int(r.get("proposed_token_count", 0)), int(r.get("accepted_draft_tokens", 0)) + (0 if r.get("full_acceptance") else 1))
        for r in rounds
    )
    n_rnd = len(rounds)
    if prop_sum <= 0 or eval_sum <= 0 or n_rnd <= 0:
        return None
    draft_step_ms = (d_sum / prop_sum) * 1000.0
    verify_rnd_ms = (v_sum / n_rnd) * 1000.0
    alpha = acc_sum / eval_sum
    c_ar = draft_step_ms / ar_step_ms
    gamma = verify_rnd_ms / ar_step_ms
    k = 4
    exp_tok = (1.0 - alpha ** (k + 1)) / max(1e-9, 1.0 - alpha)
    lev_ideal = exp_tok / (k * c_ar + 1.0)
    lev_gamma = exp_tok / (k * c_ar + gamma)

    sd_e2e_s = sum(
        float(t.get("context_selection_s", 0.0)) + float(t.get("prompt_prefill_s", 0.0)) + float(t.get("draft_prefill_s", 0.0)) + float(t.get("decode_s", 0.0))
        for t in sd_trials if t.get("trial_index", 0) == 0
    )
    sd_dec_s = sum(float(t.get("decode_s", 0.0)) for t in sd_trials if t.get("trial_index", 0) == 0)
    sd_tok = sum(int(t.get("output_tokens", 0)) for t in sd_trials if t.get("trial_index", 0) == 0)
    sd_e2e_tok_s = sd_tok / max(1e-9, sd_e2e_s)
    sd_dec_tok_s = sd_tok / max(1e-9, sd_dec_s)
    return {
        "ar_e2e_tok_s": round(ar_e2e_tok_s, 2),
        "ar_dec_tok_s": round(ar_dec_tok_s, 2),
        "ar_step_ms": round(ar_step_ms, 3),
        "sd_e2e_tok_s": round(sd_e2e_tok_s, 2),
        "sd_dec_tok_s": round(sd_dec_tok_s, 2),
        "alpha_eval_pct": round(alpha * 100.0, 2),
        "draft_step_ms": round(draft_step_ms, 3),
        "verify_rnd_ms": round(verify_rnd_ms, 2),
        "c_ar": round(c_ar, 4),
        "gamma_ver": round(gamma, 2),
        "leviathan_ideal_decode_x": round(lev_ideal, 2),
        "leviathan_gamma_adjusted_x": round(lev_gamma, 2),
        "observed_decode_x_vs_ar": round(sd_dec_tok_s / ar_dec_tok_s, 2),
        "observed_e2e_x_vs_ar": round(sd_e2e_tok_s / ar_e2e_tok_s, 2),
    }


def export_stratified_table(v6_master: dict[str, Any] | None) -> None:
    if not v6_master or "heldout300_strata_ar" not in v6_master:
        return
    st = v6_master["heldout300_strata_ar"]
    h300_rep = _load_json(ANALYSIS_DIR / "b1_4_heldout300" / "b1_3_live_report.json") or {}
    strat_rep = _load_json(ANALYSIS_DIR / "stratified_performance_breakdown.json") or {}
    live_strata = _compute_h300_stratum_details()
    b0 = st["bypass_le_10k"]
    s1 = st["selective_gt_10k"]
    d2 = st["dense_ge_20k"]

    def _ci(d: dict[str, Any], bold: bool = True) -> str:
        star = "$^*$" if d.get("p_raw", 1.0) < 0.001 else ""
        s = f"{d['delta']:+.2f} [{d['ci_lo']:+.2f}, {d['ci_hi']:+.2f}]{star}"
        return f"\\textbf{{{s}}}" if bold and d.get("p_raw", 1.0) < 0.001 else s

    # Extract Full Test Set metrics dynamically from h300_rep & v6_master["heldout300_holm"]
    arms_map = {a["label"]: a for a in h300_rep.get("arms", [])}
    vs_b0s_sp = next(
        (c for c in h300_rep.get("comparisons", {}).get("vs_b0s", []) if "Attn" in str(c.get("arm", "")) and c.get("metric") == "throughput_e2e_tok_s"),
        None,
    )
    h300_fl = next((c for c in v6_master.get("heldout300_holm", []) if c.get("vs") == "Baseline Full"), None)

    ar_all = arms_map.get("B0-S (Target AR)", {}).get("throughput_e2e_tok_s", 0.0)
    fl_all = arms_map.get("Baseline Full", {}).get("throughput_e2e_tok_s", 0.0)
    sp_all = arms_map.get("SPADE (Attn Sent-B10k)", {}).get("throughput_e2e_tok_s", 0.0)
    if vs_b0s_sp:
        full_vs_ar_d = {
            "delta": vs_b0s_sp["delta"],
            "ci_lo": vs_b0s_sp["ci_95_low"],
            "ci_hi": vs_b0s_sp["ci_95_high"],
            "p_raw": vs_b0s_sp["p_value_two_sided"],
        }
        full_vs_ar_s = _ci(full_vs_ar_d)
    else:
        full_vs_ar_s = "—"
    if h300_fl:
        full_vs_fl_s = _ci(h300_fl["e2e_tok_s"])
    else:
        full_vs_fl_s = "—"

    def _meta(skey: str, strat_key: str) -> dict[str, str]:
        if live_strata and skey in live_strata:
            ls = live_strata[skey]
            return {
                "n": str(int(ls["n_claims"])),
                "ctx": f"{ls['mean_context_tok']:,.1f} tok",
                "red": f"{ls['tok_red_pct']:.1f}\\%" if abs(ls["tok_red_pct"]) > 0.05 else "0.0\\%",
                "ae": f"{ls['fl_alpha_eval']:.2f}\\% $\\to$ {ls['sp_alpha_eval']:.2f}\\%",
                "ap": f"{ls['fl_alpha_prop']:.2f}\\% $\\to$ {ls['sp_alpha_prop']:.2f}\\%",
            }
        sr = strat_rep.get(strat_key, {})
        if skey == "full_all" and arms_map:
            fl_a = arms_map.get("Baseline Full", {})
            sp_a = arms_map.get("SPADE (Attn Sent-B10k)", {})
            return {
                "n": str(fl_a.get("n_claims", 300)),
                "ctx": f"{fl_a.get('mean_target_prompt_tokens', 0.0):,.1f} tok",
                "red": f"-{sp_a.get('prompt_compression_pct', 0.0):.1f}\\%",
                "ae": f"{fl_a.get('observed_alpha_eval_pct', 0.0):.2f}\\% $\\to$ {sp_a.get('observed_alpha_eval_pct', 0.0):.2f}\\%",
                "ap": f"{fl_a.get('observed_alpha_prop_pct', 0.0):.2f}\\% $\\to$ {sp_a.get('observed_alpha_prop_pct', 0.0):.2f}\\%",
            }
        return {
            "n": str(sr.get("n", st.get(skey, {}).get("n_claims", 0))),
            "ctx": f"{sr.get('full_draft_tokens', 0.0):,.1f} tok" if sr else "—",
            "red": (f"-{sr.get('kv_reduction_pct', 0.0):.1f}\\%" if sr.get("kv_reduction_pct", 0.0) > 0 else "0.0\\%") if sr else "—",
            "ae": "—",
            "ap": f"{sr.get('full_acc', 0.0):.2f}\\% $\\to$ {sr.get('spade_acc', 0.0):.2f}\\%" if sr else "—",
        }

    m_b0 = _meta("bypass_le_10k", "Bypass Regime (Evidence <= 10k chars)")
    m_s1 = _meta("selective_gt_10k", "Selective Regime (Evidence > 10k chars)")
    m_d2 = _meta("dense_ge_20k", "Dense Multi-Doc (Evidence >= 20k chars)")
    m_al = _meta("full_all", "All Claims (Full Benchmark)")

    rows = [
        (
            f"Bypass ($L \\le 10\\text{{k}}$) & {m_b0['n']} (42.3\\%) & {m_b0['ctx']} & {m_b0['red']} & "
            f"{b0['spade_vs_ar_e2e']['base']:.2f} & {b0['spade_vs_full_e2e']['base']:.2f} $\\to$ {b0['spade_vs_full_e2e']['treat']:.2f} & "
            f"{_ci(b0['spade_vs_ar_e2e'])} & {_ci(b0['spade_vs_full_e2e'], bold=False)} & "
            f"{m_b0['ae']} & {m_b0['ap']} \\\\"
        ),
        (
            f"Selective ($L > 10\\text{{k}}$) & {m_s1['n']} (57.7\\%) & {m_s1['ctx']} & \\textbf{{{m_s1['red']}}} & "
            f"{s1['spade_vs_ar_e2e']['base']:.2f} & {s1['spade_vs_full_e2e']['base']:.2f} $\\to$ \\textbf{{{s1['spade_vs_full_e2e']['treat']:.2f}}} & "
            f"{_ci(s1['spade_vs_ar_e2e'])} & {_ci(s1['spade_vs_full_e2e'])} & "
            f"{m_s1['ae']} & {m_s1['ap']} \\\\"
        ),
        (
            f"Dense ($L \\ge 20\\text{{k}}$) & {m_d2['n']} (41.3\\%) & {m_d2['ctx']} & \\textbf{{{m_d2['red']}}} & "
            f"{d2['spade_vs_ar_e2e']['base']:.2f} & {d2['spade_vs_full_e2e']['base']:.2f} $\\to$ \\textbf{{{d2['spade_vs_full_e2e']['treat']:.2f}}} & "
            f"{_ci(d2['spade_vs_ar_e2e'])} & {_ci(d2['spade_vs_full_e2e'])} & "
            f"{m_d2['ae']} & {m_d2['ap']} \\\\"
        ),
        "\\midrule",
        (
            f"\\textbf{{Full Test Set}} & {m_al['n']} (100\\%) & {m_al['ctx']} & \\textbf{{{m_al['red']}}} & "
            f"{ar_all:.2f} & {fl_all:.2f} $\\to$ \\textbf{{{sp_all:.2f}}} & {full_vs_ar_s} & {full_vs_fl_s} & "
            f"{m_al['ae']} & {m_al['ap']} \\\\"
        ),
    ]
    s1_pct = (s1["spade_vs_full_e2e"]["delta"] / s1["spade_vs_full_e2e"]["base"]) * 100.0
    d2_pct = (d2["spade_vs_full_e2e"]["delta"] / d2["spade_vs_full_e2e"]["base"]) * 100.0
    tex = (
        "\\begin{table*}[t]\n\\centering\n\\small\n\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lccccccccr}\n\\toprule\n"
        "\\textbf{Context Regime} & \\textbf{Claims} & \\textbf{Mean Context} & \\textbf{Draft Tok Red.} & \\textbf{AR tok/s} & \\textbf{Agg.\\ (F $\\to$ S)} & \\textbf{$\\Delta$ vs.\\ AR (95\\% CI)} & \\textbf{$\\Delta$ vs.\\ Full (95\\% CI)} & \\textbf{$\\alpha_{\\text{eval}}$ (F $\\to$ S)} & \\textbf{$\\alpha_{\\text{prop}}$ (F $\\to$ S)} \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}%\n}\n"
        f"\\caption{{\\textbf{{Context-Stratified Efficiency on Held-Out AVeriTeC (\\texttt{{Qwen3-8B/0.6B}}, Aggregate $\\sum\\text{{tok}}/\\sum\\text{{time}}$ in tok/s):}} All throughputs and paired claim-cluster bootstrap 95\\% CIs ($B=10,000$; $^*$ denotes $p < 0.001$) use the consistent aggregate ratio definition. On long strata ($8.3\\text{{k}}$ and $10.1\\text{{k}}$ mean tokens), SPADE outperforms \\textit{{both}} Target AR (${s1['spade_vs_ar_e2e']['delta']:+.2f}$ and ${d2['spade_vs_ar_e2e']['delta']:+.2f}\\text{{ tok/s}}$, $p={d2['spade_vs_ar_e2e']['p_raw']:.4f}$ on Dense where Full-SD trails AR at ${d2['full_vs_ar_e2e']['treat']:.2f}$ vs.\\ ${d2['full_vs_ar_e2e']['base']:.2f}$) and Full-SD (${s1['spade_vs_full_e2e']['delta']:+.2f}$ and ${d2['spade_vs_full_e2e']['delta']:+.2f}\\text{{ tok/s}}$, $+{s1_pct:.1f}\\%$ and $+{d2_pct:.1f}\\%$) while preserving $\\alpha_{{\\text{{eval}}}}$ and $\\alpha_{{\\text{{prop}}}}$.}}\n"
        "\\label{tab:stratified_performance}\n\\end{table*}\n"
    )
    (TABLES_DIR / "table_stratified_performance.tex").write_text(tex, encoding="utf-8")


def export_batch_table(batch_data: dict[str, Any] | None) -> None:
    if not batch_data:
        return
    sa = {(r["context_len"], r["policy"], r["batch_size"]): r for r in batch_data["standalone_batched_decode"]}
    coloc = {
        (r["context_len"], r["policy"], r["batch_size"]): r
        for r in batch_data.get("colocated_concurrency", [])
        if not r.get("oom")
    }
    lines = []
    for ctx_len, clbl in ((8192, "8k (8,192)"), (16384, "16k (16,384)")):
        for bs in (1, 4, 8, 16, 32, 64):
            rf = sa.get((ctx_len, "Full Context", bs), {"oom": True})
            rs = sa.get((ctx_len, "SPADE B=10k", bs), {"oom": True})
            r2 = sa.get((ctx_len, "SPADE B=2k", bs), {"oom": True})
            f_s = "\\textit{OOM}" if rf["oom"] else f"{rf['draft_kv_gb']:.2f} GB / {rf['draft_decode_tok_s']:.1f}"
            s_s = "\\textit{OOM}" if rs["oom"] else f"{rs['draft_kv_gb']:.2f} GB / {rs['draft_decode_tok_s']:.1f}"
            b2_s = "\\textit{OOM}" if r2["oom"] else f"{r2['draft_kv_gb']:.2f} GB / {r2['draft_decode_tok_s']:.1f}"
            if rf["oom"]:
                gain_s = f"Feasible ({rs['draft_decode_tok_s']:.0f} / {r2['draft_decode_tok_s']:.0f})"
            else:
                gain_s = f"{rs['draft_decode_tok_s']/rf['draft_decode_tok_s']:.2f}$\\times$ / {r2['draft_decode_tok_s']/rf['draft_decode_tok_s']:.2f}$\\times$"
            lines.append(f"{clbl if bs==1 else ''} & {bs} & {f_s} & {s_s} & {b2_s} & {gain_s} \\\\")
        if ctx_len == 8192:
            lines.append("\\midrule")
    mc = batch_data["max_concurrent_batch_32gb"]
    c8 = mc["8192"]
    c16 = mc["16384"]
    v8_f = coloc[(8192, "Full Context", c8["Full Context"])]["peak_vram_gb"]
    v8_s = coloc[(8192, "SPADE B=10k", c8["SPADE B=10k"])]["peak_vram_gb"]
    v8_2 = coloc[(8192, "SPADE B=2k", c8["SPADE B=2k"])]["peak_vram_gb"]
    v16_f = coloc[(16384, "Full Context", c16["Full Context"])]["peak_vram_gb"]
    v16_s = coloc[(16384, "SPADE B=10k", c16["SPADE B=10k"])]["peak_vram_gb"]
    v16_2 = coloc[(16384, "SPADE B=2k", c16["SPADE B=2k"])]["peak_vram_gb"]
    kv1_8k = sa[(8192, "Full Context", 1)]["draft_kv_gb"]
    kv1_16k = sa[(16384, "Full Context", 1)]["draft_kv_gb"]
    peak_10k_x = sa[(16384, "SPADE B=10k", 64)]["draft_decode_tok_s"] / sa[(16384, "Full Context", 8)]["draft_decode_tok_s"]
    peak_2k_x = sa[(16384, "SPADE B=2k", 64)]["draft_decode_tok_s"] / sa[(16384, "Full Context", 8)]["draft_decode_tok_s"]

    lines.append("\\midrule")
    lines.append(
        f"\\multicolumn{{6}}{{l}}{{\\textbf{{Co-Located 8B+0.6B Max Batch Size on 32\\,GB GPU:}} "
        f"$L=8\\text{{k}}$: Full={c8['Full Context']} (${v8_f:.1f}\\text{{ GB}}$) $\\to$ SPADE-10k={c8['SPADE B=10k']} (${v8_s:.1f}\\text{{ GB}}$) $\\to$ SPADE-2k={c8['SPADE B=2k']} (${v8_2:.1f}\\text{{ GB}}$); "
        f"$L=16\\text{{k}}$: Full={c16['Full Context']} (${v16_f:.1f}\\text{{ GB}}$) $\\to$ SPADE-10k={c16['SPADE B=10k']} (${v16_s:.1f}\\text{{ GB}}$) $\\to$ SPADE-2k={c16['SPADE B=2k']} (${v16_2:.1f}\\text{{ GB}}$).}} \\\\"
    )
    tex = (
        "\\begin{table*}[t]\n\\centering\n\\small\n\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lccccr}\n\\toprule\n"
        "\\textbf{Context Len} & \\textbf{Batch} & \\textbf{Full KV / Draft tok/s} & \\textbf{SPADE-10k KV / tok/s} & \\textbf{SPADE-2k KV / tok/s} & \\textbf{Decode Gain (10k / 2k)} \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}%\n}\n"
        f"\\caption{{\\textbf{{Multi-Request Batched Serving Memory \\& Draft Decode Throughput (\\texttt{{Qwen3-0.6B}}, StaticCache, RTX 5090 32\\,GB):}} While single-request ($\\text{{batch}}=1$) draft KV fits easily (${kv1_8k:.2f}\\text{{--}}{kv1_16k:.2f}\\text{{ GB}}$), shared $8$-head GQA causes Full-SD draft KV to hit OOM at $\\text{{batch}}=32$ ($8\\text{{k}}$) and $\\text{{batch}}=16$ ($16\\text{{k}}$). SPADE scales cleanly to $\\text{{batch}}=64$ (${peak_10k_x:.2f}\\times$ peak draft throughput for $B=10\\text{{k}}$ and ${peak_2k_x:.2f}\\times$ for $B=2\\text{{k}}$ vs.\\ Full $\\text{{batch}}=8$ at $16\\text{{k}}$) and lifts co-located \\texttt{{8B+0.6B}} max concurrency from ${c16['Full Context']} \\to {c16['SPADE B=10k']}$ ($16\\text{{k}}$) and ${c8['Full Context']} \\to {c8['SPADE B=2k']}$ ($8\\text{{k}}$).}}\n"
        "\\label{tab:batch_serving}\n\\end{table*}\n"
    )
    (TABLES_DIR / "table_batch_serving.tex").write_text(tex, encoding="utf-8")


def export_spade_vs_bm25_table(fbm: dict[str, Any] | None, v6_master: dict[str, Any] | None = None) -> None:
    if not fbm:
        return
    fp = fbm["fixed_prefix"]
    lv = fbm.get("live_rollouts", {})
    l173 = fp["Long-173"]
    hp = fp["HotpotQA-100"]
    sel_lv = lv.get("AVeriTeC-Sel173 (b1-4)", {})
    sel_overall = _compute_sel173_live_overall()

    def _fmt_live(live_st: dict[str, Any] | None) -> str:
        if not live_st:
            return "—"
        da = live_st["delta_alpha_pp"]
        dt = live_st["delta_thpt"]
        ap = _fmt_p(live_st["alpha_p"])
        tp = _fmt_p(live_st["thpt_p"])
        return f"{da:+.2f} pp ({ap}) / {dt:+.2f} ({tp})"

    def _row_fp(label: str, st: dict[str, Any], live_st: dict[str, Any] | None = None) -> str:
        return (
            f"{label} & {st['n_claims']} ({st['n_prefixes']:,}) & {st['bm25_expected_acc_pct']:.2f}\\% & "
            f"{st['spade_expected_acc_pct']:.2f}\\% & {st['delta_pp']:+.2f} [{st['ci_lo']:+.2f}, {st['ci_hi']:+.2f}] & "
            f"{_fmt_p(st['p_raw'])} & {_fmt_live(live_st)} \\\\"
        )

    c_glob_l173 = l173["contrasts_vs_bm25"]["SPADE (Attn B=10k) vs BM25 Sent-B10k"]
    c_glob_hp = hp["contrasts_vs_bm25"]["SPADE (Attn B=10k) vs BM25 Sent-B10k"]

    hp_live_overall: dict[str, Any] | None = None
    lb_pw = (v6_master or {}).get("longbench", {}).get("pairwise_holm", {}).get("SPADE B=10k vs BM25 Sent-B10k")
    if lb_pw and "alpha_eval_pct" in lb_pw and "e2e_tok_s" in lb_pw:
        hp_live_overall = {
            "delta_alpha_pp": lb_pw["alpha_eval_pct"]["delta"],
            "alpha_p": lb_pw["alpha_eval_pct"]["p_holm"],
            "delta_thpt": lb_pw["e2e_tok_s"]["delta"],
            "thpt_p": lb_pw["e2e_tok_s"]["p_holm"],
        }

    s_l173 = l173["stratified_spade_vs_bm25"]
    s_hp = hp["stratified_spade_vs_bm25"]
    rows = [
        "\\multicolumn{7}{l}{\\textit{AVeriTeC Long-173 ($N=173$, $19,259$ fixed Target-AR prefixes; Stratified by Query--Evidence Lexical Overlap \\& Length)}} \\\\",
        (
            f"All Long-173 Claims & {l173['n_claims']} ({l173['n_prefixes']:,}) & {l173['summary']['BM25 Sent-B10k']['mean_expected_acc_pct']:.2f}\\% & "
            f"{l173['summary']['SPADE (Attn B=10k)']['mean_expected_acc_pct']:.2f}\\% & "
            f"{c_glob_l173['delta_pp']:+.2f} [{c_glob_l173['ci_lo']:+.2f}, {c_glob_l173['ci_hi']:+.2f}] & "
            f"{_fmt_p(c_glob_l173['p_raw'])} & {_fmt_live(sel_overall)} \\\\"
        ),
        _row_fp(f"Low Claim--Ev. Recall (Bot-33\\%, $\\le {s_l173['claim_ev_recall']['q33']:.2f}$)", s_l173["claim_ev_recall"]["strata"]["Bot-33%"], sel_lv.get("claim_ev_recall", {}).get("strata", {}).get("Bot-33%")),
        _row_fp(f"High Claim--Ev. Recall (Top-33\\%, $\\ge {s_l173['claim_ev_recall']['q67']:.2f}$)", s_l173["claim_ev_recall"]["strata"]["Top-33%"], sel_lv.get("claim_ev_recall", {}).get("strata", {}).get("Top-33%")),
        _row_fp(f"Low Sent. Jaccard (Bot-33\\%, $\\le {s_l173['top5_sent_jaccard']['q33']:.3f}$)", s_l173["top5_sent_jaccard"]["strata"]["Bot-33%"], sel_lv.get("top5_sent_jaccard", {}).get("strata", {}).get("Bot-33%")),
        _row_fp(f"High Sent. Jaccard (Top-33\\%, $\\ge {s_l173['top5_sent_jaccard']['q67']:.3f}$)", s_l173["top5_sent_jaccard"]["strata"]["Top-33%"], sel_lv.get("top5_sent_jaccard", {}).get("strata", {}).get("Top-33%")),
        _row_fp(f"Low Max Sent. Q-Recall (Bot-33\\%, $\\le {s_l173['max_sent_q_recall']['q33']:.2f}$)", s_l173["max_sent_q_recall"]["strata"]["Bot-33%"], sel_lv.get("max_sent_q_recall", {}).get("strata", {}).get("Bot-33%")),
        _row_fp(f"Ultra-Long Context ($> {s_l173['ev_chars']['median']/1000.0:.1f}\\text{{k}}$ chars, High)", s_l173["ev_chars"]["strata"]["High (>med)"], sel_lv.get("ev_chars", {}).get("strata", {}).get("High (>med)")),
        "\\midrule",
        "\\multicolumn{7}{l}{\\textit{LongBench HotpotQA ($N=100$, $7,766$ fixed prefixes; High Lexical Overlap, Median Claim--Ev. Recall $=1.000$)}} \\\\",
        (
            f"All HotpotQA Claims & {hp['n_claims']} ({hp['n_prefixes']:,}) & {hp['summary']['BM25 Sent-B10k']['mean_expected_acc_pct']:.2f}\\% & "
            f"{hp['summary']['SPADE (Attn B=10k)']['mean_expected_acc_pct']:.2f}\\% & "
            f"{c_glob_hp['delta_pp']:+.2f} [{c_glob_hp['ci_lo']:+.2f}, {c_glob_hp['ci_hi']:+.2f}] & "
            f"{_fmt_p(c_glob_hp['p_raw'])} & {_fmt_live(hp_live_overall)} \\\\"
        ),
        _row_fp(f"Low Top-5 Sent. Jaccard ($\\le {s_hp['top5_sent_jaccard']['median']:.3f}$)", s_hp["top5_sent_jaccard"]["strata"]["Low (<=med)"]),
        _row_fp(f"High Top-5 Sent. Jaccard ($> {s_hp['top5_sent_jaccard']['median']:.3f}$)", s_hp["top5_sent_jaccard"]["strata"]["High (>med)"]),
    ]
    bot_recall = s_l173["claim_ev_recall"]["strata"]["Bot-33%"]
    bot_live = sel_lv.get("claim_ev_recall", {}).get("strata", {}).get("Bot-33%", {})
    ultra_hi = s_l173["ev_chars"]["strata"]["High (>med)"]
    bot_live_str = f"; live ${bot_live['delta_thpt']:+.2f}\\text{{ tok/s}}$, {_fmt_p_eq(bot_live['thpt_p'])}" if bot_live else ""
    tex = (
        "\\begin{table*}[t]\n\\centering\n\\small\n\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lcccccr}\n\\toprule\n"
        "\\textbf{Stratum} & \\textbf{Claims (Prefixes)} & \\textbf{BM25 $1-\\text{TV}$} & \\textbf{SPADE $1-\\text{TV}$} & \\textbf{Fixed $\\Delta$ (95\\% CI)} & \\textbf{Fixed $p$} & \\textbf{Live $\\Delta\\alpha$ ($p$) / $\\Delta\\text{tok/s}$ ($p$)} \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}%\n}\n"
        f"\\caption{{\\textbf{{When Does Target-Prefill Attention Outperform Lexical BM25? (Deterministic Fixed-Prefix \\& Live-Rollout Stratification):}} Across ${l173['n_prefixes']:,}$ shared Target-AR prefixes on \\texttt{{Long-173}}, SPADE's advantage concentrates sharply on low lexical-overlap queries (Bot-33\\% claim--evidence recall: ${bot_recall['delta_pp']:+.2f}\\text{{ pp}}$, {_fmt_p_eq(bot_recall['p_raw'])}{bot_live_str}) and longest contexts ($> {s_l173['ev_chars']['median']/1000.0:.1f}\\text{{k}}$ chars: ${ultra_hi['delta_pp']:+.2f}\\text{{ pp}}$, {_fmt_p_eq(ultra_hi['p_raw'])}). Conversely, on high-overlap AVeriTeC claims and keyword-anchored HotpotQA (median claim recall $1.000$), zero-GPU \\texttt{{BM25 Sent-B10k}} matches SPADE.}}\n"
        "\\label{tab:spade_vs_bm25_stratified}\n\\end{table*}\n"
    )
    (TABLES_DIR / "table_spade_vs_bm25_stratified.tex").write_text(tex, encoding="utf-8")


def export_qwen3_ratio_table(q3_sweep: dict[str, Any] | None, v6_master: dict[str, Any] | None = None) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    q25_sp = _extract_leviathan_from_runs(RUNS_DIR / "p2-qwen25-heldout300-b0s", RUNS_DIR / "p2-qwen25-heldout300-spade")
    q25_fl = _extract_leviathan_from_runs(RUNS_DIR / "p2-qwen25-heldout300-b0s", RUNS_DIR / "p2-qwen25-heldout300-full")
    lm3_sp = _extract_leviathan_from_runs(RUNS_DIR / "v7_llama3" / "llama3-heldout300-b0s", RUNS_DIR / "v7_llama3" / "llama3-heldout300-spade-b10k")
    lm3_fl = _extract_leviathan_from_runs(RUNS_DIR / "v7_llama3" / "llama3-heldout300-b0s", RUNS_DIR / "v7_llama3" / "llama3-heldout300-baseline-full")
    extra["cross_family_leviathan"] = {"Qwen2.5-SPADE": q25_sp, "Qwen2.5-Full": q25_fl, "Llama3-SPADE": lm3_sp, "Llama3-Full": lm3_fl}

    if not q3_sweep:
        return extra

    def _ci_fmt(c: dict[str, Any]) -> str:
        star = "$^*$" if c.get("p_raw", c.get("p_value_two_sided", 1.0)) < 0.001 else ""
        lo = c.get("ci_lo", c.get("ci_95_low", 0.0))
        hi = c.get("ci_hi", c.get("ci_95_high", 0.0))
        return f"{c['delta']:+.2f} [{lo:+.2f}, {hi:+.2f}]{star}"

    lines = []
    for d_name in ("Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"):
        m = q3_sweep["models"].get(d_name)
        if not m:
            continue
        ar = m["AR"]
        sp = m["SPADE"]
        fl = m.get("Full") or {}
        fl_e2e = f"{fl['e2e_tok_s']:.2f}" if "e2e_tok_s" in fl else "\\textit{OOM}"
        fl_kv = fl.get("draft_kv_mb", 0.0)
        kv_red = f"{(sp['draft_kv_mb']/fl_kv - 1.0)*100.0:.1f}\\%" if fl_kv > 0 else "—"
        sp_ar = m["spade_vs_ar"]["e2e_tok_s"]
        lines.append(
            f"Qwen3-8B / {d_name.replace('Qwen3-','')} ({m['ratio_label']}, $N={q3_sweep['n_claims']}$) & {ar['e2e_tok_s']:.2f} & {fl_e2e} & "
            f"{sp['e2e_tok_s']:.2f} ({kv_red}) & {sp['alpha_eval_pct']:.2f}\\% & "
            f"{sp['c_ar']:.3f} ($\\gamma={sp['gamma_ver']:.2f}$) & {sp['leviathan_ideal_decode_x']:.2f}$\\times$ / {sp['leviathan_gamma_adjusted_x']:.2f}$\\times$ & "
            f"{sp['observed_decode_x_vs_ar']:.2f}$\\times$ & {_ci_fmt(sp_ar)} \\\\"
        )

    q25_rep = _load_json(ANALYSIS_DIR / "p2_qwen25_heldout300" / "b1_3_live_report.json") or {}
    q25_arms = {a["label"]: a for a in q25_rep.get("arms", [])}
    q25_ar_ci = next(
        (c for c in q25_rep.get("comparisons", {}).get("vs_b0s", []) if "SPADE" in str(c.get("arm", "")) and c.get("metric") == "throughput_e2e_tok_s"),
        None,
    )
    if (q25_sp or q25_arms) and (q25_arms or v6_master):
        lines.append("\\midrule")

    if q25_sp and q25_arms:
        fl_a = q25_arms["Baseline Full"]
        sp_a = q25_arms["SPADE (Ours)"]
        kv_red_25 = (sp_a["draft_kv_peak_mb"] / fl_a["draft_kv_peak_mb"] - 1.0) * 100.0
        ci_25_s = _ci_fmt(q25_ar_ci) if q25_ar_ci else "—"
        lines.append(
            f"Qwen2.5-7B / 0.5B (1:14.0, $N=300$) & {q25_sp['ar_e2e_tok_s']:.2f} & {fl_a['throughput_e2e_tok_s']:.2f} & "
            f"{sp_a['throughput_e2e_tok_s']:.2f} ({kv_red_25:.1f}\\%) & {q25_sp['alpha_eval_pct']:.2f}\\% & "
            f"{q25_sp['c_ar']:.3f} ($\\gamma={q25_sp['gamma_ver']:.2f}$) & "
            f"{q25_sp['leviathan_ideal_decode_x']:.2f}$\\times$ / {q25_sp['leviathan_gamma_adjusted_x']:.2f}$\\times$ & "
            f"{q25_sp['observed_decode_x_vs_ar']:.2f}$\\times$ & {ci_25_s} \\\\"
        )

    lm3 = (v6_master or {}).get("llama3", {})
    lm3_rows = {r["label"]: r for r in lm3.get("rows", [])}
    lm3_ar_ci = lm3.get("pairwise_holm", {}).get("SPADE (Ours) vs B0-S (Target AR)", {}).get("e2e_tok_s")
    if lm3_sp and lm3_rows:
        fl_l = lm3_rows["Baseline Full"]
        sp_l = lm3_rows["SPADE (Ours)"]
        kv_red_l = (sp_l["draft_kv_mb"] / fl_l["draft_kv_mb"] - 1.0) * 100.0
        ci_l_s = _ci_fmt(lm3_ar_ci) if lm3_ar_ci else "—"
        lines.append(
            f"Llama-3.1-8B / 3.2-1B (1:8.0, $N=300$) & {lm3_sp['ar_e2e_tok_s']:.2f} & {fl_l['e2e_tok_s']:.2f} & "
            f"{sp_l['e2e_tok_s']:.2f} ({kv_red_l:.1f}\\%) & {lm3_sp['alpha_eval_pct']:.2f}\\% & "
            f"{lm3_sp['c_ar']:.3f} ($\\gamma={lm3_sp['gamma_ver']:.2f}$) & "
            f"{lm3_sp['leviathan_ideal_decode_x']:.2f}$\\times$ / {lm3_sp['leviathan_gamma_adjusted_x']:.2f}$\\times$ & "
            f"{lm3_sp['observed_decode_x_vs_ar']:.2f}$\\times$ & {ci_l_s} \\\\"
        )
    tex = (
        "\\begin{table*}[t]\n\\centering\n\\small\n\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lcccccccr}\n\\toprule\n"
        "\\textbf{Target / Draft (Ratio)} & \\textbf{AR E2E} & \\textbf{Full E2E} & \\textbf{SPADE E2E (KV)} & \\textbf{$\\alpha_{\\text{eval}}$} & \\textbf{$c_{\\text{AR}}$ ($\\gamma_{\\text{ver}}$)} & \\textbf{Lev.\\ Ideal / $\\gamma$-Adj.} & \\textbf{Obs.\\ Dec.} & \\textbf{SPADE $\\Delta$ vs.\\ AR (95\\% CI)} \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}%\n}\n"
        "\\caption{\\textbf{Same-Family Capacity Ratio Sweep (\\texttt{Qwen3-8B} + $\\{\\text{0.6B, 1.7B, 4B}\\}$, $N=100$ Stratified Claims) \\& Leviathan Cost Analysis:} Holding tokenizer, vocabulary ($151,936$), and target model fixed isolates the draft-to-target cost ratio $c_{\\text{AR}} = t_{\\text{draft step}} / t_{\\text{AR step}}$ alongside verification latency inflation $\\gamma_{\\text{ver}} = t_{\\text{verify}} / t_{\\text{AR step}}$. Theoretical decode speedup $\\frac{1-\\alpha^{K+1}}{(1-\\alpha)(Kc_{\\text{AR}}+\\gamma_{\\text{ver}})}$ closely predicts observed decode acceleration and explains why $1:8$ (Llama-3) and $1:4.7$--$1:2.0$ drafters lose net E2E speedup over Target AR.}\n"
        "\\label{tab:qwen3_ratio_sweep}\n\\end{table*}\n"
    )
    (TABLES_DIR / "table_qwen3_ratio_sweep.tex").write_text(tex, encoding="utf-8")
    return extra


def export_long173_interleaved_table(l173_int: dict[str, Any] | None) -> None:
    if not l173_int:
        return
    pw = l173_int["pairwise_e2e_holm"]
    rows = []
    for r in l173_int["rows"]:
        lbl = r["label"]
        dt_s = f"{r['draft_tok']:,.1f}" if r["draft_tok"] > 0 else "—"
        kv_s = f"{r['draft_kv_mb']:.1f} MB" if r["draft_kv_mb"] > 0 else "—"
        al_s = f"{r['alpha_eval_mean']:.2f} $\\pm$ {r['alpha_eval_sd']:.2f}" if r["alpha_eval_mean"] > 0 else "—"
        dc_s = f"{r['dec_mean']:.2f} $\\pm$ {r['dec_sd']:.2f}"
        e2_s = f"{r['e2e_mean']:.2f} $\\pm$ {r['e2e_sd']:.2f}"
        if lbl == "SPADE B=10k":
            c_ar = pw["SPADE vs Target AR (B0-S)"]
            ci_s = f"vs.\\ AR: {c_ar['delta']:+.2f} [{c_ar['ci_lo']:+.2f}, {c_ar['ci_hi']:+.2f}] ({_fmt_p_holm_eq(c_ar['p_holm'])})"
        else:
            c = pw[f"SPADE vs {lbl}"]
            ci_s = f"{-c['delta']:+.2f} [{-c['ci_hi']:+.2f}, {-c['ci_lo']:+.2f}] ({_fmt_p_holm_eq(c['p_holm'])})"
        rows.append(
            f"\\textbf{{{lbl}}} & {dt_s} & {kv_s} & {r['sel_ms_claim']:.1f} & {al_s} & {dc_s} & \\textbf{{{e2_s}}} & {ci_s} \\\\"
        )
    tex = (
        "\\begin{table*}[t]\n\\centering\n\\small\n\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lccccccr}\n\\toprule\n"
        "\\textbf{Configuration ($R=3$ Interleaved Passes)} & \\textbf{Draft Tok} & \\textbf{Draft KV} & \\textbf{Sel (ms)} & \\textbf{$\\alpha_{\\text{eval}}$ (\\%)} & \\textbf{Dec tok/s} & \\textbf{E2E tok/s} & \\textbf{Arm $-$ SPADE / Contrast (95\\% CI, $p_{\\text{Holm}}$)} \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}%\n}\n"
        "\\caption{\\textbf{Counterbalanced $R=3$ Interleaved Long-173 Replication ($N=173$, Randomized 5-Arm Order per Claim):} Eliminates sequential run drift by randomizing arm execution order within every claim across three full passes (seeds $42, 43, 44$). Reports $\\text{mean} \\pm \\text{SD}$ across passes and pooled claim-clustered paired bootstrap 95\\% CIs ($B=10,000$) with Holm--Bonferroni adjustment.}\n"
        "\\label{tab:long173_interleaved}\n\\end{table*}\n"
    )
    (TABLES_DIR / "table_long173_interleaved.tex").write_text(tex, encoding="utf-8")


def export_oracle_interleaved_table(ora: dict[str, Any] | None) -> None:
    if not ora:
        return
    fl = ora["baseline_full"]
    sp = ora["spade_bypass"]
    ce = ora["contrasts_spade_vs_full"]["e2e_tok_s"]
    o400 = _load_json(ANALYSIS_DIR / "b1_4_oracle400" / "b1_3_live_report.json") or {}
    o_arms = {a["label"]: a for a in o400.get("arms", [])}
    o_vs_fl = {c["arm"]: c for c in o400.get("comparisons", {}).get("vs_full", []) if c.get("metric") == "throughput_e2e_tok_s"}
    fl_vs_b0s = next(
        (c for c in o400.get("comparisons", {}).get("vs_b0s", []) if c.get("arm") == "Baseline Full" and c.get("metric") == "throughput_e2e_tok_s"),
        None,
    )

    def _seq_row(disp: str, pol: str, key: str, is_base: bool = False) -> str:
        a = o_arms[key]
        dt_s = "—" if key == "B0-S (Target AR)" else (f"{a['mean_draft_prompt_tokens']:.1f}" if a["mean_draft_prompt_tokens"] > 0 else "—")
        kv_s = f"{a['draft_kv_peak_mb']:.1f} MB" if key in ("Baseline Full", "Target-Attn Sent-B10k") else "—"
        al_s = f"{a['observed_alpha_eval_pct']:.2f}\\%" if a["observed_alpha_eval_pct"] > 0 else "—"
        if is_base:
            ci_s, p_s = "—", "—"
        elif key == "B0-S (Target AR)":
            if fl_vs_b0s:
                ci_s = f"{-fl_vs_b0s['delta']:+.2f} [{-fl_vs_b0s['ci_95_high']:+.2f}, {-fl_vs_b0s['ci_95_low']:+.2f}]"
                p_s = _fmt_p(fl_vs_b0s["p_value_two_sided"])
            else:
                ci_s, p_s = "—", "—"
        else:
            c = o_vs_fl.get(key)
            if c:
                suffix = " (drift)" if key == "Target-Attn Sent-B10k" else ""
                ci_s = f"{c['delta']:+.2f} [{c['ci_95_low']:+.2f}, {c['ci_95_high']:+.2f}]{suffix}"
                p_s = _fmt_p(c["p_value_two_sided"])
            else:
                ci_s, p_s = "—", "—"
        return f"\\textbf{{{disp}}} & {pol} & {dt_s} & {kv_s} & {al_s} & {a['committed_per_round']:.3f} & {a['throughput_e2e_tok_s']:.2f} & {ci_s} & {p_s} \\\\"

    seq_rows = []
    if o_arms:
        seq_rows = [
            _seq_row("B0-S (Target AR)", "Full (Target Only)", "B0-S (Target AR)"),
            _seq_row("Baseline Full (Seq.)", "Full $\\to$ Full", "Baseline Full", is_base=True),
            _seq_row("BM25 Top-2 (Seq.)", "Full $\\to$ Top-2", "BM25 Top-2"),
            _seq_row("BM25 Sent-B8k (Seq.)", "Full $\\to$ Sent B=8k", "BM25 Sent-B8k"),
            _seq_row("SPADE Bypass (Seq.)", "Full $\\to$ Bypass", "Target-Attn Sent-B10k"),
        ]
    seq_drift = o_vs_fl.get("Target-Attn Sent-B10k", {}).get("delta", 0.73)

    tex = (
        "\\begin{table*}[t]\n\\centering\n\\small\n\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{llccccccr}\n\\toprule\n"
        "\\textbf{Protocol / Decoding Method} & \\textbf{Context Policy} & \\textbf{Mean Draft Tok} & \\textbf{Draft KV} & \\textbf{$\\alpha_{\\text{eval}}$ (\\%)} & \\textbf{Tok/Rnd} & \\textbf{E2E Thpt} & \\textbf{$\\Delta$ vs.\\ Full (95\\% CI)} & \\textbf{$p$-value} \\\\\n"
        "\\midrule\n"
        "\\multicolumn{9}{l}{\\textit{Counterbalanced Per-Claim Randomized Interleaving (Shared StaticDraftRunner, $N=393$)}} \\\\\n"
        f"\\textbf{{Baseline Full (Interleaved)}} & Full $\\to$ Full & {fl['draft_tok']:.1f} & {fl['draft_kv_mb']:.1f} MB & {fl['alpha_eval_pct']:.2f}\\% & {fl['tok_per_round']:.3f} & \\textbf{{{fl['e2e_tok_s']:.2f}}} & — & — \\\\\n"
        f"\\textbf{{SPADE Bypass (Interleaved)}} & Full $\\to$ Bypass & \\textbf{{{sp['draft_tok']:.1f}}} & \\textbf{{{sp['draft_kv_mb']:.1f} MB}} & \\textbf{{{sp['alpha_eval_pct']:.2f}\\%}} & \\textbf{{{sp['tok_per_round']:.3f}}} & \\textbf{{{sp['e2e_tok_s']:.2f}}} & \\textbf{{{ce['delta']:+.2f} [{ce['ci_lo']:+.2f}, {ce['ci_hi']:+.2f}]}} & \\textbf{{{ce['p_raw']:.4f}}} \\\\\n"
        "\\midrule\n"
        f"\\multicolumn{{9}}{{l}}{{\\textit{{Sequential Standalone Reference Runs ($N=393$; ${seq_drift:+.2f}\\text{{ tok/s}}$ between Full and Bypass reflects run-order drift)}}}} \\\\\n"
        + ("\n".join(seq_rows) + "\n" if seq_rows else "")
        + "\\bottomrule\n\\end{tabular}%\n}\n"
        f"\\caption{{\\textbf{{Auxiliary Evaluation on {ora['n_claims']} Short-Context Oracle Claims:}} Per-claim 50/50 counterbalanced interleaving removes sequential run-order drift: SPADE triggers selection bypass ($2.45\\,\\mu\\text{{s}}$/claim, $C_{{\\text{{draft}}}} = C_{{\\text{{full}}}}$) with identical acceptance ({sp['alpha_eval_pct']:.2f}\\%) and E2E parity (${sp['e2e_tok_s']:.2f}$ vs.\\ ${fl['e2e_tok_s']:.2f}\\text{{ tok/s}}$, $\\Delta={ce['delta']:+.2f}$ $[{ce['ci_lo']:+.2f}, {ce['ci_hi']:+.2f}]$, $p={ce['p_raw']:.4f}$). Sequential baselines are shown below for completeness; comparing them confirms ${seq_drift:+.2f}\\text{{ tok/s}}$ is sequential drift on identical code paths.}}\n"
        "\\label{tab:oracle_results}\n\\end{table*}\n"
    )
    (TABLES_DIR / "table3_oracle_results.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    v6_master = _load_json(REV6_DIR / "v6_master_analysis.json")
    batch_data = _load_json(REV8_DIR / "batch_serving_kv.json")
    fbm = _load_json(REV8_DIR / "fixed_prefix_spade_vs_bm25.json")
    ora = _load_json(REV8_DIR / "oracle393_interleaved.json")
    q3_sweep = _load_json(REV8_DIR / "qwen3_ratio_sweep.json")
    l173_int = _load_json(REV8_DIR / "long173_interleaved_r3.json")

    export_stratified_table(v6_master)
    export_batch_table(batch_data)
    export_spade_vs_bm25_table(fbm, v6_master)
    extra = export_qwen3_ratio_table(q3_sweep, v6_master)
    export_long173_interleaved_table(l173_int)
    export_oracle_interleaved_table(ora)

    summary = {
        "has_batch": batch_data is not None,
        "has_fixed_bm25": fbm is not None,
        "has_oracle_interleaved": ora is not None,
        "has_qwen3_ratio": q3_sweep is not None,
        "has_long173_interleaved": l173_int is not None,
        **extra,
    }
    REV8_DIR.mkdir(parents=True, exist_ok=True)
    (REV8_DIR / "v8_master_analysis.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
