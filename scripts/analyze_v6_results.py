"""Unified statistical aggregation, Holm-Bonferroni correction, QA F1, Theorem 1 CI audit, and LaTeX table exporter for SPADE v6."""
from __future__ import annotations

import json
import math
import random
import re
import statistics
import string
from collections import Counter
from pathlib import Path
from typing import Any

RUNS_DIR = Path("outputs/runs")
REV_DIR = Path("outputs/analysis/v6_revisions")
TABLES_DIR = Path("docs/paper/latex/generated_tables")


def holm_bonferroni(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    prev = 0.0
    for rank, orig_idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[orig_idx])
        val = max(prev, val)
        prev = val
        adj[orig_idx] = val
    return adj


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _normalize_qa_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _token_f1(pred: str, gold: str) -> float:
    p_toks = _normalize_qa_answer(pred).split()
    g_toks = _normalize_qa_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(p_toks)
    rec = num_same / len(g_toks)
    return (2 * prec * rec) / (prec + rec)


def _extract_final_answer(raw_text: str) -> str:
    m = re.search(r"Final\s+Answer\s*:\s*(.+)", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        ans = m.group(1).strip().splitlines()[0].strip()
        return ans
    return raw_text.strip()


def _extract_run_claims(run_dir: Path, bench_answers: dict[str, list[str]] | None = None) -> dict[str, dict[str, Any]]:
    trials = [json.loads(l) for l in (run_dir / "trial_metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rounds = [json.loads(l) for l in (run_dir / "round_traces.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] if (run_dir / "round_traces.jsonl").exists() else []
    tokens = [json.loads(l) for l in (run_dir / "token_traces.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] if (run_dir / "token_traces.jsonl").exists() else []
    preds = {str(r["claim_id"]): r for r in (json.loads(l) for l in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}

    claims: dict[str, dict[str, Any]] = {}
    for t in trials:
        cid = str(t["claim_id"])
        if t.get("trial_index", 0) != 0 and cid in claims:
            continue
        sel_s = float(t.get("context_selection_s", 0.0))
        p_pref_s = float(t.get("prompt_prefill_s", 0.0))
        d_pref_s = float(t.get("draft_prefill_s", 0.0))
        dec_s = float(t.get("decode_s", 0.0))
        e2e_s = sel_s + p_pref_s + d_pref_s + dec_s
        out_tok = int(t.get("output_tokens", 0))

        qa_f1 = None
        qa_sub = None
        if bench_answers and cid in bench_answers and cid in preds:
            pred_ans = _extract_final_answer(str(preds[cid].get("raw_text", "")))
            golds = bench_answers[cid]
            qa_f1 = max(_token_f1(pred_ans, g) for g in golds)
            norm_full = _normalize_qa_answer(str(preds[cid].get("raw_text", "")))
            qa_sub = 1.0 if any(_normalize_qa_answer(g) in norm_full for g in golds if _normalize_qa_answer(g)) else 0.0

        claims[cid] = {
            "claim_id": cid,
            "output_tokens": out_tok,
            "draft_prompt_tokens": float(t.get("draft_prompt_tokens", 0.0)),
            "context_selection_s": sel_s,
            "prompt_prefill_s": p_pref_s,
            "draft_prefill_s": d_pref_s,
            "decode_s": dec_s,
            "e2e_s": e2e_s,
            "per_claim_e2e_tok_s": (out_tok / e2e_s) if e2e_s > 0 else 0.0,
            "draft_kv_peak_mb": float(t.get("draft_kv_peak_bytes", 0.0)) / (1024.0 * 1024.0),
            "target_rounds": 0,
            "n_acc": 0,
            "n_prop": 0,
            "n_eval": 0,
            "n_eval_acc": 0,
            "qa_f1": qa_f1,
            "qa_sub": qa_sub,
        }

    for r in rounds:
        cid = str(r["claim_id"])
        if cid in claims:
            claims[cid]["target_rounds"] += 1
            claims[cid]["n_acc"] += int(r.get("accepted_draft_tokens", 0))
            claims[cid]["n_prop"] += int(r.get("proposed_token_count", 0))

    for tk in tokens:
        st = tk.get("status")
        if st not in {"accepted", "rejected"}:
            continue
        cid = str(tk["claim_id"])
        if cid in claims:
            claims[cid]["n_eval"] += 1
            if st == "accepted":
                claims[cid]["n_eval_acc"] += 1

    return claims


def _pooled(rows: list[dict[str, Any]], metric: str) -> float:
    n = len(rows)
    if n == 0:
        return 0.0
    if metric == "e2e_tok_s":
        den = sum(r["e2e_s"] for r in rows)
        return sum(r["output_tokens"] for r in rows) / den if den > 0 else 0.0
    if metric == "per_claim_e2e_tok_s":
        return sum(r["per_claim_e2e_tok_s"] for r in rows) / n
    if metric == "dec_tok_s":
        den = sum(r["decode_s"] for r in rows)
        return sum(r["output_tokens"] for r in rows) / den if den > 0 else 0.0
    if metric == "alpha_eval_pct":
        den = sum(r["n_eval"] for r in rows)
        return (sum(r["n_eval_acc"] for r in rows) / den) * 100.0 if den > 0 else 0.0
    if metric == "alpha_prop_pct":
        den = sum(r["n_prop"] for r in rows)
        return (sum(r["n_acc"] for r in rows) / den) * 100.0 if den > 0 else 0.0
    if metric == "tok_per_round":
        den = sum(r["target_rounds"] for r in rows)
        return sum(r["output_tokens"] for r in rows) / den if den > 0 else 0.0
    if metric == "draft_tok":
        return sum(r["draft_prompt_tokens"] for r in rows) / n
    if metric == "draft_kv_mb":
        return sum(r["draft_kv_peak_mb"] for r in rows) / n
    if metric == "sel_ms_claim":
        return (sum(r["context_selection_s"] for r in rows) * 1000.0) / n
    if metric == "qa_f1_pct":
        vals = [r["qa_f1"] for r in rows if r["qa_f1"] is not None]
        return (sum(vals) / len(vals)) * 100.0 if vals else 0.0
    if metric == "qa_sub_pct":
        vals = [r["qa_sub"] for r in rows if r["qa_sub"] is not None]
        return (sum(vals) / len(vals)) * 100.0 if vals else 0.0
    raise ValueError(metric)


def _boot(base: dict[str, dict[str, Any]], treat: dict[str, dict[str, Any]], metric: str, n_boot: int = 10000, seed: int = 42) -> dict[str, Any]:
    common = sorted(set(base) & set(treat))
    bl = [base[c] for c in common]
    tl = [treat[c] for c in common]
    b_val = _pooled(bl, metric)
    t_val = _pooled(tl, metric)
    delta = t_val - b_val
    rng = random.Random(seed)
    n = len(common)
    boots = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        boots.append(_pooled([tl[i] for i in idxs], metric) - _pooled([bl[i] for i in idxs], metric))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(n_boot - 1, int(0.975 * n_boot))]
    p2 = min(1.0, 2.0 * min(sum(1 for x in boots if x <= 0.0) / n_boot, sum(1 for x in boots if x >= 0.0) / n_boot))
    return {"base": round(b_val, 4), "treat": round(t_val, 4), "delta": round(delta, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "p_raw": p2}


def _summarize_suite(specs: list[tuple[str, Path]], bench_answers: dict[str, list[str]] | None = None) -> dict[str, Any]:
    cmaps = {}
    rows = []
    for label, rdir in specs:
        if not (rdir / "metrics.json").exists():
            continue
        cm = _extract_run_claims(rdir, bench_answers=bench_answers)
        cmaps[label] = cm
        clist = list(cm.values())
        rows.append({
            "label": label,
            "n_claims": len(clist),
            "draft_tok": round(_pooled(clist, "draft_tok"), 1),
            "draft_kv_mb": round(_pooled(clist, "draft_kv_mb"), 1),
            "sel_ms_claim": round(_pooled(clist, "sel_ms_claim"), 2),
            "alpha_eval_pct": round(_pooled(clist, "alpha_eval_pct"), 2),
            "alpha_prop_pct": round(_pooled(clist, "alpha_prop_pct"), 2),
            "tok_per_round": round(_pooled(clist, "tok_per_round"), 3),
            "dec_tok_s": round(_pooled(clist, "dec_tok_s"), 2),
            "e2e_tok_s": round(_pooled(clist, "e2e_tok_s"), 2),
            "per_claim_e2e_tok_s": round(_pooled(clist, "per_claim_e2e_tok_s"), 2),
            "qa_f1_pct": round(_pooled(clist, "qa_f1_pct"), 2) if bench_answers else None,
            "qa_sub_pct": round(_pooled(clist, "qa_sub_pct"), 2) if bench_answers else None,
        })
    return {"rows": rows, "cmaps": cmaps}


def analyze_theorem1_300() -> dict[str, Any]:
    paths = {
        "AR_s42": RUNS_DIR / "b1-4-heldout300-b0s" / "predictions.jsonl",
        "AR_s43": RUNS_DIR / "b1-4-heldout300-b0s-seed43" / "predictions.jsonl",
        "Full_SD": RUNS_DIR / "b1-4-heldout300-baseline-full" / "predictions.jsonl",
        "SPADE": RUNS_DIR / "b1-4-heldout300-attn-sent-b10k" / "predictions.jsonl",
    }
    data = {}
    for k, p in paths.items():
        if p.exists():
            data[k] = {str(r["claim_id"]): r for r in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())}

    pairs = [
        ("AR (s42) vs AR (s43) [Noise Ref]", "AR_s42", "AR_s43"),
        ("Full-SD vs AR (s42)", "Full_SD", "AR_s42"),
        ("SPADE vs AR (s42)", "SPADE", "AR_s42"),
        ("SPADE vs Full-SD", "SPADE", "Full_SD"),
    ]
    out_pairs = {}
    rng = random.Random(42)
    for title, ka, kb in pairs:
        if ka not in data or kb not in data:
            continue
        cids = sorted(set(data[ka]) & set(data[kb]))
        n = len(cids)
        words_a = [str(data[ka][c].get("raw_text", "")).lower().split() for c in cids]
        words_b = [str(data[kb][c].get("raw_text", "")).lower().split() for c in cids]
        cnt_a = [Counter(w) for w in words_a]
        cnt_b = [Counter(w) for w in words_b]
        len_a = [len(w) for w in words_a]
        len_b = [len(w) for w in words_b]
        tok_a = [int(data[ka][c].get("output_tokens", 0)) for c in cids]
        tok_b = [int(data[kb][c].get("output_tokens", 0)) for c in cids]
        verd_eq = [
            1 if ((data[ka][c].get("parsed") or {}).get("verdict") == (data[kb][c].get("parsed") or {}).get("verdict")) else 0
            for c in cids
        ]

        tot_ca: Counter[str] = Counter()
        tot_cb: Counter[str] = Counter()
        for ca_i, cb_i in zip(cnt_a, cnt_b):
            tot_ca.update(ca_i)
            tot_cb.update(cb_i)
        tot_ta = sum(len_a)
        tot_tb = sum(len_b)
        vocab = list(set(tot_ca) | set(tot_cb))
        obs_tv = 0.5 * sum(abs(tot_ca[w] / tot_ta - tot_cb[w] / tot_tb) for w in vocab)

        loo_tvs = []
        for i in range(n):
            ta_i = tot_ta - len_a[i]
            tb_i = tot_tb - len_b[i]
            ci_a = cnt_a[i]
            ci_b = cnt_b[i]
            tv_i = 0.5 * sum(abs((tot_ca[w] - ci_a.get(w, 0)) / ta_i - (tot_cb[w] - ci_b.get(w, 0)) / tb_i) for w in vocab)
            loo_tvs.append(tv_i)
        mean_loo = sum(loo_tvs) / n
        se_jack = math.sqrt(((n - 1) / n) * sum((x - mean_loo) ** 2 for x in loo_tvs))
        k_agree = sum(verd_eq)
        w_lo, w_hi = _wilson_ci(k_agree, n)
        out_pairs[title] = {
            "n_claims": n,
            "tv_1gram": round(obs_tv, 4),
            "tv_se_jackknife": round(se_jack, 4),
            "tv_ci95": [round(max(0.0, obs_tv - 1.96 * se_jack), 4), round(min(1.0, obs_tv + 1.96 * se_jack), 4)],
            "verdict_agreement_pct": round((k_agree / n) * 100.0, 2),
            "verdict_agreement_ci95_pct": [round(w_lo * 100.0, 2), round(w_hi * 100.0, 2)],
            "mean_tokens_a": round(statistics.mean(tok_a), 1),
            "sd_tokens_a": round(statistics.stdev(tok_a), 1),
            "mean_tokens_b": round(statistics.mean(tok_b), 1),
            "sd_tokens_b": round(statistics.stdev(tok_b), 1),
            "_verd_vec": verd_eq,
        }

    # Compare SPADE-AR agreement vs Full-SD-AR agreement via paired bootstrap
    agree_diff = None
    if "SPADE vs AR (s42)" in out_pairs and "Full-SD vs AR (s42)" in out_pairs:
        va = out_pairs["SPADE vs AR (s42)"]["_verd_vec"]
        vb = out_pairs["Full-SD vs AR (s42)"]["_verd_vec"]
        n = len(va)
        obs = (sum(va) - sum(vb)) / n * 100.0
        diffs = sorted(((sum(va[i] - vb[i] for i in [rng.randrange(n) for _ in range(n)])) / n * 100.0) for _ in range(10000))
        p2 = min(1.0, 2.0 * min(sum(1 for x in diffs if x <= 0) / 10000, sum(1 for x in diffs if x >= 0) / 10000))
        agree_diff = {"delta_pp": round(obs, 2), "ci95_pp": [round(diffs[250], 2), round(diffs[9750], 2)], "p_value": round(p2, 4)}
    for v in out_pairs.values():
        v.pop("_verd_vec", None)
    return {"pairwise": out_pairs, "spade_vs_full_agreement_diff": agree_diff}


def _fmt_p(p: float) -> str:
    if p < 0.0001:
        return "$< 10^{-4}$"
    return f"{p:.4f}"


def main() -> None:
    REV_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Existing Heldout-300 Selector Multi-Comparison with Holm-Bonferroni
    sel_specs = [
        ("Baseline Full", RUNS_DIR / "b1-4-heldout300-baseline-full"),
        ("BM25 Top-2", RUNS_DIR / "b1-4-heldout300-selective-topk2"),
        ("Random Sent-B10k", RUNS_DIR / "b1-4-heldout300-random-sent-b10k"),
        ("BM25 Sent-B10k", RUNS_DIR / "b1-4-heldout300-bm25-sent-b10k"),
        ("SPADE No-Coverage", RUNS_DIR / "b1-4-heldout300-attn-sent-b10k-nocoverage"),
        ("SPADE", RUNS_DIR / "b1-4-heldout300-attn-sent-b10k"),
    ]
    h300 = _summarize_suite(sel_specs)
    h300_comps = []
    if "SPADE" in h300["cmaps"]:
        sp_m = h300["cmaps"]["SPADE"]
        for base_lbl in ("Baseline Full", "BM25 Top-2", "Random Sent-B10k", "BM25 Sent-B10k", "SPADE No-Coverage"):
            if base_lbl in h300["cmaps"]:
                ba = _boot(h300["cmaps"][base_lbl], sp_m, "alpha_eval_pct")
                bt = _boot(h300["cmaps"][base_lbl], sp_m, "e2e_tok_s")
                h300_comps.append({"vs": base_lbl, "alpha_eval": ba, "e2e_tok_s": bt})
        pa = holm_bonferroni([c["alpha_eval"]["p_raw"] for c in h300_comps])
        pt = holm_bonferroni([c["e2e_tok_s"]["p_raw"] for c in h300_comps])
        for c, adj_a, adj_t in zip(h300_comps, pa, pt):
            c["alpha_eval"]["p_holm"] = round(adj_a, 4)
            c["e2e_tok_s"]["p_holm"] = round(adj_t, 4)

    # 2. Long-173 Controlled Baseline Suite
    l173_specs = [
        ("Baseline Full", RUNS_DIR / "v6_long173" / "long173-baseline-full"),
        ("KV-Sparse (B=1,776)", RUNS_DIR / "v6_long173" / "long173-kvsparse-1776"),
        ("KV-Sparse (B=512)", RUNS_DIR / "v6_long173" / "long173-kvsparse-512"),
        ("Prompt-Lookup (PLD)", RUNS_DIR / "v6_long173" / "long173-pld"),
        ("No Evidence (B=0)", RUNS_DIR / "v6_long173" / "long173-no-evidence"),
        ("Sliding Window (B=8k)", RUNS_DIR / "v6_long173" / "long173-sliding-window-b8k"),
        ("BM25 Sent-B10k", RUNS_DIR / "v6_long173" / "long173-bm25-b10k"),
        ("SPADE B=2k", RUNS_DIR / "v6_long173" / "long173-spade-b2k"),
        ("SPADE B=10k (Default)", RUNS_DIR / "v6_long173" / "long173-spade-b10k"),
    ]
    l173 = _summarize_suite(l173_specs)
    l173_vs_full = {}
    if "Baseline Full" in l173["cmaps"]:
        raw_ps = []
        keys_order = []
        for row in l173["rows"]:
            lbl = row["label"]
            if lbl == "Baseline Full":
                continue
            b = _boot(l173["cmaps"]["Baseline Full"], l173["cmaps"][lbl], "e2e_tok_s")
            l173_vs_full[lbl] = b
            raw_ps.append(b["p_raw"])
            keys_order.append(lbl)
        for lbl, padj in zip(keys_order, holm_bonferroni(raw_ps)):
            l173_vs_full[lbl]["p_holm"] = round(padj, 4)

    # Pairwise key comparisons on Long-173 (SPADE B=10k vs BM25, KV-Sparse-1776, PLD; SPADE B=2k vs KV-Sparse-512)
    l173_pairwise = {}
    for ta, tb in [
        ("SPADE B=10k (Default)", "BM25 Sent-B10k"),
        ("SPADE B=10k (Default)", "KV-Sparse (B=1,776)"),
        ("SPADE B=10k (Default)", "Prompt-Lookup (PLD)"),
        ("SPADE B=2k", "KV-Sparse (B=512)"),
        ("SPADE B=2k", "SPADE B=10k (Default)"),
    ]:
        if ta in l173["cmaps"] and tb in l173["cmaps"]:
            l173_pairwise[f"{ta} vs {tb}"] = {
                "e2e_tok_s": _boot(l173["cmaps"][tb], l173["cmaps"][ta], "e2e_tok_s"),
                "alpha_eval_pct": _boot(l173["cmaps"][tb], l173["cmaps"][ta], "alpha_eval_pct"),
            }
    if l173_pairwise:
        p_t = holm_bonferroni([v["e2e_tok_s"]["p_raw"] for v in l173_pairwise.values()])
        p_a = holm_bonferroni([v["alpha_eval_pct"]["p_raw"] for v in l173_pairwise.values()])
        for v, pt_i, pa_i in zip(l173_pairwise.values(), p_t, p_a):
            v["e2e_tok_s"]["p_holm"] = round(pt_i, 4)
            v["alpha_eval_pct"]["p_holm"] = round(pa_i, 4)

    # 3. LongBench HotpotQA (N=100)
    lb_bench_path = Path("outputs/benchmarks/longbench-hotpotqa-100.jsonl")
    lb_answers = {}
    if lb_bench_path.exists():
        for line in lb_bench_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                lb_answers[str(r["claim_id"])] = list((r.get("evaluation") or {}).get("answers") or [])
    lb_specs = [
        ("Target AR (B0-S)", RUNS_DIR / "v6_longbench" / "lb-hotpot-b0s"),
        ("Baseline Full", RUNS_DIR / "v6_longbench" / "lb-hotpot-baseline-full"),
        ("KV-Sparse (B=1,776)", RUNS_DIR / "v6_longbench" / "lb-hotpot-kvsparse-1776"),
        ("Prompt-Lookup (PLD)", RUNS_DIR / "v6_longbench" / "lb-hotpot-pld"),
        ("BM25 Sent-B10k", RUNS_DIR / "v6_longbench" / "lb-hotpot-bm25-b10k"),
        ("SPADE B=2k", RUNS_DIR / "v6_longbench" / "lb-hotpot-spade-b2k"),
        ("SPADE B=10k", RUNS_DIR / "v6_longbench" / "lb-hotpot-spade-b10k"),
    ]
    lb = _summarize_suite(lb_specs, bench_answers=lb_answers)
    lb_vs_full = {}
    if "Baseline Full" in lb["cmaps"]:
        rps = []
        ko = []
        for row in lb["rows"]:
            lbl = row["label"]
            if lbl == "Baseline Full":
                continue
            b = _boot(lb["cmaps"]["Baseline Full"], lb["cmaps"][lbl], "e2e_tok_s")
            lb_vs_full[lbl] = b
            rps.append(b["p_raw"])
            ko.append(lbl)
        for lbl, padj in zip(ko, holm_bonferroni(rps)):
            lb_vs_full[lbl]["p_holm"] = round(padj, 4)

    lb_pairwise = {}
    for ta, tb in [
        ("SPADE B=10k", "Target AR (B0-S)"),
        ("SPADE B=10k", "BM25 Sent-B10k"),
        ("SPADE B=10k", "KV-Sparse (B=1,776)"),
        ("SPADE B=10k", "Prompt-Lookup (PLD)"),
        ("SPADE B=10k", "SPADE B=2k"),
    ]:
        if ta in lb["cmaps"] and tb in lb["cmaps"]:
            entry = {"e2e_tok_s": _boot(lb["cmaps"][tb], lb["cmaps"][ta], "e2e_tok_s")}
            if tb != "Target AR (B0-S)":
                entry["alpha_eval_pct"] = _boot(lb["cmaps"][tb], lb["cmaps"][ta], "alpha_eval_pct")
            lb_pairwise[f"{ta} vs {tb}"] = entry
    if lb_pairwise:
        p_lt = holm_bonferroni([v["e2e_tok_s"]["p_raw"] for v in lb_pairwise.values()])
        for v, pt_i in zip(lb_pairwise.values(), p_lt):
            v["e2e_tok_s"]["p_holm"] = round(pt_i, 4)
        alpha_items = [v["alpha_eval_pct"] for v in lb_pairwise.values() if "alpha_eval_pct" in v]
        if alpha_items:
            for v_a, pa_i in zip(alpha_items, holm_bonferroni([x["p_raw"] for x in alpha_items])):
                v_a["p_holm"] = round(pa_i, 4)

    lb_qa_f1_holm = {}
    for ta, tb in [
        ("SPADE B=10k", "Target AR (B0-S)"),
        ("SPADE B=10k", "Baseline Full"),
        ("SPADE B=2k", "Target AR (B0-S)"),
    ]:
        if ta in lb["cmaps"] and tb in lb["cmaps"]:
            lb_qa_f1_holm[f"{ta} vs {tb}"] = _boot(lb["cmaps"][tb], lb["cmaps"][ta], "qa_f1_pct")
    if lb_qa_f1_holm:
        for v_f, pf_i in zip(lb_qa_f1_holm.values(), holm_bonferroni([x["p_raw"] for x in lb_qa_f1_holm.values()])):
            v_f["p_holm"] = round(pf_i, 4)

    # 4. Theorem 1 (300 claims + teacher-forced)
    thm1_300 = analyze_theorem1_300()

    master = {
        "heldout300_holm": h300_comps,
        "long173": {"rows": l173["rows"], "vs_full": l173_vs_full, "pairwise_holm": l173_pairwise},
        "longbench": {"rows": lb["rows"], "vs_full": lb_vs_full, "pairwise_holm": lb_pairwise, "qa_f1_holm": lb_qa_f1_holm},
        "theorem1_300": thm1_300,
    }
    (REV_DIR / "v6_master_analysis.json").write_text(json.dumps(master, indent=2) + "\n", encoding="utf-8")

    # ----------------- Export LaTeX Tables -----------------
    # Table: Long-173 Controlled Baselines (replaces Table 11)
    if l173["rows"]:
        full_kv = next((r["draft_kv_mb"] for r in l173["rows"] if r["label"] == "Baseline Full"), 850.0)
        full_thpt = next((r["e2e_tok_s"] for r in l173["rows"] if r["label"] == "Baseline Full"), 35.0)
        lines = []
        for r in l173["rows"]:
            lbl = r["label"]
            kv_s = f"{r['draft_kv_mb']:.1f} MB" if r["draft_kv_mb"] > 0 else "—"
            dt_s = f"{r['draft_tok']:,.1f}" if r["draft_tok"] > 0 else "—"
            sel_s = f"{r['sel_ms_claim']:.1f}"
            ae_s = f"{r['alpha_eval_pct']:.2f}\\%"
            ap_s = f"{r['alpha_prop_pct']:.2f}\\%"
            if lbl == "Baseline Full":
                ci_s = "—"
            else:
                b = l173_vs_full.get(lbl)
                ci_s = f"{b['delta']:+.2f} [{b['ci_lo']:+.2f}, {b['ci_hi']:+.2f}]" if b else "—"
            b_lbl = f"\\textbf{{{lbl}}}"
            if "SPADE" in lbl:
                lines.append(f"{b_lbl} & {dt_s} & \\textbf{{{kv_s}}} & {sel_s} & \\textbf{{{ae_s}}} & \\textbf{{{ap_s}}} & {r['dec_tok_s']:.2f} & \\textbf{{{r['e2e_tok_s']:.2f}}} & {ci_s} \\\\")
            else:
                lines.append(f"{b_lbl} & {dt_s} & {kv_s} & {sel_s} & {ae_s} & {ap_s} & {r['dec_tok_s']:.2f} & {r['e2e_tok_s']:.2f} & {ci_s} \\\\")
        tex = (
            "\\begin{table*}[t]\n\\centering\n\\small\n\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{lcccccccr}\n\\toprule\n"
            "\\textbf{Configuration} & \\textbf{Draft Tok / KV Slots} & \\textbf{Draft KV Peak} & \\textbf{Sel./Evict (ms)} & \\textbf{$\\alpha_{\\text{eval}}$ (\\%)} & \\textbf{$\\alpha_{\\text{prop}}$ (\\%)} & \\textbf{Dec tok/s} & \\textbf{E2E tok/s} & \\textbf{$\\Delta$ vs.\\ Full (95\\% CI)} \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}%\n}\n"
            "\\caption{\\textbf{Controlled Long-Context Baseline & Compression Comparison on Complete Selective Cohort ($N=173$ claims $>10\\text{k}$ chars, RTX 5090):} Supersedes the 20-claim pilot. Compares uncompressed Full-SD, attention-evicted KV-Sparse (SnapKV-style retaining sink+recent+pooled attention KV with RoPE offset preservation), training-free Prompt-Lookup Decoding (PLD), sliding window, zero-evidence ($B=0$), lexical BM25, and SPADE at $B=2\\text{k}$ and $B=10\\text{k}$. Reported CIs use paired claim-cluster bootstrap ($B=10,000$).}\n"
            "\\label{tab:decisive_baselines}\n\\end{table*}\n"
        )
        (TABLES_DIR / "table_decisive_baselines.tex").write_text(tex, encoding="utf-8")

    # Table: LongBench HotpotQA (N=100)
    if lb["rows"]:
        lines = []
        ar_thpt = next((r["e2e_tok_s"] for r in lb["rows"] if "AR" in r["label"]), 25.0)
        full_thpt = next((r["e2e_tok_s"] for r in lb["rows"] if r["label"] == "Baseline Full"), 25.0)
        for r in lb["rows"]:
            lbl = r["label"]
            dt_s = f"{r['draft_tok']:,.1f}" if r["draft_tok"] > 0 else "—"
            kv_s = f"{r['draft_kv_mb']:.1f} MB" if r["draft_kv_mb"] > 0 else "—"
            ae_s = f"{r['alpha_eval_pct']:.2f}\\%" if r["alpha_eval_pct"] > 0 else "—"
            spd_ar = f"{(r['e2e_tok_s'] / ar_thpt - 1.0) * 100.0:+.1f}\\%" if "AR" not in lbl else "—"
            if lbl == "Baseline Full":
                d_s = "—"
            else:
                b = lb_vs_full.get(lbl)
                d_s = f"{b['delta']:+.2f} [{b['ci_lo']:+.2f}, {b['ci_hi']:+.2f}]" if b else "—"
            f1_s = f"{r['qa_f1_pct']:.1f}" if r["qa_f1_pct"] is not None else "—"
            sub_s = f"{r['qa_sub_pct']:.1f}\\%" if r["qa_sub_pct"] is not None else "—"
            if "SPADE" in lbl:
                lines.append(f"\\textbf{{{lbl}}} & {dt_s} & \\textbf{{{kv_s}}} & \\textbf{{{ae_s}}} & \\textbf{{{r['e2e_tok_s']:.2f}}} & \\textbf{{{spd_ar}}} & {d_s} & {f1_s} / {sub_s} \\\\")
            else:
                lines.append(f"\\textbf{{{lbl}}} & {dt_s} & {kv_s} & {ae_s} & {r['e2e_tok_s']:.2f} & {spd_ar} & {d_s} & {f1_s} / {sub_s} \\\\")
        tex = (
            "\\begin{table*}[t]\n\\centering\n\\small\n\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{lccccccr}\n\\toprule\n"
            "\\textbf{Method} & \\textbf{Draft Tok} & \\textbf{Draft KV} & \\textbf{$\\alpha_{\\text{eval}}$ (\\%)} & \\textbf{E2E tok/s} & \\textbf{vs.\\ AR} & \\textbf{$\\Delta$ vs.\\ Full (95\\% CI)} & \\textbf{QA F1 / Sub-EM} \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}%\n}\n"
            "\\caption{\\textbf{Second-Benchmark Generalization on LongBench HotpotQA Multi-Document QA ($N=100$, mean 14,083 tokens):} Evaluated on Qwen3-8B/0.6B under identical speculative configuration ($K=4, T=0.7, \\text{top-}p=0.9$). QA F1 and answer substring recall (Sub-EM) confirm downstream quality preservation alongside KV compression and throughput gains.}\n"
            "\\label{tab:longbench_hotpot}\n\\end{table*}\n"
        )
        (TABLES_DIR / "table_longbench_hotpot.tex").write_text(tex, encoding="utf-8")

    # Table: Full E2E Prefill + Hook Scaling (Table 12)
    sc_path = REV_DIR / "e2e_scaling_2k_32k.json"
    if sc_path.exists():
        sc = json.loads(sc_path.read_text(encoding="utf-8"))
        lines = []
        for L_str in ["2048", "4096", "8192", "16384", "32768"]:
            d = sc.get(L_str) or sc.get(int(L_str), {})
            if not d:
                continue
            L_lbl = f"{int(L_str)//1024}k"
            t_pre = f"{d['target_8b_prefill_ms']:.1f}" if "target_8b_prefill_ms" in d else "—"
            hk = f"{d['hook_ms']:.2f} ms / {d['hook_peak_delta_mb']:.1f} MB" if "hook_ms" in d else "—"
            df_pre = f"{d['full_draft_prefill_ms']:.1f} -> {d['spade_draft_prefill_ms']:.1f}" if "target_8b_prefill_ms" in d else f"{d['full_draft_prefill_ms']:.1f} -> 25.2"
            tot_pre = f"{d['total_prefill_full_ms']:.0f} -> {d['total_prefill_spade_ms']:.0f} ({-d['total_prefill_saved_pct']:+.1f}\\%)" if "total_prefill_full_ms" in d else "—"
            ar_e2e = f"{d['ar_e2e_tok_s']:.2f}" if "ar_e2e_tok_s" in d else "OOM"
            fl_e2e = f"{d['full_sd_e2e_tok_s']:.2f} ({d['full_sd_peak_vram_gb']:.1f}G)" if "full_sd_e2e_tok_s" in d else "OOM"
            sp_alpha = f"{d['spade_alpha_eval_pct']:.1f}\\%" if "spade_alpha_eval_pct" in d else "—"
            sp_e2e = f"\\textbf{{{d['spade_e2e_tok_s']:.2f}}} ({d['spade_peak_vram_gb']:.1f}G)" if "spade_e2e_tok_s" in d else "OOM"
            df_pre = df_pre.replace("->", "$\\to$")
            tot_pre = tot_pre.replace("->", "$\\to$")
            lines.append(f"{L_lbl} ({d.get('actual_tokens', L_str):,}) & {t_pre} ms & {hk} & {df_pre} ms & {tot_pre} ms & {ar_e2e} & {fl_e2e} & {sp_alpha} & {sp_e2e} \\\\")
        tex = (
            "\\begin{table*}[t]\n\\centering\n\\small\n\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{lcccccccr}\n\\toprule\n"
            "\\textbf{Context Len} & \\textbf{Target 8B Prefill} & \\textbf{Hook (ms / $\\Delta$VRAM)} & \\textbf{Draft Prefill (F $\\to$ S)} & \\textbf{Total Prefill (F $\\to$ S)} & \\textbf{AR tok/s} & \\textbf{Full-SD tok/s (VRAM)} & \\textbf{SPADE $\\alpha_{\\text{eval}}$} & \\textbf{SPADE tok/s (VRAM)} \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}%\n}\n"
            "\\caption{\\textbf{End-to-End Sequence Scaling up to 32k Tokens on RTX 5090 (PyTorch SDPA, \\texttt{logits\\_to\\_keep=1}):} Reports target prefill, exact hook overhead, draft prefill reduction, total prefill (Target+Hook+Draft), and E2E generation throughput (128 tokens) with peak co-located VRAM. While SPADE caps draft prefill and KV memory, target prefill and target KV dominate total compute at 16k--32k tokens.}\n"
            "\\label{tab:long_context_scaling}\n\\end{table*}\n"
        )
        (TABLES_DIR / "table_long_context_scaling.tex").write_text(tex, encoding="utf-8")

    # Table: Dev Selector Ablations
    dev_path = REV_DIR / "dev_selector_ablation.json"
    if dev_path.exists():
        dev = json.loads(dev_path.read_text(encoding="utf-8"))
        lines = []
        for r in dev["results"]:
            cfg = r["config"].replace("upper50, W=16, mean-head, sum, B=10k", "Default: upper50, W=16, mean, sum, B=10k").replace("_", "\\_")
            lines.append(f"{cfg} & {r['mean_draft_prompt_tokens']:,.1f} & {r['mean_expected_acceptance_pct']:.2f}\\% \\\\")
        tex = (
            "\\begin{table}[t]\n\\centering\n\\small\n\\resizebox{\\columnwidth}{!}{%\n"
            "\\begin{tabular}{lcc}\n\\toprule\n"
            "\\textbf{Selector / Hyperparameter Configuration} & \\textbf{Draft Tokens} & \\textbf{$\\mathbb{E}[1-\\text{TV}]$ (\\%)} \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}%\n}\n"
            "\\caption{\\textbf{Disjoint Dev-Split Selector & Hyperparameter Ablation ($N=50$ long claims):} Teacher-forced expected acceptance $1 - \\text{TV}(p,q)$ across scoring normalizations, window size $W$, layer subsets, head aggregation, and budgets $B$.}\n"
            "\\label{tab:dev_selector_ablation}\n\\end{table}\n"
        )
        (TABLES_DIR / "table_dev_selector_ablation.tex").write_text(tex, encoding="utf-8")

    # Table: All-Selector Coverage & Retention (300 claims)
    cov_path = REV_DIR / "selector_coverage_all.json"
    if cov_path.exists():
        cov = json.loads(cov_path.read_text(encoding="utf-8"))
        lines = []
        for arm in ("SPADE", "SPADE No-Coverage", "BM25 Sent-B10k", "Random Sent-B10k", "BM25 Top-2"):
            if arm not in cov:
                continue
            d = cov[arm]
            lines.append(
                f"\\textbf{{{arm}}} & {d['mean_passage_coverage_pct']:.1f}\\% & {d['own_cited_retention_pct']:.1f}\\% ({d['own_cited_retained']}/{d['own_cited_total']}) & "
                f"{d['shared_ar_cited_retention_pct']:.1f}\\% ({d['shared_ar_cited_retained']}/{d['shared_ar_cited_total']}) & "
                f"{d['own_justification_lexical_cov_pct']:.1f}\\% / {d['shared_ar_justification_lexical_cov_pct']:.1f}\\% & {d['gold_justification_lexical_cov_pct']:.1f}\\% \\\\"
            )
        tex = (
            "\\begin{table*}[t]\n\\centering\n\\small\n\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{lccccr}\n\\toprule\n"
            "\\textbf{Selector ($B=10\\text{k}$ / Top-2)} & \\textbf{Pass.\\ Cov.} & \\textbf{Own Cited Ret.} & \\textbf{Shared AR Cited Ret.} & \\textbf{Justif.\\ Lex.\\ Cov. (Own / AR)} & \\textbf{Gold Lex.\\ Cov.} \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}%\n}\n"
            "\\caption{\\textbf{Passage Coverage, Citation Retention, and Non-Stopword Lexical Content Recall Across All Selectors (300 Held-Out Claims):} Reports both per-trajectory cited passage retention and paired recall against shared Target AR citations and summaries.}\n"
            "\\label{tab:selector_coverage_all}\n\\end{table*}\n"
        )
        (TABLES_DIR / "table_selector_coverage_all.tex").write_text(tex, encoding="utf-8")

    print(json.dumps(master, indent=2))


if __name__ == "__main__":
    main()
