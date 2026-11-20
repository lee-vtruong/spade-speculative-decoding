"""Extract 2 crisp qualitative examples comparing SPADE vs BM25 Top-2 / BM25 Sent on long multi-doc claims."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BENCH_PATH = Path("outputs/benchmarks/retrieved-heldout300.jsonl")
DRAFTS_PATH = Path("outputs/analysis/v6_revisions/reconstructed_long173_drafts.jsonl")
RUNS_DIR = Path("outputs/runs")
OUT_PATH = Path("outputs/analysis/v6_revisions/qualitative_examples.json")


def _load_claim_stats(run_id: str) -> dict[str, dict[str, Any]]:
    t_rows = {str(r["claim_id"]): r for r in (json.loads(l) for l in (RUNS_DIR / run_id / "trial_metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    p_rows = {str(r["claim_id"]): r for r in (json.loads(l) for l in (RUNS_DIR / run_id / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    tok_rows = [json.loads(l) for l in (RUNS_DIR / run_id / "token_traces.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    acc_cnt: dict[str, int] = {}
    ev_cnt: dict[str, int] = {}
    for tk in tok_rows:
        st = tk.get("status")
        if st not in {"accepted", "rejected"}:
            continue
        cid = str(tk["claim_id"])
        ev_cnt[cid] = ev_cnt.get(cid, 0) + 1
        if st == "accepted":
            acc_cnt[cid] = acc_cnt.get(cid, 0) + 1

    out: dict[str, dict[str, Any]] = {}
    for cid, tm in t_rows.items():
        e2e_s = tm["prompt_prefill_s"] + tm.get("draft_prefill_s", 0.0) + tm["decode_s"] + tm.get("context_selection_s", 0.0)
        n_ev = ev_cnt.get(cid, 0)
        n_ac = acc_cnt.get(cid, 0)
        pr = p_rows.get(cid, {})
        parsed = pr.get("parsed") or {}
        out[cid] = {
            "e2e_tok_s": round(tm["output_tokens"] / e2e_s, 2) if e2e_s > 0 else 0.0,
            "alpha_eval_pct": round((n_ac / n_ev) * 100.0, 2) if n_ev > 0 else 0.0,
            "draft_toks": int(tm.get("draft_prompt_tokens", tm.get("prompt_tokens", 0))),
            "citations": parsed.get("citations") or [],
            "verdict": parsed.get("verdict"),
            "summary": parsed.get("summary"),
        }
    return out


def main() -> None:
    bench = {str(r["claim_id"]): r for r in (json.loads(l) for l in BENCH_PATH.read_text(encoding="utf-8").splitlines() if l.strip())}
    drafts = {str(r["claim_id"]): r for r in (json.loads(l) for l in DRAFTS_PATH.read_text(encoding="utf-8").splitlines() if l.strip())}

    full_st = _load_claim_stats("b1-4-heldout300-baseline-full")
    spade_st = _load_claim_stats("b1-4-heldout300-attn-sent-b10k")
    bm25_st = _load_claim_stats("b1-4-heldout300-bm25-sent-b10k")
    topk2_st = _load_claim_stats("b1-4-heldout300-selective-topk2")

    candidates = []
    for cid, d in drafts.items():
        sp = spade_st[cid]
        b2 = bm25_st[cid]
        tk = topk2_st[cid]
        fl = full_st[cid]
        # Prefer multi-passage claims where target cited > 1 passage and Top-2 dropped a cited passage while SPADE beats both Top-2 & BM25 in alpha_eval
        dropped_cites = set(str(c) for c in tk["citations"]) - set(str(x) for x in d["bm25_topk2_selected_ids"])
        if len(bench[cid].get("evidence", [])) >= 4 and len(dropped_cites) >= 1 and sp["alpha_eval_pct"] > max(b2["alpha_eval_pct"], tk["alpha_eval_pct"]):
            margin = (sp["alpha_eval_pct"] - b2["alpha_eval_pct"]) + (sp["alpha_eval_pct"] - tk["alpha_eval_pct"])
            candidates.append((margin, cid, sorted(dropped_cites)))

    candidates.sort(reverse=True)
    # Fallback if strict filter yields < 2
    if len(candidates) < 2:
        for cid, d in drafts.items():
            sp = spade_st[cid]
            b2 = bm25_st[cid]
            tk = topk2_st[cid]
            if any(x[1] == cid for x in candidates):
                continue
            if len(bench[cid].get("evidence", [])) >= 4 and sp["alpha_eval_pct"] >= b2["alpha_eval_pct"] and sp["alpha_eval_pct"] > tk["alpha_eval_pct"]:
                margin = sp["alpha_eval_pct"] - tk["alpha_eval_pct"]
                candidates.append((margin, cid, []))

    picked = []
    for _, cid, dropped in candidates[:4]:
        row = bench[cid]
        d = drafts[cid]
        picked.append({
            "claim_id": cid,
            "claim": row["claim"],
            "n_passages": len(row.get("evidence", [])),
            "full_evidence_chars": sum(len(str(e.get("text", ""))) for e in row.get("evidence", [])),
            "dropped_citations_by_topk2": dropped,
            "full_sd": full_st[cid],
            "bm25_top2": topk2_st[cid],
            "bm25_sent_b10k": bm25_st[cid],
            "spade_b10k": spade_st[cid],
            "spade_draft_preview": d["spade_draft_prompt"][:1500],
            "bm25_sent_preview": d["bm25_sent_draft_prompt"][:1500],
        })

    OUT_PATH.write_text(json.dumps(picked, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Extracted {len(picked)} qualitative candidates to {OUT_PATH}")
    for ex in picked[:2]:
        print(f"- {ex['claim_id']} | m={ex['n_passages']} | SPADE alpha={ex['spade_b10k']['alpha_eval_pct']}% | BM25-Sent={ex['bm25_sent_b10k']['alpha_eval_pct']}% | Top2={ex['bm25_top2']['alpha_eval_pct']}% | Top2 dropped cites={ex['dropped_citations_by_topk2']}")


if __name__ == "__main__":
    main()
