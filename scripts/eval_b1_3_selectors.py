"""Step B1.3b: Controlled Fixed-Prefix Evaluation of Evidence Selectors on retrieved100.

Evaluates zero-leakage draft context selectors on the exact same 100 long-context claims
and 9,821 fixed prefixes from `m4-2-retrieved100-baseline-full/predictions.jsonl`:
  - full (baseline)
  - bm25_topk2 (M4.2 baseline selector)
  - bm25_hybrid_topk2 (Candidate 1: BM25 Top-2 Full + 120c compact tail)
  - bm25_sent_b8k (Candidate 2: BM25+entity sentence-level budget ~8k chars)
  - attn_u18_topk2_sum / attn_u18_topk2_peak / attn_last6_topk2_sum (Candidate 3a/3b: Target-Prefill Attention Top-2)
  - attn_u18_hybrid_topk2 (Candidate 3c: Target-Prefill Attention Top-2 Full + 120c compact tail)
  - attn_u18_sent_b8k / attn_u18_sent_b10k (Candidate 3d: Target-Prefill Attention Sentence-Level Budget)

Reports cited-evidence recall, draft compression %, selection latency (ms/claim),
mean TV/JS/expected acceptance (1-TV), and 10,000-resample claim-clustered paired
bootstrap comparisons vs BOTH `full` and `bm25_topk2`.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from spade.adaptive.context_selection import (
    DraftContextResult,
    build_draft_context,
    build_draft_context_from_scores,
)
from spade.adaptive.target_attention_selector import (
    TargetPrefillAttentionHook,
    find_evidence_and_sentence_token_spans,
    score_evidence_from_token_importance,
)
from spade.analysis.distribution_audit import claim_clustered_bootstrap_mean, jensen_shannon_divergence
from spade.decoding.autoregressive import format_qwen_prompt, load_causal_lm, validate_tokenizers
from spade.decoding.runtime_opt import _supports_logits_to_keep, processed_distribution_fast
from spade.decoding.sampling import expected_acceptance, total_variation


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]


def _model_device(model: Any) -> torch.device:
    if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
        return model.get_input_embeddings().weight.device
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def _tail_logits(model: Any, input_ids: torch.Tensor, g_len: int) -> torch.Tensor:
    kw: dict[str, Any] = {"input_ids": input_ids, "use_cache": False}
    if _supports_logits_to_keep(model):
        kw["logits_to_keep"] = g_len
    out = model(**kw)
    lg = out.logits[0, -g_len:].detach().float()
    del out
    return lg


def render_b1_3_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Step B1.3b — Fixed-Prefix Evidence Selector Comparison (`m4-2-retrieved100`, 100 claims)",
        "",
        f"- Fixed prefixes evaluated: `{report['n_prefixes']}` across `{report['n_claims']}` claims | Cluster unit: `claim_id` (`B={report['n_bootstrap']}`)",
        "- All selectors are strictly zero-leakage (computed solely from input prompt & target prefill attention before token 0).",
        "",
        "## 1. Selector Summary (Recall, Compression, Selection Cost, and Fixed-Prefix Alignment)",
        "",
        "| Selector | Draft Tok | Comp % | Sel (ms) | Cited Recall (`ev>2`) | Mean $\\mathrm{TV}(p,q)$ | Mean $\\mathrm{JS}(p,q)$ | Exp Acc $1-\\mathrm{TV}$ (%) | $\\Delta(1-\\mathrm{TV})$ vs `full` (95% CI, $p$) | $\\Delta(1-\\mathrm{TV})$ vs `bm25_topk2` (95% CI, $p$) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s in report["selectors"]:
        vs_f = s.get("delta_vs_full")
        vs_b = s.get("delta_vs_bm25_topk2")
        f_str = (
            f"**{vs_f['mean']:+.3f} pp** (`[{vs_f['ci_95_low']:+.3f}, {vs_f['ci_95_high']:+.3f}]`, $p={vs_f['p_value_two_sided']:.4f}$)"
            if vs_f is not None
            else "—"
        )
        b_str = (
            f"**{vs_b['mean']:+.3f} pp** (`[{vs_b['ci_95_low']:+.3f}, {vs_b['ci_95_high']:+.3f}]`, $p={vs_b['p_value_two_sided']:.4f}$)"
            if vs_b is not None
            else "—"
        )
        lines.append(
            f"| `{s['name']}` | {s['mean_draft_prompt_tokens']:.1f} | {s['prompt_compression_pct']:.1f}% | "
            f"{s['mean_selection_ms']:.2f} | {s['cited_recall_ev_gt2_str']} | "
            f"{s['mean_tv']:.5f} | {s['mean_js']:.5f} | **{s['expected_acceptance_pct']:.3f}%** | {f_str} | {b_str} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("outputs/benchmarks/train-retrieved-bounded.jsonl"))
    p.add_argument("--source-predictions", type=Path, default=Path("outputs/runs/m4-2-retrieved100-baseline-full/predictions.jsonl"))
    p.add_argument("--target", default=os.environ.get("TARGET_MODEL", "Qwen/Qwen3-8B"))
    p.add_argument("--draft", default=os.environ.get("DRAFT_MODEL", "Qwen/Qwen3-0.6B"))
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/b1_3_selector_eval"))
    args = p.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    draft_tok = AutoTokenizer.from_pretrained(args.draft, local_files_only=True)
    validate_tokenizers(tokenizer, draft_tok)
    target_model = load_causal_lm(args.target, local_files_only=True, dtype=args.dtype)
    draft_model = load_causal_lm(args.draft, local_files_only=True, dtype=args.dtype)
    t_dev = _model_device(target_model)
    d_dev = _model_device(draft_model)

    rows = read_jsonl(args.input)[: args.limit]
    preds = read_jsonl(args.source_predictions)
    pred_by_cid = {str(r["claim_id"]): r for r in preds}

    n_layers = len(target_model.model.layers)
    u18_layers = list(range(n_layers // 2, n_layers))          # 18..35 for 36-layer Qwen3-8B
    last6_layers = list(range(max(0, n_layers - 6), n_layers))  # 30..35

    selector_names = [
        "full",
        "bm25_topk2",
        "bm25_hybrid_topk2",
        "bm25_sent_b8k",
        "attn_u18_topk2_sum",
        "attn_u18_topk2_peak",
        "attn_last6_topk2_sum",
        "attn_u18_hybrid_topk2",
        "attn_u18_sent_b8k",
        "attn_u18_sent_b10k",
    ]

    # Per-selector accumulators
    per_claim_exp: dict[str, dict[str, list[float]]] = {name: defaultdict(lambda: [0.0, 0]) for name in selector_names}
    per_sel_tv_sum: dict[str, float] = defaultdict(float)
    per_sel_js_sum: dict[str, float] = defaultdict(float)
    per_sel_exp_sum: dict[str, float] = defaultdict(float)
    per_sel_prompt_toks: dict[str, list[int]] = defaultdict(list)
    per_sel_sel_ms: dict[str, list[float]] = defaultdict(list)
    per_sel_cited_hits: dict[str, int] = defaultdict(int)
    n_ev_gt2_with_cites = 0
    total_prefixes = 0
    all_cids: list[str] = []

    t_start = time.perf_counter()
    for idx, row in enumerate(rows):
        cid = str(row["claim_id"])
        pred = pred_by_cid.get(cid)
        if not pred or not pred.get("token_ids"):
            continue
        token_ids = [int(t) for t in pred["token_ids"]]
        g_len = len(token_ids)
        all_cids.append(cid)
        total_prefixes += g_len

        evidence = list(row.get("evidence") or [])
        parsed = pred.get("parsed") or {}
        cited_ids = {str(c).strip() for c in (parsed.get("citations") or []) if isinstance(c, str) and c.strip()}
        track_recall = len(evidence) > 2 and bool(cited_ids)
        if track_recall:
            n_ev_gt2_with_cites += 1

        # 1) Tokenize target prompt with character offsets for exact passage/sentence alignment
        ser_full = format_qwen_prompt(tokenizer, row["prompt"])
        enc_full = tokenizer(ser_full, return_offsets_mapping=True, add_special_tokens=False)
        t_prompt_ids = list(enc_full["input_ids"])
        offsets = [(int(a), int(b)) for a, b in enc_full["offset_mapping"]]
        p_t = len(t_prompt_ids)
        spans = find_evidence_and_sentence_token_spans(ser_full, evidence, offsets)

        # 2) Single Target Prefill with lightweight attention hook (zero extra forward pass!) + continuation
        with TargetPrefillAttentionHook(target_model, layer_indices=u18_layers, tail_window=16) as hook:
            with torch.inference_mode():
                t_pref_inp = torch.tensor([t_prompt_ids], dtype=torch.long, device=t_dev)
                kw_pref: dict[str, Any] = {"input_ids": t_pref_inp, "use_cache": True}
                if _supports_logits_to_keep(target_model):
                    kw_pref["logits_to_keep"] = 1
                pref_out = target_model(**kw_pref)
                first_t_logit = pref_out.logits[0, -1:].detach().float()
                target_pkv = pref_out.past_key_values
                del pref_out, t_pref_inp

            imp_u18, imp_u18_s = hook.compute_token_importance(target_pkv, layer_subset=u18_layers)
            imp_last6, imp_last6_s = hook.compute_token_importance(target_pkv, layer_subset=last6_layers)

        if g_len > 1:
            with torch.inference_mode():
                cont_inp = torch.tensor([token_ids[:-1]], dtype=torch.long, device=t_dev)
                cont_out = target_model(input_ids=cont_inp, past_key_values=target_pkv, use_cache=False)
                t_logits_all = torch.cat([first_t_logit, cont_out.logits[0].detach().float()], dim=0)
                del cont_out, cont_inp
        else:
            t_logits_all = first_t_logit
        del target_pkv

        # Score evidence using target prefill attention (zero-leakage!)
        t_sc0 = time.perf_counter()
        p_u18_sum, s_u18 = score_evidence_from_token_importance(imp_u18, spans, passage_agg="sum")
        sc_u18_s = time.perf_counter() - t_sc0

        t_sc1 = time.perf_counter()
        p_u18_peak, _ = score_evidence_from_token_importance(imp_u18, spans, passage_agg="peak", peak_k=32)
        sc_peak_s = time.perf_counter() - t_sc1

        t_sc2 = time.perf_counter()
        p_last6_sum, _ = score_evidence_from_token_importance(imp_last6, spans, passage_agg="sum")
        sc_last6_s = time.perf_counter() - t_sc2

        # 3) Build all 10 candidate draft contexts
        ctxs: dict[str, DraftContextResult] = {
            "full": build_draft_context(row, policy="full"),
            "bm25_topk2": build_draft_context(row, policy="selective_topk", top_k=2),
            "bm25_hybrid_topk2": build_draft_context(row, policy="hybrid_topk_compact", top_k=2, compact_sentences=1, compact_chars=120),
            "bm25_sent_b8k": build_draft_context(row, policy="sentence_budget", budget_chars=8000),
            "attn_u18_topk2_sum": build_draft_context_from_scores(row, policy="selective_topk", passage_scores=p_u18_sum, top_k=2, extra_selection_s=imp_u18_s + sc_u18_s),
            "attn_u18_topk2_peak": build_draft_context_from_scores(row, policy="selective_topk", passage_scores=p_u18_peak, top_k=2, extra_selection_s=imp_u18_s + sc_peak_s),
            "attn_last6_topk2_sum": build_draft_context_from_scores(row, policy="selective_topk", passage_scores=p_last6_sum, top_k=2, extra_selection_s=imp_last6_s + sc_last6_s),
            "attn_u18_hybrid_topk2": build_draft_context_from_scores(row, policy="hybrid_topk_compact", passage_scores=p_u18_sum, top_k=2, compact_sentences=1, compact_chars=120, extra_selection_s=imp_u18_s + sc_u18_s),
            "attn_u18_sent_b8k": build_draft_context_from_scores(row, policy="sentence_budget", sentence_scores_by_passage=s_u18, budget_chars=8000, extra_selection_s=imp_u18_s + sc_u18_s),
            "attn_u18_sent_b10k": build_draft_context_from_scores(row, policy="sentence_budget", sentence_scores_by_passage=s_u18, budget_chars=10000, extra_selection_s=imp_u18_s + sc_u18_s),
        }

        # Precompute target distributions once in fast batched fp32
        p_dists = processed_distribution_fast(t_logits_all, temperature=args.temperature, top_k=0, top_p=args.top_p, dtype=torch.float32)

        # 4) Run draft teacher-forced forward pass per unique draft prompt (deduplicated!)
        logits_cache: dict[str, tuple[torch.Tensor, int]] = {}
        for name in selector_names:
            dctx = ctxs[name]
            per_sel_sel_ms[name].append(dctx.context_selection_s * 1000.0)
            if track_recall:
                # For hybrid/sentence_budget that keep all E-IDs, check top-2 full recall or retained full-ID recall
                if name in {"bm25_topk2", "attn_u18_topk2_sum", "attn_u18_topk2_peak", "attn_last6_topk2_sum"}:
                    if cited_ids.issubset(set(dctx.selected_evidence_ids)):
                        per_sel_cited_hits[name] += 1
                else:
                    # Hybrid/sentence policies preserve all E-IDs; count whether top-2 full passages covered cited_ids
                    if name == "full":
                        per_sel_cited_hits[name] += 1
                    elif "hybrid" in name:
                        scores_ref = p_u18_sum if name.startswith("attn_") else None
                        if scores_ref is not None:
                            top2_i = sorted(range(len(evidence)), key=lambda i: (-scores_ref[i], i))[:2]
                            top2_eids = {str(evidence[i].get("evidence_id", f"E{i+1}")) for i in top2_i}
                            if cited_ids.issubset(top2_eids):
                                per_sel_cited_hits[name] += 1
                        else:
                            if cited_ids.issubset(set(ctxs["bm25_topk2"].selected_evidence_ids)):
                                per_sel_cited_hits[name] += 1
                    else:
                        # Sentence-budget keeps all E-IDs represented
                        per_sel_cited_hits[name] += 1

            p_key = dctx.draft_prompt
            if p_key not in logits_cache:
                ser_d = format_qwen_prompt(tokenizer, p_key)
                d_prompt_ids = tokenizer(ser_d, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
                d_full_ids = d_prompt_ids + token_ids[:-1]
                with torch.inference_mode():
                    d_inp = torch.tensor([d_full_ids], dtype=torch.long, device=d_dev)
                    d_lgs = _tail_logits(draft_model, d_inp, g_len)
                    del d_inp
                if d_lgs.device != t_logits_all.device:
                    d_lgs = d_lgs.to(t_logits_all.device)
                q_dists = processed_distribution_fast(d_lgs, temperature=args.temperature, top_k=0, top_p=args.top_p, dtype=torch.float32)
                logits_cache[p_key] = (q_dists, len(d_prompt_ids))

            q_dists, d_ptok = logits_cache[p_key]
            per_sel_prompt_toks[name].append(d_ptok)

            # Vectorized TV and JS over all g_len prefixes of this claim!
            p64 = p_dists.to(torch.float64)
            q64 = q_dists.to(torch.float64)
            tv_vec = 0.5 * torch.abs(p64 - q64).sum(dim=-1)
            exp_vec = 1.0 - tv_vec
            m64 = 0.5 * (p64 + q64)
            kl_pm = torch.where(p64 > 0, p64 * torch.log(torch.clamp(p64 / torch.clamp(m64, min=1e-30), min=1e-30)), torch.zeros_like(p64)).sum(dim=-1)
            kl_qm = torch.where(q64 > 0, q64 * torch.log(torch.clamp(q64 / torch.clamp(m64, min=1e-30), min=1e-30)), torch.zeros_like(q64)).sum(dim=-1)
            js_vec = 0.5 * (kl_pm + kl_qm)

            tv_sum = float(tv_vec.sum().item())
            exp_sum = float(exp_vec.sum().item())
            js_sum = float(js_vec.sum().item())
            per_sel_tv_sum[name] += tv_sum
            per_sel_exp_sum[name] += exp_sum
            per_sel_js_sum[name] += js_sum
            per_claim_exp[name][cid][0] += exp_sum * 100.0
            per_claim_exp[name][cid][1] += g_len

        del t_logits_all, p_dists, logits_cache
        if p_t >= 4096 and t_dev.type == "cuda":
            torch.cuda.empty_cache()

        if (idx + 1) % 10 == 0 or (idx + 1) == len(rows):
            print(f"  [{idx+1}/{len(rows)}] evaluated {cid} (cum_prefixes={total_prefixes}, {time.perf_counter()-t_start:.1f}s)", flush=True)

    # 5) Aggregate & run 10,000-resample claim-clustered paired bootstrap vs `full` AND vs `bm25_topk2`
    mean_full_toks = sum(per_sel_prompt_toks["full"]) / len(all_cids)
    selector_summaries: list[dict[str, Any]] = []
    for name in selector_names:
        mean_tok = sum(per_sel_prompt_toks[name]) / len(all_cids)
        comp_pct = (1.0 - mean_tok / mean_full_toks) * 100.0 if mean_full_toks > 0 else 0.0
        mean_tv = per_sel_tv_sum[name] / total_prefixes
        mean_js = per_sel_js_sum[name] / total_prefixes
        exp_pct = (per_sel_exp_sum[name] / total_prefixes) * 100.0
        mean_sel_ms = sum(per_sel_sel_ms[name]) / len(all_cids)

        delta_vs_full = None
        if name != "full":
            diff_map = {
                cid: (per_claim_exp[name][cid][0] - per_claim_exp["full"][cid][0], int(per_claim_exp[name][cid][1]))
                for cid in all_cids
            }
            delta_vs_full = claim_clustered_bootstrap_mean(
                diff_map, all_claim_ids=all_cids, n_bootstrap=args.n_bootstrap, seed=args.seed, higher_is_better=True
            )

        delta_vs_bm25 = None
        if name not in {"full", "bm25_topk2"}:
            diff_map_b = {
                cid: (per_claim_exp[name][cid][0] - per_claim_exp["bm25_topk2"][cid][0], int(per_claim_exp[name][cid][1]))
                for cid in all_cids
            }
            delta_vs_bm25 = claim_clustered_bootstrap_mean(
                diff_map_b, all_claim_ids=all_cids, n_bootstrap=args.n_bootstrap, seed=args.seed, higher_is_better=True
            )

        hits = per_sel_cited_hits[name]
        rec_pct = (hits * 100.0 / n_ev_gt2_with_cites) if n_ev_gt2_with_cites else 0.0
        selector_summaries.append({
            "name": name,
            "mean_draft_prompt_tokens": round(mean_tok, 1),
            "prompt_compression_pct": round(comp_pct, 2),
            "mean_selection_ms": round(mean_sel_ms, 3),
            "cited_recall_ev_gt2_hits": hits,
            "cited_recall_ev_gt2_total": n_ev_gt2_with_cites,
            "cited_recall_ev_gt2_pct": round(rec_pct, 1),
            "cited_recall_ev_gt2_str": f"{hits}/{n_ev_gt2_with_cites} ({rec_pct:.1f}%)",
            "mean_tv": round(mean_tv, 6),
            "mean_js": round(mean_js, 6),
            "expected_acceptance_pct": round(exp_pct, 4),
            "delta_vs_full": delta_vs_full,
            "delta_vs_bm25_topk2": delta_vs_bm25,
        })

    report = {
        "schema_version": "b1_3_selector_eval_v1",
        "n_claims": len(all_cids),
        "n_prefixes": total_prefixes,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "selectors": selector_summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "b1_3_selector_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = render_b1_3_markdown(report)
    (args.output_dir / "b1_3_selector_report.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    print(f"Saved B1.3b selector evaluation to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
