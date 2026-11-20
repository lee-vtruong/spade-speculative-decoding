"""Evaluate lexical content-word coverage and cited-passage retention across ALL 5 selectors on 300 held-out claims."""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from spade.adaptive.context_selection import (
    build_draft_context,
    build_draft_context_from_scores,
)
from spade.adaptive.target_attention_selector import (
    TargetPrefillAttentionHook,
    find_evidence_and_sentence_token_spans,
    resolve_layer_indices,
    score_evidence_from_token_importance,
)
from spade.decoding.autoregressive import format_qwen_prompt

TARGET_SNAP = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
BENCH_PATH = Path("outputs/benchmarks/retrieved-heldout300.jsonl")
RUNS_DIR = Path("outputs/runs")
OUT_DIR = Path("outputs/analysis/v6_revisions")

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "if", "in", "into", "is", "it",
    "its", "of", "on", "or", "she", "that", "the", "their", "them", "there", "these", "they",
    "this", "to", "was", "we", "were", "what", "when", "where", "which", "who", "will", "with",
    "would", "you", "your", "could", "should", "can", "may", "might", "must", "shall",
    "about", "above", "after", "again", "all", "any", "because", "before", "below", "between",
    "both", "but", "during", "each", "few", "further", "here", "how", "more", "most", "no",
    "nor", "not", "now", "once", "only", "other", "our", "out", "over", "same", "so", "some",
    "such", "than", "then", "through", "too", "under", "until", "up", "very", "what", "when",
    "where", "while", "whom", "why",
})


def _content_tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text) if len(t) > 2 and t.lower() not in _STOPWORDS}


def _extract_ev_text(prompt: str) -> str:
    m = re.search(r"Evidence:\s*(\[.*?\])\s*\n(?:JSON:|Response:)", prompt, flags=re.DOTALL)
    if not m:
        return ""
    try:
        ev = json.loads(m.group(1))
        return " ".join(str(x.get("text", "")) for x in ev)
    except Exception:
        return ""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bench = [json.loads(l) for l in BENCH_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    run_ids = {
        "b0s": "b1-4-heldout300-b0s",
        "SPADE": "b1-4-heldout300-attn-sent-b10k",
        "SPADE No-Coverage": "b1-4-heldout300-attn-sent-b10k-nocoverage",
        "BM25 Sent-B10k": "b1-4-heldout300-bm25-sent-b10k",
        "Random Sent-B10k": "b1-4-heldout300-random-sent-b10k",
        "BM25 Top-2": "b1-4-heldout300-selective-topk2",
    }
    preds_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, rid in run_ids.items():
        p = RUNS_DIR / rid / "predictions.jsonl"
        preds_by_arm[arm] = {str(r["claim_id"]): r for r in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())}

    # Reconstruct target attention contexts once for long claims (>10k chars)
    tok = AutoTokenizer.from_pretrained(TARGET_SNAP, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(TARGET_SNAP, dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()
    n_layers = len(model.model.layers)
    layer_indices = resolve_layer_indices(n_layers, "upper50")

    reconstructed: dict[str, dict[str, tuple[list[str], str]]] = {
        arm: {} for arm in ("SPADE", "SPADE No-Coverage", "BM25 Sent-B10k", "Random Sent-B10k", "BM25 Top-2")
    }
    attn_draft_samples: list[dict[str, Any]] = []

    for idx, row in enumerate(bench):
        cid = str(row["claim_id"])
        ev_list = list(row.get("evidence") or [])
        full_chars = sum(len(str(e.get("text", ""))) for e in ev_list)
        all_ids = [str(e.get("evidence_id", f"E{i+1}")) for i, e in enumerate(ev_list)]
        full_ev_txt = " ".join(str(e.get("text", "")) for e in ev_list)

        d_bm25 = build_draft_context(row, policy="sentence_budget", budget_chars=10000, min_sentences_per_passage=1)
        d_rand = build_draft_context(row, policy="random_sentence_budget", budget_chars=10000, min_sentences_per_passage=1, seed=42 + idx)
        d_topk = build_draft_context(row, policy="selective_topk", top_k=2)

        reconstructed["BM25 Sent-B10k"][cid] = (d_bm25.selected_evidence_ids, _extract_ev_text(d_bm25.draft_prompt))
        reconstructed["Random Sent-B10k"][cid] = (d_rand.selected_evidence_ids, _extract_ev_text(d_rand.draft_prompt))
        reconstructed["BM25 Top-2"][cid] = (d_topk.selected_evidence_ids, _extract_ev_text(d_topk.draft_prompt))

        if full_chars <= 10000:
            reconstructed["SPADE"][cid] = (all_ids, full_ev_txt)
            reconstructed["SPADE No-Coverage"][cid] = (all_ids, full_ev_txt)
        else:
            ser = format_qwen_prompt(tok, row["prompt"])
            enc = tok(ser, return_offsets_mapping=True, add_special_tokens=False)
            p_ids = list(enc["input_ids"])
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
            inp = torch.tensor([p_ids], dtype=torch.long, device="cuda")
            with TargetPrefillAttentionHook(model, layer_indices=layer_indices, tail_window=16) as hook:
                with torch.inference_mode():
                    out = model(input_ids=inp, use_cache=True, logits_to_keep=1)
                imp, _ = hook.compute_token_importance(out.past_key_values)
                del out, inp
            spans = find_evidence_and_sentence_token_spans(ser, ev_list, offsets)
            _, s_scores = score_evidence_from_token_importance(imp, spans)
            d_spade = build_draft_context_from_scores(row, policy="sentence_budget", sentence_scores_by_passage=s_scores, budget_chars=10000, min_sentences_per_passage=1)
            d_nocov = build_draft_context_from_scores(row, policy="sentence_budget", sentence_scores_by_passage=s_scores, budget_chars=10000, min_sentences_per_passage=0)
            reconstructed["SPADE"][cid] = (d_spade.selected_evidence_ids, _extract_ev_text(d_spade.draft_prompt))
            reconstructed["SPADE No-Coverage"][cid] = (d_nocov.selected_evidence_ids, _extract_ev_text(d_nocov.draft_prompt))
            attn_draft_samples.append({
                "claim_id": cid,
                "spade_draft_prompt": d_spade.draft_prompt,
                "spade_selected_ids": d_spade.selected_evidence_ids,
                "bm25_sent_draft_prompt": d_bm25.draft_prompt,
                "bm25_topk2_draft_prompt": d_topk.draft_prompt,
                "bm25_topk2_selected_ids": d_topk.selected_evidence_ids,
                "nocov_selected_ids": d_nocov.selected_evidence_ids,
            })
            if len(p_ids) >= 4096:
                torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    (OUT_DIR / "reconstructed_long173_drafts.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in attn_draft_samples),
        encoding="utf-8",
    )

    summary_report: dict[str, Any] = {}
    b0s_preds = preds_by_arm["b0s"]
    for arm in ("SPADE", "SPADE No-Coverage", "BM25 Sent-B10k", "Random Sent-B10k", "BM25 Top-2"):
        arm_preds = preds_by_arm[arm]
        own_cited_tot = own_cited_hit = 0
        ar_cited_tot = ar_cited_hit = 0
        own_lex_recalls: list[float] = []
        ar_lex_recalls: list[float] = []
        gold_lex_recalls: list[float] = []
        passage_covs: list[float] = []
        id_match_count = 0

        for row in bench:
            cid = str(row["claim_id"])
            ev_list = list(row.get("evidence") or [])
            full_ev_toks = _content_tokens(" ".join(str(e.get("text", "")) for e in ev_list))
            sel_ids, draft_ev_text = reconstructed[arm][cid]
            sel_id_set = set(sel_ids)
            draft_toks = _content_tokens(draft_ev_text)
            if ev_list:
                passage_covs.append(len(sel_id_set) / len(ev_list))

            ap = arm_preds[cid]
            logged_ids = set(ap.get("draft_selected_evidence_ids") or [])
            if logged_ids == sel_id_set:
                id_match_count += 1

            own_parsed = ap.get("parsed") or {}
            own_cites = [str(c) for c in (own_parsed.get("citations") or [])]
            for c in own_cites:
                own_cited_tot += 1
                if c in sel_id_set:
                    own_cited_hit += 1

            ar_parsed = (b0s_preds[cid].get("parsed") or {})
            ar_cites = [str(c) for c in (ar_parsed.get("citations") or [])]
            for c in ar_cites:
                ar_cited_tot += 1
                if c in sel_id_set:
                    ar_cited_hit += 1

            # Lexical content recall of reasoning justifications (bounded to tokens present in full evidence)
            own_sum_toks = _content_tokens(str(own_parsed.get("summary") or "")) & full_ev_toks
            if own_sum_toks:
                own_lex_recalls.append(len(own_sum_toks & draft_toks) / len(own_sum_toks))

            ar_sum_toks = _content_tokens(str(ar_parsed.get("summary") or "")) & full_ev_toks
            if ar_sum_toks:
                ar_lex_recalls.append(len(ar_sum_toks & draft_toks) / len(ar_sum_toks))

            gold_toks = _content_tokens(str((row.get("evaluation") or {}).get("gold_justification") or "")) & full_ev_toks
            if gold_toks:
                gold_lex_recalls.append(len(gold_toks & draft_toks) / len(gold_toks))

        summary_report[arm] = {
            "id_reconstruction_match_300": id_match_count,
            "mean_passage_coverage_pct": round(statistics.mean(passage_covs) * 100.0, 2),
            "own_cited_retained": own_cited_hit,
            "own_cited_total": own_cited_tot,
            "own_cited_retention_pct": round((own_cited_hit / own_cited_tot) * 100.0, 2) if own_cited_tot else 100.0,
            "shared_ar_cited_retained": ar_cited_hit,
            "shared_ar_cited_total": ar_cited_tot,
            "shared_ar_cited_retention_pct": round((ar_cited_hit / ar_cited_tot) * 100.0, 2) if ar_cited_tot else 100.0,
            "own_justification_lexical_cov_pct": round(statistics.mean(own_lex_recalls) * 100.0, 2) if own_lex_recalls else 0.0,
            "shared_ar_justification_lexical_cov_pct": round(statistics.mean(ar_lex_recalls) * 100.0, 2) if ar_lex_recalls else 0.0,
            "gold_justification_lexical_cov_pct": round(statistics.mean(gold_lex_recalls) * 100.0, 2) if gold_lex_recalls else 0.0,
        }

    (OUT_DIR / "selector_coverage_all.json").write_text(json.dumps(summary_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary_report, indent=2))


if __name__ == "__main__":
    main()
