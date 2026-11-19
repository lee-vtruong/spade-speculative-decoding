import json
import re
import statistics
from spade.adaptive.context_selection import build_draft_context

_WORD_RE = re.compile(r"\b[a-zA-Z0-9_]{3,}\b")
_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "for", "are", "was", "were", "been", "have", "has", "had",
    "will", "would", "shall", "should", "can", "could", "about", "which", "there", "their", "they", "them",
    "what", "when", "where", "who", "whom", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "than", "too", "very", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once", "here", "why", "claim", "evidence",
    "supported", "refuted", "according", "statement", "shows", "stated", "reported", "article",
}


def get_tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)} - _STOPWORDS


def extract_evidence_text_from_prompt(prompt: str) -> str:
    m = re.search(r"Evidence:\s*(\[\{.*?\}\])\s*\n", prompt, flags=re.DOTALL)
    if m:
        try:
            ev_list = json.loads(m.group(1))
            return " ".join(e.get("text", "") for e in ev_list)
        except Exception:
            pass
    return ""


with open("outputs/benchmarks/retrieved-heldout300.jsonl", "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

policies = [
    ("Selective Top-2", lambda r, i: build_draft_context(r, policy="selective_topk", top_k=2).draft_prompt),
    ("Random Sent-B10k", lambda r, i: build_draft_context(r, policy="random_sentence_budget", budget_chars=10000, min_sentences_per_passage=1, seed=42+i).draft_prompt),
    ("BM25 Sent-B8k", lambda r, i: build_draft_context(r, policy="sentence_budget", budget_chars=8000, min_sentences_per_passage=1).draft_prompt),
    ("BM25 Sent-B10k", lambda r, i: build_draft_context(r, policy="sentence_budget", budget_chars=10000, min_sentences_per_passage=1).draft_prompt),
    ("BM25 NoCoverage", lambda r, i: build_draft_context(r, policy="sentence_budget", budget_chars=10000, min_sentences_per_passage=0).draft_prompt),
]

print("=== CLEAN EVIDENCE-ONLY GOLD JUSTIFICATION RECALL (N=300 CLAIMS) ===")
for name, fn in policies:
    recalls = []
    chars_list = []
    full_chars_list = []
    for i, r in enumerate(rows):
        gold = r.get("evaluation", {}).get("gold_justification", "")
        gold_toks = get_tokens(gold)
        full_text = " ".join(e.get("text", "") for e in r.get("evidence", []))
        full_toks = get_tokens(full_text)
        gold_in_full = gold_toks & full_toks

        draft_prompt = fn(r, i)
        draft_ev_text = extract_evidence_text_from_prompt(draft_prompt)
        draft_toks = get_tokens(draft_ev_text)

        fc = len(full_text)
        dc = len(draft_ev_text)
        chars_list.append(dc)
        full_chars_list.append(fc)

        if gold_in_full:
            # Strictly bounded in [0.0, 1.0]: what fraction of gold tokens in full evidence are preserved in draft?
            rec = len(gold_in_full & draft_toks) / len(gold_in_full)
            recalls.append(rec)

    mean_rec = statistics.mean(recalls) * 100 if recalls else 0.0
    mean_chars = statistics.mean(chars_list)
    print(f"{name:20s} | Evidence Gold Recall: {mean_rec:6.2f}% | Evidence Chars: {mean_chars:6.0f} | Evaluated: {len(recalls)}/300")
