"""Zero-leakage evidence-selective context builders for M4.2 Asymmetric Speculative Drafting.

Provides three draft context policies while keeping the target model strictly on full evidence:
  1. full          : Full claim + all evidence passages (exact baseline).
  2. selective_topk: Claim + top-k evidence passages ranked solely by claim-evidence lexical relevance.
  3. compact       : Claim + compact extractive representation of every evidence passage (preserving all E-IDs).

Strictly leak-free: uses ONLY claim text and model-visible evidence passages (never gold labels
or gold justifications). Deterministic, portable, and timed via perf_counter().
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from spade.benchmark.prompt_builder import build_prompt

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "if", "in", "into", "is", "it",
    "its", "of", "on", "or", "she", "that", "the", "their", "them", "there", "these", "they",
    "this", "to", "was", "we", "were", "what", "when", "where", "which", "who", "will", "with",
    "would", "you", "your",
})


@dataclass(frozen=True)
class DraftContextResult:
    """Result of constructing a draft-model prompt from a benchmark example."""
    policy: str
    draft_prompt: str
    draft_prompt_hash: str
    selected_evidence_ids: list[str]
    total_evidence_count: int
    selected_evidence_count: int
    full_evidence_chars: int
    draft_evidence_chars: int
    context_selection_s: float


def _content_terms(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1 and t.lower() not in _STOPWORDS]


def _score_passages_bm25(claim: str, texts: list[str], k1: float = 1.2, b: float = 0.75) -> list[float]:
    """Deterministic lightweight BM25 relevance score of each text against claim terms only."""
    if not texts:
        return []
    q_terms = _content_terms(claim)
    if not q_terms:
        return [0.0] * len(texts)
    q_set = set(q_terms)
    q_nums = {t for t in q_set if any(ch.isdigit() for ch in t)}

    docs = [_content_terms(t) for t in texts]
    n_docs = len(docs)
    avgdl = sum(len(d) for d in docs) / max(1, n_docs)
    df: Counter[str] = Counter()
    for d in docs:
        for term in set(d):
            if term in q_set:
                df[term] += 1

    scores: list[float] = []
    for d in docs:
        if not d:
            scores.append(0.0)
            continue
        tf = Counter(d)
        dl = len(d)
        denom_norm = k1 * (1.0 - b + b * (dl / max(1.0, avgdl)))
        score = 0.0
        for term in q_set:
            f = tf.get(term, 0)
            if f <= 0:
                continue
            idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            weight = 1.5 if term in q_nums else 1.0
            score += weight * idf * ((f * (k1 + 1.0)) / (f + denom_norm))
        scores.append(score)
    return scores


def _compact_passage_text(
    claim: str,
    text: str,
    *,
    max_sentences: int = 1,
    max_chars: int = 160,
) -> str:
    """Deterministically extract top claim-relevant sentence(s) from a single evidence passage."""
    clean = " ".join(text.split())
    if not clean:
        return ""
    if len(clean) <= max_chars:
        return clean

    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if len(sents) <= 1:
        snippet = clean[:max_chars].rsplit(" ", 1)[0] if " " in clean[:max_chars] else clean[:max_chars]
        return snippet.strip()

    scores = _score_passages_bm25(claim, sents)
    # Rank by (-score, original_index) for deterministic tie-breaking, then restore reading order
    top_indices = sorted(
        sorted(range(len(sents)), key=lambda i: (-scores[i], i))[: max(1, max_sentences)]
    )
    joined = " ".join(sents[i] for i in top_indices)
    if len(joined) > max_chars:
        joined = joined[:max_chars].rsplit(" ", 1)[0] if " " in joined[:max_chars] else joined[:max_chars]
    return joined.strip()


def build_draft_context(
    row: dict[str, Any],
    *,
    policy: str = "full",
    top_k: int = 2,
    compact_sentences: int = 1,
    compact_chars: int = 160,
) -> DraftContextResult:
    """Construct draft prompt under the requested policy without touching target prompt or gold labels.

    Args:
        row: Prepared benchmark row containing 'claim', 'evidence', 'prompt', and 'evidence_mode'.
        policy: One of {"full", "selective_topk", "compact"}.
        top_k: Number of evidence passages retained under 'selective_topk'.
        compact_sentences: Number of claim-salient sentences kept per passage under 'compact'.
        compact_chars: Character budget per compacted passage under 'compact'.
    """
    if policy not in {"full", "selective_topk", "compact"}:
        raise ValueError(f"Unsupported draft context policy: {policy!r}")

    claim = str(row.get("claim", ""))
    evidence = list(row.get("evidence") or [])
    ev_mode = str(row.get("evidence_mode") or "retrieved")
    if ev_mode not in ("retrieved", "oracle_qa"):
        ev_mode = "oracle_qa" if "oracle" in ev_mode else "retrieved"

    full_chars = sum(len(str(e.get("text", ""))) for e in evidence)
    all_ids = [str(e.get("evidence_id", f"E{i+1}")) for i, e in enumerate(evidence)]

    if policy == "full":
        return DraftContextResult(
            policy="full",
            draft_prompt=str(row["prompt"]),
            draft_prompt_hash=str(row.get("prompt_hash", "")),
            selected_evidence_ids=all_ids,
            total_evidence_count=len(evidence),
            selected_evidence_count=len(evidence),
            full_evidence_chars=full_chars,
            draft_evidence_chars=full_chars,
            context_selection_s=0.0,
        )

    t0 = time.perf_counter()
    if policy == "selective_topk":
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0 or not evidence:
            selected = []
        elif top_k >= len(evidence):
            selected = evidence
        else:
            scores = _score_passages_bm25(claim, [str(e.get("text", "")) for e in evidence])
            chosen_idx = sorted(
                sorted(range(len(evidence)), key=lambda i: (-scores[i], i))[:top_k]
            )
            selected = [evidence[i] for i in chosen_idx]
        draft_prompt, draft_hash = build_prompt(claim, selected, mode=ev_mode)
        sel_ids = [str(e.get("evidence_id", "")) for e in selected]
        draft_chars = sum(len(str(e.get("text", ""))) for e in selected)
    else:  # policy == "compact"
        compacted = [
            {
                "evidence_id": str(e.get("evidence_id", f"E{i+1}")),
                "text": _compact_passage_text(
                    claim,
                    str(e.get("text", "")),
                    max_sentences=compact_sentences,
                    max_chars=compact_chars,
                ),
            }
            for i, e in enumerate(evidence)
        ]
        draft_prompt, draft_hash = build_prompt(claim, compacted, mode=ev_mode)
        sel_ids = [c["evidence_id"] for c in compacted]
        draft_chars = sum(len(c["text"]) for c in compacted)

    selection_s = time.perf_counter() - t0
    return DraftContextResult(
        policy=policy,
        draft_prompt=draft_prompt,
        draft_prompt_hash=draft_hash,
        selected_evidence_ids=sel_ids,
        total_evidence_count=len(evidence),
        selected_evidence_count=len(sel_ids),
        full_evidence_chars=full_chars,
        draft_evidence_chars=draft_chars,
        context_selection_s=selection_s,
    )
