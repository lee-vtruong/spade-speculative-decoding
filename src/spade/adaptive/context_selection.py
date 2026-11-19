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
_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,}|\d[\d,./%-]*)\b")
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


def split_evidence_sentences_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Deterministically split passage text into `(sentence, char_start, char_end)` spans."""
    if not text or not text.strip():
        return []
    spans: list[tuple[str, int, int]] = []
    pos = 0
    for m in _SENT_SPLIT_RE.finditer(text):
        seg = text[pos : m.start()]
        stripped = seg.strip()
        if stripped:
            lead = seg.find(stripped)
            spans.append((stripped, pos + lead, pos + lead + len(stripped)))
        pos = m.end()
    tail = text[pos:]
    stripped = tail.strip()
    if stripped:
        lead = tail.find(stripped)
        spans.append((stripped, pos + lead, pos + lead + len(stripped)))
    return spans


def _build_hybrid_topk_compact(
    claim: str,
    evidence: list[dict[str, Any]],
    passage_scores: list[float],
    *,
    top_k: int = 2,
    tail_sentences: int = 1,
    tail_chars: int = 120,
) -> list[dict[str, str]]:
    """Keep top-k passages in full and represent remaining passages with 1 compact salient sentence."""
    if not evidence:
        return []
    if top_k >= len(evidence):
        return [{"evidence_id": str(e.get("evidence_id", f"E{i+1}")), "text": str(e.get("text", ""))} for i, e in enumerate(evidence)]
    top_set = set(sorted(range(len(evidence)), key=lambda i: (-passage_scores[i], i))[: max(0, top_k)])
    out: list[dict[str, str]] = []
    for i, e in enumerate(evidence):
        eid = str(e.get("evidence_id", f"E{i+1}"))
        txt = str(e.get("text", ""))
        if i in top_set:
            out.append({"evidence_id": eid, "text": txt})
        else:
            out.append({
                "evidence_id": eid,
                "text": _compact_passage_text(claim, txt, max_sentences=tail_sentences, max_chars=tail_chars),
            })
    return out


def _build_sentence_budget_evidence(
    claim: str,
    evidence: list[dict[str, Any]],
    *,
    sentence_scores_by_passage: list[list[float]] | None = None,
    budget_chars: int = 8000,
    min_sentences_per_passage: int = 1,
    max_chars_per_sentence: int = 400,
) -> list[dict[str, str]]:
    """Select highest-scoring sentences across all E1..Em under a character budget, preserving order & E-IDs."""
    if not evidence:
        return []
    full_chars = sum(len(str(e.get("text", ""))) for e in evidence)
    if full_chars <= budget_chars:
        return [{"evidence_id": str(e.get("evidence_id", f"E{i+1}")), "text": str(e.get("text", ""))} for i, e in enumerate(evidence)]

    # Split every passage into sentences
    pass_sents: list[list[str]] = []
    for e in evidence:
        spans = split_evidence_sentences_with_spans(str(e.get("text", "")))
        sents = [s[:max_chars_per_sentence].strip() for s, _, _ in spans] if spans else []
        pass_sents.append(sents)

    if sentence_scores_by_passage is None:
        # Zero-leakage BM25 + entity/number overlap + light passage-level BM25 prior
        claim_entities = {m.lower() for m in _ENTITY_RE.findall(claim) if len(m) > 1}
        p_scores = _score_passages_bm25(claim, [str(e.get("text", "")) for e in evidence])
        all_flat = [s for sents in pass_sents for s in sents]
        flat_bm25 = _score_passages_bm25(claim, all_flat) if all_flat else []
        sentence_scores_by_passage = []
        cursor = 0
        for p_idx, sents in enumerate(pass_sents):
            sc_list: list[float] = []
            for s in sents:
                base = flat_bm25[cursor]
                cursor += 1
                s_low = s.lower()
                ent_hits = sum(1 for ent in claim_entities if ent in s_low)
                sc_list.append(base + 0.75 * ent_hits + 0.25 * p_scores[p_idx])
            sentence_scores_by_passage.append(sc_list)

    chosen: dict[int, set[int]] = {i: set() for i in range(len(evidence))}
    used_chars = 0

    # Guarantee at least `min_sentences_per_passage` top sentence(s) per passage so every E-ID stays visible
    if min_sentences_per_passage > 0:
        for p_idx, (sents, scs) in enumerate(zip(pass_sents, sentence_scores_by_passage)):
            if not sents:
                continue
            ranked = sorted(range(len(sents)), key=lambda si: (-scs[si], si))
            for si in ranked[:min_sentences_per_passage]:
                chosen[p_idx].add(si)
                used_chars += len(sents[si]) + 1

    # Greedily add highest-scoring remaining sentences across all passages until budget_chars
    candidates: list[tuple[float, int, int]] = []
    for p_idx, (sents, scs) in enumerate(zip(pass_sents, sentence_scores_by_passage)):
        for si, sc in enumerate(scs):
            if si not in chosen[p_idx]:
                candidates.append((sc, p_idx, si))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

    for _, p_idx, si in candidates:
        slen = len(pass_sents[p_idx][si]) + 1
        if used_chars + slen > budget_chars:
            continue
        chosen[p_idx].add(si)
        used_chars += slen

    out: list[dict[str, str]] = []
    for p_idx, e in enumerate(evidence):
        eid = str(e.get("evidence_id", f"E{p_idx+1}"))
        sents = pass_sents[p_idx]
        if sents and chosen[p_idx]:
            sel_text = " ".join(sents[si] for si in sorted(chosen[p_idx]))
            out.append({"evidence_id": eid, "text": sel_text})
        elif min_sentences_per_passage > 0:
            out.append({"evidence_id": eid, "text": ""})
    return out


def build_draft_context_from_scores(
    row: dict[str, Any],
    *,
    policy: str,
    passage_scores: list[float] | None = None,
    sentence_scores_by_passage: list[list[float]] | None = None,
    top_k: int = 2,
    compact_sentences: int = 1,
    compact_chars: int = 120,
    budget_chars: int = 8000,
    min_sentences_per_passage: int = 1,
    extra_selection_s: float = 0.0,
    seed: int = 42,
) -> DraftContextResult:
    """Build draft context from externally supplied zero-leakage scores (e.g. target prefill attention)."""
    claim = str(row.get("claim", ""))
    evidence = list(row.get("evidence") or [])
    ev_mode = str(row.get("evidence_mode") or "retrieved")
    if ev_mode not in ("retrieved", "oracle_qa"):
        ev_mode = "oracle_qa" if "oracle" in ev_mode else "retrieved"
    full_chars = sum(len(str(e.get("text", ""))) for e in evidence)

    t0 = time.perf_counter()
    if policy == "selective_topk":
        scores = passage_scores if passage_scores is not None else _score_passages_bm25(claim, [str(e.get("text", "")) for e in evidence])
        if top_k == 0 or not evidence:
            selected = []
        elif top_k >= len(evidence):
            selected = evidence
        else:
            chosen_idx = sorted(sorted(range(len(evidence)), key=lambda i: (-scores[i], i))[:top_k])
            selected = [evidence[i] for i in chosen_idx]
    elif policy == "hybrid_topk_compact":
        scores = passage_scores if passage_scores is not None else _score_passages_bm25(claim, [str(e.get("text", "")) for e in evidence])
        selected = _build_hybrid_topk_compact(
            claim, evidence, scores, top_k=top_k, tail_sentences=compact_sentences, tail_chars=compact_chars
        )
    elif policy == "sentence_budget":
        selected = _build_sentence_budget_evidence(
            claim,
            evidence,
            sentence_scores_by_passage=sentence_scores_by_passage,
            budget_chars=budget_chars,
            min_sentences_per_passage=min_sentences_per_passage,
        )
    elif policy == "random_sentence_budget":
        import random
        claim_hash = sum(ord(c) for c in claim)
        rng = random.Random(seed + claim_hash)
        pass_sents_spans = [split_evidence_sentences_with_spans(str(e.get("text", ""))) for e in evidence]
        rnd_scores = [[rng.random() for _ in spans] for spans in pass_sents_spans]
        selected = _build_sentence_budget_evidence(
            claim,
            evidence,
            sentence_scores_by_passage=rnd_scores,
            budget_chars=budget_chars,
            min_sentences_per_passage=min_sentences_per_passage,
        )
    else:
        raise ValueError(f"Unsupported scored policy: {policy!r}")

    draft_prompt, draft_hash = build_prompt(claim, selected, mode=ev_mode)
    sel_ids = [str(e.get("evidence_id", "")) for e in selected]
    draft_chars = sum(len(str(e.get("text", ""))) for e in selected)
    selection_s = (time.perf_counter() - t0) + float(extra_selection_s)
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


def build_draft_context(
    row: dict[str, Any],
    *,
    policy: str = "full",
    top_k: int = 2,
    compact_sentences: int = 1,
    compact_chars: int = 160,
    budget_chars: int = 8000,
    min_sentences_per_passage: int = 1,
    seed: int = 42,
) -> DraftContextResult:
    """Construct draft prompt under the requested policy without touching target prompt or gold labels.

    Args:
        row: Prepared benchmark row containing 'claim', 'evidence', 'prompt', and 'evidence_mode'.
        policy: One of {"full", "selective_topk", "compact", "hybrid_topk_compact", "sentence_budget", "random_sentence_budget"}.
        top_k: Number of evidence passages retained under 'selective_topk' / 'hybrid_topk_compact'.
        compact_sentences: Number of claim-salient sentences kept per passage under 'compact' / tail.
        compact_chars: Character budget per compacted passage under 'compact' / tail.
        budget_chars: Total character budget across all evidence under 'sentence_budget' / 'random_sentence_budget'.
        min_sentences_per_passage: Minimum sentences retained per passage.
        seed: Random seed for randomized baselines.
    """
    valid_policies = {
        "full",
        "no_evidence",
        "sliding_window",
        "selective_topk",
        "compact",
        "hybrid_topk_compact",
        "sentence_budget",
        "random_sentence_budget",
    }
    if policy not in valid_policies:
        raise ValueError(f"Unsupported draft context policy: {policy!r}")

    if policy in {"hybrid_topk_compact", "sentence_budget", "random_sentence_budget"}:
        return build_draft_context_from_scores(
            row,
            policy=policy,
            top_k=top_k,
            compact_sentences=compact_sentences,
            compact_chars=compact_chars,
            budget_chars=budget_chars,
            min_sentences_per_passage=min_sentences_per_passage,
            seed=seed,
        )

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

    if policy == "no_evidence":
        draft_prompt, draft_hash = build_prompt(claim, [], mode=ev_mode)
        return DraftContextResult(
            policy="no_evidence",
            draft_prompt=draft_prompt,
            draft_prompt_hash=draft_hash,
            selected_evidence_ids=[],
            total_evidence_count=len(evidence),
            selected_evidence_count=0,
            full_evidence_chars=full_chars,
            draft_evidence_chars=0,
            context_selection_s=0.0,
        )

    t0 = time.perf_counter()
    if policy == "sliding_window":
        # StreamingLLM-style: initial sink (first sentence of passage 1) + trailing evidence sentences up to budget
        pass_sents = [split_evidence_sentences_with_spans(str(e.get("text", ""))) for e in evidence]
        sink_sent = pass_sents[0][0][0] if pass_sents and pass_sents[0] else ""
        tail_sents = []
        cur_chars = len(sink_sent)
        # gather from reverse (tail sliding window)
        for p_idx in reversed(range(len(evidence))):
            for tup in reversed(pass_sents[p_idx]):
                s = tup[0]
                if cur_chars + len(s) + 1 <= budget_chars:
                    tail_sents.append(s)
                    cur_chars += len(s) + 1
                else:
                    break
            if cur_chars >= budget_chars:
                break
        tail_sents.reverse()
        combined_text = (sink_sent + "\n... [omitted] ...\n" + "\n".join(tail_sents)).strip()
        selected = [{"evidence_id": "SW", "text": combined_text}]
        draft_prompt, draft_hash = build_prompt(claim, selected, mode=ev_mode)
        sel_ids = ["SW"]
        draft_chars = len(combined_text)
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
