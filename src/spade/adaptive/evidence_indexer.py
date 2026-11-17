"""Evidence Token Indexer and Trie for Evidence-Adaptive Speculative Decoding (M4).

Extracts word n-grams, entity spans, and token sequences from provided evidence passages,
allowing high-speed lookup of evidence-supported continuations during draft token generation.
"""
from __future__ import annotations
from typing import Any
from collections import defaultdict
import torch

class EvidenceIndexer:
    """Indexes evidence passages into token n-grams for fast continuation matching."""

    def __init__(self, tokenizer: Any, evidence_texts: list[str] | None = None, max_ngram: int = 4):
        self.tokenizer = tokenizer
        self.max_ngram = max_ngram
        # Mapping from tuple of prefix token IDs -> set of next token IDs
        self.ngram_continuations: dict[tuple[int, ...], set[int]] = defaultdict(set)
        # All unique token IDs appearing in evidence
        self.evidence_token_set: set[int] = set()

        if evidence_texts:
            self.index_texts(evidence_texts)

    def index_texts(self, texts: list[str]) -> None:
        """Tokenize and index passages into n-gram continuations."""
        for text in texts:
            if not text or not text.strip():
                continue
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue

            for t in token_ids:
                self.evidence_token_set.add(t)

            n = len(token_ids)
            for order in range(1, self.max_ngram):
                for i in range(n - order):
                    prefix = tuple(token_ids[i:i + order])
                    next_token = token_ids[i + order]
                    self.ngram_continuations[prefix].add(next_token)

    def get_candidate_tokens(self, context_tokens: list[int]) -> tuple[set[int], int]:
        """Find candidate next tokens that continue an evidence n-gram.

        Returns:
            (candidate_tokens, matched_ngram_order)
        """
        if not self.ngram_continuations:
            return set(), 0

        # Try longest match first (up to max_ngram - 1)
        for order in range(min(self.max_ngram - 1, len(context_tokens)), 0, -1):
            prefix = tuple(context_tokens[-order:])
            candidates = self.ngram_continuations.get(prefix)
            if candidates:
                return candidates, order

        return set(), 0

    def is_evidence_token(self, token_id: int) -> bool:
        """Check if a single token appears anywhere in evidence."""
        return token_id in self.evidence_token_set
