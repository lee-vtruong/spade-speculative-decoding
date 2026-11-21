"""FlashAttention-compatible, zero-extra-forward-pass Target-Prefill Attention Selector.

Exploits the natural asymmetry of RAG/fact-checking speculative decoding:
the target model (e.g. Qwen3-8B) ALWAYS prefills the full evidence prompt before
decoding starts. By attaching a lightweight `forward_pre_hook` that slices only
the last `tail_window` query hidden states (~130 KB/layer) and computing
`softmax(Q_tail @ K^T / sqrt(d))` directly against the already-populated target
KV cache, we obtain exact per-token target attention importance over all prompt
tokens in < 1 ms on GPU, with:
  - ZERO extra model forward passes,
  - 100% FlashAttention compatibility (never sets `output_attentions=True`),
  - 100% zero-leakage (computed strictly from the input prompt before token 0).
"""
from __future__ import annotations

import bisect
import json
import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from spade.adaptive.context_selection import split_evidence_sentences_with_spans


@dataclass(frozen=True)
class EvidenceTokenSpans:
    """Exact token-index spans `[start, end)` for every passage and its sentences."""
    passage_spans: list[tuple[int, int]]
    sentence_spans_by_passage: list[list[tuple[int, int]]]


def _char_span_to_token_span(
    char_start: int,
    char_end: int,
    tok_starts: list[int],
    tok_ends: list[int],
) -> tuple[int, int]:
    if not tok_starts or char_end <= char_start:
        return (0, 0)
    # First token whose end > char_start
    s_idx = bisect.bisect_right(tok_ends, char_start)
    # First token whose start >= char_end
    e_idx = bisect.bisect_left(tok_starts, char_end)
    s_idx = max(0, min(s_idx, len(tok_starts)))
    e_idx = max(s_idx, min(e_idx, len(tok_starts)))
    return (s_idx, e_idx)


def find_evidence_and_sentence_token_spans(
    serialized_prompt: str,
    evidence: list[dict[str, Any]],
    offsets: list[tuple[int, int]],
) -> EvidenceTokenSpans:
    """Map each evidence passage and its sentences to exact `[tok_start, tok_end)` spans."""
    tok_starts = [int(a) for a, _ in offsets]
    tok_ends = [int(b) for _, b in offsets]
    passage_spans: list[tuple[int, int]] = []
    sentence_spans_by_passage: list[list[tuple[int, int]]] = []

    cursor = 0
    for i, item in enumerate(evidence):
        eid = str(item.get("evidence_id", f"E{i+1}"))
        raw_text = str(item.get("text", ""))
        item_json = json.dumps({"evidence_id": eid, "text": raw_text}, ensure_ascii=False, separators=(",", ":"))
        idx = serialized_prompt.find(item_json, cursor)
        if idx < 0:
            idx = serialized_prompt.find(item_json)
        if idx < 0:
            passage_spans.append((0, 0))
            sent_list = split_evidence_sentences_with_spans(raw_text)
            sentence_spans_by_passage.append([(0, 0)] * len(sent_list))
            continue

        p_char_start = idx
        p_char_end = idx + len(item_json)
        cursor = p_char_end
        p_tok_span = _char_span_to_token_span(p_char_start, p_char_end, tok_starts, tok_ends)
        passage_spans.append(p_tok_span)

        # Locate the JSON-encoded text value inside item_json for sentence alignment
        text_key_prefix = f'"text":"'
        tk_pos = item_json.find(text_key_prefix)
        text_char_base = p_char_start + tk_pos + len(text_key_prefix) if tk_pos >= 0 else p_char_start
        text_char_end = p_char_end - 2 if p_char_end >= 2 else p_char_end

        sents = split_evidence_sentences_with_spans(raw_text)
        s_tok_spans: list[tuple[int, int]] = []
        s_cursor = text_char_base
        raw_len = max(1, len(raw_text))
        enc_len = max(1, text_char_end - text_char_base)

        for s_text, cs, ce in sents:
            # Try exact JSON-escaped match first (fast & exact)
            esc = json.dumps(s_text, ensure_ascii=False)[1:-1]
            s_pos = serialized_prompt.find(esc, s_cursor, text_char_end)
            if s_pos >= 0:
                sc_s = s_pos
                sc_e = s_pos + len(esc)
                s_cursor = sc_e
            else:
                # Proportional fallback if escaping crosses boundary
                sc_s = text_char_base + int((cs / raw_len) * enc_len)
                sc_e = text_char_base + int((ce / raw_len) * enc_len)
            s_tok_spans.append(_char_span_to_token_span(sc_s, sc_e, tok_starts, tok_ends))
        sentence_spans_by_passage.append(s_tok_spans)

    return EvidenceTokenSpans(
        passage_spans=passage_spans,
        sentence_spans_by_passage=sentence_spans_by_passage,
    )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def extract_layer_kv(past_key_values: Any, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrieve `(keys, values)` tensors `[B, n_kv_heads, seq_len, head_dim]` from DynamicCache / tuple."""
    layers = getattr(past_key_values, "layers", None)
    if isinstance(layers, (list, tuple)) and layer_idx < len(layers):
        k = getattr(layers[layer_idx], "keys", None)
        v = getattr(layers[layer_idx], "values", None)
        if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
            return k, v
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache") and layer_idx < len(past_key_values.key_cache):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]
    if isinstance(past_key_values, (tuple, list)) and layer_idx < len(past_key_values):
        return past_key_values[layer_idx][0], past_key_values[layer_idx][1]
    raise RuntimeError(f"Could not extract cached KV for layer {layer_idx}")


def _extract_layer_keys(past_key_values: Any, layer_idx: int) -> torch.Tensor:
    """Retrieve `[B, n_kv_heads, seq_len, head_dim]` key tensor from DynamicCache / tuple."""
    return extract_layer_kv(past_key_values, layer_idx)[0]


def resolve_layer_indices(n_layers: int, spec: str = "upper50") -> list[int]:
    """Map a named layer-subset specification into valid layer indices."""
    if n_layers <= 0:
        return []
    if spec == "upper50":
        return list(range(max(0, n_layers // 2), n_layers))
    if spec == "upper25":
        return list(range(max(0, (3 * n_layers) // 4), n_layers))
    if spec == "middle50":
        lo = max(0, n_layers // 4)
        hi = max(lo + 1, (3 * n_layers) // 4)
        return list(range(lo, min(n_layers, hi)))
    if spec == "all":
        return list(range(n_layers))
    if spec == "last1":
        return [n_layers - 1]
    raise ValueError(f"Unsupported layer subset spec: {spec!r}")


class TargetPrefillAttentionHook:
    """Lightweight pre-hook capturing only the last `tail_window` query hidden states during Target Prefill."""

    def __init__(
        self,
        target_model: Any,
        *,
        layer_indices: list[int] | None = None,
        tail_window: int = 16,
    ) -> None:
        backbone = getattr(target_model, "model", target_model)
        layers = getattr(backbone, "layers", None)
        if layers is None:
            raise ValueError("Target model does not expose `.model.layers` for attention hooking")
        self.layers = layers
        n_layers = len(layers)
        if layer_indices is None:
            # Default: upper-middle to final layers (strongest retrieval/evidence grounding signal)
            start = max(0, n_layers // 2)
            layer_indices = list(range(start, n_layers))
        self.layer_indices = [int(i) for i in layer_indices if 0 <= int(i) < n_layers]
        self.tail_window = max(1, int(tail_window))
        self._captured: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> "TargetPrefillAttentionHook":
        self._captured.clear()
        for l_idx in self.layer_indices:
            attn_mod = self.layers[l_idx].self_attn

            def _pre_hook(mod, args, kwargs, _l=l_idx):
                h = kwargs.get("hidden_states", args[0] if args else None)
                pos_emb = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
                if h is None or pos_emb is None:
                    return
                w = min(self.tail_window, int(h.shape[1]))
                cos, sin = pos_emb
                self._captured[_l] = (
                    h[:, -w:, :].detach().clone(),
                    cos[:, -w:, :].detach().clone(),
                    sin[:, -w:, :].detach().clone(),
                )

            self._handles.append(attn_mod.register_forward_pre_hook(_pre_hook, with_kwargs=True))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._captured.clear()

    @torch.inference_mode()
    def compute_layer_token_importance(
        self,
        l_idx: int,
        past_key_values: Any,
        *,
        head_agg: str = "mean",
        seq_len: int | None = None,
    ) -> torch.Tensor:
        """Compute exact 1D per-token importance `[P]` for a single hooked layer on-device."""
        if head_agg not in {"mean", "max"}:
            raise ValueError(f"Unsupported head_agg={head_agg!r}")
        h_tail, cos_tail, sin_tail = self._captured[l_idx]
        attn_mod = self.layers[l_idx].self_attn
        k_all = _extract_layer_keys(past_key_values, l_idx)  # [B, n_kv, P, D]
        if seq_len is not None:
            # Slice pre-allocated StaticCache capacity down to the populated prompt length
            k_all = k_all[:, :, : int(seq_len), :]
        bsz, n_kv, kv_len, head_dim = k_all.shape
        w = int(h_tail.shape[1])

        q_proj = attn_mod.q_proj(h_tail).view(bsz, w, -1, head_dim)
        if hasattr(attn_mod, "q_norm") and attn_mod.q_norm is not None:
            q_proj = attn_mod.q_norm(q_proj)
        q_states = q_proj.transpose(1, 2)  # [B, n_q, w, D]
        cos_u = cos_tail.unsqueeze(1)
        sin_u = sin_tail.unsqueeze(1)
        q_states = (q_states * cos_u) + (_rotate_half(q_states) * sin_u)

        n_q = int(q_states.shape[1])
        n_rep = max(1, n_q // n_kv)
        q_grouped = q_states.view(bsz, n_kv, n_rep, w, head_dim)
        k_grouped = k_all.unsqueeze(2)  # [B, n_kv, 1, P, D]
        scaling = float(getattr(attn_mod, "scaling", head_dim ** -0.5))
        scores = torch.matmul(q_grouped, k_grouped.transpose(-2, -1)) * scaling  # [B, n_kv, n_rep, w, P]

        # Causal mask on the W x W bottom-right corner
        if w > 1:
            row_idx = torch.arange(kv_len - w, kv_len, device=scores.device).view(1, 1, 1, w, 1)
            col_idx = torch.arange(kv_len, device=scores.device).view(1, 1, 1, 1, kv_len)
            scores = scores.masked_fill(col_idx > row_idx, float("-inf"))

        probs = torch.softmax(scores.float(), dim=-1)  # [B, n_kv, n_rep, w, P]
        if head_agg == "max":
            return probs[0].mean(dim=2).amax(dim=(0, 1))  # [P]
        return probs[0].mean(dim=(0, 1, 2))  # [P]

    @torch.inference_mode()
    def compute_token_importance(
        self,
        past_key_values: Any,
        *,
        layer_subset: list[int] | None = None,
        head_agg: str = "mean",
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Compute exact mean attention importance `[P]` across hooked layers from cached K & captured Q_tail."""
        t0 = time.perf_counter()
        use_layers = self.layer_indices if layer_subset is None else [l for l in layer_subset if l in self._captured]
        if not use_layers:
            raise RuntimeError("No target prefill query states were captured by TargetPrefillAttentionHook")

        acc_importance: torch.Tensor | None = None
        for l_idx in use_layers:
            layer_imp = self.compute_layer_token_importance(
                l_idx, past_key_values, head_agg=head_agg, seq_len=seq_len
            )
            acc_importance = layer_imp if acc_importance is None else (acc_importance + layer_imp)

        assert acc_importance is not None
        mean_imp = acc_importance / float(len(use_layers))
        if mean_imp.device.type == "cuda":
            torch.cuda.synchronize(mean_imp.device)
        elapsed_s = time.perf_counter() - t0
        return mean_imp.detach().cpu(), elapsed_s


def _aggregate_segment_score(seg: torch.Tensor, agg: str, peak_k: int = 32) -> float:
    if seg.numel() == 0:
        return 0.0
    if agg == "sum":
        return float(seg.sum().item())
    if agg == "mean":
        return float(seg.mean().item())
    if agg == "sqrt":
        return float(seg.sum().item() / math.sqrt(float(seg.numel())))
    if agg == "max":
        return float(seg.max().item())
    if agg == "peak":
        k = min(int(peak_k), int(seg.numel()))
        return float(torch.topk(seg, k).values.sum().item())
    raise ValueError(f"Unsupported aggregation={agg!r}")


def score_evidence_from_token_importance(
    token_importance: torch.Tensor,
    spans: EvidenceTokenSpans,
    *,
    passage_agg: str = "sum",
    sentence_agg: str = "sum",
    peak_k: int = 32,
) -> tuple[list[float], list[list[float]]]:
    """Aggregate 1D per-token target attention importance into passage & sentence scores."""
    imp = token_importance.detach().float().cpu()
    n_tok = int(imp.numel())

    passage_scores: list[float] = []
    for s, e in spans.passage_spans:
        s_c = max(0, min(s, n_tok))
        e_c = max(s_c, min(e, n_tok))
        passage_scores.append(_aggregate_segment_score(imp[s_c:e_c], passage_agg, peak_k=peak_k))

    sentence_scores_by_passage: list[list[float]] = []
    for s_spans in spans.sentence_spans_by_passage:
        s_scores: list[float] = []
        for s, e in s_spans:
            s_c = max(0, min(s, n_tok))
            e_c = max(s_c, min(e, n_tok))
            s_scores.append(_aggregate_segment_score(imp[s_c:e_c], sentence_agg, peak_k=peak_k))
        sentence_scores_by_passage.append(s_scores)

    return passage_scores, sentence_scores_by_passage
