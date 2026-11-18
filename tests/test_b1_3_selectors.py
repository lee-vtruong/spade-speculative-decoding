"""CPU unit tests for Step B1.3 evidence selectors and TargetPrefillAttentionHook exactness."""
from __future__ import annotations

import pytest
import torch

from spade.adaptive.context_selection import (
    build_draft_context,
    split_evidence_sentences_with_spans,
)
from spade.adaptive.target_attention_selector import (
    TargetPrefillAttentionHook,
    find_evidence_and_sentence_token_spans,
    score_evidence_from_token_importance,
)
from spade.benchmark.prompt_builder import build_prompt


def _sample_multi_evidence_row() -> dict:
    claim = "The Apollo 11 mission landed on the Moon in July 1969 with Neil Armstrong."
    ev = [
        {"evidence_id": "E1", "text": "Unrelated gardening guide. Tomatoes grow best in full sunlight and warm soil. Water daily."},
        {"evidence_id": "E2", "text": "Apollo 11 launched on July 16, 1969. Neil Armstrong stepped onto the lunar surface on July 20, 1969. Buzz Aldrin joined him shortly after."},
        {"evidence_id": "E3", "text": "Sports history note. Football matches last ninety minutes. Referees keep official time."},
        {"evidence_id": "E4", "text": "NASA confirmed the Apollo 11 Moon landing in July 1969. Michael Collins piloted the command module Columbia."},
    ]
    prompt, phash = build_prompt(claim, ev, mode="retrieved")
    return {
        "claim_id": "test-b1-3-1",
        "claim": claim,
        "evidence": ev,
        "evidence_mode": "retrieved",
        "prompt": prompt,
        "prompt_hash": phash,
        "evaluation": {"gold_label": "Supported"},
    }


def test_hybrid_topk_compact_and_sentence_budget_preserve_ids_and_zero_leakage():
    row = _sample_multi_evidence_row()
    row_alt = {**row, "evaluation": {"gold_label": "Refuted"}}

    h1 = build_draft_context(row, policy="hybrid_topk_compact", top_k=2, compact_sentences=1, compact_chars=80)
    h2 = build_draft_context(row_alt, policy="hybrid_topk_compact", top_k=2, compact_sentences=1, compact_chars=80)
    assert h1.draft_prompt == h2.draft_prompt
    assert h1.selected_evidence_ids == ["E1", "E2", "E3", "E4"]
    assert h1.draft_evidence_chars < h1.full_evidence_chars
    assert "Apollo 11 launched on July 16, 1969" in h1.draft_prompt
    assert "NASA confirmed the Apollo 11 Moon landing" in h1.draft_prompt

    sb1 = build_draft_context(row, policy="sentence_budget", budget_chars=320)
    sb2 = build_draft_context(row_alt, policy="sentence_budget", budget_chars=320)
    assert sb1.draft_prompt == sb2.draft_prompt
    assert sb1.selected_evidence_ids == ["E1", "E2", "E3", "E4"]
    assert sb1.draft_evidence_chars < sb1.full_evidence_chars
    assert "Neil Armstrong" in sb1.draft_prompt


def test_find_evidence_and_sentence_token_spans():
    row = _sample_multi_evidence_row()
    prompt = row["prompt"]
    # Character-level trivial offsets (1 char = 1 token) for exact deterministic boundary verification
    offsets = [(i, i + 1) for i in range(len(prompt))]
    spans = find_evidence_and_sentence_token_spans(prompt, row["evidence"], offsets)
    assert len(spans.passage_spans) == 4
    for p_idx, (ps, pe) in enumerate(spans.passage_spans):
        assert pe > ps
        assert f'"evidence_id":"E{p_idx+1}"' in prompt[ps:pe]
        assert len(spans.sentence_spans_by_passage[p_idx]) == len(split_evidence_sentences_with_spans(row["evidence"][p_idx]["text"]))
        for ss, se in spans.sentence_spans_by_passage[p_idx]:
            assert ps <= ss < se <= pe

    imp = torch.zeros(len(prompt), dtype=torch.float32)
    s2, e2 = spans.passage_spans[1]
    imp[s2:e2] = 1.0
    p_sc, s_sc = score_evidence_from_token_importance(imp, spans, passage_agg="sum")
    assert p_sc[1] > p_sc[0] and p_sc[1] > p_sc[2] and p_sc[1] > p_sc[3]
    assert sum(s_sc[1]) > 0.0 and sum(s_sc[0]) == 0.0


def test_target_prefill_attention_hook_matches_output_attentions_exact():
    transformers = pytest.importorskip("transformers")
    cfg_cls = getattr(transformers, "Qwen3Config", None)
    mdl_cls = getattr(transformers, "Qwen3ForCausalLM", None)
    if cfg_cls is None or mdl_cls is None:
        pytest.skip("Qwen3 classes unavailable")

    torch.manual_seed(17)
    cfg = cfg_cls(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attn_implementation="eager",
    )
    model = mdl_cls(cfg).eval()
    inp = torch.tensor([[1, 7, 13, 22, 5, 9, 18, 31, 4, 12, 25, 3, 19, 8, 11, 29, 6, 14, 21, 10]], dtype=torch.long)
    layer_subset = [2, 3]
    tail_w = 6

    with torch.inference_mode():
        ref_out = model(input_ids=inp, use_cache=False, output_attentions=True)
    # Reference exact mean attention over layers [2, 3], all heads, last `tail_w` query positions
    ref_imp = torch.stack(
        [ref_out.attentions[l][0, :, -tail_w:, :].float().mean(dim=(0, 1)) for l in layer_subset],
        dim=0,
    ).mean(dim=0)

    with TargetPrefillAttentionHook(model, layer_indices=layer_subset, tail_window=tail_w) as hook:
        with torch.inference_mode():
            out = model(input_ids=inp, use_cache=True)
        got_imp, elapsed = hook.compute_token_importance(out.past_key_values, layer_subset=layer_subset)

    assert elapsed >= 0.0
    assert got_imp.shape == ref_imp.shape
    max_diff = float(torch.max(torch.abs(got_imp - ref_imp)).item())
    assert max_diff < 1e-5, f"TargetPrefillAttentionHook mismatch vs output_attentions: max_diff={max_diff}"
