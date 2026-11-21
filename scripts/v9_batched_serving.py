"""Co-located batched E2E serving benchmark (Target AR vs Full-SD vs SPADE B=10k/2k) on 32 GB GPU."""
from __future__ import annotations

import gc
import statistics
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.cache_utils import Cache, CacheLayerMixin

from spade.adaptive.target_attention_selector import (
    TargetPrefillAttentionHook,
    find_evidence_and_sentence_token_spans,
    resolve_layer_indices,
    score_evidence_from_token_importance,
)
from spade.adaptive.context_selection import build_draft_context_from_scores
from spade.decoding.autoregressive import format_qwen_prompt
from spade.decoding.runtime_opt import _set_static_cache_len, processed_distribution_fast
from spade.decoding.sampling import sample_distribution
from spade.decoding.speculative import exact_speculative_round


class _SliceLayer(CacheLayerMixin):
    """View into batch row `batch_idx` of a pre-allocated batched StaticCache layer."""

    is_compileable = False

    def __init__(self, parent_layer: Any, batch_idx: int) -> None:
        self.parent = parent_layer
        self.b = int(batch_idx)
        self._pos = 0

    @property
    def keys(self) -> torch.Tensor:
        return self.parent.keys[self.b : self.b + 1, :, : self._pos, :]

    @property
    def values(self) -> torch.Tensor:
        return self.parent.values[self.b : self.b + 1, :, : self._pos, :]

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        return

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_len = int(key_states.shape[-2])
        s = self._pos
        self.parent.keys[self.b, :, s : s + kv_len, :].copy_(key_states[0])
        self.parent.values[self.b, :, s : s + kv_len, :].copy_(value_states[0])
        self._pos = s + kv_len
        return (
            self.parent.keys[self.b : self.b + 1, :, : self._pos, :],
            self.parent.values[self.b : self.b + 1, :, : self._pos, :],
        )

    def get_seq_length(self) -> int:
        return self._pos

    def get_max_cache_shape(self) -> int:
        return int(self.parent.keys.shape[-2])

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return self._pos + int(cache_position.shape[0]), 0


class _SliceStaticCache(Cache):
    """Zero-overhead adapter writing chunked prefill directly into batch row `b` of a StaticCache."""

    def __init__(self, parent_cache: StaticCache, batch_idx: int) -> None:
        super().__init__(layers=[_SliceLayer(layer, batch_idx) for layer in parent_cache.layers])


def _chunked_prefill_into_row(
    model: Any,
    parent_cache: StaticCache,
    batch_idx: int,
    token_ids: list[int],
    *,
    device: torch.device,
    chunk_size: int = 4096,
    hook_layers: list[int] | None = None,
    tail_window: int = 16,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Prefill `token_ids` chunk-by-chunk into `parent_cache[batch_idx]` with near-zero extra KV VRAM."""
    slice_cache = _SliceStaticCache(parent_cache, batch_idx)
    n = len(token_ids)
    last_logits: torch.Tensor | None = None
    imp_cpu: torch.Tensor | None = None

    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        if end < n and (n - end) < max(64, tail_window):
            end = n - max(64, tail_window)
        is_last = (end == n)
        if start >= end:
            start = end
            continue
        inp = torch.tensor([token_ids[start:end]], dtype=torch.long, device=device)
        cpos = torch.arange(start, end, dtype=torch.long, device=device)
        pids = cpos.unsqueeze(0)
        kw: dict[str, Any] = {
            "input_ids": inp,
            "past_key_values": slice_cache,
            "cache_position": cpos,
            "position_ids": pids,
            "use_cache": True,
            "logits_to_keep": 1,
        }
        if is_last and hook_layers is not None:
            with TargetPrefillAttentionHook(model, layer_indices=hook_layers, tail_window=tail_window) as hk:
                with torch.inference_mode():
                    out = model(**kw)
                last_logits = out.logits[0, -1].float()
                imp_cpu, _ = hk.compute_token_importance(slice_cache, head_agg="mean", seq_len=n)
                del out
        else:
            with torch.inference_mode():
                out = model(**kw)
            if is_last:
                last_logits = out.logits[0, -1].float()
            del out
        del inp, cpos, pids
        start = end

    assert last_logits is not None
    return last_logits, imp_cpu


def _cache_kv_bytes(cache: StaticCache) -> int:
    tot = 0
    for lay in cache.layers:
        k = getattr(lay, "keys", None)
        v = getattr(lay, "values", None)
        if k is not None:
            tot += k.numel() * k.element_size()
        if v is not None:
            tot += v.numel() * v.element_size()
    return tot


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_single_batched_ar(
    target: Any,
    tok: Any,
    prompts: list[dict[str, Any]],
    *,
    bs: int,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
) -> dict[str, Any]:
    """Run compiled StaticCache batched Target AR (no draft model resident)."""
    device = target.get_input_embeddings().weight.device
    cfg = target.config
    n_kvh = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
    hd = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    dtype = target.get_input_embeddings().weight.dtype

    p_ids_list = [
        tok(format_qwen_prompt(tok, prompts[i % len(prompts)]["prompt"]), add_special_tokens=False)["input_ids"]
        for i in range(bs)
    ]
    max_p = max(len(x) for x in p_ids_list)
    t_cap = ((max_p + max_new_tokens + 64 + 127) // 128) * 128

    t_cache = StaticCache(config=cfg, max_cache_len=t_cap)
    t_cache.early_initialization(batch_size=bs, num_heads=n_kvh, head_dim=hd, dtype=dtype, device=device)

    zero_f = torch.tensor(0.0, dtype=dtype, device=device)
    neg_f = torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device)

    def _raw_step(ii: torch.Tensor, am: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        return target(
            input_ids=ii, attention_mask=am, past_key_values=t_cache, position_ids=pi, use_cache=True, logits_to_keep=1
        ).logits[:, -1]

    compiled_step = torch.compile(_raw_step, mode="default", fullgraph=False)

    # Compile + warmup using dummy last slot before writing real prompts
    w_inp = torch.ones((bs, 1), dtype=torch.long, device=device)
    w_mask = torch.zeros((bs, 1, 1, t_cap), dtype=dtype, device=device)
    w_pids = torch.full((bs, 1), max_p, dtype=torch.long, device=device)
    for w_idx in range(3):
        _set_static_cache_len(t_cache, max_p + w_idx)
        with torch.inference_mode():
            _ = compiled_step(w_inp, w_mask, w_pids)
    torch.cuda.synchronize(device)
    del w_inp, w_mask, w_pids

    # Prefill into batched StaticCache slice-by-slice
    t_cache.reset()
    _set_static_cache_len(t_cache, 0)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t_pre0 = time.perf_counter()

    t_valid = torch.zeros((bs, t_cap), dtype=torch.bool, device=device)
    t_pos = torch.zeros((bs,), dtype=torch.long, device=device)
    next_logits = torch.empty((bs, cfg.vocab_size), dtype=torch.float32, device=device)
    for b in range(bs):
        lg, _ = _chunked_prefill_into_row(target, t_cache, b, p_ids_list[b], device=device)
        next_logits[b].copy_(lg)
        plen = len(p_ids_list[b])
        t_valid[b, :plen] = True
        t_pos[b] = plen
    torch.cuda.synchronize(device)
    prefill_s = time.perf_counter() - t_pre0

    # Decode M steps cleanly with EOS masked out so all bs requests execute full matched decode
    eos_id = tok.eos_token_id
    gen = torch.Generator(device=device).manual_seed(seed)
    t_slot = max_p
    step_inp = torch.empty((bs, 1), dtype=torch.long, device=device)
    torch.cuda.synchronize(device)
    t_dec0 = time.perf_counter()

    for _step in range(max_new_tokens - 1):
        lg_m = next_logits.clone()
        if eos_id is not None and eos_id < lg_m.shape[-1]:
            lg_m[:, eos_id] = float("-inf")
        probs = processed_distribution_fast(lg_m, temperature=temperature, top_p=top_p, dtype=torch.float32)
        next_toks = torch.multinomial(probs, num_samples=1, generator=gen)  # [bs, 1]
        step_inp.copy_(next_toks)
        t_valid[:, t_slot] = True
        _set_static_cache_len(t_cache, t_slot)
        amask = torch.where(t_valid.view(bs, 1, 1, t_cap), zero_f, neg_f)
        pids = t_pos.view(bs, 1)
        with torch.inference_mode():
            next_logits = compiled_step(step_inp, amask, pids).float()
        t_slot += 1
        t_pos.add_(1)

    torch.cuda.synchronize(device)
    decode_s = time.perf_counter() - t_dec0
    e2e_s = prefill_s + decode_s
    tot_tok = bs * max_new_tokens
    t_kv_gb = _cache_kv_bytes(t_cache) / (1024.0 ** 3)
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)

    del compiled_step, t_cache, t_valid, t_pos, next_logits, step_inp
    torch._dynamo.reset()
    _cleanup()
    return {
        "policy": "Target AR (8B Only)",
        "batch_size": bs,
        "oom": False,
        "t_cap": t_cap,
        "d_cap": 0,
        "target_kv_gb": round(t_kv_gb, 3),
        "draft_kv_gb": 0.0,
        "total_kv_gb": round(t_kv_gb, 3),
        "peak_vram_gb": round(peak_gb, 2),
        "prefill_sel_s": round(prefill_s, 4),
        "decode_s": round(decode_s, 4),
        "e2e_s": round(e2e_s, 4),
        "agg_dec_tok_s": round(tot_tok / decode_s, 2) if decode_s > 0 else 0.0,
        "agg_e2e_tok_s": round(tot_tok / e2e_s, 2) if e2e_s > 0 else 0.0,
        "per_req_e2e_tok_s": round((max_new_tokens / e2e_s), 2) if e2e_s > 0 else 0.0,
        "alpha_eval_pct": None,
        "tok_per_round": 1.0,
        "ar_step_ms": round((decode_s / max(1, max_new_tokens - 1)) * 1000.0, 3),
        "verify_rnd_ms": None,
    }


def run_single_batched_sd(
    target: Any,
    draft: Any,
    tok: Any,
    prompts: list[dict[str, Any]],
    *,
    bs: int,
    policy_label: str,
    budget_chars: int | None,
    max_new_tokens: int = 128,
    draft_length: int = 4,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
) -> dict[str, Any]:
    """Run compiled StaticCache co-located batched speculative decoding (Full-SD or SPADE)."""
    device = target.get_input_embeddings().weight.device
    t_cfg = target.config
    d_cfg = draft.config
    t_kvh = int(getattr(t_cfg, "num_key_value_heads", t_cfg.num_attention_heads))
    t_hd = int(getattr(t_cfg, "head_dim", t_cfg.hidden_size // t_cfg.num_attention_heads))
    d_kvh = int(getattr(d_cfg, "num_key_value_heads", d_cfg.num_attention_heads))
    d_hd = int(getattr(d_cfg, "head_dim", d_cfg.hidden_size // d_cfg.num_attention_heads))
    dtype = target.get_input_embeddings().weight.dtype
    hook_layers = resolve_layer_indices(len(target.model.layers), "upper50")

    row_list = [prompts[i % len(prompts)] for i in range(bs)]
    p_ser_list = [format_qwen_prompt(tok, r["prompt"]) for r in row_list]
    enc_list = [tok(s, return_offsets_mapping=True, add_special_tokens=False) for s in p_ser_list]
    p_ids_list = [list(e["input_ids"]) for e in enc_list]
    p_off_list = [[(int(a), int(b)) for a, b in e["offset_mapping"]] for e in enc_list]
    max_p = max(len(x) for x in p_ids_list)

    m_ver = draft_length + 1
    headroom_slots = max(384, max_new_tokens * 3)
    t_cap = ((max_p + headroom_slots + 127) // 128) * 128
    if budget_chars is None:
        d_cap = t_cap
    elif budget_chars >= 10000:
        d_cap = ((2800 + headroom_slots + 127) // 128) * 128
    else:
        d_cap = ((900 + headroom_slots + 127) // 128) * 128

    t_cache = StaticCache(config=t_cfg, max_cache_len=t_cap)
    t_cache.early_initialization(batch_size=bs, num_heads=t_kvh, head_dim=t_hd, dtype=dtype, device=device)
    d_cache = StaticCache(config=d_cfg, max_cache_len=d_cap)
    d_cache.early_initialization(batch_size=bs, num_heads=d_kvh, head_dim=d_hd, dtype=dtype, device=device)

    zero_f = torch.tensor(0.0, dtype=dtype, device=device)
    neg_f = torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device)

    def _raw_d_step(ii: torch.Tensor, am: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        return draft(
            input_ids=ii, attention_mask=am, past_key_values=d_cache, position_ids=pi, use_cache=True, logits_to_keep=1
        ).logits[:, -1]

    def _raw_t_ver(ii: torch.Tensor, am: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        return target(
            input_ids=ii, attention_mask=am, past_key_values=t_cache, position_ids=pi, use_cache=True
        ).logits

    compiled_d_step = torch.compile(_raw_d_step, mode="default", fullgraph=False)
    compiled_t_ver = torch.compile(_raw_t_ver, mode="default", fullgraph=False)

    # Warmup / compile fixed shapes strictly before timed benchmark
    wd_in = torch.ones((bs, 1), dtype=torch.long, device=device)
    wd_m = torch.zeros((bs, 1, 1, d_cap), dtype=dtype, device=device)
    wd_p = torch.zeros((bs, 1), dtype=torch.long, device=device)
    wt_in = torch.ones((bs, m_ver), dtype=torch.long, device=device)
    wt_m = torch.zeros((bs, 1, m_ver, t_cap), dtype=dtype, device=device)
    wt_p = torch.arange(m_ver, dtype=torch.long, device=device).view(1, m_ver).expand(bs, m_ver).contiguous()
    for w_i in range(3):
        _set_static_cache_len(d_cache, w_i)
        _set_static_cache_len(t_cache, w_i * m_ver)
        with torch.inference_mode():
            _ = compiled_d_step(wd_in, wd_m, wd_p)
            _ = compiled_t_ver(wt_in, wt_m, wt_p)
    torch.cuda.synchronize(device)
    del wd_in, wd_m, wd_p, wt_in, wt_m, wt_p

    # Prefill + SPADE selection per batch item cleanly
    t_cache.reset(); d_cache.reset()
    _set_static_cache_len(t_cache, 0); _set_static_cache_len(d_cache, 0)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t_pre0 = time.perf_counter()

    t_valid = torch.zeros((bs, t_cap), dtype=torch.bool, device=device)
    d_valid = torch.zeros((bs, d_cap), dtype=torch.bool, device=device)
    t_pos = torch.zeros((bs,), dtype=torch.long, device=device)
    d_pos = torch.zeros((bs,), dtype=torch.long, device=device)
    next_t_logits = torch.empty((bs, t_cfg.vocab_size), dtype=torch.float32, device=device)
    next_d_logits = torch.empty((bs, d_cfg.vocab_size), dtype=torch.float32, device=device)

    max_d = 0
    for b in range(bs):
        use_hook = budget_chars is not None and sum(len(str(e.get("text", ""))) for e in (row_list[b].get("evidence") or [])) > budget_chars
        t_lg, imp_cpu = _chunked_prefill_into_row(
            target, t_cache, b, p_ids_list[b], device=device, hook_layers=hook_layers if use_hook else None
        )
        next_t_logits[b].copy_(t_lg)
        plen = len(p_ids_list[b])
        t_valid[b, :plen] = True
        t_pos[b] = plen

        if use_hook:
            assert imp_cpu is not None
            ev_list = list(row_list[b].get("evidence") or [])
            spans = find_evidence_and_sentence_token_spans(p_ser_list[b], ev_list, p_off_list[b])
            _, s_scores = score_evidence_from_token_importance(imp_cpu, spans, sentence_agg="sum")
            dctx = build_draft_context_from_scores(
                row_list[b], policy="sentence_budget", sentence_scores_by_passage=s_scores, budget_chars=budget_chars, min_sentences_per_passage=1
            )
            d_ids = tok(format_qwen_prompt(tok, dctx.draft_prompt), add_special_tokens=False)["input_ids"]
            if len(d_ids) + headroom_slots >= d_cap:
                d_ids = d_ids[: max(64, d_cap - headroom_slots - 8)]
        else:
            d_ids = p_ids_list[b]
            if len(d_ids) + headroom_slots >= d_cap:
                d_ids = d_ids[: max(64, d_cap - headroom_slots - 8)]
        d_lg, _ = _chunked_prefill_into_row(draft, d_cache, b, d_ids, device=device)
        next_d_logits[b].copy_(d_lg)
        dlen = len(d_ids)
        d_valid[b, :dlen] = True
        d_pos[b] = dlen
        max_d = max(max_d, dlen)

    torch.cuda.synchronize(device)
    prefill_s = time.perf_counter() - t_pre0

    eos_id = tok.eos_token_id
    common_vocab = min(int(t_cfg.vocab_size), int(d_cfg.vocab_size))

    def _mask_and_dist(lg_2d: torch.Tensor) -> torch.Tensor:
        x = lg_2d[:, :common_vocab].clone()
        if eos_id is not None and eos_id < common_vocab:
            x[:, eos_id] = float("-inf")
        return processed_distribution_fast(x, temperature=temperature, top_p=top_p, dtype=torch.float32)

    gen = torch.Generator(device=device).manual_seed(seed)
    t_slot = max_p
    d_slot = max_d
    generated: list[list[int]] = [[] for _ in range(bs)]
    pending_tok: list[int | None] = [None] * bs
    ver_offsets = torch.arange(m_ver, dtype=torch.long, device=device).view(1, m_ver)

    n_rounds = 0
    tot_eval = 0
    tot_eval_acc = 0
    verify_s_acc = 0.0
    d_step_in = torch.empty((bs, 1), dtype=torch.long, device=device)
    t_ver_in = torch.zeros((bs, m_ver), dtype=torch.long, device=device)

    torch.cuda.synchronize(device)
    t_dec0 = time.perf_counter()

    while min(len(g) for g in generated) < max_new_tokens:
        if t_slot + m_ver > t_cap or d_slot + draft_length + 2 > d_cap:
            break
        n_rounds += 1
        d_round_start = d_slot
        cur_d_logits = next_d_logits
        proposed_b: list[list[int]] = [[] for _ in range(bs)]
        q_dists_b: list[list[torch.Tensor]] = [[] for _ in range(bs)]

        # Draft K=4 autoregressive batched steps
        for idx in range(draft_length):
            q_all = _mask_and_dist(cur_d_logits)
            toks = torch.multinomial(q_all, num_samples=1, generator=gen)  # [bs, 1]
            toks_cpu = toks.squeeze(1).tolist()
            for b in range(bs):
                proposed_b[b].append(int(toks_cpu[b]))
                q_dists_b[b].append(q_all[b])
            if idx < draft_length - 1:
                d_step_in.copy_(toks)
                d_valid[:, d_slot] = True
                _set_static_cache_len(d_cache, d_slot)
                dam = torch.where(d_valid.view(bs, 1, 1, d_cap), zero_f, neg_f)
                dpids = d_pos.view(bs, 1)
                with torch.inference_mode():
                    cur_d_logits = compiled_d_step(d_step_in, dam, dpids).float()
                d_slot += 1
                d_pos.add_(1)

        # Target parallel verify batched pass (shape [bs, 5])
        tv0 = time.perf_counter()
        t_ver_host = torch.zeros((bs, m_ver), dtype=torch.long)
        has_pending_b = [pending_tok[b] is not None for b in range(bs)]
        for b in range(bs):
            vt = ([pending_tok[b]] + proposed_b[b]) if has_pending_b[b] else proposed_b[b]
            for j, tid in enumerate(vt):
                t_ver_host[b, j] = tid
        t_ver_in.copy_(t_ver_host)

        # Build [bs, 1, m_ver, t_cap] causal mask over previously valid slots + causal verify block
        base_v = t_valid.view(bs, 1, 1, t_cap).expand(bs, 1, m_ver, t_cap).clone()
        for q_i in range(m_ver):
            base_v[:, 0, q_i, t_slot : t_slot + q_i + 1] = True
        tam = torch.where(base_v, zero_f, neg_f)
        tpids = t_pos.view(bs, 1) + ver_offsets
        _set_static_cache_len(t_cache, t_slot)
        with torch.inference_mode():
            ver_logits = compiled_t_ver(t_ver_in, tam, tpids).float()  # [bs, m_ver, V]
        torch.cuda.synchronize(device)
        verify_s_acc += time.perf_counter() - tv0

        # Exact residual verification per batch element
        finished_before = [len(generated[b]) >= max_new_tokens for b in range(bs)]
        catchup_toks: list[list[int]] = [[] for _ in range(bs)]
        d_new_pos = [0] * bs
        t_new_pos = [0] * bs
        d_valid[:, d_round_start:d_slot] = False  # selectively re-enable kept accepted prefix below

        for b in range(bs):
            d_before = int(d_pos[b].item()) - (draft_length - 1)
            if finished_before[b]:
                pending_tok[b] = None
                d_new_pos[b] = d_before
                t_new_pos[b] = int(t_pos[b].item())
                continue
            rem = max_new_tokens - len(generated[b])
            prop_use = proposed_b[b][: min(draft_length, rem)]
            q_use = q_dists_b[b][: len(prop_use)]
            hp = has_pending_b[b]
            if hp:
                t_stack = ver_logits[b, : len(prop_use) + 1]
            else:
                t_stack = torch.cat([next_t_logits[b].unsqueeze(0), ver_logits[b, : len(prop_use)]], dim=0)
            batched_p = _mask_and_dist(t_stack)
            p_dists = [batched_p[i] for i in range(len(prop_use))]
            target_extra = batched_p[len(prop_use)]

            res = exact_speculative_round(
                prop_use, q_use, p_dists, generator=gen, eos_token_id=None, remaining=rem, target_extra=target_extra
            )
            generated[b].extend(res.committed)
            n_eval_b = min(len(prop_use), res.accepted + (0 if res.rejected_position is None else 1))
            tot_eval += n_eval_b
            tot_eval_acc += res.accepted

            # Update Target KV valid mask and RoPE cursor
            t_keep = (1 if hp else 0) + res.accepted
            if t_keep > 0:
                t_valid[b, t_slot : t_slot + t_keep] = True
            t_new_pos[b] = int(t_pos[b].item()) + t_keep

            # Rollback draft valid mask to accepted prefix (< draft_length) and prepare catch-up
            d_keep = min(res.accepted, draft_length - 1)
            if d_keep > 0:
                d_valid[b, d_round_start : d_round_start + d_keep] = True
            d_new_pos[b] = d_before + d_keep

            last_tok = res.committed[-1]
            if res.rejected_position is not None:
                pending_tok[b] = last_tok
                catchup_toks[b] = [last_tok]
            elif res.extra_target is not None:
                pending_tok[b] = last_tok
                catchup_toks[b] = [prop_use[-1], res.extra_target]
            else:
                pending_tok[b] = None
                next_t_logits[b].copy_(ver_logits[b, len(prop_use) - (0 if hp else 1)])
                catchup_toks[b] = [prop_use[-1]]

        t_slot += m_ver
        t_pos = torch.tensor(t_new_pos, dtype=torch.long, device=device)
        d_pos = torch.tensor(d_new_pos, dtype=torch.long, device=device)

        if min(len(g) for g in generated) >= max_new_tokens:
            break

        # Execute up to 2 batched draft catch-up steps
        max_cu = max(len(x) for x in catchup_toks)
        for cu_i in range(max_cu):
            active_mask = torch.tensor([cu_i < len(catchup_toks[b]) for b in range(bs)], dtype=torch.bool, device=device)
            tok_vec = torch.tensor(
                [[catchup_toks[b][cu_i] if cu_i < len(catchup_toks[b]) else 0] for b in range(bs)],
                dtype=torch.long,
                device=device,
            )
            d_valid[:, d_slot] = active_mask
            _set_static_cache_len(d_cache, d_slot)
            dam = torch.where(d_valid.view(bs, 1, 1, d_cap), zero_f, neg_f)
            with torch.inference_mode():
                cu_lg = compiled_d_step(tok_vec, dam, d_pos.view(bs, 1)).float()
            for b in range(bs):
                if cu_i == len(catchup_toks[b]) - 1:
                    next_d_logits[b].copy_(cu_lg[b])
            d_slot += 1
            d_pos.add_(active_mask.long())

    torch.cuda.synchronize(device)
    decode_s = time.perf_counter() - t_dec0
    e2e_s = prefill_s + decode_s
    tot_tok = sum(len(g) for g in generated)
    t_kv_gb = _cache_kv_bytes(t_cache) / (1024.0 ** 3)
    d_kv_gb = _cache_kv_bytes(d_cache) / (1024.0 ** 3)
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)

    del compiled_d_step, compiled_t_ver, t_cache, d_cache, t_valid, d_valid, t_pos, d_pos, next_t_logits, next_d_logits, d_step_in, t_ver_in
    torch._dynamo.reset()
    _cleanup()
    return {
        "policy": policy_label,
        "batch_size": bs,
        "oom": False,
        "t_cap": t_cap,
        "d_cap": d_cap,
        "target_kv_gb": round(t_kv_gb, 3),
        "draft_kv_gb": round(d_kv_gb, 3),
        "total_kv_gb": round(t_kv_gb + d_kv_gb, 3),
        "peak_vram_gb": round(peak_gb, 2),
        "prefill_sel_s": round(prefill_s, 4),
        "decode_s": round(decode_s, 4),
        "e2e_s": round(e2e_s, 4),
        "agg_dec_tok_s": round(tot_tok / decode_s, 2) if decode_s > 0 else 0.0,
        "agg_e2e_tok_s": round(tot_tok / e2e_s, 2) if e2e_s > 0 else 0.0,
        "per_req_e2e_tok_s": round((tot_tok / bs) / e2e_s, 2) if e2e_s > 0 else 0.0,
        "alpha_eval_pct": round((tot_eval_acc / max(1, tot_eval)) * 100.0, 2),
        "tok_per_round": round(tot_tok / max(1, n_rounds * bs), 3),
        "verify_rnd_ms": round((verify_s_acc / max(1, n_rounds)) * 1000.0, 3),
    }
