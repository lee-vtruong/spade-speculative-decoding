"""Step B1.1 runtime optimization primitives for speculative drafting & sampling.

Provides:
1. `processed_distribution_fast`: GPU-friendly, zero-CPU-sync temperature/top-k/top-p
   distribution processing in float32 (supporting 1D [V] or 2D [B, V] logits) that
   preserves the exact nucleus cutoff semantics of `sampling.processed_distribution`.
2. `StaticDraftRunner`: Fixed-shape `StaticCache` wrapper for 1-token draft decoding
   supporting:
   - `eager_static`
   - `compile_default` (`torch.compile(mode="default")`)
   - `compile_reduce_overhead` (`torch.compile(mode="reduce-overhead")`)
   - `cudagraph_static` (explicit `torch.cuda.CUDAGraph()` capture)
   with explicit separation of compile/warmup/graph-capture time from steady-state
   decode time, plus safe KV-cache rollback via `cache_position` pointer reset.
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


def _supports_logits_to_keep(model: Any) -> bool:
    try:
        forward = getattr(model, "forward", model.__call__)
        return "logits_to_keep" in inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False


def processed_distribution_fast(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Fast, zero-CPU-sync temperature/top-k/top-p distribution transform.

    Supports both 1D `[V]` and 2D `[B, V]` logits tensors. Uses `scatter_` /
    `masked_fill` instead of boolean advanced indexing (`x[indices[remove]] = -inf`)
    and omits per-call host-synchronizing `.all()`/`.sum()` assertions.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for sampling")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    squeeze = logits.ndim == 1
    x = logits.detach().to(dtype) / float(temperature)
    if squeeze:
        x = x.unsqueeze(0)

    if top_k > 0:
        k = min(int(top_k), x.shape[-1])
        threshold = torch.topk(x, k, dim=-1).values[..., -1:]
        x = x.masked_fill(x < threshold, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(x, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Retain the smallest descending prefix whose cumulative mass reaches top_p.
        remove_sorted = (cumulative - sorted_probs) >= float(top_p)
        remove_sorted[..., 0] = False
        remove_mask = torch.zeros_like(remove_sorted)
        remove_mask.scatter_(dim=-1, index=sorted_indices, src=remove_sorted)
        x = x.masked_fill(remove_mask, float("-inf"))

    probs = torch.softmax(x, dim=-1)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs.squeeze(0) if squeeze else probs


def next_power_of_two_bucket(length: int, *, min_bucket: int = 512, headroom: int = 512) -> int:
    """Choose a static cache capacity bucket with decoding headroom."""
    needed = max(min_bucket, int(length) + int(headroom))
    bucket = min_bucket
    while bucket < needed:
        bucket *= 2
    return bucket


@dataclass
class RunnerSetupTimings:
    mode: str
    max_cache_len: int
    cache_init_s: float = 0.0
    compile_s: float = 0.0
    warmup_s: float = 0.0
    graph_capture_s: float = 0.0


class StaticDraftRunner:
    """Single-batch draft model runner backed by Hugging Face `StaticCache`.

    Strictly separates one-time setup (`cache_init_s`, `compile_s`, `warmup_s`,
    `graph_capture_s`) from steady-state `prefill()` and `step()` latencies.
    """

    VALID_MODES = (
        "eager_static",
        "compile_default",
        "compile_reduce_overhead",
        "cudagraph_static",
    )

    FLEX_BLOCK_SIZE = 128

    def __init__(
        self,
        model: Any,
        *,
        max_cache_len: int,
        mode: str = "eager_static",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_flex_block_mask: bool | None = None,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported mode={mode!r}; expected one of {self.VALID_MODES}")

        from transformers.cache_utils import StaticCache

        self.model = model
        self.mode = mode
        if mode in {"compile_default", "compile_reduce_overhead", "cudagraph_static"}:
            params = list(model.parameters()) if hasattr(model, "parameters") else []
            if len({p.device for p in params}) <= 1:
                try:
                    from accelerate.hooks import remove_hook_from_module

                    remove_hook_from_module(self.model, recurse=True)
                except ImportError:
                    pass
        self.max_cache_len = int(max_cache_len)
        self.device = device or self._infer_device(model)
        self.dtype = dtype or self._infer_dtype(model)
        self._supports_l2k = _supports_logits_to_keep(model)
        attn_impl = getattr(model.config, "_attn_implementation", None)
        self.use_flex_block_mask = (
            bool(use_flex_block_mask)
            if use_flex_block_mask is not None
            else (attn_impl == "flex_attention")
        )
        if self.use_flex_block_mask and (self.max_cache_len % self.FLEX_BLOCK_SIZE != 0):
            raise ValueError(
                f"max_cache_len={self.max_cache_len} must be divisible by {self.FLEX_BLOCK_SIZE} for flex_attention BlockMask"
            )
        self.cur_len = 0
        self.pos_offset = 0
        self.setup_timings = RunnerSetupTimings(mode=mode, max_cache_len=self.max_cache_len)

        t0 = time.perf_counter()
        self.cache = StaticCache(config=model.config, max_cache_len=self.max_cache_len)
        num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        head_dim = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
        self.cache.early_initialization(
            batch_size=1,
            num_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self._sync()
        self.setup_timings.cache_init_s = time.perf_counter() - t0

        # Static single-token buffers reused across steps (required for CUDA graphs)
        self._static_input_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self._static_cache_position = torch.zeros((1,), dtype=torch.long, device=self.device)
        self._static_position_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self._kv_arange = torch.arange(self.max_cache_len, dtype=torch.long, device=self.device).view(1, 1, 1, -1)
        self._mask_zero = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        self._mask_neg = torch.tensor(torch.finfo(self.dtype).min, dtype=self.dtype, device=self.device)
        self._flex_kv_num_blocks: torch.Tensor | None = None
        self._flex_decode_bm: Any = None
        if self.use_flex_block_mask:
            from torch.nn.attention.flex_attention import BlockMask

            num_kv_blocks = self.max_cache_len // self.FLEX_BLOCK_SIZE
            self._flex_kv_num_blocks = torch.ones((1, 1, 1), dtype=torch.int32, device=self.device)
            kv_indices = torch.arange(num_kv_blocks, dtype=torch.int32, device=self.device).view(1, 1, 1, -1).contiguous()
            # For Q_LEN=1 < BLOCK_SIZE=128, full_kv_num_blocks MUST remain 0; all active KV blocks live in kv_num_blocks.
            full_kv_num_blocks = torch.zeros((1, 1, 1), dtype=torch.int32, device=self.device)
            full_kv_indices = kv_indices.clone()
            cpos_ref = self._static_cache_position

            def _decode_mask_mod(b, h, q_idx, kv_idx):
                return kv_idx <= cpos_ref[0]

            self._flex_decode_bm = BlockMask.from_kv_blocks(
                kv_num_blocks=self._flex_kv_num_blocks,
                kv_indices=kv_indices,
                full_kv_num_blocks=full_kv_num_blocks,
                full_kv_indices=full_kv_indices,
                BLOCK_SIZE=self.FLEX_BLOCK_SIZE,
                mask_mod=_decode_mask_mod,
                seq_lengths=(1, self.max_cache_len),
            )
        self._static_logits_out: torch.Tensor | None = None
        self._graph: Any = None
        self._compiled_step: Any = None
        self._is_prepared = False

    @staticmethod
    def _infer_device(model: Any) -> torch.device:
        if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
            return model.get_input_embeddings().weight.device
        for p in model.parameters():
            return p.device
        return torch.device("cpu")

    @staticmethod
    def _infer_dtype(model: Any) -> torch.dtype:
        if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
            return model.get_input_embeddings().weight.dtype
        for p in model.parameters():
            return p.dtype
        return torch.float32

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _update_step_pos(self, pos: int) -> None:
        self._static_cache_position.fill_(pos)
        self._static_position_ids.fill_(pos + self.pos_offset)
        if self._flex_kv_num_blocks is not None:
            self._flex_kv_num_blocks.fill_((pos // self.FLEX_BLOCK_SIZE) + 1)

    def _raw_step(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_flex_block_mask:
            attn_mask: Any = self._flex_decode_bm
        else:
            attn_mask = torch.where(
                self._kv_arange <= cache_position.view(1, 1, 1, 1),
                self._mask_zero,
                self._mask_neg,
            )
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "past_key_values": self.cache,
            "cache_position": cache_position,
            "position_ids": position_ids,
            "use_cache": True,
        }
        if self._supports_l2k:
            kwargs["logits_to_keep"] = 1
        out = self.model(**kwargs)
        return out.logits[0, -1]

    def _mark_cudagraph_step_begin(self) -> None:
        if (
            self.mode == "compile_reduce_overhead"
            and hasattr(torch, "compiler")
            and hasattr(torch.compiler, "cudagraph_mark_step_begin")
        ):
            torch.compiler.cudagraph_mark_step_begin()

    def prepare(self, *, warmup_steps: int = 3) -> RunnerSetupTimings:
        """Run one-time compilation, warmup, and/or CUDA graph capture.

        Resets `StaticCache` afterward so caller state starts completely clean.
        """
        if self._is_prepared:
            return self.setup_timings

        self.cache.reset()
        self._set_static_cache_len(0)
        self._static_input_ids.fill_(0)
        self._update_step_pos(0)

        if self.mode in {"compile_default", "compile_reduce_overhead"}:
            compile_mode = "default" if self.mode == "compile_default" else "reduce-overhead"
            t_c0 = time.perf_counter()
            self._compiled_step = torch.compile(self._raw_step, mode=compile_mode, fullgraph=False)
            # Trigger actual compilation on the first traced call
            with torch.inference_mode():
                self._mark_cudagraph_step_begin()
                _ = self._compiled_step(
                    self._static_input_ids,
                    self._static_cache_position,
                    self._static_position_ids,
                )
            self._sync()
            self.setup_timings.compile_s = time.perf_counter() - t_c0

            t_w0 = time.perf_counter()
            with torch.inference_mode():
                for i in range(1, max(2, warmup_steps) + 1):
                    self._update_step_pos(i)
                    self._mark_cudagraph_step_begin()
                    _ = self._compiled_step(
                        self._static_input_ids,
                        self._static_cache_position,
                        self._static_position_ids,
                    )
            self._sync()
            self.setup_timings.warmup_s = time.perf_counter() - t_w0

        elif self.mode == "cudagraph_static":
            if self.device.type != "cuda":
                raise RuntimeError("cudagraph_static requires a CUDA device")
            t_w0 = time.perf_counter()
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                with torch.inference_mode():
                    for i in range(max(3, warmup_steps)):
                        self._update_step_pos(i)
                        _ = self._raw_step(
                            self._static_input_ids,
                            self._static_cache_position,
                            self._static_position_ids,
                        )
            torch.cuda.current_stream(self.device).wait_stream(stream)
            self._sync()
            self.setup_timings.warmup_s = time.perf_counter() - t_w0

            t_g0 = time.perf_counter()
            self._graph = torch.cuda.CUDAGraph()
            self._update_step_pos(max(3, warmup_steps))
            with torch.inference_mode():
                with torch.cuda.graph(self._graph):
                    self._static_logits_out = self._raw_step(
                        self._static_input_ids,
                        self._static_cache_position,
                        self._static_position_ids,
                    )
            self._sync()
            self.setup_timings.graph_capture_s = time.perf_counter() - t_g0

        else:
            # eager_static warmup
            t_w0 = time.perf_counter()
            with torch.inference_mode():
                for i in range(max(2, warmup_steps)):
                    self._update_step_pos(i)
                    _ = self._raw_step(
                        self._static_input_ids,
                        self._static_cache_position,
                        self._static_position_ids,
                    )
            self._sync()
            self.setup_timings.warmup_s = time.perf_counter() - t_w0

        # Reset cache so real prefill begins with clean state
        self.cache.reset()
        self._set_static_cache_len(0)
        self.cur_len = 0
        self.pos_offset = 0
        self._is_prepared = True
        return self.setup_timings

    @staticmethod
    def _make_fast_causal_prefill_block_mask(p_len: int, cap: int, device: torch.device, block_size: int = 128) -> Any:
        """Zero-compile analytical BlockMask for causal prefill into a StaticCache of length `cap`."""
        from torch.nn.attention.flex_attention import BlockMask

        num_q_blocks = (p_len + block_size - 1) // block_size
        num_kv_blocks = cap // block_size
        q_idx_arr = torch.arange(num_q_blocks, dtype=torch.int32, device=device)
        kv_num_blocks = (q_idx_arr + 1).view(1, 1, num_q_blocks).contiguous()
        kv_indices = (
            torch.arange(num_kv_blocks, dtype=torch.int32, device=device)
            .view(1, 1, 1, num_kv_blocks)
            .expand(1, 1, num_q_blocks, num_kv_blocks)
            .contiguous()
        )
        full_kv_num_blocks = torch.zeros((1, 1, num_q_blocks), dtype=torch.int32, device=device)
        full_kv_indices = kv_indices.clone()
        return BlockMask.from_kv_blocks(
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=full_kv_num_blocks,
            full_kv_indices=full_kv_indices,
            BLOCK_SIZE=block_size,
            mask_mod=lambda b, h, q_idx, kv_idx: kv_idx <= q_idx,
            seq_lengths=(p_len, cap),
        )

    @property
    def static_kv_capacity_bytes(self) -> int:
        from .cache_utils import get_cache_memory_bytes

        return get_cache_memory_bytes(self.cache)

    def active_kv_bytes(self, seq_len: int | None = None) -> int:
        """Active KV-cache footprint proportional to `seq_len` (comparable to DynamicCache)."""
        used = self.cur_len if seq_len is None else int(seq_len)
        cap = max(1, self.max_cache_len)
        return int(round(self.static_kv_capacity_bytes * (min(used, cap) / cap)))

    def prefill(self, prompt_ids: list[int]) -> torch.Tensor:
        """Prefill `StaticCache` with `prompt_ids` and return float32 next-token logits."""
        p_len = len(prompt_ids)
        if p_len == 0:
            raise ValueError("prompt_ids must be non-empty")
        if p_len >= self.max_cache_len:
            raise ValueError(f"prompt length {p_len} exceeds max_cache_len={self.max_cache_len}")

        self.cache.reset()
        self._set_static_cache_len(0)
        self.pos_offset = 0
        inp = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        cache_pos = torch.arange(p_len, dtype=torch.long, device=self.device)
        pos_ids = cache_pos.unsqueeze(0)
        kwargs: dict[str, Any] = {
            "input_ids": inp,
            "past_key_values": self.cache,
            "cache_position": cache_pos,
            "position_ids": pos_ids,
            "use_cache": True,
        }
        if self.use_flex_block_mask:
            kwargs["attention_mask"] = self._make_fast_causal_prefill_block_mask(
                p_len, self.max_cache_len, self.device, block_size=self.FLEX_BLOCK_SIZE
            )
        if self._supports_l2k:
            kwargs["logits_to_keep"] = 1
        with torch.inference_mode():
            out = self.model(**kwargs)
        logits = out.logits[0, -1].float()
        del out, inp, cache_pos, pos_ids
        self.cur_len = p_len
        return logits

    def prefill_kv_sparse(
        self,
        prompt_ids: list[int],
        *,
        kv_budget: int,
        sink_tokens: int = 64,
        recent_tokens: int = 128,
        obs_window: int = 16,
        pool_kernel: int = 5,
    ) -> tuple[torch.Tensor, float, float, int]:
        """SnapKV/TriForce-style attention-evicted draft KV prefill into fixed StaticCache.

        Preserves original RoPE positions via `self.pos_offset = len(prompt_ids) - kept_len`.
        Returns `(next_logits, dyn_prefill_s, eviction_s, kept_len)`.
        """
        from ..adaptive.target_attention_selector import TargetPrefillAttentionHook, extract_layer_kv

        p_len = len(prompt_ids)
        if p_len == 0:
            raise ValueError("prompt_ids must be non-empty")
        if p_len <= kv_budget:
            self._sync()
            t0 = time.perf_counter()
            logits = self.prefill(prompt_ids)
            self._sync()
            return logits, time.perf_counter() - t0, 0.0, p_len

        effective_budget = int(kv_budget)
        if effective_budget >= self.max_cache_len:
            raise ValueError(f"kv_budget={effective_budget} exceeds max_cache_len={self.max_cache_len}")

        sink_t = int(sink_tokens)
        recent_t = int(recent_tokens)
        if sink_t + recent_t >= effective_budget:
            sink_t = max(1, effective_budget // 4)
            recent_t = max(1, effective_budget // 4)
        n_mid = effective_budget - sink_t - recent_t
        if n_mid <= 0:
            raise ValueError(f"kv_budget={effective_budget} leaves no room for middle attention selection")

        backbone = getattr(self.model, "model", self.model)
        layers = getattr(backbone, "layers", None)
        if layers is None:
            raise ValueError("Draft model does not expose `.model.layers` for KV-sparse eviction")
        n_layers = len(layers)
        all_layers = list(range(n_layers))

        self.cache.reset()
        self._set_static_cache_len(0)
        inp = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        pre_kwargs: dict[str, Any] = {"input_ids": inp, "use_cache": True}
        if self._supports_l2k:
            pre_kwargs["logits_to_keep"] = 1

        self._sync()
        t_pre0 = time.perf_counter()
        with TargetPrefillAttentionHook(self.model, layer_indices=all_layers, tail_window=obs_window) as hook:
            with torch.inference_mode():
                out = self.model(**pre_kwargs)
            self._sync()
            dyn_prefill_s = time.perf_counter() - t_pre0

            next_logits = out.logits[0, -1].float()
            dyn_cache = out.past_key_values
            del out, inp

            t_ev0 = time.perf_counter()
            mid_start = sink_t
            mid_end = p_len - recent_t
            mid_len = mid_end - mid_start
            sink_idx = torch.arange(0, sink_t, device=self.device)
            recent_idx = torch.arange(mid_end, p_len, device=self.device)

            with torch.inference_mode():
                for l_idx in all_layers:
                    layer_imp = hook.compute_layer_token_importance(l_idx, dyn_cache, head_agg="mean")
                    mid_imp = layer_imp[mid_start:mid_end].view(1, 1, mid_len)
                    if pool_kernel > 1 and mid_len >= pool_kernel:
                        mid_imp = F.max_pool1d(
                            mid_imp, kernel_size=pool_kernel, stride=1, padding=pool_kernel // 2
                        )
                    mid_flat = mid_imp.view(-1)[:mid_len]
                    top_rel = torch.topk(mid_flat, k=n_mid, largest=True, sorted=False).indices + mid_start
                    keep_idx, _ = torch.sort(torch.cat([sink_idx, top_rel, recent_idx], dim=0))

                    k_src, v_src = extract_layer_kv(dyn_cache, l_idx)
                    k_dst, v_dst = extract_layer_kv(self.cache, l_idx)
                    k_dst[:, :, :effective_budget, :].copy_(k_src.index_select(2, keep_idx).to(self.dtype))
                    v_dst[:, :, :effective_budget, :].copy_(v_src.index_select(2, keep_idx).to(self.dtype))

            del dyn_cache
            self._sync()
            eviction_s = time.perf_counter() - t_ev0

        self._set_static_cache_len(effective_budget)
        self.cur_len = effective_budget
        self.pos_offset = p_len - effective_budget
        return next_logits, dyn_prefill_s, eviction_s, effective_budget

    def step(self, token_id: int) -> torch.Tensor:
        """Advance 1 token at `self.cur_len` and return float32 next-token logits."""
        if self.cur_len >= self.max_cache_len:
            raise ValueError(f"StaticCache overflow: cur_len={self.cur_len} >= max_cache_len={self.max_cache_len}")
        pos = self.cur_len
        self._static_input_ids[0, 0] = int(token_id)
        self._update_step_pos(pos)

        with torch.inference_mode():
            if self.mode == "cudagraph_static":
                assert self._graph is not None and self._static_logits_out is not None
                self._graph.replay()
                logits = self._static_logits_out.float().clone()
            elif self._compiled_step is not None:
                self._mark_cudagraph_step_begin()
                logits = self._compiled_step(
                    self._static_input_ids,
                    self._static_cache_position,
                    self._static_position_ids,
                ).float().clone()
            else:
                logits = self._raw_step(
                    self._static_input_ids,
                    self._static_cache_position,
                    self._static_position_ids,
                ).float()

        self.cur_len = pos + 1
        return logits

    def step_many(self, token_ids: list[int]) -> torch.Tensor:
        """Advance 1 or more tokens sequentially (reuses fixed-shape compiled/graphed step)."""
        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        logits = None
        for tok in token_ids:
            logits = self.step(tok)
        assert logits is not None
        return logits

    def _set_static_cache_len(self, target_len: int) -> None:
        layers = getattr(self.cache, "layers", None)
        if isinstance(layers, (list, tuple)):
            for layer in layers:
                cl = getattr(layer, "cumulative_length", None)
                if isinstance(cl, torch.Tensor):
                    cl.fill_(int(target_len))
                elif isinstance(cl, int):
                    layer.cumulative_length = int(target_len)

    def rollback(self, target_len: int) -> None:
        """Roll back logical KV-cache pointer to `target_len` (zero tensor reallocation).

        Updates `layer.cumulative_length` in-place (preserving static tensor addresses
        for CUDA Graphs) and relies on the explicit 4D causal mask in `_raw_step` so
        rolled-back positions strictly greater than `cache_position` are never attended.
        """
        if target_len < 0 or target_len > self.cur_len:
            raise ValueError(f"Invalid rollback target_len={target_len} for cur_len={self.cur_len}")
        self._set_static_cache_len(int(target_len))
        self.cur_len = int(target_len)
