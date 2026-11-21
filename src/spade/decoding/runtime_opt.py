"""Step B1.1 & v9 runtime optimization primitives for speculative drafting, target AR/verify, & sampling.

Provides:
1. `processed_distribution_fast`: GPU-friendly, zero-CPU-sync temperature/top-k/top-p
   distribution processing in float32 (supporting 1D [V] or 2D [B, V] logits) that
   preserves the exact nucleus cutoff semantics of `sampling.processed_distribution`.
2. `StaticDraftRunner`: Fixed-shape `StaticCache` wrapper for 1-token draft decoding.
3. `StaticTargetRunner`: Fixed-shape `StaticCache` wrapper for compiled 1-token Target AR
   (`step`) and `(K+1)`-token parallel target verification (`verify`) with $O(1)$ rollback,
   supporting shared-pool explicit CUDA Graphs (`compile_cudagraph`).
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

FLEX_BLOCK_SIZE = 128

VALID_RUNNER_MODES = (
    "eager_static",
    "compile_default",
    "compile_reduce_overhead",
    "cudagraph_static",
    "compile_cudagraph",
)


def _supports_logits_to_keep(model: Any) -> bool:
    try:
        forward = getattr(model, "forward", model.__call__)
        return "logits_to_keep" in inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False


def _strip_accelerate_hooks_if_single_device(model: Any) -> None:
    params = list(model.parameters()) if hasattr(model, "parameters") else []
    if len({p.device for p in params}) <= 1:
        try:
            from accelerate.hooks import remove_hook_from_module

            remove_hook_from_module(model, recurse=True)
        except ImportError:
            pass


def _infer_model_device(model: Any) -> torch.device:
    if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
        return model.get_input_embeddings().weight.device
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def _infer_model_dtype(model: Any) -> torch.dtype:
    if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
        return model.get_input_embeddings().weight.dtype
    for p in model.parameters():
        return p.dtype
    return torch.float32


def _set_static_cache_len(cache: Any, target_len: int) -> None:
    layers = getattr(cache, "layers", None)
    if isinstance(layers, (list, tuple)):
        for layer in layers:
            cl = getattr(layer, "cumulative_length", None)
            if isinstance(cl, torch.Tensor):
                cl.fill_(int(target_len))
            elif isinstance(cl, int):
                layer.cumulative_length = int(target_len)


def _make_fast_causal_prefill_block_mask(
    p_len: int, cap: int, device: torch.device, block_size: int = FLEX_BLOCK_SIZE
) -> Any:
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


def _compile_and_capture_cudagraph(
    raw_fn: Callable[..., torch.Tensor],
    static_args: tuple[torch.Tensor, ...],
    prepare_pos_fn: Callable[[int], None],
    sync_fn: Callable[[], None],
    device: torch.device,
    *,
    compiled: bool,
    graph_pool: Any = None,
    warmup_steps: int = 3,
    reset_cache_fn: Callable[[], None] | None = None,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, float, float, float]:
    """Compile (optional) on default stream, warm up on side stream, and capture CUDA graph."""
    if device.type != "cuda":
        raise RuntimeError("CUDA graph capture requires a CUDA device")

    run_fn = torch.compile(raw_fn, mode="default", fullgraph=False) if compiled else raw_fn
    compile_s = 0.0
    if compiled:
        if reset_cache_fn is not None:
            reset_cache_fn()
        t_c0 = time.perf_counter()
        prepare_pos_fn(0)
        with torch.inference_mode():
            _ = run_fn(*static_args)
        sync_fn()
        compile_s = time.perf_counter() - t_c0

    if reset_cache_fn is not None:
        reset_cache_fn()
    t_w0 = time.perf_counter()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    n_warm = max(3, int(warmup_steps))
    with torch.cuda.stream(stream):
        with torch.inference_mode():
            for i in range(n_warm):
                prepare_pos_fn(i)
                _ = run_fn(*static_args)
    torch.cuda.current_stream(device).wait_stream(stream)
    sync_fn()
    warmup_s = time.perf_counter() - t_w0

    if reset_cache_fn is not None:
        reset_cache_fn()
    t_g0 = time.perf_counter()
    graph = torch.cuda.CUDAGraph()
    prepare_pos_fn(n_warm if reset_cache_fn is None else 0)
    capture_kwargs: dict[str, Any] = {}
    if graph_pool is not None:
        capture_kwargs["pool"] = graph_pool
    with torch.inference_mode():
        with torch.cuda.graph(graph, **capture_kwargs):
            static_out = run_fn(*static_args)
    sync_fn()
    graph_capture_s = time.perf_counter() - t_g0
    return graph, static_out, compile_s, warmup_s, graph_capture_s


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


class _BaseStaticCacheRunner:
    """Shared base for fixed-shape `StaticCache` runners with O(1) pointer rollback."""

    VALID_MODES = VALID_RUNNER_MODES
    FLEX_BLOCK_SIZE = FLEX_BLOCK_SIZE
    _make_fast_causal_prefill_block_mask = staticmethod(_make_fast_causal_prefill_block_mask)
    _infer_device = staticmethod(_infer_model_device)
    _infer_dtype = staticmethod(_infer_model_dtype)

    def __init__(
        self,
        model: Any,
        *,
        max_cache_len: int,
        mode: str = "eager_static",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_flex_block_mask: bool | None = None,
        graph_pool: Any = None,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported mode={mode!r}; expected one of {self.VALID_MODES}")

        from transformers.cache_utils import StaticCache

        self.model = model
        self.mode = mode
        if mode in {"compile_default", "compile_reduce_overhead", "cudagraph_static", "compile_cudagraph"}:
            _strip_accelerate_hooks_if_single_device(self.model)

        self.max_cache_len = int(max_cache_len)
        self.device = device or _infer_model_device(model)
        self.dtype = dtype or _infer_model_dtype(model)
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
        self.graph_pool = graph_pool
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

        self._kv_arange = torch.arange(self.max_cache_len, dtype=torch.long, device=self.device).view(1, 1, 1, -1)
        self._mask_zero = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        self._mask_neg = torch.tensor(torch.finfo(self.dtype).min, dtype=self.dtype, device=self.device)
        self._is_prepared = False

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _set_static_cache_len(self, target_len: int) -> None:
        _set_static_cache_len(self.cache, int(target_len))

    def _reset_cache_state(self) -> None:
        self.cache.reset()
        self._set_static_cache_len(0)

    def _mark_cudagraph_step_begin(self) -> None:
        if (
            self.mode == "compile_reduce_overhead"
            and hasattr(torch, "compiler")
            and hasattr(torch.compiler, "cudagraph_mark_step_begin")
        ):
            torch.compiler.cudagraph_mark_step_begin()

    def _build_flex_decode_mask(self, cpos_ref: torch.Tensor, kv_num_blocks: torch.Tensor, q_len: int) -> Any:
        from torch.nn.attention.flex_attention import BlockMask

        num_kv_blocks = self.max_cache_len // self.FLEX_BLOCK_SIZE
        kv_indices = (
            torch.arange(num_kv_blocks, dtype=torch.int32, device=self.device)
            .view(1, 1, 1, -1)
            .contiguous()
        )
        full_kv_num_blocks = torch.zeros((1, 1, 1), dtype=torch.int32, device=self.device)
        full_kv_indices = kv_indices.clone()
        mask_mod = (
            (lambda b, h, q_idx, kv_idx: kv_idx <= cpos_ref[0])
            if q_len == 1
            else (lambda b, h, q_idx, kv_idx: kv_idx <= cpos_ref[q_idx])
        )
        return BlockMask.from_kv_blocks(
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=full_kv_num_blocks,
            full_kv_indices=full_kv_indices,
            BLOCK_SIZE=self.FLEX_BLOCK_SIZE,
            mask_mod=mask_mod,
            seq_lengths=(q_len, self.max_cache_len),
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

        self._reset_cache_state()
        self.pos_offset = 0

        # Guard PyTorch flex_attention Inductor NoValidChoicesError for 64 < q_len < 128
        # by right-padding prefill to FLEX_BLOCK_SIZE when room permits, then resetting length to p_len.
        needs_pad = self.use_flex_block_mask and p_len < self.FLEX_BLOCK_SIZE and self.FLEX_BLOCK_SIZE <= self.max_cache_len
        if needs_pad:
            padded_ids = list(prompt_ids) + [0] * (self.FLEX_BLOCK_SIZE - p_len)
            run_len = self.FLEX_BLOCK_SIZE
        else:
            padded_ids = prompt_ids
            run_len = p_len

        inp = torch.tensor([padded_ids], dtype=torch.long, device=self.device)
        cache_pos = torch.arange(run_len, dtype=torch.long, device=self.device)
        pos_ids = cache_pos.unsqueeze(0)
        kwargs: dict[str, Any] = {
            "input_ids": inp,
            "past_key_values": self.cache,
            "cache_position": cache_pos,
            "position_ids": pos_ids,
            "use_cache": True,
        }
        if self.use_flex_block_mask:
            kwargs["attention_mask"] = _make_fast_causal_prefill_block_mask(
                run_len, self.max_cache_len, self.device, block_size=self.FLEX_BLOCK_SIZE
            )
        if self._supports_l2k and not needs_pad:
            kwargs["logits_to_keep"] = 1
        with torch.inference_mode():
            out = self.model(**kwargs)
        logits = out.logits[0, p_len - 1].float() if needs_pad else out.logits[0, -1].float()
        del out, inp, cache_pos, pos_ids
        if needs_pad:
            self._set_static_cache_len(p_len)
        self.cur_len = p_len
        return logits

    def rollback(self, target_len: int) -> None:
        """Roll back logical KV-cache pointer to `target_len` (zero tensor reallocation)."""
        if target_len < 0 or target_len > self.cur_len:
            raise ValueError(f"Invalid rollback target_len={target_len} for cur_len={self.cur_len}")
        self._set_static_cache_len(int(target_len))
        self.cur_len = int(target_len)


class StaticDraftRunner(_BaseStaticCacheRunner):
    """Single-batch draft model runner backed by Hugging Face `StaticCache`.

    Strictly separates one-time setup (`cache_init_s`, `compile_s`, `warmup_s`,
    `graph_capture_s`) from steady-state `prefill()` and `step()` latencies.
    """

    def __init__(
        self,
        model: Any,
        *,
        max_cache_len: int,
        mode: str = "eager_static",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_flex_block_mask: bool | None = None,
        graph_pool: Any = None,
    ) -> None:
        super().__init__(
            model,
            max_cache_len=max_cache_len,
            mode=mode,
            device=device,
            dtype=dtype,
            use_flex_block_mask=use_flex_block_mask,
            graph_pool=graph_pool,
        )
        # Static single-token buffers reused across steps (required for CUDA graphs)
        self._static_input_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self._static_cache_position = torch.zeros((1,), dtype=torch.long, device=self.device)
        self._static_position_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self._flex_kv_num_blocks: torch.Tensor | None = None
        self._flex_decode_bm: Any = None
        if self.use_flex_block_mask:
            self._flex_kv_num_blocks = torch.ones((1, 1, 1), dtype=torch.int32, device=self.device)
            self._flex_decode_bm = self._build_flex_decode_mask(
                self._static_cache_position, self._flex_kv_num_blocks, q_len=1
            )
        self._static_logits_out: torch.Tensor | None = None
        self._graph: Any = None
        self._compiled_step: Any = None

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

    def prepare(self, *, warmup_steps: int = 3) -> RunnerSetupTimings:
        """Run one-time compilation, warmup, and/or CUDA graph capture.

        Resets `StaticCache` afterward so caller state starts completely clean.
        """
        if self._is_prepared:
            return self.setup_timings

        self._reset_cache_state()
        self._static_input_ids.fill_(0)
        self._update_step_pos(0)

        if self.mode in {"compile_default", "compile_reduce_overhead"}:
            compile_mode = "default" if self.mode == "compile_default" else "reduce-overhead"
            t_c0 = time.perf_counter()
            self._compiled_step = torch.compile(self._raw_step, mode=compile_mode, fullgraph=False)
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

        elif self.mode in {"cudagraph_static", "compile_cudagraph"}:
            compiled = self.mode == "compile_cudagraph"
            pool = self.graph_pool
            if pool is None and self.device.type == "cuda":
                pool = torch.cuda.graph_pool_handle()
                self.graph_pool = pool
            self._graph, self._static_logits_out, cs, ws, gs = _compile_and_capture_cudagraph(
                self._raw_step,
                (self._static_input_ids, self._static_cache_position, self._static_position_ids),
                self._update_step_pos,
                self._sync,
                self.device,
                compiled=compiled,
                graph_pool=pool,
                warmup_steps=warmup_steps,
                reset_cache_fn=self._reset_cache_state,
            )
            self.setup_timings.compile_s = cs
            self.setup_timings.warmup_s = ws
            self.setup_timings.graph_capture_s = gs

        else:
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

        self._reset_cache_state()
        self.cur_len = 0
        self.pos_offset = 0
        self._is_prepared = True
        return self.setup_timings

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

        self._reset_cache_state()
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
            if self.mode in {"cudagraph_static", "compile_cudagraph"}:
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


class StaticTargetRunner(_BaseStaticCacheRunner):
    """Single-batch target model runner supporting both 1-token AR (`step`) and `(K+1)`-token parallel `verify`."""

    def __init__(
        self,
        model: Any,
        *,
        max_cache_len: int,
        max_verify_len: int = 5,
        mode: str = "compile_cudagraph",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_flex_block_mask: bool | None = None,
        graph_pool: Any = None,
        enable_ar_step: bool = True,
        enable_verify_step: bool = True,
    ) -> None:
        if int(max_verify_len) < 1:
            raise ValueError(f"max_verify_len must be >= 1, got {max_verify_len}")
        super().__init__(
            model,
            max_cache_len=max_cache_len,
            mode=mode,
            device=device,
            dtype=dtype,
            use_flex_block_mask=use_flex_block_mask,
            graph_pool=graph_pool,
        )
        self.max_verify_len = int(max_verify_len)
        if self.use_flex_block_mask and self.max_verify_len > self.FLEX_BLOCK_SIZE:
            raise ValueError(f"max_verify_len={self.max_verify_len} exceeds FLEX_BLOCK_SIZE={self.FLEX_BLOCK_SIZE}")
        self.enable_ar_step = bool(enable_ar_step)
        self.enable_verify_step = bool(enable_verify_step)

        # Static 1-token AR buffers
        self._static_step_input_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self._static_step_cpos = torch.zeros((1,), dtype=torch.long, device=self.device)
        self._static_step_pids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self._flex_step_kv_num_blocks: torch.Tensor | None = None
        self._flex_step_bm: Any = None
        if self.use_flex_block_mask and self.enable_ar_step:
            self._flex_step_kv_num_blocks = torch.ones((1, 1, 1), dtype=torch.int32, device=self.device)
            self._flex_step_bm = self._build_flex_decode_mask(
                self._static_step_cpos, self._flex_step_kv_num_blocks, q_len=1
            )

        # Static parallel-verify buffers (shape [1, max_verify_len])
        self._static_ver_input_ids = torch.zeros((1, self.max_verify_len), dtype=torch.long, device=self.device)
        self._static_ver_cpos = torch.zeros((self.max_verify_len,), dtype=torch.long, device=self.device)
        self._static_ver_pids = torch.zeros((1, self.max_verify_len), dtype=torch.long, device=self.device)
        self._ver_pos_offsets = torch.arange(self.max_verify_len, dtype=torch.long, device=self.device)
        self._ver_host_ids = torch.zeros((self.max_verify_len,), dtype=torch.long)
        self._flex_ver_kv_num_blocks: torch.Tensor | None = None
        self._flex_ver_bm: Any = None
        if self.use_flex_block_mask and self.enable_verify_step:
            self._flex_ver_kv_num_blocks = torch.ones((1, 1, 1), dtype=torch.int32, device=self.device)
            self._flex_ver_bm = self._build_flex_decode_mask(
                self._static_ver_cpos, self._flex_ver_kv_num_blocks, q_len=self.max_verify_len
            )

        self._step_graph: Any = None
        self._step_static_out: torch.Tensor | None = None
        self._compiled_step: Any = None
        self._ver_graph: Any = None
        self._ver_static_out: torch.Tensor | None = None
        self._compiled_ver: Any = None

    def _update_step_pos(self, pos: int) -> None:
        self._static_step_cpos.fill_(pos)
        self._static_step_pids.fill_(pos + self.pos_offset)
        if self._flex_step_kv_num_blocks is not None:
            self._flex_step_kv_num_blocks.fill_((pos // self.FLEX_BLOCK_SIZE) + 1)

    def _update_verify_pos(self, pos: int) -> None:
        self._static_ver_cpos.copy_(self._ver_pos_offsets).add_(pos)
        self._static_ver_pids[0].copy_(self._static_ver_cpos)
        if self.pos_offset:
            self._static_ver_pids.add_(self.pos_offset)
        if self._flex_ver_kv_num_blocks is not None:
            self._flex_ver_kv_num_blocks.fill_(((pos + self.max_verify_len - 1) // self.FLEX_BLOCK_SIZE) + 1)

    def _raw_step(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_flex_block_mask:
            attn_mask: Any = self._flex_step_bm
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
        return self.model(**kwargs).logits[0, -1]

    def _raw_verify(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_flex_block_mask:
            attn_mask: Any = self._flex_ver_bm
        else:
            attn_mask = torch.where(
                self._kv_arange <= cache_position.view(1, 1, self.max_verify_len, 1),
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
        return self.model(**kwargs).logits[0]

    def prepare(self, *, warmup_steps: int = 3) -> RunnerSetupTimings:
        """Prepare enabled `step` and `verify` paths (sharing `self.graph_pool` on CUDA)."""
        if self._is_prepared:
            return self.setup_timings

        self._reset_cache_state()
        self._static_step_input_ids.fill_(0)
        self._static_ver_input_ids.fill_(0)

        if self.mode in {"cudagraph_static", "compile_cudagraph"}:
            compiled = self.mode == "compile_cudagraph"
            if self.graph_pool is None and self.device.type == "cuda":
                self.graph_pool = torch.cuda.graph_pool_handle()

            if self.enable_ar_step:
                self._step_graph, self._step_static_out, cs, ws, gs = _compile_and_capture_cudagraph(
                    self._raw_step,
                    (self._static_step_input_ids, self._static_step_cpos, self._static_step_pids),
                    self._update_step_pos,
                    self._sync,
                    self.device,
                    compiled=compiled,
                    graph_pool=self.graph_pool,
                    warmup_steps=warmup_steps,
                    reset_cache_fn=self._reset_cache_state,
                )
                self.setup_timings.compile_s += cs
                self.setup_timings.warmup_s += ws
                self.setup_timings.graph_capture_s += gs

            if self.enable_verify_step:
                self._ver_graph, self._ver_static_out, cs, ws, gs = _compile_and_capture_cudagraph(
                    self._raw_verify,
                    (self._static_ver_input_ids, self._static_ver_cpos, self._static_ver_pids),
                    lambda i: self._update_verify_pos(i * self.max_verify_len),
                    self._sync,
                    self.device,
                    compiled=compiled,
                    graph_pool=self.graph_pool,
                    warmup_steps=warmup_steps,
                    reset_cache_fn=self._reset_cache_state,
                )
                self.setup_timings.compile_s += cs
                self.setup_timings.warmup_s += ws
                self.setup_timings.graph_capture_s += gs

        elif self.mode in {"compile_default", "compile_reduce_overhead"}:
            compile_mode = "default" if self.mode == "compile_default" else "reduce-overhead"
            n_warm = max(2, int(warmup_steps))
            if self.enable_ar_step:
                self._reset_cache_state()
                t_c0 = time.perf_counter()
                self._compiled_step = torch.compile(self._raw_step, mode=compile_mode, fullgraph=False)
                self._update_step_pos(0)
                with torch.inference_mode():
                    self._mark_cudagraph_step_begin()
                    _ = self._compiled_step(
                        self._static_step_input_ids, self._static_step_cpos, self._static_step_pids
                    )
                self._sync()
                self.setup_timings.compile_s += time.perf_counter() - t_c0
                t_w0 = time.perf_counter()
                with torch.inference_mode():
                    for i in range(1, n_warm + 1):
                        self._update_step_pos(i)
                        self._mark_cudagraph_step_begin()
                        _ = self._compiled_step(
                            self._static_step_input_ids, self._static_step_cpos, self._static_step_pids
                        )
                self._sync()
                self.setup_timings.warmup_s += time.perf_counter() - t_w0

            if self.enable_verify_step:
                self._reset_cache_state()
                t_c0 = time.perf_counter()
                self._compiled_ver = torch.compile(self._raw_verify, mode=compile_mode, fullgraph=False)
                self._update_verify_pos(0)
                with torch.inference_mode():
                    self._mark_cudagraph_step_begin()
                    _ = self._compiled_ver(
                        self._static_ver_input_ids, self._static_ver_cpos, self._static_ver_pids
                    )
                self._sync()
                self.setup_timings.compile_s += time.perf_counter() - t_c0
                t_w0 = time.perf_counter()
                with torch.inference_mode():
                    for i in range(1, n_warm + 1):
                        self._update_verify_pos(i * self.max_verify_len)
                        self._mark_cudagraph_step_begin()
                        _ = self._compiled_ver(
                            self._static_ver_input_ids, self._static_ver_cpos, self._static_ver_pids
                        )
                self._sync()
                self.setup_timings.warmup_s += time.perf_counter() - t_w0

        else:
            # eager_static warmup
            t_w0 = time.perf_counter()
            n_warm = max(2, int(warmup_steps))
            if self.enable_ar_step:
                self._reset_cache_state()
                with torch.inference_mode():
                    for i in range(n_warm):
                        self._update_step_pos(i)
                        _ = self._raw_step(
                            self._static_step_input_ids, self._static_step_cpos, self._static_step_pids
                        )
            if self.enable_verify_step:
                self._reset_cache_state()
                with torch.inference_mode():
                    for i in range(n_warm):
                        self._update_verify_pos(i * self.max_verify_len)
                        _ = self._raw_verify(
                            self._static_ver_input_ids, self._static_ver_cpos, self._static_ver_pids
                        )
            self._sync()
            self.setup_timings.warmup_s = time.perf_counter() - t_w0

        self._reset_cache_state()
        self.cur_len = 0
        self.pos_offset = 0
        self._is_prepared = True
        return self.setup_timings

    def step(self, token_id: int) -> torch.Tensor:
        """Advance 1 autoregressive token at `self.cur_len` and return float32 `[V]` logits."""
        if not self.enable_ar_step:
            raise RuntimeError("StaticTargetRunner was initialized with enable_ar_step=False")
        if self.cur_len >= self.max_cache_len:
            raise ValueError(f"StaticCache overflow: cur_len={self.cur_len} >= max_cache_len={self.max_cache_len}")
        pos = self.cur_len
        self._static_step_input_ids[0, 0] = int(token_id)
        self._update_step_pos(pos)

        with torch.inference_mode():
            if self.mode in {"cudagraph_static", "compile_cudagraph"}:
                assert self._step_graph is not None and self._step_static_out is not None
                self._step_graph.replay()
                logits = self._step_static_out.float().clone()
            elif self._compiled_step is not None:
                self._mark_cudagraph_step_begin()
                logits = self._compiled_step(
                    self._static_step_input_ids,
                    self._static_step_cpos,
                    self._static_step_pids,
                ).float().clone()
            else:
                logits = self._raw_step(
                    self._static_step_input_ids,
                    self._static_step_cpos,
                    self._static_step_pids,
                ).float()

        self.cur_len = pos + 1
        return logits

    def verify(self, token_ids: list[int]) -> torch.Tensor:
        """Verify `m` tokens (`1 <= m <= max_verify_len`) in one static forward pass and return float32 `[m, V]` logits."""
        if not self.enable_verify_step:
            raise RuntimeError("StaticTargetRunner was initialized with enable_verify_step=False")
        m = len(token_ids)
        if m <= 0 or m > self.max_verify_len:
            raise ValueError(f"verify token_ids length {m} must be in [1, {self.max_verify_len}]")
        pos = self.cur_len
        if pos + self.max_verify_len > self.max_cache_len:
            raise ValueError(
                f"StaticCache overflow on verify: pos={pos} + max_verify_len={self.max_verify_len} > max_cache_len={self.max_cache_len}"
            )

        self._ver_host_ids.zero_()
        for i, tid in enumerate(token_ids):
            self._ver_host_ids[i] = int(tid)
        self._static_ver_input_ids[0].copy_(self._ver_host_ids)
        self._update_verify_pos(pos)

        with torch.inference_mode():
            if self.mode in {"cudagraph_static", "compile_cudagraph"}:
                assert self._ver_graph is not None and self._ver_static_out is not None
                self._ver_graph.replay()
                full_logits = self._ver_static_out[:m].float().clone()
            elif self._compiled_ver is not None:
                self._mark_cudagraph_step_begin()
                full_logits = self._compiled_ver(
                    self._static_ver_input_ids,
                    self._static_ver_cpos,
                    self._static_ver_pids,
                )[:m].float().clone()
            else:
                full_logits = self._raw_verify(
                    self._static_ver_input_ids,
                    self._static_ver_cpos,
                    self._static_ver_pids,
                )[:m].float()

        if m < self.max_verify_len:
            self._set_static_cache_len(pos + m)
        self.cur_len = pos + m
        return full_logits
