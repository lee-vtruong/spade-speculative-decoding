"""KV-cache management and rollback utilities for speculative decoding.

Supports both modern Transformers DynamicCache and legacy tuple-of-tensors
cache representations, ensuring safe rollback upon speculative rejection.
"""
from __future__ import annotations
from typing import Any
import torch

def get_cache_seq_length(past_key_values: Any) -> int:
    """Return the sequence length stored in past_key_values, or 0 if empty/None."""
    if past_key_values is None:
        return 0
    # Transformers DynamicCache
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length())
        except Exception:
            pass
    if hasattr(past_key_values, "key_cache") and past_key_values.key_cache:
        try:
            first_key = past_key_values.key_cache[0]
            if isinstance(first_key, torch.Tensor) and first_key.ndim >= 3:
                return first_key.shape[-2]
        except Exception:
            pass
    # Tuple of layers: ((k0, v0), (k1, v1), ...)
    if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
        first_layer = past_key_values[0]
        if isinstance(first_layer, (tuple, list)) and len(first_layer) > 0:
            k = first_layer[0]
            if isinstance(k, torch.Tensor) and k.ndim >= 3:
                return k.shape[-2]
        elif isinstance(first_layer, torch.Tensor) and first_layer.ndim >= 3:
            return first_layer.shape[-2]
    return 0

def crop_past_key_values(past_key_values: Any, max_length: int) -> Any:
    """Safely crop key-value cache to max_length, returning updated cache.

    Handles:
    1. DynamicCache (.crop(max_length) or key_cache/value_cache truncation)
    2. Tuple of (key, value) tensors per layer
    """
    if past_key_values is None or max_length < 0:
        return past_key_values

    # Transformers DynamicCache (handles <=5.14 positive max_length and >=5.20 negative tokens_to_remove)
    if hasattr(past_key_values, "crop"):
        cur_len = get_cache_seq_length(past_key_values)
        if cur_len > max_length:
            try:
                past_key_values.crop(max_length)
                if get_cache_seq_length(past_key_values) == max_length:
                    return past_key_values
            except Exception:
                pass
            try:
                past_key_values.crop(-(cur_len - max_length))
                if get_cache_seq_length(past_key_values) == max_length:
                    return past_key_values
            except Exception:
                pass
        else:
            return past_key_values

    # Transformers 5.x DynamicCache.layers direct fallback
    layers = getattr(past_key_values, "layers", None)
    if isinstance(layers, (list, tuple)) and layers:
        for layer in layers:
            if hasattr(layer, "keys") and isinstance(layer.keys, torch.Tensor) and layer.keys.shape[-2] > max_length:
                layer.keys = layer.keys[..., :max_length, :]
            if hasattr(layer, "values") and isinstance(layer.values, torch.Tensor) and layer.values.shape[-2] > max_length:
                layer.values = layer.values[..., :max_length, :]
        return past_key_values

    # DynamicCache with direct key_cache/value_cache list access
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        try:
            for i in range(len(past_key_values.key_cache)):
                k = past_key_values.key_cache[i]
                v = past_key_values.value_cache[i]
                if isinstance(k, torch.Tensor) and k.shape[-2] > max_length:
                    past_key_values.key_cache[i] = k[..., :max_length, :]
                if isinstance(v, torch.Tensor) and v.shape[-2] > max_length:
                    past_key_values.value_cache[i] = v[..., :max_length, :]
            return past_key_values
        except Exception:
            pass

    # Tuple of tuples: ((k0, v0), (k1, v1), ...)
    if isinstance(past_key_values, tuple):
        cropped_layers = []
        for layer in past_key_values:
            if isinstance(layer, tuple) and len(layer) >= 2:
                k, v = layer[0], layer[1]
                if isinstance(k, torch.Tensor) and k.shape[-2] > max_length:
                    k = k[..., :max_length, :]
                if isinstance(v, torch.Tensor) and v.shape[-2] > max_length:
                    v = v[..., :max_length, :]
                cropped_layers.append((k, v) + layer[2:])
            else:
                cropped_layers.append(layer)
        return tuple(cropped_layers)

    return past_key_values

def clone_past_key_values(past_key_values: Any) -> Any:
    """Create a detached copy of past_key_values for speculative branching."""
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        # Create a new DynamicCache instance if possible, or clone internal lists
        cls = past_key_values.__class__
        try:
            new_cache = cls()
            new_cache.key_cache = [k.clone() for k in past_key_values.key_cache]
            new_cache.value_cache = [v.clone() for v in past_key_values.value_cache]
            return new_cache
        except Exception:
            pass
    if isinstance(past_key_values, tuple):
        return tuple(
            tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in layer)
            if isinstance(layer, tuple) else layer
            for layer in past_key_values
        )
    return past_key_values

def build_position_ids(start_pos: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Generate 2D position_ids tensor [1, seq_len] from start_pos."""
    return torch.arange(start_pos, start_pos + seq_len, dtype=torch.long, device=device).unsqueeze(0)

def build_causal_attention_mask(total_length: int, num_new_tokens: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generate attention mask for incremental step or verification block."""
    # When using past_key_values, total_length is cache_len + num_new_tokens
    # A 2D mask of ones [1, total_length] is standard in Hugging Face Transformers
    return torch.ones((1, total_length), dtype=dtype, device=device)

class CacheManager:
    """Stateful helper for tracking KV cache and position alignment during speculative rounds."""

    def __init__(self, past_key_values: Any = None):
        self.past_key_values = past_key_values

    @property
    def seq_length(self) -> int:
        return get_cache_seq_length(self.past_key_values)

    def update(self, past_key_values: Any) -> None:
        self.past_key_values = past_key_values

    def reset(self) -> None:
        self.past_key_values = None

    def rollback(self, target_length: int) -> None:
        """Crop cache back to target_length."""
        if target_length < self.seq_length:
            self.past_key_values = crop_past_key_values(self.past_key_values, target_length)

    def position_ids_for(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        return build_position_ids(self.seq_length, num_tokens, device)

    @property
    def memory_bytes(self) -> int:
        return get_cache_memory_bytes(self.past_key_values)


def get_cache_memory_bytes(past_key_values: Any) -> int:
    """Return total byte footprint of all key/value tensors stored in past_key_values."""
    if past_key_values is None:
        return 0
    total = 0
    # Transformers 5.x DynamicCache.layers -> layer.keys / layer.values
    layers = getattr(past_key_values, "layers", None)
    if isinstance(layers, (list, tuple)) and layers:
        for layer in layers:
            for attr in ("keys", "values"):
                t = getattr(layer, attr, None)
                if isinstance(t, torch.Tensor):
                    total += int(t.numel() * t.element_size())
        if total > 0:
            return total
    # DynamicCache.key_cache / value_cache
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for t in list(past_key_values.key_cache) + list(past_key_values.value_cache):
            if isinstance(t, torch.Tensor):
                total += int(t.numel() * t.element_size())
        return total
    # Legacy tuple-of-tuples ((k0, v0), ...)
    if isinstance(past_key_values, (tuple, list)):
        for layer in past_key_values:
            if isinstance(layer, (tuple, list)):
                for t in layer:
                    if isinstance(t, torch.Tensor):
                        total += int(t.numel() * t.element_size())
            elif isinstance(layer, torch.Tensor):
                total += int(layer.numel() * layer.element_size())
    return total
