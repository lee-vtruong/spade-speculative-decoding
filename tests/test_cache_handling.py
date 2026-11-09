"""Tests for KV-cache rollback, sequence cropping, and position alignment."""
from __future__ import annotations
import pytest
import torch
from spade.decoding.cache_utils import (
    get_cache_seq_length,
    crop_past_key_values,
    clone_past_key_values,
    build_position_ids,
    build_causal_attention_mask,
    CacheManager,
)

class MockDynamicCache:
    """Mock for modern Transformers DynamicCache."""
    def __init__(self, num_layers: int = 2, batch_size: int = 1, num_heads: int = 4, seq_len: int = 10, head_dim: int = 8):
        self.key_cache = [torch.randn(batch_size, num_heads, seq_len, head_dim) for _ in range(num_layers)]
        self.value_cache = [torch.randn(batch_size, num_heads, seq_len, head_dim) for _ in range(num_layers)]

    def get_seq_length(self) -> int:
        return self.key_cache[0].shape[-2]

    def crop(self, max_length: int) -> None:
        for i in range(len(self.key_cache)):
            self.key_cache[i] = self.key_cache[i][..., :max_length, :]
            self.value_cache[i] = self.value_cache[i][..., :max_length, :]

def _make_tuple_cache(num_layers: int = 2, batch_size: int = 1, num_heads: int = 4, seq_len: int = 10, head_dim: int = 8):
    """Create legacy tuple-of-tuples KV cache: ((k0, v0), (k1, v1), ...)."""
    return tuple(
        (torch.randn(batch_size, num_heads, seq_len, head_dim), torch.randn(batch_size, num_heads, seq_len, head_dim))
        for _ in range(num_layers)
    )

def test_cache_seq_length_detection():
    assert get_cache_seq_length(None) == 0

    dynamic = MockDynamicCache(seq_len=16)
    assert get_cache_seq_length(dynamic) == 16

    legacy = _make_tuple_cache(seq_len=24)
    assert get_cache_seq_length(legacy) == 24

def test_dynamic_cache_crop_and_rollback():
    cache = MockDynamicCache(seq_len=20)
    assert get_cache_seq_length(cache) == 20

    # Rollback to 12 tokens
    cropped = crop_past_key_values(cache, max_length=12)
    assert get_cache_seq_length(cropped) == 12
    assert cropped.key_cache[0].shape[-2] == 12
    assert cropped.value_cache[0].shape[-2] == 12

def test_legacy_tuple_cache_crop_and_rollback():
    cache = _make_tuple_cache(seq_len=15)
    assert get_cache_seq_length(cache) == 15

    # Crop to 8
    cropped = crop_past_key_values(cache, max_length=8)
    assert get_cache_seq_length(cropped) == 8
    assert cropped[0][0].shape[-2] == 8
    assert cropped[0][1].shape[-2] == 8

def test_cache_manager_state_and_rollback():
    cache = MockDynamicCache(seq_len=30)
    mgr = CacheManager(cache)
    assert mgr.seq_length == 30

    # Rejection at draft position 2 with prefix 20: target length should be 22
    mgr.rollback(22)
    assert mgr.seq_length == 22

    # Verify position IDs start at current seq_length
    pos_ids = mgr.position_ids_for(num_tokens=4, device=torch.device("cpu"))
    assert pos_ids.shape == (1, 4)
    assert pos_ids[0].tolist() == [22, 23, 24, 25]

def test_clone_past_key_values():
    original = _make_tuple_cache(seq_len=10)
    cloned = clone_past_key_values(original)

    # In-place modify original
    original[0][0].zero_()
    # Cloned must remain unchanged
    assert not torch.all(cloned[0][0] == 0)

def test_causal_attention_mask_builder():
    mask = build_causal_attention_mask(total_length=50, num_new_tokens=4, device=torch.device("cpu"))
    assert mask.shape == (1, 50)
    assert (mask == 1.0).all()
