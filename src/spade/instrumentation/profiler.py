"""Timing, GPU memory tracking, and throughput profiling for speculative inference."""
from __future__ import annotations
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator
import torch

@dataclass
class ProfileReport:
    prompt_prefill_s: float = 0.0
    draft_generation_s: float = 0.0
    target_verification_s: float = 0.0
    correction_s: float = 0.0
    decode_s: float = 0.0
    end_to_end_s: float = 0.0
    output_tokens: int = 0
    prompt_tokens: int = 0
    speculative_rounds: int = 0
    accepted_tokens: int = 0
    tokens_per_second: float = 0.0
    gpu_peak_allocated_bytes: list[int] = field(default_factory=list)
    gpu_peak_reserved_bytes: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_prefill_s": self.prompt_prefill_s,
            "draft_generation_s": self.draft_generation_s,
            "target_verification_s": self.target_verification_s,
            "correction_s": self.correction_s,
            "decode_s": self.decode_s,
            "end_to_end_s": self.end_to_end_s,
            "output_tokens": self.output_tokens,
            "prompt_tokens": self.prompt_tokens,
            "speculative_rounds": self.speculative_rounds,
            "accepted_tokens": self.accepted_tokens,
            "tokens_per_second": self.tokens_per_second,
            "gpu_peak_allocated_bytes": self.gpu_peak_allocated_bytes,
            "gpu_peak_reserved_bytes": self.gpu_peak_reserved_bytes,
        }

class InferenceProfiler:
    """Manages stage timing with device synchronization and peak memory tracking."""

    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cpu")
        self.timings: dict[str, float] = {
            "prompt_prefill_s": 0.0,
            "draft_generation_s": 0.0,
            "target_verification_s": 0.0,
            "correction_s": 0.0,
            "decode_s": 0.0,
        }
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.speculative_rounds = 0
        self.accepted_tokens = 0
        self.output_tokens = 0
        self.prompt_tokens = 0

    def sync(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def reset_memory_stats(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    @contextmanager
    def time_stage(self, stage_name: str) -> Generator[None, None, None]:
        """Context manager to accurately time a decoding stage with synchronization."""
        self.sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            elapsed = time.perf_counter() - t0
            self.timings[stage_name] = self.timings.get(stage_name, 0.0) + elapsed

    def start_request(self) -> None:
        self.sync()
        self.reset_memory_stats()
        self.start_time = time.perf_counter()

    def end_request(self, prompt_tokens: int, output_tokens: int) -> ProfileReport:
        self.sync()
        self.end_time = time.perf_counter()
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens

        end_to_end = (self.end_time - self.start_time) if self.start_time else 0.0
        tps = (output_tokens / self.timings["decode_s"]) if self.timings["decode_s"] > 0 else (
            output_tokens / end_to_end if end_to_end > 0 else 0.0
        )

        peak_alloc: list[int] = []
        peak_res: list[int] = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                peak_alloc.append(torch.cuda.max_memory_allocated(i))
                peak_res.append(torch.cuda.max_memory_reserved(i))

        return ProfileReport(
            prompt_prefill_s=self.timings.get("prompt_prefill_s", 0.0),
            draft_generation_s=self.timings.get("draft_generation_s", 0.0),
            target_verification_s=self.timings.get("target_verification_s", 0.0),
            correction_s=self.timings.get("correction_s", 0.0),
            decode_s=self.timings.get("decode_s", 0.0),
            end_to_end_s=end_to_end,
            output_tokens=output_tokens,
            prompt_tokens=prompt_tokens,
            speculative_rounds=self.speculative_rounds,
            accepted_tokens=self.accepted_tokens,
            tokens_per_second=tps,
            gpu_peak_allocated_bytes=peak_alloc,
            gpu_peak_reserved_bytes=peak_res,
        )
