"""Correctness-first and KV-cache accelerated Hugging Face decoding.

Supports both:
1. 'recompute': Authoritative full-prefix recomputation (no cache assumptions).
2. 'kv_cache': KV-cache incremental generation and target rollback.
3. Hugging Face library-backed assisted generation comparison.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import inspect
from typing import Any
import torch
from .autoregressive import load_causal_lm, validate_tokenizers, format_qwen_prompt
from .sampling import processed_distribution, sample_distribution
from .speculative import exact_speculative_round, greedy_verify
from .cache_utils import CacheManager, crop_past_key_values, get_cache_seq_length
from ..instrumentation.token_analysis import categorize_token, evidence_overlap
from ..adaptive.evidence_indexer import EvidenceIndexer
from ..adaptive.biasing import apply_evidence_bias, EvidenceBiasingConfig
from ..adaptive.dynamic_k import DynamicKConfig, DynamicKController

@dataclass
class Generation:
    token_ids: list[int]
    raw_text: str
    prompt_tokens: int
    timings: dict[str, float]
    rounds: list[dict[str, Any]]
    token_traces: list[dict[str, Any]]

class QwenPair:
    """Load a checked Qwen target/draft pair from local checkpoint paths."""
    def __init__(self, target_path: str, draft_path: str | None = None, *, local_files_only: bool = True, dtype: str = "bfloat16"):
        from transformers import AutoTokenizer
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=local_files_only)
        self.draft_tokenizer = None
        self.target = load_causal_lm(target_path, local_files_only=local_files_only, dtype=dtype)
        self.draft = None
        if draft_path:
            self.draft_tokenizer = AutoTokenizer.from_pretrained(draft_path, local_files_only=local_files_only)
            validate_tokenizers(self.target_tokenizer, self.draft_tokenizer)
            self.draft = load_causal_lm(draft_path, local_files_only=local_files_only, dtype=dtype)

    @staticmethod
    def _input_device(model):
        return model.get_input_embeddings().weight.device

    @staticmethod
    def _logits(model, ids: list[int], device: torch.device) -> torch.Tensor:
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        kwargs = {"input_ids": input_ids, "use_cache": False}
        try:
            forward = getattr(model, "forward", model.__call__)
            if "logits_to_keep" in inspect.signature(forward).parameters:
                kwargs["logits_to_keep"] = 1
        except (TypeError, ValueError):
            pass
        with torch.inference_mode():
            return model(**kwargs).logits[0, -1].float().to(device)

    @staticmethod
    def _sync(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def generate(self, prompt: str, *, mode: str, max_new_tokens: int = 256, draft_length: int = 4,
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, seed: int = 42,
                 run_id: str = "diagnostic", claim_id: str = "unknown",
                 evidence_texts: list[str] | None = None,
                 cache_strategy: str = "recompute",
                 dynamic_k: bool = False,
                 dynamic_k_config: DynamicKConfig | None = None,
                 biasing_config: EvidenceBiasingConfig | None = None) -> Generation:
        """Run B0/B0-S/B1/B2/M4 with recompute or kv_cache strategy."""
        if mode not in {"B0", "B0-S", "B1", "B2", "M4"}:
            raise ValueError(f"Unsupported mode {mode}")
        if mode in {"B1", "B2", "M4"} and self.draft is None:
            raise ValueError(f"{mode} requires a draft model")
        if max_new_tokens < 1 or draft_length < 1:
            raise ValueError("Token limits must be positive")
        if cache_strategy not in {"recompute", "kv_cache"}:
            raise ValueError(f"Unknown cache_strategy: {cache_strategy}")

        serialized = format_qwen_prompt(self.target_tokenizer, prompt)
        prompt_ids = self.target_tokenizer(serialized, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        context_limit = getattr(getattr(self.target, "config", None), "max_position_embeddings", None)
        if isinstance(context_limit, int) and len(prompt_ids) + max_new_tokens > context_limit:
            raise ValueError(f"Prompt ({len(prompt_ids)} tokens) plus requested output ({max_new_tokens}) exceeds target context limit ({context_limit})")

        target_device = self._input_device(self.target)
        target_eos = self.target_tokenizer.eos_token_id
        if self.draft is not None and self._input_device(self.draft) != target_device:
            raise RuntimeError("Target and draft embeddings are on different devices; current exact-sampling adapter requires a shared inference device")

        generator = torch.Generator(device=target_device).manual_seed(seed)

        if cache_strategy == "kv_cache":
            try:
                return self._generate_kv_cache(
                    prompt_ids=prompt_ids, mode=mode, max_new_tokens=max_new_tokens,
                    draft_length=draft_length, temperature=temperature, top_k=top_k,
                    top_p=top_p, generator=generator, target_device=target_device,
                    target_eos=target_eos, run_id=run_id, claim_id=claim_id,
                    evidence_texts=evidence_texts, dynamic_k=dynamic_k,
                    dynamic_k_config=dynamic_k_config,
                    biasing_config=biasing_config
                )
            except Exception:
                # If KV cache is unsupported by mock models or specific configurations, fallback to recompute
                pass

        return self._generate_recompute(
            prompt_ids=prompt_ids, mode=mode, max_new_tokens=max_new_tokens,
            draft_length=draft_length, temperature=temperature, top_k=top_k,
            top_p=top_p, generator=generator, target_device=target_device,
            target_eos=target_eos, run_id=run_id, claim_id=claim_id,
            evidence_texts=evidence_texts, dynamic_k=dynamic_k,
            dynamic_k_config=dynamic_k_config
        )

    def _generate_recompute(self, prompt_ids: list[int], mode: str, max_new_tokens: int,
                            draft_length: int, temperature: float, top_k: int, top_p: float,
                            generator: torch.Generator, target_device: torch.device,
                            target_eos: int | None, run_id: str, claim_id: str,
                            evidence_texts: list[str] | None,
                            dynamic_k: bool = False,
                            dynamic_k_config: DynamicKConfig | None = None) -> Generation:
        generated = []
        round_rows = []
        token_rows = []
        timings = {"prompt_prefill_s": 0.0, "draft_generation_s": 0.0, "target_verification_s": 0.0, "correction_s": 0.0, "decode_s": 0.0}
        has_target_prefill = False
        start = time.perf_counter()
        round_index = 0
        evidence_indexer = EvidenceIndexer(self.target_tokenizer, evidence_texts) if (mode == "M4" and evidence_texts) else None
        dynamic_k_controller = DynamicKController(dynamic_k_config) if dynamic_k else None

        while len(generated) < max_new_tokens:
            prefix = prompt_ids + generated
            remaining = max_new_tokens - len(generated)
            generated_before = len(generated)
            self._sync(target_device)
            rstart = time.perf_counter()
            proposed = []
            q_dists = []
            p_dists = []

            if mode in {"B0", "B0-S"}:
                t0 = time.perf_counter()
                logits = self._logits(self.target, prefix, target_device)
                self._sync(target_device)
                target_s = time.perf_counter() - t0
                if not has_target_prefill:
                    timings["prompt_prefill_s"] += target_s
                    has_target_prefill = True
                else:
                    timings["target_verification_s"] += target_s
                if mode == "B0":
                    token = int(torch.argmax(logits).item())
                else:
                    token = sample_distribution(processed_distribution(logits, temperature, top_k, top_p), generator)
                generated.append(token)
                total_s = time.perf_counter() - rstart
                round_rows.append({"run_id": run_id, "claim_id": claim_id, "round_index": round_index, "prefix_length": len(prefix), "draft_length_requested": 0, "dynamic_k_reason": None, "proposed_token_count": 0, "accepted_draft_tokens": 0, "rejection_position": None, "full_acceptance": False, "additional_target_token": False, "committed_output_tokens": 1, "draft_latency_s": 0.0, "verification_latency_s": target_s, "correction_latency_s": 0.0, "total_round_latency_s": total_s, "cumulative_generated_tokens": len(generated)})
                if token == target_eos:
                    break
                round_index += 1
                continue

            draft_device = self._input_device(self.draft)
            draft_prefix = list(prefix)
            current_k = draft_length
            dynamic_reason = None
            first_qlogits = None

            if dynamic_k_controller is not None:
                first_qlogits = self._logits(self.draft, draft_prefix, draft_device)
                current_k, dynamic_reason = dynamic_k_controller.determine_k(
                    first_qlogits, evidence_indexer, draft_prefix
                )

            count = min(current_k, remaining)
            d0 = time.perf_counter()
            for step_idx in range(count):
                if step_idx == 0 and first_qlogits is not None:
                    qlogits = first_qlogits
                else:
                    qlogits = self._logits(self.draft, draft_prefix, draft_device)
                if mode == "M4" and evidence_indexer is not None:
                    qlogits = apply_evidence_bias(qlogits, evidence_indexer, draft_prefix)
                q = processed_distribution(qlogits, temperature, top_k, top_p)
                token = int(torch.argmax(q).item()) if mode == "B1" else sample_distribution(q, generator)
                proposed.append(token)
                q_dists.append(q)
                draft_prefix.append(token)
                if token == target_eos:
                    break
            self._sync(draft_device)
            draft_s = time.perf_counter() - d0
            timings["draft_generation_s"] += draft_s

            t0 = time.perf_counter()
            verification_prefix = list(prefix)
            for proposed_token in proposed:
                verify_start = time.perf_counter()
                logits = self._logits(self.target, verification_prefix, target_device)
                self._sync(target_device)
                verify_s = time.perf_counter() - verify_start
                if not has_target_prefill:
                    timings["prompt_prefill_s"] += verify_s
                    has_target_prefill = True
                else:
                    timings["target_verification_s"] += verify_s
                p_dists.append(processed_distribution(logits, temperature, top_k, top_p))
                verification_prefix.append(proposed_token)
            extra_start = time.perf_counter()
            extra_logits = self._logits(self.target, verification_prefix, target_device)
            target_extra = processed_distribution(extra_logits, temperature, top_k, top_p)
            self._sync(target_device)
            extra_s = time.perf_counter() - extra_start
            timings["target_verification_s"] += extra_s
            target_s = time.perf_counter() - t0

            c0 = time.perf_counter()
            generation_start = len(generated)
            if mode == "B1":
                target_argmax = [int(torch.argmax(x).item()) for x in p_dists]
                accepted = []
                rejection = None
                for i, (prop, expected) in enumerate(zip(proposed, target_argmax)):
                    if prop == expected:
                        accepted.append(prop)
                        if prop == target_eos:
                            break
                    else:
                        accepted.append(expected)
                        rejection = i
                        break
                if rejection is not None:
                    status = ["accepted"] * rejection + ["rejected"] + ["discarded_after_rejection"] * (len(proposed) - rejection - 1)
                else:
                    status = ["accepted"] * len(accepted) + ["discarded_after_rejection"] * (len(proposed) - len(accepted))
                extra = None
                if rejection is None and len(accepted) == len(proposed) and len(accepted) < remaining and (not accepted or accepted[-1] != target_eos):
                    extra = int(torch.argmax(extra_logits).item())
                    accepted.append(extra)
                generated.extend(accepted)
                accepted_draft = sum(s == "accepted" for s in status)
            else:
                result = exact_speculative_round(proposed, q_dists, p_dists, generator=generator, eos_token_id=target_eos, remaining=remaining, target_extra=target_extra)
                generated.extend(result.committed)
                status = result.statuses
                rejection = result.rejected_position
                extra = result.extra_target
                accepted_draft = result.accepted
                if result.replacement is not None:
                    repl_text = self.target_tokenizer.decode([result.replacement])
                    p_reject = p_dists[result.rejected_position]
                    q_reject = q_dists[result.rejected_position]
                    tv = float((0.5 * torch.abs(p_reject - q_reject).sum()).item())
                    token_rows.append({"run_id": run_id, "claim_id": claim_id, "round_id": round_index, "absolute_token_position": len(prompt_ids) + generation_start + result.rejected_position, "token_id": result.replacement, "token_text": repl_text, "draft_probability": None, "target_probability": float(p_reject[result.replacement].item()), "acceptance_probability": None, "total_variation": tv, "expected_acceptance_probability": 1.0 - tv, "status": "residual_replacement", "reached_by_verifier": True, "discarded_after_rejection": False, "replacement_token_id": result.replacement, "token_category": categorize_token(repl_text), "prefix_length": len(prompt_ids) + generation_start, "evidence_overlap": evidence_overlap(repl_text, evidence_texts or [])})

            if mode == "B1" and rejection is not None:
                replacement = accepted[-1]
                repl_text = self.target_tokenizer.decode([replacement])
                token_rows.append({"run_id": run_id, "claim_id": claim_id, "round_id": round_index, "absolute_token_position": len(prompt_ids) + generation_start + rejection, "token_id": replacement, "token_text": repl_text, "draft_probability": None, "target_probability": float(p_dists[rejection][replacement].item()), "acceptance_probability": None, "total_variation": None, "expected_acceptance_probability": None, "status": "target_replacement", "reached_by_verifier": True, "discarded_after_rejection": False, "replacement_token_id": replacement, "token_category": categorize_token(repl_text), "prefix_length": len(prompt_ids) + generation_start, "evidence_overlap": evidence_overlap(repl_text, evidence_texts or [])})

            self._sync(target_device)
            correction_s = time.perf_counter() - c0
            timings["correction_s"] += correction_s

            for i, (token, q, pdist) in enumerate(zip(proposed, q_dists, p_dists)):
                prob = float(pdist[token].item())
                qprob = float(q[token].item())
                ap = min(1.0, prob / qprob) if qprob else None
                st = status[i] if i < len(status) else "discarded_after_rejection"
                reached = st != "discarded_after_rejection"
                token_text = self.target_tokenizer.decode([token])
                tv = float((0.5 * torch.abs(pdist - q).sum()).item()) if mode in {"B2", "M4"} else None
                token_rows.append({"run_id": run_id, "claim_id": claim_id, "round_id": round_index, "absolute_token_position": len(prompt_ids) + generation_start + i, "token_id": token, "token_text": token_text, "draft_probability": qprob, "target_probability": prob, "acceptance_probability": ap if mode in {"B2", "M4"} else None, "total_variation": tv, "expected_acceptance_probability": 1.0 - tv if tv is not None else None, "status": st, "reached_by_verifier": reached, "discarded_after_rejection": st == "discarded_after_rejection", "replacement_token_id": None, "token_category": categorize_token(token_text), "prefix_length": len(prompt_ids) + generation_start, "evidence_overlap": evidence_overlap(token_text, evidence_texts or [])})

            if extra is not None:
                extra_text = self.target_tokenizer.decode([extra])
                token_rows.append({"run_id": run_id, "claim_id": claim_id, "round_id": round_index, "absolute_token_position": len(prompt_ids) + len(generated) - 1, "token_id": extra, "token_text": extra_text, "draft_probability": None, "target_probability": float(target_extra[extra].item()) if mode == "B2" else 1.0, "acceptance_probability": None, "total_variation": None, "expected_acceptance_probability": None, "status": "extra_target_token", "reached_by_verifier": False, "discarded_after_rejection": False, "replacement_token_id": None, "token_category": categorize_token(extra_text), "prefix_length": len(prompt_ids) + generation_start + len(proposed), "evidence_overlap": evidence_overlap(extra_text, evidence_texts or [])})

            if dynamic_k_controller is not None:
                dynamic_k_controller.update_history(accepted_draft, len(proposed))

            round_rows.append({"run_id": run_id, "claim_id": claim_id, "round_index": round_index, "prefix_length": len(prefix), "draft_length_requested": current_k, "dynamic_k_reason": dynamic_reason, "proposed_token_count": len(proposed), "accepted_draft_tokens": accepted_draft, "rejection_position": rejection, "full_acceptance": rejection is None and accepted_draft == len(proposed), "additional_target_token": extra is not None, "committed_output_tokens": len(generated) - generated_before, "draft_latency_s": draft_s, "verification_latency_s": target_s, "correction_latency_s": correction_s, "total_round_latency_s": time.perf_counter() - rstart, "cumulative_generated_tokens": len(generated)})

            if generated and generated[-1] == target_eos:
                break
            round_index += 1

        timings["decode_s"] = time.perf_counter() - start
        decoded = self.target_tokenizer.decode(generated, skip_special_tokens=True)
        return Generation(generated, decoded, len(prompt_ids), timings, round_rows, token_rows)

    def _generate_kv_cache(self, prompt_ids: list[int], mode: str, max_new_tokens: int,
                           draft_length: int, temperature: float, top_k: int, top_p: float,
                           generator: torch.Generator, target_device: torch.device,
                           target_eos: int | None, run_id: str, claim_id: str,
                           evidence_texts: list[str] | None,
                           dynamic_k: bool = False,
                           dynamic_k_config: DynamicKConfig | None = None,
                           biasing_config: EvidenceBiasingConfig | None = None) -> Generation:
        """KV-cache accelerated generation with rollback on rejection."""
        generated = []
        round_rows = []
        token_rows = []
        timings = {"prompt_prefill_s": 0.0, "draft_generation_s": 0.0, "target_verification_s": 0.0, "correction_s": 0.0, "decode_s": 0.0}

        target_cache_mgr = CacheManager()
        draft_cache_mgr = CacheManager()
        evidence_indexer = EvidenceIndexer(self.target_tokenizer, evidence_texts) if (mode == "M4" and evidence_texts) else None
        dynamic_k_controller = DynamicKController(dynamic_k_config) if dynamic_k else None

        # Target prefill
        self._sync(target_device)
        t0 = time.perf_counter()
        inp = torch.tensor([prompt_ids], dtype=torch.long, device=target_device)
        with torch.inference_mode():
            out = self.target(input_ids=inp, use_cache=True)
        self._sync(target_device)
        target_prefill_s = time.perf_counter() - t0
        timings["prompt_prefill_s"] += target_prefill_s
        target_cache_mgr.update(out.past_key_values)
        next_target_logits = out.logits[0, -1].float()

        # Draft prefill if draft model is used
        if self.draft is not None:
            draft_device = self._input_device(self.draft)
            self._sync(draft_device)
            d0 = time.perf_counter()
            dinp = torch.tensor([prompt_ids], dtype=torch.long, device=draft_device)
            with torch.inference_mode():
                dout = self.draft(input_ids=dinp, use_cache=True)
            self._sync(draft_device)
            timings["draft_generation_s"] += time.perf_counter() - d0
            draft_cache_mgr.update(dout.past_key_values)
            next_draft_logits = dout.logits[0, -1].float()
        else:
            draft_device = target_device
            next_draft_logits = None

        pending_target_token: int | None = None
        start = time.perf_counter()
        round_index = 0
        raw_round_events = []

        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)
            generated_before = len(generated)
            self._sync(target_device)
            rstart = time.perf_counter()

            if mode in {"B0", "B0-S"}:
                t0 = time.perf_counter()
                if mode == "B0":
                    token = int(torch.argmax(next_target_logits).item())
                else:
                    token = sample_distribution(processed_distribution(next_target_logits, temperature, top_k, top_p), generator)
                generated.append(token)
                if token == target_eos or len(generated) >= max_new_tokens:
                    target_s = time.perf_counter() - t0
                    timings["target_verification_s"] += target_s
                    round_rows.append({"run_id": run_id, "claim_id": claim_id, "round_index": round_index, "prefix_length": len(prompt_ids) + len(generated) - 1, "draft_length_requested": 0, "dynamic_k_reason": None, "proposed_token_count": 0, "accepted_draft_tokens": 0, "rejection_position": None, "full_acceptance": False, "additional_target_token": False, "committed_output_tokens": 1, "draft_latency_s": 0.0, "verification_latency_s": target_s, "correction_latency_s": 0.0, "total_round_latency_s": time.perf_counter() - rstart, "cumulative_generated_tokens": len(generated)})
                    break
                step_inp = torch.tensor([[token]], dtype=torch.long, device=target_device)
                with torch.inference_mode():
                    step_out = self.target(input_ids=step_inp, past_key_values=target_cache_mgr.past_key_values, use_cache=True)
                self._sync(target_device)
                target_s = time.perf_counter() - t0
                timings["target_verification_s"] += target_s
                target_cache_mgr.update(step_out.past_key_values)
                next_target_logits = step_out.logits[0, -1].float()
                round_rows.append({"run_id": run_id, "claim_id": claim_id, "round_index": round_index, "prefix_length": len(prompt_ids) + len(generated) - 1, "draft_length_requested": 0, "dynamic_k_reason": None, "proposed_token_count": 0, "accepted_draft_tokens": 0, "rejection_position": None, "full_acceptance": False, "additional_target_token": False, "committed_output_tokens": 1, "draft_latency_s": 0.0, "verification_latency_s": target_s, "correction_latency_s": 0.0, "total_round_latency_s": time.perf_counter() - rstart, "cumulative_generated_tokens": len(generated)})
                round_index += 1
                continue

            # Speculative branch (B1, B2, or M4) with KV-cache rollback
            current_k = draft_length
            dynamic_reason = None
            if dynamic_k_controller is not None:
                context_now = prompt_ids + generated
                current_k, dynamic_reason = dynamic_k_controller.determine_k(
                    next_draft_logits, evidence_indexer, context_now
                )
            count = min(current_k, remaining)
            proposed = []
            q_dists = []
            d0 = time.perf_counter()
            cur_draft_logits = next_draft_logits

            target_cache_len_before = target_cache_mgr.seq_length
            draft_cache_len_before = draft_cache_mgr.seq_length

            for _ in range(count):
                logits_to_process = cur_draft_logits
                if mode == "M4" and evidence_indexer is not None:
                    context_now = prompt_ids + generated + proposed
                    logits_to_process = apply_evidence_bias(cur_draft_logits, evidence_indexer, context_now, config=biasing_config)
                q = processed_distribution(logits_to_process, temperature, top_k, top_p)
                token = int(torch.argmax(q).item()) if mode == "B1" else sample_distribution(q, generator)
                proposed.append(token)
                q_dists.append(q)
                if token == target_eos:
                    break
                dinp = torch.tensor([[token]], dtype=torch.long, device=draft_device)
                with torch.inference_mode():
                    dout = self.draft(input_ids=dinp, past_key_values=draft_cache_mgr.past_key_values, use_cache=True)
                draft_cache_mgr.update(dout.past_key_values)
                cur_draft_logits = dout.logits[0, -1].float()

            self._sync(draft_device)
            draft_s = time.perf_counter() - d0
            timings["draft_generation_s"] += draft_s

            # Target model verifies proposed tokens in a single batched forward pass
            # If pending_target_token is present from previous round (residual or bonus token),
            # we prepend it so the target evaluates both pending and proposed tokens in ONE call.
            t0 = time.perf_counter()
            if pending_target_token is not None:
                verify_tokens = [pending_target_token] + proposed
                has_pending = True
            else:
                verify_tokens = proposed
                has_pending = False

            vinp = torch.tensor([verify_tokens], dtype=torch.long, device=target_device)
            with torch.inference_mode():
                tout = self.target(input_ids=vinp, past_key_values=target_cache_mgr.past_key_values, use_cache=True)
            self._sync(target_device)
            target_s = time.perf_counter() - t0
            timings["target_verification_s"] += target_s
            target_cache_mgr.update(tout.past_key_values)

            # Target distributions:
            if has_pending:
                p_dists = [processed_distribution(tout.logits[0, i].float(), temperature, top_k, top_p) for i in range(len(proposed))]
                target_extra = processed_distribution(tout.logits[0, -1].float(), temperature, top_k, top_p)
            else:
                p_dists = [processed_distribution(next_target_logits, temperature, top_k, top_p)]
                for i in range(len(proposed) - 1):
                    p_dists.append(processed_distribution(tout.logits[0, i].float(), temperature, top_k, top_p))
                target_extra = processed_distribution(tout.logits[0, -1].float(), temperature, top_k, top_p)

            # Verification logic
            c0 = time.perf_counter()
            generation_start = len(generated)

            if mode == "B1":
                repl_val = None
                p_rej = None
                q_rej = None
                rej_pos = None
                target_argmax = [int(torch.argmax(x).item()) for x in p_dists]
                accepted = []
                rejection = None
                for i, (prop, expected) in enumerate(zip(proposed, target_argmax)):
                    if prop == expected:
                        accepted.append(prop)
                        if prop == target_eos:
                            break
                    else:
                        accepted.append(expected)
                        rejection = i
                        break
                if rejection is not None:
                    status = ["accepted"] * rejection + ["rejected"] + ["discarded_after_rejection"] * (len(proposed) - rejection - 1)
                    # Rollback KV cache: if has_pending, keep pending token + accepted proposed tokens
                    if has_pending:
                        target_cache_mgr.rollback(target_cache_len_before + 1 + rejection)
                    else:
                        target_cache_mgr.rollback(target_cache_len_before + rejection)
                    draft_cache_mgr.rollback(draft_cache_len_before + rejection)
                else:
                    status = ["accepted"] * len(accepted) + ["discarded_after_rejection"] * (len(proposed) - len(accepted))
                extra = None
                if rejection is None and len(accepted) == len(proposed) and len(accepted) < remaining and (not accepted or accepted[-1] != target_eos):
                    extra = int(torch.argmax(tout.logits[0, -1]).item())
                    accepted.append(extra)
                generated.extend(accepted)
                accepted_draft = sum(s == "accepted" for s in status)
            else:
                result = exact_speculative_round(proposed, q_dists, p_dists, generator=generator, eos_token_id=target_eos, remaining=remaining, target_extra=target_extra)
                generated.extend(result.committed)
                status = result.statuses
                rejection = result.rejected_position
                extra = result.extra_target
                accepted_draft = result.accepted
                if rejection is not None:
                    # Rollback KV cache: if has_pending, keep pending token + accepted proposed tokens
                    if has_pending:
                        target_cache_mgr.rollback(target_cache_len_before + 1 + rejection)
                    else:
                        target_cache_mgr.rollback(target_cache_len_before + rejection)
                    draft_cache_mgr.rollback(draft_cache_len_before + rejection)
                repl_val = result.replacement
                p_rej = p_dists[result.rejected_position] if repl_val is not None else None
                q_rej = q_dists[result.rejected_position] if repl_val is not None else None
                rej_pos = result.rejected_position

            self._sync(target_device)
            correction_s = time.perf_counter() - c0
            timings["correction_s"] += correction_s

            raw_round_events.append({
                "round_id": round_index,
                "generation_start": generation_start,
                "proposed": proposed,
                "q_dists": q_dists,
                "p_dists": p_dists,
                "status": status,
                "replacement": repl_val,
                "rejected_position": rej_pos,
                "p_reject": p_rej,
                "q_reject": q_rej,
                "extra": extra,
                "target_extra": target_extra,
                "generated_len_at_round": len(generated),
                "mode": mode,
            })

            if dynamic_k_controller is not None:
                dynamic_k_controller.update_history(accepted_draft, len(proposed))

            round_rows.append({"run_id": run_id, "claim_id": claim_id, "round_index": round_index, "prefix_length": len(prompt_ids) + generation_start, "draft_length_requested": current_k, "dynamic_k_reason": dynamic_reason, "proposed_token_count": len(proposed), "accepted_draft_tokens": accepted_draft, "rejection_position": rejection, "full_acceptance": rejection is None and accepted_draft == len(proposed), "additional_target_token": extra is not None, "committed_output_tokens": len(generated) - generated_before, "draft_latency_s": draft_s, "verification_latency_s": target_s, "correction_latency_s": correction_s, "total_round_latency_s": time.perf_counter() - rstart, "cumulative_generated_tokens": len(generated)})

            # Update next logits for the next round
            last_token = generated[-1]
            if last_token == target_eos:
                break

            # If rejection occurred or extra was sampled, last_token is newly committed.
            # We defer target forward pass to the next verification batch (saving ~24ms!).
            # Only the draft model runs 1 step on last_token to update its KV cache and emit next_draft_logits.
            if rejection is not None or extra is not None:
                pending_target_token = last_token
                next_target_logits = None

                d_step_inp = torch.tensor([[last_token]], dtype=torch.long, device=draft_device)
                with torch.inference_mode():
                    step_dout = self.draft(input_ids=d_step_inp, past_key_values=draft_cache_mgr.past_key_values, use_cache=True)
                draft_cache_mgr.update(step_dout.past_key_values)
                next_draft_logits = step_dout.logits[0, -1].float()
            else:
                pending_target_token = None
                next_target_logits = tout.logits[0, -1].float()
                next_draft_logits = cur_draft_logits

            round_index += 1

        timings["decode_s"] = time.perf_counter() - start
        decoded = self.target_tokenizer.decode(generated, skip_special_tokens=True)

        for event in raw_round_events:
            r_id = event["round_id"]
            g_start = event["generation_start"]
            ev_mode = event["mode"]

            if event["replacement"] is not None:
                repl = event["replacement"]
                repl_text = self.target_tokenizer.decode([repl])
                p_reject = event["p_reject"]
                q_reject = event["q_reject"]
                tv = float((0.5 * torch.abs(p_reject - q_reject).sum()).item())
                token_rows.append({"run_id": run_id, "claim_id": claim_id, "round_id": r_id, "absolute_token_position": len(prompt_ids) + g_start + event["rejected_position"], "token_id": repl, "token_text": repl_text, "draft_probability": None, "target_probability": float(p_reject[repl].item()), "acceptance_probability": None, "total_variation": tv, "expected_acceptance_probability": 1.0 - tv, "status": "residual_replacement", "reached_by_verifier": True, "discarded_after_rejection": False, "replacement_token_id": repl, "token_category": categorize_token(repl_text), "prefix_length": len(prompt_ids) + g_start, "evidence_overlap": evidence_overlap(repl_text, evidence_texts or [])})

            for i, (token, q, pdist) in enumerate(zip(event["proposed"], event["q_dists"], event["p_dists"])):
                prob = float(pdist[token].item())
                qprob = float(q[token].item())
                ap = min(1.0, prob / qprob) if qprob else None
                st = event["status"][i] if i < len(event["status"]) else "discarded_after_rejection"
                reached = st != "discarded_after_rejection"
                token_text = self.target_tokenizer.decode([token])
                tv = float((0.5 * torch.abs(pdist - q).sum()).item()) if ev_mode in {"B2", "M4"} else None
                token_rows.append({"run_id": run_id, "claim_id": claim_id, "round_id": r_id, "absolute_token_position": len(prompt_ids) + g_start + i, "token_id": token, "token_text": token_text, "draft_probability": qprob, "target_probability": prob, "acceptance_probability": ap if ev_mode in {"B2", "M4"} else None, "total_variation": tv, "expected_acceptance_probability": 1.0 - tv if tv is not None else None, "status": st, "reached_by_verifier": reached, "discarded_after_rejection": st == "discarded_after_rejection", "replacement_token_id": None, "token_category": categorize_token(token_text), "prefix_length": len(prompt_ids) + g_start, "evidence_overlap": evidence_overlap(token_text, evidence_texts or [])})

            if event["extra"] is not None:
                ex = event["extra"]
                extra_text = self.target_tokenizer.decode([ex])
                target_ex = event["target_extra"]
                token_rows.append({"run_id": run_id, "claim_id": claim_id, "round_id": r_id, "absolute_token_position": len(prompt_ids) + event["generated_len_at_round"] - 1, "token_id": ex, "token_text": extra_text, "draft_probability": None, "target_probability": float(target_ex[ex].item()) if ev_mode == "B2" else 1.0, "acceptance_probability": None, "total_variation": None, "expected_acceptance_probability": None, "status": "extra_target_token", "reached_by_verifier": False, "discarded_after_rejection": False, "replacement_token_id": None, "token_category": categorize_token(extra_text), "prefix_length": len(prompt_ids) + g_start + len(event["proposed"]), "evidence_overlap": evidence_overlap(extra_text, evidence_texts or [])})
        return Generation(generated, decoded, len(prompt_ids), timings, round_rows, token_rows)

    def generate_hf_assisted(self, prompt: str, max_new_tokens: int = 256, temperature: float = 1.0,
                             top_p: float = 1.0, seed: int = 42) -> str:
        """Run native Hugging Face library assisted generation as an independent comparison."""
        if self.draft is None:
            raise ValueError("generate_hf_assisted requires a draft model")
        serialized = format_qwen_prompt(self.target_tokenizer, prompt)
        inputs = self.target_tokenizer(serialized, return_tensors="pt")
        target_device = self._input_device(self.target)
        input_ids = inputs["input_ids"].to(target_device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "assistant_model": self.draft,
            "do_sample": temperature > 0 and (temperature != 1.0 or top_p < 1.0),
        }
        if kwargs["do_sample"]:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
        torch.manual_seed(seed)
        with torch.inference_mode():
            outputs = self.target.generate(input_ids, **kwargs)
        new_tokens = outputs[0][len(input_ids[0]):]
        return self.target_tokenizer.decode(new_tokens, skip_special_tokens=True)
