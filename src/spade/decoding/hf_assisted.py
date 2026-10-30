"""Correctness-first Hugging Face decoding with explicit full-prefix recomputation.

This adapter intentionally does not use a KV cache. It is suitable for short
correctness/smoke runs and produces authoritative token decisions, but its
latency is not representative of optimized speculative decoding.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import inspect
from typing import Any
import torch
from .autoregressive import load_causal_lm, validate_tokenizers, format_qwen_prompt
from .sampling import processed_distribution, sample_distribution
from .speculative import exact_speculative_round
from ..instrumentation.token_analysis import categorize_token, evidence_overlap

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
        self.target_tokenizer=AutoTokenizer.from_pretrained(target_path,local_files_only=local_files_only)
        self.draft_tokenizer=None
        self.target=load_causal_lm(target_path,local_files_only=local_files_only,dtype=dtype)
        self.draft=None
        if draft_path:
            self.draft_tokenizer=AutoTokenizer.from_pretrained(draft_path,local_files_only=local_files_only)
            validate_tokenizers(self.target_tokenizer,self.draft_tokenizer)
            self.draft=load_causal_lm(draft_path,local_files_only=local_files_only,dtype=dtype)

    @staticmethod
    def _input_device(model):
        return model.get_input_embeddings().weight.device

    @staticmethod
    def _logits(model, ids: list[int], device: torch.device) -> torch.Tensor:
        input_ids=torch.tensor([ids],dtype=torch.long,device=device)
        kwargs={"input_ids":input_ids,"use_cache":False}
        # Qwen3 can project only the final hidden state through its large LM
        # head. Projecting every prompt position creates a [1, seq, vocab]
        # tensor (about 5 GB at the observed 16k-token context).
        try:
            forward=getattr(model,"forward",model.__call__)
            if "logits_to_keep" in inspect.signature(forward).parameters:
                kwargs["logits_to_keep"]=1
        except (TypeError,ValueError):
            pass
        with torch.inference_mode():
            return model(**kwargs).logits[0,-1].float().to(device)

    @staticmethod
    def _sync(device: torch.device) -> None:
        if device.type == "cuda": torch.cuda.synchronize(device)

    def generate(self, prompt: str, *, mode: str, max_new_tokens: int = 256, draft_length: int = 4,
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, seed: int = 42,
                 run_id: str = "diagnostic", claim_id: str = "unknown",
                 evidence_texts: list[str] | None = None) -> Generation:
        """Run B0/B0-S/B1/B2. Full-prefix recomputation avoids cache rollback assumptions."""
        if mode not in {"B0","B0-S","B1","B2"}: raise ValueError(f"Unsupported mode {mode}")
        if mode in {"B1","B2"} and self.draft is None: raise ValueError(f"{mode} requires a draft model")
        if max_new_tokens < 1 or draft_length < 1: raise ValueError("Token limits must be positive")
        serialized=format_qwen_prompt(self.target_tokenizer,prompt)
        prompt_ids=self.target_tokenizer(serialized,return_tensors="pt",add_special_tokens=False)["input_ids"][0].tolist()
        context_limit=getattr(getattr(self.target,"config",None),"max_position_embeddings",None)
        if isinstance(context_limit,int) and len(prompt_ids)+max_new_tokens>context_limit:
            raise ValueError(f"Prompt ({len(prompt_ids)} tokens) plus requested output ({max_new_tokens}) exceeds target context limit ({context_limit})")
        target_device=self._input_device(self.target); target_eos=self.target_tokenizer.eos_token_id
        if self.draft is not None and self._input_device(self.draft)!=target_device:
            raise RuntimeError("Target and draft embeddings are on different devices; current exact-sampling adapter requires a shared inference device")
        generator=torch.Generator(device=target_device).manual_seed(seed)
        generated=[]; round_rows=[]; token_rows=[]
        timings={"prompt_prefill_s":0.0,"draft_generation_s":0.0,"target_verification_s":0.0,"correction_s":0.0,"decode_s":0.0}
        has_target_prefill=False
        start=time.perf_counter(); round_index=0
        while len(generated)<max_new_tokens:
            prefix=prompt_ids+generated; remaining=max_new_tokens-len(generated); generated_before=len(generated)
            self._sync(target_device); rstart=time.perf_counter(); proposed=[]; q_dists=[]; p_dists=[]
            if mode in {"B0","B0-S"}:
                t0=time.perf_counter(); logits=self._logits(self.target,prefix,target_device); self._sync(target_device); target_s=time.perf_counter()-t0
                if not has_target_prefill: timings["prompt_prefill_s"]+=target_s; has_target_prefill=True
                else: timings["target_verification_s"]+=target_s
                if mode=="B0": token=int(torch.argmax(logits).item())
                else: token=sample_distribution(processed_distribution(logits,temperature,top_k,top_p),generator)
                generated.append(token)
                total_s=time.perf_counter()-rstart
                round_rows.append({"run_id":run_id,"claim_id":claim_id,"round_index":round_index,"prefix_length":len(prefix),"draft_length_requested":0,"proposed_token_count":0,"accepted_draft_tokens":0,"rejection_position":None,"full_acceptance":False,"additional_target_token":False,"committed_output_tokens":1,"draft_latency_s":0.0,"verification_latency_s":target_s,"correction_latency_s":0.0,"total_round_latency_s":total_s,"cumulative_generated_tokens":len(generated)})
                if token==target_eos: break
                round_index+=1; continue
            draft_device=self._input_device(self.draft)
            count=min(draft_length,remaining)
            d0=time.perf_counter(); draft_prefix=list(prefix)
            for _ in range(count):
                qlogits=self._logits(self.draft,draft_prefix,draft_device)
                q=processed_distribution(qlogits,temperature,top_k,top_p)
                token=int(torch.argmax(q).item()) if mode=="B1" else sample_distribution(q,generator)
                proposed.append(token); q_dists.append(q); draft_prefix.append(token)
                if token==target_eos: break
            self._sync(draft_device); draft_s=time.perf_counter()-d0; timings["draft_generation_s"]+=draft_s
            t0=time.perf_counter()
            # Recompute the exact committed prefix at each position. This mirrors
            # B0's call shape and avoids block-length/backend rounding differences.
            verification_prefix=list(prefix)
            for proposed_token in proposed:
                verify_start=time.perf_counter()
                logits=self._logits(self.target,verification_prefix,target_device)
                self._sync(target_device); verify_s=time.perf_counter()-verify_start
                if not has_target_prefill: timings["prompt_prefill_s"]+=verify_s; has_target_prefill=True
                else: timings["target_verification_s"]+=verify_s
                p_dists.append(processed_distribution(logits,temperature,top_k,top_p))
                verification_prefix.append(proposed_token)
            extra_start=time.perf_counter()
            extra_logits=self._logits(self.target,verification_prefix,target_device)
            target_extra=processed_distribution(extra_logits,temperature,top_k,top_p)
            self._sync(target_device); extra_s=time.perf_counter()-extra_start
            timings["target_verification_s"]+=extra_s
            target_s=time.perf_counter()-t0
            c0=time.perf_counter()
            generation_start=len(generated)
            if mode=="B1":
                target_argmax=[int(torch.argmax(x).item()) for x in p_dists]
                accepted=[]; rejection=None
                for i,(prop,expected) in enumerate(zip(proposed,target_argmax)):
                    if prop==expected:
                        accepted.append(prop)
                        if prop==target_eos: break
                    else:
                        accepted.append(expected); rejection=i; break
                if not rejection is None:
                    status=["accepted"]*rejection+["rejected"]+["discarded_after_rejection"]*(len(proposed)-rejection-1)
                else: status=["accepted"]*len(accepted)+["discarded_after_rejection"]*(len(proposed)-len(accepted))
                extra=None
                if rejection is None and len(accepted)==len(proposed) and len(accepted)<remaining and (not accepted or accepted[-1]!=target_eos):
                    extra=int(torch.argmax(extra_logits).item()); accepted.append(extra)
                generated.extend(accepted)
                accepted_draft=sum(s=="accepted" for s in status)
            else:
                result=exact_speculative_round(proposed,q_dists,p_dists,generator=generator,eos_token_id=target_eos,remaining=remaining,target_extra=target_extra)
                generated.extend(result.committed); status=result.statuses; rejection=result.rejected_position; extra=result.extra_target
                accepted_draft=result.accepted
                if result.replacement is not None:
                    repl_text=self.target_tokenizer.decode([result.replacement])
                    p_reject=p_dists[result.rejected_position]; q_reject=q_dists[result.rejected_position]
                    tv=float((0.5*torch.abs(p_reject-q_reject).sum()).item())
                    token_rows.append({"run_id":run_id,"claim_id":claim_id,"round_id":round_index,"absolute_token_position":len(prompt_ids)+generation_start+result.rejected_position,"token_id":result.replacement,"token_text":repl_text,"draft_probability":None,"target_probability":float(p_reject[result.replacement].item()),"acceptance_probability":None,"total_variation":tv,"expected_acceptance_probability":1.0-tv,"status":"residual_replacement","reached_by_verifier":True,"discarded_after_rejection":False,"replacement_token_id":result.replacement,"token_category":categorize_token(repl_text),"prefix_length":len(prompt_ids)+generation_start,"evidence_overlap":evidence_overlap(repl_text,evidence_texts or [])})
            if mode=="B1" and rejection is not None:
                replacement=accepted[-1]; repl_text=self.target_tokenizer.decode([replacement])
                token_rows.append({"run_id":run_id,"claim_id":claim_id,"round_id":round_index,"absolute_token_position":len(prompt_ids)+generation_start+rejection,"token_id":replacement,"token_text":repl_text,"draft_probability":None,"target_probability":float(p_dists[rejection][replacement].item()),"acceptance_probability":None,"total_variation":None,"expected_acceptance_probability":None,"status":"target_replacement","reached_by_verifier":True,"discarded_after_rejection":False,"replacement_token_id":replacement,"token_category":categorize_token(repl_text),"prefix_length":len(prompt_ids)+generation_start,"evidence_overlap":evidence_overlap(repl_text,evidence_texts or [])})
            self._sync(target_device); correction_s=time.perf_counter()-c0; timings["correction_s"]+=correction_s
            for i,(token,q,pdist) in enumerate(zip(proposed,q_dists,p_dists)):
                prob=float(pdist[token].item()); qprob=float(q[token].item()); ap=min(1.0,prob/qprob) if qprob else None
                st=status[i] if i<len(status) else "discarded_after_rejection"
                reached=st!="discarded_after_rejection"
                token_text=self.target_tokenizer.decode([token])
                tv=float((0.5*torch.abs(pdist-q).sum()).item()) if mode=="B2" else None
                token_rows.append({"run_id":run_id,"claim_id":claim_id,"round_id":round_index,"absolute_token_position":len(prompt_ids)+generation_start+i,"token_id":token,"token_text":token_text,"draft_probability":qprob,"target_probability":prob,"acceptance_probability":ap if mode=="B2" else None,"total_variation":tv,"expected_acceptance_probability":1.0-tv if tv is not None else None,"status":st,"reached_by_verifier":reached,"discarded_after_rejection":st=="discarded_after_rejection","replacement_token_id":None,"token_category":categorize_token(token_text),"prefix_length":len(prompt_ids)+generation_start,"evidence_overlap":evidence_overlap(token_text,evidence_texts or [])})
            if extra is not None:
                extra_text=self.target_tokenizer.decode([extra])
                token_rows.append({"run_id":run_id,"claim_id":claim_id,"round_id":round_index,"absolute_token_position":len(prompt_ids)+len(generated)-1,"token_id":extra,"token_text":extra_text,"draft_probability":None,"target_probability":float(target_extra[extra].item()) if mode=="B2" else 1.0,"acceptance_probability":None,"total_variation":None,"expected_acceptance_probability":None,"status":"extra_target_token","reached_by_verifier":False,"discarded_after_rejection":False,"replacement_token_id":None,"token_category":categorize_token(extra_text),"prefix_length":len(prompt_ids)+generation_start+len(proposed),"evidence_overlap":evidence_overlap(extra_text,evidence_texts or [])})
            round_rows.append({"run_id":run_id,"claim_id":claim_id,"round_index":round_index,"prefix_length":len(prefix),"draft_length_requested":draft_length,"proposed_token_count":len(proposed),"accepted_draft_tokens":accepted_draft,"rejection_position":rejection,"full_acceptance":rejection is None and accepted_draft==len(proposed),"additional_target_token":extra is not None,"committed_output_tokens":len(generated)-generated_before,"draft_latency_s":draft_s,"verification_latency_s":target_s,"correction_latency_s":correction_s,"total_round_latency_s":time.perf_counter()-rstart,"cumulative_generated_tokens":len(generated)})
            if generated and generated[-1]==target_eos: break
            round_index+=1
        timings["decode_s"]=time.perf_counter()-start
        decoded=self.target_tokenizer.decode(generated,skip_special_tokens=True)
        return Generation(generated,decoded,len(prompt_ids),timings,round_rows,token_rows)
