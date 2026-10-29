"""Small Hugging Face adapter helpers; imports remain optional for CPU tooling."""
from __future__ import annotations
from typing import Any

def load_causal_lm(model_name_or_path: str, *, local_files_only: bool = True, dtype: str = "bfloat16", revision: str | None = None):
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc: raise RuntimeError("Install the pinned server requirements to load models") from exc
    torch_dtype=getattr(torch,dtype,None)
    if torch_dtype is None: raise ValueError(f"Unsupported torch dtype: {dtype}")
    return AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=torch_dtype,revision=revision,local_files_only=local_files_only,device_map="auto")

def validate_tokenizers(target: Any, draft: Any) -> None:
    """Fail closed unless vocab IDs and relevant special-token/template metadata agree."""
    attrs=("bos_token_id","eos_token_id","pad_token_id","unk_token_id")
    differences=[]
    for name in attrs:
        if getattr(target,name,None)!=getattr(draft,name,None): differences.append(name)
    tv=target.get_vocab(); dv=draft.get_vocab()
    if len(tv)!=len(dv): differences.append("vocab_size")
    else:
        for token, idx in tv.items():
            if dv.get(token)!=idx: differences.append("token_id_mapping"); break
    if differences: raise ValueError("Target/draft tokenizer incompatibility: "+", ".join(differences))

def format_qwen_prompt(tokenizer: Any, prompt: str) -> str:
    messages=[{"role":"user","content":prompt}]
    try: return tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError as exc: raise RuntimeError("Tokenizer chat template does not expose enable_thinking=False; refusing undocumented formatting") from exc
