"""Verify mathematical exactness of TargetPrefillAttentionHook (RoPE + QK-Norm + GQA)
against eager full self-attention (`output_attentions=True`) on real Qwen3 and Qwen2.5 models.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from spade.adaptive.target_attention_selector import (
    TargetPrefillAttentionHook,
    resolve_layer_indices,
)
from spade.decoding.autoregressive import format_qwen_prompt

HUB = Path("/home/stackops/whale/cache/huggingface/hub")
OUT_PATH = Path("outputs/analysis/v6_revisions/hook_exactness.json")


def _find_snapshot(model_dir_name: str) -> str:
    snaps = sorted((HUB / model_dir_name / "snapshots").glob("*"))
    if not snaps:
        raise FileNotFoundError(f"No snapshot under {HUB / model_dir_name}")
    return str(snaps[0])


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    n = int(x.numel())
    if n < 2:
        return 1.0
    rx = torch.argsort(torch.argsort(x)).float()
    ry = torch.argsort(torch.argsort(y)).float()
    vx = rx - rx.mean()
    vy = ry - ry.mean()
    denom = float(vx.norm() * vy.norm())
    return float((vx * vy).sum().item() / denom) if denom > 0 else 0.0


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    vx = x - x.mean()
    vy = y - y.mean()
    denom = float(vx.norm() * vy.norm())
    return float((vx * vy).sum().item() / denom) if denom > 0 else 0.0


def eval_model_exactness(
    model_name: str,
    model_dir_name: str,
    dtype: torch.dtype,
    prompts_toks: list[torch.Tensor],
    tail_window: int = 16,
) -> dict[str, Any]:
    snap = _find_snapshot(model_dir_name)
    eager_model = AutoModelForCausalLM.from_pretrained(
        snap, dtype=dtype, attn_implementation="eager", device_map="cuda", local_files_only=True
    ).eval()
    sdpa_model = eager_model  # Reference hook works directly on SDPA too; compare against eager `output_attentions=True`
    n_layers = len(eager_model.model.layers)
    layer_subset = resolve_layer_indices(n_layers, "upper50")

    rows: list[dict[str, Any]] = []
    for idx, inp in enumerate(prompts_toks):
        seq_len = int(inp.shape[1])
        with torch.inference_mode():
            ref_out = eager_model(input_ids=inp, use_cache=False, output_attentions=True)
            ref_imp = torch.stack(
                [ref_out.attentions[l][0, :, -tail_window:, :].float().mean(dim=(0, 1)) for l in layer_subset],
                dim=0,
            ).mean(dim=0).cpu()
            del ref_out

        # Now run with SDPA / default kernel via config swap to confirm SDPA prefill + Hook matches Eager output_attentions=True
        orig_impl = getattr(sdpa_model.config, "_attn_implementation", "sdpa")
        sdpa_model.config._attn_implementation = "sdpa"
        try:
            with TargetPrefillAttentionHook(sdpa_model, layer_indices=layer_subset, tail_window=tail_window) as hook:
                with torch.inference_mode():
                    out = sdpa_model(input_ids=inp, use_cache=True, logits_to_keep=1)
                got_imp, elapsed_s = hook.compute_token_importance(out.past_key_values, layer_subset=layer_subset)
                del out
        finally:
            sdpa_model.config._attn_implementation = orig_impl

        diff = (got_imp - ref_imp).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        cos = float(torch.nn.functional.cosine_similarity(got_imp.unsqueeze(0), ref_imp.unsqueeze(0)).item())
        r = _pearson(got_imp, ref_imp)
        rho = _spearman(got_imp, ref_imp)
        k = min(64, seq_len)
        top_got = set(torch.topk(got_imp, k=k).indices.tolist())
        top_ref = set(torch.topk(ref_imp, k=k).indices.tolist())
        top_overlap = len(top_got & top_ref) / float(k)

        rows.append({
            "sample_idx": idx,
            "seq_len": seq_len,
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
            "cosine_similarity": cos,
            "pearson_r": r,
            "spearman_rho": rho,
            "top64_token_overlap": top_overlap,
            "hook_ms": elapsed_s * 1000.0,
        })
        torch.cuda.empty_cache()

    del eager_model
    torch.cuda.empty_cache()
    n = len(rows)
    return {
        "model": model_name,
        "dtype": str(dtype),
        "n_layers": n_layers,
        "hooked_layers": len(layer_subset),
        "tail_window": tail_window,
        "samples": rows,
        "mean_max_abs_error": sum(r["max_abs_error"] for r in rows) / n,
        "worst_max_abs_error": max(r["max_abs_error"] for r in rows),
        "mean_abs_error": sum(r["mean_abs_error"] for r in rows) / n,
        "mean_cosine_similarity": sum(r["cosine_similarity"] for r in rows) / n,
        "mean_pearson_r": sum(r["pearson_r"] for r in rows) / n,
        "min_spearman_rho": min(r["spearman_rho"] for r in rows),
        "mean_top64_overlap": sum(r["top64_token_overlap"] for r in rows) / n,
    }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    bench = [json.loads(l) for l in Path("outputs/benchmarks/retrieved-heldout300.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    long_rows = [r for r in bench if sum(len(e.get("text", "")) for e in r.get("evidence", [])) > 10000][:5]
    lengths = [512, 768, 1024, 1536, 2048]

    q3_tok = AutoTokenizer.from_pretrained(_find_snapshot("models--Qwen--Qwen3-8B"), local_files_only=True)
    q3_prompts = []
    for row, L in zip(long_rows, lengths):
        ids = q3_tok(format_qwen_prompt(q3_tok, row["prompt"]), add_special_tokens=False)["input_ids"][:L]
        q3_prompts.append(torch.tensor([ids], dtype=torch.long, device="cuda"))

    q25_tok = AutoTokenizer.from_pretrained(_find_snapshot("models--Qwen--Qwen2.5-7B-Instruct"), local_files_only=True)
    q25_prompts = []
    for row, L in zip(long_rows, lengths):
        ids = q25_tok(format_qwen_prompt(q25_tok, row["prompt"]), add_special_tokens=False)["input_ids"][:L]
        q25_prompts.append(torch.tensor([ids], dtype=torch.long, device="cuda"))

    report = {
        "qwen3_0.6b_fp32": eval_model_exactness("Qwen/Qwen3-0.6B", "models--Qwen--Qwen3-0.6B", torch.float32, q3_prompts[:3]),
        "qwen3_8b_bf16": eval_model_exactness("Qwen/Qwen3-8B", "models--Qwen--Qwen3-8B", torch.bfloat16, q3_prompts),
        "qwen25_7b_bf16": eval_model_exactness("Qwen/Qwen2.5-7B-Instruct", "models--Qwen--Qwen2.5-7B-Instruct", torch.bfloat16, q25_prompts),
    }
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
