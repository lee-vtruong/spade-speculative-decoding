"""Disjoint Dev-Set Selector & Hyperparameter Ablation (50 long claims from train-retrieved-bounded.jsonl).

Asserts zero claim_id overlap with retrieved-heldout300.jsonl, generates reference target trajectories once,
and runs fast teacher-forced prefix evaluations across:
  - Sentence score normalization: sum vs mean vs sqrt vs max
  - Tail query window: W in {1, 4, 8, 16, 32}
  - Layer subset: upper50 (default), upper25, middle50, all, last1
  - Head aggregation: mean vs max
  - Budget sweep: B in {1k, 2k, 4k, 8k, 10k, 16k}
  - Baselines: Full, BM25 Sent-B10k, Random Sent-B10k
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from spade.adaptive.context_selection import (
    build_draft_context,
    build_draft_context_from_scores,
)
from spade.adaptive.target_attention_selector import (
    TargetPrefillAttentionHook,
    find_evidence_and_sentence_token_spans,
    resolve_layer_indices,
    score_evidence_from_token_importance,
)
from spade.decoding.autoregressive import format_qwen_prompt
from spade.decoding.hf_assisted import QwenPair
from spade.decoding.runtime_opt import processed_distribution_fast

TARGET_SNAP = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DRAFT_SNAP = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
TRAIN_PATH = Path("outputs/benchmarks/train-retrieved-bounded.jsonl")
TEST_PATH = Path("outputs/benchmarks/retrieved-heldout300.jsonl")
DEV_BENCH_PATH = Path("outputs/benchmarks/dev-long50.jsonl")
DEV_TRAJ_PATH = Path("outputs/analysis/v6_revisions/dev50_target_trajectories.jsonl")
OUT_PATH = Path("outputs/analysis/v6_revisions/dev_selector_ablation.json")


def _build_dev50() -> list[dict[str, Any]]:
    test_ids = {str(json.loads(l)["claim_id"]) for l in TEST_PATH.read_text(encoding="utf-8").splitlines() if l.strip()}
    dev_rows = []
    with TRAIN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            cid = str(r["claim_id"])
            if cid in test_ids:
                continue
            full_c = sum(len(str(e.get("text", ""))) for e in (r.get("evidence") or []))
            if full_c > 10000 and len(r.get("evidence") or []) >= 2:
                dev_rows.append(r)
                if len(dev_rows) >= 50:
                    break
    assert len(dev_rows) == 50, f"Expected 50 long dev claims, got {len(dev_rows)}"
    DEV_BENCH_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in dev_rows), encoding="utf-8")
    return dev_rows


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dev_rows = _build_dev50()

    # Step 1: Generate or load 50 Target AR trajectories (T=0.7, top-p=0.9, max_new_tokens=128)
    if DEV_TRAJ_PATH.exists():
        traj_by_cid = {str(r["claim_id"]): r for r in (json.loads(l) for l in DEV_TRAJ_PATH.read_text(encoding="utf-8").splitlines() if l.strip())}
    else:
        traj_by_cid = {}

    if len(traj_by_cid) < 50:
        pair = QwenPair(TARGET_SNAP, None, local_files_only=True, dtype="bfloat16")
        lines = []
        for idx, row in enumerate(dev_rows):
            cid = str(row["claim_id"])
            g = pair.generate(
                row["prompt"], mode="B0-S", max_new_tokens=128, temperature=0.7, top_p=0.9,
                seed=42 + idx, cache_strategy="kv_cache", distribution_mode="fp32_fast",
            )
            rec = {"claim_id": cid, "token_ids": g.token_ids}
            traj_by_cid[cid] = rec
            lines.append(json.dumps(rec) + "\n")
        DEV_TRAJ_PATH.write_text("".join(lines), encoding="utf-8")
        del pair
        torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained(TARGET_SNAP, local_files_only=True)
    target = AutoModelForCausalLM.from_pretrained(TARGET_SNAP, dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()
    n_t_layers = len(target.model.layers)

    # Cache per-claim target distributions (on CPU fp16/fp32 sparse? Store top-p p_mat indices or full fp32? 50 claims * ~95 prefixes = 4750 * 151k -> 2.8 GB RAM, fits easily in CPU RAM!)
    claim_cache: list[dict[str, Any]] = []
    for row in dev_rows:
        cid = str(row["claim_id"])
        token_ids = list(traj_by_cid[cid]["token_ids"])
        g_len = len(token_ids)
        ser_full = format_qwen_prompt(tok, row["prompt"])
        enc = tok(ser_full, return_offsets_mapping=True, add_special_tokens=False)
        p_t_ids = list(enc["input_ids"])
        offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
        spans = find_evidence_and_sentence_token_spans(ser_full, list(row.get("evidence") or []), offsets)

        # Capture tail_window=32 across ALL layers in 1 prefill pass so any W<=32, layer_spec, head_agg can be sliced cheaply!
        # Wait: causal mask depends on W ONLY for last 32 positions, which are prompt trigger tokens OUTSIDE evidence spans!
        # Still, let's compute exact per-layer importance for W=16 on all layers, PLUS store raw K and h_tail[32] per layer? No, P<=18k -> 36 layers * [32, P] softmax requires K.
        # Even simpler and 100% exact: pass TargetPrefillAttentionHook with tail_window=32 on all layers, extract per-W=[1,4,8,16,32] importance vectors [n_layers, P] right during the single target pass!
        # Each [n_layers, P] vector is only 36 * 18000 * 4B = 2.5 MB!
        all_layers = list(range(n_t_layers))
        inp_prompt = torch.tensor([p_t_ids], dtype=torch.long, device="cuda")
        with TargetPrefillAttentionHook(target, layer_indices=all_layers, tail_window=32) as hook:
            with torch.inference_mode():
                pre_out = target(input_ids=inp_prompt, use_cache=True, logits_to_keep=1)
                dyn_kv = pre_out.past_key_values

            # Extract per-layer importance under default W=16 (head=mean and head=max) plus W in {1,4,8,32} for upper50
            per_layer_w16_mean = torch.stack([hook.compute_layer_token_importance(l, dyn_kv, head_agg="mean") for l in all_layers], dim=0).cpu()
            per_layer_w16_max = torch.stack([hook.compute_layer_token_importance(l, dyn_kv, head_agg="max") for l in all_layers], dim=0).cpu()

            # Now exact W variants over upper50 using the captured h_tail slice
            u50 = resolve_layer_indices(n_t_layers, "upper50")
            imp_by_w: dict[int, torch.Tensor] = {}
            orig_captured = {l: hook._captured[l] for l in u50}
            for w in (1, 4, 8, 16, 32):
                for l in u50:
                    h, c, s = orig_captured[l]
                    hook._captured[l] = (h[:, -w:, :], c[:, -w:, :], s[:, -w:, :])
                imp_w = torch.stack([hook.compute_layer_token_importance(l, dyn_kv, head_agg="mean") for l in u50], dim=0).mean(dim=0).cpu()
                imp_by_w[w] = imp_w

            del pre_out, dyn_kv, inp_prompt

        # Target teacher-forced distributions p_mat [g_len, V]
        t_inp = torch.tensor([p_t_ids + token_ids[:-1]], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            t_logits = target(input_ids=t_inp, use_cache=False, logits_to_keep=g_len).logits[0].float()
        p_mat = processed_distribution_fast(t_logits, temperature=0.7, top_p=0.9, dtype=torch.float32).cpu()
        del t_inp, t_logits
        torch.cuda.empty_cache()

        claim_cache.append({
            "row": row,
            "token_ids": token_ids,
            "p_mat": p_mat,
            "spans": spans,
            "per_layer_w16_mean": per_layer_w16_mean,
            "per_layer_w16_max": per_layer_w16_max,
            "imp_by_w": imp_by_w,
        })

    del target
    torch.cuda.empty_cache()

    draft = AutoModelForCausalLM.from_pretrained(DRAFT_SNAP, dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()

    def eval_config(name: str, draft_prompt_fn) -> dict[str, Any]:
        exp_acc_all: list[float] = []
        draft_tok_lens: list[int] = []
        draft_chars_list: list[int] = []
        for idx, item in enumerate(claim_cache):
            row = item["row"]
            token_ids = item["token_ids"]
            g_len = len(token_ids)
            p_mat = item["p_mat"].to("cuda")

            d_prompt, ev_chars = draft_prompt_fn(idx, item)
            d_ids = tok(format_qwen_prompt(tok, d_prompt), add_special_tokens=False)["input_ids"]
            draft_tok_lens.append(len(d_ids))
            draft_chars_list.append(ev_chars)

            d_inp = torch.tensor([d_ids + token_ids[:-1]], dtype=torch.long, device="cuda")
            with torch.inference_mode():
                d_logits = draft(input_ids=d_inp, use_cache=False, logits_to_keep=g_len).logits[0].float()
            q_mat = processed_distribution_fast(d_logits, temperature=0.7, top_p=0.9, dtype=torch.float32)

            acc_vec = torch.minimum(p_mat, q_mat).sum(dim=-1)
            exp_acc_all.extend(acc_vec.cpu().tolist())
            del d_inp, d_logits, q_mat, p_mat

        torch.cuda.empty_cache()
        return {
            "config": name,
            "mean_expected_acceptance_pct": round(statistics.mean(exp_acc_all) * 100.0, 2),
            "mean_draft_prompt_tokens": round(statistics.mean(draft_tok_lens), 1),
            "mean_draft_evidence_chars": round(statistics.mean(draft_chars_list), 1),
            "n_prefixes": len(exp_acc_all),
        }

    def _attn_builder(layer_spec="upper50", w=16, head_agg="mean", sentence_agg="sum", budget_chars=10000):
        def _fn(idx, item):
            row = item["row"]
            if w != 16 and layer_spec == "upper50" and head_agg == "mean":
                imp = item["imp_by_w"][w]
            else:
                layers = resolve_layer_indices(n_t_layers, layer_spec)
                src = item["per_layer_w16_max"] if head_agg == "max" else item["per_layer_w16_mean"]
                imp = src[layers].mean(dim=0)
            _, s_scores = score_evidence_from_token_importance(imp, item["spans"], sentence_agg=sentence_agg)
            dctx = build_draft_context_from_scores(
                row, policy="sentence_budget", sentence_scores_by_passage=s_scores, budget_chars=budget_chars, min_sentences_per_passage=1
            )
            return dctx.draft_prompt, dctx.draft_evidence_chars
        return _fn

    configs = [
        ("Baseline Full", lambda i, it: (it["row"]["prompt"], sum(len(str(e.get("text", ""))) for e in it["row"].get("evidence", [])))),
        ("BM25 Sent-B10k", lambda i, it: ((r := build_draft_context(it["row"], policy="sentence_budget", budget_chars=10000, min_sentences_per_passage=1)).draft_prompt, r.draft_evidence_chars)),
        ("Random Sent-B10k", lambda i, it: ((r := build_draft_context(it["row"], policy="random_sentence_budget", budget_chars=10000, min_sentences_per_passage=1, seed=42 + i)).draft_prompt, r.draft_evidence_chars)),
        ("SPADE Default (upper50, W=16, mean-head, sum, B=10k)", _attn_builder("upper50", 16, "mean", "sum", 10000)),
        ("Norm: sentence_agg=mean", _attn_builder("upper50", 16, "mean", "mean", 10000)),
        ("Norm: sentence_agg=sqrt", _attn_builder("upper50", 16, "mean", "sqrt", 10000)),
        ("Norm: sentence_agg=max", _attn_builder("upper50", 16, "mean", "max", 10000)),
        ("Tail Window W=1", _attn_builder("upper50", 1, "mean", "sum", 10000)),
        ("Tail Window W=4", _attn_builder("upper50", 4, "mean", "sum", 10000)),
        ("Tail Window W=8", _attn_builder("upper50", 8, "mean", "sum", 10000)),
        ("Tail Window W=32", _attn_builder("upper50", 32, "mean", "sum", 10000)),
        ("Layers: upper25", _attn_builder("upper25", 16, "mean", "sum", 10000)),
        ("Layers: middle50", _attn_builder("middle50", 16, "mean", "sum", 10000)),
        ("Layers: all", _attn_builder("all", 16, "mean", "sum", 10000)),
        ("Layers: last1", _attn_builder("last1", 16, "mean", "sum", 10000)),
        ("Head Agg: max", _attn_builder("upper50", 16, "max", "sum", 10000)),
        ("Budget B=1k", _attn_builder("upper50", 16, "mean", "sum", 1000)),
        ("Budget B=2k", _attn_builder("upper50", 16, "mean", "sum", 2000)),
        ("Budget B=4k", _attn_builder("upper50", 16, "mean", "sum", 4000)),
        ("Budget B=8k", _attn_builder("upper50", 16, "mean", "sum", 8000)),
        ("Budget B=16k", _attn_builder("upper50", 16, "mean", "sum", 16000)),
    ]

    results = [eval_config(name, fn) for name, fn in configs]
    report = {
        "n_dev_claims": len(dev_rows),
        "disjoint_verified_from": str(TEST_PATH),
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for r in results:
        print(f"{r['config']:52s} | E[1-TV]={r['mean_expected_acceptance_pct']:5.2f}% | tok={r['mean_draft_prompt_tokens']:6.1f}")


if __name__ == "__main__":
    main()
