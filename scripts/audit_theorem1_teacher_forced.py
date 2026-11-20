"""Teacher-forced fixed-prefix exactness and Monte Carlo distribution audit for Theorem 1 (SPADE vs Target)."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from spade.adaptive.context_selection import build_draft_context_from_scores
from spade.adaptive.target_attention_selector import (
    TargetPrefillAttentionHook,
    find_evidence_and_sentence_token_spans,
    resolve_layer_indices,
    score_evidence_from_token_importance,
)
from spade.decoding.autoregressive import format_qwen_prompt
from spade.decoding.runtime_opt import processed_distribution_fast
from spade.decoding.speculative import exact_speculative_round

TARGET_SNAP = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DRAFT_SNAP = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
BENCH_PATH = Path("outputs/benchmarks/retrieved-heldout300.jsonl")
AR_PREDS = Path("outputs/runs/b1-4-heldout300-b0s/predictions.jsonl")
OUT_PATH = Path("outputs/analysis/v6_revisions/theorem1_teacher_forced_audit.json")


def _chi2_sf_approx(chi2: float, df: int) -> float:
    """Wilson-Hilferty chi-square survival approximation (pure stdlib math)."""
    if df <= 0:
        return 1.0
    if chi2 <= 0:
        return 1.0
    k = float(df)
    z = ((chi2 / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    bench = [json.loads(l) for l in BENCH_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    ar_by_cid = {str(r["claim_id"]): r for r in (json.loads(l) for l in AR_PREDS.read_text(encoding="utf-8").splitlines() if l.strip())}
    long_rows = [r for r in bench if sum(len(e.get("text", "")) for e in r.get("evidence", [])) > 10000][:25]

    tok = AutoTokenizer.from_pretrained(TARGET_SNAP, local_files_only=True)
    target = AutoModelForCausalLM.from_pretrained(TARGET_SNAP, dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()
    draft = AutoModelForCausalLM.from_pretrained(DRAFT_SNAP, dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()
    t_layers = resolve_layer_indices(len(target.model.layers), "upper50")

    analytical_tvs: list[float] = []
    analytical_kls: list[float] = []
    high_div_prefixes: list[tuple[float, torch.Tensor, torch.Tensor]] = []

    for row in long_rows:
        cid = str(row["claim_id"])
        token_ids = list(ar_by_cid[cid]["token_ids"])
        if len(token_ids) < 2:
            continue
        g_len = len(token_ids)
        ser_full = format_qwen_prompt(tok, row["prompt"])
        enc = tok(ser_full, return_offsets_mapping=True, add_special_tokens=False)
        p_t_ids = list(enc["input_ids"])
        offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]

        # Step 1: Extract SPADE attention scores on clean target prompt prefill
        inp_prompt = torch.tensor([p_t_ids], dtype=torch.long, device="cuda")
        with TargetPrefillAttentionHook(target, layer_indices=t_layers, tail_window=16) as hook:
            with torch.inference_mode():
                pre_out = target(input_ids=inp_prompt, use_cache=True, logits_to_keep=1)
            imp, _ = hook.compute_token_importance(pre_out.past_key_values)
            del pre_out, inp_prompt
        spans = find_evidence_and_sentence_token_spans(ser_full, list(row.get("evidence") or []), offsets)
        _, s_scores = score_evidence_from_token_importance(imp, spans)
        dctx = build_draft_context_from_scores(row, policy="sentence_budget", sentence_scores_by_passage=s_scores, budget_chars=10000, min_sentences_per_passage=1)
        p_d_ids = tok(format_qwen_prompt(tok, dctx.draft_prompt), add_special_tokens=False)["input_ids"]

        # Step 2: Teacher-forced forward on C_full + AR_tokens[:-1] and C_draft + AR_tokens[:-1]
        t_inp = torch.tensor([p_t_ids + token_ids[:-1]], dtype=torch.long, device="cuda")
        d_inp = torch.tensor([p_d_ids + token_ids[:-1]], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            t_logits = target(input_ids=t_inp, use_cache=False, logits_to_keep=g_len).logits[0].float()
            d_logits = draft(input_ids=d_inp, use_cache=False, logits_to_keep=g_len).logits[0].float()
        del t_inp, d_inp

        p_mat = processed_distribution_fast(t_logits, temperature=0.7, top_p=0.9, dtype=torch.float32).to(torch.float64)
        q_mat = processed_distribution_fast(d_logits, temperature=0.7, top_p=0.9, dtype=torch.float32).to(torch.float64)

        # Analytical induced 1-step marginal emission distribution P_SPADE(x)
        overlap = torch.minimum(p_mat, q_mat)
        sum_overlap = overlap.sum(dim=-1, keepdim=True)
        res_num = torch.clamp(p_mat - q_mat, min=0.0)
        res_den = res_num.sum(dim=-1, keepdim=True).clamp(min=1e-18)
        p_spade_mat = overlap + (1.0 - sum_overlap) * (res_num / res_den)
        # Re-check positions where p == q identically
        eq_mask = (1.0 - sum_overlap).squeeze(-1) <= 1e-15
        p_spade_mat[eq_mask] = p_mat[eq_mask]

        tv_analytical = 0.5 * torch.abs(p_spade_mat - p_mat).sum(dim=-1)
        analytical_tvs.extend(tv_analytical.cpu().tolist())

        # KL(p || P_SPADE) over active support of p
        pos_mask = p_mat > 1e-15
        kl_vals = torch.zeros_like(tv_analytical)
        for pos in range(g_len):
            pm = pos_mask[pos]
            kl_vals[pos] = (p_mat[pos, pm] * torch.log(p_mat[pos, pm] / p_spade_mat[pos, pm].clamp(min=1e-18))).sum()
        analytical_kls.extend(kl_vals.cpu().tolist())

        # Collect candidate high-divergence prefixes (TV(p, q) > 0.15)
        tv_pq = 0.5 * torch.abs(p_mat - q_mat).sum(dim=-1)
        for pos in range(g_len):
            val = float(tv_pq[pos].item())
            if val > 0.15:
                high_div_prefixes.append((val, p_mat[pos].detach().cpu(), q_mat[pos].detach().cpu()))

        torch.cuda.empty_cache()

    del target, draft
    torch.cuda.empty_cache()

    # Step 3: Empirical Monte Carlo Check via ACTUAL exact_speculative_round AND vectorized M=10,000 draws on top 50 high-TV prefixes
    high_div_prefixes.sort(key=lambda x: -x[0])
    top50 = high_div_prefixes[:50]
    rng = torch.Generator(device="cpu").manual_seed(12345)

    M = 10000
    emp_tv_spade_ar: list[float] = []
    emp_tv_ar1_ar2: list[float] = []
    chi2_pvals: list[float] = []
    exact_fn_tvs: list[float] = []

    for idx_p, (tv_pq_val, p, q) in enumerate(top50):
        # Active support for clean multinomial / chi2
        nz = torch.nonzero((p + q) > 0, as_tuple=False).squeeze(-1)
        p_sub = p[nz]
        q_sub = q[nz]

        # Direct AR 10,000 draws (x2 independent runs for null empirical TV reference)
        draws_ar1 = torch.multinomial(p_sub, M, replacement=True, generator=rng)
        draws_ar2 = torch.multinomial(p_sub, M, replacement=True, generator=rng)

        # Speculative sampling 10,000 draws from (q, p)
        prop_s = torch.multinomial(q_sub, M, replacement=True, generator=rng)
        ratio = torch.clamp(p_sub[prop_s] / q_sub[prop_s].clamp(min=1e-18), max=1.0)
        u = torch.rand(M, generator=rng, dtype=torch.float64)
        acc_mask = u < ratio
        res = torch.clamp(p_sub - q_sub, min=0.0)
        res_sum = res.sum()
        if res_sum > 0:
            res = res / res_sum
        else:
            res = p_sub
        repl_draws = torch.multinomial(res, M, replacement=True, generator=rng)
        draws_spade = torch.where(acc_mask, prop_s, repl_draws)

        k_supp = len(nz)
        hist_spade = torch.bincount(draws_spade, minlength=k_supp).double()
        hist_ar1 = torch.bincount(draws_ar1, minlength=k_supp).double()
        hist_ar2 = torch.bincount(draws_ar2, minlength=k_supp).double()

        emp_tv_spade_ar.append(float(0.5 * torch.abs(hist_spade / M - hist_ar1 / M).sum().item()))
        emp_tv_ar1_ar2.append(float(0.5 * torch.abs(hist_ar1 / M - hist_ar2 / M).sum().item()))

        # Chi-square GOF on categories with expected count >= 5 under p_sub * M
        exp_cnt = p_sub * M
        valid = exp_cnt >= 5.0
        if int(valid.sum().item()) >= 2:
            obs_v = hist_spade[valid]
            exp_v = exp_cnt[valid]
            # Lump tail if needed
            if not bool(valid.all()):
                obs_tail = hist_spade[~valid].sum()
                exp_tail = exp_cnt[~valid].sum()
                if exp_tail >= 1.0:
                    obs_v = torch.cat([obs_v, obs_tail.unsqueeze(0)])
                    exp_v = torch.cat([exp_v, exp_tail.unsqueeze(0)])
            chi2_stat = float((((obs_v - exp_v) ** 2) / exp_v).sum().item())
            df = int(len(obs_v)) - 1
            chi2_pvals.append(_chi2_sf_approx(chi2_stat, df))

        # Verify actual `exact_speculative_round` function on first 10 high-TV prefixes (2,000 calls each)
        if idx_p < 10:
            n_real = 2000
            props = torch.multinomial(q_sub, n_real, replacement=True, generator=rng).tolist()
            cnt_real = torch.zeros(k_supp, dtype=torch.float64)
            for tok_id in props:
                rr = exact_speculative_round([int(tok_id)], [q_sub], [p_sub], generator=rng, remaining=1)
                cnt_real[rr.committed[0]] += 1.0
            exact_fn_tvs.append(float(0.5 * torch.abs(cnt_real / n_real - p_sub).sum().item()))

    report = {
        "n_claims": len(long_rows),
        "n_prefixes": len(analytical_tvs),
        "analytical_1step_exactness": {
            "mean_tv_spade_vs_target": statistics.mean(analytical_tvs),
            "max_tv_spade_vs_target": max(analytical_tvs),
            "mean_kl_target_spade": statistics.mean(analytical_kls),
            "max_kl_target_spade": max(analytical_kls),
        },
        "monte_carlo_high_divergence_audit": {
            "n_high_tv_prefixes": len(top50),
            "draws_per_prefix": M,
            "mean_proposal_tv_pq": round(statistics.mean(x[0] for x in top50), 4),
            "mean_empirical_tv_spade_vs_ar1": round(statistics.mean(emp_tv_spade_ar), 5),
            "mean_empirical_tv_ar1_vs_ar2_null": round(statistics.mean(emp_tv_ar1_ar2), 5),
            "chi2_n_tested": len(chi2_pvals),
            "chi2_pass_rate_alpha05": round(sum(1 for pv in chi2_pvals if pv >= 0.05) / max(1, len(chi2_pvals)), 4),
            "min_chi2_p_value": round(min(chi2_pvals), 4) if chi2_pvals else None,
            "exact_speculative_round_10prefix_2k_mean_tv": round(statistics.mean(exact_fn_tvs), 5) if exact_fn_tvs else None,
        },
    }
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
