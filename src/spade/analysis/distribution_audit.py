"""M4.1 Distribution-Level Evidence Biasing Audit (CPU & GPU compatible).

Evaluates per-prefix distributional divergence between target p, unbiased draft q_base,
and evidence-biased draft q_biased on fixed teacher-forced prefixes:
  - Total Variation distance TV(p, q) and exact per-prefix expected acceptance 1 - TV(p, q)
  - Jensen-Shannon divergence JS(p, q) in nats
  - Target & draft probability mass on evidence candidate sets
  - Draft entropy, entropy-gating factor, n-gram match order, candidate count, and micro-overhead
  - Subgroup stratification and claim-clustered paired bootstrap (strictly by claim_id)

Strictly separates deterministic per-prefix expected acceptance (1 - TV) from
observed stochastic acceptance measured in speculative rollout runs.
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from spade.adaptive.biasing import (
    EvidenceBiasingConfig,
    apply_evidence_bias,
    compute_entropy,
    compute_gating_and_boost,
)
from spade.adaptive.evidence_indexer import EvidenceIndexer
from spade.decoding.sampling import expected_acceptance, processed_distribution, total_variation
from spade.instrumentation.token_analysis import categorize_token, evidence_overlap

TV_EPS = 1e-7


@dataclass(frozen=True)
class AuditBiasSpec:
    """Single biasing configuration evaluated at every fixed prefix."""
    name: str
    variant: str  # "static" or "entropy"
    beta0: float
    config: EvidenceBiasingConfig


def build_default_audit_specs(
    betas: list[float],
    variants: list[str],
    order_decay: float = 0.5,
    entropy_threshold: float = 2.0,
    max_boost_cap: float = 3.0,
) -> list[AuditBiasSpec]:
    specs: list[AuditBiasSpec] = []
    for variant in variants:
        if variant not in {"static", "entropy"}:
            raise ValueError(f"Unsupported variant: {variant}")
        for b in betas:
            if b <= 0.0:
                continue
            btag = f"b{b:.2f}".replace(".", "p")
            cfg = EvidenceBiasingConfig(
                boost_weight=float(b),
                order_decay=order_decay,
                adaptive_entropy=(variant == "entropy"),
                entropy_threshold=entropy_threshold,
                max_boost_cap=max_boost_cap,
                gating_mode="sigmoid" if variant == "entropy" else "none",
            )
            specs.append(AuditBiasSpec(name=f"{variant}-{btag}", variant=variant, beta0=float(b), config=cfg))
    return specs


def build_v2_audit_specs() -> list[AuditBiasSpec]:
    """Pre-specified M4.1 v2 comparison suite testing the 2 targeted modifications.

    Includes original baselines (orig-entropy-b1p50, orig-static-b0p50) alongside:
      - Mod 1 only: precision-filtered n-gram matching (min_match_order in {2,3}, max_candidates=3)
      - Mod 1 + Mod 2: precision-filtered + band-pass entropy [0.4, 2.5] + draft-support mass gate
    """
    specs: list[AuditBiasSpec] = [
        AuditBiasSpec(
            name="orig-entropy-b1p50",
            variant="orig_entropy",
            beta0=1.5,
            config=EvidenceBiasingConfig(boost_weight=1.5, adaptive_entropy=True, gating_mode="sigmoid"),
        ),
        AuditBiasSpec(
            name="orig-static-b0p50",
            variant="orig_static",
            beta0=0.5,
            config=EvidenceBiasingConfig(boost_weight=0.5, adaptive_entropy=False, gating_mode="none"),
        ),
    ]
    # Mod 1 only: precision-filtered (min_order=2 or 3, max_candidates=3, no entropy gating)
    for min_ord in (2, 3):
        for b in (0.5, 1.0, 1.5):
            btag = f"b{b:.2f}".replace(".", "p")
            specs.append(AuditBiasSpec(
                name=f"prec-o{min_ord}-c3-{btag}",
                variant=f"prec_o{min_ord}_c3",
                beta0=b,
                config=EvidenceBiasingConfig(
                    boost_weight=b,
                    adaptive_entropy=False,
                    gating_mode="none",
                    min_match_order=min_ord,
                    max_candidates=3,
                ),
            ))

    # Mod 1 + Mod 2: precision-filtered + band-pass entropy [0.4, 2.5] + draft support mass gate
    for min_ord, min_mass, betas in [
        (2, 0.05, (0.5, 1.0, 1.5, 2.0)),
        (2, 0.10, (1.0, 1.5, 2.0)),
        (3, 0.05, (1.0, 1.5, 2.0)),
    ]:
        mtag = f"m{int(round(min_mass * 100)):02d}"
        for b in betas:
            btag = f"b{b:.2f}".replace(".", "p")
            specs.append(AuditBiasSpec(
                name=f"bandpass-o{min_ord}-c3-{mtag}-{btag}",
                variant=f"bandpass_o{min_ord}_c3_{mtag}",
                beta0=b,
                config=EvidenceBiasingConfig(
                    boost_weight=b,
                    adaptive_entropy=True,
                    gating_mode="bandpass",
                    min_match_order=min_ord,
                    max_candidates=3,
                    entropy_low=0.4,
                    entropy_high=2.5,
                    min_draft_mass=min_mass,
                ),
            ))
    return specs


def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Compute Jensen-Shannon divergence JS(p || q) in nats, bounded in [0, ln(2)].

    JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m), where m = 0.5 * (p + q).
    Handles zero-probability support cleanly with 0 * log(0) = 0.
    """
    p64 = p.detach().to(torch.float64)
    q64 = q.detach().to(torch.float64)
    m = 0.5 * (p64 + q64)

    p_mask = p64 > 0
    q_mask = q64 > 0

    kl_pm = torch.sum(p64[p_mask] * torch.log(p64[p_mask] / m[p_mask])) if bool(p_mask.any()) else torch.tensor(0.0, dtype=torch.float64)
    kl_qm = torch.sum(q64[q_mask] * torch.log(q64[q_mask] / m[q_mask])) if bool(q_mask.any()) else torch.tensor(0.0, dtype=torch.float64)
    js = 0.5 * (kl_pm + kl_qm)
    return float(max(0.0, js.item()))


def entropy_bucket(h_nats: float) -> str:
    if h_nats < 0.5:
        return "[0.0, 0.5)"
    if h_nats < 1.0:
        return "[0.5, 1.0)"
    if h_nats < 2.0:
        return "[1.0, 2.0)"
    if h_nats < 3.0:
        return "[2.0, 3.0)"
    return "[3.0, +inf)"


def match_length_bucket(matched_order: int) -> str:
    if matched_order <= 0:
        return "0 (no match)"
    if matched_order == 1:
        return "1"
    if matched_order == 2:
        return "2"
    return "3+"


def candidate_count_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 10:
        return "4-10"
    return "11+"


def target_mass_bucket(mass: float) -> str:
    if mass <= 1e-9:
        return "0.00"
    if mass <= 0.05:
        return "(0.00, 0.05]"
    if mass <= 0.25:
        return "(0.05, 0.25]"
    if mass <= 0.50:
        return "(0.25, 0.50]"
    return "(0.50, 1.00]"


def normalize_semantic_category(token_text: str) -> str:
    raw = categorize_token(token_text)
    mapping = {
        "numeric": "numeric",
        "potential_named_entity": "named_entity",
        "citation": "citation",
        "date_related": "date",
        "punctuation": "punctuation",
        "other": "general_text",
    }
    return mapping.get(raw, "general_text")


def evaluate_prefix_biasing(
    *,
    claim_id: str,
    position: int,
    realized_token_id: int,
    realized_token_text: str,
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    evidence_indexer: EvidenceIndexer,
    evidence_texts: list[str],
    context_tokens: list[int],
    specs: list[AuditBiasSpec],
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 0.9,
) -> dict[str, Any]:
    """Evaluate target p, unbiased draft q_base, and biased draft q_biased at one fixed prefix."""
    t_logits = target_logits.detach().float()
    d_logits = draft_logits.detach().float()

    p = processed_distribution(t_logits, temperature=temperature, top_k=top_k, top_p=top_p)
    q_base = processed_distribution(d_logits, temperature=temperature, top_k=top_k, top_p=top_p)

    tv_base = total_variation(p, q_base)
    exp_acc_base = expected_acceptance(p, q_base)
    js_base = jensen_shannon_divergence(p, q_base)
    entropy_nats = compute_entropy(d_logits)

    t0_lookup = time.perf_counter()
    candidates, matched_order = evidence_indexer.get_candidate_tokens(context_tokens)
    lookup_us = (time.perf_counter() - t0_lookup) * 1e6

    cand_count = len(candidates)
    if cand_count > 0:
        cand_tensor = torch.as_tensor(tuple(candidates), dtype=torch.long, device=p.device)
        target_mass_cand = float(p[cand_tensor].sum().item())
        draft_base_mass_cand = float(q_base[cand_tensor].sum().item())
        target_argmax_in_cand = int(torch.argmax(p).item()) in candidates
        realized_in_cand = int(realized_token_id) in candidates
    else:
        target_mass_cand = 0.0
        draft_base_mass_cand = 0.0
        target_argmax_in_cand = False
        realized_in_cand = False

    ev_info = evidence_overlap(realized_token_text, evidence_texts)
    has_ev_overlap = bool(ev_info["lexical_overlap"] > 0.0)
    sem_cat = normalize_semantic_category(realized_token_text)

    config_results: dict[str, dict[str, Any]] = {}
    for spec in specs:
        cfg = spec.config
        t0_bias = time.perf_counter()
        gating, applied_boost, active_cand_tensor = compute_gating_and_boost(d_logits, candidates, matched_order, cfg)
        if applied_boost > 0.0 and active_cand_tensor is not None:
            biased_logits = d_logits.clone()
            biased_logits[active_cand_tensor] += applied_boost
            bias_us = lookup_us + (time.perf_counter() - t0_bias) * 1e6

            q_biased = processed_distribution(biased_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            tv_biased = total_variation(p, q_biased)
            exp_acc_biased = expected_acceptance(p, q_biased)
            js_biased = jensen_shannon_divergence(p, q_biased)
            draft_biased_mass = float(q_biased[active_cand_tensor].sum().item())

            config_results[spec.name] = {
                "variant": spec.variant,
                "beta0": spec.beta0,
                "is_biased": True,
                "gating_value": round(gating, 6),
                "applied_boost": round(applied_boost, 6),
                "tv_biased": tv_biased,
                "delta_tv": tv_biased - tv_base,
                "js_biased": js_biased,
                "delta_js": js_biased - js_base,
                "expected_acceptance_biased": exp_acc_biased,
                "delta_expected_acceptance": exp_acc_biased - exp_acc_base,
                "draft_biased_mass_on_candidates": draft_biased_mass,
                "bias_overhead_us": round(bias_us, 3),
            }
        else:
            bias_us = lookup_us + (time.perf_counter() - t0_bias) * 1e6
            config_results[spec.name] = {
                "variant": spec.variant,
                "beta0": spec.beta0,
                "is_biased": False,
                "gating_value": round(gating, 6),
                "applied_boost": 0.0,
                "tv_biased": tv_base,
                "delta_tv": 0.0,
                "js_biased": js_base,
                "delta_js": 0.0,
                "expected_acceptance_biased": exp_acc_base,
                "delta_expected_acceptance": 0.0,
                "draft_biased_mass_on_candidates": draft_base_mass_cand,
                "bias_overhead_us": round(bias_us, 3),
            }

    return {
        "claim_id": str(claim_id),
        "position": int(position),
        "realized_token_id": int(realized_token_id),
        "realized_token_text": realized_token_text,
        "evidence_overlap": has_ev_overlap,
        "evidence_overlap_score": float(ev_info["lexical_overlap"]),
        "semantic_category": sem_cat,
        "entropy_nats": float(entropy_nats),
        "entropy_bucket": entropy_bucket(entropy_nats),
        "matched_order": int(matched_order),
        "match_length_bucket": match_length_bucket(matched_order),
        "candidate_count": int(cand_count),
        "candidate_count_bucket": candidate_count_bucket(cand_count),
        "target_mass_on_candidates": target_mass_cand,
        "target_mass_bucket": target_mass_bucket(target_mass_cand),
        "draft_base_mass_on_candidates": draft_base_mass_cand,
        "candidate_mass_deficit": target_mass_cand - draft_base_mass_cand,
        "target_argmax_in_candidates": target_argmax_in_cand,
        "realized_token_in_candidates": realized_in_cand,
        "tv_base": tv_base,
        "js_base": js_base,
        "expected_acceptance_base": exp_acc_base,
        "lookup_us": round(lookup_us, 3),
        "configs": config_results,
    }


def _rank_average_ties(values: list[float]) -> list[float]:
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda kv: kv[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return _pearson(_rank_average_ties(xs), _rank_average_ties(ys))


def claim_clustered_bootstrap_mean(
    per_claim_sums: dict[str, tuple[float, int]],
    *,
    all_claim_ids: list[str],
    n_bootstrap: int = 10000,
    seed: int = 42,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Claim-clustered bootstrap of a pooled token-level mean delta = sum_c num_c / sum_c den_c.

    Resamples strictly at the claim_id level (never assuming independent token samples).
    """
    n_total_claims = len(all_claim_ids)
    active_claims = [c for c in all_claim_ids if per_claim_sums.get(c, (0.0, 0))[1] > 0]
    total_num = sum(per_claim_sums.get(c, (0.0, 0))[0] for c in all_claim_ids)
    total_den = sum(per_claim_sums.get(c, (0.0, 0))[1] for c in all_claim_ids)

    if total_den == 0 or not active_claims:
        return {
            "n_claims_total": n_total_claims,
            "n_claims_active": 0,
            "n_prefixes": 0,
            "mean": 0.0,
            "bootstrap_se": 0.0,
            "ci_95_low": 0.0,
            "ci_95_high": 0.0,
            "p_value_two_sided": 1.0,
            "interpretation": "insufficient_data",
            "cluster_unit": "claim_id",
        }

    obs_mean = total_num / total_den
    claim_pairs = [per_claim_sums.get(c, (0.0, 0)) for c in all_claim_ids]
    rng = random.Random(seed)
    boot_vals: list[float] = []
    for _ in range(n_bootstrap):
        b_num = 0.0
        b_den = 0
        for _ in range(n_total_claims):
            num_i, den_i = claim_pairs[rng.randrange(n_total_claims)]
            b_num += num_i
            b_den += den_i
        if b_den > 0:
            boot_vals.append(b_num / b_den)

    if not boot_vals:
        boot_vals = [obs_mean]
    boot_vals.sort()
    m = len(boot_vals)
    ci_lo = boot_vals[int(0.025 * m)]
    ci_hi = boot_vals[min(m - 1, int(0.975 * m))]
    b_mean = sum(boot_vals) / m
    b_se = math.sqrt(sum((x - b_mean) ** 2 for x in boot_vals) / max(1, m - 1))

    p_le = sum(1 for x in boot_vals if x <= 0.0) / m
    p_ge = sum(1 for x in boot_vals if x >= 0.0) / m
    p_two_sided = min(1.0, 2.0 * min(p_le, p_ge))

    if ci_lo > 0.0:
        interp = "improved_ci_excludes_zero" if higher_is_better else "degraded_ci_excludes_zero"
    elif ci_hi < 0.0:
        interp = "degraded_ci_excludes_zero" if higher_is_better else "improved_ci_excludes_zero"
    else:
        interp = "inconclusive_ci_crosses_zero"

    return {
        "n_claims_total": n_total_claims,
        "n_claims_active": len(active_claims),
        "n_prefixes": int(total_den),
        "mean": round(obs_mean, 6),
        "bootstrap_se": round(b_se, 6),
        "ci_95_low": round(ci_lo, 6),
        "ci_95_high": round(ci_hi, 6),
        "p_value_two_sided": round(p_two_sided, 4),
        "interpretation": interp,
        "cluster_unit": "claim_id",
    }


def claim_clustered_correlation(
    per_claim_xy: dict[str, list[tuple[float, float]]],
    *,
    all_claim_ids: list[str],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Claim-clustered bootstrap for Spearman and Pearson correlation between x (entropy) and y (benefit)."""
    obs_x: list[float] = []
    obs_y: list[float] = []
    active = 0
    for cid in all_claim_ids:
        pts = per_claim_xy.get(cid, [])
        if pts:
            active += 1
            for x, y in pts:
                obs_x.append(x)
                obs_y.append(y)

    if len(obs_x) < 3:
        return {
            "n_claims_active": active,
            "n_prefixes": len(obs_x),
            "spearman_r": 0.0,
            "spearman_ci_95": [0.0, 0.0],
            "spearman_p_two_sided": 1.0,
            "pearson_r": 0.0,
            "pearson_ci_95": [0.0, 0.0],
            "pearson_p_two_sided": 1.0,
            "cluster_unit": "claim_id",
        }

    obs_sp = _spearman(obs_x, obs_y)
    obs_pe = _pearson(obs_x, obs_y)

    claim_lists = [per_claim_xy.get(cid, []) for cid in all_claim_ids]
    n_c = len(all_claim_ids)
    rng = random.Random(seed)
    boot_sp: list[float] = []
    boot_pe: list[float] = []
    for _ in range(n_bootstrap):
        bx: list[float] = []
        by: list[float] = []
        for _ in range(n_c):
            for x, y in claim_lists[rng.randrange(n_c)]:
                bx.append(x)
                by.append(y)
        if len(bx) >= 3:
            boot_sp.append(_spearman(bx, by))
            boot_pe.append(_pearson(bx, by))

    def _summarize(obs: float, vals: list[float]) -> tuple[float, list[float], float]:
        if not vals:
            return round(obs, 4), [round(obs, 4), round(obs, 4)], 1.0
        vals.sort()
        m = len(vals)
        lo = vals[int(0.025 * m)]
        hi = vals[min(m - 1, int(0.975 * m))]
        p2 = min(1.0, 2.0 * min(sum(1 for v in vals if v <= 0.0) / m, sum(1 for v in vals if v >= 0.0) / m))
        return round(obs, 4), [round(lo, 4), round(hi, 4)], round(p2, 4)

    sp_r, sp_ci, sp_p = _summarize(obs_sp, boot_sp)
    pe_r, pe_ci, pe_p = _summarize(obs_pe, boot_pe)
    return {
        "n_claims_active": active,
        "n_prefixes": len(obs_x),
        "spearman_r": sp_r,
        "spearman_ci_95": sp_ci,
        "spearman_p_two_sided": sp_p,
        "pearson_r": pe_r,
        "pearson_ci_95": pe_ci,
        "pearson_p_two_sided": pe_p,
        "cluster_unit": "claim_id",
    }


def _summarize_config_slice(
    records: list[dict[str, Any]],
    cfg_name: str,
    all_claim_ids: list[str],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    n_all = len(records)
    if n_all == 0:
        return {"n_prefixes": 0}

    tv_base_mean = sum(r["tv_base"] for r in records) / n_all
    js_base_mean = sum(r["js_base"] for r in records) / n_all
    exp_base_mean = sum(r["expected_acceptance_base"] for r in records) / n_all

    biased_recs = [r for r in records if r["configs"][cfg_name]["is_biased"]]
    n_biased = len(biased_recs)

    n_tv_red = sum(1 for r in biased_recs if r["configs"][cfg_name]["delta_tv"] < -TV_EPS)
    n_tv_inc = sum(1 for r in biased_recs if r["configs"][cfg_name]["delta_tv"] > TV_EPS)
    n_tv_eq = n_biased - n_tv_red - n_tv_inc

    def _agg_claim_sums(subset: list[dict[str, Any]], key: str) -> dict[str, tuple[float, int]]:
        sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
        for r in subset:
            cid = r["claim_id"]
            sums[cid][0] += float(r["configs"][cfg_name][key])
            sums[cid][1] += 1
        return {cid: (v[0], int(v[1])) for cid, v in sums.items()}

    all_dtv = claim_clustered_bootstrap_mean(
        _agg_claim_sums(records, "delta_tv"),
        all_claim_ids=all_claim_ids,
        n_bootstrap=n_bootstrap,
        seed=seed,
        higher_is_better=False,
    )
    all_dexp = claim_clustered_bootstrap_mean(
        _agg_claim_sums(records, "delta_expected_acceptance"),
        all_claim_ids=all_claim_ids,
        n_bootstrap=n_bootstrap,
        seed=seed,
        higher_is_better=True,
    )
    all_djs = claim_clustered_bootstrap_mean(
        _agg_claim_sums(records, "delta_js"),
        all_claim_ids=all_claim_ids,
        n_bootstrap=n_bootstrap,
        seed=seed,
        higher_is_better=False,
    )

    biased_dtv = claim_clustered_bootstrap_mean(
        _agg_claim_sums(biased_recs, "delta_tv"),
        all_claim_ids=all_claim_ids,
        n_bootstrap=n_bootstrap,
        seed=seed,
        higher_is_better=False,
    )
    biased_dexp = claim_clustered_bootstrap_mean(
        _agg_claim_sums(biased_recs, "delta_expected_acceptance"),
        all_claim_ids=all_claim_ids,
        n_bootstrap=n_bootstrap,
        seed=seed,
        higher_is_better=True,
    )
    biased_djs = claim_clustered_bootstrap_mean(
        _agg_claim_sums(biased_recs, "delta_js"),
        all_claim_ids=all_claim_ids,
        n_bootstrap=n_bootstrap,
        seed=seed,
        higher_is_better=False,
    )

    tv_biased_mean = sum(r["configs"][cfg_name]["tv_biased"] for r in records) / n_all
    js_biased_mean = sum(r["configs"][cfg_name]["js_biased"] for r in records) / n_all
    exp_biased_mean = sum(r["configs"][cfg_name]["expected_acceptance_biased"] for r in records) / n_all
    mean_overhead_us = sum(r["configs"][cfg_name]["bias_overhead_us"] for r in records) / n_all
    mean_biased_overhead_us = (
        sum(r["configs"][cfg_name]["bias_overhead_us"] for r in biased_recs) / n_biased if n_biased else 0.0
    )
    mean_target_mass_biased = (
        sum(r["target_mass_on_candidates"] for r in biased_recs) / n_biased if n_biased else 0.0
    )
    mean_draft_base_mass_biased = (
        sum(r["draft_base_mass_on_candidates"] for r in biased_recs) / n_biased if n_biased else 0.0
    )
    mean_draft_biased_mass = (
        sum(r["configs"][cfg_name]["draft_biased_mass_on_candidates"] for r in biased_recs) / n_biased if n_biased else 0.0
    )
    mean_gating_biased = (
        sum(r["configs"][cfg_name]["gating_value"] for r in biased_recs) / n_biased if n_biased else 0.0
    )
    mean_boost_biased = (
        sum(r["configs"][cfg_name]["applied_boost"] for r in biased_recs) / n_biased if n_biased else 0.0
    )

    first_cfg = records[0]["configs"][cfg_name]
    return {
        "config_name": cfg_name,
        "variant": first_cfg["variant"],
        "beta0": first_cfg["beta0"],
        "n_prefixes_total": n_all,
        "n_prefixes_biased": n_biased,
        "pct_prefixes_biased": round((n_biased / n_all) * 100.0, 2),
        "n_tv_reduced": n_tv_red,
        "n_tv_increased": n_tv_inc,
        "n_tv_unchanged": n_tv_eq,
        "pct_all_tv_reduced": round((n_tv_red / n_all) * 100.0, 2),
        "pct_all_tv_increased": round((n_tv_inc / n_all) * 100.0, 2),
        "pct_biased_tv_reduced": round((n_tv_red / n_biased) * 100.0, 2) if n_biased else 0.0,
        "pct_biased_tv_increased": round((n_tv_inc / n_biased) * 100.0, 2) if n_biased else 0.0,
        "pct_biased_tv_unchanged": round((n_tv_eq / n_biased) * 100.0, 2) if n_biased else 0.0,
        "tv_base_mean": round(tv_base_mean, 6),
        "tv_biased_mean": round(tv_biased_mean, 6),
        "js_base_mean": round(js_base_mean, 6),
        "js_biased_mean": round(js_biased_mean, 6),
        "expected_acceptance_base_pct": round(exp_base_mean * 100.0, 4),
        "expected_acceptance_biased_pct": round(exp_biased_mean * 100.0, 4),
        "delta_expected_acceptance_all_pp": round(all_dexp["mean"] * 100.0, 4),
        "delta_expected_acceptance_biased_pp": round(biased_dexp["mean"] * 100.0, 4),
        "all_delta_tv_bootstrap": all_dtv,
        "all_delta_expected_acceptance_bootstrap": all_dexp,
        "all_delta_js_bootstrap": all_djs,
        "biased_delta_tv_bootstrap": biased_dtv,
        "biased_delta_expected_acceptance_bootstrap": biased_dexp,
        "biased_delta_js_bootstrap": biased_djs,
        "mean_target_mass_on_candidates_biased": round(mean_target_mass_biased, 4),
        "mean_draft_base_mass_on_candidates_biased": round(mean_draft_base_mass_biased, 4),
        "mean_draft_biased_mass_on_candidates_biased": round(mean_draft_biased_mass, 4),
        "mean_gating_value_biased": round(mean_gating_biased, 4),
        "mean_applied_boost_biased": round(mean_boost_biased, 4),
        "mean_bias_overhead_us_all": round(mean_overhead_us, 2),
        "mean_bias_overhead_us_biased": round(mean_biased_overhead_us, 2),
    }


def _subgroup_table(
    records: list[dict[str, Any]],
    cfg_name: str,
    group_key: str,
    ordered_buckets: list[str],
    all_claim_ids: list[str],
    *,
    n_bootstrap: int,
    seed: int,
    biased_only: bool = False,
) -> list[dict[str, Any]]:
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if biased_only and not r["configs"][cfg_name]["is_biased"]:
            continue
        val = r[group_key]
        label = ("evidence_overlap" if val else "non_overlap") if isinstance(val, bool) else str(val)
        by_bucket[label].append(r)

    rows: list[dict[str, Any]] = []
    seen = set()
    keys_order = [k for k in ordered_buckets if k in by_bucket] + sorted(k for k in by_bucket if k not in ordered_buckets)
    for bucket in keys_order:
        if bucket in seen:
            continue
        seen.add(bucket)
        sub = by_bucket[bucket]
        n_sub = len(sub)
        n_b = sum(1 for r in sub if r["configs"][cfg_name]["is_biased"])
        n_red = sum(1 for r in sub if r["configs"][cfg_name]["delta_tv"] < -TV_EPS)
        n_inc = sum(1 for r in sub if r["configs"][cfg_name]["delta_tv"] > TV_EPS)

        claim_sums_dexp: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
        claim_sums_dtv: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
        for r in sub:
            cid = r["claim_id"]
            claim_sums_dexp[cid][0] += float(r["configs"][cfg_name]["delta_expected_acceptance"])
            claim_sums_dexp[cid][1] += 1
            claim_sums_dtv[cid][0] += float(r["configs"][cfg_name]["delta_tv"])
            claim_sums_dtv[cid][1] += 1

        boot_dexp = claim_clustered_bootstrap_mean(
            {c: (v[0], int(v[1])) for c, v in claim_sums_dexp.items()},
            all_claim_ids=all_claim_ids,
            n_bootstrap=n_bootstrap,
            seed=seed,
            higher_is_better=True,
        )
        boot_dtv = claim_clustered_bootstrap_mean(
            {c: (v[0], int(v[1])) for c, v in claim_sums_dtv.items()},
            all_claim_ids=all_claim_ids,
            n_bootstrap=n_bootstrap,
            seed=seed,
            higher_is_better=False,
        )

        rows.append({
            "group_axis": group_key,
            "bucket": bucket,
            "biased_only": biased_only,
            "n_prefixes": n_sub,
            "n_biased": n_b,
            "pct_biased": round((n_b / n_sub) * 100.0, 2) if n_sub else 0.0,
            "pct_tv_reduced": round((n_red / n_b) * 100.0, 2) if n_b else 0.0,
            "pct_tv_increased": round((n_inc / n_b) * 100.0, 2) if n_b else 0.0,
            "tv_base_mean": round(sum(r["tv_base"] for r in sub) / n_sub, 5),
            "expected_acc_base_pct": round((sum(r["expected_acceptance_base"] for r in sub) / n_sub) * 100.0, 3),
            "delta_expected_acc_pp": round(boot_dexp["mean"] * 100.0, 4),
            "delta_expected_acc_ci95_pp": [
                round(boot_dexp["ci_95_low"] * 100.0, 4),
                round(boot_dexp["ci_95_high"] * 100.0, 4),
            ],
            "p_value_two_sided": boot_dexp["p_value_two_sided"],
            "interpretation": boot_dexp["interpretation"],
            "delta_tv_mean": boot_dtv["mean"],
            "mean_target_mass_cand": round(sum(r["target_mass_on_candidates"] for r in sub) / n_sub, 4),
            "mean_draft_base_mass_cand": round(sum(r["draft_base_mass_on_candidates"] for r in sub) / n_sub, 4),
            "mean_entropy_nats": round(sum(r["entropy_nats"] for r in sub) / n_sub, 3),
            "n_claims_active": boot_dexp["n_claims_active"],
        })
    return rows


def build_distribution_audit_report(
    records: list[dict[str, Any]],
    *,
    primary_config: str | None = None,
    sensitivity_report: dict[str, Any] | None = None,
    n_bootstrap: int = 10000,
    seed: int = 42,
    eager_draft_ms_per_step: float = 21.0,
    fused_draft_ms_per_step: float = 2.0,
    target_verify_ms_per_round: float = 26.2,
    draft_length_k: int = 4,
) -> dict[str, Any]:
    """Build complete M4.1 distribution-level audit report from per-prefix replay records."""
    if not records:
        raise ValueError("No prefix audit records provided")

    all_claim_ids = sorted({str(r["claim_id"]) for r in records})
    config_names = list(records[0]["configs"].keys())
    if not config_names:
        raise ValueError("Audit records contain no evaluated biasing configs")

    if primary_config is None or primary_config not in config_names:
        # Prefer entropy-b1p50 (the default SPADE config) if present, else first entropy, else first
        if "entropy-b1p50" in config_names:
            primary_config = "entropy-b1p50"
        elif any(c.startswith("entropy-") for c in config_names):
            primary_config = next(c for c in config_names if c.startswith("entropy-"))
        else:
            primary_config = config_names[0]

    # 1. Config-level summary across all evaluated (variant, beta_0)
    config_summaries = [
        _summarize_config_slice(records, cname, all_claim_ids, n_bootstrap=n_bootstrap, seed=seed)
        for cname in config_names
    ]
    summary_by_name = {s["config_name"]: s for s in config_summaries}
    primary_summary = summary_by_name[primary_config]

    # 2. Direct entropy vs static paired cluster bootstrap at matched beta_0
    entropy_vs_static: list[dict[str, Any]] = []
    by_beta_variant: dict[float, dict[str, str]] = defaultdict(dict)
    for s in config_summaries:
        by_beta_variant[float(s["beta0"])][s["variant"]] = s["config_name"]

    for beta0 in sorted(by_beta_variant.keys()):
        if "static" in by_beta_variant[beta0] and "entropy" in by_beta_variant[beta0]:
            st_name = by_beta_variant[beta0]["static"]
            en_name = by_beta_variant[beta0]["entropy"]
            claim_diffs: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
            claim_diffs_biased: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
            for r in records:
                cid = r["claim_id"]
                diff = r["configs"][en_name]["delta_expected_acceptance"] - r["configs"][st_name]["delta_expected_acceptance"]
                claim_diffs[cid][0] += diff
                claim_diffs[cid][1] += 1
                if r["configs"][en_name]["is_biased"] or r["configs"][st_name]["is_biased"]:
                    claim_diffs_biased[cid][0] += diff
                    claim_diffs_biased[cid][1] += 1

            boot_all = claim_clustered_bootstrap_mean(
                {c: (v[0], int(v[1])) for c, v in claim_diffs.items()},
                all_claim_ids=all_claim_ids,
                n_bootstrap=n_bootstrap,
                seed=seed,
                higher_is_better=True,
            )
            boot_biased = claim_clustered_bootstrap_mean(
                {c: (v[0], int(v[1])) for c, v in claim_diffs_biased.items()},
                all_claim_ids=all_claim_ids,
                n_bootstrap=n_bootstrap,
                seed=seed,
                higher_is_better=True,
            )
            entropy_vs_static.append({
                "beta0": beta0,
                "entropy_config": en_name,
                "static_config": st_name,
                "all_prefixes_delta_exp_acc_pp": round(boot_all["mean"] * 100.0, 4),
                "all_prefixes_ci95_pp": [round(boot_all["ci_95_low"] * 100.0, 4), round(boot_all["ci_95_high"] * 100.0, 4)],
                "all_prefixes_p_two_sided": boot_all["p_value_two_sided"],
                "all_prefixes_interpretation": boot_all["interpretation"],
                "biased_only_delta_exp_acc_pp": round(boot_biased["mean"] * 100.0, 4),
                "biased_only_ci95_pp": [round(boot_biased["ci_95_low"] * 100.0, 4), round(boot_biased["ci_95_high"] * 100.0, 4)],
                "biased_only_p_two_sided": boot_biased["p_value_two_sided"],
                "biased_only_interpretation": boot_biased["interpretation"],
            })

    # 3. Subgroup breakdowns for primary config
    subgroups = {
        "evidence_overlap": _subgroup_table(
            records, primary_config, "evidence_overlap", ["evidence_overlap", "non_overlap"],
            all_claim_ids, n_bootstrap=n_bootstrap, seed=seed,
        ),
        "entropy_bucket_all": _subgroup_table(
            records, primary_config, "entropy_bucket",
            ["[0.0, 0.5)", "[0.5, 1.0)", "[1.0, 2.0)", "[2.0, 3.0)", "[3.0, +inf)"],
            all_claim_ids, n_bootstrap=n_bootstrap, seed=seed, biased_only=False,
        ),
        "entropy_bucket_biased": _subgroup_table(
            records, primary_config, "entropy_bucket",
            ["[0.0, 0.5)", "[0.5, 1.0)", "[1.0, 2.0)", "[2.0, 3.0)", "[3.0, +inf)"],
            all_claim_ids, n_bootstrap=n_bootstrap, seed=seed, biased_only=True,
        ),
        "match_length": _subgroup_table(
            records, primary_config, "match_length_bucket", ["0 (no match)", "1", "2", "3+"],
            all_claim_ids, n_bootstrap=n_bootstrap, seed=seed,
        ),
        "candidate_count": _subgroup_table(
            records, primary_config, "candidate_count_bucket", ["0", "1", "2-3", "4-10", "11+"],
            all_claim_ids, n_bootstrap=n_bootstrap, seed=seed,
        ),
        "semantic_category": _subgroup_table(
            records, primary_config, "semantic_category",
            ["numeric", "named_entity", "citation", "date", "punctuation", "general_text"],
            all_claim_ids, n_bootstrap=n_bootstrap, seed=seed,
        ),
        "target_mass_biased": _subgroup_table(
            records, primary_config, "target_mass_bucket",
            ["0.00", "(0.00, 0.05]", "(0.05, 0.25]", "(0.25, 0.50]", "(0.50, 1.00]"],
            all_claim_ids, n_bootstrap=n_bootstrap, seed=seed, biased_only=True,
        ),
    }

    # Also compute static entropy-bucket breakdown (at same beta0 if available) to test raw entropy predictiveness
    primary_beta = float(primary_summary["beta0"])
    static_matched_name = by_beta_variant.get(primary_beta, {}).get("static")
    entropy_predictiveness: dict[str, Any] = {}
    for label, cname in [("primary", primary_config), ("static_matched", static_matched_name)]:
        if cname is None:
            continue
        per_claim_xy: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in records:
            if r["configs"][cname]["is_biased"]:
                per_claim_xy[r["claim_id"]].append(
                    (float(r["entropy_nats"]), float(r["configs"][cname]["delta_expected_acceptance"]))
                )
        entropy_predictiveness[label] = {
            "config_name": cname,
            "correlation_entropy_vs_benefit": claim_clustered_correlation(
                per_claim_xy,
                all_claim_ids=all_claim_ids,
                n_bootstrap=min(2000, n_bootstrap),
                seed=seed,
            ),
        }

    # 4. Failure-mode diagnosis: TV-increased vs TV-reduced biased prefixes + worst-20 concrete cases
    biased_primary = [r for r in records if r["configs"][primary_config]["is_biased"]]
    tv_inc_recs = [r for r in biased_primary if r["configs"][primary_config]["delta_tv"] > TV_EPS]
    tv_red_recs = [r for r in biased_primary if r["configs"][primary_config]["delta_tv"] < -TV_EPS]

    def _profile(sub: list[dict[str, Any]]) -> dict[str, Any]:
        if not sub:
            return {"count": 0}
        n = len(sub)
        return {
            "count": n,
            "mean_delta_tv": round(sum(r["configs"][primary_config]["delta_tv"] for r in sub) / n, 6),
            "mean_entropy_nats": round(sum(r["entropy_nats"] for r in sub) / n, 4),
            "mean_matched_order": round(sum(r["matched_order"] for r in sub) / n, 3),
            "mean_candidate_count": round(sum(r["candidate_count"] for r in sub) / n, 2),
            "mean_target_mass_cand": round(sum(r["target_mass_on_candidates"] for r in sub) / n, 4),
            "mean_draft_base_mass_cand": round(sum(r["draft_base_mass_on_candidates"] for r in sub) / n, 4),
            "mean_candidate_mass_deficit": round(sum(r["candidate_mass_deficit"] for r in sub) / n, 4),
            "pct_target_argmax_in_cand": round((sum(1 for r in sub if r["target_argmax_in_candidates"]) / n) * 100.0, 2),
            "pct_order_1": round((sum(1 for r in sub if r["matched_order"] == 1) / n) * 100.0, 2),
            "pct_evidence_overlap": round((sum(1 for r in sub if r["evidence_overlap"]) / n) * 100.0, 2),
        }

    worst_cases = sorted(
        tv_inc_recs,
        key=lambda r: float(r["configs"][primary_config]["delta_tv"]),
        reverse=True,
    )[:20]
    worst_cases_rows = [
        {
            "claim_id": r["claim_id"],
            "position": r["position"],
            "realized_token_text": r["realized_token_text"],
            "semantic_category": r["semantic_category"],
            "evidence_overlap": r["evidence_overlap"],
            "matched_order": r["matched_order"],
            "candidate_count": r["candidate_count"],
            "entropy_nats": round(r["entropy_nats"], 3),
            "target_mass_on_candidates": round(r["target_mass_on_candidates"], 4),
            "draft_base_mass_on_candidates": round(r["draft_base_mass_on_candidates"], 4),
            "tv_base": round(r["tv_base"], 4),
            "tv_biased": round(r["configs"][primary_config]["tv_biased"], 4),
            "delta_tv": round(r["configs"][primary_config]["delta_tv"], 4),
            "applied_boost": r["configs"][primary_config]["applied_boost"],
        }
        for r in worst_cases
    ]

    # 5. Overhead vs expected benefit break-even analysis (Requirement 6, Q5)
    # Expected committed tokens per round under geometric acceptance alpha with fixed K:
    # E[committed | alpha, K] = (1 - alpha^(K+1)) / (1 - alpha)
    def _expected_committed_per_round(alpha: float, k: int) -> float:
        a = min(max(alpha, 1e-6), 1.0 - 1e-9)
        return (1.0 - (a ** (k + 1))) / (1.0 - a)

    alpha0 = primary_summary["expected_acceptance_base_pct"] / 100.0
    alpha1 = primary_summary["expected_acceptance_biased_pct"] / 100.0
    e_comm_base = _expected_committed_per_round(alpha0, draft_length_k)
    e_comm_biased = _expected_committed_per_round(alpha1, draft_length_k)
    rel_round_reduction = (e_comm_biased - e_comm_base) / e_comm_biased if e_comm_biased > 0 else 0.0
    bias_overhead_ms_per_round = (primary_summary["mean_bias_overhead_us_all"] * draft_length_k) / 1000.0

    eager_round_ms = draft_length_k * eager_draft_ms_per_step + target_verify_ms_per_round
    fused_round_ms = draft_length_k * fused_draft_ms_per_step + target_verify_ms_per_round
    eager_expected_time_saved_ms = rel_round_reduction * eager_round_ms
    fused_expected_time_saved_ms = rel_round_reduction * fused_round_ms

    overhead_vs_benefit = {
        "primary_config": primary_config,
        "draft_length_k": draft_length_k,
        "expected_acceptance_base": round(alpha0, 6),
        "expected_acceptance_biased": round(alpha1, 6),
        "delta_expected_acceptance_pp": primary_summary["delta_expected_acceptance_all_pp"],
        "theoretical_committed_per_round_base": round(e_comm_base, 4),
        "theoretical_committed_per_round_biased": round(e_comm_biased, 4),
        "theoretical_round_reduction_pct": round(rel_round_reduction * 100.0, 4),
        "measured_bias_overhead_us_per_prefix": primary_summary["mean_bias_overhead_us_all"],
        "measured_bias_overhead_ms_per_k4_round": round(bias_overhead_ms_per_round, 4),
        "eager_baseline_round_ms": round(eager_round_ms, 2),
        "eager_expected_saved_ms_per_round": round(eager_expected_time_saved_ms, 4),
        "eager_net_gain_ms_per_round": round(eager_expected_time_saved_ms - bias_overhead_ms_per_round, 4),
        "fused_baseline_round_ms": round(fused_round_ms, 2),
        "fused_expected_saved_ms_per_round": round(fused_expected_time_saved_ms, 4),
        "fused_net_gain_ms_per_round": round(fused_expected_time_saved_ms - bias_overhead_ms_per_round, 4),
    }

    # 6. Strict separation: Per-prefix expected acceptance vs Observed stochastic acceptance
    expected_vs_observed: list[dict[str, Any]] = []
    if sensitivity_report and isinstance(sensitivity_report.get("rows"), list):
        sens_lookup: dict[tuple[str, float], dict[str, Any]] = {}
        for sr in sensitivity_report["rows"]:
            sens_lookup[(str(sr.get("variant")), round(float(sr.get("beta0", 0.0)), 4))] = sr

        for s in config_summaries:
            key = (s["variant"], round(float(s["beta0"]), 4))
            zero_key = (s["variant"], 0.0)
            sr = sens_lookup.get(key)
            sr0 = sens_lookup.get(zero_key)
            if sr is None:
                continue
            obs_alpha = sr.get("acceptance_eval_pct")
            obs_alpha0 = sr0.get("acceptance_eval_pct") if sr0 else None
            obs_delta_pp = round(obs_alpha - obs_alpha0, 3) if (obs_alpha is not None and obs_alpha0 is not None) else None
            expected_vs_observed.append({
                "config_name": s["config_name"],
                "variant": s["variant"],
                "beta0": s["beta0"],
                "per_prefix_expected_acc_base_pct": s["expected_acceptance_base_pct"],
                "per_prefix_expected_acc_biased_pct": s["expected_acceptance_biased_pct"],
                "per_prefix_delta_expected_acc_pp": s["delta_expected_acceptance_all_pp"],
                "observed_stochastic_alpha_eval_pct": obs_alpha,
                "observed_stochastic_delta_alpha_eval_pp": obs_delta_pp,
                "observed_e2e_throughput_tok_s": sr.get("throughput_e2e_tok_s"),
                "metric_separation_note": (
                    "per_prefix_expected_acceptance = 1 - TV(p,q) on fixed teacher-forced prefixes; "
                    "observed_stochastic_alpha_eval comes from separate stochastic rollout runs and is NOT interchangeable."
                ),
            })

    # 7. Per-claim % biased statistics
    per_claim_biased_pcts: list[float] = []
    by_claim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_claim[r["claim_id"]].append(r)
    for cid in all_claim_ids:
        cr = by_claim[cid]
        nb = sum(1 for r in cr if r["configs"][primary_config]["is_biased"])
        per_claim_biased_pcts.append((nb / len(cr)) * 100.0 if cr else 0.0)
    mean_claim_biased = sum(per_claim_biased_pcts) / len(per_claim_biased_pcts)
    std_claim_biased = math.sqrt(
        sum((x - mean_claim_biased) ** 2 for x in per_claim_biased_pcts) / max(1, len(per_claim_biased_pcts) - 1)
    )

    # 8. Explicit pre-registered decision gate evaluation (Keep vs Modify vs Drop)
    best_exp_cfg = max(config_summaries, key=lambda s: s["delta_expected_acceptance_all_pp"])
    best_all_boot = best_exp_cfg["all_delta_expected_acceptance_bootstrap"]
    best_pp = best_exp_cfg["delta_expected_acceptance_all_pp"]

    # Check if any clean targeted subgroup (e.g. order >= 2 or low candidate_count) has strong positive lift
    strong_subgroups = [
        row for axis_rows in (subgroups["match_length"], subgroups["candidate_count"], subgroups["target_mass_biased"])
        for row in axis_rows
        if row["interpretation"] == "improved_ci_excludes_zero" and row["delta_expected_acc_pp"] >= 0.50
    ]

    if best_all_boot["interpretation"] == "improved_ci_excludes_zero" and best_pp >= 1.0:
        gate_verdict = "KEEP_CURRENT_BIASING"
    elif strong_subgroups:
        gate_verdict = "TARGETED_MODIFICATION_CANDIDATE"
    else:
        gate_verdict = "DROP_OR_REFRAME_AS_NEGATIVE_RESULT"

    return {
        "schema_version": "m4_1_distribution_audit_v1",
        "cluster_unit": "claim_id",
        "n_claims": len(all_claim_ids),
        "n_prefixes": len(records),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "primary_config": primary_config,
        "config_summaries": config_summaries,
        "entropy_vs_static_matched_beta": entropy_vs_static,
        "audit_questions": {
            "q1_pct_prefixes_biased": {
                "pooled_pct": primary_summary["pct_prefixes_biased"],
                "n_biased": primary_summary["n_prefixes_biased"],
                "n_total": primary_summary["n_prefixes_total"],
                "per_claim_mean_pct": round(mean_claim_biased, 2),
                "per_claim_std_pct": round(std_claim_biased, 2),
            },
            "q2_pct_prefixes_tv_reduced": {
                "pct_of_all_tv_reduced": primary_summary["pct_all_tv_reduced"],
                "pct_of_all_tv_increased": primary_summary["pct_all_tv_increased"],
                "pct_of_biased_tv_reduced": primary_summary["pct_biased_tv_reduced"],
                "pct_of_biased_tv_increased": primary_summary["pct_biased_tv_increased"],
                "pct_of_biased_tv_unchanged": primary_summary["pct_biased_tv_unchanged"],
                "mean_delta_expected_acc_all_pp": primary_summary["delta_expected_acceptance_all_pp"],
                "mean_delta_expected_acc_biased_pp": primary_summary["delta_expected_acceptance_biased_pp"],
                "all_ci95_pp": [
                    round(primary_summary["all_delta_expected_acceptance_bootstrap"]["ci_95_low"] * 100.0, 4),
                    round(primary_summary["all_delta_expected_acceptance_bootstrap"]["ci_95_high"] * 100.0, 4),
                ],
                "all_p_two_sided": primary_summary["all_delta_expected_acceptance_bootstrap"]["p_value_two_sided"],
            },
            "q3_where_biasing_increases_tv": {
                "tv_increased_profile": _profile(tv_inc_recs),
                "tv_reduced_profile": _profile(tv_red_recs),
                "worst_20_cases": worst_cases_rows,
            },
            "q4_does_entropy_predict_benefit": {
                "correlations": entropy_predictiveness,
                "entropy_bucket_biased": subgroups["entropy_bucket_biased"],
                "entropy_vs_static_matched_beta": entropy_vs_static,
            },
            "q5_overhead_vs_expected_benefit": overhead_vs_benefit,
        },
        "subgroups_primary": subgroups,
        "expected_vs_observed_comparison": expected_vs_observed,
        "decision_gate": {
            "verdict": gate_verdict,
            "best_config_by_expected_acceptance": best_exp_cfg["config_name"],
            "best_delta_expected_acc_all_pp": best_pp,
            "best_delta_expected_acc_ci95_pp": [
                round(best_all_boot["ci_95_low"] * 100.0, 4),
                round(best_all_boot["ci_95_high"] * 100.0, 4),
            ],
            "best_p_two_sided": best_all_boot["p_value_two_sided"],
            "strong_positive_subgroups": strong_subgroups,
            "criteria": {
                "KEEP_CURRENT_BIASING": "All-prefix delta_expected_acc >= +1.0 pp with claim-clustered 95% CI > 0 AND net expected latency gain > biasing overhead.",
                "TARGETED_MODIFICATION_CANDIDATE": "Overall lift < +1.0 pp, BUT a clean diagnostic regime (e.g. match_order >= 2 or candidate_count <= 3) exhibits >= +0.50 pp lift with 95% CI > 0 while another identifiable regime drives TV increase.",
                "DROP_OR_REFRAME_AS_NEGATIVE_RESULT": "Neither overall nor any clean subgroup achieves meaningful positive lift (>= +0.50 pp, CI > 0), or overhead exceeds maximum expected round savings.",
            },
        },
    }
