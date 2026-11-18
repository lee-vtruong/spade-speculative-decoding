"""Reusable audit, metric extraction, sensitivity selection, and paired cluster bootstrap utilities.

All functions operate purely on offline JSON/JSONL artifacts and require no GPU or model weights.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from spade.benchmark.evaluation import LABEL_MAP, classification_metrics


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def audit_report_json(report_path: Path) -> dict[str, Any]:
    """Audit internal arithmetic and metadata consistency of a unified benchmark report JSON.

    Does not modify the input file.
    """
    rows = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected non-empty list in {report_path}")

    checks: list[dict[str, Any]] = []
    commits = set()
    pytorch_versions = set()
    transformers_versions = set()

    for row in rows:
        method = row.get("Method", row.get("Run ID", "unknown"))
        claims = int(row.get("Claims", 0))
        total_tokens = int(row.get("Total Tokens", 0))
        target_rounds = int(row.get("Target Rounds", 0))
        decode_per_claim = float(row.get("Decode/Claim (s)", 0.0))
        e2e_per_claim = float(row.get("E2E/Claim (s)", 0.0))
        tp_dec = float(row.get("Throughput Decode (tok/s)", 0.0))
        tp_e2e = float(row.get("Throughput E2E (tok/s)", 0.0))
        ms_per_tok = float(row.get("Latency/Token (ms)", 0.0))
        fwd_red = float(row.get("Target Fwd Reduc", 0.0))
        draft_ms = float(row.get("Draft Lat/Rnd (ms)", 0.0))
        draft_pct = float(row.get("Draft % Decode", 0.0))

        commits.add(row.get("Git Commit"))
        pytorch_versions.add(row.get("PyTorch"))
        transformers_versions.add(row.get("Transformers"))

        method_errors: list[str] = []

        if row.get("Git Dirty") is not False:
            method_errors.append(f"git_dirty is {row.get('Git Dirty')!r}, expected False")

        if claims <= 0 or total_tokens <= 0:
            method_errors.append("claims and total_tokens must be positive")
        else:
            exp_mean_tok = round(total_tokens / claims, 2)
            if abs(exp_mean_tok - float(row.get("Mean Tokens", -1))) > 0.011:
                method_errors.append(f"Mean Tokens mismatch: {row.get('Mean Tokens')} vs {exp_mean_tok}")

            if decode_per_claim > 0:
                exp_tp_dec = total_tokens / (decode_per_claim * claims)
                if abs(exp_tp_dec - tp_dec) > 0.06:
                    method_errors.append(f"Throughput Decode mismatch: {tp_dec} vs {exp_tp_dec:.2f}")
                exp_ms_tok = (decode_per_claim * claims / total_tokens) * 1000.0
                if abs(exp_ms_tok - ms_per_tok) > 0.06:
                    method_errors.append(f"Latency/Token mismatch: {ms_per_tok} vs {exp_ms_tok:.2f}")

            if e2e_per_claim > 0:
                exp_tp_e2e = total_tokens / (e2e_per_claim * claims)
                if abs(exp_tp_e2e - tp_e2e) > 0.06:
                    method_errors.append(f"Throughput E2E mismatch: {tp_e2e} vs {exp_tp_e2e:.2f}")

            if target_rounds > 0:
                exp_fwd_red = total_tokens / target_rounds
                if abs(exp_fwd_red - fwd_red) > 0.015:
                    method_errors.append(f"Target Fwd Reduc mismatch: {fwd_red} vs {exp_fwd_red:.2f}")
                if decode_per_claim > 0:
                    exp_draft_pct = ((draft_ms * target_rounds / 1000.0) / (decode_per_claim * claims)) * 100.0
                    if abs(exp_draft_pct - draft_pct) > 0.25:
                        method_errors.append(f"Draft % Decode mismatch: {draft_pct} vs {exp_draft_pct:.1f}")

        n_prop = int(row.get("N_prop", 0) or 0)
        if n_prop > 0:
            n_acc = int(row.get("N_acc", 0))
            n_res = int(row.get("N_res", 0))
            n_bon = int(row.get("N_bon", 0))
            n_eval = int(row.get("N_eval", 0))

            if n_acc + n_res + n_bon != total_tokens:
                method_errors.append(
                    f"Token conservation failed: {n_acc}+{n_res}+{n_bon}={n_acc + n_res + n_bon} != {total_tokens}"
                )
            if not row.get("Accounting Exact"):
                method_errors.append("Accounting Exact flag is not True")
            if n_eval != n_acc + n_res:
                method_errors.append(f"N_eval ({n_eval}) != N_acc + N_res ({n_acc + n_res}); possible double-counting")

            exp_alpha_eval = round((n_acc / n_eval) * 100.0, 2) if n_eval else 0.0
            if abs(exp_alpha_eval - float(row.get("Alpha_eval (%)", -1))) > 0.011:
                method_errors.append(f"Alpha_eval mismatch: {row.get('Alpha_eval (%)')} vs {exp_alpha_eval}")

            exp_alpha_prop = round((n_acc / n_prop) * 100.0, 2)
            if abs(exp_alpha_prop - float(row.get("Alpha_prop (%)", -1))) > 0.011:
                method_errors.append(f"Alpha_prop mismatch: {row.get('Alpha_prop (%)')} vs {exp_alpha_prop}")

            alpha_ev = float(row.get("Alpha_ev_overlap (%)", -1))
            alpha_non = float(row.get("Alpha_non_overlap (%)", -1))
            alpha_eval_val = float(row.get("Alpha_eval (%)", -1))
            lo, hi = min(alpha_ev, alpha_non), max(alpha_ev, alpha_non)
            if not (lo - 0.02 <= alpha_eval_val <= hi + 0.02):
                method_errors.append(
                    f"Weighted-average bound violated: alpha_eval={alpha_eval_val} not in [{lo}, {hi}]"
                )

            exp_prop_comm = round(n_prop / total_tokens, 3)
            if abs(exp_prop_comm - float(row.get("Prop/Committed", -1))) > 0.002:
                method_errors.append(f"Prop/Committed mismatch: {row.get('Prop/Committed')} vs {exp_prop_comm}")

        checks.append({
            "method": method,
            "run_id": row.get("Run ID"),
            "passed": len(method_errors) == 0,
            "errors": method_errors,
        })

    global_errors: list[str] = []
    if len(commits) != 1:
        global_errors.append(f"Inconsistent git commits across methods: {sorted(commits)}")
    if len(pytorch_versions) != 1:
        global_errors.append(f"Inconsistent PyTorch versions across methods: {sorted(pytorch_versions)}")
    if len(transformers_versions) != 1:
        global_errors.append(f"Inconsistent Transformers versions across methods: {sorted(transformers_versions)}")

    all_passed = (not global_errors) and all(c["passed"] for c in checks)
    return {
        "report_path": str(report_path),
        "methods_checked": len(checks),
        "git_commit": next(iter(commits)) if len(commits) == 1 else sorted(commits),
        "pytorch": next(iter(pytorch_versions)) if len(pytorch_versions) == 1 else sorted(pytorch_versions),
        "transformers": next(iter(transformers_versions)) if len(transformers_versions) == 1 else sorted(transformers_versions),
        "global_errors": global_errors,
        "method_checks": checks,
        "all_passed": all_passed,
    }


def _extract_gold_verdict(pred_row: dict[str, Any]) -> str | None:
    ev = pred_row.get("evaluation") or {}
    for key in ("label", "gold_label", "verdict"):
        val = ev.get(key)
        if isinstance(val, str) and val in LABEL_MAP:
            return LABEL_MAP[val]
    return None


def _extract_pred_verdict(pred_row: dict[str, Any]) -> str:
    parsed = pred_row.get("parsed")
    if isinstance(parsed, dict):
        v = parsed.get("verdict")
        if isinstance(v, str) and v in LABEL_MAP:
            return LABEL_MAP[v]
    return "__INVALID__"


def extract_claim_level_metrics(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Aggregate per-claim metrics from raw trial, round, token, and prediction JSONL files."""
    trials = load_jsonl(run_dir / "trial_metrics.jsonl")
    rounds = load_jsonl(run_dir / "round_traces.jsonl")
    tokens = load_jsonl(run_dir / "token_traces.jsonl")
    preds = load_jsonl(run_dir / "predictions.jsonl")

    claims: dict[str, dict[str, Any]] = {}
    for t in trials:
        cid = str(t["claim_id"])
        # Use trial_index == 0 for canonical claim-level pairing when multiple trials exist
        if t.get("trial_index", 0) != 0 and cid in claims:
            continue
        prefill_s = float(t.get("prompt_prefill_s", 0.0)) + float(t.get("draft_prefill_s", 0.0))
        decode_s = float(t.get("decode_s", 0.0))
        claims[cid] = {
            "claim_id": cid,
            "output_tokens": int(t.get("output_tokens", 0)),
            "prompt_tokens": int(t.get("prompt_tokens", 0)),
            "decode_s": decode_s,
            "prefill_s": prefill_s,
            "e2e_s": decode_s + prefill_s,
            "draft_s": float(t.get("draft_generation_s", 0.0)),
            "verify_s": float(t.get("target_verification_s", 0.0)),
            "correction_s": float(t.get("correction_s", 0.0)),
            "parse_success": bool(t.get("parse_success", False)),
            "citation_valid": False,
            "target_rounds": 0,
            "n_acc": 0,
            "n_res": 0,
            "n_bon": 0,
            "n_prop": 0,
            "n_eval": 0,
            "n_eval_acc": 0,
            "n_ev_eval": 0,
            "n_ev_acc": 0,
            "n_non_ev_eval": 0,
            "n_non_ev_acc": 0,
        }

    for p in preds:
        cid = str(p["claim_id"])
        if cid in claims:
            claims[cid]["citation_valid"] = bool(p.get("citation_id_validity", False))

    for r in rounds:
        cid = str(r["claim_id"])
        if cid not in claims:
            continue
        c = claims[cid]
        c["target_rounds"] += 1
        c["n_acc"] += int(r.get("accepted_draft_tokens", 0))
        c["n_prop"] += int(r.get("proposed_token_count", 0))
        if r.get("rejection_position") is not None:
            c["n_res"] += 1
        if r.get("additional_target_token"):
            c["n_bon"] += 1

    for tok in tokens:
        status = tok.get("status")
        if status not in {"accepted", "rejected"}:
            continue
        cid = str(tok["claim_id"])
        if cid not in claims:
            continue
        c = claims[cid]
        is_acc = 1 if status == "accepted" else 0
        c["n_eval"] += 1
        c["n_eval_acc"] += is_acc
        overlap = float((tok.get("evidence_overlap") or {}).get("lexical_overlap", 0.0))
        if overlap > 0.0:
            c["n_ev_eval"] += 1
            c["n_ev_acc"] += is_acc
        else:
            c["n_non_ev_eval"] += 1
            c["n_non_ev_acc"] += is_acc

    return claims


def summarize_and_audit_run(run_dir: Path, reference_row: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute complete efficiency, quality, and audit summary from a raw run directory."""
    env = load_json(run_dir / "environment.json")
    cfg = load_json(run_dir / "config_resolved.json")
    metrics = load_json(run_dir / "metrics.json")
    trials = load_jsonl(run_dir / "trial_metrics.jsonl")
    rounds = load_jsonl(run_dir / "round_traces.jsonl")
    tokens = load_jsonl(run_dir / "token_traces.jsonl")
    preds = load_jsonl(run_dir / "predictions.jsonl")

    mode = str(cfg.get("mode", "unknown"))
    is_spec = mode in {"B1", "B2", "M4"}
    is_hf = mode in {"HF", "HF-Assisted"}

    total_claims = len(trials)
    total_output_tokens = sum(int(t.get("output_tokens", 0)) for t in trials)
    total_decode_s = sum(float(t.get("decode_s", 0.0)) for t in trials)
    total_prefill_s = sum(
        float(t.get("prompt_prefill_s", 0.0)) + float(t.get("draft_prefill_s", 0.0))
        for t in trials
    )
    total_e2e_s = total_decode_s + total_prefill_s
    total_draft_s = sum(float(t.get("draft_generation_s", 0.0)) for t in trials)
    total_verify_s = sum(float(t.get("target_verification_s", 0.0)) for t in trials)
    total_rounds = len(rounds)

    throughput_decode = total_output_tokens / total_decode_s if total_decode_s > 0 else 0.0
    throughput_e2e = total_output_tokens / total_e2e_s if total_e2e_s > 0 else 0.0
    latency_decode_ms_tok = (total_decode_s / total_output_tokens) * 1000.0 if total_output_tokens > 0 else 0.0
    latency_e2e_ms_tok = (total_e2e_s / total_output_tokens) * 1000.0 if total_output_tokens > 0 else 0.0
    draft_ms_rnd = (total_draft_s / total_rounds) * 1000.0 if total_rounds > 0 else 0.0
    verify_ms_rnd = (total_verify_s / total_rounds) * 1000.0 if total_rounds > 0 else 0.0
    draft_pct_decode = (total_draft_s / total_decode_s) * 100.0 if total_decode_s > 0 else 0.0
    verify_pct_decode = (total_verify_s / total_decode_s) * 100.0 if total_decode_s > 0 else 0.0
    target_fwd_reduction = total_output_tokens / total_rounds if total_rounds > 0 else 1.0

    # Verdict classification metrics from predictions
    gold_labels: list[str] = []
    pred_labels: list[str] = []
    for p in preds:
        g = _extract_gold_verdict(p)
        if g is not None:
            gold_labels.append(g)
            pred_labels.append(_extract_pred_verdict(p))
    cls_metrics = classification_metrics(gold_labels, pred_labels) if gold_labels else {"accuracy": None, "macro_f1": None}

    audit_errors: list[str] = []
    if metrics and len(preds) != int(metrics.get("examples", -1)):
        audit_errors.append(f"predictions count ({len(preds)}) != metrics.examples ({metrics.get('examples')})")
    if len(preds) != total_claims and cfg.get("trials", 1) == 1:
        audit_errors.append(f"predictions count ({len(preds)}) != trial_metrics count ({total_claims})")

    if rounds:
        sum_round_draft = sum(float(r.get("draft_latency_s", 0.0)) for r in rounds)
        sum_round_verify = sum(float(r.get("verification_latency_s", 0.0)) for r in rounds)
        if abs(sum_round_draft - total_draft_s) > 1e-3:
            audit_errors.append(
                f"Draft time mismatch: sum(rounds)={sum_round_draft:.6f}s vs sum(trials)={total_draft_s:.6f}s"
            )
        if abs(sum_round_verify - total_verify_s) > 1e-3:
            audit_errors.append(
                f"Verify time mismatch: sum(rounds)={sum_round_verify:.6f}s vs sum(trials)={total_verify_s:.6f}s"
            )
        if not is_hf:
            round_committed = sum(int(r.get("committed_output_tokens", 0)) for r in rounds)
            if round_committed != total_output_tokens:
                audit_errors.append(
                    f"Committed tokens mismatch: sum(rounds)={round_committed} != sum(trials)={total_output_tokens}"
                )

    n_acc = n_res = n_bon = n_prop = n_eval = n_ev_eval = n_ev_acc = n_non_ev_eval = n_non_ev_acc = 0
    alpha_eval = alpha_prop = alpha_ev = alpha_non_ev = prop_per_committed = None
    accounting_exact = True

    if is_spec and rounds:
        n_acc = sum(int(r.get("accepted_draft_tokens", 0)) for r in rounds)
        n_prop = sum(int(r.get("proposed_token_count", 0)) for r in rounds)
        n_res = sum(1 for r in rounds if r.get("rejection_position") is not None)
        n_bon = sum(1 for r in rounds if r.get("additional_target_token"))
        accounting_exact = (n_acc + n_res + n_bon == total_output_tokens)
        if not accounting_exact:
            audit_errors.append(
                f"Token conservation failed: N_acc({n_acc})+N_res({n_res})+N_bon({n_bon}) != Total({total_output_tokens})"
            )

        eval_tokens = [tok for tok in tokens if tok.get("status") in {"accepted", "rejected"}]
        n_eval = len(eval_tokens)
        n_eval_acc = sum(1 for tok in eval_tokens if tok.get("status") == "accepted")
        n_eval_rej = sum(1 for tok in eval_tokens if tok.get("status") == "rejected")

        if n_eval_acc != n_acc:
            audit_errors.append(f"Token trace accepted ({n_eval_acc}) != round trace accepted ({n_acc})")
        if n_eval_rej != n_res:
            audit_errors.append(f"Token trace rejected ({n_eval_rej}) != round trace rejections ({n_res})")
        if n_eval != n_acc + n_res:
            audit_errors.append(f"N_eval ({n_eval}) != N_acc + N_res ({n_acc + n_res})")

        ev_tokens = [tok for tok in eval_tokens if float((tok.get("evidence_overlap") or {}).get("lexical_overlap", 0.0)) > 0.0]
        non_ev_tokens = [tok for tok in eval_tokens if float((tok.get("evidence_overlap") or {}).get("lexical_overlap", 0.0)) == 0.0]
        n_ev_eval = len(ev_tokens)
        n_ev_acc = sum(1 for tok in ev_tokens if tok.get("status") == "accepted")
        n_non_ev_eval = len(non_ev_tokens)
        n_non_ev_acc = sum(1 for tok in non_ev_tokens if tok.get("status") == "accepted")

        if n_ev_eval + n_non_ev_eval != n_eval or n_ev_acc + n_non_ev_acc != n_eval_acc:
            audit_errors.append("Evidence overlap partition does not sum exactly to N_eval / N_acc")

        alpha_eval = (n_eval_acc / n_eval) * 100.0 if n_eval > 0 else 0.0
        alpha_prop = (n_acc / n_prop) * 100.0 if n_prop > 0 else 0.0
        alpha_ev = (n_ev_acc / n_ev_eval) * 100.0 if n_ev_eval > 0 else 0.0
        alpha_non_ev = (n_non_ev_acc / n_non_ev_eval) * 100.0 if n_non_ev_eval > 0 else 0.0
        prop_per_committed = n_prop / total_output_tokens if total_output_tokens > 0 else 0.0

    summary = {
        "run_id": cfg.get("run_id", run_dir.name),
        "mode": mode,
        "draft_length": cfg.get("draft_length"),
        "dynamic_k": bool(cfg.get("dynamic_k", False)),
        "enable_biasing": cfg.get("enable_biasing", mode == "M4"),
        "adaptive_entropy": not bool(cfg.get("no_entropy_gating", False)) if mode == "M4" else None,
        "beta0": cfg.get("beta0", 1.5 if mode == "M4" else 0.0),
        "claims": total_claims,
        "total_tokens": total_output_tokens,
        "mean_tokens": round(total_output_tokens / total_claims, 2) if total_claims else 0.0,
        "throughput_e2e_tok_s": round(throughput_e2e, 2),
        "throughput_decode_tok_s": round(throughput_decode, 2),
        "e2e_s_per_claim": round(total_e2e_s / total_claims, 3) if total_claims else 0.0,
        "decode_s_per_claim": round(total_decode_s / total_claims, 3) if total_claims else 0.0,
        "latency_e2e_ms_per_tok": round(latency_e2e_ms_tok, 2),
        "latency_decode_ms_per_tok": round(latency_decode_ms_tok, 2),
        "target_rounds": total_rounds,
        "target_rounds_per_claim": round(total_rounds / total_claims, 2) if total_claims else 0.0,
        "target_fwd_reduction": round(target_fwd_reduction, 2),
        "draft_latency_ms_per_round": round(draft_ms_rnd, 2),
        "draft_pct_decode": round(draft_pct_decode, 2),
        "verification_latency_ms_per_round": round(verify_ms_rnd, 2),
        "verify_pct_decode": round(verify_pct_decode, 2),
        "n_acc": n_acc,
        "n_res": n_res,
        "n_bon": n_bon,
        "n_prop": n_prop,
        "n_eval": n_eval,
        "n_ev_eval": n_ev_eval,
        "n_ev_acc": n_ev_acc,
        "n_non_ev_eval": n_non_ev_eval,
        "n_non_ev_acc": n_non_ev_acc,
        "acceptance_eval_pct": round(alpha_eval, 2) if alpha_eval is not None else None,
        "acceptance_prop_pct": round(alpha_prop, 2) if alpha_prop is not None else None,
        "evidence_overlap_acceptance_pct": round(alpha_ev, 2) if alpha_ev is not None else None,
        "non_overlap_acceptance_pct": round(alpha_non_ev, 2) if alpha_non_ev is not None else None,
        "proposals_per_committed": round(prop_per_committed, 3) if prop_per_committed is not None else None,
        "json_valid_pct": round(float(metrics.get("json_parse_success_rate", 0.0)) * 100.0, 2),
        "citation_valid_pct": round(float(metrics.get("citation_id_validity_rate", 0.0)) * 100.0, 2),
        "verdict_accuracy_pct": round(cls_metrics["accuracy"] * 100.0, 2) if cls_metrics["accuracy"] is not None else None,
        "verdict_macro_f1": round(cls_metrics["macro_f1"], 4) if cls_metrics["macro_f1"] is not None else None,
        "accounting_exact": accounting_exact,
        "git_commit": (env.get("git_commit") or "unknown")[:7],
        "git_commit_full": env.get("git_commit"),
        "git_dirty": env.get("git_dirty"),
        "pytorch": env.get("torch"),
        "transformers": env.get("transformers"),
        "audit_passed": len(audit_errors) == 0,
        "audit_errors": audit_errors,
    }

    if reference_row is not None:
        cross_checks = [
            ("Total Tokens", summary["total_tokens"], 0),
            ("Target Rounds", summary["target_rounds"], 0),
            ("Throughput Decode (tok/s)", summary["throughput_decode_tok_s"], 0.02),
            ("Throughput E2E (tok/s)", summary["throughput_e2e_tok_s"], 0.02),
            ("N_acc", summary["n_acc"], 0),
            ("N_res", summary["n_res"], 0),
            ("N_bon", summary["n_bon"], 0),
            ("N_prop", summary["n_prop"], 0),
            ("N_eval", summary["n_eval"], 0),
        ]
        for ref_key, actual_val, tol in cross_checks:
            if ref_key in reference_row and reference_row[ref_key] is not None:
                if abs(float(reference_row[ref_key]) - float(actual_val)) > tol:
                    audit_errors.append(
                        f"Reference mismatch on {ref_key}: report={reference_row[ref_key]} vs raw={actual_val}"
                    )
        summary["audit_passed"] = len(audit_errors) == 0
        summary["audit_errors"] = audit_errors

    return summary


def _compute_pooled_metric(claim_rows: list[dict[str, Any]], metric: str) -> float:
    if metric == "throughput_e2e_tok_s":
        num = sum(c["output_tokens"] for c in claim_rows)
        den = sum(c["e2e_s"] for c in claim_rows)
        return num / den if den > 0 else 0.0
    if metric == "throughput_decode_tok_s":
        num = sum(c["output_tokens"] for c in claim_rows)
        den = sum(c["decode_s"] for c in claim_rows)
        return num / den if den > 0 else 0.0
    if metric == "latency_e2e_ms_per_tok":
        num = sum(c["e2e_s"] for c in claim_rows) * 1000.0
        den = sum(c["output_tokens"] for c in claim_rows)
        return num / den if den > 0 else 0.0
    if metric == "latency_decode_ms_per_tok":
        num = sum(c["decode_s"] for c in claim_rows) * 1000.0
        den = sum(c["output_tokens"] for c in claim_rows)
        return num / den if den > 0 else 0.0
    if metric == "acceptance_eval_pct":
        num = sum(c["n_eval_acc"] for c in claim_rows)
        den = sum(c["n_eval"] for c in claim_rows)
        return (num / den) * 100.0 if den > 0 else 0.0
    if metric == "acceptance_prop_pct":
        num = sum(c["n_acc"] for c in claim_rows)
        den = sum(c["n_prop"] for c in claim_rows)
        return (num / den) * 100.0 if den > 0 else 0.0
    if metric == "evidence_overlap_acceptance_pct":
        num = sum(c["n_ev_acc"] for c in claim_rows)
        den = sum(c["n_ev_eval"] for c in claim_rows)
        return (num / den) * 100.0 if den > 0 else 0.0
    if metric == "proposals_per_committed":
        num = sum(c["n_prop"] for c in claim_rows)
        den = sum(c["output_tokens"] for c in claim_rows)
        return num / den if den > 0 else 0.0
    if metric == "target_rounds_per_claim":
        return sum(c["target_rounds"] for c in claim_rows) / len(claim_rows) if claim_rows else 0.0
    if metric == "draft_latency_ms_per_round":
        num = sum(c["draft_s"] for c in claim_rows) * 1000.0
        den = sum(c["target_rounds"] for c in claim_rows)
        return num / den if den > 0 else 0.0
    raise ValueError(f"Unsupported metric for bootstrap: {metric}")


def paired_cluster_bootstrap(
    baseline_claims: dict[str, dict[str, Any]],
    treatment_claims: dict[str, dict[str, Any]],
    metric: str,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired cluster bootstrap resampled strictly at the claim_id level.

    Reports actual effect size, 95% percentile CI, two-sided p-value, and honest interpretation
    without p-hacking or hiding negative/null effects.
    """
    common = sorted(set(baseline_claims.keys()) & set(treatment_claims.keys()))
    if not common:
        raise ValueError("No overlapping claim_ids between baseline and treatment")

    base_list = [baseline_claims[c] for c in common]
    treat_list = [treatment_claims[c] for c in common]

    base_val = _compute_pooled_metric(base_list, metric)
    treat_val = _compute_pooled_metric(treat_list, metric)
    obs_delta = treat_val - base_val

    # Per-claim win/loss/tie count on individual claim values (where denominator > 0)
    wins = losses = ties = 0
    for cid in common:
        b_c = _compute_pooled_metric([baseline_claims[cid]], metric)
        t_c = _compute_pooled_metric([treatment_claims[cid]], metric)
        diff = t_c - b_c
        if diff > 1e-9:
            wins += 1
        elif diff < -1e-9:
            losses += 1
        else:
            ties += 1

    rng = random.Random(seed)
    n = len(common)
    boot_deltas: list[float] = []
    for _ in range(n_bootstrap):
        idxs = [rng.randrange(n) for _ in range(n)]
        b_samp = [base_list[i] for i in idxs]
        t_samp = [treat_list[i] for i in idxs]
        boot_deltas.append(_compute_pooled_metric(t_samp, metric) - _compute_pooled_metric(b_samp, metric))

    boot_deltas.sort()
    ci_lo = boot_deltas[int(0.025 * n_bootstrap)]
    ci_hi = boot_deltas[min(n_bootstrap - 1, int(0.975 * n_bootstrap))]
    boot_mean = sum(boot_deltas) / n_bootstrap
    boot_se = math.sqrt(sum((x - boot_mean) ** 2 for x in boot_deltas) / max(1, n_bootstrap - 1))

    # Standard two-sided bootstrap p-value via percentile inversion
    p_le = sum(1 for x in boot_deltas if x <= 0.0) / n_bootstrap
    p_ge = sum(1 for x in boot_deltas if x >= 0.0) / n_bootstrap
    p_two_sided = min(1.0, 2.0 * min(p_le, p_ge))

    if ci_lo > 0.0:
        interpretation = "positive_ci_excludes_zero"
    elif ci_hi < 0.0:
        interpretation = "negative_ci_excludes_zero"
    else:
        interpretation = "inconclusive_ci_crosses_zero"

    return {
        "metric": metric,
        "n_claims": n,
        "n_bootstrap": n_bootstrap,
        "baseline": round(base_val, 4),
        "treatment": round(treat_val, 4),
        "delta": round(obs_delta, 4),
        "bootstrap_mean_delta": round(boot_mean, 4),
        "bootstrap_se": round(boot_se, 4),
        "ci_95_low": round(ci_lo, 4),
        "ci_95_high": round(ci_hi, 4),
        "p_value_two_sided": round(p_two_sided, 4),
        "claim_wins": wins,
        "claim_losses": losses,
        "claim_ties": ties,
        "interpretation": interpretation,
    }


def select_best_by_throughput(
    rows: list[dict[str, Any]],
    min_json_valid_pct: float = 99.0,
    min_citation_valid_pct: float = 99.0,
) -> dict[str, Any]:
    """Select optimal configuration strictly by measured E2E throughput (or lowest E2E ms/tok).

    Never selects solely by highest acceptance rate. Reports both throughput-optimal and
    acceptance-optimal configurations for full transparency.
    """
    if not rows:
        raise ValueError("No rows provided for selection")

    eligible = [
        r for r in rows
        if float(r.get("json_valid_pct", 0.0)) >= min_json_valid_pct
        and float(r.get("citation_valid_pct", 0.0)) >= min_citation_valid_pct
        and bool(r.get("audit_passed", True))
    ]
    pool = eligible if eligible else rows

    # Primary objective: highest measured end-to-end throughput, tie-break by lowest E2E ms/token
    best_e2e = max(
        pool,
        key=lambda r: (float(r["throughput_e2e_tok_s"]), -float(r["latency_e2e_ms_per_tok"])),
    )

    acc_candidates = [r for r in pool if r.get("acceptance_eval_pct") is not None]
    best_acc = max(acc_candidates, key=lambda r: float(r["acceptance_eval_pct"])) if acc_candidates else None

    return {
        "selection_criterion": "max_measured_e2e_throughput_with_quality_guardrails",
        "quality_guardrails": {
            "min_json_valid_pct": min_json_valid_pct,
            "min_citation_valid_pct": min_citation_valid_pct,
            "require_audit_passed": True,
        },
        "eligible_count": len(eligible),
        "total_count": len(rows),
        "selected": {
            "run_id": best_e2e["run_id"],
            "variant": best_e2e.get("variant"),
            "beta0": best_e2e.get("beta0"),
            "adaptive_entropy": best_e2e.get("adaptive_entropy"),
            "throughput_e2e_tok_s": best_e2e["throughput_e2e_tok_s"],
            "latency_e2e_ms_per_tok": best_e2e["latency_e2e_ms_per_tok"],
            "acceptance_eval_pct": best_e2e.get("acceptance_eval_pct"),
            "evidence_overlap_acceptance_pct": best_e2e.get("evidence_overlap_acceptance_pct"),
            "json_valid_pct": best_e2e.get("json_valid_pct"),
            "citation_valid_pct": best_e2e.get("citation_valid_pct"),
        },
        "highest_acceptance_for_comparison": (
            {
                "run_id": best_acc["run_id"],
                "variant": best_acc.get("variant"),
                "beta0": best_acc.get("beta0"),
                "acceptance_eval_pct": best_acc.get("acceptance_eval_pct"),
                "throughput_e2e_tok_s": best_acc["throughput_e2e_tok_s"],
                "latency_e2e_ms_per_tok": best_acc["latency_e2e_ms_per_tok"],
            }
            if best_acc is not None
            else None
        ),
        "selected_differs_from_highest_acceptance": (
            best_acc is not None and best_e2e["run_id"] != best_acc["run_id"]
        ),
    }
