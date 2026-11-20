"""Fast 3-claim GPU smoke check for new KV-Sparse (SnapKV-style) and Prompt-Lookup Decoding (PLD) baselines."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

PYTHON_BIN = "/home/stackops/miniconda3/bin/python"
TARGET_PATH = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DRAFT_PATH = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"


def _stats(run_dir: Path) -> dict:
    t_rows = [json.loads(l) for l in (run_dir / "trial_metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    tok_rows = [json.loads(l) for l in (run_dir / "token_traces.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    r_rows = [json.loads(l) for l in (run_dir / "round_traces.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    tot_toks = sum(r["output_tokens"] for r in t_rows)
    tot_e2e = sum(r["prompt_prefill_s"] + r.get("draft_prefill_s", 0.0) + r["decode_s"] + r.get("context_selection_s", 0.0) for r in t_rows)
    n_eval = sum(1 for tk in tok_rows if tk.get("status") in {"accepted", "rejected"})
    n_eval_acc = sum(1 for tk in tok_rows if tk.get("status") == "accepted")
    n_prop = sum(r.get("proposed_token_count", 0) for r in r_rows)
    n_acc = sum(r.get("accepted_draft_tokens", 0) for r in r_rows)
    return {
        "n_claims": len(t_rows),
        "e2e_tok_s": round(tot_toks / tot_e2e, 2) if tot_e2e > 0 else 0.0,
        "alpha_eval_pct": round((n_eval_acc / n_eval) * 100.0, 2) if n_eval else 0.0,
        "alpha_prop_pct": round((n_acc / n_prop) * 100.0, 2) if n_prop else 0.0,
        "mean_draft_tokens": round(sum(r.get("draft_prompt_tokens", 0.0) for r in t_rows) / len(t_rows), 1),
        "draft_kv_peak_mb": round(sum(r.get("draft_kv_peak_bytes", 0.0) for r in t_rows) / len(t_rows) / (1024 * 1024), 2),
        "context_sel_ms": round(sum(r.get("context_selection_s", 0.0) for r in t_rows) * 1000.0 / len(t_rows), 2),
    }


def main() -> None:
    bench_rows = [json.loads(l) for l in Path("outputs/benchmarks/retrieved-heldout300.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    long3 = [r for r in bench_rows if sum(len(e.get("text", "")) for e in r.get("evidence", [])) > 10000][:3]
    tmp_dir = Path("outputs/runs/_smoke_v6")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    smoke_jsonl = tmp_dir / "smoke3_long.jsonl"
    smoke_jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in long3), encoding="utf-8")

    # 1. KV-Sparse
    cmd_kv = [
        PYTHON_BIN, "scripts/run_baselines.py",
        "--mode", "B2",
        "--input", str(smoke_jsonl),
        "--target", TARGET_PATH,
        "--draft", DRAFT_PATH,
        "--draft-context-policy", "kv_sparse",
        "--draft-kv-budget", "1776",
        "--draft-length", "4",
        "--cache-strategy", "kv_cache",
        "--draft-runtime", "flex_compile_reduce_overhead",
        "--distribution-mode", "fp32_fast",
        "--temperature", "0.7",
        "--top-p", "0.9",
        "--max-new-tokens", "128",
        "--seed", "42",
        "--global-warmup", "1",
        "--run-id", "smoke-kvsparse",
        "--output-root", str(tmp_dir),
    ]
    subprocess.run(cmd_kv, check=True)
    kv_st = _stats(tmp_dir / "smoke-kvsparse")
    print("SMOKE KV-Sparse:", json.dumps(kv_st))

    # 2. PLD
    cmd_pld = [
        PYTHON_BIN, "scripts/run_baselines.py",
        "--mode", "PLD",
        "--input", str(smoke_jsonl),
        "--target", TARGET_PATH,
        "--draft-length", "4",
        "--cache-strategy", "kv_cache",
        "--distribution-mode", "fp32_fast",
        "--temperature", "0.7",
        "--top-p", "0.9",
        "--max-new-tokens", "128",
        "--seed", "42",
        "--global-warmup", "1",
        "--run-id", "smoke-pld",
        "--output-root", str(tmp_dir),
    ]
    subprocess.run(cmd_pld, check=True)
    pld_st = _stats(tmp_dir / "smoke-pld")
    print("SMOKE PLD:", json.dumps(pld_st))

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
