import argparse
import json
from pathlib import Path

def print_run_metrics(run_dir: Path):
    if not run_dir.exists():
        print(f"Run dir does not exist: {run_dir}")
        return

    round_file = run_dir / "round_traces.jsonl"
    trial_file = run_dir / "trial_metrics.jsonl"

    total_acc = 0
    total_draft = 0
    total_rounds = 0
    total_committed = 0

    if round_file.exists():
        with round_file.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                total_acc += r.get("accepted_draft_tokens", 0)
                total_draft += r.get("proposed_token_count", 0)
                total_rounds += 1
                total_committed += r.get("committed_output_tokens", 0)

    acc_rate = (total_acc / total_draft * 100) if total_draft > 0 else 0
    comm_per_rnd = (total_committed / total_rounds) if total_rounds > 0 else 0

    total_out_toks = 0
    total_e2e_s = 0
    draft_prompt_toks = []
    targ_prompt_toks = []
    n_claims = 0

    if trial_file.exists():
        with trial_file.open("r", encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                n_claims += 1
                total_out_toks += t.get("output_tokens", 0)
                sel_s = t.get("context_selection_s", 0)
                p_pref_s = t.get("prompt_prefill_s", 0)
                d_pref_s = t.get("draft_prefill_s", 0)
                dec_s = t.get("decode_s", 0)
                total_e2e_s += (sel_s + p_pref_s + d_pref_s + dec_s)
                draft_prompt_toks.append(t.get("draft_prompt_tokens", t.get("prompt_tokens", 0)))
                targ_prompt_toks.append(t.get("prompt_tokens", 0))

    e2e_thpt = total_out_toks / total_e2e_s if total_e2e_s > 0 else 0
    e2e_ms_tok = (total_e2e_s / total_out_toks * 1000) if total_out_toks > 0 else 0
    mean_dp = sum(draft_prompt_toks) / len(draft_prompt_toks) if draft_prompt_toks else 0
    mean_tp = sum(targ_prompt_toks) / len(targ_prompt_toks) if targ_prompt_toks else 0

    print(f"=== METRICS: {run_dir.name} (N={n_claims} claims) ===")
    print(f"Acceptance Rate (alpha): {acc_rate:.2f}%")
    print(f"Committed / Round:      {comm_per_rnd:.3f}")
    print(f"Draft Prompt Tokens:    {mean_dp:.1f} / {mean_tp:.1f} (-{(1-mean_dp/mean_tp)*100:.1f}%)" if mean_tp > 0 else "—")
    print(f"Total Output Tokens:    {total_out_toks}")
    print(f"Total E2E Seconds:      {total_e2e_s:.2f}s")
    print(f"E2E Latency:            {e2e_ms_tok:.3f} ms/tok")
    print(f"E2E Throughput:         {e2e_thpt:.2f} tok/s")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_id", type=str)
    args = p.parse_args()
    print_run_metrics(Path("outputs/runs") / args.run_id)
