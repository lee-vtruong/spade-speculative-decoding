import json
from pathlib import Path
import statistics

runs_dir = Path("outputs/runs")
methods = {
    "Baseline Full": "b1-4-heldout300-baseline-full",
    "Selective Top-2": "b1-4-heldout300-selective-topk2",
    "BM25 Sent-B8k": "b1-4-heldout300-bm25-sent-b8k",
    "BM25 Sent-B10k": "b1-4-heldout300-bm25-sent-b10k",
    "SPADE": "b1-4-heldout300-attn-sent-b10k",
    "Random Sent-B10k": "b1-4-heldout300-random-sent-b10k",
    "SPADE NoCoverage": "b1-4-heldout300-attn-sent-b10k-nocoverage",
}

print("=== EXACT CONTEXT COMPRESSION & CHARACTER RETENTION ===")
for m_name, r_id in methods.items():
    t_file = runs_dir / r_id / "trial_metrics.jsonl"
    if not t_file.exists():
        continue
    char_ratios = []
    tok_ratios = []
    full_chars_list = []
    draft_chars_list = []
    draft_tok_list = []
    targ_tok_list = []
    with t_file.open("r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            fc = float(t.get("full_evidence_chars", 0))
            dc = float(t.get("draft_evidence_chars", fc))
            ttok = float(t.get("prompt_tokens", 0))
            dtok = float(t.get("draft_prompt_tokens", ttok))
            full_chars_list.append(fc)
            draft_chars_list.append(dc)
            draft_tok_list.append(dtok)
            targ_tok_list.append(ttok)
            if fc > 0:
                char_ratios.append(dc / fc)
            if ttok > 0:
                tok_ratios.append(dtok / ttok)

    mean_fc = statistics.mean(full_chars_list) if full_chars_list else 0
    mean_dc = statistics.mean(draft_chars_list) if draft_chars_list else 0
    mean_char_ratio = statistics.mean(char_ratios) * 100 if char_ratios else 100
    mean_tok_ratio = statistics.mean(tok_ratios) * 100 if tok_ratios else 100
    mean_draft_tok = statistics.mean(draft_tok_list) if draft_tok_list else 0
    mean_targ_tok = statistics.mean(targ_tok_list) if targ_tok_list else 0
    print(f"{m_name:20s} | Full Chars: {mean_fc:6.0f} -> Draft Chars: {mean_dc:6.0f} (-{100-mean_char_ratio:4.1f}%) | Tokens: {mean_targ_tok:5.0f} -> {mean_draft_tok:5.0f} (-{100-mean_tok_ratio:4.1f}%) | N={len(full_chars_list)}")
