import json
import statistics
from pathlib import Path

def analyze_bench(path_str):
    p = Path(path_str)
    if not p.exists():
        print(f"Missing {p}")
        return
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").split("\n") if line.strip()]
    chars = []
    gt_10k = 0
    gt_8k = 0
    gt_16k = 0
    
    # Also check token lengths if available or estimate from characters (approx 4 chars/tok)
    for r in rows:
        ev_text = ""
        if "evidence" in r:
            if isinstance(r["evidence"], list):
                ev_text = " ".join(e.get("text", "") if isinstance(e, dict) else str(e) for e in r["evidence"])
            else:
                ev_text = str(r["evidence"])
        elif "prompt" in r and isinstance(r["prompt"], dict) and "evidence" in r["prompt"]:
            ev_text = str(r["prompt"]["evidence"])
        else:
            ev_text = str(r.get("prompt", ""))
        
        c_len = len(ev_text)
        chars.append(c_len)
        if c_len > 8000:
            gt_8k += 1
        if c_len > 10000:
            gt_10k += 1
        if c_len > 16000:
            gt_16k += 1

    sorted_chars = sorted(chars)
    n = len(chars)
    p10 = sorted_chars[int(0.10 * n)]
    p25 = sorted_chars[int(0.25 * n)]
    p50 = sorted_chars[int(0.50 * n)]
    p75 = sorted_chars[int(0.75 * n)]
    p90 = sorted_chars[int(0.90 * n)]

    print(f"=== {p.name} ({n} claims) ===")
    print(f"Evidence Chars: Mean={statistics.mean(chars):.1f}, Std={statistics.stdev(chars):.1f}, Median={p50}, Min={min(chars)}, Max={max(chars)}")
    print(f"Percentiles: p10={p10}, p25={p25}, p50={p50}, p75={p75}, p90={p90}")
    print(f"Evidence > 8k chars:  {gt_8k}/{n} ({gt_8k/n*100:.2f}%)")
    print(f"Evidence > 10k chars: {gt_10k}/{n} ({gt_10k/n*100:.2f}%)")
    print(f"Evidence > 16k chars: {gt_16k}/{n} ({gt_16k/n*100:.2f}%)")
    print()

def analyze_predictions(pred_path_str):
    p = Path(pred_path_str)
    if not p.exists():
        print(f"Missing predictions: {p}")
        return
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").split("\n") if line.strip()]
    out_lens = []
    prompt_lens = []
    for r in rows:
        if "output_tokens" in r:
            out_lens.append(r["output_tokens"])
        elif "generated_tokens" in r:
            out_lens.append(len(r["generated_tokens"]) if isinstance(r["generated_tokens"], list) else r["generated_tokens"])
        elif "prediction" in r:
            # estimate or count words/chars
            out_lens.append(len(r["prediction"].split()))
        if "prompt_tokens" in r:
            prompt_lens.append(r["prompt_tokens"])
    
    if out_lens:
        sorted_lens = sorted(out_lens)
        n = len(out_lens)
        print(f"=== Predictions in {p.parent.name} ({n} samples) ===")
        print(f"Output Length: Mean={statistics.mean(out_lens):.1f}, Std={statistics.stdev(out_lens):.1f}, Median={statistics.median(out_lens)}, Min={min(out_lens)}, Max={max(out_lens)}, p90={sorted_lens[int(0.9*n)]}")
    if prompt_lens:
        sorted_p = sorted(prompt_lens)
        n = len(prompt_lens)
        print(f"Prompt Length (tokens): Mean={statistics.mean(prompt_lens):.1f}, Std={statistics.stdev(prompt_lens):.1f}, Median={statistics.median(prompt_lens)}, Min={min(prompt_lens)}, Max={max(prompt_lens)}, p90={sorted_p[int(0.9*n)]}")
    print()

if __name__ == "__main__":
    analyze_bench("outputs/benchmarks/retrieved-heldout300.jsonl")
    analyze_bench("outputs/benchmarks/dev-oracle-heldout400.jsonl")
    analyze_bench("outputs/benchmarks/dev-oracle-pilot100.jsonl")

    # Analyze predictions from the main run
    analyze_predictions("outputs/runs/b1-4-heldout300-attn-sent-b10k/predictions.jsonl")
    analyze_predictions("outputs/runs/b1-4-heldout300-baseline-full/predictions.jsonl")
    analyze_predictions("outputs/runs/b1-4-oracle400-baseline-full/predictions.jsonl")
