"""Qualitative win/loss case study between SPADE (Attention) and BM25 sentence selection.
Analyzes paired claims on retrieved-heldout300 to identify specific failure and success patterns.
"""
import json
import statistics
from pathlib import Path

def main():
    bench_file = Path("outputs/benchmarks/retrieved-heldout300.jsonl")
    spade_dir = Path("outputs/runs/b1-4-heldout300-attn-sent-b10k")
    bm25_dir = Path("outputs/runs/b1-4-heldout300-bm25-sent-b10k")

    bench_rows = {json.loads(l)["claim_id"]: json.loads(l) for l in bench_file.read_text(encoding="utf-8").splitlines() if l.strip()}
    spade_trials = {x["claim_id"]: x for x in [json.loads(l) for l in (spade_dir / "trial_metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]}
    bm25_trials = {x["claim_id"]: x for x in [json.loads(l) for l in (bm25_dir / "trial_metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]}

    # Round traces to compute per-claim acceptance
    spade_rounds = [json.loads(l) for l in (spade_dir / "round_traces.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    bm25_rounds = [json.loads(l) for l in (bm25_dir / "round_traces.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    def get_claim_acc(r_list):
        m = {}
        for r in r_list:
            cid = r["claim_id"]
            if cid not in m:
                m[cid] = [0, 0]
            m[cid][0] += r.get("accepted_draft_tokens", 0)
            m[cid][1] += r.get("proposed_token_count", 0)
        return {cid: (acc/tot if tot > 0 else 0) for cid, (acc, tot) in m.items()}

    spade_acc = get_claim_acc(spade_rounds)
    bm25_acc = get_claim_acc(bm25_rounds)

    common_cids = set(spade_acc.keys()) & set(bm25_acc.keys()) & set(bench_rows.keys())
    print(f"Total overlapping claims: {len(common_cids)}")

    diffs = []
    for cid in common_cids:
        delta_acc = spade_acc[cid] - bm25_acc[cid]
        diffs.append((delta_acc, cid))

    diffs.sort(reverse=True)
    top_wins = diffs[:5]
    top_losses = diffs[-5:]

    print("\n================ TOP 5 WINS FOR ATTENTION OVER BM25 ================")
    for delta, cid in top_wins:
        row = bench_rows[cid]
        print(f"\nClaim ID: {cid} | Delta Acceptance: {delta*100:+.2f} pp (SPADE: {spade_acc[cid]*100:.1f}%, BM25: {bm25_acc[cid]*100:.1f}%)")
        print(f"Claim: {row.get('claim')}")
        print(f"Number of retrieved passages: {len(row.get('evidence', []))}")

    print("\n================ TOP 5 LOSSES FOR ATTENTION VS BM25 ================")
    for delta, cid in reversed(top_losses):
        row = bench_rows[cid]
        print(f"\nClaim ID: {cid} | Delta Acceptance: {delta*100:+.2f} pp (SPADE: {spade_acc[cid]*100:.1f}%, BM25: {bm25_acc[cid]*100:.1f}%)")
        print(f"Claim: {row.get('claim')}")
        print(f"Number of retrieved passages: {len(row.get('evidence', []))}")

if __name__ == "__main__":
    main()
