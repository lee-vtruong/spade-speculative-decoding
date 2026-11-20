import json
import statistics
from collections import Counter

ar_preds = [json.loads(l) for l in open('/home/stackops/whale/spade-speculative-decoding/outputs/runs/b1-4-heldout300-b0s/predictions.jsonl')]
spade_preds = [json.loads(l) for l in open('/home/stackops/whale/spade-speculative-decoding/outputs/runs/b1-4-heldout300-attn-sent-b10k/predictions.jsonl')]
full_preds = [json.loads(l) for l in open('/home/stackops/whale/spade-speculative-decoding/outputs/runs/b1-4-heldout300-baseline-full/predictions.jsonl')]

print(f"Loaded: AR={len(ar_preds)}, SPADE={len(spade_preds)}, Full={len(full_preds)}")

ar_lens = [len(x.get("raw_text", "")) for x in ar_preds]
spade_lens = [len(x.get("raw_text", "")) for x in spade_preds]
full_lens = [len(x.get("raw_text", "")) for x in full_preds]

print(f"AR text char len:    mean={statistics.mean(ar_lens):.1f}, std={statistics.stdev(ar_lens):.1f}")
print(f"SPADE text char len: mean={statistics.mean(spade_lens):.1f}, std={statistics.stdev(spade_lens):.1f}")
print(f"Full text char len:  mean={statistics.mean(full_lens):.1f}, std={statistics.stdev(full_lens):.1f}")

# Compare output tokens count
ar_toks = [x.get("output_tokens", 0) for x in ar_preds]
spade_toks = [x.get("output_tokens", 0) for x in spade_preds]
full_toks = [x.get("output_tokens", 0) for x in full_preds]

print(f"AR output tokens:    mean={statistics.mean(ar_toks):.1f}, std={statistics.stdev(ar_toks):.1f}")
print(f"SPADE output tokens: mean={statistics.mean(spade_toks):.1f}, std={statistics.stdev(spade_toks):.1f}")
print(f"Full output tokens:  mean={statistics.mean(full_toks):.1f}, std={statistics.stdev(full_toks):.1f}")

# Word 1-gram distribution
def get_words(preds):
    words = []
    for p in preds:
        words.extend(p.get("raw_text", "").lower().split())
    return Counter(words)

c_ar = get_words(ar_preds)
c_spade = get_words(spade_preds)
c_full = get_words(full_preds)

print(f"Total emitted words: AR={sum(c_ar.values())}, SPADE={sum(c_spade.values())}, Full={sum(c_full.values())}")
print(f"Unique vocab count:  AR={len(c_ar)}, SPADE={len(c_spade)}, Full={len(c_full)}")

all_vocab = set(c_ar.keys()) | set(c_spade.keys()) | set(c_full.keys())
tot_ar = sum(c_ar.values())
tot_spade = sum(c_spade.values())
tot_full = sum(c_full.values())

tv_spade_ar = 0.5 * sum(abs(c_spade.get(w, 0)/tot_spade - c_ar.get(w, 0)/tot_ar) for w in all_vocab)
tv_full_ar = 0.5 * sum(abs(c_full.get(w, 0)/tot_full - c_ar.get(w, 0)/tot_ar) for w in all_vocab)
tv_spade_full = 0.5 * sum(abs(c_spade.get(w, 0)/tot_spade - c_full.get(w, 0)/tot_full) for w in all_vocab)

print(f"1-Gram Total Variation Distance (TV):")
print(f"  TV(SPADE, Target AR):   {tv_spade_ar:.4f}")
print(f"  TV(Full-SD, Target AR): {tv_full_ar:.4f}")
print(f"  TV(SPADE, Full-SD):     {tv_spade_full:.4f}")

# Verdict classification agreement
def get_verdicts(preds):
    return {p["claim_id"]: (p.get("parsed") or {}).get("verdict", "Unknown") for p in preds}

v_ar = get_verdicts(ar_preds)
v_spade = get_verdicts(spade_preds)
v_full = get_verdicts(full_preds)

cids = sorted(set(v_ar.keys()) & set(v_spade.keys()) & set(v_full.keys()))
match_spade_ar = sum(1 for c in cids if v_spade[c] == v_ar[c]) / len(cids)
match_full_ar = sum(1 for c in cids if v_full[c] == v_ar[c]) / len(cids)
match_spade_full = sum(1 for c in cids if v_spade[c] == v_full[c]) / len(cids)

print(f"\nVerdict Exact Agreement Rate (N={len(cids)}):")
print(f"  SPADE vs Target AR:   {match_spade_ar*100:.2f}%")
print(f"  Full-SD vs Target AR: {match_full_ar*100:.2f}%")
print(f"  SPADE vs Full-SD:     {match_spade_full*100:.2f}%")

from collections import defaultdict
vd = defaultdict(lambda: defaultdict(int))
for c in cids:
    vd['AR'][v_ar[c]] += 1
    vd['SPADE'][v_spade[c]] += 1
    vd['Full'][v_full[c]] += 1
print("\nVerdict breakdown:")
for m in ['AR', 'Full', 'SPADE']:
    print(m, dict(vd[m]))
