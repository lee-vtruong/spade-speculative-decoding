import json
import sys

def analyze_categories(trace_path):
    cats = {}
    with open(trace_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get('reached_by_verifier', False):
                continue
            c = rec.get('token_category', 'other')
            if c not in cats:
                cats[c] = {'proposed': 0, 'accepted': 0}
            cats[c]['proposed'] += 1
            if rec.get('status') == 'accepted':
                cats[c]['accepted'] += 1
    res = {}
    for c, v in cats.items():
        res[c] = {
            'proposed': v['proposed'],
            'accepted': v['accepted'],
            'rate': v['accepted'] / v['proposed'] if v['proposed'] > 0 else 0
        }
    return res

b2_k4 = analyze_categories('outputs/runs/full-b2-k4-lim100/token_traces.jsonl')
m4_k4 = analyze_categories('outputs/runs/full-m4-k4-lim100/token_traces.jsonl')

b2_k6 = analyze_categories('outputs/runs/full-b2-k6-lim100/token_traces.jsonl')
m4_k6 = analyze_categories('outputs/runs/full-m4-k6-lim100/token_traces.jsonl')

print("="*85)
print(f"{'Category':24s} | {'B2 (K=4)':18s} | {'M4 (K=4)':18s} | {'Delta (K=4)':12s}")
print("="*85)
for c in sorted(b2_k4.keys()):
    b2_r = b2_k4[c]['rate']
    m4_r = m4_k4.get(c, {}).get('rate', 0)
    diff = m4_r - b2_r
    print(f"{c:24s} | {b2_r*100:6.2f}% ({b2_k4[c]['accepted']:4d}/{b2_k4[c]['proposed']:4d}) | {m4_r*100:6.2f}% ({m4_k4.get(c,{}).get('accepted',0):4d}/{m4_k4.get(c,{}).get('proposed',0):4d}) | {diff*100:+6.2f}%")

print("\n" + "="*85)
print(f"{'Category':24s} | {'B2 (K=6)':18s} | {'M4 (K=6)':18s} | {'Delta (K=6)':12s}")
print("="*85)
for c in sorted(b2_k6.keys()):
    b2_r = b2_k6[c]['rate']
    m4_r = m4_k6.get(c, {}).get('rate', 0)
    diff = m4_r - b2_r
    print(f"{c:24s} | {b2_r*100:6.2f}% ({b2_k6[c]['accepted']:4d}/{b2_k6[c]['proposed']:4d}) | {m4_r*100:6.2f}% ({m4_k6.get(c,{}).get('accepted',0):4d}/{m4_k6.get(c,{}).get('proposed',0):4d}) | {diff*100:+6.2f}%")
print("="*85)
