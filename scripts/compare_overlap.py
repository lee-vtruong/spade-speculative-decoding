import json

def analyze_evidence_overlap(trace_path):
    stats = {
        'evidence_overlap': {'proposed': 0, 'accepted': 0},
        'non_overlap': {'proposed': 0, 'accepted': 0},
    }
    with open(trace_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get('reached_by_verifier', False):
                continue
            overlap = rec.get('evidence_overlap', {}).get('lexical_overlap', 0.0)
            key = 'evidence_overlap' if overlap > 0 else 'non_overlap'
            stats[key]['proposed'] += 1
            if rec.get('status') == 'accepted':
                stats[key]['accepted'] += 1
    res = {}
    for k, v in stats.items():
        res[k] = {
            'proposed': v['proposed'],
            'accepted': v['accepted'],
            'rate': v['accepted'] / v['proposed'] if v['proposed'] > 0 else 0
        }
    return res

b2_k4 = analyze_evidence_overlap('outputs/runs/full-b2-k4-lim100/token_traces.jsonl')
m4_k4 = analyze_evidence_overlap('outputs/runs/full-m4-k4-lim100/token_traces.jsonl')

b2_k6 = analyze_evidence_overlap('outputs/runs/full-b2-k6-lim100/token_traces.jsonl')
m4_k6 = analyze_evidence_overlap('outputs/runs/full-m4-k6-lim100/token_traces.jsonl')

print("="*85)
print("EVIDENCE OVERLAP ACCEPTANCE COMPARISON")
print("="*85)
print(f"K=4 | Evidence Overlap: B2={b2_k4['evidence_overlap']['rate']*100:.2f}% ({b2_k4['evidence_overlap']['accepted']}/{b2_k4['evidence_overlap']['proposed']}) vs M4={m4_k4['evidence_overlap']['rate']*100:.2f}% ({m4_k4['evidence_overlap']['accepted']}/{m4_k4['evidence_overlap']['proposed']}) | Delta: {(m4_k4['evidence_overlap']['rate'] - b2_k4['evidence_overlap']['rate'])*100:+.2f}%")
print(f"K=4 | Non-Overlap:      B2={b2_k4['non_overlap']['rate']*100:.2f}% ({b2_k4['non_overlap']['accepted']}/{b2_k4['non_overlap']['proposed']}) vs M4={m4_k4['non_overlap']['rate']*100:.2f}% ({m4_k4['non_overlap']['accepted']}/{m4_k4['non_overlap']['proposed']}) | Delta: {(m4_k4['non_overlap']['rate'] - b2_k4['non_overlap']['rate'])*100:+.2f}%")
print("-"*85)
print(f"K=6 | Evidence Overlap: B2={b2_k6['evidence_overlap']['rate']*100:.2f}% ({b2_k6['evidence_overlap']['accepted']}/{b2_k6['evidence_overlap']['proposed']}) vs M4={m4_k6['evidence_overlap']['rate']*100:.2f}% ({m4_k6['evidence_overlap']['accepted']}/{m4_k6['evidence_overlap']['proposed']}) | Delta: {(m4_k6['evidence_overlap']['rate'] - b2_k6['evidence_overlap']['rate'])*100:+.2f}%")
print(f"K=6 | Non-Overlap:      B2={b2_k6['non_overlap']['rate']*100:.2f}% ({b2_k6['non_overlap']['accepted']}/{b2_k6['non_overlap']['proposed']}) vs M4={m4_k6['non_overlap']['rate']*100:.2f}% ({m4_k6['non_overlap']['accepted']}/{m4_k6['non_overlap']['proposed']}) | Delta: {(m4_k6['non_overlap']['rate'] - b2_k6['non_overlap']['rate'])*100:+.2f}%")
print("="*85)
