import json
from collections import Counter
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

gold_rows = [json.loads(l) for l in Path('outputs/benchmarks/retrieved-heldout300.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
gold_map = {r['claim_id']: r.get('evaluation', {}).get('gold_label') for r in gold_rows}
print('Gold label distribution:', Counter(gold_map.values()))

for run_id in ['b1-4-heldout300-b0s', 'b1-4-heldout300-baseline-full', 'b1-4-heldout300-attn-sent-b10k']:
    pfile = Path(f'outputs/runs/{run_id}/predictions.jsonl')
    if pfile.exists():
        preds = [json.loads(l) for l in pfile.read_text(encoding='utf-8').splitlines() if l.strip()]
        y_true = []
        y_pred = []
        for p in preds:
            cid = p['claim_id']
            gl = gold_map.get(cid)
            parsed = p.get('parsed') or {}
            pl = parsed.get('verdict')
            if gl and pl:
                y_true.append(gl)
                y_pred.append(pl)
        print(f'\n=== {run_id} (N={len(y_pred)}) ===')
        print('Predicted distribution:', Counter(y_pred))
        labels = sorted(list(set(y_true + y_pred)))
        print('Confusion matrix (labels: ' + ', '.join(labels) + '):')
        print(confusion_matrix(y_true, y_pred, labels=labels))
