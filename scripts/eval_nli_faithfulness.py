"""Evaluate Factual Consistency / Faithfulness using DeBERTa-v3 NLI on model predictions."""
import json
import os
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def evaluate_faithfulness(
    run_dir: str,
    benchmark_path: str = "outputs/benchmarks/dev-oracle-pilot100.jsonl",
    nli_model_path: str = "/home/stackops/whale/cache/huggingface/hub/models--MoritzLaurer--DeBERTa-v3-large-mnli-fever-anli-ling-wanli/snapshots/b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
):
    predictions_path = Path(run_dir) / "predictions.jsonl"
    if not predictions_path.exists():
        print(f"File not found: {predictions_path}")
        return
        
    # Load benchmark evidence map
    evidence_map = {}
    if Path(benchmark_path).exists():
        with open(benchmark_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    cid = str(item.get("claim_id"))
                    ev_texts = [e.get("text", "") for e in item.get("evidence", []) if e.get("text")]
                    evidence_map[cid] = " ".join(ev_texts)
        print(f"Loaded {len(evidence_map)} evidence premises from {benchmark_path}")
    else:
        print(f"Warning: benchmark file {benchmark_path} not found")

    print(f"Loading NLI model from {nli_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(nli_model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(nli_model_path, local_files_only=True, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    
    id2label = model.config.id2label
    print(f"NLI label mapping: {id2label}")
    entail_idx = None
    for idx, label in id2label.items():
        if "entail" in label.lower():
            entail_idx = int(idx)
            break
            
    if entail_idx is None:
        raise ValueError("Could not find entailment label index in model config")
        
    records = []
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    print(f"Evaluating {len(records)} predictions from {run_dir}...")
    
    entailment_scores = []
    total_sentences = 0
    entailed_sentences = 0
    
    with torch.inference_mode():
        for rec in records:
            parsed = rec.get("parsed")
            if not parsed or not isinstance(parsed, dict):
                continue
                
            summary = parsed.get("summary", "")
            if not summary or not isinstance(summary, str):
                continue
                
            cid = str(rec.get("claim_id"))
            premise_str = evidence_map.get(cid, "")
            if not premise_str:
                continue
                
            # Sentence splitting on summary
            sentences = [s.strip() for s in summary.replace("\n", " ").split(".") if len(s.strip()) > 8]
            if not sentences:
                sentences = [summary.strip()]
                
            claim_scores = []
            for sent in sentences:
                inputs = tokenizer(premise_str, sent, return_tensors="pt", truncation=True, max_length=1024).to("cuda:0")
                logits = model(**inputs).logits[0]
                probs = torch.softmax(logits, dim=-1)
                p_entail = float(probs[entail_idx].item())
                claim_scores.append(p_entail)
                total_sentences += 1
                if p_entail >= 0.5:
                    entailed_sentences += 1
                    
            if claim_scores:
                entailment_scores.append(sum(claim_scores) / len(claim_scores))
                
    mean_entail = sum(entailment_scores) / len(entailment_scores) if entailment_scores else 0.0
    sentence_faithfulness = (entailed_sentences / total_sentences) * 100.0 if total_sentences else 0.0
    
    result = {
        "run_id": Path(run_dir).name,
        "total_claims_evaluated": len(entailment_scores),
        "total_sentences": total_sentences,
        "mean_entailment_probability": round(mean_entail, 4),
        "sentence_level_faithfulness_pct": round(sentence_faithfulness, 2),
    }
    
    print("\n=== FAITHFULNESS EVALUATION RESULT ===")
    print(json.dumps(result, indent=2))
    
    out_path = Path(run_dir) / "nli_faithfulness.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/runs/full-m4-dyn-lim100"
    bench = sys.argv[2] if len(sys.argv) > 2 else "outputs/benchmarks/dev-oracle-pilot100.jsonl"
    evaluate_faithfulness(run_dir, benchmark_path=bench)
