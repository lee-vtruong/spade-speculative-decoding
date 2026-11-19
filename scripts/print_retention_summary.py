import json

with open("outputs/analysis/b1_4_heldout300/content_retention_report.json", "r", encoding="utf-8") as f:
    rep = json.load(f)

print("=== RETENTION REPORT OVERALL ===")
for m, d in rep["overall_retention"].items():
    cov = d["mean_passage_coverage_pct"]
    rec = d["target_cited_recall_pct"]
    ret = d["target_cited_passages_retained"]
    tot = d["target_cited_passages_total"]
    comp = d["mean_evidence_compression_pct"]
    print(f"{m:20s} | Cov: {cov:5.1f}% | Cited Recall: {rec:5.1f}% ({ret:3d}/{tot:3d}) | Retained Chars: {comp:5.1f}%")

print("\n=== STRATIFIED BY PASSAGE COUNT (Acceptance Rate) ===")
for bin_name, bdata in rep["stratified_breakdown"]["by_passage_count"].items():
    print(f"-- {bin_name} --")
    for m, md in bdata.items():
        cnt = md["count"]
        acc = md["mean_acceptance_pct"]
        cmp_pct = md["mean_compression_pct"]
        print(f"   {m:20s}: N={cnt:2d}, Acc={acc:5.2f}%, Chars={cmp_pct:5.1f}%")

print("\n=== STRATIFIED BY CONTEXT LENGTH (Acceptance Rate) ===")
for bin_name, bdata in rep["stratified_breakdown"]["by_evidence_chars"].items():
    print(f"-- {bin_name} --")
    for m, md in bdata.items():
        cnt = md["count"]
        acc = md["mean_acceptance_pct"]
        cmp_pct = md["mean_compression_pct"]
        print(f"   {m:20s}: N={cnt:2d}, Acc={acc:5.2f}%, Chars={cmp_pct:5.1f}%")
