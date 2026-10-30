"""Rejection metrics with explicit token/round/claim aggregation weights."""
from __future__ import annotations
from collections import Counter, defaultdict
from statistics import mean

PROPOSALS={"accepted","rejected"}

def acceptance_summary(token_rows: list[dict], round_rows: list[dict]) -> dict:
    proposals=[x for x in token_rows if x.get("status") in PROPOSALS and x.get("reached_by_verifier")]
    naccepted=sum(x.get("status")=="accepted" for x in proposals)
    round_rates=[int(x.get("accepted_draft_tokens",0))/int(x.get("proposed_token_count",0)) for x in round_rows if x.get("proposed_token_count",0)>0]
    by_claim=defaultdict(list)
    for row in proposals: by_claim[str(row.get("claim_id"))].append(row)
    claim_rates=[sum(x.get("status")=="accepted" for x in group)/len(group) for group in by_claim.values() if group]
    expected=[x["expected_acceptance_probability"] for x in proposals if isinstance(x.get("expected_acceptance_probability"),(int,float))]
    p=[x["target_probability"] for x in proposals if isinstance(x.get("target_probability"),(int,float))]
    q=[x["draft_probability"] for x in proposals if isinstance(x.get("draft_probability"),(int,float))]
    ratios=[float(x["target_probability"])/float(x["draft_probability"]) for x in proposals if isinstance(x.get("target_probability"),(int,float)) and isinstance(x.get("draft_probability"),(int,float)) and x["draft_probability"]>0]
    tv=[x["total_variation"] for x in proposals if isinstance(x.get("total_variation"),(int,float))]
    return {
      "aggregation_definitions":{"token_weighted":"accepted reached proposals / all reached proposals","round_weighted":"mean accepted/proposed within each round","claim_weighted":"mean per-claim acceptance rate; claims weighted equally"},
      "observed_acceptance":{"token_weighted":naccepted/len(proposals) if proposals else 0.0,"round_weighted":mean(round_rates) if round_rates else 0.0,"claim_weighted":mean(claim_rates) if claim_rates else 0.0},
      "expected_acceptance_probability_mean":mean(expected) if expected else None,
      "mean_target_probability":mean(p) if p else None,"mean_draft_probability":mean(q) if q else None,
      "mean_p_over_q":mean(ratios) if ratios else None,"mean_total_variation":mean(tv) if tv else None,
      "reached_proposals":len(proposals),"accepted_reached_proposals":naccepted,
      "mean_accepted_tokens_per_round":mean([int(x.get("accepted_draft_tokens",0)) for x in round_rows]) if round_rows else 0.0,
      "mean_committed_tokens_per_round":mean([int(x.get("committed_output_tokens",0)) for x in round_rows]) if round_rows else 0.0,
      "acceptance_length_distribution":dict(Counter(str(int(x.get("accepted_draft_tokens",0))) for x in round_rows)),
      "rejection_position_counts":dict(Counter(str(x.get("rejection_position")) for x in round_rows if x.get("rejection_position") is not None)),
      "token_category_acceptance":_category_rates(proposals),
    }

def _category_rates(proposals: list[dict]) -> dict:
    groups=defaultdict(list)
    for row in proposals: groups[row.get("token_category","other")].append(row)
    return {cat:{"accepted":sum(x.get("status")=="accepted" for x in rows),"proposals":len(rows),"acceptance_rate":sum(x.get("status")=="accepted" for x in rows)/len(rows) if rows else 0.0} for cat,rows in sorted(groups.items())}
