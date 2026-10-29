"""Aggregation helpers with explicit token versus round denominators."""
from __future__ import annotations
from collections import Counter
def acceptance_summary(token_rows: list[dict], round_rows: list[dict]) -> dict:
    reached=[x for x in token_rows if x.get("reached_by_verifier")]
    accepted=sum(x.get("status")=="accepted" for x in reached)
    return {"token_weighted_acceptance_rate":accepted/len(reached) if reached else 0.0,"reached_proposals":len(reached),"accepted_reached_proposals":accepted,
            "mean_accepted_tokens_per_round":sum(int(x.get("accepted_draft_tokens",0)) for x in round_rows)/len(round_rows) if round_rows else 0.0,
            "mean_committed_tokens_per_round":sum(int(x.get("committed_output_tokens",0)) for x in round_rows)/len(round_rows) if round_rows else 0.0,
            "rejection_position_counts":dict(Counter(str(x.get("rejection_position")) for x in round_rows if x.get("rejection_position") is not None))}
