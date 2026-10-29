"""Versioned JSONL trace field definitions and status validation."""
ROUND_FIELDS={"run_id","claim_id","round_index","prefix_length","draft_length_requested","proposed_token_count","accepted_draft_tokens","rejection_position","full_acceptance","additional_target_token","committed_output_tokens","draft_latency_s","verification_latency_s","correction_latency_s","total_round_latency_s","cumulative_generated_tokens"}
TOKEN_STATUSES={"accepted","rejected","discarded_after_rejection","residual_replacement","extra_target_token"}
TOKEN_FIELDS={"run_id","claim_id","round_id","absolute_token_position","token_id","token_text","draft_probability","target_probability","acceptance_probability","status","reached_by_verifier","discarded_after_rejection","replacement_token_id","token_category","prefix_length","evidence_overlap"}

def validate_record(record: dict, fields: set[str]) -> None:
    missing=fields-record.keys()
    if missing: raise ValueError(f"Missing trace fields: {sorted(missing)}")
    if "status" in fields and record["status"] not in TOKEN_STATUSES: raise ValueError(f"Unknown token status: {record['status']}")
