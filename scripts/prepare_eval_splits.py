"""Create a strictly disjoint held-out final evaluation split excluding all tuning claim IDs.

Ensures zero claim-level leakage between exploratory tuning (e.g., dev-oracle-pilot100.jsonl)
and final held-out evaluation (e.g., dev-oracle-heldout400.jsonl).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from spade.benchmark.benchmark_builder import stratified_sample


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _label_of(row: dict[str, Any]) -> str:
    ev = row.get("evaluation") or {}
    return str(ev.get("label") or ev.get("gold_label") or row.get("label") or "unknown")


def build_disjoint_heldout_split(
    full_benchmark: Path,
    tuning_benchmark: Path,
    heldout_output: Path,
    heldout_size: int | None = None,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build held-out benchmark JSONL strictly disjoint from tuning_benchmark."""
    full_rows = read_jsonl(full_benchmark)
    tuning_rows = read_jsonl(tuning_benchmark)

    if not full_rows:
        raise ValueError(f"Full benchmark is empty: {full_benchmark}")
    if not tuning_rows:
        raise ValueError(f"Tuning benchmark is empty: {tuning_benchmark}")

    tuning_ids = [str(r["claim_id"]) for r in tuning_rows]
    tuning_id_set = set(tuning_ids)
    if len(tuning_id_set) != len(tuning_ids):
        raise ValueError("Duplicate claim_ids detected inside tuning benchmark")

    remaining = [r for r in full_rows if str(r["claim_id"]) not in tuning_id_set]
    if not remaining:
        raise ValueError("No disjoint claims remain after excluding tuning claim_ids")

    # Deduplicate remaining by claim_id while preserving order
    seen_rem: set[str] = set()
    dedup_remaining: list[dict[str, Any]] = []
    for r in remaining:
        cid = str(r["claim_id"])
        if cid not in seen_rem:
            seen_rem.add(cid)
            dedup_remaining.append(r)

    if heldout_size is not None:
        if heldout_size < 1 or heldout_size > len(dedup_remaining):
            raise ValueError(
                f"Requested heldout_size={heldout_size} out of valid range [1, {len(dedup_remaining)}]"
            )
        # Attach temporary top-level label if needed by stratified_sample
        wrapped = []
        by_cid = {}
        for r in dedup_remaining:
            cid = str(r["claim_id"])
            by_cid[cid] = r
            wrapped.append({"claim_id": cid, "label": _label_of(r)})
        selected_wrapped = stratified_sample(wrapped, heldout_size, seed)
        heldout_rows = [by_cid[w["claim_id"]] for w in selected_wrapped]
    else:
        heldout_rows = dedup_remaining

    heldout_ids = [str(r["claim_id"]) for r in heldout_rows]
    overlap = tuning_id_set & set(heldout_ids)
    if overlap:
        raise RuntimeError(f"Leakage detected: {len(overlap)} overlapping claim_ids")

    manifest_path = heldout_output.with_suffix(heldout_output.suffix + ".manifest.json")
    if not overwrite and (heldout_output.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"Refusing to overwrite existing output/manifest: {heldout_output} (pass --overwrite to replace)"
        )

    heldout_output.parent.mkdir(parents=True, exist_ok=True)
    with heldout_output.open("w", encoding="utf-8") as f:
        for row in heldout_rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    manifest = {
        "split_role": "heldout_final_evaluation",
        "strictly_disjoint": True,
        "overlap_count": 0,
        "seed": seed,
        "source_full_benchmark": str(full_benchmark.resolve()),
        "source_full_sha256": sha256_file(full_benchmark),
        "source_full_count": len(full_rows),
        "tuning_benchmark": str(tuning_benchmark.resolve()),
        "tuning_benchmark_sha256": sha256_file(tuning_benchmark),
        "tuning_claim_count": len(tuning_rows),
        "tuning_class_distribution": dict(sorted(Counter(_label_of(r) for r in tuning_rows).items())),
        "heldout_benchmark": str(heldout_output.resolve()),
        "heldout_benchmark_sha256": sha256_file(heldout_output),
        "heldout_claim_count": len(heldout_rows),
        "heldout_class_distribution": dict(sorted(Counter(_label_of(r) for r in heldout_rows).items())),
        "heldout_claim_ids": heldout_ids,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-benchmark", type=Path, required=True, help="Full prepared benchmark JSONL (e.g. dev-oracle-500.jsonl)")
    p.add_argument("--tuning-benchmark", type=Path, required=True, help="Fixed tuning benchmark JSONL (e.g. dev-oracle-pilot100.jsonl)")
    p.add_argument("--heldout-output", type=Path, required=True, help="Output path for strictly disjoint held-out JSONL")
    p.add_argument("--heldout-size", type=int, default=None, help="Optional size cap (stratified); defaults to all disjoint claims")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", default=False)
    args = p.parse_args()

    manifest = build_disjoint_heldout_split(
        full_benchmark=args.full_benchmark,
        tuning_benchmark=args.tuning_benchmark,
        heldout_output=args.heldout_output,
        heldout_size=args.heldout_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "heldout_benchmark": manifest["heldout_benchmark"],
        "heldout_claim_count": manifest["heldout_claim_count"],
        "tuning_claim_count": manifest["tuning_claim_count"],
        "overlap_count": manifest["overlap_count"],
        "strictly_disjoint": manifest["strictly_disjoint"],
        "heldout_class_distribution": manifest["heldout_class_distribution"],
    }, indent=2))


if __name__ == "__main__":
    main()
