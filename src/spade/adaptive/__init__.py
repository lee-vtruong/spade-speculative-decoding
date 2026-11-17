"""SPADE Evidence-Adaptive Speculative Decoding Package (M4)."""
from spade.adaptive.evidence_indexer import EvidenceIndexer
from spade.adaptive.biasing import EvidenceBiasingConfig, apply_evidence_bias

__all__ = ["EvidenceIndexer", "EvidenceBiasingConfig", "apply_evidence_bias"]
