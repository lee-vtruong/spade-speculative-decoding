"""Baseline CLI guard until a cache-safe, version-tested HF adapter exists."""
import argparse
p=argparse.ArgumentParser(description="SPADE GPU baseline preparation")
p.add_argument("--mode",choices=["B0","B0-S","B1","B2"],required=True); p.add_argument("--input",required=True); p.add_argument("--output",required=True)
p.add_argument("--draft-length",type=int,choices=[2,4,6,8],default=4); p.add_argument("--max-new-tokens",type=int,default=256); p.add_argument("--warmup",type=int,default=1); p.add_argument("--trials",type=int,default=1); p.add_argument("--seed",type=int,default=42)
p.parse_args(); p.error("Model-backed inference is not safe to run until the Transformers-version-specific Qwen KV-cache adapter is implemented and validated.")
