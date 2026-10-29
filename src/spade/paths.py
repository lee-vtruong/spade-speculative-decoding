"""Portable project/data/model/output paths configurable through environment."""
from __future__ import annotations
import os
from pathlib import Path

def project_root() -> Path:
    return Path(os.environ.get("PROJECT_ROOT",Path(__file__).resolve().parents[2])).expanduser().resolve()
def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT",project_root()/"data")).expanduser().resolve()
def model_root() -> Path:
    return Path(os.environ.get("MODEL_ROOT",project_root()/"models")).expanduser().resolve()
def output_root() -> Path:
    return Path(os.environ.get("OUTPUT_ROOT",project_root()/"outputs")).expanduser().resolve()
