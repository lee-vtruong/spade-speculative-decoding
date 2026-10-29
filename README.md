# SPADE: Speculative Prediction for Adaptive Decoding with Evidence

Research foundation for M1 benchmark construction, M2 decoding baselines, and M3 rejection analysis. **Model-backed generation is pending** until a cache-safe adapter is implemented and GPU-validated; no experiment results are included. CPU mathematical primitives and benchmark utilities are available.

## Local setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pytest
```

No model or dataset downloads are needed for CPU checks. See [docs/SERVER_RUNBOOK.md](docs/SERVER_RUNBOOK.md) for server setup and current limitations.
