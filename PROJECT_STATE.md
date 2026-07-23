# Project State

- Current milestone: M2 — training/evaluation infrastructure
- Completed work: target verified; Git repository initialized; Python
  environment/caches confined to F drive; PyTorch 2.12.1+cu130 installed; RTX
  2050 CUDA tensor smoke passed; strict output schema, tiny joint detector,
  classification/localization metrics, ECE, Brier score, and validation-only
  threshold selection implemented; deterministic safe fixtures, grouped split,
  atomic checkpoints, and joint smoke training implemented; 10 tests pass
- Active experiment: none; SMOKE-0001 completed
- Last verified command: `.venv\Scripts\python.exe scripts\train_smoke.py`
- Available checkpoints:
  `artifacts/experiments/SMOKE-0001/last.pt` (pipeline verification only)
- Current metrics: smoke tests only; no research result claimed
- Next action: add validated experiment configuration, dataset manifests,
  evaluation artifacts, resume testing, and a research-dataset adapter
- User blockers: none
