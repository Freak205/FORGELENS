# Project State

- Current milestone: M3 — genuine and published baselines
- Completed work: target verified; Git repository initialized; Python
  environment/caches confined to F drive; PyTorch 2.12.1+cu130 installed; RTX
  2050 CUDA tensor smoke passed; strict output schema, tiny joint detector,
  classification/localization metrics, ECE, Brier score, and validation-only
  threshold selection implemented; deterministic safe fixtures, grouped split,
  atomic checkpoints, joint smoke training, experiment config validation,
  ROC/PR AUC with bootstrap intervals, explicit abstention policy, strict
  AIForge-Doc adapter, nine deterministic corruptions, secure revision-pinned
  dataset acquisition, AMP/accumulation training engine, strict structured
  inference, and a private free-Kaggle SmolVLM2 LoRA route implemented; CORD v2
  is revision-pinned, downloaded, extracted, checksummed,
  and adapted with 800/100/100 official split counts; TruFor source and
  licences pinned, strict output adapter implemented; 40 tests pass
- Active experiment: none; SMOKE-0001 completed
- Last verified command: `.\tools\run.ps1 verify` (40 passed), followed by
  `.\tools\run.ps1 prepare-trufor` (100 immutable CORD test inputs)
- Last verified Git commit before current state update:
  `3d215db` (immutable leakage-safe CORD manifest)
- Available checkpoints:
  `artifacts/experiments/SMOKE-0001/last.pt` (pipeline verification only)
- Current metrics: smoke tests only; no research result claimed
- Next action: obtain paired AIForge/CORD samples, build immutable joint
  manifests, train the first real baseline, then run published/VLM comparisons
- User blockers: accept free AIForge-Doc v2 terms and provide a read-only
  Hugging Face token for genuine dataset training
