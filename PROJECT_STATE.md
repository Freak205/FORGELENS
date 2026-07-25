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
  licences pinned, strict output adapter implemented; deterministic 2,000-row
  CORD copy-move benchmark and real GPU training path prepared; 43 tests pass
- Active experiment: `RESIDUAL-COPYMOVE-001` completed and rejected; test
  ROC-AUC 0.525 [0.452, 0.610], 100% false positives, validation-threshold
  pixel IoU 0.051
- Last verified command: `.\tools\run.ps1 verify` (46 passed); generated and
  validated 1,000 forged images, 1,000 exact masks, and 2,000 manifest rows
- Packaging verification: 46 tests pass; localhost demo returned HTTP 200,
  strict `uncertain/manual_review`, and a mask for a licensed CORD test image
- Deployment packaging: Git-connected Vercel configuration, Python 3.14
  serverless prediction handler, 4 MiB upload guard, embedded mask response,
  security headers, and explicit checkpoint inclusion added
- Report verification: actual reliability/ablation figures and three
  qualitative failures generated; clean wheel install and isolated model
  forward passed under `F:\HYPERVERGE\.tmp\clean-audit-605bb90`
- Last verified Git commit before current state update:
  `3d215db` (immutable leakage-safe CORD manifest)
- Available checkpoints:
  `artifacts/experiments/SMOKE-0001/last.pt` (pipeline verification only)
- Current metrics: smoke tests only; no research result claimed
- Next action: obtain AIForge access; run the primary benchmark and free-Kaggle
  VLM comparison. All currently possible offline deliverables are complete.
- User blockers: accept free AIForge-Doc v2 terms and provide a read-only
  Hugging Face token for genuine dataset training
