# Project State

- Current milestone: M2 — training/evaluation infrastructure
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
  and adapted with 800/100/100 official split counts; 36 tests pass
- Active experiment: none; SMOKE-0001 completed
- Last verified command: real CORD adapter load (800 rows, RGB 256×256, authentic
  label and empty mask verified) after `.\tools\run.ps1 verify` (36 passed)
- Last verified Git commit before current state update:
  `bf4b32d` (strict structured inference)
- Available checkpoints:
  `artifacts/experiments/SMOKE-0001/last.pt` (pipeline verification only)
- Current metrics: smoke tests only; no research result claimed
- Next action: generate immutable paired split manifests and execute the first
  real CORD/AIForge baseline as soon as AIForge-Doc v2 read access arrives
- User blockers: accept free AIForge-Doc v2 terms and provide a read-only
  Hugging Face token for genuine dataset training
