# ForgeLens

**Calibrated document-forgery detection and pixel localization under generator
shift.**

> Research prototype—not forensic proof. Current local checkpoints failed
> operational validation and default to manual review.

ForgeLens is a typed, tested PyTorch research pipeline spanning licensed-data
provenance, leakage-safe splits, joint classification/localization, calibration,
abstention, robustness, strict JSON inference, published-baseline isolation,
and a free-GPU VLM LoRA route.

## Verified state

- 50 tests; formatting, linting, and strict typing pass.
- Four genuine RTX 2050 training runs are preserved as negative results.
- Best proxy ROC-AUC: 0.548 (95% CI 0.468–0.628)—rejected, not advertised.
- Paired GPT-Image-2 CORD test ROC-AUC: 0.502 (95% CI
  0.444–0.558)—rejected, not advertised.
- Batch-one residual-model inference: 2.45 ms median, 27 MiB peak VRAM.
- AIForge-Doc v2 is revision-pinned and fully downloaded under
  `F:\HYPERVERGE`; the private Kaggle VLM kernel is submitted but the account
  currently launches it without CUDA.

## Reproduce

```powershell
cd F:\HYPERVERGE\forgelens
.\tools\run.ps1 sync
.\tools\run.ps1 verify
.\tools\run.ps1 demo
```

Open `http://127.0.0.1:7860`. The demo binds only to localhost and accepts
approved PNG/JPEG/WebP files up to 10 MiB.

Research evidence: `reports/technical_report.md`, `docs/experiment_log.md`,
`docs/dataset_register.md`, and `results/`.
