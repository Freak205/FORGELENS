# Portfolio Materials — Hold Until Primary Results

## Recruiter summary

Built ForgeLens, a typed and tested PyTorch research system spanning licensed
data provenance, leakage-safe splits, CUDA training, localization, calibration,
robustness, published-baseline isolation, strict inference, and a free-GPU VLM
LoRA route. Preserved three negative experiments instead of presenting
unsupported improvements.

## Resume bullets

- Engineered a 46-test PyTorch document-forensics pipeline with joint
  classification/localization, validation-only calibration, abstention,
  bootstrap intervals, and reproducible CUDA checkpoints.
- Built a licence-audited 2,000-sample CORD benchmark and ran three controlled
  ablations, rejecting models whose 95% ROC-AUC intervals included chance.
- Profiled RTX 2050 batch-one inference at 2.45 ms median and 27 MiB peak VRAM;
  packaged a revision-pinned free-Kaggle SmolVLM2 QLoRA workflow.

Do not publish performance-focused claims until AIForge and VLM experiments
complete.
