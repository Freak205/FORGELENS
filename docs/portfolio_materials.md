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

## LinkedIn draft

Built ForgeLens, an evidence-first PyTorch document-forensics research system.
It covers licence-audited data, leakage-safe splits, joint detection and
localization, calibration, abstention, robustness, CUDA profiling, and a
free-GPU VLM LoRA route. Three baseline experiments were genuine negative
results; I rejected them because bootstrap intervals included chance rather
than turning weak metrics into a claim.

## Interview explanation

- Start with the threat model and why source-document leakage matters.
- Explain validation-only temperature and image/pixel thresholds.
- Show why 0.66 F1 was misleading when the false-positive rate was 1.0.
- Describe the ablation decision: global RGB → U-Net → fixed residuals.
- Emphasize the stop decision: proxy optimization ended when evidence showed
  it did not represent AI inpainting.
- Finish with the gated primary experiment and the packaged SmolVLM2 LoRA job.
