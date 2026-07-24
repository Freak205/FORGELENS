# Portfolio Materials — Evidence-Checked Draft

## Recruiter summary

Built ForgeLens, a typed and tested PyTorch research system spanning licensed
data provenance, leakage-safe splits, CUDA training, localization, calibration,
robustness, published-baseline isolation, strict inference, and a free-GPU VLM
LoRA route. Preserved four negative experiments, including a paired
GPT-Image-2 benchmark, instead of presenting unsupported improvements.

## Resume bullets

- Engineered a 50-test PyTorch document-forensics pipeline with joint
  classification/localization, validation-only calibration, abstention,
  bootstrap intervals, and reproducible CUDA checkpoints.
- Built a licence-audited 2,000-sample CORD benchmark and ran three controlled
  ablations, rejecting models whose 95% ROC-AUC intervals included chance.
- Ran a leakage-safe 1,966-sample paired GPT-Image-2/CORD experiment with
  bootstrap CIs, few-shot calibration, risk–coverage, and nine corruptions;
  rejected the chance-level model rather than overstating its F1.
- Profiled RTX 2050 batch-one inference at 2.45 ms median and 27 MiB peak VRAM;
  completed a revision-pinned free-Kaggle SmolVLM2 LoRA workflow with a hashed
  37 MB adapter and audited zero-shot/SFT proxy comparison.

Do not present ForgeLens as an accurate detector; the strongest claim is the
reproducible research/evaluation system and its evidence-based stop decisions.

## LinkedIn draft

Built ForgeLens, an evidence-first PyTorch document-forensics research system.
It covers licence-audited data, leakage-safe splits, joint detection and
localization, calibration, abstention, robustness, CUDA profiling, and a
free-GPU VLM LoRA route. Four experiments, including paired GPT-Image-2
forgeries, were genuine negative results; I rejected them because bootstrap
intervals included chance rather than turning weak metrics into a claim.

## Interview explanation

- Start with the threat model and why source-document leakage matters.
- Explain validation-only temperature and image/pixel thresholds.
- Show why 0.66 F1 was misleading when the false-positive rate was 1.0.
- Describe the ablation decision: global RGB → U-Net → fixed residuals.
- Emphasize the stop decision: proxy optimization ended when evidence showed
  it did not represent AI inpainting.
- Finish with the paired GPT-Image-2 negative result and safe public-CORD
  SmolVLM2 LoRA proxy, clearly separating its small proxy improvement from
  AI-inpainting or production claims.
