# Research Plan

## Question

Can few-shot domain calibration and multimodal forensic reasoning improve
detection and localization of previously unseen AI-generated document
forgeries without retraining the full detector for every generator?

## Evaluation design

Establish RGB classification and segmentation baselines, reproduce a licensed
published detector where feasible, then compare residual and OCR/layout
features through controlled ablations. Evaluate traditional-to-generative and
cross-generator shifts with leakage-safe manifests. Fit thresholds and
calibrators on validation data only. Report classification, localization,
calibration, risk-coverage, robustness, latency, memory, and bootstrap
confidence intervals.

The VLM consumes detector and OCR evidence and produces validated structured
reasoning; it is never treated as the primary forensic detector. Compare
zero-shot prompting with LoRA/QLoRA SFT on free GPU compute.

## Milestones

M0–M10 follow `TASKS.md`. Each milestone ends with verification, artifact
registration, state updates, and an evidence-based next decision.

## Definition of done

The authoritative definition is in `AGENTS.md`. No result may be reported
without its exact configuration, split manifest, checkpoint, evaluation
command, and committed artifact.
