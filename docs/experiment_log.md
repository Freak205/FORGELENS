# Experiment Log

## ENV-0001 — CUDA smoke

- Timestamp: 2026-07-23
- Hypothesis: the local RTX 2050 can execute the pinned PyTorch CUDA build
- Config: PyTorch 2.12.1+cu130, Python 3.14.6
- Hardware: NVIDIA GeForce RTX 2050, 4,096 MiB
- Result: CUDA available; a 64×64 random tensor allocation and reduction passed
- Peak measured allocation: 0.02 MiB for the smoke tensor
- Decision: use local CUDA for tiny baselines and inference smoke tests; retain
  free remote GPU route for VLM tuning

## SMOKE-0001 — joint training pipeline

- Timestamp: 2026-07-23
- Hypothesis: the joint baseline can train end-to-end on safe fixtures
- Dataset: blank-canvas-fixtures-v1, 12 samples; no research split
- Config: one epoch, batch 4, seed 20260723, TinyJointDetector(base=8), AdamW
- Git commit: `50fde438feb396653abc0308308d01f4fd1534df`
- Hardware: RTX 2050; PyTorch 2.12.1+cu130
- Duration: 0.762 seconds on the committed rerun
- Peak allocated VRAM: 26.55 MiB
- Checkpoint: `artifacts/experiments/SMOKE-0001/last.pt`
- Metrics: loss 1.3022, image F1 0.0, pixel IoU 0.0
- Observation: training, evaluation, GPU, and atomic checkpoint paths work.
  Performance is intentionally meaningless after one epoch on plumbing-only
  fixtures and must not appear in portfolio results.
- Decision: proceed to full experiment configuration and real approved data
  adapter; do not tune against this fixture.

Each entry must include ID, timestamp, hypothesis, dataset version, split
manifest, resolved config, model, seed, Git commit, hardware, training time,
peak VRAM, checkpoint, metrics, observations, failures, and decision.
