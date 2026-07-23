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

## DATA-0001 — CORD v2 accession and extraction

- Timestamp: 2026-07-23
- Dataset: `naver-clova-ix/cord-v2`
- Revision: `7f0115a4b758a71d6473b8d085751692da2fef98`
- Licence: CC BY 4.0 according to official dataset metadata
- Download: six Parquet shards, 2,307,284,272 bytes total
- Extracted: 800 train, 100 validation, 100 test images with matching JSONL
  provenance rows and per-image SHA-256
- Failure: PyArrow 25 could not load its unsigned `_dataset` extension under
  Windows Application Control.
- Resolution: pinned pure-JavaScript/WASM Parquet extraction; exact counts
  verified after correcting train-manifest append behavior.
- Decision: CORD is approved as the first authentic receipt corpus. Do not
  alter official splits.

## BASELINE-0001 — TruFor reproduction preparation

- Timestamp: 2026-07-23
- Hypothesis: the official published baseline can be evaluated through a
  revision-pinned, licence-aware adapter without contaminating ForgeLens
- Source: `grip-unina/TruFor` commit
  `ae54475df6f41a491d7615100feb19263dec13f7`
- Licence: free informational/nonprofit use only; notices and attribution
  required
- Split manifest: CORD v2 official test split, 100 checksum-carrying inputs
- Result: strict `.npz` normalization implemented; 40 project checks pass
- Failure: official weights host timed out through both PowerShell and curl
- Decision: retain the official URL and MD5, reject unverified mirrors, and run
  the old upstream stack in an isolated container when official weights become
  reachable; do not claim baseline metrics before paired forged data exists

Each entry must include ID, timestamp, hypothesis, dataset version, split
manifest, resolved config, model, seed, Git commit, hardware, training time,
peak VRAM, checkpoint, metrics, observations, failures, and decision.
