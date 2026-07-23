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

## RGB-COPYMOVE-001 — global-pooled RGB baseline

- Timestamp: 2026-07-23
- Hypothesis: a tiny joint RGB model learns a non-trivial traditional-tampering
  baseline on real CORD documents
- Dataset: `forgelens/cord-copy-move-v1`, 1,600/200/200 samples, official
  source-group-preserving splits
- Config: five epochs, 128×192, batch 12, seed 20260723, mixed precision,
  TinyJointDetector(base=12), AdamW
- Git commit: `6e90f4f8ea855051ef3afdbced101c1c0b3f2617`
- Hardware: RTX 2050; PyTorch 2.12.1+cu130
- Duration and peak VRAM: 703.16 seconds, 92.98 MiB
- Test: ROC-AUC 0.548 [0.468, 0.628], PR-AUC 0.566 [0.470, 0.659], pixel IoU
  0.191, ECE 0.015, Brier 0.248
- Failure: validation-selected threshold produced a 99% test false-positive
  rate; global average pooling diluted local tampering evidence
- Checkpoint SHA-256:
  `b5c997d33aa151b6a007efa5e6e9ae6ffcd66a1eecec7ffa3dc0769011f89305`
- Decision: reject operationally and test a skip-connected localizer with
  explicit top-region evidence aggregation

## UNET-COPYMOVE-001 — localization-first RGB ablation

- Timestamp: 2026-07-23
- Hypothesis: skip connections and top-region pooling recover local evidence
  lost by global average pooling
- Dataset/config: same locked split and seed as `RGB-COPYMOVE-001`; 128×192,
  five epochs, batch 12, TinyUNetJointDetector(base=12)
- Git commit: `a578c316e5504def7622cf8e1f059240a0e4a95c`
- Duration and peak VRAM: 102.75 seconds after one-time RAM cache, 320.31 MiB
- Test: ROC-AUC 0.509 [0.436, 0.587], PR-AUC 0.516 [0.429, 0.614],
  fixed-threshold pixel IoU 0.0, false-positive rate 1.0
- Failure: downsampling likely erased thin paste-boundary evidence; fixed 0.5
  pixel threshold was not validation-selected
- Checkpoint SHA-256:
  `8c4520764b7bfa2f065c68af518f3abc29feea536b6c3554fa5dba55aa86c358`
- Decision: reject and test higher-resolution fixed high-pass residual evidence;
  add validation-only localization threshold selection

## RESIDUAL-COPYMOVE-001 — high-resolution residual ablation

- Timestamp: 2026-07-23
- Hypothesis: 192×288 inputs plus fixed Laplacian/Sobel residuals preserve and
  expose copy-paste seam evidence
- Dataset/config: same locked split and seed; five epochs, batch 8,
  ResidualUNetJointDetector(base=8)
- Git commit: `ffce4cc97d1337d1631ed8f151544e05d6931da4`
- Duration and peak VRAM: 182.20 seconds, 326.92 MiB
- Test: ROC-AUC 0.525 [0.452, 0.610], PR-AUC 0.532 [0.441, 0.641],
  validation-threshold pixel IoU 0.051, false-positive rate 1.0
- Failure: neither fixed residuals nor extra resolution provided a reliable
  ranking/localization signal on the proxy copy-move task
- Checkpoint SHA-256:
  `18d61a4aa7c8a3f5ad87704d4726de81d2dd3348c536824bd2e9ebeaa0aa37c3`
- Decision: reject operationally and stop optimizing the proxy benchmark;
  prioritize licensed AI-inpainting data

## AIFORGE-CORD-UNET-001 — paired GPT-Image-2 baseline

- Timestamp: 2026-07-24
- Hypothesis: a localization-first RGB model learns generator-specific evidence
  on paired GPT-Image-2 CORD forgeries
- Dataset: `Scam-AI/AIForge-Doc-v2` revision
  `9fe6f52f073c01b42966d0fd0dda87db7c9725f9`, CORD-only paired subset;
  1,258/314/394 train/validation/test samples
- Split policy: preserve the official AIForge test partition, derive validation
  only from official training, and keep each authentic/forged pair together
- Config: eight epochs, 256×384, batch 4, seed 20260723, mixed precision,
  TinyUNetJointDetector(base=8), AdamW
- Git commit: `0ea95f8b642df8d7c75ff2bc71416740fda31534`
- Hardware: RTX 2050; PyTorch 2.12.1+cu130
- Duration and peak VRAM: 565.91 seconds, 280.98 MiB
- Test: ROC-AUC 0.502 [0.444, 0.558], PR-AUC 0.508 [0.437, 0.578],
  validation-threshold pixel IoU 0.020, false-positive rate 1.0
- Calibration/robustness: N={1,5,10,25} group calibration did not change
  ranking; 50% coverage error was 0.492; severity-3 corruption AUCs ranged
  0.500–0.503 across nine capture proxies
- Checkpoint SHA-256:
  `769991b8698c5c01fba47dfab93e2323fc952a7ca8298b1a39fbbc89a7e0af93`
- Failure: the small RGB model learned neither reliable paired classification
  nor localization evidence; low ECE reflects near-constant 0.5 predictions,
  not useful confidence
- Decision: reject operationally; preserve the negative result and do not
  convert its F1 of 0.667 into a performance claim

## VLM-SFT-001 — private free-Kaggle route

- Timestamp: 2026-07-24
- Model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct` revision
  `482adb537c021c86670beed01cd58990d01e72e4`, Apache 2.0
- Dataset policy: code-only kernel; download public pinned CORD v2 in the
  session and generate a deterministic paired copy-move proxy; never upload the
  gated AIForge derivative
- Kernel: private `ivsanirudh/forgelens-vlm-lora-sft`
- Compatibility work: removed unavailable network installs, bitsandbytes, and
  TRL; version 5 uses Kaggle-installed Transformers and PEFT
- Result: Kaggle launched version 5 without a CUDA device despite the requested
  free P100 accelerator, so training correctly stopped before model download
  or optimization
- Decision: external account/runtime blocker. Enable a GPU accelerator for the
  private notebook, then rerun the already-submitted version; do not claim VLM
  results until `record.json` is downloaded and audited

Each entry must include ID, timestamp, hypothesis, dataset version, split
manifest, resolved config, model, seed, Git commit, hardware, training time,
peak VRAM, checkpoint, metrics, observations, failures, and decision.
