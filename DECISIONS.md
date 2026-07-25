# Decision Log

## D-0001 — Storage boundary

All environments, caches, datasets, checkpoints, reports, and generated
artifacts live below `F:\HYPERVERGE`. This is a user requirement.

## D-0002 — Runtime strategy

Use the already-installed trusted Python 3.14 interpreter with a virtual
environment and every installed package under `F:\HYPERVERGE`. PyTorch 2.12
publishes CPython 3.14 Windows wheels. A downloaded standalone Python on `F:`
was rejected because Windows Application Control blocked one of its extension
modules; project-local native wheels loaded successfully when hosted by the
trusted system interpreter. No global package is installed.

## D-0003 — Development sequence

Start with tiny deterministic RGB classification and segmentation baselines.
Add residual, OCR/layout, and VLM components only after baseline failure
analysis. This avoids unsupported architectural complexity.

## D-0004 — VLM candidate

Use `HuggingFaceTB/SmolVLM2-2.2B-Instruct` at revision
`482adb537c021c86670beed01cd58990d01e72e4` for the first zero-shot and LoRA SFT
comparison. It is Apache-2.0, 2.2B parameters, supported by Transformers/TRL,
and sized for a free 16 GB Kaggle GPU with 4-bit loading and conservative image
tokens. Qwen2.5-VL-3B was rejected for this portfolio route because its official
3B checkpoint uses a non-commercial Qwen research licence.

## D-0005 — Local CORD extraction

The CORD v2 Hugging Face snapshot is stored as Parquet. PyArrow 25's unsigned
`_dataset` extension was blocked by Windows Application Control on this
machine. Use pinned MIT-licensed `hyparquet` 1.26.2 and
`hyparquet-compressors` 1.1.1 under the trusted system Node runtime to extract
the embedded images. Every output image receives a SHA-256 entry and official
train/validation/test splits remain unchanged.

## D-0006 — Published baseline isolation

Pin TruFor to official commit
`ae54475df6f41a491d7615100feb19263dec13f7` and run it only in its upstream
Docker environment. Its free licence is limited to informational and nonprofit
use. Do not redistribute the source or weights, do not use an unverified weight
mirror, and do not load its pickle-based checkpoint inside the ForgeLens
environment.

## D-0007 — Vercel inference runtime

Deploy the rejected residual U-Net baseline through a numerically verified ONNX
export and CPU ONNX Runtime. Keep PyTorch, torchvision, Hugging Face Hub, and
other research-only libraries in the optional `research` dependency set. The
original CPU PyTorch function dependencies measured 817.76 MiB before source
files and produced Vercel's 820.66 MB bundle failure. The locked ONNX Runtime
function measures 126.59 MiB uncompressed while preserving the `POST /predict`
contract, thresholds, calibration, mask, and safety language. Research assets
remain in the repository but are excluded from uploads and function tracing.
