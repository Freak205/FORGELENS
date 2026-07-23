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
