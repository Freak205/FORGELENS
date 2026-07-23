# Free Kaggle GPU Route

This job fine-tunes only a mixed-precision LoRA adapter for the open Apache-2.0
`HuggingFaceTB/SmolVLM2-2.2B-Instruct` revision
`482adb537c021c86670beed01cd58990d01e72e4`. The VLM receives detector/OCR
evidence and is not the primary forensic detector.

By default, the code-only kernel downloads pinned public CORD v2 and constructs a
small deterministic copy-move proxy inside Kaggle. This avoids uploading gated
AIForge data or credentials. An optional private `vlm_sft.jsonl` bundle is
supported only when its upload is separately reviewed and authorized.

The job uses a private Kaggle kernel with free GPU enabled and emits a compact
adapter plus `record.json` to `/kaggle/working/forgelens-output`. Download all
outputs back below `F:\HYPERVERGE` immediately after completion.

Execution requires only Kaggle API authentication. The proxy result must not be
reported as AI-inpainting performance.
