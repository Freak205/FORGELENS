# Free Kaggle GPU Route

This job fine-tunes only a LoRA adapter for the open Apache-2.0
`HuggingFaceTB/SmolVLM2-2.2B-Instruct` revision
`482adb537c021c86670beed01cd58990d01e72e4`. The VLM receives detector/OCR
evidence and is not the primary forensic detector.

The input bundle must contain `vlm_sft.jsonl` with authorized local image paths
and prompt-completion messages. It must not contain credentials, real identity
documents, biometrics, or gated data that Kaggle terms prohibit uploading.

The job uses a private Kaggle kernel with free GPU enabled, saves resumable
checkpoints to `/kaggle/working/forgelens-output`, and emits a compact adapter
plus `record.json`. Download all outputs back below `F:\HYPERVERGE` immediately
after completion.

Execution requires Kaggle login/API credentials and is intentionally not
attempted until the evidence dataset passes licence and privacy checks.
