"""Free-GPU LoRA SFT entry point for the evidence-grounded VLM."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "-r",
        str(Path(__file__).resolve().parents[1] / "requirements-vlm.txt"),
    ],
    check=True,
)

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
MODEL_REVISION = "482adb537c021c86670beed01cd58990d01e72e4"
INPUT_ROOT = Path(os.environ.get("FORGELENS_INPUT", "/kaggle/input/forgelens-bundle"))
OUTPUT_ROOT = Path(os.environ.get("FORGELENS_OUTPUT", "/kaggle/working/forgelens-output"))


def main() -> None:
    """Train a resumable LoRA adapter and export a compact result bundle."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the VLM SFT job")
    dataset_path = INPUT_ROOT / "vlm_sft.jsonl"
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(OUTPUT_ROOT / "cache" / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(OUTPUT_ROOT / "cache"))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    split = dataset.train_test_split(test_size=0.1, seed=20260723)
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=os.environ.get("HF_TOKEN"),
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=(
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ),
        bnb_4bit_use_double_quant=True,
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    checkpoint = OUTPUT_ROOT / "checkpoints"
    args = SFTConfig(
        output_dir=str(checkpoint),
        seed=20260723,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_length=None,
        completion_only_loss=True,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=MODEL_ID,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=processor,
        peft_config=lora,
        model_init_kwargs={
            "revision": MODEL_REVISION,
            "quantization_config": quantization,
            "device_map": "auto",
        },
    )
    resume = True if any(checkpoint.glob("checkpoint-*")) else None
    result = trainer.train(resume_from_checkpoint=resume)
    adapter = OUTPUT_ROOT / "adapter"
    trainer.save_model(str(adapter))
    processor.save_pretrained(str(adapter))
    record = {
        "experiment_id": "VLM-SFT-001",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "method": "4-bit NF4 QLoRA-style PEFT SFT",
        "completion_only_loss": True,
        "dataset_rows": len(dataset),
        "train_metrics": result.metrics,
        "duration_seconds": time.perf_counter() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1048576,
        "gpu": torch.cuda.get_device_name(0),
        "adapter_path": str(adapter),
    }
    (OUTPUT_ROOT / "record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
