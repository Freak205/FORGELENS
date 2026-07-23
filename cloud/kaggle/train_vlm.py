"""Free-GPU LoRA SFT entry point for the evidence-grounded VLM."""

# ruff: noqa: E402

import json
import os
import time
from pathlib import Path

import accelerate
import bitsandbytes
import datasets
import peft
import torch
import transformers
import trl
from datasets import Image as DatasetImage
from datasets import Dataset, load_dataset
from PIL import Image as PILImage
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
MODEL_REVISION = "482adb537c021c86670beed01cd58990d01e72e4"
CORD_REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
INPUT_ROOT = Path(os.environ.get("FORGELENS_INPUT", "/kaggle/input/forgelens-bundle"))
OUTPUT_ROOT = Path(
    os.environ.get("FORGELENS_OUTPUT", "/kaggle/working/forgelens-output")
)


def prepare_dataset(dataset_path: Path) -> Dataset:
    """Load relative bundle paths as decoded images without random re-splitting."""
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = dataset.map(
        lambda row: {"image": str(INPUT_ROOT / row["image"])},
        desc="Resolving private bundle image paths",
    )
    return dataset.cast_column("image", DatasetImage())


def messages(label: int) -> list[dict]:
    """Return label-first supervision without personal-data inference."""
    prompt = (
        "Inspect this receipt for image tampering. Return exactly one verdict "
        "(AUTHENTIC or FORGED), then one short visual-evidence sentence. Do not "
        "infer identity or personal attributes."
    )
    verdict = "FORGED" if label else "AUTHENTIC"
    evidence = (
        "A copied image region was deterministically relocated in this proxy."
        if label
        else "No synthetic copied region is present in this proxy source."
    )
    return [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"VERDICT: {verdict}\nEVIDENCE: {evidence}"}
            ],
        },
    ]


def copy_move(image: PILImage.Image, index: int) -> PILImage.Image:
    """Create a deterministic visible-but-localized copy-move proxy."""
    forged = image.convert("RGB").copy()
    width, height = forged.size
    patch_width = max(12, width // 5)
    patch_height = max(12, height // 10)
    source_x = (index * 17) % max(1, width - patch_width)
    source_y = (index * 29) % max(1, height - patch_height)
    target_x = (source_x + width // 3) % max(1, width - patch_width)
    target_y = (source_y + height // 3) % max(1, height - patch_height)
    patch = forged.crop(
        (source_x, source_y, source_x + patch_width, source_y + patch_height)
    )
    forged.paste(patch, (target_x, target_y))
    return forged


def public_cord_proxy_dataset() -> Dataset:
    """Build a no-secret, non-gated VLM proxy directly on Kaggle."""
    source = load_dataset(
        "naver-clova-ix/cord-v2",
        revision=CORD_REVISION,
    )
    rows: list[dict] = []
    plan = {
        "train": ("train", 128),
        "validation": ("validation", 32),
        "test": ("test", 64),
    }
    for output_split, (source_split, count) in plan.items():
        selected = source[source_split].select(range(count))
        for index, row in enumerate(selected):
            authentic = row["image"].convert("RGB")
            group = f"{source_split}:{index:06d}"
            for label, image in (
                (0, authentic),
                (1, copy_move(authentic, index)),
            ):
                rows.append(
                    {
                        "sample_id": f"{group}:{label}",
                        "source_group": group,
                        "split": output_split,
                        "label": label,
                        "image": image,
                        "messages": messages(label),
                    }
                )
    return Dataset.from_list(rows)


def balanced_test_subset(dataset: Dataset, per_class: int = 64) -> Dataset:
    """Select a deterministic, balanced evaluation subset."""
    authentic = dataset.filter(lambda row: int(row["label"]) == 0).shuffle(
        seed=20260723
    )
    forged = dataset.filter(lambda row: int(row["label"]) == 1).shuffle(seed=20260723)
    count = min(per_class, len(authentic), len(forged))
    return Dataset.from_list(
        [
            *[authentic[index] for index in range(count)],
            *[forged[index] for index in range(count)],
        ]
    ).shuffle(seed=20260723)


@torch.no_grad()
def evaluate_verdicts(model: object, processor: object, dataset: Dataset) -> dict:
    """Measure deterministic balanced verdict accuracy and macro recall."""
    predictions: list[int] = []
    targets: list[int] = []
    model.eval()
    for row in dataset:
        messages = [row["messages"][0]]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        answer = processor.batch_decode(generated, skip_special_tokens=True)[0].upper()
        predictions.append(1 if "FORGED" in answer else 0)
        targets.append(int(row["label"]))
    recalls = []
    for label in (0, 1):
        matches = [
            prediction == label
            for prediction, target in zip(predictions, targets)
            if target == label
        ]
        recalls.append(sum(matches) / len(matches))
    return {
        "samples": len(targets),
        "accuracy": sum(
            prediction == target for prediction, target in zip(predictions, targets)
        )
        / len(targets),
        "balanced_accuracy": sum(recalls) / 2,
        "authentic_recall": recalls[0],
        "forged_recall": recalls[1],
    }


def main() -> None:
    """Train a resumable LoRA adapter and export a compact result bundle."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the VLM SFT job")
    dataset_path = INPUT_ROOT / "vlm_sft.jsonl"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(OUTPUT_ROOT / "cache" / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(OUTPUT_ROOT / "cache"))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    using_private_bundle = dataset_path.is_file()
    dataset = (
        prepare_dataset(dataset_path)
        if using_private_bundle
        else public_cord_proxy_dataset()
    )
    train_dataset = dataset.filter(lambda row: row["split"] == "train")
    validation_dataset = dataset.filter(lambda row: row["split"] == "validation")
    test_dataset = balanced_test_subset(
        dataset.filter(lambda row: row["split"] == "test")
    )
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
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=processor,
        peft_config=lora,
        model_init_kwargs={
            "revision": MODEL_REVISION,
            "quantization_config": quantization,
            "device_map": "auto",
        },
    )
    zero_shot = evaluate_verdicts(trainer.model, processor, test_dataset)
    resume = True if any(checkpoint.glob("checkpoint-*")) else None
    result = trainer.train(resume_from_checkpoint=resume)
    fine_tuned = evaluate_verdicts(trainer.model, processor, test_dataset)
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
        "dataset": (
            "private licensed bundle"
            if using_private_bundle
            else "public CORD v2 deterministic copy-move proxy"
        ),
        "cord_revision": None if using_private_bundle else CORD_REVISION,
        "train_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
        "test_evaluation": {
            "selection": "deterministic balanced subset; 64 per class maximum",
            "zero_shot": zero_shot,
            "fine_tuned": fine_tuned,
        },
        "train_metrics": result.metrics,
        "duration_seconds": time.perf_counter() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1048576,
        "gpu": torch.cuda.get_device_name(0),
        "adapter_path": str(adapter),
        "runtime_versions": {
            "accelerate": accelerate.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "datasets": datasets.__version__,
            "peft": peft.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
        },
    }
    (OUTPUT_ROOT / "record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
