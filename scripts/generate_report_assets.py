"""Generate evidence-backed report figures from the committed checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader

from forgelens.data import DocumentSample, ManifestDocumentDataset
from forgelens.models import ResidualUNetJointDetector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = PROJECT_ROOT.parent
FIGURES = PROJECT_ROOT / "reports" / "figures"
CHECKPOINT = PROJECT_ROOT / "artifacts/experiments/RESIDUAL-COPYMOVE-001/best.pt"
TEMPERATURE = 1.0000354051589966


def collate(samples: list[DocumentSample]) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.stack([sample.image for sample in samples]),
        torch.stack([sample.label for sample in samples]),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def calibration_svg(bins: list[dict[str, float | int]]) -> str:
    width, height, margin = 720, 460, 60
    plot_width, plot_height = width - 2 * margin, height - 2 * margin
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#0d1117"/>',
        '<g stroke="#8b949e" fill="none">',
        f'<path d="M{margin} {margin}V{height - margin}H{width - margin}"/>',
        f'<path d="M{margin} {height - margin}L{width - margin} {margin}" stroke-dasharray="6 5"/>',
        "</g>",
        '<g fill="#e6edf3" font-family="system-ui" font-size="14">',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="20">Residual baseline reliability diagram</text>',
        f'<text x="{width / 2}" y="{height - 12}" text-anchor="middle">Mean predicted risk</text>',
        f'<text transform="translate(18 {height / 2}) rotate(-90)" text-anchor="middle">Observed forged rate</text>',
        "</g>",
    ]
    points: list[str] = []
    for item in bins:
        count = int(item["count"])
        if count == 0:
            continue
        confidence = float(item["confidence"])
        accuracy = float(item["accuracy"])
        x = margin + confidence * plot_width
        y = height - margin - accuracy * plot_height
        points.append(f"{x:.2f},{y:.2f}")
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{4 + min(count, 50) / 12:.2f}" '
            'fill="#58a6ff" opacity="0.85"/>'
        )
    if points:
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="#58a6ff" stroke-width="2"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def ablation_svg() -> str:
    records = [
        ("RGB", 0.5476, 0.4679, 0.6280),
        ("U-Net", 0.5090, 0.4360, 0.5870),
        ("Residual", 0.5255, 0.4521, 0.6103),
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420">',
        '<rect width="100%" height="100%" fill="#0d1117"/>',
        '<text x="360" y="30" fill="#e6edf3" font-family="system-ui" font-size="20" text-anchor="middle">Proxy benchmark ROC-AUC (95% bootstrap CI)</text>',
        '<path d="M80 330H680" stroke="#8b949e"/>',
        '<path d="M80 280H680" stroke="#d29922" stroke-dasharray="6 5"/>',
        '<text x="70" y="285" fill="#d29922" text-anchor="end">0.5</text>',
    ]
    for index, (name, estimate, lower, upper) in enumerate(records):
        x = 180 + index * 190
        y = 380 - estimate * 200
        low_y = 380 - lower * 200
        high_y = 380 - upper * 200
        parts.extend(
            [
                f'<line x1="{x}" y1="{low_y:.1f}" x2="{x}" y2="{high_y:.1f}" stroke="#e6edf3" stroke-width="3"/>',
                f'<circle cx="{x}" cy="{y:.1f}" r="9" fill="#f85149"/>',
                f'<text x="{x}" y="360" fill="#e6edf3" font-family="system-ui" text-anchor="middle">{name}</text>',
                f'<text x="{x}" y="{y - 16:.1f}" fill="#e6edf3" font-family="system-ui" text-anchor="middle">{estimate:.3f}</text>',
            ]
        )
    parts.append("</svg>")
    return "".join(parts)


def tensor_image(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(array)


def save_examples(
    model: ResidualUNetJointDetector,
    dataset: ManifestDocumentDataset,
    device: torch.device,
) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    forged_indices = [
        index for index, record in enumerate(dataset.records) if record["label"] == 1
    ][:3]
    for ordinal, index in enumerate(forged_indices, start=1):
        sample = dataset[index]
        with torch.inference_mode():
            probability = (
                model(sample.image.unsqueeze(0).to(device))
                .mask_logits.sigmoid()[0, 0]
                .cpu()
            )
        input_path = FIGURES / f"failure_{ordinal}_input.png"
        target_path = FIGURES / f"failure_{ordinal}_target.png"
        overlay_path = FIGURES / f"failure_{ordinal}_prediction.png"
        tensor_image(sample.image).save(input_path)
        Image.fromarray((sample.mask[0].numpy() * 255).astype("uint8")).save(
            target_path
        )
        base = tensor_image(sample.image).convert("RGBA")
        red = Image.new("RGBA", base.size, (255, 45, 45, 0))
        red.putalpha(
            Image.fromarray((probability.numpy().clip(0, 1) * 180).astype("uint8"))
        )
        Image.alpha_composite(base, red).convert("RGB").save(overlay_path)
        examples.append(
            {
                "sample_id": sample.sample_id,
                "input": str(input_path.relative_to(PROJECT_ROOT)),
                "target": str(target_path.relative_to(PROJECT_ROOT)),
                "prediction": str(overlay_path.relative_to(PROJECT_ROOT)),
            }
        )
    return examples


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualUNetJointDetector(base_channels=8)
    model.load_state_dict(payload["model"])
    model = model.to(device).eval()
    dataset = ManifestDocumentDataset(
        PROJECT_ROOT / "configs/data/cord_copy_move_v1.json",
        STORAGE_ROOT,
        "test",
        (192, 288),
    )
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for images, batch_labels in DataLoader(
        dataset, batch_size=8, collate_fn=collate, num_workers=0
    ):
        with torch.inference_mode():
            logits.append(model(images.to(device)).image_logits.cpu())
        labels.append(batch_labels)
    probabilities = (torch.cat(logits) / TEMPERATURE).sigmoid()
    targets = torch.cat(labels)
    bins: list[dict[str, float | int]] = []
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        selected = (probabilities >= lower) & (
            probabilities <= upper if index == 9 else probabilities < upper
        )
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": int(selected.sum()),
                "confidence": (
                    float(probabilities[selected].mean()) if selected.any() else 0.0
                ),
                "accuracy": (
                    float(targets[selected].float().mean()) if selected.any() else 0.0
                ),
            }
        )
    calibration_path = FIGURES / "calibration_residual.svg"
    ablation_path = FIGURES / "ablation_auc.svg"
    calibration_path.write_text(calibration_svg(bins), encoding="utf-8")
    ablation_path.write_text(ablation_svg(), encoding="utf-8")
    examples = save_examples(model, dataset, device)
    figure_hashes = {
        path.name: sha256(path) for path in FIGURES.iterdir() if path.is_file()
    }
    record = {
        "checkpoint_sha256": sha256(CHECKPOINT),
        "temperature": TEMPERATURE,
        "calibration_bins": bins,
        "examples": examples,
        "figure_sha256": figure_hashes,
    }
    (FIGURES / "assets_manifest.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps({"figures": len(figure_hashes)}, indent=2))


if __name__ == "__main__":
    main()
