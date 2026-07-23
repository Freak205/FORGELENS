"""Deterministic document-capture corruptions for evaluation only."""

from io import BytesIO
from typing import Literal

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import InterpolationMode  # type: ignore[import-untyped]
from torchvision.transforms import functional

CorruptionName = Literal[
    "jpeg",
    "blur",
    "rotation",
    "perspective",
    "low_illumination",
    "noise",
    "resize",
    "screenshot",
    "print_scan",
]


def _validate(image: Tensor, severity: int) -> None:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape [3, height, width]")
    if severity not in {1, 2, 3, 4, 5}:
        raise ValueError("severity must be an integer from 1 to 5")


def apply_corruption(
    image: Tensor,
    name: CorruptionName,
    severity: int,
    seed: int = 20260723,
) -> Tensor:
    """Apply a deterministic corruption while preserving tensor shape."""
    _validate(image, severity)
    height, width = image.shape[-2:]
    generator = torch.Generator(device=image.device).manual_seed(seed)
    result = image.detach().clone().clamp(0.0, 1.0)

    if name == "jpeg":
        quality = 95 - 14 * severity
        pil_image = functional.to_pil_image(result.cpu())
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            result = functional.pil_to_tensor(decoded.convert("RGB")).float() / 255.0
        result = result.to(image.device)
    elif name == "blur":
        kernel = 1 + 2 * severity
        result = functional.gaussian_blur(result, [kernel, kernel])
    elif name == "rotation":
        angle = float(severity * 1.5)
        result = functional.rotate(
            result,
            angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=1.0,
        )
    elif name == "perspective":
        displacement = max(1, round(min(height, width) * severity * 0.012))
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [
            [displacement, displacement],
            [width - 1 - displacement, 0],
            [width - 1, height - 1 - displacement],
            [0, height - 1],
        ]
        result = functional.perspective(
            result,
            startpoints,
            endpoints,
            interpolation=InterpolationMode.BILINEAR,
            fill=1.0,
        )
    elif name == "low_illumination":
        result = (result * (1.0 - 0.12 * severity)).pow(1.0 + 0.12 * severity)
    elif name == "noise":
        noise = torch.randn(
            result.shape,
            generator=generator,
            device=result.device,
            dtype=result.dtype,
        )
        result = result + noise * (0.01 * severity)
    elif name == "resize":
        scale = 1.0 - 0.12 * severity
        reduced = functional.resize(
            result,
            [max(8, round(height * scale)), max(8, round(width * scale))],
            interpolation=InterpolationMode.BILINEAR,
        )
        result = functional.resize(
            reduced, [height, width], interpolation=InterpolationMode.BILINEAR
        )
    elif name == "screenshot":
        scale = 1.0 - 0.08 * severity
        reduced = functional.resize(
            result,
            [max(8, round(height * scale)), max(8, round(width * scale))],
            interpolation=InterpolationMode.BILINEAR,
        )
        result = functional.resize(
            reduced, [height, width], interpolation=InterpolationMode.NEAREST
        )
        levels = float(max(8, 64 - 8 * severity))
        result = torch.round(result * levels) / levels
    elif name == "print_scan":
        grayscale = functional.rgb_to_grayscale(result, num_output_channels=3)
        result = 0.75 * grayscale + 0.25 * result
        kernel = 1 + 2 * min(severity, 3)
        result = functional.gaussian_blur(result, [kernel, kernel])
        noise = torch.randn(
            result.shape,
            generator=generator,
            device=result.device,
            dtype=result.dtype,
        )
        result = result + noise * (0.006 * severity)
    else:
        raise ValueError(f"unsupported corruption: {name}")
    return result.clamp(0.0, 1.0)
