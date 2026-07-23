"""Small, auditable baselines used before adding multimodal complexity."""

from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class DetectorOutput:
    """Joint image-level and pixel-level logits."""

    image_logits: Tensor
    mask_logits: Tensor


class ConvBlock(nn.Sequential):
    """Two convolutions at a fixed spatial resolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class TinyJointDetector(nn.Module):
    """Tiny RGB classification and localization baseline."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        self.encoder1 = ConvBlock(3, base_channels)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Linear(base_channels * 4, 1)
        self.mask_head = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1),
        )

    def forward(self, images: Tensor) -> DetectorOutput:
        """Return logits at image level and original spatial resolution."""
        input_size = images.shape[-2:]
        features = self.pool(self.encoder1(images))
        features = self.pool(self.encoder2(features))
        features = self.encoder3(features)
        pooled = functional.adaptive_avg_pool2d(features, 1).flatten(1)
        image_logits = self.classifier(pooled).squeeze(1)
        mask_logits = functional.interpolate(
            self.mask_head(features),
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        return DetectorOutput(image_logits=image_logits, mask_logits=mask_logits)
