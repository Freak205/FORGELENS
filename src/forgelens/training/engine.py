"""Reusable joint classification/localization training engine."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
from torch.optim import Optimizer

from forgelens.models import TinyJointDetector


@dataclass(frozen=True)
class EpochMetrics:
    """Losses measured over one epoch."""

    total_loss: float
    classification_loss: float
    localization_loss: float
    batches: int


class JointTrainer:
    """Memory-aware trainer for the baseline joint detector."""

    def __init__(
        self,
        model: TinyJointDetector,
        optimizer: Optimizer,
        device: torch.device,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = True,
        localization_weight: float = 1.0,
    ) -> None:
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient accumulation steps must be positive")
        if localization_weight <= 0.0:
            raise ValueError("localization weight must be positive")
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision and device.type == "cuda"
        self.localization_weight = localization_weight
        self.scaler = GradScaler(device.type, enabled=self.mixed_precision)
        self.classification_criterion = nn.BCEWithLogitsLoss()
        self.localization_criterion = nn.BCEWithLogitsLoss()

    def _losses(
        self, images: Tensor, masks: Tensor, labels: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        output = self.model(images)
        classification = self.classification_criterion(output.image_logits, labels)
        localization = self.localization_criterion(output.mask_logits, masks)
        total = classification + self.localization_weight * localization
        return total, classification, localization

    def train_epoch(self, batches: list[tuple[Tensor, Tensor, Tensor]]) -> EpochMetrics:
        """Train one epoch with AMP, accumulation, and NaN detection."""
        if not batches:
            raise ValueError("training batches cannot be empty")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(3)
        for index, (images, masks, labels) in enumerate(batches):
            images = images.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            with autocast(
                device_type=self.device.type,
                enabled=self.mixed_precision,
            ):
                total, classification, localization = self._losses(
                    images, masks, labels
                )
                scaled_total = total / self.gradient_accumulation_steps
            if not torch.isfinite(total):
                raise FloatingPointError("non-finite loss detected")
            self.scaler.scale(scaled_total).backward()  # type: ignore[no-untyped-call]
            should_step = (
                index + 1
            ) % self.gradient_accumulation_steps == 0 or index + 1 == len(batches)
            if should_step:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            totals += torch.tensor(
                [
                    float(total.item()),
                    float(classification.item()),
                    float(localization.item()),
                ]
            )
        averages = totals / len(batches)
        return EpochMetrics(
            total_loss=float(averages[0].item()),
            classification_loss=float(averages[1].item()),
            localization_loss=float(averages[2].item()),
            batches=len(batches),
        )

    @torch.no_grad()
    def evaluate(self, batches: list[tuple[Tensor, Tensor, Tensor]]) -> EpochMetrics:
        """Evaluate losses without optimizer state changes."""
        if not batches:
            raise ValueError("evaluation batches cannot be empty")
        self.model.eval()
        totals = torch.zeros(3)
        for images, masks, labels in batches:
            total, classification, localization = self._losses(
                images.to(self.device),
                masks.to(self.device),
                labels.to(self.device),
            )
            totals += torch.tensor(
                [
                    float(total.item()),
                    float(classification.item()),
                    float(localization.item()),
                ]
            )
        averages = totals / len(batches)
        return EpochMetrics(
            total_loss=float(averages[0].item()),
            classification_loss=float(averages[1].item()),
            localization_loss=float(averages[2].item()),
            batches=len(batches),
        )
