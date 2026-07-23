import torch

from forgelens.models import (
    ResidualUNetJointDetector,
    TinyJointDetector,
    TinyUNetJointDetector,
)


def test_joint_detector_shapes_and_gradients() -> None:
    model = TinyJointDetector(base_channels=4)
    images = torch.rand(2, 3, 32, 48)
    output = model(images)
    assert output.image_logits.shape == (2,)
    assert output.mask_logits.shape == (2, 1, 32, 48)
    (output.image_logits.mean() + output.mask_logits.mean()).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_unet_joint_detector_shapes_and_gradients() -> None:
    model = TinyUNetJointDetector(base_channels=4)
    images = torch.rand(2, 3, 32, 48)
    output = model(images)
    assert output.image_logits.shape == (2,)
    assert output.mask_logits.shape == (2, 1, 32, 48)
    (output.image_logits.mean() + output.mask_logits.mean()).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_residual_unet_has_fixed_evidence_filters() -> None:
    model = ResidualUNetJointDetector(base_channels=4)
    images = torch.rand(2, 3, 32, 48)
    output = model(images)
    assert output.mask_logits.shape == (2, 1, 32, 48)
    assert model.residual_kernels.requires_grad is False
    output.image_logits.mean().backward()
    assert model.encoder1[0].weight.grad is not None
