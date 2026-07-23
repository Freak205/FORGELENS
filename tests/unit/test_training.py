import torch

from forgelens.data import FictionalDocumentFixtures
from forgelens.models import TinyJointDetector
from forgelens.training import JointTrainer, seed_everything


def fixture_batches() -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    dataset = FictionalDocumentFixtures(size=4, height=32, width=48)
    samples = [dataset[index] for index in range(4)]
    return [
        (
            torch.stack([sample.image for sample in samples]),
            torch.stack([sample.mask for sample in samples]),
            torch.stack([sample.label for sample in samples]),
        )
    ]


def test_joint_trainer_cpu_epoch_and_evaluation() -> None:
    seed_everything(7)
    model = TinyJointDetector(base_channels=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = JointTrainer(
        model,
        optimizer,
        torch.device("cpu"),
        gradient_accumulation_steps=2,
        mixed_precision=True,
    )
    train_metrics = trainer.train_epoch(fixture_batches())
    validation_metrics = trainer.evaluate(fixture_batches())
    assert train_metrics.total_loss > 0.0
    assert validation_metrics.total_loss > 0.0
    assert train_metrics.batches == 1


def test_joint_trainer_rejects_empty_batches() -> None:
    model = TinyJointDetector(base_channels=2)
    trainer = JointTrainer(
        model,
        torch.optim.AdamW(model.parameters()),
        torch.device("cpu"),
    )
    try:
        trainer.train_epoch([])
    except ValueError as error:
        assert "cannot be empty" in str(error)
    else:
        raise AssertionError("empty batches should be rejected")
