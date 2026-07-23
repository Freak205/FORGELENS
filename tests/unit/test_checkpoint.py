from pathlib import Path

import torch

from forgelens.models import TinyJointDetector
from forgelens.training import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = TinyJointDetector(base_channels=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, 3, {"loss": 0.5}, {"seed": 7})
    expected = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    payload = load_checkpoint(path, model, optimizer)
    assert payload["epoch"] == 3
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, expected[name])
