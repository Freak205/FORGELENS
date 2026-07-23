"""Temperature scaling fitted on validation logits only."""

import torch
from torch import Tensor, nn


class TemperatureScaler(nn.Module):
    """Positive scalar temperature for binary logits."""

    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> Tensor:
        """Return the positive temperature."""
        return self.log_temperature.exp()

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature.clamp_min(1e-4)

    def fit(self, validation_logits: Tensor, validation_targets: Tensor) -> float:
        """Fit on held-out validation data and return final loss."""
        self.train()
        optimizer = torch.optim.LBFGS(
            [self.log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe"
        )
        criterion = nn.BCEWithLogitsLoss()

        def closure() -> Tensor:
            optimizer.zero_grad()
            loss: Tensor = criterion(
                self(validation_logits), validation_targets.float()
            )
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        optimizer.step(closure)  # type: ignore[no-untyped-call]
        self.eval()
        with torch.no_grad():
            final_loss = criterion(self(validation_logits), validation_targets.float())
        return float(final_loss.item())
