"""Inference logic: load a checkpoint and forecast future cost.

Kept separate from `main.py` to keep the functions pure (no FastAPI) and
independently testable. The multi-step forecast is autoregressive, ported
from `dsa_previsao_futuro` in the original notebook: at each step the model
predicts the next value, it gets appended to the end of the window, and the
oldest value is dropped, so the same fixed-size window always feeds the
next step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from aws_cost_forecast.model.transformer import TimeSeriesTransformer
from aws_cost_forecast.training.train import load_checkpoint, select_device


@dataclass
class ModelBundle:
    """Trained model, scaler and metadata needed to serve forecasts."""

    model: nn.Module
    scaler: MinMaxScaler
    input_window: int
    device: torch.device


def load_model_bundle(
    checkpoint_path: str | Path, device: torch.device | None = None
) -> ModelBundle | None:
    """Loads a checkpoint saved by `training.train.save_checkpoint`.

    Returns:
        ``None`` if the checkpoint file doesn't exist (e.g. before the
        first training run), so the API can still start without a
        trained model.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None

    device = device or select_device()
    checkpoint = load_checkpoint(checkpoint_path)

    model = TimeSeriesTransformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return ModelBundle(
        model=model,
        scaler=checkpoint["scaler"],
        input_window=checkpoint["input_window"],
        device=device,
    )


def forecast_future(
    model: nn.Module, start_sequence: torch.Tensor, steps: int, device: torch.device
) -> np.ndarray:
    """Forecasts `steps` steps ahead, autoregressively.

    Args:
        start_sequence: normalized window, shape ``(input_window, 1)``.
        steps: number of future steps to forecast.

    Returns:
        Array of shape ``(steps,)`` with the forecasts, still on the
        normalized scale.
    """
    model.eval()
    future_predictions: list[float] = []
    current_sequence = start_sequence.clone().detach().to(device)

    with torch.no_grad():
        for _ in range(steps):
            next_pred = model(current_sequence.unsqueeze(0))
            future_predictions.append(next_pred.item())

            next_pred_tensor = torch.tensor([[next_pred.item()]], device=device)
            current_sequence = torch.cat((current_sequence[1:], next_pred_tensor), dim=0)

    return np.array(future_predictions)


def forecast_from_history(
    model: nn.Module,
    scaler: MinMaxScaler,
    historical_costs: list[float],
    steps: int,
    device: torch.device,
) -> list[float]:
    """Normalizes the history, forecasts the future and denormalizes the result."""
    history = np.array(historical_costs, dtype=np.float32).reshape(-1, 1)
    history_scaled = scaler.transform(history).astype(np.float32)
    start_sequence = torch.tensor(history_scaled, dtype=torch.float32)

    forecast_scaled = forecast_future(model, start_sequence, steps, device)
    forecast_original = scaler.inverse_transform(forecast_scaled.reshape(-1, 1))

    return forecast_original.flatten().tolist()
