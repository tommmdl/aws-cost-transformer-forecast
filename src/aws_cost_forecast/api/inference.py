"""Lógica de inferência: carregar checkpoint e prever custo futuro.

Separado de `main.py` para manter as funções puras (sem FastAPI) e
testáveis isoladamente. O forecast multi-passo é autorregressivo, portado
de `dsa_previsao_futuro` no notebook original: a cada passo o modelo prevê
o próximo valor, ele é anexado ao final da janela e o valor mais antigo é
descartado, então a mesma janela (tamanho fixo) sempre alimenta o próximo
passo.
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
    """Modelo treinado, scaler e metadados necessários para servir previsões."""

    model: nn.Module
    scaler: MinMaxScaler
    input_window: int
    device: torch.device


def load_model_bundle(
    checkpoint_path: str | Path, device: torch.device | None = None
) -> ModelBundle | None:
    """Carrega um checkpoint salvo por `training.train.save_checkpoint`.

    Returns:
        ``None`` se o arquivo de checkpoint não existir (ex.: antes do
        primeiro treino), para que a API suba mesmo sem modelo treinado.
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
    """Prevê `steps` passos à frente, de forma autorregressiva.

    Args:
        start_sequence: janela normalizada, shape ``(input_window, 1)``.
        steps: quantidade de passos futuros a prever.

    Returns:
        Array de shape ``(steps,)`` com as previsões, ainda na escala
        normalizada.
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
    """Normaliza o histórico, prevê o futuro e desnormaliza o resultado."""
    history = np.array(historical_costs, dtype=np.float32).reshape(-1, 1)
    history_scaled = scaler.transform(history).astype(np.float32)
    start_sequence = torch.tensor(history_scaled, dtype=torch.float32)

    forecast_scaled = forecast_future(model, start_sequence, steps, device)
    forecast_original = scaler.inverse_transform(forecast_scaled.reshape(-1, 1))

    return forecast_original.flatten().tolist()
