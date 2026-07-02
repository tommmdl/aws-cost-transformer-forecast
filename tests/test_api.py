import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from aws_cost_forecast.api.inference import (
    ModelBundle,
    forecast_from_history,
    forecast_future,
    load_model_bundle,
)
from aws_cost_forecast.api.main import app, get_model_bundle
from aws_cost_forecast.model.transformer import TimeSeriesTransformer
from aws_cost_forecast.training.train import save_checkpoint


def _build_tiny_bundle(input_window: int = 5) -> ModelBundle:
    model = TimeSeriesTransformer(input_dim=1, d_model=8, nhead=2, num_layers=1, dropout=0.0)
    model.eval()
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(np.array([[0.0], [100.0]]))
    return ModelBundle(
        model=model, scaler=scaler, input_window=input_window, device=torch.device("cpu")
    )


@pytest.fixture
def client_with_model():
    bundle = _build_tiny_bundle(input_window=5)
    app.dependency_overrides[get_model_bundle] = lambda: bundle
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.pop(get_model_bundle, None)


@pytest.fixture
def client_without_model():
    app.dependency_overrides[get_model_bundle] = lambda: None
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.pop(get_model_bundle, None)


def test_health_returns_ok_and_model_loaded_true(client_with_model):
    response = client_with_model.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_returns_model_loaded_false_when_no_model(client_without_model):
    response = client_without_model.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}


def test_forecast_returns_requested_number_of_steps(client_with_model):
    payload = {"historical_costs": [10.0, 20.0, 30.0, 40.0, 50.0], "steps": 3}

    response = client_with_model.post("/forecast", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["steps"] == 3
    assert len(body["forecast"]) == 3


def test_forecast_rejects_wrong_history_length(client_with_model):
    payload = {"historical_costs": [10.0, 20.0], "steps": 3}

    response = client_with_model.post("/forecast", json=payload)

    assert response.status_code == 422


def test_forecast_returns_503_when_model_not_loaded(client_without_model):
    payload = {"historical_costs": [10.0, 20.0, 30.0, 40.0, 50.0], "steps": 3}

    response = client_without_model.post("/forecast", json=payload)

    assert response.status_code == 503


def test_forecast_future_output_length_matches_steps():
    torch.manual_seed(0)
    model = TimeSeriesTransformer(input_dim=1, d_model=8, nhead=2, num_layers=1, dropout=0.0)
    start_sequence = torch.zeros(5, 1)

    result = forecast_future(model, start_sequence, steps=4, device=torch.device("cpu"))

    assert result.shape == (4,)


class _ConstantModel(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full((x.shape[0], 1), self.value)


def test_forecast_from_history_inverse_transforms_correctly():
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(np.array([[0.0], [100.0]]))
    model = _ConstantModel(value=0.0)

    result = forecast_from_history(
        model, scaler, historical_costs=[10.0, 20.0, 30.0], steps=2, device=torch.device("cpu")
    )

    assert len(result) == 2
    assert result[0] == pytest.approx(50.0)
    assert result[1] == pytest.approx(50.0)


def test_load_model_bundle_returns_none_when_checkpoint_missing(tmp_path):
    result = load_model_bundle(tmp_path / "does_not_exist.pt")

    assert result is None


def test_load_model_bundle_round_trip(tmp_path):
    model = TimeSeriesTransformer(input_dim=1, d_model=8, nhead=2, num_layers=1, dropout=0.0)
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(np.array([[0.0], [100.0]]))
    config = {"input_dim": 1, "d_model": 8, "nhead": 2, "num_layers": 1, "dropout": 0.0}
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(checkpoint_path, model, scaler, config, input_window=5)

    bundle = load_model_bundle(checkpoint_path, device=torch.device("cpu"))

    assert bundle is not None
    assert bundle.input_window == 5
    assert isinstance(bundle.model, TimeSeriesTransformer)
