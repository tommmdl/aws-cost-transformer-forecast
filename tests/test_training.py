import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aws_cost_forecast.model.transformer import TimeSeriesTransformer
from aws_cost_forecast.training.train import (
    build_dataloaders,
    create_sequences,
    evaluate,
    load_checkpoint,
    main,
    save_checkpoint,
    select_device,
    sequences_to_tensors,
    train_one_epoch,
)


def test_create_sequences_produces_expected_count():
    data = np.arange(10, dtype=np.float32).reshape(-1, 1)

    sequences = create_sequences(data, input_window=4)

    assert len(sequences) == 6


def test_create_sequences_matches_expected_values():
    data = np.arange(6, dtype=np.float32).reshape(-1, 1)

    sequences = create_sequences(data, input_window=3)

    seq0, label0 = sequences[0]
    np.testing.assert_array_equal(seq0, data[0:3])
    np.testing.assert_array_equal(label0, data[3:4])


def test_sequences_to_tensors_shapes():
    data = np.arange(10, dtype=np.float32).reshape(-1, 1)
    sequences = create_sequences(data, input_window=4)

    X, y = sequences_to_tensors(sequences)

    assert X.shape == (len(sequences), 4, 1)
    assert y.shape == (len(sequences), 1, 1)
    assert X.dtype == torch.float32


def test_build_dataloaders_returns_correct_batch_counts():
    X_train, y_train = torch.zeros(10, 4, 1), torch.zeros(10, 1, 1)
    X_test, y_test = torch.zeros(6, 4, 1), torch.zeros(6, 1, 1)

    train_loader, test_loader = build_dataloaders(
        X_train, y_train, X_test, y_test, batch_size=4
    )

    assert len(train_loader) == 3
    assert len(test_loader) == 2


def test_train_one_epoch_reduces_loss_over_multiple_epochs():
    torch.manual_seed(0)
    X = torch.rand(16, 5, 1)
    y = X.mean(dim=1, keepdim=True)
    loader = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=False)

    model = TimeSeriesTransformer(input_dim=1, d_model=8, nhead=2, num_layers=1, dropout=0.0)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    first_epoch_loss = train_one_epoch(model, loader, criterion, optimizer, device)
    for _ in range(29):
        last_epoch_loss = train_one_epoch(model, loader, criterion, optimizer, device)

    assert last_epoch_loss < first_epoch_loss


def test_evaluate_returns_nonnegative_metrics():
    torch.manual_seed(0)
    X, y = torch.randn(8, 5, 1), torch.randn(8, 1, 1)
    loader = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=False)
    model = TimeSeriesTransformer(input_dim=1, d_model=8, nhead=2, num_layers=1, dropout=0.0)

    metrics = evaluate(model, loader, torch.device("cpu"))

    assert set(metrics) == {"mse", "rmse", "mae"}
    assert metrics["mse"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    model = TimeSeriesTransformer(input_dim=1, d_model=8, nhead=2, num_layers=1, dropout=0.0)
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(np.array([[0.0], [10.0]]))
    config = {"input_dim": 1, "d_model": 8, "nhead": 2, "num_layers": 1, "dropout": 0.0}
    path = tmp_path / "model.pt"

    save_checkpoint(path, model, scaler, config, input_window=5)
    checkpoint = load_checkpoint(path)

    assert checkpoint["model_config"] == config
    assert checkpoint["input_window"] == 5
    assert checkpoint["model_state_dict"].keys() == model.state_dict().keys()
    assert np.allclose(checkpoint["scaler"].data_min_, scaler.data_min_)


def test_select_device_returns_torch_device():
    assert isinstance(select_device(), torch.device)


def test_main_runs_end_to_end_and_saves_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "model.pt"

    metrics = main(
        [
            "--n-days", "150",
            "--seed", "1",
            "--input-window", "5",
            "--train-split", "0.8",
            "--batch-size", "8",
            "--epochs", "1",
            "--d-model", "8",
            "--nhead", "2",
            "--num-layers", "1",
            "--checkpoint-path", str(checkpoint_path),
        ]
    )

    assert checkpoint_path.exists()
    assert set(metrics) == {"mse", "rmse", "mae"}
    assert metrics["mse"] >= 0
