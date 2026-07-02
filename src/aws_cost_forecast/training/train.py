"""Script de treino do TimeSeriesTransformer sobre custo AWS sintético.

Pipeline portado de `Projetos/Projeto4.ipynb`: gera a série sintética,
normaliza com `MinMaxScaler` (ajustado **somente** nos dados de treino, para
não vazar a distribuição do período de teste para dentro do treino), cria
sequências por janela deslizante, treina com `nn.MSELoss` + `torch.optim.Adam`
e salva um checkpoint com os pesos do modelo, o scaler e a configuração
necessária para reconstruir o modelo depois (usado pela API de inferência).

MSE penaliza quadraticamente os erros, forçando o modelo a evitar grandes
desvios de custo. Adam combina momentum (média móvel do gradiente) com
RMSProp (taxa de aprendizado adaptada por parâmetro, dividida pela raiz da
média móvel do quadrado do gradiente) — uma forma adaptativa da regra da
cadeia usada em gradiente descendente.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aws_cost_forecast.data.synthetic_aws_cost import generate_synthetic_aws_cost
from aws_cost_forecast.model.transformer import TimeSeriesTransformer

DEFAULT_CHECKPOINT_PATH = Path("checkpoints/model.pt")


def select_device() -> torch.device:
    """Seleciona GPU (CUDA ou Apple MPS) se disponível, senão CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_sequences(
    data: np.ndarray, input_window: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cria janelas deslizantes (sequência de entrada, rótulo seguinte).

    Args:
        data: série normalizada, shape ``(n_pontos, n_features)``.
        input_window: tamanho da janela de entrada.

    Returns:
        Lista de tuplas ``(seq, label)``, ``seq`` com shape
        ``(input_window, n_features)`` e ``label`` com shape
        ``(1, n_features)``.
    """
    sequences = []
    length = len(data)
    for i in range(length - input_window):
        seq = data[i : i + input_window]
        label = data[i + input_window : i + input_window + 1]
        sequences.append((seq, label))
    return sequences


def sequences_to_tensors(
    sequences: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Converte a lista de sequências em tensores ``(X, y)``."""
    X = torch.tensor(np.array([item[0] for item in sequences]), dtype=torch.float32)
    y = torch.tensor(np.array([item[1] for item in sequences]), dtype=torch.float32)
    return X, y


def build_dataloaders(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Constrói os DataLoaders de treino (embaralhado) e teste (em ordem)."""
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Executa uma época de treino e retorna a perda média."""
    model.train()
    total_loss = 0.0
    for seq, label in loader:
        seq, label = seq.to(device), label.to(device)

        optimizer.zero_grad()
        y_pred = model(seq)
        loss = criterion(y_pred, label.squeeze(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Avalia o modelo em modo eval, retornando MSE, RMSE e MAE."""
    model.eval()
    predictions: list[np.ndarray] = []
    actuals: list[np.ndarray] = []

    with torch.no_grad():
        for seq, label in loader:
            seq, label = seq.to(device), label.to(device)
            y_pred = model(seq)
            predictions.extend(y_pred.cpu().numpy())
            actuals.extend(label.cpu().numpy())

    predictions_arr = np.array(predictions)
    actuals_arr = np.array(actuals).reshape(-1, predictions_arr.shape[-1])

    mse = mean_squared_error(actuals_arr, predictions_arr)
    mae = mean_absolute_error(actuals_arr, predictions_arr)
    return {"mse": mse, "rmse": mse**0.5, "mae": mae}


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    scaler: MinMaxScaler,
    model_config: dict[str, Any],
    input_window: int,
) -> None:
    """Salva pesos do modelo, scaler e configuração para reconstrução futura."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "input_window": input_window,
            "scaler": scaler,
        },
        path,
    )


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Carrega um checkpoint salvo por :func:`save_checkpoint`."""
    return torch.load(path, weights_only=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treina o TimeSeriesTransformer sobre custo AWS sintético."
    )
    parser.add_argument("--n-days", type=int, default=730)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-window", type=int, default=50)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, float]:
    args = build_arg_parser().parse_args(argv)
    device = select_device()

    df = generate_synthetic_aws_cost(n_days=args.n_days, seed=args.seed)
    series = df["cost"].to_numpy().reshape(-1, 1)

    cutoff = int(len(series) * args.train_split)
    series_train, series_test = series[:cutoff], series[cutoff:]

    # Ajusta o scaler SOMENTE no treino: usar min/max da série completa
    # vazaria a distribuição do período de teste para dentro do treino.
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(series_train)
    series_train_scaled = scaler.transform(series_train).astype(np.float32)
    series_test_scaled = scaler.transform(series_test).astype(np.float32)

    train_sequences = create_sequences(series_train_scaled, args.input_window)
    test_sequences = create_sequences(series_test_scaled, args.input_window)

    X_train, y_train = sequences_to_tensors(train_sequences)
    X_test, y_test = sequences_to_tensors(test_sequences)

    train_loader, test_loader = build_dataloaders(
        X_train, y_train, X_test, y_test, args.batch_size
    )

    model_config = {
        "input_dim": 1,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }
    model = TimeSeriesTransformer(**model_config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss média: {avg_loss:.6f}")

    metrics = evaluate(model, test_loader, device)
    print(
        f"MSE: {metrics['mse']:.6f} | RMSE: {metrics['rmse']:.6f} | "
        f"MAE: {metrics['mae']:.6f}"
    )

    save_checkpoint(args.checkpoint_path, model, scaler, model_config, args.input_window)
    print(f"Checkpoint salvo em: {args.checkpoint_path}")

    return metrics


if __name__ == "__main__":
    main()
