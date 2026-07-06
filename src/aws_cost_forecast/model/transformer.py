"""Transformer for time series forecasting.

Ported from `Projetos/Projeto4.ipynb` (DSA postgraduate coursework),
keeping the original math intact — only adds typing and docstrings.

Architecture: a linear projection takes the input value (the day's cost)
into a `d_model`-dimensional space (a weight matrix, same idea as the
matrices from Projeto 2), positional encoding is added, it goes through
self-attention layers (`nn.TransformerEncoder`), and the result is
aggregated by averaging over time before a final linear projection back
to the input dimension.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from aws_cost_forecast.model.positional_encoding import PositionalEncoding


class TimeSeriesTransformer(nn.Module):
    """Encoder-only Transformer for time series regression."""

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 32,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.model_type = "Transformer"
        self.d_model = d_model

        # Projects the scalar (or multivariate) input into the
        # Transformer's embedding space.
        self.encoder = nn.Linear(input_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )

        # Projects back to the original input dimension (the forecast).
        self.decoder = nn.Linear(d_model, input_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Runs the model's forward pass.

        Args:
            src: tensor of shape ``(batch, seq_len, input_dim)``.

        Returns:
            Tensor of shape ``(batch, input_dim)`` with the forecast for
            the series' next value.
        """
        # Scales by the square root of d_model (convention from
        # "Attention Is All You Need"): without it, the learned embedding
        # would be too small compared to the positional encoding, which
        # is already O(1).
        src = self.encoder(src) * math.sqrt(self.d_model)

        src = self.pos_encoder(src)

        output = self.transformer_encoder(src)

        # Averaging over time is more stable than using only the
        # sequence's last step (reduces variance, like a sample mean).
        output = output.mean(dim=1)

        output = self.decoder(output)

        return output
