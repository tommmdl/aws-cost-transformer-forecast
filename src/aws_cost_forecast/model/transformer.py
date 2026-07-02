"""Transformer para previsão de séries temporais.

Portado de `Projetos/Projeto4.ipynb` (pós-graduação DSA), mantendo a lógica
matemática original intacta — só adiciona tipagem e docstrings.

Arquitetura: uma projeção linear leva o valor de entrada (custo do dia) para
um espaço de dimensão `d_model` (matriz de pesos, igual às matrizes do
Projeto 2), soma-se a codificação posicional, passa por camadas de
self-attention (`nn.TransformerEncoder`) e o resultado é agregado por média
ao longo do tempo antes de uma projeção linear final de volta à dimensão de
entrada.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from aws_cost_forecast.model.positional_encoding import PositionalEncoding


class TimeSeriesTransformer(nn.Module):
    """Transformer encoder-only para regressão de séries temporais."""

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

        # Projeta a entrada escalar (ou multivariada) para o espaço de
        # embedding do Transformer.
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

        # Projeta de volta à dimensão original da entrada (a previsão).
        self.decoder = nn.Linear(d_model, input_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Executa o forward pass do modelo.

        Args:
            src: tensor de shape ``(batch, seq_len, input_dim)``.

        Returns:
            Tensor de shape ``(batch, input_dim)`` com a previsão do
            próximo valor da série.
        """
        # Escala pela raiz de d_model (convenção de "Attention Is All You
        # Need"): sem isso, o embedding aprendido ficaria pequeno demais
        # frente à codificação posicional, que já é O(1).
        src = self.encoder(src) * math.sqrt(self.d_model)

        src = self.pos_encoder(src)

        output = self.transformer_encoder(src)

        # Média ao longo do tempo é mais estável que usar só o último
        # passo da sequência (reduz variância, como uma média amostral).
        output = output.mean(dim=1)

        output = self.decoder(output)

        return output
