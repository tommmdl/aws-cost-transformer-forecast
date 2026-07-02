"""Codificação posicional sinusoidal para embeddings de um Transformer.

Portado de `Projetos/Projeto4.ipynb` (pós-graduação DSA), mantendo a lógica
matemática original intacta — só adiciona tipagem e docstrings.

Self-attention é permutation-equivariant: o mecanismo calcula uma soma
ponderada por similaridade entre posições, sem nenhuma noção implícita de
ordem. Sem codificação posicional, embaralhar a sequência de entrada não
mudaria a saída do bloco de atenção. A codificação abaixo injeta a posição
somando, a cada dimensão do embedding, uma onda senoidal com frequência
diferente — uma base tipo Fourier para posição.

Por que seno/cosseno (e não o índice de posição puro): os valores ficam
limitados a [-1, 1] independente do tamanho da sequência, e a identidade de
soma de ângulos (`sin(a+b) = sin(a)cos(b) + cos(a)sin(b)`) garante que
`PE(pos + k)` é uma transformação linear fixa de `PE(pos)` — deslocar a
posição em `k` passos equivale a aplicar uma matriz de rotação sobre o vetor
de codificação, análogo à mudança de base via matriz de autovetores em PCA,
só que aqui a base é fixa (não aprendida).
"""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Soma uma codificação posicional sinusoidal fixa ao embedding de entrada."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Matriz de codificação posicional: uma linha por posição, uma coluna
        # por dimensão do embedding.
        pe = torch.zeros(max_len, d_model)

        # Vetor de posições 0..max_len-1.
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Termo divisor: controla a frequência de oscilação de cada dimensão.
        # Calculado via exp(log(...)) por estabilidade numérica, evitando
        # calcular 10000**(2i/d_model) diretamente.
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Seno nas dimensões pares, cosseno nas ímpares.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Dimensão extra no início para broadcast sobre o batch.
        pe = pe.unsqueeze(0)

        # Buffer, não parâmetro: não é aprendido via gradiente, mas viaja
        # com o modelo (device, state_dict).
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Soma a codificação posicional ao embedding e aplica dropout.

        Args:
            x: tensor de shape ``(batch, seq_len, d_model)``.

        Returns:
            Tensor de mesma shape que ``x``.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
