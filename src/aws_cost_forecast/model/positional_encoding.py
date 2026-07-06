"""Sinusoidal positional encoding for Transformer embeddings.

Ported from `Projetos/Projeto4.ipynb` (DSA postgraduate coursework),
keeping the original math intact — only adds typing and docstrings.

Self-attention is permutation-equivariant: the mechanism computes a
weighted sum based on similarity between positions, with no implicit
notion of order. Without positional encoding, shuffling the input
sequence would not change the attention block's output. The encoding
below injects position by adding, to each embedding dimension, a
sinusoidal wave with a different frequency — a Fourier-like basis for
position.

Why sine/cosine (and not the raw position index): the values stay bounded
to [-1, 1] regardless of sequence length, and the angle-sum identity
(`sin(a+b) = sin(a)cos(b) + cos(a)sin(b)`) guarantees that `PE(pos + k)`
is a fixed linear transformation of `PE(pos)` — shifting the position by
`k` steps is equivalent to applying a rotation matrix to the encoding
vector, analogous to the change of basis via the eigenvector matrix in
PCA, except here the basis is fixed (not learned).
"""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Adds a fixed sinusoidal positional encoding to the input embedding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Positional encoding matrix: one row per position, one column
        # per embedding dimension.
        pe = torch.zeros(max_len, d_model)

        # Position vector 0..max_len-1.
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Divisor term: controls the oscillation frequency of each
        # dimension. Computed via exp(log(...)) for numerical stability,
        # avoiding computing 10000**(2i/d_model) directly.
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Sine on even dimensions, cosine on odd ones.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Extra leading dimension for broadcasting over the batch.
        pe = pe.unsqueeze(0)

        # Buffer, not a parameter: not learned via gradient, but travels
        # with the model (device, state_dict).
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds the positional encoding to the embedding and applies dropout.

        Args:
            x: tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Tensor of the same shape as ``x``.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
