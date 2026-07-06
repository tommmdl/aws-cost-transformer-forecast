# Architecture

This document explains the math behind `TimeSeriesTransformer`, the model
implemented from scratch in this project (`src/aws_cost_forecast/model/`).
It's meant as the deep dive behind the summary in the main
[README](../README.md).

## Data flow

```
cost(t-49) .. cost(t)              →  input window, shape (batch, 50, 1)
        │
        ▼
Linear(1 → d_model) * sqrt(d_model)  →  learned embedding, scaled
        │
        ▼
   + PositionalEncoding               →  injects order into the embedding
        │
        ▼
TransformerEncoder (nhead=2, num_layers=2, self-attention)
        │
        ▼
   mean(dim=time)                     →  aggregates the sequence into one vector
        │
        ▼
Linear(d_model → 1)                  →  next-day cost forecast
```

Default hyperparameters (`src/aws_cost_forecast/model/transformer.py`):
`d_model=32`, `nhead=2`, `num_layers=2`, `dropout=0.1`, `input_window=50`.

## Why a Transformer needs positional encoding

Self-attention is permutation-equivariant: it computes a weighted sum based
on similarity between positions, with no built-in notion of order. Shuffle
the 50 days in the input window and, without positional encoding, the
model's output would be identical — clearly wrong for a time series, where
"yesterday" and "seven weeks ago" must be distinguishable.

`PositionalEncoding` (`positional_encoding.py`) fixes this by adding a
sinusoidal signal to the embedding — one sine/cosine pair per pair of
dimensions, each pair oscillating at a different frequency (a Fourier-like
basis for position):

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Two properties make this choice work well:

- **Bounded values**: sine/cosine stay in `[-1, 1]` regardless of sequence
  length, unlike using the raw position index.
- **Linear shift**: the angle-sum identity
  (`sin(a+b) = sin(a)cos(b) + cos(a)sin(b)`) guarantees that `PE(pos + k)`
  is a fixed linear transformation of `PE(pos)`. Shifting position by `k`
  steps is equivalent to applying a rotation matrix to the encoding vector —
  the same idea as a change of basis via an eigenvector matrix in PCA,
  except here the basis is fixed and not learned from data.

The encoding is stored as a `register_buffer`, not a learnable parameter:
it isn't updated by gradient descent, but it still moves with the model
(device placement, `state_dict`) — the right tool when a tensor is derived
math, not something to be learned.

## Why scale the embedding by `sqrt(d_model)`

`transformer.py` scales the linear projection's output by
`sqrt(d_model)` before adding the positional encoding — the same
convention from *Attention Is All You Need*. Without it, the learned
embedding (whose scale depends on the weight initialization) would be tiny
compared to the positional encoding, which is already `O(1)` from the
bounded sine/cosine values. Scaling keeps the two signals on comparable
magnitudes so attention can weigh content and position together, instead
of one drowning out the other.

## Why mean-pool over time instead of using the last step

After the `TransformerEncoder` layers, the model averages the output over
the time dimension (`output.mean(dim=1)`) rather than taking only the
sequence's last position. Averaging is more stable than picking a single
step — the same idea as a sample mean having lower variance than any
single observation — and it lets every position in the window contribute
to the forecast, not just the most recent one.

## Training pipeline

`training/train.py` ports the pipeline from the original postgraduate
notebook (`Projetos/Projeto4.ipynb`):

1. **Sliding-window sequencing** (`create_sequences`): each training
   example is a window of `input_window` consecutive days predicting the
   next day.
2. **Normalization without leakage**: `MinMaxScaler(feature_range=(-1, 1))`
   is fitted **only on the training split**. Fitting it on the full series
   would leak statistics from the test period into training — a mistake
   that's easy to make and hard to notice, since it doesn't raise an error,
   it just makes the reported test error look better than it should.
3. **Loss and optimizer**: `nn.MSELoss` penalizes errors quadratically,
   pushing the model to avoid large cost deviations. `torch.optim.Adam`
   combines momentum (a moving average of the gradient) with RMSProp (a
   per-parameter learning rate divided by the root of the moving average of
   the squared gradient) — an adaptive form of the chain rule used in
   gradient descent.
4. **Checkpointing**: `save_checkpoint`/`load_checkpoint` persist the model
   weights, the fitted scaler, and the model config together, so the
   inference API (`api/inference.py`) can reconstruct the exact same model
   and normalization used at training time.

## Autoregressive forecasting

`api/inference.py` generates multi-step forecasts autoregressively
(`forecast_future`): at each step the model predicts the next value, that
value is appended to the end of the window, and the oldest value is
dropped — so a fixed-size window always feeds the next prediction. This
mirrors how the original notebook's `dsa_previsao_futuro` worked, and is
what lets `/forecast` return an arbitrary number of `steps` even though the
model itself only ever predicts one day at a time.
