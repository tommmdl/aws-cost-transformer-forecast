import torch

from aws_cost_forecast.model.positional_encoding import PositionalEncoding
from aws_cost_forecast.model.transformer import TimeSeriesTransformer


def test_positional_encoding_preserves_input_shape():
    pe = PositionalEncoding(d_model=16, dropout=0.0)
    x = torch.zeros(2, 10, 16)

    out = pe(x)

    assert out.shape == x.shape


def test_positional_encoding_matches_sinusoidal_formula_at_position_zero():
    pe = PositionalEncoding(d_model=16, dropout=0.0)

    pos_zero = pe.pe[0, 0, :]

    assert torch.allclose(pos_zero[0::2], torch.zeros(8))
    assert torch.allclose(pos_zero[1::2], torch.ones(8))


def test_positional_encoding_buffer_is_not_a_learnable_parameter():
    pe = PositionalEncoding(d_model=16, dropout=0.0)

    param_names = {name for name, _ in pe.named_parameters()}
    buffer_names = {name for name, _ in pe.named_buffers()}

    assert "pe" not in param_names
    assert "pe" in buffer_names


def test_positional_encoding_without_dropout_is_deterministic_addition():
    pe = PositionalEncoding(d_model=8, dropout=0.0)
    pe.eval()
    x = torch.randn(3, 5, 8)

    out = pe(x)

    assert torch.allclose(out, x + pe.pe[:, :5, :])


def test_transformer_forward_output_shape_matches_input_dim():
    model = TimeSeriesTransformer(
        input_dim=1, d_model=16, nhead=2, num_layers=1, dropout=0.0
    )
    src = torch.randn(4, 20, 1)

    out = model(src)

    assert out.shape == (4, 1)


def test_transformer_supports_multivariate_input_dim():
    model = TimeSeriesTransformer(
        input_dim=3, d_model=16, nhead=2, num_layers=1, dropout=0.0
    )
    src = torch.randn(2, 12, 3)

    out = model(src)

    assert out.shape == (2, 3)


def test_transformer_is_end_to_end_differentiable():
    model = TimeSeriesTransformer(
        input_dim=1, d_model=16, nhead=2, num_layers=1, dropout=0.0
    )
    src = torch.randn(4, 20, 1)

    out = model(src)
    out.sum().backward()

    assert model.encoder.weight.grad is not None
    assert not torch.all(model.encoder.weight.grad == 0)
