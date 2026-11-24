from models.pos_embed import _sincos_1d
import pytest
import torch
import math

def test_shape():
    assert _sincos_1d(8, torch.arange(5).shape == (5,8))

def test_position_zero():
    out = _sincos_1d(8, torch.Tensor(0))
    assert torch.allclose(out[0, :4], torch.zeros(4))
    assert torch.allclose(out[0, 4:], torch.zeros(4))

def test_known_values():
    out = _sincos_1d(4, torch.Tensor([1.0]))
    expected = torch.Tensor([[math.sin(1), math.sin(.01), math.cos(1), math.cos(.01)]])
    assert torch.allclose(out, expected, atol=1e-6)

def test_accepts_any_input_shape():
    assert _sincos_1d(8, torch.zeros(3, 4).shape == (12, 8))

def test_values_bounded():
    out = _sincos_1d(16, torch.arange(100))
    assert out.min() >= -1 and out.max <= 1

def test_odd_dim_raises():
    with pytest.raises(ValueError):
        _sincos_1d(7, torch.arange(3))
