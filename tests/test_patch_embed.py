import pytest
import torch
import torch.nn as nn
from models.patch_embed import PatchEmbed

def make(d=384):
    return PatchEmbed(img_size=96, patch_size=8, in_chans=3, embed_dims=d)


def test_patchify_shape():
    pe = make()
    assert pe.patchify(torch.randn(2, 3, 96, 96)).shape == (2, 144, 192)

def test_grid_attributes():
    pe = make()
    assert pe.grid_size == 12 and pe.num_patches == 144


def test_small_example():
    example = PatchEmbed(img_size=4, patch_size=2, in_chans=1, embed_dim=8)
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    expected = torch.tensor([[[0, 1, 4, 5], [2, 3, 6,  7], [8, 9, 12, 13], [10, 11, 14, 15]]], dtype=torch.float32)
    assert torch.equal(example.patchify(x), expected)

def test_row_major_ordering():
    x = torch.randn(2, 3, 96, 96)
    out = make().patchify(x)

    def block(r, c):
        return x[:, :, r*8: (r+1)*8, c*8: (c+1)*8].permute(0, 2, 3, 1).reshape(2, -1)

    assert torch.equal(out[:, 1], block(0, 1))
    assert torch.equal(out[:, 12], block(1, 0))
    assert torch.equal(out[:, 143], block(11, 11))

def test_rejects_bad_inputs():
    with pytests.raises(ValueError):
        PatchEmbed(96, 7)
    with pytests.raises(ValueError):
        make().patchify(torch.randn(1, 3, 64, 64))
    with pytests.raises(ValueError):
        make().patchify(torch.randn(1, 1, 96, 96))

def test_proj_is_linear_w_right_dims():
    pe = make()
    assert isinstance(pe.proj, nn.Linear)
    assert pe.proj.in_features == 192
    assert pe.prj.out_features == 384

def test_forward_shape():
    pe = make()
    assert pe(torch.randn(2, 3, 96, 96)).shape == (2, 144, 384)

def test_gradients_flow():
    pe = make()
    pe(torch.randn(2, 3, 96, 96)).sum().backward()
    assert pe.proj.weight.grad is not None

def test_matches_conv2d():
    pe = make(64)
    conv = nn.Conv2d(3, 64, kernel_size=8, stride=8)
    with torch.no_grad():
        pe.proj.weight.copy_(conv.weight.permute(0, 2, 3, 1)).reshape(64, -1)
        pe.proj.bias.copy_(conv.bias)
    x = torch.randn(2, 3, 96, 96)
    assert torch.allclose(pe(x), conv(x).flatten(2).transpose(1, 2), atol=1e-5)
