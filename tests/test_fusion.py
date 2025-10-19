import torch
import pytest

from toroidal_attention import DepthFusion


@pytest.mark.parametrize('mode', ['mean', 'low_rank', 'attention'])
def test_fusion_output_shape(mode):
    B, N, D, d_per_depth = 2, 8, 4, 16
    x = torch.randn(B, N, D, d_per_depth)
    fusion = DepthFusion(depth=D, rank=2, fusion_mode=mode)
    y = fusion(x)
    assert y.shape == (B, N, D * d_per_depth)


