import torch

from toroidal_attention import ToroidalAttention
from toroidal_attention.window import build_toroidal_window_mask


def test_core_shapes_and_nan():
    B, N, d_model = 2, 32, 128
    x = torch.randn(B, N, d_model)
    attn = ToroidalAttention(d_model=d_model, n_heads=8, max_len=64, depth=4, lambda_distance=0.1)
    y, attn_viz = attn(x, return_attention=True)
    assert y.shape == (B, N, d_model)
    assert attn_viz.shape == (B, N, N)
    assert not torch.isnan(y).any()


def test_lambda_has_effect():
    B, N, d_model = 2, 32, 128
    x = torch.randn(B, N, d_model)
    attn0 = ToroidalAttention(d_model=d_model, n_heads=8, max_len=64, depth=4, lambda_distance=0.0)
    attn1 = ToroidalAttention(d_model=d_model, n_heads=8, max_len=64, depth=4, lambda_distance=0.2)
    y0, _ = attn0(x, return_attention=False)
    y1, _ = attn1(x, return_attention=False)
    assert not torch.allclose(y0, y1)


def test_window_mask_wrap_and_shape():
    N, D, W = 8, 3, 2
    device = torch.device('cpu')
    mask = build_toroidal_window_mask(N, D, W, device)
    ND = N * D
    assert mask.shape == (ND, ND)
    # Check that row 0 (all depths) can attend to last indices due to wrap
    rows = torch.arange(0, D)
    # Allowed columns include indices from (N-W)..(N-1) across depths
    allowed_seq = [(N - i) % N for i in range(1, W + 1)]
    allowed_cols = []
    for s in allowed_seq:
        for l in range(D):
            allowed_cols.append(s * D + l)
    allowed_cols = torch.tensor(sorted(set(allowed_cols)), dtype=torch.long)
    # Values at allowed columns should be zero (no mask)
    assert (mask[rows.unsqueeze(1), allowed_cols.unsqueeze(0)] == 0.0).all()


