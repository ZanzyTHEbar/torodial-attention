import torch
import time

from toroidal_attention import ToroidalAttention


def test_perf_smoke_cpu_small():
    B, N, d_model = 1, 64, 128
    x = torch.randn(B, N, d_model)
    attn = ToroidalAttention(d_model=d_model, n_heads=8, max_len=128, depth=2, lambda_distance=0.1)
    t0 = time.time()
    y, _ = attn(x, return_attention=False)
    elapsed = time.time() - t0
    assert y.shape == (B, N, d_model)
    assert elapsed < 2.0  # sanity threshold


