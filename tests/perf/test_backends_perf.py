import os
import time
import torch
import pytest

from toroidal_attention import ToroidalAttention


@pytest.mark.skipif(os.environ.get('RUN_PERF_SMOKE', '0') != '1', reason='Perf smoke disabled by default')
@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA GPU for meaningful perf')
def test_flash2_vs_sdpa_smoke():
    B, N, d_model, H, D = 4, 256, 512, 8, 2

    x = torch.randn(B, N, d_model, device='cuda')

    # SDPA baseline (no bias for fair comparison)
    sdpa = ToroidalAttention(
        d_model=d_model,
        n_heads=H,
        max_len=N,
        depth=D,
        lambda_distance=0.0,
        backend='sdpa',
    ).cuda()

    flash = ToroidalAttention(
        d_model=d_model,
        n_heads=H,
        max_len=N,
        depth=D,
        lambda_distance=0.0,
        backend='flash2',
    ).cuda()

    def bench(m):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(10):
                m(x)[0]
        torch.cuda.synchronize()
        return time.time() - t0

    t_sdpa = bench(sdpa)
    try:
        t_flash = bench(flash)
    except Exception:
        pytest.skip('flash-attn not available or incompatible')

    # Expect flash to be at least modestly faster
    assert t_flash < t_sdpa * 0.9
