import torch

from toroidal_attention.latent import LatentCfg, LatentKV
from toroidal_attention import ToroidalAttention


def test_latent_update_and_attend_shapes():
    B, H, d_k, m = 2, 4, 16, 8
    cfg = LatentCfg(latent_dim=m, update="gru")
    kv = LatentKV(d_k=d_k, cfg=cfg)
    device = torch.device('cpu')

    z = kv.init_state(B, H, device)
    k_t = torch.randn(B, H, d_k, device=device)
    v_t = torch.randn(B, H, d_k, device=device)
    q_t = torch.randn(B, H, d_k, device=device)

    z_new = kv.update(z, k_t, v_t)
    out = kv.attend(q_t, z_new)

    assert z_new.shape == (B, H, m)
    assert out.shape == (B, H, d_k)


def test_forward_streaming_end_to_end():
    B, d_model = 2, 64
    N, H, D = 4, 4, 2

    attn = ToroidalAttention(
        d_model=d_model,
        n_heads=H,
        max_len=16,
        depth=D,
        lambda_distance=0.0,
        backend='sdpa',
        latent_cfg=LatentCfg(latent_dim=8, update='gru'),
    )

    state = None
    outputs = []
    for _ in range(N):
        x_t = torch.randn(B, 1, d_model)
        y_t, state = attn.forward_streaming(x_t, state)
        outputs.append(y_t)

    y = torch.cat(outputs, dim=1)
    assert y.shape == (B, N, d_model)


