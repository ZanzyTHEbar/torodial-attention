"""
Parametrized tests for comprehensive coverage of toroidal attention.

Tests sweep across:
- Different depths (1, 2, 4, 8)
- Different sequence lengths (64, 128, 256, 512)
- Different fusion modes (low_rank, attention, mean)
- Orthogonal vs standard PE
"""

import pytest
import torch
from toroidal_attention import ToroidalAttention


@pytest.mark.parametrize("depth,seq_len", [
    (1, 64),
    (2, 128),
    (4, 256),
    (8, 512),
])
def test_forward_backward_parametrized(depth, seq_len, device):
    """Test forward/backward pass for various configs."""
    d_model = 256
    n_heads = 8
    batch_size = 2
    
    attn = ToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        max_len=seq_len * 2,
        use_orthogonal_pe=True,
    ).to(device)
    
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    
    # Forward
    output, _ = attn(x)
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    
    # Backward
    loss = output.mean()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


@pytest.mark.parametrize("fusion_mode", ['low_rank', 'attention', 'mean'])
def test_fusion_modes_parametrized(fusion_mode, device):
    """Test all fusion modes produce valid output."""
    d_model = 128
    n_heads = 4
    depth = 4
    seq_len = 64
    batch_size = 2
    
    attn = ToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        fusion_mode=fusion_mode,
        fusion_rank=1 if fusion_mode == 'low_rank' else None,
        use_orthogonal_pe=True,
    ).to(device)
    
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    output, _ = attn(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


@pytest.mark.parametrize("use_orth", [True, False])
def test_pe_variants_parametrized(use_orth, device):
    """Test orthogonal vs standard PE."""
    d_model = 128
    n_heads = 4
    depth = 2
    seq_len = 128
    batch_size = 2
    
    # Standard PE warns for weak orthogonality but still works
    import warnings
    with warnings.catch_warnings():
        if not use_orth:
            warnings.simplefilter("ignore")  # Ignore weak orthogonality warning
        
        attn = ToroidalAttention(
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            use_orthogonal_pe=use_orth,
        ).to(device)
    
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    output, _ = attn(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    
    # Check orthogonality score
    score = attn.pos_encoding.get_orthogonality_score()
    if use_orth:
        assert score < 0.1, f"Orthogonal PE should have score < 0.1, got {score}"
    else:
        # Standard PE may have weaker orthogonality
        assert score < 1.0, f"Standard PE score should be < 1.0, got {score}"


@pytest.mark.parametrize("lambda_distance", [0.0, 0.1, 0.5])
def test_distance_bias_parametrized(lambda_distance, device):
    """Test different distance bias strengths."""
    d_model = 128
    n_heads = 4
    depth = 2
    seq_len = 64
    batch_size = 2
    
    attn = ToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        lambda_distance=lambda_distance,
        use_orthogonal_pe=True,
    ).to(device)
    
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    output, _ = attn(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()


@pytest.mark.parametrize("window_size", [None, 32, 64])
def test_window_attention_parametrized(window_size, device):
    """Test sliding window attention."""
    d_model = 128
    n_heads = 4
    depth = 2
    seq_len = 128
    batch_size = 2
    
    attn = ToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        window_size=window_size,
        use_orthogonal_pe=True,
    ).to(device)
    
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    output, _ = attn(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()


@pytest.mark.parametrize("backend", ['sdpa', 'flash2'])
def test_backends_parametrized(backend, device):
    """Test different backends (flash2 auto-disables if unavailable)."""
    if backend == 'flash2' and device.type != 'cuda':
        pytest.skip("Flash2 requires CUDA")
    
    d_model = 128
    n_heads = 4
    depth = 2
    seq_len = 64
    batch_size = 2
    
    # Flash2 requires lambda=0, no window
    lambda_val = 0.0 if backend == 'flash2' else 0.1
    
    attn = ToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        backend=backend,
        lambda_distance=lambda_val,
        window_size=None,
        allow_flash2=True,
        use_orthogonal_pe=True,
    ).to(device)
    
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    output, _ = attn(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()


@pytest.mark.parametrize("return_attention", [True, False])
def test_return_attention_parametrized(return_attention, device):
    """Test with/without attention weight return."""
    d_model = 128
    n_heads = 4
    depth = 2
    seq_len = 64
    batch_size = 2
    
    attn = ToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        use_orthogonal_pe=True,
    ).to(device)
    
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    output, attn_weights = attn(x, return_attention=return_attention)
    
    assert output.shape == (batch_size, seq_len, d_model)
    
    if return_attention:
        assert attn_weights is not None
        assert attn_weights.shape == (batch_size, seq_len, seq_len)
    else:
        assert attn_weights is None


@pytest.mark.gpu
@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_gpu_memory_scaling_parametrized(depth, gpu_device):
    """Test memory scaling on GPU."""
    d_model = 256
    n_heads = 8
    seq_len = 256
    batch_size = 4
    
    torch.cuda.reset_peak_memory_stats()
    
    attn = ToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        use_orthogonal_pe=True,
    ).to(gpu_device)
    
    x = torch.randn(batch_size, seq_len, d_model, device=gpu_device)
    
    # Forward pass
    output, _ = attn(x)
    
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
    
    # Memory should scale roughly with depth (not exactly linear due to overhead)
    print(f"Depth {depth}: {peak_mem_mb:.1f} MB peak")
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()


@pytest.mark.parametrize("mask_type", ['causal', 'padding', 'custom'])
def test_masking_parametrized(mask_type, device):
    """Test different masking types."""
    d_model = 128
    n_heads = 4
    depth = 2
    seq_len = 64
    batch_size = 2
    
    attn = ToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        use_orthogonal_pe=True,
    ).to(device)
    
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    if mask_type == 'causal':
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    elif mask_type == 'padding':
        mask = torch.ones(batch_size, seq_len, device=device).bool()
        mask[:, seq_len//2:] = False  # Mask second half
    elif mask_type == 'custom':
        mask = torch.rand(batch_size, seq_len, seq_len, device=device) > 0.5
    
    output, _ = attn(x, mask=mask)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()

