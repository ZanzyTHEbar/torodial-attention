"""
Tests for FFT approximation and adaptive depth learning.
"""

import pytest
import torch
from toroidal_attention.fft import (
    fft_circulant_attention,
    FFTToroidalAttention,
    hybrid_fft_attention,
    AdaptiveFFTAttention,
)
from toroidal_attention.adaptive import (
    AdaptiveDepthSelector,
    AdaptiveDepthToroidalAttention,
    DepthAnnealer,
)


# FFT Tests

def test_fft_attention_shape():
    """Test FFT attention produces correct output shape."""
    B, H, N, d_k = 2, 4, 256, 32
    
    Q = torch.randn(B, H, N, d_k)
    K = torch.randn(B, H, N, d_k)
    V = torch.randn(B, H, N, d_k)
    
    output = fft_circulant_attention(Q, K, V, use_rfft=True)
    
    assert output.shape == (B, H, N, d_k)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_fft_toroidal_attention_module():
    """Test FFTToroidalAttention module."""
    d_model = 256
    n_heads = 8
    seq_len = 512
    batch_size = 2
    
    fft_attn = FFTToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
        max_len=2048,
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, _ = fft_attn(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()


@pytest.mark.parametrize("seq_len", [256, 512, 1024])
def test_adaptive_fft_attention(seq_len):
    """Test adaptive FFT attention with different sequence lengths."""
    d_model = 128
    n_heads = 4
    batch_size = 2
    
    attn = AdaptiveFFTAttention(
        d_model=d_model,
        n_heads=n_heads,
        fft_threshold=512,
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = attn(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()


def test_hybrid_fft_attention():
    """Test hybrid FFT attention with depth."""
    B, H, N, D, d_k = 2, 4, 256, 2, 32
    ND = N * D
    
    Q = torch.randn(B, H, ND, d_k)
    K = torch.randn(B, H, ND, d_k)
    V = torch.randn(B, H, ND, d_k)
    
    output = hybrid_fft_attention(Q, K, V, depth=D, use_fft_threshold=128)
    
    assert output.shape == (B, H, ND, d_k)
    assert not torch.isnan(output).any()


# Adaptive Depth Tests

def test_adaptive_depth_selector():
    """Test adaptive depth selector."""
    d_model = 128
    candidate_depths = [1, 2, 4, 8]
    
    selector = AdaptiveDepthSelector(
        d_model=d_model,
        candidate_depths=candidate_depths,
        temperature=1.0,
    )
    
    # Test static selection
    depth_weights, depth_probs = selector(return_depth_probs=True)
    
    assert depth_weights.shape == (len(candidate_depths),)
    assert depth_probs.shape == (len(candidate_depths),)
    assert torch.allclose(depth_weights.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(depth_probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_adaptive_depth_input_routing():
    """Test input-dependent depth routing."""
    d_model = 128
    candidate_depths = [1, 2, 4]
    batch_size = 4
    seq_len = 64
    
    selector = AdaptiveDepthSelector(
        d_model=d_model,
        candidate_depths=candidate_depths,
    )
    selector.enable_input_routing()
    
    x = torch.randn(batch_size, seq_len, d_model)
    depth_weights, _ = selector(x)
    
    assert depth_weights.shape == (len(candidate_depths),)
    assert torch.allclose(depth_weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_adaptive_depth_toroidal_attention():
    """Test adaptive depth toroidal attention."""
    d_model = 256
    n_heads = 8
    seq_len = 128
    batch_size = 2
    candidate_depths = [1, 2, 4]
    
    attn = AdaptiveDepthToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
        candidate_depths=candidate_depths,
        max_len=256,
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, attn_weights, depth_info = attn(x, return_depth_info=True)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()
    assert depth_info is not None
    assert 'selected_depth' in depth_info
    assert depth_info['selected_depth'] in candidate_depths


def test_depth_annealer():
    """Test depth annealer."""
    annealer = DepthAnnealer(
        initial_temp=5.0,
        final_temp=0.5,
        anneal_steps=100,
        anneal_strategy='exponential',
    )
    
    # Test temperature decreases
    temp_start = annealer.step()
    for _ in range(50):
        annealer.step()
    temp_mid = annealer.step()
    for _ in range(48):
        annealer.step()
    temp_end = annealer.step()
    
    assert temp_start > temp_mid > temp_end
    assert temp_start == pytest.approx(5.0, abs=0.1)
    assert temp_end == pytest.approx(0.5, abs=0.1)


@pytest.mark.parametrize("use_input_routing", [True, False])
def test_adaptive_depth_training_mode(use_input_routing):
    """Test adaptive depth in training vs eval mode."""
    d_model = 128
    n_heads = 4
    seq_len = 64
    batch_size = 2
    
    attn = AdaptiveDepthToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
        candidate_depths=[1, 2, 4],
        use_input_routing=use_input_routing,
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Training mode: soft mixture
    attn.train()
    output_train, _, info_train = attn(x, return_depth_info=True)
    
    # Eval mode: hard selection
    attn.eval()
    output_eval, _, info_eval = attn(x, return_depth_info=True)
    
    assert output_train.shape == output_eval.shape
    assert not torch.isnan(output_train).any()
    assert not torch.isnan(output_eval).any()
    
    # In eval mode, should select single depth (one weight = 1.0)
    eval_weights = torch.tensor(info_eval['depth_weights'])
    assert (eval_weights == 1.0).sum() == 1


def test_get_selected_depth():
    """Test get_selected_depth method."""
    d_model = 128
    n_heads = 4
    candidate_depths = [1, 2, 4, 8]
    
    attn = AdaptiveDepthToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
        candidate_depths=candidate_depths,
    )
    
    selected = attn.get_selected_depth()
    assert selected in candidate_depths


def test_set_temperature():
    """Test temperature adjustment."""
    d_model = 128
    n_heads = 4
    
    attn = AdaptiveDepthToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
        candidate_depths=[1, 2, 4],
        temperature=1.0,
    )
    
    # Change temperature
    attn.set_temperature(0.5)
    assert attn.depth_selector.temperature == 0.5
    
    attn.set_temperature(2.0)
    assert attn.depth_selector.temperature == 2.0


@pytest.mark.gpu
def test_adaptive_depth_gpu(gpu_device):
    """Test adaptive depth on GPU."""
    d_model = 256
    n_heads = 8
    seq_len = 256
    batch_size = 4
    
    attn = AdaptiveDepthToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
        candidate_depths=[1, 2, 4],
    ).to(gpu_device)
    
    x = torch.randn(batch_size, seq_len, d_model, device=gpu_device)
    output, _, depth_info = attn(x, return_depth_info=True)
    
    assert output.device.type == gpu_device.type
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()


@pytest.mark.gpu
def test_fft_attention_gpu(gpu_device):
    """Test FFT attention on GPU."""
    d_model = 256
    n_heads = 8
    seq_len = 1024
    batch_size = 2
    
    fft_attn = FFTToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
    ).to(gpu_device)
    
    x = torch.randn(batch_size, seq_len, d_model, device=gpu_device)
    output, _ = fft_attn(x)
    
    assert output.device.type == gpu_device.type
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any()

