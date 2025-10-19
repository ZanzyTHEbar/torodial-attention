import torch
import pytest

from toroidal_attention import Toroidal3DPositionalEncoding
from toroidal_attention.positional_encoding_orthogonal import Toroidal3DPositionalEncodingOrthogonal


def test_pe_orthogonality_threshold():
    """Test that orthogonal PE achieves good orthogonality (<0.3)."""
    d_model, depth = 128, 4
    # Use orthogonal version for this test
    pe = Toroidal3DPositionalEncodingOrthogonal(d_model=d_model, max_len=256, depth=depth)
    score = pe.get_orthogonality_score()
    assert score < 0.3, f"Orthogonal PE score {score:.6f} should be < 0.3"


def test_standard_pe_orthogonality():
    """Document that standard PE has ~0.66 orthogonality (known limitation)."""
    d_model, depth = 128, 4
    pe = Toroidal3DPositionalEncoding(d_model=d_model, max_len=256, depth=depth)
    score = pe.get_orthogonality_score()
    # Standard PE has weak orthogonality - this is expected and documented
    assert 0.5 < score < 0.8, f"Standard PE score {score:.6f} expected in range [0.5, 0.8]"


def test_pe_shape_all_depths():
    d_model, depth = 128, 4
    pe = Toroidal3DPositionalEncoding(d_model=d_model, max_len=256, depth=depth)
    sin_emb, cos_emb = pe(seq_len=32)
    assert sin_emb.shape == (32, depth, d_model // 2)
    assert cos_emb.shape == (32, depth, d_model // 2)


def test_pe_shape_single_depth():
    d_model, depth = 128, 4
    pe = Toroidal3DPositionalEncoding(d_model=d_model, max_len=256, depth=depth)
    sin_emb, cos_emb = pe(seq_len=32, depth_idx=0)
    assert sin_emb.shape == (32, d_model // 2)
    assert cos_emb.shape == (32, d_model // 2)


def test_rotary_preserves_norm():
    """Test that RoPE preserves vector norms (isometry property)."""
    B, N, depth, n_heads = 2, 16, 4, 8
    # Use head_dim_per_depth as PE expects (not full d_model)
    head_dim_per_depth = 32  # Must be even
    
    pe = Toroidal3DPositionalEncoding(d_model=head_dim_per_depth, max_len=256, depth=depth)
    
    # Create input with correct shape matching core.py usage: (B, N, depth, heads, head_dim_per_depth)
    x = torch.randn(B, N, depth, n_heads, head_dim_per_depth)
    
    # Get embeddings
    sin_emb, cos_emb = pe(N)  # Shape: (N, depth, head_dim_per_depth//2)
    
    # Apply rotation with proper broadcasting (matches core.py pattern)
    sin_emb = sin_emb.unsqueeze(0).unsqueeze(3)  # (1, N, depth, 1, d//2) - broadcasts over heads
    cos_emb = cos_emb.unsqueeze(0).unsqueeze(3)  # (1, N, depth, 1, d//2)
    
    x_rot = pe.apply_rotary_embedding(x, sin_emb, cos_emb)
    
    # RoPE should preserve vector norms (rotation is isometry)
    original_norms = x.norm(dim=-1)
    rotated_norms = x_rot.norm(dim=-1)
    
    assert torch.allclose(original_norms, rotated_norms, rtol=1e-3, atol=1e-5), \
        "RoPE should preserve vector norms (isometry property)"


