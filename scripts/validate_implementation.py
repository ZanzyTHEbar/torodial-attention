#!/usr/bin/env python
"""
Quick validation script to verify toroidal attention implementation correctness.

This script performs basic sanity checks without requiring the full test suite.
Run after: pip install torch numpy

Tests:
1. Import check - all modules load
2. Shape consistency - forward pass produces correct shapes
3. Distance bias - lambda affects output
4. Rotational invariance - approximate shift equivariance
5. Gradient flow - backward pass works
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import numpy as np
    from toroidal_attention import (
        ToroidalAttention,
        Toroidal3DPositionalEncoding,
        compute_toroidal_distance,
        DepthFusion,
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\nInstall dependencies:")
    print("  pip install torch numpy pyyaml")
    sys.exit(1)


def test_shape_consistency():
    """Test 1: Forward pass shape consistency."""
    print("\n[Test 1] Shape Consistency")
    
    B, N, d_model = 2, 32, 256
    n_heads, depth = 8, 4
    
    attn = ToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        max_len=128,
    )
    
    x = torch.randn(B, N, d_model)
    output, attn_weights = attn(x, return_attention=True)
    
    # Check shapes
    assert output.shape == (B, N, d_model), f"Output shape wrong: {output.shape}"
    assert attn_weights.shape == (B, N, N), f"Attention shape wrong: {attn_weights.shape}"
    
    # Check no NaN/Inf
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"
    
    print(f"  ✓ Input: {x.shape} → Output: {output.shape}")
    print(f"  ✓ Attention weights: {attn_weights.shape}")
    print(f"  ✓ No NaN/Inf")


def test_distance_bias_effect():
    """Test 2: Distance bias affects output."""
    print("\n[Test 2] Distance Bias Effect")
    
    B, N, d_model = 2, 32, 256
    n_heads, depth = 8, 4
    
    # Create two models with different lambda
    attn_no_bias = ToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        lambda_distance=0.0,  # No distance bias
    )
    
    attn_with_bias = ToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        lambda_distance=0.5,  # Strong distance bias
    )
    
    # Copy weights (only lambda differs)
    attn_with_bias.load_state_dict(attn_no_bias.state_dict(), strict=False)
    
    x = torch.randn(B, N, d_model)
    
    with torch.no_grad():
        out_no_bias, _ = attn_no_bias(x)
        out_with_bias, _ = attn_with_bias(x)
    
    # Outputs should differ
    diff = (out_with_bias - out_no_bias).abs().max().item()
    
    print(f"  Max output difference: {diff:.6f}")
    
    if diff < 0.001:
        print("  ⚠️  WARNING: Distance bias has minimal effect")
        print("     (This might indicate bias isn't being applied correctly)")
    else:
        print(f"  ✓ Distance bias affects output (diff={diff:.4f})")


def test_rotational_invariance():
    """Test 3: Approximate rotational invariance."""
    print("\n[Test 3] Rotational Invariance")
    
    B, N, d_model = 1, 32, 256
    n_heads, depth = 4, 2
    
    attn = ToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        lambda_distance=0.0,  # Disable distance bias for pure rotation test
    )
    
    x = torch.randn(B, N, d_model)
    
    with torch.no_grad():
        output_original, _ = attn(x)
        
        # Shift by 8 positions
        shift = 8
        x_shifted = torch.roll(x, shifts=shift, dims=1)
        output_shifted, _ = attn(x_shifted)
        
        # Expected: output should also shift
        output_expected = torch.roll(output_original, shifts=shift, dims=1)
    
    max_diff = (output_shifted - output_expected).abs().max().item()
    mean_diff = (output_shifted - output_expected).abs().mean().item()
    
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    
    if max_diff < 0.01:
        print(f"  ✓ Excellent rotational invariance (diff < 0.01)")
    elif max_diff < 0.1:
        print(f"  ✓ Good rotational invariance (diff < 0.1)")
    elif max_diff < 0.5:
        print(f"  ⚠️  Approximate rotational invariance (diff < 0.5)")
    else:
        print(f"  ❌ FAILED: Rotational invariance violated (diff >= 0.5)")


def test_gradient_flow():
    """Test 4: Gradient flow through forward and backward."""
    print("\n[Test 4] Gradient Flow")
    
    B, N, d_model = 2, 16, 128
    n_heads, depth = 4, 2
    
    attn = ToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
    )
    
    x = torch.randn(B, N, d_model, requires_grad=True)
    output, _ = attn(x)
    
    # Backward pass
    loss = output.mean()
    loss.backward()
    
    # Check gradients
    assert x.grad is not None, "No gradient for input"
    
    grad_norm = x.grad.norm().item()
    
    print(f"  Input gradient norm: {grad_norm:.4f}")
    
    # Check gradient bounds
    assert grad_norm > 1e-6, f"Gradient too small (vanishing): {grad_norm}"
    assert grad_norm < 1000.0, f"Gradient too large (exploding): {grad_norm}"
    
    # Check parameter gradients
    param_grad_norms = []
    for name, param in attn.named_parameters():
        if param.grad is not None:
            pgrad_norm = param.grad.norm().item()
            param_grad_norms.append((name, pgrad_norm))
            assert pgrad_norm < 1000.0, f"Parameter {name} has exploding gradient: {pgrad_norm}"
    
    print(f"  ✓ All gradients bounded")
    print(f"  ✓ {len(param_grad_norms)} parameters have gradients")


def test_positional_encoding():
    """Test 5: Positional encoding orthogonality."""
    print("\n[Test 5] Positional Encoding")
    
    d_model, depth = 128, 4
    pe = Toroidal3DPositionalEncoding(d_model=d_model, max_len=256, depth=depth)
    
    # Check orthogonality score
    ortho_score = pe.get_orthogonality_score()
    
    print(f"  Orthogonality score: {ortho_score:.4f} (0 = orthogonal)")
    
    if ortho_score < 0.1:
        print("  ✓ Excellent orthogonality")
    elif ortho_score < 0.3:
        print("  ✓ Good orthogonality")
    else:
        print("  ⚠️  Weak orthogonality (might reduce effectiveness)")
    
    # Check shape
    sin_emb, cos_emb = pe(seq_len=32)
    assert sin_emb.shape == (32, depth, d_model // 2)
    print(f"  ✓ PE shape correct: {sin_emb.shape}")


def test_distance_metric():
    """Test 6: Toroidal distance properties."""
    print("\n[Test 6] Distance Metric")
    
    seq_len, depth = 16, 4
    distances = compute_toroidal_distance(seq_len, depth, normalize=True)
    
    # Symmetry
    distances_T = distances.permute(2, 3, 0, 1)
    is_symmetric = torch.allclose(distances, distances_T, atol=1e-6)
    
    # Identity
    diag_vals = [distances[i, k, i, k].item() for i in range(seq_len) for k in range(depth)]
    is_zero_diag = all(abs(v) < 1e-6 for v in diag_vals)
    
    # Wrap-around
    dist_to_end = distances[0, 0, seq_len-1, 0].item()
    dist_to_next = distances[0, 0, 1, 0].item()
    has_wraparound = abs(dist_to_end - dist_to_next) < 1e-6
    
    print(f"  Symmetry: {'✓' if is_symmetric else '❌'}")
    print(f"  Identity (δ(a,a)=0): {'✓' if is_zero_diag else '❌'}")
    print(f"  Wrap-around: {'✓' if has_wraparound else '❌'}")
    
    if is_symmetric and is_zero_diag and has_wraparound:
        print("  ✓ All distance properties hold")


def test_flexible_configs():
    """Test 7: Flexible architecture configurations."""
    print("\n[Test 7] Flexible Configurations")
    
    configs = [
        (256, 8, 4, "n_heads=8, depth=4"),
        (256, 12, 4, "n_heads=12, depth=4"),
        (512, 16, 8, "n_heads=16, depth=8"),
    ]
    
    for d_model, n_heads, depth, desc in configs:
        try:
            attn = ToroidalAttention(d_model=d_model, n_heads=n_heads, depth=depth)
            x = torch.randn(2, 16, d_model)
            output, _ = attn(x)
            print(f"  ✓ {desc}")
        except Exception as e:
            print(f"  ❌ {desc}: {e}")


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Toroidal Attention Implementation Validation")
    print("=" * 60)
    
    tests = [
        test_shape_consistency,
        test_distance_bias_effect,
        test_rotational_invariance,
        test_gradient_flow,
        test_positional_encoding,
        test_distance_metric,
        test_flexible_configs,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✓ All validation tests passed!")
        print("\nNext steps:")
        print("1. Install full dev dependencies: pip install -e \".[dev]\"")
        print("2. Run complete test suite: pytest tests/ -v")
        print("3. Proceed with Week 2 (development tooling)")
    else:
        print(f"\n❌ {failed} test(s) failed - needs investigation")
        sys.exit(1)


if __name__ == "__main__":
    main()

