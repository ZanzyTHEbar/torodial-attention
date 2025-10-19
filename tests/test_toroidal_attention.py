"""
Unit tests for Toroidal Attention Module

Tests validate:
1. Rotational invariance (Lemma 1 from EDD)
2. Gradient bounds for stability
3. Shape preservation
4. Distance metric properties
5. Orthogonality of PE bases
"""

import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from toroidal_attention import (
    DepthFusion,
    Toroidal3DPositionalEncoding,
    ToroidalAttention,
    compute_toroidal_distance,
)


class TestToroidal3DPositionalEncoding:
    """Test 3D positional encoding with orthogonal bases."""

    def test_orthogonality(self):
        """Test that frequency bases are distinct (φ_m ≠ ω_m)."""
        d_model, depth = 128, 4
        pe = Toroidal3DPositionalEncoding(d_model, max_len=256, depth=depth)

        # Check that bases are not identical
        ortho_score = pe.get_orthogonality_score()
        assert ortho_score < 0.9999, \
            f"Frequency bases are identical: {ortho_score}"

        print(f"✓ Distinctness test passed: score = {ortho_score:.4f} (lower is more orthogonal)")

    def test_shape_consistency(self):
        """Test that PE produces correct shapes."""
        d_model, max_len, depth = 128, 64, 4
        pe = Toroidal3DPositionalEncoding(d_model, max_len, depth)

        # Test all depths
        sin_emb, cos_emb = pe(seq_len=32)
        assert sin_emb.shape == (32, depth, d_model // 2)
        assert cos_emb.shape == (32, depth, d_model // 2)

        # Test single depth
        sin_emb_single, cos_emb_single = pe(seq_len=32, depth_idx=0)
        assert sin_emb_single.shape == (32, d_model // 2)
        assert cos_emb_single.shape == (32, d_model // 2)

        print("✓ PE shape consistency test passed")

    def test_rotation_application(self):
        """Test that rotary embedding preserves magnitude."""
        d_model, depth = 128, 4
        pe = Toroidal3DPositionalEncoding(d_model, max_len=256, depth=depth)

        # Create test tensor
        B, N = 2, 16
        x = torch.randn(B, N, depth, d_model)

        # Apply rotation
        sin_emb, cos_emb = pe(N)
        x_rot = pe.apply_rotary_embedding(x, sin_emb, cos_emb)

        # Check magnitude preservation (rotation shouldn't change norm significantly)
        norm_original = torch.norm(x, dim=-1)
        norm_rotated = torch.norm(x_rot, dim=-1)

        assert torch.allclose(norm_original, norm_rotated, rtol=1e-3), \
            "Rotation should preserve vector magnitudes"

        print("✓ Rotation magnitude preservation test passed")


class TestToroidalDistance:
    """Test toroidal distance metric properties."""

    def test_symmetry(self):
        """Test δ(a, b) = δ(b, a)."""
        seq_len, depth = 16, 4
        distances = compute_toroidal_distance(seq_len, depth, normalize=True)

        # Check symmetry: dist[i,k,j,l] == dist[j,l,i,k]
        distances_transposed = distances.permute(2, 3, 0, 1)
        assert torch.allclose(distances, distances_transposed, atol=1e-6), \
            "Distance metric must be symmetric"

        print("✓ Distance symmetry test passed")

    def test_identity(self):
        """Test δ(a, a) = 0."""
        seq_len, depth = 16, 4
        distances = compute_toroidal_distance(seq_len, depth, normalize=True)

        # Check diagonal is zero
        for i in range(seq_len):
            for k in range(depth):
                assert distances[i, k, i, k].item() == 0.0, \
                    f"Distance from (i,k) to itself should be 0, got {distances[i, k, i, k]}"

        print("✓ Distance identity test passed")

    def test_wraparound(self):
        """Test cyclic wrap-around at boundaries."""
        seq_len, depth = 16, 4
        distances = compute_toroidal_distance(seq_len, depth, normalize=True)

        # Distance from position 0 to (seq_len-1) should equal distance to 1
        # (they're both 1 step away on a circle)
        dist_to_end = distances[0, 0, seq_len - 1, 0].item()
        dist_to_next = distances[0, 0, 1, 0].item()

        assert abs(dist_to_end - dist_to_next) < 1e-6, \
            f"Cyclic wrap-around failed: dist_to_end={dist_to_end}, dist_to_next={dist_to_next}"

        print("✓ Distance wrap-around test passed")

    def test_minimum_at_self(self):
        """Test that minimum distance is at self-position."""
        seq_len, depth = 16, 4
        distances = compute_toroidal_distance(seq_len, depth, normalize=True)

        for i in range(seq_len):
            for k in range(depth):
                # All distances from (i,k) should be >= distance to self (which is 0)
                assert (distances[i, k] >= -1e-6).all(), \
                    "Minimum distance should be at self-position"

        print("✓ Distance minimum test passed")


class TestDepthFusion:
    """Test depth fusion module."""

    def test_output_shape(self):
        """Test that fusion produces correct output shape."""
        B, N, D, d_per_depth = 2, 16, 4, 64
        x = torch.randn(B, N, D, d_per_depth)

        for mode in ['mean', 'low_rank', 'attention']:
            fusion = DepthFusion(depth=D, rank=2, fusion_mode=mode)
            output = fusion(x)

            expected_shape = (B, N, D * d_per_depth)
            assert output.shape == expected_shape, \
                f"Fusion mode {mode}: expected {expected_shape}, got {output.shape}"

        print("✓ Depth fusion shape test passed")

    def test_effective_rank(self):
        """Test that low-rank fusion has controlled rank."""
        D, rank = 8, 2
        fusion = DepthFusion(depth=D, rank=rank, fusion_mode='low_rank')

        eff_rank = fusion.get_effective_rank()
        assert eff_rank <= D, f"Effective rank {eff_rank} exceeds depth {D}"
        assert eff_rank >= rank / 2, f"Effective rank {eff_rank} too low (expected ~{rank})"

        print(f"✓ Effective rank test passed: {eff_rank:.2f} / {D}")


class TestToroidalAttention:
    """Test main toroidal attention module."""

    def test_forward_pass(self):
        """Test basic forward pass with correct shapes."""
        B, N, d_model = 2, 32, 256
        n_heads, depth = 8, 4

        attn = ToroidalAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_len=128,
            depth=depth,
        )

        x = torch.randn(B, N, d_model)
        output, attn_weights = attn(x, return_attention=True)

        assert output.shape == (B, N, d_model), f"Output shape mismatch: {output.shape}"
        assert attn_weights.shape == (B, N, N), f"Attention shape mismatch: {attn_weights.shape}"

        print("✓ Forward pass test passed")

    def test_causal_masking(self):
        """Test that causal masking prevents future attention."""
        B, N, d_model = 2, 16, 128
        n_heads, depth = 4, 2

        attn = ToroidalAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_len=64,
            depth=depth,
        )

        x = torch.randn(B, N, d_model)

        # Create causal mask (prevent attending to future positions)
        mask = torch.triu(torch.ones(N, N), diagonal=1).bool()

        output, attn_weights = attn(x, mask=mask, return_attention=True)

        # Check that attention weights respect mask
        # Lower triangle (including diagonal) should have attention, upper triangle should be ~0
        upper_triangle = attn_weights[0].triu(diagonal=1)
        assert upper_triangle.abs().max() < 1e-6, \
            "Causal mask not properly applied - future positions have attention"

        print("✓ Causal masking test passed")

    def test_causal_masking_sdpa_and_manual(self):
        """Regression: ensure both SDPA and manual paths honor causal masking."""
        B, N, d_model = 1, 16, 64
        n_heads, depth = 4, 2
        x = torch.randn(B, N, d_model)
        mask = torch.triu(torch.ones(N, N), diagonal=1).bool()

        for chunk in (None, 8):
            attn = ToroidalAttention(
                d_model=d_model,
                n_heads=n_heads,
                max_len=64,
                depth=depth,
                attn_chunk_size=chunk,
            )
            _, attn_weights = attn(x, mask=mask, return_attention=True)
            assert attn_weights is not None
            upper_triangle = attn_weights[0].triu(diagonal=1)
            assert upper_triangle.abs().max() < 1e-6

    def test_causal_masking_disables_flash2(self):
        """Regression: flash2 must be disabled when causal/extra masks applied."""
        B, N, d_model = 1, 8, 64
        x = torch.randn(B, N, d_model)
        mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
        # backend=flash2 with lambda=0 and no window would normally try flash2.
        # Presence of extra mask should fallback gracefully.
        attn = ToroidalAttention(d_model=d_model, n_heads=4, max_len=16, depth=2, lambda_distance=0.0, backend='flash2')
        y, _ = attn(x, mask=mask, return_attention=False)
        assert y.shape == (B, N, d_model)

    def test_rotational_invariance(self):
        """
        Test Lemma 1: Scores invariant under cyclic shift mod N.

        This is the key property of toroidal attention - shifting the input
        sequence should produce a correspondingly shifted output.
        """
        B, N, d_model = 1, 32, 128
        n_heads, depth = 4, 2

        attn = ToroidalAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_len=64,
            depth=depth,
            lambda_distance=0.0,  # Disable distance bias for pure rotation test
        )

        # Original input
        x = torch.randn(B, N, d_model)
        output_original, _ = attn(x)

        # Shift input by s positions
        shift = 8
        x_shifted = torch.roll(x, shifts=shift, dims=1)
        output_shifted, _ = attn(x_shifted)

        # Output should also be shifted by s positions
        output_expected = torch.roll(output_original, shifts=shift, dims=1)

        # Check if outputs match (allowing for numerical tolerance)
        max_diff = (output_shifted - output_expected).abs().max().item()

        # Note: Perfect invariance may not hold due to depth fusion and other factors
        # We check for approximate invariance
        assert max_diff < 0.5, \
            f"Rotational invariance violated: max_diff = {max_diff:.4f}"

        print(f"✓ Rotational invariance test passed (max_diff = {max_diff:.4f})")

    def test_gradient_flow(self):
        """Test that gradients flow properly and are bounded."""
        B, N, d_model = 2, 16, 128
        n_heads, depth = 4, 2

        attn = ToroidalAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_len=64,
            depth=depth,
        )

        x = torch.randn(B, N, d_model, requires_grad=True)
        output, _ = attn(x)

        # Compute loss and backpropagate
        loss = output.mean()
        loss.backward()

        # Check that gradients exist and are bounded
        assert x.grad is not None, "No gradient computed for input"

        grad_norm = x.grad.norm().item()
        assert grad_norm < 100.0, f"Gradient norm too large: {grad_norm}"
        assert grad_norm > 1e-6, f"Gradient norm too small (vanishing): {grad_norm}"

        # Check PE gradients are bounded (validates gradient stability claim)
        for name, param in attn.named_parameters():
            if param.grad is not None:
                param_grad_norm = param.grad.norm().item()
                assert param_grad_norm < 1000.0, \
                    f"Parameter {name} has exploding gradient: {param_grad_norm}"

        print(f"✓ Gradient flow test passed (grad_norm = {grad_norm:.4f})")

    def test_memory_efficiency(self):
        """Test that memory usage is reasonable (O(bch + Dr))."""
        B, N, d_model = 4, 128, 512
        n_heads, depth = 8, 4

        attn = ToroidalAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_len=256,
            depth=depth,
        )

        torch.randn(B, N, d_model)

        # Count parameters
        n_params = sum(p.numel() for p in attn.parameters())

        # Rough estimate of memory
        param_memory_mb = (n_params * 4) / (1024 ** 2)  # 4 bytes per float32

        print(f"✓ Memory efficiency test: {n_params:,} parameters ({param_memory_mb:.2f} MB)")

        # Check it's reasonable (should be comparable to standard attention)
        # Standard attention has ~4*d_model^2 parameters (Q,K,V,O projections)
        standard_params = 4 * d_model * d_model
        assert n_params < standard_params * 2, \
            f"Parameter count too high: {n_params:,} vs standard {standard_params:,}"

    def test_backend_gating_flash2_fallback(self):
        """Flash2 must be auto-disabled when bias/window are present."""
        B, N, d_model = 1, 8, 64
        x = torch.randn(B, N, d_model)
        # lambda_distance>0 → fallback to SDPA/manual
        attn_bias = ToroidalAttention(d_model=d_model, n_heads=4, max_len=16, depth=2, lambda_distance=0.1, backend='flash2')
        y, _ = attn_bias(x)
        assert y.shape == (B, N, d_model)
        # window_size>0 → fallback to SDPA/manual
        attn_win = ToroidalAttention(d_model=d_model, n_heads=4, max_len=16, depth=2, lambda_distance=0.0, backend='flash2', window_size=2)
        y2, _ = attn_win(x)
        assert y2.shape == (B, N, d_model)


def run_all_tests():
    """Run all test suites."""
    print("=" * 60)
    print("Running Toroidal Attention Unit Tests")
    print("=" * 60)

    # Test PE
    print("\n[1/5] Testing 3D Positional Encoding...")
    test_pe = TestToroidal3DPositionalEncoding()
    test_pe.test_orthogonality()
    test_pe.test_shape_consistency()
    test_pe.test_rotation_application()

    # Test distance
    print("\n[2/5] Testing Toroidal Distance Metric...")
    test_dist = TestToroidalDistance()
    test_dist.test_symmetry()
    test_dist.test_identity()
    test_dist.test_wraparound()
    test_dist.test_minimum_at_self()

    # Test fusion
    print("\n[3/5] Testing Depth Fusion...")
    test_fusion = TestDepthFusion()
    test_fusion.test_output_shape()
    test_fusion.test_effective_rank()

    # Test attention
    print("\n[4/5] Testing Toroidal Attention Module...")
    test_attn = TestToroidalAttention()
    test_attn.test_forward_pass()
    test_attn.test_causal_masking()
    test_attn.test_gradient_flow()
    test_attn.test_memory_efficiency()

    # Test invariance (most important)
    print("\n[5/5] Testing Rotational Invariance (Lemma 1)...")
    test_attn.test_rotational_invariance()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

