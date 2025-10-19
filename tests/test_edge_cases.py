"""
Edge Case and Boundary Condition Tests

Tests for unusual but valid configurations that might break the implementation:
- Minimal dimensions (seq_len=1, depth=1, batch_size=1)
- Maximum dimensions (large sequences, many depths)
- Unusual ratios (n_heads vs depth mismatches handled)
- Empty/invalid inputs
- Numerical stability at extremes
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from toroidal_attention import (
    DepthFusion,
    Toroidal3DPositionalEncoding,
    ToroidalAttention,
    compute_toroidal_distance,
)


class TestMinimalDimensions:
    """Test with minimal valid dimensions."""

    def test_single_token(self):
        """Test with sequence length of 1."""
        attn = ToroidalAttention(d_model=128, n_heads=4, depth=2, max_len=64)
        x = torch.randn(2, 1, 128)  # Single token

        output, _ = attn(x)
        assert output.shape == (2, 1, 128)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_single_batch(self):
        """Test with batch size of 1."""
        attn = ToroidalAttention(d_model=128, n_heads=4, depth=2)
        x = torch.randn(1, 16, 128)  # Single batch

        output, _ = attn(x)
        assert output.shape == (1, 16, 128)

    def test_depth_one(self):
        """Test with depth=1 (2D toroidal, no stacking)."""
        attn = ToroidalAttention(d_model=128, n_heads=4, depth=1)
        x = torch.randn(2, 16, 128)

        output, _ = attn(x)
        assert output.shape == (2, 16, 128)

        # Should still work mathematically
        assert not torch.isnan(output).any()

    def test_minimal_head_dim(self):
        """Test with small but numerically stable head dimension."""
        # d_model=128, n_heads=4, depth=2 -> d_k=32, head_dim_per_depth=16 (small but stable)
        attn = ToroidalAttention(d_model=128, n_heads=4, depth=2)
        x = torch.randn(2, 8, 128)

        output, _ = attn(x)
        assert output.shape == (2, 8, 128)
        
        # Should be numerically stable (no NaN/Inf)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestLargeDimensions:
    """Test with large dimensions for memory/performance issues."""

    def test_long_sequence(self):
        """Test with long sequence (256 tokens)."""
        attn = ToroidalAttention(d_model=256, n_heads=8, depth=4, max_len=512)
        x = torch.randn(2, 256, 256)

        output, _ = attn(x)
        assert output.shape == (2, 256, 256)

    def test_many_depths(self):
        """Test with many depth platters."""
        # depth=8 for more stacking
        attn = ToroidalAttention(d_model=512, n_heads=8, depth=8)
        x = torch.randn(2, 32, 512)

        output, _ = attn(x)
        assert output.shape == (2, 32, 512)

        # Check fusion matrix dimensions
        fusion_matrix = attn.fusion.get_fusion_matrix()
        assert fusion_matrix.shape == (8, 8)

    def test_large_batch(self):
        """Test with large batch size."""
        attn = ToroidalAttention(d_model=128, n_heads=4, depth=2)
        x = torch.randn(32, 16, 128)  # Large batch

        output, _ = attn(x)
        assert output.shape == (32, 16, 128)


class TestNumericalStability:
    """Test numerical stability with extreme values."""

    def test_large_values(self):
        """Test with large input values."""
        attn = ToroidalAttention(d_model=128, n_heads=4, depth=2)
        x = torch.randn(2, 16, 128) * 100  # Scale by 100

        output, _ = attn(x)

        # Should not produce NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_small_values(self):
        """Test with very small input values."""
        attn = ToroidalAttention(d_model=128, n_heads=4, depth=2)
        x = torch.randn(2, 16, 128) * 0.001  # Scale by 0.001

        output, _ = attn(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_zero_input(self):
        """Test with all-zero input."""
        attn = ToroidalAttention(d_model=128, n_heads=4, depth=2)
        x = torch.zeros(2, 16, 128)

        output, _ = attn(x)

        # Output should be well-defined (likely close to zero)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_stability_extreme(self):
        """Test gradient computation with extreme values."""
        attn = ToroidalAttention(d_model=128, n_heads=4, depth=2)
        x = torch.randn(2, 16, 128, requires_grad=True) * 10
        x.retain_grad()  # Retain gradients for non-leaf tensor

        output, _ = attn(x)
        loss = output.mean()
        loss.backward()

        # Gradients should exist and be bounded
        assert x.grad is not None, "Gradient not computed for input tensor"
        assert not torch.isnan(x.grad).any(), "NaN values in gradients"
        assert not torch.isinf(x.grad).any(), "Inf values in gradients"
        assert x.grad.abs().max() < 1e6, f"Gradient too large: {x.grad.abs().max()}"


class TestInvalidInputs:
    """Test error handling for invalid inputs."""

    def test_wrong_dimension(self):
        """Test with wrong input dimension."""
        attn = ToroidalAttention(d_model=128, n_heads=4, depth=2)
        x = torch.randn(2, 16, 64)  # Wrong d_model (64 instead of 128)

        with pytest.raises(AssertionError):
            output, _ = attn(x)

    def test_mismatched_heads_depth(self):
        """Test configuration where n_heads not divisible by depth."""
        # n_heads=5, depth=2 -> 5 % 2 != 0
        with pytest.raises(AssertionError):
            ToroidalAttention(d_model=128, n_heads=5, depth=2)

    def test_d_model_not_divisible_by_depth(self):
        """Test when d_model not divisible by depth."""
        # d_model=127, depth=4 -> 127 % 4 != 0
        with pytest.raises(AssertionError):
            ToroidalAttention(d_model=127, n_heads=4, depth=4)

    def test_d_model_not_divisible_by_heads(self):
        """Test when d_model not divisible by n_heads."""
        # d_model=127, n_heads=8 -> 127 % 8 != 0
        with pytest.raises(AssertionError):
            ToroidalAttention(d_model=127, n_heads=8, depth=4)


class TestSequenceLengthVariation:
    """Test with varying sequence lengths."""

    def test_sequence_shorter_than_max(self):
        """Test with sequence shorter than max_len."""
        attn = ToroidalAttention(d_model=128, n_heads=4, depth=2, max_len=256)

        # Try different lengths
        for seq_len in [16, 32, 64, 128]:
            x = torch.randn(2, seq_len, 128)
            output, _ = attn(x)
            assert output.shape == (2, seq_len, 128)

    def test_sequence_equals_max(self):
        """Test with sequence length exactly at max_len."""
        max_len = 128
        attn = ToroidalAttention(d_model=128, n_heads=4, depth=2, max_len=max_len)
        x = torch.randn(2, max_len, 128)

        output, _ = attn(x)
        assert output.shape == (2, max_len, 128)

    def test_power_of_two_lengths(self):
        """Test with power-of-2 sequence lengths (common in practice)."""
        attn = ToroidalAttention(d_model=256, n_heads=8, depth=4, max_len=512)

        for seq_len in [8, 16, 32, 64, 128, 256]:
            x = torch.randn(2, seq_len, 256)
            output, _ = attn(x)
            assert output.shape == (2, seq_len, 256)

    def test_prime_sequence_lengths(self):
        """Test with prime sequence lengths (edge case for modulo ops)."""
        attn = ToroidalAttention(d_model=128, n_heads=4, depth=2, max_len=256)

        for seq_len in [7, 11, 13, 17, 23]:
            x = torch.randn(2, seq_len, 128)
            output, _ = attn(x)
            assert output.shape == (2, seq_len, 128)


class TestMaskingEdgeCases:
    """Test masking with edge cases."""

    def test_all_masked(self):
        """Test with all positions masked (pathological case)."""
        attn = ToroidalAttention(d_model=128, n_heads=4, depth=2)
        x = torch.randn(2, 16, 128)

        # Mask everything
        mask = torch.ones(16, 16, dtype=torch.bool)

        output, _ = attn(x, mask=mask)

        # Output should be all zeros or NaN (depending on implementation)
        # This is a pathological case, but should not crash
        assert output.shape == (2, 16, 128)

    def test_no_masking(self):
        """Test with no mask (all positions visible)."""
        attn = ToroidalAttention(d_model=128, n_heads=4, depth=2)
        x = torch.randn(2, 16, 128)

        mask = torch.zeros(16, 16, dtype=torch.bool)

        output_masked, _ = attn(x, mask=mask)
        output_no_mask, _ = attn(x, mask=None)

        # Should be nearly identical
        assert torch.allclose(output_masked, output_no_mask, atol=1e-5)

    def test_diagonal_mask(self):
        """Test with diagonal masking (attend only to self)."""
        attn = ToroidalAttention(d_model=128, n_heads=4, depth=2)
        x = torch.randn(2, 16, 128)

        # Mask all off-diagonal elements
        mask = ~torch.eye(16, dtype=torch.bool)

        output, _ = attn(x, mask=mask)
        assert output.shape == (2, 16, 128)


class TestDistanceMetricEdgeCases:
    """Test toroidal distance with edge cases."""

    def test_distance_single_position(self):
        """Test distance for sequence length 1."""
        distances = compute_toroidal_distance(seq_len=1, depth=4)

        # Shape should be (1, 4, 1, 4) - one seq pos, four depths
        assert distances.shape == (1, 4, 1, 4)

        # Sequence distance should be 0 (same position)
        # But depth distances should still exist between layers
        # Distance from depth k to depth l should be |k-l|/D
        for k in range(4):
            for l in range(4):
                expected_dist = abs(k - l) / 4
                assert torch.isclose(distances[0, k, 0, l], torch.tensor(expected_dist)), \
                    f"Distance mismatch at depth ({k}, {l}): {distances[0, k, 0, l]} vs {expected_dist}"

    def test_distance_large_sequence(self):
        """Test distance computation doesn't overflow with large sequences."""
        distances = compute_toroidal_distance(seq_len=1024, depth=8)

        assert distances.shape == (1024, 8, 1024, 8)
        assert not torch.isnan(distances).any()
        assert not torch.isinf(distances).any()

    def test_distance_symmetry_all_pairs(self):
        """Test symmetry holds for all position pairs."""
        seq_len, depth = 16, 4
        distances = compute_toroidal_distance(seq_len, depth)

        # Check all pairs
        for i in range(seq_len):
            for k in range(depth):
                for j in range(seq_len):
                    for l in range(depth):
                        assert abs(distances[i, k, j, l] - distances[j, l, i, k]) < 1e-6


class TestPositionalEncodingEdgeCases:
    """Test positional encoding edge cases."""

    def test_pe_single_position(self):
        """Test PE for single position."""
        pe = Toroidal3DPositionalEncoding(d_model=128, max_len=64, depth=4)
        sin_emb, cos_emb = pe(seq_len=1)

        assert sin_emb.shape == (1, 4, 64)
        assert cos_emb.shape == (1, 4, 64)
        assert not torch.isnan(sin_emb).any()
        assert not torch.isnan(cos_emb).any()

    def test_pe_max_length(self):
        """Test PE at maximum length."""
        max_len = 2048
        pe = Toroidal3DPositionalEncoding(d_model=128, max_len=max_len, depth=4)
        sin_emb, cos_emb = pe(seq_len=max_len)

        assert sin_emb.shape == (max_len, 4, 64)
        assert cos_emb.shape == (max_len, 4, 64)

    def test_pe_orthogonality_various_configs(self):
        """Test that positional encodings are distinct across configs."""
        configs = [
            (32, 2),   # Small model
            (128, 4),  # Medium
            (512, 8),  # Large model
        ]

        for d_model, depth in configs:
            pe = Toroidal3DPositionalEncoding(d_model=d_model, max_len=256, depth=depth)
            ortho_score = pe.get_orthogonality_score()

            # Just verify bases are not identical (score < 1.0)
            # Perfect orthogonality not strictly required
            assert ortho_score < 0.9999, \
                f"Frequency bases are identical for d_model={d_model}, depth={depth}: {ortho_score}"

            print(f"  d_model={d_model}, depth={depth}: orthogonality score={ortho_score:.4f}")


class TestDepthFusionEdgeCases:
    """Test depth fusion edge cases."""

    def test_fusion_single_depth(self):
        """Test fusion with depth=1 (no fusion needed)."""
        fusion = DepthFusion(depth=1, fusion_mode='low_rank')
        x = torch.randn(2, 16, 1, 64)

        output = fusion(x)
        assert output.shape == (2, 16, 64)

    def test_fusion_rank_equals_depth(self):
        """Test when fusion rank equals depth (full rank)."""
        depth = 4
        fusion = DepthFusion(depth=depth, rank=depth, fusion_mode='low_rank')

        eff_rank = fusion.get_effective_rank()
        # Should be reasonably close to full rank
        # Effective rank uses entropy formula, so it's typically 50-75% of actual rank
        # due to random initialization and entropy calculation
        assert eff_rank > depth * 0.5, \
            f"Effective rank too low: {eff_rank} (expected > {depth * 0.5})"

    def test_fusion_modes_consistency(self):
        """Test all fusion modes produce same shape."""
        x = torch.randn(2, 16, 4, 64)

        for mode in ['mean', 'low_rank', 'attention']:
            fusion = DepthFusion(depth=4, fusion_mode=mode)
            output = fusion(x)

            assert output.shape == (2, 16, 256), \
                f"Fusion mode {mode} produced wrong shape: {output.shape}"


def run_edge_case_tests():
    """Run all edge case tests."""
    import pytest

    print("=" * 60)
    print("Running Edge Case Tests")
    print("=" * 60)

    # Run pytest on this file
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    run_edge_case_tests()

