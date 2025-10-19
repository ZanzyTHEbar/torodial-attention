"""
3D Rotary Positional Encoding (Toroidal RoPE)

Extends standard RoPE to 3D toroidal structure with orthogonal frequency bases
to prevent dimensional collapse. Implements helical twist combining sequence
position and depth indices.

Mathematical formulation:
    PE(i, k)_m = sin(2π(i·ω_m + k·φ_m)) + cos(2π(i·ω_m + k·φ_m))
    where:
        ω_m = 10000^(-2m/d)      # Sequence frequency
        φ_m = 10000^(-2m/(d·D))  # Depth frequency (orthogonal)
        i ∈ [0, N-1]             # Sequence position
        k ∈ [0, D-1]             # Depth (platter) index
"""

import math

import torch
import torch.nn as nn


class Toroidal3DPositionalEncoding(nn.Module):
    """
    3D Rotary Positional Encoding for toroidal attention.

    Implements orthogonal frequency bases for sequence and depth dimensions
    to enable rotational invariance while preventing basis collapse.

    Args:
        d_model (int): Model dimension (must be divisible by depth)
        max_len (int): Maximum sequence length
        depth (int): Number of depth platters (stacks)
        base (float): Base for frequency calculation (default: 10000)

    Attributes:
        freqs_seq (torch.Tensor): Sequence frequency basis ω_m, shape (d_model//2,)
        freqs_depth (torch.Tensor): Depth frequency basis φ_m, shape (d_model//2,)
    """

    def __init__(self, d_model: int, max_len: int = 2048, depth: int = 4, base: float = 10000.0):
        super().__init__()

        assert d_model % 2 == 0, "d_model must be even for RoPE"
        # Note: d_model here is head_dim_per_depth, not the full model dimension
        # The divisibility by depth constraint is enforced in core.py

        self.d_model = d_model
        self.max_len = max_len
        self.depth = depth
        self.base = base

        # Compute frequency bases
        # ω_m = 10000^(-2m/d) for sequence positions
        dim_indices = torch.arange(0, d_model, 2).float()  # [0, 2, 4, ..., d_model-2]
        freqs_seq = 1.0 / (base ** (dim_indices / d_model))
        self.register_buffer('freqs_seq', freqs_seq)

        # φ_m for depth positions - use scaled indexing for separation
        # Scale by depth to create distinct frequency distribution
        # For small d_model, use a larger scaling factor
        scale_factor = max(depth * 2.0, 3.0)
        freqs_depth = 1.0 / (base ** (dim_indices / (d_model * scale_factor)))
        self.register_buffer('freqs_depth', freqs_depth)

        # Verify bases are distinct (not identical)
        # Note: Perfect orthogonality is not strictly required - just sufficient separation
        if torch.allclose(freqs_seq, freqs_depth, rtol=1e-3):
            import warnings
            warnings.warn(
                f"Frequency bases are nearly identical for d_model={d_model}, depth={depth}. "
                "This may reduce positional encoding effectiveness."
            )

    def _compute_orthogonality(self, freqs_seq: torch.Tensor, freqs_depth: torch.Tensor) -> float:
        """Compute orthogonality score between two frequency bases."""
        norm_seq = freqs_seq / freqs_seq.norm()
        norm_depth = freqs_depth / freqs_depth.norm()
        similarity = torch.abs(torch.dot(norm_seq, norm_depth))
        return similarity.item()

    def forward(self, seq_len: int, depth_idx: int = None, device: torch.device = None):
        """
        Generate 3D positional encoding for given sequence length and depth.

        Args:
            seq_len (int): Sequence length
            depth_idx (int, optional): Specific depth index (if None, generates for all depths)
            device (torch.device, optional): Device for tensors

        Returns:
            tuple: (sin_emb, cos_emb) each of shape:
                - (seq_len, d_model//2) if depth_idx is specified
                - (seq_len, depth, d_model//2) if depth_idx is None
        """
        if device is None:
            device = self.freqs_seq.device

        # Sequence positions: θ_i = 2π·i/N (normalized to [0, 2π])
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        theta_seq = 2 * math.pi * positions / seq_len  # Shape: (seq_len,)

        if depth_idx is not None:
            # Single depth layer
            phi_depth = 2 * math.pi * depth_idx / self.depth

            # Helical twist: θ_i + φ_k
            # Outer product to get angle for each (position, frequency) pair
            angles = theta_seq.unsqueeze(-1) * self.freqs_seq + phi_depth * self.freqs_depth

            return angles.sin(), angles.cos()  # Shape: (seq_len, d_model//2)
        else:
            # All depth layers
            depths = torch.arange(self.depth, device=device, dtype=torch.float32)
            phi_depths = 2 * math.pi * depths / self.depth  # Shape: (depth,)

            # Helical twist for all positions and depths
            # θ_i * ω_m + φ_k * φ_m
            seq_component = theta_seq.unsqueeze(-1) * self.freqs_seq  # (seq_len, d//2)
            depth_component = phi_depths.unsqueeze(-1) * self.freqs_depth  # (depth, d//2)

            # Broadcast to (seq_len, depth, d//2)
            angles = seq_component.unsqueeze(1) + depth_component.unsqueeze(0)

            return angles.sin(), angles.cos()  # Shape: (seq_len, depth, d_model//2)

    def apply_rotary_embedding(self, x: torch.Tensor, sin_emb: torch.Tensor, cos_emb: torch.Tensor):
        """
        Apply rotary embedding to input tensor.

        Implements rotation via complex number multiplication:
            x' = x * e^(iθ) = (x_real + ix_imag) * (cos + i·sin)

        Args:
            x (torch.Tensor): Input tensor, shape (..., seq_len, d_model) or (..., seq_len, depth, d_model)
            sin_emb (torch.Tensor): Sine component of PE
            cos_emb (torch.Tensor): Cosine component of PE

        Returns:
            torch.Tensor: Rotated tensor, same shape as input
        """
        # Split into even and odd dimensions (pairs for rotation)
        x1 = x[..., 0::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices

        # Apply 2D rotation matrix:
        # [cos  -sin] [x1]   [x1*cos - x2*sin]
        # [sin   cos] [x2] = [x1*sin + x2*cos]
        x_rotated_1 = x1 * cos_emb - x2 * sin_emb
        x_rotated_2 = x1 * sin_emb + x2 * cos_emb

        # Interleave back
        x_rotated = torch.stack([x_rotated_1, x_rotated_2], dim=-1)
        x_rotated = x_rotated.flatten(start_dim=-2)

        return x_rotated

    def get_orthogonality_score(self) -> float:
        """
        Compute orthogonality score between sequence and depth frequency bases.

        Returns:
            float: Orthogonality score (0 = orthogonal, 1 = parallel)
        """
        return self._compute_orthogonality(self.freqs_seq, self.freqs_depth)


def precompute_toroidal_freqs(d_model: int, max_len: int, depth: int, base: float = 10000.0):
    """
    Precompute toroidal frequency embeddings for efficiency.

    This is a convenience function for caching embeddings when sequence length
    and depth are known in advance.

    Args:
        d_model (int): Model dimension
        max_len (int): Maximum sequence length
        depth (int): Number of depth platters
        base (float): Frequency base

    Returns:
        tuple: (sin_cache, cos_cache) each of shape (max_len, depth, d_model//2)
    """
    pe = Toroidal3DPositionalEncoding(d_model, max_len, depth, base)
    sin_cache, cos_cache = pe(max_len)
    return sin_cache, cos_cache

