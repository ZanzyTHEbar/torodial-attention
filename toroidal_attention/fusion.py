"""
Low-Rank Depth Fusion

Implements efficient cross-platter fusion for 3D toroidal attention.
Mimics HDD platter independence with sparse vertical connections.

Mathematical formulation:
    Ω ∈ R^(D×r), where r << D
    O_fused = f(O, Ω)

    where O has shape (B, N, D, d/D) and fusion reduces depth dimension.
"""

import math

import torch
import torch.nn as nn


class DepthFusion(nn.Module):
    """
    Low-rank fusion across depth platters.

    Implements learnable cross-depth mixing while maintaining computational
    efficiency through low-rank projection. Allows information flow between
    depth platters while mimicking HDD platter independence.

    Args:
        depth (int): Number of depth platters D
        rank (int, optional): Fusion rank r (if None, uses D//4)
        fusion_mode (str): Fusion strategy - 'low_rank', 'attention', or 'mean'

    Fusion modes:
        - 'low_rank': Linear low-rank projection Ω ∈ R^(D×r)
        - 'attention': Learned attention weights across depths
        - 'mean': Simple averaging (no parameters)
    """

    def __init__(
        self,
        depth: int,
        rank: int = None,
        fusion_mode: str = 'low_rank',
        dropout: float = 0.0
    ):
        super().__init__()

        self.depth = depth
        self.rank = rank if rank is not None else max(1, depth // 4)
        self.fusion_mode = fusion_mode

        assert self.rank <= depth, f"Rank {self.rank} must be <= depth {depth}"

        if fusion_mode == 'low_rank':
            # Low-rank factorization: Ω = U @ V^T
            # where U ∈ R^(D×r), V ∈ R^(D×r)
            # This ensures rank(Ω) <= r
            # Initialize closer to orthonormal columns to avoid overly low effective rank
            self.U = nn.Parameter(torch.randn(depth, self.rank))
            self.V = nn.Parameter(torch.randn(depth, self.rank))
            with torch.no_grad():
                # Orthonormalize columns via QR for better initial rank spread
                qU, _ = torch.linalg.qr(self.U, mode='reduced')
                qV, _ = torch.linalg.qr(self.V, mode='reduced')
                # Pad if reduced rank smaller than requested
                if qU.shape[1] < self.rank:
                    pad = torch.zeros(depth, self.rank - qU.shape[1])
                    qU = torch.cat([qU, pad], dim=1)
                if qV.shape[1] < self.rank:
                    pad = torch.zeros(depth, self.rank - qV.shape[1])
                    qV = torch.cat([qV, pad], dim=1)
                self.U.copy_(qU[:, : self.rank])
                self.V.copy_(qV[:, : self.rank])
            self.dropout = nn.Dropout(dropout) if dropout > 0 else None
            self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        elif fusion_mode == 'attention':
            # Learnable attention across depths
            self.depth_query = nn.Parameter(torch.randn(depth, depth))
            self.depth_key = nn.Parameter(torch.randn(depth, depth))

        elif fusion_mode == 'mean':
            # Simple averaging (no parameters)
            pass
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fuse information across depth dimension.

        Args:
            x (torch.Tensor): Input tensor with depth dimension
                             Shape: (B, N, D, d/D) or (B, N, D, H, d_k)

        Returns:
            torch.Tensor: Fused output
                         Shape: (B, N, d) or (B, N, H, d_k)
        """
        if self.fusion_mode == 'mean':
            # Simple mean pooling across depth
            # (B, N, D, d/D) -> (B, N, d/D) -> (B, N, d) via reshape
            return self._mean_fusion(x)

        elif self.fusion_mode == 'low_rank':
            return self._lowrank_fusion(x)

        elif self.fusion_mode == 'attention':
            return self._attention_fusion(x)

    def _mean_fusion(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple mean pooling across depth dimension.

        Args:
            x: Shape (B, N, D, d/D) or (B, N, D, H, d_k)

        Returns:
            Fused tensor with depth dimension removed
        """
        # Average across depth dimension (dim=2)
        x_fused = x.mean(dim=2)

        # If shape is (B, N, d/D), expand to (B, N, d) by scaling
        if len(x.shape) == 4:
            # (B, N, d/D) -> (B, N, d) by repeating
            B, N, d_per_depth = x_fused.shape
            x_fused = x_fused.unsqueeze(2).expand(B, N, self.depth, d_per_depth)
            x_fused = x_fused.reshape(B, N, self.depth * d_per_depth)

        return x_fused

    def _lowrank_fusion(self, x: torch.Tensor) -> torch.Tensor:
        """
        Low-rank projection fusion.

        Computes: x_fused = x @ Ω, where Ω = U @ V^T

        Args:
            x: Shape (B, N, D, d/D) or (B, N, D, H, d_k)

        Returns:
            Fused tensor
        """
        original_shape = x.shape
        B, N, D = original_shape[:3]

        # Compute low-rank mixing matrix Ω = U @ V^T
        # Shape: (D, D)
        Omega = torch.matmul(self.U, self.V.transpose(0, 1))

        # Normalize to maintain scale
        Omega = Omega / math.sqrt(self.rank)

        if self.dropout is not None:
            Omega = self.dropout(Omega)

        # Reshape x for matrix multiplication
        if len(original_shape) == 4:
            # (B, N, D, d/D) -> (B*N, D, d/D)
            d_per_depth = original_shape[3]
            x_flat = x.reshape(B * N, D, d_per_depth)

            # Apply fusion: (B*N, D, d/D) @ (D, D) -> (B*N, D, d/D)
            # Actually we want to mix across depth, so:
            # (B*N, D, d/D) -> (B*N, d/D, D) @ (D, D) -> (B*N, d/D, D) -> (B*N, D, d/D)
            x_flat = x_flat.transpose(1, 2)  # (B*N, d/D, D)
            x_fused = torch.matmul(x_flat, Omega)  # (B*N, d/D, D)
            x_fused = x_fused.transpose(1, 2)  # (B*N, D, d/D)

            # Reshape back and flatten depth
            x_fused = x_fused.reshape(B, N, D, d_per_depth)
            x_fused = x_fused.reshape(B, N, D * d_per_depth)

        else:
            # (B, N, D, H, d_k) case
            H, d_k = original_shape[3], original_shape[4]
            x_flat = x.reshape(B * N * H, D, d_k)

            x_flat = x_flat.transpose(1, 2)  # (B*N*H, d_k, D)
            x_fused = torch.matmul(x_flat, Omega)  # (B*N*H, d_k, D)
            x_fused = x_fused.transpose(1, 2)  # (B*N*H, D, d_k)

            x_fused = x_fused.reshape(B, N, D, H, d_k)
            x_fused = x_fused.reshape(B, N, H, D * d_k)

        return x_fused

    def _attention_fusion(self, x: torch.Tensor) -> torch.Tensor:
        """
        Attention-based fusion across depths.

        Computes attention weights between depth platters and mixes accordingly.

        Args:
            x: Shape (B, N, D, d/D) or (B, N, D, H, d_k)

        Returns:
            Fused tensor
        """
        original_shape = x.shape
        B, N, D = original_shape[:3]

        # Compute attention scores between depths
        # Q = depth_query, K = depth_key
        attn_scores = torch.matmul(self.depth_query, self.depth_key.transpose(0, 1))
        attn_scores = attn_scores / math.sqrt(D)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (D, D)

        # Apply attention-based mixing
        if len(original_shape) == 4:
            d_per_depth = original_shape[3]
            x_flat = x.reshape(B * N, D, d_per_depth)

            # Mix: (B*N, D, d/D) with (D, D)
            x_flat = x_flat.transpose(1, 2)  # (B*N, d/D, D)
            x_fused = torch.matmul(x_flat, attn_weights)  # (B*N, d/D, D)
            x_fused = x_fused.transpose(1, 2)  # (B*N, D, d/D)

            x_fused = x_fused.reshape(B, N, D, d_per_depth)
            x_fused = x_fused.reshape(B, N, D * d_per_depth)
        else:
            H, d_k = original_shape[3], original_shape[4]
            x_flat = x.reshape(B * N * H, D, d_k)

            x_flat = x_flat.transpose(1, 2)
            x_fused = torch.matmul(x_flat, attn_weights)
            x_fused = x_fused.transpose(1, 2)

            x_fused = x_fused.reshape(B, N, D, H, d_k)
            x_fused = x_fused.reshape(B, N, H, D * d_k)

        return x_fused

    def get_fusion_matrix(self) -> torch.Tensor:
        """
        Get the effective fusion matrix Ω.

        Returns:
            torch.Tensor: Fusion matrix of shape (D, D)
        """
        if self.fusion_mode == 'low_rank':
            Omega = torch.matmul(self.U, self.V.transpose(0, 1))
            return Omega / math.sqrt(self.rank)

        elif self.fusion_mode == 'attention':
            attn_scores = torch.matmul(self.depth_query, self.depth_key.transpose(0, 1))
            attn_scores = attn_scores / math.sqrt(self.depth)
            return torch.softmax(attn_scores, dim=-1)

        else:  # mean
            return torch.ones(self.depth, self.depth) / self.depth

    def get_effective_rank(self) -> float:
        """
        Compute effective rank of fusion matrix.

        Returns:
            float: Effective rank (sum of normalized singular values squared)
        """
        Omega = self.get_fusion_matrix()

        # Compute SVD
        _, S, _ = torch.svd(Omega)

        # Normalize singular values
        S_norm = S / S.sum()

        # Effective rank = exp(entropy of singular value distribution)
        entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum()
        effective_rank = torch.exp(entropy)

        return effective_rank.item()


class IdentityFusion(nn.Module):
    """
    Identity fusion that simply flattens depth dimension.

    Useful for ablation studies or as a baseline.
    """

    def __init__(self, depth: int):
        super().__init__()
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flatten depth dimension without any mixing.

        Args:
            x: Shape (B, N, D, d/D)

        Returns:
            Shape (B, N, d)
        """
        B, N, D = x.shape[:3]

        if len(x.shape) == 4:
            d_per_depth = x.shape[3]
            return x.reshape(B, N, D * d_per_depth)
        else:
            H, d_k = x.shape[3], x.shape[4]
            return x.reshape(B, N, H, D * d_k)


def test_fusion_properties():
    """Test depth fusion module properties."""
    print("Testing Depth Fusion Properties:")

    B, N, D, d_per_depth = 2, 16, 4, 64
    x = torch.randn(B, N, D, d_per_depth)

    # Test different fusion modes
    for mode in ['mean', 'low_rank', 'attention']:
        fusion = DepthFusion(depth=D, rank=2, fusion_mode=mode)
        output = fusion(x)

        expected_shape = (B, N, D * d_per_depth)
        assert output.shape == expected_shape, \
            f"Mode {mode}: Expected shape {expected_shape}, got {output.shape}"

        print(f"  ✓ {mode} fusion: shape {output.shape}")

        if mode in ['low_rank', 'attention']:
            eff_rank = fusion.get_effective_rank()
            print(f"    Effective rank: {eff_rank:.2f} / {D}")

    print("\nDepth fusion validation complete!")


if __name__ == "__main__":
    test_fusion_properties()

