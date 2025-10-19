"""
Toroidal Attention Core Module

Main implementation of 3D toroidal attention mechanism with circular wrapping
and depth stacking, inspired by HDD platter geometry.

Key features:
- Circular sequence wrapping (periodic mod N)
- 3D depth stacking (D platters)
- Orthogonal rotary positional encoding
- Toroidal distance-based attention bias
- Low-rank depth fusion

Mathematical formulation from EDD:
    A_{(i,k)(j,l)} = (Q_{ik} K_{jl}^T / √d_k) - λ·δ((i,k),(j,l))
    P = softmax_c(A)  # Cyclic softmax
    O = P V
    O_fused = fusion(O, Ω)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .distance import create_distance_bias
from .fusion import DepthFusion
from .positional_encoding import Toroidal3DPositionalEncoding
from .positional_encoding_orthogonal import Toroidal3DPositionalEncodingOrthogonal
from .backends import (
    compute_attention_sdpa,
    compute_attention_flash2,
    has_flash2,
)
from .window import build_toroidal_window_mask
from .latent import LatentCfg, LatentKV


class ToroidalAttention(nn.Module):
    """
    3D Toroidal Attention Mechanism.

    Implements attention with toroidal topology: sequence positions wrap
    circularly (like HDD tracks) and stack vertically (like HDD platters).

    Args:
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        max_len (int): Maximum sequence length
        depth (int): Number of depth platters (D)
        lambda_distance (float): Distance bias scaling factor λ
        fusion_mode (str): Depth fusion strategy ('low_rank', 'attention', 'mean')
        fusion_rank (int, optional): Rank for low-rank fusion (default: depth//4)
        dropout (float): Dropout probability
        bias (bool): Whether to use bias in linear projections

    Shape:
        - Input: (B, N, d_model)
        - Output: (B, N, d_model)

    Example:
        >>> attn = ToroidalAttention(d_model=512, n_heads=8, depth=4)
        >>> x = torch.randn(2, 128, 512)
        >>> output = attn(x)
        >>> output.shape
        torch.Size([2, 128, 512])
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_len: int = 2048,
        depth: int = 4,
        lambda_distance: float = 0.1,
        fusion_mode: str = 'low_rank',
        fusion_rank: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        attn_chunk_size: Optional[int] = None,
        use_checkpoint: bool = False,
        backend: str = 'sdpa',
        window_size: Optional[int] = None,
        allow_flash2: bool = True,
        latent_cfg: Optional[LatentCfg] = None,
        use_orthogonal_pe: bool = True,
    ):
        super().__init__()

        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        assert d_model % depth == 0, f"d_model ({d_model}) must be divisible by depth ({depth})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.max_len = max_len
        self.depth = depth
        self.lambda_distance = lambda_distance

        # Attention dimensions
        self.d_k = d_model // n_heads  # Head dimension
        self.d_per_depth = d_model // depth  # Dimension per platter
        
        # Critical: Compute head_dim_per_depth for RoPE initialization
        # This is the actual dimension of tensors after splitting across depth
        assert self.d_k % depth == 0, \
            f"head_dim d_k ({self.d_k}) must be divisible by depth ({depth})"
        self.head_dim_per_depth = self.d_k // depth

        # Linear projections for Q, K, V
        # Note: We project to d_model, then reshape to (n_heads, depth, d_k)
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        # 3D Rotary Positional Encoding
        # Initialize with head_dim_per_depth, not d_k!
        # This is the actual dimension we apply RoPE to after splitting
        # Use orthogonal version for better sequence/depth separation
        PE_Class = Toroidal3DPositionalEncodingOrthogonal if use_orthogonal_pe else Toroidal3DPositionalEncoding
        self.pos_encoding = PE_Class(
            d_model=self.head_dim_per_depth,  # Apply to split head dimension
            max_len=max_len,
            depth=depth
        )

        # Depth fusion module
        self.fusion = DepthFusion(
            depth=depth,
            rank=fusion_rank,
            fusion_mode=fusion_mode,
            dropout=dropout
        )

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Cache for distance bias (computed once per sequence length)
        self.register_buffer('_cached_distance_bias', None, persistent=False)
        self._cached_seq_len = None

        # Cache for RoPE sin/cos per sequence length (device-specific)
        self._rope_cache = {}
        
        # Optimization toggles
        self.attn_chunk_size = attn_chunk_size
        self.use_checkpoint = use_checkpoint
        self.backend = backend
        self.window_size = window_size
        self.allow_flash2 = allow_flash2

        # Optional streaming latent path
        self._latent: Optional[LatentKV] = None
        if latent_cfg is not None:
            self._latent = LatentKV(d_k=self.head_dim_per_depth, cfg=latent_cfg)

    def _get_rope_embeddings(self, seq_len: int, device: torch.device):
        """
        Returns cached (sin_emb, cos_emb) for given sequence length and device.
        Shapes: (seq_len, depth, d_k//2)
        """
        cache_key = (seq_len, str(device))
        if cache_key not in self._rope_cache:
            sin_emb, cos_emb = self.pos_encoding(seq_len, device=device)
            self._rope_cache[cache_key] = (sin_emb, cos_emb)
        return self._rope_cache[cache_key]

    def _get_distance_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Get or compute distance bias tensor.

        Caches result for efficiency when sequence length doesn't change.

        Returns:
            torch.Tensor: Distance bias, shape (seq_len, depth, seq_len, depth)
        """
        if self._cached_seq_len != seq_len or self._cached_distance_bias is None:
            bias = create_distance_bias(
                seq_len=seq_len,
                depth=self.depth,
                lambda_param=self.lambda_distance,
                device=device
            )
            self._cached_distance_bias = bias
            self._cached_seq_len = seq_len

        return self._cached_distance_bias

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of toroidal attention.

        Args:
            x (torch.Tensor): Input tensor, shape (B, N, d_model)
            mask (torch.Tensor, optional): Attention mask, shape (B, N, N) or (N, N)
                                          True/1 for positions to mask out
            return_attention (bool): Whether to return attention weights

        Returns:
            output (torch.Tensor): Attention output, shape (B, N, d_model)
            attention_weights (torch.Tensor, optional): If return_attention=True,
                                                        shape (B, H, N, N)
        """
        B, N, d_model = x.shape
        device = x.device

        assert d_model == self.d_model, f"Input d_model {d_model} != expected {self.d_model}"

        # 1. Project to Q, K, V
        Q = self.W_q(x)  # (B, N, d_model)
        K = self.W_k(x)  # (B, N, d_model)
        V = self.W_v(x)  # (B, N, d_model)

        # 2. Split into heads, then shard head_dim across depth: d_k = head_dim_per_depth * depth
        # Shapes: Qh,Kh,Vh -> (B, N, H, d_k)
        H = self.n_heads
        d_k = self.d_k
        head_dim_per_depth = self.head_dim_per_depth  # Use pre-computed value

        Qh = Q.view(B, N, H, d_k)
        Kh = K.view(B, N, H, d_k)
        Vh = V.view(B, N, H, d_k)

        # Reshape to 3D toroidal per head: (B, N, D, H, head_dim_per_depth)
        Q_3d = Qh.view(B, N, H, self.depth, head_dim_per_depth).permute(0, 1, 3, 2, 4)
        K_3d = Kh.view(B, N, H, self.depth, head_dim_per_depth).permute(0, 1, 3, 2, 4)
        V_3d = Vh.view(B, N, H, self.depth, head_dim_per_depth).permute(0, 1, 3, 2, 4)

        # 3. Apply 3D RoPE to Q and K (cached)
        sin_emb, cos_emb = self._get_rope_embeddings(N, device)  # (N, D, head_dim_per_depth//2)
        sin_emb = sin_emb.unsqueeze(0).unsqueeze(3)  # (1, N, D, 1, head_dim//2)
        cos_emb = cos_emb.unsqueeze(0).unsqueeze(3)  # (1, N, D, 1, head_dim//2)
        Q_rot = self.pos_encoding.apply_rotary_embedding(Q_3d, sin_emb, cos_emb)
        K_rot = self.pos_encoding.apply_rotary_embedding(K_3d, sin_emb, cos_emb)

        # 4. Flatten (N,D) → ND for full 3D bias application
        # (B, N, D, H, hdpd) -> (B, H, ND, hdpd)
        Q_flat = Q_rot.permute(0, 3, 1, 2, 4).reshape(B, H, N * self.depth, head_dim_per_depth)
        K_flat = K_rot.permute(0, 3, 1, 2, 4).reshape(B, H, N * self.depth, head_dim_per_depth)
        V_flat = V_3d.permute(0, 3, 1, 2, 4).reshape(B, H, N * self.depth, head_dim_per_depth)

        # 5. Build distance bias and masks
        dist_bias = self._get_distance_bias(N, device)  # (N, D, N, D)
        dist_bias_2d = dist_bias.view(N * self.depth, N * self.depth)  # (ND, ND)
        attn_bias_2d = -dist_bias_2d  # additive bias to logits

        # Use -1e9 instead of -inf to prevent NaN in softmax when entire rows are masked
        MASK_VALUE = -1e9
        ND = N * self.depth

        # Optional sliding window mask (additive 2D)
        window_mask_2d = None
        if self.window_size is not None and self.window_size > 0:
            window_mask_2d = build_toroidal_window_mask(N, self.depth, self.window_size, device, MASK_VALUE)

        # Optional extra masks; if batch-specific, we will fall back to manual path
        batch_specific_mask = False
        extra_mask_2d = None
        if mask is not None:
            if mask.dim() == 2 and mask.shape == (N, N):
                base = mask.to(torch.bool)
                mask_nd = base.unsqueeze(1).unsqueeze(3).expand(N, self.depth, N, self.depth).reshape(ND, ND)
                extra_mask_2d = mask_nd.to(Q_flat.dtype) * MASK_VALUE
            elif mask.dim() == 3 and mask.shape[1:] == (N, N):
                batch_specific_mask = True
            elif mask.dim() == 2 and mask.shape[0] == B and mask.shape[1] == N:
                batch_specific_mask = True

        # Decide computation path
        compute_manual = return_attention or self.attn_chunk_size is not None or batch_specific_mask

        # Backend gating for flash2
        want_flash = (
            self.backend == 'flash2'
            and self.allow_flash2
            and has_flash2()
            and self.window_size in (None, 0)
            and self.lambda_distance == 0.0
            and not compute_manual
            and extra_mask_2d is None  # any extra additive mask disables flash2
        )

        dropout_p = (self.dropout.p if (self.dropout is not None and self.training) else 0.0)

        if not compute_manual and not want_flash:
            # SDPA path with additive bias and optional window mask
            combined_mask_2d = None
            if window_mask_2d is not None:
                combined_mask_2d = window_mask_2d
            if extra_mask_2d is not None:
                combined_mask_2d = extra_mask_2d if combined_mask_2d is None else (combined_mask_2d + extra_mask_2d)

            attn_output_flat = compute_attention_sdpa(
                Q_flat, K_flat, V_flat, attn_bias_2d, combined_mask_2d, is_causal=False, dropout_p=dropout_p
            )
        elif want_flash:
            # FlashAttention v2 path (no bias/window/masks)
            attn_output_flat = compute_attention_flash2(
                Q_flat, K_flat, V_flat, is_causal=False, dropout_p=dropout_p
            )
        else:
            # Manual path: compute scores → softmax → matmul (supports batch-specific masks)
            chunk = self.attn_chunk_size
            if chunk is None:
                scores = torch.matmul(Q_flat, K_flat.transpose(-2, -1)) / math.sqrt(head_dim_per_depth)  # (B,H,ND,ND)
                scores = scores + attn_bias_2d.unsqueeze(0).unsqueeze(0)
                if window_mask_2d is not None:
                    scores = scores + window_mask_2d.unsqueeze(0).unsqueeze(0)
                if extra_mask_2d is not None:
                    scores = scores + extra_mask_2d.unsqueeze(0).unsqueeze(0)

                # Batch masks
                if mask is not None:
                    if mask.dim() == 3 and mask.shape[1:] == (N, N):
                        base = mask.to(torch.bool)
                        mask_nd = base.unsqueeze(2).unsqueeze(4).expand(B, N, self.depth, N, self.depth).reshape(B, ND, ND)
                        scores = scores + mask_nd.unsqueeze(1).to(Q_flat.dtype) * MASK_VALUE
                    elif mask.dim() == 2 and mask.shape == (B, N):
                        pad = mask.to(torch.bool)
                        pad_nd = pad.unsqueeze(-1).expand(B, N, self.depth).reshape(B, ND)
                        key_invalid = ~pad_nd
                        scores = scores + key_invalid.unsqueeze(1).unsqueeze(2).to(Q_flat.dtype) * MASK_VALUE

                attn_weights = torch.softmax(scores, dim=-1)
                if self.dropout is not None:
                    attn_weights = self.dropout(attn_weights)
                attn_output_flat = torch.matmul(attn_weights, V_flat)
            else:
                outputs = []
                attn_weights_chunks = [] if return_attention else None
                for start in range(0, ND, chunk):
                    end = min(start + chunk, ND)
                    Qc = Q_flat[:, :, start:end, :]

                    def _chunk_forward(qs):
                        sc = torch.matmul(qs, K_flat.transpose(-2, -1)) / math.sqrt(head_dim_per_depth)
                        sc = sc + attn_bias_2d[start:end, :].unsqueeze(0).unsqueeze(0)
                        if window_mask_2d is not None:
                            sc = sc + window_mask_2d[start:end, :].unsqueeze(0).unsqueeze(0)
                        if extra_mask_2d is not None:
                            sc = sc + extra_mask_2d[start:end, :].unsqueeze(0).unsqueeze(0)
                        if mask is not None and mask.dim() == 3 and mask.shape[1:] == (N, N):
                            base = mask.to(torch.bool)
                            mask_nd = base.unsqueeze(2).unsqueeze(4).expand(B, N, self.depth, N, self.depth).reshape(B, ND, ND)
                            sc = sc + mask_nd[:, start:end, :].unsqueeze(1).to(Q_flat.dtype) * MASK_VALUE
                        elif mask is not None and mask.dim() == 2 and mask.shape == (B, N):
                            pad = mask.to(torch.bool)
                            pad_nd = pad.unsqueeze(-1).expand(B, N, self.depth).reshape(B, ND)
                            key_invalid = ~pad_nd
                            sc = sc + key_invalid.unsqueeze(1).unsqueeze(2).to(Q_flat.dtype) * MASK_VALUE
                        aw = torch.softmax(sc, dim=-1)
                        if self.dropout is not None:
                            aw = self.dropout(aw)
                        return torch.matmul(aw, V_flat), aw

                    if self.use_checkpoint and Qc.requires_grad and not return_attention:
                        from torch.utils.checkpoint import checkpoint
                        def _wrapped(qs):
                            out, _ = _chunk_forward(qs)
                            return out
                        outc = checkpoint(_wrapped, Qc)
                        outputs.append(outc)
                    else:
                        outc, awc = _chunk_forward(Qc)
                        outputs.append(outc)
                        if return_attention:
                            attn_weights_chunks.append(awc)

                attn_output_flat = torch.cat(outputs, dim=2)
                if return_attention:
                    attn_weights = torch.cat(attn_weights_chunks, dim=2)

        # Reshape back to (B, N, D, d_per_depth)
        attn_output = attn_output_flat.reshape(B, H, N, self.depth, head_dim_per_depth)
        attn_output = attn_output.permute(0, 3, 2, 1, 4)  # (B, D, N, H, hdpd)
        attn_output = attn_output.permute(0, 2, 1, 3, 4).reshape(B, N, self.depth, H * head_dim_per_depth)

        # 9. Fuse depths via fusion module → (B, N, d_model)
        output = self.fusion(attn_output)

        # 11. Output projection
        output = self.W_o(output)

        if self.dropout is not None:
            output = self.dropout(output)

        # Optionally return attention weights (averaged over depth and heads for visualization)
        if return_attention and 'attn_weights' in locals():
            attn_viz_nd = attn_weights.mean(dim=1)  # (B, ND, ND)
            attn_viz_blocks = attn_viz_nd.view(B, N, self.depth, N, self.depth)
            attn_viz = attn_viz_blocks.mean(dim=(2, 4))  # (B, N, N)
            return output, attn_viz

        return output, None

    def forward_streaming(
        self,
        x_t: torch.Tensor,  # (B, 1, d_model)
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Streaming forward using latent KV compression.

        Args:
            x_t: Single-step input (B, 1, d_model)
            state: Latent state (B, H, m); if None, initialized to zeros

        Returns:
            y_t: Output (B, 1, d_model)
            new_state: Updated latent state (B, H, m)
        """
        if self._latent is None:
            raise RuntimeError("Latent streaming not configured; provide latent_cfg in __init__")

        B, one, d_model = x_t.shape
        assert one == 1 and d_model == self.d_model
        device = x_t.device

        # Project
        Q = self.W_q(x_t)  # (B,1,d)
        K = self.W_k(x_t)
        V = self.W_v(x_t)

        H = self.n_heads
        d_k = self.d_k
        hdpd = self.head_dim_per_depth

        Qh = Q.view(B, 1, H, d_k).view(B, 1, H, self.depth, hdpd)
        Kh = K.view(B, 1, H, d_k).view(B, 1, H, self.depth, hdpd)
        Vh = V.view(B, 1, H, d_k).view(B, 1, H, self.depth, hdpd)

        # Apply RoPE at position 0
        sin_emb, cos_emb = self._get_rope_embeddings(1, device)
        # Shapes from _get_rope_embeddings: (seq_len, D, hdpd//2)
        sin_emb = sin_emb.unsqueeze(0).unsqueeze(3)  # (1,1,D,1,hdpd/2)
        cos_emb = cos_emb.unsqueeze(0).unsqueeze(3)
        # Inputs expected by PE: (B, seq_len, D, H, hdpd)
        Q_rot = self.pos_encoding.apply_rotary_embedding(Qh.view(B, 1, H, self.depth, hdpd).permute(0, 1, 3, 2, 4), sin_emb, cos_emb)
        K_rot = self.pos_encoding.apply_rotary_embedding(Kh.view(B, 1, H, self.depth, hdpd).permute(0, 1, 3, 2, 4), sin_emb, cos_emb)

        # Aggregate across depth (mean) to single per-head vector for latent
        Qh_avg = Q_rot.mean(dim=2).squeeze(1)  # (B,H,hdpd)
        Kh_avg = K_rot.mean(dim=2).squeeze(1)  # (B,H,hdpd)
        # Average V across depth to (B,H,hdpd)
        Vh_avg = Vh.mean(dim=3).squeeze(1)  # (B,H,hdpd)

        # Init state if needed
        if state is None:
            state = self._latent.init_state(B, H, device)

        # Update and attend
        z_new = self._latent.update(state, Kh_avg, Vh_avg)
        head_out = self._latent.attend(Qh_avg, z_new)  # (B,H,hdpd)

        # Broadcast across depth and fuse
        out_per_depth = head_out.unsqueeze(1).unsqueeze(2).expand(B, self.depth, 1, H, hdpd)
        out_per_depth = out_per_depth.permute(0, 2, 1, 3, 4).reshape(B, 1, self.depth, H * hdpd)
        fused = self.fusion(out_per_depth)  # (B,1,d_model)
        y_t = self.W_o(fused)
        if self.dropout is not None:
            y_t = self.dropout(y_t)

        return y_t, z_new

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return (
            f'd_model={self.d_model}, n_heads={self.n_heads}, '
            f'depth={self.depth}, max_len={self.max_len}, '
            f'lambda={self.lambda_distance}'
        )


class ToroidalAttentionLayer(nn.Module):
    """
    Complete attention layer with toroidal attention + feedforward.

    Combines:
    - Toroidal attention
    - Residual connection
    - Layer normalization
    - Feedforward network (optional)

    This matches the structure of a standard Transformer layer.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_len: int = 2048,
        depth: int = 4,
        lambda_distance: float = 0.1,
        fusion_mode: str = 'low_rank',
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        super().__init__()

        self.attention = ToroidalAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_len=max_len,
            depth=depth,
            lambda_distance=lambda_distance,
            fusion_mode=fusion_mode,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(d_model)

        # Optional feedforward network
        if d_ff is not None:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            self.norm2 = nn.LayerNorm(d_model)
        else:
            self.ffn = None
            self.norm2 = None

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through attention layer.

        Args:
            x: Input tensor (B, N, d_model)
            mask: Attention mask (optional)

        Returns:
            Output tensor (B, N, d_model)
        """
        # Self-attention with residual
        attn_output, _ = self.attention(x, mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feedforward with residual (if present)
        if self.ffn is not None:
            ffn_output = self.ffn(x)
            x = self.norm2(x + ffn_output)

        return x


def test_toroidal_attention():
    """Test toroidal attention module."""
    print("Testing Toroidal Attention Module:")

    B, N, d_model = 2, 32, 256
    n_heads, depth = 8, 4

    # Create module
    attn = ToroidalAttention(
        d_model=d_model,
        n_heads=n_heads,
        max_len=128,
        depth=depth,
        lambda_distance=0.1,
    )

    # Test input
    x = torch.randn(B, N, d_model)

    # Forward pass
    output, attn_weights = attn(x, return_attention=True)

    # Check shapes
    assert output.shape == (B, N, d_model), f"Output shape mismatch: {output.shape}"
    assert attn_weights.shape == (B, N, N), f"Attention shape mismatch: {attn_weights.shape}"

    print(f"  ✓ Forward pass: {x.shape} -> {output.shape}")
    print(f"  ✓ Attention weights shape: {attn_weights.shape}")

    # Test with mask
    mask = torch.triu(torch.ones(N, N), diagonal=1).bool()  # Causal mask
    output_masked, _ = attn(x, mask=mask)
    assert output_masked.shape == (B, N, d_model)
    print("  ✓ Causal masking works")

    # Check parameter count
    n_params = sum(p.numel() for p in attn.parameters())
    print(f"  ✓ Parameter count: {n_params:,}")

    # Test orthogonality
    ortho_score = attn.pos_encoding.get_orthogonality_score()
    print(f"  ✓ PE orthogonality score: {ortho_score:.4f} (0 = orthogonal)")

    print("\nToroidal attention validation complete!")


if __name__ == "__main__":
    test_toroidal_attention()

