"""
Attention computation backends for Toroidal Attention.

Provides:
- SDPA-based attention using torch.scaled_dot_product_attention with additive bias
- Optional FlashAttention v2 wrapper (gated; no additive bias support)
"""

from typing import Optional

import importlib.util
import torch
import torch.nn.functional as F


def _expand_bias_for_sdpa(
    bias_2d: Optional[torch.Tensor],
    batch_size: int,
    num_heads: int,
    query_len: int,
    key_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """
    Expand a (L,S) additive bias to (B, H, L, S) for SDPA.
    Accepts None and returns None.
    """
    if bias_2d is None:
        return None
    # Ensure correct shape
    assert bias_2d.shape == (query_len, key_len), (
        f"bias_2d shape {bias_2d.shape} != ({query_len}, {key_len})"
    )
    bias = bias_2d.to(device=device, dtype=dtype)
    bias = bias.unsqueeze(0).unsqueeze(0)  # (1,1,L,S)
    bias = bias.expand(batch_size, num_heads, query_len, key_len)
    return bias


def compute_attention_sdpa(
    q: torch.Tensor,  # (B, H, L, d)
    k: torch.Tensor,  # (B, H, S, d)
    v: torch.Tensor,  # (B, H, S, d)
    attn_bias_2d: Optional[torch.Tensor] = None,  # (L,S) additive bias
    extra_mask_2d: Optional[torch.Tensor] = None,  # (L,S) additive mask (e.g., window, padding)
    is_causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Compute attention using PyTorch SDPA with additive bias.

    Notes:
    - SDPA performs scaling internally; q,k should be unscaled.
    - attn_mask is additive and broadcastable to (B,H,L,S).
    - If both attn_bias_2d and extra_mask_2d provided, they are summed.
    """
    B, H, L, _ = q.shape
    S = k.shape[2]
    device = q.device
    dtype = q.dtype

    bias = None
    if attn_bias_2d is not None or extra_mask_2d is not None:
        # Sum into a single 2D bias
        bias_2d = None
        if attn_bias_2d is not None:
            bias_2d = attn_bias_2d.to(device=device, dtype=dtype)
        if extra_mask_2d is not None:
            extra = extra_mask_2d.to(device=device, dtype=dtype)
            bias_2d = extra if bias_2d is None else (bias_2d + extra)
        bias = _expand_bias_for_sdpa(bias_2d, B, H, L, S, device, dtype)

    # SDPA expects (B,H,L,S) mask
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=bias, dropout_p=dropout_p, is_causal=is_causal
    )
    return out


def has_flash2() -> bool:
    """Return True if flash_attn package is available."""
    return importlib.util.find_spec("flash_attn") is not None


def compute_attention_flash2(
    q: torch.Tensor,  # (B, H, L, d)
    k: torch.Tensor,  # (B, H, S, d)
    v: torch.Tensor,  # (B, H, S, d)
    is_causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Compute attention using FlashAttention v2.

    Constraints:
    - No additive bias/masks (beyond causal) supported in this wrapper.
    - Requires flash_attn to be installed.

    Implementation note:
    FlashAttention v2 Python API variants differ across versions. Here we fall back to
    SDPA-like behavior if the expected symbols are unavailable.
    """
    if not has_flash2():
        raise RuntimeError("flash_attn is not available")

    # Try common API: flash_attn.flash_attn_func(q, k, v, ...) with (B*H, L, d)
    try:
        from flash_attn import flash_attn_func  # type: ignore

        B, H, L, d = q.shape
        S = k.shape[2]
        assert L == S, "flash_attn requires square attention in this simple wrapper"

        q_ = q.transpose(1, 0).reshape(H * B, L, d)
        k_ = k.transpose(1, 0).reshape(H * B, L, d)
        v_ = v.transpose(1, 0).reshape(H * B, L, d)

        out_ = flash_attn_func(
            q_, k_, v_, dropout_p if q.requires_grad else 0.0, softmax_scale=None, causal=is_causal
        )
        out = out_.reshape(H, B, L, d).transpose(1, 0)
        return out
    except Exception as e:
        # Last resort: indicate unavailability to caller
        raise RuntimeError(f"flash_attn v2 path unavailable: {e}")


