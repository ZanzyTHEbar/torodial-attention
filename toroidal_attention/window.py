"""
Sliding Window utilities for toroidal attention.

Builds boolean or additive masks for local attention with circular wrap along
the sequence dimension and replication across depth platters.
"""

import torch


def build_toroidal_window_mask(
    seq_len: int,
    depth: int,
    window_size: int,
    device: torch.device,
    mask_value: float = -1e9,
) -> torch.Tensor:
    """
    Create an additive attention mask of shape (ND, ND) with circular windowing.

    For each position i in [0, N-1] and depth k in [0, D-1], allow attention to
    positions j within +/- window_size in circular sense, at any depth by default.

    Returns:
        mask_2d (torch.Tensor): (ND, ND) where disallowed positions are set to mask_value,
                                and allowed positions are 0.
    """
    N, D = seq_len, depth
    ND = N * D
    mask = torch.full((ND, ND), mask_value, device=device)

    # For each (i,k) compute allowed j indices with wrap
    # We allow across all depths; if per-depth restriction is desired, it can be added later.
    idx = torch.arange(N, device=device)
    for i in range(N):
        # Circular window [i-W, i+W]
        left = (i - window_size) % N
        right = (i + window_size) % N
        if left <= right:
            allowed_seq = idx[(idx >= left) & (idx <= right)]
        else:
            # Wrap-around interval
            allowed_seq = torch.cat([idx[idx >= left], idx[idx <= right]])

        # Broadcast across depths: all (j, l) for l in [0,D)
        allowed_flat = (allowed_seq.unsqueeze(1) * D + torch.arange(D, device=device).unsqueeze(0)).reshape(-1)

        # Set rows corresponding to all depths at position i
        row_base = i * D
        rows = row_base + torch.arange(D, device=device)
        mask[rows.unsqueeze(1), allowed_flat.unsqueeze(0)] = 0.0

    return mask


