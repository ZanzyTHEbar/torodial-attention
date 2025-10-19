"""
Streaming Latent Attention components.

LatentKV maintains a compact per-head latent state updated from (k_t, v_t)
and supports attending queries against this state for O(1) memory inference.
"""

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LatentCfg:
    latent_dim: int = 64
    update: Literal["gru", "linear"] = "gru"


class LatentKV(nn.Module):
    """
    Per-head latent KV compressor.

    Shapes:
        - State z: (B, H, m) where m=latent_dim
        - k_t, v_t: (B, H, d_k)
        - q_t: (B, H, d_k)
    """

    def __init__(self, d_k: int, cfg: LatentCfg):
        super().__init__()
        self.cfg = cfg
        m = cfg.latent_dim
        self.kv_proj = nn.Linear(2 * d_k, m, bias=True)
        if cfg.update == "gru":
            self.cell = nn.GRUCell(m, m)
        else:
            self.Wz = nn.Linear(m, m, bias=True)
            self.Wa = nn.Linear(m, m, bias=True)
        self.q_to_m = nn.Linear(d_k, m, bias=False)
        self.out_proj = nn.Linear(m, d_k, bias=False)

    def init_state(self, batch_size: int, num_heads: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, num_heads, self.cfg.latent_dim, device=device)

    def update(self, z_prev: torch.Tensor, k_t: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
        B, H, d = k_t.shape
        kv = torch.cat([k_t, v_t], dim=-1)  # (B,H,2*d_k)
        kv_lat = self.kv_proj(kv)  # (B,H,m)
        z_prev_flat = z_prev.reshape(B * H, -1)
        kv_flat = kv_lat.reshape(B * H, -1)
        if self.cfg.update == "gru":
            z_new = self.cell(kv_flat, z_prev_flat)
        else:
            z_new = self.Wz(z_prev_flat) + self.Wa(kv_flat)
        return z_new.reshape(B, H, -1)

    def attend(self, q_t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # q_t: (B,H,d_k), z: (B,H,m)
        # Project query to latent dim and compute a scalar gate
        q_m = self.q_to_m(q_t)  # (B,H,m)
        scale = z.shape[-1] ** 0.5
        gate = torch.sigmoid(((q_m * z).sum(dim=-1, keepdim=True)) / scale)  # (B,H,1)
        mixed = z * gate  # (B,H,m)
        return self.out_proj(mixed)  # (B,H,d_k)


