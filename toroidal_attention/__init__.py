"""
Toroidal Attention Mechanism for Large Language Models

This package implements a novel 3D toroidal attention mechanism inspired by HDD storage geometry,
featuring circular context wrapping and depth stacking to mitigate boundary effects in long sequences.

Key components:
- ToroidalAttention: Main attention module with 3D toroidal structure
- Toroidal3DPositionalEncoding: Orthogonal rotary embeddings for sequence and depth
- ToroidalDistance: Cyclic distance metric with wrap-around
- DepthFusion: Low-rank fusion across depth platters
"""

from .core import ToroidalAttention
from .distance import compute_toroidal_distance
from .fusion import DepthFusion
from .positional_encoding import Toroidal3DPositionalEncoding
from .positional_encoding_orthogonal import Toroidal3DPositionalEncodingOrthogonal
from .latent import LatentCfg, LatentKV

__version__ = "0.1.0"
__all__ = [
    "ToroidalAttention",
    "Toroidal3DPositionalEncoding",
    "Toroidal3DPositionalEncodingOrthogonal",
    "compute_toroidal_distance",
    "DepthFusion",
    "LatentCfg",
    "LatentKV",
]

