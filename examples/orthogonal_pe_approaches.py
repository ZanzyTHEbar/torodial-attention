"""
Alternative Approaches for Orthogonal Positional Encoding

This file demonstrates three mathematically sound approaches to achieve
true orthogonality between sequence and depth frequency bases.
"""

import math
import torch
import torch.nn as nn


class OrthogonalPE_Approach1_AlternatingDimensions(nn.Module):
    """
    Approach 1: Alternating Dimension Assignment
    
    Assign different dimensions to sequence vs depth:
    - Even dimensions (0, 2, 4, ...) encode sequence position
    - Odd dimensions (1, 3, 5, ...) encode depth position
    
    This guarantees perfect orthogonality (dot product = 0) since
    there's no overlap in which dimensions are affected.
    
    Pros: Mathematically perfect orthogonality
    Cons: Halves the capacity for each dimension
    """
    
    def __init__(self, d_model: int, max_len: int = 2048, depth: int = 4, base: float = 10000.0):
        super().__init__()
        
        assert d_model % 2 == 0, "d_model must be even"
        
        self.d_model = d_model
        self.max_len = max_len
        self.depth = depth
        self.base = base
        
        # Split dimensions: half for sequence, half for depth
        n_dims = d_model // 2
        seq_dims = n_dims // 2
        depth_dims = n_dims - seq_dims
        
        # Sequence frequencies (use first half of dimension pairs)
        seq_indices = torch.arange(0, seq_dims * 2, 2).float()
        freqs_seq = 1.0 / (base ** (seq_indices / d_model))
        self.register_buffer('freqs_seq', freqs_seq)
        self.register_buffer('seq_dim_mask', torch.arange(n_dims) < seq_dims)
        
        # Depth frequencies (use second half of dimension pairs)
        depth_indices = torch.arange(0, depth_dims * 2, 2).float()
        freqs_depth = 1.0 / (base ** (depth_indices / d_model))
        self.register_buffer('freqs_depth', freqs_depth)
        self.register_buffer('depth_dim_mask', torch.arange(n_dims) >= seq_dims)
        
        # Verify perfect orthogonality
        ortho_score = self._compute_orthogonality()
        print(f"Approach 1 orthogonality score: {ortho_score:.6f} (should be ~0)")
    
    def _compute_orthogonality(self) -> float:
        """Compute orthogonality between the two bases."""
        # Create full frequency vectors with zeros in non-used dimensions
        full_seq = torch.zeros(self.d_model // 2)
        full_depth = torch.zeros(self.d_model // 2)
        
        full_seq[self.seq_dim_mask] = self.freqs_seq
        full_depth[self.depth_dim_mask] = self.freqs_depth
        
        # Normalize and compute dot product
        norm_seq = full_seq / (full_seq.norm() + 1e-8)
        norm_depth = full_depth / (full_depth.norm() + 1e-8)
        
        return torch.abs(torch.dot(norm_seq, norm_depth)).item()
    
    def forward(self, seq_len: int, depth_idx: int = None, device: torch.device = None):
        """Generate positional encodings."""
        if device is None:
            device = self.freqs_seq.device
        
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        theta_seq = 2 * math.pi * positions / seq_len
        
        if depth_idx is not None:
            phi_depth = 2 * math.pi * depth_idx / self.depth
            
            # Create full angle array
            angles = torch.zeros(seq_len, self.d_model // 2, device=device)
            angles[:, self.seq_dim_mask] = theta_seq.unsqueeze(-1) * self.freqs_seq
            angles[:, self.depth_dim_mask] = phi_depth * self.freqs_depth
            
            return angles.sin(), angles.cos()
        else:
            depths = torch.arange(self.depth, device=device, dtype=torch.float32)
            phi_depths = 2 * math.pi * depths / self.depth
            
            angles = torch.zeros(seq_len, self.depth, self.d_model // 2, device=device)
            angles[:, :, self.seq_dim_mask] = theta_seq.unsqueeze(-1).unsqueeze(-1) * self.freqs_seq
            angles[:, :, self.depth_dim_mask] = phi_depths.unsqueeze(0).unsqueeze(-1) * self.freqs_depth
            
            return angles.sin(), angles.cos()


class OrthogonalPE_Approach2_DifferentBases(nn.Module):
    """
    Approach 2: Different Base Values
    
    Use different base values for sequence vs depth:
    - Sequence: base = 10000 (standard RoPE)
    - Depth: base = different value (e.g., 500, or sqrt(10000))
    
    This creates frequency distributions in different "octaves"
    that are less correlated.
    
    Pros: Maintains full capacity for both dimensions
    Cons: Not perfectly orthogonal, but much better than single-base
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 2048,
        depth: int = 4,
        base_seq: float = 10000.0,
        base_depth: float = 500.0  # Different base for depth
    ):
        super().__init__()
        
        assert d_model % 2 == 0, "d_model must be even"
        
        self.d_model = d_model
        self.max_len = max_len
        self.depth = depth
        self.base_seq = base_seq
        self.base_depth = base_depth
        
        dim_indices = torch.arange(0, d_model, 2).float()
        
        # Sequence frequencies with standard base
        freqs_seq = 1.0 / (base_seq ** (dim_indices / d_model))
        self.register_buffer('freqs_seq', freqs_seq)
        
        # Depth frequencies with different base
        freqs_depth = 1.0 / (base_depth ** (dim_indices / d_model))
        self.register_buffer('freqs_depth', freqs_depth)
        
        # Check orthogonality
        ortho_score = self._compute_orthogonality()
        print(f"Approach 2 orthogonality score: {ortho_score:.6f}")
    
    def _compute_orthogonality(self) -> float:
        """Compute orthogonality score."""
        norm_seq = self.freqs_seq / self.freqs_seq.norm()
        norm_depth = self.freqs_depth / self.freqs_depth.norm()
        return torch.abs(torch.dot(norm_seq, norm_depth)).item()
    
    def forward(self, seq_len: int, depth_idx: int = None, device: torch.device = None):
        """Generate positional encodings."""
        if device is None:
            device = self.freqs_seq.device
        
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        theta_seq = 2 * math.pi * positions / seq_len
        
        if depth_idx is not None:
            phi_depth = 2 * math.pi * depth_idx / self.depth
            
            angles = theta_seq.unsqueeze(-1) * self.freqs_seq + phi_depth * self.freqs_depth
            return angles.sin(), angles.cos()
        else:
            depths = torch.arange(self.depth, device=device, dtype=torch.float32)
            phi_depths = 2 * math.pi * depths / self.depth
            
            seq_component = theta_seq.unsqueeze(-1) * self.freqs_seq
            depth_component = phi_depths.unsqueeze(-1) * self.freqs_depth
            
            angles = seq_component.unsqueeze(1) + depth_component.unsqueeze(0)
            return angles.sin(), angles.cos()


class OrthogonalPE_Approach3_GramSchmidt(nn.Module):
    """
    Approach 3: Gram-Schmidt Orthogonalization
    
    Start with two frequency sets, then explicitly orthogonalize them
    using the Gram-Schmidt process.
    
    Process:
    1. Start with seq and depth frequencies
    2. Keep freqs_seq as-is
    3. Project freqs_depth onto freqs_seq and subtract (orthogonalize)
    4. Normalize the result
    
    Pros: Guarantees perfect orthogonality while keeping full capacity
    Cons: Destroys the nice exponential structure of depth frequencies
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 2048,
        depth: int = 4,
        base: float = 10000.0
    ):
        super().__init__()
        
        assert d_model % 2 == 0, "d_model must be even"
        
        self.d_model = d_model
        self.max_len = max_len
        self.depth = depth
        self.base = base
        
        dim_indices = torch.arange(0, d_model, 2).float()
        
        # Sequence frequencies (unchanged)
        freqs_seq = 1.0 / (base ** (dim_indices / d_model))
        self.register_buffer('freqs_seq', freqs_seq)
        
        # Initial depth frequencies
        freqs_depth_initial = 1.0 / (base ** (dim_indices / (d_model * depth * 2)))
        
        # Gram-Schmidt orthogonalization
        # Project depth onto seq
        projection = torch.dot(freqs_depth_initial, freqs_seq) / torch.dot(freqs_seq, freqs_seq)
        
        # Subtract projection to make orthogonal
        freqs_depth_ortho = freqs_depth_initial - projection * freqs_seq
        
        # Normalize to have similar magnitude
        freqs_depth_ortho = freqs_depth_ortho * (freqs_seq.norm() / freqs_depth_ortho.norm())
        
        self.register_buffer('freqs_depth', freqs_depth_ortho)
        
        # Verify perfect orthogonality
        ortho_score = self._compute_orthogonality()
        print(f"Approach 3 orthogonality score: {ortho_score:.10f} (should be ~0)")
    
    def _compute_orthogonality(self) -> float:
        """Compute orthogonality score."""
        norm_seq = self.freqs_seq / self.freqs_seq.norm()
        norm_depth = self.freqs_depth / self.freqs_depth.norm()
        return torch.abs(torch.dot(norm_seq, norm_depth)).item()
    
    def forward(self, seq_len: int, depth_idx: int = None, device: torch.device = None):
        """Generate positional encodings."""
        if device is None:
            device = self.freqs_seq.device
        
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        theta_seq = 2 * math.pi * positions / seq_len
        
        if depth_idx is not None:
            phi_depth = 2 * math.pi * depth_idx / self.depth
            
            angles = theta_seq.unsqueeze(-1) * self.freqs_seq + phi_depth * self.freqs_depth
            return angles.sin(), angles.cos()
        else:
            depths = torch.arange(self.depth, device=device, dtype=torch.float32)
            phi_depths = 2 * math.pi * depths / self.depth
            
            seq_component = theta_seq.unsqueeze(-1) * self.freqs_seq
            depth_component = phi_depths.unsqueeze(-1) * self.freqs_depth
            
            angles = seq_component.unsqueeze(1) + depth_component.unsqueeze(0)
            return angles.sin(), angles.cos()


def compare_approaches():
    """Compare the three approaches."""
    print("=" * 80)
    print("COMPARISON OF ORTHOGONAL POSITIONAL ENCODING APPROACHES")
    print("=" * 80)
    
    configs = [(128, 4), (256, 8), (64, 2)]
    
    for d_model, depth in configs:
        print(f"\nConfiguration: d_model={d_model}, depth={depth}")
        print("-" * 60)
        
        # Approach 1
        pe1 = OrthogonalPE_Approach1_AlternatingDimensions(d_model, depth=depth)
        
        # Approach 2
        pe2 = OrthogonalPE_Approach2_DifferentBases(d_model, depth=depth)
        
        # Approach 3
        pe3 = OrthogonalPE_Approach3_GramSchmidt(d_model, depth=depth)
        
        # Test forward pass
        sin1, cos1 = pe1(seq_len=16, depth_idx=2)
        sin2, cos2 = pe2(seq_len=16, depth_idx=2)
        sin3, cos3 = pe3(seq_len=16, depth_idx=2)
        
        print(f"  Approach 1 output shape: {sin1.shape}")
        print(f"  Approach 2 output shape: {sin2.shape}")
        print(f"  Approach 3 output shape: {sin3.shape}")
        print()


if __name__ == "__main__":
    compare_approaches()
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("""
For the toroidal attention mechanism, I recommend:

**Best: Approach 3 (Gram-Schmidt)** if you need perfect orthogonality
  ✓ Mathematically perfect orthogonality (dot product ≈ 0)
  ✓ Full capacity for both dimensions
  ✗ Destroys exponential frequency structure for depth
  
**Good: Approach 2 (Different Bases)** for practical use
  ✓ Much better orthogonality than current approach
  ✓ Maintains exponential structure for both
  ✓ Simple to implement
  ✗ Not perfectly orthogonal
  
**Academic: Approach 1 (Alternating)** for theoretical guarantees
  ✓ Perfect orthogonality by construction
  ✗ Halves the effective capacity
  ✗ Asymmetric treatment of dimensions

For most applications, **Approach 2 with base_depth ≈ 500-1000** provides 
the best balance of orthogonality, simplicity, and maintaining the nice
properties of RoPE.
""")
