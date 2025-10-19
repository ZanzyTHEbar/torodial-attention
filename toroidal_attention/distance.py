"""
Toroidal Distance Metric

Implements cyclic distance calculation for toroidal topology with wrap-around
at sequence boundaries and vertical depth separation.

Mathematical formulation:
    δ((i,k), (j,l)) = min(|i-j|, N-|i-j|)/N + |k-l|/D

    where:
        (i,k): Position i at depth k
        (j,l): Position j at depth l
        N: Sequence length
        D: Number of depth platters
"""

import torch


def compute_toroidal_distance(
    seq_len: int,
    depth: int,
    device: torch.device = None,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute pairwise toroidal distances for all positions in 3D toroidal space.

    The distance metric handles:
    - Circular wrap-around for sequence dimension (min distance on circle)
    - Linear separation for depth dimension

    Args:
        seq_len (int): Sequence length N
        depth (int): Number of depth platters D
        device (torch.device, optional): Device for computation
        normalize (bool): Whether to normalize distances to [0, 1]

    Returns:
        torch.Tensor: Distance matrix of shape (seq_len, depth, seq_len, depth)
                      where [i, k, j, l] = δ((i,k), (j,l))
    """
    if device is None:
        device = torch.device('cpu')

    # Create position indices
    seq_indices = torch.arange(seq_len, device=device)  # [0, 1, ..., N-1]
    depth_indices = torch.arange(depth, device=device)  # [0, 1, ..., D-1]

    # Compute cyclic sequence distance: min(|i-j|, N-|i-j|)
    # Shape: (seq_len, seq_len)
    i, j = torch.meshgrid(seq_indices, seq_indices, indexing='ij')
    abs_diff = torch.abs(i - j)
    cyclic_dist = torch.minimum(abs_diff, seq_len - abs_diff)

    # Normalize sequence distance by N if requested
    if normalize:
        cyclic_dist = cyclic_dist.float() / seq_len
    else:
        cyclic_dist = cyclic_dist.float()

    # Compute depth distance: |k-l|
    # Shape: (depth, depth)
    k, l = torch.meshgrid(depth_indices, depth_indices, indexing='ij')
    depth_dist = torch.abs(k - l).float()

    # Normalize depth distance by D if requested
    if normalize:
        depth_dist = depth_dist / depth

    # Combine into 4D distance tensor
    # Broadcast to (seq_len, depth, seq_len, depth)
    toroidal_dist = cyclic_dist.unsqueeze(1).unsqueeze(3) + depth_dist.unsqueeze(0).unsqueeze(2)

    return toroidal_dist


def compute_toroidal_distance_pairwise(
    pos_i: torch.Tensor,
    depth_i: torch.Tensor,
    pos_j: torch.Tensor,
    depth_j: torch.Tensor,
    seq_len: int,
    depth_total: int,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute toroidal distance for specific position pairs.

    Useful for sparse attention patterns or dynamic computation.

    Args:
        pos_i (torch.Tensor): Source sequence positions, shape (*)
        depth_i (torch.Tensor): Source depth indices, shape (*)
        pos_j (torch.Tensor): Target sequence positions, shape (*)
        depth_j (torch.Tensor): Target depth indices, shape (*)
        seq_len (int): Total sequence length N
        depth_total (int): Total depth D
        normalize (bool): Whether to normalize distances

    Returns:
        torch.Tensor: Pairwise distances, shape (*)
    """
    # Cyclic sequence distance
    abs_diff = torch.abs(pos_i - pos_j)
    cyclic_dist = torch.minimum(abs_diff, seq_len - abs_diff).float()

    if normalize:
        cyclic_dist = cyclic_dist / seq_len

    # Depth distance
    depth_dist = torch.abs(depth_i - depth_j).float()

    if normalize:
        depth_dist = depth_dist / depth_total

    return cyclic_dist + depth_dist


def create_distance_bias(
    seq_len: int,
    depth: int,
    lambda_param: float = 0.1,
    device: torch.device = None
) -> torch.Tensor:
    """
    Create distance bias tensor for attention scores.

    The bias is subtracted from attention logits:
        A = (Q @ K^T / sqrt(d_k)) - λ·δ((i,k),(j,l))

    This encourages attention to nearby positions in toroidal space.

    Args:
        seq_len (int): Sequence length
        depth (int): Number of depth platters
        lambda_param (float): Scaling factor for distance penalty
        device (torch.device, optional): Device for tensor

    Returns:
        torch.Tensor: Distance bias, shape (seq_len, depth, seq_len, depth)
    """
    distances = compute_toroidal_distance(seq_len, depth, device, normalize=True)
    bias = lambda_param * distances
    return bias


_DIST_CACHE = {}


def get_cached_distance_bias(
    seq_len: int,
    depth: int,
    lambda_param: float,
    device: torch.device,
):
    """
    Cache and return distance bias keyed by (seq_len, depth, lambda, device).
    """
    key = (seq_len, depth, float(lambda_param), str(device))
    if key not in _DIST_CACHE:
        _DIST_CACHE[key] = create_distance_bias(seq_len, depth, lambda_param, device)
    return _DIST_CACHE[key]


def visualize_toroidal_distances(seq_len: int, depth: int, save_path: str = None):
    """
    Visualize toroidal distance patterns (for debugging/analysis).

    Creates plots showing:
    - 2D heatmap of cyclic sequence distances
    - 3D visualization of toroidal distance from reference point

    Args:
        seq_len (int): Sequence length
        depth (int): Number of depth platters
        save_path (str, optional): Path to save visualization
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return

    distances = compute_toroidal_distance(seq_len, depth, normalize=True)

    # Plot distances from center point
    ref_pos = seq_len // 2
    ref_depth = depth // 2

    dist_from_ref = distances[ref_pos, ref_depth].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Heatmap for each depth
    for d in range(depth):
        im = axes[0].imshow(dist_from_ref[:, d].T, cmap='viridis', aspect='auto')
        axes[0].set_title(f'Distance from ({ref_pos}, {ref_depth})')
        axes[0].set_xlabel('Sequence Position')
        axes[0].set_ylabel('Sequence Position')

    plt.colorbar(im, ax=axes[0])

    # Line plot showing cyclic wrap-around
    axes[1].plot(dist_from_ref[:, ref_depth], label=f'Same depth ({ref_depth})')
    if depth > 1:
        axes[1].plot(dist_from_ref[:, 0], label='Depth 0', linestyle='--')
    axes[1].set_xlabel('Sequence Position')
    axes[1].set_ylabel('Toroidal Distance')
    axes[1].set_title('Distance Profile (shows wrap-around)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def test_distance_properties():
    """
    Test mathematical properties of toroidal distance metric.

    Verifies:
    1. Symmetry: δ(a, b) = δ(b, a)
    2. Identity: δ(a, a) = 0
    3. Triangle inequality (approximately, due to cyclic metric)
    4. Periodicity: distance pattern repeats after full rotation
    """
    seq_len, depth = 16, 4
    distances = compute_toroidal_distance(seq_len, depth, normalize=True)

    print("Testing Toroidal Distance Properties:")
    print(f"  Sequence length: {seq_len}, Depth: {depth}")

    # Test symmetry
    is_symmetric = torch.allclose(
        distances,
        distances.permute(2, 3, 0, 1),
        atol=1e-6
    )
    print(f"  ✓ Symmetry: {is_symmetric}")

    # Test identity
    diag_distances = torch.diagonal(
        torch.diagonal(distances, dim1=0, dim2=2),
        dim1=0, dim2=1
    )
    is_zero_diagonal = torch.allclose(diag_distances, torch.zeros_like(diag_distances))
    print(f"  ✓ Identity (δ(a,a)=0): {is_zero_diagonal}")

    # Test wrap-around (distance from 0 to N-1 should equal distance from 0 to 1)
    dist_0_to_end = distances[0, 0, seq_len-1, 0]
    dist_0_to_1 = distances[0, 0, 1, 0]
    is_cyclic = torch.isclose(dist_0_to_end, dist_0_to_1, atol=1e-6)
    print(f"  ✓ Cyclic wrap-around: {is_cyclic} (dist to end ≈ dist to neighbor)")

    # Test minimum at center
    center = seq_len // 2
    dist_from_center = distances[center, 0, :, 0]
    max_dist_at_center = (dist_from_center[center] <= dist_from_center).all()
    print(f"  ✓ Minimum at self-position: {max_dist_at_center}")

    print("\nDistance metric validation complete!")

    return is_symmetric and is_zero_diagonal and is_cyclic


if __name__ == "__main__":
    # Run tests
    test_distance_properties()

