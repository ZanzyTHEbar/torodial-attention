import torch

from toroidal_attention import compute_toroidal_distance


def test_distance_symmetry_and_identity():
    N, D = 16, 4
    dist = compute_toroidal_distance(N, D, normalize=True)
    assert torch.allclose(dist, dist.permute(2, 3, 0, 1))
    diag = torch.diagonal(torch.diagonal(dist, dim1=0, dim2=2), dim1=0, dim2=1)
    assert torch.allclose(diag, torch.zeros_like(diag))


def test_distance_wraparound():
    N, D = 16, 2
    dist = compute_toroidal_distance(N, D, normalize=True)
    assert torch.isclose(dist[0, 0, N-1, 0], dist[0, 0, 1, 0])


