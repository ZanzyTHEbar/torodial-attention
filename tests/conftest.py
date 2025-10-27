"""
Pytest configuration and fixtures for reproducible testing.

Provides:
- Deterministic seeds for torch, numpy, random
- Device fixtures (CPU/GPU auto-selection)
- Common test configurations
"""

import pytest
import torch
import numpy as np
import random


@pytest.fixture(scope="session", autouse=True)
def deterministic_seeds():
    """Set deterministic seeds for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    yield
    
    # Cleanup (optional)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@pytest.fixture(scope="session")
def device():
    """Return available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def cpu_device():
    """Force CPU device."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def gpu_device():
    """Return GPU device if available, skip test otherwise."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    return torch.device("cuda")


@pytest.fixture
def default_config():
    """Default test configuration."""
    return {
        'd_model': 128,
        'n_heads': 4,
        'depth': 2,
        'max_len': 256,
        'lambda_distance': 0.1,
        'fusion_mode': 'low_rank',
        'fusion_rank': 1,
        'use_orthogonal_pe': True,
    }


@pytest.fixture
def small_config():
    """Small configuration for quick tests."""
    return {
        'd_model': 64,
        'n_heads': 2,
        'depth': 2,
        'max_len': 64,
        'lambda_distance': 0.1,
        'fusion_mode': 'low_rank',
        'fusion_rank': 1,
        'use_orthogonal_pe': True,
    }


@pytest.fixture
def large_config():
    """Large configuration for stress tests."""
    return {
        'd_model': 512,
        'n_heads': 8,
        'depth': 4,
        'max_len': 1024,
        'lambda_distance': 0.1,
        'fusion_mode': 'low_rank',
        'fusion_rank': 1,
        'use_orthogonal_pe': True,
    }


@pytest.fixture
def phi2_config():
    """Phi-2 compatible configuration."""
    return {
        'd_model': 2560,
        'n_heads': 32,
        'depth': 4,
        'max_len': 2048,
        'lambda_distance': 0.1,
        'fusion_mode': 'low_rank',
        'fusion_rank': 1,
        'use_orthogonal_pe': True,
    }


@pytest.fixture(params=[
    {'depth': 1, 'seq_len': 128},
    {'depth': 2, 'seq_len': 256},
    {'depth': 4, 'seq_len': 512},
])
def depth_seq_sweep(request):
    """Parametrize tests across depth and sequence length combinations."""
    return request.param


@pytest.fixture(params=['low_rank', 'attention', 'mean'])
def fusion_mode(request):
    """Parametrize tests across fusion modes."""
    return request.param


@pytest.fixture(params=[True, False])
def use_orthogonal_pe(request):
    """Parametrize tests for orthogonal vs standard PE."""
    return request.param


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "phi2: mark test as requiring Phi-2 model"
    )
    config.addinivalue_line(
        "markers", "flash2: mark test as requiring FlashAttention v2"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests based on markers and environment."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    skip_phi2 = pytest.mark.skip(reason="Phi-2 model not downloaded (set RUN_PHI2_TESTS=1)")
    skip_flash2 = pytest.mark.skip(reason="FlashAttention v2 not available")
    
    import os
    run_phi2 = os.environ.get('RUN_PHI2_TESTS', '0') == '1'
    
    for item in items:
        # Skip GPU tests if no GPU
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
        
        # Skip Phi-2 tests unless explicitly enabled
        if "phi2" in item.keywords and not run_phi2:
            item.add_marker(skip_phi2)
        
        # Skip Flash2 tests if not available
        if "flash2" in item.keywords:
            try:
                from toroidal_attention.backends import has_flash2
                if not has_flash2():
                    item.add_marker(skip_flash2)
            except:
                item.add_marker(skip_flash2)

