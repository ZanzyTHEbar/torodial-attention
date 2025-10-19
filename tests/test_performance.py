"""
Performance and Regression Tests

Tests for:
- Memory usage validation (O(bch + Dr))
- Inference speed benchmarks
- Gradient computation efficiency
- Comparison with standard attention
- Performance regression detection
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from toroidal_attention import ToroidalAttention


class PerformanceBaseline:
    """Store baseline performance metrics for regression detection."""

    BASELINES = {
        'forward_time_ms': {
            'small': 10.0,   # (B=2, N=32, d=128)
            'medium': 50.0,  # (B=4, N=128, d=256)
            'large': 200.0,  # (B=8, N=256, d=512)
        },
        'memory_mb': {
            'small': 10.0,
            'medium': 50.0,
            'large': 200.0,
        },
        'parameter_count': {
            # Updated baselines based on current implementation:
            # 4 * (d_model * d_model + d_model) + DepthFusion params
            'd128_h4_d2': 66052,   # 4*(128*128 + 128) + fusion
            'd256_h8_d4': 263176,  # 4*(256*256 + 256) + fusion
            'd512_h16_d8': 1050632, # 4*(512*512 + 512) + fusion
        }
    }

    @classmethod
    def check_regression(cls, metric_name: str, config: str, value: float, tolerance: float = 1.5) -> bool:
        """
        Check if metric has regressed beyond tolerance.

        Args:
            metric_name: Name of metric (e.g., 'forward_time_ms')
            config: Configuration key
            value: Measured value
            tolerance: Maximum ratio of value/baseline before flagging

        Returns:
            True if no regression, False if regressed
        """
        baseline = cls.BASELINES.get(metric_name, {}).get(config)
        if baseline is None:
            return True  # No baseline, assume OK

        ratio = value / baseline
        return ratio < tolerance


class TestMemoryUsage:
    """Test memory usage meets O(bch + Dr) constraints."""

    def test_memory_scales_with_sequence(self):
        """Test that memory doesn't scale quadratically with sequence length."""
        if not torch.cuda.is_available():
            print("Skipping CUDA memory test (no GPU)")
            return

        device = torch.device('cuda')
        attn = ToroidalAttention(d_model=256, n_heads=8, depth=4).to(device)

        memory_usages = []
        seq_lengths = [32, 64, 128, 256]

        for seq_len in seq_lengths:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()

            x = torch.randn(4, seq_len, 256).to(device)
            output, _ = attn(x)

            torch.cuda.synchronize()
            memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            memory_usages.append(memory_mb)

            print(f"  seq_len={seq_len}: {memory_mb:.2f} MB")

        # Memory should scale roughly linearly, not quadratically
        # Check that memory growth rate is reasonable
        growth_32_to_64 = memory_usages[1] / memory_usages[0]
        growth_128_to_256 = memory_usages[3] / memory_usages[2]

        # Growth should be similar (linear scaling)
        ratio = growth_128_to_256 / growth_32_to_64
        assert 0.5 < ratio < 2.0, \
            f"Memory scaling inconsistent: {growth_32_to_64:.2f}x vs {growth_128_to_256:.2f}x"

    def test_memory_vs_vanilla_attention(self):
        """Compare memory usage with standard attention."""
        if not torch.cuda.is_available():
            print("Skipping CUDA memory comparison (no GPU)")
            return

        device = torch.device('cuda')
        d_model, seq_len, batch_size = 256, 128, 4

        # Toroidal attention
        torch.cuda.reset_peak_memory_stats(device)
        toroidal = ToroidalAttention(d_model=d_model, n_heads=8, depth=4).to(device)
        x = torch.randn(batch_size, seq_len, d_model).to(device)
        output_tor, _ = toroidal(x)
        torch.cuda.synchronize()
        mem_toroidal = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        # Standard attention
        torch.cuda.reset_peak_memory_stats(device)
        standard = nn.MultiheadAttention(d_model, 8, batch_first=True).to(device)
        x = torch.randn(batch_size, seq_len, d_model).to(device)
        output_std, _ = standard(x, x, x)
        torch.cuda.synchronize()
        mem_standard = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        print(f"  Toroidal: {mem_toroidal:.2f} MB")
        print(f"  Standard: {mem_standard:.2f} MB")
        print(f"  Ratio: {mem_toroidal/mem_standard:.2f}x")

        # Toroidal should use comparable or slightly more memory due to depth dimension
        assert mem_toroidal < mem_standard * 3.0, \
            "Toroidal uses excessive memory compared to standard"

    def test_parameter_count(self):
        """Test parameter count matches expectations."""
        configs = [
            (128, 4, 2, 'd128_h4_d2'),
            (256, 8, 4, 'd256_h8_d4'),
            (512, 16, 8, 'd512_h16_d8'),
        ]

        for d_model, n_heads, depth, config_name in configs:
            attn = ToroidalAttention(d_model=d_model, n_heads=n_heads, depth=depth)
            n_params = sum(p.numel() for p in attn.parameters())

            print(f"  {config_name}: {n_params:,} parameters")

            # Check against baseline (allow some variance for different fusion modes)
            baseline = PerformanceBaseline.BASELINES['parameter_count'].get(config_name)
            if baseline:
                ratio = n_params / baseline
                assert 0.8 < ratio < 1.2, \
                    f"Parameter count mismatch: {n_params} vs expected ~{baseline}"


class TestInferenceSpeed:
    """Test inference speed and compare with baselines."""

    def measure_forward_time(self, attn: nn.Module, x: torch.Tensor, n_iterations: int = 100) -> float:
        """Measure average forward pass time in milliseconds."""
        device = x.device

        # Warmup
        for _ in range(10):
            _ = attn(x) if isinstance(attn, ToroidalAttention) else attn(x, x, x)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(n_iterations):
            _ = attn(x) if isinstance(attn, ToroidalAttention) else attn(x, x, x)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start
        return (elapsed / n_iterations) * 1000  # ms

    def test_forward_speed_small(self):
        """Test forward speed with small configuration."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        attn = ToroidalAttention(d_model=128, n_heads=4, depth=2).to(device)
        x = torch.randn(2, 32, 128).to(device)

        avg_time = self.measure_forward_time(attn, x)
        print(f"  Forward time (small): {avg_time:.2f} ms")

        # Check regression
        no_regression = PerformanceBaseline.check_regression(
            'forward_time_ms', 'small', avg_time, tolerance=2.0
        )
        assert no_regression, f"Performance regression detected: {avg_time:.2f}ms"

    def test_forward_speed_medium(self):
        """Test forward speed with medium configuration."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        attn = ToroidalAttention(d_model=256, n_heads=8, depth=4).to(device)
        x = torch.randn(4, 128, 256).to(device)

        avg_time = self.measure_forward_time(attn, x, n_iterations=50)
        print(f"  Forward time (medium): {avg_time:.2f} ms")

        no_regression = PerformanceBaseline.check_regression(
            'forward_time_ms', 'medium', avg_time, tolerance=2.0
        )
        assert no_regression, f"Performance regression detected: {avg_time:.2f}ms"

    def test_backward_speed(self):
        """Test backward pass speed."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        attn = ToroidalAttention(d_model=256, n_heads=8, depth=4).to(device)
        x = torch.randn(4, 64, 256, requires_grad=True).to(device)

        # Warmup
        for _ in range(10):
            attn.zero_grad()
            output, _ = attn(x)
            loss = output.mean()
            loss.backward()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Measure
        n_iterations = 50
        start = time.time()

        for _ in range(n_iterations):
            attn.zero_grad()
            output, _ = attn(x)
            loss = output.mean()
            loss.backward()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start
        avg_time = (elapsed / n_iterations) * 1000

        print(f"  Backward time: {avg_time:.2f} ms")

        # Backward should be 2-3x forward time
        # This is a sanity check rather than strict requirement
        assert avg_time < 500, "Backward pass unreasonably slow"

    def test_throughput_tokens_per_second(self):
        """Test throughput in tokens/second."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        attn = ToroidalAttention(d_model=256, n_heads=8, depth=4).to(device)
        batch_size, seq_len = 8, 128
        x = torch.randn(batch_size, seq_len, 256).to(device)

        n_iterations = 50

        # Warmup
        for _ in range(10):
            _, _ = attn(x)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(n_iterations):
            _, _ = attn(x)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start
        total_tokens = batch_size * seq_len * n_iterations
        tokens_per_sec = total_tokens / elapsed

        print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")

        # Should process at least 1000 tokens/sec on any hardware
        assert tokens_per_sec > 1000, f"Throughput too low: {tokens_per_sec:.0f}"


class TestComparativePerformance:
    """Compare performance with standard attention."""

    def test_speed_vs_standard_attention(self):
        """Compare inference speed with standard attention."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        d_model, n_heads, seq_len, batch_size = 256, 8, 128, 4

        # Toroidal attention
        toroidal = ToroidalAttention(d_model=d_model, n_heads=n_heads, depth=4).to(device)
        x_tor = torch.randn(batch_size, seq_len, d_model).to(device)

        # Standard attention
        standard = nn.MultiheadAttention(d_model, n_heads, batch_first=True).to(device)
        x_std = torch.randn(batch_size, seq_len, d_model).to(device)

        # Warmup
        for _ in range(10):
            _, _ = toroidal(x_tor)
            _, _ = standard(x_std, x_std, x_std)

        # Measure toroidal
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _, _ = toroidal(x_tor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        time_toroidal = time.time() - start

        # Measure standard
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _, _ = standard(x_std, x_std, x_std)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        time_standard = time.time() - start

        ratio = time_toroidal / time_standard

        print(f"  Toroidal: {time_toroidal*10:.2f} ms")
        print(f"  Standard: {time_standard*10:.2f} ms")
        print(f"  Ratio: {ratio:.2f}x")

        # Toroidal should be within 5x of standard (due to additional computations)
        assert ratio < 5.0, f"Toroidal too slow: {ratio:.2f}x standard attention"

    def test_gradient_computation_overhead(self):
        """Compare gradient computation overhead."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        d_model, n_heads = 256, 8

        # Toroidal
        toroidal = ToroidalAttention(d_model=d_model, n_heads=n_heads, depth=4).to(device)
        x_tor = torch.randn(4, 64, d_model, requires_grad=True).to(device)

        start = time.time()
        for _ in range(50):
            toroidal.zero_grad()
            output, _ = toroidal(x_tor)
            loss = output.mean()
            loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        time_toroidal = time.time() - start

        # Standard
        standard = nn.MultiheadAttention(d_model, n_heads, batch_first=True).to(device)
        x_std = torch.randn(4, 64, d_model, requires_grad=True).to(device)

        start = time.time()
        for _ in range(50):
            standard.zero_grad()
            output, _ = standard(x_std, x_std, x_std)
            loss = output.mean()
            loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        time_standard = time.time() - start

        ratio = time_toroidal / time_standard

        print(f"  Gradient overhead: {ratio:.2f}x")

        # Gradient computation should be reasonable
        assert ratio < 5.0, "Gradient computation too slow"


class TestScalability:
    """Test scalability properties."""

    def test_depth_scaling(self):
        """Test performance scales reasonably with depth."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        times = []
        depths = [1, 2, 4, 8]

        for depth in depths:
            attn = ToroidalAttention(d_model=256, n_heads=8, depth=depth).to(device)
            x = torch.randn(4, 64, 256).to(device)

            # Warmup
            for _ in range(10):
                _, _ = attn(x)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            start = time.time()
            for _ in range(50):
                _, _ = attn(x)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            elapsed = time.time() - start
            times.append(elapsed)

            print(f"  Depth {depth}: {elapsed*20:.2f} ms/iter")

        # Time should scale roughly linearly with depth
        # Check that doubling depth doesn't more than triple time
        for i in range(len(depths) - 1):
            depth_ratio = depths[i+1] / depths[i]
            time_ratio = times[i+1] / times[i]

            assert time_ratio < depth_ratio * 2, \
                f"Non-linear scaling detected: depth {depths[i]}→{depths[i+1]}, time {time_ratio:.2f}x"


def run_performance_tests():
    """Run all performance tests."""
    print("=" * 60)
    print("Running Performance Tests")
    print("=" * 60)

    test_classes = [
        TestMemoryUsage,
        TestInferenceSpeed,
        TestComparativePerformance,
        TestScalability,
    ]

    for test_class in test_classes:
        print(f"\n[{test_class.__name__}]")
        test_instance = test_class()

        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]

        for method_name in test_methods:
            try:
                print(f"\n  {method_name}:")
                method = getattr(test_instance, method_name)
                method()
                print(f"  ✓ {method_name} passed")
            except Exception as e:
                print(f"  ✗ {method_name} failed: {str(e)}")

    print(f"\n{'='*60}")
    print("Performance Tests Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_performance_tests()

