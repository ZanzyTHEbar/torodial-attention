#!/usr/bin/env python
"""
Comprehensive benchmark script for toroidal attention backends.

Benchmarks:
- CPU vs GPU performance
- SDPA vs Flash2 (when available)
- Different depths and sequence lengths
- Memory usage tracking

Output: JSON with metrics for analysis and reporting.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from toroidal_attention import ToroidalAttention


def get_device_info(device):
    """Get device information."""
    if device.type == 'cuda':
        return {
            'device': 'cuda',
            'name': torch.cuda.get_device_name(0),
            'capability': torch.cuda.get_device_capability(0),
            'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
        }
    else:
        return {
            'device': 'cpu',
            'name': 'CPU',
        }


def benchmark_forward(attn, x, n_iters=10, warmup=3):
    """Benchmark forward pass."""
    device = x.device
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = attn(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
    
    # Benchmark
    times = []
    start = time.perf_counter()
    
    for _ in range(n_iters):
        iter_start = time.perf_counter()
        with torch.no_grad():
            _ = attn(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - iter_start)
    
    total_time = time.perf_counter() - start
    
    # Memory
    peak_mem_mb = 0
    if device.type == 'cuda':
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
    
    return {
        'total_time_s': total_time,
        'avg_time_s': sum(times) / len(times),
        'min_time_s': min(times),
        'max_time_s': max(times),
        'std_time_s': torch.tensor(times).std().item(),
        'ms_per_iter': (sum(times) / len(times)) * 1000,
        'peak_memory_mb': peak_mem_mb,
        'n_iters': n_iters,
    }


def benchmark_config(d_model, n_heads, depth, seq_len, batch_size, device, backend='sdpa',
                     lambda_distance=0.1, window_size=None):
    """Benchmark a specific configuration."""
    print(f"  Config: depth={depth}, seq={seq_len}, batch={batch_size}, backend={backend}")
    
    try:
        attn = ToroidalAttention(
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            max_len=seq_len * 2,
            lambda_distance=lambda_distance,
            window_size=window_size,
            backend=backend,
            use_orthogonal_pe=True,
        ).to(device)
        
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        # Run benchmark
        results = benchmark_forward(attn, x)
        
        # Calculate throughput
        total_tokens = batch_size * seq_len * results['n_iters']
        results['seqs_per_sec'] = results['n_iters'] / results['total_time_s']
        results['tokens_per_sec'] = total_tokens / results['total_time_s']
        
        # Config info
        n_params = sum(p.numel() for p in attn.parameters())
        results['n_params'] = n_params
        results['config'] = {
            'd_model': d_model,
            'n_heads': n_heads,
            'depth': depth,
            'seq_len': seq_len,
            'batch_size': batch_size,
            'backend': backend,
            'lambda_distance': lambda_distance,
            'window_size': window_size,
        }
        
        print(f"    ✓ {results['ms_per_iter']:.2f} ms/iter, "
              f"{results['tokens_per_sec']:.0f} tok/s, "
              f"{results['peak_memory_mb']:.0f} MB peak")
        
        return results
        
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        return None


def run_benchmarks(device_str, output_path):
    """Run comprehensive benchmark suite."""
    device = torch.device(device_str)
    print(f"Benchmarking on device: {device}")
    print(f"Device info: {get_device_info(device)}")
    
    # Fixed model config
    d_model = 512
    n_heads = 8
    
    # Sweep parameters
    depths = [1, 2, 4, 8]
    seq_lens = [128, 256, 512]
    batch_size = 4
    
    # Backends to test
    backends = ['sdpa']
    if device.type == 'cuda':
        try:
            from toroidal_attention.backends import has_flash2
            if has_flash2():
                backends.append('flash2')
                print("Flash2 available")
        except:
            pass
    
    results = {
        'device_info': get_device_info(device),
        'benchmarks': [],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    print("\nRunning benchmarks...")
    
    for depth in depths:
        for seq_len in seq_lens:
            for backend in backends:
                # SDPA with distance bias
                if backend == 'sdpa':
                    res = benchmark_config(
                        d_model, n_heads, depth, seq_len, batch_size,
                        device, backend='sdpa', lambda_distance=0.1
                    )
                    if res:
                        results['benchmarks'].append(res)
                
                # Flash2 (no bias/window to enable it)
                if backend == 'flash2':
                    res = benchmark_config(
                        d_model, n_heads, depth, seq_len, batch_size,
                        device, backend='flash2', lambda_distance=0.0, window_size=None
                    )
                    if res:
                        results['benchmarks'].append(res)
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Benchmarks complete. Results saved to: {output_path}")
    print(f"Total configurations tested: {len(results['benchmarks'])}")
    
    # Print summary
    print("\nSummary:")
    for bench in results['benchmarks']:
        cfg = bench['config']
        print(f"  depth={cfg['depth']}, seq={cfg['seq_len']}, backend={cfg['backend']}: "
              f"{bench['ms_per_iter']:.1f} ms/iter, {bench['peak_memory_mb']:.0f} MB")


def main():
    parser = argparse.ArgumentParser(description='Benchmark toroidal attention backends')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to benchmark on')
    parser.add_argument('--out', type=str, default='results/benchmark.json',
                       help='Output JSON path')
    
    args = parser.parse_args()
    
    output_path = Path(args.out)
    
    try:
        run_benchmarks(args.device, output_path)
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
