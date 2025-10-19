#!/usr/bin/env python3
"""
Benchmark different attention backends.

Usage:
    python scripts/benchmark_backends.py --device cuda --batch 8 --seq 256 --iters 100
    ENABLE_FLASH2=1 python scripts/benchmark_backends.py --device cuda --out bench_gpu.json
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

from toroidal_attention import ToroidalAttention


def bench(model, x, iters=20, device='cpu'):
    """Benchmark a model with detailed metrics."""
    if x.is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(x)[0]

    if x.is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            model(x)[0]
    if x.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    batch, seq = x.shape[:2]
    total_tokens = batch * seq * iters
    tokens_per_sec = total_tokens / elapsed
    ms_per_iter = (elapsed / iters) * 1000
    seqs_per_sec = (batch * iters) / elapsed

    result = {
        'total_seconds': elapsed,
        'ms_per_iter': ms_per_iter,
        'tokens_per_sec': tokens_per_sec,
        'seqs_per_sec': seqs_per_sec,
        'total_tokens': total_tokens,
    }

    if x.is_cuda:
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        result['peak_memory_mb'] = peak_mem_mb

    return result


def main():
    parser = argparse.ArgumentParser(description='Benchmark TAM backends')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--seq', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--out', type=str, default='bench_results.json')
    args = parser.parse_args()

    device = torch.device(args.device)
    x = torch.randn(args.batch, args.seq, args.d_model, device=device)

    configs = [
        ('sdpa', dict(backend='sdpa', lambda_distance=0.0)),
        ('manual', dict(backend='sdpa', lambda_distance=0.0, attn_chunk_size=64)),
    ]
    if os.environ.get('ENABLE_FLASH2', '0') == '1':
        configs.append(('flash2', dict(backend='flash2', lambda_distance=0.0)))

    results = {
        'config': {
            'device': args.device,
            'batch': args.batch,
            'seq': args.seq,
            'd_model': args.d_model,
            'heads': args.heads,
            'depth': args.depth,
            'iters': args.iters,
        },
        'backends': {}
    }

    for name, kwargs in configs:
        print(f"Benchmarking {name}...")
        model = ToroidalAttention(
            d_model=args.d_model,
            n_heads=args.heads,
            max_len=args.seq,
            depth=args.depth,
            **kwargs
        ).to(device)
        results['backends'][name] = bench(model, x, iters=args.iters, device=args.device)
        print(f"  {results['backends'][name]['ms_per_iter']:.2f} ms/iter, "
              f"{results['backends'][name]['seqs_per_sec']:.1f} seq/s")

    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(json.dumps(results, indent=2))

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {args.out}")


if __name__ == '__main__':
    main()
