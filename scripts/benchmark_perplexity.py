"""
Perplexity comparison benchmark for Phi-2 with/without toroidal attention.

Compares:
- Baseline Phi-2
- Phi-2 + ToroidalAttention (single layer)
- Phi-2 + ToroidalAttention (multi-layer)
- Phi-2 + ToroidalAttention (full replacement)

Datasets:
- WikiText-2
- WikiText-103
- OpenWebText (subset)
- Custom long-context datasets
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

from scripts.load_phi2 import load_phi2_model, replace_attention_layer
from toroidal_attention import ToroidalAttention


def compute_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute perplexity on a dataset.
    
    Returns:
        dict with perplexity, loss, and timing metrics
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Computing perplexity")):
            if max_batches and i >= max_batches:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Shift for autoregressive loss
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100  # Ignore last token
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss
            
            # Count valid tokens (not padding or ignored)
            valid_tokens = (labels != -100).sum().item()
            
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
    
    end_time = time.time()
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    time_elapsed = end_time - start_time
    tokens_per_second = total_tokens / time_elapsed
    
    return {
        'perplexity': float(perplexity),
        'loss': float(avg_loss),
        'total_tokens': int(total_tokens),
        'time_seconds': float(time_elapsed),
        'tokens_per_second': float(tokens_per_second),
    }


def load_dataset_for_eval(
    dataset_name: str,
    tokenizer,
    max_length: int = 512,
    num_samples: Optional[int] = None,
) -> DataLoader:
    """Load and tokenize dataset for evaluation."""
    
    if dataset_name == 'wikitext2':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    elif dataset_name == 'wikitext103':
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
    elif dataset_name == 'openwebtext':
        dataset = load_dataset('openwebtext', split='train[:1%]')  # Small subset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Limit samples if specified
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
        )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    dataloader = DataLoader(tokenized, batch_size=1, shuffle=False)
    
    return dataloader


def create_toroidal_config(
    base_config,
    depth: int,
    lambda_distance: float,
    fusion_mode: str = 'low_rank',
    max_len: int = 2048,
) -> ToroidalAttention:
    """Create toroidal attention configuration."""
    return ToroidalAttention(
        d_model=base_config.hidden_size,
        n_heads=base_config.num_attention_heads,
        depth=depth,
        max_len=max_len,
        lambda_distance=lambda_distance,
        fusion_mode=fusion_mode,
        use_orthogonal_pe=True,
    )


def run_baseline_benchmark(
    model,
    tokenizer,
    config,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict:
    """Run baseline Phi-2 benchmark."""
    print("\n" + "="*80)
    print("BASELINE: Standard Phi-2")
    print("="*80)
    
    # Keep model on CPU if device is CUDA (Phi-2 is too large)
    # Use CPU for baseline to save GPU memory
    actual_device = torch.device('cpu') if device.type == 'cuda' else device
    print(f"  Running baseline on: {actual_device} (model too large for GPU)")
    model = model.to(actual_device)
    results = compute_perplexity(model, dataloader, actual_device, max_batches)
    
    results['config'] = {
        'model': 'phi-2',
        'variant': 'baseline',
        'num_layers': config.num_hidden_layers,
        'hidden_size': config.hidden_size,
        'num_heads': config.num_attention_heads,
    }
    
    return results


def run_toroidal_benchmark(
    model,
    tokenizer,
    config,
    dataloader: DataLoader,
    device: torch.device,
    layer_indices: List[int],
    depth: int,
    lambda_distance: float,
    max_batches: Optional[int] = None,
) -> Dict:
    """Run Phi-2 + ToroidalAttention benchmark."""
    variant_name = f"{len(layer_indices)}layer_d{depth}_l{lambda_distance}"
    print("\n" + "="*80)
    print(f"TOROIDAL: {variant_name}")
    print(f"  Layers: {layer_indices}")
    print(f"  Depth: {depth}")
    print(f"  Lambda: {lambda_distance}")
    print("="*80)
    
    # Replace layers
    for layer_idx in layer_indices:
        toroidal = create_toroidal_config(
            config,
            depth=depth,
            lambda_distance=lambda_distance,
        )
        replace_attention_layer(model, layer_idx=layer_idx, toroidal_attn=toroidal, copy_weights=False)
    
    # Use CPU for all benchmarks (model + toroidal layers too large for 8GB GPU)
    actual_device = torch.device('cpu') if device.type == 'cuda' else device
    print(f"  Running on: {actual_device} (model too large for GPU)")
    model = model.to(actual_device)
    results = compute_perplexity(model, dataloader, actual_device, max_batches)
    
    results['config'] = {
        'model': 'phi-2',
        'variant': f'toroidal_{variant_name}',
        'replaced_layers': layer_indices,
        'depth': depth,
        'lambda_distance': lambda_distance,
        'num_replaced_layers': len(layer_indices),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Perplexity comparison benchmark')
    parser.add_argument('--dataset', type=str, default='wikitext2', choices=['wikitext2', 'wikitext103', 'openwebtext'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to evaluate')
    parser.add_argument('--max_batches', type=int, default=None, help='Max batches per config')
    parser.add_argument('--output', type=str, default='results/perplexity_comparison.json', help='Output file')
    parser.add_argument('--skip_baseline', action='store_true', help='Skip baseline benchmark')
    parser.add_argument('--depths', type=str, default='1,2,4,8', help='Comma-separated depth values to test')
    parser.add_argument('--lambdas', type=str, default='0.0,0.1,0.2', help='Comma-separated lambda values to test')
    
    args = parser.parse_args()
    
    # Parse depths and lambdas
    depths = [int(d) for d in args.depths.split(',')]
    lambdas = [float(l) for l in args.lambdas.split(',')]
    
    print(f"Loading model and dataset...")
    model, tokenizer, config = load_phi2_model(device='cpu', torch_dtype=torch.float32)
    dataloader = load_dataset_for_eval(args.dataset, tokenizer, args.max_length, args.num_samples)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    results = []
    
    # Baseline
    if not args.skip_baseline:
        baseline_results = run_baseline_benchmark(
            model, tokenizer, config, dataloader, device, args.max_batches
        )
        results.append(baseline_results)
        print(f"\nBaseline Perplexity: {baseline_results['perplexity']:.2f}")
        print(f"Baseline Loss: {baseline_results['loss']:.4f}")
        print(f"Throughput: {baseline_results['tokens_per_second']:.0f} tokens/s")
    
    # Single-layer replacement (layer 0)
    for depth in depths:
        for lambda_dist in lambdas:
            # Reload model for each config
            model, _, _ = load_phi2_model(device='cpu', torch_dtype=torch.float32)
            
            single_results = run_toroidal_benchmark(
                model, tokenizer, config, dataloader, device,
                layer_indices=[0],
                depth=depth,
                lambda_distance=lambda_dist,
                max_batches=args.max_batches,
            )
            results.append(single_results)
            print(f"\nToroidal (1 layer, d={depth}, λ={lambda_dist}) Perplexity: {single_results['perplexity']:.2f}")
            print(f"Loss: {single_results['loss']:.4f}")
            print(f"Throughput: {single_results['tokens_per_second']:.0f} tokens/s")
    
    # Multi-layer replacement (4 layers: 0, 8, 16, 24)
    for depth in depths[:2]:  # Only test depths 1,2 for multi-layer (memory)
        for lambda_dist in lambdas[:2]:  # Only test lambdas 0.0, 0.1
            model, _, _ = load_phi2_model(device='cpu', torch_dtype=torch.float32)
            
            multi_results = run_toroidal_benchmark(
                model, tokenizer, config, dataloader, device,
                layer_indices=[0, 8, 16, 24],
                depth=depth,
                lambda_distance=lambda_dist,
                max_batches=args.max_batches,
            )
            results.append(multi_results)
            print(f"\nToroidal (4 layers, d={depth}, λ={lambda_dist}) Perplexity: {multi_results['perplexity']:.2f}")
            print(f"Loss: {multi_results['loss']:.4f}")
            print(f"Throughput: {multi_results['tokens_per_second']:.0f} tokens/s")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Config':<40} {'Perplexity':>12} {'Loss':>10} {'Tokens/s':>12}")
    print("-"*80)
    for result in results:
        variant = result['config']['variant']
        ppl = result['perplexity']
        loss = result['loss']
        tps = result['tokens_per_second']
        print(f"{variant:<40} {ppl:>12.2f} {loss:>10.4f} {tps:>12.0f}")
    
    # Compute improvements vs baseline
    if not args.skip_baseline:
        baseline_ppl = results[0]['perplexity']
        print("\n" + "="*80)
        print("IMPROVEMENTS vs BASELINE")
        print("="*80)
        print(f"{'Config':<40} {'Δ Perplexity':>15} {'% Change':>12}")
        print("-"*80)
        for result in results[1:]:
            variant = result['config']['variant']
            ppl = result['perplexity']
            delta = baseline_ppl - ppl
            pct = (delta / baseline_ppl) * 100
            print(f"{variant:<40} {delta:>15.2f} {pct:>11.1f}%")


if __name__ == '__main__':
    main()

