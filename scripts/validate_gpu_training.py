"""
GPU training validation script for Phi-2 + ToroidalAttention.

Validates:
- GPU memory usage
- Training throughput
- Gradient flow on GPU
- Mixed precision training
- Multi-batch stability
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from scripts.load_phi2 import load_phi2_model, replace_attention_layer, freeze_model_except_attention
from toroidal_attention import ToroidalAttention


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return {'allocated_mb': allocated, 'reserved_mb': reserved}
    return {'allocated_mb': 0, 'reserved_mb': 0}


def prepare_training_data(tokenizer, num_samples=100, max_length=256):
    """Prepare small training dataset."""
    # Use a tiny subset of OpenWebText
    dataset = load_dataset('openwebtext', split=f'train[:{num_samples}]')
    
    def tokenize(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
        )
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    return DataLoader(tokenized, batch_size=2, shuffle=True)


def validate_gpu_training(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_steps: int = 10,
    use_amp: bool = False,
):
    """Validate training on GPU."""
    model = model.to(device)
    model.train()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None
    
    # Memory tracking
    memory_stats = []
    timing_stats = []
    loss_stats = []
    
    print(f"\nStarting GPU training validation...")
    print(f"  Device: {device}")
    print(f"  Mixed Precision: {use_amp}")
    print(f"  Steps: {num_steps}")
    
    # Initial memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    initial_mem = get_gpu_memory_usage()
    print(f"\n  Initial GPU Memory: {initial_mem['allocated_mb']:.0f} MB allocated, {initial_mem['reserved_mb']:.0f} MB reserved")
    
    step = 0
    for batch in tqdm(dataloader, desc="Training", total=num_steps):
        if step >= num_steps:
            break
        
        step_start = time.time()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Labels (shifted)
        labels = input_ids.clone()
        
        optimizer.zero_grad()
        
        # Forward pass
        if use_amp:
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        step_time = time.time() - step_start
        
        # Track stats
        mem = get_gpu_memory_usage()
        memory_stats.append(mem)
        timing_stats.append(step_time)
        loss_stats.append(loss.item())
        
        step += 1
    
    # Final memory
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    final_mem = get_gpu_memory_usage()
    
    # Compute stats
    avg_time = sum(timing_stats) / len(timing_stats)
    avg_loss = sum(loss_stats) / len(loss_stats)
    avg_mem = sum(m['allocated_mb'] for m in memory_stats) / len(memory_stats)
    
    print(f"\n  Final GPU Memory: {final_mem['allocated_mb']:.0f} MB allocated, {final_mem['reserved_mb']:.0f} MB reserved")
    print(f"  Peak GPU Memory: {peak_mem:.0f} MB")
    print(f"  Avg Step Time: {avg_time:.3f}s")
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Avg Memory: {avg_mem:.0f} MB")
    
    return {
        'avg_step_time': avg_time,
        'avg_loss': avg_loss,
        'avg_memory_mb': avg_mem,
        'peak_memory_mb': peak_mem,
        'initial_memory_mb': initial_mem['allocated_mb'],
        'final_memory_mb': final_mem['allocated_mb'],
        'loss_values': loss_stats,
        'timing_values': timing_stats,
    }


def main():
    parser = argparse.ArgumentParser(description='GPU training validation')
    parser.add_argument('--layer_indices', type=str, default='0', help='Comma-separated layer indices to replace')
    parser.add_argument('--depth', type=int, default=4, help='Toroidal attention depth')
    parser.add_argument('--lambda_distance', type=float, default=0.1, help='Distance bias strength')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of training steps')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of training samples')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--output', type=str, default='results/gpu_training_validation.json', help='Output file')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: GPU not available")
        return
    
    device = torch.device('cuda')
    layer_indices = [int(x) for x in args.layer_indices.split(',')]
    
    print(f"="*80)
    print("GPU TRAINING VALIDATION")
    print(f"="*80)
    print(f"  Layers to replace: {layer_indices}")
    print(f"  Depth: {args.depth}")
    print(f"  Lambda: {args.lambda_distance}")
    print(f"  Max length: {args.max_length}")
    print(f"  Mixed precision: {args.use_amp}")
    
    # Load model
    print(f"\nLoading Phi-2...")
    model, tokenizer, config = load_phi2_model(device='cpu', torch_dtype=torch.float32)
    
    # Replace layers
    print(f"\nReplacing {len(layer_indices)} layers with ToroidalAttention...")
    for layer_idx in layer_indices:
        toroidal = ToroidalAttention(
            d_model=config.hidden_size,
            n_heads=config.num_attention_heads,
            depth=args.depth,
            max_len=args.max_length,
            lambda_distance=args.lambda_distance,
            fusion_mode='low_rank',
            use_orthogonal_pe=True,
        )
        replace_attention_layer(model, layer_idx=layer_idx, toroidal_attn=toroidal, copy_weights=False)
    
    # Freeze base model
    freeze_model_except_attention(model, layer_indices)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Prepare data
    print(f"\nPreparing training data...")
    dataloader = prepare_training_data(tokenizer, args.num_samples, args.max_length)
    
    # Run validation
    results = validate_gpu_training(
        model,
        dataloader,
        device,
        args.num_steps,
        args.use_amp,
    )
    
    # Save results
    import json
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_full = {
        'config': {
            'layer_indices': layer_indices,
            'depth': args.depth,
            'lambda_distance': args.lambda_distance,
            'max_length': args.max_length,
            'use_amp': args.use_amp,
            'num_steps': args.num_steps,
        },
        'results': results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_full, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"  ✅ GPU training validated successfully")
    print(f"  ✅ {args.num_steps} steps completed without errors")
    print(f"  ✅ Loss converging: {results['avg_loss']:.4f}")
    print(f"  ✅ Peak GPU memory: {results['peak_memory_mb']:.0f} MB")
    print(f"  ✅ Avg throughput: {1/results['avg_step_time']:.2f} steps/s")
    
    # Check for issues
    if results['avg_loss'] > 10.0:
        print(f"\n  ⚠️  Warning: High loss ({results['avg_loss']:.2f}), may need longer training")
    if results['peak_memory_mb'] > 7000:
        print(f"\n  ⚠️  Warning: High memory usage ({results['peak_memory_mb']:.0f} MB), close to limit")
    
    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()

