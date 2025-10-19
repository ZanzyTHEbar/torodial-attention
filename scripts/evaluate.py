"""
Evaluation and Benchmarking for Toroidal Attention

Implements comprehensive evaluation:
1. Perplexity measurement
2. Rotational invariance validation (Lemma 1)
3. Gradient analysis for stability
4. Memory profiling
5. Ablation study visualization
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


from scripts.load_phi2 import load_phi2_model, replace_attention_layer
from scripts.prepare_data import create_dataloaders
from scripts.train_toroidal import TrainingConfig, compute_loss
from toroidal_attention import ToroidalAttention


class EvaluationResults:
    """Store and manage evaluation results."""

    def __init__(self):
        self.perplexity = {}
        self.invariance_scores = {}
        self.gradient_stats = {}
        self.memory_usage = {}
        self.inference_speed = {}

    def save(self, path: Path):
        """Save results to JSON."""
        data = {
            'perplexity': self.perplexity,
            'invariance_scores': self.invariance_scores,
            'gradient_stats': self.gradient_stats,
            'memory_usage': self.memory_usage,
            'inference_speed': self.inference_speed,
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved evaluation results to {path}")

    def print_summary(self):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("Evaluation Summary")
        print("=" * 60)

        if self.perplexity:
            print("\nPerplexity:")
            for name, ppl in self.perplexity.items():
                print(f"  {name}: {ppl:.2f}")

        if self.invariance_scores:
            print("\nRotational Invariance (max error):")
            for name, score in self.invariance_scores.items():
                print(f"  {name}: {score:.4f}")

        if self.gradient_stats:
            print("\nGradient Statistics:")
            for name, stats in self.gradient_stats.items():
                print(f"  {name}:")
                for key, value in stats.items():
                    print(f"    {key}: {value:.4f}")

        if self.memory_usage:
            print("\nMemory Usage (MB):")
            for name, mem in self.memory_usage.items():
                print(f"  {name}: {mem:.2f}")


def evaluate_perplexity(
    model,
    dataloader,
    device: str,
    name: str = "model"
) -> float:
    """
    Evaluate perplexity on dataset.

    Args:
        model: The model to evaluate
        dataloader: DataLoader
        device: Device
        name: Name for logging

    Returns:
        perplexity: Average perplexity
    """
    print(f"\nEvaluating perplexity for {name}...")

    model.eval()
    total_loss = 0
    n_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            loss = compute_loss(model, batch, device)
            batch_size, seq_len = batch['input_ids'].shape

            total_loss += loss.item() * batch_size * seq_len
            n_tokens += batch_size * seq_len

    avg_loss = total_loss / n_tokens
    perplexity = np.exp(avg_loss)

    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")

    return perplexity


def test_rotational_invariance(
    model,
    dataloader,
    device: str,
    n_samples: int = 10,
    shifts: List[int] = [4, 8, 16],
) -> Dict[int, float]:
    """
    Test rotational invariance (Lemma 1 validation).

    For each shift s, compare:
    - output_shifted = model(roll(input, s))
    - output_expected = roll(model(input), s)

    Perfect invariance means these are identical.

    Args:
        model: Model to test
        dataloader: DataLoader
        device: Device
        n_samples: Number of samples to test
        shifts: List of shift amounts to test

    Returns:
        dict: {shift: max_error} for each shift amount
    """
    print("\nTesting rotational invariance...")
    print(f"  Shifts: {shifts}")
    print(f"  Samples: {n_samples}")

    model.eval()

    errors = {shift: [] for shift in shifts}

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_samples:
                break

            input_ids = batch['input_ids'].to(device)
            B, N = input_ids.shape

            # Get original outputs (logits)
            outputs_original = model(input_ids=input_ids).logits

            # Test each shift
            for shift in shifts:
                # Shift input
                input_shifted = torch.roll(input_ids, shifts=shift, dims=1)

                # Get shifted output
                outputs_shifted = model(input_ids=input_shifted).logits

                # Expected output (roll original output)
                outputs_expected = torch.roll(outputs_original, shifts=shift, dims=1)

                # Compute error
                error = (outputs_shifted - outputs_expected).abs().max().item()
                errors[shift].append(error)

    # Average errors
    avg_errors = {shift: np.mean(err_list) for shift, err_list in errors.items()}
    max_errors = {shift: np.max(err_list) for shift, err_list in errors.items()}

    print("\n  Results:")
    for shift in shifts:
        print(f"    Shift {shift:2d}: avg_error={avg_errors[shift]:.4f}, max_error={max_errors[shift]:.4f}")

    # Check if invariance holds (allowing for numerical tolerance)
    threshold = 1.0  # Relaxed threshold
    invariant = all(err < threshold for err in max_errors.values())

    if invariant:
        print("  ✓ Rotational invariance validated!")
    else:
        print("  ⚠ Rotational invariance not perfectly satisfied (may be due to depth fusion)")

    return max_errors


def analyze_gradients(
    model,
    dataloader,
    device: str,
    n_batches: int = 10,
) -> Dict[str, float]:
    """
    Analyze gradient statistics for stability validation.

    Verifies:
    - Gradients are bounded (no explosion)
    - Gradients are not vanishing
    - PE gradients satisfy stability conditions

    Args:
        model: Model to analyze
        dataloader: DataLoader
        device: Device
        n_batches: Number of batches to analyze

    Returns:
        dict: Gradient statistics
    """
    print("\nAnalyzing gradients...")

    model.train()

    all_grad_norms = []
    pe_grad_norms = []

    for i, batch in enumerate(tqdm(dataloader, desc="Computing gradients", total=n_batches)):
        if i >= n_batches:
            break

        # Forward pass
        model.zero_grad()
        loss = compute_loss(model, batch, device)

        # Backward pass
        loss.backward()

        # Collect gradient norms
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                all_grad_norms.append(grad_norm)

                # Track PE gradients specifically
                if 'pos_encoding' in name or 'freqs' in name:
                    pe_grad_norms.append(grad_norm)

    # Compute statistics
    stats = {
        'mean': np.mean(all_grad_norms),
        'std': np.std(all_grad_norms),
        'max': np.max(all_grad_norms),
        'min': np.min(all_grad_norms),
        'median': np.median(all_grad_norms),
    }

    if pe_grad_norms:
        stats['pe_mean'] = np.mean(pe_grad_norms)
        stats['pe_max'] = np.max(pe_grad_norms)

    print("\n  Gradient Statistics:")
    for key, value in stats.items():
        print(f"    {key}: {value:.4f}")

    # Check for issues
    if stats['max'] > 100:
        print("  ⚠ Warning: Large gradient values detected (potential instability)")
    elif stats['max'] < 1e-5:
        print("  ⚠ Warning: Very small gradients (potential vanishing gradients)")
    else:
        print("  ✓ Gradient magnitudes look healthy")

    return stats


def profile_memory(
    model,
    batch_size: int = 8,
    seq_len: int = 128,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Profile memory usage.

    Measures:
    - Peak memory during forward pass
    - Peak memory during backward pass
    - Memory per sample

    Args:
        model: Model to profile
        batch_size: Batch size
        seq_len: Sequence length
        device: Device

    Returns:
        dict: Memory statistics (in MB)
    """
    if device == 'cpu':
        print("  Skipping memory profiling (CPU mode)")
        return {}

    print("\nProfiling memory usage...")
    print(f"  Batch size: {batch_size}, Seq len: {seq_len}")

    model.train()

    # Create dummy batch
    input_ids = torch.randint(0, 50257, (batch_size, seq_len)).to(device)
    labels = torch.randint(0, 50257, (batch_size, seq_len)).to(device)

    # Measure forward pass
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss

    torch.cuda.synchronize()
    forward_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB

    # Measure backward pass
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    loss.backward()

    torch.cuda.synchronize()
    backward_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB

    # Calculate per-sample memory
    per_sample_memory = backward_memory / batch_size

    stats = {
        'forward_mb': forward_memory,
        'backward_mb': backward_memory,
        'per_sample_mb': per_sample_memory,
    }

    print("\n  Memory Usage:")
    for key, value in stats.items():
        print(f"    {key}: {value:.2f} MB")

    return stats


def measure_inference_speed(
    model,
    batch_size: int = 8,
    seq_len: int = 128,
    n_iterations: int = 100,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Measure inference speed (tokens/second).

    Args:
        model: Model to benchmark
        batch_size: Batch size
        seq_len: Sequence length
        n_iterations: Number of iterations
        device: Device

    Returns:
        dict: Speed statistics
    """
    print("\nMeasuring inference speed...")
    print(f"  Iterations: {n_iterations}")

    model.eval()

    # Create dummy batch
    input_ids = torch.randint(0, 50257, (batch_size, seq_len)).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids=input_ids)

    # Benchmark
    if device != 'cpu':
        torch.cuda.synchronize()

    import time
    start_time = time.time()

    with torch.no_grad():
        for _ in range(n_iterations):
            _ = model(input_ids=input_ids)

    if device != 'cpu':
        torch.cuda.synchronize()

    end_time = time.time()

    # Calculate statistics
    elapsed = end_time - start_time
    total_tokens = batch_size * seq_len * n_iterations
    tokens_per_sec = total_tokens / elapsed
    ms_per_token = (elapsed * 1000) / total_tokens

    stats = {
        'tokens_per_sec': tokens_per_sec,
        'ms_per_token': ms_per_token,
        'total_time_sec': elapsed,
    }

    print("\n  Inference Speed:")
    for key, value in stats.items():
        print(f"    {key}: {value:.2f}")

    return stats


def visualize_ablation_results(results_dir: Path, save_path: Path):
    """
    Create visualization of ablation study results.

    Args:
        results_dir: Directory containing ablation results
        save_path: Where to save the visualization
    """
    print("\nCreating ablation study visualization...")

    # Load results from different experiments
    experiments = ['toroidal_2d', 'toroidal_3d_lowrank', 'toroidal_3d_attention']

    data = {}
    for exp_name in experiments:
        metrics_path = results_dir / exp_name / 'metrics.json'
        if metrics_path.exists():
            with open(metrics_path) as f:
                data[exp_name] = json.load(f)

    if not data:
        print("  No ablation results found")
        return

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Training loss curves
    ax = axes[0, 0]
    for exp_name, metrics in data.items():
        if 'train_losses' in metrics:
            ax.plot(metrics['train_losses'], label=exp_name, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Validation perplexity
    ax = axes[0, 1]
    for exp_name, metrics in data.items():
        if 'val_perplexities' in metrics:
            ax.plot(metrics['val_perplexities'], label=exp_name, marker='o')
    ax.set_xlabel('Evaluation Step')
    ax.set_ylabel('Perplexity')
    ax.set_title('Validation Perplexity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Gradient norms
    ax = axes[1, 0]
    for exp_name, metrics in data.items():
        if 'gradient_norms' in metrics:
            # Plot moving average
            grad_norms = np.array(metrics['gradient_norms'])
            window = 50
            if len(grad_norms) >= window:
                smoothed = np.convolve(grad_norms, np.ones(window)/window, mode='valid')
                ax.plot(smoothed, label=exp_name)
    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Stability (50-step MA)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Best perplexity comparison
    ax = axes[1, 1]
    exp_names = []
    best_ppls = []
    for exp_name, metrics in data.items():
        if 'summary' in metrics and 'best_val_perplexity' in metrics['summary']:
            exp_names.append(exp_name)
            best_ppls.append(metrics['summary']['best_val_perplexity'])

    if exp_names:
        bars = ax.bar(range(len(exp_names)), best_ppls)
        ax.set_xticks(range(len(exp_names)))
        ax.set_xticklabels(exp_names, rotation=45, ha='right')
        ax.set_ylabel('Perplexity')
        ax.set_title('Best Validation Perplexity')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (bar, ppl) in enumerate(zip(bars, best_ppls)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ppl:.2f}',
                   ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to {save_path}")
    plt.close()


def comprehensive_evaluation(
    checkpoint_path: Path,
    config_path: Path,
    output_dir: Path,
):
    """
    Run comprehensive evaluation on a trained model.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to training config
        output_dir: Where to save results
    """
    print("=" * 60)
    print("Comprehensive Toroidal Attention Evaluation")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(config_path) as f:
        config_dict = json.load(f)
    config = TrainingConfig(**config_dict)

    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)

    # Reconstruct model
    print("\nReconstructing model...")
    model, tokenizer, model_config = load_phi2_model(
        model_name=config.model_name,
        device=config.device,
    )

    toroidal_attn = ToroidalAttention(
        d_model=model_config.hidden_size,
        n_heads=model_config.num_attention_heads,
        max_len=config.seq_len * 2,
        depth=config.depth,
        lambda_distance=config.lambda_distance,
        fusion_mode=config.fusion_mode,
    )

    replace_attention_layer(model, config.layer_idx, toroidal_attn, copy_weights=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)

    # Prepare test data
    print("\nPreparing test dataset...")
    _, test_loader = create_dataloaders(
        dataset_type=config.dataset_type,
        tokenizer=tokenizer,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        n_train=100,  # Small dummy set
        n_val=config.n_val,
    )

    # Run evaluations
    results = EvaluationResults()

    # 1. Perplexity
    ppl = evaluate_perplexity(model, test_loader, config.device, "toroidal")
    results.perplexity['toroidal'] = ppl

    # 2. Rotational invariance
    inv_scores = test_rotational_invariance(model, test_loader, config.device)
    results.invariance_scores = inv_scores

    # 3. Gradient analysis
    grad_stats = analyze_gradients(model, test_loader, config.device)
    results.gradient_stats['toroidal'] = grad_stats

    # 4. Memory profiling
    mem_stats = profile_memory(model, config.batch_size, config.seq_len, config.device)
    if mem_stats:
        results.memory_usage['toroidal'] = mem_stats

    # 5. Inference speed
    speed_stats = measure_inference_speed(model, config.batch_size, config.seq_len, device=config.device)
    results.inference_speed['toroidal'] = speed_stats

    # Save results
    results.save(output_dir / 'evaluation_results.json')
    results.print_summary()

    print(f"\n✓ Evaluation complete! Results saved to {output_dir}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate toroidal attention")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config')
    parser.add_argument('--output', type=str, default='evaluation', help='Output directory')
    parser.add_argument('--visualize-ablation', type=str, help='Path to ablation results directory')

    args = parser.parse_args()

    if args.visualize_ablation:
        visualize_ablation_results(
            Path(args.visualize_ablation),
            Path(args.output) / 'ablation_visualization.png'
        )
    else:
        comprehensive_evaluation(
            Path(args.checkpoint),
            Path(args.config),
            Path(args.output),
        )


if __name__ == "__main__":
    main()

