"""
Training Script for Toroidal Attention

Fine-tunes Phi-2 model with toroidal attention on prepared datasets.
Implements:
- Training loop with AdamW optimizer
- Gradient monitoring for stability validation
- Checkpoint saving
- Perplexity evaluation
- Ablation study support (2D vs 3D, different fusion modes)
"""

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.load_phi2 import (
    freeze_model_except_attention,
    load_phi2_model,
    replace_attention_layer,
)
from scripts.prepare_data import create_dataloaders
from toroidal_attention import ToroidalAttention
from toroidal_attention.latent import LatentCfg


class TrainingConfig:
    """Configuration for toroidal attention training."""

    def __init__(self, **kwargs):
        # Model
        self.model_name = kwargs.get('model_name', 'microsoft/phi-2')
        self.layer_idx = kwargs.get('layer_idx', 0)  # Which layer to replace

        # Toroidal attention
        self.depth = kwargs.get('depth', 4)
        self.lambda_distance = kwargs.get('lambda_distance', 0.1)
        self.fusion_mode = kwargs.get('fusion_mode', 'low_rank')
        self.fusion_rank = kwargs.get('fusion_rank', None)
        # Backends & options
        self.backend = kwargs.get('backend', 'sdpa')
        self.window_size = kwargs.get('window_size', None)
        self.allow_flash2 = kwargs.get('allow_flash2', True)
        # Latent streaming (inference-oriented)
        self.latent_dim = kwargs.get('latent_dim', None)
        self.latent_update = kwargs.get('latent_update', 'gru')

        # Dataset
        self.dataset_type = kwargs.get('dataset_type', 'periodic')
        self.seq_len = kwargs.get('seq_len', 128)
        self.n_train = kwargs.get('n_train', 1000)
        self.n_val = kwargs.get('n_val', 200)
        # Optional dataset params
        self.period = kwargs.get('period', 32)
        self.frequency = kwargs.get('frequency', 0.1)

        # Training
        self.batch_size = kwargs.get('batch_size', 8)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.num_epochs = kwargs.get('num_epochs', 10)
        self.warmup_steps = kwargs.get('warmup_steps', 100)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        self.eval_every = kwargs.get('eval_every', 100)
        self.freeze_base = kwargs.get('freeze_base', True)

        # System
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path(kwargs.get('save_dir', 'checkpoints'))
        self.log_dir = Path(kwargs.get('log_dir', 'logs'))

        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        return {k: str(v) if isinstance(v, Path) else v
                for k, v in self.__dict__.items()}

    def save(self, path: Path):
        """Save configuration."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def load_training_config_yaml(path: Path) -> TrainingConfig:
    """
    Load TrainingConfig from a YAML file structured as:
    data: {dataset, max_len, num_samples, period, frequency}
    toroidal_attention: {depth, lambda_distance, fusion_mode, fusion_rank, dropout, target_layer_idx}
    training: {batch_size, epochs, learning_rate, warmup_steps, freeze_base}
    runtime: {device, dtype}
    """
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get('data', {})
    tor_cfg = cfg.get('toroidal_attention', {})
    train_cfg = cfg.get('training', {})
    run_cfg = cfg.get('runtime', {})

    kwargs = {
        'model_name': 'microsoft/phi-2',
        'layer_idx': tor_cfg.get('target_layer_idx', 0),
        'depth': tor_cfg.get('depth', 4),
        'lambda_distance': tor_cfg.get('lambda_distance', 0.1),
        'fusion_mode': tor_cfg.get('fusion_mode', 'low_rank'),
        'fusion_rank': tor_cfg.get('fusion_rank', None),
        'backend': tor_cfg.get('backend', 'sdpa'),
        'window_size': tor_cfg.get('window_size', None),
        'allow_flash2': tor_cfg.get('allow_flash2', True),
        'latent_dim': tor_cfg.get('latent_dim', None),
        'latent_update': tor_cfg.get('latent_update', 'gru'),
        'dataset_type': data_cfg.get('dataset', 'periodic'),
        'seq_len': data_cfg.get('max_len', 128),
        'n_train': data_cfg.get('num_samples', 1000),
        'n_val': max(1, int(0.2 * data_cfg.get('num_samples', 1000))),
        'period': data_cfg.get('period', 32),
        'frequency': data_cfg.get('frequency', 0.1),
        'batch_size': train_cfg.get('batch_size', 4),
        'learning_rate': train_cfg.get('learning_rate', 5e-5),
        'num_epochs': train_cfg.get('epochs', 1),
        'warmup_steps': train_cfg.get('warmup_steps', 0),
        'freeze_base': train_cfg.get('freeze_base', True),
        'device': run_cfg.get('device', 'auto'),
    }

    # Normalize device 'auto'
    if kwargs['device'] == 'auto':
        kwargs['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    return TrainingConfig(**kwargs)


class MetricsTracker:
    """Track training metrics and gradients."""

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
        self.gradient_norms = []
        self.learning_rates = []
        self.best_val_loss = float('inf')
        self.best_val_perplexity = float('inf')

    def update_train(self, loss: float, grad_norm: float, lr: float):
        """Update training metrics."""
        self.train_losses.append(loss)
        self.train_perplexities.append(np.exp(loss))
        self.gradient_norms.append(grad_norm)
        self.learning_rates.append(lr)

    def update_val(self, loss: float):
        """Update validation metrics."""
        self.val_losses.append(loss)
        perplexity = np.exp(loss)
        self.val_perplexities.append(perplexity)

        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.best_val_perplexity = perplexity
            return True  # New best
        return False

    def get_summary(self) -> Dict:
        """Get summary of metrics."""
        return {
            'train_loss': self.train_losses[-1] if self.train_losses else None,
            'val_loss': self.val_losses[-1] if self.val_losses else None,
            'train_perplexity': self.train_perplexities[-1] if self.train_perplexities else None,
            'val_perplexity': self.val_perplexities[-1] if self.val_perplexities else None,
            'best_val_loss': self.best_val_loss,
            'best_val_perplexity': self.best_val_perplexity,
            'avg_grad_norm': np.mean(self.gradient_norms[-100:]) if self.gradient_norms else None,
        }

    def save(self, path: Path):
        """Save metrics to JSON."""
        data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_perplexities': self.train_perplexities,
            'val_perplexities': self.val_perplexities,
            'gradient_norms': self.gradient_norms,
            'learning_rates': self.learning_rates,
            'summary': self.get_summary(),
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


def compute_loss(model, batch, device):
    """
    Compute language modeling loss.

    Args:
        model: The model
        batch: Batch with 'input_ids' and 'labels'
        device: Device

    Returns:
        loss: Cross-entropy loss
    """
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)

    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels)

    return outputs.loss


def evaluate(model, dataloader, device):
    """
    Evaluate model on validation set.

    Returns:
        Average loss and perplexity
    """
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            loss = compute_loss(model, batch, device)
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    perplexity = np.exp(avg_loss)

    return avg_loss, perplexity


def train_epoch(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    config: TrainingConfig,
    metrics: MetricsTracker,
    epoch: int,
):
    """Train for one epoch."""
    model.train()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")

    for step, batch in enumerate(pbar):
        # Forward pass
        loss = compute_loss(model, batch, config.device)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Compute gradient norm (for stability validation)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.max_grad_norm
        )

        # Optimizer step
        optimizer.step()
        scheduler.step()

        # Update metrics
        current_lr = scheduler.get_last_lr()[0]
        metrics.update_train(loss.item(), grad_norm.item(), current_lr)

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'ppl': f"{np.exp(loss.item()):.2f}",
            'grad': f"{grad_norm.item():.3f}",
            'lr': f"{current_lr:.2e}",
        })

        # Validation
        if (step + 1) % config.eval_every == 0:
            val_loss, val_ppl = evaluate(model, val_loader, config.device)
            is_best = metrics.update_val(val_loss)

            print(f"\n  Validation: loss={val_loss:.4f}, perplexity={val_ppl:.2f}")

            if is_best:
                print("  ✓ New best model!")
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    step,
                    metrics,
                    config,
                    is_best=True
                )

            model.train()


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    step: int,
    metrics: MetricsTracker,
    config: TrainingConfig,
    is_best: bool = False,
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics.get_summary(),
        'config': config.to_dict(),
    }

    # Save regular checkpoint
    filename = f"checkpoint_epoch{epoch}_step{step}.pt"
    filepath = config.save_dir / filename
    torch.save(checkpoint, filepath)

    # Save best model separately
    if is_best:
        best_path = config.save_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"  Saved best model to {best_path}")


def train_toroidal_attention(config: TrainingConfig):
    """
    Main training function.

    Args:
        config: Training configuration
    """
    print("=" * 60)
    print("Toroidal Attention Training")
    print("=" * 60)
    print("\nConfiguration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")

    # Save config
    config.save(config.save_dir / "config.json")

    # Initialize metrics tracker
    metrics = MetricsTracker()

    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("\n[2/5] Loading Phi-2 model...")
    model, _, model_config = load_phi2_model(
        model_name=config.model_name,
        device=config.device,
    )

    # Create toroidal attention
    print("\n[3/5] Creating toroidal attention...")
    latent_cfg = None
    if config.latent_dim is not None:
        latent_cfg = LatentCfg(latent_dim=int(config.latent_dim), update=str(config.latent_update))

    toroidal_attn = ToroidalAttention(
        d_model=model_config.hidden_size,
        n_heads=model_config.num_attention_heads,
        max_len=config.seq_len * 2,  # Allow some headroom
        depth=config.depth,
        lambda_distance=config.lambda_distance,
        fusion_mode=config.fusion_mode,
        fusion_rank=config.fusion_rank,
        backend=config.backend,
        window_size=config.window_size,
        allow_flash2=bool(config.allow_flash2),
        latent_cfg=latent_cfg,
    )

    # Replace attention layer
    replace_attention_layer(
        model,
        layer_idx=config.layer_idx,
        toroidal_attn=toroidal_attn,
        copy_weights=False,  # Start with random initialization
    )

    # Optionally freeze non-toroidal parameters
    if config.freeze_base:
        freeze_model_except_attention(model, config.layer_idx)

    # Prepare datasets
    print("\n[4/5] Preparing datasets...")
    train_loader, val_loader = create_dataloaders(
        dataset_type=config.dataset_type,
        tokenizer=tokenizer,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        n_train=config.n_train,
        n_val=config.n_val,
        period=config.period,
        frequency=config.frequency,
    )

    # Setup optimizer and scheduler
    print("\n[5/5] Setting up training...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )

    # Evaluate baseline
    print("\nEvaluating baseline...")
    val_loss, val_ppl = evaluate(model, val_loader, config.device)
    print(f"  Baseline: loss={val_loss:.4f}, perplexity={val_ppl:.2f}")
    metrics.update_val(val_loss)

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training")
    print("=" * 60)

    for epoch in range(config.num_epochs):
        train_epoch(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            config,
            metrics,
            epoch,
        )

        # End-of-epoch evaluation
        val_loss, val_ppl = evaluate(model, val_loader, config.device)
        is_best = metrics.update_val(val_loss)

        print(f"\nEpoch {epoch+1} complete:")
        print(f"  Val loss: {val_loss:.4f}")
        print(f"  Val perplexity: {val_ppl:.2f}")
        print(f"  Best val perplexity: {metrics.best_val_perplexity:.2f}")

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, -1, metrics, config, is_best)

    # Save final metrics
    metrics.save(config.log_dir / "metrics.json")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("\nFinal results:")
    for key, value in metrics.get_summary().items():
        print(f"  {key}: {value}")

    return model, metrics


def run_ablation_study():
    """
    Run ablation study comparing different configurations.

    Tests:
    1. Baseline (frozen Phi-2)
    2. Toroidal 2D (depth=1, circular only)
    3. Toroidal 3D (depth=4, full stacking)
    4. Different fusion modes
    """
    print("=" * 60)
    print("Toroidal Attention Ablation Study")
    print("=" * 60)

    base_config = {
        'dataset_type': 'periodic',
        'seq_len': 128,
        'n_train': 1000,
        'n_val': 200,
        'batch_size': 8,
        'num_epochs': 5,
        'learning_rate': 1e-4,
    }

    experiments = [
        {'name': 'toroidal_2d', 'depth': 1, 'fusion_mode': 'mean'},
        {'name': 'toroidal_3d_lowrank', 'depth': 4, 'fusion_mode': 'low_rank'},
        {'name': 'toroidal_3d_attention', 'depth': 4, 'fusion_mode': 'attention'},
    ]

    results = {}

    for exp in experiments:
        print(f"\n\n{'='*60}")
        print(f"Experiment: {exp['name']}")
        print(f"{'='*60}")

        config = TrainingConfig(
            **base_config,
            **exp,
            save_dir=f"checkpoints/{exp['name']}",
            log_dir=f"logs/{exp['name']}",
        )

        model, metrics = train_toroidal_attention(config)

        results[exp['name']] = metrics.get_summary()

    # Print comparison
    print("\n" + "=" * 60)
    print("Ablation Study Results")
    print("=" * 60)
    print(f"\n{'Experiment':<30} {'Best Val PPL':<15} {'Improvement':<15}")
    print("-" * 60)

    baseline_ppl = None
    for name, summary in results.items():
        ppl = summary['best_val_perplexity']

        if baseline_ppl is None:
            baseline_ppl = ppl
            improvement = "baseline"
        else:
            improvement = f"{(baseline_ppl - ppl) / baseline_ppl * 100:.1f}%"

        print(f"{name:<30} {ppl:<15.2f} {improvement:<15}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train toroidal attention")
    parser.add_argument('--depth', type=int, default=4, help='Number of depth platters')
    parser.add_argument('--fusion_mode', type=str, default='low_rank', choices=['low_rank', 'attention', 'mean'])
    parser.add_argument('--dataset', type=str, default='periodic', choices=['periodic', 'sinusoidal', 'openwebtext'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')

    args = parser.parse_args()

    if args.ablation:
        run_ablation_study()
    else:
        config = TrainingConfig(
            depth=args.depth,
            fusion_mode=args.fusion_mode,
            dataset_type=args.dataset,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
        )

        train_toroidal_attention(config)


if __name__ == "__main__":
    main()

