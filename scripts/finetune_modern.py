"""
Modern fine-tuning script for Phi-2 + ToroidalAttention with best practices.

Features:
- Gradient accumulation
- Mixed precision training (AMP)
- Early stopping with patience
- Learning rate scheduling (warmup + cosine decay)
- Gradient clipping
- W&B integration
- Checkpoint saving (best + periodic)
- Validation during training
- Memory-efficient data loading
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional
import warnings

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import numpy as np

from scripts.load_phi2 import load_phi2_model, replace_attention_layer, freeze_model_except_attention
from toroidal_attention import ToroidalAttention


class EarlyStopping:
    """Early stopping to avoid overfitting."""
    def __init__(self, patience=3, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
    
    def __call__(self, value):
        if self.mode == 'min':
            if value < self.best_value - self.min_delta:
                self.best_value = value
                self.counter = 0
                return True  # New best
            else:
                self.counter += 1
        else:
            if value > self.best_value + self.min_delta:
                self.best_value = value
                self.counter = 0
                return True
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return False


def prepare_dataset(
    dataset_name: str,
    tokenizer,
    max_length: int = 512,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
):
    """Prepare train/validation datasets."""
    if dataset_name == 'wikitext2':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        train_data = dataset['train']
        val_data = dataset['validation']
    elif dataset_name == 'wikitext103':
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
        train_data = dataset['train']
        val_data = dataset['validation']
    elif dataset_name == 'openwebtext':
        dataset = load_dataset('openwebtext', split='train')
        # Split into train/val
        split_data = dataset.train_test_split(test_size=0.05, seed=42)
        train_data = split_data['train']
        val_data = split_data['test']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Limit sizes if specified
    if train_size:
        train_data = train_data.select(range(min(train_size, len(train_data))))
    if val_size:
        val_data = val_data.select(range(min(val_size, len(val_data))))
    
    def tokenize(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
        )
    
    train_tokenized = train_data.map(
        tokenize,
        batched=True,
        remove_columns=train_data.column_names,
        desc="Tokenizing train"
    )
    val_tokenized = val_data.map(
        tokenize,
        batched=True,
        remove_columns=val_data.column_names,
        desc="Tokenizing val"
    )
    
    train_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    val_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    return train_tokenized, val_tokenized


def compute_metrics(model, dataloader, device, max_batches=None):
    """Compute loss and perplexity on a dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            if max_batches and i >= max_batches:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Shift for autoregressive loss
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss
            valid_tokens = (labels != -100).sum().item()
            
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return {
        'loss': float(avg_loss),
        'perplexity': float(perplexity),
    }


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    gradient_accumulation_steps,
    max_grad_norm,
    use_amp,
    wandb_enabled,
    epoch,
    global_step,
    master_params=None,
    trainable_params=None,
):
    """Train for one epoch with optional FP32 master weights."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Shift for autoregressive loss
        labels = input_ids.clone()
        
        # Forward pass
        if use_amp:
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Accumulate loss for logging
        total_loss += loss.item() * gradient_accumulation_steps
        
        # Optimizer step (after accumulation)
        if (step + 1) % gradient_accumulation_steps == 0:
            # Copy gradients from fp16 model to fp32 master weights if needed
            if master_params is not None and trainable_params is not None:
                for model_param, master_param in zip(trainable_params, master_params):
                    if model_param.grad is not None:
                        master_param.grad = model_param.grad.detach().clone().float()
            
            if use_amp:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    master_params if master_params is not None else model.parameters(),
                    max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    master_params if master_params is not None else model.parameters(),
                    max_grad_norm
                )
                optimizer.step()
            
            # Copy updated fp32 master weights back to fp16 model if needed
            if master_params is not None and trainable_params is not None:
                with torch.no_grad():
                    for model_param, master_param in zip(trainable_params, master_params):
                        model_param.copy_(master_param.half())
            
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            # Logging
            if wandb_enabled and global_step % 10 == 0:
                try:
                    import wandb
                    wandb.log({
                        'train/loss': total_loss / (step + 1),
                        'train/perplexity': np.exp(total_loss / (step + 1)),
                        'train/learning_rate': scheduler.get_last_lr()[0],
                        'train/grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        'epoch': epoch,
                        'global_step': global_step,
                    })
                except:
                    pass
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss / (step + 1):.4f}",
                'ppl': f"{np.exp(total_loss / (step + 1)):.2f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
            })
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss, global_step


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, metrics, save_path):
    """Save model checkpoint."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
    }
    
    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Phi-2 with ToroidalAttention (Best Practices)')
    
    # Model args
    parser.add_argument('--model_name', type=str, default='microsoft/phi-2')
    parser.add_argument('--layer_indices', type=str, default='0', help='Comma-separated layer indices')
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--lambda_distance', type=float, default=0.1)
    parser.add_argument('--fusion_mode', type=str, default='low_rank')
    
    # Data args
    parser.add_argument('--dataset', type=str, default='wikitext2')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--train_size', type=int, default=None)
    parser.add_argument('--val_size', type=int, default=None)
    
    # Training args
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    # Optimization args
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluation args
    parser.add_argument('--eval_steps', type=int, default=500, help='Evaluate every N steps')
    parser.add_argument('--eval_batches', type=int, default=None, help='Max batches for evaluation')
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    
    # Checkpoint args
    parser.add_argument('--output_dir', type=str, default='results/checkpoints')
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--save_best_only', action='store_true')
    
    # Wandb args
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='toroidal-phi2')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    
    args = parser.parse_args()
    
    # Parse layer indices
    layer_indices = [int(x) for x in args.layer_indices.split(',')]
    
    # Initialize W&B
    wandb_enabled = args.use_wandb
    if wandb_enabled:
        try:
            import wandb
            run_name = args.wandb_run_name or f"phi2_toroidal_d{args.depth}_l{args.lambda_distance}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                tags=[f"depth{args.depth}", f"layers{len(layer_indices)}", args.dataset],
            )
            print("✓ W&B logging enabled")
        except ImportError:
            print("⚠️  W&B not installed, disabling logging")
            wandb_enabled = False
        except Exception as e:
            print(f"⚠️  W&B initialization failed: {e}")
            wandb_enabled = False
    
    print("="*80)
    print("FINE-TUNING PHI-2 + TOROIDAL ATTENTION (BEST PRACTICES)")
    print("="*80)
    print(f"Layers to replace: {layer_indices}")
    print(f"Depth: {args.depth}")
    print(f"Lambda: {args.lambda_distance}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Mixed precision: {args.use_amp}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Load model
    print("\n[1/6] Loading Phi-2 model...")
    # Use float16 for GPU to save memory, float32 for CPU
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    model, tokenizer, config = load_phi2_model(device='cpu', torch_dtype=dtype)
    
    # Disable AMP if already using fp16 (redundant and causes errors)
    if dtype == torch.float16 and args.use_amp:
        print("  ⚠️  Disabling AMP (model already in fp16)")
        args.use_amp = False
    
    # Replace layers
    print(f"\n[2/6] Replacing {len(layer_indices)} layers with ToroidalAttention...")
    for layer_idx in layer_indices:
        toroidal = ToroidalAttention(
            d_model=config.hidden_size,
            n_heads=config.num_attention_heads,
            depth=args.depth,
            max_len=args.max_length,
            lambda_distance=args.lambda_distance,
            fusion_mode=args.fusion_mode,
            use_orthogonal_pe=True,
        )
        # Convert toroidal attention to same dtype as model
        toroidal = toroidal.to(dtype)
        replace_attention_layer(model, layer_idx=layer_idx, toroidal_attn=toroidal, copy_weights=False)
    
    # Freeze base model
    freeze_model_except_attention(model, layer_indices)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Move to device
    print(f"\n  Moving model to {device}...")
    model = model.to(device)
    print(f"  ✓ Model ready on {device}")
    if device.type == 'cuda':
        print(f"  GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    
    # Prepare datasets
    print(f"\n[3/6] Preparing {args.dataset} dataset...")
    train_dataset, val_dataset = prepare_dataset(
        args.dataset,
        tokenizer,
        args.max_length,
        args.train_size,
        args.val_size,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Setup optimizer and scheduler
    print("\n[4/6] Setting up optimizer and scheduler...")
    
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # CRITICAL FIX: Use FP32 master weights when model is in fp16
    # This prevents optimizer state corruption in fp16
    master_params = None
    if dtype == torch.float16:
        print("  ⚠️  Model is fp16: creating FP32 master weights for optimizer")
        master_params = [p.detach().clone().float().requires_grad_(True) for p in trainable_params]
        optimizer_params = master_params
    else:
        optimizer_params = trainable_params
    
    optimizer = AdamW(
        optimizer_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    total_steps = len(train_loader) // args.gradient_accumulation_steps * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    scaler = GradScaler() if args.use_amp else None
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, mode='min')
    
    print(f"  Total training steps: {total_steps}")
    print(f"  Warmup steps: {args.warmup_steps}")
    
    # Training loop
    print(f"\n[5/6] Training for {args.epochs} epochs...")
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            args.gradient_accumulation_steps,
            args.max_grad_norm,
            args.use_amp,
            wandb_enabled,
            epoch,
            global_step,
            master_params=master_params,
            trainable_params=trainable_params,
        )
        
        print(f"\nTrain Loss: {train_loss:.4f}, Train PPL: {np.exp(train_loss):.2f}")
        
        # Validate
        print("\nValidating...")
        val_metrics = compute_metrics(model, val_loader, device, args.eval_batches)
        val_loss = val_metrics['loss']
        val_ppl = val_metrics['perplexity']
        
        print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
        
        # Log to W&B
        if wandb_enabled:
            try:
                import wandb
                wandb.log({
                    'val/loss': val_loss,
                    'val/perplexity': val_ppl,
                    'epoch': epoch,
                    'global_step': global_step,
                })
            except:
                pass
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                {'train_loss': train_loss, 'val_loss': val_loss, 'val_ppl': val_ppl},
                Path(args.output_dir) / f"best_model_d{args.depth}_l{args.lambda_distance}.pt"
            )
        
        if not args.save_best_only:
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                {'train_loss': train_loss, 'val_loss': val_loss, 'val_ppl': val_ppl},
                Path(args.output_dir) / f"checkpoint_epoch{epoch}.pt"
            )
        
        # Early stopping check
        if early_stopping(val_loss):
            if early_stopping.early_stop:
                print(f"\n⚠️  Early stopping triggered after epoch {epoch}")
                break
    
    print(f"\n[6/6] Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best val PPL: {np.exp(best_val_loss):.2f}")
    
    if wandb_enabled:
        try:
            import wandb
            wandb.finish()
        except:
            pass


if __name__ == '__main__':
    main()

