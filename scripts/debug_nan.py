#!/usr/bin/env python3
"""
Debug script to pinpoint where NaN originates during training.
Adds hooks to track NaN propagation through the model.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from toroidal_attention import ToroidalAttention
from scripts.load_phi2 import replace_attention_layer, freeze_model_except_attention
from datasets import load_dataset

def add_nan_hooks(model, name_prefix=""):
    """Add forward hooks to detect NaN in activations."""
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"🔴 NaN/Inf detected in {name} output")
                    print(f"   Shape: {output.shape}")
                    print(f"   Min: {output.min().item():.4f}, Max: {output.max().item():.4f}")
                    print(f"   Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")
                    raise ValueError(f"NaN/Inf in {name}")
            elif isinstance(output, tuple):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        if torch.isnan(o).any() or torch.isinf(o).any():
                            print(f"🔴 NaN/Inf detected in {name} output[{i}]")
                            raise ValueError(f"NaN/Inf in {name}")
        return hook
    
    for name, module in model.named_modules():
        full_name = f"{name_prefix}.{name}" if name_prefix else name
        if isinstance(module, (nn.Linear, nn.LayerNorm, ToroidalAttention)):
            hook = module.register_forward_hook(make_hook(full_name))
            hooks.append(hook)
    
    return hooks

def test_training_step():
    """Run a single training step with full diagnostics."""
    print("="*80)
    print("DIAGNOSTIC NaN DEBUGGING")
    print("="*80)
    
    # Setup
    device = torch.device("cuda")
    dtype = torch.float16
    
    print("\n[1/5] Loading Phi-2 model...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=dtype,
        device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n[2/5] Replacing layer 0 with ToroidalAttention...")
    toroidal = ToroidalAttention(
        d_model=2560,
        n_heads=32,
        max_len=2048,
        depth=2,
        lambda_distance=0.1,
        fusion_mode='low_rank',
        use_orthogonal_pe=True,
    ).to(dtype)
    
    replace_attention_layer(model, 0, toroidal)
    freeze_model_except_attention(model, [0])
    model = model.to(device)
    
    print(f"   GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    print("\n[3/5] Adding NaN detection hooks...")
    hooks = add_nan_hooks(model, "model")
    
    print("\n[4/5] Preparing data...")
    # Use single sample to avoid OOM during debugging
    text = "The quick brown fox jumps over the lazy dog. " * 20  # Repeat to get decent length
    
    inputs = tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
        max_length=128  # Shorter for debugging
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    print(f"   Batch shape: {input_ids.shape}")
    
    print("\n[5/5] Running forward pass with diagnostics...")
    model.train()
    
    try:
        # Forward pass
        print("   → Forward pass...")
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        print(f"   ✅ Forward pass complete")
        print(f"      Loss: {loss.item():.4f}")
        print(f"      Loss dtype: {loss.dtype}")
        
        # Check logits
        logits = outputs.logits
        print(f"      Logits shape: {logits.shape}")
        print(f"      Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
        print(f"      Logits has NaN: {torch.isnan(logits).any()}")
        print(f"      Logits has Inf: {torch.isinf(logits).any()}")
        
        # Backward pass
        print("   → Backward pass...")
        loss.backward()
        
        print(f"   ✅ Backward pass complete")
        
        # Check gradients
        print("   → Checking gradients...")
        nan_grads = []
        inf_grads = []
        large_grads = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_grads.append(name)
                if torch.isinf(param.grad).any():
                    inf_grads.append(name)
                grad_max = param.grad.abs().max().item()
                if grad_max > 1000:
                    large_grads.append((name, grad_max))
        
        if nan_grads:
            print(f"   🔴 NaN gradients in: {nan_grads}")
        if inf_grads:
            print(f"   🔴 Inf gradients in: {inf_grads}")
        if large_grads:
            print(f"   ⚠️  Large gradients (>1000):")
            for name, val in large_grads[:5]:
                print(f"      {name}: {val:.2e}")
        
        if not nan_grads and not inf_grads:
            print("   ✅ All gradients are finite")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        for hook in hooks:
            hook.remove()
    
    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_training_step()

