#!/usr/bin/env python3
"""
Debug script: Run 20 training steps to reproduce NaN at step 15-16.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from toroidal_attention import ToroidalAttention
from scripts.load_phi2 import replace_attention_layer, freeze_model_except_attention

def run_training_steps():
    print("="*80)
    print("20-STEP TRAINING DEBUG")
    print("="*80)
    
    # Setup
    device = torch.device("cuda")
    dtype = torch.float16
    
    print("\n[1/4] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=dtype,
        device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
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
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5,
        weight_decay=0.01
    )
    
    print(f"   GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    print("\n[2/4] Preparing data...")
    text = "The quick brown fox jumps over the lazy dog. " * 20
    inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    print("\n[3/4] Running 20 training steps...")
    model.train()
    
    for step in range(1, 21):
        optimizer.zero_grad()
        
        # Forward
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        # Check for NaN in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n🔴 Step {step}: Loss is {loss.item()}")
            print("   Checking logits...")
            if torch.isnan(outputs.logits).any():
                print("   → NaN in logits")
            if torch.isinf(outputs.logits).any():
                print("   → Inf in logits")
            break
        
        # Backward
        loss.backward()
        
        # Check gradients
        nan_grads = []
        inf_grads = []
        max_grad = 0.0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_grads.append(name)
                if torch.isinf(param.grad).any():
                    inf_grads.append(name)
                g_max = param.grad.abs().max().item()
                if g_max > max_grad:
                    max_grad = g_max
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Report
        status = "✅" if not nan_grads and not inf_grads else "🔴"
        print(f"   Step {step:2d}: Loss={loss.item():7.4f}, MaxGrad={max_grad:8.2e} {status}")
        
        if nan_grads:
            print(f"      NaN grads: {nan_grads[:3]}")
            break
        if inf_grads:
            print(f"      Inf grads: {inf_grads[:3]}")
            break
    
    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)

if __name__ == "__main__":
    run_training_steps()

