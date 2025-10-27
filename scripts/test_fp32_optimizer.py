#!/usr/bin/env python3
"""
Debug: Test with FP32 optimizer (master weights).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from toroidal_attention import ToroidalAttention
from scripts.load_phi2 import replace_attention_layer, freeze_model_except_attention

def test_fp32_optimizer():
    print("="*80)
    print("FP32 OPTIMIZER TEST")
    print("="*80)
    
    device = torch.device("cuda")
    dtype = torch.float16
    
    print("\n[1/3] Loading model (fp16)...")
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
    
    print("\n[2/3] Creating FP32 optimizer (master weights)...")
    # Collect trainable params and create FP32 copies
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Create FP32 master weights
    master_params = [p.detach().clone().float().requires_grad_(True) for p in trainable_params]
    
    optimizer = torch.optim.AdamW(master_params, lr=1e-5, weight_decay=0.01)
    
    print(f"   Model params dtype: {trainable_params[0].dtype}")
    print(f"   Master params dtype: {master_params[0].dtype}")
    print(f"   Total trainable params: {sum(p.numel() for p in trainable_params):,}")
    
    print("\n[3/3] Running 10 training steps with FP32 master weights...")
    text = "The quick brown fox jumps over the lazy dog. " * 20
    inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    model.train()
    
    for step in range(1, 11):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward (model in fp16)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        if torch.isnan(loss):
            print(f"🔴 Step {step}: Loss is NaN")
            break
        
        # Backward (gradients in fp16)
        loss.backward()
        
        # Copy fp16 gradients to fp32 master weights
        for model_param, master_param in zip(trainable_params, master_params):
            if model_param.grad is not None:
                master_param.grad = model_param.grad.detach().clone().float()
        
        # Clip gradients in fp32
        max_grad_before = max(p.grad.abs().max().item() for p in master_params if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(master_params, 1.0)
        max_grad_after = max(p.grad.abs().max().item() for p in master_params if p.grad is not None)
        
        # Optimizer step on fp32 master weights
        optimizer.step()
        
        # Copy updated fp32 master weights back to fp16 model
        with torch.no_grad():
            for model_param, master_param in zip(trainable_params, master_params):
                model_param.copy_(master_param.half())
        
        print(f"   Step {step:2d}: Loss={loss.item():7.4f}, MaxGrad={max_grad_before:8.2e} → {max_grad_after:8.2e} ✅")
    
    print("\n" + "="*80)
    print("RESULT: FP32 master weights approach")
    print("="*80)

if __name__ == "__main__":
    test_fp32_optimizer()

