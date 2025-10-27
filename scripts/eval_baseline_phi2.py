"""Evaluate baseline Phi-2 on WikiText-2"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import json
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create results directory
os.makedirs('results', exist_ok=True)

print("\n" + "="*80)
print("BASELINE PHI-2 EVALUATION")
print("="*80)
print("\n[1/4] Loading Phi-2 model...")

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", 
    torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
    device_map=None
)
model = model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
tokenizer.pad_token = tokenizer.eos_token

print(f"  ✓ Model loaded on {device}")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\n[2/4] Loading WikiText-2 validation set...")

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
dataset = dataset.select(range(50))  # Same as toroidal validation

print(f"  ✓ Loaded {len(dataset)} samples")

print("\n[3/4] Evaluating...")

# Evaluate
losses = []
for sample in tqdm(dataset, desc="Evaluating"):
    text = sample['text']
    if not text.strip():
        continue
    
    inputs = tokenizer(
        text, 
        return_tensors='pt', 
        max_length=256, 
        truncation=True, 
        padding='max_length'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        losses.append(outputs.loss.item())

avg_loss = np.mean(losses)
perplexity = np.exp(avg_loss)

results = {
    'model': 'Phi-2 Baseline (no toroidal attention)',
    'dataset': 'WikiText-2 validation',
    'samples': len(losses),
    'avg_loss': float(avg_loss),
    'perplexity': float(perplexity),
    'device': str(device),
    'dtype': 'float16' if device.type == 'cuda' else 'float32'
}

print("\n[4/4] Results:")
print("="*80)
print("BASELINE PHI-2 RESULTS")
print("="*80)
print(f"Model: {results['model']}")
print(f"Dataset: {results['dataset']}")
print(f"Samples: {results['samples']}")
print(f"Average Loss: {avg_loss:.4f}")
print(f"Perplexity: {perplexity:.2f}")
print(f"Device: {results['device']}")
print(f"Dtype: {results['dtype']}")
print("="*80)

with open('results/baseline_phi2.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ Results saved to: results/baseline_phi2.json")
print("\nNext: Run comparison with toroidal attention results")

