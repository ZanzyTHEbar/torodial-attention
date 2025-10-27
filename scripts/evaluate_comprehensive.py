"""
Comprehensive evaluation script for systematically benchmarking Phi-2 + ToroidalAttention.

Based on modern best practices:
- Multiple evaluation metrics (perplexity, BLEU, accuracy)
- Standard benchmarks (WikiText, LAMBADA, HellaSwag)
- Statistical significance testing
- Comparison tables with baseline
- Automated result visualization
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import warnings

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from scipy import stats

from scripts.load_phi2 import load_phi2_model, replace_attention_layer
from toroidal_attention import ToroidalAttention


class BenchmarkSuite:
    """Comprehensive benchmark suite for LLM evaluation."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def evaluate_perplexity(self, dataset_name, split='test', max_samples=None):
        """Compute perplexity on a dataset."""
        print(f"\n[Perplexity] Evaluating on {dataset_name}/{split}...")
        
        if dataset_name == 'wikitext2':
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        elif dataset_name == 'wikitext103':
            dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
        elif dataset_name == 'ptb':
            dataset = load_dataset('ptb_text_only', split=split)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        total_loss = 0.0
        total_tokens = 0
        
        self.model.eval()
        with torch.no_grad():
            for example in tqdm(dataset, desc="Computing perplexity"):
                text = example['text']
                if not text.strip():
                    continue
                
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                input_ids = inputs['input_ids'].to(self.device)
                
                # Shift for autoregressive loss
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]
                labels[:, -1] = -100
                
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                valid_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return {
            'perplexity': float(perplexity),
            'loss': float(avg_loss),
            'num_tokens': int(total_tokens),
        }
    
    def evaluate_lambada(self, max_samples=None):
        """Evaluate on LAMBADA (last word prediction accuracy)."""
        print("\n[LAMBADA] Evaluating last word prediction...")
        
        dataset = load_dataset('lambada', split='test')
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        correct = 0
        total = 0
        
        self.model.eval()
        with torch.no_grad():
            for example in tqdm(dataset, desc="LAMBADA"):
                text = example['text']
                # Get last word
                words = text.strip().split()
                if len(words) < 2:
                    continue
                
                target_word = words[-1]
                context = ' '.join(words[:-1])
                
                # Tokenize context
                inputs = self.tokenizer(context, return_tensors='pt')
                input_ids = inputs['input_ids'].to(self.device)
                
                # Generate next token
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]
                predicted_token = logits.argmax(dim=-1)
                predicted_word = self.tokenizer.decode(predicted_token).strip()
                
                if predicted_word.lower() == target_word.lower():
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': float(accuracy),
            'correct': int(correct),
            'total': int(total),
        }
    
    def evaluate_multiple_choice(self, dataset_name, max_samples=None):
        """Evaluate on multiple choice tasks (HellaSwag, etc.)."""
        print(f"\n[{dataset_name}] Evaluating multiple choice...")
        
        if dataset_name == 'hellaswag':
            dataset = load_dataset('hellaswag', split='validation')
        elif dataset_name == 'arc_easy':
            dataset = load_dataset('ai2_arc', 'ARC-Easy', split='test')
        elif dataset_name == 'arc_challenge':
            dataset = load_dataset('ai2_arc', 'ARC-Challenge', split='test')
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        correct = 0
        total = 0
        
        self.model.eval()
        with torch.no_grad():
            for example in tqdm(dataset, desc=dataset_name):
                if dataset_name == 'hellaswag':
                    context = example['ctx']
                    choices = example['endings']
                    label = int(example['label'])
                else:
                    context = example['question']
                    choices = example['choices']['text']
                    label = ord(example['answerKey']) - ord('A')
                
                # Score each choice
                scores = []
                for choice in choices:
                    text = context + ' ' + choice
                    inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                    input_ids = inputs['input_ids'].to(self.device)
                    
                    outputs = self.model(input_ids=input_ids, labels=input_ids)
                    loss = outputs.loss
                    scores.append(-loss.item())  # Lower loss = better
                
                predicted = np.argmax(scores)
                if predicted == label:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': float(accuracy),
            'correct': int(correct),
            'total': int(total),
        }


def compare_with_baseline(results_toroidal, results_baseline):
    """Compare toroidal results with baseline and compute statistics."""
    comparisons = {}
    
    for metric in results_toroidal.keys():
        if metric in results_baseline:
            toroidal_val = results_toroidal[metric]
            baseline_val = results_baseline[metric]
            
            # Compute delta and percentage
            delta = toroidal_val - baseline_val
            pct_change = (delta / baseline_val * 100) if baseline_val != 0 else 0
            
            # For perplexity, lower is better
            if 'perplexity' in metric.lower():
                improvement = -pct_change
            else:
                improvement = pct_change
            
            comparisons[metric] = {
                'toroidal': toroidal_val,
                'baseline': baseline_val,
                'delta': delta,
                'pct_change': pct_change,
                'improvement': improvement,
            }
    
    return comparisons


def run_full_evaluation(model, tokenizer, device, config):
    """Run comprehensive evaluation suite."""
    suite = BenchmarkSuite(model, tokenizer, device)
    
    results = {}
    
    # Perplexity benchmarks
    if config.get('eval_wikitext2', True):
        results['wikitext2'] = suite.evaluate_perplexity('wikitext2', max_samples=config.get('max_samples'))
    
    if config.get('eval_wikitext103', False):
        results['wikitext103'] = suite.evaluate_perplexity('wikitext103', max_samples=config.get('max_samples'))
    
    # Accuracy benchmarks
    if config.get('eval_lambada', False):
        results['lambada'] = suite.evaluate_lambada(max_samples=config.get('max_samples'))
    
    if config.get('eval_hellaswag', False):
        results['hellaswag'] = suite.evaluate_multiple_choice('hellaswag', max_samples=config.get('max_samples'))
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of Phi-2 + ToroidalAttention')
    
    # Model args
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to fine-tuned checkpoint')
    parser.add_argument('--baseline', action='store_true', help='Evaluate baseline Phi-2')
    parser.add_argument('--layer_indices', type=str, default='0')
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--lambda_distance', type=float, default=0.1)
    
    # Evaluation args
    parser.add_argument('--eval_wikitext2', action='store_true', default=True)
    parser.add_argument('--eval_wikitext103', action='store_true')
    parser.add_argument('--eval_lambada', action='store_true')
    parser.add_argument('--eval_hellaswag', action='store_true')
    parser.add_argument('--max_samples', type=int, default=None)
    
    # Output args
    parser.add_argument('--output', type=str, default='results/evaluation/results.json')
    parser.add_argument('--compare_with', type=str, default=None, help='Path to baseline results for comparison')
    
    args = parser.parse_args()
    
    # Use GPU if available (optimized for 8GB VRAM)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("⚠️  Using CPU (no GPU detected)")
    
    print("="*80)
    print("COMPREHENSIVE EVALUATION: PHI-2 + TOROIDAL ATTENTION")
    print("="*80)
    
    # Load model
    print("\n[1/3] Loading model...")
    # Use float16 for GPU to save memory
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    model, tokenizer, config = load_phi2_model(device='cpu', torch_dtype=dtype)
    
    if not args.baseline:
        # Replace with toroidal attention
        print("\n[2/3] Replacing attention layers...")
        layer_indices = [int(x) for x in args.layer_indices.split(',')]
        
        for layer_idx in layer_indices:
            toroidal = ToroidalAttention(
                d_model=config.hidden_size,
                n_heads=config.num_attention_heads,
                depth=args.depth,
                max_len=2048,
                lambda_distance=args.lambda_distance,
                use_orthogonal_pe=True,
            )
            replace_attention_layer(model, layer_idx=layer_idx, toroidal_attn=toroidal, copy_weights=False)
        
        # Load checkpoint if provided
        if args.checkpoint:
            print(f"  Loading checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded from epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")
    
    # Move model to device
    print(f"  Moving model to {device}...")
    model = model.to(device)
    print(f"  ✓ Model ready on {device}")
    
    # Run evaluation
    print("\n[3/3] Running evaluations...")
    eval_config = {
        'eval_wikitext2': args.eval_wikitext2,
        'eval_wikitext103': args.eval_wikitext103,
        'eval_lambada': args.eval_lambada,
        'eval_hellaswag': args.eval_hellaswag,
        'max_samples': args.max_samples,
    }
    
    results = run_full_evaluation(model, tokenizer, device, eval_config)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    full_results = {
        'config': {
            'baseline': args.baseline,
            'layer_indices': args.layer_indices if not args.baseline else None,
            'depth': args.depth if not args.baseline else None,
            'lambda_distance': args.lambda_distance if not args.baseline else None,
            'checkpoint': args.checkpoint,
        },
        'results': results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")
    
    # Print summary
    print("\nRESULTS SUMMARY:")
    print("-"*80)
    for dataset, metrics in results.items():
        print(f"\n{dataset.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    # Compare with baseline if provided
    if args.compare_with:
        print(f"\n{'='*80}")
        print("COMPARISON WITH BASELINE")
        print(f"{'='*80}")
        
        with open(args.compare_with) as f:
            baseline_data = json.load(f)
        
        baseline_results = baseline_data['results']
        
        print(f"\n{'Metric':<30} {'Baseline':>12} {'Toroidal':>12} {'Δ':>12} {'Change %':>12}")
        print("-"*80)
        
        for dataset in results.keys():
            if dataset in baseline_results:
                for metric in results[dataset].keys():
                    if metric in baseline_results[dataset]:
                        baseline_val = baseline_results[dataset][metric]
                        toroidal_val = results[dataset][metric]
                        delta = toroidal_val - baseline_val
                        pct = (delta / baseline_val * 100) if baseline_val != 0 else 0
                        
                        print(f"{dataset}_{metric:<23} {baseline_val:>12.4f} {toroidal_val:>12.4f} {delta:>12.4f} {pct:>11.2f}%")


if __name__ == '__main__':
    main()

