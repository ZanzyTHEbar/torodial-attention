#!/usr/bin/env python3
"""
LongBench Evaluation Harness for Toroidal Attention.

LongBench is a benchmark for long-context understanding with 21 datasets across 6 categories:
- Single-Doc QA
- Multi-Doc QA
- Summarization
- Few-shot Learning
- Synthetic Tasks
- Code Completion

Usage:
    # Evaluate TAM on LongBench (env-gated, requires HF datasets)
    RUN_LONGBENCH=1 python scripts/eval_longbench.py --model tam --depth 4 --out results/longbench_tam.json

    # Evaluate baseline Phi-2
    RUN_LONGBENCH=1 python scripts/eval_longbench.py --model phi2 --out results/longbench_phi2.json
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.load_phi2 import (
    freeze_model_except_attention,
    load_phi2_model,
    replace_attention_layer,
)
from toroidal_attention import ToroidalAttention


def load_longbench_dataset(task: str, split: str = 'test'):
    """
    Load a LongBench dataset.

    Args:
        task: Task name (e.g., 'narrativeqa', 'qasper', 'multifieldqa_en')
        split: Dataset split ('test' for LongBench)

    Returns:
        Dataset with fields: 'input', 'context', 'answers', 'length', etc.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: uv add datasets")

    # LongBench is hosted on HuggingFace
    dataset = load_dataset('THUDM/LongBench', task, split=split)
    return dataset


def evaluate_model_on_longbench(
    model,
    tokenizer,
    task: str,
    max_samples: Optional[int] = None,
    device: str = 'cuda',
) -> Dict:
    """
    Evaluate a model on a LongBench task.

    Args:
        model: Language model
        tokenizer: Tokenizer
        task: LongBench task name
        max_samples: Limit number of samples (for quick testing)
        device: Device to run on

    Returns:
        Dict with metrics: accuracy, f1, exact_match, etc. (task-dependent)
    """
    dataset = load_longbench_dataset(task)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    results = []

    for example in tqdm(dataset, desc=f"Evaluating {task}"):
        # Prepare input
        input_text = example.get('input', '')
        context = example.get('context', '')

        # Combine context and input
        prompt = f"{context}\n\n{input_text}" if context else input_text

        # Tokenize
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=tokenizer.model_max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Store result
        results.append({
            'input': input_text,
            'context_length': len(context.split()),
            'generated': generated,
            'reference': example.get('answers', []),
        })

    # Compute metrics (task-specific; placeholder for now)
    metrics = compute_longbench_metrics(task, results)

    return {
        'task': task,
        'num_samples': len(results),
        'metrics': metrics,
        'results': results,
    }


def compute_longbench_metrics(task: str, results: List[Dict]) -> Dict:
    """
    Compute task-specific metrics for LongBench.

    This is a placeholder; actual implementation would use task-specific scorers.
    """
    # TODO: Implement task-specific metrics (F1, EM, ROUGE, etc.)
    # For now, return dummy metrics
    return {
        'accuracy': 0.0,
        'f1': 0.0,
        'exact_match': 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate on LongBench')
    parser.add_argument('--model', choices=['tam', 'phi2'], default='tam', help='Model to evaluate')
    parser.add_argument('--task', type=str, default='narrativeqa', help='LongBench task')
    parser.add_argument('--depth', type=int, default=4, help='TAM depth (if model=tam)')
    parser.add_argument('--lambda-distance', type=float, default=0.1, help='TAM lambda (if model=tam)')
    parser.add_argument('--fusion', type=str, default='low_rank', help='TAM fusion mode')
    parser.add_argument('--layer-idx', type=int, default=0, help='Layer to replace (if model=tam)')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit samples for testing')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', type=str, default='results/longbench.json', help='Output JSON')
    args = parser.parse_args()

    # Check env gate
    if os.getenv('RUN_LONGBENCH') != '1':
        print("LongBench evaluation is env-gated. Set RUN_LONGBENCH=1 to run.")
        sys.exit(0)

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, config = load_phi2_model(device=args.device, torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32)

    if args.model == 'tam':
        print(f"Replacing layer {args.layer_idx} with ToroidalAttention (depth={args.depth})")
        toroidal_attn = ToroidalAttention(
            d_model=config.hidden_size,
            n_heads=config.num_attention_heads,
            max_len=2048,
            depth=args.depth,
            lambda_distance=args.lambda_distance,
            fusion_mode=args.fusion,
        )
        replace_attention_layer(model, args.layer_idx, toroidal_attn, copy_weights=False)
        freeze_model_except_attention(model, args.layer_idx)

    # Evaluate
    print(f"Evaluating on task: {args.task}")
    results = evaluate_model_on_longbench(
        model=model,
        tokenizer=tokenizer,
        task=args.task,
        max_samples=args.max_samples,
        device=args.device,
    )

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {args.out}")
    print(f"Metrics: {results['metrics']}")


if __name__ == '__main__':
    main()

