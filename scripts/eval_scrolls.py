#!/usr/bin/env python3
"""
SCROLLS Evaluation Harness for Toroidal Attention.

SCROLLS (Standardized CompaRison Over Long Language Sequences) is a suite of 7 datasets
testing long-context understanding:
- QAsper (scientific QA)
- Narrative QA (story understanding)
- Qua LI TY (multiple-choice QA)
- QuALITY (question answering)
- SummScreen FD (TV show summarization)
- Gov Report (government report summarization)
- Contract NLI (contract entailment)

Usage:
    # Evaluate TAM on SCROLLS (env-gated)
    RUN_SCROLLS=1 python scripts/eval_scrolls.py --model tam --depth 4 --out results/scrolls_tam.json

    # Evaluate baseline
    RUN_SCROLLS=1 python scripts/eval_scrolls.py --model phi2 --out results/scrolls_phi2.json
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.load_phi2 import (
    freeze_model_except_attention,
    load_phi2_model,
    replace_attention_layer,
)
from toroidal_attention import ToroidalAttention


def load_scrolls_dataset(task: str, split: str = 'validation'):
    """
    Load a SCROLLS dataset.

    Args:
        task: Task name (e.g., 'qasper', 'narrative_qa', 'quality')
        split: Dataset split ('validation' or 'test')

    Returns:
        Dataset with fields: 'input', 'output', 'id', etc.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: uv add datasets")

    # SCROLLS is hosted on HuggingFace
    dataset = load_dataset('tau/scrolls', task, split=split)
    return dataset


def evaluate_model_on_scrolls(
    model,
    tokenizer,
    task: str,
    max_samples: Optional[int] = None,
    device: str = 'cuda',
) -> Dict:
    """
    Evaluate a model on a SCROLLS task.

    Args:
        model: Language model
        tokenizer: Tokenizer
        task: SCROLLS task name
        max_samples: Limit number of samples
        device: Device to run on

    Returns:
        Dict with metrics: F1, EM, ROUGE, etc. (task-dependent)
    """
    dataset = load_scrolls_dataset(task)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    results = []

    for example in tqdm(dataset, desc=f"Evaluating {task}"):
        # Prepare input
        input_text = example.get('input', '')

        # Tokenize
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=tokenizer.model_max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Store result
        results.append({
            'id': example.get('id', ''),
            'input_length': len(input_text.split()),
            'generated': generated,
            'reference': example.get('output', ''),
        })

    # Compute metrics
    metrics = compute_scrolls_metrics(task, results)

    return {
        'task': task,
        'num_samples': len(results),
        'metrics': metrics,
        'results': results,
    }


def compute_scrolls_metrics(task: str, results: List[Dict]) -> Dict:
    """
    Compute task-specific metrics for SCROLLS.

    SCROLLS uses different metrics per task:
    - QAsper, Narrative QA, QuALITY: F1, EM
    - SummScreen, Gov Report: ROUGE-1, ROUGE-2, ROUGE-L
    - Contract NLI: Accuracy, F1
    """
    # TODO: Implement actual SCROLLS metrics
    # For now, return placeholders
    return {
        'f1': 0.0,
        'exact_match': 0.0,
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate on SCROLLS')
    parser.add_argument('--model', choices=['tam', 'phi2'], default='tam')
    parser.add_argument('--task', type=str, default='qasper', help='SCROLLS task')
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--lambda-distance', type=float, default=0.1)
    parser.add_argument('--fusion', type=str, default='low_rank')
    parser.add_argument('--layer-idx', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', type=str, default='results/scrolls.json')
    args = parser.parse_args()

    # Check env gate
    if os.getenv('RUN_SCROLLS') != '1':
        print("SCROLLS evaluation is env-gated. Set RUN_SCROLLS=1 to run.")
        sys.exit(0)

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, config = load_phi2_model(
        device=args.device,
        torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32,
    )

    if args.model == 'tam':
        print(f"Replacing layer {args.layer_idx} with ToroidalAttention")
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
    results = evaluate_model_on_scrolls(
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

