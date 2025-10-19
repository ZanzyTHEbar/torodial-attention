#!/usr/bin/env python3
"""
Experiment Runner for TAM vs Baselines.

Runs systematic comparisons across:
- Models: TAM (various configs), vanilla Phi-2, SWA-only, SDPA+Flash2
- Tasks: LongBench, SCROLLS, synthetic periodic/sinusoidal
- Metrics: Perplexity, accuracy, throughput, memory

Usage:
    # Run full ablation grid (env-gated)
    RUN_EXPERIMENTS=1 python scripts/run_experiments.py --config configs/experiments.yaml --out results/

    # Quick test
    python scripts/run_experiments.py --quick --out results/quick/
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.load_phi2 import load_phi2_model, replace_attention_layer
from toroidal_attention import ToroidalAttention


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    model_type: str  # 'tam', 'phi2', 'swa', 'sdpa'
    depth: Optional[int] = None
    lambda_distance: Optional[float] = None
    fusion_mode: Optional[str] = None
    window_size: Optional[int] = None
    backend: str = 'sdpa'
    use_orthogonal_pe: bool = False
    layer_idx: int = 0


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    config: ExperimentConfig
    task: str
    metrics: Dict
    runtime_seconds: float
    peak_memory_mb: Optional[float] = None


class ExperimentRunner:
    """Runs and logs experiments."""

    def __init__(self, output_dir: Path, use_wandb: bool = False):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.results: List[ExperimentResult] = []

        if use_wandb:
            try:
                import wandb
                wandb.init(project='toroidal-attention', dir=str(output_dir))
                self.wandb = wandb
            except ImportError:
                print("W&B not available, logging to JSON/CSV only")
                self.use_wandb = False

    def run_experiment(
        self,
        config: ExperimentConfig,
        task: str,
        device: str = 'cuda',
    ) -> ExperimentResult:
        """Run a single experiment."""
        print(f"\n{'='*60}")
        print(f"Running: {config.name} on {task}")
        print(f"{'='*60}")

        # Load model
        model, tokenizer, phi2_config = load_phi2_model(
            device=device,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        )

        # Configure model
        if config.model_type == 'tam':
            toroidal_attn = ToroidalAttention(
                d_model=phi2_config.hidden_size,
                n_heads=phi2_config.num_attention_heads,
                max_len=2048,
                depth=config.depth or 4,
                lambda_distance=config.lambda_distance or 0.1,
                fusion_mode=config.fusion_mode or 'low_rank',
                window_size=config.window_size,
                backend=config.backend,
                use_orthogonal_pe=config.use_orthogonal_pe,
            )
            replace_attention_layer(model, config.layer_idx, toroidal_attn, copy_weights=False)
        elif config.model_type == 'swa':
            # SWA-only: TAM with window but no distance bias
            toroidal_attn = ToroidalAttention(
                d_model=phi2_config.hidden_size,
                n_heads=phi2_config.num_attention_heads,
                max_len=2048,
                depth=1,  # No depth for SWA-only
                lambda_distance=0.0,
                window_size=config.window_size or 256,
                backend='sdpa',
            )
            replace_attention_layer(model, config.layer_idx, toroidal_attn, copy_weights=False)
        # elif config.model_type == 'phi2': vanilla, no replacement

        # Run evaluation
        start_time = time.time()
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        metrics = self._evaluate_on_task(model, tokenizer, task, device)

        runtime = time.time() - start_time
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == 'cuda' else None

        result = ExperimentResult(
            config=config,
            task=task,
            metrics=metrics,
            runtime_seconds=runtime,
            peak_memory_mb=peak_mem,
        )

        self.results.append(result)

        # Log to W&B
        if self.use_wandb:
            self.wandb.log({
                'experiment': config.name,
                'task': task,
                **metrics,
                'runtime_seconds': runtime,
                'peak_memory_mb': peak_mem,
            })

        # Save incremental results
        self._save_results()

        return result

    def _evaluate_on_task(self, model, tokenizer, task: str, device: str) -> Dict:
        """Evaluate model on a task (placeholder)."""
        # TODO: Implement actual evaluation logic
        # For now, return dummy metrics
        return {
            'perplexity': 0.0,
            'accuracy': 0.0,
            'throughput_tokens_per_sec': 0.0,
        }

    def _save_results(self):
        """Save results to JSON and CSV."""
        # JSON
        json_path = self.output_dir / 'results.json'
        results_dict = [
            {
                'config': asdict(r.config),
                'task': r.task,
                'metrics': r.metrics,
                'runtime_seconds': r.runtime_seconds,
                'peak_memory_mb': r.peak_memory_mb,
            }
            for r in self.results
        ]
        json_path.write_text(json.dumps(results_dict, indent=2))

        # CSV
        csv_path = self.output_dir / 'results.csv'
        if self.results:
            import csv
            with open(csv_path, 'w', newline='') as f:
                # Flatten structure for CSV
                fieldnames = ['name', 'model_type', 'task', 'runtime_seconds', 'peak_memory_mb']
                fieldnames.extend(self.results[0].metrics.keys())

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for r in self.results:
                    row = {
                        'name': r.config.name,
                        'model_type': r.config.model_type,
                        'task': r.task,
                        'runtime_seconds': r.runtime_seconds,
                        'peak_memory_mb': r.peak_memory_mb,
                        **r.metrics,
                    }
                    writer.writerow(row)


def load_experiment_configs(config_path: Path) -> List[ExperimentConfig]:
    """Load experiment configs from YAML."""
    with open(config_path) as f:
        data = yaml.safe_load(f)

    configs = []
    for exp in data.get('experiments', []):
        configs.append(ExperimentConfig(**exp))

    return configs


def main():
    parser = argparse.ArgumentParser(description='Run TAM experiments')
    parser.add_argument('--config', type=str, default='configs/experiments.yaml', help='Experiment config YAML')
    parser.add_argument('--out', type=str, default='results/', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    args = parser.parse_args()

    # Check env gate (unless quick mode)
    if not args.quick and os.getenv('RUN_EXPERIMENTS') != '1':
        print("Experiment runner is env-gated. Set RUN_EXPERIMENTS=1 to run full suite, or use --quick.")
        sys.exit(0)

    # Load configs
    if args.quick:
        # Quick test: single TAM config
        configs = [
            ExperimentConfig(name='tam_quick', model_type='tam', depth=2, lambda_distance=0.1),
            ExperimentConfig(name='phi2_baseline', model_type='phi2'),
        ]
        tasks = ['synthetic_periodic']
    else:
        configs = load_experiment_configs(Path(args.config))
        tasks = ['narrativeqa', 'qasper', 'synthetic_periodic']  # Example tasks

    # Run experiments
    runner = ExperimentRunner(output_dir=Path(args.out), use_wandb=args.wandb)

    for config in configs:
        for task in tasks:
            try:
                runner.run_experiment(config, task, device=args.device)
            except Exception as e:
                print(f"Error in {config.name} on {task}: {e}")
                continue

    print(f"\n{'='*60}")
    print(f"All experiments complete. Results saved to {args.out}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

