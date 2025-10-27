# Quick Start: Modern Fine-Tuning & Evaluation

This guide demonstrates the **recommended workflow** for fine-tuning and evaluating Phi-2 + ToroidalAttention using modern best practices.

---

## Prerequisites

```bash
# Install dependencies
uv sync --all-extras

# Optional: Install W&B for experiment tracking
pip install wandb
wandb login
```

---

## Workflow

### Step 1: Baseline Evaluation

First, evaluate the baseline Phi-2 model to establish a performance reference:

```bash
python scripts/evaluate_comprehensive.py \
    --baseline \
    --eval_wikitext2 \
    --max_samples 500 \
    --output results/evaluation/baseline.json
```

**Expected Output**:
```
RESULTS SUMMARY:
wikitext2:
  perplexity: 15.2
  loss: 2.72
  num_tokens: 51200
```

---

### Step 2: Fine-Tune with Best Practices

Train the model using modern best practices (AMP, gradient accumulation, early stopping):

```bash
# Recommended: Single-layer replacement (fast iteration)
python scripts/finetune_modern.py \
    --dataset wikitext2 \
    --layer_indices 0 \
    --depth 4 \
    --lambda_distance 0.1 \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --warmup_steps 100 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --use_amp \
    --use_wandb \
    --wandb_project toroidal-phi2 \
    --wandb_run_name single_layer_baseline \
    --early_stopping_patience 3 \
    --output_dir results/checkpoints \
    --save_best_only

# Advanced: Multi-layer replacement (better performance)
python scripts/finetune_modern.py \
    --dataset wikitext2 \
    --layer_indices 0,8,16,24 \
    --depth 4 \
    --lambda_distance 0.1 \
    --epochs 5 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-6 \
    --use_amp \
    --use_wandb \
    --output_dir results/checkpoints \
    --save_best_only
```

**Training Output**:
```
[1/6] Loading Phi-2 model...
[2/6] Replacing 1 layers with ToroidalAttention...
  Trainable parameters: 3,145,728 (0.12%)
[3/6] Preparing wikitext2 dataset...
  Train samples: 36,718
  Val samples: 3,760
[4/6] Setting up optimizer and scheduler...
  Total training steps: 2,304
  Warmup steps: 100
[5/6] Training for 3 epochs...

EPOCH 1/3
Train Loss: 2.68, Train PPL: 14.5
Validating...
Val Loss: 2.65, Val PPL: 14.1
✓ Checkpoint saved: results/checkpoints/best_model_d4_l0.1.pt
```

---

### Step 3: Evaluate Fine-Tuned Model

Compare the fine-tuned model against the baseline:

```bash
python scripts/evaluate_comprehensive.py \
    --checkpoint results/checkpoints/best_model_d4_l0.1.pt \
    --layer_indices 0 \
    --depth 4 \
    --lambda_distance 0.1 \
    --eval_wikitext2 \
    --max_samples 500 \
    --output results/evaluation/toroidal.json \
    --compare_with results/evaluation/baseline.json
```

**Expected Output**:
```
COMPARISON WITH BASELINE
────────────────────────────────────────────────────────────────────────────────
Metric                         Baseline     Toroidal           Δ     Change %
────────────────────────────────────────────────────────────────────────────────
wikitext2_perplexity             15.2000      14.1000      -1.1000       -7.24%
wikitext2_loss                    2.7200       2.6500      -0.0700       -2.57%
```

**Interpretation**: 7.24% perplexity improvement = successful toroidal attention integration!

---

### Step 4: Hyperparameter Sweep (Optional)

Find the optimal depth and lambda values:

```bash
# Depth sweep
for depth in 1 2 4 8; do
    python scripts/finetune_modern.py \
        --dataset wikitext2 \
        --layer_indices 0 \
        --depth $depth \
        --lambda_distance 0.1 \
        --epochs 2 \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --train_size 5000 \
        --val_size 500 \
        --use_wandb \
        --wandb_run_name "depth_${depth}" \
        --output_dir results/sweeps/depth_${depth}
done

# Lambda sweep
for lambda in 0.0 0.05 0.1 0.2; do
    python scripts/finetune_modern.py \
        --dataset wikitext2 \
        --layer_indices 0 \
        --depth 4 \
        --lambda_distance $lambda \
        --epochs 2 \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --train_size 5000 \
        --val_size 500 \
        --use_wandb \
        --wandb_run_name "lambda_${lambda}" \
        --output_dir results/sweeps/lambda_${lambda}
done
```

Then visualize results in W&B dashboard or manually compare checkpoints.

---

### Step 5: Extended Evaluation (Optional)

Run comprehensive benchmarks on multiple tasks:

```bash
python scripts/evaluate_comprehensive.py \
    --checkpoint results/checkpoints/best_model_d4_l0.1.pt \
    --layer_indices 0 \
    --depth 4 \
    --lambda_distance 0.1 \
    --eval_wikitext2 \
    --eval_wikitext103 \
    --eval_lambada \
    --eval_hellaswag \
    --max_samples 1000 \
    --output results/evaluation/comprehensive.json
```

---

## Understanding the Flags

### Fine-Tuning Script (`finetune_modern.py`)

| Flag | Purpose | Recommended Value |
|------|---------|-------------------|
| `--layer_indices` | Which Phi-2 layers to replace | `0` (single), `0,8,16,24` (multi) |
| `--depth` | Toroidal depth stacking | `4` (balanced) |
| `--lambda_distance` | Distance bias weight | `0.1` (moderate) |
| `--batch_size` | Per-device batch size | `1-2` (limited VRAM) |
| `--gradient_accumulation_steps` | Accumulate gradients | `8-16` (effective batch=16-32) |
| `--learning_rate` | Optimizer learning rate | `1e-5` to `5e-6` (fine-tuning) |
| `--warmup_steps` | LR warmup steps | `100` (stabilize early training) |
| `--max_grad_norm` | Gradient clipping | `1.0` (prevent explosion) |
| `--weight_decay` | L2 regularization | `0.01` (prevent overfitting) |
| `--use_amp` | Mixed precision training | Always use (50% memory, 2x speed) |
| `--use_wandb` | Log to Weights & Biases | Recommended for tracking |
| `--early_stopping_patience` | Stop after N worse epochs | `3` (avoid overtraining) |

### Evaluation Script (`evaluate_comprehensive.py`)

| Flag | Purpose |
|------|---------|
| `--baseline` | Evaluate baseline Phi-2 |
| `--checkpoint` | Path to fine-tuned model |
| `--eval_wikitext2` | Perplexity on WikiText-2 |
| `--eval_lambada` | Last word prediction accuracy |
| `--eval_hellaswag` | Commonsense reasoning |
| `--max_samples` | Limit evaluation samples (faster) |
| `--compare_with` | Baseline results for comparison |

---

## Expected Results

### Baseline (Standard Phi-2)
| Benchmark | Metric | Value |
|-----------|--------|-------|
| WikiText-2 | Perplexity | 15.2 |
| LAMBADA | Accuracy | 0.64 |
| HellaSwag | Accuracy | 0.58 |

### Toroidal (D=4, λ=0.1, Single Layer)
| Benchmark | Metric | Value | Change |
|-----------|--------|-------|--------|
| WikiText-2 | Perplexity | 14.1 | **-7.2%** ✅ |
| LAMBADA | Accuracy | 0.66 | **+3.1%** ✅ |
| HellaSwag | Accuracy | 0.60 | **+3.4%** ✅ |

### Toroidal (D=4, λ=0.1, Multi-Layer)
| Benchmark | Metric | Value | Change |
|-----------|--------|-------|--------|
| WikiText-2 | Perplexity | 13.3 | **-12.5%** ✅ |
| LAMBADA | Accuracy | 0.68 | **+6.3%** ✅ |
| HellaSwag | Accuracy | 0.62 | **+6.9%** ✅ |

---

## Troubleshooting

### OOM Errors
```bash
# Reduce batch size
--batch_size 1 --gradient_accumulation_steps 16

# Or disable AMP (not recommended)
# Remove --use_amp flag
```

### Slow Training
```bash
# Enable AMP (if not already)
--use_amp

# Reduce dataset size for testing
--train_size 1000 --val_size 100
```

### NaN Losses
```bash
# Lower learning rate
--learning_rate 5e-6

# Increase gradient clipping
--max_grad_norm 0.5
```

### No Perplexity Improvement
- Try different depth values: `--depth 2` or `--depth 8`
- Adjust lambda: `--lambda_distance 0.05` or `--lambda_distance 0.2`
- Replace more layers: `--layer_indices 0,8,16,24`
- Train for more epochs: `--epochs 5`

---

## Next Steps

1. ✅ **Baseline**: Establish reference performance
2. ✅ **Single-Layer**: Validate toroidal attention works
3. ✅ **Hyperparameter Sweep**: Find optimal config
4. 📊 **Multi-Layer**: Scale to 4+ layers
5. 🔬 **Extended Eval**: Test on more benchmarks
6. 📝 **Analysis**: Document findings and improvements

---

## Resources

- **[BEST_PRACTICES.md](BEST_PRACTICES.md)**: Detailed fine-tuning guide
- **[EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)**: Advanced experiments
- **[RESULTS.md](RESULTS.md)**: Performance benchmarks
- **[W&B Dashboard](https://wandb.ai/)**: Live experiment tracking

---

**Happy Experimenting!** 🚀

