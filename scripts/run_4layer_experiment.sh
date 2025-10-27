#!/bin/bash
set -euo pipefail

echo "================================"
echo "4-LAYER TOROIDAL ATTENTION EXPERIMENT"
echo "================================"
echo ""
echo "Configuration:"
echo "  Layers: [0, 8, 16, 24]"
echo "  Depth: 2"
echo "  Lambda: 0.1"
echo "  Dataset: WikiText-2"
echo "  Samples: 200 train, 50 val"
echo "  Epochs: 2"
echo ""

uv run python scripts/finetune_modern.py \
    --dataset wikitext2 \
    --layer_indices "0,8,16,24" \
    --depth 2 \
    --lambda_distance 0.1 \
    --epochs 2 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --warmup_steps 12 \
    --max_length 256 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --device cuda \
    --train_size 200 \
    --val_size 50 \
    --eval_batches 25 \
    --output_dir results/checkpoints/4layer \
    --save_best_only

echo ""
echo "✅ Experiment complete!"
echo "Results saved to: results/checkpoints/4layer/"
