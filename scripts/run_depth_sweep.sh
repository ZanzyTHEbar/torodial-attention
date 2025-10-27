#!/bin/bash
set -euo pipefail

echo "========================================"
echo "DEPTH SWEEP: SINGLE-LAYER EXPERIMENTS"
echo "========================================"
echo ""
echo "Testing depths: [4, 8] on layer 0"
echo "Goal: Find optimal depth before multi-layer"
echo ""

# Depth 4
echo "----------------------------------------"
echo "EXPERIMENT 1: Depth 4"
echo "----------------------------------------"
mkdir -p results/checkpoints/depth_sweep/d4

uv run python scripts/finetune_modern.py \
    --dataset wikitext2 \
    --layer_indices "0" \
    --depth 4 \
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
    --output_dir results/checkpoints/depth_sweep/d4 \
    --save_best_only

echo "✅ Depth 4 complete!"
echo ""

# Depth 8
echo "----------------------------------------"
echo "EXPERIMENT 2: Depth 8"
echo "----------------------------------------"
mkdir -p results/checkpoints/depth_sweep/d8

uv run python scripts/finetune_modern.py \
    --dataset wikitext2 \
    --layer_indices "0" \
    --depth 8 \
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
    --output_dir results/checkpoints/depth_sweep/d8 \
    --save_best_only

echo "✅ Depth 8 complete!"
echo ""

# Summary
echo "========================================"
echo "DEPTH SWEEP COMPLETE!"
echo "========================================"
echo ""
echo "Comparing results..."

python3 << 'PYTHON'
import json
import glob

results = []
for depth in [2, 4, 8]:
    log_path = f"results/checkpoints/depth_sweep/d{depth}/train*.log"
    logs = glob.glob(log_path)
    if logs:
        with open(logs[0]) as f:
            for line in f:
                if "Best val PPL" in line:
                    ppl = float(line.split(":")[-1].strip())
                    results.append({"depth": depth, "perplexity": ppl})
                    break

# Add baseline depth=2 from previous run
baseline_log = glob.glob("results/checkpoints/train*.log")
if baseline_log:
    with open(baseline_log[0]) as f:
        for line in f:
            if "Best val PPL" in line:
                ppl = float(line.split(":")[-1].strip())
                results.insert(0, {"depth": 2, "perplexity": ppl})
                break

if results:
    results.sort(key=lambda x: x["perplexity"])
    print("\nDEPTH SWEEP RESULTS (Sorted by PPL):")
    print("="*50)
    for i, r in enumerate(results, 1):
        print(f"{i}. Depth {r['depth']}: PPL = {r['perplexity']:.2f}")
    print("="*50)
    print(f"\n🏆 BEST: Depth {results[0]['depth']} with PPL {results[0]['perplexity']:.2f}")
    
    with open("results/checkpoints/depth_sweep/summary.json", "w") as f:
        json.dump({"results": results, "best": results[0]}, f, indent=2)
    print("\n✅ Results saved to: results/checkpoints/depth_sweep/summary.json")
else:
    print("\n⚠️  No results found")
PYTHON

echo ""
echo "========================================"

