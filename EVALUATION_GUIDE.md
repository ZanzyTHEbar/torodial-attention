# Evaluation Guide — Toroidal Attention Mechanism

**Purpose**: Step-by-step guide for running comprehensive evaluations of TAM vs baselines.

---

## Quick Start

### 1. Enhanced Performance Benchmarks

```bash
# CPU benchmarks (no env gate needed)
uv run python scripts/benchmark_backends.py --device cpu --batch 8 --seq 256 --iters 50 --out bench_cpu.json

# GPU benchmarks with Flash2 (env-gated)
ENABLE_FLASH2=1 uv run python scripts/benchmark_backends.py --device cuda --batch 8 --seq 512 --iters 100 --out bench_gpu.json
```

**Output**: JSON with `ms_per_iter`, `seqs_per_sec`, `tokens_per_sec`, `peak_memory_mb`

---

### 2. LongBench Evaluation

```bash
# Single task (env-gated, requires HF datasets)
RUN_LONGBENCH=1 uv run python scripts/eval_longbench.py \
    --model tam \
    --task narrativeqa \
    --depth 4 \
    --lambda-distance 0.1 \
    --max-samples 100 \
    --out results/longbench_tam_narrativeqa.json

# Baseline
RUN_LONGBENCH=1 uv run python scripts/eval_longbench.py \
    --model phi2 \
    --task narrativeqa \
    --max-samples 100 \
    --out results/longbench_phi2_narrativeqa.json
```

**Tasks**: narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique, gov_report, qmsum, multi_news, trec, triviaqa, samsum, passage_count, passage_retrieval_en, lcc, repobench-p

---

### 3. SCROLLS Evaluation

```bash
# Single task (env-gated)
RUN_SCROLLS=1 uv run python scripts/eval_scrolls.py \
    --model tam \
    --task qasper \
    --depth 4 \
    --max-samples 50 \
    --out results/scrolls_tam_qasper.json

# Baseline
RUN_SCROLLS=1 uv run python scripts/eval_scrolls.py \
    --model phi2 \
    --task qasper \
    --max-samples 50 \
    --out results/scrolls_phi2_qasper.json
```

**Tasks**: qasper, narrative_qa, quality, qmsum, summ_screen_fd, gov_report, contract_nli

---

### 4. Systematic Experiments

```bash
# Quick test (no env gate)
uv run python scripts/run_experiments.py --quick --out results/quick/

# Full ablation grid (env-gated)
RUN_EXPERIMENTS=1 uv run python scripts/run_experiments.py \
    --config configs/experiments.yaml \
    --out results/full/ \
    --device cuda

# With W&B logging
RUN_EXPERIMENTS=1 uv run python scripts/run_experiments.py \
    --config configs/experiments.yaml \
    --out results/full/ \
    --wandb
```

**Output**: `results.json`, `results.csv`, optional W&B dashboard

---

## Evaluation Workflow

### Phase 1: Baseline Performance

1. **CPU Benchmarks** (no GPU needed):

   ```bash
   uv run python scripts/benchmark_backends.py --device cpu --out bench_cpu.json
   ```

2. **Verify Implementation**:

   ```bash
   uv run python scripts/validate_implementation.py
   ```

3. **Quick Synthetic Test**:

   ```bash
   uv run python main.py train --dataset periodic --depth 4 --epochs 5 --out logs/quick/
   ```

---

### Phase 2: Long-Context Benchmarks (GPU Recommended)

1. **LongBench Subset** (narrativeqa, qasper, passage_count):

   ```bash
   for task in narrativeqa qasper passage_count; do
       RUN_LONGBENCH=1 uv run python scripts/eval_longbench.py \
           --model tam --task $task --max-samples 100 \
           --out results/longbench_tam_$task.json
   done
   ```

2. **SCROLLS Subset** (qasper, narrative_qa):

   ```bash
   for task in qasper narrative_qa; do
       RUN_SCROLLS=1 uv run python scripts/eval_scrolls.py \
           --model tam --task $task --max-samples 50 \
           --out results/scrolls_tam_$task.json
   done
   ```

---

### Phase 3: Ablation Studies

1. **Depth Ablation**:

   ```bash
   for depth in 2 4 8; do
       uv run python main.py train \
           --dataset periodic \
           --depth $depth \
           --lambda-distance 0.1 \
           --epochs 10 \
           --out logs/ablation_depth_$depth/
   done
   ```

2. **Lambda Ablation**:

   ```bash
   for lambda in 0.0 0.05 0.1 0.2; do
       uv run python main.py train \
           --dataset periodic \
           --depth 4 \
           --lambda-distance $lambda \
           --epochs 10 \
           --out logs/ablation_lambda_$lambda/
   done
   ```

3. **Fusion Ablation**:

   ```bash
   for fusion in mean low_rank attention; do
       uv run python main.py train \
           --dataset periodic \
           --depth 4 \
           --fusion-mode $fusion \
           --epochs 10 \
           --out logs/ablation_fusion_$fusion/
   done
   ```

---

### Phase 4: GPU Performance (A100 Recommended)

1. **Backend Comparison**:

   ```bash
   # SDPA + manual
   uv run python scripts/benchmark_backends.py --device cuda --out bench_gpu_base.json
   
   # With Flash2
   ENABLE_FLASH2=1 uv run python scripts/benchmark_backends.py --device cuda --out bench_gpu_flash2.json
   ```

2. **Memory Scaling**:

   ```bash
   for seq in 128 256 512 1024; do
       uv run python scripts/benchmark_backends.py \
           --device cuda \
           --seq $seq \
           --batch 4 \
           --out bench_gpu_seq_$seq.json
   done
   ```

---

### Phase 5: Phi-2 Integration (GPU Required)

1. **Single-Layer Replacement**:

   ```bash
   RUN_PHI2_INTEGRATION=1 uv run python tests/integration/test_phi2_integration.py
   ```

2. **Training Run** (requires substantial compute):

   ```bash
   uv run python scripts/train_toroidal.py \
       --model phi2 \
       --replace-layer 0 \
       --depth 4 \
       --lambda-distance 0.1 \
       --dataset openwebtext \
       --epochs 1 \
       --batch-size 4 \
       --out logs/phi2_layer0/
   ```

---

## Results Aggregation

### 1. Collect Raw Results

```bash
# Ensure all results are in results/ directory
ls -lh results/*.json
```

### 2. Generate Summary Tables

```python
# Example Python script to aggregate results
import json
from pathlib import Path
import pandas as pd

results = []
for path in Path('results/').glob('*.json'):
    with open(path) as f:
        data = json.load(f)
        results.append({
            'experiment': path.stem,
            **data.get('metrics', {}),
            'runtime': data.get('runtime_seconds', 0),
        })

df = pd.DataFrame(results)
df.to_csv('results/summary.csv', index=False)
print(df)
```

### 3. Update RESULTS.md

- Fill in TBD entries with actual metrics
- Add plots (matplotlib/seaborn)
- Write analysis and conclusions

---

## Environment Variables Reference

| Variable | Purpose | Default |
|----------|---------|---------|
| `RUN_LONGBENCH` | Enable LongBench evaluation | `0` (disabled) |
| `RUN_SCROLLS` | Enable SCROLLS evaluation | `0` (disabled) |
| `RUN_EXPERIMENTS` | Enable full experiment runner | `0` (disabled) |
| `RUN_PHI2_INTEGRATION` | Enable Phi-2 integration tests | `0` (disabled) |
| `ENABLE_FLASH2` | Enable Flash Attention v2 benchmarks | `0` (disabled) |
| `RUN_PERF_SMOKE` | Enable performance smoke tests | `0` (disabled) |

---

## Troubleshooting

### Issue: Out of Memory on GPU

**Solution**: Reduce batch size or sequence length:

```bash
uv run python scripts/benchmark_backends.py --device cuda --batch 2 --seq 256
```

### Issue: LongBench/SCROLLS datasets not downloading

**Solution**: Check HuggingFace access and internet connection:

```bash
python -c "from datasets import load_dataset; load_dataset('THUDM/LongBench', 'narrativeqa', split='test')"
```

### Issue: Flash2 not available

**Solution**: Install flash-attn (requires CUDA 11.6+):

```bash
uv add flash-attn --extra-index-url https://download.pytorch.org/whl/cu118
```

Or skip Flash2 benchmarks and use SDPA only.

### Issue: Slow CPU benchmarks

**Expected**: CPU is 10-50x slower than GPU for attention. Use smaller configs:

```bash
uv run python scripts/benchmark_backends.py --device cpu --batch 2 --seq 128 --iters 20
```

---

## Next Steps

1. **Run Phase 1-2** to establish baselines
2. **Run Phase 3** to identify optimal hyperparameters
3. **Run Phase 4-5** on GPU for production-scale evaluation
4. **Aggregate results** and update RESULTS.md
5. **Write analysis** and publish findings

---

## Citation

If you use this evaluation framework, please cite:

```bibtex
@misc{toroidal-attention-2025,
  title={Toroidal Attention Mechanism: 3D Structured Context for Long-Range Dependencies},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[your-repo]/toroidal-attention}
}
```
