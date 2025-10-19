# Toroidal Attention Mechanism — Evaluation Results

**Version**: 0.1.0  
**Last Updated**: 2025-10-19  
**Status**: In Progress

---

## Executive Summary

This document presents comprehensive evaluation results for the Toroidal Attention Mechanism (TAM) compared to state-of-the-art baselines across multiple dimensions:

1. **Long-Context Understanding**: LongBench and SCROLLS benchmarks
2. **Performance**: Speed and memory efficiency
3. **Ablation Studies**: Impact of depth, lambda, fusion, window, and PE choices

**Key Findings** (to be populated):
- TAM achieves X% perplexity improvement on periodic tasks
- TAM maintains Y% of baseline speed with Z% memory overhead
- Optimal configuration: depth=4, lambda=0.1, fusion=low_rank, orthogonal PE

---

## 1. Long-Context Benchmarks

### 1.1 LongBench Results

LongBench is a comprehensive benchmark for long-context understanding with 21 datasets across 6 categories.

#### Overall Performance

| Model | Avg Score | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code |
|-------|-----------|---------------|--------------|---------------|----------|-----------|------|
| Phi-2 Baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TAM (d=4, λ=0.1) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TAM + Orthogonal PE | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TAM + SWA (w=256) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

#### Per-Task Breakdown

**Single-Document QA**
- NarrativeQA: TBD
- Qasper: TBD
- MultiFieldQA-en: TBD

**Multi-Document QA**
- HotpotQA: TBD
- 2WikiMQA: TBD
- MuSiQue: TBD

**Summarization**
- GovReport: TBD
- QMSum: TBD
- Multi-News: TBD

**Few-shot Learning**
- TREC: TBD
- TriviaQA: TBD
- SAMSum: TBD

**Synthetic Tasks**
- PassageCount: TBD
- PassageRetrieval-en: TBD

**Code Completion**
- LCC: TBD
- RepoBench-P: TBD

---

### 1.2 SCROLLS Results

SCROLLS (Standardized CompaRison Over Long Language Sequences) focuses on 7 diverse long-context tasks.

| Task | Metric | Phi-2 | TAM | TAM + Ortho PE | TAM + SWA |
|------|--------|-------|-----|----------------|-----------|
| QAsper | F1 | TBD | TBD | TBD | TBD |
| Narrative QA | F1 | TBD | TBD | TBD | TBD |
| QuALITY | EM | TBD | TBD | TBD | TBD |
| QMSum | ROUGE-L | TBD | TBD | TBD | TBD |
| SummScreen FD | ROUGE-L | TBD | TBD | TBD | TBD |
| Gov Report | ROUGE-L | TBD | TBD | TBD | TBD |
| Contract NLI | Accuracy | TBD | TBD | TBD | TBD |

---

## 2. Performance Benchmarks

### 2.1 Speed Comparison

**Configuration**: batch=8, seq=256, d_model=512, heads=8, depth=2

#### CPU Performance

| Backend | ms/iter | seq/s | tokens/s | Relative Speed |
|---------|---------|-------|----------|----------------|
| SDPA | 16.2 | 494 | 126k | 1.0x (baseline) |
| Manual (chunked) | 46.6 | 172 | 44k | 0.35x |
| Flash2 | N/A | N/A | N/A | N/A (CPU only) |

#### GPU Performance (A100 40GB)

| Backend | ms/iter | seq/s | tokens/s | Peak Memory (MB) | Relative Speed |
|---------|---------|-------|----------|------------------|----------------|
| SDPA | TBD | TBD | TBD | TBD | 1.0x |
| Manual (chunked) | TBD | TBD | TBD | TBD | TBD |
| Flash2 | TBD | TBD | TBD | TBD | TBD |

**Key Observations**:
- SDPA is the default backend and provides good performance
- Manual chunked path is slower but supports `return_attention=True`
- Flash2 (when available) expected to be 1.5-2x faster than SDPA for large sequences

---

### 2.2 Memory Scaling

**Theoretical Complexity**: O(N²D) for attention matrix, O(N·d) for input/output

| Config | Seq Length | Depth | Peak Memory (MB) | Memory per Token (KB) |
|--------|------------|-------|------------------|-----------------------|
| TAM d=2 | 128 | 2 | TBD | TBD |
| TAM d=2 | 256 | 2 | TBD | TBD |
| TAM d=2 | 512 | 2 | TBD | TBD |
| TAM d=4 | 128 | 4 | TBD | TBD |
| TAM d=4 | 256 | 4 | TBD | TBD |
| TAM d=4 | 512 | 4 | TBD | TBD |
| TAM d=8 | 128 | 8 | TBD | TBD |
| TAM d=8 | 256 | 8 | TBD | TBD |

**Scaling Analysis**:
- Memory grows quadratically with sequence length (as expected)
- Memory grows linearly with depth
- Gradient checkpointing (future work) can reduce memory by ~50%

---

## 3. Ablation Studies

### 3.1 Depth Ablation

**Question**: How does depth (number of platters) affect performance?

| Depth | Perplexity (Periodic) | Accuracy (NarrativeQA) | Speed (ms/iter) | Memory (MB) |
|-------|----------------------|------------------------|-----------------|-------------|
| 1 | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD | TBD |
| 16 | TBD | TBD | TBD | TBD |

**Findings**:
- TBD

---

### 3.2 Lambda (Distance Penalty) Ablation

**Question**: How does the toroidal distance penalty affect performance?

| Lambda | Perplexity (Periodic) | Perplexity (Sinusoidal) | Accuracy (QAsper) |
|--------|----------------------|-------------------------|-------------------|
| 0.0 | TBD | TBD | TBD |
| 0.05 | TBD | TBD | TBD |
| 0.1 | TBD | TBD | TBD |
| 0.2 | TBD | TBD | TBD |
| 0.5 | TBD | TBD | TBD |

**Findings**:
- TBD

---

### 3.3 Fusion Mode Ablation

**Question**: Which depth fusion strategy is most effective?

| Fusion Mode | Params | Perplexity | Accuracy | Speed |
|-------------|--------|------------|----------|-------|
| Mean | 0 | TBD | TBD | TBD |
| Low-rank (r=1) | 2·D·r | TBD | TBD | TBD |
| Low-rank (r=D/4) | 2·D·r | TBD | TBD | TBD |
| Attention | 3·D·D | TBD | TBD | TBD |

**Findings**:
- TBD

---

### 3.4 Sliding Window Ablation

**Question**: Does adding a sliding window improve efficiency without hurting accuracy?

| Window Size | Perplexity | Accuracy | Speed | Memory |
|-------------|------------|----------|-------|--------|
| None | TBD | TBD | TBD | TBD |
| 128 | TBD | TBD | TBD | TBD |
| 256 | TBD | TBD | TBD | TBD |
| 512 | TBD | TBD | TBD | TBD |

**Findings**:
- TBD

---

### 3.5 Positional Encoding Ablation

**Question**: Does orthogonal PE improve performance?

| PE Type | Orthogonality Score | Perplexity | Accuracy |
|---------|---------------------|------------|----------|
| Standard | ~0.90 | TBD | TBD |
| Orthogonal (Gram-Schmidt) | <0.10 | TBD | TBD |

**Findings**:
- TBD

---

## 4. Phi-2 Integration Results

### 4.1 Single-Layer Replacement

**Setup**: Replace layer 0 of Phi-2 with TAM, freeze rest of model

| Config | Perplexity (OWT) | Accuracy (HellaSwag) | Training Time (1 epoch) |
|--------|------------------|----------------------|-------------------------|
| Phi-2 Baseline | TBD | TBD | TBD |
| TAM (d=4, λ=0.1) | TBD | TBD | TBD |
| TAM + Ortho PE | TBD | TBD | TBD |
| TAM + SWA | TBD | TBD | TBD |

---

### 4.2 Multi-Layer Replacement

**Setup**: Replace layers 0-3 of Phi-2 with TAM

| Layers Replaced | Perplexity | Accuracy | Memory Overhead |
|-----------------|------------|----------|-----------------|
| 0 | TBD | TBD | TBD |
| 0-1 | TBD | TBD | TBD |
| 0-3 | TBD | TBD | TBD |
| All (0-31) | TBD | TBD | TBD |

---

## 5. Comparison with State-of-the-Art

### 5.1 Long-Context Methods

| Method | Context Length | Perplexity | Speed | Memory |
|--------|----------------|------------|-------|--------|
| Standard Attention | 2048 | TBD | 1.0x | 1.0x |
| Sliding Window (Longformer) | 4096 | TBD | TBD | TBD |
| Flash Attention v2 | 2048 | TBD | TBD | TBD |
| Ring Attention | 8192+ | TBD | TBD | TBD |
| **TAM (ours)** | 2048 | TBD | TBD | TBD |

---

### 5.2 Efficiency Methods

| Method | Complexity | Memory | Speed | Accuracy |
|--------|------------|--------|-------|----------|
| Standard | O(N²) | O(N²) | 1.0x | 1.0x |
| Linear Attention | O(N) | O(N) | TBD | TBD |
| Sparse Attention | O(N√N) | O(N√N) | TBD | TBD |
| Flash Attention | O(N²) | O(N) | TBD | 1.0x |
| **TAM** | O(N²D) | O(N²D) | TBD | TBD |

---

## 6. Qualitative Analysis

### 6.1 Attention Pattern Visualization

**TODO**: Add attention heatmaps showing:
- Toroidal wrap-around patterns
- Distance bias effects
- Depth fusion behavior

---

### 6.2 Case Studies

**TODO**: Analyze specific examples where TAM excels or struggles:
- Periodic sequences (expected strength)
- Long-range dependencies
- Multi-document reasoning

---

## 7. Limitations and Future Work

### 7.1 Current Limitations

1. **Memory Scaling**: O(N²D) limits maximum sequence length
2. **Training Stability**: Requires careful hyperparameter tuning
3. **Flash2 Integration**: Not yet fully compatible with distance bias
4. **Latent Streaming**: Inference-only in v1, not used in training

---

### 7.2 Future Improvements

1. **Gradient Checkpointing**: Reduce memory by 50%+
2. **Ring Attention**: Enable 10K+ token contexts
3. **FFT Approximation**: O(N log N) speed via convolution
4. **Adaptive Depth**: Learn optimal depth per layer
5. **Multi-Layer TAM**: Replace all Phi-2 layers

---

## 8. Reproducibility

### 8.1 Hardware

- **CPU Benchmarks**: AMD Ryzen 9 5950X, 64GB RAM
- **GPU Benchmarks**: NVIDIA A100 40GB (target)
- **OS**: Linux 6.17.3-3-cachyos

---

### 8.2 Software

- **Python**: 3.10+
- **PyTorch**: 2.0.0+
- **Transformers**: 4.30.0+
- **Flash Attention**: 2.0+ (optional)

---

### 8.3 Commands

```bash
# Run full evaluation suite
RUN_EXPERIMENTS=1 python scripts/run_experiments.py --config configs/experiments.yaml --out results/

# Run specific benchmarks
RUN_LONGBENCH=1 python scripts/eval_longbench.py --model tam --depth 4 --out results/longbench_tam.json
RUN_SCROLLS=1 python scripts/eval_scrolls.py --model tam --depth 4 --out results/scrolls_tam.json

# Performance benchmarks
python scripts/benchmark_backends.py --device cpu --out bench_cpu.json
ENABLE_FLASH2=1 python scripts/benchmark_backends.py --device cuda --out bench_gpu.json
```

---

## 9. Conclusion

**Summary** (to be written after results are collected):
- TAM demonstrates [X]% improvement on [Y] tasks
- Optimal configuration is [Z]
- Trade-offs: [speed/memory/accuracy]
- Recommended use cases: [periodic data, long-context, etc.]

---

## Appendix

### A. Full Configuration Details

See `configs/experiments.yaml` for complete experiment specifications.

### B. Raw Data

All raw results are available in `results/` directory:
- `results.json`: Structured experiment results
- `results.csv`: Flattened results for analysis
- `bench_*.json`: Performance benchmark data

### C. Plots and Visualizations

**TODO**: Add plots for:
- Perplexity vs depth/lambda
- Speed vs sequence length
- Memory vs depth
- Attention pattern heatmaps

