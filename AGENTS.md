# AI Agent Guide: Toroidal Attention Codebase

**Version**: 0.1.0  
**Last Updated**: October 19, 2025  
**Purpose**: Comprehensive reference for AI agents working on this codebase

---

## 🎯 Executive Summary

This is a **research implementation** of a novel 3D toroidal attention mechanism for large language models. The core innovation is treating transformer context windows as 3D toroidal structures (like HDD platters) rather than linear sequences, enabling:

- **Circular wrapping** to eliminate edge biases
- **Depth stacking** with sparse fusion for efficient long-context modeling
- **Rotational invariance** with proven mathematical properties

**Current Status**: 90% complete - in validation phase. Core implementation is correct; needs testing verification and tooling improvements.

**Primary Goal**: Demonstrate 5-10% perplexity improvement on periodic/long-context tasks vs. standard attention.

---

## 📚 Project Architecture

### High-Level Design Philosophy

This codebase follows these principles:

1. **Mathematical Correctness First**: Implementation directly mirrors the mathematical formalism from the Engineering Design Document (EDD)
2. **Modular Components**: Clean separation between positional encoding, distance metrics, attention, and fusion
3. **Drop-in Replacement**: Toroidal attention can replace any standard attention layer in existing models
4. **Research-Ready**: Designed for experimentation with ablation studies, configurable hyperparameters, and comprehensive testing

### Mathematical Foundation

The core attention mechanism is defined as:

```
Input: X ∈ R^(B×N×d)
1. Reshape to 3D: X_3D ∈ R^(B×N×D×(d/D))
2. Apply 3D RoPE: PE(i,k)_m = sin(2π(i·ω_m + k·φ_m))
3. Compute attention: A_{(i,k)(j,l)} = (Q_{ik} K_{jl}^T / √d_k) - λ·δ((i,k),(j,l))
4. Apply cyclic softmax: P = softmax_c(A)
5. Attend to values: O = P V
6. Low-rank depth fusion: O_fused = fusion(O, Ω)
7. Output: Y ∈ R^(B×N×d)
```

**Key Properties** (from EDD):

- **Lemma 1 (Modular Invariance)**: Attention scores invariant under cyclic shift mod N
- **Lemma 2 (Efficiency Bound)**: Memory scales as O(bch + Dr) per device in ring variant
- **Postulate 1 (Hybrid Topology)**: Toroidal topology enables ring distribution
- **Postulate 2 (Scalable Stacking)**: Depth shards independent intra-platter, fused inter-platter

---

## 🗂️ Directory Structure

```
toroidal-attention/
├── toroidal_attention/           # Core implementation (installable package)
│   ├── __init__.py               # Package exports: ToroidalAttention, etc.
│   ├── core.py                   # Main ToroidalAttention module
│   ├── positional_encoding.py   # 3D RoPE (standard implementation)
│   ├── positional_encoding_orthogonal.py  # Orthogonal RoPE (Gram-Schmidt)
│   ├── distance.py               # Toroidal distance metric
│   └── fusion.py                 # Depth fusion strategies
│
├── scripts/                      # Training & evaluation scripts
│   ├── load_phi2.py              # Phi-2 integration utilities
│   ├── prepare_data.py           # Dataset creation (periodic, sinusoidal, OWT)
│   ├── train_toroidal.py         # Main training script with ablation support
│   ├── evaluate.py               # Evaluation & benchmarking
│   └── validate_implementation.py # Standalone validation (no pytest)
│
├── tests/                        # Comprehensive test suite
│   ├── test_toroidal_attention.py       # Core mechanism tests
│   ├── test_pe.py                       # Positional encoding tests
│   ├── test_distance.py                 # Distance metric tests
│   ├── test_fusion.py                   # Depth fusion tests
│   ├── test_mathematical_correctness.py # Lemma validation
│   ├── test_integration.py              # Phi-2 integration
│   ├── test_performance.py              # Memory & speed benchmarks
│   ├── test_edge_cases.py               # Boundary conditions
│   ├── integration/
│   │   └── test_phi2_integration.py     # End-to-end Phi-2 tests
│   ├── perf/
│   │   └── test_perf_smoke.py           # Quick performance checks
│   └── run_all_tests.py                 # Test runner
│
├── examples/                     # Example usage and comparisons
│   ├── orthogonal_pe_approaches.py      # Compare orthogonalization methods
│   └── test_orthogonal_pe.py            # Orthogonality validation
│
├── configs/
│   └── training_config.yaml      # Hydra-style training configuration
│
├── checkpoints/                  # Model weights (gitignored)
├── logs/                         # Training logs (gitignored)
│
├── main.py                       # Unified CLI entry point
├── pyproject.toml                # Project metadata, dependencies, tool configs
├── uv.lock                       # Locked dependency versions
├── README.md                     # User-facing documentation
├── ANALYSIS_COMPLETE.md          # Implementation status analysis
├── ORTHOGONALITY_ANALYSIS.md     # Orthogonal PE theory & recommendations
├── DEPENDENCY_OPTIMIZATION.md    # Dependency management notes
└── AGENTS.md                     # This file
```

---

## 🔧 Core Components

### 1. `toroidal_attention/core.py` - Main Attention Module

**Class**: `ToroidalAttention(nn.Module)`

**Key Responsibilities**:

- Projects input to Q, K, V (standard attention projections)
- Applies 3D RoPE to queries and keys
- Computes 3D→2D flattened attention over (N×D) × (N×D) matrix
- Applies toroidal distance bias
- Handles three mask types: causal, padding, per-batch
- Performs depth fusion
- Returns output and optional attention weights

**Critical Implementation Details**:

1. **3D to ND Flattening** (Lines ~207-218):

   ```python
   # CORRECT: Flatten depth into sequence dimension for attention
   # Shape: (B, heads, N, D, d_k) → (B, heads, N*D, d_k)
   Q_2d = Q.reshape(B, self.n_heads, N * self.depth, self.d_k)
   K_2d = K.reshape(B, self.n_heads, N * self.depth, self.d_k)
   ```

   - **Why**: Allows standard attention over 4D toroidal space represented as 2D
   - **Gotcha**: DO NOT use `.view()` here; use `.reshape()` for safety

2. **Distance Bias Application** (Lines ~220-240):

   ```python
   # Compute distance bias in 4D: (N, D, N, D)
   dist_bias_4d = create_distance_bias(N, self.depth, self.lambda_distance, device)
   # Flatten to 2D: (N*D, N*D)
   dist_bias_2d = dist_bias_4d.view(N * self.depth, N * self.depth)
   # Apply to attention scores
   attn_scores = attn_scores - dist_bias_2d
   ```

   - **Why**: Penalizes attention to distant positions in toroidal space
   - **Gotcha**: Distance bias computed BEFORE softmax, not after

3. **Masking Priority** (Lines ~240-260):

   ```python
   # Order matters: causal → padding → per-batch
   if causal_mask is not None:
       attn_scores.masked_fill_(causal_mask, float('-inf'))
   if padding_mask is not None:
       attn_scores.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
   if mask is not None:
       attn_scores.masked_fill_(~mask, float('-inf'))
   ```

   - **Why**: Causal must come first; per-batch mask is most specific
   - **Gotcha**: Padding mask needs dimension expansion

**Configuration Constraints**:

- `d_model % n_heads == 0` (standard attention requirement)
- `d_model % depth == 0` (enables clean depth sharding)
- `d_k = d_model // n_heads` (head dimension)
- `d_per_depth = d_model // depth` (dimension per platter)

**Extension Points**:

- Add new fusion modes in `fusion.py`
- Modify distance metric in `distance.py`
- Swap positional encoding (e.g., use orthogonal version)

---

### 2. `toroidal_attention/positional_encoding.py` - 3D RoPE

**Class**: `Toroidal3DPositionalEncoding(nn.Module)`

**Purpose**: Generates orthogonal rotary embeddings for (sequence, depth) coordinates.

**Mathematical Formulation**:

```python
PE(i, k)_m = sin(2π(i·ω_m + k·φ_m)) + cos(2π(i·ω_m + k·φ_m))
```

Where:

- `i ∈ [0, N-1]`: sequence position
- `k ∈ [0, D-1]`: depth layer
- `ω_m`: sequence frequency basis (RoPE-style exponential decay)
- `φ_m`: depth frequency basis (scaled exponential decay)

**Critical Details**:

1. **Frequency Computation**:

   ```python
   # Sequence frequencies (standard RoPE)
   freqs_seq = 1.0 / (base ** (dim_indices / d_model))
   
   # Depth frequencies (scaled to prevent collapse)
   freqs_depth = 1.0 / (base ** (dim_indices / (d_model * depth_scale_factor)))
   ```

   - `base=10000.0` by default (standard RoPE)
   - `depth_scale_factor` prevents dimensional collapse

2. **Caching** (Lines ~121-130):

   ```python
   # Cache sin/cos embeddings up to max_len
   # Shape: (max_len, depth, d_model//2)
   self.register_buffer('sin_cached', sin_emb)
   self.register_buffer('cos_cached', cos_emb)
   ```

   - **Why**: Avoid recomputing PE every forward pass
   - **Benefit**: ~2-3x speedup on repeated sequences

3. **Orthogonality Issue**:
   - **Problem**: Standard implementation achieves ~0.88-0.98 correlation between `ω` and `φ`
   - **Impact**: Depth and sequence dimensions not fully independent
   - **Solution**: Use `positional_encoding_orthogonal.py` (Gram-Schmidt approach)
   - See `ORTHOGONALITY_ANALYSIS.md` for full details

**When to Use Standard vs Orthogonal**:

- **Standard**: Quick experiments, baseline comparisons
- **Orthogonal**: Production use, research claims about 3D structure

---

### 3. `toroidal_attention/distance.py` - Toroidal Distance Metric

**Function**: `create_distance_bias(N, D, lambda_distance, device)`

**Purpose**: Computes distance between all pairs of (i,k) and (j,l) positions in toroidal space.

**Mathematical Formulation**:

```python
δ((i,k), (j,l)) = δ_circular(i, j) + δ_linear(k, l)

where:
  δ_circular(i, j) = min(|i-j|, N-|i-j|) / N    # Wrap-around distance
  δ_linear(k, l) = |k-l| / D                     # Euclidean depth distance
```

**Key Properties**:

1. **Symmetry**: `δ(A, B) = δ(B, A)`
2. **Identity**: `δ(A, A) = 0`
3. **Wrap-around**: `δ(0, N-1) = 1/N` (not N-1)
4. **Bounded**: `δ ∈ [0, 1 + 1] = [0, 2]`

**Output Shape**: `(N, D, N, D)` - full 4D distance tensor

**Usage**:

```python
dist_bias_4d = create_distance_bias(seq_len, depth, lambda_val, device)
# Flatten for attention: (N*D, N*D)
dist_bias_2d = dist_bias_4d.view(seq_len * depth, seq_len * depth)
# Apply to attention scores
attn_scores = attn_scores - dist_bias_2d
```

**Gotchas**:

- Always **subtract** distance bias from attention scores (not add)
- Apply **before** softmax
- `lambda_distance=0.0` disables distance penalty (useful for ablation)

---

### 4. `toroidal_attention/fusion.py` - Depth Fusion

**Class**: `DepthFusion(nn.Module)`

**Purpose**: Fuses information across depth platters after attention.

**Modes**:

1. **Low-Rank** (Recommended):

   ```python
   Ω = U @ V^T
   where U, V ∈ R^(D×r), r << D
   ```

   - **Parameters**: `2 * D * r`
   - **Default rank**: `r = D // 4`
   - **Why**: Efficient, learnable, mimics HDD platter independence

2. **Attention**:

   ```python
   Ω = softmax(Q_depth @ K_depth^T / √D) @ V_depth
   ```

   - **Parameters**: `3 * D * D` (Q, K, V projections)
   - **Why**: Most expressive, allows data-dependent fusion
   - **Gotcha**: Slower and more memory-intensive

3. **Mean**:

   ```python
   Ω = mean(O, dim=depth)
   ```

   - **Parameters**: 0
   - **Why**: Simplest baseline, no learnable parameters
   - **When**: Quick ablations

**Configuration**:

```python
fusion = DepthFusion(
    depth=4,
    d_per_depth=128,
    mode='low_rank',
    rank=1  # or None for depth//4
)
```

---

## 🧪 Testing Strategy

### Test Categories

1. **Unit Tests** (`test_*.py`):
   - Test individual components in isolation
   - Fast execution (<1 second per test)
   - Example: `test_distance.py` validates wrap-around distance

2. **Integration Tests** (`integration/`):
   - Test component interactions
   - Example: `test_phi2_integration.py` validates Phi-2 layer replacement

3. **Mathematical Correctness** (`test_mathematical_correctness.py`):
   - Validates EDD lemmas
   - Example: Rotational invariance (Lemma 1)

4. **Performance Tests** (`perf/`):
   - Memory scaling, speed benchmarks
   - Example: Verify O(N²D) complexity

5. **Edge Cases** (`test_edge_cases.py`):
   - Boundary conditions, error handling
   - Example: Zero-length sequences, singular depth

### Running Tests

```bash
# Quick validation (no pytest needed)
python scripts/validate_implementation.py

# Full test suite
uv run pytest

# Specific category
uv run pytest tests/test_mathematical_correctness.py -v

# With coverage
uv run pytest --cov=toroidal_attention --cov-report=term-missing

# Comprehensive (all test files)
uv run python tests/run_all_tests.py
```

### Critical Test Cases

**Must Pass** (blockers for any changes):

1. ✅ **Rotational invariance**: `shift(input) → shift(output)`
2. ✅ **Distance symmetry**: `δ(i,j) == δ(j,i)`
3. ✅ **Gradient stability**: `|∂L/∂θ| < 10`
4. ✅ **Shape consistency**: All tensors match expected dimensions
5. ✅ **Mask correctness**: Causal mask prevents future attention

**Should Pass** (warnings if fail):

- Orthogonality score < 0.1 (for orthogonal PE)
- Memory scaling within 20% of O(N²D)
- Training convergence on periodic data

---

## 🛠️ Development Workflow

### Toolchain

- **Package Manager**: [uv](https://github.com/astral-sh/uv) (10-100x faster than pip)
- **Testing**: pytest with coverage
- **Linting**: ruff (replaces flake8, black, isort)
- **Type Checking**: Not yet implemented (planned: mypy)
- **Experiment Tracking**: Not yet implemented (planned: W&B)

### Setup

```bash
# Clone and install
git clone <repo-url>
cd toroidal-attention

# Install with development dependencies
uv sync --all-extras

# Verify installation
uv run python scripts/validate_implementation.py
```

### Common Development Tasks

#### 1. Adding a New Feature

Example: Adding a new fusion mode

```bash
# 1. Create feature branch
git checkout -b feature/new-fusion-mode

# 2. Implement in toroidal_attention/fusion.py
# Add new mode to DepthFusion class

# 3. Write unit tests
# Create tests/test_new_fusion.py

# 4. Run tests
uv run pytest tests/test_new_fusion.py -v

# 5. Lint and format
uv run ruff check --fix .
uv run ruff format .

# 6. Run full test suite
uv run pytest

# 7. Update documentation
# Edit README.md, this file (AGENTS.md)

# 8. Commit with conventional commit
git add .
git commit -m "feat(fusion): add hybrid attention-lowrank fusion mode"
```

#### 2. Fixing a Bug

```bash
# 1. Write failing test first
# Edit tests/test_<component>.py to reproduce bug

# 2. Verify test fails
uv run pytest tests/test_<component>.py::test_bugfix -v

# 3. Fix implementation
# Edit toroidal_attention/<component>.py

# 4. Verify test passes
uv run pytest tests/test_<component>.py::test_bugfix -v

# 5. Run full suite
uv run pytest

# 6. Commit
git commit -m "fix(component): resolve edge case with zero depth"
```

#### 3. Running Experiments

```bash
# Quick experiment (periodic data)
uv run python main.py train --dataset periodic --depth 4 --epochs 5

# Full ablation study
uv run python main.py train --ablation

# Custom config
uv run python scripts/train_toroidal.py --config configs/training_config.yaml

# Evaluate results
uv run python main.py eval \
    --checkpoint checkpoints/best_model.pt \
    --config checkpoints/config.json \
    --output results/
```

#### 4. Profiling Performance

```bash
# Memory profiling
python -m memory_profiler scripts/train_toroidal.py

# Speed profiling
python -m cProfile -o profile.stats scripts/train_toroidal.py
python -m pstats profile.stats

# PyTorch profiler
# Add to training script:
# with torch.profiler.profile(...) as prof:
#     train_step()
# prof.export_chrome_trace("trace.json")
```

### Code Quality Checks

```bash
# Run all quality checks before committing
uv run ruff check . && \
uv run ruff format . && \
uv run pytest --cov=toroidal_attention

# Auto-fix common issues
uv run ruff check --fix .

# Check formatting without applying
uv run ruff format --check .
```

---

## 📦 Dependencies

### Core Runtime Dependencies

```toml
torch>=2.0.0              # Core deep learning framework
transformers>=4.30.0      # Hugging Face models (Phi-2)
datasets>=2.12.0          # Dataset loading
numpy>=1.26.0,<2.0.0      # Numerical operations
tqdm>=4.65.0              # Progress bars
pyyaml>=6.0.0             # Config loading
```

### Development Dependencies

```toml
pytest>=7.3.0             # Testing framework
pytest-cov>=4.1.0         # Coverage reporting
ruff>=0.1.0               # Linting & formatting
matplotlib>=3.7.0         # Visualization
tensorboard>=2.13.0       # Training visualization
```

### Training Dependencies

```toml
accelerate>=0.20.0        # Distributed training (Hugging Face)
wandb>=0.15.0             # Experiment tracking (planned)
```

### Version Constraints

- **Python**: >=3.10 (uses match statements, modern type hints)
- **PyTorch**: >=2.0.0 (uses `torch.compile`, improved performance)
- **NumPy**: <2.0.0 (compatibility with torch 2.0)

### Dependency Management

```bash
# Add runtime dependency
uv add torch transformers

# Add dev dependency
uv add --dev pytest ruff

# Add to optional group
uv add --optional training wandb

# Update all dependencies
uv sync --all-extras --upgrade

# Lock dependencies
uv lock

# Verify lockfile
uv sync --frozen
```

---

## 🧩 Integration Points

### Phi-2 Integration

**Purpose**: Demonstrate toroidal attention as a drop-in replacement for standard attention in a production LLM.

**Key Files**:

- `scripts/load_phi2.py`: Utilities for loading and modifying Phi-2
- `tests/integration/test_phi2_integration.py`: Integration tests

**Usage**:

```python
from scripts.load_phi2 import load_phi2_model, replace_attention_layer
from toroidal_attention import ToroidalAttention

# Load Phi-2
model, tokenizer, config = load_phi2_model()

# Create toroidal attention
toroidal_attn = ToroidalAttention(
    d_model=2560,      # Phi-2 hidden size
    n_heads=32,        # Phi-2 attention heads
    max_len=2048,
    depth=4,
    lambda_distance=0.1,
    fusion_mode='low_rank',
)

# Replace layer 0
replace_attention_layer(model, layer_idx=0, toroidal_attn=toroidal_attn)

# Train or evaluate
# See scripts/train_toroidal.py for full training loop
```

**Critical Details**:

1. **Dimension Matching**:
   - Phi-2: `d_model=2560`, `n_heads=32`
   - Ensure `2560 % depth == 0` when choosing depth
   - Suggested depths: 2, 4, 8, 16, 20, 32

2. **Freezing Base Model**:

   ```python
   # Freeze all Phi-2 parameters except toroidal layer
   for param in model.parameters():
       param.requires_grad = False
   for param in model.layers[layer_idx].parameters():
       param.requires_grad = True
   ```

3. **Forward Pass**:
   - Toroidal attention returns tuple: `(output, attn_weights)`
   - Standard Phi-2 attention returns: `output`
   - Handle this in `replace_attention_layer()` wrapper

4. **Masking**:
   - Phi-2 uses causal masking (autoregressive)
   - Toroidal attention must respect this: `causal=True`

---

## 🎓 Mathematical Foundations (for AI Agents)

### Why Toroidal Topology?

**Problem**: Standard attention treats sequences linearly:

- Edge positions have asymmetric context (can't attend backward from position 0)
- Boundary effects hurt performance on periodic tasks
- No inherent notion of "depth" for hierarchical context

**Solution**: Toroidal topology:

- **Circular wrapping**: Position N wraps to position 0
- **Depth stacking**: Context distributed across D vertical platters
- **Uniform attention**: All positions equivalent under rotation

**Analogy**: Hard Disk Drive geometry

- **Tracks**: Circular sequence positions (wrap around)
- **Platters**: Depth layers (stacked vertically)
- **Sectors**: Individual tokens

### Key Mathematical Objects

1. **Toroidal Space**: `T = S¹ × R × Z_D`
   - `S¹`: Circle (sequence positions mod N)
   - `R`: Real line (continuous embedding space)
   - `Z_D`: Discrete depth {0, 1, ..., D-1}

2. **Distance Metric**: `δ: T × T → R⁺`
   - Cyclic in sequence: wraps at boundaries
   - Euclidean in depth: linear separation

3. **Rotary Embedding**: `PE: Z_N × Z_D → R^d`
   - Injective (one-to-one) if orthogonal
   - Periodic in sequence: `PE(i+N, k) = PE(i, k)`
   - Depth-sensitive: `PE(i, k) ≠ PE(i, k')`

### Proving Correctness

When making changes, verify these invariants:

1. **Rotational Invariance**:

   ```python
   for s in range(1, N):
       input_shifted = torch.roll(input, shifts=s, dims=1)
       output_shifted = model(input_shifted)
       expected = torch.roll(model(input), shifts=s, dims=1)
       assert torch.allclose(output_shifted, expected, atol=1e-2)
   ```

2. **Distance Symmetry**:

   ```python
   for i, k, j, l in all_pairs:
       assert abs(distance(i, k, j, l) - distance(j, l, i, k)) < 1e-6
   ```

3. **Gradient Boundedness**:

   ```python
   output = model(input)
   loss = criterion(output, target)
   loss.backward()
   for param in model.parameters():
       assert param.grad.abs().max() < 10.0
   ```

---

## 🚨 Known Issues and Gotchas

### 1. **Orthogonality Correlation**

**Issue**: Standard PE has high correlation (~0.9) between sequence and depth bases.

**Impact**: Depth dimension partially collapses into sequence dimension.

**Solution**: Use `positional_encoding_orthogonal.py` (Gram-Schmidt approach).

**Detection**:

```python
pe = Toroidal3DPositionalEncoding(d_model, max_len, depth)
score = pe.get_orthogonality_score()
if score > 0.1:
    print(f"WARNING: High correlation {score:.3f}")
```

**Reference**: See `ORTHOGONALITY_ANALYSIS.md`

---

### 2. **Memory Scaling**

**Issue**: Attention memory scales as O(N²D), not O(N²).

**Impact**: 4x depth → 4x memory usage.

**Mitigation**:

- Use gradient checkpointing (planned)
- Reduce batch size proportionally
- Use flash attention (future work)

**Example**:

```python
# Standard attention: 8 GB for batch=8, seq=128
# Toroidal (depth=4): 32 GB for same config
# Solution: batch=2, seq=128, depth=4 → 8 GB
```

---

### 3. **Mask Broadcasting**

**Issue**: Padding mask has shape `(B, N)`, but attention is over `(N*D)`.

**Solution**: Expand mask to `(B, 1, 1, N*D)` with proper broadcasting.

**Code** (in `core.py`):

```python
# WRONG: padding_mask.unsqueeze(1)
# RIGHT: padding_mask.unsqueeze(1).unsqueeze(2)
```

---

### 4. **Depth Divisibility**

**Issue**: `d_model % depth != 0` causes shape mismatch.

**Impact**: Runtime error in reshape operations.

**Prevention**: Add assertion in `__init__`:

```python
assert d_model % depth == 0, f"d_model ({d_model}) must be divisible by depth ({depth})"
```

**Valid Configurations** (for Phi-2, d_model=2560):

- ✅ depth ∈ {1, 2, 4, 5, 8, 10, 16, 20, 32, 64, ...}
- ❌ depth ∈ {3, 6, 7, 9, 11, ...}

---

### 5. **Causal Mask Expansion**

**Issue**: Standard causal mask is `(N, N)`, but toroidal attention is `(N*D, N*D)`.

**Solution**: Expand causal mask by repeating for each depth layer.

**Implementation**:

```python
# Create standard causal mask
causal_mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
# Expand to (N*D, N*D)
causal_mask_3d = causal_mask.repeat_interleave(D, dim=0).repeat_interleave(D, dim=1)
```

**Gotcha**: This creates a "layered" causal structure where depth k can attend to all previous depths, not just depth k. If you want per-depth causality, use different logic.

---

## 🔮 Future Work and Extension Points

### Near-Term (Next 1-2 Months)

1. **Gradient Checkpointing**:
   - **Goal**: Reduce memory by 50%+
   - **File**: `toroidal_attention/core.py`
   - **Approach**: Wrap attention computation in `torch.utils.checkpoint.checkpoint()`

2. **Flash Attention Integration**:
   - **Goal**: 2-3x speedup
   - **Library**: `flash-attn`
   - **Challenge**: Adapt to toroidal distance bias

3. **Experiment Tracking (W&B)**:
   - **Goal**: Automatic experiment logging
   - **Files**: `scripts/train_toroidal.py`
   - **Integration**: Add `wandb.init()`, `wandb.log()`

4. **Hydra Configuration**:
   - **Goal**: CLI overrides like `python train.py depth=8`
   - **Files**: `configs/`, `scripts/train_toroidal.py`
   - **Benefit**: Easier hyperparameter sweeps

### Mid-Term (3-6 Months)

1. **Ring Attention Integration**:
   - **Goal**: Enable >10K token contexts
   - **Paper**: Liu et al., "Ring Attention with Blockwise Transformers" (2023)
   - **Approach**: Blockwise toroidal attention with ring all-reduce

2. **Multi-Layer Replacement**:
   - **Goal**: Replace all Phi-2 layers with toroidal attention
   - **Challenge**: Training stability, memory scaling

3. **Long-Context Benchmarks**:
   - **Datasets**: LongBench, SCROLLS, ZeroSCROLLS
   - **Metrics**: Perplexity, accuracy, efficiency

4. **Theoretical Analysis**:
   - **Goal**: Convergence proofs, sample complexity bounds
   - **Approach**: Collaborate with theory researchers

### Long-Term (6-12 Months)

1. **Architecture Extensions**:
   - Other models: Llama, Mistral, GPT-NeoX
   - Vision transformers: Toroidal 2D attention for images
   - Multimodal: Toroidal attention for text+image

2. **FFT Approximation**:
   - **Goal**: O(N log N) speed via convolution theorem
   - **Approach**: Circular convolution in frequency domain

3. **Adaptive Depth**:
   - **Goal**: Learn optimal depth per layer
   - **Approach**: Differentiable architecture search

4. **Production Deployment**:
   - **Goal**: Inference optimization, quantization
   - **Tools**: TorchScript, ONNX, TensorRT

---

## 📐 Code Conventions

### Naming

- **Classes**: PascalCase (`ToroidalAttention`)
- **Functions**: snake_case (`create_distance_bias`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_LEN`)
- **Private**: Leading underscore (`_compute_attention`)

### Tensor Naming

Follow ML conventions for dimension names:

- `B`: Batch size
- `N`: Sequence length
- `D`: Depth (number of platters)
- `d`: Model dimension
- `h`: Number of heads
- `d_k`: Head dimension (`d // h`)

**Example**:

```python
# Input: (B, N, d)
# After 3D reshape: (B, N, D, d/D)
# After head split: (B, h, N, D, d_k)
# After flattening: (B, h, N*D, d_k)
```

### Docstrings

Use Google-style docstrings:

```python
def create_distance_bias(seq_len: int, depth: int, lambda_distance: float, device: torch.device) -> torch.Tensor:
    """
    Compute toroidal distance bias for attention.

    Args:
        seq_len (int): Sequence length N
        depth (int): Number of depth platters D
        lambda_distance (float): Scaling factor for distance penalty
        device (torch.device): Device to create tensor on

    Returns:
        torch.Tensor: Distance bias of shape (N, D, N, D)

    Example:
        >>> bias = create_distance_bias(128, 4, 0.1, torch.device('cuda'))
        >>> bias.shape
        torch.Size([128, 4, 128, 4])
    """
```

### Type Hints

Use type hints for all public functions:

```python
from typing import Optional, Tuple

def forward(
    self,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    return_attention: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Forward pass."""
    ...
```

### Comments

- **Why, not what**: Explain reasoning, not obvious operations
- **References**: Link to papers/equations when implementing algorithms
- **TODO**: Use for future improvements with issue numbers if available

**Good**:

```python
# Apply distance bias BEFORE softmax to penalize distant attention
# (EDD Equation 12)
attn_scores = attn_scores - dist_bias
```

**Bad**:

```python
# Subtract distance bias
attn_scores = attn_scores - dist_bias
```

---

## 🤖 AI Agent Best Practices

### When Making Changes

1. **Read context first**: Understand the mathematical foundation and existing implementation
2. **Write tests first**: Create failing test before fixing bug or adding feature
3. **Verify invariants**: Ensure mathematical properties still hold after changes
4. **Check all test categories**: Run unit, integration, and mathematical correctness tests
5. **Update documentation**: Modify README, AGENTS.md, docstrings as needed

### When Debugging

1. **Check tensor shapes**: Most bugs are shape mismatches

   ```python
   print(f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
   ```

2. **Verify device placement**: Ensure all tensors on same device

   ```python
   assert Q.device == K.device == V.device
   ```

3. **Inspect attention weights**: Visualize attention patterns

   ```python
   import matplotlib.pyplot as plt
   plt.imshow(attn_weights[0, 0].detach().cpu(), cmap='viridis')
   ```

4. **Check gradients**: Look for NaN or exploding gradients

   ```python
   for name, param in model.named_parameters():
       if param.grad is not None:
           print(f"{name}: grad mean={param.grad.mean():.4f}, max={param.grad.max():.4f}")
   ```

### When Extending

1. **Follow existing patterns**: Match style of surrounding code
2. **Add configuration options**: Make new features controllable via config
3. **Provide defaults**: Choose sensible defaults that work out of the box
4. **Write examples**: Add usage examples in docstrings
5. **Benchmark performance**: Profile new code to ensure no regressions

### When Optimizing

1. **Profile first**: Use `cProfile` or `torch.profiler` to find bottlenecks
2. **Benchmark**: Measure before and after with `scripts/benchmark.py`
3. **Verify correctness**: Ensure optimization doesn't change outputs

   ```python
   output_before = model_slow(input)
   output_after = model_fast(input)
   assert torch.allclose(output_before, output_after, atol=1e-4)
   ```

4. **Document trade-offs**: Note memory/speed trade-offs in comments

---

## 📊 Performance Characteristics

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Forward pass | O(B·N²·D·d) | Dominated by attention computation |
| Backward pass | O(B·N²·D·d) | Similar to forward |
| Memory | O(B·N·d + N²·D) | Input + attention matrix |
| Parameters | O(d² + D·r) | Projections + fusion |

### Scaling Behavior

**Sequence Length** (N):

- Memory: Quadratic O(N²)
- Time: Quadratic O(N²)
- Mitigation: Ring attention (future work)

**Depth** (D):

- Memory: Linear O(D) (for attention matrix)
- Time: Linear O(D) (more layers to fuse)
- Parameters: Linear O(D·r) (fusion only)

**Model Dimension** (d):

- Memory: Linear O(d)
- Time: Linear O(d)
- Parameters: Quadratic O(d²) (projections)

### Benchmarks (Approximate)

**Hardware**: A100 40GB GPU, PyTorch 2.0

| Config | Batch | Seq | Depth | Memory | Time/Step | Throughput |
|--------|-------|-----|-------|--------|-----------|------------|
| Small | 16 | 128 | 2 | 8 GB | 50 ms | 320 seq/s |
| Medium | 8 | 256 | 4 | 18 GB | 120 ms | 67 seq/s |
| Large | 4 | 512 | 8 | 35 GB | 400 ms | 10 seq/s |

*Note: Actual values depend on hardware, PyTorch version, and optimization settings.*

---

## 🎯 Common Tasks Reference

### Task: Add a New Fusion Mode

1. Edit `toroidal_attention/fusion.py`:

   ```python
   class DepthFusion(nn.Module):
       def __init__(self, depth, d_per_depth, mode='low_rank', rank=None):
           ...
           if mode == 'my_new_mode':
               self.my_param = nn.Parameter(torch.randn(depth, depth))
   
       def forward(self, x):
           if self.mode == 'my_new_mode':
               return self._fuse_my_new_mode(x)
   ```

2. Add test in `tests/test_fusion.py`:

   ```python
   def test_my_new_mode():
       fusion = DepthFusion(depth=4, d_per_depth=128, mode='my_new_mode')
       x = torch.randn(2, 128, 4, 128)  # (B, N, D, d/D)
       output = fusion(x)
       assert output.shape == (2, 128, 512)  # (B, N, d)
   ```

3. Update config schema in `configs/training_config.yaml`

### Task: Change Positional Encoding

1. Create new PE class in `toroidal_attention/positional_encoding_custom.py`
2. Import in `toroidal_attention/core.py`:

   ```python
   from .positional_encoding_custom import CustomPE
   
   class ToroidalAttention(nn.Module):
       def __init__(self, ..., pe_type='standard'):
           if pe_type == 'custom':
               self.pos_encoding = CustomPE(...)
   ```

3. Add tests comparing outputs with standard PE

### Task: Add a New Distance Metric

1. Edit `toroidal_attention/distance.py`:

   ```python
   def create_custom_distance_bias(seq_len, depth, ...):
       # Implement custom distance logic
       return bias  # Shape: (N, D, N, D)
   ```

2. Update `core.py` to accept distance metric as parameter
3. Validate properties: symmetry, identity, boundedness

### Task: Integrate with a New Model (e.g., Llama)

1. Create `scripts/load_llama.py` (mirror `load_phi2.py`)
2. Handle model-specific details:
   - Attention layer structure
   - Hidden size, number of heads
   - Masking conventions
3. Add integration tests in `tests/integration/`

---

## 🔍 Quick Debugging Guide

### Problem: Shape Mismatch Error

**Symptoms**: `RuntimeError: expected shape [...] but got [...]`

**Check**:

1. `d_model % depth == 0`
2. `d_model % n_heads == 0`
3. Mask shapes match expected dimensions
4. Verify reshape operations use correct dimensions

**Fix**:

```python
# Add assertions
assert x.shape[-1] % self.depth == 0
assert x.shape[-1] % self.n_heads == 0
```

### Problem: NaN Loss During Training

**Symptoms**: Loss becomes NaN after a few steps

**Check**:

1. Gradient clipping enabled: `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`
2. Learning rate not too high (try 1e-5 instead of 1e-4)
3. Distance bias not too large (reduce `lambda_distance`)
4. No division by zero in custom code

**Fix**:

```python
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Reduce lambda
lambda_distance = 0.01  # instead of 0.1
```

### Problem: Out of Memory (OOM)

**Symptoms**: `RuntimeError: CUDA out of memory`

**Check**:

1. Batch size too large
2. Sequence length too long
3. Depth too high (quadratic memory with depth)

**Fix**:

```python
# Reduce batch size
batch_size = 4  # instead of 8

# Or reduce sequence length
seq_len = 128  # instead of 256

# Or reduce depth
depth = 2  # instead of 4

# Or use gradient checkpointing (future work)
```

### Problem: Slow Training

**Symptoms**: Seconds per step >> expected

**Check**:

1. Profiler output (see "Profiling Performance" above)
2. Device placement (CPU vs GPU)
3. Unnecessary tensor copies (.cpu() or .cuda() in hot loop)
4. Inefficient loops (replace with vectorized ops)

**Fix**:

```python
# Use profiler
with torch.profiler.profile() as prof:
    train_step()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Ensure GPU usage
assert next(model.parameters()).is_cuda
```

---

## 📚 Additional Resources

### Internal Documentation

- **README.md**: User-facing overview, installation, quick start
- **ANALYSIS_COMPLETE.md**: Implementation status, validation results
- **ORTHOGONALITY_ANALYSIS.md**: Deep dive on PE orthogonality
- **DEPENDENCY_OPTIMIZATION.md**: Dependency management notes
- **configs/training_config.yaml**: Annotated training configuration

### External References

1. **RoFormer**: Su et al. (2021) - Rotary Position Embedding
   - [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

2. **Ring Attention**: Liu et al. (2023) - Blockwise transformers
   - [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)

3. **Phi-2**: Microsoft (2023) - Small language model
   - [Hugging Face](https://huggingface.co/microsoft/phi-2)

4. **Flash Attention**: Dao et al. (2022) - Fast attention
   - [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

### Useful Commands Cheat Sheet

```bash
# Setup
uv sync --all-extras

# Testing
uv run pytest                                    # All tests
uv run pytest -k "test_rotational"               # Specific test
uv run pytest --cov --cov-report=html            # With coverage
python scripts/validate_implementation.py        # Standalone validation

# Training
uv run python main.py train --dataset periodic --depth 4 --epochs 10
uv run python main.py train --ablation           # Ablation study
uv run python main.py eval --checkpoint <path> --config <path>

# Code Quality
uv run ruff check --fix .                        # Auto-fix linting
uv run ruff format .                             # Format code

# Profiling
python -m cProfile -o profile.stats <script>
python -m memory_profiler <script>

# Dependency Management
uv add <package>                                 # Add dependency
uv add --dev <package>                           # Add dev dependency
uv sync --upgrade                                # Update all
```

---

## 🎓 Learning Path for New AI Agents

If you're a new AI agent encountering this codebase:

1. **Read this file** (AGENTS.md) - 30 minutes
2. **Read README.md** - 15 minutes
3. **Read ANALYSIS_COMPLETE.md** - 20 minutes
4. **Skim key implementation files**:
   - `toroidal_attention/core.py` - Main attention logic
   - `toroidal_attention/positional_encoding.py` - RoPE implementation
   - `toroidal_attention/distance.py` - Distance metric
5. **Run validation script**: `python scripts/validate_implementation.py` - 5 minutes
6. **Read one test file**: `tests/test_mathematical_correctness.py` - 10 minutes
7. **Try a simple change**: Modify `lambda_distance` and retrain - 30 minutes

**Total Time**: ~2 hours to full proficiency

---

## 📝 Version History

- **v0.1.0** (2025-10-19): Initial AGENTS.md creation
  - Core implementation 90% complete
  - Documentation comprehensive
  - Testing infrastructure in place

---

## 🤝 Contributing

When contributing as an AI agent:

1. **Understand the why**: Don't just fix symptoms; understand root causes
2. **Preserve mathematical properties**: Ensure changes maintain EDD lemmas
3. **Test thoroughly**: Run full test suite, not just affected tests
4. **Document decisions**: Explain trade-offs in comments and commit messages
5. **Ask questions**: If uncertain about architecture decisions, flag for human review

---

## 📞 Getting Help

If you encounter issues not covered here:

1. **Check existing issues**: May already be documented
2. **Review test failures**: Often reveal root cause
3. **Consult analysis docs**: ANALYSIS_COMPLETE.md, ORTHOGONALITY_ANALYSIS.md
4. **Examine git history**: See why code was written a certain way
5. **Flag for human review**: Some decisions require domain expertise

---

**End of AGENTS.md**

*This document is intended to be comprehensive. If you find gaps, please extend it!*

### Backend Gating and Streaming

- Flash v2 path is taken only if: `backend='flash2'`, `allow_flash2=True`, `lambda_distance==0.0`, `window_size is None`, `flash_attn` is installed, and manual/return_attention/chunked paths are not required. Otherwise SDPA/manual path is used.
- Sliding Window Attention uses circular wrap over `(N*D)` and is implemented as an additive 2D mask; it disables flash2.
- Streaming API: `ToroidalAttention.forward_streaming(x_t, state)` is available when `latent_cfg` is provided. This maintains a per-head latent KV state for O(1) memory inference. Training continues to use full attention in v1.
