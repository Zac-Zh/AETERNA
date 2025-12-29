# Project AETERNA 🔮

**A**dvanced **E**ntropy-**T**uned **E**fficient **R**ecurrent **N**eural **A**rchitecture

A complete PyTorch research codebase for variable-length packed training with Mamba-2/SSD-style selective state-space models and differentiable Von Neumann Graph Entropy (VNGE) regularization.

## 🎯 Key Features

- **PackMamba**: Variable-length packed training with automatic state resets at sequence boundaries
- **Mamba-2/SSD Blocks**: Selective state-space model layers with efficient recurrence
- **VNGE Regularization**: Sparse graph-based structural regularization using Hutchinson trace estimation and Chebyshev polynomial approximation
- **Sparse Graphs**: O(N·window) complexity graph construction (no dense O(N²) adjacency matrices)
- **Full Reproducibility**: Deterministic packing, pinned seeds, logged metadata
- **Triton/PyTorch Acceleration**: Optimized kernels with automatic fallback

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-org/aeterna.git
cd aeterna

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA toolkit (optional, for GPU acceleration)
- Triton >= 2.0.0 (optional, for optimized kernels)

## 🚀 Quickstart

### 1. Run Tests

Verify the installation by running the test suite:

```bash
pytest -v
```

**Expected output**: All tests pass, including:
- State contamination test (packed sequences don't leak across boundaries)
- VNGE estimator accuracy test (MAE < 1e-2 vs exact)
- Sparse graph scaling test (O(N·window) not O(N²))

### 2. Validate VNGE Estimator

Check VNGE computation on small graphs:

```bash
python experiments/validate_vnge.py
```

**Expected output**:
```
VNGE exact=2.3456 est=2.3489 mae=0.0033
Gradient finite: True
```

### 3. Train Language Model (200 steps)

Run end-to-end training with packed sequences and VNGE:

```bash
python experiments/train_lm.py --config configs/aeterna_small.yaml
```

**Expected behavior**:
- Loss decreases over 200 steps
- JSONL logs written to `logs/train_lambda_0.0.jsonl` and `logs/train_lambda_0.1.jsonl`
- No NaN/Inf losses
- Throughput reported in tokens/sec

### 4. Benchmark Packing vs Padding

Compare throughput and efficiency:

```bash
python experiments/benchmark_packing.py --config configs/aeterna_small.yaml
```

**Expected output**:
```
Packed tokens/sec: 12500.3
Padded tokens/sec: 8200.1
Utilization (packed/padded): 1.52x
```

## 📊 Running Experiments

### Basic Training

Train with custom lambda value:

```bash
python experiments/orchestrator.py --config configs/aeterna_small.yaml --mode train --lambda_vnge 0.1
```

### Lambda Sweep

Run hyperparameter sweep over VNGE regularization strengths:

```bash
python experiments/orchestrator.py --config configs/sweeps/vnge_lambda.yaml --mode sweep
```

This will train with λ ∈ {0.0, 0.01, 0.05, 0.1, 0.2, 0.5} and save results to `logs/sweeps/`.

### Evaluation

Evaluate trained model on validation set:

```bash
python experiments/eval_lm.py --checkpoint checkpoints/step_500_lambda_0.1.pt --split validation
```

## 🏗️ Architecture

### PackMamba: Variable-Length Packed Training

**Problem**: Standard batching pads sequences to uniform length, wasting computation.

**Solution**: Pack multiple variable-length sequences into a single batch and reset hidden states at boundaries.

**Implementation**:
- `sample_indices[t]` tracks which sequence token `t` belongs to
- When `sample_indices[t] != sample_indices[t-1]`, hidden state is reset to zero
- Ensures no information leakage between sequences

**Acceptance Test**: `tests/test_state_contamination.py`
```python
# Pack sequences A and B together: [A... B...]
packed_output = model(packed_tokens, sample_indices)
b_from_packed = packed_output[len(A):]  # Extract B's outputs

# Run B alone
b_alone_output = model(B_tokens, sample_indices_B)

# These MUST match (tolerance: 1e-5 fp32)
assert torch.allclose(b_from_packed, b_alone_output, atol=1e-5)
```

### VNGE: Von Neumann Graph Entropy Regularization

**Goal**: Encourage diverse, well-structured latent representations.

**Computation**:
1. **Sparse Graph Construction** (per sequence):
   - Build local-window graph: connect token i to neighbors i±w
   - Edge weights from cosine similarity or Gaussian kernel
   - Complexity: O(N·window), NOT O(N²)

2. **Normalized Laplacian**:
   - L = I - D^{-1/2} A D^{-1/2}
   - Density matrix: ρ = L / Tr(L)

3. **Entropy Estimation**:
   - H(ρ) = -Tr(ρ log ρ)
   - Hutchinson trace estimator with Rademacher probes
   - Chebyshev polynomial approximation for f(x) = -x log x

**Key Constraint**: Graph construction MUST be sparse. The codebase enforces:
- No dense N×N adjacency matrices allowed (except in tests for N ≤ 512)
- Assertions check edge count: `num_edges < N * window * 3`

### Mamba-2/SSD Blocks

Simplified selective state-space model:
- **State update**: h_t = (I + dt·A)h_{t-1} + dt·B·x_t
- **Output**: y_t = C·h_t + D·x_t
- **Selectivity**: Parameters A, B, C adapt based on input
- **Packed support**: Automatic state reset at boundaries

## 🗂️ Repository Structure

```
aeterna/
  ops/
    pack_reset.py          # Boundary detection utilities
    varlen_scan.py         # Variable-length Mamba scan (reference + optimized)
    triton_kernels.py      # Triton-accelerated kernels (when available)
  layers/
    mamba2_block.py        # Mamba-2/SSD block implementation
    norms.py               # RMSNorm, LayerNorm
  graphs/
    sparse_graph.py        # Local-window sparse graph construction
    laplacian_ops.py       # Normalized Laplacian operations
  loss/
    vnge_loss.py           # VNGE estimator (Hutchinson + Chebyshev)
  data/
    packing.py             # Deterministic packing utilities
    loader.py              # Dataset loading (HuggingFace datasets)
  model/
    aeterna_model.py       # Main language model wrapper
    mamba_backbone.py      # Simple recurrent backbone
  utils/
    seed.py                # Reproducibility (set_seed)
    logging.py             # JSONL logging
    checkpoint.py          # Model checkpointing
    profiling.py           # Timing and memory profiling

configs/
  aeterna_small.yaml       # Default config (200 steps, lambda=0.0/0.1)
  sweeps/
    vnge_lambda.yaml       # Lambda sweep config
    chebyshev_degree.yaml  # Chebyshev degree/probe grid

experiments/
  train_lm.py              # Basic training script
  orchestrator.py          # CLI for sweeps and experiments
  eval_lm.py               # Evaluation script (perplexity, etc.)
  benchmark_packing.py     # Packing vs padding throughput
  validate_vnge.py         # VNGE estimator validation

tests/
  test_state_contamination.py     # PackMamba correctness
  test_vnge_exact_small.py         # VNGE estimator accuracy
  test_varlen_scan_correctness.py  # Varlen scan tests
  test_sparse_graph_scaling.py     # Graph construction scaling
```

## 🔬 How It Works

### Packing with sample_indices

**Packed Batch Format**:
```python
# Three sequences: [5, 3, 7] tokens
input_ids:       [tok0, tok1, ..., tok4, | tok5, tok6, tok7, | tok8, ..., tok14]
sample_indices:  [   0,    0, ...,    0, |    1,    1,    1, |    2, ...,    2]
```

**State Reset Logic** (`aeterna/ops/pack_reset.py`):
```python
reset_mask[0] = True  # Always reset at start
reset_mask[1:] = (sample_indices[1:] != sample_indices[:-1])
```

### VNGE Sparse Graph Construction

**Default**: Local window with radius `w`

For sequence of length N:
- Token i connects to tokens in range [i-w, i+w]
- Edge weights: cosine similarity or Gaussian kernel
- Total edges: ~2·N·w (bidirectional)

**Assertion**: Code enforces no dense adjacency for N > 512

### Hutchinson + Chebyshev VNGE Estimator

**Goal**: Estimate Tr(f(L)) where f(x) = -x log x, L is normalized Laplacian

**Steps**:
1. Map Laplacian eigenvalues [0, 2] to Chebyshev domain [-1, 1]
2. Approximate f using Chebyshev polynomials of degree K
3. Compute coefficients c_k via Chebyshev-Gauss quadrature
4. For each Rademacher probe z ∈ {-1, +1}^N:
   - Compute T_k(L)z via recurrence (no eigendecomposition!)
   - Estimate Tr(f(L)) ≈ (1/n_probes) Σ z^T · Σ c_k T_k(L) z

**Hyperparameters**:
- `cheb_order`: Polynomial degree K (trade accuracy vs compute)
- `num_probes`: Number of random probes (trade variance vs compute)

**Typical values**: K=8-20, probes=4-8

## 🧪 Testing & Validation

### Acceptance Criteria

✅ **1. End-to-End Training**
```bash
python experiments/train_lm.py --config configs/aeterna_small.yaml
```
- Loss decreases monotonically (no divergence)
- Reproducible with fixed seed
- No NaN/Inf for 200+ steps

✅ **2. PackMamba Correctness**
```bash
pytest tests/test_state_contamination.py -v
```
- Packed(A+B) outputs match B-alone outputs (tolerance: 1e-5 fp32)

✅ **3. No Dense O(N²) Adjacency**
```bash
pytest tests/test_sparse_graph_scaling.py -v
```
- Edge count assertion: `num_edges < N * window * 3`
- No OOM for N=2048

✅ **4. VNGE Estimator Sanity**
```bash
pytest tests/test_vnge_exact_small.py -v
python experiments/validate_vnge.py
```
- MAE < 1e-2 vs exact (small graphs)
- Gradients finite and non-zero

### Known Failure Modes & Debugging

**1. VNGE NaN/Inf**
- **Cause**: Chebyshev polynomial overflow, log(0) in entropy
- **Fix**: Reduce `cheb_order` (try 8 instead of 20), increase `eps` (try 1e-6)
- **Debug**: Add gradient clipping, check Laplacian trace

**2. State Contamination**
- **Cause**: Incorrect boundary detection in `sample_indices`
- **Fix**: Verify `reset_state_on_boundaries` logic
- **Debug**: Print `sample_indices` and `reset_mask`, inspect hidden states

**3. OOM on Large Sequences**
- **Cause**: Accidentally creating dense N×N matrix
- **Fix**: Ensure `build_local_window_graph` returns sparse tensor
- **Debug**: Add assertion `assert adj.is_sparse`

**4. Slow Training**
- **Cause**: CPU-only execution, inefficient graph construction
- **Fix**: Use CUDA device, reduce `vnge_window` or `cheb_order`
- **Debug**: Profile with `aeterna.utils.profiling.ProfilerContext`

## 🔄 Reproducibility Checklist

- ✅ **Seeds**: `set_seed(42)` called before dataset loading and training
- ✅ **Dataset Revision**: HuggingFace datasets pinned (can specify `revision` param)
- ✅ **Deterministic Packing**: Same seed → identical `sample_indices`
- ✅ **Config Hashing**: All hyperparameters logged in JSONL
- ✅ **Checkpoint Metadata**: Saves step, optimizer state, RNG states

## 📈 Profiling & Performance

### Timing

```python
from aeterna.utils.profiling import Timer

with Timer("forward_pass"):
    logits = model(input_ids, sample_indices)
```

### Memory

```python
from aeterna.utils.profiling import memory_summary

stats = memory_summary()
print(f"Allocated: {stats['allocated_mb']:.1f} MB")
```

### Nsight Systems (NVIDIA GPUs)

```bash
nsys profile -o aeterna_profile python experiments/train_lm.py --config configs/aeterna_small.yaml
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project builds on ideas from:
- **Mamba** (Gu & Dao, 2023): Selective state-space models
- **PackMamba** (Variable-length packing for efficient training)
- **FINGER** (Hutchinson trace + Chebyshev for graph kernels)
- **VNGE** (Von Neumann graph entropy for representation learning)

---

**Need help?** Open an issue on GitHub or check the tests for usage examples.
