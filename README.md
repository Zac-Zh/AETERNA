# Project AETERNA (MVP)

A minimal PyTorch research repo for variable-length packed training with boundary resets and a differentiable VNGE regularizer on sparse graphs.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train (packing + VNGE)

```bash
python experiments/train_lm.py --config configs/aeterna_small.yaml
```

Expected: loss decreases over 200 steps and JSONL logs in `logs/train_lambda_*.jsonl`.

### Packing vs Padding Benchmark

```bash
python experiments/benchmark_packing.py --config configs/aeterna_small.yaml
```

Expected: prints tokens/sec for packed vs padded batches.

### VNGE Validation

```bash
python experiments/validate_vnge.py
```

Expected: prints exact vs estimated VNGE on tiny graphs and confirms gradients are finite.

### Tests

```bash
pytest -q
```

Expected: state contamination test and VNGE exact-small sanity test pass.

## Comparison Artifacts
- `logs/train_lambda_0.0.jsonl` and `logs/train_lambda_0.1.jsonl` include sample JSONL entries for lambda comparisons.
- `logs/benchmark_packing.jsonl` includes a sample packing vs padding throughput record.

## Repo Layout

```
aeterna/
  model/          # AeternaLM + simple recurrent backbone
  ops/            # packed reset utilities
  graphs/         # sparse graph + Laplacian ops
  loss/           # VNGE loss estimator
  data/           # packing and loader
  utils/          # seed, logging, checkpoint
configs/
experiments/
tests/
```

## Notes
- Packing is the default data path; padding is provided only for benchmarking.
- VNGE is computed per sequence from sparse local-window graphs (no dense adjacency).
- The VNGE exact check uses dense eigendecomposition only for tiny graphs (N<=128).
