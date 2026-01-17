# CF-HoT Training Pipeline

## Overview

Two-phase training system for Control Field Holonomy Transformers:

- **Phase A**: Train small standalone CF-HoT (~50M params) to validate architecture
- **Phase B**: Inject CF-HoT adapters into existing large models (e.g., Llama-8B)

## Quick Start

### Phase A: Architecture Validation

```bash
# Basic run
TOKENIZERS_PARALLELISM=false python phase_a_validation.py \
    --output_dir ./results/phase_a \
    --batch_size 16 \
    --max_steps 10000 \
    --lr 1e-4

# With baseline comparison
TOKENIZERS_PARALLELISM=false python phase_a_validation.py \
    --output_dir ./results/phase_a \
    --batch_size 16 \
    --max_steps 10000 \
    --lr 1e-4
    # (remove --no_baseline flag)
```

### Phase B: 8B Model Adaptation

```bash
python phase_b_8b_adapters.py \
    --model_path /path/to/llama-8b \
    --output_dir ./results/phase_b \
    --batch_size 2 \
    --max_steps 5000
```

## Hardware Requirements

| Phase | GPU Memory | Time (RTX 3090) |
|-------|------------|-----------------|
| Phase A | ~14GB | ~45 min |
| Phase B | ~20GB | ~6-8 hours |

## Key Hyperparameters

### Phase A (Small Model)
```python
d_model = 512       # Hidden dimension
n_layers = 8        # Transformer layers
d_fiber = 32        # Consistency subspace
d_control = 64      # Predictor hidden
momentum = 0.9      # EMA decay
```

### Phase B (Adapters)
```python
d_fiber = 16        # Keep small for adapters
d_control = 64      # Predictor hidden
lambda_hol = 1e-4   # Weak regularization
```

## Critical Notes

1. **Clamping**: Delta_h and field are clamped to prevent explosion
2. **Learning rate**: Use 1e-4, not 3e-4 (prevents NaN)
3. **TOKENIZERS_PARALLELISM**: Set to false to avoid DataLoader hangs

## Output Files

```
results/
├── phase_a/
│   ├── cf_hot_step_*.pt      # Checkpoints
│   └── phase_a_results.json  # Final metrics
└── phase_b/
    └── cf_adapter_final.pt   # Trained adapters
```

## Success Criteria

| Metric | Good | Bad |
|--------|------|-----|
| CF-HoT PPL vs Baseline | Lower | Higher |
| Gate values | 0.2-0.8 | Near 0 or 1 |
| Holonomy | Stable, non-zero | Exploding or zero |
