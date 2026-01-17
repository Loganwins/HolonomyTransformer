# CF-HoT: Adaptive Repetition Suppression

**A working system for reducing LLM repetition via learned risk prediction**

## Results

| Metric | Baseline | CF-HoT | Change |
|--------|----------|--------|--------|
| Repetition Rate | 33.9% | 17.5% | **-48%** |
| Distinct-2 | 0.836 | 0.976 | **+17%** |

## What This Actually Is

This is **NOT** the full Lie Holonomy Transformer as described in the theoretical paper. After 5 failed attempts at attention gating, we pivoted to a simpler approach that works:

1. **Train a risk predictor** — Learns to predict token repetition from hidden states (F1 > 0.96)
2. **Apply at decode-time** — Penalize recent tokens when risk is high
3. **Base model unchanged** — No attention modification, no training/inference mismatch

## Quick Start
```bash
# Install
pip install torch transformers peft datasets bitsandbytes

# Train risk predictor (~1 hour on RTX 3090)
python training/cfhot_risk_v2.py

# Run guided generation
python cfhot_guided_generation.py
```

## How It Works
```python
# At each generation step:
risk = risk_predictor(hidden_states)  # 0.0 to 1.0

if risk > 0.1:
    recent_tokens = input_ids[-32:]
    logits[recent_tokens] -= risk * penalty_scale
```

High risk → suppress recent tokens → model forced to use different words.

## Files

| File | Purpose |
|------|---------|
| `training/cfhot_risk_v2.py` | Train risk predictor (Phase 1) |
| `cfhot_guided_generation.py` | Decode-time generation (Phase 2) |
| `cfhot_development_log.md` | Full documentation of what worked and what didn't |

## What We Proved

✅ Hidden states predict repetition (F1 > 0.96)  
✅ Decode-time intervention reduces repetition (48%)  
✅ Output quality preserved  

## What We Did NOT Prove

❌ Attention gating works on pretrained models (it doesn't)  
❌ The geometric interpretation (holonomy, fiber bundles) is meaningful  
❌ This outperforms tuned standard repetition penalty  

## Development Log

See `cfhot_development_log.md` for the full journey:
- The passive hook bug discovery
- 5 failed attention gating attempts
- The pivot to supervised learning
- Final working system

## Citation
```bibtex
@misc{cfhot2026,
  author = {Napolitano, Logan Matthew},
  title = {CF-HoT: Adaptive Repetition Suppression via Learned Risk Prediction},
  year = {2026},
  url = {https://github.com/Loganwins/HolonomyTransformer}
}
```

## License

MIT
