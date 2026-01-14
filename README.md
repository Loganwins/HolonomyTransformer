# The Holonomy Transformer (HoT)

**A Geometrically-Native Neural Architecture for Consistent Reasoning**

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/holonomy_transformer_paper.pdf)
[![License](https://img.shields.io/badge/License-Custom-orange.svg)](LICENSE)

---

## Abstract

We introduce the **Holonomy Transformer (HoT)**, a novel neural architecture that embeds geometric consistency constraints directly into its computational structure. Unlike standard transformers that represent tokens as flat vectors and compute attention via dot-product similarity, HoT represents tokens as **sections of a principal fiber bundle** and computes attention via **parallel transport costs**.

This architectural choice makes logical inconsistency not merely unlikely but *structurally suppressed*: information cannot flow efficiently along high-holonomy (inconsistent) paths because the geometry itself prevents it.

---

## Key Innovation

**Standard Transformer:**
```
Tokens → Flat Vectors → Dot-Product Attention → Output
```

**Holonomy Transformer:**
```
Tokens → Fiber Bundle Sections → Parallel Transport Attention → Curvature-Gated Flow → Output
           ↑                            ↑                              ↑
    Geometry native            Consistency native              Reasoning native
```

---

## Core Equation

$$\text{Attention}_{ij} \propto \exp\left(\frac{q_i \cdot k_j}{\sqrt{d}} - \lambda \cdot \text{Hol}(i \to j \to i)\right)$$

**Translation:** Pay attention only to positions you can reach *consistently*.

---

## Architecture Components

| Component | Standard Transformer | Holonomy Transformer |
|-----------|---------------------|----------------------|
| **Embedding** | Flat vectors | Fiber bundle sections (base + fiber + connection) |
| **Attention** | Dot-product similarity | Parallel transport holonomy cost |
| **FFN** | Unconstrained flow | Curvature-gated information flow |
| **Structure** | No consistency guarantee | Geometric consistency enforcement |

### 1. Fiber Bundle Embeddings
Each token carries:
- **Base position** (semantic content)
- **Fiber orientation** (reasoning state)
- **Local connection** (geometric relations)

### 2. Parallel Transport Attention
Attention weights determined by holonomy cost:
- Low holonomy → High attention (consistent path)
- High holonomy → Low attention (inconsistent path)

### 3. Curvature-Gated FFN
Information flow gated by local curvature:
- Low curvature → Gate open → Information flows
- High curvature → Gate closed → Information blocked

### 4. Waypoint Crystallization
Stable reasoning anchors emerge naturally at low-curvature positions.

---

## Model Configurations

| Model | d_base | d_fiber | Layers | Heads | Params |
|-------|--------|---------|--------|-------|--------|
| HoT-Small | 512 | 16 | 6 | 8 | ~85M |
| HoT-Base | 768 | 32 | 12 | 12 | ~350M |
| HoT-Large | 1024 | 64 | 24 | 16 | ~1.3B |

---

## Installation
```bash
git clone https://github.com/Loganwins/HolonomyTransformer.git
cd HolonomyTransformer
pip install -r requirements.txt
```

---

## Quick Start
```python
from src.holonomy_transformer import HolonomyTransformer, HoTConfig

# Create model
config = HoTConfig(
    vocab_size=50257,
    d_base=512,
    d_fiber=32,
    n_layers=6,
    n_heads=8,
)
model = HolonomyTransformer(config)

# Forward pass
input_ids = torch.randint(0, 50257, (1, 128))
outputs = model(input_ids, labels=input_ids)

print(f"LM Loss: {outputs['lm_loss']:.4f}")
print(f"Holonomy Loss: {outputs['holonomy_loss']:.4f}")
print(f"Curvature Loss: {outputs['curvature_loss']:.4f}")
```

---

## Why This Matters

**Standard transformers** learn consistency statistically—they see consistent examples and hope to generalize. This means inconsistency is always *possible*.

**The Holonomy Transformer** enforces consistency geometrically—inconsistent reasoning paths have vanishing information flow by construction. This means inconsistency is *structurally suppressed*.

This is not a filter on top of a transformer. This is a new kind of transformer where **the geometry of computation is the consistency mechanism**.

---

## Citation
```bibtex
@article{napolitano2026holonomy,
  title={The Holonomy Transformer: A Geometrically-Native Neural Architecture for Consistent Reasoning},
  author={Napolitano, Logan},
  journal={arXiv preprint},
  year={2026}
}
```

---

## Related Work

- [Holonomy Crushing](https://github.com/Loganwins/Holonomy_Crusher) - Our prior work on decode-time consistency enforcement
- [Geometric Deep Learning](https://arxiv.org/abs/2104.13478) - Bronstein et al.'s foundational survey
- [Gauge Equivariant CNNs](https://arxiv.org/abs/1902.04615) - Cohen et al.'s work on gauge symmetry

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Author

**Logan Napolitano**  
Independent Researcher  
github.com/Loganwins

---

*"Consistency is not a statistical regularity to be learned. It is a geometric property to be enforced."*
