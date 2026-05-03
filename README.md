# zwingen

A mathematically rigorous research library for advanced machine learning.

## Overview

zwingen provides production-grade implementations of machine learning methods
with formal mathematical guarantees, uncertainty quantification, causal inference,
and reproducibility infrastructure. Every estimator exposes its assumptions,
theoretical guarantees, and falsification procedures as first-class objects.

## Key Features

- **Rigorous estimator contracts**: every model declares its hypothesis class,
  loss function, convergence guarantees, and falsifiable assumptions.
- **Bayesian inference**: NUTS, SMC, variational inference, normalising flows,
  Laplace approximation, expectation propagation, pathfinder.
- **Causal inference**: structural causal models, do-calculus, AIPW, TMLE,
  double machine learning, sensitivity analysis, discovery algorithms.
- **Uncertainty quantification**: conformal prediction (split, jackknife-plus,
  CV-plus), PAC-Bayes bounds, calibration diagnostics.
- **Optimal transport**: entropic Sinkhorn, sliced Wasserstein, Gromov-Wasserstein,
  unbalanced OT.
- **Differential geometry**: Stiefel, Grassmann, and SPD manifolds; Fisher
  information and natural gradient; mirror descent on Bregman geometries.
- **Reproducibility**: content-addressed receipts binding code, data, hardware,
  and PRNG seeds; provenance graphs in W3C PROV and OpenLineage; audit replay.
- **Formal verification**: machine-readable proof obligations, Lean 4 theorem
  cards, runtime contract checking.

## Installation

```bash
pip install zwingen
```

With optional dependencies:

```bash
pip install zwingen[jax]       # JAX backend
pip install zwingen[torch]     # PyTorch backend
pip install zwingen[serve]     # FastAPI serving
pip install zwingen[all]       # everything
pip install zwingen[dev]       # development tools
```

## Quick Start

```python
from zwingen.models.linear import LinearRegression
from zwingen.random import key

import numpy as np

rng = key(42)
X = np.random.default_rng(0).normal(size=(100, 3))
y = X @ np.array([1.0, 2.0, 3.0]) + 0.1 * np.random.default_rng(0).normal(size=100)

model = LinearRegression()
model.fit(X, y, rng=rng)

prediction = model.predict(X, return_interval=True)
print(prediction.mean[:5])
print(prediction.lower[:5])
print(prediction.upper[:5])

# Inspect theory
print(model.specification)

# Falsify assumptions on data
report = model.falsify(X, y)
print(report)
```

## CLI

```bash
zw doctor              # check backends and dependencies
zw theory --to latex   # export estimator theory to LaTeX
zw registry list       # browse the estimator registry
zw replay receipt.json # replay a reproducibility receipt
```

## Documentation

- [Getting Started](https://docs.zwingen.dev/quickstart/)
- [API Reference](https://docs.zwingen.dev/api/)
- [Mathematical Foundations](https://docs.zwingen.dev/theory/)
- [Contributing](https://docs.zwingen.dev/contributing/)

## Citation

If you use zwingen in your research, please cite:

```bibtex
@software{zwingen,
  author = {Laitinen-Fredriksson Lundstroem-Imanov, Gustav Olaf Yunus},
  title = {zwingen: A Mathematically Rigorous Research Library for Advanced Machine Learning},
  version = {1.0.0},
  year = {2026},
  url = {https://github.com/olaflaitinen/zwingen},
  license = {Apache-2.0},
}
```

## License

Apache-2.0. See [LICENSE](LICENSE) for the full text.
