# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-03

### Added

- Core array protocol with backend-agnostic shape and dtype handling.
- Backend dispatcher supporting NumPy (always), JAX (optional), PyTorch (optional).
- Deterministic PRNG keys with split and fold_in semantics.
- Device placement and mixed precision policies.
- Structured logging with JSON and human-readable output.
- Domain-specific error hierarchy.
- Floating-point tolerance helpers.
- Numerically stable linear algebra: pivoted Cholesky, randomised SVD, Krylov solvers.
- Stochastic Lanczos quadrature for log determinants.
- Calculus utilities: hvp, jvp, vjp, Stein operator.
- Exponential family distributions in canonical form.
- Divergence measures: KL, Renyi, Jensen-Shannon, Wasserstein, Sinkhorn, MMD.
- Statistical tests: permutation, MMD, HSIC, KCIT.
- Multiple testing corrections: Benjamini-Hochberg, Storey q-values, e-values.
- PSD kernel protocol with RBF, Matern, polynomial, linear, periodic, spectral mixture.
- Equivariant kernels: permutation, rotation, translation.
- Nystrom approximation and random Fourier features.
- Estimator base with specification septuple and falsification framework.
- GLM, GAM, KernelRidge, gradient boosting, quantile forests, mixture models.
- Bayesian inference: NUTS, SMC, variational inference, normalising flows, Laplace, EP.
- Pathfinder algorithm.
- MCMC diagnostics: R-hat, ESS, SBC, posterior predictive checks.
- Structural causal models with do-calculus identification.
- Causal estimators: AIPW, TMLE, DML, R-learner, X-learner.
- Sensitivity analysis: E-values, Rosenbaum bounds.
- Discovery algorithms: PC, FCI, NOTEARS, DiBS.
- Manifold implementations: Stiefel, Grassmann, SPD.
- Optimal transport: Sinkhorn, sliced Wasserstein, Gromov-Wasserstein, unbalanced OT.
- Fisher information, natural gradient, mirror descent.
- Neural network layers with equivariant and spectral-normalised variants.
- Neural ODE and SDE solvers.
- Conformal prediction: split, jackknife-plus, CV-plus, Mondrian, weighted.
- PAC-Bayes bounds: McAllester, Catoni, Maurer.
- Calibration: temperature scaling, isotonic, Platt, vector scaling.
- Posterior-to-conformal bridge.
- Formal contracts: @requires, @ensures, @invariant.
- Lean 4 theorem card bridge.
- Epistemic assumption registry and reproducibility receipts.
- W3C PROV and OpenLineage provenance graphs.
- Audit replay infrastructure.
- Experimental free-threaded execution paths.
- Online monitoring: drift detectors, coverage trackers.
- scikit-learn compatible pipeline and cross-validation.
- Hyperparameter optimisation: Bayesian, TPE, Hyperband, BOHB.
- Model export to ONNX, TorchScript, StableHLO.
- FastAPI server template.
- Synthetic and real-world dataset loaders.
- Optimisers: Adam, SGD, natural gradient, mirror descent.
- Learning-rate schedulers.
- Public estimator registry with taxonomy.
- CLI with doctor, theory, registry, and replay subcommands.
- Full test suite: unit, property, numerical, contracts, reproducibility, integration.
- MkDocs Material documentation site.
- Ten GitHub Actions CI/CD workflows.
- SLSA level 3 provenance and sigstore signing.
- Trusted Publisher PyPI release pipeline.

[1.0.0]: https://github.com/olaflaitinen/zwingen/releases/tag/v1.0.0
