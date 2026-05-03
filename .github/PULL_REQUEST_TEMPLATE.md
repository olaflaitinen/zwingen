<!--
  Thanks for contributing to zwingen!

  Please complete every section. Sections that do not apply should be marked
  with "N/A" rather than removed, so that reviewers can see the full audit
  trail for the change.

  Read CONTRIBUTING.md and docs/theory/overview.md before opening this PR.
-->

## Summary

<!-- One paragraph: what does this PR change, and why? -->

Fixes # (issue number, if any)
Relates to # (RFC or design proposal, if any)

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New estimator or numerical primitive
- [ ] New uncertainty / calibration component
- [ ] Causal-inference component (identification, estimation, sensitivity)
- [ ] Performance / numerical-stability improvement
- [ ] Documentation / tutorial / theorem-card change
- [ ] Build, packaging, or release-engineering change
- [ ] Breaking change (requires a deprecation cycle and migration note)

## Mathematical specification

<!--
  If this PR adds or modifies an estimator, paste the septuple
  M = (X, Y, H, l, Omega, A, G) here, or link to the section of
  Estimator.specification that captures it.
-->

- Input space `X`:
- Output space `Y`:
- Hypothesis class `H`:
- Loss `l`:
- Regulariser `Omega`:
- Estimation operator `A`:
- Guarantees `G`:

## Assumptions and falsifiers

<!--
  List the named assumptions registered with `zwingen.epistemic` and the
  falsifier attached to each one. If you are introducing a new assumption,
  link to the predicate definition and explain why the falsifier is sound.
-->

- Assumption: `...` — Falsifier: `...`

## Uncertainty quantification

- [ ] Closed-form `(1 - alpha)` confidence set
- [ ] Posterior samples / variational posterior
- [ ] Split / jackknife+ / CV+ conformal interval
- [ ] PAC-Bayes bound
- [ ] `UncertaintyAbsent` with written justification (please attach)

## Numerical contract

- [ ] Analytic baselines added or updated under `tests/numerical/`
- [ ] Tolerance thresholds documented in `docs/reference/tolerances.md`
- [ ] No tolerance has been **tightened** without an accompanying CHANGELOG entry
- [ ] No tolerance has been **loosened** without justification

## Reproducibility

- [ ] Run produces a deterministic receipt (`zw.epistemic.receipt`)
- [ ] Bitwise replay test exercises the new code path
- [ ] Provenance graph (`docs/theory/provenance.md`) is unchanged or updated
- [ ] Receipt schema version: `zwingen.epistemic.RECEIPT_SCHEMA_VERSION = ...`

## Public API impact

- [ ] Public-API surface is unchanged
- [ ] New public symbol(s) added (list below) and documented under `docs/api/`
- [ ] Symbol(s) deprecated with `since=`/`removed_in=` metadata
- [ ] Migration note added under `docs/migrations/`

New or changed symbols:

```

```

## Tests

- [ ] Unit tests under `tests/unit/`
- [ ] Property tests under `tests/property/` (hypothesis)
- [ ] Numerical-regression tests under `tests/numerical/`
- [ ] Contract tests under `tests/contracts/`
- [ ] Integration tests under `tests/integration/`
- [ ] Reproducibility tests under `tests/reproducibility/`

Run locally:

```

uv run pytest -q

uv run pytest tests/numerical -q --runslow

```

## Benchmarks

- [ ] No measurable performance impact expected
- [ ] `asv` benchmark added or updated under `benchmarks/benchmarks/`
- [ ] Benchmark results attached (CPU and, if relevant, GPU/TPU)

## Documentation

- [ ] API docs updated under `docs/api/`
- [ ] Theory docs updated under `docs/theory/`
- [ ] Tutorial / example notebook updated under `docs/tutorials/` or `examples/`
- [ ] Theorem card added under `docs/theory/formal/theorem-cards/`

## Compatibility

Tested on:

- [ ] CPython 3.10
- [ ] CPython 3.11
- [ ] CPython 3.12
- [ ] CPython 3.13
- [ ] CPython 3.13t (free-threaded, advisory)
- [ ] CPython 3.14 / 3.14t (experimental, advisory)
- [ ] CPython 3.15 / 3.15t (experimental, advisory)
- [ ] Linux x86-64 / aarch64
- [ ] macOS x86-64 / arm64
- [ ] Windows x86-64

## Security and provenance

- [ ] No new secrets, tokens, or credentials introduced
- [ ] `gitleaks` passes locally
- [ ] `codeql` passes (or the new alerts have been triaged in the PR description)
- [ ] Wheels still build deterministically (`make verify-build`)

## Checklist

- [ ] I have read `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`
- [ ] I have signed off my commits (DCO) or signed them with my GPG / sigstore key
- [ ] I have added an entry to `CHANGELOG.md`
- [ ] I have added or updated `CITATION.cff` if this change warrants a new release
- [ ] My commits follow the Conventional Commits format
