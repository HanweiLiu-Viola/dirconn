# Changelog

All notable changes to dirconn are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/); before 1.0 a minor version can change the API.

## [Unreleased]

### Added

- `GC(method="state_space")`: Granger causality from the state-space (innovations) form of the VAR
  model (Barnett and Seth 2015). Only the model of all channels is fitted; every reduced model,
  conditional or pairwise, follows from it exactly through one discrete algebraic Riccati equation.
  The refit of a reduced VAR at the same order is biased at short orders because the reduced
  process is VARMA (in a three-oscillator chain about +60 % for one link at the true order 3); the
  state-space value at order 3 agrees with the refit at order 60. The method is recorded in the
  provenance (`details["method"]`); it needs a stable fit and raises when the stationarity check
  fails. The default stays `method="refit"`; `analytic_gc_test` accepts only the refit.
- `SSGC` and `ssgc()`: conditional (or pairwise) spectral Granger causality from the same
  state-space form, `(source, target, freq)`. A parametric alternative to `MTGC` that conditions
  on the other channels and works on short data; its mean over frequency is the time-domain value.
- `dirconn.statespace`: `var_to_state_space`, `reduced_model`, `state_space_gc` and
  `spectral_state_space_gc` for VAR coefficients known in advance.

## [0.1.0] - 2026-11-07

### Added

- First public version. Data structures: `TimeSeries`, `Network`, `Connectivity` with provenance and
  an assumption checklist, NetCDF input and output, region time series in a BIDS-like layout.
- Estimators: conditional and pairwise Granger causality, DTF, PDC, generalised PDC, renormalised PDC
  (`RPDC`); time-reversed Granger causality (`TRGC`), imaginary coherence (`ImCoh`), phase slope index
  (`PSI`); multitaper Granger causality (`MTGC`); PCMCI with the partial-correlation test (extra
  `pcmci`); transfer entropy with the KSG estimator (`TE`, multivariate by default, optional PyTorch
  backend); time-varying `ADTF`, `TPDC` and windowed `WDTF`.
- Statistics: surrogate and permutation tests, asymptotic per-frequency tests of PDC and rPDC,
  multiple-comparison correction, bootstrap intervals, a residual whiteness check on every VAR fit.
- Simulation of networks with a known truth at region, sensor and source level through the head
  models of the MNE `sample` dataset; the benchmark protocols, their saved results and the generated
  report.
- Sixteen tutorials, the documentation site, and a Docker image of the analysis environment.
- A workflow that builds the documentation with every tutorial executed and publishes it to
  GitHub Pages, and a release workflow that publishes to TestPyPI and PyPI with Trusted
  Publishing.

[Unreleased]: https://github.com/HanweiLiu-Viola/dirconn/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/HanweiLiu-Viola/dirconn/releases/tag/v0.1.0
