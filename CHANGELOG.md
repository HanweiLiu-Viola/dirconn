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
- Per-sample masks on `TimeSeries`: `valid` (gaps and bad segments) and `predict` (the samples that
  contribute a regression row, for example bursts). A VAR row at time `t` is used only when
  `predict[t]` and `valid[t - p .. t]` are all true; order selection, cross-validation, the
  residual whiteness check, the sampling check and the provenance (`n_obs`, `details["masks"]`,
  an input hash that covers the masks) follow the selected rows. `GC`, `DTF`, `PDC`, `GPDC` and
  `RPDC` support masks (`MethodInfo.supports_mask`); every other estimator raises on masked data.
- `TimeSeries.from_array(..., nan="mask")`, `TimeSeries.from_mne(..., nan="mask")` and
  `read_bids(..., nan="mask")` read NaN gaps as invalid samples; `write_roi_bids` writes invalid
  samples as NaN.
- `TrialShuffle`, `bootstrap` and `contrast_test` carry the masks with the trials or blocks; the
  surrogates that move samples within a trial raise on masked data.
- `TimeSeries.lock(events, tmin, tmax)` cuts epochs around events and drops locks that cross the
  record's edges or contain invalid samples; `TimeSeries.control_locks(eligible, n, tmin, tmax,
  seed=...)` draws matched control locks. Both are recorded in `TimeSeries.history`, which every
  estimator copies into the provenance (`preprocessing["data_history"]`).
- Documentation page "Gaps, selected samples and event locks".

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
