# Changelog

All notable changes to dirconn are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/); before 1.0 a minor version can change the API.

## [Unreleased]

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
  GitHub Pages.
