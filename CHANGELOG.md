# Changelog

All notable changes to dirconn are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/); before 1.0 a minor version can change the API.

## [Unreleased]

### Added

- `dirconn-bids`, a command line following the BIDS App convention, at the participant level:
  `dirconn-bids BIDS_DIR OUTPUT_DIR participant [options]` reads the region-level (or sensor-level)
  recordings of a BIDS dataset, prepares them, estimates directed connectivity with the configured
  methods, tests it, runs the controls and writes a derivatives dataset. Entity filters
  (`--participant-label`, `--session-id`, `--task-id`, `--run-id`) and `--bids-filter-file`, which
  is intersected with them; `--config` and `--set KEY=VALUE` overrides; `--work-dir`, `--nprocs`,
  `--omp-nthreads`, `--mem-mb`; optional `bids-validator`; the subcommands `validate-config` and
  `list-methods`. Exit codes 0 (done), 1 (a task failed, the other outputs kept), 2 (the run did
  not start: invalid arguments, configuration, inputs or output location, missing extra; or it
  stopped on a file system error in its own files) and 3 (BIDS validation failed). The group level, `--reports-only` and `--boilerplate-only` arrive in 0.3.0.
  Documentation pages "Command line", "Pipeline configuration" and "Outputs".
- Tutorial 17: a simulated network with a known truth written as a BIDS dataset, eight directed
  methods run on it with one `dirconn-bids` command and a TOML configuration, and the outputs read
  back (edge tables against the truth, diagnostics, run provenance).
- `dirconn.workflows`: the pipeline behind the command. A typed TOML configuration (pydantic,
  frozen and strict about unknown keys; `load_config`, `apply_overrides`, `dump_config`,
  `config_hash`, `analysis_hash`, a commented built-in default; `"null"` turns nullable options
  off; control methods such as `trgc` are refused in `estimation.methods`, and so is the analytic
  test with a frequency-resolved method), discovery of the BIDS inputs with content hashes, and pure steps for preparation (mask
  padding, resampling that guards the filter support around invalid samples and cuts the filter
  transients at the record edges, moving `tmin`, so clean records stay unmasked; z-score, event lock),
  diagnostics, estimation with band reduction and excluded frequencies, significance tests with
  multiple-comparison correction, and the time-reversal control.
- An idempotent task executor: one task per recording and method plus one controls task per
  recording; a task whose result carries its key (input content, analysis hash, method, dirconn
  version) is skipped, so runs can be repeated and resumed; a failing task leaves a crash file and
  does not stop the others. Every run gets its own directory `logs/run-<UTC time>-<pid>/` with the
  log, `progress.jsonl`, crash files, `provenance.json` (command, times, machine, versions, image,
  git revision, input hashes, configuration hashes, resources, task counts),
  `environment-lock.txt` and its effective configuration. A run whose analysis or input
  configuration differs from the one an output dataset was created with is refused. Prepared
  records are cached in the work directory.
- The dirconn derivatives layout, version 1 (`dirconn.io.derivatives`, decision record 0003): per
  result a NetCDF file, a long `_edges.tsv` table with p-values and decisions (header only for
  time-resolved results) and a JSON sidecar (measure, direction, parameters, nodes and their kinds,
  sampling rate, bands, sources, statistics, task key, versions), written last and atomically as
  the mark of a complete result.
- `dirconn.methods.registry`: `REGISTRY`, `get_estimator` and `list_methods`, the fifteen shipped
  estimators by name.
- `dirconn.diagnostics.spectral_range`: `spectral_dynamic_range` and `spectral_range_check`, the
  power ratio between the analysed bands per channel, which flags records on which a VAR fit is
  ill-conditioned. The default Welch segment is capped at the longest run of valid samples, so
  short epochs are used whole.
- `BlockShuffle` supports masked data: blocks are cut inside every run of valid samples, `predict`
  travels with them, and the numbers of blocks kept are recorded in the history. `TRGC` supports
  masks (`reverse_time` reverses them with the samples). In the pipeline, `statistics.surrogate =
  "auto"` picks block shuffling for continuous records and trial shuffling for epoched ones.
- `TimeSeries.from_mne(..., bad_annotations="mask")` and `read_bids(..., bad_annotations="mask")`
  make the samples under `BAD_` annotations invalid (with `nan="mask"`, `Raw` only); the default
  `"ignore"` keeps the earlier behaviour. The pipeline's sensor-level reader uses `"mask"`, and
  `"ignore"` with `prepare.bad_segments = "ignore"`, so the recorded samples enter the estimate.
- `tests/fixtures/bids_tiny`, a tiny synthetic BIDS dataset (five region-level units and one raw
  BrainVision recording, generated by `tools/make_bids_fixture.py`), and a numerical golden
  regression of the pipeline on it (`tools/make_golden.py`).
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
- `dirconn.methods.statespace`: `var_to_state_space`, `reduced_model`, `state_space_gc` and
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
- `dirconn.io.netcdf.write_connectivity` and `read_connectivity` (behind `Connectivity.to_netcdf` and
  `Connectivity.from_netcdf`), and `dirconn.io.mne.timeseries_from_mne` (behind
  `TimeSeries.from_mne`).
- `dirconn.io.simulation_bids`: `write_simulation_bids`, `read_simulation_bids` and
  `SimulationRecord` in a module of their own; `dirconn.io.bids` still provides them.
- `dirconn.viz.plot(conn, kind=...)`, the dispatcher behind `Connectivity.plot`, and
  `dirconn.viz.PLOT_KINDS`.
- `import-linter` in CI (`lint-imports`) checks the dependency direction between the subpackages.
- Decision record 0002 on the package layout.
- The leakage controls, a benchmark add-on (issue #9): how often time-reversed GC, imaginary
  coherence and the phase slope index remove the leakage false positives of conditional GC, and
  how often they add their own. Four more neighbour scenarios (`leakage_adjacent_temporal_lh`,
  `leakage_adjacent_cingulate_lh`, `leakage_adjacent_orbitofrontal_rh`,
  `leakage_adjacent_cingulate_rh`); the protocols `leakage_adjacent_controls` (five geometries, 50
  seeds) and `leakage_pairs_controls` (150 region pairs) with their results; in
  `dirconn.diagnostics.leakage`, `pairs_controls_report` and `adjacent_controls_report` (each
  estimator against the reference on the same simulations) with `adjacent_roles` and
  `is_pair_measure`; `dirconn.stats.counts.wilson_interval`, `dirconn.stats.counts.mcnemar_exact`
  and `dirconn.stats.correction.holm`. The benchmark report has a section for each add-on, and the
  conclusions items 25 to 31. `tools/pick_regions.py leakage` prints the far, quiet region of every
  pair, and `tools/make_leakage_pairs_protocol.py` takes `--name` (a preset) and `--n`,
  `--estimators` and `--budget-hours`; its default output is unchanged.

### Changed

- Pipeline speed: a criterion (`estimation.order = "bic"`, ...) selects the VAR order once per
  recording, and every method with an order, TRGC and the surrogate fits use it (before, each
  method and each surrogate selected again); the tests of all bands share their surrogates,
  so each surrogate is fitted once instead of once per band. The null distribution no longer
  includes the variability of the order selection. New `dirconn.stats.surrogate_test.surrogate_tests`
  and `dirconn.stats.significance.significance_tests` (several reductions, one set of fits).
- Pipeline defaults for the options the library leaves open: `segment_length = 2.0` s for
  mtgc, imcoh and psi on continuous records, `window` 2.0 s (continuous) or 0.5 s (epoched)
  for wdtf; psi takes `estimation.bands` (its tasks failed before).
- The edge table of a time-resolved result (adtf, tpdc, wdtf) has one row per pair, band and
  time point with a `time` column, and keeps the test (before: header only, the test was
  lost). On a continuous record the default block shuffling fits the observed value on blocks,
  which have no time points of the result, so a time-resolved result is not tested there
  (`StatisticsNote` in the sidecar; this also saves its surrogate fits); epoched records are
  tested per time point with trial shuffling.
- `pcmci` and multivariate `te` are tested with their own p-values (PCMCI's MCI test, TE's
  permutation tests; test `"builtin"` in the sidecar) instead of surrogates around the whole
  method. Pairwise and conditional `te` have no test of their own and keep the surrogate test.
- The version is `0.2.0.devN` until the 0.2.0 release; it is part of the pipeline's task keys, of
  the prepared-data cache key and of the `DirconnVersion` of every result. A change to what a
  pipeline step computes bumps `N` (`0.2.0.dev1` for the edge cropping in resampling,
  `0.2.0.dev2` for the shared order selection and surrogates), so results
  and caches written before are recomputed instead of reused.
- `MethodInfo` gains `report_kind` (`"matrix"`, `"spectrum"`, `"time_frequency"` or `"control"`, how
  a report draws the result) and `citations` (bibliography keys); both have defaults, so existing
  estimators keep working.
- New extra `app` (pydantic, joblib and, on Python 3.10, tomli) for the command line and
  `dirconn.workflows`; `all` includes it. Without it `dirconn-bids --version`, `--help` and
  `list-methods` still work. `lint-imports` gains two contracts: the library does not import
  `dirconn.workflows` or `dirconn.cli`, and `dirconn.workflows` does not import `simulation`,
  `validation`, `viz` or `cli`.
- The package is organised in eight subpackages: `dirconn.data`, `dirconn.io`, `dirconn.methods`,
  `dirconn.stats`, `dirconn.diagnostics`, `dirconn.simulation`, `dirconn.validation` and
  `dirconn.viz`. `dirconn.__all__` is unchanged; everything importable from `dirconn` stays there.
- The old top-level module paths (`dirconn.granger`, `dirconn.bids`, `dirconn.plotting`, ... 47 in
  all) are deprecated: they still work, emit a `DeprecationWarning` that names the new module, and
  are removed in dirconn 0.4.0. `dirconn.simulation`, `dirconn.validation` and `dirconn.viz` are
  packages now; the names of the former `dirconn.validation` module moved to
  `dirconn.validation.method_validation` and are forwarded with the same warning.
  `dirconn.datasets` stays available without a warning.
- Every public name of an old module stays reachable through its old path, attribute access
  (`import dirconn; dirconn.mvar.fit_var`) included. Private helpers that moved to another module
  (`dirconn.plotting._select`, `dirconn.simulation._pink_noise`, `dirconn.viz._yeo_table`,
  `dirconn.timeseries._channel_index`, ...) are reachable only at their new homes (for example
  `dirconn.viz.matrices._select`).
- Repository directories: `scripts/` is now `tools/`, `protocols/` is now `validation/protocols/`,
  and `benchmark_results/` is now `validation/results/`. `tests/` mirrors the subpackages.

### Fixed

- `dirconn.diagnostics.leakage.leakage_correlation` ranks |score|, so the signed net measures (TRGC,
  PSI) are read by strength and not by sign. The published results are unchanged: their estimators
  have no negative scores.
- The description of the `leakage_pairs` protocol names `tools/`, where its generator lives.

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
