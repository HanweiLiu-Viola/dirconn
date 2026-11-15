# 0002 — Package layout: eight subpackages and a fixed dependency direction

Date: 2026-11-15. Status: accepted. Revisit when the deprecated module paths are removed (0.4.0).

## Question

Up to 0.1.0 every module of `dirconn` sat directly under `src/dirconn/` (about fifty files: data
structures, readers, estimators, tests, simulation, the benchmark, figures). Nothing said which module
may import which, so the estimators could reach the benchmark code and the data classes the plotting
code. How should the package be organised for 0.2.0 and later, and how do existing imports keep
working?

## Decision

Taken from the 0.2.0 design:

- **D1 — one package, organised in subpackages.** `dirconn` stays a single distribution with a single
  import name; applications built on it may become subpackages later rather than separate packages.
  The library is split into eight subpackages:

  | Subpackage | Contents |
  |---|---|
  | `dirconn.data` | `TimeSeries`, `ChannelMeta`, `Network`/`Edge`, `Connectivity`, `Provenance`, bands, random generators, exceptions |
  | `dirconn.io` | BIDS (`bids`, `simulation_bids`), NetCDF, MNE conversion, `datasets` |
  | `dirconn.methods` | `Estimator`/`MethodInfo` (`base`), the VAR core, every estimator; `methods.timevarying` holds `tvmvar`, `timevarying`, `windowed` |
  | `dirconn.stats` | surrogates, surrogate, analytic and significance tests, bootstrap, contrasts, correction, baseline |
  | `dirconn.diagnostics` | order stability, leakage, region geometry |
  | `dirconn.simulation` | network simulation, scenarios (`scenarios/*.yaml`), head models, inverse, empty-room noise |
  | `dirconn.validation` | protocols, sweeps, benchmark and its report, metrics, the method-validation runner, noise, significance and non-stationary studies |
  | `dirconn.viz` | matrices, spectra, graphs, brain maps, and the `plot()` dispatcher |

- **D4 — one move, with shims.** All modules moved in one change (the 0.2.0 restructure) instead of a
  gradual migration, so the layout is never half old and half new. Every old module path is left
  behind as a three-line shim (see below). The public API, `dirconn.__all__` (78 names), does not
  change and is pinned by `tests/test_public_api.py`.

## Dependency direction

- `dirconn.data` imports no other subpackage.
- `dirconn.methods` does not import `io`, `viz`, `simulation` or `validation`.
- No library subpackage (`data`, `io`, `methods`, `stats`, `diagnostics`, `viz`) imports
  `dirconn.validation`.

`import-linter` enforces these rules (`[tool.importlinter]` in `pyproject.toml`; `lint-imports` runs in
CI next to `ruff` and `mypy`).

Three exceptions are accepted, listed in `ignore_imports`. All three are imports made **inside a
method of a data class, at call time**, so that the convenient method keeps living on the object the
user holds while the code that does the work lives where it belongs:

| Import | Why |
|---|---|
| `dirconn.data.result -> dirconn.viz` | `Connectivity.plot(kind=...)` delegates to the `viz.plot` dispatcher. |
| `dirconn.data.result -> dirconn.io.netcdf` | `Connectivity.to_netcdf` / `from_netcdf` delegate to `write_connectivity` / `read_connectivity`. |
| `dirconn.data.timeseries -> dirconn.io.mne` | `TimeSeries.from_mne` delegates to `timeseries_from_mne`. |

They are repeated in the `methods` contract because `methods` reaches them transitively through
`data`.

## Deprecated module paths

- Each of the 47 old top-level module paths (`dirconn.granger`, `dirconn.bids`, `dirconn.plotting`, …;
  the full map is `MOVED` in `tests/test_shims.py`) is a shim built with `dirconn._compat.redirect`. On
  first import it emits one `DeprecationWarning`, "`<old>` moved to `<new>`; the old path is removed in
  dirconn 0.4.0", and forwards every attribute to the new module.
- **The shims are removed in dirconn 0.4.0**, which leaves 0.2.x and 0.3.x for users to move.
- Private modules (`_csd`, `_ksg`, `_greedy`, `_resampling`) get no shim. `dirconn.plotting` forwards to
  `dirconn.viz`; its private helpers (such as `_select`) now live in `dirconn.viz.matrices` and are no
  longer reachable through the old path.
- `dirconn.simulation`, `dirconn.validation` and `dirconn.viz` are packages now. `dirconn.simulation`
  re-exports the public names of `simulation.simulation`; `dirconn.validation` forwards the names of
  the old module (now `validation.method_validation`) through a module `__getattr__`, with the same
  warning.
- `dirconn.datasets` stays a public alias of `dirconn.io.datasets` without a warning, because the
  tutorials and the error messages of `require_sample()` use that name.

### Guard on the package module

`dirconn/__init__.py` replaces the package's module class with `_Package`, whose `__setattr__`
refuses to rebind a public function to a module. Three public functions share their name with an old
module path: `bootstrap`, `pcmci` and `surrogate_test`. Importing such a path (`import
dirconn.bootstrap`) loads the shim, and Python then binds the submodule as an attribute of the parent
package, which would replace the function `dirconn.bootstrap` with the shim module. Before the move
the real submodule was imported while the package initialised and the function was bound afterwards,
so the function always won; the guard keeps that behaviour.

## Deviation from the design

The design placed `nonstationary` under `methods/timevarying/`. It is an evaluation harness (it scores
the time-varying estimators against a known truth and imports the simulation), not an estimator, and
`methods` may not import `simulation` or `validation`. It lives in `dirconn.validation.nonstationary`.

## Top-level directories

Renamed together with the package:

- `scripts/` → `tools/`
- `protocols/` → `validation/protocols/`
- `benchmark_results/` → `validation/results/`

`tests/` mirrors the subpackages (`tests/data_model`, `tests/io`, `tests/methods`, `tests/stats`,
`tests/diagnostics`, `tests/simulation`, `tests/validation`, `tests/viz`, `tests/tools`); shared
helpers are imported with `from tests.helpers import ...`.

## Consequences

- New code goes into the subpackage that matches its role; `lint-imports` fails when it imports
  against the direction above. A further exception needs an entry in `ignore_imports` and in this
  record.
- Code inside `dirconn` and its tests imports the new paths only (apart from the shim tests); the
  full test suite passes with the shims' warning turned into an error
  (`pytest -W "error:.*moved to:DeprecationWarning"`).
- In 0.4.0 the shims, `MOVED` in `tests/test_shims.py`, the `__getattr__` of `dirconn.validation` and
  the `_Package` guard are removed together.
