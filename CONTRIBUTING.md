# Contributing to dirconn

`dirconn` estimates *predictive* directed connectivity from neurophysiology signals. The conventions
below hold for all of it; decisions with a lasting effect are recorded in
[docs/decisions](docs/decisions).

## Questions, problems and proposals

- **Report a bug** or a result that looks wrong by opening an issue on GitHub
  (<https://github.com/HanweiLiu-Viola/dirconn/issues>). Include the dirconn version, the Python and
  numpy versions, a short script that shows the problem (simulated data are best), and what you
  expected instead.
- **Ask a question** about using dirconn in an issue as well; please say what you tried.
- **Propose a change** (a new method, a new default, a change of convention) in an issue first, so
  that it can be discussed before you write code. Then open a pull request against `main`; it needs
  the checks below to pass and, for a new method, the tests and validation described in
  [Adding a method](#adding-a-method).

## Set up

Use a conda environment (the project is developed with Python 3.11; 3.10 to 3.13 are supported):

```bash
conda create -n dirconn-dev -c conda-forge python=3.11 numpy scipy pandas xarray netcdf4 matplotlib mne mne-bids pyyaml jupyterlab pytest ruff mypy
conda activate dirconn-dev
pip install -e ".[dev,all]"
```

The optional extras are `connectivity` (`mne-connectivity`), `pcmci` (`tigramite`), `stats`
(`statsmodels`), `parallel` (`joblib`), `speed` (`numba`) and `plot` (`matplotlib`). `all` installs them. The `dev`
extra includes `tigramite` as well, because tutorial 11 and the PCMCI tests need it.

## Checks

All of these must pass before a change is committed; CI runs the same commands on Windows, Linux and
macOS with Python 3.10 to 3.13.

```bash
ruff format .
ruff check .
mypy
pytest
```

`pytest` turns `RuntimeWarning` into an error, so numerical warnings cannot slip through. Tests marked
`sample_data` need the MNE sample dataset and never run by default.

## Conventions

### Code

- Python 3.10+, line length 100, English docstrings in numpydoc style, type hints on public code.
- Estimators are **frozen dataclasses** deriving from `dirconn.Estimator`, with typed parameters, a
  class attribute `info` (a `MethodInfo`) and a `_fit` method. `fit(data)` returns a `Connectivity`
  and keeps no state. Unknown keyword arguments raise `TypeError` for free.
- **Names are short acronyms**: the class is the upper-case acronym (`GC`, `TE`, `DTF`, `PDC`),
  `MethodInfo.name` (and so `provenance.method`) is the lower-case acronym (`"gc"`), and a
  lower-case convenience function does `gc(data, **params)` = `GC(**params).fit(data)`.
- Results always have the dimensions `(source, target, freq, time)`, with NaN coordinates on axes a
  method does not resolve. Never let the shape depend on a parameter.
- **Direction is `source -> target`** everywhere: `values[i, j]` is the influence of `i` on `j`.
  VAR coefficients use the usual `[lag - 1, target, source]` layout. Papers often index `i = target`;
  transpose on purpose and say so in a comment.
- Every result carries a `Provenance` (method, parameters, seed, input hash, backend, versions,
  preprocessing, assumption checklist). Put the implementation actually used in `MethodInfo.backend`;
  a method name must not hide what is computed.
- Never choose a scientific default silently: no default frequency band, no hidden filtering. If a
  default matters scientifically, make it an explicit parameter and record it.

### Failing and warning

- **Failures raise.** Do not use a bare `except`, do not catch an error to return NaN or a fallback
  model, and do not swallow unknown arguments. A silent fallback (for example refitting with a lower
  order, or replacing a failed inverse by a pseudo-inverse) is a bug.
- **Diagnostics warn and are recorded.** Assumption checks warn with `dirconn.DirconnWarning` and are
  written into the result's assumption checklist; they never stop a run. Input errors (wrong number
  of channels, misaligned delays) raise `ValueError`.
- Undefined entries (such as the diagonal) are NaN, excluded from statistics families, and never
  counted as detections.
- Metrics that are mathematically undefined return NaN with a warning, never 0.

### Tests

Write the test first and watch it fail.

- Prefer **closed forms and independent references**: a hand-picked VAR whose DTF, PDC or Granger
  value is known exactly, or a short reference implementation written from the definition. Do not
  validate a formula with the code under test.
- **Cross-check against other libraries** where one exists (`statsmodels`, `tigramite`), with
  explicit tolerances.
- **Check outputs are finite.** Numerical bugs rarely raise; they return plausible numbers.
- Stochastic assertions use explicit seeds and, where an outcome depends on the draw, a **median over
  several seeds**, never one lucky seed. Chaotic simulators change with library versions, so quote
  numbers together with the environment.
- Property tests for invariances: channel permutation, scale invariance, time reversal.
- Platform-bound tests (fonts, paths) carry a `skipif`.
- Tests must not touch the network or download data.

### Reproducibility

- Seeds are explicit and required wherever randomness is used; derive per-task generators from one
  `SeedSequence` so results do not depend on the number of workers.
- Benchmark runs set `OMP_NUM_THREADS=1`, record library versions in the provenance, and published
  numbers use the exact versions of `constraints/benchmark.txt`. CI deliberately pins nothing
  (`constraints/ci.txt` is empty on purpose): its matrix covers several Python and numpy versions to
  find what depends on one. A reported number without its environment is not a result.
- The Docker image (`Dockerfile`) installs with `constraints/docker.txt`, which repeats the
  benchmark pins and adds the rest of the stack; a test keeps the shared pins equal. Push only
  images built from a clean commit (`scripts/build_image.py --push` refuses otherwise).
- `__version__` is read from the package metadata and is never hard-coded.

## Tutorials

The tutorials are the scripts `examples/plot_*.py`: plain Python with `# %%` cells (the sphinx-gallery
layout), so they diff cleanly and run with `python`. Notebooks are generated for viewing and are never
committed: `python scripts/make_notebooks.py` writes them to `docs/_build/notebooks` (needs `nbformat`).

Every tutorial follows the same rules, and `tests/test_examples.py` checks them and runs the script:

- The module docstring is an RST title (`=====` above and below) and a few sentences for the reader.
- The first cell sets `QUICK = os.environ.get("DIRCONN_QUICK") == "1"` (the tests set it; readers get the full
  version, so `QUICK` only shortens data and the number of seeds), sets `SEED = <int>` and prints
  `dirconn.__version__` and `np.__version__`.
- It uses the public API only and does not quote benchmark numbers; a comparison inside a tutorial is a
  demonstration with one or two seeds and says so. Measured numbers live in `docs/benchmark_results.md`.
- It ends with a cell headed "What this does and does not show": the result is predictive and not causal,
  the assumptions are printed with `for check in result.provenance.assumptions.checks`, and what the
  tutorial did not test is listed.
- A tutorial that needs the `sample` dataset has the line `NEEDS_SAMPLE = True` and calls
  `dirconn.datasets.require_sample()`; the tests mark it `sample_data` and skip it when the data are missing.

Figures come from `Connectivity.plot(...)`, MNE or a published package; do not copy plotting code from other
repositories.

Scripts named `examples/view_*.py` open interactive windows (for example the 3D brain maps of tutorial 8). They are not
tutorials: the tests only check that they compile, and the documentation build ignores them.

## Documentation

The documentation (`docs/index.rst`, `docs/api.rst`, and the tutorial gallery that sphinx-gallery builds from
`examples/`) is published at <https://hanweiliu-viola.github.io/dirconn/>. To build it locally:

```bash
pip install -e ".[docs,plot,plot-extra]"
python -m sphinx -b html docs docs/_build/html          # open docs/_build/html/index.html
```

Every tutorial except 7, 8, 13 and 14 is run during the build (about ten minutes; `DIRCONN_QUICK=1` shortens the simulations for a quick
look). Tutorials 7, 8, 13 and 14 need the `sample` dataset (7 and 8 also MNE's 3D plotting; 13 runs for about an hour and 14
for about ten minutes in their full versions): they are run when the dataset is already on the
machine (the build never downloads it) and shown without output otherwise; `DIRCONN_DOCS_SAMPLE=1` forces them to run
and `DIRCONN_DOCS_SAMPLE=0` to be skipped (use it when MNE's 3D plotting does not work on the machine). The generated folders `docs/auto_tutorials` and `docs/generated` are ignored by git.
`tests/test_docs_build.py` builds a copy of the documentation without running the tutorials and fails on any warning.

## The MNE sample dataset

The simulation benchmark uses MNE's `sample` dataset (about 1.5 GB). It is **never downloaded
implicitly**:

- Unit tests use a tiny artificial leadfield (`dirconn.simulation.artificial_leadfield`) and never need
  the dataset.
- Benchmarks and tutorials obtain it through `dirconn.datasets.require_sample()`. It checks locally
  first (including the files the benchmark reads) and, if the data are missing, stops with the size,
  the `MNE_DATA` setting and the download command; it downloads only when called with
  `download=True`. It never changes MNE's global configuration.
- Never call `mne.datasets.sample.data_path(path=...)` in code or tests without `update_path=False`:
  by default it rewrites `~/.mne/mne-python.json` and can leave `MNE_DATASETS_SAMPLE_PATH` pointing at a
  temporary directory.
- Developers who want to run the benchmarks download it once, in a place of their choice:

  ```bash
  python -c "import mne; print(mne.datasets.sample.data_path(download=True))"
  ```

  MNE looks in `~/mne_data` by default; set `MNE_DATA` to use another location.

## What each method can and cannot see

[docs/method_validation.md](docs/method_validation.md) lists, for known networks (chain, common
driver, collider, bidirectional pair, feedback loop, zero-lag coupling, hidden driver, observation
noise), how every method scores. It is generated, not written: rerun

```bash
python scripts/validate_methods.py            # full run, about ten minutes
python scripts/validate_methods.py --seeds 3  # quick look
python scripts/validate_methods.py --from-results   # re-render the saved rows only
```

after a change that can move the numbers (a new method, a new default, a change in the VAR core,
a new numpy or scipy), and commit the report together with the saved rows. The scenarios and the
runner live in `dirconn.validation`; the metrics are in `dirconn.evaluation`. Add a method to
`METHODS` and it appears in every table; add a scenario to `SCENARIOS` and it appears in the report.

## Adding a method

1. Write the tests: an exact or independently computed reference, the `info` guardrails, finiteness,
   and the axis convention.
2. Subclass `Estimator`, define `info` (`name`, `backend`, `directed`, node limits, `scale_dependent`,
   frequency or time resolution, relevant assumptions) and implement `_fit`.
3. Build the provenance with `self._provenance(...)` and return a `Connectivity`; use
   `_extra_checks` for method-specific diagnostics.
4. If the method is stochastic or iterative, expose `seed` and the order or tuning parameters, and
   record them.
5. Add it to `dirconn.validation.METHODS` and regenerate the validation report, so that it is
   evaluated against known truth from the day it exists (the simulation benchmark builds on the
   same scenarios).

## Data formats

- Sensor-level data are plain BIDS read through `mne-bids`.
- Region-level series and invasive channels use the **dirconn convention v0**
  ([decision 0001](docs/decisions/0001-bids-roi-layout.md)); it is not part of the BIDS specification
  and says so in every sidecar.
- Results are NetCDF (`Connectivity.to_netcdf`), networks and scenarios are YAML.

## Releasing (maintainers)

1. On a branch: set `version` in `pyproject.toml` (for example `0.1.0`), move the entries under
   `[Unreleased]` in `CHANGELOG.md` to a section with that version and the date, and open a pull
   request. Merge it when CI is green.
2. Publish a GitHub release from `main` with the tag `v<version>` (for example `v0.1.0`). The
   `Release` workflow builds the sdist and the wheel once, checks that the tag and the version
   agree, uploads to TestPyPI and then to PyPI (Trusted Publishing, no token). Zenodo archives the
   release and gives it a DOI.
3. After the release, set the version to the next `.dev0` (for example `0.2.0.dev0`).

To try the upload without releasing, run the `Release` workflow by hand: it builds the current
commit and uploads it to TestPyPI only.

## Licence

`dirconn` is released under the BSD 3-Clause licence (see [LICENSE](LICENSE)). By contributing you
agree that your contribution is distributed under it.
