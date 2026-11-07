# dirconn

[![PyPI](https://img.shields.io/pypi/v/dirconn)](https://pypi.org/project/dirconn/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24036991.svg)](https://doi.org/10.5281/zenodo.24036991)

Directed functional connectivity for neurophysiology signals (EEG, MEG, iEEG, LFP) at sensor and
region level.

dirconn estimates *predictive* directed influence (Granger causality and related measures). It does
not establish interventional causality, and every result records the assumptions it relies on and
the checks that were run on them.

**Status: alpha.** The API can still change between minor versions.

Documentation, tutorials and API reference: <https://hanweiliu-viola.github.io/dirconn/>

## What it does

- **Data and results**: a `TimeSeries` of trials × channels × samples for sensors, cortical regions
  (Desikan-Killiany names) and directly recorded nodes such as a subthalamic LFP; a `Connectivity`
  result that carries its provenance (method, model order, assumptions, checks) and is saved as
  NetCDF; region time series in a BIDS-like layout.
- **Methods**: conditional and pairwise Granger causality, DTF, PDC, generalised and renormalised
  PDC (rPDC); controls for zero-delay mixing (time-reversed Granger causality, imaginary coherence,
  phase slope index); Granger causality from multitaper spectra without a fitted model; PCMCI; transfer
  entropy with the KSG estimator (multivariate by default, optionally on a GPU); time-varying
  measures (ADTF, TPDC, windowed DTF).
- **Statistics**: surrogate and permutation tests, asymptotic per-frequency tests of PDC and rPDC,
  multiple-comparison correction, bootstrap intervals, and a residual whiteness check on every
  VAR fit.
- **Simulation and validation**: networks with a known truth, simulated through the head models of
  the MNE `sample` dataset, so that leakage, observation noise and hidden drivers can be measured
  instead of assumed. The benchmark protocols, their saved results and what they do and do not show
  are part of the repository (`protocols/`, `benchmark_results/`, `docs/benchmark*.md`).

## Install

```bash
pip install dirconn
pip install "dirconn[plot,stats]"   # with plotting and the statsmodels-based tests
```

Optional extras: `connectivity` (mne-connectivity), `pcmci` (tigramite), `stats` (statsmodels),
`parallel` (joblib), `speed` (numba), `gpu` (torch), `plot` (matplotlib, pyCirclize), `plot-extra`
(3D maps of the cortical surface), `all`. Python 3.10 to 3.13.

From a clone, for development:

```bash
pip install -e ".[dev,all]"
pytest
```

The MNE `sample` dataset (about 1.5 GB) is needed only for the benchmark and for tutorials 7, 8, 13
and 14. It is never downloaded without being asked; see [CONTRIBUTING.md](CONTRIBUTING.md).

## A first example

```python
import dirconn
from dirconn import Edge, Network, dk_node
from dirconn.simulation import NodeDynamics, simulate

A, B, C = "cuneus-rh", "precuneus-rh", "insula-rh"
network = Network(
    nodes=[dk_node(A), dk_node(B), dk_node(C)],
    edges=[Edge(A, B, strength=0.4, delay_ms=10.0), Edge(B, C, strength=0.4, delay_ms=10.0)],
    name="chain",
)
dynamics = {
    A: NodeDynamics(kind="oscillator", freq_hz=10.0),
    B: NodeDynamics(kind="oscillator", freq_hz=20.0),
    C: NodeDynamics(kind="oscillator", freq_hz=15.0),
}
sim = simulate(
    network,
    sfreq=100.0,
    n_times=12000,
    seed=0,
    dynamics=dynamics,
    snr_db=10.0,
    n_background=0,
    node_scale="unit",
)
data = sim.roi_observations(snr_db=10.0, seed=1)

result = dirconn.gc(data, order="bic")  # conditional Granger causality
print(result.provenance.claim)
for check in result.provenance.assumptions.checks:
    print(check.name, check.status)
```

The tutorials go on from here: model order, frequency-domain measures, significance, results that
mislead (indirect paths, shared and hidden drivers), time-varying connectivity, sensors to regions and
leakage, sources and sinks, zero-delay controls, PCMCI, transfer entropy, and the renormalised PDC.
Each ends on what its result does and does not show.

## Reproducible Docker environment

The [Dockerfile](Dockerfile) freezes the whole scientific stack (exact versions in
[constraints/docker.txt](constraints/docker.txt); the image also carries the full `pip freeze` in
`/opt/environment-lock.txt`) for analyses whose numbers must be computable again later. It is an
environment, not a pipeline: there is no entrypoint; mount your scripts or notebooks into `/work`
and the data read-only into `/data`. It holds no data. The tutorials are in `/opt/dirconn/tutorials`
as notebooks; 7, 8, 13 and 14 need the MNE `sample` dataset mounted at `/mne_data`.

```bash
docker pull ghcr.io/hanweiliu-viola/dirconn:<tag>
docker run --rm -it -v /path/to/analysis:/work -v /path/to/data:/data:ro ghcr.io/hanweiliu-viola/dirconn:<tag>
docker run --rm -p 127.0.0.1:8888:8888 -v /path/to/analysis:/work -v /path/to/data:/data:ro \
    ghcr.io/hanweiliu-viola/dirconn:<tag> jupyter lab --ip=0.0.0.0 --no-browser
```

Jupyter listens only on the local machine (`127.0.0.1`); open the URL with the token it prints.
`OMP_NUM_THREADS=1` is set on purpose, because the number of threads changes floating-point sums;
parallelise over subjects or surrogates (joblib, explicit seeds) instead, and on a shared machine
cap the container with `--cpus` and `--memory`. Reading through a Docker Desktop bind mount is slow;
keep large data on an SSD, or copy it into a Docker volume first. The image contains no torch and no
3D plotting: surface maps are drawn on the host from saved results. Under Apptainer the image runs
unchanged (`apptainer pull docker://ghcr.io/hanweiliu-viola/dirconn:<tag>`); nothing is written into
it at run time. The image is checked against the published validation numbers with
`scripts/validate_methods.py` and `scripts/compare_validation.py`.

## Citing

If you use dirconn, please cite the version you used; the metadata are in
[CITATION.cff](CITATION.cff), and every release gets a DOI on Zenodo. The DOI
[10.5281/zenodo.24036991](https://doi.org/10.5281/zenodo.24036991) always resolves to the latest
version; the Zenodo page lists the DOI of each version (0.1.0:
[10.5281/zenodo.24036992](https://doi.org/10.5281/zenodo.24036992)).

## Author and licence

Hanwei Liu, University Hospital of Wuerzburg. BSD-3-Clause, see [LICENSE](LICENSE).

Parts of the code and documentation were written with the help of an AI coding assistant; the
author reviewed all of it, and it is covered by the tests and the simulation benchmark described
above.
