"""The real runner of the benchmark: simulate one task and score every estimator of a protocol.

:func:`run_task` is what :func:`dirconn.protocol.run_protocol` calls for every task: it simulates
the scenario once through the head model of the task's modality and then applies every estimator
(and every inverse method) of the protocol to that same data, at the three stages of
:func:`dirconn.benchmark.evaluate_stages`. :func:`measure_gate` times a few tasks to estimate the
wall-clock time of a whole protocol before it is started.
"""

from __future__ import annotations

import functools
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from dirconn.benchmark import add_artifact, evaluate_stages, roi_observations
from dirconn.coherency import PSI, ImCoh
from dirconn.emptyroom import empty_room_window, n_windows
from dirconn.estimator import Estimator
from dirconn.exceptions import DirconnWarning
from dirconn.granger import GC
from dirconn.headmodel import MODALITIES, HeadModel, load_head_model
from dirconn.inverse import crosstalk
from dirconn.multitaper_granger import MTGC
from dirconn.pcmci import PCMCI
from dirconn.protocol import (
    ESTIMATOR_NAMES,
    EXTENDED_ESTIMATOR_NAMES,
    Protocol,
    Task,
    estimate_hours,
    expand,
    task_seed,
)
from dirconn.rng import RNGController
from dirconn.scenario import ScenarioSpec
from dirconn.significance import significance_test
from dirconn.simulation import simulate, simulate_on_head
from dirconn.spectral_estimators import DTF, GPDC, PDC, RPDC
from dirconn.time_reversal import TRGC
from dirconn.transfer_entropy import TE

DEFAULT_SETTINGS: dict[str, Any] = {
    "max_order": 40,
    "fixed_order_seconds": 0.2,
    "band_low_hz": 1.0,
    "band_high_hz": 45.0,
    "store_scores": False,
    "n_surrogates": 39,
    "low_order": 2,
    "n_freqs": None,
}
"""Settings of a protocol and their defaults: the cap on the BIC search, the length of the long
fixed order in seconds, and the band over which spectral measures are scored when the scenario
defines none, and whether the cards keep the score matrices (``store_scores``, for the
leakage studies; off elsewhere because it makes the results larger) and the number of
surrogates of a surrogate test (``n_surrogates``; with 39 the smallest p-value is 0.025 and
``p <= 0.05`` has probability exactly 0.05 under the null). ``low_order`` is the order in samples of
the ``low`` labels, an order too low for the scenario on purpose (two, the order of the chain
without observation noise); ``n_freqs`` is the frequency grid of the spectral estimators (``None``:
the estimator's default), which a significance study stores per frequency."""
MULTITAPER_SEGMENT_SECONDS = 2.0
"""Multitaper segment of the ``mt`` estimators (imcoh, PSI, MTGC), in seconds."""
REFERENCE_DURATION_S = 120.0
SAMPLING_RULE = 0.1
"""At most one coefficient per ten observations (:func:`dirconn.estimator.sampling_check`)."""


def settings_for(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the defaults updated with a protocol's ``settings``; unknown names are errors."""
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(DEFAULT_SETTINGS))
    if unknown:
        raise ValueError(f"Unknown setting(s) {unknown}; known: {sorted(DEFAULT_SETTINGS)}.")
    return {**DEFAULT_SETTINGS, **overrides}


def estimator_for(
    label: str, *, sfreq: float, n_nodes: int, n_times: int, settings: Mapping[str, Any]
) -> Estimator:
    """Build the estimator named by a label such as ``"gc/bic"`` or ``"pdc/fixed"``.

    ``bic`` selects the order by BIC up to a cap: the smaller of ``settings["max_order"]`` and the
    order at which the model would break the sampling rule (``n_nodes * order`` coefficients per
    ten observations). ``fixed`` uses ``round(fixed_order_seconds * sfreq)`` samples, ``low``
    ``low_order`` samples. The order-stability check is off: it multiplies the cost of a fit by 2
    to 4. Spectral estimators get ``n_freqs`` when it is set. The further labels: TRGC with the
    ``bic`` order; PCMCI with ``tau_max`` and TE with ``target_history`` chosen the same way; imcoh,
    PSI and MTGC on multitaper segments of ``MULTITAPER_SEGMENT_SECONDS``, PSI over the default
    scoring band (``band_low_hz`` to ``band_high_hz``).
    """
    if label not in ESTIMATOR_NAMES:
        raise ValueError(f"Unknown estimator {label!r}; known: {list(ESTIMATOR_NAMES)}.")
    method, setting = label.split("/")
    if setting == "bic":
        cap = max(1, min(int(settings["max_order"]), int(SAMPLING_RULE * n_times // n_nodes)))
        order_args: dict[str, Any] = {"order": "bic", "max_order": cap}
    elif setting == "low":
        order_args = {"order": int(settings["low_order"])}
    else:
        order_args = {"order": max(1, round(float(settings["fixed_order_seconds"]) * sfreq))}
    if label in EXTENDED_ESTIMATOR_NAMES:
        return _extended_estimator(method, order_args, settings)
    common = {**order_args, "check_order_stability": False}
    if method == "gc":
        return GC(**common)
    if method == "gc_pairwise":
        return GC(**common, conditional=False)
    if settings["n_freqs"] is not None:
        common["n_freqs"] = int(settings["n_freqs"])
    spectral = {"dtf": DTF, "pdc": PDC, "gpdc": GPDC, "rpdc": RPDC}[method]
    return spectral(**common)


def _extended_estimator(
    method: str, order_args: dict[str, Any], settings: Mapping[str, Any]
) -> Estimator:
    cap = int(order_args["max_order"]) if "max_order" in order_args else 1
    if method == "trgc":
        return TRGC(**order_args, check_order_stability=False)
    if method == "pcmci":
        return PCMCI(tau_max="bic", max_order=cap)
    if method == "te":
        return TE(mode="pairwise", target_history="bic", max_order=cap)
    if method == "imcoh":
        return ImCoh(segment_length=MULTITAPER_SEGMENT_SECONDS)
    if method == "psi":
        band = (float(settings["band_low_hz"]), float(settings["band_high_hz"]))
        return PSI(segment_length=MULTITAPER_SEGMENT_SECONDS, bands=[band])
    return MTGC(segment_length=MULTITAPER_SEGMENT_SECONDS)


def jsonable(value: Any) -> Any:
    """Return ``value`` with numpy types turned into plain Python so ``json`` accepts it."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Cannot serialise an object of type {type(value).__name__}: {value!r}.")


@functools.cache
def _head(modality: str) -> HeadModel:
    return load_head_model(modality)


def run_task(task: Task, protocol: Protocol) -> dict[str, Any]:
    """Simulate a task and score every estimator of the protocol on the same data.

    Returns
    -------
    dict
        One entry per estimator label (``"gc/bic"``), or per ``"label@inverse"`` when the protocol
        has several inverse methods: the output of :func:`dirconn.benchmark.evaluate_stages`
        made JSON-safe.
    """
    if protocol.kind == "significance":
        return _run_significance(task, protocol)
    settings = settings_for(protocol.settings)
    spec = task.scenario_spec()
    head = _head(task.modality)
    seed = task_seed(protocol.master_seed, task)
    n_background = spec.n_background
    noise_series = None
    if task.noise in ("white", "empty_room"):
        n_background = 0  # no brain in an empty room, and none in the white-noise control
    if task.noise == "empty_room":
        # consecutive blocks that do not overlap; a seed index beyond the last block wraps around
        blocks = max(1, n_windows(head, spec.n_times, sfreq=spec.sfreq))
        noise_series = empty_room_window(
            head, task.seed_index % blocks, spec.n_times, sfreq=spec.sfreq
        )
    sim = simulate_on_head(
        spec.network,
        head,
        sfreq=spec.sfreq,
        n_times=spec.n_times,
        seed=seed,
        dynamics=spec.dynamics,
        snr_db=spec.snr_db,
        n_background=n_background,
        background_fraction=spec.background_fraction,
        node_scale="unit",
        source_model=task.source_model,
        noise_series=noise_series,
    )
    if spec.artifact is not None:
        sim = add_artifact(sim, head, spec.artifact)
    default_band = (settings["band_low_hz"], settings["band_high_hz"])
    roi_seed = int(RNGController(seed).child(1).integers(0, 2**31 - 1))
    payload: dict[str, Any] = {}
    for label in protocol.estimators:
        for method in protocol.inverse:
            estimator = estimator_for(
                label,
                sfreq=spec.sfreq,
                n_nodes=spec.network.n_nodes,
                n_times=spec.n_times,
                settings=settings,
            )
            with warnings.catch_warnings():
                if task.zero_network:
                    # no true edge: AUROC and friends are NaN by design, once per score card
                    warnings.simplefilter("ignore", DirconnWarning)
                card = evaluate_stages(
                    estimator,
                    sim,
                    head,
                    seed=roi_seed,
                    default_band=default_band,
                    inverse={"method": method},
                    hidden=spec.hidden,
                    by_frequency=True,
                    store_scores=bool(settings["store_scores"]),
                )
            key = label if len(protocol.inverse) == 1 else f"{label}@{method}"
            payload[key] = jsonable(card)
    return payload


def crosstalk_matrices(modalities: Sequence[str], *, seed: int = 0) -> dict[str, dict[str, Any]]:
    """Return the crosstalk matrix of every modality, ready for a results header.

    ``matrix[y][x]`` is the leak of region ``x`` into the series of region ``y``, relative to the
    response of that series to its own region (:func:`dirconn.inverse.crosstalk` with its defaults:
    minimum norm, ``pca_flip``, a 10 dB probe), rounded to six decimals.
    """
    out: dict[str, dict[str, Any]] = {}
    for modality in modalities:
        result = crosstalk(_head(modality), seed=seed)
        out[modality] = {
            "names": list(result.rois),
            "matrix": np.round(result.matrix, 6).tolist(),
        }
    return out


def _run_significance(task: Task, protocol: Protocol) -> dict[str, Any]:
    """Simulate one region-level dataset and apply every test of the protocol to it.

    The data are the node activity of the scenario plus independent white observation noise at the
    task's SNR (:func:`dirconn.benchmark.roi_observations`), as at stage 1 of the benchmark: no
    head model. The result per estimator is the p-value matrix of every test (six decimals),
    ``[source, target]``; for a spectral estimator ``[source, target, freq]`` with the grid in
    ``freqs``. What is significant at which level is decided later, from the p-values, by
    :func:`dirconn.significance_study.significance_report` and
    :func:`~dirconn.significance_study.frequency_report`.
    """
    settings = settings_for(protocol.settings)
    spec = task.scenario_spec()
    seed = task_seed(protocol.master_seed, task)
    sim = simulate(
        spec.network,
        sfreq=spec.sfreq,
        n_times=spec.n_times,
        seed=seed,
        dynamics=spec.dynamics,
        snr_db=spec.snr_db,
        n_background=0,
        node_scale="unit",
    )
    streams = RNGController(seed)
    observation_seed = int(streams.child(1).integers(0, 2**31 - 1))
    test_seed = int(streams.child(2).integers(0, 2**31 - 1))
    data = roi_observations(sim, snr_db=spec.snr_db, seed=observation_seed)
    if spec.hidden:
        data = data.pick(list(spec.analysed))
    payload: dict[str, Any] = {}
    for label in protocol.estimators:
        estimator = estimator_for(
            label,
            sfreq=spec.sfreq,
            n_nodes=data.n_channels,
            n_times=spec.n_times,
            settings=settings,
        )
        tests: dict[str, Any] = {}
        freqs = None
        for test in protocol.tests:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DirconnWarning)
                if test == "analytic":
                    result = significance_test(estimator, data, test="analytic")
                else:
                    result = significance_test(
                        estimator,
                        data,
                        test="surrogate",
                        surrogate=test.split(":", 1)[1],
                        n_surrogates=int(settings["n_surrogates"]),
                        seed=test_seed,
                    )
            pvalues = np.array(result.pvalues, float)[..., 0]  # one time point
            pvalues[np.eye(len(pvalues), dtype=bool)] = np.nan  # a node with itself is no edge
            if pvalues.shape[2] == 1:
                pvalues = pvalues[:, :, 0]  # one value per cell (Granger causality)
            else:
                freqs = result.observed.freqs
            tests[test] = {"pvalues": np.round(pvalues, 6).tolist()}
        payload[label] = {"analysed": list(data.names), "tests": tests}
        if freqs is not None:
            payload[label]["freqs"] = np.asarray(freqs, float)
    return jsonable(payload)


def make_runner(protocol: Protocol) -> Callable[[Task], dict[str, Any]]:
    """Return ``run_task`` bound to a protocol; picklable, for :func:`run_protocol`."""
    return functools.partial(run_task, protocol=protocol)


def _warm(modality: str) -> None:
    _head(modality)


def _parallel_probe(
    task: Task, protocol: Protocol, workers: int, modalities: Sequence[str]
) -> tuple[float, float]:
    """Run ``workers`` copies of a task at once; return the startup and the batch time in seconds.

    The first call loads the head models in the worker processes (the startup time); the second
    runs one copy of the task in every worker at the same time, so its duration is what one task
    costs when the machine is busy, which is what decides how much parallelism buys.
    """
    from joblib import Parallel, delayed

    pool = Parallel(n_jobs=workers)
    started = time.perf_counter()
    pool(delayed(_warm)(m) for m in modalities for _ in range(2 * workers))
    startup = time.perf_counter() - started
    started = time.perf_counter()
    pool(delayed(run_task)(task, protocol) for _ in range(workers))
    return startup, time.perf_counter() - started


def measure_gate(
    protocol: Protocol,
    *,
    workers: int,
    efficiency: float | None = None,
    scenarios: Mapping[str, ScenarioSpec] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Time the tasks of a protocol and estimate the wall-clock time of the whole run.

    The head models are loaded first (their loading is the startup time, counted once). For every
    number of nodes, the first task of each modality is run once (all estimators, all stages) and
    its time is scaled to a data length of :data:`REFERENCE_DURATION_S`, linearly; the slowest
    modality stands for the size, which makes the estimate a cautious one.

    With several workers the parallel efficiency is **measured**, not assumed: the heaviest task
    is run in every worker at once (:func:`_parallel_probe`) and the efficiency is its serial time
    over the time of that batch, between 0.01 and 1. On a machine with a power limit or too little
    memory for the workers this is far below the 0.8 one might hope for. Pass ``efficiency`` to
    skip the probe and use a value of your own.

    Returns
    -------
    dict
        ``seconds_per_task`` (by number of nodes, as text), ``reference_duration_s``, ``workers``,
        ``efficiency``, ``startup_seconds``, ``n_tasks``, ``estimate_hours``, the list of
        ``measured`` tasks and, when probed, ``probe``.
    """
    tasks = expand(protocol, scenarios)
    if not tasks:
        raise ValueError("The protocol expands to no task.")
    chosen: dict[tuple[int, str], Task] = {}
    for task in tasks:
        chosen.setdefault((task.n_nodes, task.modality), task)
    modalities = sorted({m for _, m in chosen if m in MODALITIES})  # none at region level
    started = time.perf_counter()
    for modality in modalities:
        _head(modality)  # loading a head model is once per process, not part of a task's cost
    startup = time.perf_counter() - started
    seconds: dict[int, float] = {}
    measured = []
    slowest: tuple[float, Task] | None = None
    for (n_nodes, modality), task in sorted(chosen.items()):
        started = time.perf_counter()
        run_task(task, protocol)
        elapsed = time.perf_counter() - started
        scaled = elapsed * REFERENCE_DURATION_S / task.duration_s
        seconds[n_nodes] = max(seconds.get(n_nodes, 0.0), scaled)
        if slowest is None or elapsed > slowest[0]:
            slowest = (elapsed, task)
        measured.append(
            {
                "n_nodes": n_nodes,
                "modality": modality,
                "scenario": task.scenario,
                "duration_s": task.duration_s,
                "seconds": elapsed,
            }
        )
        if progress is not None:
            progress(f"gate: {task.scenario} ({n_nodes} nodes, {modality}): {elapsed:.1f} s")
    assert slowest is not None
    probe: dict[str, Any] | None = None
    if efficiency is not None:
        used = float(efficiency)
    elif workers == 1:
        used = 1.0
    else:
        serial, heaviest = slowest
        if progress is not None:
            progress(f"gate: probing {workers} workers with {heaviest.scenario}")
        startup, batch = _parallel_probe(heaviest, protocol, workers, modalities)
        used = min(1.0, max(0.01, serial / batch))
        probe = {
            "scenario": heaviest.scenario,
            "serial_seconds": serial,
            "batch_seconds": batch,
        }
        if progress is not None:
            progress(f"gate: a batch of {workers} takes {batch:.1f} s: efficiency {used:.2f}")
    out: dict[str, Any] = {
        "seconds_per_task": {str(n): s for n, s in sorted(seconds.items())},
        "reference_duration_s": REFERENCE_DURATION_S,
        "workers": workers,
        "efficiency": used,
        "startup_seconds": startup,
        "n_tasks": len(tasks),
        "estimate_hours": estimate_hours(
            tasks,
            seconds,
            reference_duration_s=REFERENCE_DURATION_S,
            workers=workers,
            efficiency=used,
            startup_seconds=startup,
        ),
        "measured": measured,
    }
    if probe is not None:
        out["probe"] = probe
    return out
