"""Evaluation of time-resolved methods on non-stationary region-level scenarios.

Four three-region scenarios (``switch_on``, ``switch_off``, ``direction_flip``, ``burst``;
shipped as scenario files) switch a coupling on, off, around or briefly on in the middle of a 60 s
recording. Region-level series with independent white observation noise at a given SNR (no
forward model) are scored with the time-resolved metrics of :mod:`dirconn.evaluation`:

* the ranking of the edges inside the *stable* segments, outside a guard band after each switch;
* the *tracking latency*, how long the score of a switching edge takes to follow the switch. One
  fit is too noisy for it (a score that is already beyond half the way when the switch happens
  counts as an instant arrival), so every row keeps the score course of each switching pair and
  the latency is taken from the mean course over the datasets of a cell
  (:func:`latency_of_mean_course`).

The two are reported separately. The methods are :class:`~dirconn.ADTF` (``dual`` and ``dekf``;
``rls`` has the same coefficients as ``dual``) and :class:`~dirconn.TPDC` over a range of
forgetting factors, and :class:`~dirconn.WDTF` with two window lengths. A second experiment
compares the two ways of getting one value per segment: a continuous ADTF followed by
``reduce_time`` against ``WDTF(segments=...)``.

Everything here reports what was observed in these scenarios; it does not rank the methods.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dirconn.estimator import Estimator
from dirconn.evaluation import (
    auroc,
    guard_seconds,
    latency_from_course,
    score_stable,
    switch_times,
)
from dirconn.exceptions import DirconnWarning
from dirconn.result import Connectivity
from dirconn.scenario import ScenarioSpec, load_scenario
from dirconn.simulation import simulate
from dirconn.timeseries import TimeSeries
from dirconn.timevarying import ADTF, TPDC
from dirconn.tvmvar import TvarFit, fit_tvar_grid
from dirconn.validation import add_observation_noise, environment
from dirconn.windowed import WDTF

SCENARIOS: tuple[str, ...] = ("switch_on", "switch_off", "direction_flip", "burst")
FORGETTING: tuple[float, ...] = (
    0.95,
    0.98,
    0.99,
    0.995,
    0.999,
    0.9999,
    0.99999,
    0.999999,
    0.9999999,
)
"""The first four are the confirmed grid. They turned out to give a memory of 1 to 3 samples,
so the grid runs on up to the factors whose measured memory is about a second."""
SNR_DB: tuple[float, ...] = (20.0, 10.0, 0.0)
WINDOWS_S: tuple[float, ...] = (4.0, 8.0)
N_SEEDS = 10
BAND = (1.0, 45.0)
"""Band in Hz over which the spectral results are averaged before scoring (the Nyquist is 50 Hz)."""
GUARD_FRACTION = 1.0 / 3.0
"""The guard band after a switch is this fraction of the shortest stretch of the scenario."""
SEGMENTS = {"A": (10.0, 29.0), "B": (31.0, 36.0)}
"""The two segments of the segment comparison: well before the switch, and 1 s to 6 s after it."""
SEGMENT_FORGETTING = 0.999999
SEGMENT_SCENARIOS: tuple[str, ...] = ("switch_on", "direction_flip")
RESULTS_KIND = "dirconn-nonstationary"

ADTF_UPDATES: tuple[str, ...] = ("dual", "dekf")
"""``rls`` is left out of the sweep: its coefficients are identical to those of ``dual``."""
TIME_STEP_S = 0.1
"""Distance between the returned time points of ADTF and TPDC, in seconds."""
BURN_IN_SAMPLES = 300
"""Samples (3 s at 100 Hz) discarded before the first output of ADTF and TPDC. Fixed so that all
seeds share one time axis (the default depends on the order the criterion picks) and large
enough for the seed fit of every order up to 5 with three nodes (``20 * order * n_nodes``)."""
HOLD_S = 1.0
"""The score has to stay beyond half the way for this long for a latency to count."""
CURVE_STEP_S = 0.5
"""Spacing of the stored score courses of the switching pairs, in seconds."""


def method_labels() -> list[str]:
    """Return the labels of the methods of the sweep, in report order."""
    return [
        *[f"adtf-{update}" for update in ADTF_UPDATES],
        "tpdc",
        *[f"wdtf-{int(w)}s" for w in WINDOWS_S],
    ]


def has_forgetting(label: str) -> bool:
    """Whether the method of ``label`` has a forgetting factor (the windowed ones do not)."""
    return not label.startswith("wdtf")


def make_estimator(label: str, forgetting: float | None) -> Estimator:
    """Build the estimator of a sweep label; the order is chosen by BIC (cap 5)."""
    if label.startswith("adtf-") or label == "tpdc":
        if forgetting is None:
            raise ValueError(f"The method {label!r} needs a forgetting factor.")
        options: dict[str, Any] = {
            "forgetting": forgetting,
            "time_step": TIME_STEP_S,
            "burn_in": BURN_IN_SAMPLES,
        }
        if label == "tpdc":
            return TPDC(**options)
        return ADTF(update=label.split("-", 1)[1], **options)
    if label.startswith("wdtf-") and label.endswith("s"):
        return WDTF(window=float(label[5:-1]))
    raise ValueError(f"Unknown method label {label!r}; known: {method_labels()}.")


def dataset_seed(scenario: str, snr_db: float, k: int) -> int:
    """Return the seed of the ``k``-th dataset of a scenario and SNR (stable under new axes)."""
    return 10_000 * (SCENARIOS.index(scenario) + 1) + 100 * SNR_DB.index(float(snr_db)) + int(k)


def simulate_roi(spec: ScenarioSpec, *, snr_db: float, seed: int) -> TimeSeries:
    """Simulate a scenario at the region level and add white observation noise at ``snr_db``."""
    sim = simulate(
        spec.network,
        sfreq=spec.sfreq,
        n_times=spec.n_times,
        dynamics=spec.dynamics,
        seed=seed,
        n_sensors=4,
        n_background=0,
    )
    return add_observation_noise(sim.sources, snr_db, seed)


def _fit(estimator: Estimator, series: TimeSeries) -> tuple[Connectivity, list[str], float]:
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DirconnWarning)
        result = estimator.fit(series)
    seconds = time.perf_counter() - started
    flagged = sorted(
        c.name for c in result.provenance.assumptions.checks if c.status in ("warn", "fail")
    )
    return result, flagged, seconds


def switching_pairs(spec: ScenarioSpec, duration: float) -> dict[tuple[str, str], list[float]]:
    """Return ``{(source, target): switch times}`` for the pairs whose coupling switches."""
    names = spec.network.names
    out = {}
    for source in names:
        for target in names:
            if source == target:
                continue
            times = switch_times(spec.network, pair=(source, target), duration=duration)
            if times:
                out[(source, target)] = times
    return out


def scenario_guard(spec: ScenarioSpec) -> float:
    """Return the guard band of a scenario in seconds: a third of its shortest stretch.

    A stretch lies between neighbouring switches (or the ends of the recording). One guard for
    all methods keeps the scores comparable; the methods differ in how long they take to settle,
    which :func:`latency_of_mean_course` measures.
    """
    duration = spec.n_times / spec.sfreq
    edges = [0.0, *switch_times(spec.network, duration=duration), duration]
    return GUARD_FRACTION * min(b - a for a, b in zip(edges, edges[1:], strict=False))


def evaluate_result(result: Connectivity, spec: ScenarioSpec) -> dict[str, Any]:
    """Score one time-resolved result of a scenario: stable-segment metrics and score courses.

    The main scores use the guard band of the scenario (:func:`scenario_guard`). For comparison
    the AUROC is also computed with a guard equal to the measured memory of the method
    (``auroc_memory``; none for a method without a memory guard) and without any guard
    (``auroc_all``). The score course of every switching pair (magnitude averaged over ``BAND``)
    is kept at a spacing of about :data:`CURVE_STEP_S` for :func:`latency_of_mean_course`; its
    time axis is stored once as ``curve_axis`` (``start``, ``step``, ``n``).
    """
    duration = spec.n_times / spec.sfreq
    switches = switch_times(spec.network, duration=duration)
    truth = spec.network.truth(freqs=result.freqs, times=result.times)
    guard = scenario_guard(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DirconnWarning)
        stable = score_stable(result, truth, band=BAND, guard_s=guard, switches=switches)
        everything = score_stable(result, truth, band=BAND, guard_s=0.0, switches=switches)
        try:
            memory_guard: float | None = guard_seconds(result)
        except ValueError:
            memory_guard = None
        by_memory = (
            None
            if memory_guard is None
            else score_stable(result, truth, band=BAND, guard_s=memory_guard, switches=switches)
        )
    magnitudes = np.abs(result.reduce_freq(BAND, how="mean").values[:, :, 0, :])
    step = float(np.median(np.diff(result.times))) if result.times.size > 1 else math.inf
    every = 1 if not math.isfinite(step) else max(1, round(CURVE_STEP_S / step))
    keep = np.arange(0, result.times.size, every)
    index = {name: i for i, name in enumerate(spec.network.names)}
    kept = result.times[keep]
    axis = {
        "start": round(float(kept[0]), 6),
        "step": round(float(np.median(np.diff(kept))), 6) if kept.size > 1 else 0.0,
        "n": int(kept.size),
    }
    curves = [
        {
            "source": source,
            "target": target,
            "switches": times,
            "scores": [round(float(v), 3) for v in magnitudes[index[source], index[target], keep]],
        }
        for (source, target), times in switching_pairs(spec, duration).items()
    ]
    return {
        "guard_s": guard,
        "memory_guard_s": memory_guard,
        "auroc_stable": stable["auroc"],
        "direction_stable": stable["direction_accuracy"],
        "separation_stable": stable["separation"],
        "auroc_memory": None if by_memory is None else by_memory["auroc"],
        "auroc_all": everything["auroc"],
        "direction_all": everything["direction_accuracy"],
        "n_stable": stable["n_stable"],
        "n_defined": stable["n_defined"],
        "n_times": int(result.times.size),
        "switches": switches,
        "curve_axis": axis,
        "curves": curves,
    }


def _cell_key(row: dict[str, Any]) -> tuple[str, float, str, float | None]:
    return (row["scenario"], float(row["snr_db"]), row["method"], row["forgetting"])


def aggregate_course(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Average the score courses of the rows of one cell (one scenario, SNR, method, factor).

    The rows differ in their seed only. Returns the cell with the mean course of every switching
    pair, which is what :func:`latency_of_cell` and the results file keep instead of every course.
    """
    if not rows:
        raise ValueError("rows is empty; there is nothing to average.")
    first = rows[0]
    for row in rows:
        if _cell_key(row) != _cell_key(first) or row["curve_axis"] != first["curve_axis"]:
            raise ValueError("The rows are not one cell: they differ in more than the seed.")
    courses = []
    for k, curve in enumerate(first["curves"]):
        mean = np.mean([r["curves"][k]["scores"] for r in rows], axis=0)
        courses.append(
            {
                "source": curve["source"],
                "target": curve["target"],
                "switches": curve["switches"],
                "mean": [round(float(v), 4) for v in mean],
            }
        )
    scenario, snr_db, method, forgetting = _cell_key(first)
    return {
        "scenario": scenario,
        "snr_db": snr_db,
        "method": method,
        "forgetting": forgetting,
        "n": len(rows),
        "guard_s": first["guard_s"],
        "switches": first["switches"],
        "curve_axis": first["curve_axis"],
        "courses": courses,
    }


def aggregate_courses(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return :func:`aggregate_course` for every cell of the sweep rows."""
    cells: dict[tuple[str, float, str, float | None], list[dict[str, Any]]] = {}
    for row in rows:
        if row["kind"] == "sweep":
            cells.setdefault(_cell_key(row), []).append(row)
    return [aggregate_course(cell) for cell in cells.values()]


def latency_of_cell(cell: dict[str, Any], *, hold_s: float = HOLD_S) -> list[dict[str, Any]]:
    """Return the tracking latency of the mean course of a cell, per switching pair and switch.

    The mean course of the pair is analysed by :func:`~dirconn.evaluation.latency_from_course`: the
    time the mean takes to cover half the way between its levels before and after each switch,
    holding for ``hold_s`` seconds.
    """
    axis = cell["curve_axis"]
    times = axis["start"] + axis["step"] * np.arange(axis["n"])
    step = axis["step"] if axis["n"] > 1 else math.inf
    hold_points = 1 if not math.isfinite(step) else max(1, round(hold_s / step))
    out = []
    for course in cell["courses"]:
        for switch in course["switches"]:
            record = latency_from_course(
                times,
                np.asarray(course["mean"], dtype=float),
                switch,
                switches=cell["switches"],
                guard_s=cell["guard_s"],
                hold_points=hold_points,
            )
            out.append({"source": course["source"], "target": course["target"], **record})
    return out


def latency_of_mean_course(
    rows: Sequence[dict[str, Any]], *, hold_s: float = HOLD_S
) -> list[dict[str, Any]]:
    """Return the tracking latency of the mean score course of the rows of one cell.

    One fit is too noisy for a latency (a score that is already beyond half the way when the switch
    happens counts as an instant arrival), so the courses of the seeds are averaged first
    (:func:`aggregate_course`) and :func:`latency_of_cell` measures the mean.
    """
    return latency_of_cell(aggregate_course(rows), hold_s=hold_s)


def _update_of(label: str) -> str:
    """Return the recursion a method with a forgetting factor uses (``tpdc`` runs the DEKF)."""
    return "dekf" if label == "tpdc" else label.split("-", 1)[1]


def _row(
    result: Connectivity,
    spec: ScenarioSpec,
    *,
    scenario: str,
    snr_db: float,
    k: int,
    seed: int,
    label: str,
    forgetting: float | None,
    flagged: list[str],
    seconds: float,
) -> dict[str, Any]:
    details = result.provenance.details
    memory = details.get("memory_samples")
    sfreq = details.get("sfreq")
    return {
        "kind": "sweep",
        "scenario": scenario,
        "snr_db": float(snr_db),
        "seed_index": int(k),
        "seed": seed,
        "method": label,
        "forgetting": forgetting,
        "order": int(result.provenance.model_order or 0),
        "diagnostic_warnings": flagged,
        "unstable_fraction": float(details.get("unstable_fraction", float("nan"))),
        "memory_s": None if memory is None or not sfreq else float(memory) / float(sfreq),
        "fit_seconds": seconds,
        **evaluate_result(result, spec),
    }


def evaluate_dataset(
    scenario: str,
    snr_db: float,
    k: int,
    *,
    methods: Sequence[str] | None = None,
    forgetting: Sequence[float] = FORGETTING,
    segments: bool = True,
) -> list[dict[str, Any]]:
    """Simulate one dataset and evaluate every method on it; returns one row per method fit.

    Every method sees the same series. The recursions of ADTF and TPDC run once per update for all
    values of ``forgetting`` together (:func:`~dirconn.tvmvar.fit_tvar_grid`), ADTF-DEKF and TPDC
    share the DEKF fit (they differ only in the measure computed from it), and the time of a
    batch is split evenly over its rows as ``fit_seconds``. With ``segments`` the segment comparison
    rows (``kind="segments"``) are added for the scenarios that have them.
    """
    spec = load_scenario(scenario)
    seed = dataset_seed(scenario, snr_db, k)
    series = simulate_roi(spec, snr_db=snr_db, seed=seed)
    factors = [float(f) for f in forgetting]
    batches: dict[str, tuple[list[TvarFit], float]] = {}

    def batch(update: str) -> tuple[list[TvarFit], float]:
        if update not in batches:
            estimator = ADTF(update=update, time_step=TIME_STEP_S, burn_in=BURN_IN_SAMPLES)
            options = estimator.tvar_options(series)
            started = time.perf_counter()
            fits = fit_tvar_grid(series, forgetting=factors, **options)
            batches[update] = (fits, (time.perf_counter() - started) / len(factors))
        return batches[update]

    rows: list[dict[str, Any]] = []
    for label in method_labels() if methods is None else methods:
        common: dict[str, Any] = {
            "scenario": scenario,
            "snr_db": snr_db,
            "k": k,
            "seed": seed,
            "label": label,
        }
        if not has_forgetting(label):
            result, flagged, seconds = _fit(make_estimator(label, None), series)
            rows.append(
                _row(result, spec, forgetting=None, flagged=flagged, seconds=seconds, **common)
            )
            continue
        fits, share = batch(_update_of(label))
        for factor, fit in zip(factors, fits, strict=True):
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DirconnWarning)
                estimator = make_estimator(label, factor)
                assert isinstance(estimator, ADTF | TPDC)
                result = estimator.from_fit(series, fit)
            flagged = sorted(
                c.name for c in result.provenance.assumptions.checks if c.status in ("warn", "fail")
            )
            seconds = share + time.perf_counter() - started
            rows.append(
                _row(result, spec, forgetting=factor, flagged=flagged, seconds=seconds, **common)
            )
    if segments and scenario in SEGMENT_SCENARIOS:
        rows.extend(compare_segments(scenario, snr_db, k, spec=spec, series=series, seed=seed))
    return rows


def _segment_truth(spec: ScenarioSpec, segment: tuple[float, float]) -> np.ndarray:
    """Return the truth matrix at the centre of a segment (inside one stretch of the scenario)."""
    centre = (segment[0] + segment[1]) / 2.0
    return np.asarray(spec.network.truth(times=[centre]).values[:, :, 0, 0])


def compare_segments(
    scenario: str,
    snr_db: float,
    k: int,
    *,
    spec: ScenarioSpec | None = None,
    series: TimeSeries | None = None,
    seed: int | None = None,
    forgetting: float = SEGMENT_FORGETTING,
) -> list[dict[str, Any]]:
    """Compare a continuous ADTF reduced over segments with a fit on each segment alone.

    The segments ``A`` (well before the switch) and ``B`` (1 s to 6 s after it) are scored with the
    AUROC against the truth at the segment centre, and the score of every switching pair is kept, so
    that the memory of the continuous estimate (B still carries the old state) and the variance of
    the short fit (B has only 5 s) can be read off.
    """
    spec = load_scenario(scenario) if spec is None else spec
    seed = dataset_seed(scenario, snr_db, k) if seed is None else seed
    series = simulate_roi(spec, snr_db=snr_db, seed=seed) if series is None else series
    segment_list = list(SEGMENTS.values())
    duration = spec.n_times / spec.sfreq
    pairs = list(switching_pairs(spec, duration))
    names = spec.network.names
    out = []
    approaches: dict[str, Callable[[], Connectivity]] = {
        "adtf_reduce_time": lambda: (
            ADTF(forgetting=forgetting, time_step=TIME_STEP_S, burn_in=BURN_IN_SAMPLES)
            .fit(series)
            .reduce_time(segment_list)
        ),
        "wdtf_segments": lambda: WDTF(segments=segment_list).fit(series),
    }
    for approach, build in approaches.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DirconnWarning)
            result = build()
        reduced = result.reduce_freq(BAND, how="mean")
        for index, (label, segment) in enumerate(SEGMENTS.items()):
            scores = np.abs(reduced.values[:, :, 0, index])
            truth = _segment_truth(spec, segment)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DirconnWarning)
                value = auroc(scores, truth)
            out.append(
                {
                    "kind": "segments",
                    "scenario": scenario,
                    "snr_db": float(snr_db),
                    "seed_index": int(k),
                    "seed": seed,
                    "approach": approach,
                    "segment": label,
                    "span": list(segment),
                    "forgetting": forgetting if approach == "adtf_reduce_time" else None,
                    "auroc": value,
                    "pair_scores": {
                        f"{s}->{t}": float(scores[names.index(s), names.index(t)]) for s, t in pairs
                    },
                }
            )
    return out


# ----------------------------------------------------------------------------------------------
# Results files
# ----------------------------------------------------------------------------------------------


def save_results(
    rows: list[dict[str, Any]], path: str | Path, *, meta: dict[str, Any] | None = None
) -> None:
    """Write the rows and the settings of the run to a JSON file (NaN and Infinity allowed).

    The score courses of the single fits are not kept, only their mean per cell (``courses``),
    which is all the latencies need and keeps the file small.
    """
    payload = {
        "kind": RESULTS_KIND,
        "meta": {
            "scenarios": list(SCENARIOS),
            "forgetting": list(FORGETTING),
            "snr_db": list(SNR_DB),
            "windows_s": list(WINDOWS_S),
            "band": list(BAND),
            "segments": {k: list(v) for k, v in SEGMENTS.items()},
            "segment_forgetting": SEGMENT_FORGETTING,
            "environment": environment(),
            **(meta or {}),
        },
        "rows": [
            {k: v for k, v in row.items() if k not in ("curves", "curve_axis")} for row in rows
        ],
        "courses": aggregate_courses(rows),
    }
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        # one row or cell per line: compact, and a change shows up as a line in a diff
        handle.write("{\n")
        handle.write(f'"kind": {json.dumps(payload["kind"])},\n')
        handle.write(f'"meta": {json.dumps(payload["meta"], indent=1)},\n')
        for key in ("rows", "courses"):
            handle.write(f'"{key}": [\n')
            handle.write(",\n".join(json.dumps(item) for item in payload[key]))
            handle.write("\n]" + ("," if key == "rows" else "") + "\n")
        handle.write("}\n")


def load_results(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read what :func:`save_results` wrote: ``(rows, meta)`` with ``meta["courses"]`` included."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("kind") != RESULTS_KIND:
        raise ValueError(f"{path} is not a {RESULTS_KIND} results file.")
    return payload["rows"], {**payload["meta"], "courses": payload.get("courses", [])}


def finite_mean(values: Sequence[float]) -> float:
    """Mean over the finite values, NaN when there are none."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else math.nan


# ----------------------------------------------------------------------------------------------
# The report
# ----------------------------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "–"
    return f"{value:.{digits}f}"


def _cells(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str, float | None], list[dict[str, Any]]]:
    out: dict[tuple[str, str, float | None], list[dict[str, Any]]] = {}
    for row in rows:
        if row["kind"] == "sweep":
            out.setdefault((row["scenario"], row["method"], row["forgetting"]), []).append(row)
    return out


def _cell_key_of(cell: dict[str, Any]) -> tuple[str, float, str, float | None]:
    return (cell["scenario"], float(cell["snr_db"]), cell["method"], cell["forgetting"])


def _by_snr(rows: Sequence[dict[str, Any]], snr_db: float) -> list[dict[str, Any]]:
    return [r for r in rows if r["snr_db"] == snr_db]


def _fmt_factor(factor: float | None) -> str:
    return "–" if factor is None else format(factor, ".10g")


def _latency_text(cell: dict[str, Any] | None) -> str:
    """Latencies of the mean course of a cell, one entry per switch, ``–`` if not tracked."""
    if cell is None:
        return "–"
    return " / ".join(
        _fmt(r["latency_s"], 1) if r["tracked"] else "–" for r in latency_of_cell(cell)
    )


def _switch_legend(cell: dict[str, Any], names: Sequence[str]) -> str:
    letter = {name: chr(ord("A") + i) for i, name in enumerate(names)}
    parts = [
        f"{letter[c['source']]}→{letter[c['target']]} at {sw:g} s"
        for c in cell["courses"]
        for sw in c["switches"]
    ]
    return ", ".join(parts)


def _run_lines(meta: dict[str, Any]) -> list[str]:
    """Describe the run (workers, efficiency, estimated and actual time) if it is known."""
    gate = meta.get("gate")
    if not gate or not meta.get("actual_wall_s"):
        return []
    return [
        f"- Run: {gate['n_datasets']} datasets on {gate['jobs']} workers. The gate measured "
        f"{gate['single_dataset_s']:.1f} s for one dataset alone and a parallel efficiency of "
        f"{gate['efficiency']:.2f} on this machine and estimated "
        f"{gate['estimate_s'] / 60:.1f} min; the run took {meta['actual_wall_s'] / 60:.1f} min.",
    ]


def _order_labels(labels: Sequence[str]) -> list[str]:
    known = method_labels()
    return sorted(set(labels), key=lambda m: known.index(m) if m in known else len(known))


def render_report(rows: list[dict[str, Any]], meta: dict[str, Any], *, findings: str = "") -> str:
    """Render the rows as a Markdown report: setup, one table per scenario, segment comparison.

    The tables hold means over the seeds (latencies come from the mean course over the seeds); the
    text says what was observed and ranks nothing.
    """
    sweep = _cells(rows)
    cells = {_cell_key_of(c): c for c in (meta.get("courses") or aggregate_courses(rows))}
    scenarios = [sc for sc in SCENARIOS if any(k[0] == sc for k in sweep)]
    methods = _order_labels([k[1] for k in sweep])
    snrs = sorted({float(r["snr_db"]) for r in rows}, reverse=True)
    seeds = sorted({r["seed_index"] for r in rows})
    env = meta.get("environment", {})
    out = [
        "# Non-stationary validation",
        "",
        "How well the time-resolved methods follow a coupling that switches on, off, around or "
        "briefly on, on region-level series with known truth. Generated by "
        "`scripts/run_nonstationary.py`; do not edit the tables by hand.",
        "",
    ]
    if findings.strip():
        out += [findings.strip(), ""]
    out += [
        "## Setup",
        "",
        f"- {len(seeds)} seeds per scenario and SNR; 60 s at 100 Hz; three regions oscillating "
        "at 10, 20 and 15 Hz (A, B, C in the order of the scenario file); coupling 0.4, "
        "delay 10 ms. "
        "The scenarios are the shipped files `"
        + "`, `".join(scenarios)
        + "`; B → C is a constant control edge in all of them, so the truth always has an edge "
        "and a non-edge and the AUROC is defined at every time point.",
        f"- Independent white observation noise at {', '.join(f'{x:g}' for x in snrs)} dB "
        "(signal over noise, per channel); no forward model.",
        f"- Every method sees the same series. Spectral results are averaged over {BAND[0]:g} to "
        f"{BAND[1]:g} Hz and scored by magnitude.",
        "- Methods: `adtf-dual` and `adtf-dekf` (ADTF with the dual and the dual extended "
        "Kalman recursion), `tpdc` (time-varying PDC, DEKF), `wdtf-4s` and `wdtf-8s` (one fit "
        "per window of 4 s or 8 s, step half a window). `adtf-rls` is not run: its coefficients "
        "are identical to those of `adtf-dual`. The order is chosen once by "
        "BIC (cap 5) on the whole recording; ADTF and TPDC output every 0.1 s.",
        "- **forgetting** is the factor `lambda` of ADTF and TPDC. In this recursion it sets the "
        "process noise `1 / lambda - 1`; it is not the memory of a forgetting-factor least-squares "
        "filter. The **memory** column is *measured* (the reciprocal of the mean gain along the "
        "regressor, in seconds) and is far shorter than `1 / (1 - lambda)` samples.",
        "- Environment: " + ", ".join(f"{k} {v}" for k, v in env.items()) + ".",
        *_run_lines(meta),
        "",
        "## How to read the numbers",
        "",
        "- **AUROC (stable)**: at every time point, how often a true edge outranks a non-edge, "
        "averaged over the time points outside the *guard band* after each switch and over "
        "the seeds. The guard is one third of the shortest stretch between switches (10 s; "
        "3.3 s for `burst`) and the same for every method. The memory-guard and no-guard "
        "variants are in the results file (`auroc_memory`, `auroc_all`).",
        "- **direction accuracy (stable)**: among the pairs that are an edge in one direction "
        "only, how often the stronger direction is the true one (chance 0.5), same time points.",
        "- **latency (s)**: the time from the switch until the mean score of the switching edge "
        "has covered half the way between its levels before and after (levels over the stable "
        "parts of the neighbouring stretches), and stayed beyond it for 1 s. It is taken from "
        "the mean course over the seeds because a single fit is too noisy (a score already "
        "beyond half the way when the switch happens counts as an instant arrival). Entries "
        "are in the order "
        "listed under the table; `–` means not tracked or not measurable (the stretch is shorter "
        "than the guard, or the levels do not differ). The windowed methods give times on a grid "
        "of half a window, so their latency is a multiple of 2 s or 4 s at best.",
        "- **unstable**: median share of the time points whose model has a companion spectral "
        "radius of 1 or more (windowed methods: share of windows). **warned**: share of fits "
        "that raised a diagnostic.",
        "",
    ]

    # overview: mean stable AUROC over scenarios, per method and factor, by SNR
    out += [
        "## Overview",
        "",
        "Mean AUROC (stable) over the four scenarios, by SNR, with the median measured memory.",
        "",
        "| method | forgetting | memory (s) | " + " | ".join(f"{x:g} dB" for x in snrs) + " |",
        "|---|---|---|" + "---|" * len(snrs),
    ]
    for method in methods:
        for factor in sorted(
            {k[2] for k in sweep if k[1] == method}, key=lambda f: -1 if f is None else f
        ):
            per_snr = []
            memory = []
            for snr in snrs:
                values = [
                    r["auroc_stable"]
                    for sc in scenarios
                    for r in _by_snr(sweep.get((sc, method, factor), []), snr)
                ]
                per_snr.append(finite_mean(values))
            for sc in scenarios:
                memory += [
                    r["memory_s"]
                    for r in sweep.get((sc, method, factor), [])
                    if r["memory_s"] is not None
                ]
            out.append(
                f"| {method} | {_fmt_factor(factor)} "
                f"| {_fmt(float(np.median(memory)) if memory else None, 2)} | "
                + " | ".join(_fmt(v, 3) for v in per_snr)
                + " |"
            )
    out.append("")

    # one table per scenario
    out += ["## Scenarios", ""]
    for scenario in scenarios:
        spec = load_scenario(scenario)
        any_cell = next(c for k, c in cells.items() if k[0] == scenario)
        out += [
            f"### {scenario}",
            "",
            spec.description,
            "",
            f"Guard band {any_cell['guard_s']:.1f} s. Latency entries: "
            + _switch_legend(any_cell, spec.network.names)
            + ".",
            "",
            "| method | forgetting | memory (s) | "
            + " | ".join(f"AUROC {x:g} dB" for x in snrs)
            + " | direction acc. | "
            + " | ".join(f"latency {x:g} dB" for x in snrs)
            + " | unstable | warned |",
            "|---|---|---|" + "---|" * (2 * len(snrs) + 3),
        ]
        for method in methods:
            for factor in sorted(
                {k[2] for k in sweep if k[0] == scenario and k[1] == method},
                key=lambda f: -1 if f is None else f,
            ):
                cell = sweep[(scenario, method, factor)]
                memory = [r["memory_s"] for r in cell if r["memory_s"] is not None]
                warned = sum(1 for r in cell if r["diagnostic_warnings"]) / len(cell)
                unstable = [
                    r["unstable_fraction"] for r in cell if math.isfinite(r["unstable_fraction"])
                ]
                out.append(
                    f"| {method} | {_fmt_factor(factor)} "
                    f"| {_fmt(float(np.median(memory)) if memory else None, 2)} | "
                    + " | ".join(
                        _fmt(finite_mean([r["auroc_stable"] for r in _by_snr(cell, snr)]), 3)
                        for snr in snrs
                    )
                    + f" | {_fmt(finite_mean([r['direction_stable'] for r in cell]), 2)} | "
                    + " | ".join(
                        _latency_text(cells.get((scenario, snr, method, factor))) for snr in snrs
                    )
                    + f" | {_fmt(float(np.median(unstable)) if unstable else None, 2)} "
                    f"| {warned:.0%} |"
                )
        out.append("")

    out += segment_section(rows)
    return "\n".join(out) + "\n"


def segment_section(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Render the comparison of ``reduce_time`` on an ADTF with ``WDTF(segments=...)``."""
    seg = [r for r in rows if r["kind"] == "segments"]
    if not seg:
        return []
    out = [
        "## One value per segment: continuous ADTF or a fit per segment",
        "",
        "Two ways to get a result for a stretch of the recording: run ADTF over the whole "
        f"recording and average it over the segment (`reduce_time`; forgetting "
        f"{SEGMENT_FORGETTING:g}), or fit the segment on its own (`WDTF(segments=...)`). The "
        f"segments are A = {SEGMENTS['A'][0]:g} to {SEGMENTS['A'][1]:g} s (well before the "
        f"switch at 30 s) and B = {SEGMENTS['B'][0]:g} to {SEGMENTS['B'][1]:g} s (1 s to 6 s "
        "after it). The first approach carries the memory of the preceding samples and the "
        "transition after the switch; the second has only the 5 s of segment B to estimate "
        "from and depends on the order chosen for the whole recording. Scores are the AUROC "
        "against the truth at the centre of the segment and the mean score of the switching "
        "pair(s); the table shows what was observed in these two scenarios, not which "
        "approach is better.",
        "",
        "| scenario | SNR | approach | AUROC A | AUROC B | switching pair | score A | score B |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for scenario in SEGMENT_SCENARIOS:
        snrs = sorted({r["snr_db"] for r in seg if r["scenario"] == scenario}, reverse=True)
        for snr in snrs:
            for approach in ("adtf_reduce_time", "wdtf_segments"):
                cell = [
                    r
                    for r in seg
                    if r["scenario"] == scenario
                    and r["snr_db"] == snr
                    and r["approach"] == approach
                ]
                if not cell:
                    continue
                a = [r for r in cell if r["segment"] == "A"]
                b = [r for r in cell if r["segment"] == "B"]
                for pair in a[0]["pair_scores"]:
                    out.append(
                        f"| {scenario} | {snr:g} dB | {approach} "
                        f"| {_fmt(finite_mean([r['auroc'] for r in a]), 3)} "
                        f"| {_fmt(finite_mean([r['auroc'] for r in b]), 3)} | {pair} "
                        f"| {_fmt(finite_mean([r['pair_scores'][pair] for r in a]), 3)} "
                        f"| {_fmt(finite_mean([r['pair_scores'][pair] for r in b]), 3)} |"
                    )
    out.append("")
    return out


def datasets(
    *, seeds: int = N_SEEDS, snr_db: Sequence[float] = SNR_DB, scenarios: Sequence[str] = SCENARIOS
) -> list[tuple[str, float, int]]:
    """Return the datasets of a run as ``(scenario, snr_db, seed index)``, in a fixed order."""
    return [(sc, float(snr), k) for sc in scenarios for snr in snr_db for k in range(seeds)]


def run_dataset(task: tuple[str, float, int]) -> dict[str, Any]:
    """Evaluate one dataset with the default settings; the unit of work of a (parallel) run."""
    started = time.perf_counter()
    scenario, snr_db, k = task
    rows = evaluate_dataset(scenario, snr_db, k)
    return {"task": [scenario, snr_db, k], "rows": rows, "seconds": time.perf_counter() - started}
