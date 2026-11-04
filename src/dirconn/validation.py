"""Reference scenarios for checking what each method can and cannot see.

Every scenario is a known network of oscillating regions, simulated at the region level (no
forward model, no mixing). The methods are scored on the estimated matrix against the network's
ground truth with :mod:`dirconn.evaluation`. The scenarios include cases the methods are expected
to get right (chain, common driver, collider, bidirectional pair, feedback loop) and cases that
break their assumptions on purpose (instantaneous coupling, a hidden common driver, observation
noise), so that the limits are measured rather than assumed.

This is the first, small version of the benchmark scenarios (``docs/benchmark.md``).
"""

from __future__ import annotations

import json
import math
import platform
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from dirconn.coherency import PSI, ImCoh
from dirconn.estimator import Estimator
from dirconn.evaluation import (
    chance_auprc,
    convention_metrics,
    edge_density,
)
from dirconn.exceptions import DirconnWarning
from dirconn.granger import GC
from dirconn.multitaper_granger import MTGC
from dirconn.network import Edge, Network, dk_node
from dirconn.pcmci import PCMCI
from dirconn.result import Connectivity
from dirconn.simulation import NodeDynamics, simulate
from dirconn.spectral_estimators import DTF, GPDC, PDC, RPDC
from dirconn.time_reversal import TRGC
from dirconn.timeseries import TimeSeries
from dirconn.transfer_entropy import TE

SFREQ = 250.0
SPECTRAL_BAND = (1.0, 80.0)
SEGMENT_LENGTH = 2.0
"""Multitaper segment of the coherency-based methods and MTGC, in seconds (0.5 Hz resolution)."""
"""Band over which the spectral measures are averaged before scoring."""

A, B, C, D, H = (
    "bankssts-lh",
    "superiorparietal-lh",
    "precuneus-rh",
    "insula-rh",
    "lingual-lh",
)

_DYNAMICS = {
    A: NodeDynamics(freq_hz=10.0, bandwidth_hz=4.0),
    B: NodeDynamics(freq_hz=20.0, bandwidth_hz=4.0),
    C: NodeDynamics(freq_hz=15.0, bandwidth_hz=4.0),
    D: NodeDynamics(freq_hz=25.0, bandwidth_hz=4.0),
    H: NodeDynamics(freq_hz=12.0, bandwidth_hz=4.0),
}


@dataclass(frozen=True)
class Scenario:
    """A known network and what to expect from the methods on it.

    Attributes
    ----------
    name : str
        Short identifier.
    description : str
        The network in words.
    network : Network
        All nodes, including any hidden one.
    analysed : tuple of str
        Nodes the methods see; a node left out is hidden from them.
    noise_db : float or None
        Observation noise added to the analysed series at this SNR in dB (None: none).
    white_nodes : tuple of str
        Nodes with white (memoryless) dynamics instead of an oscillator.
    expectation : str
        What a correct method should show and what limitation the scenario probes.
    """

    name: str
    description: str
    network: Network
    analysed: tuple[str, ...]
    noise_db: float | None
    expectation: str
    white_nodes: tuple[str, ...] = ()

    @property
    def dynamics(self) -> dict[str, NodeDynamics]:
        """Own dynamics of the nodes: a 10 to 25 Hz oscillator each, or white where requested."""
        white = NodeDynamics(kind="white")
        return {
            name: white if name in self.white_nodes else _DYNAMICS[name]
            for name in self.network.names
        }


def _network(nodes: list[str], edges: list[Edge]) -> Network:
    return Network(nodes=tuple(dk_node(n) for n in nodes), edges=tuple(edges))


def _scenarios() -> dict[str, Scenario]:
    four = [A, B, C, D]
    items = [
        Scenario(
            "chain",
            "A -> B -> C; D unconnected.",
            _network(four, [Edge(A, B, 0.4, 8.0), Edge(B, C, 0.4, 8.0)]),
            (A, B, C, D),
            None,
            "Both edges found with the right direction. Pairwise methods and the transfer-function "
            "measure also report the indirect path A -> C; conditional Granger and PDC should not.",
        ),
        Scenario(
            "common_driver",
            "A -> B and A -> C with different delays; D unconnected.",
            _network(four, [Edge(A, B, 0.4, 8.0), Edge(A, C, 0.4, 16.0)]),
            (A, B, C, D),
            None,
            "Both edges found. Pairwise methods may report B <-> C, which share a driver.",
        ),
        Scenario(
            "collider",
            "A -> C and B -> C; D unconnected.",
            _network(four, [Edge(A, C, 0.4, 8.0), Edge(B, C, 0.4, 8.0)]),
            (A, B, C, D),
            None,
            "Both edges found, nothing between A and B.",
        ),
        Scenario(
            "bidirectional",
            "A <-> B with different delays, C -> D; the direction of A <-> B is not scored.",
            _network(four, [Edge(A, B, 0.05, 8.0), Edge(B, A, 0.05, 12.0), Edge(C, D, 0.4, 8.0)]),
            (A, B, C, D),
            None,
            "Edges in both directions of A <-> B (weak, 0.05: a feedback loop of resonant regions "
            "becomes unstable at much lower coupling than a one-way edge), and C -> D with the "
            "right direction.",
        ),
        Scenario(
            "feedback_loop",
            "A -> B -> C -> A; D unconnected.",
            _network(
                four,
                [Edge(A, B, 0.05, 8.0), Edge(B, C, 0.05, 8.0), Edge(C, A, 0.05, 8.0)],
            ),
            (A, B, C, D),
            None,
            "All three edges found (weak, 0.05, for the same stability reason); the loop is the "
            "case where pairwise methods see an edge between every pair of the loop.",
        ),
        Scenario(
            "instantaneous",
            "A -> B with zero delay between two white (memoryless) regions, C -> D with a delay "
            "of 8 ms between oscillators.",
            _network(four, [Edge(A, B, 0.5, 0.0), Edge(C, D, 0.4, 8.0)]),
            (A, B, C, D),
            None,
            "Limit: a zero-lag influence between memoryless regions leaves no temporal precedence "
            "at all, so A -> B cannot be found; C -> D is the control that must still work. The "
            "direction score of a run is 0.5 or 1 (A -> B is a coin flip, C -> D is right), so "
            "chance is a mean of 0.75. (With an oscillating source the zero-lag edge is found, "
            "through the source's own autocorrelation: 6 of 6 seeds.)",
            white_nodes=(A, B),
        ),
        Scenario(
            "hidden_driver",
            "A -> C, plus an unobserved H driving both A and B; the methods see only A, B, C.",
            _network(
                [H, A, B, C],
                [Edge(H, A, 0.5, 8.0), Edge(H, B, 0.5, 16.0), Edge(A, C, 0.4, 8.0)],
            ),
            (A, B, C),
            None,
            "Limit: causal sufficiency is violated, so an edge between A and B appears although "
            "there is none. Granger causality still ranks the true edge A -> C first but the "
            "spurious one is about half as strong; PDC and DTF are shares, and rank it above.",
        ),
        Scenario(
            "noise_m10db",
            "The chain A -> B -> C with independent observation noise ten times as strong as the "
            "signal (-10 dB) on every channel.",
            _network(four, [Edge(A, B, 0.4, 8.0), Edge(B, C, 0.4, 8.0)]),
            (A, B, C, D),
            -10.0,
            "Limit: measurement noise turns the process into a VARMA one and weakens "
            "Granger-type measures; the ranking starts to degrade.",
        ),
        Scenario(
            "noise_m15db",
            "The same chain with observation noise 30 times as strong as the signal (-15 dB).",
            _network(four, [Edge(A, B, 0.4, 8.0), Edge(B, C, 0.4, 8.0)]),
            (A, B, C, D),
            -15.0,
            "Limit: the breaking point; the ranking is close to chance.",
        ),
    ]
    return {item.name: item for item in items}


SCENARIOS: dict[str, Scenario] = _scenarios()
"""The reference scenarios by name."""

METHODS: dict[str, Callable[[], Estimator]] = {
    "gc": lambda: GC(order="bic", max_order=12),
    "gc_pairwise": lambda: GC(order="bic", max_order=12, conditional=False),
    "dtf": lambda: DTF(order="bic", max_order=12),
    "pdc": lambda: PDC(order="bic", max_order=12),
    "rpdc": lambda: RPDC(order="bic", max_order=12),
    "gpdc": lambda: GPDC(order="bic", max_order=12),
    "trgc": lambda: TRGC(order="bic", max_order=12),
    "imcoh": lambda: ImCoh(segment_length=SEGMENT_LENGTH),
    "psi": lambda: PSI(segment_length=SEGMENT_LENGTH, bands=[SPECTRAL_BAND]),
    "mtgc": lambda: MTGC(segment_length=SEGMENT_LENGTH),
    "pcmci": lambda: PCMCI(tau_max="bic", max_order=12),
    "te": lambda: TE(mode="pairwise"),
}
"""Estimator factories by label; the spectral measures are scored on their mean over 1 to 80 Hz.

An undirected measure (``imcoh``) is scored on its magnitude against the undirected truth (an edge
in either direction), and its direction accuracy is undefined. A directed antisymmetric measure
(``trgc``, ``psi``: a net flow) is scored against the net truth: a pair with true edges in both
directions is left out, and a true edge that scores below zero (the net flow points the other way)
counts as zero in the separation. A signed measure (``pcmci``: a partial correlation) is scored
on its magnitude, because a negative coupling is still an edge."""

VAR_METHODS: tuple[str, ...] = ("gc", "gc_pairwise", "dtf", "pdc", "rpdc", "gpdc")
"""The methods read from one pooled VAR fit (they share its order and its failures)."""


def simulate_scenario(scenario: Scenario, *, seed: int, n_times: int = 10000) -> TimeSeries:
    """Simulate a scenario and return the series the methods are allowed to see.

    The noise-free node activity is taken at the region level, restricted to ``scenario.analysed``
    and, if the scenario asks for it, with independent white observation noise added to every
    channel at exactly ``noise_db``.
    """
    sim = simulate(
        scenario.network,
        sfreq=SFREQ,
        n_times=n_times,
        dynamics=scenario.dynamics,
        seed=seed,
        n_sensors=4,
        n_background=0,
    )
    series = sim.sources.pick(list(scenario.analysed))
    if scenario.noise_db is None:
        return series
    return add_observation_noise(series, scenario.noise_db, seed)


def add_observation_noise(series: TimeSeries, noise_db: float, seed: int) -> TimeSeries:
    """Add independent white noise to every channel at exactly ``noise_db`` (signal over noise).

    The noise power of a channel is its signal power over ``10 ** (noise_db / 10)``, measured over
    all trials and samples; the noise comes from ``seed + 10_000`` so that it does not depend on the
    random numbers of the simulation.
    """
    rng = np.random.default_rng(seed + 10_000)
    data = np.array(series.data)
    noise = rng.standard_normal(data.shape)
    power = np.mean(data**2, axis=(0, 2)) / 10.0 ** (noise_db / 10.0)
    noise *= np.sqrt(power / np.mean(noise**2, axis=(0, 2)))[None, :, None]
    return TimeSeries(
        data=data + noise,
        sfreq=series.sfreq,
        channels=series.channels,
        segmentation=series.segmentation,
        tmin=series.tmin,
    )


def truth_matrix(scenario: Scenario) -> np.ndarray:
    """Binary ``[source, target]`` truth over the analysed nodes (diagonal NaN)."""
    names = scenario.network.names
    keep = [names.index(n) for n in scenario.analysed]
    return scenario.network.truth().values[:, :, 0, 0][np.ix_(keep, keep)]


def score_matrix(estimator: Estimator, series: TimeSeries) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit ``estimator`` and return a 2-D score matrix plus facts about the fit.

    Time-domain results are used as they are; spectral results are averaged over
    ``SPECTRAL_BAND``. Diagnostics are collected, not raised.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DirconnWarning)
        result = estimator.fit(series)
    if not estimator.info.directed or estimator.info.signed:
        # the sign of an undirected measure is a phase and that of a signed one (a correlation)
        # the sign of the coupling, not its strength: score the magnitude
        result = Connectivity(
            values=np.abs(result.values),
            source=result.source,
            target=result.target,
            freqs=result.freqs,
            times=result.times,
            provenance=result.provenance.with_step({"op": "abs"}),
        )
    if result.is_frequency_resolved:
        result = result.reduce_freq(SPECTRAL_BAND, how="mean")
    facts = {
        "order": result.provenance.model_order,
        "diagnostic_warnings": sorted(
            c.name for c in result.provenance.assumptions.checks if c.status in ("warn", "fail")
        ),
    }
    return result.values[:, :, 0, 0], facts


def evaluate(scenario: Scenario, method: str, *, seed: int, n_times: int = 10000) -> dict[str, Any]:
    """Run one method on one simulated scenario and score it against the truth.

    Returns
    -------
    dict
        ``auroc``, ``auprc`` and ``direction_accuracy`` (NaN where undefined), the selected VAR
        ``order`` and the names of any diagnostics that warned.
    """
    series = simulate_scenario(scenario, seed=seed, n_times=n_times)
    return evaluate_series(scenario, method, series)


def evaluate_series(scenario: Scenario, method: str, series: TimeSeries) -> dict[str, Any]:
    """Score one method on an already simulated series of ``scenario``; see :func:`evaluate`."""
    estimator = METHODS[method]()
    scores, facts = score_matrix(estimator, series)
    truth = truth_matrix(scenario)
    metrics = convention_metrics(scores, truth, estimator.info)
    return {**metrics, **facts}


def run_all(
    *,
    seeds: Sequence[int],
    n_times: int = 10000,
    scenarios: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Run methods on scenarios for several seeds and collect one row per run.

    Each scenario is simulated once per seed and every method is scored on that same series, so
    the methods are compared on identical data.

    Parameters
    ----------
    seeds : sequence of int
        Random seeds; one simulation per scenario and seed.
    n_times : int
        Samples per simulation (at 250 Hz).
    scenarios, methods : sequence of str, optional
        Names from :data:`SCENARIOS` and :data:`METHODS`; default all.
    progress : callable, optional
        Called with a short message after every simulation.

    Returns
    -------
    list of dict
        Rows with ``scenario``, ``method``, ``seed`` and the output of :func:`evaluate`.
    """
    scenario_names = list(SCENARIOS) if scenarios is None else list(scenarios)
    method_names = list(METHODS) if methods is None else list(methods)
    unknown = [n for n in scenario_names if n not in SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenario(s) {unknown}; known: {list(SCENARIOS)}.")
    unknown = [m for m in method_names if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown method(s) {unknown}; known: {list(METHODS)}.")
    rows: list[dict[str, Any]] = []
    for name in scenario_names:
        scenario = SCENARIOS[name]
        for seed in seeds:
            series = simulate_scenario(scenario, seed=seed, n_times=n_times)
            for method in method_names:
                rows.append(
                    {
                        "scenario": name,
                        "method": method,
                        "seed": seed,
                        **evaluate_series(scenario, method, series),
                    }
                )
            if progress is not None:
                progress(f"{name} seed {seed}")
    return rows


def environment() -> dict[str, str]:
    """Return the versions and platform a set of numbers was produced with."""
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "scipy", "dirconn"):
        info[package] = metadata.version(package)
    return info


def save_rows(
    rows: list[dict[str, Any]], path: str | Path, *, n_times: int, seeds: Sequence[int]
) -> None:
    """Write the rows and the run settings to a JSON file (``NaN`` and ``Infinity`` allowed)."""
    payload = {
        "meta": {
            "n_times": n_times,
            "seeds": list(seeds),
            "sfreq": SFREQ,
            "spectral_band": list(SPECTRAL_BAND),
            "environment": environment(),
        },
        "rows": rows,
    }
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")


def load_rows(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read what :func:`save_rows` wrote; returns ``(rows, meta)``."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["rows"], payload["meta"]


def _median(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[~np.isnan(array)]
    return float("nan") if finite.size == 0 else float(np.median(finite))


def _mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[~np.isnan(array)]
    return float("nan") if finite.size == 0 else float(np.mean(finite))


def _fmt(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}"


def _fmt_separation(value: float) -> str:
    """Format a separation ratio; above 100 the exact value (kept in the saved rows) is noise."""
    if not (math.isnan(value) or math.isinf(value)) and value >= 100.0:
        return ">100"
    return _fmt(value)


def render_report(
    rows: list[dict[str, Any]],
    *,
    n_times: int,
    seeds: Sequence[int],
    findings: str = "",
    env: dict[str, str] | None = None,
) -> str:
    """Render the rows as a Markdown report: overview tables, then one table per scenario.

    Numbers are medians over the seeds; AUROC also shows the range. Only the scenarios and methods
    present in ``rows`` appear.
    """
    env = environment() if env is None else env
    scenario_names = [n for n in SCENARIOS if any(r["scenario"] == n for r in rows)]
    method_names = [m for m in METHODS if any(r["method"] == m for r in rows)]

    def column(scenario: str, method: str, key: str) -> list[float]:
        # a method without a VAR model has no order (None): NaN, shown as "n/a"
        return [
            math.nan if r[key] is None else float(r[key])
            for r in rows
            if r["scenario"] == scenario and r["method"] == method
        ]

    def overview(
        key: str,
        title: str,
        aggregate: Callable[[Sequence[float]], float] = _median,
        fmt: Callable[[float], str] = _fmt,
    ) -> list[str]:
        lines = [f"**{title}**", "", "| scenario | " + " | ".join(method_names) + " |"]
        lines.append("|---|" + "---|" * len(method_names))
        for name in scenario_names:
            cells = [fmt(aggregate(column(name, m, key))) for m in method_names]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        return [*lines, ""]

    out = [
        "# Method validation",
        "",
        "What each method can and cannot see on networks whose truth is known. Generated by "
        "`scripts/validate_methods.py`; do not edit the tables by hand.",
        "",
    ]
    if findings.strip():
        out += [findings.strip(), ""]
    out += [
        "## Setup",
        "",
        f"- {len(seeds)} seeds ({min(seeds)} to {max(seeds)}), {n_times} samples at {SFREQ:g} Hz "
        "per simulation; every method sees the same simulated series.",
        "- Region-level series (no forward model, no mixing). Four oscillating regions of 10 to "
        "25 Hz unless a scenario says otherwise.",
        "- Time-domain measures are scored as they are; the spectral measures are averaged "
        f"over {SPECTRAL_BAND[0]:g} to {SPECTRAL_BAND[1]:g} Hz first. `imcoh` is undirected: its "
        "magnitude is scored against an edge in either direction, without direction accuracy. "
        "`trgc` and `psi` are net flows: a pair coupled both ways is left out (in "
        "`bidirectional` only C -> D is scored), and a true edge whose net flow points the other "
        "way counts as 0 in the separation. `pcmci` is signed (a partial correlation): its "
        "magnitude is scored.",
        "- Estimators (the VAR-based ones and `pcmci`'s `tau_max` choose the order once by BIC, "
        "at most 12 lags; `imcoh`, "
        f"`psi` and `mtgc` use {SEGMENT_LENGTH:g} s multitaper segments):",
        *[f"  - `{m}`: `{METHODS[m]()!r}`" for m in method_names],
        "- Environment: " + ", ".join(f"{k} {v}" for k, v in env.items()) + ".",
        "",
        "## How to read the numbers",
        "",
        "- **AUROC**: how often a true edge outranks a non-edge (0.5 is chance, 1 is perfect). "
        "Optimistic on sparse networks, so it is shown with the range over seeds.",
        "- **AUPRC**: average precision; its chance level is listed for every scenario.",
        "- **direction accuracy**: among pairs that are an edge in one direction only, how often "
        "the stronger direction is the true one (chance 0.5). With only one or two such pairs "
        "a run scores 0.5 or 1, so it is averaged over the seeds instead of taking a median.",
        "- **separation**: weakest true edge divided by strongest non-edge. Above 1 every true "
        "edge beats every non-edge, and the number says by how much; below 1 a spurious edge "
        "beats a true one.",
        "- **order**: median selected VAR order. **warned**: seeds in which a diagnostic "
        "(stationarity, sampling, order at its limit) raised a warning.",
        "",
        "## Overview",
        "",
        *overview("auroc", "Median AUROC"),
        *overview(
            "separation",
            "Median separation (weakest true edge / strongest spurious edge)",
            fmt=_fmt_separation,
        ),
        *overview("direction_accuracy", "Mean direction accuracy", aggregate=_mean),
        "## Scenarios",
        "",
    ]
    for name in scenario_names:
        scenario = SCENARIOS[name]
        truth = truth_matrix(scenario)
        out += [
            f"### {name}",
            "",
            scenario.description,
            "",
            f"*Expected:* {scenario.expectation}",
            "",
            f"Chance AUPRC for this truth: {_fmt(chance_auprc(truth))} "
            f"(edge density {_fmt(edge_density(truth))}).",
            "",
            "| method | AUROC (range) | AUPRC | direction acc. (mean) | separation "
            "| order | warned |",
            "|---|---|---|---|---|---|---|",
        ]
        for method in method_names:
            if not column(name, method, "auroc"):
                continue
            aurocs = column(name, method, "auroc")
            warned = sum(
                1
                for r in rows
                if r["scenario"] == name and r["method"] == method and r["diagnostic_warnings"]
            )
            out.append(
                f"| {method} | {_fmt(_median(aurocs))} ({_fmt(min(aurocs))}-{_fmt(max(aurocs))}) "
                f"| {_fmt(_median(column(name, method, 'auprc')))} "
                f"| {_fmt(_mean(column(name, method, 'direction_accuracy')))} "
                f"| {_fmt_separation(_median(column(name, method, 'separation')))} "
                f"| {_median(column(name, method, 'order')):g} "
                f"| {warned}/{len(aurocs)} |"
            )
        out.append("")
    return "\n".join(out) + "\n"
