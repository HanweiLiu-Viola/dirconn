"""Benchmark protocols: what to run, how to run it again, and how to read what came out.

A protocol is a YAML file (``version: 1``) that lists axes; the runner takes their Cartesian
product. A **task** is one simulation: a scenario, an SNR, a data length, a coupling strength, a
source model, a modality and a seed number. The estimators are not part of the task: the runner
simulates once and applies every estimator of the protocol to the same data. Everything here is
plain logic; the function that actually simulates and estimates is passed in as ``runner``
(see :mod:`dirconn.sweep`), so the bookkeeping can be tested without any data set.

Seeds come from the content of a task, not from its position in the list, so adding a value to an
axis never changes the seed of a task that already exists. Results go to a JSON Lines file, one
line per finished task, which is what makes a run resumable.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import json
import math
import os
import platform
import subprocess
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dirconn.headmodel import MODALITIES
from dirconn.inverse import INVERSE_METHODS
from dirconn.leakage import adjacent_report, leakage_correlation
from dirconn.network import Network, dk_node
from dirconn.noise_study import noise_report
from dirconn.rng import RNGController
from dirconn.scenario import ScenarioSpec, list_scenarios, load_scenario
from dirconn.significance_study import frequency_report, significance_report
from dirconn.simulation import SOURCE_MODELS, NodeDynamics

PROTOCOL_VERSION = 1
ESTIMATOR_METHODS: tuple[str, ...] = ("gc", "gc_pairwise", "dtf", "pdc", "gpdc")
ORDER_SETTINGS: tuple[str, ...] = ("bic", "fixed")
VAR_ESTIMATOR_NAMES: tuple[str, ...] = tuple(
    f"{method}/{order}" for method in ESTIMATOR_METHODS for order in ORDER_SETTINGS
)
"""Labels of the VAR-based estimators: a method and an order setting, ``"gc/bic"``."""
EXTENDED_ESTIMATOR_NAMES: tuple[str, ...] = (
    "trgc/bic",
    "imcoh/mt",
    "psi/mt",
    "mtgc/mt",
    "pcmci/bic",
    "te/bic",
)
"""Labels of the estimators beyond the VAR measures (zero-lag controls, MTGC, PCMCI, TE): ``bic``
orders (or ``tau_max``, target history) by BIC as for the VAR-based ones; ``mt`` uses multitaper
segments of ``MULTITAPER_SEGMENT_SECONDS``."""
CALIBRATION_ESTIMATOR_NAMES: tuple[str, ...] = ("pdc/low", "rpdc/bic", "rpdc/fixed", "rpdc/low")
"""Labels of the calibration of the asymptotic PDC and rPDC tests: rPDC at
the two order settings, and ``low``, a fixed order that is too low for the scenario on purpose
(``settings["low_order"]`` samples), for PDC and rPDC."""
ESTIMATOR_NAMES: tuple[str, ...] = (
    VAR_ESTIMATOR_NAMES + EXTENDED_ESTIMATOR_NAMES + CALIBRATION_ESTIMATOR_NAMES
)
"""Every estimator label a protocol may name."""
AXES: tuple[str, ...] = (
    "scenario",
    "snr_db",
    "duration_s",
    "coupling",
    "source_model",
    "modality",
    "noise",
)
NOISE_KINDS: tuple[str, ...] = ("simulated", "white", "empty_room")
"""Noise of the sensors in a head-model task: ``simulated`` (1/f background sources plus white
noise, the default), ``white`` (white noise only, no background) or ``empty_room`` (the real
MEG empty-room recording of the sample dataset, no background; :mod:`dirconn.emptyroom`)."""
RESULTS_KIND = "dirconn-benchmark"
DEFAULT_BUDGET_HOURS = 2.0
DEFAULT_EFFICIENCY = 0.8
_UNVERSIONED = {"platform"}
"""Environment entries that may change between a run and its resume."""

_PROTOCOL_KEYS = {
    "version",
    "name",
    "description",
    "scenarios",
    "estimators",
    "n_seeds",
    "master_seed",
    "snr_db",
    "duration_s",
    "coupling",
    "source_model",
    "modality",
    "inverse",
    "exclude",
    "budget_hours",
    "settings",
    "saturation_guard",
    "kind",
    "pairs",
    "tests",
    "noise",
}
_REQUIRED = ("name", "estimators", "n_seeds")
KINDS: tuple[str, ...] = ("scenarios", "region_pairs", "significance")
SURROGATE_NAMES: tuple[str, ...] = (
    "phase_randomize",
    "aaft",
    "iaaft",
    "circular_shift",
    "trial_shuffle",
)
TEST_NAMES: tuple[str, ...] = (
    "analytic",
    *(f"surrogate:{name}" for name in SURROGATE_NAMES),
)
"""Tests of a ``significance`` protocol: the analytic test of the estimator's class (the F test of
Granger causality, the asymptotic tests of PDC and rPDC), or a surrogate test with the named
generator (:func:`dirconn.significance_test`)."""
ANALYTIC_METHODS: tuple[str, ...] = ("gc", "gc_pairwise", "pdc", "rpdc")
"""Methods with an analytic test, the only ones a protocol with the ``analytic`` test may name."""
_PAIRS_KEYS = {"n", "seed", "regions"}
PAIR_FREQS_HZ = (10.0, 20.0)
"""Oscillation frequencies of the two nodes of a region-pair scenario."""


@dataclass(frozen=True)
class Protocol:
    """A benchmark protocol.

    Attributes
    ----------
    name, description : str
        Identifier (the file name without ``.yaml``) and a sentence on what it is for.
    scenarios : tuple of str
        Shipped scenario names (:func:`dirconn.scenario.list_scenarios`).
    estimators : tuple of str
        Labels from :data:`ESTIMATOR_NAMES`.
    n_seeds : int
        Seeds per grid point, numbered ``0`` to ``n_seeds - 1``.
    master_seed : int
        Seed all task seeds derive from.
    snr_db, duration_s : tuple of float or None
        Axis values; ``None`` takes the default of the scenario file.
    coupling : tuple of float or None
        Absolute strength that replaces every edge of the scenario (sign kept); ``0`` removes all
        edges (a zero network). ``None`` keeps the file's strengths. Only scenarios whose edges
        all share one strength can be swept.
    source_model, modality : tuple of str
        Source models and modalities (separate head models).
    inverse : tuple of str
        Inverse methods applied at stage 3 of every task.
    exclude : tuple of dict
        Entries ``{axis: value or list of values}``; a task matching every axis of an entry is
        dropped.
    budget_hours : float
        Estimated wall-clock time above which the run refuses to start without ``force``.
    settings : dict
        Scalar settings handed to the runner (for example ``max_order``); part of what a result
        file is compatible with.
    saturation_guard : bool
        Whether the results must pass :func:`saturation_report`: a grid on which even the best
        estimator is perfect everywhere on the easy chain says nothing about the methods.
    kind : {"scenarios", "region_pairs"}
        Where the scenarios come from. ``scenarios`` (default): the shipped files named in
        ``scenarios``. ``region_pairs``: ``pairs`` (``n``, ``seed``, ``regions``) makes ``n`` random
        pairs of the listed regions and every pair is a two-node network without any edge
        (:func:`region_pair_scenario`), the coupling fixed at 0: a study of what leaks between
        regions, not of what the methods find in a network.
    pairs : dict, optional
        The block of a ``region_pairs`` protocol.
    noise : tuple of str
        Noise of the sensors, an axis of ``kind: scenarios`` (:data:`NOISE_KINDS`). The three kinds
        of one seed share their sources, so the difference between them is the noise alone;
        ``empty_room`` is for MEG head models only. A task names its noise in its key only when
        it is not ``simulated``, so the ids and seeds of tasks made before the axis existed do
        not change.
    tests : tuple of str
        For ``kind: significance``: a study of the tests of :func:`dirconn.significance_test`
        at region level (no head model, no inverse step, so no ``modality``, ``source_model`` or
        ``inverse``). Every task is one simulated dataset and every test in ``tests`` is applied to
        it; the results are the p-value matrices (:mod:`dirconn.significance_study`).
    """

    name: str
    scenarios: tuple[str, ...]
    estimators: tuple[str, ...]
    n_seeds: int
    description: str = ""
    master_seed: int = 0
    snr_db: tuple[float | None, ...] = (None,)
    duration_s: tuple[float | None, ...] = (None,)
    coupling: tuple[float | None, ...] = (None,)
    source_model: tuple[str, ...] = ("partial",)
    modality: tuple[str, ...] = ("eeg",)
    inverse: tuple[str, ...] = ("mne",)
    exclude: tuple[dict[str, Any], ...] = ()
    budget_hours: float = DEFAULT_BUDGET_HOURS
    settings: dict[str, Any] = field(default_factory=dict)
    saturation_guard: bool = False
    kind: str = "scenarios"
    pairs: dict[str, Any] | None = None
    tests: tuple[str, ...] = ()
    noise: tuple[str, ...] = ("simulated",)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Protocol:
        """Build a protocol from a dictionary, for example loaded from YAML.

        Raises
        ------
        ValueError
            For unknown keys, missing keys, unknown names and numbers out of range; the message
            names the offender.
        """
        _check_keys(data, _PROTOCOL_KEYS, "the protocol")
        if "version" not in data:
            raise ValueError(f"The protocol is missing 'version' (supported: {PROTOCOL_VERSION}).")
        if data["version"] != PROTOCOL_VERSION or isinstance(data["version"], bool):
            raise ValueError(
                f"Unsupported protocol version {data['version']!r}; supported: {PROTOCOL_VERSION}."
            )
        for key in _REQUIRED:
            if key not in data:
                raise ValueError(f"The protocol is missing the required key {key!r}.")
        kind = data.get("kind", "scenarios")
        if kind not in KINDS:
            raise ValueError(f"Unknown kind value {kind!r}; known: {list(KINDS)}.")
        pairs: dict[str, Any] | None = None
        tests: tuple[str, ...] = ()
        if kind != "significance" and "tests" in data:
            raise ValueError("The key 'tests' belongs to a protocol of kind 'significance'.")
        if kind != "scenarios" and "noise" in data:
            raise ValueError(
                f"The key 'noise' belongs to a protocol of kind 'scenarios'; a protocol of kind "
                f"{kind!r} has no head-model noise to choose."
            )
        if kind in ("scenarios", "significance"):
            if "pairs" in data:
                raise ValueError("The key 'pairs' belongs to a protocol of kind 'region_pairs'.")
            shipped = list_scenarios()
            scenarios = _axis(data, "scenarios", None, lambda v: _choice(v, shipped, "scenarios"))
            for name in scenarios:
                if load_scenario(name).network.has_intervals:
                    raise ValueError(
                        f"Scenario {name!r} has edges that act only in some time intervals. The "
                        "stages of this benchmark are scored against one static truth, so its "
                        "numbers would be about another experiment; use the time-resolved study "
                        "(scripts/run_nonstationary.py)."
                    )
            if kind == "significance":
                for key in ("modality", "source_model", "inverse"):
                    if key in data:
                        raise ValueError(
                            f"A protocol of kind 'significance' works at region level and has no "
                            f"{key!r}."
                        )
                tests = _axis(data, "tests", None, lambda v: _choice(v, TEST_NAMES, "tests"))
        else:
            if "scenarios" in data:
                raise ValueError(
                    "A protocol of kind 'region_pairs' makes its scenarios from "
                    "'pairs'; 'scenarios' must not be given."
                )
            if "coupling" in data:
                raise ValueError(
                    "A protocol of kind 'region_pairs' has no 'coupling' axis: its "
                    "networks have no edge."
                )
            pairs = _pairs_block(data.get("pairs"))
            scenarios = ()
        estimators = _axis(
            data, "estimators", None, lambda v: _choice(v, ESTIMATOR_NAMES, "estimators")
        )
        snr = _axis(data, "snr_db", (None,), lambda v: _number(v, "snr_db"), allow_none=True)
        duration = _axis(
            data,
            "duration_s",
            (None,),
            lambda v: _number(v, "duration_s", low=0.0, open_low=True),
            allow_none=True,
        )
        coupling = _axis(
            data,
            "coupling",
            (None,),
            lambda v: _number(v, "coupling", low=0.0),
            allow_none=True,
        )
        source_model = _axis(
            data, "source_model", ("partial",), lambda v: _choice(v, SOURCE_MODELS, "source_model")
        )
        modality = _axis(data, "modality", ("eeg",), lambda v: _choice(v, MODALITIES, "modality"))
        noise = _axis(data, "noise", ("simulated",), lambda v: _choice(v, NOISE_KINDS, "noise"))
        if "empty_room" in noise and any(m not in ("mag", "grad") for m in modality):
            raise ValueError(
                "noise 'empty_room' is a recording of MEG and exists for 'mag' and 'grad' only; "
                f"the modalities are {list(modality)}."
            )
        inverse = _axis(data, "inverse", ("mne",), lambda v: _choice(v, INVERSE_METHODS, "inverse"))
        without = [e for e in estimators if e.split("/")[0] not in ANALYTIC_METHODS]
        if "analytic" in tests and without:
            raise ValueError(
                "The analytic test is the F test of Granger causality or the asymptotic test of "
                f"PDC and rPDC: every estimator of the protocol must be one of "
                f"{list(ANALYTIC_METHODS)}; {without} has no analytic test."
            )
        protocol = cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            scenarios=scenarios,
            estimators=estimators,
            n_seeds=_integer(data["n_seeds"], "n_seeds", minimum=1),
            master_seed=_integer(data.get("master_seed", 0), "master_seed", minimum=0),
            snr_db=snr,
            duration_s=duration,
            coupling=coupling,
            source_model=source_model,
            modality=modality,
            inverse=inverse,
            exclude=_exclude(data.get("exclude")),
            budget_hours=_number(
                data.get("budget_hours", DEFAULT_BUDGET_HOURS),
                "budget_hours",
                low=0.0,
                open_low=True,
            ),
            settings=_settings(data.get("settings")),
            saturation_guard=_boolean(data.get("saturation_guard", False), "saturation_guard"),
            kind=kind,
            pairs=pairs,
            tests=tests,
            noise=noise,
        )
        if any(c is not None for c in protocol.coupling):
            for name in protocol.scenarios:
                _uniform_strength(load_scenario(name))
        return protocol

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-compatible dictionary that :meth:`from_dict` restores exactly."""
        out = {
            "version": PROTOCOL_VERSION,
            "name": self.name,
            "description": self.description,
            "scenarios": list(self.scenarios),
            "estimators": list(self.estimators),
            "n_seeds": self.n_seeds,
            "master_seed": self.master_seed,
            "snr_db": list(self.snr_db),
            "duration_s": list(self.duration_s),
            "coupling": list(self.coupling),
            "source_model": list(self.source_model),
            "modality": list(self.modality),
            "inverse": list(self.inverse),
            "exclude": copy.deepcopy(list(self.exclude)),
            "budget_hours": self.budget_hours,
            "settings": dict(self.settings),
            "saturation_guard": self.saturation_guard,
            "kind": self.kind,
            "pairs": None if self.pairs is None else copy.deepcopy(self.pairs),
        }
        out["tests"] = list(self.tests)
        out["noise"] = list(self.noise)
        if self.kind == "region_pairs":
            del out["scenarios"], out["coupling"], out["tests"], out["noise"]
        elif self.kind == "significance":
            del out["pairs"], out["modality"], out["source_model"], out["inverse"], out["noise"]
        else:
            del out["pairs"], out["tests"]
        return out

    def numeric_settings(self) -> dict[str, Any]:
        """Return what decides the numbers of a task, apart from the task itself."""
        out: dict[str, Any] = {
            "master_seed": self.master_seed,
            "estimators": list(self.estimators),
            "inverse": list(self.inverse),
            "settings": dict(self.settings),
        }
        if self.kind == "significance":
            out["tests"] = list(self.tests)
        return out


def load_protocol(source: str | os.PathLike[str]) -> Protocol:
    """Load a protocol from a YAML file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        For every problem of :meth:`Protocol.from_dict`; the message starts with the file name.
    """
    file = Path(source)
    if not file.exists():
        raise FileNotFoundError(file)
    with open(file, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    try:
        if not isinstance(data, Mapping):
            raise ValueError("it must contain a YAML mapping at the top level.")
        return Protocol.from_dict(data)
    except ValueError as error:
        raise ValueError(f"{file.name}: {error}") from None


@dataclass(frozen=True)
class Task:
    """One simulation of a protocol.

    ``snr_db`` and ``duration_s`` are the effective values (the scenario's default when the axis
    leaves them open); ``coupling`` is ``None`` when the scenario keeps its own strengths.
    """

    scenario: str
    snr_db: float
    duration_s: float
    coupling: float | None
    source_model: str
    modality: str
    seed_index: int
    spec: ScenarioSpec = field(compare=False, repr=False)
    scenario_hash: str = ""
    noise: str = "simulated"

    @property
    def zero_network(self) -> bool:
        """True when the coupling is 0: no edges, nothing to find, only false positives to count."""
        return self.coupling == 0.0

    @property
    def n_nodes(self) -> int:
        """Number of nodes of the scenario."""
        return self.spec.network.n_nodes

    def to_dict(self) -> dict[str, Any]:
        """Return the readable key of the task (without the scenario hash)."""
        return {
            "scenario": self.scenario,
            "snr_db": self.snr_db,
            "duration_s": self.duration_s,
            "coupling": self.coupling,
            "source_model": self.source_model,
            "modality": self.modality,
            "seed_index": self.seed_index,
            **({} if self.noise == "simulated" else {"noise": self.noise}),
        }

    @property
    def task_id(self) -> str:
        """Identifier of the task *and* of the scenario it was built from."""
        return _digest({**self.to_dict(), "scenario_hash": self.scenario_hash})[:16]

    def scenario_spec(self) -> ScenarioSpec:
        """Return the scenario with this task's coupling, SNR and data length applied."""
        spec = self.spec if self.coupling is None else apply_coupling(self.spec, self.coupling)
        return dataclasses.replace(spec, snr_db=self.snr_db, duration_s=self.duration_s)


def apply_coupling(spec: ScenarioSpec, value: float) -> ScenarioSpec:
    """Return the scenario with every edge set to the strength ``value`` (sign kept).

    ``0`` removes all edges. Raises ``ValueError`` when the edges of the scenario do not share one
    strength, naming the scenario and two edges that differ.
    """
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"coupling must be finite and non-negative, got {value!r}.")
    _uniform_strength(spec)
    edges = (
        ()
        if value == 0.0
        else tuple(
            dataclasses.replace(e, strength=math.copysign(value, e.strength))
            for e in spec.network.edges
        )
    )
    network = Network(nodes=spec.network.nodes, edges=edges, name=spec.network.name)
    return dataclasses.replace(spec, network=network)


def select_pairs(regions: Sequence[str], n: int, *, seed: int) -> list[tuple[str, str]]:
    """Return ``n`` distinct unordered pairs of ``regions``, in a random order fixed by ``seed``.

    Every pair is ``(x, y)`` with ``x`` before ``y`` in ``regions``. The pairs are the first ``n``
    of one seeded permutation of all pairs, so asking for more keeps the pairs already chosen.
    """
    names = list(regions)
    total = len(names) * (len(names) - 1) // 2
    if n > total:
        raise ValueError(f"n ({n}) is more than the {total} pairs of {len(names)} regions.")
    everything = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    order = np.random.default_rng(seed).permutation(total)
    return [everything[k] for k in order[:n]]


def region_pair_scenario(x: str, y: str) -> ScenarioSpec:
    """Return the scenario of a region pair: two oscillating nodes (10 and 20 Hz), no edge.

    Both nodes oscillate so that each has an autocorrelation: leakage of one into the series of the
    other is only visible to a lagged method when the leaking activity has a past.
    """
    name = f"pair:{x}:{y}"
    network = Network(nodes=(dk_node(x), dk_node(y)), edges=(), name=name)
    dynamics = {
        node: NodeDynamics(kind="oscillator", freq_hz=freq, bandwidth_hz=4.0)
        for node, freq in zip((x, y), PAIR_FREQS_HZ, strict=True)
    }
    description = (
        f"The regions {x} and {y}, no edge between them: whatever an estimator finds between "
        "their series is leakage (or noise)."
    )
    return ScenarioSpec(name=name, description=description, network=network, dynamics=dynamics)


def _pair_scenarios(protocol: Protocol) -> dict[str, ScenarioSpec]:
    assert protocol.pairs is not None
    pairs = select_pairs(
        protocol.pairs["regions"], protocol.pairs["n"], seed=protocol.pairs["seed"]
    )
    return {spec.name: spec for spec in (region_pair_scenario(x, y) for x, y in pairs)}


def expand(protocol: Protocol, scenarios: Mapping[str, ScenarioSpec] | None = None) -> list[Task]:
    """Return the tasks of a protocol in a fixed order, with the exclusions applied.

    Parameters
    ----------
    protocol : Protocol
    scenarios : mapping of str to ScenarioSpec, optional
        Replaces the shipped file of a scenario (for tests and for trying an edited scenario).
    """
    if protocol.kind == "region_pairs":
        specs = _pair_scenarios(protocol)
        couplings: tuple[float | None, ...] = (0.0,)  # a zero network, whatever the file says
    else:
        specs = {name: _spec(name, scenarios) for name in protocol.scenarios}
        couplings = protocol.coupling
    noises = protocol.noise if protocol.kind == "scenarios" else ("simulated",)
    source_models = ("none",) if protocol.kind == "significance" else protocol.source_model
    modalities = ("roi",) if protocol.kind == "significance" else protocol.modality
    tasks: list[Task] = []
    for name in specs:
        base = specs[name]
        for snr in protocol.snr_db:
            for duration in protocol.duration_s:
                for coupling in couplings:
                    for source_model in source_models:
                        for modality in modalities:
                            for noise in noises:
                                for seed_index in range(protocol.n_seeds):
                                    task = Task(
                                        scenario=name,
                                        snr_db=base.snr_db if snr is None else snr,
                                        duration_s=base.duration_s
                                        if duration is None
                                        else duration,
                                        coupling=coupling,
                                        source_model=source_model,
                                        modality=modality,
                                        seed_index=seed_index,
                                        spec=base,
                                        noise=noise,
                                    )
                                    if _excluded(task, protocol.exclude):
                                        continue
                                    digest = _digest(task.scenario_spec().to_dict())
                                    tasks.append(
                                        dataclasses.replace(task, scenario_hash=digest[:16])
                                    )
    return tasks


def task_seed(master_seed: int, task: Task) -> int:
    """Return the integer seed of a task: a function of the master seed and the task's content."""
    # without the noise: the arms of one seed share their sources and differ in the noise only
    key = {k: v for k, v in task.to_dict().items() if k != "noise"}
    index = int(_digest(key)[:15], 16)  # 60 bits
    rng = RNGController(master_seed).child(index)
    return int(rng.integers(0, 2**31 - 1))


@dataclass(frozen=True)
class RunSummary:
    """What one call of :func:`run_protocol` did."""

    n_tasks: int
    n_done: int
    n_skipped: int
    n_failed: int
    seconds: float


def run_protocol(
    protocol: Protocol,
    runner: Callable[[Task], dict[str, Any]],
    path: str | os.PathLike[str],
    *,
    workers: int = 1,
    new_file: bool = False,
    environment: Mapping[str, str] | None = None,
    gate: Mapping[str, Any] | None = None,
    header_extra: Mapping[str, Any] | None = None,
    scenarios: Mapping[str, ScenarioSpec] | None = None,
    progress: Callable[[str], None] | None = None,
) -> RunSummary:
    """Run every task that has no result yet and append the results to ``path``.

    A task that raises is recorded as failed and the run goes on; the next call retries it.

    Parameters
    ----------
    runner : callable
        ``runner(task) -> dict`` returning something JSON-serialisable (``NaN`` allowed). It must
        be picklable when ``workers`` is above 1.
    path : path
        The results file (JSON Lines). If it exists the run resumes, after checking that the
        settings that decide the numbers and the package versions are those of the file.
    workers : int
        Number of processes; above 1 needs ``joblib``.
    new_file : bool
        Start over: an existing file is replaced.
    environment : mapping, optional
        Versions to record and to compare; default :func:`environment`.
    gate : mapping, optional
        The budget-gate measurement, recorded in the header.
    header_extra : mapping, optional
        More entries for the header (for example the crosstalk matrices of a region-pair study);
        a resume without it keeps the entries the file already has.
    scenarios : mapping, optional
        As in :func:`expand`.
    progress : callable, optional
        Called with a short message after every task.

    Raises
    ------
    ValueError
        For a file that is not a results file, or whose settings or versions differ (the message
        names the differences and mentions ``new_file``).
    ImportError
        For ``workers`` above 1 without joblib.
    """
    workers = _integer(workers, "workers", minimum=1)
    if workers > 1:
        try:
            import joblib  # noqa: F401
        except ImportError:
            raise ImportError(
                "workers > 1 needs joblib; install it with: pip install 'dirconn[parallel]'."
            ) from None
    env = dict(environment) if environment is not None else environment_versions()
    path = Path(path)
    tasks = expand(protocol, scenarios)
    header = {
        "type": "header",
        "kind": RESULTS_KIND,
        "version": PROTOCOL_VERSION,
        "protocol": protocol.name,
        "settings": protocol.numeric_settings(),
        "settings_hash": _digest(protocol.numeric_settings())[:16],
        "environment": env,
        "code": code_version(),
        "gate": None if gate is None else dict(gate),
        "saturation_guard": protocol.saturation_guard,
        "protocol_kind": protocol.kind,
        "pairs": None if protocol.pairs is None else copy.deepcopy(protocol.pairs),
        "started": datetime.now(timezone.utc).isoformat(),
    }
    if header_extra:
        header.update(copy.deepcopy(dict(header_extra)))
    kept: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0 and not new_file:
        old_header, old_records, _ = read_results(path)
        _check_compatible(old_header, header)
        header["started"] = old_header.get("started", header["started"])
        header["gate"] = header["gate"] if gate is not None else old_header.get("gate")
        for key, value in old_header.items():
            header.setdefault(key, value)  # entries only the file knows, such as the crosstalk
        current = {_digest(t.to_dict()): t.task_id for t in tasks}
        for record in old_records:
            if record.get("status") != "ok":
                continue
            superseded = current.get(_digest(record["task"]), record["id"]) != record["id"]
            if not superseded:
                kept.append(record)
    _write_lines(path, [header, *kept])
    done = {r["id"] for r in kept}
    todo = [t for t in tasks if t.task_id not in done]
    started = time.perf_counter()
    n_done = n_failed = 0

    def finish(task: Task, outcome: dict[str, Any]) -> None:
        nonlocal n_done, n_failed
        record = {
            "type": "task",
            "id": task.task_id,
            "task": task.to_dict(),
            "seed": task_seed(protocol.master_seed, task),
            "scenario_hash": task.scenario_hash,
            **outcome,
        }
        _append_line(path, record)
        if outcome["status"] == "ok":
            n_done += 1
        else:
            n_failed += 1
        if progress is not None:
            progress(f"{task.scenario} seed {task.seed_index}: {outcome['status']}")

    if workers == 1:
        for task in todo:
            finish(task, _execute(runner, task))
    elif todo:
        from joblib import Parallel, delayed

        pairs = Parallel(n_jobs=workers, return_as="generator_unordered")(
            delayed(_execute_indexed)(runner, i, task) for i, task in enumerate(todo)
        )
        for i, outcome in pairs:
            finish(todo[i], outcome)

    seconds = time.perf_counter() - started
    summary = RunSummary(
        n_tasks=len(tasks),
        n_done=n_done,
        n_skipped=len(tasks) - len(todo),
        n_failed=n_failed,
        seconds=seconds,
    )
    _append_line(
        path,
        {"type": "summary", **dataclasses.asdict(summary)},
    )
    return summary


def read_results(
    path: str | os.PathLike[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    """Read a results file; returns ``(header, task records, summary or None)``.

    A last line that is cut off (a run that was killed while writing) is ignored.

    Raises
    ------
    ValueError
        If the file is not a results file or a line other than the last one is not JSON.
    """
    lines = [x for x in Path(path).read_text(encoding="utf-8").split("\n") if x.strip()]
    parsed: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1 and i > 0:
                break
            raise ValueError(f"{Path(path).name}: line {i + 1} is not valid JSON.") from None
    if not parsed or parsed[0].get("kind") != RESULTS_KIND:
        raise ValueError(f"{Path(path).name} is not a dirconn benchmark results file.")
    records = [p for p in parsed[1:] if p.get("type") == "task"]
    final = next((p for p in reversed(parsed[1:]) if p.get("type") == "summary"), None)
    return parsed[0], records, final


def code_version(directory: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
    """Return the git commit of the code and whether the tree had uncommitted changes.

    ``{"git_commit": <40 hex digits>, "git_dirty": <bool>}``, or ``None`` when ``directory``
    (default: where this package is) is not in a git repository or git is not available.
    Untracked files and the results themselves do not make a tree dirty; changed tracked files do.
    An installed copy outside a repository, as in the Docker image, falls back to the commit it
    was built from, which the image records in ``DIRCONN_GIT_COMMIT`` and ``DIRCONN_GIT_DIRTY``.
    """
    where = str(Path(__file__).resolve().parent if directory is None else directory)

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=where, capture_output=True, text=True, timeout=30, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None or len(commit) != 40:
        return _built_from() if directory is None else None
    changed = git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": commit, "git_dirty": bool(changed)}


def _built_from() -> dict[str, Any] | None:
    """Return the commit an installed copy was built from (Docker image), or ``None``."""
    commit = os.environ.get("DIRCONN_GIT_COMMIT", "")
    if len(commit) != 40 or not set(commit) <= set("0123456789abcdef"):
        return None
    return {"git_commit": commit, "git_dirty": os.environ.get("DIRCONN_GIT_DIRTY") == "1"}


def environment_versions() -> dict[str, str]:
    """Return the Python, platform and package versions a set of numbers is tied to."""
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "scipy", "mne", "statsmodels", "joblib", "dirconn"):
        try:
            info[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            info[package] = "not installed"
    return info


def estimate_hours(
    tasks: Iterable[Task],
    seconds_by_nodes: Mapping[int, float],
    *,
    reference_duration_s: float,
    workers: int,
    efficiency: float = DEFAULT_EFFICIENCY,
    startup_seconds: float = 0.0,
) -> float:
    """Estimate the wall-clock hours of ``tasks`` from timings of one task per network size.

    ``seconds_by_nodes[n]`` is the measured time of a task with ``n`` nodes and a data length of
    ``reference_duration_s``; the time of other lengths is taken as proportional. With several
    workers the total is divided by ``workers * efficiency``; ``startup_seconds`` (loading the
    head models in every process) is added once.
    """
    total = 0.0
    for task in tasks:
        if task.n_nodes not in seconds_by_nodes:
            raise ValueError(
                f"No timing for {task.n_nodes} nodes (scenario {task.scenario!r}); "
                f"measured: {sorted(seconds_by_nodes)}."
            )
        total += seconds_by_nodes[task.n_nodes] * task.duration_s / reference_duration_s
    divisor = 1.0 if workers == 1 else workers * efficiency
    return total / 3600.0 / divisor + startup_seconds / 3600.0


def check_budget(estimate: float, budget_hours: float, *, force: bool = False) -> None:
    """Raise ``ValueError`` when the estimated hours exceed the budget and ``force`` is not set."""
    if not math.isfinite(estimate):
        raise ValueError(f"The run-time estimate must be a finite number, got {estimate!r}.")
    if estimate > budget_hours and not force:
        raise ValueError(
            f"The estimated run time of {estimate:g} h exceeds the budget of {budget_hours:g} h; "
            "shrink the grid or pass force to go ahead."
        )


SATURATION_THRESHOLD = 0.99
"""A mean AUROC at or above this is called perfect; below it, imperfect."""
GUARD_STAGES: tuple[str, ...] = ("sensor", "source")
"""The stages the guard judges; the region level is the idealised observation and may stay easy."""
_GRID_KEYS = ("snr_db", "duration_s", "coupling", "source_model", "modality")


@dataclass(frozen=True)
class StageSaturation:
    """Where the best estimator is worst and best on the reference scenario, at one stage."""

    hardest_auroc: float
    hardest_grid: dict[str, Any]
    easiest_auroc: float
    easiest_grid: dict[str, Any]


@dataclass(frozen=True)
class SaturationReport:
    """Outcome of :func:`saturation_report`.

    Attributes
    ----------
    scenario : str
        The reference scenario.
    ok : bool
        True when there is no problem.
    problems : list of str
        What is wrong, one sentence each; empty when ``ok``.
    stages : dict of str to StageSaturation
        For ``sensor`` and ``source`` the hardest and the easiest grid point (as the best
        estimator's mean AUROC over seeds); empty when there are no results of the scenario.
    """

    scenario: str
    ok: bool
    problems: list[str]
    stages: dict[str, StageSaturation]


def saturation_report(
    records: Sequence[Mapping[str, Any]], *, scenario: str = "chain"
) -> SaturationReport:
    """Check that the grid reaches from the easy to the hard on a reference scenario.

    Findings A29: in easy settings all three stages score perfectly and nothing can be said about
    the methods. For the sensor and the source stage separately, the grid must contain (a) a point
    where even the **best** estimator, averaged over seeds, is below :data:`SATURATION_THRESHOLD`
    (a hard point) and (b) a point where it is at or above it (an anchor, so the grid is not all
    ruin). The best estimator decides because some methods are never perfect on a chain for
    structural reasons (DTF reports the indirect path). Failed tasks, other scenarios and ``NaN``
    scores are left out; grid points are the combinations of SNR, data length, coupling, source
    model and modality.
    """
    cells: dict[str, dict[tuple[Any, ...], dict[str, list[float]]]] = {s: {} for s in GUARD_STAGES}
    for record in records:
        task = record.get("task", {})
        if record.get("status") != "ok" or task.get("scenario") != scenario:
            continue
        grid = tuple(task.get(k) for k in _GRID_KEYS)
        for label, stages in record.get("result", {}).items():
            for stage in GUARD_STAGES:
                card = stages.get(stage)
                value = card.get("auroc") if isinstance(card, Mapping) else None
                if value is None or math.isnan(value):
                    continue
                cells[stage].setdefault(grid, {}).setdefault(label, []).append(float(value))
    if not any(cells[s] for s in GUARD_STAGES):
        problem = f"There are no results of the reference scenario {scenario!r}."
        return SaturationReport(scenario, False, [problem], {})
    problems: list[str] = []
    stages_out: dict[str, StageSaturation] = {}
    for stage in GUARD_STAGES:
        if not cells[stage]:
            problems.append(f"{stage}: there are no results of {scenario!r} at this stage.")
            continue
        best = {
            grid: max(float(np.mean(v)) for v in by_label.values())
            for grid, by_label in cells[stage].items()
        }
        hardest = min(best, key=lambda g: best[g])
        easiest = max(best, key=lambda g: best[g])
        stages_out[stage] = StageSaturation(
            hardest_auroc=best[hardest],
            hardest_grid=dict(zip(_GRID_KEYS, hardest, strict=True)),
            easiest_auroc=best[easiest],
            easiest_grid=dict(zip(_GRID_KEYS, easiest, strict=True)),
        )
        if best[hardest] >= SATURATION_THRESHOLD:
            problems.append(
                f"{stage}: the best estimator is perfect at every grid point of {scenario!r} "
                f"(lowest mean AUROC {best[hardest]:.3f}); the grid is too easy and must be "
                "hardened."
            )
        if best[easiest] < SATURATION_THRESHOLD:
            problems.append(
                f"{stage}: there is no anchor, no grid point of {scenario!r} where the best "
                f"estimator reaches {SATURATION_THRESHOLD:g} (highest {best[easiest]:.3f})."
            )
    return SaturationReport(scenario, not problems, problems, stages_out)


def _describe_grid(grid: Mapping[str, Any]) -> str:
    def number(value: Any) -> str:
        return "?" if value is None else f"{value:g}"

    coupling = (
        "file strengths" if grid["coupling"] is None else f"coupling {number(grid['coupling'])}"
    )
    return (
        f"SNR {number(grid['snr_db'])} dB, {number(grid['duration_s'])} s, {coupling}, "
        f"{grid['source_model']}, {grid['modality']}"
    )


def render_summary(
    header: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    final: Mapping[str, Any] | None,
) -> str:
    """Return a Markdown summary of a results file.

    It has the environment, the gate, the counts, a table per estimator and stage (mean AUROC, mean
    direction accuracy, median separation, over the tasks with a defined AUROC; ``NaN`` and an
    infinite separation left out), the agreement between the stages, and, for a protocol with a
    ``saturation_guard``, the verdict of :func:`saturation_report`. This is a check that the
    results can be read back, not the report of the benchmark.
    """
    ok = [r for r in records if r.get("status") == "ok"]
    failed = [r for r in records if r.get("status") == "failed"]
    env = header.get("environment", {})
    lines = [f"# Benchmark results: {header.get('protocol', '?')}", ""]
    packages = ", ".join(f"{k} {v}" for k, v in env.items() if k not in ("platform", "python"))
    lines.append(f"Environment: Python {env.get('python', '?')}; {packages}.")
    gate = header.get("gate")
    estimate = None if not gate else gate.get("estimate_hours")
    if estimate is not None:
        lines.append(f"Budget gate: estimated {float(estimate):.2f} h before the run.")
    status = f"{len(ok)} tasks done, {len(failed)} failed"
    if final is None:
        status += "; the run is unfinished (no summary line)"
    else:
        status += f"; last session {float(final.get('seconds', 0.0)) / 3600.0:.2f} h"
    lines += [status + ".", ""]

    names = list(header.get("settings", {}).get("estimators", []))
    cards: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in ok:
        for estimator, stages in record.get("result", {}).items():
            if estimator not in names:
                names.append(estimator)
            for stage, card in stages.items():
                if not isinstance(card, Mapping) or "auroc" not in card:
                    continue
                cards.setdefault((estimator, stage), [])
                value = card["auroc"]
                if value is not None and not math.isnan(value):
                    cards[(estimator, stage)].append(card)

    def mean(values: Sequence[Any]) -> str:
        finite = [v for v in values if v is not None and math.isfinite(v)]
        return f"{np.mean(finite):.3f}" if finite else "nan"

    def median(values: Sequence[Any]) -> str:
        finite = [v for v in values if v is not None and math.isfinite(v)]
        return f"{np.median(finite):.2f}" if finite else "nan"

    if header.get("protocol_kind") in ("region_pairs", "significance"):
        lines += [
            "These networks have no true edge, so AUROC, direction accuracy and separation are "
            "undefined here; see the leakage section below."
            if header["protocol_kind"] == "region_pairs"
            else "The edge metrics of the benchmark do not apply to a study of tests; the p-values "
            "are summarised in the significance section below.",
        ]
    else:
        lines += [
            "| estimator | stage | mean AUROC | mean direction | median separation | n |",
            "|---|---|---|---|---|---|",
        ]
        stage_order = {"roi": 0, "sensor": 1, "source": 2}
        for name in names:
            for stage in sorted(
                (s for e, s in cards if e == name), key=lambda s: stage_order.get(s, 9)
            ):
                used = cards[(name, stage)]
                lines.append(
                    f"| {name} | {stage} | {mean([c['auroc'] for c in used])} "
                    f"| {mean([c.get('direction_accuracy') for c in used])} "
                    f"| {median([c.get('separation') for c in used])} | {len(used)} |"
                )

        lines += ["", "| estimator | ROI vs sensor | ROI vs source |", "|---|---|---|"]
        for name in names:
            agreements = [
                r["result"][name].get("agreement", {}) for r in ok if name in r.get("result", {})
            ]
            sensor = mean([a.get("spearman") for a in agreements])
            source = mean([a.get("spearman_roi_source") for a in agreements])
            lines.append(f"| {name} | {sensor} | {source} |")

    if header.get("saturation_guard"):
        report = saturation_report(records)
        lines += ["", f"## Saturation guard (scenario {report.scenario!r})", ""]
        lines.append("Verdict: ok." if report.ok else "Verdict: **VIOLATED**.")
        if report.stages:
            lines += [
                "",
                "| stage | hardest point | best AUROC there | easiest point | best AUROC there |",
                "|---|---|---|---|---|",
            ]
            for stage, s in report.stages.items():
                lines.append(
                    f"| {stage} | {_describe_grid(s.hardest_grid)} | {s.hardest_auroc:.3f} "
                    f"| {_describe_grid(s.easiest_grid)} | {s.easiest_auroc:.3f} |"
                )
        lines += ["", *[f"- {p}" for p in report.problems]]
    lines += _leakage_sections(header, records)
    lines += _significance_section(header, records)
    lines += _noise_section(records)
    return "\n".join(lines) + "\n"


def _fmt(value: float, digits: int = 3) -> str:
    return "nan" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def _leakage_sections(header: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return the Markdown of the two leakage studies, if the results hold what they need."""
    lines: list[str] = []
    if header.get("protocol_kind") == "region_pairs":
        rows = leakage_correlation(header, records) if header.get("crosstalk") else []
        lines += [
            "",
            "## Leakage and crosstalk",
            "",
            "Region pairs without any edge. Spearman correlation between the score of a false edge "
            "x -> y and the leak of x into the series of y (the crosstalk matrix of the header), "
            "over the cells of all pairs; 95% percentile bootstrap over pairs. The region-level "
            "stage has no leakage and is the control.",
            "",
        ]
        if not rows:
            lines.append("No score matrices to correlate (the protocol needs `store_scores`).")
        else:
            lines += [
                "| estimator | modality | stage | pairs | cells | Spearman | 95% interval |",
                "|---|---|---|---|---|---|---|",
            ]
            for r in rows:
                interval = f"[{_fmt(r['ci_low'], 2)}, {_fmt(r['ci_high'], 2)}]"
                lines.append(
                    f"| {r['estimator']} | {r['modality']} | {r['stage']} | {r['n_pairs']} "
                    f"| {r['n_cells']} | {_fmt(r['spearman'], 2)} | {interval} |"
                )
    if any(r.get("task", {}).get("scenario") == "leakage_adjacent" for r in records):
        rows = adjacent_report(records)
        if rows:
            lines += [
                "",
                "## Neighbour leakage (leakage_adjacent)",
                "",
                "Z -> X is the only edge; Y is X's neighbour. Relative scores are the score of an "
                "edge into Y over the score of Z -> X (mean over seeds, [min, max]).",
                "",
                "| estimator | modality | SNR | stage | n | AUROC | direction | Z -> Y | X -> Y |",
                "|---|---|---|---|---|---|---|---|---|",
            ]

            def span(s: Mapping[str, float]) -> str:
                return f"{_fmt(s['mean'], 2)} [{_fmt(s['min'], 2)}, {_fmt(s['max'], 2)}]"

            for r in rows:
                lines.append(
                    f"| {r['estimator']} | {r['modality']} | {r['snr_db']:g} | {r['stage']} "
                    f"| {r['n']} | {_fmt(r['auroc'])} | {_fmt(r['direction_accuracy'])} "
                    f"| {span(r['z_to_y'])} | {span(r['x_to_y'])} |"
                )
            lines += [
                "",
                "Source minus region level, paired by seed (mean, sd over seeds):",
                "",
                "| estimator | modality | SNR | n | AUROC | direction | Z -> Y | X -> Y |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for r in adjacent_report(records, paired=True):

                def diff(m: Mapping[str, float]) -> str:
                    return f"{_fmt(m['mean'], 2)} ({_fmt(m['sd'], 2)})"

                lines.append(
                    f"| {r['estimator']} | {r['modality']} | {r['snr_db']:g} | {r['n']} "
                    f"| {diff(r['auroc'])} | {diff(r['direction_accuracy'])} "
                    f"| {diff(r['z_to_y'])} | {diff(r['x_to_y'])} |"
                )
    return lines


def _noise_section(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return the Markdown of the noise study, if any task used another noise than the default."""
    if not any(
        r.get("task", {}).get("noise", "simulated") != "simulated"
        for r in records
        if r.get("status") == "ok"
    ):
        return []
    rows = noise_report(records)

    def arm(row: Mapping[str, Any], name: str) -> str:
        entry = row["arms"].get(name)
        return "-" if entry is None else f"{_fmt(entry['auroc'])} (n={entry['n']})"

    def difference(row: Mapping[str, Any], name: str) -> str:
        entry = row["paired"].get(name)
        if entry is None:
            return "-"
        mean, sd = entry["auroc"]["mean"], entry["auroc"]["sd"]
        sign = "" if math.isnan(mean) or mean < 0 else "+"
        return f"{sign}{_fmt(mean)} (sd {_fmt(sd)}, n={entry['n']})"

    lines = [
        "",
        "## Noise study",
        "",
        "Every seed has up to three arms with the same sources: the simulated noise (1/f "
        "background sources plus white noise), white noise only (no background) and the real "
        "empty-room recording of the sample dataset (no background), all scaled to the same "
        "sensor SNR. Mean AUROC over seeds, and the difference between arms taken seed by seed "
        "(mean, sd, number of seeds). The empty-room windows are blocks of one 110 s recording "
        "and are reused by several seeds, so the noise samples are not independent even where "
        "the seeds are. The "
        "region-level stage does not see the sensor noise and is the control.",
        "",
        "| modality | SNR | estimator | stage | simulated | white | empty room "
        "| white - simulated | empty room - white | empty room - simulated |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['modality']} | {r['snr_db']:g} | {r['estimator']} | {r['stage']} "
            f"| {arm(r, 'simulated')} | {arm(r, 'white')} | {arm(r, 'empty_room')} "
            f"| {difference(r, 'white_minus_simulated')} "
            f"| {difference(r, 'empty_room_minus_white')} "
            f"| {difference(r, 'empty_room_minus_simulated')} |"
        )
    return lines


def _interval(value: Mapping[str, float] | None) -> str:
    if value is None:
        return "n/a"
    return f"{_fmt(value['estimate'])} [{_fmt(value['low'])}, {_fmt(value['high'])}]"


def _significance_section(
    header: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Return the Markdown of a significance study, if the header says it is one."""
    if header.get("protocol_kind") != "significance":
        return []
    n_surrogates = int(header.get("settings", {}).get("settings", {}).get("n_surrogates", 39))
    rows = significance_report(records, n_surrogates=n_surrogates)
    lines = [
        "",
        "## Significance study",
        "",
        "Datasets are simulated at region level (node activity plus white observation noise). A "
        "cell is *null* where the scenario has no edge. Rates are over cells; the intervals are "
        "bootstrapped over datasets (the cells of a dataset share its data), a share of datasets "
        f"has a Wilson interval. A surrogate test with {n_surrogates} surrogates has p-values in "
        f"steps of 1/{n_surrogates + 1}, so a level below {1 / (n_surrogates + 1):.3f} cannot be "
        "reached (n/a).",
        "",
    ]
    frequency_rows = frequency_report(records, n_surrogates=n_surrogates)
    if not rows and not frequency_rows:
        return [*lines, "No finished tasks with p-values."]
    if rows:
        lines += _cell_tables(rows)
    if frequency_rows:
        lines += _frequency_tables(frequency_rows)
    return lines


def _significance_head(row: Mapping[str, Any]) -> str:
    coupling = "file" if row["coupling"] is None else f"{row['coupling']:g}"
    return (
        f"| {row['scenario']} | {coupling} | {row['snr_db']:g} | {row['duration_s']:g} "
        f"| {row['estimator']} | {row['test']} | {row['n_datasets']} "
    )


def _frequency_tables(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Tables of the per-frequency tests (PDC, rPDC): pointwise and per pair after BH."""
    head = _significance_head
    alphas = list(rows[0]["null_rate"])
    lines = [
        "",
        "### Frequency-resolved tests",
        "",
        "One p-value per pair and frequency. *Pointwise* rates are over pairs and the frequencies "
        "of the band (`all`: the whole grid); *after BH* a pair counts once if any frequency of "
        "the band survives a BH correction over those frequencies, q = 0.05. *Liberal*: the lower "
        "end of the interval of the pointwise rate at 0.05 is above 0.05.",
        "",
        "False positives, pairs without an edge:",
        "",
        "| scenario | coupling | SNR | s | estimator | test | datasets | band | freqs "
        "| null pairs | "
        + " | ".join(f"rate at {a:g}" for a in alphas)
        + " | liberal at 0.05 | pairs after BH | freqs needed for BH |",
        "|" + "---|" * (10 + len(alphas) + 3),
    ]
    for r in rows:
        if not r["n_null_pairs"]:
            continue
        liberal = r["liberal"].get(0.05)
        lines.append(
            head(r)
            + f"| {r['band']} | {r['n_freqs']} | {r['n_null_pairs']} | "
            + " | ".join(_interval(r["null_rate"][a]) for a in alphas)
            + f" | {'n/a' if liberal is None else ('yes' if liberal else 'no')} "
            + f"| {_interval(r['pair_bh']['null_rate'])} | {r['bh_freqs_needed']} |"
        )
    edge_rows = [r for r in rows if r["n_edge_pairs"]]
    if edge_rows:
        lines += [
            "",
            "Power, pairs with a true edge:",
            "",
            "| scenario | coupling | SNR | s | estimator | test | datasets | band | freqs "
            "| edge pairs | "
            + " | ".join(f"power at {a:g}" for a in alphas)
            + " | pairs after BH |",
            "|" + "---|" * (10 + len(alphas) + 1),
        ]
        for r in edge_rows:
            lines.append(
                head(r)
                + f"| {r['band']} | {r['n_freqs']} | {r['n_edge_pairs']} | "
                + " | ".join(_interval(r["power"][a]) for a in alphas)
                + f" | {_interval(r['pair_bh']['power'])} |"
            )
    return lines


def _cell_tables(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Tables of the tests with one p-value per cell (Granger causality)."""
    head = _significance_head
    lines: list[str] = ["BH is over the six cells of a dataset, q = 0.05.", ""]
    alphas = list(rows[0]["null_rate"])
    null_rows = [r for r in rows if r["n_null_cells"]]
    lines += [
        "False positives, cells without an edge:",
        "",
        "| scenario | coupling | SNR | s | estimator | test | datasets | null cells | "
        + " | ".join(f"rate at {a:g}" for a in alphas)
        + " | mean p (expected) | datasets with a BH false rejection |",
        "|" + "---|" * (8 + len(alphas) + 2),
    ]
    for r in null_rows:
        lines.append(
            head(r)
            + f"| {r['n_null_cells']} | "
            + " | ".join(_interval(r["null_rate"][a]) for a in alphas)
            + f" | {_fmt(r['mean_p_null'])} ({_fmt(r['expected_mean_p'])}) "
            + f"| {_interval(r['bh']['any_false_rejection'])} |"
        )
    edge_rows = [r for r in rows if r["n_edge_cells"]]
    if edge_rows:
        lines += [
            "",
            "Power, cells with a true edge:",
            "",
            "| scenario | coupling | SNR | s | estimator | test | datasets | edge cells | "
            + " | ".join(f"power at {a:g}" for a in alphas)
            + " | BH power | BH false-discovery proportion | cells needed for BH |",
            "|" + "---|" * (8 + len(alphas) + 3),
        ]
        for r in edge_rows:
            lines.append(
                head(r)
                + f"| {r['n_edge_cells']} | "
                + " | ".join(_interval(r["power"][a]) for a in alphas)
                + f" | {_interval(r['bh']['power'])} | {_fmt(r['bh']['fdp'])} "
                + f"| {r['bh_cells_needed']} |"
            )
    return lines


def _spec(name: str, scenarios: Mapping[str, ScenarioSpec] | None) -> ScenarioSpec:
    if scenarios is not None and name in scenarios:
        return scenarios[name]
    return load_scenario(name)


def _uniform_strength(spec: ScenarioSpec) -> None:
    edges = spec.network.edges
    if not edges:
        return
    first = edges[0]
    for edge in edges[1:]:
        if not math.isclose(abs(edge.strength), abs(first.strength)):
            raise ValueError(
                f"Scenario {spec.name!r}: the strength of {first.source!r} -> {first.target!r} "
                f"({first.strength:g}) and of {edge.source!r} -> {edge.target!r} "
                f"({edge.strength:g}) differ; a coupling axis needs one strength for all edges."
            )


def _excluded(task: Task, exclude: Sequence[Mapping[str, Any]]) -> bool:
    values = {"noise": "simulated", **task.to_dict()}
    for entry in exclude:
        if all(values[axis] in options for axis, options in entry.items()):
            return True
    return False


def _execute(runner: Callable[[Task], dict[str, Any]], task: Task) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = runner(task)
    except Exception as error:  # a failed task is data, not a reason to stop the run
        return {
            "status": "failed",
            "seconds": time.perf_counter() - started,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
    return {"status": "ok", "seconds": time.perf_counter() - started, "result": result}


def _execute_indexed(
    runner: Callable[[Task], dict[str, Any]], index: int, task: Task
) -> tuple[int, dict[str, Any]]:
    return index, _execute(runner, task)


def _check_compatible(old: Mapping[str, Any], new: Mapping[str, Any]) -> None:
    differences = []
    for key, value in new["settings"].items():
        if old.get("settings", {}).get(key) != value:
            differences.append(
                f"{key}: file has {old.get('settings', {}).get(key)!r}, protocol has {value!r}"
            )
    old_env, new_env = old.get("environment", {}), new["environment"]
    for key in sorted((set(old_env) | set(new_env)) - _UNVERSIONED):
        if old_env.get(key) != new_env.get(key):
            differences.append(f"{key}: file has {old_env.get(key)!r}, now {new_env.get(key)!r}")
    if differences:
        raise ValueError(
            "The results file cannot be resumed: "
            + "; ".join(differences)
            + "; pass new_file=True to start over."
        )


def _write_lines(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    os.replace(tmp, path)


def _append_line(path: Path, record: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        with contextlib.suppress(OSError):
            os.fsync(handle.fileno())


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_keys(mapping: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown key(s) {unknown} in {where}; allowed keys: {sorted(allowed)}.")


def _integer(value: Any, key: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer of at least {minimum}, got {value!r}.")
    return value


def _number(value: Any, key: str, *, low: float | None = None, open_low: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number, got {value!r}.")
    value = float(value)
    if low is not None and (value < low or (open_low and value == low)):
        relation = "greater than" if open_low else "at least"
        raise ValueError(f"{key} must be {relation} {low:g}, got {value:g}.")
    return value


def _choice(value: Any, options: Sequence[str], key: str) -> str:
    if value not in options:
        raise ValueError(f"Unknown {key} value {value!r}; known: {list(options)}.")
    return str(value)


def _axis(
    data: Mapping[str, Any],
    key: str,
    default: tuple[Any, ...] | None,
    convert: Callable[[Any], Any],
    *,
    allow_none: bool = False,
) -> tuple[Any, ...]:
    if key not in data:
        if default is None:
            raise ValueError(f"The protocol is missing the required key {key!r}.")
        return default
    values = data[key]
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{key} must be a non-empty list, got {values!r}.")
    out = tuple(None if (v is None and allow_none) else convert(v) for v in values)
    if len(set(out)) != len(out):
        raise ValueError(f"{key} lists a value twice: {list(values)}.")
    return out


def _exclude(value: Any) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"exclude must be a list of mappings, got {value!r}.")
    entries = []
    for entry in value:
        if not isinstance(entry, Mapping) or not entry:
            raise ValueError(f"Each exclude entry must be a non-empty mapping, got {entry!r}.")
        unknown = sorted(set(entry) - set(AXES))
        if unknown:
            raise ValueError(f"exclude names unknown axis {unknown}; axes: {list(AXES)}.")
        entries.append(
            {
                axis: [
                    _exclude_value(axis, v)
                    for v in (options if isinstance(options, list) else [options])
                ]
                for axis, options in entry.items()
            }
        )
    return tuple(entries)


def _exclude_value(axis: str, value: Any) -> Any:
    if axis in ("snr_db", "duration_s", "coupling") and value is not None:
        return float(value)
    return value


def _pairs_block(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"A protocol of kind 'region_pairs' needs a 'pairs' block, got {value!r}.")
    _check_keys(value, _PAIRS_KEYS, "'pairs'")
    for key in sorted(_PAIRS_KEYS):
        if key not in value:
            raise ValueError(f"'pairs' is missing the required key {key!r}.")
    regions = value["regions"]
    if not isinstance(regions, (list, tuple)) or len(regions) < 2:
        raise ValueError(f"pairs.regions must be a list of at least two regions, got {regions!r}.")
    if len(set(regions)) != len(regions):
        raise ValueError("pairs.regions lists a region twice.")
    for region in regions:
        if not isinstance(region, str) or not region.endswith(("-lh", "-rh")):
            raise ValueError(
                "pairs.regions must be Desikan-Killiany labels ending in -lh or -rh, "
                f"got {region!r}."
            )
    n = _integer(value["n"], "pairs.n", minimum=1)
    total = len(regions) * (len(regions) - 1) // 2
    if n > total:
        raise ValueError(f"pairs.n ({n}) is more than the {total} pairs of {len(regions)} regions.")
    return {
        "n": n,
        "seed": _integer(value["seed"], "pairs.seed", minimum=0),
        "regions": list(regions),
    }


def _boolean(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false, got {value!r}.")
    return value


def _settings(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"settings must be a mapping of scalars, got {value!r}.")
    for key, item in value.items():
        if not isinstance(key, str) or not (
            item is None or isinstance(item, (bool, int, float, str))
        ):
            raise ValueError(f"settings entries must be scalars, got {key!r}: {item!r}.")
    return dict(value)
