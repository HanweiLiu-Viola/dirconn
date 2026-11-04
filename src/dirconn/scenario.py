"""Versioned scenario files: a ground-truth network together with how to simulate it.

A scenario is one YAML file with ``version: 1``. It holds the network (nodes and edges, as in
:meth:`dirconn.network.Network.from_dict`), the own dynamics of the nodes, defaults for the
sampling rate, the SNR and the data length, the background sources, and optionally nodes that are
simulated but hidden from the methods. :func:`load_scenario` reads a file or one of the files
shipped in :data:`SCENARIO_DIR`; :func:`list_scenarios` names the shipped ones.

All keys are checked: an unknown key, an edge to an unknown node, a band or oscillator above the
Nyquist frequency, a delay that is not a whole number of samples and an unstable network are errors
with a message that names the key, the node or the edge.
"""

from __future__ import annotations

import dataclasses
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dirconn.network import Edge, Network, dk_node
from dirconn.simulation import NodeDynamics, network_var, network_var_schedule

SCENARIO_VERSION = 1
SCENARIO_DIR = Path(__file__).parent / "scenarios"
"""Directory of the shipped scenario files, one ``<name>.yaml`` each."""

DEFAULT_SFREQ = 100.0
DEFAULT_SNR_DB = 10.0
DEFAULT_DURATION_S = 120.0
DEFAULT_N_BACKGROUND = 10
DEFAULT_BACKGROUND_FRACTION = 0.5

_TOP_KEYS = {
    "version",
    "name",
    "description",
    "network",
    "dynamics",
    "defaults",
    "background",
    "hidden",
    "artifact",
}
_NETWORK_KEYS = {"nodes", "edges"}
_DYNAMICS_KEYS = {"default", "nodes"}
_DYNAMICS_FIELDS = {f.name for f in dataclasses.fields(NodeDynamics)}
_DEFAULTS_KEYS = {"sfreq", "snr_db", "duration_s"}
_BACKGROUND_KEYS = {"n_sources", "fraction"}
_ARTIFACT_REQUIRED = ("rate_hz", "freq_hz", "decay_ms", "amplitude", "anchor")
_ARTIFACT_KEYS = set(_ARTIFACT_REQUIRED) | {"n_sensors"}
DEFAULT_ARTIFACT_SENSORS = 6


@dataclass(frozen=True)
class ArtifactSpec:
    """A stimulus-locked artifact added at sensor level.

    A stimulus occurs every ``1 / rate_hz`` seconds, starting at the first sample of every trial;
    each one adds the same damped sine (``freq_hz``, decay time ``decay_ms``) to a fixed group of
    sensors: the ``n_sensors`` where the topography of the node ``anchor`` is strongest, with that
    topography's signs. Its power over the group is ``amplitude ** 2`` times the power of the
    noise-free signal over the same sensors. It exists only at sensor level, so it is meant for
    stage 2 and 3 of the benchmark; the region-level stage does not see it.
    """

    rate_hz: float
    freq_hz: float
    decay_ms: float
    amplitude: float
    anchor: str
    n_sensors: int = DEFAULT_ARTIFACT_SENSORS


@dataclass(frozen=True)
class ScenarioSpec:
    """A ground-truth network and how to simulate it.

    Attributes
    ----------
    name, description : str
        Identifier (the file name without ``.yaml`` for shipped files) and the network in words.
    network : Network
        All nodes, including hidden ones.
    dynamics : dict of str to NodeDynamics
        Own dynamics of every node.
    sfreq, snr_db, duration_s : float
        Defaults for the sampling rate (Hz), the sensor SNR (dB) and the data length (s); a
        protocol may override the last two.
    n_background, background_fraction : int, float
        Number of background 1/f sources and the share of the noise power they carry.
    hidden : tuple of str
        Nodes that are simulated but not given to the methods.
    artifact : ArtifactSpec, optional
        A stimulus-locked sensor-level artifact; ``None`` for none.
    """

    name: str
    description: str
    network: Network
    dynamics: dict[str, NodeDynamics]
    sfreq: float = DEFAULT_SFREQ
    snr_db: float = DEFAULT_SNR_DB
    duration_s: float = DEFAULT_DURATION_S
    n_background: int = DEFAULT_N_BACKGROUND
    background_fraction: float = DEFAULT_BACKGROUND_FRACTION
    hidden: tuple[str, ...] = ()
    artifact: ArtifactSpec | None = None

    @property
    def n_times(self) -> int:
        """Samples of the default data length."""
        return round(self.duration_s * self.sfreq)

    @property
    def analysed(self) -> tuple[str, ...]:
        """Nodes the methods see: all but the hidden ones, in network order."""
        return tuple(n for n in self.network.names if n not in self.hidden)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScenarioSpec:
        """Build a scenario from a dictionary, for example loaded from YAML.

        Raises
        ------
        ValueError
            For anything the module docstring lists; the message names the offending key.
        """
        _check_keys(data, _TOP_KEYS, "the scenario")
        _check_version(data)
        for key in ("name", "network"):
            if key not in data:
                raise ValueError(f"The scenario is missing the required key {key!r}.")
        name = str(data["name"])
        network_data = _mapping(data["network"], "network")
        _check_keys(network_data, _NETWORK_KEYS, "'network'")
        network = Network.from_dict({"name": name, **network_data})

        defaults = _mapping(data.get("defaults"), "defaults")
        _check_keys(defaults, _DEFAULTS_KEYS, "'defaults'")
        sfreq = _number(defaults, "sfreq", DEFAULT_SFREQ, low=0.0)
        snr_db = _number(defaults, "snr_db", DEFAULT_SNR_DB)
        duration_s = _number(defaults, "duration_s", DEFAULT_DURATION_S, low=0.0)
        if round(duration_s * sfreq) < 1:
            raise ValueError(
                f"duration_s ({duration_s:g} s) is shorter than one sample at {sfreq:g} Hz."
            )

        background = _mapping(data.get("background"), "background")
        _check_keys(background, _BACKGROUND_KEYS, "'background'")
        n_sources = background.get("n_sources", DEFAULT_N_BACKGROUND)
        if isinstance(n_sources, bool) or not isinstance(n_sources, int) or n_sources < 0:
            raise ValueError(f"n_sources must be a non-negative integer, got {n_sources!r}.")
        fraction = _number(
            background, "fraction", DEFAULT_BACKGROUND_FRACTION, low=0.0, high=1.0, low_open=False
        )

        dynamics = _dynamics(data.get("dynamics"), network)
        for node, own in dynamics.items():
            try:
                own.ar_coefficients(sfreq)
            except ValueError as error:
                raise ValueError(f"Node {node!r}: {error}") from None
        try:
            network_var_schedule(network, sfreq, round(duration_s * sfreq), dynamics)
        except ValueError as error:
            raise ValueError(f"The network {name!r} cannot be simulated: {error}") from None

        hidden = data.get("hidden") or ()
        if isinstance(hidden, str) or not isinstance(hidden, (list, tuple)):
            raise ValueError(f"hidden must be a list of node names, got {hidden!r}.")
        unknown = sorted(set(hidden) - set(network.names))
        if unknown:
            raise ValueError(f"hidden names nodes that are not in the network: {unknown}.")
        if network.n_nodes - len(set(hidden)) < 2:
            raise ValueError("hidden leaves fewer than two analysed nodes.")

        artifact = _artifact(data.get("artifact"), network, sfreq)

        return cls(
            name=name,
            description=str(data.get("description", "")),
            network=network,
            dynamics=dynamics,
            sfreq=sfreq,
            snr_db=snr_db,
            duration_s=duration_s,
            n_background=n_sources,
            background_fraction=fraction,
            hidden=tuple(hidden),
            artifact=artifact,
        )

    def to_yaml(self, path: str | os.PathLike[str]) -> None:
        """Write the scenario to a YAML file that :func:`load_scenario` reads back."""
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-compatible dictionary that :meth:`from_dict` restores exactly."""
        network = self.network.to_dict()
        del network["name"]
        out: dict[str, Any] = {
            "version": SCENARIO_VERSION,
            "name": self.name,
            "description": self.description,
            "network": network,
            "dynamics": {
                "nodes": {n: dataclasses.asdict(d) for n, d in self.dynamics.items()},
            },
            "defaults": {
                "sfreq": self.sfreq,
                "snr_db": self.snr_db,
                "duration_s": self.duration_s,
            },
            "background": {
                "n_sources": self.n_background,
                "fraction": self.background_fraction,
            },
            "hidden": list(self.hidden),
        }
        if self.artifact is not None:
            out["artifact"] = dataclasses.asdict(self.artifact)
        return out


def _artifact(value: Any, network: Network, sfreq: float) -> ArtifactSpec | None:
    if value is None:
        return None
    block = _mapping(value, "artifact")
    _check_keys(block, _ARTIFACT_KEYS, "'artifact'")
    for key in _ARTIFACT_REQUIRED:
        if key not in block:
            raise ValueError(f"'artifact' is missing the required key {key!r}.")
    rate_hz = _number(block, "rate_hz", 0.0, low=0.0)
    freq_hz = _number(block, "freq_hz", 0.0, low=0.0)
    decay_ms = _number(block, "decay_ms", 0.0, low=0.0)
    amplitude = _number(block, "amplitude", 0.0, low=0.0)
    if freq_hz >= sfreq / 2.0:
        raise ValueError(
            f"freq_hz ({freq_hz:g} Hz) of the artifact must be below the Nyquist frequency "
            f"({sfreq / 2.0:g} Hz)."
        )
    period = sfreq / rate_hz
    if abs(period - round(period)) > 1e-6 or round(period) < 1:
        raise ValueError(
            f"rate_hz ({rate_hz:g} Hz) must give a whole number of samples per stimulus at "
            f"{sfreq:g} Hz, got {period:g}."
        )
    if decay_ms / 1000.0 > period / sfreq:
        raise ValueError(
            f"decay_ms ({decay_ms:g} ms) of the artifact is longer than the interval between "
            f"stimuli ({1000.0 * period / sfreq:g} ms)."
        )
    anchor = block["anchor"]
    if anchor not in network.names:
        raise ValueError(f"anchor {anchor!r} of the artifact is not a node of the network.")
    n_sensors = block.get("n_sensors", DEFAULT_ARTIFACT_SENSORS)
    if isinstance(n_sensors, bool) or not isinstance(n_sensors, int) or n_sensors < 1:
        raise ValueError(f"n_sensors must be a positive integer, got {n_sensors!r}.")
    return ArtifactSpec(rate_hz, freq_hz, decay_ms, amplitude, anchor, n_sensors)


def artifact_waveform(spec: ArtifactSpec, sfreq: float, n_times: int) -> np.ndarray:
    """Return the artifact's time course: one damped sine per stimulus, shape ``(n_times,)``.

    The waveform is exactly periodic with the stimulus interval and zero at each stimulus.
    """
    period = round(sfreq / spec.rate_hz)
    tau = (np.arange(n_times) % period) / sfreq
    wave = np.exp(-tau / (spec.decay_ms / 1000.0)) * np.sin(2.0 * np.pi * spec.freq_hz * tau)
    return np.asarray(wave)


def artifact_pattern(topography: np.ndarray, n_sensors: int) -> np.ndarray:
    """Return the spatial pattern of the artifact: the strongest sensors of a topography.

    Everything but the ``n_sensors`` sensors with the largest ``abs`` topography is set to 0 and
    the result has unit norm; the signs of the topography are kept.
    """
    topography = np.asarray(topography, dtype=float)
    if not 1 <= n_sensors <= topography.size:
        raise ValueError(
            f"n_sensors must be between 1 and the number of sensors ({topography.size}), "
            f"got {n_sensors!r}."
        )
    keep = np.argsort(-np.abs(topography), kind="stable")[:n_sensors]
    pattern = np.zeros_like(topography)
    pattern[keep] = topography[keep]
    return np.asarray(pattern / np.linalg.norm(pattern))


def stimulation_artifact(
    spec: ArtifactSpec, topography: np.ndarray, signal: np.ndarray, sfreq: float
) -> np.ndarray:
    """Return the artifact to add to sensor data, with the shape of ``signal``.

    Parameters
    ----------
    spec : ArtifactSpec
    topography : ndarray, shape (n_sensors,)
        Topography of ``spec.anchor`` on the sensors of ``signal``.
    signal : ndarray, shape (n_trials, n_sensors, n_times)
        The noise-free sensor signal; it sets the scale (see :class:`ArtifactSpec`).
    sfreq : float
    """
    signal = np.asarray(signal, dtype=float)
    pattern = artifact_pattern(topography, spec.n_sensors)
    group = pattern != 0.0
    wave = artifact_waveform(spec, sfreq, signal.shape[2])
    unit = np.broadcast_to(pattern[None, :, None] * wave[None, None, :], signal.shape)
    scale = spec.amplitude * np.sqrt(np.mean(signal[:, group] ** 2) / np.mean(unit[:, group] ** 2))
    return np.asarray(unit * scale)


def list_scenarios() -> list[str]:
    """Return the names of the shipped scenarios, sorted."""
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))


def load_scenario(source: str | os.PathLike[str]) -> ScenarioSpec:
    """Load a scenario from a YAML file or by the name of a shipped one.

    Parameters
    ----------
    source : str or path
        A file path, or a bare name such as ``"chain"`` for ``SCENARIO_DIR / "chain.yaml"``.

    Raises
    ------
    FileNotFoundError
        If a path does not exist.
    ValueError
        For an unknown shipped name and for every problem of :meth:`ScenarioSpec.from_dict`; the
        message starts with the file name.
    """
    if isinstance(source, str) and source == Path(source).name and not Path(source).suffix:
        file = SCENARIO_DIR / f"{source}.yaml"
        if not file.is_file():
            raise ValueError(f"Unknown scenario {source!r}; shipped scenarios: {list_scenarios()}.")
    else:
        file = Path(source)
        if not file.exists():
            raise FileNotFoundError(file)
    with open(file, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    try:
        if not isinstance(data, Mapping):
            raise ValueError("it must contain a YAML mapping at the top level.")
        return ScenarioSpec.from_dict(data)
    except ValueError as error:
        raise ValueError(f"{file.name}: {error}") from None


def _check_keys(mapping: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown key(s) {unknown} in {where}; allowed keys: {sorted(allowed)}.")


def _check_version(data: Mapping[str, Any]) -> None:
    if "version" not in data:
        raise ValueError(f"The scenario is missing 'version' (supported: {SCENARIO_VERSION}).")
    if data["version"] != SCENARIO_VERSION or isinstance(data["version"], bool):
        raise ValueError(
            f"Unsupported scenario version {data['version']!r}; supported: {SCENARIO_VERSION}."
        )


def _mapping(value: Any, key: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{key!r} must be a mapping, got {type(value).__name__}.")
    return value


def _number(
    block: Mapping[str, Any],
    key: str,
    default: float,
    *,
    low: float | None = None,
    high: float | None = None,
    low_open: bool = True,
) -> float:
    value = block.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number, got {value!r}.")
    value = float(value)
    if low is not None and (value < low or (low_open and value == low)):
        relation = "greater than" if low_open else "at least"
        raise ValueError(f"{key} must be {relation} {low:g}, got {value:g}.")
    if high is not None and value > high:
        raise ValueError(f"{key} must be at most {high:g}, got {value:g}.")
    return value


def _dynamics(value: Any, network: Network) -> dict[str, NodeDynamics]:
    block = _mapping(value, "dynamics")
    _check_keys(block, _DYNAMICS_KEYS, "'dynamics'")
    default = _node_dynamics(block.get("default"), "dynamics.default")
    per_node = _mapping(block.get("nodes"), "dynamics.nodes")
    unknown = sorted(set(per_node) - set(network.names))
    if unknown:
        raise ValueError(f"dynamics.nodes names nodes that are not in the network: {unknown}.")
    return {
        name: _node_dynamics(per_node[name], f"dynamics.nodes.{name}")
        if name in per_node
        else default
        for name in network.names
    }


def _node_dynamics(value: Any, where: str) -> NodeDynamics:
    fields = _mapping(value, where)
    _check_keys(fields, _DYNAMICS_FIELDS, f"'{where}'")
    return NodeDynamics(**fields)


RANDOM_SEEDS: dict[str, int] = {"random_5": 1005, "random_10": 1010, "random_20": 1020}
"""Seeds of the shipped random scenarios; ``scripts/make_random_scenarios.py`` regenerates them."""


@dataclass(frozen=True)
class RandomScenario:
    """A generated scenario with how it came about.

    Attributes
    ----------
    spec : ScenarioSpec
        The scenario; its description records the seed and the number of rejected graphs.
    n_rejected : int
        Graphs drawn and thrown away because their spectral radius reached ``max_radius``.
    n_acyclic : int
        Graphs drawn and thrown away because they had feedback edges but no cycle (a feedback
        edge closes a cycle only when a forward path leads back to it).
    n_back_edges : int
        Edges that go against the hidden order of the nodes (the rest of the graph is acyclic).
    order : tuple of str
        Node names in that order; every other edge goes from an earlier to a later node.
    """

    spec: ScenarioSpec
    n_rejected: int
    n_acyclic: int
    n_back_edges: int
    order: tuple[str, ...]


def _has_cycle(adjacency: np.ndarray) -> bool:
    reach = np.asarray(adjacency, dtype=bool)
    closure = reach.copy()
    for _ in range(reach.shape[0]):
        closure = closure | (closure.astype(int) @ reach.astype(int) > 0)
    return bool(np.diag(closure).any())


def random_scenario(
    n_nodes: int,
    *,
    seed: int,
    regions: Sequence[str],
    name: str | None = None,
    edges_per_node: float = 1.5,
    n_back_edges: int = 2,
    strength: tuple[float, float] = (0.2, 0.4),
    delay_samples: tuple[int, int] = (1, 3),
    freq_hz: tuple[float, float] = (8.0, 30.0),
    bandwidth_hz: float = 4.0,
    sfreq: float = DEFAULT_SFREQ,
    max_radius: float = 0.95,
    max_tries: int = 1000,
) -> RandomScenario:
    """Draw a random sparse directed network that is stable.

    ``n_nodes`` regions are drawn without replacement from ``regions``; their order is the
    hidden order. ``round(edges_per_node * n_nodes)`` edges are drawn on distinct unordered
    pairs, all but ``n_back_edges`` of them from the earlier to the later node of the pair, so
    the graph is acyclic except for the feedback edges. Strengths are uniform in ``strength``
    (positive), delays a uniform whole number of samples in ``delay_samples``, and every node
    is an oscillator with a frequency uniform in ``freq_hz``. A graph whose spectral radius
    (:attr:`~dirconn.simulation.VarModel.spectral_radius`) is ``max_radius`` or more is thrown
    away and drawn again, and so is a graph without any cycle when ``n_back_edges`` is above 0;
    both numbers of throw-aways are returned and the first is written into the description.
    The regions are drawn once; only the graph and the frequencies are redrawn.

    Raises
    ------
    ValueError
        For arguments that cannot work (the message names them) and when ``max_tries`` graphs
        in a row are rejected.
    """
    n = int(n_nodes)
    if n < 2:
        raise ValueError(f"n_nodes must be at least 2, got {n_nodes!r}.")
    pool = list(regions)
    if len(set(pool)) != len(pool):
        raise ValueError("regions must be unique.")
    if len(pool) < n:
        raise ValueError(f"regions has {len(pool)} names, fewer than n_nodes ({n}).")
    n_edges = round(edges_per_node * n)
    max_edges = n * (n - 1) // 2
    if not 1 <= n_edges <= max_edges:
        raise ValueError(
            f"edges_per_node={edges_per_node:g} gives {n_edges} edges; "
            f"{n} nodes allow 1 to {max_edges}."
        )
    if not 0 <= n_back_edges <= n_edges:
        raise ValueError(f"n_back_edges must be between 0 and {n_edges}, got {n_back_edges!r}.")
    low, high = strength
    if not 0.0 < low <= high:
        raise ValueError(f"strength must be (low, high) with 0 < low <= high, got {strength!r}.")
    dmin, dmax = delay_samples
    if not 1 <= dmin <= dmax:
        raise ValueError(
            f"delay_samples must be (min, max) with 1 <= min <= max, got {delay_samples!r}."
        )
    if not 0.0 < max_radius < 1.0:
        raise ValueError(f"max_radius must be in (0, 1), got {max_radius!r}.")

    rng = np.random.default_rng(seed)
    names = tuple(pool[i] for i in rng.choice(len(pool), size=n, replace=False))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rejected = 0
    acyclic = 0
    tries = 0
    while True:
        tries += 1
        chosen = rng.choice(len(pairs), size=n_edges, replace=False)
        is_back = np.zeros(n_edges, dtype=bool)
        is_back[rng.choice(n_edges, size=n_back_edges, replace=False)] = True
        edges = []
        for k, back in zip(chosen, is_back, strict=True):
            i, j = pairs[k]
            source, target = (j, i) if back else (i, j)
            edges.append(
                Edge(
                    names[source],
                    names[target],
                    round(float(rng.uniform(low, high)), 3),
                    int(rng.integers(dmin, dmax + 1)) * 1000.0 / sfreq,
                )
            )
        freqs = np.round(rng.uniform(*freq_hz, size=n), 1)
        dynamics = {
            node: NodeDynamics(freq_hz=float(f), bandwidth_hz=bandwidth_hz)
            for node, f in zip(names, freqs, strict=True)
        }
        label = name or f"random_{n}"
        network = Network(tuple(dk_node(x) for x in names), tuple(edges), name=label)
        if n_back_edges > 0 and not _has_cycle(network.adjacency()):
            acyclic += 1
        else:
            try:
                radius = network_var(network, sfreq, dynamics).spectral_radius
            except ValueError:  # at or above the stability limit of network_var itself
                radius = float("inf")
            if radius < max_radius:
                break
            rejected += 1
        if tries >= max_tries:
            raise ValueError(
                f"No acceptable graph in {max_tries} draws ({rejected} with a spectral radius of "
                f"{max_radius:g} or more, {acyclic} without a cycle); raise max_radius or "
                "max_tries, or lower the strengths (max_tries)."
            )
    cycle = ", and the graph has a cycle" if n_back_edges else ""
    description = (
        f"A random sparse network of {n} regions with {n_edges} edges ({n_back_edges} against "
        f"the hidden order{cycle}), strengths {low:g} to {high:g}, delays {dmin} to {dmax} "
        f"samples, oscillators of {freq_hz[0]:g} to {freq_hz[1]:g} Hz. Drawn with seed {seed}; "
        f"{rejected} rejected (spectral radius {max_radius:g} or more) and {acyclic} redrawn "
        f"(no cycle) before this one, whose spectral radius is {radius:.3f}."
    )
    spec = ScenarioSpec(
        name=label, description=description, network=network, dynamics=dynamics, sfreq=sfreq
    )
    return RandomScenario(spec, rejected, acyclic, n_back_edges, names)
