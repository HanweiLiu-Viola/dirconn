"""Ground-truth network specification: nodes, directed edges, delays and bands."""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dirconn.channels import ChannelMeta
from dirconn.exceptions import DirconnWarning
from dirconn.provenance import Provenance
from dirconn.result import Connectivity

ROUNDING_MODES: tuple[str, ...] = ("error", "round")
KINDS: tuple[str, ...] = ("cortical", "direct")
_ALIGNMENT_TOLERANCE = 1e-6  # samples

_NODE_KEYS = {"name", "kind", "type", "region", "hemisphere", "atlas"}
_EDGE_KEYS = {"source", "target", "strength", "delay_ms", "band", "active"}
_TIME_TOLERANCE = 1e-9
_NETWORK_KEYS = {"name", "nodes", "edges"}


def dk_node(name: str, **overrides: Any) -> ChannelMeta:
    """Return a cortical node for a Desikan-Killiany label such as ``"inferiortemporal-lh"``.

    Region, hemisphere and atlas are derived from the label; keyword arguments override them.
    """
    for suffix, hemisphere in (("-lh", "lh"), ("-rh", "rh")):
        if name.endswith(suffix) and len(name) > len(suffix):
            fields: dict[str, Any] = {
                "type": "misc",
                "node_kind": "cortical",
                "region": name[: -len(suffix)],
                "hemisphere": hemisphere,
                "atlas": "DK",
            }
            fields.update(overrides)
            return ChannelMeta(name, **fields)
    raise ValueError(
        "A Desikan-Killiany label must end with '-lh' or '-rh' to give the hemisphere, "
        f"got {name!r}."
    )


def direct_node(
    name: str, region: str | None = None, hemisphere: str | None = None, **overrides: Any
) -> ChannelMeta:
    """Return a direct node: an invasive channel used as recorded, for example ``"STN-L"``."""
    fields: dict[str, Any] = {
        "type": "misc",
        "node_kind": "direct",
        "region": region,
        "hemisphere": hemisphere,
    }
    fields.update(overrides)
    return ChannelMeta(name, **fields)


@dataclass(frozen=True)
class Edge:
    """A directed coupling from ``source`` to ``target``.

    Parameters
    ----------
    source, target : str
        Node names; they must differ.
    strength : float
        Coupling strength: finite and non-zero, negative for inhibitory coupling.
    delay_ms : float
        Transmission delay in milliseconds, 0 for instantaneous coupling.
    band : (float, float), optional
        Frequency band in Hz where the coupling acts; ``None`` means broadband.
    active : sequence of (float, float), optional
        Time intervals ``(start, stop)`` in seconds from the start of the recording in which the
        edge acts; ``None`` (default) means all the time. An interval contains ``start`` and not
        ``stop`` (it is half open, so intervals can follow each other), they are ascending and
        do not overlap, and the switching is hard: the coupling is on or off, never faded.
        The simulator needs them to lie inside the recording.
    """

    source: str
    target: str
    strength: float
    delay_ms: float = 0.0
    band: tuple[float, float] | None = None
    active: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        """Validate and normalise the fields."""
        for field_name in ("source", "target"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty node name.")
        if self.source == self.target:
            raise ValueError(f"self-loops are not allowed: {self.source!r} -> {self.target!r}.")
        strength = float(self.strength)
        if not math.isfinite(strength) or strength == 0.0:
            raise ValueError(f"strength must be finite and non-zero, got {self.strength!r}.")
        delay = float(self.delay_ms)
        if not math.isfinite(delay) or delay < 0.0:
            raise ValueError(f"delay_ms must be finite and non-negative, got {self.delay_ms!r}.")
        object.__setattr__(self, "strength", strength)
        object.__setattr__(self, "delay_ms", delay)
        if self.band is not None:
            try:
                low, high = (float(b) for b in self.band)
            except (TypeError, ValueError):
                raise ValueError(f"band must be (low, high) in Hz, got {self.band!r}.") from None
            if not (math.isfinite(low) and math.isfinite(high)) or low < 0.0 or low >= high:
                raise ValueError(f"band must be finite with 0 <= low < high, got {self.band!r}.")
            object.__setattr__(self, "band", (low, high))
        if self.active is not None:
            object.__setattr__(self, "active", _check_active(self.active))

    def is_active(self, times: Any) -> np.ndarray:
        """Return a boolean array: whether the edge acts at each time in ``times`` (seconds).

        An interval contains its start and not its stop; times within a nanosecond of a limit are
        treated as being on it, so that ``arange(n) * 0.1`` works as expected.
        """
        t = np.asarray(times, dtype=float)
        if self.active is None:
            return np.ones(t.shape, dtype=bool)
        on = np.zeros(t.shape, dtype=bool)
        for start, stop in self.active:
            on |= (t >= start - _TIME_TOLERANCE) & (t < stop - _TIME_TOLERANCE)
        return on

    def delay_samples(self, sfreq: float, rounding: str = "error") -> int:
        """Return the delay as a whole number of samples at ``sfreq``.

        Parameters
        ----------
        sfreq : float
            Sampling frequency in Hz.
        rounding : {"error", "round"}
            What to do when the delay is not a whole number of samples. ``"error"`` raises
            ``ValueError``; ``"round"`` rounds half up and warns with :class:`DirconnWarning`.
        """
        if rounding not in ROUNDING_MODES:
            raise ValueError(f"rounding must be one of {ROUNDING_MODES}, got {rounding!r}.")
        sfreq = float(sfreq)
        if not math.isfinite(sfreq) or sfreq <= 0.0:
            raise ValueError(f"sfreq must be a positive finite number, got {sfreq!r}.")
        exact = self.delay_ms * sfreq / 1000.0
        samples = math.floor(exact + 0.5)
        if abs(exact - samples) > _ALIGNMENT_TOLERANCE:
            if rounding == "error":
                raise ValueError(
                    f"A delay of {self.delay_ms:g} ms is not a whole number of samples at "
                    f"{sfreq:g} Hz ({exact:g} samples); use a multiple of {1000.0 / sfreq:g} ms "
                    "or pass rounding='round'."
                )
            warnings.warn(
                f"The delay of {self.delay_ms:g} ms was rounded to {samples} samples at "
                f"{sfreq:g} Hz.",
                DirconnWarning,
                stacklevel=2,
            )
        return int(samples)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-compatible dictionary; ``band`` is left out when broadband."""
        out: dict[str, Any] = {
            "source": self.source,
            "target": self.target,
            "strength": self.strength,
            "delay_ms": self.delay_ms,
        }
        if self.band is not None:
            out["band"] = list(self.band)
        if self.active is not None:
            out["active"] = [list(pair) for pair in self.active]
        return out


@dataclass(frozen=True)
class Network:
    """A directed network of cortical regions and direct (invasive) nodes.

    Parameters
    ----------
    nodes : sequence of ChannelMeta
        At least two nodes with unique names and ``node_kind`` ``"cortical"`` or ``"direct"``;
        see :func:`dk_node` and :func:`direct_node`.
    edges : sequence of Edge
        Directed couplings between the nodes. The same pair may appear several times with
        different bands; exact duplicates are rejected.
    name : str
        Optional label, for example the scenario name.
    """

    nodes: tuple[ChannelMeta, ...]
    edges: tuple[Edge, ...] = ()
    name: str = ""

    def __post_init__(self) -> None:
        """Validate and normalise the fields."""
        nodes = tuple(self.nodes)
        edges = tuple(self.edges)
        if len(nodes) < 2:
            raise ValueError(f"A network needs at least two nodes, got {len(nodes)}.")
        names = [n.name for n in nodes]
        if len(set(names)) != len(names):
            raise ValueError("Node names must be unique.")
        for node in nodes:
            if node.node_kind not in KINDS:
                raise ValueError(
                    f"Node {node.name!r} has node_kind {node.node_kind!r}; networks contain "
                    f"only {KINDS} nodes."
                )
        known = set(names)
        seen: dict[tuple[str, str, tuple[float, float] | None], list[Edge]] = {}
        for edge in edges:
            for end in (edge.source, edge.target):
                if end not in known:
                    raise ValueError(
                        f"Edge {edge.source!r} -> {edge.target!r} uses unknown node {end!r}."
                    )
            key = (edge.source, edge.target, edge.band)
            if any(_overlap(edge, other) for other in seen.get(key, ())):
                raise ValueError(
                    f"duplicate edge {edge.source!r} -> {edge.target!r} with band {edge.band} "
                    "(the same pair and band may only repeat with active intervals that do not "
                    "overlap)."
                )
            seen.setdefault(key, []).append(edge)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)

    @property
    def names(self) -> tuple[str, ...]:
        """Node names in order."""
        return tuple(n.name for n in self.nodes)

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return len(self.nodes)

    @property
    def has_intervals(self) -> bool:
        """Whether any edge acts only in some time intervals (see :attr:`Edge.active`)."""
        return any(edge.active is not None for edge in self.edges)

    def adjacency(self) -> np.ndarray:
        """Return a boolean matrix; ``[i, j]`` is True when any edge goes from node i to node j."""
        index = {name: i for i, name in enumerate(self.names)}
        matrix = np.zeros((self.n_nodes, self.n_nodes), dtype=bool)
        for edge in self.edges:
            matrix[index[edge.source], index[edge.target]] = True
        return matrix

    def truth(self, freqs: Any = None, times: Any = None) -> Connectivity:
        """Return the ground truth as a binary :class:`Connectivity`.

        Entry ``[i, j]`` is 1 where node ``i`` drives node ``j`` and 0 elsewhere; the diagonal is
        NaN (undefined), like the output of estimators.

        Parameters
        ----------
        freqs : array_like, optional
            Frequency axis in Hz, normally the one of the result to be evaluated. A banded edge
            is 1 only at frequencies inside its band (limits included). Without ``freqs`` the
            truth is broadband: the axis has length 1 with a NaN coordinate and every edge,
            banded or not, counts as present.
        times : array_like, optional
            Time axis in seconds, from the start of the recording. An edge with
            :attr:`Edge.active` intervals is 1 only at the times inside them; an edge without
            intervals is 1 at every time. Without ``times`` the truth has no time axis (length 1,
            NaN coordinate) and an edge with intervals counts as present, as a banded edge does
            without ``freqs``.

        Returns
        -------
        Connectivity
        """
        index = {name: i for i, name in enumerate(self.names)}
        n = self.n_nodes
        freq_axis = None if freqs is None else np.asarray(freqs, dtype=float)
        n_freqs = 1 if freq_axis is None else freq_axis.shape[0]
        time_axis = None if times is None else np.asarray(times, dtype=float)
        n_times = 1 if time_axis is None else time_axis.shape[0]
        values4 = np.zeros((n, n, n_freqs, n_times))
        for edge in self.edges:
            i, j = index[edge.source], index[edge.target]
            on_freq: np.ndarray = np.ones(n_freqs, dtype=bool)
            if edge.band is not None and freq_axis is not None:
                low, high = edge.band
                on_freq = (freq_axis >= low) & (freq_axis <= high)
            on_time: np.ndarray = np.ones(n_times, dtype=bool)
            if edge.active is not None and time_axis is not None:
                on_time = edge.is_active(time_axis)
            values4[i, j][np.ix_(on_freq, on_time)] = 1.0
        values4[np.arange(n), np.arange(n)] = np.nan
        provenance = Provenance(
            method="truth",
            parameters={"name": self.name, "edges": [e.to_dict() for e in self.edges]},
            backend="Network.truth",
        )
        return Connectivity.from_array(
            values4,
            source=self.names,
            freqs=freq_axis,
            times=time_axis,
            provenance=provenance,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-compatible dictionary that :meth:`from_dict` restores exactly."""
        nodes = []
        for node in self.nodes:
            entry: dict[str, Any] = {
                "name": node.name,
                "kind": node.node_kind,
                "region": node.region,
                "hemisphere": node.hemisphere,
                "atlas": node.atlas,
            }
            if node.type != "misc":
                entry["type"] = node.type
            nodes.append(entry)
        return {
            "name": self.name,
            "nodes": nodes,
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Network:
        """Build a network from a dictionary, for example loaded from YAML.

        Unknown keys are errors, so a misspelt field is never ignored silently. Nodes have
        ``kind`` ``cortical`` (default) or ``direct``. For a cortical node whose name is a
        Desikan-Killiany label, ``region``, ``hemisphere`` and ``atlas`` are derived from the
        name unless given; an explicit ``null`` switches the derivation off for that field.
        """
        _check_keys(data, _NETWORK_KEYS, "the network")
        if "nodes" not in data:
            raise ValueError("Network data needs a 'nodes' list.")
        nodes = [_node_from_dict(item) for item in data["nodes"]]
        edges = [_edge_from_dict(item) for item in data.get("edges") or ()]
        return cls(nodes=tuple(nodes), edges=tuple(edges), name=str(data.get("name", "")))

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> Network:
        """Load a network from a YAML file; see :meth:`from_dict` for the format."""
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(file)
        with open(file, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, Mapping):
            raise ValueError(f"{file.name} must contain a YAML mapping at the top level.")
        return cls.from_dict(data)

    def to_yaml(self, path: str | os.PathLike[str]) -> None:
        """Write the network to a YAML file."""
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)


def _check_keys(mapping: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown key(s) {unknown} in {where}; allowed keys: {sorted(allowed)}.")


def _node_from_dict(item: Mapping[str, Any]) -> ChannelMeta:
    _check_keys(item, _NODE_KEYS, "a node")
    if "name" not in item:
        raise ValueError("A node is missing the required key 'name'.")
    name = str(item["name"])
    kind = item.get("kind", "cortical")
    if kind not in KINDS:
        raise ValueError(f"A node's kind must be one of {KINDS}, got {kind!r}.")
    fields: dict[str, Any] = {"region": None, "hemisphere": None, "atlas": None}
    if kind == "cortical" and name.endswith(("-lh", "-rh")) and len(name) > 3:
        derived = dk_node(name)
        fields = {
            "region": derived.region,
            "hemisphere": derived.hemisphere,
            "atlas": derived.atlas,
        }
    for key in ("region", "hemisphere", "atlas"):
        if key in item:
            fields[key] = item[key]
    return ChannelMeta(name, type=item.get("type", "misc"), node_kind=kind, **fields)


def _edge_from_dict(item: Mapping[str, Any]) -> Edge:
    _check_keys(item, _EDGE_KEYS, "an edge")
    for key in ("source", "target", "strength"):
        if key not in item:
            raise ValueError(f"An edge is missing the required key {key!r}.")
    band: Sequence[float] | None = item.get("band")
    return Edge(
        source=item["source"],
        target=item["target"],
        strength=item["strength"],
        delay_ms=item.get("delay_ms", 0.0),
        band=None if band is None else tuple(band),  # type: ignore[arg-type]
        active=item.get("active"),
    )


def _check_active(active: Any) -> tuple[tuple[float, float], ...]:
    """Normalise the ``active`` intervals of an edge to a tuple of ``(start, stop)`` floats."""
    expected = "active must be a non-empty sequence of (start, stop) pairs in seconds, or None"
    if isinstance(active, (str, bytes)):
        raise ValueError(f"{expected}, got {active!r}.")
    try:
        items = list(active)
    except TypeError:
        raise ValueError(f"{expected}, got {active!r}.") from None
    if not items:
        raise ValueError(f"{expected}, got {active!r}.")
    pairs: list[tuple[float, float]] = []
    for item in items:
        try:
            start, stop = (float(v) for v in item)
        except (TypeError, ValueError):
            raise ValueError(f"{expected}, got the entry {item!r}.") from None
        if not (math.isfinite(start) and math.isfinite(stop)):
            raise ValueError(f"The limits of an active interval must be finite, got {item!r}.")
        if start < 0.0:
            raise ValueError(f"The start of an active interval must not be negative, got {item!r}.")
        if start >= stop:
            raise ValueError(
                f"The start must be smaller than the stop of an active interval, got {item!r}."
            )
        pairs.append((start, stop))
    for (_, previous_stop), (next_start, _) in zip(pairs, pairs[1:], strict=False):
        if next_start < previous_stop:
            raise ValueError(
                f"The active intervals must be ascending and must not overlap, got {pairs!r}."
            )
    return tuple(pairs)


def _overlap(first: Edge, second: Edge) -> bool:
    """Whether two edges of the same pair and band could act at the same time."""
    if first.active is None or second.active is None:
        return True
    return any(a0 < b1 and b0 < a1 for a0, a1 in first.active for b0, b1 in second.active)
