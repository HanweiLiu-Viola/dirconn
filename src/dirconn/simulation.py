"""Simulation of directed networks with a tiny artificial forward model.

A :class:`~dirconn.network.Network` is turned into a coupled linear system: every node has its
own dynamics (an AR(2) oscillator or white noise) and every edge adds ``strength * x_source[t -
delay]`` to its target. Edges with a band pass the source through a causal band-pass filter
first. The node activity is mixed to sensors with a leadfield, background 1/f sources and white
noise are added at an exactly defined signal-to-noise ratio, and invasive ("direct") nodes are
recorded as they are, with their own noise.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal

from dirconn.channels import ChannelMeta
from dirconn.headmodel import HeadModel
from dirconn.network import ROUNDING_MODES, Edge, Network
from dirconn.provenance import Provenance
from dirconn.result import Connectivity
from dirconn.timeseries import TimeSeries

NODE_SCALES: tuple[str, ...] = ("none", "unit")
SOURCE_MODELS: tuple[str, ...] = ("partial", "focal", "patch")
DYNAMICS_KINDS: tuple[str, ...] = ("oscillator", "white")
DEFAULT_BURN_IN = 1000
STABILITY_LIMIT = 0.999
_BLOW_UP = 1e8


@dataclass(frozen=True)
class NodeDynamics:
    """Own dynamics of a network node.

    Parameters
    ----------
    kind : {"oscillator", "white"}
        ``"oscillator"`` is an AR(2) process with a spectral peak, ``"white"`` has no memory.
    freq_hz : float
        Centre frequency of the oscillator in Hz (must stay below the Nyquist frequency).
    bandwidth_hz : float
        Width of the oscillator peak in Hz; the pole radius is ``exp(-pi * bandwidth / sfreq)``.
    noise_std : float
        Standard deviation of the innovations driving the node.
    """

    kind: str = "oscillator"
    freq_hz: float = 10.0
    bandwidth_hz: float = 2.0
    noise_std: float = 1.0

    def __post_init__(self) -> None:
        """Validate the fields."""
        if self.kind not in DYNAMICS_KINDS:
            raise ValueError(f"kind must be one of {DYNAMICS_KINDS}, got {self.kind!r}.")
        for field_name in ("freq_hz", "bandwidth_hz", "noise_std"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be positive and finite, got {value!r}.")
            object.__setattr__(self, field_name, value)

    def ar_coefficients(self, sfreq: float) -> tuple[float, float]:
        """Return the AR(2) coefficients ``(a1, a2)`` at the sampling frequency ``sfreq``."""
        if self.kind == "white":
            return 0.0, 0.0
        if self.freq_hz >= sfreq / 2.0:
            raise ValueError(
                f"freq_hz ({self.freq_hz:g} Hz) must be below the Nyquist frequency "
                f"({sfreq / 2.0:g} Hz)."
            )
        radius = math.exp(-math.pi * self.bandwidth_hz / sfreq)
        return 2.0 * radius * math.cos(2.0 * math.pi * self.freq_hz / sfreq), -(radius**2)


@dataclass(frozen=True, eq=False)
class VarModel:
    """The linear part of a simulated network, in the usual VAR convention.

    ``x[t] = instantaneous @ x[t] + sum_k lagged[k - 1] @ x[t - k] + noise`` with matrices indexed
    ``[target, source]``. Edges with a band are not part of these matrices; they are listed in
    ``banded_edges`` with their delays in samples.
    """

    names: tuple[str, ...]
    sfreq: float
    instantaneous: np.ndarray
    lagged: np.ndarray
    noise_std: np.ndarray
    banded_edges: tuple[Edge, ...] = ()
    banded_delays: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Make the arrays read-only."""
        for field_name in ("instantaneous", "lagged", "noise_std"):
            view = np.asarray(getattr(self, field_name)).view()
            view.flags.writeable = False
            object.__setattr__(self, field_name, view)

    @property
    def order(self) -> int:
        """Number of lags in ``lagged``."""
        return int(self.lagged.shape[0])

    @property
    def spectral_radius(self) -> float:
        """Largest eigenvalue modulus of the linear part (below 1 means stable)."""
        return _spectral_radius(np.asarray(self.instantaneous), np.asarray(self.lagged))


def _check_sfreq(sfreq: float) -> float:
    value = float(sfreq)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"sfreq must be a positive finite number, got {sfreq!r}.")
    return value


def _resolve_dynamics(
    network: Network, dynamics: Mapping[str, NodeDynamics] | None
) -> dict[str, NodeDynamics]:
    dynamics = dict(dynamics or {})
    unknown = sorted(set(dynamics) - set(network.names))
    if unknown:
        raise ValueError(f"dynamics names nodes that are not in the network: {unknown}.")
    return {name: dynamics.get(name, NodeDynamics()) for name in network.names}


def network_var(
    network: Network,
    sfreq: float,
    dynamics: Mapping[str, NodeDynamics] | None = None,
    delay_rounding: str = "error",
    at_time: float | None = None,
) -> VarModel:
    """Translate a network into VAR coefficients.

    Parameters
    ----------
    network : Network
        The coupling structure.
    sfreq : float
        Sampling frequency in Hz; delays are converted to samples with it.
    dynamics : mapping of str to NodeDynamics, optional
        Own dynamics per node name; nodes that are not listed get ``NodeDynamics()``.
    delay_rounding : {"error", "round"}
        See :meth:`Edge.delay_samples`.
    at_time : float, optional
        Moment in seconds, needed when edges have ``active`` intervals: only the edges that act
        then are included. Ignored by a network without intervals. For the whole recording see
        :func:`network_var_schedule`.

    Returns
    -------
    VarModel

    Raises
    ------
    ValueError
        For misaligned delays, banded edges without at least one sample of delay or with a
        band above the Nyquist frequency, singular instantaneous couplings, and systems whose
        linear part is unstable.
    """
    if delay_rounding not in ROUNDING_MODES:
        raise ValueError(f"rounding must be one of {ROUNDING_MODES}, got {delay_rounding!r}.")
    sfreq = _check_sfreq(sfreq)
    if network.has_intervals:
        if at_time is None:
            raise ValueError(
                "The network has edges with active intervals; pass at_time (seconds) to choose "
                "the moment, or use network_var_schedule for the whole recording."
            )
        network = _network_at(network, [at_time])[0]
    dyn = _resolve_dynamics(network, dynamics)
    n = network.n_nodes
    index = {name: i for i, name in enumerate(network.names)}

    linear: list[tuple[int, int, float, int]] = []
    banded_edges: list[Edge] = []
    banded_delays: list[int] = []
    for edge in network.edges:
        delay = edge.delay_samples(sfreq, delay_rounding)
        if edge.band is None:
            linear.append((index[edge.source], index[edge.target], edge.strength, delay))
            continue
        label = f"{edge.source!r} -> {edge.target!r}"
        if delay < 1:
            raise ValueError(
                f"The banded edge {label} needs a delay of at least one sample "
                f"({1000.0 / sfreq:g} ms at {sfreq:g} Hz)."
            )
        if edge.band[1] >= sfreq / 2.0:
            raise ValueError(
                f"The upper edge of the band of {label} ({edge.band[1]:g} Hz) must be below the "
                f"Nyquist frequency ({sfreq / 2.0:g} Hz)."
            )
        banded_edges.append(edge)
        banded_delays.append(delay)

    order = max([2] + [d for *_, d in linear])
    lagged = np.zeros((order, n, n))
    instantaneous = np.zeros((n, n))
    noise_std = np.zeros(n)
    for name, i in index.items():
        a1, a2 = dyn[name].ar_coefficients(sfreq)
        lagged[0, i, i] = a1
        lagged[1, i, i] = a2
        noise_std[i] = dyn[name].noise_std
    for source, target, strength, delay in linear:
        if delay == 0:
            instantaneous[target, source] += strength
        else:
            lagged[delay - 1, target, source] += strength

    _check_stable(instantaneous, lagged)
    return VarModel(
        names=network.names,
        sfreq=sfreq,
        instantaneous=instantaneous,
        lagged=lagged,
        noise_std=noise_std,
        banded_edges=tuple(banded_edges),
        banded_delays=tuple(banded_delays),
    )


def _network_at(network: Network, times: Sequence[float]) -> list[Network]:
    """Return, for each time, the static network of the edges that act then."""
    out = []
    for t in times:
        edges = tuple(
            dataclasses.replace(edge, active=None)
            for edge in network.edges
            if bool(edge.is_active(t))
        )
        out.append(Network(network.nodes, edges, name=network.name))
    return out


def network_var_schedule(
    network: Network,
    sfreq: float,
    n_times: int,
    dynamics: Mapping[str, NodeDynamics] | None = None,
    delay_rounding: str = "error",
) -> list[tuple[int, VarModel]]:
    """Split a recording into stretches in which the set of acting edges is constant.

    Sample ``n`` (counted from the start of the recording, after any burn-in) belongs to the
    interval ``[start, stop)`` when ``start <= n / sfreq < stop``. Stretches that follow each other
    with the same edges are merged, so a network without intervals gives one stretch, equal to
    :func:`network_var`.

    Parameters
    ----------
    network, sfreq, dynamics, delay_rounding
        As for :func:`network_var`.
    n_times : int
        Number of samples of the recording; the intervals must lie inside ``0`` to
        ``n_times / sfreq`` seconds.

    Returns
    -------
    list of (int, VarModel)
        The first sample of each stretch (the first is 0) and its model.

    Raises
    ------
    ValueError
        For an interval outside the recording or one that contains no sample, and for a stretch
        whose coupled system is unstable (the message names the stretch).
    """
    sfreq = _check_sfreq(sfreq)
    n_times = int(n_times)
    if n_times < 1:
        raise ValueError(f"n_times must be at least 1, got {n_times}.")
    if not network.has_intervals:
        return [(0, network_var(network, sfreq, dynamics, delay_rounding))]

    duration = n_times / sfreq
    spans: list[list[tuple[int, int]]] = []
    bounds = {0}
    for edge in network.edges:
        edge_spans = []
        for start, stop in edge.active or ():
            if start >= duration - 1e-9 or stop > duration + 1e-9:
                raise ValueError(
                    f"The active interval {(start, stop)} of the edge {edge.source!r} -> "
                    f"{edge.target!r} must lie inside the recording (0 to {duration:g} s)."
                )
            first = math.ceil(start * sfreq - 1e-9)
            end = min(math.ceil(stop * sfreq - 1e-9), n_times)
            if end <= first:
                raise ValueError(
                    f"The active interval {(start, stop)} of the edge {edge.source!r} -> "
                    f"{edge.target!r} contains no sample at {sfreq:g} Hz."
                )
            edge_spans.append((first, end))
            bounds.update((first, end))
        spans.append(edge_spans)

    starts = sorted(b for b in bounds if b < n_times)
    stretches: list[tuple[int, tuple[bool, ...]]] = []
    for first in starts:
        on = tuple(
            edge.active is None or any(a <= first < b for a, b in edge_spans)
            for edge, edge_spans in zip(network.edges, spans, strict=True)
        )
        if not stretches or stretches[-1][1] != on:
            stretches.append((first, on))

    schedule = []
    for k, (first, on) in enumerate(stretches):
        stop_sample = stretches[k + 1][0] if k + 1 < len(stretches) else n_times
        edges = tuple(
            dataclasses.replace(edge, active=None)
            for edge, acting in zip(network.edges, on, strict=True)
            if acting
        )
        static = Network(network.nodes, edges, name=network.name)
        try:
            model = network_var(static, sfreq, dynamics, delay_rounding)
        except ValueError as error:
            raise ValueError(
                f"{error} (the stretch from {first / sfreq:g} s to {stop_sample / sfreq:g} s)"
            ) from None
        schedule.append((first, model))
    return schedule


def _spectral_radius(instantaneous: np.ndarray, lagged: np.ndarray) -> float:
    n = instantaneous.shape[0]
    mix = np.eye(n) - instantaneous
    if np.linalg.svd(mix, compute_uv=False).min() < 1e-8:
        raise ValueError("The instantaneous couplings make the system singular.")
    effective = np.einsum("ab,kbc->kac", np.linalg.inv(mix), lagged)
    order = effective.shape[0]
    companion = np.zeros((n * order, n * order))
    companion[:n, :] = np.hstack(list(effective))
    companion[n:, :-n] = np.eye(n * (order - 1))
    return float(np.abs(np.linalg.eigvals(companion)).max())


def _check_stable(instantaneous: np.ndarray, lagged: np.ndarray) -> None:
    radius = _spectral_radius(instantaneous, lagged)
    if radius >= STABILITY_LIMIT:
        raise ValueError(
            f"The coupled system is unstable (spectral radius {radius:.3f}); "
            "lower the coupling strengths."
        )


def _sos_step(sos: np.ndarray, state: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Advance a cascade of second-order sections by one sample (direct form II transposed)."""
    for s in range(sos.shape[0]):
        b0, b1, b2, _, a1, a2 = sos[s]
        y = b0 * x + state[s, 0]
        state[s, 0] = b1 * x - a1 * y + state[s, 1]
        state[s, 1] = b2 * x - a2 * y
        x = y
    return x


def _simulate_nodes(
    models: Sequence[VarModel],
    starts: Sequence[int],
    n_trials: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run the coupled system; returns ``(n_trials, n_nodes, n_samples)``.

    ``models[k]`` governs the samples from ``starts[k]`` on (``starts[0]`` is 0). All models share
    the nodes and their own dynamics and differ in the couplings. The band-pass filters of banded
    edges run through the whole recording whether their edge is acting or not, so switching an
    edge on does not restart its filter.
    """
    n = len(models[0].names)
    order = max(model.order for model in models)
    index = {name: i for i, name in enumerate(models[0].names)}
    filters: dict[tuple[int, Any], int] = {}
    sections: list[np.ndarray] = []
    banded: list[list[tuple[int, int, float, int]]] = []
    for model in models:
        entries = []
        for edge, delay in zip(model.banded_edges, model.banded_delays, strict=True):
            band = edge.band
            assert band is not None  # banded edges only
            key = (index[edge.source], band)
            if key not in filters:
                filters[key] = len(sections)
                sections.append(
                    signal.butter(2, list(band), btype="bandpass", fs=model.sfreq, output="sos")
                )
            entries.append((filters[key], index[edge.target], edge.strength, delay))
        banded.append(entries)
    filter_source = {f: source for (source, _), f in filters.items()}
    pad = max([order] + [entry[3] for entries in banded for entry in entries])

    noise = rng.standard_normal((n_trials, n, n_samples)) * models[0].noise_std[None, :, None]
    x = np.zeros((n_trials, n, pad + n_samples))
    z = np.zeros((n_trials, max(len(sections), 1), pad + n_samples))
    states = [np.zeros((sos.shape[0], 2, n_trials)) for sos in sections]
    weights, has_instantaneous, unmix_t = [], [], []
    for model in models:
        lagged = np.zeros((order, n, n))
        lagged[: model.order] = model.lagged
        weights.append(lagged[::-1].transpose(1, 2, 0).reshape(n, n * order))
        has_instantaneous.append(bool(model.instantaneous.any()))
        unmix_t.append(np.linalg.inv(np.eye(n) - model.instantaneous).T)

    current = 0
    for t in range(pad, pad + n_samples):
        while current + 1 < len(models) and t - pad >= starts[current + 1]:
            current += 1
        rhs = (
            noise[:, :, t - pad]
            + x[:, :, t - order : t].reshape(n_trials, n * order) @ weights[current].T
        )
        for f, target, strength, delay in banded[current]:
            rhs[:, target] += strength * z[:, f, t - delay]
        step = rhs @ unmix_t[current] if has_instantaneous[current] else rhs
        x[:, :, t] = step
        for f, sos in enumerate(sections):
            z[:, f, t] = _sos_step(sos, states[f], step[:, filter_source[f]])

    out = x[:, :, pad:]
    if not np.isfinite(out).all() or np.abs(out).max() > _BLOW_UP:
        raise ValueError("The simulated system is unstable: the activity grows without bound.")
    return out


def artificial_leadfield(
    n_sensors: int, n_sources: int, seed: int | np.random.Generator
) -> np.ndarray:
    """Return a tiny artificial leadfield of shape ``(n_sensors, n_sources)``.

    Sensors sit on a circle and sources at random positions inside it; the gain decays with the
    squared distance and every column has unit norm. Nearby sources therefore have similar
    sensor patterns, which gives realistic spatial leakage.
    """
    if n_sensors < 1 or n_sources < 1:
        raise ValueError(
            f"n_sensors and n_sources must be positive, got {n_sensors} and {n_sources}."
        )
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, n_sensors, endpoint=False)
    sensors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    radius = 0.9 * np.sqrt(rng.uniform(size=n_sources))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_sources)
    sources = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    distance2 = ((sensors[:, None, :] - sources[None, :, :]) ** 2).sum(axis=2)
    lead = 1.0 / (distance2 + 0.1)
    return lead / np.linalg.norm(lead, axis=0, keepdims=True)


def _pink_noise(rng: np.random.Generator, shape: tuple[int, int, int]) -> np.ndarray:
    """Unit-variance 1/f (power) noise along the last axis."""
    white = rng.standard_normal(shape)
    spectrum = np.fft.rfft(white, axis=-1)
    freqs = np.fft.rfftfreq(shape[-1])
    amplitude = np.zeros_like(freqs)
    amplitude[1:] = 1.0 / np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum * amplitude, n=shape[-1], axis=-1)
    return pink / pink.std(axis=-1, keepdims=True)


def _readonly(array: np.ndarray) -> np.ndarray:
    view = np.asarray(array).view()
    view.flags.writeable = False
    return view


@dataclass(frozen=True, eq=False, repr=False)
class SimulatedData:
    """Output of :func:`simulate`.

    Attributes
    ----------
    data : TimeSeries
        Sensor-level input for estimators: the sensors followed by the direct channels.
    sources : TimeSeries
        Noise-free activity of every network node, for evaluating estimators directly on the
        region level.
    network : Network
        The network that generated the data.
    leadfield : ndarray, shape (n_sensors, n_cortical + n_background)
        Mixing matrix: cortical nodes first, then background sources.
    signal, noise : ndarray, shape (n_trials, n_sensors, n_times)
        Noise-free sensor signal and the noise added to it; ``data`` is their sum on the sensor
        channels.
    provenance : Provenance
        Everything needed to reproduce the data.
    """

    data: TimeSeries
    sources: TimeSeries
    network: Network
    leadfield: np.ndarray
    signal: np.ndarray
    noise: np.ndarray
    provenance: Provenance

    def __repr__(self) -> str:
        """Return a short summary."""
        return (
            f"SimulatedData(n_trials={self.data.n_trials}, n_nodes={self.network.n_nodes}, "
            f"n_channels={self.data.n_channels}, n_times={self.data.n_times}, "
            f"snr_db={self.snr_db:.3g})"
        )

    @property
    def snr_db(self) -> float:
        """Sensor SNR of the realisation: mean signal power over mean noise power, in dB."""
        return float(10.0 * np.log10(np.mean(self.signal**2) / np.mean(self.noise**2)))

    @property
    def truth(self) -> Connectivity:
        """Broadband ground truth of the network."""
        return self.network.truth()

    def truth_at(self, freqs: Any = None, times: Any = None) -> Connectivity:
        """Ground truth on the frequency and time axes of the result to be evaluated."""
        return self.network.truth(freqs=freqs, times=times)

    def roi_observations(self, *, snr_db: float, seed: int) -> TimeSeries:
        """Return region-level data: every node's activity plus independent white noise.

        This is the data to hand to an estimator when the question is about the estimator and not
        about the head model. The noise of each node is scaled so that its SNR, relative to the
        node's own power, equals ``snr_db`` exactly. This is not the sensor SNR of the simulation
        (a mean over sensors); it only says how well the node series are observed.

        Parameters
        ----------
        snr_db : float
            Signal-to-noise ratio of every node series, in dB.
        seed : int
            Seed of the observation noise (required, so the data can be reproduced).

        Returns
        -------
        TimeSeries
            Channels in network order with the node metadata, same segmentation and time axis as
            ``sources``.
        """
        rng = np.random.default_rng(seed)
        activity = np.asarray(self.sources.data)
        noise = rng.standard_normal(activity.shape)
        target = np.mean(activity**2, axis=(0, 2)) / 10.0 ** (float(snr_db) / 10.0)
        noise *= np.sqrt(target / np.mean(noise**2, axis=(0, 2)))[None, :, None]
        return TimeSeries(
            activity + noise,
            self.sources.sfreq,
            self.sources.channels,
            self.sources.segmentation,
            self.sources.tmin,
        )


def simulate(
    network: Network,
    *,
    sfreq: float,
    n_times: int,
    seed: int,
    n_trials: int = 1,
    dynamics: Mapping[str, NodeDynamics] | None = None,
    snr_db: float = 10.0,
    n_sensors: int | None = None,
    n_background: int = 10,
    background_fraction: float = 0.5,
    leadfield: np.ndarray | None = None,
    sensor_channels: Sequence[ChannelMeta] | None = None,
    node_scale: str = "none",
    extra_signal: np.ndarray | None = None,
    noise_series: np.ndarray | None = None,
    delay_rounding: str = "error",
    burn_in: int = DEFAULT_BURN_IN,
) -> SimulatedData:
    """Simulate sensor data generated by a directed network.

    The sensor SNR is defined as the mean power of the noise-free sensor signal divided by the
    mean power of the noise, over all sensors, trials and samples, in dB. The noise is made of
    background 1/f sources (carrying ``background_fraction`` of the noise power, before the
    joint rescaling) and white sensor noise, and is rescaled so that the realised SNR equals
    ``snr_db`` exactly. Direct channels bypass the leadfield; each gets white noise at
    ``snr_db`` relative to its own power.

    Parameters
    ----------
    network : Network
        Needs at least one cortical node.
    sfreq : float
        Sampling frequency in Hz.
    n_times : int
        Samples per trial.
    seed : int
        Random seed (required). Everything, including the default leadfield, derives from it.
    n_trials : int
        Number of independent trials; with more than one the data are ``"epoched"``.
    dynamics : mapping of str to NodeDynamics, optional
        Own dynamics per node name; default ``NodeDynamics()`` (10 Hz oscillator).
    snr_db : float
        Target sensor SNR in dB.
    n_sensors : int, optional
        Number of sensors for the default leadfield (default 16); must agree with ``leadfield``.
    n_background : int
        Number of background 1/f sources; 0 turns the background off (all noise is white).
    background_fraction : float
        Share of the noise power carried by the background sources, between 0 and 1.
    leadfield : ndarray, shape (n_sensors, n_cortical + n_background), optional
        Replaces the artificial leadfield.
    sensor_channels : sequence of ChannelMeta, optional
        Names and types of the sensors, one per row of the leadfield; default ``S01``, ``S02``, ...
        of type ``"eeg"``.
    node_scale : {"none", "unit"}
        ``"none"`` keeps the node powers the dynamics produce, which in a chain grow with every
        step. ``"unit"`` rescales every node to unit variance before mixing. Rescaling a node by a
        positive constant changes neither the structure of the network nor the truth, but the
        sensor SNR (defined over all sensors) is dominated by the strongest node and a weak node
        can be buried in noise; ``"unit"`` makes nodes comparable.
    noise_series : ndarray, shape (n_trials, n_sensors, n_times), optional
        A recorded noise (for example an empty-room measurement) used **instead of** the
        simulated noise; it is rescaled so that the realised SNR equals ``snr_db`` exactly, so
        only its shape in time and across sensors matters, not its size. It replaces the
        background sources and the white noise of the sensors, so ``n_background`` must be 0.
        The activity of the nodes does not depend on it: the same seed gives the same signal
        with any noise. Direct channels keep their own white noise.
    extra_signal : ndarray, shape (n_trials, n_sensors, n_times), optional
        Activity that reaches the sensors besides the nodes, for example the local activity of the
        regions that is unrelated to the network. It is added to the signal and counted in the
        SNR, so ``signal`` is the leadfield times the node activity plus this array.
    delay_rounding : {"error", "round"}
        See :meth:`Edge.delay_samples`.
    burn_in : int
        Samples discarded at the start so that the dynamics are stationary.

    Returns
    -------
    SimulatedData
    """
    sfreq = _check_sfreq(sfreq)
    if node_scale not in NODE_SCALES:
        raise ValueError(f"node_scale must be one of {NODE_SCALES}, got {node_scale!r}.")
    _check_count("n_times", n_times, 1)
    _check_count("n_trials", n_trials, 1)
    _check_count("n_background", n_background, 0)
    if noise_series is not None and n_background != 0:
        raise ValueError(
            "noise_series replaces the background sources and the white noise, so n_background "
            f"must be 0, got {n_background}."
        )
    _check_count("burn_in", burn_in, 0)
    if n_sensors is not None:
        _check_count("n_sensors", n_sensors, 1)
    if not math.isfinite(float(snr_db)):
        raise ValueError(f"snr_db must be finite, got {snr_db!r}.")
    if not 0.0 <= float(background_fraction) <= 1.0:
        raise ValueError(
            f"background_fraction must be between 0 and 1, got {background_fraction!r}."
        )
    if n_times < 2:
        raise ValueError("n_times must be at least 2.")

    cortical = [i for i, node in enumerate(network.nodes) if node.node_kind == "cortical"]
    direct = [i for i, node in enumerate(network.nodes) if node.node_kind == "direct"]
    if not cortical:
        raise ValueError("The network needs at least one cortical node to produce sensor data.")

    schedule = network_var_schedule(network, sfreq, n_times, dynamics, delay_rounding)
    resolved = _resolve_dynamics(network, dynamics)
    rng = np.random.default_rng(seed)

    n_columns = len(cortical) + n_background
    if leadfield is None:
        lead = artificial_leadfield(16 if n_sensors is None else n_sensors, n_columns, rng)
        leadfield_kind = "artificial"
    else:
        lead = np.asarray(leadfield, dtype=float)
        if lead.ndim != 2 or lead.shape[1] != n_columns or not np.isfinite(lead).all():
            raise ValueError(
                f"leadfield must be finite with {n_columns} columns "
                f"({len(cortical)} cortical nodes + {n_background} background sources), "
                f"got shape {lead.shape}."
            )
        if n_sensors is not None and n_sensors != lead.shape[0]:
            raise ValueError(
                f"n_sensors ({n_sensors}) disagrees with the {lead.shape[0]} rows of leadfield."
            )
        leadfield_kind = "custom"
    n_sens = lead.shape[0]
    if sensor_channels is not None:
        sensor_channels = tuple(sensor_channels)
        if len(sensor_channels) != n_sens:
            raise ValueError(
                f"sensor_channels has {len(sensor_channels)} entries for {n_sens} sensors."
            )

    # The first stretch also governs the burn-in; later ones start after it.
    starts = [0] + [burn_in + first for first, _ in schedule[1:]]
    activity = _simulate_nodes(
        [model for _, model in schedule], starts, n_trials, n_times + burn_in, rng
    )[:, :, burn_in:]
    activity = np.ascontiguousarray(activity)
    if node_scale == "unit":
        activity = activity / activity.std(axis=(0, 2), keepdims=True)
    clean = np.einsum("sk,bkt->bst", lead[:, : len(cortical)], activity[:, cortical])
    if extra_signal is not None:
        extra = np.asarray(extra_signal, dtype=float)
        if extra.shape != clean.shape or not np.isfinite(extra).all():
            raise ValueError(
                f"extra_signal must be finite with shape {clean.shape} "
                f"(trials, sensors, times), got {extra.shape}."
            )
        clean = clean + extra
    power_noise = float(np.mean(clean**2)) / 10.0 ** (float(snr_db) / 10.0)

    white = rng.standard_normal(clean.shape)  # drawn even if unused: the stream stays the same
    if noise_series is not None:
        given = np.asarray(noise_series, dtype=float)
        if given.shape != clean.shape or not np.isfinite(given).all():
            raise ValueError(
                f"noise_series must be finite with shape {clean.shape} "
                f"(trials, sensors, times), got {given.shape}."
            )
        if not np.any(given):
            raise ValueError("noise_series must not be all zero: it cannot be scaled to an SNR.")
        noise = given
    elif n_background > 0:
        background = np.einsum(
            "sk,bkt->bst",
            lead[:, len(cortical) :],
            _pink_noise(rng, (n_trials, n_background, n_times)),
        )
        fraction = float(background_fraction)
        noise = background * math.sqrt(fraction / np.mean(background**2)) + white * math.sqrt(
            (1.0 - fraction) / np.mean(white**2)
        )
    else:
        noise = white
    noise = noise * math.sqrt(power_noise / float(np.mean(noise**2)))

    sensor_data = clean + noise
    direct_data = []
    for i in direct:
        node_noise = rng.standard_normal((n_trials, n_times))
        target_power = float(np.mean(activity[:, i] ** 2)) / 10.0 ** (float(snr_db) / 10.0)
        node_noise *= math.sqrt(target_power / float(np.mean(node_noise**2)))
        direct_data.append(activity[:, i] + node_noise)
    full = (
        np.concatenate([sensor_data, np.stack(direct_data, axis=1)], axis=1)
        if direct
        else sensor_data
    )

    segmentation = "continuous" if n_trials == 1 else "epoched"
    if sensor_channels is None:
        sensor_channels = tuple(ChannelMeta(f"S{k + 1:02d}", type="eeg") for k in range(n_sens))
    data = TimeSeries.from_array(
        full,
        sfreq,
        channels=[*sensor_channels, *(network.nodes[i] for i in direct)],
        segmentation=segmentation,
    )
    sources = TimeSeries.from_array(
        activity, sfreq, channels=network.nodes, segmentation=segmentation
    )
    provenance = Provenance(
        method="simulate",
        parameters={
            "network": network.to_dict(),
            "dynamics": {name: dataclasses.asdict(d) for name, d in resolved.items()},
            "sfreq": sfreq,
            "n_times": n_times,
            "n_trials": n_trials,
            "snr_db": float(snr_db),
            "n_sensors": n_sens,
            "n_background": n_background,
            "noise_series": noise_series is not None,
            "background_fraction": float(background_fraction),
            "leadfield": leadfield_kind,
            "node_scale": node_scale,
            "extra_signal": extra_signal is not None,
            "delay_rounding": delay_rounding,
            "burn_in": burn_in,
        },
        seed=seed,
        backend="dirconn.simulation",
    )
    return SimulatedData(
        data=data,
        sources=sources,
        network=network,
        leadfield=_readonly(lead),
        signal=_readonly(clean),
        noise=_readonly(noise),
        provenance=provenance,
    )


def simulate_on_head(
    network: Network,
    head: HeadModel,
    *,
    sfreq: float,
    n_times: int,
    seed: int,
    n_trials: int = 1,
    dynamics: Mapping[str, NodeDynamics] | None = None,
    snr_db: float = 10.0,
    n_background: int = 10,
    background_fraction: float = 0.5,
    node_scale: str = "unit",
    source_model: str = "partial",
    coherence: float | None = None,
    noise_series: np.ndarray | None = None,
    delay_rounding: str = "error",
    burn_in: int = DEFAULT_BURN_IN,
) -> SimulatedData:
    """Simulate sensor data of a network through a realistic head model.

    Every cortical node must be named after a region of the head model (a DK label such as
    ``"inferiortemporal-lh"``) and enters the sensors through its region, according to
    ``source_model``. The background 1/f sources are drawn without replacement from the sources
    that are **not inside the regions of the network's nodes**, using a generator derived from
    ``seed`` (any other source of the brain, labelled or not, can be a background source). Direct
    nodes bypass the head model.
    The rest, including the exact SNR, is :func:`simulate`.

    Parameters
    ----------
    network : Network
        Cortical nodes named like the head model's regions.
    head : HeadModel
        See :func:`dirconn.headmodel.load_head_model`.
    source_model : {"partial", "focal", "patch"}
        How a region's activity reaches the sensors. ``"partial"`` (the default): every source of
        the region carries ``c * node + sqrt(1 - c**2) * its own 1/f activity`` with unit-variance
        parts, so the sources of a region are partly coherent (correlation ``c**2`` between two
        of them); the sensors see their mean. The coherent part enters as ``c`` times the region's
        mean topography and the local part as the sum of the independent activities, which is added
        to the signal and counted in the SNR. ``"focal"``: one source, the region's centre source
        (:attr:`HeadModel.roi_centre`), carries the node. ``"patch"``: every source carries exactly
        the node (coherence 1), so the sensors see the region's mean topography times the node.
    coherence : float, optional
        ``c`` of ``"partial"``, in ``(0, 1]``; default 0.7. Refused for the other models.
    noise_series : ndarray, shape (n_trials, n_sensors, n_times), optional
        A recorded noise of the head model's sensors, used instead of the simulated noise (see
        :func:`simulate`); needs ``n_background=0``.
    sfreq, n_times, seed, n_trials, dynamics, snr_db, n_background, background_fraction
        As in :func:`simulate`; ``seed`` is required.
    node_scale, delay_rounding, burn_in
        As in :func:`simulate`. ``node_scale`` defaults to ``"unit"`` here, unlike in
        :func:`simulate`: through a head model the node powers decide how much each node leaks
        into the sensors of the others, so they are equalised unless you say otherwise.

    Returns
    -------
    SimulatedData
        The sensors carry the head model's channel names and type. The provenance records the
        head model, the visibility of each node's region and the background sources used.
    """
    if not isinstance(head, HeadModel):
        raise TypeError(f"head must be a HeadModel, got {type(head).__name__}.")
    _check_count("n_background", n_background, 0)
    if source_model not in SOURCE_MODELS:
        raise ValueError(f"source_model must be one of {SOURCE_MODELS}, got {source_model!r}.")
    if source_model != "partial" and coherence is not None:
        raise ValueError(f"coherence is not used by source_model={source_model!r}; remove it.")
    level = 0.7 if coherence is None else float(coherence)
    if source_model == "partial" and not 0.0 < level <= 1.0:
        raise ValueError(f"coherence must lie in (0, 1], got {coherence!r}.")
    cortical = [node.name for node in network.nodes if node.node_kind == "cortical"]
    unknown = [name for name in cortical if name not in head.roi_names]
    if unknown:
        raise ValueError(
            f"Node(s) {unknown} are not regions of the head model; use DK labels such as "
            f"{list(head.roi_names[:3])}."
        )
    roi_index = [head.roi_names.index(name) for name in cortical]
    eligible = np.flatnonzero(~np.isin(head.source_roi, roi_index))
    if n_background > eligible.size:
        raise ValueError(
            f"n_background={n_background} exceeds the {eligible.size} sources outside the nodes' "
            "regions that can serve as background."
        )
    rng = np.random.default_rng([int(seed), 1])
    chosen = np.sort(rng.choice(eligible, size=n_background, replace=False))
    if source_model == "focal":
        node_columns = head.gain[:, head.roi_centre[roi_index]]
    elif source_model == "patch":
        node_columns = head.roi_leadfield[:, roi_index]
    else:
        node_columns = level * head.roi_leadfield[:, roi_index]
    leadfield = np.concatenate([node_columns, head.gain[:, chosen]], axis=1)
    local = None
    if source_model == "partial" and level < 1.0:
        local = _local_activity(head, roi_index, level, n_trials, n_times, seed)

    simulated = simulate(
        network,
        sfreq=sfreq,
        n_times=n_times,
        seed=seed,
        n_trials=n_trials,
        dynamics=dynamics,
        snr_db=snr_db,
        n_background=n_background,
        background_fraction=background_fraction,
        leadfield=leadfield,
        sensor_channels=[ChannelMeta(name, type=head.modality) for name in head.channel_names],
        node_scale=node_scale,
        extra_signal=local,
        noise_series=noise_series,
        delay_rounding=delay_rounding,
        burn_in=burn_in,
    )
    parameters = dict(simulated.provenance.parameters)
    parameters["leadfield"] = "head model"
    parameters["head_model"] = {
        "modality": head.modality,
        "n_channels": head.n_channels,
        "scale": float(head.scale),
        "node_visibility": {
            name: float(head.visibility[k]) for name, k in zip(cortical, roi_index, strict=True)
        },
        "background_sources": [int(i) for i in chosen],
        "source_model": source_model,
        "coherence": level if source_model == "partial" else None,
        **head.provenance,
    }
    return dataclasses.replace(
        simulated, provenance=dataclasses.replace(simulated.provenance, parameters=parameters)
    )


def _local_activity(
    head: HeadModel,
    roi_index: Sequence[int],
    coherence: float,
    n_trials: int,
    n_times: int,
    seed: int,
) -> np.ndarray:
    """Return what the sources of the nodes' regions do besides carrying the node.

    Every source has independent unit-variance 1/f activity, weighted by ``sqrt(1 - c**2)``; the
    sensors see the mean over the sources of a region, as for the node itself.
    """
    rng = np.random.default_rng([int(seed), 2])
    weight = float(np.sqrt(1.0 - coherence**2))
    out = np.zeros((n_trials, head.n_channels, n_times))
    for k in roi_index:
        sources = np.flatnonzero(head.source_roi == k)
        activity = _pink_noise(rng, (n_trials, sources.size, n_times))
        out += weight * np.einsum("sv,bvt->bst", head.gain[:, sources] / sources.size, activity)
    return out


def _check_count(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}, got {value!r}.")


NONLINEAR_COUPLINGS: tuple[str, ...] = ("quadratic", "linear")
"""Coupling functions of :func:`coupled_nonlinear`."""


def coupled_nonlinear(
    n_times: int,
    *,
    sfreq: float,
    seed: int,
    coupling: str = "quadratic",
    strength: float = 0.5,
    delay: int = 2,
    n_trials: int = 1,
) -> tuple[TimeSeries, Connectivity]:
    """Three channels where X drives Y through a chosen function and Z is independent.

    ``X_t = 0.6 X_{t-1} + e``, ``Y_t = 0.4 Y_{t-1} + strength * g(X_{t-delay}) + e``,
    ``Z_t = 0.5 Z_{t-1} + e``, with independent unit-variance Gaussian noise ``e``. The coupling
    ``g`` has zero mean and unit variance: ``"quadratic"`` is ``(x^2 - s^2) / (sqrt(2) s^2)`` with
    ``s^2`` the stationary variance of X, whose linear correlation with ``x`` is zero, so a linear
    method sees no influence; ``"linear"`` is ``x / s``. This is a small test bed for methods
    that claim to see non-linear dependence, not a model of brain signals; it is not part of the
    scenario library or the head-model pipeline.

    Parameters
    ----------
    n_times : int
        Samples per trial (after a burn-in of 100 samples).
    sfreq : float
        Sampling frequency in Hz (the model is in samples; ``sfreq`` only labels the time axis).
    seed : int
        Seed of the random numbers; required.
    coupling : {"quadratic", "linear"}
    strength : float
        Weight of ``g`` in Y.
    delay : int
        Lag of the coupling in samples (at least 1).
    n_trials : int
        Independent trials; one trial gives continuous data, more give epoched data.

    Returns
    -------
    data : TimeSeries
        Channels ``X``, ``Y``, ``Z``.
    truth : Connectivity
        ``1`` for X -> Y, ``0`` for the other pairs, NaN on the diagonal.
    """
    if coupling not in NONLINEAR_COUPLINGS:
        raise ValueError(f"coupling must be one of {NONLINEAR_COUPLINGS}, got {coupling!r}.")
    _check_count("n_times", n_times, 2)
    _check_count("delay", delay, 1)
    _check_count("n_trials", n_trials, 1)
    if not math.isfinite(float(strength)):
        raise ValueError(f"strength must be finite, got {strength!r}.")
    burn = 100
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal((int(n_trials), 3, burn + int(n_times)))
    x = np.zeros_like(noise)
    variance = 1.0 / (1.0 - 0.6**2)
    for t in range(1, x.shape[-1]):
        x[:, 0, t] = 0.6 * x[:, 0, t - 1] + noise[:, 0, t]
        x[:, 2, t] = 0.5 * x[:, 2, t - 1] + noise[:, 2, t]
        drive = x[:, 0, t - delay] if t >= delay else np.zeros(x.shape[0])
        if coupling == "quadratic":
            g = (drive**2 - variance) / (math.sqrt(2.0) * variance)
        else:
            g = drive / math.sqrt(variance)
        x[:, 1, t] = 0.4 * x[:, 1, t - 1] + float(strength) * g + noise[:, 1, t]
    x = x[:, :, burn:]
    segmentation = "continuous" if n_trials == 1 else "epoched"
    data = TimeSeries.from_array(x, sfreq, names=["X", "Y", "Z"], segmentation=segmentation)
    truth_values = np.zeros((3, 3))
    truth_values[0, 1] = 1.0
    np.fill_diagonal(truth_values, np.nan)
    provenance = Provenance(
        method="coupled_nonlinear",
        parameters={
            "n_times": int(n_times),
            "sfreq": float(sfreq),
            "coupling": coupling,
            "strength": float(strength),
            "delay": int(delay),
            "n_trials": int(n_trials),
        },
        seed=int(seed),
    )
    truth = Connectivity.from_array(truth_values, source=["X", "Y", "Z"], provenance=provenance)
    return data, truth
