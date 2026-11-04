"""Evaluation of an estimator on simulated data at ROI level and at sensor level.

Stage 1 gives the estimator the activity of the network's nodes, observed with independent white
noise: the best case, without any spatial mixing. Stage 2 gives it sensor series generated from
the same nodes through a head model. A cortical node is represented by one sensor, so that the
truth keeps its meaning; the sensor is the one where the node's region is strongest (a distinct
sensor per node), and the share of the region's field and of the sensor's signal that this accounts
for is reported, because it is small: a single sensor sees many regions. Stage 2 is therefore the
best case *for the sensor level*, not a typical sensor-space analysis. Direct nodes (for example an
STN channel) are used as recorded at both stages.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from dirconn.datasets import require_sample
from dirconn.estimator import Estimator
from dirconn.evaluation import (
    auroc_by_frequency,
    convention_scores,
    score_result,
    scoring_band,
)
from dirconn.exceptions import DirconnWarning
from dirconn.headmodel import HeadModel
from dirconn.inverse import roi_series
from dirconn.network import Network
from dirconn.result import Connectivity
from dirconn.scenario import ArtifactSpec, stimulation_artifact
from dirconn.simulation import SimulatedData
from dirconn.timeseries import TimeSeries


@dataclass(frozen=True, eq=False)
class RepresentativeSensors:
    """One sensor per cortical node.

    Attributes
    ----------
    node_names, sensor_names : tuple of str
        The cortical nodes in network order and the sensor chosen for each.
    sensor_index : tuple of int
        Row of each chosen sensor in the head model.
    dominance : ndarray
        Share of the node's region topography energy that lies on its sensor (``1 / n_channels``
        would be an even spread).
    purity : ndarray
        Share of the signal energy that the network's regions put on the sensor that comes from
        its own node; 1 means the sensor sees only its node. The energy of a region is its
        topography energy times the power of its node, so a weak node next to a strong one is
        impure even when the topographies are separate.
    """

    node_names: tuple[str, ...]
    sensor_names: tuple[str, ...]
    sensor_index: tuple[int, ...]
    dominance: np.ndarray
    purity: np.ndarray


def representative_sensors(
    head: HeadModel, network: Network, node_power: Sequence[float] | np.ndarray | None = None
) -> RepresentativeSensors:
    """Choose a distinct sensor for every cortical node of ``network``.

    The sensors are assigned so that the total dominance is largest (a Hungarian assignment), which
    gives distinct sensors even when two regions peak on the same one.

    Parameters
    ----------
    head : HeadModel
    network : Network
    node_power : sequence of float, optional
        Power (variance) of each cortical node, in network order, used to weight the purity. By
        default every node has the same power, which describes the leadfield alone and overstates
        the purity whenever node powers differ.
    """
    if not isinstance(head, HeadModel):
        raise TypeError(f"head must be a HeadModel, got {type(head).__name__}.")
    nodes = [node.name for node in network.nodes if node.node_kind == "cortical"]
    if not nodes:
        raise ValueError(
            "The network has no cortical node, so there is nothing to pick sensors for."
        )
    unknown = [name for name in nodes if name not in head.roi_names]
    if unknown:
        raise ValueError(f"Node(s) {unknown} are not regions of the head model.")
    if len(nodes) > head.n_channels:
        raise ValueError(
            f"{len(nodes)} cortical nodes need distinct sensors but the head model has only "
            f"{head.n_channels} sensors."
        )
    power = np.ones(len(nodes)) if node_power is None else np.asarray(node_power, dtype=float)
    if power.shape != (len(nodes),) or not (power > 0).all():
        raise ValueError(
            f"node_power needs one positive value per cortical node ({len(nodes)}), got {power}."
        )
    energy = head.roi_leadfield[:, [head.roi_names.index(n) for n in nodes]] ** 2
    share = energy / energy.sum(axis=0, keepdims=True)  # (sensors, nodes)
    node_rows, sensors = linear_sum_assignment(-share.T)
    order = np.argsort(node_rows)
    sensors = sensors[order]
    k = np.arange(len(nodes))
    return RepresentativeSensors(
        node_names=tuple(nodes),
        sensor_names=tuple(head.channel_names[s] for s in sensors),
        sensor_index=tuple(int(s) for s in sensors),
        dominance=share[sensors, k],
        purity=(energy[sensors, k] * power) / (energy[sensors] * power[None, :]).sum(axis=1),
    )


def roi_observations(sim: SimulatedData, *, snr_db: float, seed: int) -> TimeSeries:
    """Return the stage-1 data: every node's activity plus independent white noise.

    The noise of each node is scaled so that its SNR, relative to the node's own power, equals
    ``snr_db`` exactly. This is not the sensor SNR of the simulation (which is a mean over
    sensors); it only says how well the node series are observed. The implementation is
    :meth:`dirconn.simulation.SimulatedData.roi_observations`.
    """
    return sim.roi_observations(snr_db=snr_db, seed=seed)


def sensor_series(sim: SimulatedData, rep: RepresentativeSensors) -> TimeSeries:
    """Return the stage-2 data: the representative sensors and direct channels, in node order."""
    names = []
    for node in sim.network.nodes:
        if node.node_kind == "cortical":
            names.append(rep.sensor_names[rep.node_names.index(node.name)])
        else:
            names.append(node.name)
    return sim.data.pick(names)


def source_series(
    sim: SimulatedData,
    head: HeadModel,
    *,
    method: str = "mne",
    reg: float | None = None,
    mode: str = "pca_flip",
) -> TimeSeries:
    """Return the stage-3 data: region series estimated from the sensors, in network order.

    The sensors of ``head`` go through the inverse method (:func:`dirconn.inverse.roi_series`) for
    every cortical node; direct nodes (for example an STN channel) are used as recorded.
    """
    sensors = sim.data.pick(list(head.channel_names))
    cortical = [node.name for node in sim.network.nodes if node.node_kind == "cortical"]
    estimated = roi_series(sensors, head, cortical, method=method, reg=reg, mode=mode)
    columns = []
    channels = []
    for node in sim.network.nodes:
        if node.node_kind == "cortical":
            k = estimated.names.index(node.name)
            columns.append(estimated.data[:, k])
            channels.append(estimated.channels[k])
        else:
            k = sim.data.names.index(node.name)
            columns.append(sim.data.data[:, k])
            channels.append(sim.data.channels[k])
    return TimeSeries(
        np.stack(columns, axis=1),
        sim.data.sfreq,
        tuple(channels),
        sim.data.segmentation,
        sim.data.tmin,
    )


def add_artifact(sim: SimulatedData, head: HeadModel, spec: ArtifactSpec) -> SimulatedData:
    """Return the simulation with a stimulation artifact added to its sensor data.

    The artifact (:func:`dirconn.scenario.stimulation_artifact`) goes onto the sensors where the
    topography of ``spec.anchor`` is strongest, with that topography's signs, at ``spec.amplitude``
    times the power of the noise-free signal there. Region-level data (``sources``) and the noise
    and signal arrays, hence ``snr_db``, are untouched: the artifact is a separate contamination of
    the sensors, recorded in the provenance. Stage 1 of :func:`evaluate_stages` does not see it.
    """
    if spec.anchor not in head.roi_names:
        raise ValueError(
            f"The artifact's anchor {spec.anchor!r} is not a region of the head model."
        )
    topography = head.roi_leadfield[:, head.roi_names.index(spec.anchor)]
    artifact = stimulation_artifact(spec, topography, sim.signal, sim.data.sfreq)
    data = np.array(sim.data.data)
    data[:, : artifact.shape[1]] += artifact  # the sensors come first, then the direct channels
    contaminated = TimeSeries(
        data, sim.data.sfreq, sim.data.channels, sim.data.segmentation, sim.data.tmin
    )
    parameters = dict(sim.provenance.parameters)
    parameters["artifact"] = dataclasses.asdict(spec)
    provenance = dataclasses.replace(sim.provenance, parameters=parameters)
    return dataclasses.replace(sim, data=contaminated, provenance=provenance)


_INVERSE_KEYS = ("method", "reg", "mode")


def _fit(estimator: Estimator, series: TimeSeries) -> tuple[Connectivity, list[str]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DirconnWarning)
        result = estimator.fit(series)
    flagged = sorted(
        c.name for c in result.provenance.assumptions.checks if c.status in ("warn", "fail")
    )
    return result, flagged


def evaluate_stages(
    estimator: Estimator,
    sim: SimulatedData,
    head: HeadModel,
    *,
    seed: int,
    band: tuple[float, float] | str | None = None,
    roi_snr_db: float | None = None,
    inverse: Mapping[str, Any] | None = None,
    hidden: Sequence[str] = (),
    by_frequency: bool = False,
    default_band: tuple[float, float] | None = None,
    store_scores: bool = False,
) -> dict[str, Any]:
    """Score an estimator at ROI level (stage 1), sensor level (stage 2) and source level (stage 3).

    Parameters
    ----------
    estimator : Estimator
        Applied to both stages with the same settings.
    sim : SimulatedData
        Simulation produced with ``head`` (:func:`dirconn.simulation.simulate_on_head`).
    head : HeadModel
        The head model of the simulation, used to choose the representative sensors.
    seed : int
        Seed of the observation noise of stage 1.
    band : (float, float) or str, optional
        Band for scoring a spectral result; defaults to the band of the network's banded edges
        (:func:`~dirconn.evaluation.scoring_band`), which a broadband network does not define.
    roi_snr_db : float, optional
        Stage-1 observation SNR per node; defaults to the SNR of the simulation.
    inverse : mapping, optional
        Switches stage 3 on. Keys ``method``, ``reg`` and ``mode`` of :func:`source_series`; ``{}``
        means the defaults (minimum norm, ``pca_flip``). Without it only stages 1 and 2 are run.
    hidden : sequence of str
        Nodes that are simulated but hidden from the estimator. They take part in the choice of
        sensors and in the inverse step (they are in the head and leak into the others) but their
        series are not given to the estimator and the truth is the subgraph between the remaining
        nodes, so an edge that a hidden common cause creates counts as a false positive.
    by_frequency : bool
        Add the AUROC at every frequency to the card of a spectral estimator
        (``auroc_by_frequency``: ``freqs`` and ``auroc`` rounded to four decimals, NaN where the
        truth has no edge or only edges).
    default_band : (float, float), optional
        Scoring band for a spectral estimator when ``band`` is not given and the observed network
        has no banded edge (which would define one); without it that case is an error.
    store_scores : bool
        Add ``scores`` to every stage card: the score matrix of the analysed nodes that the
        agreement and the metrics are computed from (``[source, target]``, band-reduced for a
        spectral estimator, NaN diagonal), rounded to six decimals. It is what a study of single
        cells, such as the false edges of a leakage scenario, needs.

    Returns
    -------
    dict
        ``roi``, ``sensor`` and, with ``inverse``, ``source``: the score card of
        :func:`~dirconn.evaluation.score_result` plus the selected ``order`` and the names of the
        diagnostics that warned; ``agreement``: the Spearman correlation of the off-diagonal scores
        of stages 1 and 2 (``spearman``) and, with stage 3, of stages 1 and 3
        (``spearman_roi_source``); ``representative``: the chosen sensors with their dominance and
        purity; ``inverse``: the settings of stage 3; ``band`` and ``roi_snr_db``; ``analysed`` and
        ``hidden``: the node names the estimator saw and the ones it did not.
    """
    if not isinstance(estimator, Estimator):
        raise TypeError(f"estimator must be an Estimator, got {type(estimator).__name__}.")
    if not isinstance(head, HeadModel):
        raise TypeError(f"head must be a HeadModel, got {type(head).__name__}.")
    settings: dict[str, Any] | None = None
    if inverse is not None:
        if not isinstance(inverse, Mapping):
            raise TypeError(f"inverse must be a mapping of settings or None, got {inverse!r}.")
        unknown = sorted(set(inverse) - set(_INVERSE_KEYS))
        if unknown:
            raise ValueError(f"unknown inverse setting(s) {unknown}; use {list(_INVERSE_KEYS)}.")
        settings = {
            "method": inverse.get("method", "mne"),
            "reg": inverse.get("reg"),
            "mode": inverse.get("mode", "pca_flip"),
        }
    snr = float(sim.provenance.parameters["snr_db"]) if roi_snr_db is None else float(roi_snr_db)
    cortical = [i for i, node in enumerate(sim.network.nodes) if node.node_kind == "cortical"]
    power = np.var(sim.sources.data, axis=(0, 2))[cortical]
    rep = representative_sensors(head, sim.network, node_power=power)
    series = {
        "roi": roi_observations(sim, snr_db=snr, seed=seed),
        "sensor": sensor_series(sim, rep),
    }
    if settings is not None:
        series["source"] = source_series(sim, head, **settings)

    names = list(sim.network.names)
    hidden = list(hidden)
    unknown = sorted(set(hidden) - set(names))
    if unknown:
        raise ValueError(f"hidden names nodes that are not in the network: {unknown}.")
    analysed = [n for n in names if n not in set(hidden)]
    if len(analysed) < 2:
        raise ValueError("hidden leaves fewer than two analysed nodes.")
    observed = sim.network
    if hidden:
        # every series is in network order, with the node name or the sensor chosen for it
        keep = [i for i, n in enumerate(names) if n in set(analysed)]
        series = {stage: ts.pick([ts.names[i] for i in keep]) for stage, ts in series.items()}
        observed = Network(
            nodes=tuple(node for node in sim.network.nodes if node.name in set(analysed)),
            edges=tuple(
                e
                for e in sim.network.edges
                if e.source in set(analysed) and e.target in set(analysed)
            ),
            name=sim.network.name,
        )

    results: dict[str, Connectivity] = {}
    cards: dict[str, dict[str, Any]] = {}
    used_band: Any = None
    for stage, data in series.items():
        result, flagged = _fit(estimator, data)
        results[stage] = result
        if result.is_frequency_resolved:
            used_band = scoring_band(observed, default_band) if band is None else band
            truth: Any = observed.truth(freqs=result.freqs)
        else:
            used_band = None
            truth = observed.truth()
        card = score_result(result, truth, band=used_band, info=estimator.info)
        cards[stage] = {
            **card,
            "order": result.provenance.model_order,
            "diagnostics": flagged,
        }
        if by_frequency and result.is_frequency_resolved:
            freqs, curve = auroc_by_frequency(result, truth)
            cards[stage]["auroc_by_frequency"] = {
                "freqs": [float(f) for f in freqs],
                "auroc": np.round(curve, 4).tolist(),
            }

    scores = {
        stage: convention_scores(results[stage], estimator.info, used_band) for stage in results
    }
    if store_scores:
        for stage, matrix in scores.items():
            cards[stage]["scores"] = np.round(matrix, 6).tolist()
    off = ~np.eye(scores["roi"].shape[0], dtype=bool)

    def agreement(first: str, second: str) -> float:
        a, b = scores[first][off], scores[second][off]
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            return float("nan")
        return float(spearmanr(a, b).statistic)

    out: dict[str, Any] = {
        "roi": cards["roi"],
        "sensor": cards["sensor"],
        "agreement": {"spearman": agreement("roi", "sensor"), "n_pairs": int(off.sum())},
        "representative": {
            "nodes": list(rep.node_names),
            "sensors": list(rep.sensor_names),
            "dominance": [float(v) for v in rep.dominance],
            "purity": [float(v) for v in rep.purity],
            "node_power": [float(v) for v in power],
        },
        "band": cards["roi"]["band"],
        "roi_snr_db": snr,
        "analysed": analysed,
        "hidden": hidden,
    }
    if settings is not None:
        out["source"] = cards["source"]
        out["inverse"] = settings
        out["agreement"]["spearman_roi_source"] = agreement("roi", "source")
    return out


def delayed_bandpass(
    x: np.ndarray, *, sfreq: float, band: tuple[float, float], delay_samples: int
) -> np.ndarray:
    """Return ``x`` band-passed by a causal filter and delayed by ``delay_samples``.

    The filter is a 4th-order Butterworth band-pass run forward only, so the output at sample
    ``t`` depends on ``x`` up to ``t - delay_samples`` and the first ``delay_samples`` samples are
    zero.
    """
    low, high = band
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    filtered = signal.sosfilt(sos, np.asarray(x, dtype=float))
    out = np.zeros_like(filtered)
    out[delay_samples:] = filtered[: filtered.size - delay_samples]
    return out


def _standardise(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def constructed_pair(
    real: TimeSeries,
    *,
    source_channel: str,
    noise_channel: str,
    source_start: float,
    noise_start: float,
    duration: float,
    delay_ms: float,
    strength: float = 1.0,
    band: tuple[float, float] = (8.0, 12.0),
) -> tuple[TimeSeries, np.ndarray]:
    """Build a two-channel series with a known direction from two real channels.

    ``source`` is a window of a real channel, standardised. ``target`` is ``strength`` times the
    causal band-passed and delayed (:func:`delayed_bandpass`) source, standardised, plus a window of
    a second real channel, standardised, which is the real background. The noise window is taken
    from a different, non-overlapping time than the source window, so the noise is independent of
    the source and the truth is exact: ``source -> target`` exists, the reverse does not. Real
    channels recorded at the same time are not independent (they share sources and may influence
    each other), which would make the reverse direction unknowable.

    Parameters
    ----------
    real : TimeSeries
        A continuous recording.
    source_channel, noise_channel : str
        Two different channels of ``real``.
    source_start, noise_start : float
        Start of each window in seconds; the windows must not overlap.
    duration : float
        Length of both windows in seconds.
    delay_ms : float
        Delay of the influence; a positive whole number of samples.
    strength : float
        Amplitude of the influence relative to the noise (both have unit variance), so the power
        ratio is ``strength ** 2``.
    band : (float, float)
        Band of the influence in Hz, between 0 and the Nyquist frequency.

    Returns
    -------
    series : TimeSeries
        Channels ``"source"`` and ``"target"``.
    truth : ndarray
        ``[[nan, 1], [0, nan]]``.
    """
    if real.segmentation != "continuous":
        raise ValueError("constructed_pair needs a continuous recording.")
    for name in (source_channel, noise_channel):
        if name not in real.names:
            raise ValueError(f"Channel {name!r} is not in the recording.")
    if source_channel == noise_channel:
        raise ValueError("The source and noise channels must differ.")
    strength = float(strength)
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError(f"strength must be positive and finite, got {strength!r}.")
    low, high = float(band[0]), float(band[1])
    if not 0 < low < high < real.sfreq / 2:
        raise ValueError(
            f"band must satisfy 0 < low < high < Nyquist ({real.sfreq / 2:g} Hz), got {band!r}."
        )
    delay = float(delay_ms) * real.sfreq / 1000.0
    if delay < 1:
        raise ValueError(f"delay_ms must be at least one sample ({1000.0 / real.sfreq:g} ms).")
    if abs(delay - round(delay)) > 1e-9:
        raise ValueError(
            f"delay_ms must be a whole number of samples ({1000.0 / real.sfreq:g} ms each), "
            f"got {delay_ms!r}."
        )
    n = int(round(float(duration) * real.sfreq))
    starts = [int(round(float(t) * real.sfreq)) for t in (source_start, noise_start)]
    if min(starts) < 0 or max(starts) + n > real.n_times or n < 2:
        raise ValueError("A window reaches outside the recording.")
    if abs(starts[0] - starts[1]) < n:
        raise ValueError(
            "The source and noise windows overlap; use non-overlapping time windows so that the "
            "noise is independent of the source."
        )

    source = real.data[0, real.names.index(source_channel), starts[0] : starts[0] + n]
    noise = real.data[0, real.names.index(noise_channel), starts[1] : starts[1] + n]
    a = _standardise(source)
    influence = _standardise(
        delayed_bandpass(a, sfreq=real.sfreq, band=(low, high), delay_samples=int(round(delay)))
    )
    target = strength * influence + _standardise(noise)
    series = TimeSeries.from_array(np.stack([a, target]), real.sfreq, names=["source", "target"])
    return series, np.array([[np.nan, 1.0], [0.0, np.nan]])


def evaluate_pair(
    estimator: Estimator,
    series: TimeSeries,
    truth: np.ndarray,
    *,
    band: tuple[float, float] | str | None = None,
) -> dict[str, Any]:
    """Score an estimator on a constructed pair.

    Returns the :func:`~dirconn.evaluation.score_result` card plus ``scores`` (the band-reduced
    ``forward`` and ``reverse`` scores), the selected ``order`` and the names of the diagnostics
    that warned. A spectral estimator needs ``band``, normally the band of the construction.
    """
    result, flagged = _fit(estimator, series)
    card = score_result(result, truth, band=band, info=estimator.info)
    scores = convention_scores(result, estimator.info, band)
    return {
        **card,
        "scores": {"forward": float(scores[0, 1]), "reverse": float(scores[1, 0])},
        "order": result.provenance.model_order,
        "diagnostics": flagged,
    }


@cache
def _real_eeg(sfreq: float, l_freq: float, h_freq: float, root: str) -> TimeSeries:
    import mne

    raw = mne.io.read_raw_fif(
        Path(root) / "MEG" / "sample" / "sample_audvis_raw.fif", preload=True, verbose="ERROR"
    )
    names = [raw.ch_names[i] for i in mne.pick_types(raw.info, meg=False, eeg=True, exclude="bads")]
    raw.pick(names)
    raw.filter(l_freq, h_freq, verbose="ERROR")
    raw.resample(sfreq, verbose="ERROR")
    return TimeSeries.from_mne(raw, picks=names)


def real_eeg_series(
    *,
    sfreq: float = 150.0,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    path: str | Path | None = None,
) -> TimeSeries:
    """Return the good EEG channels of the MNE sample recording as a continuous series.

    The recording is band-pass filtered and resampled (zero-phase filter, applied once to the whole
    recording) and cached. It is a task recording (auditory and visual stimulation), not resting
    state, so it contains evoked responses and real connectivity; use it as realistic background,
    not as a connectivity-free baseline.
    """
    root = require_sample(path=path)
    return _real_eeg(float(sfreq), float(l_freq), float(h_freq), str(root))
