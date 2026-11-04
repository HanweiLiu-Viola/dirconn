"""Small estimators and datasets shared by the statistics tests."""

from dataclasses import dataclass, field

import numpy as np

from dirconn import Connectivity, Edge, Network, TimeSeries, dk_node
from dirconn.estimator import Estimator, MethodInfo
from dirconn.headmodel import HeadModel


@dataclass(frozen=True)
class AbsCorr(Estimator):
    """Symmetric, undirected stand-in: absolute correlation, NaN diagonal."""

    info = MethodInfo(name="abscorr", backend="test stub", directed=False)

    def _fit(self, data: TimeSeries) -> Connectivity:
        flat = np.moveaxis(data.data, 1, 0).reshape(data.n_channels, -1)
        values = np.abs(np.corrcoef(flat))
        np.fill_diagonal(values, np.nan)
        return Connectivity.from_array(values, source=data.names, provenance=self._provenance(data))


def coupled(n_trials=4, n_times=300, seed=0, strength=0.8, n_channels=3):
    """Channel 0 drives channel 1; the other channels are independent noise."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_trials, n_channels, n_times))
    x[:, 1] = strength * x[:, 0] + np.sqrt(1 - strength**2) * x[:, 1]
    return TimeSeries.from_array(x, 100.0, segmentation="epoched")


def independent(**kwargs):
    return coupled(strength=0.0, **kwargs)


@dataclass(frozen=True)
class MeanDifference(Estimator):
    """Cheap directed stand-in: ``values[i, j] = mean(ch_i) - mean(ch_j)``, NaN diagonal."""

    info = MethodInfo(name="meandiff", backend="test stub", directed=True)

    def _fit(self, data: TimeSeries) -> Connectivity:
        means = data.data.mean(axis=(0, 2))
        values = means[:, None] - means[None, :]
        np.fill_diagonal(values, np.nan)
        return Connectivity.from_array(values, source=data.names, provenance=self._provenance(data))


def epochs(n_trials=30, n_times=50, means=(0.0, 1.0, 3.0), seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_trials, len(means), n_times)) + np.asarray(means)[None, :, None]
    return TimeSeries.from_array(x, 100.0, segmentation="epoched")


@dataclass(frozen=True)
class Spy(Estimator):
    """Records every dataset it is fitted on."""

    info = MethodInfo(name="spy", backend="test stub", directed=False)
    seen: list = field(default_factory=list, init=False, compare=False)

    def _fit(self, data: TimeSeries) -> Connectivity:
        self.seen.append(data)
        values = np.ones((data.n_channels, data.n_channels))
        np.fill_diagonal(values, np.nan)
        return Connectivity.from_array(values, source=data.names, provenance=self._provenance(data))


ROIS = ("bankssts-lh", "insula-rh", "precuneus-lh", "caudalmiddlefrontal-rh", "lingual-lh")


def head(modality="eeg", n_channels=12, n_sources=80, seed=0, topographies=None):
    """A small made-up head model: five regions of eight sources each.

    By default the gains are random. ``topographies`` (n_channels x 5) fixes the topography of
    each region instead: its sources get that topography plus perturbations that average to zero.
    """
    rng = np.random.default_rng(seed)
    gain = rng.standard_normal((n_channels, n_sources))
    source_roi = np.full(n_sources, -1)
    for k in range(len(ROIS)):
        source_roi[k * 8 : (k + 1) * 8] = k
        if topographies is not None:
            spread = rng.standard_normal((n_channels, 8)) * 0.5
            gain[:, k * 8 : (k + 1) * 8] = (
                np.asarray(topographies)[:, [k]] + spread - spread.mean(axis=1, keepdims=True)
            )
    roi_leadfield = np.stack([gain[:, source_roi == k].mean(axis=1) for k in range(len(ROIS))], 1)
    norms = np.linalg.norm(roi_leadfield, axis=0)
    names = tuple(f"{modality.upper()}{i:02d}" for i in range(n_channels))
    return HeadModel(
        modality=modality,
        channel_names=names,
        roi_names=ROIS,
        roi_leadfield=roi_leadfield,
        scale=1.0,
        visibility=norms / np.median(norms),
        gain=gain,
        source_roi=source_roi,
        source_flip=np.where(source_roi >= 0, 1, 0),
        roi_centre=np.array([int(np.flatnonzero(source_roi == k)[0]) for k in range(len(ROIS))]),
        forward=None,
        info=None,
        provenance={"forward_file": "made up"},
    )


def chain(*names):
    nodes = [dk_node(n) for n in names]
    edges = [
        Edge(a, b, strength=0.4, delay_ms=10.0) for a, b in zip(names[:-1], names[1:], strict=True)
    ]
    return Network(nodes=nodes, edges=edges)


SFREQ = 150.0


def fake_recording(n_channels=6, seconds=300, seed=0):
    """Stands in for a real recording: coloured noise, different in every channel."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SFREQ)
    white = rng.standard_normal((n_channels, n))
    spectrum = np.fft.rfft(white, axis=1)
    freqs = np.fft.rfftfreq(n, 1 / SFREQ)
    amplitude = 1.0 / np.sqrt(np.maximum(freqs, 1.0))
    coloured = np.fft.irfft(spectrum * amplitude, n=n, axis=1)
    return TimeSeries.from_array(coloured * 1e-5, SFREQ, names=[f"E{i}" for i in range(n_channels)])
