"""Imaginary coherence and phase slope index: lag-based measures from multitaper cross-spectra.

Both are built from the coherency ``C_ij(f) = S_ij(f) / sqrt(S_ii(f) S_jj(f))`` of the shared
multitaper cross-spectra (:mod:`dirconn._csd`), with ``S_ij = <X_i conj(X_j)>``. An instantaneous
(zero-lag) mixture of independent sources, the way volume conduction and source leakage act, makes
coherency real, so both measures are near zero for it; a coupling with a delay gives an imaginary
part. They are controls for leakage, not models of the interaction: neither conditions on other
channels, and a zero-lag mixture of *coupled* sources still distorts them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from dirconn._csd import DEFAULT_TIME_HALFBANDWIDTH, CrossSpectra, cross_spectra
from dirconn.bands import BANDS
from dirconn.estimator import Estimator, MethodInfo
from dirconn.result import Connectivity
from dirconn.timeseries import TimeSeries

_BACKEND = "own coherency from multitaper cross-spectra (mne.time_frequency.csd_array_multitaper)"
_ASSUMPTIONS = ("stationarity", "linearity", "filtering")


def coherency(spectra: CrossSpectra) -> np.ndarray:
    """Return the complex coherency ``S_ij / sqrt(S_ii S_jj)``, shape (n, n, n_freqs)."""
    power = np.einsum("iif->if", spectra.values).real
    return spectra.values / np.sqrt(power[:, None, :] * power[None, :, :])


@dataclass(frozen=True)
class _CrossSpectral(Estimator):
    """Shared parameters and the cross-spectra step of the coherency-based measures."""

    segment_length: float | None = None
    time_halfbandwidth: float = DEFAULT_TIME_HALFBANDWIDTH

    def _spectra(self, data: TimeSeries, fmin: float, fmax: float | None) -> CrossSpectra:
        return cross_spectra(
            data,
            segment_length=self.segment_length,
            time_halfbandwidth=self.time_halfbandwidth,
            fmin=fmin,
            fmax=fmax,
        )


@dataclass(frozen=True)
class ImCoh(_CrossSpectral):
    """Imaginary part of coherency (Nolte et al. 2004), per frequency.

    ``values[i, j, f] = Im(C_ij(f))``, between -1 and 1. It is antisymmetric
    (``values[j, i] = -values[i, j]``), and positive when ``j`` lags ``i`` at that frequency
    (phase between 0 and pi). The sign is a phase, not a direction of influence: it flips every
    half cycle of delay, so the measure is undirected (``info.directed`` is ``False``); score its
    magnitude. The diagonal is undefined (NaN).

    Parameters
    ----------
    segment_length : float, optional
        Segment length in seconds; required for continuous data, optional for epoched data (every
        trial is then cut on its own, otherwise every trial is one segment). It sets the
        frequency resolution, ``1 / segment_length`` Hz, so it has no default.
    time_halfbandwidth : float
        Time-half-bandwidth product ``NW`` of the tapers (at least 1).
    fmin, fmax : float
        Frequency range in Hz, inclusive; ``fmax`` defaults to the Nyquist frequency. The
        frequencies are the native bins of the segment length, so there is no ``n_freqs``.
    """

    info = MethodInfo(
        name="imcoh",
        backend=_BACKEND,
        directed=False,
        antisymmetric=True,
        min_nodes=2,
        value_range=(-1.0, 1.0),
        scale_dependent=False,
        frequency_resolved=True,
        assumptions=_ASSUMPTIONS,
    )

    fmin: float = 0.0
    fmax: float | None = None

    def _fit(self, data: TimeSeries) -> Connectivity:
        spectra = self._spectra(data, self.fmin, self.fmax)
        values = coherency(spectra).imag
        index = np.arange(data.n_channels)
        values[index, index] = np.nan
        provenance = self._provenance(data, details={"cross_spectra": spectra.record})
        return Connectivity.from_array(
            values, source=list(data.names), freqs=spectra.freqs, provenance=provenance
        )


def _band(band: Any) -> tuple[float, float]:
    if isinstance(band, str):
        key = band.lower()
        if key not in BANDS:
            raise ValueError(f"Unknown band {band!r}; the named bands are {sorted(BANDS)}.")
        return BANDS[key]
    try:
        low, high = (float(b) for b in band)
    except (TypeError, ValueError):
        raise ValueError(f"A band must be a name or (fmin, fmax) in Hz, got {band!r}.") from None
    if not (math.isfinite(low) and math.isfinite(high)) or low < 0 or low >= high:
        raise ValueError(f"A band needs 0 <= fmin < fmax, got {band!r}.")
    return low, high


@dataclass(frozen=True)
class PSI(_CrossSpectral):
    """Phase slope index (Nolte et al. 2008) in one or more frequency bands.

    ``values[i, j, b] = Im(sum_f conj(C_ij(f)) C_ij(f + df))`` over the neighbouring frequency
    bins of band ``b``: the slope of the coherency phase across the band, weighted by the
    coherence. It is antisymmetric and positive when ``i`` leads ``j`` (the phase of ``C_ij``
    grows with frequency when ``j`` is a delayed copy of ``i``). The value is not normalised by
    its standard deviation (as in ``mne_connectivity.phase_slope_index``); use
    :func:`dirconn.significance_test` for inference. The diagonal is undefined (NaN).

    Parameters
    ----------
    bands : sequence of str or (float, float)
        Required. Each band is a name from :data:`dirconn.BANDS` or ``(fmin, fmax)`` in Hz, both
        limits inclusive, and must hold at least two frequency bins. The band centres become the
        ``freq`` coordinate and must increase; the band edges are recorded in the provenance.
    segment_length, time_halfbandwidth
        As for :class:`ImCoh`.
    """

    info = MethodInfo(
        name="psi",
        backend=_BACKEND,
        directed=True,
        antisymmetric=True,
        min_nodes=2,
        value_range=(-math.inf, math.inf),
        scale_dependent=False,
        frequency_resolved=True,
        assumptions=_ASSUMPTIONS,
    )

    bands: Any = None

    def __post_init__(self) -> None:
        """Validate the bands."""
        super().__post_init__()
        if self.bands is None:
            raise ValueError("bands is required: PSI is a slope over a band, e.g. [(8, 13)].")
        if isinstance(self.bands, str) or not hasattr(self.bands, "__iter__"):
            raise ValueError(f"bands must be a sequence of bands, got {self.bands!r}.")
        edges = [_band(b) for b in self.bands]
        if not edges:
            raise ValueError("bands must hold at least one band.")
        centres = np.array([(lo + hi) / 2 for lo, hi in edges])
        if not (np.diff(centres) > 0).all():
            raise ValueError("The band centres must increase; order the bands by frequency.")

    def _edges(self) -> list[tuple[float, float]]:
        return [_band(b) for b in self.bands]

    def _fit(self, data: TimeSeries) -> Connectivity:
        edges = self._edges()
        nyquist = data.sfreq / 2
        high = max(hi for _, hi in edges)
        if high > nyquist:
            raise ValueError(f"A band reaches {high:g} Hz, above Nyquist ({nyquist:g} Hz).")
        spectra = self._spectra(data, min(lo for lo, _ in edges), high)
        coh = coherency(spectra)
        values = np.empty((data.n_channels, data.n_channels, len(edges)))
        bins = []
        for b, (lo, hi) in enumerate(edges):
            keep = np.flatnonzero((spectra.freqs >= lo) & (spectra.freqs <= hi))
            if keep.size < 2:
                raise ValueError(
                    f"Band ({lo:g}, {hi:g}) Hz holds {keep.size} frequency bin(s) at a resolution "
                    f"of {spectra.record['frequency_resolution_hz']:g} Hz; it needs at least 2."
                )
            part = coh[:, :, keep]
            values[:, :, b] = np.sum(np.conj(part[:, :, :-1]) * part[:, :, 1:], axis=-1).imag
            bins.append(int(keep.size))
        index = np.arange(data.n_channels)
        values[index, index] = np.nan
        provenance = self._provenance(
            data,
            details={
                "cross_spectra": spectra.record,
                "bands": [[lo, hi] for lo, hi in edges],
                "bins_per_band": bins,
            },
        )
        centres = [(lo + hi) / 2 for lo, hi in edges]
        return Connectivity.from_array(
            values, source=list(data.names), freqs=centres, provenance=provenance
        )


def imcoh(data: TimeSeries, **params: Any) -> Connectivity:
    """Imaginary coherence of ``data``; shorthand for ``ImCoh(**params).fit(data)``."""
    return ImCoh(**params).fit(data)


def psi(data: TimeSeries, **params: Any) -> Connectivity:
    """Phase slope index of ``data``; shorthand for ``PSI(**params).fit(data)``."""
    return PSI(**params).fit(data)
