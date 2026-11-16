"""Spectral dynamic range of a record across the analysed bands.

A VAR model fitted to data whose power differs by many orders of magnitude between the bands of
interest is numerically ill-conditioned (a low-passed or strongly 1/f record, for example), so the
estimate in the weak band is unreliable. The diagnostic compares the Welch power of the bands per
channel and flags a large ratio.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal

from dirconn.data.provenance import AssumptionCheck
from dirconn.data.timeseries import TimeSeries

__all__ = [
    "SPECTRAL_RANGE_WARN_DB",
    "SpectralRange",
    "spectral_dynamic_range",
    "spectral_range_check",
]

SPECTRAL_RANGE_WARN_DB: float = 60.0
"""Dynamic range in dB above which :func:`spectral_range_check` warns.

The threshold is a conservative rule of thumb; it has not been calibrated on a benchmark.
"""


def _round(x: Any) -> Any:
    """Round floats (scalars or nested lists of them) to 6 significant digits."""
    if isinstance(x, np.ndarray):
        return _round(x.tolist())
    if isinstance(x, list):
        return [_round(v) for v in x]
    return float(f"{float(x):.6g}")


@dataclass(frozen=True)
class SpectralRange:
    """Welch spectrum and band powers of a record.

    Parameters
    ----------
    names : tuple of str
        Channel names.
    freqs : numpy.ndarray
        Frequency grid in Hz, shape ``(n_freqs,)``.
    psd : numpy.ndarray
        Power spectral density, shape ``(n_channels, n_freqs)``.
    band_power : dict of str to numpy.ndarray
        Mean PSD inside each band per channel, shape ``(n_channels,)``.
    dynamic_range_db : numpy.ndarray
        ``10 * log10(max band power / min band power)`` per channel.
    worst_channel : str
        Channel with the largest dynamic range.
    worst_db : float
        The dynamic range of ``worst_channel``.
    """

    names: tuple[str, ...]
    freqs: np.ndarray
    psd: np.ndarray
    band_power: dict[str, np.ndarray]
    dynamic_range_db: np.ndarray
    worst_channel: str
    worst_db: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary with floats rounded to 6 significant digits."""
        return {
            "names": list(self.names),
            "freqs": _round(self.freqs),
            "psd": _round(self.psd),
            "band_power": {k: _round(v) for k, v in self.band_power.items()},
            "dynamic_range_db": _round(self.dynamic_range_db),
            "worst_channel": self.worst_channel,
            "worst_db": _round(self.worst_db),
        }


def _valid_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` of the maximal runs of True in a 1-D mask."""
    padded = np.concatenate(([False], valid, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def spectral_dynamic_range(
    ts: TimeSeries,
    bands: Mapping[str, tuple[float, float]],
    *,
    nperseg: int | None = None,
) -> SpectralRange:
    """Compare the Welch power of the bands per channel.

    Every trial is cut into its maximal runs of valid samples (the whole trial without a mask) and
    Welch's method is applied to each run longer than ``nperseg``; the spectra are averaged over
    runs and trials, weighted by run length. Shorter runs are skipped.

    Parameters
    ----------
    ts : TimeSeries
        The record.
    bands : mapping of str to (float, float)
        Named bands ``(lo, hi)`` in Hz; the power of a band is the mean PSD over ``lo <= f < hi``.
    nperseg : int, optional
        Welch segment length; default ``min(n_times, int(2 * sfreq))``, at least 64.

    Returns
    -------
    SpectralRange
        The spectrum, band powers and per-channel dynamic range.

    Raises
    ------
    ValueError
        If no run of valid samples is longer than ``nperseg``, or a band holds no frequency bin.
    """
    data = np.asarray(ts.data, dtype=float)
    n_trials, n_channels, n_times = data.shape
    if nperseg is None:
        nperseg = max(64, min(n_times, int(2 * ts.sfreq)))
    nperseg = int(nperseg)
    valid = ts.valid if ts.valid is not None else np.ones((n_trials, n_times), dtype=bool)

    total = np.zeros((n_channels, nperseg // 2 + 1))
    weight = 0
    freqs = np.fft.rfftfreq(nperseg, 1.0 / ts.sfreq)
    for trial in range(n_trials):
        for start, stop in _valid_runs(np.asarray(valid[trial], dtype=bool)):
            if stop - start <= nperseg:
                continue
            _, pxx = signal.welch(data[trial, :, start:stop], fs=ts.sfreq, nperseg=nperseg, axis=-1)
            total += pxx * (stop - start)
            weight += stop - start
    if weight == 0:
        raise ValueError(
            f"No run of valid samples is longer than nperseg={nperseg} "
            f"({nperseg / ts.sfreq:.2f} s); cannot estimate the spectrum."
        )
    psd = total / weight

    band_power: dict[str, np.ndarray] = {}
    for name, (lo, hi) in bands.items():
        sel = (freqs >= lo) & (freqs < hi)
        if not sel.any():
            raise ValueError(f"Band {name!r} ({lo}-{hi} Hz) contains no frequency bin.")
        band_power[name] = psd[:, sel].mean(axis=1)
    stacked = np.stack(list(band_power.values()))
    with np.errstate(divide="ignore"):
        db = 10.0 * np.log10(stacked.max(axis=0) / stacked.min(axis=0))
    worst = int(np.argmax(db))
    return SpectralRange(
        names=tuple(ts.names),
        freqs=freqs,
        psd=psd,
        band_power=band_power,
        dynamic_range_db=db,
        worst_channel=ts.names[worst],
        worst_db=float(db[worst]),
    )


def spectral_range_check(sr: SpectralRange) -> AssumptionCheck:
    """Turn a :class:`SpectralRange` into an assumption check.

    Parameters
    ----------
    sr : SpectralRange
        Result of :func:`spectral_dynamic_range`.

    Returns
    -------
    AssumptionCheck
        ``"warn"`` when the worst dynamic range exceeds :data:`SPECTRAL_RANGE_WARN_DB`, else
        ``"pass"``; the value is the worst range in dB.
    """
    if not math.isfinite(sr.worst_db):
        return AssumptionCheck(
            "spectral_dynamic_range", "warn", "band power is zero or not finite", None
        )
    status = "warn" if sr.worst_db > SPECTRAL_RANGE_WARN_DB else "pass"
    idx = sr.names.index(sr.worst_channel)
    powers = {k: float(v[idx]) for k, v in sr.band_power.items()}
    hi_band = max(powers, key=powers.__getitem__)
    lo_band = min(powers, key=powers.__getitem__)
    detail = (
        f"{sr.worst_channel}: {sr.worst_db:.1f} dB between bands {hi_band} and {lo_band} "
        f"(threshold {SPECTRAL_RANGE_WARN_DB} dB, uncalibrated)"
    )
    return AssumptionCheck("spectral_dynamic_range", status, detail, sr.worst_db)
