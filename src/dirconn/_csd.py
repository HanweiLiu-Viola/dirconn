"""Multitaper cross-spectra shared by the spectral methods (imcoh, PSI, multitaper Granger).

The spectra are an own multitaper transform on non-overlapping segments of ``segment_length``
seconds, with the tapers, weights and scaling of :func:`mne.time_frequency.csd_array_multitaper`
(``tests/test_csd.py`` checks equality on every frequency MNE returns). MNE is not called at run
time because it never returns 0 Hz, which the spectral factorisation of the multitaper Granger
causality needs (:func:`two_sided_spectra`). Continuous data must give ``segment_length``;
epoched data may (every trial is then cut on its own) or use every trial as one segment. No
segment crosses a trial boundary, and the tail that does not fill a segment is dropped and
recorded. The mean of every channel is removed within every segment, as MNE always does.
:func:`cross_spectra` returns the native FFT bins of the segment length between ``fmin`` and
``fmax`` without 0 Hz, as MNE does.

Convention: ``values[i, j, f] = <X_i(f) conj(X_j(f))>``, averaged over tapers and segments. When
channel ``j`` is channel ``i`` delayed by ``d`` samples, the phase of ``values[i, j, f]`` is
``+2 pi f d / sfreq`` (measured in ``tests/test_csd.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from dirconn.timeseries import TimeSeries
from dirconn.tvmvar import seconds_to_samples

DEFAULT_TIME_HALFBANDWIDTH = 4.0
"""Default time-half-bandwidth product ``NW`` (MNE's default as well); about ``2 NW - 1`` tapers."""

_LOW_BIAS_RATIO = 0.9
_BACKEND = "own multitaper transform (scipy DPSS; tapers, weights and scaling as in MNE)"


@dataclass(frozen=True)
class CrossSpectra:
    """Cross-spectral density matrices on a frequency grid.

    Attributes
    ----------
    values : ndarray of complex, shape (n_channels, n_channels, n_freqs)
        ``values[i, j, f] = <X_i(f) conj(X_j(f))>``; Hermitian in ``(i, j)`` at every frequency.
    freqs : ndarray, shape (n_freqs,)
        Frequencies in Hz.
    record : dict
        Segments, tapers and bandwidth, for the provenance of the result that uses the spectra.
    """

    values: np.ndarray
    freqs: np.ndarray
    record: dict[str, Any]


def _number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    return float(value)


def _segments(data: TimeSeries, segment_length: Any) -> tuple[np.ndarray, int, int]:
    """Return ``(segments, segment_samples, discarded_samples)``; segments are (n, ch, times)."""
    if segment_length is None:
        if data.segmentation == "epoched":
            return np.asarray(data.data), data.n_times, 0
        raise ValueError(
            "segment_length is required for continuous data: it sets the frequency resolution "
            "(1 / segment_length Hz), so it has no default."
        )
    seg = seconds_to_samples(
        _number(segment_length, "segment_length"), data.sfreq, "segment_length"
    )
    per_trial = data.n_times // seg
    if per_trial < 1:
        where = "a trial" if data.segmentation == "epoched" else "the data"
        raise ValueError(
            f"segment_length ({segment_length!r} s, {seg} samples) is longer than {where} "
            f"({data.n_times} samples)."
        )
    used = per_trial * seg
    x = np.asarray(data.data)[:, :, :used]
    # (trial, channel, segment, sample) -> (trial, segment, channel, sample): trials never mix
    split = x.reshape(data.n_trials, data.n_channels, per_trial, seg).transpose(0, 2, 1, 3)
    segments = split.reshape(data.n_trials * per_trial, data.n_channels, seg)
    return segments, seg, data.n_trials * (data.n_times - used)


def _tapers(n_times: int, half_nbw: float) -> tuple[np.ndarray, np.ndarray]:
    """DPSS tapers and their eigenvalues, chosen as MNE does (``sym=False``, eigenvalue > 0.9)."""
    from scipy.signal.windows import dpss

    tapers, ratios = dpss(
        n_times, half_nbw, Kmax=max(int(2 * half_nbw), 1), sym=False, norm=2, return_ratios=True
    )
    tapers, ratios = np.atleast_2d(tapers), np.atleast_1d(ratios)
    keep = ratios > _LOW_BIAS_RATIO
    return tapers[keep], ratios[keep]


def _prepare(
    data: TimeSeries, segment_length: Any, time_halfbandwidth: Any
) -> tuple[np.ndarray, int, int, float, np.ndarray, np.ndarray]:
    segments, seg, discarded = _segments(data, segment_length)
    nw = _number(time_halfbandwidth, "time_halfbandwidth")
    if nw < 1:
        raise ValueError(f"time_halfbandwidth must be at least 1, got {time_halfbandwidth!r}.")
    tapers, eigenvalues = _tapers(seg, nw)
    if tapers.shape[0] < 1:
        raise ValueError(f"time_halfbandwidth={nw:g} leaves no taper with low bias.")
    return segments, seg, discarded, nw, tapers, eigenvalues


def _periodogram(
    segments: np.ndarray,
    tapers: np.ndarray,
    eigenvalues: np.ndarray,
    sfreq: float,
    n_fft: int | None = None,
) -> np.ndarray:
    """Two-sided multitaper cross-periodogram on every rfft bin, averaged over segments.

    ``P[i, j, f] = sum_k w_k X_ik(f) conj(X_jk(f)) / sfreq`` with ``w_k`` the eigenvalues divided by
    their sum (MNE's non-adaptive weights), after removing the mean of every segment. ``n_fft``
    zero-pads the transform (default: the segment length).
    """
    x = segments - segments.mean(axis=-1, keepdims=True)
    tapered = x[:, :, np.newaxis, :] * tapers[np.newaxis, np.newaxis]
    spectra = np.fft.rfft(tapered, n=n_fft, axis=-1)
    weights = eigenvalues / eigenvalues.sum()
    total = np.einsum("k,sikf,sjkf->ijf", weights, spectra, np.conj(spectra))
    # exactly Hermitian, as MNE's upper-triangle computation is: otherwise rounding makes
    # |imcoh[i, j]| and |imcoh[j, i]| differ in the last bit and breaks ties when scoring
    total = 0.5 * (total + np.conj(np.swapaxes(total, 0, 1)))
    return total / (segments.shape[0] * sfreq)


def _record(
    data: TimeSeries, segments: np.ndarray, seg: int, discarded: int, nw: float, n_tapers: int
) -> dict[str, Any]:
    return {
        "backend": _BACKEND,
        "segmentation": data.segmentation,
        "segment_length_s": seg / data.sfreq,
        "segment_samples": seg,
        "n_segments": int(segments.shape[0]),
        "discarded_samples": int(discarded),
        "time_halfbandwidth": nw,
        "bandwidth_hz": 2 * nw * data.sfreq / seg,
        "n_tapers": n_tapers,
        "frequency_resolution_hz": data.sfreq / seg,
        "mean_removed_per_segment": True,
    }


def cross_spectra(
    data: TimeSeries,
    *,
    segment_length: float | None = None,
    time_halfbandwidth: float = DEFAULT_TIME_HALFBANDWIDTH,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> CrossSpectra:
    """Multitaper cross-spectra of all channel pairs, averaged over segments.

    One-sided densities with the conventions of :func:`mne.time_frequency.csd_array_multitaper`
    (equal to it on every frequency it returns, which ``tests/test_csd.py`` checks): 0 Hz is left
    out and the Nyquist bin of an even segment is not doubled.

    Parameters
    ----------
    data : TimeSeries
        Continuous or epoched input.
    segment_length : float, optional
        Segment length in seconds, a multiple of the sampling interval. Required for continuous
        data; for epoched data every trial is cut on its own, and without it every trial is one
        segment.
    time_halfbandwidth : float
        Time-half-bandwidth product ``NW`` (at least 1); the full bandwidth in Hz is
        ``2 NW / segment duration``.
    fmin, fmax : float
        Frequency range in Hz, inclusive; ``fmax`` defaults to the Nyquist frequency.

    Returns
    -------
    CrossSpectra
    """
    segments, seg, discarded, nw, tapers, eigenvalues = _prepare(
        data, segment_length, time_halfbandwidth
    )
    nyquist = data.sfreq / 2
    lo = _number(fmin, "fmin")
    hi = nyquist if fmax is None else _number(fmax, "fmax")
    if lo < 0:
        raise ValueError(f"fmin must be non-negative, got {fmin!r}.")
    if hi > nyquist:
        raise ValueError(f"fmax ({fmax!r} Hz) is above the Nyquist frequency ({nyquist:g} Hz).")
    if lo >= hi:
        raise ValueError(f"fmin ({fmin!r}) must be below fmax ({hi!r}).")
    bins = np.fft.rfftfreq(seg, 1.0 / data.sfreq)
    keep = (bins > 0) & (bins >= lo) & (bins <= hi)
    if not keep.any():
        raise ValueError(
            f"There is no frequency bin between fmin={lo:g} and fmax={hi:g} Hz at a resolution "
            f"of {data.sfreq / seg:g} Hz (segment of {seg} samples)."
        )
    one_sided = 2 * _periodogram(segments, tapers, eigenvalues, data.sfreq)
    if seg % 2 == 0:
        one_sided[:, :, -1] /= 2  # the Nyquist bin has no mirror image, as in MNE
    record = _record(data, segments, seg, discarded, nw, tapers.shape[0])
    record.update({"fmin": lo, "fmax": hi, "two_sided": False})
    return CrossSpectra(values=one_sided[:, :, keep], freqs=bins[keep], record=record)


def two_sided_spectra(
    data: TimeSeries,
    *,
    segment_length: float | None = None,
    time_halfbandwidth: float = DEFAULT_TIME_HALFBANDWIDTH,
    pad_factor: int = 1,
) -> CrossSpectra:
    """Two-sided multitaper cross-spectra on every bin from 0 Hz to Nyquist, for factorisation.

    ``values[..., f]`` is the two-sided density at ``freqs[f]`` (half the one-sided one away from
    0 Hz and Nyquist); the negative frequencies are its complex conjugate. Same segments and
    tapers as :func:`cross_spectra`. ``pad_factor`` zero-pads every segment to that many times its
    length, which samples the same spectrum on a finer grid (every ``pad_factor``-th bin is a
    native bin); the record keeps ``n_fft``.
    """
    if isinstance(pad_factor, bool) or not isinstance(pad_factor, int) or pad_factor < 1:
        raise ValueError(f"pad_factor must be a positive integer, got {pad_factor!r}.")
    segments, seg, discarded, nw, tapers, eigenvalues = _prepare(
        data, segment_length, time_halfbandwidth
    )
    n_fft = seg * pad_factor
    values = _periodogram(segments, tapers, eigenvalues, data.sfreq, n_fft)
    record = _record(data, segments, seg, discarded, nw, tapers.shape[0])
    record.update({"two_sided": True, "n_fft": n_fft})
    freqs = np.fft.rfftfreq(n_fft, 1.0 / data.sfreq)
    return CrossSpectra(values=values, freqs=freqs, record=record)
