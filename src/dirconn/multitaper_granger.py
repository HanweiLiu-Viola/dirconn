"""Non-parametric spectral Granger causality from multitaper spectra (Dhamala et al. 2008).

No autoregressive model is fitted. The two-sided multitaper spectral matrix of a channel pair
(:func:`dirconn._csd.two_sided_spectra`) is factorised as ``S(f) = psi(f) psi(f)^H`` with Wilson's
algorithm (Wilson 1972), which gives the transfer function ``H`` and the noise covariance
``Sigma`` of the pair; Geweke's spectral decomposition then gives the Granger causality at every
frequency. The pairs are judged one at a time (pairwise, not conditional).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from dirconn._csd import DEFAULT_TIME_HALFBANDWIDTH, two_sided_spectra
from dirconn.estimator import Estimator, MethodInfo
from dirconn.result import Connectivity
from dirconn.timeseries import TimeSeries

WILSON_TOL = 1e-10
"""Convergence: largest relative error ``||S - psi psi^H|| / ||S||`` over the frequencies."""
WILSON_MAX_ITER = 1000
WILSON_STALL = 1e-6
"""The iteration has stalled when ten more steps lower the error by less than this fraction."""
WILSON_MAX_RESIDUAL = 1e-6
"""Largest relative error accepted when the iteration stalls above ``WILSON_TOL``.

With ``PAD_FACTOR = 16`` the largest stall measured on the reference scenarios was 3e-8."""
PAD_FACTOR = 16
"""Zero-padding of the multitaper transform before the factorisation.

An estimated spectrum on the native grid of a segment has no exact causal factor on that grid
(its autocovariance wraps around the circle), so the iteration stalls. Measured on the reference
scenarios: padding 4 stalls at up to 1.5e-3 and moves sharp spectral
peaks by up to 10 % against padding 16; padding 16 stalls below 3e-8 and agrees with padding 32 to
1.2e-5, at about twice the run time of padding 8."""


def full_circle(values: np.ndarray, n_fft: int) -> np.ndarray:
    """Extend a two-sided spectrum on the rfft bins to all ``n_fft`` bins of the unit circle.

    ``values`` has the frequency on the last axis; the negative frequencies are the complex
    conjugates of the positive ones. Returns the frequency on the first axis, ``(n_fft, m, m)``.
    """
    positive = np.moveaxis(values, -1, 0)
    n_rfft = positive.shape[0]
    if n_rfft != n_fft // 2 + 1:
        raise ValueError(f"{n_rfft} bins do not belong to an FFT of length {n_fft}.")
    negative = np.conj(positive[1 : n_fft - n_rfft + 1][::-1])
    return np.concatenate([positive, negative], axis=0)


@dataclass(frozen=True)
class WilsonFactor:
    """Result of :func:`wilson_factorisation`.

    Attributes
    ----------
    transfer : ndarray, shape (n_fft, m, m)
        ``H(f)``, indexed ``[target, source]``, with ``H`` equal to the identity at lag 0.
    noise_cov : ndarray, shape (m, m)
        ``Sigma``, so that ``S(f) = H(f) Sigma H(f)^H``.
    n_iter : int
        Iterations used.
    error : float
        Final ``max_f ||S - psi psi^H|| / ||S||``.
    """

    transfer: np.ndarray
    noise_cov: np.ndarray
    n_iter: int
    error: float


def _plus(g: np.ndarray) -> np.ndarray:
    """Wilson's plus operator: keep the causal part (positive lags, half the upper lag-0 term)."""
    n_fft = g.shape[0]
    coefs = np.fft.ifft(g, axis=0)
    coefs[0] = np.triu(0.5 * coefs[0])
    coefs[n_fft // 2 + 1 :] = 0.0
    return np.fft.fft(coefs, axis=0)


def _error(spectrum: np.ndarray, psi: np.ndarray) -> float:
    residual = spectrum - psi @ np.conj(np.swapaxes(psi, 1, 2))
    norms = np.linalg.norm(spectrum, axis=(1, 2))
    return float(np.max(np.linalg.norm(residual, axis=(1, 2)) / norms))


def wilson_factorisation(
    spectrum: np.ndarray,
    *,
    tol: float = WILSON_TOL,
    max_iter: int = WILSON_MAX_ITER,
    max_residual: float = WILSON_MAX_RESIDUAL,
) -> WilsonFactor:
    """Minimum-phase factorisation ``S(f) = psi(f) psi(f)^H`` of a spectral matrix (Wilson 1972).

    The iteration stops when the relative error ``max_f ||S - psi psi^H|| / ||S||`` is below
    ``tol``, or when it has stalled (ten more steps lower it by less than ``WILSON_STALL`` of its
    value) at an error of at most ``max_residual``: an estimated spectrum has no exact causal
    factor on a finite grid, so its error stops at a small floor (see ``PAD_FACTOR``).

    Parameters
    ----------
    spectrum : ndarray, shape (n_fft, m, m)
        Hermitian, positive definite spectral matrices on all ``n_fft`` bins of the unit circle
        (see :func:`full_circle`).
    tol, max_iter, max_residual
        Convergence criterion, the largest number of iterations and the largest error accepted
        when the iteration stalls.

    Returns
    -------
    WilsonFactor

    Raises
    ------
    ValueError
        When the iteration neither converges nor stalls below ``max_residual`` within
        ``max_iter`` steps, or a matrix is not positive definite. No partial result is returned.
    """
    spectrum = np.asarray(spectrum, dtype=complex)
    n_fft, m, _ = spectrum.shape
    gamma0 = np.fft.ifft(spectrum, axis=0)[0].real
    try:
        upper = np.linalg.cholesky(gamma0).T
    except np.linalg.LinAlgError:
        raise ValueError("The spectral matrix is not positive definite at lag 0.") from None
    psi = np.broadcast_to(upper, spectrum.shape).astype(complex)
    identity = np.eye(m)
    history = [_error(spectrum, psi)]
    for n_iter in range(1, max_iter + 1):
        try:
            inverse = np.linalg.inv(psi)
        except np.linalg.LinAlgError:
            raise ValueError("The Wilson iteration met a singular factor.") from None
        g = inverse @ spectrum @ np.conj(np.swapaxes(inverse, 1, 2)) + identity
        psi = psi @ _plus(g)
        error = _error(spectrum, psi)
        history.append(error)
        if error < tol:
            break
        stalled = n_iter >= 10 and error > (1.0 - WILSON_STALL) * history[-11]
        if stalled:
            if error <= max_residual:
                break
            raise ValueError(
                f"The Wilson factorisation stalled at a relative error of {error:.3g} after "
                f"{n_iter} iterations (accepted up to {max_residual:g})."
            )
    else:
        raise ValueError(
            f"The Wilson factorisation did not converge in {max_iter} iterations "
            f"(relative error {error:.3g}, tolerance {tol:g})."
        )
    a0 = np.fft.ifft(psi, axis=0)[0].real
    try:
        transfer = psi @ np.linalg.inv(a0)
    except np.linalg.LinAlgError:
        raise ValueError("The lag-0 factor of the Wilson iteration is singular.") from None
    return WilsonFactor(transfer=transfer, noise_cov=a0 @ a0.T, n_iter=n_iter, error=error)


def geweke_spectral_gc(transfer: np.ndarray, noise_cov: np.ndarray) -> np.ndarray:
    """Geweke's spectral Granger causality of a two-channel system, ``[source, target, freq]``.

    ``values[x, y, f] = log(S_yy / (S_yy - (Sigma_xx - Sigma_xy^2 / Sigma_yy) |H_yx|^2))`` with
    ``S = H Sigma H^H``; ``transfer`` is ``(n_freqs, 2, 2)`` indexed ``[target, source]``. The
    diagonal is NaN.
    """
    if transfer.shape[1:] != (2, 2) or noise_cov.shape != (2, 2):
        raise ValueError("geweke_spectral_gc needs a two-channel system.")
    spectrum = transfer @ noise_cov @ np.conj(np.swapaxes(transfer, 1, 2))
    values = np.full((2, 2, transfer.shape[0]), np.nan)
    for x, y in ((0, 1), (1, 0)):
        s_yy = spectrum[:, y, y].real
        partial = noise_cov[x, x] - noise_cov[x, y] ** 2 / noise_cov[y, y]
        values[x, y] = np.log(s_yy / (s_yy - partial * np.abs(transfer[:, y, x]) ** 2))
    return values


@dataclass(frozen=True)
class MTGC(Estimator):
    """Pairwise spectral Granger causality from multitaper spectra, without a fitted model.

    ``values[i, j, f]`` is Geweke's spectral Granger causality from ``i`` to ``j`` at frequency
    ``f`` (in nats), computed from the Wilson factorisation of the two-sided multitaper spectral
    matrix of the pair ``(i, j)`` alone (Dhamala et al. 2008). Each pair is judged on its own, so
    an indirect or shared influence through a third channel is not removed (as in
    ``GC(conditional=False)``); the conditional version is not implemented. The mean over the
    whole unit circle approximates the time-domain pairwise Granger causality; it is recorded in
    the provenance (``integral``). The diagonal is undefined (NaN). This is predictive influence,
    not interventional causality.

    Parameters
    ----------
    segment_length : float, optional
        Segment length in seconds; required for continuous data, optional for epoched data. It
        sets the frequency resolution, ``1 / segment_length`` Hz, and so has no default.
    time_halfbandwidth : float
        Time-half-bandwidth product ``NW`` of the tapers (at least 1).
    fmin, fmax : float
        Frequency range of the result in Hz, inclusive (the factorisation always uses the whole
        spectrum). The frequencies are the native bins of the segment length, 0 Hz included.
    """

    info = MethodInfo(
        name="mtgc",
        backend="own multitaper spectra, Wilson factorisation, Geweke decomposition (pairwise)",
        directed=True,
        min_nodes=2,
        value_range=(0.0, math.inf),
        scale_dependent=False,
        frequency_resolved=True,
        assumptions=("stationarity", "linearity", "causal_sufficiency", "filtering"),
    )

    segment_length: float | None = None
    time_halfbandwidth: float = DEFAULT_TIME_HALFBANDWIDTH
    fmin: float = 0.0
    fmax: float | None = None

    def _fit(self, data: TimeSeries) -> Connectivity:
        spectra = two_sided_spectra(
            data,
            segment_length=self.segment_length,
            time_halfbandwidth=self.time_halfbandwidth,
            pad_factor=PAD_FACTOR,
        )
        nyquist = data.sfreq / 2
        fmax = nyquist if self.fmax is None else float(self.fmax)
        if not 0.0 <= float(self.fmin) < fmax <= nyquist:
            raise ValueError(
                f"Need 0 <= fmin < fmax <= Nyquist ({nyquist:g} Hz), got fmin={self.fmin!r}, "
                f"fmax={self.fmax!r}."
            )
        # the result is given on the native bins of the segment, every PAD_FACTOR-th padded bin
        native = np.zeros(spectra.freqs.size, dtype=bool)
        native[::PAD_FACTOR] = True
        keep = native & (spectra.freqs >= self.fmin) & (spectra.freqs <= fmax)
        if not keep.any():
            raise ValueError(f"No frequency bin between {self.fmin!r} and {fmax!r} Hz.")
        n_fft = spectra.record["n_fft"]
        n = data.n_channels
        values = np.full((n, n, int(keep.sum())), np.nan)
        integral = np.full((n, n), np.nan)
        n_iter = np.zeros((n, n), dtype=int)
        errors = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                pair = spectra.values[np.ix_([i, j], [i, j])]
                try:
                    factor = wilson_factorisation(full_circle(pair, n_fft))
                except ValueError as exc:
                    raise ValueError(f"Pair {data.names[i]!r}, {data.names[j]!r}: {exc}") from exc
                gc = geweke_spectral_gc(factor.transfer, factor.noise_cov)
                rfft = gc[:, :, : spectra.freqs.size]
                for (a, b), (x, y) in (((i, j), (0, 1)), ((j, i), (1, 0))):
                    values[a, b] = rfft[x, y, keep]
                    integral[a, b] = float(np.mean(gc[x, y]))
                n_iter[i, j] = n_iter[j, i] = factor.n_iter
                errors[i, j] = errors[j, i] = factor.error
        provenance = self._provenance(
            data,
            details={
                "cross_spectra": spectra.record,
                "pairwise": True,
                "integral": [[None if math.isnan(v) else v for v in row] for row in integral],
                "wilson_iterations_max": int(np.max(n_iter)),
                "wilson_error_max": float(np.max(errors)),
                "wilson_tol": WILSON_TOL,
                "wilson_max_residual": WILSON_MAX_RESIDUAL,
                "pad_factor": PAD_FACTOR,
            },
        )
        return Connectivity.from_array(
            values, source=list(data.names), freqs=spectra.freqs[keep], provenance=provenance
        )


def mtgc(data: TimeSeries, **params: Any) -> Connectivity:
    """Multitaper Granger causality of ``data``; shorthand for ``MTGC(**params).fit(data)``."""
    return MTGC(**params).fit(data)
