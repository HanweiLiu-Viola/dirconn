"""Frequency-domain quantities of a VAR model: transfer function, DTF, PDC, gPDC and rPDC.

For a VAR(p) model ``x[t] = sum_k A_k x[t - k] + e[t]`` with ``A_k = coefs[k - 1]`` indexed
``[target, source]``, the frequency-domain polynomial and its inverse are::

    Abar(f) = I - sum_k A_k exp(-2 pi i f k / fs)        (the AR polynomial)
    H(f)    = Abar(f)^-1                                  (the transfer function)

Both are indexed ``[target, source]``. The measures returned by :func:`dtf_from_coefs` and
:func:`pdc_from_coefs` use the layout ``[source, target, freq]`` of ``dirconn.Connectivity``, so
that ``values[i, j]`` is the influence of channel ``i`` on channel ``j``. Papers usually index the
other way round (``i`` is the target); the transposition is done here, once.

:func:`pdc_null_scale` and :func:`rpdc_from_coefs` also need the innovation covariance and the
covariance of the lagged regressors of the fit (``VarFit.noise_cov``, ``VarFit.regressor_cov``):
they build the asymptotic covariance of the least-squares coefficients.

:func:`dtf_from_coefs_series` and :func:`pdc_from_coefs_series` do the same for a trajectory of
coefficients ``(n_times, order, n, n)`` of a time-varying model and return
``[source, target, freq, time]``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve

SERIES_CHUNK_ELEMENTS = 4_000_000
"""Complex entries (time x freq x n x n) the series transforms handle at once, to bound memory."""


def _check_coefs(coefs: Any) -> np.ndarray:
    array = np.asarray(coefs, dtype=float)
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError(
            f"coefs must have shape (order, n_channels, n_channels), got {array.shape}."
        )
    return array


def _check_freq_vector(freqs: Any) -> np.ndarray:
    array = np.asarray(freqs, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"freqs must be a one-dimensional array, got shape {array.shape}.")
    return array


def frequency_grid(
    sfreq: float,
    *,
    n_freqs: int = 128,
    fmax: float | None = None,
    freqs: Any = None,
) -> np.ndarray:
    """Return the frequency axis of a spectral estimate, in Hz.

    By default the grid runs from 0 to the Nyquist frequency in ``n_freqs`` evenly spaced points,
    both ends included: it hides no choice of frequency band, and a band can be picked afterwards
    with :meth:`dirconn.Connectivity.reduce_freq`.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    n_freqs : int
        Number of grid points, at least 2. Ignored when ``freqs`` is given.
    fmax : float, optional
        Upper end of the grid in Hz, at most the Nyquist frequency (the default).
    freqs : array_like, optional
        An explicit axis: finite, non-negative, strictly increasing and not above the Nyquist
        frequency. Use it to match the axis of another result, for example
        ``SimulatedData.truth_at(freqs)``.

    Returns
    -------
    ndarray
        A read-only, strictly increasing vector.
    """
    sfreq = float(sfreq)
    if not math.isfinite(sfreq) or sfreq <= 0.0:
        raise ValueError(f"sfreq must be a positive finite number, got {sfreq!r}.")
    nyquist = sfreq / 2.0

    if freqs is not None:
        if fmax is not None:
            raise ValueError("Pass either freqs or fmax, not both.")
        axis = np.asarray(freqs, dtype=float)
        if axis.ndim != 1 or axis.size == 0:
            raise ValueError(f"freqs must be a non-empty one-dimensional array, got {axis.shape}.")
        if not np.isfinite(axis).all():
            raise ValueError("freqs must be finite.")
        if (axis < 0).any():
            raise ValueError("freqs must be non-negative.")
        if (axis > nyquist).any():
            raise ValueError(f"freqs must not exceed the Nyquist frequency ({nyquist:g} Hz).")
        if axis.size > 1 and not (np.diff(axis) > 0).all():
            raise ValueError("freqs must be strictly increasing.")
    else:
        if isinstance(n_freqs, bool) or not isinstance(n_freqs, (int, np.integer)) or n_freqs < 2:
            raise ValueError(f"n_freqs must be an integer of at least 2, got {n_freqs!r}.")
        top = nyquist if fmax is None else float(fmax)
        if not math.isfinite(top) or top <= 0.0:
            raise ValueError(f"fmax must be a positive finite number, got {fmax!r}.")
        if top > nyquist:
            raise ValueError(f"fmax ({top:g} Hz) exceeds the Nyquist frequency ({nyquist:g} Hz).")
        axis = np.linspace(0.0, top, int(n_freqs))
    out = axis.copy()
    out.flags.writeable = False
    return out


def ar_polynomial(coefs: Any, freqs: Any, sfreq: float) -> np.ndarray:
    """Return ``Abar(f) = I - sum_k A_k exp(-2 pi i f k / fs)``, shape ``(n_freqs, n, n)``.

    The matrices are indexed ``[target, source]``; this is the quantity partial directed coherence
    is built from.
    """
    array = _check_coefs(coefs)
    freq = _check_freq_vector(freqs)
    order, n_channels, _ = array.shape
    phase = np.exp(-2j * np.pi * np.outer(freq, np.arange(1, order + 1)) / float(sfreq))
    a_of_f = np.einsum("fk,kij->fij", phase, array)
    return np.eye(n_channels)[None] - a_of_f


def transfer_function(coefs: Any, freqs: Any, sfreq: float) -> np.ndarray:
    """Return the transfer function ``H(f) = Abar(f)^-1``, shape ``(n_freqs, n, n)``.

    The matrices are indexed ``[target, source]``. A singular ``Abar(f)`` (a unit root) is an error:
    the model is not stationary and the transfer function does not exist; it is never replaced by a
    pseudo-inverse.
    """
    abar = ar_polynomial(coefs, freqs, sfreq)
    identity = np.broadcast_to(np.eye(abar.shape[1]), abar.shape)
    try:
        return np.linalg.solve(abar, identity)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "I - A(f) is singular at one of the frequencies (a unit root): the VAR model is not "
            "stationary and its transfer function does not exist."
        ) from exc


def _normalise(power: np.ndarray, axis: int) -> np.ndarray:
    """Divide by the sum over ``axis``; entries whose sum is zero are undefined (NaN)."""
    total = power.sum(axis=axis, keepdims=True)
    return np.divide(power, total, out=np.full_like(power, np.nan), where=total != 0)


def dtf_from_coefs(coefs: Any, freqs: Any, sfreq: float) -> np.ndarray:
    """Directed transfer function, normalised over the sources of each target.

    ``DTF(i -> j, f) = |H_ji(f)|^2 / sum_m |H_jm(f)|^2``: the share of the signal that reaches
    channel ``j`` at frequency ``f`` and comes from channel ``i``. For every target the values over
    all sources sum to 1.

    Parameters
    ----------
    coefs : array_like, shape (order, n_channels, n_channels)
        VAR coefficients ``[lag - 1, target, source]``.
    freqs : array_like, shape (n_freqs,)
        Frequencies in Hz.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    ndarray, shape (n_channels, n_channels, n_freqs)
        Layout ``[source, target, freq]``.
    """
    power = np.abs(transfer_function(coefs, freqs, sfreq)) ** 2  # [freq, target, source]
    return _normalise(power, axis=2).transpose(2, 1, 0)


def pdc_from_coefs(
    coefs: Any,
    freqs: Any,
    sfreq: float,
    *,
    squared: bool = True,
    noise_var: Any = None,
) -> np.ndarray:
    """Partial directed coherence, normalised over the targets of each source.

    ``PDC(i -> j, f) = |Abar_ji(f)|^2 / sum_m |Abar_mi(f)|^2``: the share of the outflow of channel
    ``i`` at frequency ``f`` that goes to channel ``j``. For every source the squared values over
    all targets sum to 1. With ``noise_var`` the terms are weighted by the inverse innovation
    variance of the target (generalised PDC), which removes the dependence on the relative scale of
    the channels::

        gPDC(i -> j, f) = (|Abar_ji|^2 / s_j) / sum_m (|Abar_mi|^2 / s_m)

    Parameters
    ----------
    coefs, freqs, sfreq
        As in :func:`dtf_from_coefs`.
    squared : bool
        ``True`` returns the squared form (the values themselves sum to 1 over the targets);
        ``False`` the amplitude form ``|Abar_ji| / sqrt(sum_m |Abar_mi|^2)`` of Baccala and
        Sameshima (2001), the square root of the former.
    noise_var : array_like, shape (n_channels,), optional
        Innovation variance of every channel, for example the diagonal of ``VarFit.noise_cov``.

    Returns
    -------
    ndarray, shape (n_channels, n_channels, n_freqs)
        Layout ``[source, target, freq]``. A cell whose denominator is zero (a source column of
        ``Abar`` that vanishes, as at a unit root) is NaN.
    """
    abar = ar_polynomial(coefs, freqs, sfreq)
    power = np.abs(abar) ** 2  # [freq, target, source]
    if noise_var is not None:
        variance = np.asarray(noise_var, dtype=float)
        if (
            variance.shape != (abar.shape[1],)
            or not np.isfinite(variance).all()
            or (variance <= 0).any()
        ):
            raise ValueError(
                f"noise_var must be a vector of {abar.shape[1]} positive finite values, "
                f"got {noise_var!r}."
            )
        power = power / variance[None, :, None]
    result = _normalise(power, axis=1).transpose(2, 1, 0)
    return result if squared else np.sqrt(result)


def _lag_vectors(order: int, freqs: np.ndarray, sfreq: float) -> np.ndarray:
    """Rows ``u_k(f) = [cos(k w), -sin(k w)]``, ``w = 2 pi f / fs``, shape ``(n_freqs, order, 2)``.

    With them ``[Re, Im] Abar_ij(f) = [delta_ij, 0] - sum_k a_ij(k) u_k(f)``.
    """
    phase = np.outer(2.0 * np.pi * freqs / float(sfreq), np.arange(1, order + 1))
    return np.stack([np.cos(phase), -np.sin(phase)], axis=-1)


def _source_blocks(regressor_cov: Any, order: int, n_channels: int) -> np.ndarray:
    """Return the lag blocks of ``inv(regressor_cov)`` of every source, ``(n, order, order)``.

    Block ``j`` holds the entries ``H_jj(k, l)`` of channel ``j`` at lags ``k`` and ``l``, which
    with the noise variance of a target give the covariance of the coefficients ``a_ij(k)``.
    """
    cov = np.asarray(regressor_cov, dtype=float)
    size = order * n_channels
    if cov.shape != (size, size) or not np.isfinite(cov).all():
        raise ValueError(
            f"regressor_cov must be a finite ({size}, {size}) matrix for order {order} and "
            f"{n_channels} channels, got shape {cov.shape}."
        )
    try:
        factor = cho_factor(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "The covariance of the lagged regressors is singular: the coefficients have no "
            "asymptotic covariance. Use fewer lags or fewer channels."
        ) from exc
    inverse = cho_solve(factor, np.eye(size))
    return np.einsum("kjlj->jkl", inverse.reshape(order, n_channels, order, n_channels))


def _noise_variance(noise_cov: Any, n_channels: int) -> np.ndarray:
    cov = np.asarray(noise_cov, dtype=float)
    if cov.shape != (n_channels, n_channels):
        raise ValueError(
            f"noise_cov must have shape ({n_channels}, {n_channels}), got {cov.shape}."
        )
    variance = np.diag(cov)
    if not np.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError("The innovation variances (diagonal of noise_cov) must be positive.")
    return variance


def _coefficient_gram(
    coefs: np.ndarray, regressor_cov: Any, freqs: np.ndarray, sfreq: float
) -> tuple[np.ndarray, np.ndarray]:
    """``W_j(f) = U(f)^T H_jj U(f)``, shape ``(n, n_freqs, 2, 2)``, and the rows ``U(f)``.

    ``S_ii W_j(f) / N`` is the asymptotic covariance of ``[Re, Im] Abar_ij(f)`` (target ``i``,
    source ``j``) of a least-squares fit on ``N`` rows.
    """
    order, n_channels, _ = coefs.shape
    rows = _lag_vectors(order, freqs, sfreq)
    blocks = _source_blocks(regressor_cov, order, n_channels)
    return np.einsum("fkd,jkl,fle->jfde", rows, blocks, rows), rows


def rank_one_freqs(order: int, freqs: Any, sfreq: float) -> np.ndarray:
    """Frequencies at which the covariance ``V`` of rPDC has rank 1, as a boolean mask.

    At order 1 the real and imaginary part of ``Abar_ij(f)`` are proportional to one coefficient,
    at every frequency. At ``f = 0`` and at the Nyquist frequency ``sin(k w) = 0`` for every lag,
    so the imaginary part vanishes. In both cases only one direction is tested (``df = 1``).
    """
    freq = _check_freq_vector(freqs)
    if order == 1:
        return np.ones(freq.shape, dtype=bool)
    return (freq == 0.0) | np.isclose(freq, float(sfreq) / 2.0, rtol=1e-12, atol=0.0)


def pdc_null_scale(
    coefs: Any, noise_cov: Any, regressor_cov: Any, freqs: Any, sfreq: float
) -> np.ndarray:
    """``C_ij(f)`` of Schelter et al. (2006), layout ``[source, target, freq]``.

    In the paper's indices (``i`` target, ``j`` source), ``S`` the innovation covariance and
    ``H = inv(regressor_cov)``::

        C_ij(f) = S_ii sum_{k,l} H_jj(k, l) [cos(k w) cos(l w) + sin(k w) sin(l w)]

    ``C_ij(f) / N`` is the sum of the asymptotic variances of the real and imaginary part of
    ``Abar_ij(f)``. Under the null ``Abar_ij(f) = 0``, ``N |Abar_ij(f)|^2 / C_ij(f)`` is
    approximated by a chi-square variable with one degree of freedom.

    Parameters
    ----------
    coefs, freqs, sfreq
        As in :func:`dtf_from_coefs`.
    noise_cov : array_like, shape (n_channels, n_channels)
        Innovation covariance, ``VarFit.noise_cov``.
    regressor_cov : array_like, shape (order * n_channels, order * n_channels)
        Covariance of the lagged regressors, ``VarFit.regressor_cov``.
    """
    array = _check_coefs(coefs)
    freq = _check_freq_vector(freqs)
    variance = _noise_variance(noise_cov, array.shape[1])
    gram, _ = _coefficient_gram(array, regressor_cov, freq, sfreq)
    trace = gram[..., 0, 0] + gram[..., 1, 1]  # [source, freq]
    return trace[:, None, :] * variance[None, :, None]


def rpdc_from_coefs(
    coefs: Any, noise_cov: Any, regressor_cov: Any, freqs: Any, sfreq: float
) -> tuple[np.ndarray, np.ndarray]:
    """Renormalised partial directed coherence (Schelter et al. 2009), ``[source, target, freq]``.

    In the paper's indices (``i`` target, ``j`` source)::

        lambda_ij(f) = Z^T V_ij(f)^-1 Z,    Z = [Re Abar_ij(f), Im Abar_ij(f)]^T
        V_ij(f) = S_ii sum_{k,l} H_jj(k, l) u_k(f) u_l(f)^T,    u_k(f) = [cos(k w), -sin(k w)]^T

    ``V_ij(f) / N`` is the asymptotic covariance of ``Z`` of a least-squares fit on ``N`` rows,
    derived from ``cov(a_ij(k), a_ij(l)) = S_ii H_jj(k, l) / N``; ``N lambda`` is the Wald
    statistic of ``Abar_ij(f) = 0``. Where ``V`` has rank 1 (:func:`rank_one_freqs`) the
    quadratic form is taken on the one direction ``u_1(f)`` of its range. The diagonal is NaN.

    Parameters
    ----------
    coefs, noise_cov, regressor_cov, freqs, sfreq
        As in :func:`pdc_null_scale`.

    Returns
    -------
    values : ndarray, shape (n_channels, n_channels, n_freqs)
        ``lambda``, layout ``[source, target, freq]``.
    df : ndarray, shape (n_freqs,)
        Degrees of freedom of ``N lambda`` under the null: 1 where ``V`` has rank 1, else 2.
    """
    array = _check_coefs(coefs)
    freq = _check_freq_vector(freqs)
    order, n_channels, _ = array.shape
    variance = _noise_variance(noise_cov, n_channels)
    gram, rows = _coefficient_gram(array, regressor_cov, freq, sfreq)  # [source, freq, 2, 2]
    abar = ar_polynomial(array, freq, sfreq).transpose(2, 1, 0)  # [source, target, freq]
    re, im = abar.real, abar.imag
    single = rank_one_freqs(order, freq, sfreq)
    values = np.empty(abar.shape)

    full = ~single
    if full.any():
        g = gram[:, full]
        det = g[..., 0, 0] * g[..., 1, 1] - g[..., 0, 1] * g[..., 1, 0]  # [source, freq]
        r, m = re[:, :, full], im[:, :, full]
        quad = (
            g[:, None, :, 1, 1] * r**2
            - 2.0 * g[:, None, :, 0, 1] * r * m
            + g[:, None, :, 0, 0] * m**2
        )
        values[:, :, full] = quad / det[:, None, :]
    if single.any():
        u = rows[single, 0, :]  # [freq, 2], a unit vector spanning the range of V
        g = gram[:, single]
        spread = np.einsum("fd,jfde,fe->jf", u, g, u)  # [source, freq]
        along = re[:, :, single] * u[None, None, :, 0] + im[:, :, single] * u[None, None, :, 1]
        values[:, :, single] = along**2 / spread[:, None, :]
    values /= variance[None, :, None]
    values[np.arange(n_channels), np.arange(n_channels)] = np.nan
    return values, np.where(single, 1, 2)


def spectral_radius(coefs: Any) -> float:
    """Spectral radius of the companion matrix of a VAR model.

    The model is stationary when the radius is below 1; values close to 1 mean a near unit root,
    where the transfer function becomes extremely large.
    """
    array = _check_coefs(coefs)
    order, n_channels, _ = array.shape
    companion = np.zeros((n_channels * order, n_channels * order))
    companion[:n_channels, :] = np.hstack(list(array))
    if order > 1:
        companion[n_channels:, :-n_channels] = np.eye(n_channels * (order - 1))
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def _check_series(coefs: Any) -> np.ndarray:
    array = np.asarray(coefs, dtype=float)
    if array.ndim != 4 or array.shape[2] != array.shape[3] or 0 in array.shape:
        raise ValueError(
            "coefs must have shape (n_times, order, n_channels, n_channels) with no empty axis, "
            f"got {array.shape}."
        )
    return array


def _series_measure(
    coefs: Any, freqs: Any, sfreq: float, *, kind: str, squared: bool = True
) -> np.ndarray:
    """Evaluate DTF (``kind="dtf"``) or PDC (``kind="pdc"``) on every slice, chunk by chunk."""
    array = _check_series(coefs)
    freq = _check_freq_vector(freqs)
    n_times, order, n_channels, _ = array.shape
    phase = np.exp(-2j * np.pi * np.outer(freq, np.arange(1, order + 1)) / float(sfreq))
    identity = np.eye(n_channels)
    out = np.empty((n_channels, n_channels, freq.size, n_times))
    chunk = max(1, SERIES_CHUNK_ELEMENTS // (freq.size * n_channels**2))
    for start in range(0, n_times, chunk):
        part = array[start : start + chunk]
        abar = identity - np.einsum("fk,tkij->tfij", phase, part)  # [time, freq, target, source]
        if kind == "dtf":
            try:
                transfer = np.linalg.solve(abar, np.broadcast_to(identity, abar.shape))
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "I - A(f) is singular at one of the frequencies (a unit root): the VAR model "
                    "is not stationary and its transfer function does not exist."
                ) from exc
            normalised = _normalise(np.abs(transfer) ** 2, axis=3)
        else:
            normalised = _normalise(np.abs(abar) ** 2, axis=2)
            if not squared:
                normalised = np.sqrt(normalised)
        out[..., start : start + chunk] = normalised.transpose(3, 2, 1, 0)
    return out


def dtf_from_coefs_series(coefs: Any, freqs: Any, sfreq: float) -> np.ndarray:
    """Directed transfer function of every time slice of a time-varying model.

    Each slice is evaluated exactly as :func:`dtf_from_coefs` does. A slice with non-finite
    coefficients gives NaN; a slice whose ``I - A(f)`` is exactly singular is an error.

    Parameters
    ----------
    coefs : array_like, shape (n_times, order, n_channels, n_channels)
        VAR coefficients ``[time, lag - 1, target, source]``.
    freqs : array_like, shape (n_freqs,)
        Frequencies in Hz.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    ndarray, shape (n_channels, n_channels, n_freqs, n_times)
        Layout ``[source, target, freq, time]``.
    """
    return _series_measure(coefs, freqs, sfreq, kind="dtf")


def pdc_from_coefs_series(
    coefs: Any, freqs: Any, sfreq: float, *, squared: bool = True
) -> np.ndarray:
    """Partial directed coherence of every time slice of a time-varying model.

    Each slice is evaluated exactly as :func:`pdc_from_coefs` does (without ``noise_var``);
    parameters and layout are those of :func:`dtf_from_coefs_series`.
    """
    return _series_measure(coefs, freqs, sfreq, kind="pdc", squared=squared)
