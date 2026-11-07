"""Granger causality from the state-space (innovations) form of a VAR model.

Barnett and Seth (2015, Phys. Rev. E 91:040101) compute Granger causality without fitting a
second, reduced model. The full VAR model is written in innovations form,

    z[t + 1] = A z[t] + K e[t],    x[t] = C z[t] + e[t],    cov(e[t]) = Sigma,

with the companion state ``z[t] = (x[t - 1], ..., x[t - p])``. Any subset ``R`` of the channels is
a state-space process with the same ``A`` and the rows ``C_R``; one discrete algebraic Riccati
equation (DARE) gives its innovations form, so the reduced model is the one *implied* by the fit
of all channels. The reduced process is in general a VARMA process, not a VAR of the same order,
which is why refitting a reduced VAR(p) by least squares (the classical recipe of
:class:`dirconn.GC` with ``method="refit"``) is biased at short orders.

The same reduced models give the conditional spectral Granger causality ``f_{X->Y|Z}(f)``
(Geweke 1984; Barnett and Seth 2015), whose mean over the whole unit circle equals the time-domain
value.

Arrays follow the package conventions: VAR coefficients ``coefs[lag - 1, target, source]``,
transfer functions ``[freq, target, source]`` and Granger causality ``[source, target]`` (time
domain) or ``[source, target, freq]`` (spectral), with a NaN diagonal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg

from dirconn.spectral import (
    _check_coefs,
    _check_freq_vector,
    spectral_radius,
    transfer_function,
)

__all__ = [
    "InnovationsForm",
    "reduced_model",
    "spectral_state_space_gc",
    "state_space_gc",
    "var_to_state_space",
]


def _readonly(array: np.ndarray) -> np.ndarray:
    out = np.array(array)
    out.flags.writeable = False
    return out


@dataclass(frozen=True, eq=False)
class InnovationsForm:
    """A state-space model in innovations form.

    ``z[t + 1] = transition @ z[t] + gain @ e[t]`` and ``y[t] = observation @ z[t] + e[t]`` with
    ``cov(e[t]) = noise_cov``, where ``y`` are the channels ``channels`` of the VAR model the form
    was derived from.

    Attributes
    ----------
    transition : ndarray, shape (n_states, n_states)
        ``A``; for a VAR model of order ``p`` and ``n`` channels the companion matrix,
        ``n_states = n p``.
    observation : ndarray, shape (n_observed, n_states)
        ``C``.
    gain : ndarray, shape (n_states, n_observed)
        ``K``, the steady-state Kalman gain.
    noise_cov : ndarray, shape (n_observed, n_observed)
        ``Sigma``, the covariance of the innovations ``e`` (the one-step prediction errors of
        ``y`` from its own past).
    channels : tuple of int
        Indices, in the original VAR model, of the observed channels.
    """

    transition: np.ndarray
    observation: np.ndarray
    gain: np.ndarray
    noise_cov: np.ndarray
    channels: tuple[int, ...]

    def __post_init__(self) -> None:
        """Make the arrays read-only."""
        for name in ("transition", "observation", "gain", "noise_cov"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(self, "channels", tuple(int(c) for c in self.channels))

    @property
    def n_channels(self) -> int:
        """Number of observed channels."""
        return len(self.channels)

    def transfer_function(self, freqs: Any, sfreq: float) -> np.ndarray:
        """Return ``H(f) = I + C (exp(2 pi i f / sfreq) I - A)^-1 K``, shape ``(n_freqs, n, n)``.

        ``y = H(L) e``; the matrices are indexed ``[target, source]``. For the form of a whole
        VAR model this is :func:`dirconn.spectral.transfer_function`.
        """
        freq = _check_freq_vector(freqs)
        z = np.exp(2j * np.pi * freq / float(sfreq))
        n_states = self.transition.shape[0]
        resolvent = z[:, None, None] * np.eye(n_states) - self.transition
        gain = np.broadcast_to(self.gain, (freq.size, *self.gain.shape))
        return np.eye(self.n_channels) + self.observation @ np.linalg.solve(resolvent, gain)


def _check_noise_cov(noise_cov: Any, n_channels: int) -> np.ndarray:
    cov = np.asarray(noise_cov, dtype=float)
    if cov.shape != (n_channels, n_channels):
        raise ValueError(
            f"noise_cov must have shape ({n_channels}, {n_channels}), got {cov.shape}."
        )
    if not np.isfinite(cov).all():
        raise ValueError("noise_cov must be finite.")
    cov = (cov + cov.T) / 2.0
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        raise ValueError(
            "noise_cov must be positive definite: an innovation with zero variance (a channel "
            "predicted perfectly, or an exact linear combination of the others) makes Granger "
            "causality infinite or undefined."
        ) from None
    return cov


def var_to_state_space(coefs: Any, noise_cov: Any) -> InnovationsForm:
    """Write a VAR model in innovations form.

    ``A`` is the companion matrix, ``C = [A_1 ... A_p]`` (the top block row of ``A``),
    ``K = [I 0 ... 0]^T`` and ``Sigma = noise_cov``.

    Parameters
    ----------
    coefs : array_like, shape (n_lags, n_channels, n_channels)
        VAR coefficients ``[lag - 1, target, source]``.
    noise_cov : array_like, shape (n_channels, n_channels)
        Innovation covariance, symmetric positive definite.

    Returns
    -------
    InnovationsForm

    Raises
    ------
    ValueError
        For an unstable model (spectral radius at least 1), whose reduced models do not exist,
        or a covariance that is not positive definite.
    """
    array = _check_coefs(coefs)
    if not np.isfinite(array).all():
        raise ValueError("coefs must be finite.")
    order, n_channels, _ = array.shape
    cov = _check_noise_cov(noise_cov, n_channels)
    radius = spectral_radius(array)
    if radius >= 1.0:
        raise ValueError(
            f"The VAR model is not stable (spectral radius {radius:.4g} >= 1); state-space "
            "Granger causality needs a stable model."
        )
    n_states = n_channels * order
    transition = np.zeros((n_states, n_states))
    transition[:n_channels] = np.hstack(list(array))
    transition[n_channels:, :-n_channels] = np.eye(n_states - n_channels)
    gain = np.zeros((n_states, n_channels))
    gain[:n_channels] = np.eye(n_channels)
    return InnovationsForm(
        transition=transition,
        observation=transition[:n_channels].copy(),
        gain=gain,
        noise_cov=cov,
        channels=tuple(range(n_channels)),
    )


def reduced_model(model: InnovationsForm, channels: Sequence[int]) -> InnovationsForm:
    """Return the innovations form of a subset of the observed channels of ``model``.

    The subprocess ``y_R = C_R z + e_R`` has the state noise ``K e`` correlated with the
    observation noise ``e_R``. Its steady-state Kalman predictor solves the DARE

        P = A P A^T + Q - (A P C_R^T + S)(C_R P C_R^T + R)^-1 (A P C_R^T + S)^T

    with ``Q = K Sigma K^T``, ``R = Sigma_RR`` and ``S = K Sigma_{:, R}``; then
    ``Sigma_R = C_R P C_R^T + R`` and ``K_R = (A P C_R^T + S) Sigma_R^-1``
    (Barnett and Seth 2015). It is solved with :func:`scipy.linalg.solve_discrete_are`.

    Parameters
    ----------
    model : InnovationsForm
        A stable innovations form, for example from :func:`var_to_state_space`.
    channels : sequence of int
        Positions among ``model``'s observed channels to keep, in the order wanted.

    Returns
    -------
    InnovationsForm
        Same ``transition``, the rows ``channels`` of ``observation``; ``channels`` holds the
        indices in the original VAR model.

    Raises
    ------
    ValueError
        For an empty or repeated selection, or when the Riccati equation has no stabilising
        solution.
    """
    keep = [int(c) for c in channels]
    if not keep:
        raise ValueError("channels must select at least one channel.")
    if len(set(keep)) != len(keep) or min(keep) < 0 or max(keep) >= model.n_channels:
        raise ValueError(
            f"channels must be distinct positions below {model.n_channels}, got {list(channels)}."
        )
    names = tuple(model.channels[c] for c in keep)
    if keep == list(range(model.n_channels)):
        return model
    a, c, k, sigma = model.transition, model.observation, model.gain, model.noise_cov
    c_r = c[keep]
    q = k @ sigma @ k.T
    r = sigma[np.ix_(keep, keep)]
    s = k @ sigma[:, keep]
    try:
        p = linalg.solve_discrete_are(a.T, c_r.T, (q + q.T) / 2.0, r, s=s)
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise ValueError(
            f"The Riccati equation of the reduced model of channels {list(names)} has no "
            f"stabilising solution ({exc}); the model may be close to a unit root."
        ) from exc
    noise_cov = c_r @ p @ c_r.T + r
    noise_cov = (noise_cov + noise_cov.T) / 2.0
    gain = np.linalg.solve(noise_cov, (a @ p @ c_r.T + s).T).T
    return InnovationsForm(
        transition=a, observation=c_r, gain=gain, noise_cov=noise_cov, channels=names
    )


def _var_form_transfer(
    coefs: np.ndarray, full_transfer: np.ndarray, model: InnovationsForm, phase: np.ndarray
) -> np.ndarray:
    """Transfer function of a reduced form of a VAR model, without inverting ``zI - A``.

    For the companion ``A`` and any ``b = (b_1, ..., b_p)``,
    ``C_full (zI - A)^-1 b = H(z) c(z) - b_1`` with ``c = b_1 + sum_{k>=2} A_k D_k``,
    ``D_1 = 0``, ``D_k = z^-1 (D_{k-1} + b_k)`` and ``H`` the transfer function of the VAR, so
    ``H_R = I + rows R of (H c - b_1)`` costs ``O(p n^2 |R|)`` per frequency instead of a solve of
    size ``n p``. ``phase`` is ``z^-1 = exp(-2 pi i f / sfreq)``.
    """
    order, n_channels, _ = coefs.shape
    blocks = model.gain.reshape(order, n_channels, model.n_channels)
    folded = np.broadcast_to(blocks[0], (phase.size, *blocks[0].shape)).astype(complex)
    delayed = np.zeros_like(folded)
    for lag in range(1, order):
        delayed = phase[:, None, None] * (delayed + blocks[lag])
        folded = folded + coefs[lag] @ delayed
    response = full_transfer @ folded - blocks[0]
    return np.eye(model.n_channels) + response[:, list(model.channels), :]


def _granger(
    coefs: Any,
    noise_cov: Any,
    *,
    conditional: bool,
    freqs: Any = None,
    sfreq: float | None = None,
) -> np.ndarray:
    """Time-domain (``freqs`` is None) or spectral state-space Granger causality."""
    if not isinstance(conditional, bool):
        raise TypeError(f"conditional must be a bool, got {conditional!r}.")
    model = var_to_state_space(coefs, noise_cov)
    array = np.asarray(coefs, dtype=float)
    n = model.n_channels
    spectral = freqs is not None
    if spectral:
        freq = _check_freq_vector(freqs)
        phase = np.exp(-2j * np.pi * freq / float(sfreq))  # type: ignore[arg-type]
        full_transfer = transfer_function(array, freq, float(sfreq))  # type: ignore[arg-type]
        values: np.ndarray = np.full((n, n, freq.size), np.nan)
    else:
        values = np.full((n, n), np.nan)

    cache: dict[tuple[int, ...], InnovationsForm] = {}

    def reduced(channels: list[int]) -> InnovationsForm:
        key = tuple(channels)
        if key not in cache:
            cache[key] = reduced_model(model, channels)
        return cache[key]

    groups = (
        [list(range(n))] if conditional else [[i, j] for i in range(n) for j in range(i + 1, n)]
    )
    for group in groups:
        full = reduced(group)
        if spectral:
            group_transfer = (
                full_transfer
                if full is model
                else _var_form_transfer(array, full_transfer, full, phase)
            )
        sigma = full.noise_cov
        for position, source in enumerate(group):
            rest = [i for i in range(len(group)) if i != position]
            targets = [group[i] for i in rest]
            reduced_form = reduced(targets)
            reduced_var = np.diag(reduced_form.noise_cov)
            if not spectral:
                values[source, targets] = np.log(reduced_var / np.diag(sigma)[rest])
                continue
            # Psi = H_R^-1 H_{R, group}: how the innovations of the group reach the innovations of
            # the reduced model
            reduced_transfer = _var_form_transfer(array, full_transfer, reduced_form, phase)
            psi = np.linalg.solve(reduced_transfer, group_transfer[:, rest, :])
            for row, (local, target) in enumerate(zip(rest, targets, strict=True)):
                # Geweke (1984): the unconditional causality from (source, Z*) to the reduced
                # innovation of the target, with the part of the other innovations that the
                # target's innovation does not explain
                others = [i for i in range(len(group)) if i != local]
                partial = (
                    sigma[np.ix_(others, others)]
                    - np.outer(sigma[others, local], sigma[local, others]) / (sigma[local, local])
                )
                gain = psi[:, row, others]
                explained = np.einsum("fa,ab,fb->f", gain, partial, gain.conj()).real
                values[source, target] = np.log(reduced_var[row] / (reduced_var[row] - explained))
    return values


def state_space_gc(coefs: Any, noise_cov: Any, *, conditional: bool = True) -> np.ndarray:
    """Time-domain Granger causality of a VAR model from its state-space form.

    ``values[i, j] = log(Sigma^R_jj / Sigma_jj)`` where ``Sigma`` is the innovation covariance of
    the full model and ``Sigma^R`` that of the reduced model without channel ``i``
    (:func:`reduced_model`), both implied by the same VAR model.

    Parameters
    ----------
    coefs : array_like, shape (n_lags, n_channels, n_channels)
        VAR coefficients ``[lag - 1, target, source]``; the model must be stable.
    noise_cov : array_like, shape (n_channels, n_channels)
        Innovation covariance, positive definite.
    conditional : bool
        ``True``: condition on all other channels (one Riccati equation per source).
        ``False``: judge every pair on its own; the "full" model of a pair is then its bivariate
        subprocess, also from the same VAR model (one Riccati equation per pair and per channel).

    Returns
    -------
    ndarray, shape (n_channels, n_channels)
        ``[source, target]`` in nats, NaN on the diagonal.
    """
    return _granger(coefs, noise_cov, conditional=conditional)


def spectral_state_space_gc(
    coefs: Any, noise_cov: Any, freqs: Any, sfreq: float, *, conditional: bool = True
) -> np.ndarray:
    """Spectral (Geweke) Granger causality of a VAR model from its state-space form.

    For source ``X``, target ``Y`` and the other channels ``Z`` of the model (all other channels
    when ``conditional``, none otherwise), Geweke's (1984) conditional measure is the
    unconditional spectral causality from ``(X, Z*)`` to ``Y*``, where ``*`` marks the
    innovations of the reduced model of ``(Y, Z)`` (Chen, Bressler and Ding 2006; Barnett and
    Seth 2015):

        f_{X->Y|Z}(f) = log( Sigma^R_YY / (Sigma^R_YY - Psi_YS(f) Sigma_{SS|Y} Psi_YS(f)^H) ),

    with ``S = (X, Z)``, ``Sigma^R`` the innovation covariance of the reduced model,
    ``Psi = H_R^-1 H_{(Y,Z),:}`` the filter from the full innovations to the reduced ones and
    ``Sigma_{SS|Y}`` the partial covariance of the full innovations of ``S`` given that of ``Y``.
    The mean of ``f`` over the whole unit circle is the time-domain value of
    :func:`state_space_gc`; for two channels ``f`` is Geweke's (1982) unconditional formula.

    Parameters
    ----------
    coefs, noise_cov, conditional
        As for :func:`state_space_gc`.
    freqs : array_like
        Frequencies in Hz.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    ndarray, shape (n_channels, n_channels, n_freqs)
        ``[source, target, freq]`` in nats, NaN on the diagonal.
    """
    return _granger(coefs, noise_cov, conditional=conditional, freqs=freqs, sfreq=sfreq)
