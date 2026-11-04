"""Time-varying multivariate autoregressive (MVAR) models fitted by a Kalman recursion.

A sliding window assumes stationarity inside each window and throws the estimate away at every
boundary, so its time resolution is capped by the window length. A recursion carries the
coefficient estimate forward sample by sample, which lets it follow a coupling that switches on in
the middle of a recording. This is the estimator behind the adaptive DTF and the temporal PDC of
the literature.

The model is the VAR of :mod:`dirconn.mvar` with coefficients that change with time::

    x[t] = sum_{k=1..p} A_k(t) @ x[t - k] + e[t]

and the same layout ``coefs[time, k - 1, target, source]``, so every time slice can be handed to
the spectral transforms of :mod:`dirconn.spectral`.

Structure of the filter
-----------------------
Putting every coefficient into one state vector costs ``O((p N^2)^2)`` per sample. The observation
matrix has Kronecker structure ``I_N (x) phi^T`` with ``phi = [x[t-1]; ...; x[t-p]]``, so with a
block-diagonal state covariance and a diagonal observation noise the ``N`` equations decouple. The
recursion therefore runs ``N`` filters with states of length ``p N`` and scalar-observation updates
(Joseph form, no matrix inverse), vectorised over the target channels and over the trials.

``update="dual"`` adds a recursion for the full innovation covariance, ``update="rls"`` tracks only
its diagonal, and ``update="dekf"`` is the dual extended Kalman filter of Wan and
Nelson (2001): the observation is modelled as ``y = x + n`` with explicit measurement noise and a
companion-form state filter estimates the denoised signal that the coefficient filter regresses
on. With ``measurement_noise=0`` the DEKF reduces exactly to ``"dual"``.

The coefficient estimates of ``"dual"`` and ``"rls"`` are **identical** (checked to the last bit):
the coefficient filter only uses the diagonal of the innovation covariance, which both recursions
update in the same way. They differ in :attr:`TvarFit.noise_cov`, whose off-diagonal entries only
``"dual"`` fills, so ADTF and TPDC give the same result for either.

Several trials are filtered separately and the **coefficients are averaged over trials at every
time point**; the spectral measure is then computed once from the averaged coefficients. This
assumes that the non-stationarity is time-locked to the trial start.

Nothing is replaced silently. A seed fit that fails raises; ``init="zeros"`` is an explicit choice
that is recorded. The recursion counts the updates it had to skip and the state-filter resets, and
:func:`tvar_checks` reports them.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from dirconn.exceptions import DirconnWarning
from dirconn.mvar import (
    OrderSelection,
    _check_rows,
    _readonly,
    _solve,
    check_order_spec,
    choose_order,
    lagged_design,
)
from dirconn.provenance import AssumptionCheck
from dirconn.timeseries import TimeSeries

UPDATES: tuple[str, ...] = ("dual", "rls", "dekf")
STANDARDISERS: tuple[str | None, ...] = (None, "zscore", "demean")
INITS: tuple[str, ...] = ("var", "zeros")
DEFAULT_MAX_ORDER = 5
"""Largest order the criterion may choose by default; the state length is ``order * n_nodes``."""
UNSTABLE_WARN_FRACTION = 0.05
"""Fraction of unstable time points above which :func:`tvar_checks` warns."""
AUTO_TIME_POINTS = 200
"""Number of returned time points that ``time_step=None`` aims at."""
MEASUREMENT_NOISE_FRACTION = 0.1
"""Default DEKF measurement noise: this fraction of each channel's variance."""

_TINY = float(np.finfo(float).tiny)


@dataclass(frozen=True, eq=False)
class TvarFit:
    """Coefficient trajectory of a time-varying MVAR model, averaged over trials.

    Attributes
    ----------
    coefs : ndarray, shape (n_times, order, n_nodes, n_nodes)
        ``coefs[i, k - 1, target, source]`` at time point ``i``; the layout of
        :attr:`dirconn.mvar.VarFit.coefs` for every time slice.
    noise_cov : ndarray, shape (n_times, n_nodes, n_nodes)
        Recursively estimated innovation covariance. For ``"dual"`` it estimates the driving
        noise. For ``"dekf"`` the innovation mixes driving and measurement noise and estimates
        their sum. For ``"rls"`` only the diagonal, the running innovation variance, is filled.
    sample_index : ndarray of int, shape (n_times,)
        Sample index (within a trial) each retained estimate belongs to; starts at ``burn_in``.
    times : ndarray, shape (n_times,)
        ``tmin + sample_index / sfreq`` in seconds.
    radius : ndarray, shape (n_times,)
        Spectral radius of the companion matrix of every retained slice (``inf`` if not finite).
    order : int
        Model order used.
    n_trials : int
        Number of trials that were filtered and averaged.
    n_unstable : int
        Number of retained time points with a companion spectral radius of 1 or more.
    n_skipped_updates : int
        Coefficient updates skipped because the innovation variance was not finite and positive,
        summed over samples, trials and target channels.
    n_state_resets : int
        DEKF state-filter updates that failed and were replaced by a restart from the raw
        observations, summed over samples and trials.
    memory_samples : float
        Effective memory of the recursion, **measured**: the reciprocal of the mean gain along the
        regressor, ``1 - obs_var / innovation_var``, over all samples after the burn-in. For
        a recursive least-squares filter with forgetting factor ``lambda`` it is ``1 / (1 -
        lambda)``. Here the forgetting factor only sets the process noise ``1 / lambda - 1``, and
        the resulting memory is far shorter (a few samples for ``lambda`` up to 0.99, about
        ``sqrt(noise variance / (process noise * regressor variance))`` in general); ``inf``
        when the filter never updates.
    update, forgetting, process_noise, state_cov_init, init, init_samples, burn_in, time_step :
        The settings actually used, with the defaults resolved.
    measurement_noise : ndarray or None
        DEKF measurement-noise variance per channel; ``None`` for the other updates.
    standardize : {"zscore", "demean"} or None
        Preprocessing applied to all trials at once.
    scale : ndarray or None
        The per-channel divisor of ``"zscore"``.
    order_selection : OrderSelection or None
        The criterion curves when the order was chosen; ``None`` for a fixed order. The curves come
        from a **stationary** pooled fit of the whole recording.
    seed_rows : int
        Regression rows of the seed fit (0 for ``init="zeros"``).
    names : tuple of str
        Channel names.
    sfreq : float
        Sampling frequency in Hz.
    """

    coefs: np.ndarray
    noise_cov: np.ndarray
    sample_index: np.ndarray
    times: np.ndarray
    radius: np.ndarray
    order: int
    n_trials: int
    n_unstable: int
    n_skipped_updates: int
    n_state_resets: int
    memory_samples: float
    update: str
    forgetting: float
    process_noise: float
    state_cov_init: float
    init: str
    init_samples: int
    burn_in: int
    time_step: int
    measurement_noise: np.ndarray | None
    standardize: str | None
    scale: np.ndarray | None
    order_selection: OrderSelection | None
    seed_rows: int
    names: tuple[str, ...]
    sfreq: float

    def __post_init__(self) -> None:
        """Make the arrays read-only."""
        for name in ("coefs", "noise_cov", "sample_index", "times", "radius"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        for name in ("measurement_noise", "scale"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _readonly(value))

    @property
    def n_times(self) -> int:
        """Number of retained time points."""
        return int(self.coefs.shape[0])

    @property
    def n_nodes(self) -> int:
        """Number of channels."""
        return int(self.coefs.shape[-1])

    @property
    def unstable_fraction(self) -> float:
        """Fraction of the retained time points whose companion spectral radius is 1 or more."""
        return self.n_unstable / self.n_times if self.n_times else 0.0

    def details(self) -> dict[str, Any]:
        """Return the settings and facts of the fit as a JSON-compatible dictionary."""
        return {
            "update": self.update,
            "order": self.order,
            "forgetting": self.forgetting,
            "process_noise": self.process_noise,
            "state_cov_init": self.state_cov_init,
            "measurement_noise": (
                None if self.measurement_noise is None else self.measurement_noise.tolist()
            ),
            "init": self.init,
            "init_samples": self.init_samples,
            "seed_rows": self.seed_rows,
            "burn_in": self.burn_in,
            "time_step": self.time_step,
            "standardize": self.standardize,
            "scale": None if self.scale is None else self.scale.tolist(),
            "order_selection": (
                None if self.order_selection is None else self.order_selection.to_dict()
            ),
            "sfreq": self.sfreq,
            "memory_samples": (self.memory_samples if math.isfinite(self.memory_samples) else None),
            "n_trials": self.n_trials,
            "n_times": self.n_times,
            "n_unstable": self.n_unstable,
            "unstable_fraction": self.unstable_fraction,
            "n_skipped_updates": self.n_skipped_updates,
            "n_state_resets": self.n_state_resets,
        }


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        kind = "a positive" if minimum == 1 else "a non-negative"
        raise ValueError(f"{name} must be {kind} integer, got {value!r}.")
    return int(value)


def _finite(value: Any, name: str, *, strictly_positive: bool) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}.") from None
    if not math.isfinite(number) or number < 0 or (strictly_positive and number == 0):
        kind = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must be a finite {kind} number, got {value!r}.")
    return number


def _standardise(
    x: np.ndarray, mode: str | None, names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray | None]:
    """Centre and scale all trials with one per-channel mean and standard deviation."""
    if mode is None:
        return x, None
    centred = x - x.mean(axis=(0, 2), keepdims=True)
    if mode == "demean":
        return centred, None
    scale = centred.std(axis=(0, 2))
    flat = np.flatnonzero(scale == 0)
    if flat.size:
        raise ValueError(
            f"Channels {[names[i] for i in flat]} have zero variance and cannot be standardised."
        )
    return centred / scale[None, :, None], scale


def _seed(
    x: np.ndarray, order: int, init_samples: int, ridge: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Stationary pooled fit on the first samples of every trial.

    Returns the state matrix ``(n_nodes, order * n_nodes)`` (row ``target``: the coefficients of
    lag 1, then lag 2, ... over the sources), the residual covariance and the number of rows.
    Failures raise, exactly as in :func:`dirconn.mvar.fit_var`.
    """
    y, design = lagged_design(x[:, :, :init_samples], order)
    _check_rows(y.shape[0], design.shape[1])
    beta, residuals = _solve(y, design, ridge)
    noise_cov = residuals.T @ residuals / (y.shape[0] - design.shape[1])
    return beta.T.copy(), noise_cov, int(y.shape[0])


def _companion_radius(coefs: np.ndarray) -> np.ndarray:
    """Spectral radius of the companion matrix of every time slice, ``inf`` if not finite."""
    n_times, order, n, _ = coefs.shape
    size = order * n
    radius = np.full(n_times, np.inf)
    finite = np.isfinite(coefs).all(axis=(1, 2, 3))
    if not finite.any():
        return radius
    companion = np.zeros((int(finite.sum()), size, size))
    companion[:, :n, :] = np.concatenate(list(coefs[finite].transpose(1, 0, 2, 3)), axis=2)
    if order > 1:
        companion[:, n:, : size - n] = np.eye(size - n)
    try:
        radius[finite] = np.abs(np.linalg.eigvals(companion)).max(axis=1)
    except np.linalg.LinAlgError:  # pragma: no cover - finite input does not fail in practice
        pass
    return radius


def _solve_gain(innovation_cov: np.ndarray, cross: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kalman gain ``cross @ inv(innovation_cov)`` per trial, and which trials could be solved."""
    n_trials = innovation_cov.shape[0]
    try:
        gain = np.linalg.solve(innovation_cov, cross.swapaxes(1, 2)).swapaxes(1, 2)
        return gain, np.ones(n_trials, dtype=bool)
    except np.linalg.LinAlgError:
        gain = np.zeros_like(cross)
        ok = np.zeros(n_trials, dtype=bool)
        for k in range(n_trials):
            try:
                gain[k] = np.linalg.solve(innovation_cov[k], cross[k].T).T
                ok[k] = True
            except np.linalg.LinAlgError:
                pass
        return gain, ok


@dataclass(frozen=True)
class _Run:
    """What the recursion returns, with a leading axis over the forgetting factors."""

    coefs: np.ndarray  # (kept times, n_forgetting, order, n, n)
    noise_cov: np.ndarray  # (kept times, n_forgetting, n, n)
    sample_index: np.ndarray
    n_skipped_updates: np.ndarray  # (n_forgetting,)
    n_state_resets: np.ndarray  # (n_forgetting,)
    mean_gain: np.ndarray  # (n_forgetting,)


def _run_filter(
    x: np.ndarray,
    *,
    order: int,
    update: str,
    forgetting: np.ndarray,
    process_noise: np.ndarray,
    seed_states: np.ndarray,
    seed_noise: np.ndarray,
    obs_variance: np.ndarray,
    measurement_variance: np.ndarray | None,
    state_cov_init: float,
    burn_in: int,
    time_step: int,
) -> _Run:
    """Run the recursion on ``(n_trials, n_nodes, n_times)`` for every forgetting factor at once.

    The trials and the forgetting factors form one batch (element ``l * n_trials + k`` is trial
    ``k`` with factor ``l``); the elements never interact, so every factor gets exactly the
    numbers a separate run would give. The coefficients are averaged over the trials of each factor.
    """
    n_trials, n, n_times = x.shape
    n_factors = forgetting.size
    batch = n_factors * n_trials
    size = order * n
    lam = np.repeat(forgetting, n_trials)  # (batch,)
    noise_level = np.repeat(process_noise, n_trials)[:, None, None, None]
    x = np.tile(x, (n_factors, 1, 1))
    diag = np.arange(n)
    eye = np.eye(size)

    states = np.broadcast_to(seed_states, (batch, n, size)).copy()
    covs = np.broadcast_to(eye * state_cov_init, (batch, n, size, size)).copy()
    noise = np.broadcast_to(seed_noise, (batch, n, n)).copy()
    obs_var = np.broadcast_to(obs_variance, (batch, n)).copy()
    lagged = np.concatenate(
        [x[:, :, order - lag : n_times - lag] for lag in range(1, order + 1)], 1
    )

    dekf = update == "dekf"
    if dekf:
        assert measurement_variance is not None
        meas = measurement_variance
        z = np.concatenate([x[:, :, order - 1 - lag] for lag in range(order)], axis=1)
        reset_cov = np.diag(np.tile(meas, order))
        state_cov = np.broadcast_to(reset_cov, (batch, size, size)).copy()
        transition = np.zeros((batch, size, size))
        if order > 1:
            transition[:, n:, : size - n] = np.eye(size - n)

    indices = np.arange(burn_in, n_times, time_step)
    kept_coefs = np.empty((indices.size, n_factors, order, n, n))
    kept_noise = np.empty((indices.size, n_factors, n, n))
    next_kept = 0
    skipped = np.zeros(n_factors, dtype=int)
    resets = np.zeros(n_factors, dtype=int)
    gain_sum = np.zeros(n_factors)
    gain_count = np.zeros(n_factors)

    def per_factor(values: np.ndarray) -> np.ndarray:
        """Sum a ``(batch, ...)`` array over everything but the factor."""
        return values.reshape(n_factors, -1).sum(axis=1)

    with np.errstate(all="ignore"):
        for t in range(order, n_times):
            phi = z if dekf else lagged[:, :, t - order]
            target = x[:, :, t]

            cov = covs + noise_level * eye
            innovation = target - np.einsum("ks,kns->kn", phi, states)
            cov_phi = np.einsum("knst,kt->kns", cov, phi)
            innovation_var = np.einsum("ks,kns->kn", phi, cov_phi) + obs_var
            ok = np.isfinite(innovation_var) & (innovation_var > 0.0)
            skipped += per_factor(~ok).astype(int)
            if t >= burn_in:
                # gain along the regressor, phi' K = 1 - obs_var / innovation_var: the
                # share of the latest sample's prediction error that goes into the state
                along = 1.0 - obs_var / np.where(ok, innovation_var, 1.0)
                gain_sum += per_factor(np.where(ok, along, 0.0))
                gain_count += per_factor(ok)
            gain = np.where(
                ok[..., None], cov_phi / np.where(ok, innovation_var, 1.0)[..., None], 0
            )
            states = np.where(ok[..., None], states + gain * innovation[..., None], states)
            # Joseph form: far better behaved over long recordings than P <- (I - k phi^T) P,
            # which loses symmetry and can drift indefinite.
            factor = eye - gain[..., :, None] * phi[:, None, None, :]
            updated = factor @ cov @ factor.swapaxes(-1, -2)
            updated += obs_var[..., None, None] * gain[..., :, None] * gain[..., None, :]
            updated = 0.5 * (updated + updated.swapaxes(-1, -2))
            covs = np.where(ok[..., None, None], updated, cov)

            if update == "rls":
                obs_var = lam[:, None] * obs_var + (1.0 - lam[:, None]) * innovation**2
                noise = np.zeros((batch, n, n))
                noise[:, diag, diag] = obs_var
            else:
                noise = lam[:, None, None] * noise + (1.0 - lam[:, None, None]) * np.einsum(
                    "kn,km->knm", innovation, innovation
                )
                obs_var = np.maximum(noise[:, diag, diag], _TINY)

            if dekf:
                # State filter, with the freshly updated coefficients in the transition matrix.
                # The coefficient filter's innovation variance estimates var(v + n), so the
                # driving-noise variance is what is left after removing the measurement noise.
                driving = np.maximum(obs_var - meas, _TINY)
                transition[:, :n, :] = states
                z_pred = np.einsum("kst,kt->ks", transition, z)
                cov_pred = transition @ state_cov @ transition.swapaxes(1, 2)
                cov_pred[:, diag, diag] += driving
                innovation_cov = cov_pred[:, :n, :n].copy()
                innovation_cov[:, diag, diag] += meas
                gain_x, solved = _solve_gain(innovation_cov, cov_pred[:, :, :n])
                z_new = z_pred + np.einsum("ksn,kn->ks", gain_x, target - z_pred[:, :n])
                factor_x = np.broadcast_to(eye, (batch, size, size)).copy()
                factor_x[:, :, :n] -= gain_x
                cov_new = factor_x @ cov_pred @ factor_x.swapaxes(1, 2)
                cov_new += (gain_x * meas) @ gain_x.swapaxes(1, 2)
                cov_new = 0.5 * (cov_new + cov_new.swapaxes(1, 2))
                good = solved & np.isfinite(z_new).all(1) & np.isfinite(cov_new).all((1, 2))
                bad = ~good
                if bad.any():
                    # A transiently unstable companion matrix can blow the covariance up;
                    # restart the state filter from the raw observations and count it.
                    resets += per_factor(bad).astype(int)
                    raw = np.concatenate([x[:, :, t - lag] for lag in range(order)], axis=1)
                    z_new = np.where(bad[:, None], raw, z_new)
                    cov_new = np.where(bad[:, None, None], reset_cov, cov_new)
                z, state_cov = z_new, cov_new

            if t >= burn_in and (t - burn_in) % time_step == 0:
                slice_coefs = states.reshape(batch, n, order, n).transpose(0, 2, 1, 3)
                kept_coefs[next_kept] = slice_coefs.reshape(n_factors, n_trials, order, n, n).mean(
                    1
                )
                kept_noise[next_kept] = noise.reshape(n_factors, n_trials, n, n).mean(1)
                next_kept += 1

    mean_gain = np.divide(gain_sum, gain_count, out=np.zeros(n_factors), where=gain_count > 0)
    return _Run(kept_coefs, kept_noise, indices, skipped, resets, mean_gain)


def check_tvar_options(
    *,
    order: int | str,
    max_order: int,
    update: str,
    forgetting: float,
    process_noise: float | None,
    obs_noise: float | None,
    measurement_noise: float | None,
    init: str,
    init_samples: int | None,
    state_cov_init: float,
    burn_in: int | None,
    time_step: int | None,
    standardize: str | None,
    ridge: float,
    n_folds: int,
) -> dict[str, Any]:
    """Validate the options of :func:`fit_tvar` and return the normalised numeric ones.

    The estimators call this when they are built, so a wrong option fails before any data are
    seen. Raises ``ValueError`` (``TypeError`` for a wrong type of ``order``) with a message that
    names the option.
    """
    check_order_spec(order, max_order, ridge, n_folds)
    if update not in UPDATES:
        raise ValueError(f"update must be one of {UPDATES}, got {update!r}.")
    if standardize not in STANDARDISERS:
        raise ValueError(f"standardize must be one of {STANDARDISERS}, got {standardize!r}.")
    if init not in INITS:
        raise ValueError(f"init must be 'var' or 'zeros', got {init!r}.")
    try:
        forgetting = float(forgetting)
    except (TypeError, ValueError):
        raise ValueError(f"forgetting must be in (0, 1], got {forgetting!r}.") from None
    if not 0.0 < forgetting <= 1.0:
        raise ValueError(f"forgetting must be in (0, 1], got {forgetting}.")
    if time_step is not None:
        time_step = _positive_int(time_step, "time_step")
    state_cov_init = _finite(state_cov_init, "state_cov_init", strictly_positive=True)
    if process_noise is not None:
        process_noise = _finite(process_noise, "process_noise", strictly_positive=False)
    if obs_noise is not None:
        obs_noise = _finite(obs_noise, "obs_noise", strictly_positive=True)
    if measurement_noise is not None:
        if update != "dekf":
            raise ValueError(
                "measurement_noise is only meaningful for update='dekf'; "
                f"got measurement_noise={measurement_noise!r} with update={update!r}."
            )
        measurement_noise = _finite(measurement_noise, "measurement_noise", strictly_positive=False)
    if init_samples is not None:
        init_samples = _positive_int(init_samples, "init_samples")
    if burn_in is not None:
        burn_in = _positive_int(burn_in, "burn_in", minimum=0)
    return {
        "forgetting": forgetting,
        "process_noise": process_noise,
        "obs_noise": obs_noise,
        "measurement_noise": measurement_noise,
        "init_samples": init_samples,
        "state_cov_init": state_cov_init,
        "burn_in": burn_in,
        "time_step": time_step,
    }


def fit_tvar(
    data: TimeSeries,
    *,
    order: int | str = "bic",
    max_order: int = DEFAULT_MAX_ORDER,
    update: str = "dual",
    forgetting: float = 0.99999,
    process_noise: float | None = None,
    obs_noise: float | None = None,
    measurement_noise: float | None = None,
    init: str = "var",
    init_samples: int | None = None,
    state_cov_init: float = 1.0,
    burn_in: int | None = None,
    time_step: int | None = 1,
    standardize: str | None = "zscore",
    ridge: float = 0.0,
    n_folds: int = 5,
) -> TvarFit:
    """Track MVAR coefficients over time with a Kalman, RLS or DEKF recursion.

    The recursion runs at every sample; ``time_step`` only thins what is returned. Every trial is
    filtered separately from the same seed and the coefficients are averaged over trials at each
    time point.

    Parameters
    ----------
    data : TimeSeries
        Continuous (one trial) or epoched data.
    order : int or {"aic", "bic", "hqc", "cv"}
        Model order, or the criterion that chooses it. The criterion is evaluated once, on a
        **stationary** pooled fit of the whole recording (after ``standardize``); under a coupling
        that switches it may therefore pick a different order than the switching model needs.
    max_order : int
        Largest candidate order of the criterion; the default is 5 because the state length is
        ``order * n_nodes``.
    update : {"dual", "rls", "dekf"}
        ``"dual"`` runs a second recursion for the innovation covariance, ``"rls"`` the same for
        its diagonal only; their **coefficients are identical** and only ``noise_cov`` differs.
        ``"dekf"`` is the dual extended Kalman filter for recordings with measurement noise (it
        reduces exactly to ``"dual"`` when ``measurement_noise=0``).
    forgetting : float
        ``lambda`` in ``(0, 1]``. It sets the process noise ``1 / lambda - 1`` and the speed of the
        innovation-variance recursion; it is **not** the memory of a forgetting-factor least-squares
        filter (see :attr:`TvarFit.memory_samples`, which is measured). The default 0.99999 gives a
        memory of about 0.2 s for three nodes at 100 Hz; 0.99, the old package's default, a few
        samples. Lower tracks
        change faster at the cost of a noisier estimate. The main tuning knob.
    process_noise : float, optional
        Diagonal of the state noise. Default ``1 / lambda - 1``, the usual correspondence between
        a random-walk state model and a forgetting factor (0 for ``lambda = 1``).
    obs_noise : float, optional
        Initial observation-noise variance of the coefficient filter. Default: the diagonal of the
        residual covariance of the seed fit (1 for ``init="zeros"``).
    measurement_noise : float, optional
        DEKF only: variance of the observation noise ``n`` in ``y = x + n``, equal for all
        channels. Default: 10 % of each channel's variance. Passing it for another update raises.
    init : {"var", "zeros"}
        ``"var"`` seeds the coefficients from a stationary pooled VAR fit on the first
        ``init_samples`` samples of every trial and **raises** if that fit fails. ``"zeros"``
        starts from zero coefficients; it is an explicit choice and is recorded.
    init_samples : int, optional
        Samples per trial of the seed fit; default ``min(n_times // 4, 20 * order * n_nodes)``, at
        least ``order + 2`` and at most ``n_times``.
    state_cov_init : float
        Initial state covariance scale: larger means less confidence in the seed.
    burn_in : int, optional
        Samples per trial discarded before the first returned estimate; default
        ``max(init_samples, 10 * order)``, at least ``order``.
    time_step : int or None
        Keep every ``time_step``-th estimate after the burn-in. ``None`` keeps about
        ``AUTO_TIME_POINTS`` (200) estimates over the returned span.
    standardize : {"zscore", "demean"} or None
        Applied to all trials at once: one mean and one standard deviation per channel. DTF and
        PDC are sensitive to relative channel scaling, so ``"zscore"`` is the default.
    ridge : float
        Relative ridge penalty of the seed fit and of the order selection.
    n_folds : int
        Folds of the cross-validated order selection (only used for ``order="cv"``).

    Returns
    -------
    TvarFit

    Raises
    ------
    ValueError
        For invalid options, a recording too short for the model, a channel without variance
        under ``standardize="zscore"``, or a seed fit that fails.
    TypeError
        If ``data`` is not a :class:`~dirconn.timeseries.TimeSeries`.
    """
    return fit_tvar_grid(
        data,
        forgetting=[forgetting],
        order=order,
        max_order=max_order,
        update=update,
        process_noise=process_noise,
        obs_noise=obs_noise,
        measurement_noise=measurement_noise,
        init=init,
        init_samples=init_samples,
        state_cov_init=state_cov_init,
        burn_in=burn_in,
        time_step=time_step,
        standardize=standardize,
        ridge=ridge,
        n_folds=n_folds,
    )[0]


def fit_tvar_grid(
    data: TimeSeries,
    *,
    order: int | str = "bic",
    max_order: int = DEFAULT_MAX_ORDER,
    update: str = "dual",
    forgetting: Sequence[float] = (0.99999,),
    process_noise: float | None = None,
    obs_noise: float | None = None,
    measurement_noise: float | None = None,
    init: str = "var",
    init_samples: int | None = None,
    state_cov_init: float = 1.0,
    burn_in: int | None = None,
    time_step: int | None = 1,
    standardize: str | None = "zscore",
    ridge: float = 0.0,
    n_folds: int = 5,
) -> list[TvarFit]:
    """Run :func:`fit_tvar` for several forgetting factors in one pass over the data.

    Everything but ``forgetting`` is as in :func:`fit_tvar` and shared: the standardisation, the
    order selection and the seed fit are done once, and the recursion runs for all factors as one
    batch. The factors never interact, so every result equals the one a separate :func:`fit_tvar`
    call gives, to the last bit; a sweep over the forgetting factor costs little more than one fit.

    Parameters
    ----------
    forgetting : sequence of float
        The forgetting factors, each in ``(0, 1]``; at least one.

    Returns
    -------
    list of TvarFit
        One fit per factor, in the order given.
    """
    if not isinstance(data, TimeSeries):
        raise TypeError(f"fit_tvar expects a TimeSeries, got {type(data).__name__}.")
    factors_in = list(forgetting)
    factors: list[float] = []
    if not factors_in:
        raise ValueError("forgetting must contain at least one value.")
    options = {}
    for value in factors_in:
        options = check_tvar_options(
            order=order,
            max_order=max_order,
            update=update,
            forgetting=value,
            process_noise=process_noise,
            obs_noise=obs_noise,
            measurement_noise=measurement_noise,
            init=init,
            init_samples=init_samples,
            state_cov_init=state_cov_init,
            burn_in=burn_in,
            time_step=time_step,
            standardize=standardize,
            ridge=ridge,
            n_folds=n_folds,
        )
        factors.append(options["forgetting"])
    process_noise = options["process_noise"]
    obs_noise = options["obs_noise"]
    measurement_noise = options["measurement_noise"]
    init_samples = options["init_samples"]
    state_cov_init = options["state_cov_init"]
    burn_in = options["burn_in"]
    time_step = options["time_step"]

    names = data.names
    x, scale = _standardise(np.asarray(data.data, dtype=float), standardize, names)
    n_trials, n, n_times = x.shape

    if isinstance(order, str):
        series = TimeSeries(
            data=x,
            sfreq=data.sfreq,
            channels=data.channels,
            segmentation=data.segmentation,
            tmin=data.tmin,
        )
        # No intercept and no further centring: the filter has none either.
        order_value, selection = choose_order(
            series, order, max_order, ridge=ridge, demean=False, n_folds=n_folds
        )
    else:
        order_value, selection = int(order), None
    size = order_value * n
    if n_times <= order_value + 1:
        raise ValueError(
            f"The recording has {n_times} samples, too few for an order-{order_value} model."
        )

    if init_samples is None:
        init_samples = min(n_times // 4, 20 * size)
    init_samples = min(max(init_samples, order_value + 2), n_times)
    if burn_in is None:
        burn_in = min(max(init_samples, 10 * order_value), n_times - 1)
    elif burn_in >= n_times:
        raise ValueError(
            f"burn_in ({burn_in}) must be smaller than the number of samples ({n_times}); "
            "nothing would be left to return."
        )
    burn_in = max(burn_in, order_value)
    if time_step is None:
        time_step = max(1, math.ceil((n_times - burn_in) / AUTO_TIME_POINTS))

    if init == "var":
        seed_states, seed_noise, seed_rows = _seed(x, order_value, init_samples, ridge)
    else:
        seed_states, seed_noise, seed_rows = np.zeros((n, size)), np.eye(n), 0

    if obs_noise is None:
        diagonal = np.diag(seed_noise).astype(float)
        obs_variance = np.where(np.isfinite(diagonal) & (diagonal > 0), diagonal, 1.0)
    else:
        obs_variance = np.full(n, obs_noise)

    if process_noise is None:
        noise_levels = np.array([(1.0 / f - 1.0) if f < 1.0 else 0.0 for f in factors])
    else:
        noise_levels = np.full(len(factors), process_noise)

    measurement_variance: np.ndarray | None = None
    if update == "dekf":
        if measurement_noise is None:
            measurement_variance = MEASUREMENT_NOISE_FRACTION * x.var(axis=(0, 2))
        else:
            measurement_variance = np.full(n, measurement_noise)

    run = _run_filter(
        x,
        order=order_value,
        update=update,
        forgetting=np.array(factors),
        process_noise=noise_levels,
        seed_states=seed_states,
        seed_noise=seed_noise,
        obs_variance=obs_variance,
        measurement_variance=measurement_variance,
        state_cov_init=state_cov_init,
        burn_in=burn_in,
        time_step=time_step,
    )
    fits = []
    for index, factor in enumerate(factors):
        coefs = np.ascontiguousarray(run.coefs[:, index])
        radius = _companion_radius(coefs)
        mean_gain = float(run.mean_gain[index])
        fits.append(
            TvarFit(
                coefs=coefs,
                noise_cov=np.ascontiguousarray(run.noise_cov[:, index]),
                sample_index=run.sample_index,
                times=data.tmin + run.sample_index / data.sfreq,
                radius=radius,
                order=order_value,
                n_trials=n_trials,
                n_unstable=int((radius >= 1.0).sum()),
                n_skipped_updates=int(run.n_skipped_updates[index]),
                n_state_resets=int(run.n_state_resets[index]),
                memory_samples=1.0 / mean_gain if mean_gain > 0.0 else math.inf,
                update=update,
                forgetting=factor,
                process_noise=float(noise_levels[index]),
                state_cov_init=state_cov_init,
                init=init,
                init_samples=init_samples,
                burn_in=burn_in,
                time_step=time_step,
                measurement_noise=measurement_variance,
                standardize=standardize,
                scale=scale,
                order_selection=selection,
                seed_rows=seed_rows,
                names=names,
                sfreq=data.sfreq,
            )
        )
    return fits


def stationarity_check(
    n_unstable: int, n_total: int, *, unit: str, stacklevel: int = 3
) -> AssumptionCheck:
    """Check how many slices of a trajectory of fitted models are unstable.

    A slice is unstable when the spectral radius of its companion matrix is 1 or more. The check
    is ``"warn"`` with a :class:`~dirconn.exceptions.DirconnWarning` when more than 5 % of the
    ``n_total`` slices (``unit``: for example ``"time points"`` or ``"windows"``) are unstable;
    ``value`` is the unstable fraction.
    """
    fraction = n_unstable / n_total if n_total else 0.0
    detail = (
        f"{n_unstable} of {n_total} {unit} ({fraction:.0%}) have a companion spectral radius of "
        "1 or more."
    )
    if fraction > UNSTABLE_WARN_FRACTION:
        warnings.warn(
            f"{detail} The coefficient trajectory left the stationary region and the spectral "
            "transforms of those slices are unreliable; consider a larger forgetting factor, a "
            "longer window or a lower order.",
            DirconnWarning,
            stacklevel=stacklevel,
        )
        return AssumptionCheck("stationarity", "warn", detail=detail, value=fraction)
    return AssumptionCheck("stationarity", "pass", detail=detail, value=fraction)


def seconds_to_samples(value: float, sfreq: float, name: str) -> int:
    """Convert a duration in seconds to samples; it must be a multiple of the sample interval.

    Raises ``ValueError`` when it is not (nothing is rounded silently) or when it is shorter than
    one sample.
    """
    samples = float(value) * float(sfreq)
    rounded = round(samples)
    if rounded < 1 or abs(samples - rounded) > 1e-6 * max(1.0, samples):
        raise ValueError(
            f"{name} ({value!r} s) must be a multiple of the sampling interval "
            f"({1.0 / sfreq:g} s at {sfreq:g} Hz); it corresponds to {samples:g} samples."
        )
    return int(rounded)


def tvar_checks(fit: TvarFit, *, stacklevel: int = 3) -> list[AssumptionCheck]:
    """Diagnostics of a time-varying fit, for the assumption checklist of a result.

    Diagnostics warn with :class:`~dirconn.exceptions.DirconnWarning` and never stop a run.

    Returns
    -------
    list of AssumptionCheck
        ``"stationarity"`` (``value``: the fraction of time points whose companion spectral radius
        is 1 or more; warns above 5 %), ``"filter_health"`` (skipped updates and state resets;
        warns if there were any), ``"order_source"`` (where the order came from) and, for several
        trials, ``"trial_locking"``. The last two are assumptions that cannot be checked from the
        data and are marked ``"not_checked"``.
    """
    checks: list[AssumptionCheck] = [
        stationarity_check(
            fit.n_unstable, fit.n_times, unit="time points", stacklevel=stacklevel + 1
        )
    ]

    if fit.n_skipped_updates or fit.n_state_resets:
        detail = (
            f"{fit.n_skipped_updates} coefficient updates were skipped (non-finite or non-positive "
            f"innovation variance) and {fit.n_state_resets} DEKF state-filter updates were reset "
            "to the raw observations."
        )
        warnings.warn(f"Time-varying fit: {detail}", DirconnWarning, stacklevel=stacklevel)
        checks.append(
            AssumptionCheck(
                "filter_health",
                "warn",
                detail=detail,
                value=float(fit.n_skipped_updates + fit.n_state_resets),
            )
        )
    else:
        checks.append(
            AssumptionCheck(
                "filter_health",
                "pass",
                detail="no update was skipped and no state was reset.",
                value=0.0,
            )
        )

    if fit.order_selection is not None:
        source = (
            f"the {fit.order_selection.criterion} criterion chose order {fit.order} of "
            f"{fit.order_selection.max_order} candidates on a stationary pooled fit of the whole "
            "recording; when the coefficients change this order can be too low or too high."
        )
    else:
        source = f"order {fit.order} was fixed by the caller, not selected."
    checks.append(AssumptionCheck("order_source", "not_checked", detail=source, value=fit.order))

    if fit.n_trials > 1:
        checks.append(
            AssumptionCheck(
                "trial_locking",
                "not_checked",
                detail=(
                    f"coefficients are averaged over {fit.n_trials} trials at every time point; "
                    "this assumes the non-stationarity is time-locked to the trial start."
                ),
            )
        )
    return checks
