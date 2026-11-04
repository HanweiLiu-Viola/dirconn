"""ADTF and TPDC: time-resolved DTF and PDC from a time-varying MVAR model.

The coefficients come from the recursion of :mod:`dirconn.tvmvar` (Kalman, RLS or DEKF); every
retained time point is turned into a transfer function or an AR polynomial exactly as the static
:class:`~dirconn.DTF` and :class:`~dirconn.PDC` do it, so the time-varying and the static measures
share their normalisation by construction.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from dirconn.estimator import Estimator, MethodInfo, sampling_check
from dirconn.provenance import STANDARD_ASSUMPTIONS, AssumptionCheck
from dirconn.result import Connectivity
from dirconn.spectral import dtf_from_coefs_series, frequency_grid, pdc_from_coefs_series
from dirconn.timeseries import TimeSeries
from dirconn.tvmvar import (
    DEFAULT_MAX_ORDER,
    TvarFit,
    check_tvar_options,
    fit_tvar,
    seconds_to_samples,
    tvar_checks,
)

_BACKEND = "own numpy time-varying MVAR (Kalman/RLS/DEKF recursion), vectorised frequency response"


@dataclass(frozen=True)
class _TimeVaryingVAR(Estimator):
    """Shared machinery: fit a time-varying MVAR, then evaluate a spectral measure per time point.

    Parameters
    ----------
    order : int or {"aic", "bic", "hqc", "cv"}
        Fixed order, or the criterion that picks it **once, on a stationary pooled fit** of the
        whole recording. When the coupling switches this order can be off; the checklist says so.
    max_order : int
        Largest candidate order; the default is 5 because the state of the filter has
        ``order * n_nodes`` entries per target.
    update : {"dual", "rls", "dekf"}
        The recursion, see :func:`dirconn.tvmvar.fit_tvar`. ``"dual"`` and ``"rls"`` give the
        same coefficients and therefore the same measure.
    forgetting : float
        Forgetting factor in ``(0, 1]``. It sets the process noise ``1 / forgetting - 1`` of
        the recursion; the memory that results is measured and recorded (``memory_samples`` in
        the provenance details) and is much shorter than ``1 / (1 - forgetting)`` samples.
    process_noise, obs_noise, measurement_noise, state_cov_init : float, optional
        Filter settings, see :func:`dirconn.tvmvar.fit_tvar`. ``measurement_noise`` is for
        ``update="dekf"`` only.
    init : {"var", "zeros"}
        Seed of the coefficients: a stationary pooled fit on the first ``init_samples`` samples
        (raises if that fit fails), or an explicit zero start.
    init_samples, burn_in : int, optional
        Samples per trial used for the seed fit and discarded before the first output.
    time_step : float, optional
        Distance between returned time points in seconds, a multiple of the sampling interval.
        Default: about 200 points over the returned span. The recursion itself always runs at
        every sample.
    n_freqs, fmax, freqs
        Frequency axis, as for :class:`~dirconn.DTF`; the default is 64 points from 0 to the
        Nyquist frequency (a time-resolved result carries one spectrum per time point).
    ridge, n_folds
        For the seed fit and the order selection.
    standardize : bool
        Divide every channel by its standard deviation (over all trials and samples) first,
        because the result depends on the relative scale of the channels.
    demean : bool
        Remove the per-channel mean when ``standardize`` is false (standardising always centres).

    Several trials are filtered separately, their coefficients are averaged at every time point and
    the measure is computed once from the average; this assumes that the non-stationarity is
    time-locked to the trial start.
    """

    order: int | str = "bic"
    max_order: int = DEFAULT_MAX_ORDER
    update: str = "dual"
    forgetting: float = 0.99999
    process_noise: float | None = None
    obs_noise: float | None = None
    measurement_noise: float | None = None
    init: str = "var"
    init_samples: int | None = None
    state_cov_init: float = 1.0
    burn_in: int | None = None
    time_step: float | None = None
    n_freqs: int = 64
    fmax: float | None = None
    freqs: Any = None
    ridge: float = 0.0
    n_folds: int = 5
    standardize: bool = True
    demean: bool = True

    def __post_init__(self) -> None:
        """Validate the parameters."""
        super().__post_init__()
        for name in ("standardize", "demean"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool, got {getattr(self, name)!r}.")
        check_tvar_options(
            order=self.order,
            max_order=self.max_order,
            update=self.update,
            forgetting=self.forgetting,
            process_noise=self.process_noise,
            obs_noise=self.obs_noise,
            measurement_noise=self.measurement_noise,
            init=self.init,
            init_samples=self.init_samples,
            state_cov_init=self.state_cov_init,
            burn_in=self.burn_in,
            time_step=None,
            standardize=self._mode(),
            ridge=self.ridge,
            n_folds=self.n_folds,
        )
        if self.time_step is not None:
            step = self.time_step
            if (
                isinstance(step, bool)
                or not isinstance(step, (int, float, np.integer, np.floating))
                or not math.isfinite(step)
                or step <= 0
            ):
                raise ValueError(
                    f"time_step must be a positive finite number of seconds or None, got {step!r}."
                )
        if self.freqs is not None:
            if self.fmax is not None:
                raise ValueError("Pass either freqs or fmax, not both.")
            return
        if (
            isinstance(self.n_freqs, bool)
            or not isinstance(self.n_freqs, (int, np.integer))
            or self.n_freqs < 2
        ):
            raise ValueError(f"n_freqs must be an integer of at least 2, got {self.n_freqs!r}.")
        if self.fmax is not None and (not math.isfinite(float(self.fmax)) or self.fmax <= 0):
            raise ValueError(f"fmax must be a positive finite number, got {self.fmax!r}.")

    def _mode(self) -> str | None:
        if self.standardize:
            return "zscore"
        return "demean" if self.demean else None

    def _depends_on_scale(self) -> bool:
        return not self.standardize

    @abstractmethod
    def _measure(self, fit: TvarFit, freqs: np.ndarray) -> np.ndarray:
        """Return the measure with layout ``[source, target, freq, time]``."""

    def _time_step_samples(self, data: TimeSeries) -> int | None:
        if self.time_step is None:
            return None
        return seconds_to_samples(self.time_step, data.sfreq, "time_step")

    def tvar_options(self, data: TimeSeries) -> dict[str, Any]:
        """Return the options of :func:`~dirconn.tvmvar.fit_tvar` other than ``forgetting``.

        With them, :func:`~dirconn.tvmvar.fit_tvar_grid` can compute the fits of several forgetting
        factors in one pass, and :meth:`from_fit` turns each into a result.
        """
        return {
            "order": self.order,
            "max_order": self.max_order,
            "update": self.update,
            "process_noise": self.process_noise,
            "obs_noise": self.obs_noise,
            "measurement_noise": self.measurement_noise,
            "init": self.init,
            "init_samples": self.init_samples,
            "state_cov_init": self.state_cov_init,
            "burn_in": self.burn_in,
            "time_step": self._time_step_samples(data),
            "standardize": self._mode(),
            "ridge": self.ridge,
            "n_folds": self.n_folds,
        }

    def _fit(self, data: TimeSeries) -> Connectivity:
        fit = fit_tvar(data, forgetting=self.forgetting, **self.tvar_options(data))
        return self._result(data, fit, stacklevel=5)

    def from_fit(self, data: TimeSeries, fit: TvarFit) -> Connectivity:
        """Return the result for a fit that was computed beforehand.

        ``fit`` must have been computed from ``data`` with this estimator's options and forgetting
        factor, for example by :func:`~dirconn.tvmvar.fit_tvar_grid` with :meth:`tvar_options`;
        its update and forgetting factor are checked. Unlike :meth:`fit` this does not run the
        channel-scale guard, so it is meant for data that are standardised.
        """
        if fit.update != self.update or fit.forgetting != self.forgetting:
            raise ValueError(
                f"The fit used update={fit.update!r} and forgetting={fit.forgetting!r}, but this "
                f"estimator has update={self.update!r} and forgetting={self.forgetting!r}."
            )
        if fit.names != data.names or fit.sfreq != data.sfreq:
            raise ValueError("The fit was computed from other data (channels or sampling rate).")
        return self._result(data, fit, stacklevel=4)

    def _result(self, data: TimeSeries, fit: TvarFit, *, stacklevel: int) -> Connectivity:
        freqs = frequency_grid(data.sfreq, n_freqs=self.n_freqs, fmax=self.fmax, freqs=self.freqs)
        values = self._measure(fit, freqs)

        checks = [
            self._sampling(fit, data, stacklevel + 1),
            *tvar_checks(fit, stacklevel=stacklevel),
        ]
        details = fit.details()
        details["time_step_seconds"] = fit.time_step / data.sfreq
        provenance = self._provenance(
            data,
            model_order=fit.order,
            preprocessing={
                "standardized": self.standardize,
                "scale": None if fit.scale is None else fit.scale.tolist(),
                "demeaned": fit.standardize is not None,
            },
            details=details,
            checks=checks,
        )
        return Connectivity.from_array(
            values, source=list(data.names), freqs=freqs, times=fit.times, provenance=provenance
        )

    def _sampling(self, fit: TvarFit, data: TimeSeries, stacklevel: int) -> AssumptionCheck:
        """Coefficients per equation against the effective number of observations.

        The recursion's measured memory (:attr:`~dirconn.tvmvar.TvarFit.memory_samples`) counts as
        the observations of every trial (at most the trial itself); averaging over trials
        multiplies that by the number of trials.
        """
        per_trial = data.n_times - fit.order
        memory = fit.memory_samples
        effective = max(1, round(fit.n_trials * min(per_trial, memory)))
        return sampling_check(fit.n_nodes * fit.order, effective, stacklevel=stacklevel)


@dataclass(frozen=True)
class ADTF(_TimeVaryingVAR):
    """Adaptive directed transfer function: the DTF at every time point.

    ``values[i, j, f, t]`` is the share of the signal reaching channel ``j`` at frequency ``f`` and
    time ``t`` that comes from channel ``i``, computed from the time-varying MVAR. Like the DTF it
    mixes direct and indirect paths and is predictive, not interventional. The recursion
    estimates the coefficients at every sample, so the time resolution is set by the forgetting
    factor through the measured memory of the recursion, not by a window; a switch is followed
    with a lag of that order, and values just after a switch still carry the previous state.

    The default recursion is ``update="dual"``. See :class:`dirconn.tvmvar.TvarFit` for what is
    recorded, and the parameters of this class's base for the settings.
    """

    info = MethodInfo(
        name="adtf",
        backend=_BACKEND,
        directed=True,
        min_nodes=2,
        value_range=(0.0, 1.0),
        scale_dependent=True,
        frequency_resolved=True,
        time_resolved=True,
        assumptions=STANDARD_ASSUMPTIONS,
    )

    def _measure(self, fit: TvarFit, freqs: np.ndarray) -> np.ndarray:
        return dtf_from_coefs_series(fit.coefs, freqs, fit.sfreq)


@dataclass(frozen=True)
class TPDC(_TimeVaryingVAR):
    """Time-varying partial directed coherence: the PDC at every time point.

    ``values[i, j, f, t]`` is the share of the outflow of channel ``i`` at frequency ``f`` and time
    ``t`` that goes to channel ``j``, built from the time-varying coefficients and therefore
    reflecting direct influences only. The default recursion is ``update="dekf"``, which suits
    recordings with observation noise (as in the literature on temporal PDC). The amplitude form
    of the old package is ``squared=False``; the default here is the squared form, as in
    :class:`~dirconn.PDC`.

    Parameters
    ----------
    squared : bool
        ``True`` (default): the squared form, whose values sum to 1 over the targets. ``False``:
        the amplitude form, its square root.

    The other parameters are those of the base class.
    """

    info = MethodInfo(
        name="tpdc",
        backend=_BACKEND,
        directed=True,
        min_nodes=2,
        value_range=(0.0, 1.0),
        scale_dependent=True,
        frequency_resolved=True,
        time_resolved=True,
        assumptions=STANDARD_ASSUMPTIONS,
    )

    update: str = "dekf"
    squared: bool = True

    def _measure(self, fit: TvarFit, freqs: np.ndarray) -> np.ndarray:
        return pdc_from_coefs_series(fit.coefs, freqs, fit.sfreq, squared=self.squared)


def adtf(data: TimeSeries, **params: Any) -> Connectivity:
    """Adaptive DTF of ``data``; shorthand for ``ADTF(**params).fit(data)``."""
    return ADTF(**params).fit(data)


def tpdc(data: TimeSeries, **params: Any) -> Connectivity:
    """Time-varying PDC of ``data``; shorthand for ``TPDC(**params).fit(data)``."""
    return TPDC(**params).fit(data)
