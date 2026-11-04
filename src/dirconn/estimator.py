"""Base class and metadata shared by all estimators."""

from __future__ import annotations

import dataclasses
import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy import stats

from dirconn.exceptions import DirconnWarning
from dirconn.mvar import OrderSelection
from dirconn.provenance import (
    STANDARD_ASSUMPTIONS,
    AssumptionCheck,
    AssumptionChecklist,
    Provenance,
    array_hash,
)
from dirconn.result import Connectivity
from dirconn.spectral import spectral_radius
from dirconn.timeseries import TimeSeries

SCALE_RATIO_WARN = 10.0
"""Ratio between the largest and smallest channel standard deviation that triggers a warning."""

SAMPLING_RATIO_WARN = 0.1
"""Coefficients per equation divided by observations above which estimates become unreliable."""

STABILITY_RATIO_WARN = 0.99
"""Spectral radius of a fitted VAR model from which it counts as close to a unit root."""


def stability_check(coefs: Any, *, stacklevel: int = 3) -> AssumptionCheck:
    """Check that a fitted VAR model is stationary, through its spectral radius.

    A radius below 0.99 passes. From 0.99 to 1 the model is close to a unit root and the result
    is ``"warn"``; at 1 or above the model is unstable and the result is ``"fail"``. Both raise a
    :class:`DirconnWarning` and neither stops the run: a transfer function that needs a stationary
    model raises its own error where it cannot be computed.

    Parameters
    ----------
    coefs : array_like, shape (order, n_channels, n_channels)
        VAR coefficients ``[lag - 1, target, source]``.
    stacklevel : int
        Stack level passed to :func:`warnings.warn`.

    Returns
    -------
    AssumptionCheck
        The ``"stationarity"`` entry for the result's assumption checklist; ``value`` is the
        spectral radius.
    """
    radius = spectral_radius(coefs)
    if radius >= 1.0:
        detail = f"the fitted VAR model is unstable (spectral radius {radius:.4g} >= 1)."
        status = "fail"
    elif radius >= STABILITY_RATIO_WARN:
        detail = (
            f"the fitted VAR model is close to a unit root (spectral radius {radius:.4g}, "
            f"limit {STABILITY_RATIO_WARN:g})."
        )
        status = "warn"
    else:
        return AssumptionCheck(
            "stationarity",
            "pass",
            detail=f"the fitted VAR model is stable (spectral radius {radius:.4g}).",
            value=radius,
        )
    warnings.warn(
        f"{detail[0].upper()}{detail[1:]} The data may be non-stationary; spectral measures built "
        "on this model are unreliable.",
        DirconnWarning,
        stacklevel=stacklevel,
    )
    return AssumptionCheck("stationarity", status, detail=detail, value=radius)


def order_boundary_check(selection: OrderSelection, *, stacklevel: int = 3) -> AssumptionCheck:
    """Check that an order selection did not stop at the largest candidate.

    When the criterion picks ``max_order`` its minimum may lie beyond the range that was searched,
    which is typical for coupling through a band-pass filter or for long delays. The result is
    ``"warn"`` with a :class:`DirconnWarning`; a search with a single candidate is no choice and
    passes.

    Parameters
    ----------
    selection : OrderSelection
        The selection to inspect.
    stacklevel : int
        Stack level passed to :func:`warnings.warn`.

    Returns
    -------
    AssumptionCheck
        The ``"order_selection"`` entry for the result's assumption checklist; ``value`` is the
        selected order.
    """
    if selection.max_order > 1 and selection.order == selection.max_order:
        detail = (
            f"the {selection.criterion} criterion chose the largest candidate order "
            f"({selection.order}); the best order may be higher, consider raising max_order."
        )
        warnings.warn(f"Order selection: {detail}", DirconnWarning, stacklevel=stacklevel)
        return AssumptionCheck("order_selection", "warn", detail=detail, value=selection.order)
    return AssumptionCheck(
        "order_selection",
        "pass",
        detail=f"the {selection.criterion} criterion chose order {selection.order} of "
        f"{selection.max_order} candidates.",
        value=selection.order,
    )


def sampling_check(n_coefficients: int, n_obs: int, *, stacklevel: int = 3) -> AssumptionCheck:
    """Check that a regression has enough observations for the coefficients it estimates.

    The rule of thumb is at most one coefficient per ten observations. Above it the check is
    ``"warn"`` and a :class:`DirconnWarning` is raised; the diagnostic never stops the run.

    Parameters
    ----------
    n_coefficients : int
        Coefficients estimated per equation, for example ``n_channels * order`` of a VAR model.
    n_obs : int
        Number of observations (regression rows) the equations are fitted on.
    stacklevel : int
        Stack level passed to :func:`warnings.warn`.

    Returns
    -------
    AssumptionCheck
        The ``"sampling_adequacy"`` entry for the result's assumption checklist; ``value`` is
        ``n_coefficients / n_obs``.
    """
    if n_coefficients <= 0 or n_obs <= 0:
        raise ValueError(
            f"n_coefficients and n_obs must be positive, got {n_coefficients} and {n_obs}."
        )
    ratio = n_coefficients / n_obs
    detail = (
        f"{n_coefficients} coefficients per equation are estimated from {n_obs} observations "
        f"(ratio {ratio:.3g}; the rule of thumb is at most {SAMPLING_RATIO_WARN:g})."
    )
    if ratio > SAMPLING_RATIO_WARN:
        warnings.warn(
            f"{detail} Estimates will be unreliable; use more data, a lower order, fewer channels "
            "or a ridge penalty.",
            DirconnWarning,
            stacklevel=stacklevel,
        )
        return AssumptionCheck("sampling_adequacy", "warn", detail=detail, value=ratio)
    return AssumptionCheck("sampling_adequacy", "pass", detail=detail, value=ratio)


WHITENESS_EXTRA_LAGS = 10
"""Lags beyond the model order that the whiteness check tests: ``h = order + 10``."""

WHITENESS_LEVEL = 0.05
"""Level of the portmanteau test below which the whiteness check can warn."""

WHITENESS_MIN_AUTOCORR = 0.1
"""Largest absolute residual autocorrelation above which a significant test warns.

Chosen from simulations: a too-low order warned in every fit (in 95 % of them the
largest autocorrelation was above 0.2); an adequate order warned in 6.5 % of fits of 600 samples
(the level of the test itself), 1.8 % at 1000 and never from 3000 on; the mild misfit that white
observation noise leaves at the BIC order (largest autocorrelation 0.02 to 0.09) warned in 5 % of
30 s recordings and in none of 120 s.
"""


def check_switches(estimator: Any, *names: str) -> None:
    """Raise ``TypeError`` unless each named parameter of ``estimator`` is a bool."""
    for name in names:
        value = getattr(estimator, name)
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool, got {value!r}.")


def residual_whiteness_check(
    residuals: Any, n_trials: int, order: int, *, stacklevel: int = 3
) -> AssumptionCheck:
    """Check that the residuals of a VAR fit are white, with a multivariate portmanteau test.

    The statistic is the modified portmanteau statistic of Hosking (1980) over the lags
    ``s = 1..h``, ``h = order + 10``, with autocovariances ``C_s`` computed within trials only
    (``N`` residual rows, ``K`` channels, residuals centred over all rows)::

        Q = N^2 sum_s tr(C_s^T C_0^-1 C_s C_0^-1) / (N - n_trials s),    df = K^2 (h - order)

    For one trial it equals statsmodels ``VARResults.test_whiteness(nlags=h, adjusted=True)``.
    On a long recording the test rejects for a negligible misfit, so the check warns, with a
    :class:`DirconnWarning`, only when the p-value is below 0.05 **and** the largest absolute
    residual autocorrelation ``C_s[a, b] / sqrt(C_0[a, a] C_0[b, b])`` over those lags exceeds
    ``WHITENESS_MIN_AUTOCORR``. It never fails and never stops the run.

    Parameters
    ----------
    residuals : array_like, shape (n_obs, n_channels)
        Residuals of a pooled fit, trial after trial, the same number of rows per trial.
    n_trials : int
        Number of trials the rows belong to.
    order : int
        Order of the fitted model.
    stacklevel : int
        Stack level passed to :func:`warnings.warn`.

    Returns
    -------
    AssumptionCheck
        The ``"residual_whiteness"`` entry for the result's assumption checklist; ``value`` is
        the p-value, or ``None`` when the trials are too short for the lags.
    """
    u = np.asarray(residuals, dtype=float)
    n_obs, n_channels = u.shape
    if n_trials < 1 or n_obs % n_trials:
        raise ValueError(f"{n_obs} residual rows cannot be split into {n_trials} equal trials.")
    rows = n_obs // n_trials
    lags = order + WHITENESS_EXTRA_LAGS
    if rows <= lags:
        return AssumptionCheck(
            "residual_whiteness",
            "not_checked",
            detail=f"trials of {rows} residuals are too short for the {lags} lags of the test.",
        )
    blocks = (u - u.mean(axis=0)).reshape(n_trials, rows, n_channels)
    cov0 = np.einsum("tra,trb->ab", blocks, blocks) / n_obs
    try:
        cov0_inv = np.linalg.inv(cov0)
    except np.linalg.LinAlgError:
        return AssumptionCheck(
            "residual_whiteness", "not_checked", detail="the residual covariance is singular."
        )
    scale = np.sqrt(np.diag(cov0))
    statistic = 0.0
    largest = 0.0
    for lag in range(1, lags + 1):
        cov = np.einsum("tra,trb->ab", blocks[:, lag:], blocks[:, :-lag]) / n_obs
        statistic += float(np.trace(cov.T @ cov0_inv @ cov @ cov0_inv)) / (n_obs - n_trials * lag)
        largest = max(largest, float(np.max(np.abs(cov / np.outer(scale, scale)))))
    statistic *= n_obs**2
    df = n_channels**2 * (lags - order)
    pvalue = float(stats.chi2.sf(statistic, df))
    detail = (
        f"Hosking portmanteau statistic {statistic:.4g} over lags 1 to {lags} (df {df}, "
        f"p = {pvalue:.3g}); largest absolute residual autocorrelation {largest:.3g} "
        f"(warning when p < {WHITENESS_LEVEL:g} and it exceeds {WHITENESS_MIN_AUTOCORR:g})."
    )
    if pvalue < WHITENESS_LEVEL and largest > WHITENESS_MIN_AUTOCORR:
        warnings.warn(
            f"The residuals of the VAR fit are not white: {detail} The model order may be too "
            "low or the process not a finite VAR (for example with observation noise); "
            "estimates and asymptotic tests built on it are less reliable.",
            DirconnWarning,
            stacklevel=stacklevel,
        )
        return AssumptionCheck("residual_whiteness", "warn", detail=detail, value=pvalue)
    return AssumptionCheck("residual_whiteness", "pass", detail=detail, value=pvalue)


@dataclass(frozen=True)
class MethodInfo:
    """Structured description of a method, enforced at run time.

    Parameters
    ----------
    name : str
        Method name recorded in the provenance, for example ``"granger"``.
    backend : str
        The implementation actually used, for example ``"own numpy OLS MVAR"``. Stating it
        avoids a method name hiding what is computed.
    directed : bool
        Whether ``values[i, j] != values[j, i]`` in general.
    antisymmetric : bool
        Whether ``values[j, i] = -values[i, j]`` by construction (a net flow or a phase), so one
        number describes a pair and a pair coupled both ways cannot be shown as two edges.
    signed : bool
        Whether the sign of a value is the sign of the coupling (a correlation), not its
        strength: a strong negative value is a strong link.
    min_nodes, max_nodes : int
        Allowed number of channels; ``max_nodes=None`` means no upper limit. A directed measure
        needs at least two channels.
    value_range : (float, float), optional
        Range of the values, ``inf`` allowed, for documentation and plotting.
    scale_dependent : bool
        Whether the values depend on the channel amplitudes. Such methods warn when the channel
        scales differ strongly.
    complex_valued : bool
        Whether the result is complex.
    frequency_resolved, time_resolved : bool
        Whether the result has a real frequency or time axis.
    assumptions : tuple of str
        Standard assumptions the method relies on (names from ``STANDARD_ASSUMPTIONS``).
    """

    name: str
    backend: str
    directed: bool
    antisymmetric: bool = False
    signed: bool = False
    min_nodes: int = 2
    max_nodes: int | None = None
    value_range: tuple[float, float] | None = None
    scale_dependent: bool = False
    complex_valued: bool = False
    frequency_resolved: bool = False
    time_resolved: bool = False
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and normalise the fields."""
        for field_name in ("name", "backend"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
        if self.min_nodes < 2:
            raise ValueError(f"min_nodes must be at least 2, got {self.min_nodes}.")
        if self.max_nodes is not None and self.max_nodes < self.min_nodes:
            raise ValueError(
                f"max_nodes must be None or at least min_nodes ({self.min_nodes}), "
                f"got {self.max_nodes}."
            )
        if self.value_range is not None:
            try:
                low, high = (float(v) for v in self.value_range)
            except (TypeError, ValueError):
                raise ValueError(
                    f"value_range must be (low, high), got {self.value_range!r}."
                ) from None
            if math.isnan(low) or math.isnan(high) or low > high:
                raise ValueError(f"value_range must satisfy low <= high, got {self.value_range!r}.")
            object.__setattr__(self, "value_range", (low, high))
        assumptions = tuple(self.assumptions)
        unknown = [a for a in assumptions if a not in STANDARD_ASSUMPTIONS]
        if unknown:
            raise ValueError(
                f"assumptions must be among {STANDARD_ASSUMPTIONS}, got unknown {unknown}."
            )
        object.__setattr__(self, "assumptions", assumptions)


class Estimator(ABC):
    """Base class for all directed-connectivity estimators.

    A concrete estimator is a frozen dataclass whose fields are its typed parameters, with a
    class attribute ``info`` (a :class:`MethodInfo`) and a ``_fit`` method::

        @dataclass(frozen=True)
        class Granger(Estimator):
            info = MethodInfo(name="granger", backend="own numpy OLS MVAR", directed=True)
            order: int = 5

            def _fit(self, data: TimeSeries) -> Connectivity: ...

    Using a dataclass means unknown keyword arguments raise ``TypeError``, parameters have types
    and defaults the IDE can show, and an estimator never changes after construction.
    Subclasses that override ``__post_init__`` must call ``super().__post_init__()``.

    :meth:`fit` is stateless: it returns the result and the estimator keeps nothing.
    """

    info: ClassVar[MethodInfo]

    def __post_init__(self) -> None:
        """Check that the subclass is defined correctly."""
        self._validate_definition()

    def _validate_definition(self) -> None:
        cls = type(self)
        if not isinstance(getattr(cls, "info", None), MethodInfo):
            raise TypeError(f"{cls.__name__} must define a class attribute 'info' (a MethodInfo).")
        params = getattr(cls, "__dataclass_params__", None)
        if params is None or not params.frozen:
            raise TypeError(
                f"{cls.__name__} must be a frozen dataclass: decorate it with "
                "@dataclass(frozen=True)."
            )

    def get_params(self) -> dict[str, Any]:
        """Return the constructor parameters as a dictionary."""
        self._validate_definition()
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.init}  # type: ignore[arg-type]

    def fit(self, data: TimeSeries) -> Connectivity:
        """Estimate directed connectivity.

        Checks the number of channels (a violation raises ``ValueError``), runs the diagnostics
        (these warn with :class:`DirconnWarning` and are recorded in the result's assumption
        checklist; they never stop the computation) and then calls the estimator.

        Parameters
        ----------
        data : TimeSeries
            Input data.

        Returns
        -------
        Connectivity
            The result, with provenance.
        """
        self._validate_definition()
        if not isinstance(data, TimeSeries):
            raise TypeError(
                "fit expects a TimeSeries; build one with TimeSeries.from_array or "
                f"TimeSeries.from_mne, got {type(data).__name__}."
            )
        checks = self._guard(data)
        result = self._fit(data)
        if not isinstance(result, Connectivity):
            raise TypeError(
                f"{type(self).__name__}._fit must return a Connectivity, "
                f"got {type(result).__name__}."
            )
        if not checks:
            return result
        assumptions = result.provenance.assumptions
        for check in checks:
            assumptions = assumptions.with_check(check)
        provenance = dataclasses.replace(result.provenance, assumptions=assumptions)
        return result.with_provenance(provenance)

    @abstractmethod
    def _fit(self, data: TimeSeries) -> Connectivity:
        """Compute the result; implemented by every concrete estimator."""

    def _extra_checks(self, data: TimeSeries) -> list[AssumptionCheck]:
        """Return method-specific diagnostics; override to add some. Must not raise."""
        return []

    def _depends_on_scale(self) -> bool:
        """Whether this fit depends on the channel amplitudes; defaults to ``info``.

        Override when a parameter changes the answer, for example a method that standardises the
        channels itself does not depend on their scale while it does so.
        """
        return self.info.scale_dependent

    def _provenance(
        self,
        data: TimeSeries,
        *,
        model_order: int | None = None,
        seed: int | None = None,
        preprocessing: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
        checks: Sequence[AssumptionCheck] = (),
    ) -> Provenance:
        """Build the provenance for a result computed from ``data``.

        ``details`` holds method-specific derived facts (order-selection curves, number of
        observations) and ``checks`` are diagnostics found while estimating; both end up in the
        result's provenance.
        """
        assumptions = AssumptionChecklist.unchecked()
        for check in checks:
            assumptions = assumptions.with_check(check)
        return Provenance(
            method=self.info.name,
            parameters=self.get_params(),
            model_order=model_order,
            seed=seed,
            backend=self.info.backend,
            input_hash=array_hash(data.data),
            preprocessing=preprocessing or {},
            assumptions=assumptions,
            details=details or {},
        )

    def _guard(self, data: TimeSeries) -> list[AssumptionCheck]:
        info = self.info
        n = data.n_channels
        if n < info.min_nodes:
            raise ValueError(f"{info.name} needs at least {info.min_nodes} channels, got {n}.")
        if info.max_nodes is not None and n > info.max_nodes:
            raise ValueError(f"{info.name} accepts at most {info.max_nodes} channels, got {n}.")
        checks: list[AssumptionCheck] = []
        if self._depends_on_scale():
            checks.append(self._scale_check(data))
        checks.extend(self._extra_checks(data))
        return checks

    def _scale_check(self, data: TimeSeries) -> AssumptionCheck:
        flat = np.moveaxis(data.data, 1, 0).reshape(data.n_channels, -1)
        std = flat.std(axis=1)
        smin, smax = float(np.min(std)), float(np.max(std))
        if smin == 0.0:
            flat_names = [data.names[i] for i in np.flatnonzero(std == 0.0)]
            detail = (
                f"channels {flat_names} are flat (zero variance); "
                f"{self.info.name} is scale dependent."
            )
            warnings.warn(detail, DirconnWarning, stacklevel=4)
            return AssumptionCheck("channel_scale", "warn", detail=detail, value=None)
        ratio = smax / smin
        if ratio > SCALE_RATIO_WARN:
            detail = (
                f"{self.info.name} depends on channel amplitudes but the channel standard "
                f"deviations differ by a factor of {ratio:.3g}; standardise the channels "
                "(z-score) before fitting."
            )
            warnings.warn(detail, DirconnWarning, stacklevel=4)
            return AssumptionCheck("channel_scale", "warn", detail=detail, value=ratio)
        return AssumptionCheck(
            "channel_scale",
            "pass",
            detail=f"channel standard deviations differ by a factor of {ratio:.3g}.",
            value=ratio,
        )


def method_info(name: str) -> MethodInfo | None:
    """Return the :class:`MethodInfo` of the estimator class whose ``info.name`` is ``name``.

    A result records only its method name (``provenance.method``); this finds the description
    of that method among the estimator classes that are loaded, for code that reads results,
    such as the plots. Returns ``None`` for a name that no loaded estimator has.
    """
    stack: list[type] = list(Estimator.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        info = cls.__dict__.get("info")
        if isinstance(info, MethodInfo) and info.name == name:
            return info
    return None
