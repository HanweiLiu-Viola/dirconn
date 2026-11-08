"""Time-domain Granger causality on the pooled least-squares VAR core."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from dirconn.estimator import (
    Estimator,
    MethodInfo,
    check_switches,
    order_boundary_check,
    residual_whiteness_check,
    sampling_check,
    stability_check,
)
from dirconn.mvar import (
    Masks,
    _check_rows,
    _prepare,
    _solve,
    check_order_spec,
    choose_order,
    fit_var,
    lagged_design,
    masks_of,
    row_mask,
)
from dirconn.provenance import STANDARD_ASSUMPTIONS, AssumptionCheck
from dirconn.result import Connectivity
from dirconn.statespace import state_space_gc
from dirconn.timeseries import TimeSeries

_PERFECT_FIT = 1e-12

GC_METHODS: tuple[str, ...] = ("refit", "state_space")
"""How :class:`GC` obtains the reduced models."""


def _check_not_perfect(sse: np.ndarray, y: np.ndarray, names: list[str], where: str) -> None:
    perfect = np.flatnonzero(sse <= _PERFECT_FIT * np.sum(y**2, axis=0))
    if perfect.size:
        raise ValueError(
            f"{where} predicts {[names[i] for i in perfect]} perfectly (zero residual); "
            "Granger causality is infinite or undefined for them."
        )


@dataclass(frozen=True)
class GC(Estimator):
    """Granger causality ``log(var_reduced / var_full)`` from a pooled least-squares VAR model.

    ``values[i, j]`` measures how much the past of channel ``i`` improves the prediction of channel
    ``j``: the log ratio of the residual sums of squares of the model without ``i`` and the model
    with it, in nats. Both models are fitted on the same observations (all trials, regressors never
    crossing a trial boundary) and at the same order, which is chosen **once**, on the full
    multivariate model. This is predictive influence, not interventional causality.

    With ``conditional=True`` the reduced model keeps every other channel, so an influence that
    runs through a third channel is not mistaken for a direct one. With ``conditional=False`` each
    ordered pair is judged on its own, from a bivariate model and the target's own past; this
    needs fewer coefficients but cannot tell direct from indirect influence. For two channels the
    modes coincide.

    ``method`` decides where the reduced model comes from. ``"refit"`` (the default) fits it again
    by least squares at the same order. The reduced process of a VAR model is in general a VARMA
    process, so a reduced VAR of the same order is misspecified and the value is biased upwards
    at short orders; the bias shrinks quickly as the order grows. ``"state_space"`` (Barnett and
    Seth 2015) fits only the model of all channels, writes it in innovations form and obtains
    every reduced model exactly from it by solving one discrete algebraic Riccati equation per
    conditioning set (see :mod:`dirconn.statespace`); the values are then non-negative and
    consistent at the true order. In the pairwise mode the "full" model of a pair is its bivariate
    subprocess of that same fit. :class:`SSGC` gives the matching frequency-resolved values.

    Data with ``valid`` or ``predict`` masks use only the rows those select (see
    :func:`dirconn.mvar.row_mask`), for the order selection and for every model.

    Parameters
    ----------
    order : int or {"aic", "bic", "hqc", "cv"}
        A fixed model order, or the criterion that selects it with :func:`dirconn.select_order`.
        The selection curves are stored in the provenance.
    max_order : int
        Largest candidate order when ``order`` is a criterion.
    conditional : bool
        Condition on all other channels (default) or judge channel pairs separately.
    method : {"refit", "state_space"}
        Refit every reduced model by least squares (default), or derive it from the fit of all
        channels through its state-space form. ``"state_space"`` needs a stable fit and raises
        ``ValueError`` when the stationarity check fails; the check is recorded in the
        provenance. :func:`dirconn.analytic_gc_test` (the F test of nested least-squares models)
        accepts only ``"refit"``.
    ridge : float
        Relative ridge penalty (see :func:`dirconn.fit_var`). With ``method="refit"`` the models
        are no longer nested and small negative values can occur. With ``"state_space"`` the
        penalised fit of all channels and its residual covariance define the model, and the
        reduced models follow from it, so the values stay non-negative.
    demean : bool
        Remove the per-channel mean over trials and time before fitting.
    n_folds : int
        Number of folds when ``order="cv"``.
    check_order_stability : bool
        Refit at longer orders and warn when the direction of a pair reverses (see
        :func:`dirconn.order_stability.order_stability_check`; the check always compares with
        least-squares refits). It costs a few extra fits; the resampling tests switch it off for
        the resampled fits.
    check_whiteness : bool
        Test the residuals of the model of all channels for autocorrelation and warn when they are
        clearly not white (see :func:`dirconn.estimator.residual_whiteness_check`). The resampling
        tests switch it off for the resampled fits.
    """

    info = MethodInfo(
        name="gc",
        backend="own numpy pooled OLS VAR, nested equation fits",
        directed=True,
        min_nodes=2,
        value_range=(0.0, math.inf),
        scale_dependent=False,
        assumptions=STANDARD_ASSUMPTIONS,
        supports_mask=True,
    )

    order: int | str = "bic"
    max_order: int = 10
    conditional: bool = True
    method: str = "refit"
    ridge: float = 0.0
    demean: bool = True
    n_folds: int = 5
    check_order_stability: bool = True
    check_whiteness: bool = True

    def __post_init__(self) -> None:
        """Validate the parameters."""
        super().__post_init__()
        check_order_spec(self.order, self.max_order, self.ridge, self.n_folds)
        check_switches(self, "check_order_stability", "check_whiteness")
        if self.method not in GC_METHODS:
            raise ValueError(f"method must be one of {GC_METHODS}, got {self.method!r}.")

    def _fit(self, data: TimeSeries) -> Connectivity:
        order, selection = choose_order(
            data,
            self.order,
            self.max_order,
            ridge=self.ridge,
            demean=self.demean,
            n_folds=self.n_folds,
        )

        x, _ = _prepare(data, self.demean)
        names = list(data.names)
        masks = masks_of(data)
        residuals: np.ndarray | None
        stability: AssumptionCheck | None = None
        if self.method == "state_space":
            values, n_obs, n_coefficients, residuals, stability = self._state_space(
                data, x, order, names, masks
            )
        elif self.conditional:
            values, n_obs, n_coefficients, residuals = self._conditional(x, order, names, masks)
        else:
            values, n_obs, n_coefficients = self._pairwise(x, order, names, masks)
            residuals = self._full_residuals(x, order, masks) if self.check_whiteness else None

        checks = [sampling_check(n_coefficients, n_obs, stacklevel=4)]
        if stability is not None:
            checks.append(stability)
        if self.check_whiteness and residuals is None:
            checks.append(
                AssumptionCheck(
                    "residual_whiteness",
                    "not_checked",
                    detail="the model of all channels could not be fitted at this order.",
                )
            )
        elif self.check_whiteness:
            rows = row_mask(data.valid, data.predict, order, data.n_times)
            checks.append(
                residual_whiteness_check(residuals, data.n_trials, order, rows=rows, stacklevel=4)
            )
        if selection is not None:
            checks.append(order_boundary_check(selection, stacklevel=4))
        if self.check_order_stability:
            from dirconn.order_stability import order_stability_check

            checks.append(
                order_stability_check(
                    data,
                    order,
                    demean=self.demean,
                    conditional=self.conditional,
                    stacklevel=4,
                )
            )
        provenance = self._provenance(
            data,
            model_order=order,
            details={
                "n_obs": n_obs,
                "conditional": self.conditional,
                "method": self.method,
                "order_selection": None if selection is None else selection.to_dict(),
            },
            checks=checks,
        )
        return Connectivity.from_array(values, source=names, provenance=provenance)

    def _state_space(
        self, data: TimeSeries, x: np.ndarray, order: int, names: list[str], masks: Masks
    ) -> tuple[np.ndarray, int, int, np.ndarray, AssumptionCheck]:
        """Fit the model of all channels once and derive every reduced model from it."""
        fit = fit_var(data, order, ridge=self.ridge, demean=self.demean)
        observed, _ = lagged_design(x, order, **masks)
        _check_not_perfect(np.sum(fit.residuals**2, axis=0), observed, names, "The full model")
        stability = stability_check(fit.coefs, stacklevel=5)
        if stability.status == "fail":
            raise ValueError(
                f"State-space Granger causality needs a stable VAR model, but {stability.detail} "
                "Use method='refit', check the data for trends or non-stationarity, or lower "
                "the order."
            )
        values = state_space_gc(fit.coefs, fit.noise_cov, conditional=self.conditional)
        return values, fit.n_obs, fit.n_channels * order, np.asarray(fit.residuals), stability

    def _full_residuals(self, x: np.ndarray, order: int, masks: Masks) -> np.ndarray | None:
        """Residuals of the model of all channels, for the whiteness check of the pairwise mode.

        ``None`` when that model cannot be fitted (too few rows or a rank-deficient design); the
        pairwise values do not need it.
        """
        y, design = lagged_design(x, order, **masks)
        if y.shape[0] <= design.shape[1]:
            return None
        try:
            _, residuals = _solve(y, design, self.ridge)
        except ValueError:
            return None
        return residuals

    def _conditional(
        self, x: np.ndarray, order: int, names: list[str], masks: Masks
    ) -> tuple[np.ndarray, int, int, np.ndarray]:
        n_channels = x.shape[1]
        y, design = lagged_design(x, order, **masks)
        _check_rows(y.shape[0], n_channels * order)
        try:
            _, full_residuals = _solve(y, design, self.ridge)
        except ValueError as exc:
            raise ValueError(f"Full model: {exc}") from exc
        full_sse = np.sum(full_residuals**2, axis=0)
        _check_not_perfect(full_sse, y, names, "The full model")

        values = np.full((n_channels, n_channels), np.nan)
        for source in range(n_channels):
            keep = [
                lag * n_channels + channel
                for lag in range(order)
                for channel in range(n_channels)
                if channel != source
            ]
            try:
                _, reduced_residuals = _solve(y, design[:, keep], self.ridge)
            except ValueError as exc:
                raise ValueError(f"Model without {names[source]!r}: {exc}") from exc
            reduced_sse = np.sum(reduced_residuals**2, axis=0)
            for target in range(n_channels):
                if target != source:
                    values[source, target] = math.log(reduced_sse[target] / full_sse[target])
        return values, int(y.shape[0]), n_channels * order, full_residuals

    def _pairwise(
        self, x: np.ndarray, order: int, names: list[str], masks: Masks
    ) -> tuple[np.ndarray, int, int]:
        n_channels = x.shape[1]
        values = np.full((n_channels, n_channels), np.nan)
        n_obs = 0
        for first in range(n_channels):
            for second in range(first + 1, n_channels):
                label = f"Pair {names[first]!r} and {names[second]!r}"
                y, design = lagged_design(x[:, [first, second], :], order, **masks)
                n_obs = int(y.shape[0])
                _check_rows(n_obs, 2 * order)
                try:
                    _, full_residuals = _solve(y, design, self.ridge)
                except ValueError as exc:
                    raise ValueError(f"{label}: {exc}") from exc
                full_sse = np.sum(full_residuals**2, axis=0)
                _check_not_perfect(full_sse, y, [names[first], names[second]], label)
                for column, (source, target) in enumerate(((second, first), (first, second))):
                    own_past = [lag * 2 + column for lag in range(order)]
                    try:
                        _, reduced = _solve(y[:, [column]], design[:, own_past], self.ridge)
                    except ValueError as exc:
                        raise ValueError(f"{label}: {exc}") from exc
                    values[source, target] = math.log(np.sum(reduced**2) / full_sse[column])
        return values, n_obs, 2 * order


def gc(data: TimeSeries, **params: Any) -> Connectivity:
    """Granger causality of ``data``; shorthand for ``GC(**params).fit(data)``.

    See :class:`GC` for the parameters.
    """
    return GC(**params).fit(data)
