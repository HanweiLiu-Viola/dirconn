"""Time reversal: the time-reversed Granger estimator and a per-cell asymmetry diagnostic.

Reversing time turns a lagged influence ``i -> j`` into ``j -> i`` but leaves an instantaneous
(zero-lag) mixture as it is, because the mixture has the same structure in both directions of
time. Two tools use this, named apart:

- :class:`TRGC`, the difference-based time-reversed Granger causality of Haufe et al. (2013) and
  Winkler et al. (2016): the net Granger flow of the data minus that of its time reversal.
- :func:`time_reversal_asymmetry`, the heuristic of the old package: any estimator on the data
  and on its reversal, compared cell by cell. It is a diagnostic, not an estimator or a test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from dirconn.estimator import Estimator, MethodInfo, check_switches
from dirconn.granger import GC
from dirconn.mvar import check_order_spec
from dirconn.provenance import STANDARD_ASSUMPTIONS
from dirconn.result import Connectivity
from dirconn.timeseries import TimeSeries


def reverse_time(data: TimeSeries) -> TimeSeries:
    """Return ``data`` with every trial reversed in time (same channels, segmentation, tmin)."""
    return TimeSeries(
        data=np.ascontiguousarray(np.asarray(data.data)[:, :, ::-1]),
        sfreq=data.sfreq,
        channels=data.channels,
        segmentation=data.segmentation,
        tmin=data.tmin,
    )


def _as_list(matrix: np.ndarray) -> list[list[float | None]]:
    """Return a matrix as nested lists for the provenance, NaN written as None."""
    return [[None if math.isnan(v) else float(v) for v in row] for row in matrix]


def _net(values: np.ndarray) -> np.ndarray:
    net = values - values.T
    np.fill_diagonal(net, np.nan)
    return net


@dataclass(frozen=True)
class TRGC(Estimator):
    """Time-reversed Granger causality, difference-based (Haufe et al. 2013; Winkler et al. 2016).

    ``values[i, j] = (F_ij - F_ji) - (R_ij - R_ji)``, where ``F`` is :class:`GC` on the data and
    ``R`` is :class:`GC` on the data reversed in time. A lagged influence of ``i`` on ``j`` makes
    the first bracket positive and, because the reversal turns it around, the second negative, so
    both add up; an instantaneous mixture makes both brackets alike, so it cancels. The result is
    antisymmetric (``values[j, i] = -values[i, j]``), in nats, with an undefined (NaN) diagonal;
    a positive value means net predictive flow from ``i`` to ``j``. This is predictive influence,
    not interventional causality.

    The order is chosen once, on the data, and reused for the reversed data. The two net flows are
    recorded in the provenance (``net_forward`` and ``net_reversed``, NaN as ``null``), and so are
    the two Granger matrices.

    Parameters
    ----------
    order, max_order, conditional, ridge, demean, n_folds, check_order_stability, check_whiteness
        As for :class:`GC`; ``conditional`` decides whether both Granger fits condition on all
        other channels. The order-stability and whiteness checks run on the forward fit only.
    """

    info = MethodInfo(
        name="trgc",
        backend="own numpy pooled OLS VAR (GC) on the data and on its time reversal",
        directed=True,
        antisymmetric=True,
        min_nodes=2,
        value_range=(-math.inf, math.inf),
        scale_dependent=False,
        assumptions=STANDARD_ASSUMPTIONS,
    )

    order: int | str = "bic"
    max_order: int = 10
    conditional: bool = True
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

    def _gc(self, order: int | str, *, forward: bool) -> GC:
        return GC(
            order=order,
            max_order=self.max_order,
            conditional=self.conditional,
            ridge=self.ridge,
            demean=self.demean,
            n_folds=self.n_folds,
            check_order_stability=forward and self.check_order_stability,
            check_whiteness=forward and self.check_whiteness,
        )

    def _fit(self, data: TimeSeries) -> Connectivity:
        forward = self._gc(self.order, forward=True).fit(data)
        order = forward.provenance.model_order
        assert order is not None
        backward = self._gc(order, forward=False).fit(reverse_time(data))
        f = forward.values[:, :, 0, 0]
        r = backward.values[:, :, 0, 0]
        net_forward, net_reversed = _net(f), _net(r)
        values = net_forward - net_reversed
        checks = [c for c in forward.provenance.assumptions.checks if c.status != "not_checked"]
        provenance = self._provenance(
            data,
            model_order=order,
            details={
                "conditional": self.conditional,
                "n_obs": forward.provenance.details.get("n_obs"),
                "order_selection": forward.provenance.details.get("order_selection"),
                "gc_forward": _as_list(f),
                "gc_reversed": _as_list(r),
                "net_forward": _as_list(net_forward),
                "net_reversed": _as_list(net_reversed),
            },
            checks=checks,
        )
        return Connectivity.from_array(values, source=list(data.names), provenance=provenance)


def trgc(data: TimeSeries, **params: Any) -> Connectivity:
    """Time-reversed Granger causality of ``data``; shorthand for ``TRGC(**params).fit(data)``."""
    return TRGC(**params).fit(data)


@dataclass(frozen=True)
class TimeReversalAsymmetry:
    """Result of :func:`time_reversal_asymmetry`.

    Attributes
    ----------
    asymmetry : Connectivity
        ``(orig - rev) / (|orig| + |rev|)`` per cell, between -1 and 1; NaN where both are zero
        or either is undefined.
    original, reversed : Connectivity
        The estimator's result on the data and on the time-reversed data.
    """

    asymmetry: Connectivity
    original: Connectivity
    reversed: Connectivity

    @property
    def robustness(self) -> np.ndarray:
        """``|asymmetry|`` clipped to [0, 1]: how much a cell changes when time is reversed."""
        return np.clip(np.abs(self.asymmetry.values), 0.0, 1.0)


def time_reversal_asymmetry(estimator: Estimator, data: TimeSeries) -> TimeReversalAsymmetry:
    """Compare an estimator's result on the data with its result on the time-reversed data.

    A heuristic, not a test (there is no p-value): a cell whose value comes from a lagged
    influence changes when time runs backwards, while one that comes from an instantaneous
    mixture stays about the same. For a directed measure, a true edge ``i -> j`` gives a positive
    asymmetry at ``[i, j]``. Large values do not prove an influence and small values do not rule
    one out; use it next to :class:`TRGC` or a significance test.

    Ported from the old package's ``TimeReversalTest``; the one change is that a cell where both
    values are zero is NaN instead of 0 (the old code divided by ``1e-12``).

    Parameters
    ----------
    estimator : Estimator
        Any estimator with real-valued results.
    data : TimeSeries
        Input data.

    Returns
    -------
    TimeReversalAsymmetry
    """
    if not isinstance(estimator, Estimator):
        raise TypeError(f"estimator must be a dirconn Estimator, got {type(estimator).__name__}.")
    original = estimator.fit(data)
    reversed_ = estimator.fit(reverse_time(data))
    if original.is_complex or reversed_.is_complex:
        raise ValueError("time_reversal_asymmetry needs real values; take abs() or real first.")
    a, b = original.values, reversed_.values
    denominator = np.abs(a) + np.abs(b)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(
            denominator > 0, (a - b) / np.where(denominator > 0, denominator, 1), np.nan
        )
    asymmetry = Connectivity(
        values=score,
        source=original.source,
        target=original.target,
        freqs=original.freqs,
        times=original.times,
        provenance=original.provenance.with_step(
            {
                "op": "time_reversal_asymmetry",
                "reversed_input_hash": reversed_.provenance.input_hash,
            }
        ),
    )
    return TimeReversalAsymmetry(asymmetry=asymmetry, original=original, reversed=reversed_)
