"""Check that the direction of a connection survives a longer model.

An influence confined to a narrow band, on a background with a long memory, cannot be summarised by
a short autoregressive model, and a short model can hand the memory to the other channel: the
direction then comes out reversed while every value is small and no other diagnostic fires.
Measured on real EEG (``docs/benchmark.md``), Granger causality of order 8 recovered the
constructed direction in 7 of 24 cases, orders 32 and above in all 24.

The check refits the Granger model at longer orders (multiples of the chosen order, as far as the
data allow) and uses the exact F test of nested models (:func:`dirconn.analytic_gc_test`). A pair is
flagged when, at some longer order, one direction is significant at ``alpha`` and the other is not
significant at ``alpha_other``, while the chosen order favours the opposite direction.
Independent real channels were flagged in 0 of 30 pairs. Where the chosen direction was wrong
the check flagged 76 and 74 percent of the cases at orders 4 and 8 (strength 1) and 60 and 74
percent (strength 3), and 1 of 90 correct ones. It is a warning about the direction of a pair,
not a proof, and it cannot see a reversal that no longer model resolves. It costs two to four
times the fit it accompanies.
"""

from __future__ import annotations

import warnings

import numpy as np

from dirconn.estimator import SAMPLING_RATIO_WARN
from dirconn.exceptions import DirconnWarning
from dirconn.provenance import AssumptionCheck
from dirconn.rng import _as_count
from dirconn.timeseries import TimeSeries

#: Multiples of the chosen order that are tried.
MULTIPLIERS: tuple[int, ...] = (2, 4, 8, 16, 32)
#: A direction counts as established when its F-test p-value is below this ...
ALPHA = 0.001
#: ... and the opposite direction is not significant at this level.
ALPHA_OTHER = 0.05
#: Largest number of regressors per equation (``order * n_channels``) that is refitted. The
#: cost grows with its square: 512 made the check 14 times slower than the fit on four channels,
#: 160 two to four times, for a few percent less sensitivity.
MAX_REGRESSORS = 160
#: Pairs named in a warning before it says "and N more".
_NAMED_PAIRS = 3


def candidate_orders(order: int, *, n_channels: int, n_times: int, n_trials: int) -> list[int]:
    """Return the longer orders the data allow, in increasing order.

    An order ``k`` is allowed if it is a multiple of ``order`` from :data:`MULTIPLIERS`, a
    regression row never crosses a trial boundary (``k < n_times``), there is at most one
    coefficient per ten observations (``k * n_channels <= 0.1 * n_trials * (n_times - k)``) and at
    most :data:`MAX_REGRESSORS` regressors per equation.
    """
    allowed = []
    for multiple in MULTIPLIERS:
        k = order * multiple
        if k >= n_times:
            continue
        n_obs = n_trials * (n_times - k)
        if k * n_channels <= SAMPLING_RATIO_WARN * n_obs and k * n_channels <= MAX_REGRESSORS:
            allowed.append(k)
    return allowed


def order_stability_check(
    data: TimeSeries,
    order: int,
    *,
    demean: bool = True,
    conditional: bool = True,
    stacklevel: int = 3,
) -> AssumptionCheck:
    """Check that the direction of every pair at ``order`` agrees with longer models.

    Parameters
    ----------
    data : TimeSeries
        The data the model of order ``order`` was fitted to.
    order : int
        The chosen model order.
    demean, conditional : bool
        As in :class:`dirconn.GC`; the check always uses ordinary least squares.
    stacklevel : int
        Stack level passed to :func:`warnings.warn`.

    Returns
    -------
    AssumptionCheck
        ``"order_stability"``: ``"pass"``, ``"warn"`` (``value`` is the number of flagged pairs,
        and a :class:`~dirconn.DirconnWarning` names them) or ``"not_checked"`` when the data
        allow no longer order. The check never raises for a fit that fails at a longer order; that
        order is skipped.
    """
    from dirconn.analytic import analytic_gc_test
    from dirconn.granger import GC

    order = _as_count(order, "order", minimum=1)
    orders = candidate_orders(
        order, n_channels=data.n_channels, n_times=data.n_times, n_trials=data.n_trials
    )
    if data.has_mask:
        # the masks select fewer rows than the full grid that candidate_orders assumes
        from dirconn.mvar import row_mask

        def n_rows(k: int) -> int:
            rows = row_mask(data.valid, data.predict, k, data.n_times)
            return int(rows.sum()) if rows is not None else data.n_trials * (data.n_times - k)

        orders = [k for k in orders if k * data.n_channels <= SAMPLING_RATIO_WARN * n_rows(k)]
    if not orders:
        return AssumptionCheck(
            "order_stability",
            "not_checked",
            detail=(
                f"no longer order than {order} fits these data under the rule of at most one "
                "coefficient per ten observations; the direction could not be compared across "
                "model orders."
            ),
        )

    def estimator(k: int) -> GC:
        return GC(order=k, demean=demean, conditional=conditional, check_order_stability=False)

    n = data.n_channels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DirconnWarning)
        try:
            base = estimator(order).fit(data).values[:, :, 0, 0]
        except ValueError as exc:
            return AssumptionCheck(
                "order_stability",
                "not_checked",
                detail=f"the model of order {order} could not be refitted for the check: {exc}",
            )
        favoured = np.sign(base - base.T)  # +1 where i -> j is favoured at the chosen order
        first_order: np.ndarray = np.zeros(
            (n, n), dtype=int
        )  # order at which i -> j is established
        for k in orders:
            try:
                pvalues = analytic_gc_test(estimator(k), data).pvalues[:, :, 0, 0]
            except ValueError:
                continue
            established = (pvalues < ALPHA) & (pvalues.T >= ALPHA_OTHER)  # i -> j established
            against = established & (favoured == -1)  # but the chosen order favours j -> i
            first_order = np.where((first_order == 0) & against, k, first_order)

    pairs = [
        (int(i), int(j), int(first_order[i, j]))
        for i, j in zip(*np.nonzero(first_order), strict=True)
    ]
    if not pairs:
        return AssumptionCheck(
            "order_stability",
            "pass",
            detail=(
                f"the direction of every pair at order {order} agrees with models of order "
                f"{', '.join(str(k) for k in orders)}."
            ),
            value=0.0,
        )

    names = data.names
    shown = [
        f"{names[i]} -> {names[j]} (established at order {k}, while order {order} favours "
        f"{names[j]} -> {names[i]})"
        for i, j, k in pairs[:_NAMED_PAIRS]
    ]
    more = f" and {len(pairs) - _NAMED_PAIRS} more" if len(pairs) > _NAMED_PAIRS else ""
    needed = max(k for _, _, k in pairs)
    detail = (
        f"the direction reverses with model order for {len(pairs)} pair(s): "
        f"{'; '.join(shown)}{more}. The order {order} may be too short for narrow-band or "
        f"long-memory dynamics, which can reverse a direction without any other warning. Refit "
        f"with an order of at least {needed} and trust the direction of these pairs only if it "
        "persists."
    )
    warnings.warn(detail, DirconnWarning, stacklevel=stacklevel)
    return AssumptionCheck("order_stability", "warn", detail=detail, value=float(len(pairs)))
