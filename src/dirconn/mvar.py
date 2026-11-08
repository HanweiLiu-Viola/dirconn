"""Multivariate autoregressive (VAR) models fitted by pooled least squares.

This is the numerical core shared by every VAR-based method. A VAR(p) model explains the value of
every channel at time ``t`` by the values of all channels at the ``p`` previous samples::

    x[t] = sum_{k=1..p} A_k @ x[t - k] + e[t]

The coefficient matrices use the layout ``coefs[k - 1, target, source]``: ``coefs[k - 1][i, j]`` is
the effect of channel ``j`` at lag ``k`` on channel ``i``. With several trials the regressions of
all trials are stacked into one least-squares problem and no regressor ever reaches across a trial
boundary.

Data with per-sample masks (``TimeSeries.valid`` and ``TimeSeries.predict``) use only the rows
whose sample is to be predicted and whose whole window ``t - order .. t`` is valid; see
:func:`lagged_design`. Without masks, or with masks that are all true, nothing changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np

from dirconn.timeseries import TimeSeries

INFORMATION_CRITERIA: tuple[str, ...] = ("aic", "bic", "hqc")
CRITERIA: tuple[str, ...] = (*INFORMATION_CRITERIA, "cv")


def _check_order(order: Any, name: str = "order") -> int:
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError(f"{name} must be a positive integer, got {order!r}.")
    return int(order)


def _check_ridge(ridge: float) -> float:
    value = float(ridge)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"ridge must be a finite non-negative number, got {ridge!r}.")
    return value


def _readonly(array: np.ndarray) -> np.ndarray:
    view = np.asarray(array).view()
    view.flags.writeable = False
    return view


def row_mask(
    valid: np.ndarray | None,
    predict: np.ndarray | None,
    order: int,
    n_times: int,
    start: int | None = None,
) -> np.ndarray | None:
    """Return which regression rows of a VAR problem are used, or ``None`` when all are.

    The row that predicts sample ``t`` (``start <= t < n_times``) of a trial is used when
    ``predict[t]`` is true and ``valid[t - order .. t]`` are all true: the predicted value and every
    regressor are valid samples.

    Parameters
    ----------
    valid, predict : ndarray of bool, shape (n_trials, n_times), or None
        The masks of a :class:`~dirconn.TimeSeries`; ``None`` means all true.
    order : int
        Number of lags.
    n_times : int
        Samples per trial.
    start : int, optional
        First predicted sample, as in :func:`lagged_design`; defaults to ``order``.

    Returns
    -------
    ndarray of bool, shape (n_trials, n_times - start), or None
        Rows in the layout of :func:`lagged_design` (trial after trial); ``None`` without masks.
    """
    if valid is None and predict is None:
        return None
    order = _check_order(order)
    start = order if start is None else int(start)
    n_trials = (valid if valid is not None else predict).shape[0]  # type: ignore[union-attr]
    for name, mask in (("valid", valid), ("predict", predict)):
        if mask is not None and mask.shape != (n_trials, n_times):
            raise ValueError(
                f"{name} must have shape (n_trials, n_times) = {(n_trials, n_times)}, "
                f"got {mask.shape}."
            )
    rows = np.ones((n_trials, n_times - start), dtype=bool)
    if predict is not None:
        rows &= predict[:, start:]
    if valid is not None:
        for k in range(order + 1):
            rows &= valid[:, start - k : n_times - k]
    return rows


def lagged_design(
    x: np.ndarray,
    order: int,
    start: int | None = None,
    *,
    valid: np.ndarray | None = None,
    predict: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the regression problem of a VAR model from trials.

    Parameters
    ----------
    x : ndarray, shape (n_trials, n_channels, n_times)
        The data.
    order : int
        Number of lags.
    start : int, optional
        Index of the first predicted sample in every trial; defaults to ``order``. A larger value
        gives all models of different orders one common set of observations.
    valid, predict : ndarray of bool, shape (n_trials, n_times), optional
        Per-sample masks (see :class:`~dirconn.TimeSeries`). Only the rows selected by
        :func:`row_mask` are returned, in the same order; without masks every row is.

    Returns
    -------
    y : ndarray, shape (n_rows, n_channels)
        The predicted samples, trial after trial; ``n_rows = n_trials * (n_times - start)``
        without masks.
    design : ndarray, shape (n_rows, n_channels * order)
        Column block ``k - 1`` (``n_channels`` columns) holds the samples ``k`` steps earlier, so
        the first ``n_channels * m`` columns are the design of a model of order ``m``. A row only
        contains samples of the trial it predicts.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 3:
        raise ValueError(f"x must have shape (trials, channels, times), got {x.shape}.")
    order = _check_order(order)
    start = order if start is None else int(start)
    n_times = x.shape[2]
    if start < order:
        raise ValueError(f"start ({start}) must be at least the order ({order}).")
    if start >= n_times:
        raise ValueError(f"start ({start}) must be smaller than the number of samples ({n_times}).")
    n_channels = x.shape[1]

    def block(first: int, last: int) -> np.ndarray:
        return x[:, :, first:last].transpose(0, 2, 1).reshape(-1, n_channels)

    y = block(start, n_times)
    design = np.hstack([block(start - k, n_times - k) for k in range(1, order + 1)])
    for name, mask in (("valid", valid), ("predict", predict)):
        if mask is not None and np.shape(mask) != (x.shape[0], n_times):
            raise ValueError(
                f"{name} must have the shape (n_trials, n_times) of the data, "
                f"{(x.shape[0], n_times)}, got {np.shape(mask)}."
            )
    rows = row_mask(valid, predict, order, n_times, start)
    if rows is not None:
        used = rows.ravel()
        y, design = y[used], design[used]
    return y, design


def _solve(y: np.ndarray, design: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    """Least squares (optionally ridge); returns ``(beta, residuals)``.

    The ridge penalty is relative: the absolute penalty is ``ridge`` times the mean power of a
    regressor column, so the same ``ridge`` means the same thing for data of any scale.
    """
    n_params = design.shape[1]
    if ridge == 0.0:
        beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        if rank < n_params:
            raise ValueError(
                f"The regression design has rank {rank} but {n_params} coefficients per equation; "
                "channels are collinear or the lags carry too little information. Use a ridge "
                "penalty, fewer lags or fewer channels."
            )
    else:
        scale = float(np.mean(np.sum(design**2, axis=0)))
        if scale == 0.0:
            raise ValueError("All regressors are zero; there is nothing to fit.")
        penalty = math.sqrt(ridge * scale)
        augmented_design = np.vstack([design, penalty * np.eye(n_params)])
        augmented_y = np.vstack([y, np.zeros((n_params, y.shape[1]))])
        beta = np.linalg.lstsq(augmented_design, augmented_y, rcond=None)[0]
    return beta, y - design @ beta


def _check_rows(n_rows: int, n_params: int) -> None:
    if n_rows <= n_params:
        raise ValueError(
            f"Too few samples: {n_rows} regression rows for {n_params} coefficients per equation. "
            "Use more data, fewer lags or fewer channels."
        )


def _prepare(data: TimeSeries, demean: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return the samples (demeaned if asked) and the mean that was removed.

    With a ``valid`` mask the mean is taken over the valid samples only.
    """
    if not isinstance(data, TimeSeries):
        raise TypeError(f"Expected a TimeSeries, got {type(data).__name__}.")
    x = np.asarray(data.data)
    if demean:
        if data.valid is None:
            mean = x.mean(axis=(0, 2))
        else:
            weight = data.valid[:, None, :]
            mean = np.sum(np.where(weight, x, 0.0), axis=(0, 2)) / np.count_nonzero(data.valid)
        return x - mean[None, :, None], mean
    return x, np.zeros(x.shape[1])


class Masks(TypedDict):
    """The per-sample masks of a :class:`~dirconn.TimeSeries`, as keyword arguments."""

    valid: np.ndarray | None
    predict: np.ndarray | None


def masks_of(data: TimeSeries) -> Masks:
    """Return the masks of ``data`` as keyword arguments for :func:`lagged_design`."""
    return {"valid": data.valid, "predict": data.predict}


@dataclass(frozen=True, eq=False)
class VarFit:
    """A fitted VAR model.

    Attributes
    ----------
    coefs : ndarray, shape (order, n_channels, n_channels)
        ``coefs[k - 1, target, source]``, the layout of ``statsmodels`` ``VARResults.coefs``.
    noise_cov : ndarray, shape (n_channels, n_channels)
        Residual covariance with the degrees-of-freedom correction ``n_obs - n_channels * order``.
    residuals : ndarray, shape (n_obs, n_channels)
        Residuals of the pooled regression, trial after trial (only the rows that were used, for
        masked data).
    regressor_cov : ndarray, shape (order * n_channels, order * n_channels)
        Covariance of the lagged regressors, ``X^T X / n_obs`` of the design of
        :func:`lagged_design` (after demeaning, without the ridge penalty): row and column
        ``(k - 1) * n_channels + j`` belong to channel ``j`` at lag ``k``. With ``noise_cov`` it
        gives the asymptotic covariance of the least-squares coefficients,
        ``noise_cov (x) inv(regressor_cov) / n_obs``.
    order : int
        Number of lags.
    n_obs : int
        Number of regression rows (the samples that were predicted, over all trials; for masked
        data the rows selected by :func:`row_mask`).
    ridge : float
        Relative ridge penalty used (0 for plain least squares).
    demeaned : bool
        Whether the per-channel mean over trials and time was removed before fitting.
    mean : ndarray, shape (n_channels,)
        The mean that was removed (zeros when ``demeaned`` is false).
    names : tuple of str
        Channel names.
    sfreq : float
        Sampling frequency in Hz.
    """

    coefs: np.ndarray
    noise_cov: np.ndarray
    residuals: np.ndarray
    regressor_cov: np.ndarray
    order: int
    n_obs: int
    ridge: float
    demeaned: bool
    mean: np.ndarray
    names: tuple[str, ...]
    sfreq: float

    def __post_init__(self) -> None:
        """Make the arrays read-only."""
        for field_name in ("coefs", "noise_cov", "residuals", "regressor_cov", "mean"):
            object.__setattr__(self, field_name, _readonly(getattr(self, field_name)))

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return int(self.coefs.shape[1])


def fit_var(data: TimeSeries, order: int, *, ridge: float = 0.0, demean: bool = True) -> VarFit:
    """Fit a VAR model by pooled least squares.

    All trials contribute to one regression; the regressors of a sample never come from another
    trial. Failures raise: a rank-deficient design or too few samples are errors, never silently
    replaced by another model.

    Parameters
    ----------
    data : TimeSeries
        The data; every trial is treated as an independent realisation. With masks only the rows
        selected by :func:`row_mask` are fitted, and the mean is that of the valid samples.
    order : int
        Number of lags, at least 1.
    ridge : float
        Relative ridge penalty, at least 0. The absolute penalty is ``ridge`` times the mean power
        of a regressor column, so it is scale free. Use it when channels are many or nearly
        collinear.
    demean : bool
        Remove the per-channel mean over trials and time first (the model has no intercept). An
        evoked response, if the data contain one, must be removed beforehand.

    Returns
    -------
    VarFit

    Raises
    ------
    ValueError
        For an invalid ``order`` or ``ridge``, too few samples, or a rank-deficient design.
    """
    order = _check_order(order)
    ridge = _check_ridge(ridge)
    x, mean = _prepare(data, demean)
    y, design = lagged_design(x, order, **masks_of(data))
    _check_rows(y.shape[0], design.shape[1])
    beta, residuals = _solve(y, design, ridge)
    n_channels = x.shape[1]
    coefs = beta.T.reshape(n_channels, order, n_channels).transpose(1, 0, 2)
    noise_cov = residuals.T @ residuals / (y.shape[0] - design.shape[1])
    return VarFit(
        coefs=coefs,
        noise_cov=noise_cov,
        residuals=residuals,
        regressor_cov=design.T @ design / y.shape[0],
        order=order,
        n_obs=int(y.shape[0]),
        ridge=ridge,
        demeaned=bool(demean),
        mean=mean,
        names=data.names,
        sfreq=data.sfreq,
    )


@dataclass(frozen=True, eq=False)
class OrderSelection:
    """Result of :func:`select_order`.

    Attributes
    ----------
    criterion : str
        The criterion that decided ``order``.
    orders : ndarray
        The candidate orders ``1 .. max_order``.
    aic, bic, hqc : ndarray
        The criterion curves, one value per candidate order.
    order : int
        The order that minimises ``criterion``.
    max_order : int
        Largest candidate order.
    n_obs : int
        Number of regression rows, the same for every candidate order.
    cv : ndarray or None
        Cross-validated normalised prediction error per candidate order; only computed when
        ``criterion="cv"``.
    cv_scheme : {"trials", "blocks"} or None
        How the data were held out: whole trials or contiguous time blocks.
    n_folds : int or None
        Number of folds actually used.
    """

    criterion: str
    orders: np.ndarray
    aic: np.ndarray
    bic: np.ndarray
    hqc: np.ndarray
    order: int
    max_order: int
    n_obs: int
    cv: np.ndarray | None = None
    cv_scheme: str | None = None
    n_folds: int | None = None

    def __post_init__(self) -> None:
        """Make the arrays read-only."""
        for field_name in ("orders", "aic", "bic", "hqc"):
            object.__setattr__(self, field_name, _readonly(getattr(self, field_name)))
        if self.cv is not None:
            object.__setattr__(self, "cv", _readonly(self.cv))

    def to_dict(self) -> dict[str, Any]:
        """Return the selection as JSON-compatible lists, for example for a provenance record."""
        return {
            "criterion": self.criterion,
            "orders": self.orders.tolist(),
            "aic": self.aic.tolist(),
            "bic": self.bic.tolist(),
            "hqc": self.hqc.tolist(),
            "cv": None if self.cv is None else self.cv.tolist(),
            "cv_scheme": self.cv_scheme,
            "n_folds": self.n_folds,
            "n_obs": self.n_obs,
        }


def _check_folds(n_folds: Any) -> int:
    if isinstance(n_folds, bool) or not isinstance(n_folds, (int, np.integer)) or n_folds < 2:
        raise ValueError(f"n_folds must be an integer of at least 2, got {n_folds!r}.")
    return int(n_folds)


def cv_splits(
    n_trials: int, rows_per_trial: int, n_folds: int, gap: int
) -> tuple[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Split the regression rows of a VAR problem into cross-validation folds.

    Rows are numbered trial after trial, as returned by :func:`lagged_design`. With at least four
    trials, whole trials are held out (round-robin groups, at most ``n_trials`` folds): trials are
    independent, so nothing leaks. With fewer trials the same contiguous block of every trial is
    held out. A row uses ``order`` earlier samples, so the training rows within ``gap`` rows of a
    held-out block are dropped; otherwise the held-out predictions would reuse training samples.
    Every row is held out exactly once.

    Parameters
    ----------
    n_trials : int
        Number of trials.
    rows_per_trial : int
        Regression rows per trial.
    n_folds : int
        Requested number of folds (at least 2).
    gap : int
        Rows to drop on each side of a held-out block; use the largest candidate order.

    Returns
    -------
    scheme : {"trials", "blocks"}
    splits : list of (train, test)
        Row index arrays, one pair per fold.
    """
    return split_rows(np.ones((n_trials, rows_per_trial), dtype=bool), n_folds, gap)


def split_rows(
    rows: np.ndarray, n_folds: int, gap: int
) -> tuple[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Cross-validation folds of the used rows of a VAR problem, as in :func:`cv_splits`.

    Parameters
    ----------
    rows : ndarray of bool, shape (n_trials, rows_per_trial)
        Which rows are used (see :func:`row_mask`); the used rows are numbered trial after trial.
    n_folds, gap : int
        As in :func:`cv_splits`. With at least four trials that have rows, whole trials are held
        out; otherwise contiguous blocks of the time points that have rows, and ``gap`` is
        measured in samples.

    Returns
    -------
    scheme : {"trials", "blocks"}
    splits : list of (train, test)
        Index arrays into the used rows.
    """
    n_folds = _check_folds(n_folds)
    if gap < 0:
        raise ValueError(f"gap must be non-negative, got {gap}.")
    trial, time = np.nonzero(rows)
    index = np.arange(trial.size)
    present = np.unique(trial)
    splits = []
    if present.size >= 4:
        n_splits = min(n_folds, int(present.size))
        for fold in range(n_splits):
            held = np.isin(trial, present[fold::n_splits])
            splits.append((index[~held], index[held]))
        return "trials", splits
    positions = np.unique(time)
    if positions.size < n_folds:
        raise ValueError(
            f"Cannot split {positions.size} rows per trial into {n_folds} contiguous blocks."
        )
    for block in np.array_split(positions, n_folds):
        first, last = int(block[0]), int(block[-1]) + 1
        test = index[(time >= first) & (time < last)]
        dropped = (time >= first - gap) & (time < last + gap)
        splits.append((index[~dropped], test))
    return "blocks", splits


def _cv_curve(
    y: np.ndarray,
    design: np.ndarray,
    n_channels: int,
    max_order: int,
    ridge: float,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Normalised held-out prediction error for every order ``1 .. max_order``.

    The error of a channel is its mean squared held-out error divided by its own variance, and the
    curve is the mean over channels, so the criterion does not depend on the scale of the data.
    """
    variance = np.mean(y**2, axis=0)
    if np.any(variance == 0.0):
        raise ValueError("A channel has zero variance; cross-validation is undefined.")
    n_params = n_channels * max_order
    for train, _ in splits:
        if train.size <= n_params:
            raise ValueError(
                f"Too few samples for cross-validation: a training set has {train.size} rows for "
                f"{n_params} coefficients per equation. Use more data, fewer folds, a lower "
                "maximum order or fewer channels."
            )
    curve = []
    for order in range(1, max_order + 1):
        columns = n_channels * order
        squared_error = np.zeros(n_channels)
        for train, test in splits:
            try:
                beta, _ = _solve(y[train], design[train, :columns], ridge)
            except ValueError as exc:
                raise ValueError(f"Order {order}, cross-validation: {exc}") from exc
            squared_error += np.sum((y[test] - design[test, :columns] @ beta) ** 2, axis=0)
        curve.append(float(np.mean(squared_error / y.shape[0] / variance)))
    return np.asarray(curve)


def select_order(
    data: TimeSeries,
    max_order: int,
    *,
    criterion: str = "bic",
    ridge: float = 0.0,
    demean: bool = True,
    n_folds: int = 5,
) -> OrderSelection:
    """Choose the VAR order with an information criterion or by cross-validation.

    Every candidate order is fitted on one common set of observations (the samples from
    ``max_order`` onwards in each trial; for masked data the rows that :func:`row_mask` selects
    at ``max_order``), so the criteria compare models on the same data. With
    ``K`` channels, ``n`` observations, residual covariance ``S`` (maximum likelihood) and
    ``m K^2`` free coefficients::

        aic = ln|S| + 2 m K^2 / n
        bic = ln|S| + ln(n) m K^2 / n
        hqc = ln|S| + 2 ln(ln(n)) m K^2 / n

    The cross-validation criterion ``"cv"`` is the held-out prediction error, normalised per
    channel by its variance and averaged over channels, so it is scale free. With at least four
    trials whole trials are held out; otherwise contiguous time blocks of each trial are held out
    with a gap that prevents leakage (see :func:`cv_splits`). It makes no assumption about the
    noise and is the more trustworthy choice for short records and many channels, at a higher
    cost. It has no penalty term, so beyond the true order the curve is nearly flat and the
    minimum can land a lag or more above the BIC choice: inspect ``OrderSelection.cv`` rather than
    trusting the single number.

    Parameters
    ----------
    data : TimeSeries
        The data.
    max_order : int
        Largest candidate order.
    criterion : {"aic", "bic", "hqc", "cv"}
        Criterion that picks the order. The default BIC is conservative and consistent.
    ridge, demean
        As in :func:`fit_var`.
    n_folds : int
        Number of folds for ``criterion="cv"`` (at least 2); with whole-trial folds it is capped at
        the number of trials.

    Returns
    -------
    OrderSelection
        Carries the three information-criterion curves so the choice can be inspected, and the
        cross-validation curve when it was requested.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}.")
    max_order = _check_order(max_order, "max_order")
    ridge = _check_ridge(ridge)
    n_folds = _check_folds(n_folds)
    x, _ = _prepare(data, demean)
    n_channels = x.shape[1]
    y, design = lagged_design(x, max_order, **masks_of(data))
    n_obs = int(y.shape[0])
    _check_rows(n_obs, n_channels * max_order)

    curves: dict[str, list[float]] = {name: [] for name in INFORMATION_CRITERIA}
    for order in range(1, max_order + 1):
        try:
            _, residuals = _solve(y, design[:, : n_channels * order], ridge)
        except ValueError as exc:
            raise ValueError(f"Order {order}: {exc}") from exc
        sign, logdet = np.linalg.slogdet(residuals.T @ residuals / n_obs)
        if sign <= 0:
            raise ValueError(f"The residual covariance is singular at order {order}.")
        n_free = order * n_channels**2
        curves["aic"].append(logdet + 2.0 * n_free / n_obs)
        curves["bic"].append(logdet + math.log(n_obs) * n_free / n_obs)
        curves["hqc"].append(logdet + 2.0 * math.log(math.log(n_obs)) * n_free / n_obs)

    arrays = {name: np.asarray(values) for name, values in curves.items()}
    cv_curve = scheme = folds = None
    if criterion == "cv":
        rows = row_mask(data.valid, data.predict, max_order, x.shape[2])
        if rows is None:
            rows = np.ones((x.shape[0], x.shape[2] - max_order), dtype=bool)
        scheme, splits = split_rows(rows, n_folds, gap=max_order)
        folds = len(splits)
        cv_curve = _cv_curve(y, design, n_channels, max_order, ridge, splits)
        arrays["cv"] = cv_curve
    return OrderSelection(
        criterion=criterion,
        orders=np.arange(1, max_order + 1),
        aic=arrays["aic"],
        bic=arrays["bic"],
        hqc=arrays["hqc"],
        order=int(np.argmin(arrays[criterion])) + 1,
        max_order=max_order,
        n_obs=n_obs,
        cv=cv_curve,
        cv_scheme=scheme,
        n_folds=folds,
    )


def check_order_spec(order: Any, max_order: Any, ridge: float, n_folds: Any) -> None:
    """Validate the order-related parameters shared by the VAR-based estimators.

    ``order`` is a positive integer or the name of a criterion in ``CRITERIA``.
    """
    if isinstance(order, str):
        if order not in CRITERIA:
            raise ValueError(
                f"order must be a positive integer or one of {CRITERIA}, got {order!r}."
            )
    else:
        _check_order(order)
    _check_order(max_order, "max_order")
    _check_ridge(ridge)
    _check_folds(n_folds)


def choose_order(
    data: TimeSeries,
    order: int | str,
    max_order: int,
    *,
    ridge: float,
    demean: bool,
    n_folds: int,
) -> tuple[int, OrderSelection | None]:
    """Resolve an order specification: use a fixed order, or select one with a criterion.

    Returns the order and the :class:`OrderSelection` behind it (``None`` for a fixed order).
    The order is chosen once, on the model of all channels, and then used for every fit of an
    analysis.
    """
    if isinstance(order, str):
        selection = select_order(
            data, max_order, criterion=order, ridge=ridge, demean=demean, n_folds=n_folds
        )
        return selection.order, selection
    return int(order), None
