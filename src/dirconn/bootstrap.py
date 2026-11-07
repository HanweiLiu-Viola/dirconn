"""Bootstrap confidence bands for any estimator.

Epoched data are resampled by trial, continuous data by non-overlapping blocks. Every resample is
passed through the estimator's whole procedure, and the bands are plain percentile intervals of
the resampled magnitudes. A percentile band describes the sampling variability of the estimate
around *its own* value; it is not corrected for bias, so for a positively biased measure such as
Granger causality in short data it can sit entirely above the true value. The ``valid`` and
``predict`` masks of the data travel with the trials or blocks. Percentile bands
are also slightly too narrow in small samples: a nominal 90% band over 30 trials covered the
true value 86% of the time in the calibration test (tests/test_bootstrap.py).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from dirconn._resampling import (
    Reduction,
    make_reducer,
    reissue,
    sort_warnings,
    summary_check,
    without_data_checks,
)
from dirconn.correction import check_alpha
from dirconn.estimator import Estimator
from dirconn.exceptions import DirconnWarning
from dirconn.provenance import AssumptionCheck, AssumptionChecklist
from dirconn.result import Connectivity
from dirconn.rng import RNGController, _as_count
from dirconn.timeseries import TimeSeries

MIN_UNITS = 20


@dataclass(frozen=True, eq=False)
class BootstrapResult:
    """Percentile bands of the magnitude of a connectivity result.

    All arrays have the shape of ``observed.values``; undefined cells are NaN.

    Attributes
    ----------
    observed : Connectivity
        The estimate on the original data (after ``reduce``), with its provenance.
    lower, upper : ndarray
        Percentiles ``(1 - ci) / 2`` and ``(1 + ci) / 2`` of ``|value|`` over the resamples.
    std : ndarray
        Standard deviation (ddof 1) of ``|value|`` over the resamples.
    ci : float
        Confidence level of the band.
    n_boot, seed : int
        How the bootstrap was run.
    interval : str
        Always ``"percentile"``: no bias correction.
    resampling : dict
        ``unit`` (``"trial"`` or ``"block"``), ``n_units`` and, for blocks, ``block_size`` and
        ``dropped_samples`` (samples after the last whole block).
    checks : AssumptionChecklist
        ``bootstrap_units`` (enough trials or blocks) and ``bootstrap_fits`` (diagnostics raised
        inside the resampled fits).
    """

    observed: Connectivity
    lower: np.ndarray
    upper: np.ndarray
    std: np.ndarray
    ci: float
    n_boot: int
    seed: int
    interval: str
    resampling: dict[str, Any]
    checks: AssumptionChecklist

    def __post_init__(self) -> None:
        """Make the arrays read-only."""
        for name in ("lower", "upper", "std"):
            view = np.asarray(getattr(self, name), dtype=float).view()
            view.flags.writeable = False
            object.__setattr__(self, name, view)

    @property
    def method(self) -> str:
        """Name of the estimator."""
        return self.observed.provenance.method

    @property
    def defined(self) -> np.ndarray:
        """Boolean mask of the cells that have a band."""
        return ~np.isnan(self.observed.values)

    def __repr__(self) -> str:
        """Return a short summary."""
        shape = "x".join(str(n) for n in self.lower.shape)
        return (
            f"BootstrapResult(method={self.method!r}, shape={shape}, ci={self.ci:g}, "
            f"n_boot={self.n_boot}, unit={self.resampling['unit']!r}, "
            f"interval={self.interval!r})"
        )


def bootstrap(
    estimator: Estimator,
    data: TimeSeries,
    *,
    n_boot: int,
    seed: int,
    ci: float = 0.95,
    block_size: int | None = None,
    reduce: Reduction = None,
) -> BootstrapResult:
    """Bootstrap an estimator's result.

    Resample ``i`` uses ``RNGController(seed).child(i)`` to draw ``n_units`` indices with
    replacement, so results do not depend on how the work is split.

    Parameters
    ----------
    estimator : Estimator
        Any dirconn estimator; its whole procedure, including order selection, is repeated.
    data : TimeSeries
        Epoched data are resampled by trial and need at least two trials; warns below 20.
        Continuous data need ``block_size``: the record is cut into non-overlapping blocks that
        are resampled and stored as *separate trials*, so no model is fitted across a seam. The
        observed result is still computed on the unbroken record, which uses a few more
        regression rows than a resample does.
    n_boot : int
        Number of resamples.
    seed : int
        Required.
    ci : float
        Confidence level, strictly between 0 and 1.
    block_size : int, optional
        Block length in samples; continuous data only. Choose it longer than the memory of the
        process (model order, autocorrelation time), or the blocks break the dependence being
        estimated.
    reduce : callable, str or (float, float), optional
        Applied to every fit before the bands are computed (see :func:`surrogate_test`).

    Returns
    -------
    BootstrapResult

    Raises
    ------
    ValueError
        For invalid arguments, fewer than two trials or blocks, or a resampled fit that leaves a
        cell undefined that is defined on the data.
    """
    if not isinstance(estimator, Estimator):
        raise TypeError(f"estimator must be an Estimator, got {type(estimator).__name__}.")
    if not isinstance(data, TimeSeries):
        raise TypeError(f"data must be a TimeSeries, got {type(data).__name__}.")
    n_boot = _as_count(n_boot, "n_boot", minimum=1)
    controller = RNGController(seed)
    try:
        ci = check_alpha(ci)
    except ValueError:
        raise ValueError(f"ci must lie strictly between 0 and 1, got {ci}.") from None
    reducer = make_reducer(reduce)

    units, valid, predict, resampling = _units(data, block_size)
    n_units = units.shape[0]
    units_check = _units_check(resampling)

    observed = reducer(estimator.fit(data))
    magnitude = np.abs(observed.values)
    defined = ~np.isnan(magnitude)
    samples = np.empty((n_boot, int(defined.sum())))
    notes: list[str] = []
    other: dict[tuple[type[Warning], str], warnings.WarningMessage] = {}

    light = without_data_checks(estimator)  # the model checks belong to the fit on the data
    for i in range(n_boot):
        index = controller.child(i).integers(0, n_units, size=n_units)
        resample = TimeSeries(
            units[index],
            data.sfreq,
            data.channels,
            "epoched",
            data.tmin,
            valid=None if valid is None else valid[index],
            predict=None if predict is None else predict[index],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = reducer(light.fit(resample))
        notes.extend(sort_warnings(caught, other)[:1])
        if result.values.shape != magnitude.shape:
            raise ValueError(
                f"resample {i} gave a result of shape {result.values.shape}, "
                f"expected {magnitude.shape}."
            )
        values = np.abs(result.values)
        lost = defined & np.isnan(values)
        if lost.any():
            raise ValueError(
                f"resample {i} left {int(lost.sum())} cell(s) undefined that are defined on "
                "the data; refusing to compute bands from degraded resamples."
            )
        samples[i] = values[defined]
    reissue(other)

    low, high = (1.0 - ci) / 2.0, (1.0 + ci) / 2.0
    lower = np.full(magnitude.shape, np.nan)
    upper = np.full(magnitude.shape, np.nan)
    std = np.full(magnitude.shape, np.nan)
    lower[defined] = np.quantile(samples, low, axis=0)
    upper[defined] = np.quantile(samples, high, axis=0)
    std[defined] = samples.std(axis=0, ddof=1) if n_boot > 1 else np.nan

    checks = AssumptionChecklist(
        (
            units_check,
            summary_check(
                "bootstrap_fits", notes, n_boot, f"the {estimator.info.name} fits on resamples"
            ),
        )
    )
    return BootstrapResult(
        observed=observed,
        lower=lower,
        upper=upper,
        std=std,
        ci=ci,
        n_boot=n_boot,
        seed=controller.seed,
        interval="percentile",
        resampling=resampling,
        checks=checks,
    )


def _units(
    data: TimeSeries, block_size: int | None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Return the resampling units ``(n_units, n_channels, n_times)`` and their masks."""
    if data.segmentation == "epoched":
        if block_size is not None:
            raise ValueError(
                "block_size applies to continuous data only; epoched data are resampled by trial."
            )
        if data.n_trials < 2:
            raise ValueError("Bootstrapping by trial needs at least 2 trials.")
        return data.data, data.valid, data.predict, {"unit": "trial", "n_units": data.n_trials}
    if block_size is None:
        raise ValueError(
            "Continuous data need block_size (in samples): the record is resampled in blocks."
        )
    size = _as_count(block_size, "block_size", minimum=1)
    n_blocks = data.n_times // size
    if n_blocks < 2:
        raise ValueError(
            f"block_size={size} leaves {n_blocks} block(s) in {data.n_times} samples; "
            "the bootstrap needs at least two blocks."
        )
    used = n_blocks * size
    blocks = data.data[0, :, :used].reshape(data.n_channels, n_blocks, size).transpose(1, 0, 2)

    def cut(mask: np.ndarray | None) -> np.ndarray | None:
        return None if mask is None else mask[0, :used].reshape(n_blocks, size)

    return (
        blocks,
        cut(data.valid),
        cut(data.predict),
        {
            "unit": "block",
            "n_units": n_blocks,
            "block_size": size,
            "dropped_samples": data.n_times - used,
        },
    )


def _units_check(resampling: dict[str, Any]) -> AssumptionCheck:
    n, unit = resampling["n_units"], resampling["unit"]
    if n < MIN_UNITS:
        detail = (
            f"only {n} {unit}s to resample; percentile bands need at least {MIN_UNITS} and are "
            "unreliable below that."
        )
        warnings.warn(detail, DirconnWarning, stacklevel=3)
        return AssumptionCheck("bootstrap_units", "warn", detail=detail, value=float(n))
    return AssumptionCheck(
        "bootstrap_units", "pass", detail=f"{n} {unit}s to resample.", value=float(n)
    )
