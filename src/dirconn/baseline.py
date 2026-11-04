"""Normalise a connectivity result to a baseline.

Task-related changes in connectivity are usually read relative to a baseline window. The
normalisation is a separate step from any significance test: it returns an ordinary
:class:`~dirconn.Connectivity` (of magnitudes) with the step recorded in its provenance. To
compare two *conditions*, use :func:`dirconn.contrast_test` on the raw results; to normalise each
condition to its own baseline first, call this function per condition.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from dirconn._resampling import Reduction, make_reducer
from dirconn.bootstrap import bootstrap
from dirconn.estimator import Estimator
from dirconn.exceptions import DirconnWarning
from dirconn.provenance import array_hash
from dirconn.result import Connectivity
from dirconn.timeseries import TimeSeries

NORMALISATIONS: tuple[str, ...] = ("difference", "ratio", "zscore")


def baseline_normalise(
    estimator: Estimator,
    active: TimeSeries,
    baseline: TimeSeries,
    *,
    how: str = "difference",
    n_boot: int | None = None,
    seed: int | None = None,
    block_size: int | None = None,
    reduce: Reduction = None,
) -> Connectivity:
    """Express the connectivity of ``active`` relative to that of ``baseline``.

    All variants work on magnitudes, so the result is real; undefined cells stay NaN.

    Parameters
    ----------
    estimator : Estimator
        Applied to both datasets with the same settings.
    active, baseline : TimeSeries
        Same channels and sampling frequency. The lengths may differ, but note that estimates
        from windows of different length differ in bias and variance (most visibly for Granger
        causality), which a baseline correction does not remove.
    how : {"difference", "ratio", "zscore"}
        ``"difference"``: ``|active| - |baseline|``. ``"ratio"``: ``|active| / |baseline|``;
        cells with a baseline of exactly zero are NaN and a warning says so. ``"zscore"``:
        ``(|active| - |baseline|)`` over the standard deviation of ``|baseline|`` across a
        bootstrap of the baseline (:func:`dirconn.bootstrap`); cells without spread are NaN.
    n_boot, seed : int, optional
        Required for ``"zscore"`` and refused otherwise, so that an ignored argument never looks
        like a setting that was used.
    block_size : int, optional
        Block length for a continuous baseline, as in :func:`dirconn.bootstrap`; ``"zscore"``
        only.
    reduce : callable, str or (float, float), optional
        Applied to both results before normalising.

    Returns
    -------
    Connectivity
        Real-valued, with the axes of the active result. Its provenance is the active
        result's, with the normalisation (and the hash of the baseline input) appended to
        ``history``.
    """
    if not isinstance(estimator, Estimator):
        raise TypeError(f"estimator must be an Estimator, got {type(estimator).__name__}.")
    for label, data in (("active", active), ("baseline", baseline)):
        if not isinstance(data, TimeSeries):
            raise TypeError(f"{label} must be a TimeSeries, got {type(data).__name__}.")
    if how not in NORMALISATIONS:
        raise ValueError(f"how must be one of {NORMALISATIONS}, got {how!r}.")
    if how == "zscore":
        if n_boot is None:
            raise ValueError("how='zscore' needs n_boot (the size of the baseline bootstrap).")
        if seed is None:
            raise ValueError("how='zscore' needs a seed for the baseline bootstrap.")
    elif n_boot is not None or seed is not None or block_size is not None:
        raise ValueError(
            f"n_boot, seed and block_size are only used by how='zscore', not by how={how!r}."
        )
    if active.names != baseline.names:
        raise ValueError(
            f"the datasets have different channels: {active.names} and {baseline.names}."
        )
    if active.sfreq != baseline.sfreq:
        raise ValueError(f"the datasets have different sfreq: {active.sfreq} and {baseline.sfreq}.")

    reducer = make_reducer(reduce)
    observed = reducer(estimator.fit(active))
    spread: np.ndarray | None = None
    if how == "zscore":
        assert n_boot is not None and seed is not None
        boot = bootstrap(
            estimator, baseline, n_boot=n_boot, seed=seed, block_size=block_size, reduce=reduce
        )
        base, spread = boot.observed, boot.std
    else:
        base = reducer(estimator.fit(baseline))
    if base.values.shape != observed.values.shape:
        raise ValueError(
            f"the datasets gave results of shapes {observed.values.shape} and "
            f"{base.values.shape}; they cannot be compared."
        )

    active_magnitude, base_magnitude = np.abs(observed.values), np.abs(base.values)
    values = _normalise(how, active_magnitude, base_magnitude, spread)
    step: dict[str, Any] = {
        "step": "baseline_normalise",
        "how": how,
        "baseline_input_hash": array_hash(baseline.data),
    }
    if how == "zscore":
        step.update({"n_boot": n_boot, "seed": seed, "block_size": block_size})
    return Connectivity(
        values=values,
        source=observed.source,
        target=observed.target,
        freqs=observed.freqs,
        times=observed.times,
        provenance=observed.provenance.with_step(step),
    )


def _normalise(
    how: str, active: np.ndarray, baseline: np.ndarray, spread: np.ndarray | None
) -> np.ndarray:
    if how == "difference":
        return active - baseline
    if how == "ratio":
        zero = baseline == 0
        if zero.any():
            warnings.warn(
                f"{int(zero.sum())} cell(s) have a baseline of exactly zero; their ratio is "
                "undefined (NaN).",
                DirconnWarning,
                stacklevel=3,
            )
        safe = np.where(zero, np.nan, baseline)
        return active / safe
    assert spread is not None
    safe_spread = np.where(spread > 0, spread, np.nan)
    return (active - baseline) / safe_spread
