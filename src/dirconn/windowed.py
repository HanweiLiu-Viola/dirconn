"""WDTF: the DTF of one pooled VAR model per window or per segment.

The cheap time-resolved baseline next to :class:`~dirconn.ADTF`: every window (or every segment
that the caller names) is fitted on its own as a stationary pooled VAR and evaluated exactly like
:class:`~dirconn.DTF`. Nothing is carried over from one window to the next, so there is no memory
of the previous state, but every window needs enough samples for its own model.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from dirconn.estimator import Estimator, MethodInfo
from dirconn.exceptions import DirconnWarning
from dirconn.mvar import check_order_spec, choose_order, fit_var
from dirconn.provenance import STANDARD_ASSUMPTIONS, AssumptionCheck
from dirconn.result import Connectivity, check_segments
from dirconn.spectral import dtf_from_coefs, frequency_grid, spectral_radius
from dirconn.spectral_estimators import _standardise
from dirconn.timeseries import TimeSeries
from dirconn.tvmvar import DEFAULT_MAX_ORDER, seconds_to_samples, stationarity_check

SAMPLES_PER_COEFFICIENT_WARN = 50
"""Regression rows per coefficient of an equation below which a window warns."""

_EDGE_TOLERANCE = 1e-9
_BACKEND = "own numpy pooled OLS VAR per window, vectorised frequency response"


def _seconds(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number of seconds, got {value!r}.")
    return float(value)


@dataclass(frozen=True)
class WDTF(Estimator):
    """Windowed directed transfer function: one DTF per window or per segment.

    ``values[i, j, f, k]`` is the DTF of window ``k``. Windows are either sliding (``window`` and
    ``step``) or named by the caller (``segments``); exactly one of ``window`` and ``segments`` is
    required. The time axis holds window or segment centres in seconds. With several trials the
    samples of the same window of all trials are pooled into one regression, which assumes that
    the non-stationarity is time-locked to the trial start.

    Parameters
    ----------
    window : float, optional
        Window length in seconds, a multiple of the sampling interval. Windows start at the first
        sample and move by ``step``; a window that does not fit is not used.
    step : float, optional
        Distance between window starts in seconds (a multiple of the sampling interval); default
        half a window. Only with ``window``.
    segments : sequence of (float, float), optional
        ``(start, stop)`` pairs in seconds on the time axis of the data, both ends included,
        ascending and not overlapping. Every segment is fitted on its own.
    order : int or {"aic", "bic", "hqc", "cv"}
        Fixed VAR order, or the criterion that picks it **once, on the whole recording**; a window
        then uses that order whatever its own data would prefer.
    max_order : int
        Largest candidate order; the default is 5, as for the other time-resolved methods.
    n_freqs, fmax, freqs
        Frequency axis, as for :class:`~dirconn.DTF`; the default is 64 points from 0 to the
        Nyquist frequency.
    ridge, demean, n_folds
        As for :func:`dirconn.fit_var` (``demean`` removes the mean of each window) and
        :func:`dirconn.select_order`.
    standardize : bool
        Divide every channel by its standard deviation over the whole recording first (one scale
        for all windows), because the result depends on the relative scale of the channels.

    A window fit that fails (too few samples, collinear channels) raises; diagnostics warn and are
    recorded: a window with fewer than 50 regression rows per coefficient of an equation, and a
    share of windows above 5 % whose model is unstable.
    """

    info = MethodInfo(
        name="wdtf",
        backend=_BACKEND,
        directed=True,
        min_nodes=2,
        value_range=(0.0, 1.0),
        scale_dependent=True,
        frequency_resolved=True,
        time_resolved=True,
        assumptions=STANDARD_ASSUMPTIONS,
    )

    window: float | None = None
    step: float | None = None
    segments: Any = None
    order: int | str = "bic"
    max_order: int = DEFAULT_MAX_ORDER
    n_freqs: int = 64
    fmax: float | None = None
    freqs: Any = None
    ridge: float = 0.0
    demean: bool = True
    n_folds: int = 5
    standardize: bool = True

    def __post_init__(self) -> None:
        """Validate the parameters."""
        super().__post_init__()
        if (self.window is None) == (self.segments is None):
            raise ValueError("Pass exactly one of window or segments.")
        if self.window is not None:
            _seconds(self.window, "window")
            if self.step is not None:
                _seconds(self.step, "step")
        else:
            if self.step is not None:
                raise ValueError("step is only used with window, not with segments.")
            object.__setattr__(self, "segments", check_segments(self.segments))
        check_order_spec(self.order, self.max_order, self.ridge, self.n_folds)
        for name in ("standardize", "demean"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool, got {getattr(self, name)!r}.")
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

    def _depends_on_scale(self) -> bool:
        return not self.standardize

    def _sample_ranges(self, data: TimeSeries) -> list[tuple[int, int]]:
        """Inclusive ``(first, last)`` sample indices of every window or segment."""
        n_times = data.n_times
        if self.window is not None:
            length = seconds_to_samples(self.window, data.sfreq, "window")
            step = seconds_to_samples(
                self.window / 2.0 if self.step is None else self.step, data.sfreq, "step"
            )
            if length > n_times:
                raise ValueError(
                    f"The window ({self.window:g} s, {length} samples) is longer than the "
                    f"recording ({n_times} samples)."
                )
            return [(s, s + length - 1) for s in range(0, n_times - length + 1, step)]
        ranges = []
        for start, stop in self.segments:
            first = math.ceil((start - data.tmin) * data.sfreq - _EDGE_TOLERANCE)
            last = math.floor((stop - data.tmin) * data.sfreq + _EDGE_TOLERANCE)
            if first < 0 or last > n_times - 1:
                raise ValueError(
                    f"The segment {(start, stop)} lies outside the recording, which covers "
                    f"{data.tmin:g} to {data.tmin + (n_times - 1) / data.sfreq:g} s."
                )
            if last <= first:
                raise ValueError(
                    f"The segment {(start, stop)} must contain at least two samples "
                    f"(sampling interval {1.0 / data.sfreq:g} s)."
                )
            ranges.append((first, last))
        return ranges

    def _fit(self, data: TimeSeries) -> Connectivity:
        freqs = frequency_grid(data.sfreq, n_freqs=self.n_freqs, fmax=self.fmax, freqs=self.freqs)
        ranges = self._sample_ranges(data)
        if self.standardize:
            series, preprocessing = _standardise(data)
        else:
            series, preprocessing = data, {"standardized": False}
        order, selection = choose_order(
            series,
            self.order,
            self.max_order,
            ridge=self.ridge,
            demean=self.demean,
            n_folds=self.n_folds,
        )

        values = np.empty((data.n_channels, data.n_channels, freqs.size, len(ranges)))
        radius = np.empty(len(ranges))
        rows = np.empty(len(ranges), dtype=int)
        centres = np.empty(len(ranges))
        for k, (first, last) in enumerate(ranges):
            window = TimeSeries(
                data=series.data[:, :, first : last + 1],
                sfreq=data.sfreq,
                channels=data.channels,
                segmentation=data.segmentation,
                tmin=data.tmin + first / data.sfreq,
            )
            fit = fit_var(window, order, ridge=self.ridge, demean=self.demean)
            values[..., k] = dtf_from_coefs(fit.coefs, freqs, data.sfreq)
            radius[k] = spectral_radius(fit.coefs)
            rows[k] = fit.n_obs
            centres[k] = data.tmin + (first + last) / 2.0 / data.sfreq

        n_coefficients = data.n_channels * order
        checks = [
            self._sampling(n_coefficients, int(rows.min())),
            stationarity_check(
                int((radius >= 1.0).sum()), radius.size, unit="windows", stacklevel=4
            ),
            AssumptionCheck(
                "order_source",
                "not_checked",
                detail=(
                    f"the {selection.criterion} criterion chose order {order} of "
                    f"{selection.max_order} candidates on the whole recording and every window "
                    "uses it."
                    if selection is not None
                    else f"order {order} was fixed by the caller, not selected."
                ),
                value=order,
            ),
        ]
        if data.n_trials > 1:
            checks.append(
                AssumptionCheck(
                    "trial_locking",
                    "not_checked",
                    detail=(
                        f"the samples of the same window of {data.n_trials} trials are pooled "
                        "into one regression; this assumes the non-stationarity is time-locked "
                        "to the trial start."
                    ),
                )
            )
        details: dict[str, Any] = {
            "order": order,
            "n_trials": data.n_trials,
            "n_windows": len(ranges),
            "min_rows": int(rows.min()),
            "sfreq": data.sfreq,
            "spectral_radius": radius.tolist(),
            "order_selection": None if selection is None else selection.to_dict(),
        }
        if self.window is not None:
            details["window_samples"] = ranges[0][1] - ranges[0][0] + 1
            details["window_seconds"] = (ranges[0][1] - ranges[0][0] + 1) / data.sfreq
            details["step_samples"] = (
                ranges[1][0] - ranges[0][0]
                if len(ranges) > 1
                else seconds_to_samples(
                    self.window / 2.0 if self.step is None else self.step, data.sfreq, "step"
                )
            )
        else:
            details["segments"] = [list(pair) for pair in self.segments]
            details["segment_samples"] = [last - first + 1 for first, last in ranges]
        provenance = self._provenance(
            data, model_order=order, preprocessing=preprocessing, details=details, checks=checks
        )
        return Connectivity.from_array(
            values, source=list(data.names), freqs=freqs, times=centres, provenance=provenance
        )

    def _sampling(self, n_coefficients: int, min_rows: int) -> AssumptionCheck:
        """Regression rows of the smallest window against 50 per coefficient of an equation."""
        ratio = n_coefficients / min_rows
        detail = (
            f"the smallest window has {min_rows} regression rows for {n_coefficients} "
            f"coefficients per equation (ratio {ratio:.3g}; at least "
            f"{SAMPLES_PER_COEFFICIENT_WARN} rows per coefficient are expected)."
        )
        if min_rows < SAMPLES_PER_COEFFICIENT_WARN * n_coefficients:
            warnings.warn(
                f"{detail} Estimates will be unreliable; use longer windows, a lower order or "
                "fewer channels.",
                DirconnWarning,
                stacklevel=5,
            )
            return AssumptionCheck("sampling_adequacy", "warn", detail=detail, value=ratio)
        return AssumptionCheck("sampling_adequacy", "pass", detail=detail, value=ratio)


def wdtf(data: TimeSeries, **params: Any) -> Connectivity:
    """Windowed DTF of ``data``; shorthand for ``WDTF(**params).fit(data)``."""
    return WDTF(**params).fit(data)
