"""Connectivity estimation of a prepared record: build an estimator, fit it, reduce to bands."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from dirconn.data.result import Connectivity
from dirconn.data.timeseries import TimeSeries
from dirconn.methods.base import Estimator
from dirconn.methods.registry import get_estimator
from dirconn.workflows.config import EstimationConfig

__all__ = ["Estimate", "build_estimator", "estimate", "refuse_masked"]


def refuse_masked(ts: TimeSeries, method: str, supports_mask: bool) -> None:
    """Raise a clear error when masked data meets a method that cannot handle masks.

    Parameters
    ----------
    ts : TimeSeries
        The record to be fitted.
    method : str
        Method name used in the message.
    supports_mask : bool
        The method's ``info.supports_mask``.

    Raises
    ------
    ValueError
        If ``ts`` carries a validity mask and the method does not support it.
    """
    if ts.has_mask and not supports_mask:
        raise ValueError(
            f'{method} does not support masked data; set prepare.bad_segments = "ignore" to run it'
        )


def build_estimator(name: str, cfg: EstimationConfig) -> Estimator:
    """Instantiate the estimator ``name`` from the configuration.

    Parameters
    ----------
    name : str
        Registry name of the method.
    cfg : EstimationConfig
        Supplies ``params[name]`` as keyword arguments, and ``order`` / ``order_max`` for the
        classes that have such fields and do not get them from ``params``.

    Returns
    -------
    Estimator
        The configured estimator. Unknown keyword arguments raise the dataclass ``TypeError``.
    """
    cls = get_estimator(name)
    kwargs = dict(cfg.params.get(name, {}))
    fields = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    if "order" in fields and "order" not in kwargs:
        kwargs["order"] = cfg.order
    if "max_order" in fields and "max_order" not in kwargs:
        kwargs["max_order"] = cfg.order_max
    return cls(**kwargs)


@dataclass(frozen=True)
class Estimate:
    """Result of one method on one record.

    Parameters
    ----------
    method : str
        Method name.
    full : Connectivity
        The estimator output (with excluded frequencies set to NaN).
    bands : dict of str to Connectivity
        One band-reduced result per configured band; empty for methods that are not
        frequency resolved.
    """

    method: str
    full: Connectivity
    bands: dict[str, Connectivity]


def estimate(ts: TimeSeries, method: str, cfg: EstimationConfig) -> Estimate:
    """Fit ``method`` on ``ts`` and reduce a spectrum to the configured bands.

    Parameters
    ----------
    ts : TimeSeries
        The prepared record.
    method : str
        Registry name of the method.
    cfg : EstimationConfig
        Estimator parameters, bands, reduction and excluded frequency ranges.

    Returns
    -------
    Estimate
        For frequency-resolved methods the excluded ranges are set to NaN in ``full`` (recorded
        as an ``exclude_freqs`` step in the provenance) before every band is reduced with
        ``cfg.reduce``.

    Raises
    ------
    ValueError
        If ``ts`` is masked and the method does not support masks.
    """
    estimator = build_estimator(method, cfg)
    info = estimator.info
    refuse_masked(ts, method, info.supports_mask)
    full = estimator.fit(ts)
    if not info.frequency_resolved:
        return Estimate(method=method, full=full, bands={})
    if cfg.exclude_freqs:
        freqs = np.asarray(full.freqs, dtype=float)
        hit = np.zeros(freqs.shape, dtype=bool)
        for lo, hi in cfg.exclude_freqs:
            hit |= (freqs >= lo) & (freqs <= hi)
        values = np.array(full.values)
        values[:, :, hit, :] = np.nan
        step = {"step": "exclude_freqs", "ranges": [[lo, hi] for lo, hi in cfg.exclude_freqs]}
        full = Connectivity.from_array(
            values,
            source=full.source,
            target=full.target,
            freqs=full.freqs,
            times=full.times,
            provenance=full.provenance.with_step(step),
        )
    bands = {name: full.reduce_freq((lo, hi), cfg.reduce) for name, (lo, hi) in cfg.bands.items()}
    return Estimate(method=method, full=full, bands=bands)
