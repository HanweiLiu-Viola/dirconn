"""Connectivity estimation of a prepared record: build an estimator, fit it, reduce to bands."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from dirconn.data.result import Connectivity
from dirconn.data.timeseries import TimeSeries
from dirconn.methods.base import Estimator
from dirconn.methods.registry import get_estimator
from dirconn.workflows.config import EstimationConfig

__all__ = [
    "CONTINUOUS_DEFAULTS",
    "EPOCHED_DEFAULTS",
    "Estimate",
    "build_estimator",
    "estimate",
    "method_kwargs",
    "refuse_masked",
]

CONTINUOUS_DEFAULTS: dict[str, dict[str, Any]] = {
    "mtgc": {"segment_length": 2.0},
    "imcoh": {"segment_length": 2.0},
    "psi": {"segment_length": 2.0},
    "wdtf": {"window": 2.0},
}
"""Pipeline defaults for parameters the library leaves without a default, on continuous records.

The library has no default for them on purpose: ``segment_length`` sets the frequency
resolution (1 / segment_length Hz) and ``window`` the time resolution. The pipeline fills them so
that every method runs with a plain configuration; ``[estimation.<method>]`` overrides them.
"""

EPOCHED_DEFAULTS: dict[str, dict[str, Any]] = {"wdtf": {"window": 0.5}}
"""Pipeline defaults on epoched records; the multitaper methods use whole epochs there."""


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


def method_kwargs(name: str, cfg: EstimationConfig, ts: TimeSeries | None = None) -> dict[str, Any]:
    """Return the keyword arguments the pipeline builds estimator ``name`` with.

    Parameters
    ----------
    name : str
        Registry name of the method.
    cfg : EstimationConfig
        Supplies ``params[name]``, and ``order`` / ``order_max`` for the classes that have such
        fields and do not get them from ``params``.
    ts : TimeSeries, optional
        The record the estimator will be fitted on. When given, the pipeline defaults of
        `CONTINUOUS_DEFAULTS` or `EPOCHED_DEFAULTS` fill the keys ``params`` does not set, and
        ``psi`` gets the configured bands, ordered by their centres.

    Returns
    -------
    dict
        Keyword arguments for the estimator class.
    """
    cls = get_estimator(name)
    kwargs = dict(cfg.params.get(name, {}))
    fields = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    if "order" in fields and "order" not in kwargs:
        kwargs["order"] = cfg.order
    if "max_order" in fields and "max_order" not in kwargs:
        kwargs["max_order"] = cfg.order_max
    if ts is not None:
        defaults = CONTINUOUS_DEFAULTS if ts.segmentation == "continuous" else EPOCHED_DEFAULTS
        for key, value in defaults.get(name, {}).items():
            kwargs.setdefault(key, value)
        if name == "psi":
            kwargs["bands"] = sorted(
                ([float(lo), float(hi)] for lo, hi in cfg.bands.values()),
                key=lambda band: band[0] + band[1],
            )
    return kwargs


def build_estimator(name: str, cfg: EstimationConfig, ts: TimeSeries | None = None) -> Estimator:
    """Instantiate the estimator ``name`` from the configuration.

    Parameters
    ----------
    name : str
        Registry name of the method.
    cfg : EstimationConfig
        Estimation configuration (see `method_kwargs`).
    ts : TimeSeries, optional
        The record the estimator will be fitted on; it selects the pipeline defaults.

    Returns
    -------
    Estimator
        The configured estimator. Unknown keyword arguments raise the dataclass ``TypeError``.
    """
    return get_estimator(name)(**method_kwargs(name, cfg, ts))


def _check_psi_bands(full: Connectivity, cfg: EstimationConfig) -> None:
    """PSI has one value per band, at the band centre: each band must hold its centre only."""
    centres = np.asarray(full.freqs, dtype=float)
    for name, (lo, hi) in cfg.bands.items():
        inside = int(((centres >= lo) & (centres <= hi)).sum())
        if inside != 1:
            raise ValueError(
                f"psi gives one value per band, at its centre; band {name!r} ({lo:g}-{hi:g} Hz) "
                f"holds {inside} band centres. Use bands that do not contain each other's centres "
                "to run psi."
            )


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
    estimator = build_estimator(method, cfg, ts)
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
    if method == "psi":
        _check_psi_bands(full, cfg)
    bands = {name: full.reduce_freq((lo, hi), cfg.reduce) for name, (lo, hi) in cfg.bands.items()}
    return Estimate(method=method, full=full, bands=bands)
