"""Connectivity estimation of a prepared record: build an estimator, fit it, reduce to bands."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from dirconn.data.result import Connectivity
from dirconn.data.timeseries import TimeSeries
from dirconn.methods.base import Estimator, order_boundary_check
from dirconn.methods.mvar import OrderSelection, select_order
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
    "select_shared_order",
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


def method_kwargs(
    name: str,
    cfg: EstimationConfig,
    ts: TimeSeries | None = None,
    order: int | None = None,
) -> dict[str, Any]:
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
    order : int, optional
        The order selected once for the record (`select_shared_order`); it replaces the
        criterion for every class with an ``order`` field, unless ``params`` sets one.

    Returns
    -------
    dict
        Keyword arguments for the estimator class.
    """
    cls = get_estimator(name)
    kwargs = dict(cfg.params.get(name, {}))
    fields = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    if "order" in fields and "order" not in kwargs:
        kwargs["order"] = cfg.order if order is None else int(order)
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


def build_estimator(
    name: str,
    cfg: EstimationConfig,
    ts: TimeSeries | None = None,
    order: int | None = None,
) -> Estimator:
    """Instantiate the estimator ``name`` from the configuration.

    Parameters
    ----------
    name : str
        Registry name of the method.
    cfg : EstimationConfig
        Estimation configuration (see `method_kwargs`).
    ts : TimeSeries, optional
        The record the estimator will be fitted on; it selects the pipeline defaults.
    order : int, optional
        The order selected once for the record; see `method_kwargs`.

    Returns
    -------
    Estimator
        The configured estimator. Unknown keyword arguments raise the dataclass ``TypeError``.
    """
    return get_estimator(name)(**method_kwargs(name, cfg, ts, order))


def select_shared_order(ts: TimeSeries, cfg: EstimationConfig) -> OrderSelection | None:
    """Select the VAR order of a record once, for every method of the record.

    Parameters
    ----------
    ts : TimeSeries
        The prepared record.
    cfg : EstimationConfig
        ``order`` (a criterion, or a fixed order) and ``order_max``.

    Returns
    -------
    OrderSelection or None
        The selection by ``cfg.order`` up to ``cfg.order_max`` (:func:`dirconn.select_order`), or
        ``None`` when ``cfg.order`` is a fixed order.

    Notes
    -----
    The criterion curves are averaged over channels and do not depend on the scale of the
    data, so the methods that standardise their input select the same order on it. The order
    is then fixed for the surrogates as well: the null distribution of a surrogate test does
    not include the variability of the order selection.
    """
    if not isinstance(cfg.order, str):
        return None
    return select_order(ts, cfg.order_max, criterion=cfg.order)


def with_shared_order(conn: Connectivity, selection: OrderSelection | None) -> Connectivity:
    """Record a shared order selection in a result fitted with that order.

    The selection goes where an estimator that selects its own order puts it
    (``details["order_selection"]`` and the ``order_selection`` assumption check), plus a
    ``shared_order_selection`` history step.
    """
    if selection is None:
        return conn
    prov = conn.provenance
    if prov.model_order is None and "order" not in prov.parameters:
        return conn  # a method without a VAR order (imcoh, mtgc, psi, pcmci, te)
    prov = dataclasses.replace(
        prov,
        details={**prov.details, "order_selection": selection.to_dict()},
        assumptions=prov.assumptions.with_check(order_boundary_check(selection, stacklevel=4)),
    ).with_step({"step": "shared_order_selection", "criterion": selection.criterion})
    return conn.with_provenance(prov)


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


def estimate(
    ts: TimeSeries,
    method: str,
    cfg: EstimationConfig,
    selection: OrderSelection | None = None,
) -> Estimate:
    """Fit ``method`` on ``ts`` and reduce a spectrum to the configured bands.

    Parameters
    ----------
    ts : TimeSeries
        The prepared record.
    method : str
        Registry name of the method.
    cfg : EstimationConfig
        Estimator parameters, bands, reduction and excluded frequency ranges.
    selection : OrderSelection, optional
        The record's shared order selection (`select_shared_order`); the estimator is fitted
        with its order and the result records it.

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
    estimator = build_estimator(method, cfg, ts, None if selection is None else selection.order)
    info = estimator.info
    refuse_masked(ts, method, info.supports_mask)
    full = with_shared_order(estimator.fit(ts), selection)
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
