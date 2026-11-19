"""Control analyses run next to the main estimate."""

from __future__ import annotations

from dirconn.data.result import Connectivity
from dirconn.data.timeseries import TimeSeries
from dirconn.methods.mvar import OrderSelection
from dirconn.methods.time_reversal import trgc
from dirconn.workflows.config import ControlsConfig, EstimationConfig
from dirconn.workflows.steps.estimate import with_shared_order

__all__ = ["time_reversal_control"]


def time_reversal_control(
    ts: TimeSeries,
    cfg: EstimationConfig,
    controls: ControlsConfig,
    selection: OrderSelection | None = None,
) -> Connectivity | None:
    """Time-reversed Granger causality as a control for the directed estimates.

    Parameters
    ----------
    ts : TimeSeries
        The prepared record.
    cfg : EstimationConfig
        Supplies the model order and its upper bound.
    controls : ControlsConfig
        The control is skipped unless ``controls.time_reversal`` is set.
    selection : OrderSelection, optional
        The record's shared order selection; TRGC is fitted with its order.

    Returns
    -------
    Connectivity or None
        The TRGC result, or ``None`` when the control is off.
    """
    if not controls.time_reversal:
        return None
    if selection is None:
        return trgc(ts, order=cfg.order, max_order=cfg.order_max)
    return with_shared_order(trgc(ts, order=selection.order), selection)
