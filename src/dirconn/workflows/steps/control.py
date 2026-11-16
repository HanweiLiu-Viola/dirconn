"""Control analyses run next to the main estimate."""

from __future__ import annotations

from dirconn.data.result import Connectivity
from dirconn.data.timeseries import TimeSeries
from dirconn.methods.time_reversal import TRGC, trgc
from dirconn.workflows.config import ControlsConfig, EstimationConfig
from dirconn.workflows.steps.estimate import refuse_masked

__all__ = ["time_reversal_control"]


def time_reversal_control(
    ts: TimeSeries, cfg: EstimationConfig, controls: ControlsConfig
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

    Returns
    -------
    Connectivity or None
        The TRGC result, or ``None`` when the control is off.

    Raises
    ------
    ValueError
        If ``ts`` is masked (TRGC does not support masks).
    """
    if not controls.time_reversal:
        return None
    refuse_masked(ts, "trgc", TRGC.info.supports_mask)
    return trgc(ts, order=cfg.order, max_order=cfg.order_max)
