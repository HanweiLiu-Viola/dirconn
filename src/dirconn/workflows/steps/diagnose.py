"""Cheap diagnostics of a prepared record: valid fraction, channel scale and spectral range.

The step is a pure function; it reads nothing from disk and logs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dirconn.data.provenance import AssumptionCheck
from dirconn.data.timeseries import TimeSeries
from dirconn.diagnostics.spectral_range import (
    SpectralRange,
    spectral_dynamic_range,
    spectral_range_check,
)
from dirconn.workflows.config import PipelineConfig

__all__ = ["Diagnostics", "diagnose"]


@dataclass(frozen=True)
class Diagnostics:
    """Diagnostics of one prepared record.

    Parameters
    ----------
    valid_fraction : float
        Share of valid samples; 1.0 without a mask.
    n_trials, n_times : int
        Shape of the record.
    sfreq : float
        Sampling frequency in Hz.
    channel_std : dict of str to float
        Standard deviation per channel over the valid samples.
    spectral : SpectralRange or None
        The spectral dynamic range, or ``None`` when the check is switched off.
    checks : tuple of AssumptionCheck
        Assumption checks derived from the diagnostics.
    """

    valid_fraction: float
    n_trials: int
    n_times: int
    sfreq: float
    channel_std: dict[str, float]
    spectral: SpectralRange | None
    checks: tuple[AssumptionCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "valid_fraction": float(f"{self.valid_fraction:.6g}"),
            "n_trials": self.n_trials,
            "n_times": self.n_times,
            "sfreq": self.sfreq,
            "channel_std": {k: float(f"{v:.6g}") for k, v in self.channel_std.items()},
            "spectral": None if self.spectral is None else self.spectral.to_dict(),
            "checks": [c.to_dict() for c in self.checks],
        }


def diagnose(ts: TimeSeries, cfg: PipelineConfig) -> Diagnostics:
    """Diagnose a prepared record.

    Parameters
    ----------
    ts : TimeSeries
        The prepared record.
    cfg : PipelineConfig
        Uses ``estimation.bands`` and ``controls.spectral_range_check``.

    Returns
    -------
    Diagnostics
        Valid fraction, per-channel standard deviation and, when enabled, the spectral range check.
    """
    n_trials, n_channels, n_times = ts.data.shape
    data = np.asarray(ts.data, dtype=float)
    if ts.valid is None:
        valid_fraction = 1.0
        std = data.std(axis=(0, 2))
    else:
        mask = np.asarray(ts.valid, dtype=bool)
        valid_fraction = float(mask.sum() / mask.size)
        samples = np.moveaxis(data, 1, 0)[:, mask]  # (n_channels, n_valid)
        std = samples.std(axis=1) if samples.shape[1] else np.full(n_channels, np.nan)
    spectral = None
    checks: tuple[AssumptionCheck, ...] = ()
    if cfg.controls.spectral_range_check:
        spectral = spectral_dynamic_range(ts, cfg.estimation.bands)
        checks = (spectral_range_check(spectral),)
    return Diagnostics(
        valid_fraction=valid_fraction,
        n_trials=int(n_trials),
        n_times=int(n_times),
        sfreq=float(ts.sfreq),
        channel_std={name: float(s) for name, s in zip(ts.names, std, strict=True)},
        spectral=spectral,
        checks=checks,
    )
