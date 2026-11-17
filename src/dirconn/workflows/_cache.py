"""On-disk cache of prepared time series (``.npz`` arrays plus a ``.json`` sidecar)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dirconn.data.channels import ChannelMeta
from dirconn.data.timeseries import TimeSeries

__all__ = ["load_timeseries", "save_timeseries"]

_EMPTY = np.zeros((0, 0), dtype=bool)


def save_timeseries(ts: TimeSeries, path: Path) -> None:
    """Write ``ts`` to ``path`` (``.npz``) and its metadata to ``path.with_suffix(".json")``.

    Parameters
    ----------
    ts : TimeSeries
        Series to store. Absent masks are stored as empty arrays.
    path : Path
        Destination of the arrays; the parent directory must exist.
    """
    path = Path(path)
    np.savez_compressed(
        path,
        data=ts.data,
        valid=_EMPTY if ts.valid is None else ts.valid,
        predict=_EMPTY if ts.predict is None else ts.predict,
    )
    meta = {
        "sfreq": ts.sfreq,
        "segmentation": ts.segmentation,
        "tmin": ts.tmin,
        "channels": [asdict(ch) for ch in ts.channels],
        "history": list(ts.history),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def load_timeseries(path: Path) -> TimeSeries:
    """Read a series written by :func:`save_timeseries`.

    Parameters
    ----------
    path : Path
        The ``.npz`` file.

    Returns
    -------
    TimeSeries
        Equal to the stored series, masks and history included.
    """
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path) as arrays:
        data = np.array(arrays["data"])
        valid = np.array(arrays["valid"])
        predict = np.array(arrays["predict"])
    return TimeSeries(
        data,
        sfreq=meta["sfreq"],
        channels=tuple(ChannelMeta(**ch) for ch in meta["channels"]),
        segmentation=meta["segmentation"],
        tmin=meta["tmin"],
        valid=valid if valid.size else None,
        predict=predict if predict.size else None,
        history=tuple(meta["history"]),
    )
