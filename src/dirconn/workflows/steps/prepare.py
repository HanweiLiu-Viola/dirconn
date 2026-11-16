"""Minimal preparation of a participant record: mask padding, resampling, z-score, event lock.

Every step is a pure function: it returns a new :class:`~dirconn.data.timeseries.TimeSeries` with
one entry appended to ``history`` and never touches its input, the disk or any global state.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage, signal

from dirconn.data.timeseries import TimeSeries
from dirconn.workflows.config import LockConfig, PrepareConfig

__all__ = ["lock_events", "pad_invalid", "prepare", "resample", "zscore"]


def _with_step(ts: TimeSeries, step: dict[str, Any], **changes: Any) -> TimeSeries:
    return dataclasses.replace(ts, history=(*ts.history, step), **changes)


def pad_invalid(ts: TimeSeries, seconds: float) -> TimeSeries:
    """Widen every invalid region by ``seconds`` on both sides.

    Parameters
    ----------
    ts : TimeSeries
        The record. Without a ``valid`` mask it is returned unchanged.
    seconds : float
        Padding in seconds; ``round(seconds * sfreq)`` samples are added on each side of every
        invalid run, per trial.

    Returns
    -------
    TimeSeries
        The record with a wider invalid region; ``predict`` is untouched.
    """
    if ts.valid is None:
        return ts
    n = round(seconds * ts.sfreq)
    invalid = ~np.asarray(ts.valid, dtype=bool)
    before = int(invalid.sum())
    if n > 0:
        invalid = ndimage.binary_dilation(invalid, structure=np.ones((1, 2 * n + 1), dtype=bool))
    step = {
        "step": "pad_invalid",
        "seconds": seconds,
        "invalid_before": before,
        "invalid_after": int(invalid.sum()),
    }
    return _with_step(ts, step, valid=~invalid)


def _resample_mask(
    mask: np.ndarray, up: int, down: int, n_out: int, *, require_all: bool
) -> np.ndarray:
    """Resample a mask; output sample ``j`` covers the original samples ``[j*d/u, (j+1)*d/u)``."""
    n_trials, n_in = mask.shape
    j = np.arange(n_out)
    start = np.minimum((j * down) // up, n_in - 1)
    end = np.minimum(-((-(j + 1) * down) // up), n_in)  # ceil((j + 1) * down / up)
    end = np.maximum(end, start + 1)
    counted = mask if require_all else ~mask
    counts = np.zeros((n_trials, n_in + 1), dtype=np.int64)
    np.cumsum(counted, axis=1, out=counts[:, 1:])
    inside = counts[:, end] - counts[:, start]
    if require_all:
        return inside == (end - start)[None, :]
    return inside == 0


def resample(ts: TimeSeries, sfreq: float) -> TimeSeries:
    """Resample to ``sfreq`` Hz with a polyphase filter.

    Parameters
    ----------
    ts : TimeSeries
        The record.
    sfreq : float
        Target rate in Hz; the ratio to the current rate is approximated by a fraction with
        denominator at most 1000. An equal rate returns the record unchanged.

    Returns
    -------
    TimeSeries
        The resampled record. An output sample is invalid when any original sample it covers was
        invalid; it is a ``predict`` sample only when all original samples it covers were.
        ``tmin`` is unchanged.
    """
    if sfreq == ts.sfreq:
        return ts
    frac = Fraction(sfreq / ts.sfreq).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    data = signal.resample_poly(np.asarray(ts.data), up, down, axis=-1)
    n_out = data.shape[-1]
    valid = predict = None
    if ts.valid is not None:
        valid = _resample_mask(np.asarray(ts.valid, dtype=bool), up, down, n_out, require_all=False)
    if ts.predict is not None:
        predict = _resample_mask(
            np.asarray(ts.predict, dtype=bool), up, down, n_out, require_all=True
        )
    step = {"step": "resample", "from": ts.sfreq, "to": sfreq}
    return _with_step(ts, step, data=data, sfreq=float(sfreq), valid=valid, predict=predict)


def zscore(ts: TimeSeries) -> TimeSeries:
    """Standardise every channel with the mean and std of its valid samples.

    Parameters
    ----------
    ts : TimeSeries
        The record; without a ``valid`` mask all samples count.

    Returns
    -------
    TimeSeries
        Linearly rescaled data (invalid samples are transformed too).

    Raises
    ------
    ValueError
        If a channel has zero standard deviation over its valid samples.
    """
    data = np.asarray(ts.data)
    keep = np.ones((data.shape[0], data.shape[2]), dtype=bool)
    if ts.valid is not None:
        keep = np.asarray(ts.valid, dtype=bool)
    mean = np.zeros(data.shape[1])
    std = np.zeros(data.shape[1])
    for c in range(data.shape[1]):
        values = data[:, c, :][keep]
        if values.size:
            mean[c] = values.mean()
            std[c] = values.std()
    flat = np.flatnonzero(~(std > 0))
    if flat.size:
        raise ValueError(
            f"Cannot z-score channel {ts.names[int(flat[0])]!r}: zero variance over its valid "
            "samples."
        )
    out = (data - mean[None, :, None]) / std[None, :, None]
    return _with_step(ts, {"step": "zscore", "scope": "per_channel_valid_samples"}, data=out)


def lock_events(ts: TimeSeries, events_tsv: Path | None, cfg: LockConfig) -> TimeSeries:
    """Cut epochs around the configured events listed in a BIDS ``events.tsv``.

    Parameters
    ----------
    ts : TimeSeries
        A continuous record.
    events_tsv : Path or None
        The events file with ``onset`` (seconds) and ``trial_type`` columns.
    cfg : LockConfig
        Event names and the window; no events returns the record unchanged.

    Returns
    -------
    TimeSeries
        The epoched record, as produced by :meth:`TimeSeries.lock`.

    Raises
    ------
    ValueError
        If the record is not continuous, or events are requested without an events file.
    """
    if not cfg.events:
        return ts
    if ts.segmentation != "continuous":
        raise ValueError("Event locking needs a continuous record, got epoched data.")
    if events_tsv is None:
        raise ValueError(f"Events {list(cfg.events)} were requested but there is no events.tsv.")
    table = pd.read_csv(events_tsv, sep="\t")
    onsets = table.loc[table["trial_type"].isin(cfg.events), "onset"].to_numpy(dtype=float)
    return ts.lock(onsets, cfg.tmin, cfg.tmax)


def prepare(ts: TimeSeries, cfg: PrepareConfig, *, events_tsv: Path | None) -> TimeSeries:
    """Apply the preparation steps in order.

    Parameters
    ----------
    ts : TimeSeries
        The record as read.
    cfg : PrepareConfig
        Bad-segment handling (``"ignore"`` zeroes invalid samples and drops the mask, otherwise
        the mask is padded by ``pad_bad_seconds``), then resampling, z-score and event lock.
    events_tsv : Path or None
        Events file used by the event lock.

    Returns
    -------
    TimeSeries
        The prepared record.
    """
    if cfg.bad_segments == "ignore" and ts.valid is not None:
        invalid = ~np.asarray(ts.valid, dtype=bool)
        data = np.where(invalid[:, None, :], 0.0, np.asarray(ts.data))
        step = {"step": "ignore_bad_segments", "n_invalid": int(invalid.sum())}
        ts = _with_step(ts, step, data=data, valid=None)
    elif cfg.pad_bad_seconds > 0:
        ts = pad_invalid(ts, cfg.pad_bad_seconds)
    if cfg.resample_hz is not None:
        ts = resample(ts, cfg.resample_hz)
    if cfg.zscore:
        ts = zscore(ts)
    return lock_events(ts, events_tsv, cfg.lock)
