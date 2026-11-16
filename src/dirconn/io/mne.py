"""Conversion of MNE ``Raw`` and ``Epochs`` objects to :class:`~dirconn.TimeSeries`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from dirconn.data.channels import ChannelMeta
from dirconn.data.timeseries import TimeSeries


def timeseries_from_mne(
    inst: Any,
    picks: Sequence[str | int] | None = None,
    *,
    nan: str = "raise",
    bad_annotations: str = "ignore",
    cls: type[TimeSeries] = TimeSeries,
) -> TimeSeries:
    """Build a time series from an MNE ``Raw`` or ``Epochs`` object.

    See :meth:`dirconn.TimeSeries.from_mne` for the parameters. ``cls`` is the class to build,
    so that :meth:`TimeSeries.from_mne` called on a subclass returns an instance of it.
    """
    import mne
    from mne.io import BaseRaw

    if bad_annotations not in ("ignore", "mask"):
        raise ValueError(f"bad_annotations must be 'ignore' or 'mask', got {bad_annotations!r}.")
    if bad_annotations == "mask" and nan != "mask":
        raise ValueError('bad_annotations="mask" needs nan="mask".')

    if isinstance(inst, BaseRaw):
        segmentation, tmin = "continuous", 0.0
    elif isinstance(inst, mne.BaseEpochs):
        segmentation, tmin = "epoched", float(inst.tmin)
    else:
        raise TypeError(f"inst must be an MNE Raw or Epochs object, got {type(inst).__name__}.")

    info = inst.info
    if picks is None:
        idx = [
            int(i)
            for i in mne.pick_types(
                info,
                meg=True,
                eeg=True,
                seeg=True,
                ecog=True,
                dbs=True,
                misc=True,
                exclude="bads",
            )
        ]
        if not idx:
            raise ValueError("No data channels found to convert.")
    else:
        idx = [_channel_index(info["ch_names"], p) for p in picks]
        if not idx:
            raise ValueError("picks is empty.")

    if bad_annotations == "mask":
        if not isinstance(inst, BaseRaw):
            raise ValueError('bad_annotations="mask" is only supported for Raw objects.')
        data = inst.get_data(picks=idx, reject_by_annotation="nan", verbose="error")
    else:
        data = inst.get_data(picks=idx)
    channels = tuple(
        ChannelMeta(
            info["ch_names"][i],
            type=str(mne.channel_type(info, i)),
            pos=_mne_position(info["chs"][i]["loc"]),
        )
        for i in idx
    )
    return cls.from_array(
        data,
        float(info["sfreq"]),
        channels=channels,
        segmentation=segmentation,
        tmin=tmin,
        nan=nan,
    )


def _channel_index(ch_names: Sequence[str], pick: str | int) -> int:
    if isinstance(pick, str):
        try:
            return list(ch_names).index(pick)
        except ValueError:
            raise ValueError(f"Channel {pick!r} is not in the data.") from None
    index = int(pick)
    if not 0 <= index < len(ch_names):
        raise ValueError(f"Channel index {index} is out of range.")
    return index


def _mne_position(loc: Any) -> tuple[float, float, float] | None:
    xyz = np.asarray(loc[:3], dtype=float)
    if not np.isfinite(xyz).all() or not xyz.any():
        return None
    return (float(xyz[0]), float(xyz[1]), float(xyz[2]))
