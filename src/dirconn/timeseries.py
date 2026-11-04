"""The input container for all estimators: a trials x channels x times array with metadata."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from dirconn.channels import ChannelMeta

SEGMENTATIONS: tuple[str, ...] = ("continuous", "epoched")


@dataclass(frozen=True, eq=False, repr=False)
class TimeSeries:
    """Multichannel time series with explicit segmentation.

    The data always have shape ``(n_trials, n_channels, n_times)``. Whether the data are one
    continuous record or a set of epochs is stored explicitly in ``segmentation`` and is never
    inferred from ``n_trials``: a single epoch is still ``"epoched"``.

    Parameters
    ----------
    data : ndarray, shape (n_trials, n_channels, n_times)
        Real-valued, finite samples. Converted to ``float64`` and exposed as a read-only view;
        a ``float64`` input array is not copied.
    sfreq : float
        Sampling frequency in Hz.
    channels : tuple of ChannelMeta
        One entry per channel, with unique names.
    segmentation : {"continuous", "epoched"}
        ``"continuous"`` requires ``n_trials == 1``.
    tmin : float
        Time of the first sample in seconds, relative to the epoch event (0 for continuous data).
    """

    data: np.ndarray
    sfreq: float
    channels: tuple[ChannelMeta, ...]
    segmentation: str
    tmin: float = 0.0

    def __post_init__(self) -> None:
        """Validate and normalise the fields."""
        arr = np.asarray(self.data)
        if arr.ndim != 3:
            raise ValueError(
                "data must have 3 dimensions (trials, channels, times), "
                f"got {arr.ndim} dimension(s)."
            )
        if arr.dtype.kind not in "fiu":
            raise ValueError(f"data must be real-valued, got dtype {arr.dtype}.")
        if arr.dtype != np.float64:
            arr = arr.astype(np.float64)
        if not np.isfinite(arr).all():
            raise ValueError("data must be finite: found NaN or infinite samples.")
        if arr.shape[2] < 1 or arr.shape[0] < 1 or arr.shape[1] < 1:
            raise ValueError(f"data must not be empty, got shape {arr.shape}.")
        view = arr.view()
        view.flags.writeable = False
        object.__setattr__(self, "data", view)

        sfreq = float(self.sfreq)
        if not math.isfinite(sfreq) or sfreq <= 0:
            raise ValueError(f"sfreq must be a positive finite number, got {self.sfreq!r}.")
        object.__setattr__(self, "sfreq", sfreq)

        tmin = float(self.tmin)
        if not math.isfinite(tmin):
            raise ValueError(f"tmin must be finite, got {self.tmin!r}.")
        object.__setattr__(self, "tmin", tmin)

        if self.segmentation not in SEGMENTATIONS:
            raise ValueError(
                f"segmentation must be one of {SEGMENTATIONS}, got {self.segmentation!r}."
            )
        if self.segmentation == "continuous" and arr.shape[0] != 1:
            raise ValueError(
                "continuous data must have a single trial; "
                f"got {arr.shape[0]} trials (use segmentation='epoched')."
            )

        channels = tuple(self.channels)
        if len(channels) != arr.shape[1]:
            raise ValueError(
                f"Got {len(channels)} channel descriptions for data with {arr.shape[1]} channels."
            )
        names = [ch.name for ch in channels]
        if len(set(names)) != len(names):
            duplicated = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Channel names must be unique; duplicated: {duplicated}.")
        object.__setattr__(self, "channels", channels)

    def __repr__(self) -> str:
        """Return a short summary that does not print the samples."""
        return (
            f"TimeSeries(n_trials={self.n_trials}, n_channels={self.n_channels}, "
            f"n_times={self.n_times}, sfreq={self.sfreq:g}, segmentation={self.segmentation!r})"
        )

    @property
    def n_trials(self) -> int:
        """Number of trials (1 for continuous data)."""
        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return int(self.data.shape[1])

    @property
    def n_times(self) -> int:
        """Number of samples per trial."""
        return int(self.data.shape[2])

    @property
    def names(self) -> tuple[str, ...]:
        """Channel names in data order."""
        return tuple(ch.name for ch in self.channels)

    @property
    def duration(self) -> float:
        """Duration of one trial in seconds."""
        return self.n_times / self.sfreq

    def pick(self, channels: Sequence[str | int]) -> TimeSeries:
        """Return a series with only the selected channels, in the given order.

        Parameters
        ----------
        channels : sequence of str or int
            Channel names or zero-based indices; they may be mixed. Every channel can appear
            once. Negative indices are rejected so that a selection never means something
            different from what it says.

        Returns
        -------
        TimeSeries
            A new series that owns a copy of the selected data and keeps ``sfreq``,
            ``segmentation`` and ``tmin``. The channel metadata travels with the channels.

        Raises
        ------
        ValueError
            For an unknown name, an index outside the channels, a duplicate or an empty selection.
        TypeError
            For a single string instead of a sequence, or an entry that is neither a name nor an
            integer index.
        """
        if isinstance(channels, (str, bytes)):
            raise TypeError(
                "channels must be a sequence of names or indices, not a single string; "
                f"use [{channels!r}]."
            )
        position = {name: i for i, name in enumerate(self.names)}
        chosen: list[int] = []
        for entry in channels:
            if isinstance(entry, str):
                if entry not in position:
                    raise ValueError(f"Channel {entry!r} is not in the data.")
                chosen.append(position[entry])
            elif isinstance(entry, (int, np.integer)) and not isinstance(entry, bool):
                index = int(entry)
                if not 0 <= index < self.n_channels:
                    raise ValueError(
                        f"Channel index {index} is out of range for {self.n_channels} channels."
                    )
                chosen.append(index)
            else:
                raise TypeError(f"Each channel must be a name or an integer index, got {entry!r}.")
        if not chosen:
            raise ValueError("Pick at least one channel.")
        repeated = sorted({self.names[i] for i in chosen if chosen.count(i) > 1})
        if repeated:
            raise ValueError(f"The selection contains duplicate channels: {repeated}.")
        return TimeSeries(
            data=self.data[:, chosen, :],
            sfreq=self.sfreq,
            channels=tuple(self.channels[i] for i in chosen),
            segmentation=self.segmentation,
            tmin=self.tmin,
        )

    @classmethod
    def from_array(
        cls,
        data: Any,
        sfreq: float,
        *,
        names: Sequence[str] | None = None,
        channels: Sequence[ChannelMeta] | None = None,
        segmentation: str | None = None,
        tmin: float = 0.0,
    ) -> TimeSeries:
        """Build a time series from a numpy array.

        Parameters
        ----------
        data : array_like, shape (n_channels, n_times) or (n_trials, n_channels, n_times)
            Two-dimensional input is one trial and defaults to ``"continuous"``. Three-dimensional
            input requires an explicit ``segmentation``.
        sfreq : float
            Sampling frequency in Hz.
        names : sequence of str, optional
            Channel names; defaults to ``"ch0"``, ``"ch1"``, ... Cannot be combined with
            ``channels``.
        channels : sequence of ChannelMeta, optional
            Full channel descriptions.
        segmentation : {"continuous", "epoched"}, optional
            See the class documentation.
        tmin : float
            Time of the first sample in seconds.

        Returns
        -------
        TimeSeries
        """
        arr = np.asarray(data)
        if arr.ndim == 2:
            segmentation = segmentation or "continuous"
            arr = arr[np.newaxis]
        elif arr.ndim == 3:
            if segmentation is None:
                raise ValueError(
                    "segmentation is required for three-dimensional input: pass "
                    "'continuous' or 'epoched' (it is not inferred from the number of trials)."
                )
        else:
            raise ValueError(f"data must have 2 or 3 dimensions, got {arr.ndim} dimension(s).")

        if names is not None and channels is not None:
            raise ValueError("Pass either names or channels, not both.")
        n_channels = arr.shape[1]
        if channels is None:
            if names is None:
                names = [f"ch{i}" for i in range(n_channels)]
            if len(names) != n_channels:
                raise ValueError(f"Got {len(names)} names for data with {n_channels} channels.")
            channels = [ChannelMeta(str(n)) for n in names]
        return cls(
            data=arr, sfreq=sfreq, channels=tuple(channels), segmentation=segmentation, tmin=tmin
        )

    @classmethod
    def from_mne(cls, inst: Any, picks: Sequence[str | int] | None = None) -> TimeSeries:
        """Build a time series from an MNE ``Raw`` or ``Epochs`` object.

        ``Raw`` becomes ``"continuous"`` and ``Epochs`` becomes ``"epoched"``. By default the
        data channels (MEG, EEG, sEEG, ECoG, DBS and misc) are used and bad channels are
        dropped; stimulus, EOG and ECG channels are ignored. Channel positions are taken from
        the info when they are set. ``node_kind`` is ``"sensor"`` for every channel; region
        metadata for ROI series is supplied by the BIDS reader or built by the caller.

        Parameters
        ----------
        inst : mne.io.BaseRaw or mne.BaseEpochs
            The data to convert.
        picks : sequence of str or int, optional
            Channel names or indices, in the order wanted. Overrides the default selection.

        Returns
        -------
        TimeSeries
        """
        import mne
        from mne.io import BaseRaw

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
            data, float(info["sfreq"]), channels=channels, segmentation=segmentation, tmin=tmin
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
