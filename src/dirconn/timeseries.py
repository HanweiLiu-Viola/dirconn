"""The input container for all estimators: a trials x channels x times array with metadata."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
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
    valid : ndarray of bool, shape (n_trials, n_times), optional
        Which samples can be used at all, as a regressor or as a predicted value. Gaps and bad
        segments are ``False``. The samples themselves must still be finite: put any number there
        (:meth:`from_array` with ``nan="mask"`` puts 0). ``None`` (default) means every sample is
        valid.
    predict : ndarray of bool, shape (n_trials, n_times), optional
        Which samples contribute an equation (a regression row) to a model fit, for example the
        samples inside beta bursts. ``None`` (default) means every sample.

        A VAR model of order ``p`` uses the row at time ``t`` only when ``predict[t]`` and
        ``valid[t - p .. t]`` are all ``True`` (see :func:`dirconn.mvar.lagged_design`). Masks
        that are all ``True`` are stored as ``None``. Estimators that cannot use the masks refuse
        masked data (``MethodInfo.supports_mask``).
    history : tuple of dict
        Steps that produced these data from a recording, oldest first, for example
        :meth:`lock`. Every estimator copies them into the provenance of its result
        (``preprocessing["data_history"]``). Must be JSON serialisable.
    """

    data: np.ndarray
    sfreq: float
    channels: tuple[ChannelMeta, ...]
    segmentation: str
    tmin: float = 0.0
    valid: np.ndarray | None = None
    predict: np.ndarray | None = None
    history: tuple[dict[str, Any], ...] = ()

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
            raise ValueError(
                "data must be finite: found NaN or infinite samples. To treat them as gaps, "
                "build the series with TimeSeries.from_array(..., nan='mask')."
            )
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

        for name in ("valid", "predict"):
            object.__setattr__(self, name, _check_mask(getattr(self, name), name, arr.shape))
        history = tuple(self.history)
        if not all(isinstance(step, dict) for step in history):
            raise TypeError("history must be a sequence of dictionaries.")
        object.__setattr__(self, "history", history)

    def __repr__(self) -> str:
        """Return a short summary that does not print the samples."""
        masked = ", masked" if self.has_mask else ""
        return (
            f"TimeSeries(n_trials={self.n_trials}, n_channels={self.n_channels}, "
            f"n_times={self.n_times}, sfreq={self.sfreq:g}, segmentation={self.segmentation!r}"
            f"{masked})"
        )

    @property
    def has_mask(self) -> bool:
        """Whether the series has a ``valid`` or a ``predict`` mask."""
        return self.valid is not None or self.predict is not None

    def mask_summary(self) -> dict[str, int]:
        """Return the number of samples, valid samples and samples to predict, for provenance.

        Counts are over trials and times (one sample is one time point of all channels).
        """
        n_samples = self.n_trials * self.n_times
        return {
            "n_samples": n_samples,
            "n_valid": n_samples if self.valid is None else int(self.valid.sum()),
            "n_predict": n_samples if self.predict is None else int(self.predict.sum()),
        }

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
            ``segmentation``, ``tmin``, the masks and the history. The channel metadata travels
            with the channels.

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
        return replace(
            self,
            data=self.data[:, chosen, :],
            channels=tuple(self.channels[i] for i in chosen),
        )

    def lock(self, events: Any, tmin: float, tmax: float) -> TimeSeries:
        """Cut epochs around events of a continuous record.

        Every event is rounded to the nearest sample; its epoch runs from ``tmin`` to ``tmax``
        seconds around it, both ends included. An epoch that would reach beyond the record or that
        contains an invalid sample is dropped. The events that were kept and the ones that were
        dropped (with the reason) are recorded as the last entry of ``history`` of the result,
        and from there in the provenance of every result computed on it.

        Parameters
        ----------
        events : array_like of float, shape (n_events,)
            Event times in seconds on the time axis of the record (the first sample is at
            ``self.tmin``). Sample indices ``i`` are times ``self.tmin + i / sfreq``.
        tmin, tmax : float
            Start and end of every epoch relative to its event, in seconds.

        Returns
        -------
        TimeSeries
            Epoched data, one trial per kept event in the order given, with ``tmin`` the rounded
            epoch start. A ``predict`` mask of the record is cut with the data; every kept sample
            is valid, so the result has no ``valid`` mask.

        Raises
        ------
        ValueError
            For epoched data, ``tmin > tmax``, events that are not a non-empty one-dimensional
            array of finite times, or when no event is left.
        """
        first, last = self._window(tmin, tmax)
        times = _event_times(events)
        onsets = np.rint((times - self.tmin) * self.sfreq).astype(int)
        valid = np.ones(self.n_times, dtype=bool) if self.valid is None else self.valid[0]
        kept: list[int] = []
        dropped_edge: list[float] = []
        dropped_invalid: list[float] = []
        for k, onset in enumerate(onsets):
            start, stop = onset + first, onset + last + 1
            if start < 0 or stop > self.n_times:
                dropped_edge.append(float(times[k]))
            elif not valid[start:stop].all():
                dropped_invalid.append(float(times[k]))
            else:
                kept.append(k)
        if not kept:
            raise ValueError(
                f"No event is left: {len(dropped_edge)} epoch(s) cross the edges of the record "
                f"and {len(dropped_invalid)} contain invalid samples."
            )
        step = {
            "step": "lock",
            "tmin": first / self.sfreq,
            "tmax": last / self.sfreq,
            "events": [float(times[k]) for k in kept],
            "dropped_edge": dropped_edge,
            "dropped_invalid": dropped_invalid,
        }
        return self._epochs(onsets[kept], first, last, step)

    def control_locks(
        self, eligible: Any, n: int, tmin: float, tmax: float, *, seed: int
    ) -> TimeSeries:
        """Draw ``n`` control epochs from eligible time points of a continuous record.

        The candidates are the samples where ``eligible`` is ``True`` and whose epoch from
        ``tmin`` to ``tmax`` lies inside the record and contains only valid samples. ``n``
        distinct candidates are drawn without replacement with
        ``RNGController(seed).child(0)`` and sorted in time, so the same seed always gives the
        same locks. Use it to match a set of event locks (:meth:`lock`) with the same number of
        locks at, for example, quiet time points. The draw (seed, number of candidates, times) is
        recorded as the last entry of ``history``.

        Parameters
        ----------
        eligible : array_like of bool, shape (n_times,) or (1, n_times)
            Time points where a control epoch may be locked.
        n : int
            Number of epochs, at least 1.
        tmin, tmax : float
            As in :meth:`lock`.
        seed : int
            Required, non-negative.

        Returns
        -------
        TimeSeries
            Epoched data, one trial per drawn time point, as from :meth:`lock`.

        Raises
        ------
        ValueError
            For epoched data, a mask of the wrong shape, or fewer than ``n`` candidates.
        """
        from dirconn.rng import RNGController, _as_count

        first, last = self._window(tmin, tmax)
        n = _as_count(n, "n", minimum=1)
        controller = RNGController(seed)
        mask = np.asarray(eligible)
        if mask.dtype != bool:
            raise TypeError(f"eligible must be a boolean array, got dtype {mask.dtype}.")
        if mask.shape not in ((self.n_times,), (1, self.n_times)):
            raise ValueError(
                f"eligible must have shape ({self.n_times},), one value per sample of the record, "
                f"got {mask.shape}."
            )
        mask = mask.reshape(-1)
        valid = np.ones(self.n_times, dtype=bool) if self.valid is None else self.valid[0]
        length = last - first + 1
        # windows[s] is True when the window that starts at sample s is entirely valid
        invalid = np.concatenate([[0], np.cumsum(~valid)])
        windows = (invalid[length:] - invalid[:-length]) == 0
        onsets = np.arange(self.n_times)
        starts = onsets + first
        inside = (starts >= 0) & (onsets + last < self.n_times)
        ok = np.zeros(self.n_times, dtype=bool)
        ok[inside] = windows[starts[inside]]
        candidates = np.flatnonzero(mask & ok)
        if candidates.size < n:
            raise ValueError(
                f"There are only {candidates.size} eligible time points whose epoch lies inside "
                f"the record and is valid; cannot draw {n}."
            )
        drawn = np.sort(controller.child(0).choice(candidates, size=n, replace=False))
        step = {
            "step": "control_locks",
            "seed": controller.seed,
            "n": n,
            "n_candidates": int(candidates.size),
            "tmin": first / self.sfreq,
            "tmax": last / self.sfreq,
            "events": [float(self.tmin + s / self.sfreq) for s in drawn],
        }
        return self._epochs(drawn, first, last, step)

    def _window(self, tmin: float, tmax: float) -> tuple[int, int]:
        """Check a locking request; return the epoch's first and last sample offsets."""
        if self.segmentation != "continuous":
            raise ValueError("Locking needs continuous data (one record); these are epoched.")
        tmin, tmax = float(tmin), float(tmax)
        if not (math.isfinite(tmin) and math.isfinite(tmax)) or tmin > tmax:
            raise ValueError(f"tmin and tmax must be finite with tmin <= tmax, got {tmin}, {tmax}.")
        return round(tmin * self.sfreq), round(tmax * self.sfreq)

    def _epochs(
        self, onsets: np.ndarray, first: int, last: int, step: dict[str, Any]
    ) -> TimeSeries:
        index = np.asarray(onsets)[:, None] + np.arange(first, last + 1)[None, :]
        data = self.data[0][:, index].transpose(1, 0, 2)
        predict = None if self.predict is None else self.predict[0][index]
        return TimeSeries(
            data=data,
            sfreq=self.sfreq,
            channels=self.channels,
            segmentation="epoched",
            tmin=first / self.sfreq,
            predict=predict,
            history=(*self.history, step),
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
        valid: Any = None,
        predict: Any = None,
        nan: str = "raise",
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
        valid, predict : array_like of bool, optional
            The masks (see the class documentation), shape ``(n_trials, n_times)``, or
            ``(n_times,)`` for two-dimensional input.
        nan : {"raise", "mask"}
            What to do with samples that are not finite (NaN or infinite). ``"raise"`` (default)
            rejects them. ``"mask"`` replaces them by 0 and marks the time point invalid in
            ``valid`` (for all channels, since a model row needs every channel); this is how a
            record with gaps is read.

        Returns
        -------
        TimeSeries
        """
        if nan not in ("raise", "mask"):
            raise ValueError(f"nan must be 'raise' or 'mask', got {nan!r}.")
        arr = np.asarray(data)
        if arr.ndim == 2:
            segmentation = segmentation or "continuous"
            arr = arr[np.newaxis]
            valid = _one_trial(valid)
            predict = _one_trial(predict)
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
        if nan == "mask" and arr.dtype.kind in "fc":
            bad = ~np.isfinite(arr)
            if bad.any():
                arr = np.where(bad, 0.0, arr)
                good = ~bad.any(axis=1)
                if valid is not None:
                    valid = _check_mask(valid, "valid", arr.shape)
                    good = good if valid is None else good & valid
                valid = good
        return cls(
            data=arr,
            sfreq=sfreq,
            channels=tuple(channels),
            segmentation=segmentation,
            tmin=tmin,
            valid=valid,
            predict=predict,
        )

    @classmethod
    def from_mne(
        cls, inst: Any, picks: Sequence[str | int] | None = None, *, nan: str = "raise"
    ) -> TimeSeries:
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
        nan : {"raise", "mask"}
            As in :meth:`from_array`: ``"mask"`` turns NaN samples (gaps) into invalid samples.

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
            data,
            float(info["sfreq"]),
            channels=channels,
            segmentation=segmentation,
            tmin=tmin,
            nan=nan,
        )


def _check_mask(mask: Any, name: str, shape: tuple[int, ...]) -> np.ndarray | None:
    """Validate a per-sample mask; return a read-only copy, or ``None`` when it is all True."""
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.dtype != bool:
        raise TypeError(f"{name} must be a boolean array, got dtype {arr.dtype}.")
    expected = (shape[0], shape[2])
    if arr.shape != expected:
        raise ValueError(
            f"{name} must have shape (n_trials, n_times) = {expected}, got {arr.shape}."
        )
    if arr.all():
        return None
    out = arr.copy()
    out.flags.writeable = False
    return out


def _one_trial(mask: Any) -> Any:
    """Give a one-dimensional mask of a single trial its trial axis."""
    if mask is None:
        return None
    arr = np.asarray(mask)
    return arr[np.newaxis] if arr.ndim == 1 else arr


def _event_times(events: Any) -> np.ndarray:
    times = np.asarray(events, dtype=float)
    if times.ndim != 1:
        raise ValueError(
            f"events must be a one-dimensional array of times, got shape {times.shape}."
        )
    if times.size == 0:
        raise ValueError("Pass at least one event.")
    if not np.isfinite(times).all():
        raise ValueError("Event times must be finite.")
    return times


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
