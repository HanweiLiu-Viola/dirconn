"""The empty-room recording of the MNE ``sample`` dataset as a source of real MEG noise.

``ernoise_raw.fif`` is 110 s of MEG without a subject: the sensor noise and the magnetic
environment of the room, which a simulation can only imitate. For a head model of magnetometers
or gradiometers this module returns that recording at the sampling rate of a simulation, for the
channels of the head model and in its order, so that it can replace the simulated noise
(:func:`dirconn.simulation.simulate`, ``noise_series``).

The recording is used as it is stored: components above the new Nyquist frequency are removed by
MNE's FFT resampling, the three projectors of the file (PCA of the environmental noise) are **not**
applied and no signal-space separation is done. The noise of the two sensor types is very
different: between magnetometers it is strongly shared (mean absolute correlation of the series
0.57 at 100 Hz) and dominated by slow fluctuations (lag-1 autocorrelation 0.89), between
gradiometers it is nearly independent (0.04) and less slow (0.55).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np

from dirconn.datasets import require_sample
from dirconn.headmodel import HeadModel

EMPTY_ROOM_FILE = "MEG/sample/ernoise_raw.fif"
MEG_MODALITIES = ("mag", "grad")


@cache
def _recording(channel_names: tuple[str, ...], sfreq: float, root: str) -> np.ndarray:
    import mne

    raw = mne.io.read_raw_fif(Path(root) / EMPTY_ROOM_FILE, preload=True, verbose="ERROR")
    missing = [n for n in channel_names if n not in raw.ch_names]
    if missing:
        raise ValueError(f"The empty-room recording lacks the channels {missing[:5]}.")
    raw.pick(list(channel_names))
    raw.resample(float(sfreq), verbose="ERROR")
    data = np.vstack([raw.get_data(picks=[raw.ch_names.index(n)]) for n in channel_names])
    data.flags.writeable = False
    return data


def empty_room_series(
    head: HeadModel, *, sfreq: float = 100.0, path: str | Path | None = None
) -> np.ndarray:
    """Return the whole empty-room recording for the sensors of ``head``.

    Parameters
    ----------
    head : HeadModel
        A magnetometer or gradiometer head model; the channels and their order are its own.
    sfreq : float
        Sampling rate in Hz to resample to (default 100, the one of the scenarios).
    path : path-like, optional
        Where to look for the sample dataset (see :func:`dirconn.datasets.require_sample`).

    Returns
    -------
    ndarray, shape (n_sensors, n_samples)
        Read-only; cached, so asking again is free. About 110 s long.

    Raises
    ------
    ValueError
        For a head model that is not MEG: there is no empty room for EEG.
    """
    if head.modality not in MEG_MODALITIES:
        raise ValueError(
            f"There is an empty-room recording for MEG only ({MEG_MODALITIES}), "
            f"not for {head.modality!r}."
        )
    root = require_sample(path=path)
    return _recording(tuple(head.channel_names), float(sfreq), str(root))


def n_windows(
    head: HeadModel, n_times: int, *, sfreq: float = 100.0, path: str | Path | None = None
) -> int:
    """Return how many non-overlapping windows of ``n_times`` samples the recording holds."""
    return int(empty_room_series(head, sfreq=sfreq, path=path).shape[1] // int(n_times))


def empty_room_window(
    head: HeadModel,
    window: int,
    n_times: int,
    *,
    sfreq: float = 100.0,
    path: str | Path | None = None,
) -> np.ndarray:
    """Return block ``window`` of the recording: ``n_times`` samples from ``window * n_times``.

    Raises
    ------
    ValueError
        If the recording (about 110 s) is shorter than one block, or has no such window.
    """
    data = empty_room_series(head, sfreq=sfreq, path=path)
    available = data.shape[1] // int(n_times)
    seconds = data.shape[1] / float(sfreq)
    if available == 0:
        raise ValueError(
            f"The empty-room recording ({seconds:.0f} s) is shorter than {n_times / sfreq:g} s."
        )
    if not 0 <= int(window) < available:
        raise ValueError(
            f"There is no window {window}: the {seconds:.0f} s recording holds {available} "
            f"windows of {n_times / sfreq:g} s (0 to {available - 1})."
        )
    start = int(window) * int(n_times)
    return np.array(data[:, start : start + int(n_times)])[None]
