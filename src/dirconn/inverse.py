"""From sensor series to region series: a linear spatial filter and region extraction.

Every method here is linear in the data: an estimate of the activity of all sources is a
``(n_sources, n_channels)`` matrix applied to the sensor data. Applying it to all 7498 sources of
the sample head model for a long recording would need gigabytes, so the filter is built once and
applied one region at a time (:func:`roi_series`); region extraction follows MNE's ``mean_flip`` and
``pca_flip`` definitions and is checked against :func:`mne.extract_label_time_course`.

The default method is minimum norm. Of the ways to summarise a region, it is the only one that
recovered the node under every source model tried (focal, partly coherent patch, coherent patch:
correlation 0.95 to 0.99); the standard beamformer recipe, an LCMV filter per vertex and the first
principal component of the region, recovered almost nothing (0.01 to 0.09), although the beamformer
itself is right at the vertex of a focal source. See ``docs/benchmark.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dirconn.channels import ChannelMeta
from dirconn.headmodel import HeadModel
from dirconn.network import dk_node
from dirconn.timeseries import TimeSeries

INVERSE_METHODS: tuple[str, ...] = ("lcmv", "mne", "dspm", "eloreta")
ROI_MODES: tuple[str, ...] = ("pca_flip", "mean_flip")

_DEFAULT_REG = {"lcmv": 0.05, "mne": 1.0 / 9.0, "dspm": 1.0 / 9.0, "eloreta": 1.0 / 9.0}
_MNE_METHOD = {"mne": "MNE", "dspm": "dSPM", "eloreta": "eLORETA"}


def extract_roi(source_data: np.ndarray, flip: np.ndarray, mode: str = "pca_flip") -> np.ndarray:
    """Summarise the sources of one region as a single time course per trial.

    Parameters
    ----------
    source_data : ndarray, shape (n_trials, n_sources, n_times)
        Activity of the sources of the region.
    flip : ndarray, shape (n_sources,)
        ``+1`` or ``-1`` per source: its sign relative to the mean orientation of the region
        (:attr:`dirconn.headmodel.HeadModel.source_flip`).
    mode : {"pca_flip", "mean_flip"}
        ``"mean_flip"`` averages the sign-flipped sources. ``"pca_flip"`` takes the first
        principal component of the sources, with the sign that agrees with the flips and the
        power of the average source (MNE's definition). With several trials one decomposition is
        done over all of them, so every trial is projected on the same pattern and the sign cannot
        change from trial to trial.

    Returns
    -------
    ndarray, shape (n_trials, n_times)
    """
    if mode not in ROI_MODES:
        raise ValueError(f"mode must be one of {ROI_MODES}, got {mode!r}.")
    data = np.asarray(source_data, dtype=float)
    if data.ndim != 3:
        raise ValueError(
            f"source_data must have 3 dimensions (trials, sources, times), got {data.ndim}."
        )
    n_trials, n_sources, n_times = data.shape
    if n_sources == 0:
        raise ValueError("The region has no source.")
    flip = np.asarray(flip, dtype=float)
    if flip.shape != (n_sources,) or not np.isin(flip, (-1.0, 1.0)).all():
        raise ValueError(
            f"flip must hold -1 or 1 for each of the {n_sources} sources, got {flip.tolist()}."
        )
    if mode == "mean_flip":
        return np.mean(flip[None, :, None] * data, axis=1)
    pooled = np.moveaxis(data, 1, 0).reshape(n_sources, n_trials * n_times)
    u, s, _ = np.linalg.svd(pooled, full_matrices=False)
    sign = np.sign(u[:, 0] @ flip)
    if sign == 0:  # exactly orthogonal to the flips: the sign is arbitrary, keep it positive
        sign = 1.0
    scale = np.linalg.norm(s) / np.sqrt(n_sources)
    return sign * scale * np.einsum("s,bst->bt", u[:, 0], data) / s[0]


def _average_reference_data(head: HeadModel, data: np.ndarray) -> np.ndarray:
    """Average-reference EEG data ``(trials, channels, times)``; other modalities are unchanged."""
    if head.modality != "eeg":
        return data
    return data - data.mean(axis=1, keepdims=True)


def spatial_filter(
    head: HeadModel,
    data: np.ndarray,
    *,
    method: str = "mne",
    reg: float | None = None,
) -> np.ndarray:
    """Return the linear map from sensor data to source activity.

    The source estimate of data ``x`` is ``weights @ x``; for every method it equals the estimate
    MNE itself computes (checked to rounding error). The data are needed by the beamformer, which
    builds its weights from their covariance; the minimum-norm methods use them only to set the
    scale of a white noise covariance.

    Parameters
    ----------
    head : HeadModel
        Must carry ``forward`` and ``info`` (a head model loaded from the sample data).
    data : ndarray, shape (n_trials, n_channels, n_times)
        Sensor data in the channel order of ``head``. EEG is average-referenced for the covariance.
    method : {"mne", "lcmv", "dspm", "eloreta"}
        ``"mne"`` (the default), ``"dspm"`` and ``"eloreta"``: minimum-norm estimates with a white
        noise covariance scaled to the mean channel variance, fixed source orientation and no depth
        weighting (the stored forward is already fixed-orientation, for which MNE offers no depth
        weighting). ``"lcmv"``: unit-noise-gain LCMV beamformer; its region series depend strongly
        on how the region is summarised (see the module documentation).
    reg : float, optional
        LCMV: the covariance regularisation (default 0.05). Others: ``lambda2`` (default 1/9).

    Returns
    -------
    ndarray, shape (n_sources, n_channels)
    """
    import mne

    if method not in INVERSE_METHODS:
        raise ValueError(f"method must be one of {INVERSE_METHODS}, got {method!r}.")
    if head.forward is None or head.info is None:
        raise ValueError(
            "The head model has no forward solution or info; load it with load_head_model."
        )
    x = np.asarray(data, dtype=float)
    if x.ndim != 3 or x.shape[1] != head.n_channels:
        raise ValueError(
            f"data must have shape (trials, {head.n_channels} channels, times), got {x.shape}."
        )
    level = _DEFAULT_REG[method] if reg is None else float(reg)
    if not np.isfinite(level) or level <= 0:
        raise ValueError(f"reg must be positive and finite, got {reg!r}.")

    x = _average_reference_data(head, x)
    names = list(head.channel_names)
    if method == "lcmv":
        pooled = np.moveaxis(x, 1, 0).reshape(head.n_channels, -1)
        cov = mne.Covariance(
            np.cov(pooled), names, [], [], nfree=pooled.shape[1] - 1, verbose="ERROR"
        )
        filters = mne.beamformer.make_lcmv(
            head.info,
            head.forward,
            cov,
            reg=level,
            pick_ori=None,
            weight_norm="unit-noise-gain",
            rank="info",
            verbose="ERROR",
        )
        weights = np.asarray(filters["weights"], dtype=float)
    else:
        variance = float(np.mean(np.var(x, axis=(0, 2))))
        noise = mne.Covariance(
            np.eye(head.n_channels) * variance, names, [], [], nfree=1000, verbose="ERROR"
        )
        inverse = mne.minimum_norm.make_inverse_operator(
            head.info,
            head.forward,
            noise,
            loose=0.0,
            depth=None,
            fixed=True,
            rank="info",
            verbose="ERROR",
        )
        identity = mne.EvokedArray(np.eye(head.n_channels), head.info, tmin=0.0, verbose="ERROR")
        weights = np.asarray(
            mne.minimum_norm.apply_inverse(
                identity,
                inverse,
                lambda2=level,
                method=_MNE_METHOD[method],
                pick_ori=None,
                verbose="ERROR",
            ).data,
            dtype=float,
        )
    view = weights.view()
    view.flags.writeable = False
    return view


def roi_series(
    data: TimeSeries,
    head: HeadModel,
    rois: Sequence[str],
    *,
    method: str = "mne",
    reg: float | None = None,
    mode: str = "pca_flip",
) -> TimeSeries:
    """Return the time courses of regions estimated from sensor data.

    Parameters
    ----------
    data : TimeSeries
        Sensor data with exactly the channels of ``head``, in its order.
    head : HeadModel
    rois : sequence of str
        DK labels such as ``"precuneus-rh"``; the series come back in this order.
    method, reg : see :func:`spatial_filter`.
    mode : {"pca_flip", "mean_flip"}
        See :func:`extract_roi`.

    Returns
    -------
    TimeSeries
        One cortical node per region, with the segmentation and time axis of ``data``. The
        amplitude is in the arbitrary units of the filter.
    """
    if tuple(data.names) != tuple(head.channel_names):
        raise ValueError(
            "data must have exactly the channels of the head model, in its order "
            f"(got {data.names[:3]}... for {head.channel_names[:3]}...)."
        )
    unknown = [name for name in rois if name not in head.roi_names]
    if unknown:
        raise ValueError(f"Region(s) {unknown} are not in the head model.")
    weights = spatial_filter(head, data.data, method=method, reg=reg)
    x = _average_reference_data(head, np.asarray(data.data, dtype=float))
    series = []
    for name in rois:
        idx = np.flatnonzero(head.source_roi == head.roi_names.index(name))
        sources = np.einsum("sc,bct->bst", weights[idx], x)
        series.append(extract_roi(sources, head.source_flip[idx], mode))
    channels: list[ChannelMeta] = [dk_node(name) for name in rois]
    return TimeSeries(
        np.stack(series, axis=1), data.sfreq, tuple(channels), data.segmentation, data.tmin
    )


def crosstalk_from_series(
    extracted: np.ndarray,
    sources: np.ndarray,
    own: Sequence[int],
    *,
    return_outside: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return how much of each source appears in each extracted region series.

    The extracted series are regressed on the (independent, active) sources; row ``i`` of the
    coefficients is divided by its own coefficient, so the diagonal is 1 and the scale of a series
    (or its sign) does not matter. ``matrix[i, j]`` is the leakage of source ``own[j]`` into the
    series ``i``, relative to the response of series ``i`` to its own source.

    Parameters
    ----------
    extracted : ndarray, shape (n_extracted, n_times)
    sources : ndarray, shape (n_sources, n_times)
        Independent activity of all sources that were active, in the same samples.
    own : sequence of int
        For every extracted series the row of ``sources`` it belongs to.
    return_outside : bool
        Also return, per extracted series, the root mean square of the relative leakage from the
        active sources that are not in ``own``.

    Returns
    -------
    matrix : ndarray, shape (n_extracted, n_extracted)
    outside : ndarray, shape (n_extracted,), only if ``return_outside``
    """
    e = np.asarray(extracted, dtype=float)
    x = np.asarray(sources, dtype=float)
    if e.ndim != 2 or x.ndim != 2 or e.shape[1] != x.shape[1]:
        raise ValueError(
            "extracted and sources must be 2-dimensional with the same number of samples, "
            f"got {e.shape} and {x.shape}."
        )
    own_index = np.asarray(list(own), dtype=int)
    if own_index.shape != (e.shape[0],) or own_index.min() < 0 or own_index.max() >= x.shape[0]:
        raise ValueError(
            f"own must name, for each of the {e.shape[0]} extracted series, a row of sources "
            f"(0 to {x.shape[0] - 1}); got {list(own)}."
        )
    coefficients = np.linalg.lstsq(x.T, e.T, rcond=None)[0].T  # (n_extracted, n_sources)
    diagonal = coefficients[np.arange(e.shape[0]), own_index]
    scale = np.abs(coefficients).max(axis=1)
    if (np.abs(diagonal) <= 1e-8 * np.maximum(scale, 1e-300)).any():
        raise ValueError("A series does not respond to its own source; its leakage is undefined.")
    relative = coefficients / diagonal[:, None]
    matrix = relative[:, own_index]
    if not return_outside:
        return matrix
    others = np.setdiff1d(np.arange(x.shape[0]), own_index)
    outside = (
        np.sqrt(np.mean(relative[:, others] ** 2, axis=1)) if others.size else np.zeros(e.shape[0])
    )
    return matrix, outside


@dataclass(frozen=True, eq=False)
class Crosstalk:
    """Leakage between regions for one inverse method.

    Attributes
    ----------
    rois : tuple of str
        The regions, in the order of the rows and columns.
    matrix : ndarray, shape (n_rois, n_rois)
        ``matrix[i, j]`` is the response of the series of region ``i`` to activity of region ``j``
        relative to its response to its own activity; the diagonal is 1.
    outside : ndarray, shape (n_rois,)
        Root mean square of the relative leakage into each series from the active regions that
        are not in ``rois``.
    """

    rois: tuple[str, ...]
    matrix: np.ndarray
    outside: np.ndarray


def crosstalk(
    head: HeadModel,
    rois: Sequence[str] | None = None,
    *,
    method: str = "mne",
    reg: float | None = None,
    mode: str = "pca_flip",
    snr_db: float = 10.0,
    n_samples: int = 3000,
    seed: int,
) -> Crosstalk:
    """Measure the leakage between regions of a head model for an inverse method and region mode.

    Every region of the head model is made active with independent unit-variance white activity,
    seen through its mean topography, with white sensor noise at ``snr_db``; the series of
    ``rois`` are extracted exactly as :func:`roi_series` does it (for a beamformer the filter is
    built from these probe data) and :func:`crosstalk_from_series` gives the leakage. A probe with
    all regions active is a population measure: a region that leaks into a network node need not
    be a node itself.

    Parameters
    ----------
    head : HeadModel
    rois : sequence of str, optional
        Regions whose series are extracted; default all.
    method, reg, mode : see :func:`roi_series`.
    snr_db : float
        SNR of the probe, over all sensors.
    n_samples : int
        Samples of the probe.
    seed : int
        Required.
    """
    names = tuple(head.roi_names) if rois is None else tuple(rois)
    unknown = [name for name in names if name not in head.roi_names]
    if unknown:
        raise ValueError(f"Region(s) {unknown} are not in the head model.")
    if not np.isfinite(snr_db):
        raise ValueError(f"snr_db must be finite, got {snr_db!r}.")
    rng = np.random.default_rng(seed)
    activity = rng.standard_normal((head.n_rois, int(n_samples)))
    signal = head.roi_leadfield @ activity
    noise = rng.standard_normal(signal.shape)
    noise *= np.sqrt(np.mean(signal**2) / 10.0 ** (float(snr_db) / 10.0) / np.mean(noise**2))
    probe = TimeSeries(
        (signal + noise)[None],
        100.0,
        tuple(ChannelMeta(name, type=head.modality) for name in head.channel_names),
        "continuous",
    )
    extracted = roi_series(probe, head, names, method=method, reg=reg, mode=mode).data[0]
    own = [head.roi_names.index(name) for name in names]
    result = crosstalk_from_series(extracted, activity, own, return_outside=True)
    assert isinstance(result, tuple)
    matrix, outside = result
    return Crosstalk(names, matrix, outside)
