"""The result container returned by every estimator."""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from dirconn.bands import BANDS
from dirconn.provenance import Provenance

DIMS: tuple[str, ...] = ("source", "target", "freq", "time")
REDUCTIONS: tuple[str, ...] = ("mean", "median", "max", "peak_freq")
TIME_REDUCTIONS: tuple[str, ...] = ("mean", "median", "max", "peak_time")
_EDGE_TOLERANCE = 1e-9
FORMAT_VERSION = "1"


def _names(names: Sequence[str], what: str, expected: int) -> tuple[str, ...]:
    out = tuple(str(n) for n in names)
    if len(out) != expected:
        raise ValueError(
            f"{what} has {len(out)} names but values has {expected} entries on that axis."
        )
    if len(set(out)) != len(out):
        raise ValueError(f"{what} names must be unique.")
    return out


def _axis(values: Any, what: str, expected: int, *, non_negative: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != expected:
        raise ValueError(
            f"{what} must be one-dimensional with {expected} entries, got shape {arr.shape}."
        )
    if expected > 1:
        if not np.isfinite(arr).all() or (non_negative and (arr < 0).any()):
            kind = "finite and non-negative" if non_negative else "finite"
            raise ValueError(f"{what} must be {kind} when the axis has more than one entry.")
        if not (np.diff(arr) > 0).all():
            raise ValueError(f"{what} must be strictly increasing.")
    elif np.isfinite(arr[0]) and non_negative and arr[0] < 0:
        raise ValueError(f"{what} must be non-negative.")
    out = arr.copy()
    out.flags.writeable = False
    return out


@dataclass(frozen=True, eq=False, repr=False)
class Connectivity:
    """Directed connectivity with fixed dimensions ``(source, target, freq, time)``.

    ``values[i, j]`` is the influence of ``source[i]`` on ``target[j]``. All four dimensions are
    always present. A dimension a method does not resolve has length 1 and a NaN coordinate, so a
    result's shape never depends on parameters. NaN values mark undefined entries (for example
    the diagonal); infinite values are rejected.

    Use :meth:`from_array` to build a result from fewer dimensions.

    Parameters
    ----------
    values : ndarray, shape (n_source, n_target, n_freq, n_time)
        Real or complex values; stored as a read-only ``float64`` or ``complex128`` array.
    source, target : tuple of str
        Unique node names for the first two axes.
    freqs : ndarray, shape (n_freq,)
        Frequencies in Hz, strictly increasing when ``n_freq > 1``; ``[nan]`` when the result is
        not frequency-resolved.
    times : ndarray, shape (n_time,)
        Times in seconds, strictly increasing when ``n_time > 1``; ``[nan]`` when the result is
        not time-resolved.
    provenance : Provenance
        Required: every result states how it was produced.
    """

    values: np.ndarray
    source: tuple[str, ...]
    target: tuple[str, ...]
    freqs: np.ndarray
    times: np.ndarray
    provenance: Provenance

    def __post_init__(self) -> None:
        """Validate and normalise the fields."""
        arr = np.asarray(self.values)
        if arr.ndim != 4:
            raise ValueError(f"values must have 4 dimensions {DIMS}, got {arr.ndim} dimension(s).")
        if arr.dtype.kind == "c":
            arr = arr.astype(np.complex128, copy=False)
        elif arr.dtype.kind in "fiu":
            arr = arr.astype(np.float64, copy=False)
        else:
            raise ValueError(f"values must be real or complex, got dtype {arr.dtype}.")
        if np.isinf(arr).any():
            raise ValueError("values must not contain infinite entries (NaN marks undefined).")
        view = arr.view()
        view.flags.writeable = False

        set_ = object.__setattr__
        set_(self, "values", view)
        set_(self, "source", _names(self.source, "source", arr.shape[0]))
        set_(self, "target", _names(self.target, "target", arr.shape[1]))
        set_(self, "freqs", _axis(self.freqs, "freqs", arr.shape[2], non_negative=True))
        set_(self, "times", _axis(self.times, "times", arr.shape[3], non_negative=False))
        if not isinstance(self.provenance, Provenance):
            raise TypeError("provenance must be a Provenance instance.")

    def __repr__(self) -> str:
        """Return a short summary that does not print the values."""
        shape = "x".join(str(n) for n in self.values.shape)
        return (
            f"Connectivity(method={self.provenance.method!r}, shape={shape}, "
            f"complex={self.is_complex})"
        )

    @property
    def is_complex(self) -> bool:
        """Whether the values are complex."""
        return bool(np.iscomplexobj(self.values))

    @property
    def is_frequency_resolved(self) -> bool:
        """Whether the frequency axis carries real frequencies."""
        return not (self.freqs.shape[0] == 1 and np.isnan(self.freqs[0]))

    @property
    def is_time_resolved(self) -> bool:
        """Whether the time axis carries real times."""
        return not (self.times.shape[0] == 1 and np.isnan(self.times[0]))

    @classmethod
    def from_array(
        cls,
        values: Any,
        *,
        source: Sequence[str],
        target: Sequence[str] | None = None,
        freqs: Any = None,
        times: Any = None,
        provenance: Provenance,
    ) -> Connectivity:
        """Build a result from an array with two to four dimensions.

        Trailing dimensions may be omitted and are added with length 1:
        ``(source, target)`` is broadband and static, ``(source, target, freq)`` is a spectrum.
        A broadband time-resolved result must pass four dimensions with ``freq`` of length 1.

        Parameters
        ----------
        values : array_like
            Real or complex values with 2, 3 or 4 dimensions.
        source : sequence of str
            Names for the first axis.
        target : sequence of str, optional
            Names for the second axis; defaults to ``source``.
        freqs : array_like, optional
            Frequencies in Hz; required when the frequency axis is longer than one.
        times : array_like, optional
            Times in seconds; required when the time axis is longer than one.
        provenance : Provenance
            How the values were produced.

        Returns
        -------
        Connectivity
        """
        arr = np.asarray(values)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis, np.newaxis]
        elif arr.ndim == 3:
            arr = arr[..., np.newaxis]
        elif arr.ndim != 4:
            raise ValueError(f"values must have 2, 3 or 4 dimensions, got {arr.ndim} dimension(s).")
        if freqs is None:
            if arr.shape[2] > 1:
                raise ValueError(
                    "freqs is required when the frequency axis has more than one entry."
                )
            freqs = [np.nan]
        if times is None:
            if arr.shape[3] > 1:
                raise ValueError("times is required when the time axis has more than one entry.")
            times = [np.nan]
        return cls(
            values=arr,
            source=tuple(source),
            target=tuple(source if target is None else target),
            freqs=np.asarray(freqs, dtype=float),
            times=np.asarray(times, dtype=float),
            provenance=provenance,
        )

    def with_provenance(self, provenance: Provenance) -> Connectivity:
        """Return a copy that carries ``provenance`` and shares the same values."""
        return Connectivity(
            values=self.values,
            source=self.source,
            target=self.target,
            freqs=self.freqs,
            times=self.times,
            provenance=provenance,
        )

    def _require_square_real(self, what: str) -> None:
        if self.source != self.target:
            raise ValueError(
                f"{what} needs the same nodes as sources and targets, in the same order "
                f"(sources: {', '.join(self.source)}; targets: {', '.join(self.target)})."
            )
        if self.is_complex:
            raise ValueError(f"Complex values cannot be used by {what}; take abs() or real first.")

    def net_direction(self) -> Connectivity:
        """Return the net direction of every pair: ``values[i, j] - values[j, i]``.

        A positive entry ``[i, j]`` means the influence of ``i`` on ``j`` is stronger than the
        reverse. The result is antisymmetric and its diagonal is undefined (NaN); a pair is
        undefined when either direction is. Frequency and time axes are kept.

        Returns
        -------
        Connectivity
            Same nodes and axes, with one ``net_direction`` step added to the provenance.
        """
        self._require_square_real("net_direction")
        values = self.values - self.values.transpose(1, 0, 2, 3)
        index = np.arange(values.shape[0])
        values[index, index] = np.nan
        return Connectivity(
            values=values,
            source=self.source,
            target=self.target,
            freqs=self.freqs,
            times=self.times,
            provenance=self.provenance.with_step({"op": "net_direction"}),
        )

    def degree_imbalance(self, mask: np.ndarray | None = None) -> xr.DataArray:
        """Return ``log(D_out) - log(D_in)`` of every node, the sum over its edges in and out.

        ``D_out`` of node ``i`` is the sum of ``values[i, j]`` over the other nodes ``j`` and
        ``D_in`` the sum of ``values[j, i]``. A positive value means the node sends more than it
        receives (a source), a negative one the reverse (a sink). The diagonal and undefined cells
        are not counted. The value is NaN where either degree is zero, because the ratio is not
        defined there. Values must be non-negative.

        Parameters
        ----------
        mask : ndarray of bool, optional
            Edges to count: shape ``(n, n)`` for all frequencies and times, or the full shape
            ``(n, n, n_freq, n_time)``, for example ``SurrogateTestResult.significant()``. The
            default counts every defined edge, with the raw values.

        Returns
        -------
        xarray.DataArray
            Dimensions ``("node", "freq", "time")``; the attribute ``provenance`` is the JSON of the
            input's provenance with one ``degree_imbalance`` step that records whether a mask was
            given and how many edges were counted.
        """
        self._require_square_real("degree_imbalance")
        n = len(self.source)
        counted = np.isfinite(self.values) & ~np.eye(n, dtype=bool)[:, :, None, None]
        if (self.values[counted] < 0).any():
            raise ValueError(
                "degree_imbalance needs non-negative values (a degree must not cancel); "
                "a net direction, for example, has negative entries."
            )
        if mask is not None:
            chosen = np.asarray(mask)
            if chosen.dtype != bool:
                raise ValueError("mask must be a boolean array.")
            if chosen.shape == (n, n):
                chosen = chosen[:, :, None, None]
            elif chosen.shape != self.values.shape:
                raise ValueError(
                    f"mask has shape {chosen.shape}; expected {(n, n)} or {self.values.shape}."
                )
            counted = counted & chosen
        weights = np.where(counted, self.values, 0.0)
        out_degree = weights.sum(axis=1)  # over the targets of every source
        in_degree = weights.sum(axis=0)  # over the sources of every target
        defined = (out_degree > 0) & (in_degree > 0)
        result = np.full(out_degree.shape, np.nan)
        result[defined] = np.log(out_degree[defined]) - np.log(in_degree[defined])
        step = {
            "op": "degree_imbalance",
            "mask": "none" if mask is None else "given",
            "edges_counted": int(counted.sum()),
        }
        return xr.DataArray(
            result,
            dims=("node", "freq", "time"),
            coords={
                "node": list(self.source),
                "freq": self.freqs.copy(),
                "time": self.times.copy(),
            },
            name="degree_imbalance",
            attrs={"provenance": self.provenance.with_step(step).to_json()},
        )

    def plot(self, kind: str = "connectivity_grid", **kwargs: Any) -> Any:
        """Draw the result with matplotlib and return the axes.

        Needs the ``plot`` extra (``pip install 'dirconn[plot]'``). The plot only reads the result.

        Parameters
        ----------
        kind : str
            One of ``"connectivity_grid"``, ``"directed_spectrum"``, ``"truth_comparison"``,
            ``"time_frequency"`` and ``"network_graph"``.

            ``"connectivity_grid"``: one ``(source, target)`` matrix, row ``i`` source ``i`` and
            column ``j`` target ``j``, the diagonal left empty, also where an estimator defines it.
            A frequency-resolved result needs ``freq`` (Hz, the nearest bin) or ``band`` (a name
            from ``BANDS`` or ``(fmin, fmax)``, the mean over the band); a time-resolved result
            needs ``time`` (s, the nearest point).
            Other options: ``ax``, ``cmap``, ``vmin``, ``vmax`` and ``colorbar``. Values that are
            all non-negative use a sequential scale from 0; otherwise the scale is diverging and
            centred on 0 (``vmax`` is then the half range). Complex values are refused.

            ``"directed_spectrum"``: both directions of one pair over frequency (``source=``,
            ``target=``, and ``time=`` for a time-resolved result); needs a frequency axis.

            ``"truth_comparison"``: the truth (left) and the estimate (right) as two matrices.
            ``truth=`` is a :class:`Connectivity` on the same nodes and axes, for example
            ``SimulatedData.truth_at(freqs=est.freqs, times=est.times)``; ``freq``, ``band`` and
            ``time`` apply to both panels; ``ax`` is a pair of axes.

            ``"time_frequency"``: one pair as a panel with frequency on the vertical and time on
            the horizontal axis (``source=``, ``target=``); needs at least two frequencies and two
            times.

            ``"network_graph"``: the nodes on a circle (the first at the top, then clockwise) and
            an arrow from source to target for every chosen edge. The edges are always chosen
            explicitly with ``threshold`` (cells of at least that value) or a boolean ``(n, n)``
            ``mask``; the arrow width grows with the value. Same ``freq``/``band``/``time`` choice
            as the matrix.

        Returns
        -------
        matplotlib.axes.Axes or tuple of two Axes
            The axes drawn on (the pair ``(truth, estimate)`` for ``"truth_comparison"``).
        """
        from dirconn.plotting import plot

        return plot(self, kind, **kwargs)

    def to_xarray(self) -> xr.DataArray:
        """Return a copy as an ``xarray.DataArray`` with named dimensions and coordinates."""
        return xr.DataArray(
            self.values.copy(),
            dims=DIMS,
            coords={
                "source": list(self.source),
                "target": list(self.target),
                "freq": self.freqs.copy(),
                "time": self.times.copy(),
            },
            name="connectivity",
            attrs={"provenance": self.provenance.to_json()},
        )

    def reduce_freq(self, band: str | Sequence[float], how: str = "mean") -> Connectivity:
        """Reduce the frequency axis over a band.

        The band is inclusive at both ends. The result has a frequency axis of length 1 whose
        coordinate is the centre of the band. Reductions ignore NaN entries; an entry that is NaN
        at every frequency stays NaN. Complex values are reduced in the complex plane.

        Parameters
        ----------
        band : str or (float, float)
            Lower and upper frequency in Hz, or the name of a band from :data:`dirconn.BANDS`
            (case-insensitive), whose limits are recorded in the provenance with the name.
        how : {"mean", "median", "max", "peak_freq"}
            ``"mean"``: complex mean. ``"median"``: median of real and imaginary parts
            separately. ``"max"``: the entry with the largest value (largest magnitude for
            complex data), kept as is. ``"peak_freq"``: the frequency in Hz of that entry
            (first one on ties); the values of the result are then frequencies.

        Returns
        -------
        Connectivity
            A new result whose provenance records the step.
        """
        if how not in REDUCTIONS:
            raise ValueError(f"how must be one of {REDUCTIONS}, got {how!r}.")
        if not self.is_frequency_resolved:
            raise ValueError("This result is not frequency-resolved; there is nothing to reduce.")
        band_name = None
        if isinstance(band, str):
            band_name = band.strip().lower()
            if band_name not in BANDS:
                raise ValueError(
                    f"Unknown band name {band!r}; the known bands are {sorted(BANDS)} "
                    "(see dirconn.BANDS), or pass (fmin, fmax) in Hz."
                )
            band = BANDS[band_name]
        try:
            fmin, fmax = (float(b) for b in band)
        except (TypeError, ValueError):
            raise ValueError(f"band must be (fmin, fmax) in Hz, got {band!r}.") from None
        if not (math.isfinite(fmin) and math.isfinite(fmax)):
            raise ValueError(f"band must be finite, got {band!r}.")
        if fmin > fmax:
            raise ValueError(f"band must satisfy fmin <= fmax, got {band!r}.")
        mask = (self.freqs >= fmin) & (self.freqs <= fmax)
        if not mask.any():
            raise ValueError(f"No frequencies of the result fall inside the band {(fmin, fmax)}.")

        sub = self.values[:, :, mask, :]
        fsub = self.freqs[mask]
        out = _reduce(sub, fsub, how, axis=2)

        step: dict[str, Any] = {
            "op": "reduce_freq",
            "band": [fmin, fmax],
            "how": how,
            "n_freqs": int(mask.sum()),
        }
        if band_name is not None:
            step["band_name"] = band_name
        if how == "peak_freq":
            step["values_unit"] = "Hz"
        return Connectivity(
            values=out,
            source=self.source,
            target=self.target,
            freqs=np.array([(fmin + fmax) / 2.0]),
            times=self.times,
            provenance=self.provenance.with_step(step),
        )

    def reduce_time(self, segments: Any, how: str = "mean") -> Connectivity:
        """Reduce the time axis over segments, symmetric to :meth:`reduce_freq`.

        Every segment ``(start, stop)`` in seconds, both ends included, becomes one entry of the
        new time axis whose coordinate is the centre of the segment. Reductions ignore NaN
        entries; an entry that is NaN at every time point stays NaN. Complex values are reduced in
        the complex plane.

        Parameters
        ----------
        segments : (float, float) or sequence of (float, float)
            One segment or several, ascending and not overlapping. Each must lie inside the time
            range of the result and contain at least one time point; nothing is clipped silently.
        how : {"mean", "median", "max", "peak_time"}
            ``"mean"``: complex mean. ``"median"``: median of real and imaginary parts separately.
            ``"max"``: the entry with the largest value (largest magnitude for complex data),
            kept as is. ``"peak_time"``: the time in seconds of that entry (first one on ties);
            the values of the result are then times.

        Returns
        -------
        Connectivity
            A new result whose provenance records the step.

        Notes
        -----
        For a time-resolved estimator such as :class:`~dirconn.ADTF` this averages a trajectory
        that has been estimated continuously, so each segment still carries memory of the
        preceding samples. Fitting every segment on its own is a different procedure; see
        :class:`~dirconn.WDTF` with ``segments``.
        """
        if how not in TIME_REDUCTIONS:
            raise ValueError(f"how must be one of {TIME_REDUCTIONS}, got {how!r}.")
        if not self.is_time_resolved:
            raise ValueError("This result is not time-resolved; there is nothing to reduce.")
        pairs = check_segments(segments, allow_single=True)
        times = self.times
        low, high = float(times[0]), float(times[-1])
        tolerance = _EDGE_TOLERANCE * max(1.0, abs(low), abs(high))
        parts = []
        counts = []
        for start, stop in pairs:
            if start < low - tolerance or stop > high + tolerance:
                raise ValueError(
                    f"The segment {(start, stop)} lies outside the time range of the result "
                    f"({low:g} to {high:g} s)."
                )
            mask = (times >= start - tolerance) & (times <= stop + tolerance)
            if not mask.any():
                raise ValueError(
                    f"The segment {(start, stop)} contains no time point of the result."
                )
            parts.append(_reduce(self.values[:, :, :, mask], times[mask], how, axis=3))
            counts.append(int(mask.sum()))
        step: dict[str, Any] = {
            "op": "reduce_time",
            "segments": [[start, stop] for start, stop in pairs],
            "how": how,
            "n_times": counts,
        }
        if how == "peak_time":
            step["values_unit"] = "s"
        return Connectivity(
            values=np.concatenate(parts, axis=3),
            source=self.source,
            target=self.target,
            freqs=self.freqs,
            times=np.array([(start + stop) / 2.0 for start, stop in pairs]),
            provenance=self.provenance.with_step(step),
        )

    def to_netcdf(self, path: str | os.PathLike[str]) -> None:
        """Write the result to a NetCDF file, provenance included.

        Complex values are stored as ``values_real`` and ``values_imag``.
        """
        coords = {
            "source": list(self.source),
            "target": list(self.target),
            "freq": self.freqs,
            "time": self.times,
        }
        if self.is_complex:
            data_vars = {
                "values_real": (DIMS, np.ascontiguousarray(self.values.real)),
                "values_imag": (DIMS, np.ascontiguousarray(self.values.imag)),
            }
        else:
            data_vars = {"values": (DIMS, np.ascontiguousarray(self.values))}
        dataset = xr.Dataset(
            data_vars,
            coords=coords,
            attrs={
                "dirconn_format_version": FORMAT_VERSION,
                "provenance": self.provenance.to_json(),
            },
        )
        dataset.to_netcdf(os.fspath(path), engine="netcdf4")

    @classmethod
    def from_netcdf(cls, path: str | os.PathLike[str]) -> Connectivity:
        """Read a result written by :meth:`to_netcdf`."""
        file = os.fspath(path)
        if not os.path.exists(file):
            raise FileNotFoundError(file)
        with xr.open_dataset(file, engine="netcdf4") as ds:
            version = ds.attrs.get("dirconn_format_version")
            if version is None or str(version) != FORMAT_VERSION:
                raise ValueError(
                    f"Unsupported dirconn format version {version!r} (expected {FORMAT_VERSION!r})."
                )
            if "provenance" not in ds.attrs:
                raise ValueError("The file has no provenance attribute.")
            provenance = Provenance.from_json(str(ds.attrs["provenance"]))
            if "values_real" in ds:
                values = ds["values_real"].values + 1j * ds["values_imag"].values
            else:
                values = ds["values"].values
            return cls(
                values=np.array(values),
                source=tuple(str(s) for s in ds["source"].values),
                target=tuple(str(t) for t in ds["target"].values),
                freqs=np.array(ds["freq"].values, dtype=float),
                times=np.array(ds["time"].values, dtype=float),
                provenance=provenance,
            )


def _reduce(sub: np.ndarray, coord: np.ndarray, how: str, axis: int = 2) -> np.ndarray:
    """Reduce ``axis`` of ``sub`` (source, target, freq, time) with NaN-ignoring semantics.

    ``coord`` holds the coordinate (frequency or time) of every entry along ``axis``; the
    ``peak_*`` reductions return it.
    """
    if how in ("max", "peak_freq", "peak_time"):
        key = np.abs(sub) if np.iscomplexobj(sub) else sub
        idx = np.where(np.isnan(key), -np.inf, key).argmax(axis=axis)
        if how == "max":
            return np.take_along_axis(sub, np.expand_dims(idx, axis), axis=axis)
        peak = coord[idx]
        peak = np.where(np.isnan(sub).all(axis=axis), np.nan, peak)
        return np.expand_dims(peak, axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if how == "mean":
            return np.nanmean(sub, axis=axis, keepdims=True)
        if np.iscomplexobj(sub):
            real = np.nanmedian(sub.real, axis=axis, keepdims=True)
            imag = np.nanmedian(sub.imag, axis=axis, keepdims=True)
            return real + 1j * imag
        return np.nanmedian(sub, axis=axis, keepdims=True)


def check_segments(segments: Any, *, allow_single: bool = False) -> tuple[tuple[float, float], ...]:
    """Normalise time segments to a tuple of ``(start, stop)`` pairs in seconds.

    The pairs must be finite with ``start <= stop`` and ascending without overlap (the next start
    is after the previous stop), so that the centres form a valid, strictly increasing time axis.
    With ``allow_single`` a flat ``(start, stop)`` is one segment.

    Raises
    ------
    ValueError
        For anything else.
    """
    expected = "segments must be a non-empty sequence of (start, stop) pairs in seconds"
    if isinstance(segments, (str, bytes)):
        raise ValueError(f"{expected}, got {segments!r}.")
    try:
        items = list(segments)
    except TypeError:
        raise ValueError(f"{expected}, got {segments!r}.") from None
    if not items:
        raise ValueError(f"{expected}, got {segments!r}.")
    scalar = all(isinstance(v, (int, float, np.integer, np.floating)) for v in items)
    if allow_single and scalar and len(items) == 2:
        items = [items]
    pairs: list[tuple[float, float]] = []
    for item in items:
        try:
            start, stop = (float(v) for v in item)
        except (TypeError, ValueError):
            raise ValueError(f"{expected}, got the entry {item!r}.") from None
        if not (math.isfinite(start) and math.isfinite(stop)):
            raise ValueError(f"segment limits must be finite, got {(start, stop)!r}.")
        if start > stop:
            raise ValueError(
                f"The start must not exceed the stop of a segment, got {(start, stop)}."
            )
        pairs.append((start, stop))
    for (_, previous_stop), (next_start, _) in zip(pairs, pairs[1:], strict=False):
        if next_start <= previous_stop:
            raise ValueError(
                f"segments must be ascending and must not overlap, got {tuple(pairs)!r}."
            )
    return tuple(pairs)
