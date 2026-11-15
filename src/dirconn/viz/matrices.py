"""Matrix figures: the connectivity grid, truth against estimate, and time-frequency panels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from dirconn.data.result import Connectivity
from dirconn.viz._backend import _pyplot


def _nearest(axis: np.ndarray, value: float, name: str, unit: str) -> int:
    """Return the index of the ``axis`` point nearest to ``value``; refuse values off the axis."""
    if not np.isfinite(value) or value < axis[0] or value > axis[-1]:
        raise ValueError(
            f"{name}={value:g} {unit} is outside the {name} axis of the result "
            f"({axis[0]:g} to {axis[-1]:g} {unit})."
        )
    return int(np.argmin(np.abs(axis - value)))


def _time_index(conn: Connectivity, time: float | None) -> tuple[int, list[str]]:
    """Return the time index to draw (0 without a time axis) and how it was chosen."""
    if time is not None and not conn.is_time_resolved:
        raise ValueError("The result has no time axis, so time does not apply.")
    if not conn.is_time_resolved:
        return 0, []
    if time is None:
        raise ValueError("The result is time-resolved: choose one time point with time=.")
    index = _nearest(conn.times, float(time), "time", "s")
    return index, [f"{conn.times[index]:g} s"]


def _select(
    conn: Connectivity,
    freq: float | None,
    band: str | Sequence[float] | None,
    time: float | None,
) -> tuple[np.ndarray, list[str]]:
    """Return the ``(source, target)`` matrix to draw and a description of how it was chosen."""
    if freq is not None and band is not None:
        raise ValueError("Give freq or band, not both.")
    if (freq is not None or band is not None) and not conn.is_frequency_resolved:
        raise ValueError("The result has no frequency axis, so freq and band do not apply.")
    notes: list[str] = []
    values = conn.values
    if conn.is_frequency_resolved:
        if band is not None:
            values = conn.reduce_freq(band).values
            notes.append(f"{band} band" if isinstance(band, str) else f"{band[0]:g}-{band[1]:g} Hz")
        elif freq is not None:
            index = _nearest(conn.freqs, float(freq), "freq", "Hz")
            values = values[:, :, index : index + 1, :]
            notes.append(f"{conn.freqs[index]:g} Hz")
        else:
            raise ValueError(
                "The result is frequency-resolved: choose one frequency with freq= "
                "or a band with band=."
            )
    index, time_note = _time_index(conn, time)
    return values[:, :, 0, index], notes + time_note


def _colour_scale(
    defined: np.ndarray, vmin: float | None, vmax: float | None
) -> tuple[float, float, str]:
    """Return the colour limits and the default colour map for the defined values.

    All values non-negative: a sequential scale from 0 (``Blues``). Any negative value: a diverging
    scale centred on 0 (``RdBu_r``), where ``vmax`` is the half range.
    """
    if defined.min() < 0:
        half = float(np.abs(defined).max()) if vmax is None else float(vmax)
        if vmin is not None and float(vmin) != -half:
            raise ValueError(
                "A diverging scale is centred on zero: give vmax alone (the half range), "
                "or vmin equal to -vmax."
            )
        low, high, default_cmap = -half, half, "RdBu_r"
    else:
        low = 0.0 if vmin is None else float(vmin)
        high = float(defined.max()) if vmax is None else float(vmax)
        default_cmap = "Blues"
    if high <= low:
        high = low + 1.0
    return low, high, default_cmap


def _empty_cells_transparent(plt: Any, cmap: Any, default: str) -> Any:
    """Return the colour map with a transparent "bad" colour, so undefined cells stay empty."""
    return plt.get_cmap(default if cmap is None else cmap).with_extremes(bad=(0.0, 0.0, 0.0, 0.0))


def _pair(conn: Connectivity, source: str, target: str) -> tuple[int, int, int, int]:
    """Return the indices of ``source -> target`` and of the reverse direction in the values."""
    if source == target:
        raise ValueError(f"A node cannot be paired with itself ({source!r}).")
    for name in (source, target):
        if name not in conn.source or name not in conn.target:
            raise ValueError(
                f"{name!r} is not a node of the result (sources: {', '.join(conn.source)}; "
                f"targets: {', '.join(conn.target)})."
            )
    return (
        conn.source.index(source),
        conn.target.index(target),
        conn.source.index(target),
        conn.target.index(source),
    )


def connectivity_grid(
    conn: Connectivity,
    *,
    freq: float | None = None,
    band: str | Sequence[float] | None = None,
    time: float | None = None,
    ax: Any = None,
    cmap: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
) -> Any:
    """Draw one ``(source, target)`` matrix: row ``i`` is source ``i``, column ``j`` target ``j``.

    Undefined cells are left empty, and so is the diagonal of a result whose sources and targets
    are the same nodes, even where the estimator defines it. Values that are all non-negative use a
    sequential scale from 0; as soon as one value is negative the scale is diverging and centred
    on 0.
    """
    if conn.is_complex:
        raise ValueError("Complex values cannot be drawn as one matrix; take abs() or real first.")
    matrix, notes = _select(conn, freq, band, time)
    if conn.source == conn.target:
        # a node's relation to itself is not a connection (the spectral measures define it as a
        # share of the node's own activity); leave it empty and keep it out of the colour scale
        matrix = matrix.copy()
        np.fill_diagonal(matrix, np.nan)
    defined = matrix[np.isfinite(matrix)]
    if defined.size == 0:
        raise ValueError("The selected slice has no defined value, so there is nothing to draw.")

    low, high, default_cmap = _colour_scale(defined, vmin, vmax)

    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    # undefined cells stay empty: they show the axes background, which is grey so that they are not
    # mistaken for a cell whose value is 0
    colours = _empty_cells_transparent(plt, cmap, default_cmap)
    image = ax.imshow(
        np.ma.masked_invalid(matrix), cmap=colours, vmin=low, vmax=high, interpolation="nearest"
    )

    rotation = 45 if max(len(name) for name in conn.target) > 4 else 0
    ax.set_facecolor("0.85")
    ax.set_xticks(range(len(conn.target)))
    ax.set_xticklabels(conn.target, rotation=rotation, ha="right" if rotation else "center")
    ax.set_yticks(range(len(conn.source)))
    ax.set_yticklabels(conn.source)
    ax.set_xlabel("target")
    ax.set_ylabel("source")
    ax.set_title(", ".join([conn.provenance.method, *notes]))
    if colorbar:
        ax.figure.colorbar(image, ax=ax)
    return ax


def truth_comparison(
    conn: Connectivity,
    *,
    truth: Connectivity,
    freq: float | None = None,
    band: str | Sequence[float] | None = None,
    time: float | None = None,
    ax: Any = None,
) -> tuple[Any, Any]:
    """Draw the truth (left) and the estimate (right) as two matrices of the same nodes.

    ``truth`` must be on the axes of ``conn``: take it from ``SimulatedData.truth_at(freqs=...,
    times=...)``. The slice options apply to both panels. The truth has its own scale (0 to 1).
    """
    if truth.source != conn.source or truth.target != conn.target:
        raise ValueError(
            "The truth and the estimate must have the same nodes in the same order "
            f"(truth: {', '.join(truth.source)}; estimate: {', '.join(conn.source)})."
        )
    same_freqs = truth.is_frequency_resolved == conn.is_frequency_resolved and (
        not conn.is_frequency_resolved or np.array_equal(truth.freqs, conn.freqs)
    )
    same_times = truth.is_time_resolved == conn.is_time_resolved and (
        not conn.is_time_resolved or np.array_equal(truth.times, conn.times)
    )
    if not (same_freqs and same_times):
        raise ValueError(
            "The truth is not on the frequency and time axes of the estimate; "
            "use SimulatedData.truth_at(freqs=estimate.freqs, times=estimate.times)."
        )
    if ax is None:
        _, axes = _pyplot().subplots(1, 2, figsize=(10.0, 4.5), layout="constrained")
    else:
        try:
            axes = list(ax)
        except TypeError:
            axes = []
        if len(axes) != 2:
            raise ValueError("Give two axes (truth, estimate) or none.")
    left = connectivity_grid(
        truth, freq=freq, band=band, time=time, ax=axes[0], vmin=0.0, vmax=1.0, colorbar=False
    )
    left.set_title("truth")
    right = connectivity_grid(conn, freq=freq, band=band, time=time, ax=axes[1])
    return left, right


def time_frequency(
    conn: Connectivity,
    *,
    source: str,
    target: str,
    ax: Any = None,
    cmap: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
) -> Any:
    """Draw one pair as a panel: frequency on the vertical axis, time on the horizontal axis."""
    if conn.is_complex:
        raise ValueError("Complex values cannot be drawn as a panel; take abs() or real first.")
    if not conn.is_frequency_resolved:
        raise ValueError("The result has no frequency axis, so there is no time-frequency panel.")
    if not conn.is_time_resolved:
        raise ValueError("The result has no time axis, so there is no time-frequency panel.")
    if conn.freqs.shape[0] < 2 or conn.times.shape[0] < 2:
        raise ValueError("A time-frequency panel needs at least two frequencies and two times.")
    i, j, _, _ = _pair(conn, source, target)
    panel = conn.values[i, j]  # (frequency, time)
    defined = panel[np.isfinite(panel)]
    if defined.size == 0:
        raise ValueError("The pair has no defined value, so there is nothing to draw.")
    low, high, default_cmap = _colour_scale(defined, vmin, vmax)

    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    colours = _empty_cells_transparent(plt, cmap, default_cmap)
    mesh = ax.pcolormesh(
        conn.times,
        conn.freqs,
        np.ma.masked_invalid(panel),
        cmap=colours,
        vmin=low,
        vmax=high,
        shading="nearest",
    )
    ax.set_facecolor("0.85")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(f"{conn.provenance.method}, {source} → {target}")
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax)
    return ax
