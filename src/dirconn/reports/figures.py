"""Deterministic inline SVG figures for the participant report.

Every function returns SVG text that starts with ``<svg`` (no XML declaration, DOCTYPE or leading
comment) and can be embedded in HTML. Figures are drawn on a ``matplotlib.figure.Figure`` created
here and rendered with the SVG canvas; ``pyplot`` global state and the user's backend are never
touched. Rendering runs inside an ``rc_context`` with a fixed hash salt, no date metadata and text
kept as text (``svg.fonttype = "none"``), so the same input gives byte-identical output.

Functions return the empty string when there is nothing meaningful to draw (see each docstring).
"""

from __future__ import annotations

import dataclasses
import io
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, NullFormatter

from dirconn.data.result import Connectivity
from dirconn.reports.collect import selected_order

__all__ = [
    "band_matrix_svg",
    "directed_spectra_svg",
    "net_direction_svg",
    "order_curve_svg",
    "psd_svg",
    "pvalue_hist_svg",
    "significant_over_time_svg",
    "svg_of",
    "time_frequency_svg",
]

_RC: dict[Any, Any] = {"svg.hashsalt": "dirconn", "svg.fonttype": "none"}
_MAX_COLUMNS = 3
_SVG_START = re.compile(r"<svg\b")


@contextmanager
def _figure(**kwargs: Any) -> Iterator[Figure]:
    """Yield a new ``Figure`` created inside the fixed ``rc_context`` (no global state changes)."""
    with matplotlib.rc_context(_RC):
        kwargs.setdefault("layout", "constrained")
        yield Figure(**kwargs)


def svg_of(fig: Figure) -> str:
    """Render ``fig`` as SVG text starting at the ``<svg`` tag.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        A figure that is not managed by ``pyplot``.

    Returns
    -------
    str
        The SVG document without XML declaration, DOCTYPE and leading comments; deterministic for
        equal figures (fixed hash salt, no date).
    """
    buffer = io.BytesIO()
    with matplotlib.rc_context(_RC):
        FigureCanvasSVG(fig)
        fig.savefig(buffer, format="svg", metadata={"Date": None})
    text = buffer.getvalue().decode("utf-8")
    match = _SVG_START.search(text)
    if match is None:  # pragma: no cover - matplotlib always writes an svg element
        raise RuntimeError("matplotlib produced no <svg> element")
    return text[match.start() :]


def _is_true(value: Any) -> bool:
    """Interpret a ``significant`` cell: bool, NaN or the strings "True", "False", "n/a"."""
    if isinstance(value, str):
        return value.strip().lower() == "true"
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
        return bool(value)
    except (TypeError, ValueError):
        return False


def _refuse_time(edges: pd.DataFrame, what: str) -> None:
    if "time" in edges.columns:
        raise ValueError(
            f"{what} needs an edge table without a 'time' column; time-resolved results have one "
            "row per pair, band and time point."
        )


def _band_rows(edges: pd.DataFrame, band: str) -> pd.DataFrame:
    return edges.loc[edges["band"].astype(str) == band]


def band_matrix_svg(edges: pd.DataFrame, band: str, nodes: Sequence[str], *, title: str) -> str:
    """Draw the value of one band as a source (rows) by target (columns) heat map.

    Cells whose ``significant`` flag is true carry a ``*``; the diagonal is left empty.

    Parameters
    ----------
    edges : pandas.DataFrame
        Edge table (``source, target, band, value, ..., significant``) without a ``time`` column.
    band : str
        The band to draw.
    nodes : sequence of str
        Node names; the order of rows and columns.
    title : str
        Figure title.

    Returns
    -------
    str
        SVG text.

    Raises
    ------
    ValueError
        If ``edges`` has a ``time`` column.
    """
    _refuse_time(edges, "band_matrix_svg")
    names = [str(n) for n in nodes]
    index = {n: i for i, n in enumerate(names)}
    n = len(names)
    values = np.full((n, n), np.nan)
    stars: list[tuple[int, int]] = []
    sub = _band_rows(edges, band)
    flags = sub["significant"] if "significant" in sub.columns else [False] * len(sub)
    for src, tgt, value, sig in zip(
        sub["source"], sub["target"], sub["value"], flags, strict=False
    ):
        i, j = index.get(str(src)), index.get(str(tgt))
        if i is None or j is None or i == j:
            continue
        values[i, j] = float(value)
        if _is_true(sig):
            stars.append((i, j))
    side = max(3.2, 0.6 * n + 2.0)
    with _figure(figsize=(side, side * 0.85)) as fig:
        ax = fig.add_subplot()
        mesh = ax.imshow(np.ma.masked_invalid(values), cmap="viridis", aspect="equal")
        ax.set_xticks(range(n), labels=names, rotation=45, ha="right")
        ax.set_yticks(range(n), labels=names)
        ax.set_xlabel("target")
        ax.set_ylabel("source")
        ax.set_title(title)
        for i, j in stars:
            ax.text(j, i, "*", ha="center", va="center", color="white", fontsize=14)
        fig.colorbar(mesh, ax=ax, label="value")
        return svg_of(fig)


def net_direction_svg(edges: pd.DataFrame, kinds: Mapping[str, str], band: str) -> str:
    """Draw ``value(i -> j) - value(j -> i)`` for node pairs of different kinds.

    Parameters
    ----------
    edges : pandas.DataFrame
        Edge table without a ``time`` column.
    kinds : mapping of str to str
        Node kind by node name.
    band : str
        The band to draw.

    Returns
    -------
    str
        SVG text with one bar per unordered pair, or ``""`` when fewer than two kinds are present
        or no pair of different kinds has a value.

    Raises
    ------
    ValueError
        If ``edges`` has a ``time`` column.
    """
    _refuse_time(edges, "net_direction_svg")
    if len(set(kinds.values())) < 2:
        return ""
    sub = _band_rows(edges, band)
    table: dict[tuple[str, str], float] = {}
    for src, tgt, value in zip(sub["source"], sub["target"], sub["value"], strict=False):
        table[(str(src), str(tgt))] = float(value)
    pairs: list[tuple[str, str, float]] = []
    seen: set[frozenset[str]] = set()
    for (i, j), forward in table.items():
        key = frozenset((i, j))
        if i == j or key in seen or i not in kinds or j not in kinds or kinds[i] == kinds[j]:
            continue
        seen.add(key)
        diff = forward - table.get((j, i), float("nan"))
        if math.isfinite(diff):
            pairs.append((i, j, diff))
    if not pairs:
        return ""
    with _figure(figsize=(6.0, 0.4 * len(pairs) + 1.6)) as fig:
        ax = fig.add_subplot()
        positions = list(range(len(pairs)))
        diffs = [p[2] for p in pairs]
        ax.barh(positions, diffs, color=["C0" if d >= 0 else "C1" for d in diffs])
        ax.set_yticks(positions, labels=[f"{i} → {j}" for i, j, _ in pairs])
        ax.invert_yaxis()
        ax.axvline(0.0, color="0.3", linewidth=0.8)
        ax.set_xlabel("value(i → j) − value(j → i)")
        ax.set_title(f"Net direction, {band}")
        return svg_of(fig)


def order_curve_svg(selection: Mapping[str, Any]) -> str:
    """Draw the order-selection criteria against the model order.

    Each criterion (aic, bic, hqc and cv when present) is scaled to its own minimum and maximum so
    that curves on different scales share one axis; the y label says so. A vertical line marks the
    order picked by ``selection["criterion"]``.

    Parameters
    ----------
    selection : mapping
        ``OrderSelection.to_dict()`` as stored in the diagnostics JSON.

    Returns
    -------
    str
        SVG text.
    """
    orders = [int(o) for o in selection["orders"]]
    chosen = selected_order(selection)
    criterion = str(selection["criterion"])
    with _figure(figsize=(5.5, 3.4)) as fig:
        ax = fig.add_subplot()
        for name, style in (("aic", "-"), ("bic", "--"), ("hqc", ":"), ("cv", "-.")):
            curve = selection.get(name)
            if curve is None:
                continue
            y = np.asarray(curve, dtype=float)
            if not np.isfinite(y).any():
                continue
            low, high = np.nanmin(y), np.nanmax(y)
            scaled = (y - low) / (high - low) if high > low else np.zeros_like(y)
            ax.plot(orders, scaled, linestyle=style, marker="o", markersize=3, label=name)
        ax.axvline(
            chosen, color="0.2", linewidth=1.0, label=f"selected order {chosen} ({criterion})"
        )
        ax.set_xlabel("model order")
        ax.set_ylabel("criterion (each scaled to its own min–max)")
        ax.set_title(f"Order selection by {criterion}")
        ax.legend()
        return svg_of(fig)


def psd_svg(spectral: Mapping[str, Any], bands: Mapping[str, Sequence[float]]) -> str:
    """Draw the power spectral density of every channel on a log axis with the bands shaded.

    Parameters
    ----------
    spectral : mapping
        The ``spectral`` entry of the diagnostics JSON (``names``, ``freqs``, ``psd``).
    bands : mapping of str to sequence of float
        Band name to ``[low, high]`` in Hz.

    Returns
    -------
    str
        SVG text.
    """
    names = [str(n) for n in spectral["names"]]
    freqs = np.asarray(spectral["freqs"], dtype=float)
    psd = np.asarray(spectral["psd"], dtype=float)
    if psd.ndim == 2 and psd.shape[0] != len(names) and psd.shape[1] == len(names):
        psd = psd.T
    shades = matplotlib.colormaps["Pastel1"].colors  # type: ignore[attr-defined]
    with _figure(figsize=(6.0, 3.6)) as fig:
        ax = fig.add_subplot()
        for k, (band, limits) in enumerate(bands.items()):
            ax.axvspan(
                float(limits[0]),
                float(limits[1]),
                color=shades[k % len(shades)],
                alpha=0.6,
                label=f"{band} band",
                zorder=0,
            )
        for name, row in zip(names, psd, strict=False):
            ax.plot(freqs, row, linewidth=1.0, label=name)
        ax.set_yscale("log")
        # plain "1e-5" labels: no mathtext, so the SVG keeps ordinary text
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("power spectral density")
        ax.set_title("Spectral range of the prepared channels")
        ax.legend(fontsize="small", ncols=2)
        return svg_of(fig)


def _magnitude(conn: Connectivity) -> Connectivity:
    """Return ``conn`` with absolute values when it is complex, else ``conn``."""
    if not conn.is_complex:
        return conn
    return dataclasses.replace(conn, values=np.abs(conn.values))


def _grid(n: int, cell: tuple[float, float]) -> tuple[int, int, tuple[float, float]]:
    columns = min(_MAX_COLUMNS, n)
    rows = math.ceil(n / columns)
    return rows, columns, (cell[0] * columns, cell[1] * rows)


def directed_spectra_svg(conn: Connectivity, pairs: Sequence[tuple[str, str]]) -> str:
    """Draw both directions over frequency, one panel per pair (at most three per row).

    Parameters
    ----------
    conn : Connectivity
        A frequency-resolved result; a time axis of length one is accepted. Complex values are
        replaced by their magnitude.
    pairs : sequence of (str, str)
        ``(source, target)`` names.

    Returns
    -------
    str
        SVG text, or ``""`` when ``pairs`` is empty.
    """
    from dirconn.viz.spectra import directed_spectrum

    if not pairs:
        return ""
    conn = _magnitude(conn)
    rows, columns, size = _grid(len(pairs), (4.2, 3.2))
    with _figure(figsize=size) as fig:
        axes = fig.subplots(rows, columns, squeeze=False)
        for ax, (source, target) in zip(axes.flat, pairs, strict=False):
            directed_spectrum(conn, source=source, target=target, ax=ax)
        for ax in list(axes.flat)[len(pairs) :]:
            ax.set_visible(False)
        return svg_of(fig)


def time_frequency_svg(conn: Connectivity, pairs: Sequence[tuple[str, str]]) -> str:
    """Draw one time-frequency panel per pair (at most three per row).

    Parameters
    ----------
    conn : Connectivity
        A result resolved in frequency and time. Complex values are replaced by their magnitude.
    pairs : sequence of (str, str)
        ``(source, target)`` names.

    Returns
    -------
    str
        SVG text, or ``""`` when ``pairs`` is empty.
    """
    from dirconn.viz.matrices import time_frequency

    if not pairs:
        return ""
    conn = _magnitude(conn)
    rows, columns, size = _grid(len(pairs), (4.6, 3.4))
    with _figure(figsize=size) as fig:
        axes = fig.subplots(rows, columns, squeeze=False)
        for ax, (source, target) in zip(axes.flat, pairs, strict=False):
            time_frequency(conn, source=source, target=target, ax=ax)
        for ax in list(axes.flat)[len(pairs) :]:
            ax.set_visible(False)
        return svg_of(fig)


def _finite_p(edges: pd.DataFrame) -> pd.Series:
    if "p" not in edges.columns:
        return pd.Series([], dtype=float)
    p = pd.to_numeric(edges["p"], errors="coerce")
    return p[np.isfinite(p) & (p >= 0.0) & (p <= 1.0)]


def pvalue_hist_svg(edges: pd.DataFrame) -> str:
    """Draw a 20-bin histogram of the finite p values.

    Parameters
    ----------
    edges : pandas.DataFrame
        An edge table with a ``p`` column.

    Returns
    -------
    str
        SVG text, or ``""`` when the table is empty or all p values are NaN.
    """
    p = _finite_p(edges)
    if p.empty:
        return ""
    with _figure(figsize=(4.6, 3.2)) as fig:
        ax = fig.add_subplot()
        ax.hist(p.to_numpy(), bins=20, range=(0.0, 1.0), color="C0", edgecolor="white")
        ax.set_xlabel("p value")
        ax.set_ylabel("number of edges")
        ax.set_title("p-value distribution")
        return svg_of(fig)


def significant_over_time_svg(edges: pd.DataFrame) -> str:
    """Draw the number of significant pairs against time, one line per band.

    Parameters
    ----------
    edges : pandas.DataFrame
        A time-resolved edge table (``time`` column) with p values.

    Returns
    -------
    str
        SVG text, or ``""`` when there is no ``time`` column or no finite p value.
    """
    if "time" not in edges.columns or _finite_p(edges).empty:
        return ""
    flags = edges["significant"].map(_is_true)
    frame = pd.DataFrame(
        {"band": edges["band"].astype(str), "time": edges["time"], "n": flags.astype(int)}
    )
    counts = frame.groupby(["band", "time"])["n"].sum()
    with _figure(figsize=(6.0, 3.2)) as fig:
        ax = fig.add_subplot()
        for band in sorted(set(counts.index.get_level_values("band"))):
            series = counts.loc[band].sort_index()
            ax.plot(series.index.to_numpy(), series.to_numpy(), label=band)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("significant pairs")
        ax.set_title("Significant pairs over time")
        ax.legend()
        return svg_of(fig)
