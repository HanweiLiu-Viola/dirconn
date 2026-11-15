"""Graph figures: nodes on a circle with arrows for the chosen edges."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from dirconn.data.result import Connectivity
from dirconn.viz._backend import _pyplot
from dirconn.viz.matrices import _select


def network_graph(
    conn: Connectivity,
    *,
    threshold: float | None = None,
    mask: Any = None,
    freq: float | None = None,
    band: str | Sequence[float] | None = None,
    time: float | None = None,
    ax: Any = None,
) -> Any:
    """Draw the nodes on a circle with an arrow from source to target for every chosen edge.

    The edges are chosen explicitly, never by a default: ``threshold`` keeps the cells with a value
    of at least that, ``mask`` is a boolean ``(n, n)`` array. The width of an arrow grows with the
    value. The first node is at the top and the others follow clockwise.
    """
    if threshold is not None and mask is not None:
        raise ValueError("Choose the edges to draw with threshold or mask, not both.")
    if threshold is None and mask is None:
        raise ValueError("Choose the edges to draw with threshold or mask.")
    if conn.is_complex:
        raise ValueError("Complex values cannot be drawn as a graph; take abs() or real first.")
    if conn.source != conn.target:
        raise ValueError(
            "A network graph needs the same nodes as sources and targets, in the same order."
        )
    matrix, notes = _select(conn, freq, band, time)
    n = len(conn.source)
    matrix = matrix.copy()
    np.fill_diagonal(matrix, np.nan)
    if mask is not None:
        chosen = np.asarray(mask)
        if chosen.dtype != bool:
            raise ValueError("mask must be a boolean array.")
        if chosen.shape != (n, n):
            raise ValueError(f"mask has shape {chosen.shape}; expected {(n, n)}.")
        notes.append("edges: mask")
    else:
        level = float(threshold)  # type: ignore[arg-type]
        with np.errstate(invalid="ignore"):
            chosen = matrix >= level
        notes.append(f"edges: value >= {level:g}")
    edges = chosen & np.isfinite(matrix)

    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    angles = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    ax.scatter(points[:, 0], points[:, 1], s=500, facecolor="white", edgecolor="0.2", zorder=3)
    for name, (x, y) in zip(conn.source, points, strict=True):
        ax.text(1.28 * x, 1.28 * y, name, ha="center", va="center")
    largest = float(matrix[edges].max()) if edges.any() else 0.0
    for i, j in zip(*np.nonzero(edges), strict=True):
        width = 0.5 + 4.0 * float(matrix[i, j]) / largest if largest > 0 else 1.0
        ax.annotate(
            "",
            xy=points[j],
            xytext=points[i],
            arrowprops={
                "arrowstyle": "-|>",
                "lw": width,
                "color": "0.2",
                "shrinkA": 14,
                "shrinkB": 14,
                "connectionstyle": "arc3,rad=0.15",
            },
        )
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(", ".join([conn.provenance.method, *notes]))
    return ax
