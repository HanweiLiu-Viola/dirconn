"""Figures of connectivity results.

matplotlib is an optional dependency (extra ``plot``). It is imported when a figure is drawn, never
when :mod:`dirconn` is imported. The plot code only reads the result: choosing a frequency, a band
or a time point is the only selection it does, and it says which one it took in the title.

:func:`plot` draws a :class:`~dirconn.Connectivity` as one of :data:`PLOT_KINDS`; the figure
functions are in :mod:`dirconn.viz.matrices`, :mod:`dirconn.viz.spectra` and
:mod:`dirconn.viz.graphs`. The figures that need more than a result (a grouping of the nodes, the
cortical surface) are in :mod:`dirconn.viz.brain`. All of them are re-exported here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dirconn.data.result import Connectivity
from dirconn.viz.brain import (
    GUARD_CAPTION,
    MIN_DOMINANT_SHARE,
    YEO_NETWORKS,
    chord_diagram,
    degree_imbalance_map,
    dominant_networks,
    group_matrix,
    region_values,
    yeo_groups,
    yeo_networks,
)
from dirconn.viz.graphs import network_graph
from dirconn.viz.matrices import connectivity_grid, time_frequency, truth_comparison
from dirconn.viz.spectra import directed_spectrum

__all__ = [
    "GUARD_CAPTION",
    "MIN_DOMINANT_SHARE",
    "PLOT_KINDS",
    "YEO_NETWORKS",
    "chord_diagram",
    "connectivity_grid",
    "degree_imbalance_map",
    "directed_spectrum",
    "dominant_networks",
    "group_matrix",
    "network_graph",
    "plot",
    "region_values",
    "time_frequency",
    "truth_comparison",
    "yeo_groups",
    "yeo_networks",
]


def plot(conn: Connectivity, kind: str = "connectivity_grid", **kwargs: Any) -> Any:
    """Draw ``conn`` as ``kind`` and return the matplotlib axes (see :meth:`Connectivity.plot`)."""
    if kind not in PLOT_KINDS:
        raise ValueError(f"Unknown plot kind {kind!r}; known kinds: {', '.join(PLOT_KINDS)}.")
    return _KINDS[kind](conn, **kwargs)


_KINDS: dict[str, Callable[..., Any]] = {
    "connectivity_grid": connectivity_grid,
    "directed_spectrum": directed_spectrum,
    "truth_comparison": truth_comparison,
    "time_frequency": time_frequency,
    "network_graph": network_graph,
}
PLOT_KINDS: tuple[str, ...] = tuple(_KINDS)
