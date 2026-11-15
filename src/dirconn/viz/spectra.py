"""Spectrum figures: both directions of one pair over frequency."""

from __future__ import annotations

from typing import Any

from dirconn.data.result import Connectivity
from dirconn.viz._backend import _pyplot
from dirconn.viz.matrices import _pair, _time_index


def directed_spectrum(
    conn: Connectivity, *, source: str, target: str, time: float | None = None, ax: Any = None
) -> Any:
    """Draw both directions of one pair over frequency: ``source -> target`` and the reverse.

    For an undirected method (``MethodInfo.directed`` false, such as imaginary coherence) the two
    curves are the two orders of one pair, not two directions: the legend says ``(A, B)`` and
    ``(B, A)`` instead of arrows, and the title says that the measure is undirected.
    """
    from dirconn.methods.base import method_info

    if conn.is_complex:
        raise ValueError("Complex values cannot be drawn as a curve; take abs() or real first.")
    if not conn.is_frequency_resolved:
        raise ValueError("The result has no frequency axis, so there is no spectrum to draw.")
    i, j, k, m = _pair(conn, source, target)
    index, notes = _time_index(conn, time)
    forward = conn.values[i, j, :, index]
    backward = conn.values[k, m, :, index]
    info = method_info(conn.provenance.method)
    undirected = info is not None and not info.directed
    if undirected:
        labels = (f"({source}, {target})", f"({target}, {source})")
        notes = ["undirected", *notes]
    else:
        labels = (f"{source} → {target}", f"{target} → {source}")

    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots()
    # two line styles as well as two colours, so the directions stay apart without colour
    ax.plot(conn.freqs, forward, color="C0", linestyle="-", label=labels[0])
    ax.plot(conn.freqs, backward, color="C1", linestyle="--", label=labels[1])
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(conn.provenance.method)
    ax.set_title(", ".join([conn.provenance.method, *notes]))
    ax.legend()
    return ax
