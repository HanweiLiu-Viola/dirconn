"""Lazy import of matplotlib, which is an optional dependency."""

from __future__ import annotations

from typing import Any


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting needs matplotlib. Install it with: pip install 'dirconn[plot]'."
        ) from exc
    return plt
