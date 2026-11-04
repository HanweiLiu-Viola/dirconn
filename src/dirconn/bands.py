"""Named frequency bands.

Band definitions differ between laboratories and between papers, so these are neither hidden
defaults nor a standard: they are a plain, inspectable table. Nothing in dirconn picks a band for
you; ``Connectivity.reduce_freq`` uses one only when you ask for it by name. To use other limits,
pass ``(fmin, fmax)`` in Hz instead.

Both limits of a band are inclusive when a band is applied, so the frequency shared by two
neighbouring bands (for example 4 Hz) belongs to both.
"""

from __future__ import annotations

from types import MappingProxyType

BANDS = MappingProxyType(
    {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 80.0),
    }
)
"""Read-only mapping from band name to ``(fmin, fmax)`` in Hz."""
