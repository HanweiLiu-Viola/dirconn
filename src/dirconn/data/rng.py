"""Reproducible random generators for resampling statistics."""

from __future__ import annotations

import numbers

import numpy as np


def _as_count(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
    count = int(value)
    if count < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {count}.")
    return count


class RNGController:
    """Derive one independent generator per task from a single seed.

    Child ``i`` is ``SeedSequence(seed, spawn_key=(i,))``. The generators therefore depend only
    on the seed and the index, never on how many workers share the work or in which order they
    run, and asking for more children keeps the earlier ones unchanged.

    Parameters
    ----------
    seed : int
        Non-negative integer. There is no default: every random procedure in dirconn needs an
        explicit seed so that a reported number can be reproduced.
    """

    def __init__(self, seed: int) -> None:
        self.seed = _as_count(seed, "seed", minimum=0)

    def child(self, index: int) -> np.random.Generator:
        """Return generator number ``index`` (zero-based)."""
        index = _as_count(index, "index", minimum=0)
        return np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(index,)))

    def spawn(self, n: int) -> list[np.random.Generator]:
        """Return generators ``0`` to ``n - 1``.

        Calling it again returns generators that produce the same streams; it does not advance.
        """
        n = _as_count(n, "n", minimum=1)
        return [self.child(i) for i in range(n)]

    def __repr__(self) -> str:
        """Return a short summary."""
        return f"RNGController(seed={self.seed})"
