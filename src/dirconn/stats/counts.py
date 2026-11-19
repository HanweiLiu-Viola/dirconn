"""Inference on counts: a binomial proportion and two paired proportions.

:func:`wilson_interval` is the Wilson score interval of a rate ``k / n``; it stays inside
``[0, 1]`` and keeps its coverage for small ``n`` and for rates near 0 or 1, where the normal
(Wald) interval fails. :func:`mcnemar_exact` compares two rates measured on the same units (for
example two estimators on the same simulations): only the discordant units, where exactly one of
the two fires, carry information, and under the null either one is equally likely to be the one.
"""

from __future__ import annotations

import math

from scipy.stats import binom, norm


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return the Wilson score interval of the proportion ``k / n``.

    Parameters
    ----------
    k : int
        Number of events, ``0 <= k <= n``.
    n : int
        Number of trials.
    confidence : float
        Coverage of the interval, strictly between 0 and 1.

    Returns
    -------
    tuple of float
        ``(low, high)``; ``(nan, nan)`` when ``n == 0``.

    Raises
    ------
    ValueError
        If the counts or the confidence are out of range.
    """
    k, n = int(k), int(n)
    if n < 0 or not 0 <= k <= max(n, 0):
        raise ValueError(f"Need 0 <= k <= n, got k={k}, n={n}.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly between 0 and 1, got {confidence}.")
    if n == 0:
        return math.nan, math.nan
    z = float(norm.ppf(0.5 + confidence / 2.0))
    z2 = z * z
    centre = (k + z2 / 2.0) / (n + z2)
    half = z / (n + z2) * math.sqrt(k * (n - k) / n + z2 / 4.0)
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Return the exact two-sided McNemar p-value from the two discordant counts.

    Under the null of equal paired rates, ``b`` is binomial with ``n = b + c`` trials and
    probability 1/2; the two-sided p-value is ``min(1, 2 * P(X <= min(b, c)))``.

    Parameters
    ----------
    b, c : int
        Units where only the first, and only the second, of the two classifiers fires.

    Returns
    -------
    float
        The p-value; 1 when there is no discordant unit.

    Raises
    ------
    ValueError
        If a count is negative.
    """
    b, c = int(b), int(c)
    if b < 0 or c < 0:
        raise ValueError(f"Discordant counts must be non-negative, got b={b}, c={c}.")
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2.0 * binom.cdf(min(b, c), n, 0.5)))
