"""Multiple-comparison correction on p-value arrays with undefined cells.

The family of tests is the set of **defined** (non-NaN) cells. Undefined cells, such as the
diagonal of a connectivity matrix, have no p-value, are returned as NaN and never enter the
family size.
"""

from __future__ import annotations

import math

import numpy as np

METHODS: tuple[str, ...] = ("fdr_bh", "bonferroni", "none")


def check_alpha(alpha: float) -> float:
    """Return ``alpha`` as a float, raising unless ``0 < alpha < 1``."""
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0 or math.isnan(alpha):
        raise ValueError(f"alpha must lie strictly between 0 and 1, got {alpha}.")
    return alpha


def adjust_pvalues(pvalues: np.ndarray, method: str = "fdr_bh") -> np.ndarray:
    """Adjust p-values so that ``adjusted <= alpha`` is the corrected decision.

    Parameters
    ----------
    pvalues : ndarray
        Any shape; NaN marks undefined cells and is kept.
    method : {"fdr_bh", "bonferroni", "none"}
        ``"fdr_bh"`` is the Benjamini-Hochberg step-up procedure (valid for independent or
        positively dependent tests), ``"bonferroni"`` multiplies by the family size, ``"none"``
        returns the raw values and must be chosen explicitly.

    Returns
    -------
    ndarray
        Same shape, values in ``[0, 1]``.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}.")
    p = np.asarray(pvalues, dtype=float)
    defined = ~np.isnan(p)
    values = p[defined]
    if values.size and (values.min() < 0.0 or values.max() > 1.0 or not np.isfinite(values).all()):
        raise ValueError("p-values must lie between 0 and 1.")
    adjusted = np.full(p.shape, np.nan)
    m = values.size
    if m == 0 or method == "none":
        adjusted[defined] = values
    elif method == "bonferroni":
        adjusted[defined] = np.minimum(1.0, values * m)
    else:
        order = np.argsort(values, kind="stable")
        scaled = values[order] * m / np.arange(1, m + 1)
        # step-up: the adjusted value of rank i is the smallest scaled value at rank >= i
        stepped = np.minimum.accumulate(scaled[::-1])[::-1]
        out = np.empty(m)
        # mathematically stepped >= p; the clamp removes a rounding error of one ulp
        out[order] = np.minimum(1.0, np.maximum(stepped, values[order]))
        adjusted[defined] = out
    return adjusted


class Correctable:
    """Mixin for test results that have ``pvalues``, ``alpha``, ``null`` and ``limitations``."""

    pvalues: np.ndarray
    alpha: float
    null: str
    limitations: tuple[str, ...]

    def adjusted(self, method: str = "fdr_bh") -> np.ndarray:
        """Return the p-values adjusted over the defined cells (see :func:`adjust_pvalues`)."""
        return adjust_pvalues(self.pvalues, method)

    def significant(self, alpha: float | None = None, method: str = "fdr_bh") -> np.ndarray:
        """Return a boolean mask of the cells that survive correction.

        Undefined cells are never significant. The family is the set of defined cells of this
        result, so choose ``reduce=`` when testing to define the family you mean. Significance
        rejects :attr:`null`, with the reservations listed in :attr:`limitations`.

        Parameters
        ----------
        alpha : float, optional
            Defaults to the ``alpha`` the test was run with.
        method : {"fdr_bh", "bonferroni", "none"}
        """
        level = self.alpha if alpha is None else check_alpha(alpha)
        adjusted = self.adjusted(method)
        return np.where(np.isnan(adjusted), False, adjusted <= level)
