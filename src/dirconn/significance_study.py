"""Analysis of the significance study of the benchmark.

A protocol of ``kind: significance`` keeps, for every simulated dataset, the p-value matrix of every
test. :func:`significance_report` turns them into the numbers that say how the tests behave: how
often a cell without an edge is called significant (the false-positive rate, at several levels),
how often a true edge is found (the power), what the Benjamini-Hochberg correction does, and how
far the p-values are from what the null hypothesis predicts. It works on the records of
:func:`dirconn.protocol.read_results` and needs no data.

Cells are not independent (the six cells of a dataset share the data), so the intervals of the
rates are bootstrapped over **datasets**, not cells; a proportion of datasets has a Wilson interval.

Spectral estimators (PDC, rPDC) have a p-value per cell and frequency; :func:`frequency_report`
summarises them, per band, pointwise and per pair after a BH correction over the frequencies of
the band.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

import numpy as np

from dirconn.bands import BANDS
from dirconn.correction import adjust_pvalues
from dirconn.scenario import list_scenarios, load_scenario

ALPHAS: tuple[float, ...] = (0.01, 0.05, 0.10)
Z_95 = 1.959963984540054


def _wilson(k: int, n: int) -> dict[str, float]:
    if n == 0:
        return {"estimate": math.nan, "low": math.nan, "high": math.nan}
    p = k / n
    denominator = 1.0 + Z_95**2 / n
    centre = (p + Z_95**2 / (2 * n)) / denominator
    half = Z_95 * math.sqrt(p * (1 - p) / n + Z_95**2 / (4 * n**2)) / denominator
    return {"estimate": p, "low": max(0.0, centre - half), "high": min(1.0, centre + half)}


def _cluster_rate(
    hits: np.ndarray, cells: np.ndarray, rng: np.random.Generator, n_boot: int
) -> dict[str, float]:
    """Return ``sum(hits) / sum(cells)`` with a percentile bootstrap interval over datasets."""
    total = float(cells.sum())
    estimate = float(hits.sum() / total)
    n = len(hits)
    draws = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, n, n)
        denominator = float(cells[pick].sum())
        if denominator > 0:
            draws.append(float(hits[pick].sum() / denominator))
    if len(draws) < 10:
        return {"estimate": estimate, "low": math.nan, "high": math.nan}
    return {
        "estimate": estimate,
        "low": float(np.percentile(draws, 2.5)),
        "high": float(np.percentile(draws, 97.5)),
    }


@cache
def _truth(scenario: str, coupling: float | None, names: tuple[str, ...]) -> np.ndarray:
    """Boolean matrix of the true edges of a task, over the analysed nodes in the given order."""
    from dirconn.protocol import apply_coupling  # protocol imports this module for its summary

    spec = load_scenario(scenario)
    if coupling is not None:
        spec = apply_coupling(spec, coupling)
    unknown = [n for n in names if n not in spec.network.names]
    if unknown:
        raise ValueError(
            f"The analysed nodes {unknown} are not nodes of the scenario {scenario!r}."
        )
    order = [spec.network.names.index(n) for n in names]
    return spec.network.adjacency()[np.ix_(order, order)]


def _bh_decisions(p: np.ndarray, q: float) -> np.ndarray:
    """Return which off-diagonal cells survive the BH correction over the cells of the matrix."""
    off = ~np.eye(p.shape[0], dtype=bool)
    adjusted = np.full(p.shape, np.nan)
    adjusted[off] = adjust_pvalues(p[off], "fdr_bh")
    return np.where(np.isnan(adjusted), False, adjusted <= q)


def significance_report(
    records: Sequence[Mapping[str, Any]],
    *,
    n_surrogates: int = 39,
    alphas: Sequence[float] = ALPHAS,
    q: float = 0.05,
    n_boot: int = 1000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Summarise the tests of a significance study, per group of datasets.

    A group is a scenario, coupling, SNR, data length, estimator and test. The truth of a task is
    the edge set of its scenario (with the coupling applied): a cell is *null* where there is no
    edge and an *edge* cell where there is one. Every p-value of a surrogate test is a multiple of
    ``1 / (n_surrogates + 1)``, so a level below that cannot be reached and is reported as ``None``.

    Parameters
    ----------
    records : sequence of mapping
        Task records; only finished ones of a significance protocol are used.
    n_surrogates : int
        Surrogates of the surrogate tests (to know their smallest p-value).
    alphas : sequence of float
        Levels of the uncorrected rates.
    q : float
        Level of the Benjamini-Hochberg correction over the six cells of a dataset.
    n_boot, seed : int
        Bootstrap resamples over datasets and their seed.

    Returns
    -------
    list of dict
        Per group: ``scenario``, ``coupling``, ``snr_db``, ``duration_s``, ``estimator``,
        ``test``, ``n_datasets``, ``n_null_cells``, ``n_edge_cells``; ``null_rate`` and ``power``
        (by level: ``estimate``, ``low``, ``high``, or ``None`` when the level cannot be reached
        or there is no such cell); ``bh``: ``any_false_rejection`` (share of datasets with a
        null cell rejected, Wilson interval), ``power``, ``fdp`` (mean false-discovery
        proportion); ``mean_p_null`` and ``expected_mean_p``; ``bh_cells_needed``: how many cells
        must reach the smallest p-value for BH to reject any.
    """
    shipped = set(list_scenarios())
    groups: dict[tuple[Any, ...], list[tuple[np.ndarray, np.ndarray]]] = {}
    for record in records:
        task = record.get("task", {})
        if record.get("status") != "ok":
            continue
        for estimator, payload in record.get("result", {}).items():
            if "tests" not in payload or "freqs" in payload:
                continue  # no p-values, or one per frequency (frequency_report)
            if task["scenario"] not in shipped:
                continue  # for example the pairs of a leakage study
            truth = _truth(task["scenario"], task["coupling"], tuple(payload["analysed"]))
            for test, result in payload["tests"].items():
                key = (
                    task["scenario"],
                    task["coupling"],
                    task["snr_db"],
                    task["duration_s"],
                    estimator,
                    test,
                )
                groups.setdefault(key, []).append((np.asarray(result["pvalues"], float), truth))

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for key, datasets in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        scenario, coupling, snr, duration, estimator, test = key
        is_surrogate = test.startswith("surrogate")
        smallest = 1.0 / (n_surrogates + 1) if is_surrogate else 0.0
        n_cells = int((~np.eye(datasets[0][0].shape[0], dtype=bool)).sum())

        null_hits: dict[float, list[int]] = {a: [] for a in alphas}
        edge_hits: dict[float, list[int]] = {a: [] for a in alphas}
        null_counts, edge_counts = [], []
        bh_null, bh_edge, false_any, fdp, null_p = [], [], [], [], []
        for p, truth in datasets:
            off = ~np.eye(p.shape[0], dtype=bool)
            is_edge = truth & off
            is_null = ~truth & off
            null_counts.append(int(is_null.sum()))
            edge_counts.append(int(is_edge.sum()))
            null_p.extend(p[is_null].tolist())
            for a in alphas:
                null_hits[a].append(int((p[is_null] <= a).sum()))
                edge_hits[a].append(int((p[is_edge] <= a).sum()))
            rejected = _bh_decisions(p, q)
            false = int((rejected & is_null).sum())
            total = int(rejected.sum())
            bh_null.append(false)
            bh_edge.append(int((rejected & is_edge).sum()))
            false_any.append(int(false > 0))
            fdp.append(false / max(total, 1))

        nulls, edges = np.array(null_counts), np.array(edge_counts)

        def rate(
            hits: list[int],
            cells: np.ndarray,
            level: float | None,
            smallest: float = smallest,
        ) -> dict[str, float] | None:
            if cells.sum() == 0 or (level is not None and level < smallest - 1e-12):
                return None
            return _cluster_rate(np.array(hits, float), cells.astype(float), rng, n_boot)

        rows.append(
            {
                "scenario": scenario,
                "coupling": coupling,
                "snr_db": snr,
                "duration_s": duration,
                "estimator": estimator,
                "test": test,
                "n_datasets": len(datasets),
                "n_null_cells": int(nulls.sum()),
                "n_edge_cells": int(edges.sum()),
                "null_rate": {a: rate(null_hits[a], nulls, a) for a in alphas},
                "power": {a: rate(edge_hits[a], edges, a) for a in alphas},
                "bh": {
                    "any_false_rejection": _wilson(sum(false_any), len(datasets)),
                    "power": rate(bh_edge, edges, None),
                    "fdp": float(np.mean(fdp)),
                },
                "mean_p_null": float(np.mean(null_p)) if null_p else math.nan,
                "expected_mean_p": (
                    (n_surrogates + 2) / (2 * (n_surrogates + 1)) if is_surrogate else 0.5
                ),
                "bh_cells_needed": (
                    max(1, math.ceil(smallest * n_cells / q - 1e-9)) if is_surrogate else 1
                ),
            }
        )
    return rows


def _rate(
    hits: Sequence[int],
    cells: Sequence[int],
    level: float | None,
    smallest: float,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, float] | None:
    """Return a rate over datasets; ``None`` without cells or below the smallest p-value."""
    total = np.asarray(cells, float)
    if total.sum() == 0 or (level is not None and level < smallest - 1e-12):
        return None
    return _cluster_rate(np.asarray(hits, float), total, rng, n_boot)


def _any_after_bh(p: np.ndarray, q: float) -> bool:
    """Whether any of the p-values of one pair survives BH over them (NaN counts as not)."""
    return bool((adjust_pvalues(np.nan_to_num(p, nan=1.0), "fdr_bh") <= q).any())


def frequency_report(
    records: Sequence[Mapping[str, Any]],
    *,
    n_surrogates: int = 39,
    alphas: Sequence[float] = ALPHAS,
    q: float = 0.05,
    bands: Sequence[str] = ("alpha", "beta"),
    profile_alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Summarise the per-frequency tests of a significance study, per group and band.

    A group is as in :func:`significance_report`; only results that carry a frequency grid
    (``freqs``) are used. Every group has a row for the whole grid (``band`` ``"all"``) and one
    for each band of ``bands`` (names of :data:`dirconn.bands.BANDS`, both limits inclusive). A
    *pair* is an off-diagonal cell, null or an edge as in :func:`significance_report`.

    Parameters
    ----------
    records : sequence of mapping
        Task records; only finished ones of a significance protocol are used.
    n_surrogates : int
        Surrogates of the surrogate tests (to know their smallest p-value).
    alphas : sequence of float
        Levels of the pointwise rates.
    q : float
        Level of the Benjamini-Hochberg correction over the frequencies of a band, per pair.
    bands : sequence of str
        Bands with a row of their own.
    profile_alpha : float
        Level of the pointwise rate per frequency (``null_profile``).
    n_boot, seed : int
        Bootstrap resamples over datasets and their seed.

    Returns
    -------
    list of dict
        Per group and band: ``scenario``, ``coupling``, ``snr_db``, ``duration_s``,
        ``estimator``, ``test``, ``band``, ``fmin``, ``fmax``, ``n_freqs`` (grid points in the
        band), ``n_datasets``, ``n_null_pairs``, ``n_edge_pairs``; ``null_rate`` and ``power``,
        pointwise over pairs and frequencies (by level: ``estimate``, ``low``, ``high``, or
        ``None``); ``liberal`` (by level: whether the lower end of the interval of the null rate
        is above the level, ``None`` without a rate); ``pair_bh``: ``null_rate`` (share of null
        pairs with any frequency rejected after BH within the band) and ``power`` (the same for
        edges); ``mean_p_null`` and ``expected_mean_p``; ``bh_freqs_needed`` (how many
        frequencies of the band must reach the smallest p-value for BH to reject any);
        ``null_profile``: the grid of the band and the pointwise null rate at ``profile_alpha``
        at each of its frequencies.

    Raises
    ------
    ValueError
        For a band that is not in :data:`dirconn.bands.BANDS`.
    """
    unknown = [b for b in bands if b not in BANDS]
    if unknown:
        raise ValueError(f"Unknown band(s) {unknown}; known: {list(BANDS)}.")
    shipped = set(list_scenarios())
    groups: dict[tuple[Any, ...], list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for record in records:
        task = record.get("task", {})
        if record.get("status") != "ok":
            continue
        for estimator, payload in record.get("result", {}).items():
            if "tests" not in payload or "freqs" not in payload:
                continue
            if task["scenario"] not in shipped:
                continue
            truth = _truth(task["scenario"], task["coupling"], tuple(payload["analysed"]))
            freqs = np.asarray(payload["freqs"], float)
            for test, result in payload["tests"].items():
                key = (
                    task["scenario"],
                    task["coupling"],
                    task["snr_db"],
                    task["duration_s"],
                    estimator,
                    test,
                )
                p = np.asarray(result["pvalues"], float)
                groups.setdefault(key, []).append((p, truth, freqs))

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for key, datasets in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        scenario, coupling, snr, duration, estimator, test = key
        is_surrogate = test.startswith("surrogate")
        smallest = 1.0 / (n_surrogates + 1) if is_surrogate else 0.0
        freqs = datasets[0][2]
        limits = [("all", float(freqs.min()), float(freqs.max()))]
        limits += [(b, float(BANDS[b][0]), float(BANDS[b][1])) for b in bands]
        for band, fmin, fmax in limits:
            in_band = (freqs >= fmin - 1e-9) & (freqs <= fmax + 1e-9)
            n_freqs = int(in_band.sum())
            if n_freqs == 0:
                continue
            null_hits: dict[float, list[int]] = {a: [] for a in alphas}
            edge_hits: dict[float, list[int]] = {a: [] for a in alphas}
            null_cells, edge_cells, null_pairs, edge_pairs = [], [], [], []
            pair_null, pair_edge, null_p = [], [], []
            profile = np.zeros(n_freqs)
            for p, truth, _ in datasets:
                off = ~np.eye(p.shape[0], dtype=bool)
                band_p = p[:, :, in_band]
                null = band_p[~truth & off]  # (null pairs, frequencies)
                edge = band_p[truth & off]
                null_pairs.append(len(null))
                edge_pairs.append(len(edge))
                null_cells.append(null.size)
                edge_cells.append(edge.size)
                null_p.extend(null.ravel().tolist())
                for a in alphas:
                    null_hits[a].append(int((null <= a).sum()))
                    edge_hits[a].append(int((edge <= a).sum()))
                profile += (null <= profile_alpha).sum(axis=0)
                pair_null.append(sum(_any_after_bh(row, q) for row in null))
                pair_edge.append(sum(_any_after_bh(row, q) for row in edge))

            null_rate = {
                a: _rate(null_hits[a], null_cells, a, smallest, rng, n_boot) for a in alphas
            }
            liberal: dict[float, bool | None] = {}
            for a, r in null_rate.items():
                low = None if r is None else r["low"]
                liberal[a] = None if low is None or math.isnan(low) else bool(low > a)
            n_null = int(sum(null_pairs))
            rows.append(
                {
                    "scenario": scenario,
                    "coupling": coupling,
                    "snr_db": snr,
                    "duration_s": duration,
                    "estimator": estimator,
                    "test": test,
                    "band": band,
                    "fmin": fmin,
                    "fmax": fmax,
                    "n_freqs": n_freqs,
                    "n_datasets": len(datasets),
                    "n_null_pairs": n_null,
                    "n_edge_pairs": int(sum(edge_pairs)),
                    "null_rate": null_rate,
                    "power": {
                        a: _rate(edge_hits[a], edge_cells, a, smallest, rng, n_boot) for a in alphas
                    },
                    "liberal": liberal,
                    "pair_bh": {
                        "null_rate": _rate(pair_null, null_pairs, None, smallest, rng, n_boot),
                        "power": _rate(pair_edge, edge_pairs, None, smallest, rng, n_boot),
                    },
                    "mean_p_null": float(np.nanmean(null_p)) if null_p else math.nan,
                    "expected_mean_p": (
                        (n_surrogates + 2) / (2 * (n_surrogates + 1)) if is_surrogate else 0.5
                    ),
                    "bh_freqs_needed": (
                        max(1, math.ceil(smallest * n_freqs / q - 1e-9)) if is_surrogate else 1
                    ),
                    "null_profile": {
                        "freqs": freqs[in_band].tolist(),
                        "rate": (profile / n_null).tolist() if n_null else None,
                    },
                }
            )
    return rows
