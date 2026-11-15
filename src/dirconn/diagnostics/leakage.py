"""Analysis of the leakage studies of the benchmark.

Two studies use the score matrices that a protocol with ``store_scores`` keeps in every stage card:

* **Region pairs** (:func:`leakage_correlation`): for many pairs of regions without any edge, the
  score of the false edge ``x -> y`` is compared with how much of region ``x`` the inverse method
  puts into the series of region ``y`` (the crosstalk matrix of
  :func:`dirconn.inverse.crosstalk`, stored in the results header). The region-level stage has no
  leakage and is the control.
* **The neighbour** (:func:`adjacent_report`): in ``leakage_adjacent`` the only true edge is
  ``Z -> X`` and ``Y`` is the neighbour into which ``X`` leaks; every edge into ``Y`` is a false
  edge. Its score is divided by the score of the true edge, so estimators with different units
  can be read side by side.

Both work on the records of :func:`dirconn.protocol.read_results` and need no head model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata

from dirconn.scenario import load_scenario

STAGES: tuple[str, ...] = ("roi", "sensor", "source")
PAIR_PREFIX = "pair:"


def _spearman(scores: np.ndarray, leaks: np.ndarray) -> float:
    """Spearman correlation; NaN (without a warning) when one of the two is constant."""
    if scores.size < 3:
        return math.nan
    a, b = rankdata(scores), rankdata(leaks)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def leakage_correlation(
    header: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Return the Spearman correlation between false-edge scores and the crosstalk of the pair.

    Every finished task of a region pair gives two cells per estimator and stage: the score of
    ``x -> y`` is paired with ``|crosstalk[y, x]|``, the leak of ``x`` into the series of ``y``, and
    the score of ``y -> x`` with ``|crosstalk[x, y]|``. A score is a rank-only quantity here, so
    estimators with different units need no normalisation. The interval is a percentile bootstrap
    over **pairs** (all the cells and seeds of a pair move together), 95%.

    Parameters
    ----------
    header : mapping
        The results header; ``header["crosstalk"][modality]`` has ``names`` and ``matrix``.
    records : sequence of mapping
        Task records; only finished tasks whose scenario is ``pair:<x>:<y>`` are used.
    n_boot : int
        Bootstrap resamples.
    seed : int
        Seed of the bootstrap.

    Returns
    -------
    list of dict
        One row per estimator, modality and stage: ``estimator``, ``modality``, ``stage``,
        ``n_cells``, ``n_pairs``, ``spearman``, ``ci_low``, ``ci_high``. ``NaN`` where the
        correlation is undefined (every score equal, fewer than three cells).

    Raises
    ------
    ValueError
        If a modality has no crosstalk matrix or a region of a pair is not in it.
    """
    crosstalk = header.get("crosstalk", {})
    cells: dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]] = {}
    for record in records:
        task = record.get("task", {})
        scenario = str(task.get("scenario", ""))
        if record.get("status") != "ok" or not scenario.startswith(PAIR_PREFIX):
            continue
        modality = task["modality"]
        if modality not in crosstalk:
            raise ValueError(
                f"The results header has no crosstalk matrix for the modality {modality!r}; "
                f"it has {sorted(crosstalk)}."
            )
        names = list(crosstalk[modality]["names"])
        matrix = np.abs(np.asarray(crosstalk[modality]["matrix"], dtype=float))
        for estimator, payload in record.get("result", {}).items():
            if "analysed" not in payload:
                continue  # a protocol that did not store the scores
            x, y = payload["analysed"]
            for region in (x, y):
                if region not in names:
                    raise ValueError(f"The region {region!r} is not in the crosstalk matrix.")
            ix, iy = names.index(x), names.index(y)
            for stage in STAGES:
                card = payload.get(stage)
                if not isinstance(card, Mapping) or "scores" not in card:
                    continue
                scores = np.asarray(card["scores"], dtype=float)
                points = [(scores[0, 1], matrix[iy, ix]), (scores[1, 0], matrix[ix, iy])]
                bucket = cells.setdefault((estimator, modality, stage), {}).setdefault(scenario, [])
                bucket.extend((float(s), float(m)) for s, m in points if np.isfinite(s))
    rng = np.random.default_rng(seed)
    order = {s: i for i, s in enumerate(STAGES)}
    rows = []
    for (estimator, modality, stage), by_pair in sorted(
        cells.items(), key=lambda kv: (kv[0][0], kv[0][1], order[kv[0][2]])
    ):
        pairs = [np.asarray(v, dtype=float) for v in by_pair.values() if v]
        if not pairs:
            continue
        flat = np.concatenate(pairs)
        estimate = _spearman(flat[:, 0], flat[:, 1])
        draws = []
        if len(pairs) >= 2:
            for _ in range(int(n_boot)):
                picked = np.concatenate([pairs[k] for k in rng.integers(0, len(pairs), len(pairs))])
                value = _spearman(picked[:, 0], picked[:, 1])
                if not math.isnan(value):
                    draws.append(value)
        low, high = (
            (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
            if len(draws) >= 10
            else (math.nan, math.nan)
        )
        rows.append(
            {
                "estimator": estimator,
                "modality": modality,
                "stage": stage,
                "n_cells": int(flat.shape[0]),
                "n_pairs": len(pairs),
                "spearman": estimate,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return rows


def _summary(values: Sequence[float]) -> dict[str, float]:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return {"mean": math.nan, "min": math.nan, "max": math.nan, "sd": math.nan}
    sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan
    return {
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "sd": sd,
    }


def _relative(scores: np.ndarray, cell: tuple[int, int], true_cell: tuple[int, int]) -> float:
    """``scores[cell]`` over the score of the true edge; NaN if that is not positive."""
    true = scores[true_cell]
    if not math.isfinite(true) or true <= 0:
        return math.nan
    value = scores[cell]
    return float(value / true) if math.isfinite(value) else math.nan


def adjacent_report(
    records: Sequence[Mapping[str, Any]],
    *,
    scenario: str = "leakage_adjacent",
    paired: bool = False,
) -> list[dict[str, Any]]:
    """Summarise the neighbour scenario per estimator, modality and SNR.

    The scenario has nodes ``Z``, ``X``, ``Y`` in this order, the single true edge ``Z -> X`` and no
    edge into ``Y``; ``Y`` is the last node. A false edge is a cell into ``Y`` (``z_to_y`` and
    ``x_to_y``); its relative score is its score over the score of the true edge of the same
    card, so 0 means nothing and 1 means as strong as the real edge.

    Parameters
    ----------
    records : sequence of mapping
        Task records of a protocol with ``store_scores``; only finished tasks of ``scenario``.
    scenario : str
        A shipped scenario with this shape.
    paired : bool
        ``False`` (default): one row per group and stage with ``auroc`` and ``direction_accuracy``
        (means over the seeds), ``z_to_y`` and ``x_to_y`` (``mean``, ``min``, ``max`` of the
        relative scores) and ``n``. ``True``: one row per group with the **source minus the
        region level** difference, paired by seed, of ``auroc``, ``direction_accuracy``,
        ``z_to_y`` and ``x_to_y`` (``mean`` and ``sd``) and ``n``.

    Returns
    -------
    list of dict
        Empty when there is no finished task of the scenario.
    """
    network = load_scenario(scenario).network
    names = list(network.names)
    true_cells = np.argwhere(network.adjacency())
    if len(true_cells) != 1:
        raise ValueError(f"The scenario {scenario!r} must have exactly one true edge.")
    true_cell = (int(true_cells[0][0]), int(true_cells[0][1]))
    receiver = len(names) - 1
    z_cell, x_cell = (0, receiver), (1, receiver)

    groups: dict[tuple[str, str, float], dict[int, dict[str, dict[str, float]]]] = {}
    for record in records:
        task = record.get("task", {})
        if record.get("status") != "ok" or task.get("scenario") != scenario:
            continue
        for estimator, payload in record.get("result", {}).items():
            if "analysed" not in payload:
                continue
            if list(payload["analysed"]) != names:
                raise ValueError(
                    f"The nodes of a record, {payload.get('analysed')}, are not {names}."
                )
            per_stage = {}
            for stage in STAGES:
                card = payload.get(stage)
                if not isinstance(card, Mapping) or "scores" not in card:
                    continue
                scores = np.asarray(card["scores"], dtype=float)
                per_stage[stage] = {
                    "auroc": float(card.get("auroc", math.nan)),
                    "direction_accuracy": float(card.get("direction_accuracy", math.nan)),
                    "z_to_y": _relative(scores, z_cell, true_cell),
                    "x_to_y": _relative(scores, x_cell, true_cell),
                }
            key = (estimator, task["modality"], float(task["snr_db"]))
            groups.setdefault(key, {})[int(task["seed_index"])] = per_stage

    rows: list[dict[str, Any]] = []
    for (estimator, modality, snr), by_seed in sorted(groups.items()):
        base = {"estimator": estimator, "modality": modality, "snr_db": snr}
        if paired:
            both = [s for s in by_seed.values() if "roi" in s and "source" in s]
            if not both:
                continue
            row: dict[str, Any] = {**base, "n": len(both)}
            for measure in ("auroc", "direction_accuracy", "z_to_y", "x_to_y"):
                diffs = [s["source"][measure] - s["roi"][measure] for s in both]
                summary = _summary(diffs)
                row[measure] = {"mean": summary["mean"], "sd": summary["sd"]}
            rows.append(row)
            continue
        for stage in STAGES:
            stage_cards = [s[stage] for s in by_seed.values() if stage in s]
            if not stage_cards:
                continue
            rows.append(
                {
                    **base,
                    "stage": stage,
                    "n": len(stage_cards),
                    "auroc": _summary([c["auroc"] for c in stage_cards])["mean"],
                    "direction_accuracy": _summary([c["direction_accuracy"] for c in stage_cards])[
                        "mean"
                    ],
                    "z_to_y": {
                        k: v
                        for k, v in _summary([c["z_to_y"] for c in stage_cards]).items()
                        if k != "sd"
                    },
                    "x_to_y": {
                        k: v
                        for k, v in _summary([c["x_to_y"] for c in stage_cards]).items()
                        if k != "sd"
                    },
                }
            )
    return rows
