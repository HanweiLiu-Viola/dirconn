"""Analysis of the leakage studies of the benchmark.

Two studies use the score matrices that a protocol with ``store_scores`` keeps in every stage card:

* **Region pairs** (:func:`leakage_correlation`): for many pairs of regions without any edge, the
  score of the false edge ``x -> y`` is compared with how much of region ``x`` the inverse method
  puts into the series of region ``y`` (the crosstalk matrix of
  :func:`dirconn.simulation.inverse.crosstalk`, stored in the results header). The region-level
  stage has no
  leakage and is the control.
* **The neighbour** (:func:`adjacent_report`): in ``leakage_adjacent`` the only true edge is
  ``Z -> X`` and ``Y`` is the neighbour into which ``X`` leaks; every edge into ``Y`` is a false
  edge. Its score is divided by the score of the true edge, so estimators with different units
  can be read side by side.

Two more compare estimators that discount zero-delay mixing (time-reversed GC, imaginary coherence,
PSI) with a reference estimator (conditional GC), unit by unit:

* **Region pairs** (:func:`pairs_controls_report`): how many pairs without an edge each estimator
  flags as a false positive, against a threshold taken at the region level.
* **The neighbour** (:func:`adjacent_controls_report`): in how many seeds a false edge of a
  ``leakage_adjacent*`` scenario scores at least as high as the true edge.

Rates carry a Wilson 95% interval; each estimator is compared with the reference by an exact
McNemar test on the discordant units, with Holm-adjusted p-values over the leaky stages.

All work on the records of :func:`dirconn.validation.protocol.read_results` and need no head model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata

from dirconn.methods.base import method_info
from dirconn.simulation.scenario import load_scenario
from dirconn.stats.correction import holm
from dirconn.stats.counts import mcnemar_exact, wilson_interval

STAGES: tuple[str, ...] = ("roi", "sensor", "source")
PAIR_PREFIX = "pair:"
ADJACENT_PREFIX = "leakage_adjacent"
"""Every scenario whose name starts with this has the neighbour shape (Z -> X, Y next to X)."""
HOLM_STAGES: tuple[str, ...] = ("sensor", "source")
"""The stages in the Holm family of the control reports. The region level has no leakage: in the
pairs study it defines the threshold (about 5% flagged by construction), so a comparison there
tests nothing about leakage and is reported with its raw p-value only."""


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


# ---- controls against a reference estimator --------------------------------------------


def is_pair_measure(estimator: str) -> bool:
    """Whether ``|score|`` of an estimator is the same in both directions of a pair.

    True for undirected measures (imaginary coherence) and directed net flows, which are
    antisymmetric (time-reversed GC, PSI): the cell ``j -> i`` of such a measure repeats the cell
    ``i -> j`` and is not a second edge. ``estimator`` is a protocol label such as ``"psi/mt"``
    or ``"gc/bic@lcmv"``; a method that is not known counts as an ordinary directed one.
    """
    method = estimator.partition("@")[0].partition("/")[0]
    info = method_info(method)
    return info is not None and (not info.directed or info.antisymmetric)


def _estimator_order(found: Sequence[str], reference: str, listed: Sequence[str] = ()) -> list[str]:
    """Put the reference first, then the order of the protocol's list, then the rest sorted."""
    rest = [e for e in listed if e in found and e != reference]
    rest += sorted(e for e in found if e != reference and e not in rest)
    return ([reference] if reference in found else []) + rest


def _unit(task: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the simulation of a task, the unit the estimators are paired on (no modality)."""
    return (
        task.get("scenario"),
        task.get("seed_index"),
        task.get("snr_db"),
        task.get("duration_s"),
        task.get("source_model"),
        task.get("coupling"),
        task.get("noise", "simulated"),
    )


def _count_row(flags: Mapping[Any, bool]) -> dict[str, Any]:
    n, k = len(flags), sum(bool(v) for v in flags.values())
    low, high = wilson_interval(k, n)
    return {"n": n, "count": k, "rate": k / n if n else math.nan, "ci_low": low, "ci_high": high}


def _versus(flags: Mapping[Any, bool], reference: Mapping[Any, bool] | None) -> dict[str, Any]:
    """Discordant counts against the reference on their common units, and the McNemar p-value."""
    if reference is None:
        return {"n_paired": 0, "only_reference": None, "only_this": None, "mcnemar_p": math.nan}
    common = [u for u in flags if u in reference]
    only_reference = sum(1 for u in common if reference[u] and not flags[u])
    only_this = sum(1 for u in common if flags[u] and not reference[u])
    return {
        "n_paired": len(common),
        "only_reference": only_reference,
        "only_this": only_this,
        "mcnemar_p": mcnemar_exact(only_reference, only_this),
    }


def _add_holm(rows: list[dict[str, Any]], reference: str) -> None:
    """Holm over the non-reference rows of the leaky stages; NaN for the others."""
    family = [
        i
        for i, r in enumerate(rows)
        if r["estimator"] != reference
        and r["stage"] in HOLM_STAGES
        and math.isfinite(r["mcnemar_p"])
    ]
    adjusted = holm(np.asarray([rows[i]["mcnemar_p"] for i in family], dtype=float))
    for r in rows:
        r["holm_p"] = math.nan
    for i, value in zip(family, adjusted, strict=True):
        rows[i]["holm_p"] = float(value)


def pairs_controls_report(
    header: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    reference: str = "gc/bic",
    quantile: float = 95.0,
) -> list[dict[str, Any]]:
    """Count the leakage false positives of region pairs per estimator, against a reference.

    A pair has no edge. Its strength for an estimator and stage is the strongest ``|score|`` of
    its two cells ``x -> y`` and ``y -> x`` (``|score|`` because net measures are signed). The
    threshold of an estimator is the ``quantile`` percentile of its strengths **at the region
    level** of the same modality, where there is no leakage; a pair is a false positive at a
    stage when its strength is above that threshold. At the region level about
    ``100 - quantile`` percent of the pairs are flagged by construction.

    Every estimator other than ``reference`` is compared with it on the same simulations: the
    reference false positives this one does not flag (``removed``), the false positives it adds
    (``added``), and the exact McNemar p-value from these two discordant counts. ``holm_p`` is the
    Holm adjustment over all the non-reference rows of the sensor and source stages (the
    region level is not in the family, see :data:`HOLM_STAGES`).

    Parameters
    ----------
    header : mapping
        The results header; the order of ``header["settings"]["estimators"]`` orders the rows.
    records : sequence of mapping
        Task records of a ``region_pairs`` protocol with ``store_scores``; only finished tasks
        whose scenario is ``pair:<x>:<y>`` are used.
    reference : str
        The estimator label the others are compared with.
    quantile : float
        Percentile (0 to 100) of the region-level strengths that is the threshold.

    Returns
    -------
    list of dict
        One row per modality, stage and estimator: ``modality``, ``stage``, ``estimator``,
        ``threshold``, ``n_pairs``, ``false_positives``, ``rate``, ``ci_low``, ``ci_high``
        (Wilson 95%), and ``n_paired``, ``removed``, ``added``, ``mcnemar_p``, ``holm_p``
        (``None`` counts and NaN p-values for the reference and when the reference is absent).
        An estimator without region-level scores in a modality has no threshold and no rows.
    """
    if not 0.0 <= quantile <= 100.0:
        raise ValueError(f"quantile must lie between 0 and 100, got {quantile}.")
    strengths: dict[tuple[str, str, str], dict[tuple[Any, ...], float]] = {}
    for record in records:
        task = record.get("task", {})
        if record.get("status") != "ok" or not str(task.get("scenario", "")).startswith(
            PAIR_PREFIX
        ):
            continue
        for estimator, payload in record.get("result", {}).items():
            for stage in STAGES:
                card = payload.get(stage) if isinstance(payload, Mapping) else None
                if not isinstance(card, Mapping) or "scores" not in card:
                    continue
                scores = np.abs(np.asarray(card["scores"], dtype=float))
                cells = np.array([scores[0, 1], scores[1, 0]])
                if not np.isfinite(cells).any():
                    continue
                strongest = float(np.nanmax(cells))
                strengths.setdefault((task["modality"], stage, estimator), {})[_unit(task)] = (
                    strongest
                )
    listed = list(header.get("settings", {}).get("estimators", []) or [])
    order = _estimator_order(sorted({e for _, _, e in strengths}), reference, listed)
    rows: list[dict[str, Any]] = []
    for modality in sorted({m for m, _, _ in strengths}):
        thresholds = {
            e: float(np.percentile(list(strengths[(modality, "roi", e)].values()), quantile))
            for e in order
            if strengths.get((modality, "roi", e))
        }
        for stage in STAGES:
            flagged = {
                e: {u: v > thresholds[e] for u, v in strengths[(modality, stage, e)].items()}
                for e in order
                if e in thresholds and strengths.get((modality, stage, e))
            }
            ref_flags = flagged.get(reference)
            for estimator, flags in flagged.items():
                counts = _count_row(flags)
                if estimator == reference:
                    versus = _versus(flags, None)
                else:
                    versus = _versus(flags, ref_flags)
                rows.append(
                    {
                        "modality": modality,
                        "stage": stage,
                        "estimator": estimator,
                        "threshold": thresholds[estimator],
                        "n_pairs": counts["n"],
                        "false_positives": counts["count"],
                        "rate": counts["rate"],
                        "ci_low": counts["ci_low"],
                        "ci_high": counts["ci_high"],
                        "n_paired": versus["n_paired"],
                        "removed": versus["only_reference"],
                        "added": versus["only_this"],
                        "mcnemar_p": versus["mcnemar_p"],
                    }
                )
    _add_holm(rows, reference)
    return rows


def adjacent_roles(scenario: str) -> tuple[list[str], int, int, int]:
    """Return the node names and the indices of ``Z``, ``X`` and ``Y`` of a neighbour scenario.

    ``Z`` is the source of the only edge, ``X`` its target and ``Y`` the remaining node, the
    neighbour of ``X``; the names are taken from the scenario file, not assumed.

    Raises
    ------
    ValueError
        If the scenario does not have three nodes and exactly one edge.
    """
    network = load_scenario(scenario).network
    names = list(network.names)
    edges = np.argwhere(network.adjacency())
    if len(names) != 3 or len(edges) != 1:
        raise ValueError(
            f"The scenario {scenario!r} must have three nodes and exactly one edge, "
            f"it has {len(names)} nodes and {len(edges)} edges."
        )
    z, x = int(edges[0][0]), int(edges[0][1])
    (y,) = {0, 1, 2} - {z, x}
    return names, z, x, y


def _adjacent_failure(scores: np.ndarray, z: int, x: int, pair_measure: bool) -> bool | None:
    """Whether some false edge has ``|score| >=`` the true edge ``z -> x``; None if undefined."""
    magnitude = np.abs(np.asarray(scores, dtype=float))
    true = magnitude[z, x]
    if not math.isfinite(true):
        return None
    false = ~np.eye(magnitude.shape[0], dtype=bool) & np.isfinite(magnitude)
    false[z, x] = False
    if pair_measure:
        false[x, z] = False  # the same pair as the true edge, not a second edge
    if not false.any():
        return None
    return bool(np.max(magnitude[false]) >= true)


def adjacent_controls_report(
    records: Sequence[Mapping[str, Any]],
    *,
    reference: str = "gc/bic",
) -> list[dict[str, Any]]:
    """Count the seeds of the neighbour scenarios in which a false edge beats the true edge.

    In a ``leakage_adjacent*`` scenario (see :func:`adjacent_roles`) the only true edge is
    ``Z -> X``. A seed **fails** at a stage when some false edge has a ``|score|`` at least as
    high as the true edge (``|score|`` because net measures are signed). For an undirected or a
    net measure (:func:`is_pair_measure`) the cell ``X -> Z`` is the true pair again and not a
    false edge; for an ordinary directed measure it is the reversed, false, direction.

    Every estimator other than ``reference`` is compared with it on the same simulations: the
    reference failures this one avoids (``rescued``), its failures where the reference succeeds
    (``new_failures``), the exact McNemar p-value from these two counts and ``holm_p``, the Holm
    adjustment over all the non-reference rows of the sensor and source stages of all scenarios.

    Parameters
    ----------
    records : sequence of mapping
        Task records of a protocol with ``store_scores``; finished tasks of every scenario whose
        name starts with ``leakage_adjacent`` are used.
    reference : str
        The estimator label the others are compared with.

    Returns
    -------
    list of dict
        One row per scenario, modality, SNR, stage and estimator: ``scenario``, ``modality``,
        ``snr_db``, ``stage``, ``estimator``, ``n`` (seeds), ``failures``, ``rate``, ``ci_low``,
        ``ci_high`` (Wilson 95%), ``n_paired``, ``rescued``, ``new_failures``, ``mcnemar_p``,
        ``holm_p`` (``None`` counts and NaN p-values for the reference and when it is absent).
        Empty when there is no finished task of such a scenario.

    Raises
    ------
    ValueError
        If the nodes of a record are not the nodes of its scenario.
    """
    roles: dict[str, tuple[list[str], int, int, int]] = {}
    fails: dict[tuple[str, str, float, str, str], dict[tuple[Any, ...], bool]] = {}
    listed: list[str] = []
    for record in records:
        task = record.get("task", {})
        scenario = str(task.get("scenario", ""))
        if record.get("status") != "ok" or not scenario.startswith(ADJACENT_PREFIX):
            continue
        if scenario not in roles:
            roles[scenario] = adjacent_roles(scenario)
        names, z, x, _ = roles[scenario]
        for estimator, payload in record.get("result", {}).items():
            if not isinstance(payload, Mapping) or "analysed" not in payload:
                continue
            if list(payload["analysed"]) != names:
                raise ValueError(
                    f"The nodes of a record of {scenario!r}, {payload.get('analysed')}, "
                    f"are not {names}."
                )
            if estimator not in listed:
                listed.append(estimator)
            pair_measure = is_pair_measure(estimator)
            for stage in STAGES:
                card = payload.get(stage)
                if not isinstance(card, Mapping) or "scores" not in card:
                    continue
                failed = _adjacent_failure(
                    np.asarray(card["scores"], dtype=float), z, x, pair_measure
                )
                if failed is None:
                    continue
                key = (scenario, task["modality"], float(task["snr_db"]), stage, estimator)
                fails.setdefault(key, {})[_unit(task)] = failed
    order = _estimator_order(listed, reference, listed)
    stage_order = {s: i for i, s in enumerate(STAGES)}
    groups = sorted({k[:4] for k in fails}, key=lambda k: (k[0], k[1], k[2], stage_order[k[3]]))
    rows: list[dict[str, Any]] = []
    for scenario, modality, snr, stage in groups:
        ref_flags = fails.get((scenario, modality, snr, stage, reference))
        for estimator in order:
            flags = fails.get((scenario, modality, snr, stage, estimator))
            if flags is None:
                continue
            counts = _count_row(flags)
            versus = _versus(flags, None if estimator == reference else ref_flags)
            rows.append(
                {
                    "scenario": scenario,
                    "modality": modality,
                    "snr_db": snr,
                    "stage": stage,
                    "estimator": estimator,
                    "n": counts["n"],
                    "failures": counts["count"],
                    "rate": counts["rate"],
                    "ci_low": counts["ci_low"],
                    "ci_high": counts["ci_high"],
                    "n_paired": versus["n_paired"],
                    "rescued": versus["only_reference"],
                    "new_failures": versus["only_this"],
                    "mcnemar_p": versus["mcnemar_p"],
                }
            )
    _add_holm(rows, reference)
    return rows
