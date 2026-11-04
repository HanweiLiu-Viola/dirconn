"""Analysis of the noise study of the benchmark.

The ``noise`` axis of a protocol gives every seed three arms: the default simulated noise (1/f
background sources plus white noise), white noise only, and the real empty-room recording. The
arms of one seed share their sources, so their difference is the noise alone. :func:`noise_report`
puts the arms side by side, per estimator and stage, and takes the differences seed by seed.
It works on the records of :func:`dirconn.protocol.read_results`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

ARMS: tuple[str, ...] = ("simulated", "white", "empty_room")
STAGES: tuple[str, ...] = ("roi", "sensor", "source")
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("white_minus_simulated", "white", "simulated"),
    ("empty_room_minus_white", "empty_room", "white"),
    ("empty_room_minus_simulated", "empty_room", "simulated"),
)
"""Name, minuend and subtrahend of the paired differences."""
MEASURES: tuple[str, ...] = ("auroc", "direction_accuracy")


def _mean(values: Sequence[float]) -> float:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(finite)) if finite else math.nan


def _difference(minuend: Mapping[int, float], subtrahend: Mapping[int, float]) -> dict[str, float]:
    diffs = [
        minuend[s] - subtrahend[s]
        for s in sorted(set(minuend) & set(subtrahend))
        if math.isfinite(minuend[s]) and math.isfinite(subtrahend[s])
    ]
    return {
        "n": len(diffs),
        "mean": float(np.mean(diffs)) if diffs else math.nan,
        "sd": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else math.nan,
    }


def noise_report(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the arms of the noise study side by side, with the paired differences.

    A group is a scenario, modality, SNR, data length, source model, estimator and stage.
    Per group: ``arms``, by noise (in the order simulated, white, empty_room): ``n`` (seeds with a
    defined AUROC), ``auroc`` and ``direction_accuracy`` (means over seeds); ``paired``: for each
    of white minus simulated, empty room minus white and empty room minus simulated that both of
    whose arms are present, ``n`` (seeds with a defined AUROC in both), and ``auroc`` and
    ``direction_accuracy`` as ``mean`` and ``sd`` (ddof 1, NaN for a single pair) of the
    differences, taken seed by seed. A positive difference means the first arm scores higher.

    Parameters
    ----------
    records : sequence of mapping
        Task records; only finished ones. A task without a ``noise`` key has the simulated noise.

    Returns
    -------
    list of dict
        Empty when there is no finished task.
    """
    cells: dict[tuple[Any, ...], dict[str, dict[int, dict[str, float]]]] = {}
    for record in records:
        task = record.get("task", {})
        if record.get("status") != "ok" or str(task.get("scenario", "")).startswith("pair:"):
            continue
        noise = task.get("noise", "simulated")
        for estimator, payload in record.get("result", {}).items():
            for stage in STAGES:
                card = payload.get(stage)
                if not isinstance(card, Mapping) or "auroc" not in card:
                    continue
                key = (
                    task["scenario"],
                    task["modality"],
                    task["snr_db"],
                    task["duration_s"],
                    task["source_model"],
                    estimator,
                    stage,
                )
                cells.setdefault(key, {}).setdefault(noise, {})[int(task["seed_index"])] = {
                    "auroc": float(card["auroc"]),
                    "direction_accuracy": float(card.get("direction_accuracy", math.nan)),
                }
    rows: list[dict[str, Any]] = []
    for key, arms in sorted(cells.items(), key=lambda kv: tuple(map(str, kv[0]))):
        scenario, modality, snr, duration, source_model, estimator, stage = key
        present = [a for a in ARMS if a in arms]
        out_arms = {}
        for arm in present:
            finite = [v for v in arms[arm].values() if math.isfinite(v["auroc"])]
            out_arms[arm] = {
                "n": len(finite),
                "auroc": _mean([v["auroc"] for v in finite]),
                "direction_accuracy": _mean([v["direction_accuracy"] for v in finite]),
            }
        paired: dict[str, Any] = {}
        for name, first, second in CONTRASTS:
            if first in arms and second in arms:
                entry: dict[str, Any] = {}
                for measure in MEASURES:
                    d = _difference(
                        {
                            s: v[measure]
                            for s, v in arms[first].items()
                            if math.isfinite(v["auroc"])
                        },
                        {
                            s: v[measure]
                            for s, v in arms[second].items()
                            if math.isfinite(v["auroc"])
                        },
                    )
                    entry[measure] = {"mean": d["mean"], "sd": d["sd"]}
                    entry["n"] = d["n"]  # the same seeds for both measures
                paired[name] = entry
        rows.append(
            {
                "scenario": scenario,
                "modality": modality,
                "snr_db": snr,
                "duration_s": duration,
                "source_model": source_model,
                "estimator": estimator,
                "stage": stage,
                "arms": out_arms,
                "paired": paired,
            }
        )
    return rows
