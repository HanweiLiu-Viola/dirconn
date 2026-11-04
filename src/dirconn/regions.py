"""Choosing the regions of benchmark scenarios from the properties of a head model.

The scenario files carry region names, not rules. These functions are the rules, written on plain
arrays so they can be tested without a head model; ``scripts/pick_regions.py`` applies them to the
sample head model and ``tests/test_scenario_library.py`` checks that the shipped files are what
they give. Leakage matrices are the ones of :func:`dirconn.inverse.crosstalk`: ``matrix[i, j]`` is
the response of the series of region ``i`` to region ``j``, so the leak of region ``x`` *into the
series of* ``y`` is ``matrix[y, x]``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from dirconn.headmodel import HeadModel

POOL_VISIBILITY = (1.0, 1.8)
POOL_MIN_DISTANCE = 60.0
LEAKAGE_QUANTILE = 0.9
STRATA_VISIBILITY = (0.9, 1.5)
STRATA_NAMES = ("near", "mid", "far")


def min_visibility(heads: Iterable[HeadModel]) -> np.ndarray:
    """Return for every region its smallest visibility over the given modalities."""
    return np.asarray(np.min([h.visibility for h in heads], axis=0))


def centre_distances(head: HeadModel) -> np.ndarray:
    """Return the distances in mm between the centre sources of all regions."""
    positions = np.vstack([s["rr"][s["vertno"]] for s in head.forward["src"]]) * 1000.0
    centres = positions[head.roi_centre]
    return np.asarray(np.linalg.norm(centres[:, None] - centres[None], axis=2))


def pool(
    visibility: np.ndarray,
    distances: np.ndarray,
    *,
    low: float = POOL_VISIBILITY[0],
    high: float = POOL_VISIBILITY[1],
    min_distance: float = POOL_MIN_DISTANCE,
) -> list[int]:
    """Return the indices of a pool of visible, well separated regions.

    Regions with ``low <= visibility <= high`` are taken in order of decreasing visibility and
    kept when their distance to every region kept so far is at least ``min_distance``.
    """
    visibility = np.asarray(visibility, dtype=float)
    kept: list[int] = []
    for i in np.argsort(-visibility, kind="stable"):
        if not low <= visibility[i] <= high:
            continue
        if all(distances[i, j] >= min_distance for j in kept):
            kept.append(int(i))
    return kept


@dataclass(frozen=True)
class LeakagePair:
    """A region ``source`` that leaks into the series of the region ``receiver``.

    ``score`` is the smaller of the two modalities' percentile ranks of the leak, in ``[0, 1]``.
    """

    source: int
    receiver: int
    score: float


def _check_matrices(matrices: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for key, matrix in matrices.items():
        array = np.abs(np.asarray(matrix, dtype=float))
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(f"The leakage matrix {key!r} must be square, got shape {array.shape}.")
        out[key] = array
    if not out:
        raise ValueError("At least one leakage matrix is needed.")
    return out


def _off_diagonal(matrix: np.ndarray) -> np.ndarray:
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


def leakage_pairs(
    matrices: Mapping[str, np.ndarray],
    visibility: np.ndarray,
    *,
    min_visibility: float = POOL_VISIBILITY[0],
    quantile: float = LEAKAGE_QUANTILE,
) -> list[LeakagePair]:
    """Return the region pairs whose leak is among the strongest on every modality, best first.

    A pair ``x -> y`` qualifies when the leak of ``x`` into the series of ``y`` is above the
    ``quantile`` of all off-diagonal leaks (signs ignored) on **every** modality in ``matrices``
    and both regions have at least ``min_visibility``. The score is the smaller of the percentile
    ranks over the modalities; pairs are ordered by it, then by source and receiver index.
    """
    mats = _check_matrices(matrices)
    visibility = np.asarray(visibility, dtype=float)
    n = visibility.size
    ranks = {}
    thresholds = {}
    for key, matrix in mats.items():
        if matrix.shape != (n, n):
            raise ValueError(f"The leakage matrix {key!r} must be {n} x {n}, got {matrix.shape}.")
        values = np.sort(_off_diagonal(matrix))
        thresholds[key] = np.quantile(values, quantile)
        ranks[key] = np.searchsorted(values, matrix, side="left") / (values.size - 1)
    pairs = []
    for receiver in range(n):
        for source in range(n):
            if source == receiver or min(visibility[source], visibility[receiver]) < min_visibility:
                continue
            if all(mats[k][receiver, source] > thresholds[k] for k in mats):
                score = min(float(ranks[k][receiver, source]) for k in mats)
                pairs.append(LeakagePair(source, receiver, min(score, 1.0)))
    pairs.sort(key=lambda p: (-p.score, p.source, p.receiver))
    return pairs


def distant_region(
    pair: tuple[int, int],
    matrices: Mapping[str, np.ndarray],
    visibility: np.ndarray,
    distances: np.ndarray,
    *,
    min_distance: float = POOL_MIN_DISTANCE,
    min_visibility: float = POOL_VISIBILITY[0],
    max_visibility: float = POOL_VISIBILITY[1],
    quantile: float = LEAKAGE_QUANTILE,
) -> int:
    """Return the most visible region that is far from both regions of ``pair`` and does not leak.

    ``pair`` is ``(source, receiver)``. A candidate must be at least ``min_distance`` mm from
    both, have a visibility between ``min_visibility`` and ``max_visibility`` (the window of the
    pool, so that it does not dominate the sensors) and leak into the series of neither of them
    above the ``quantile`` of all leaks on any modality.

    Raises
    ------
    ValueError
        If no region qualifies.
    """
    mats = _check_matrices(matrices)
    visibility = np.asarray(visibility, dtype=float)
    thresholds = {k: np.quantile(_off_diagonal(m), quantile) for k, m in mats.items()}
    best: int | None = None
    for z in range(visibility.size):
        if z in pair or not min_visibility <= visibility[z] <= max_visibility:
            continue
        if any(distances[z, member] < min_distance for member in pair):
            continue
        if any(mats[k][member, z] > thresholds[k] for k in mats for member in pair):
            continue
        if best is None or visibility[z] > visibility[best]:
            best = z
    if best is None:
        raise ValueError(f"No region is far from and quiet towards the pair {pair}.")
    return best


@dataclass(frozen=True)
class DistanceStrata:
    """One pair of regions per distance tercile.

    Attributes
    ----------
    pairs : dict of str to (int, int)
        ``"near"``, ``"mid"``, ``"far"`` to the indices of the two regions (the first has the
        lower index).
    bounds : (float, float)
        The terciles of the distances in mm that separate the strata: ``near`` is up to the first,
        ``mid`` above it up to the second, ``far`` above the second.
    """

    pairs: dict[str, tuple[int, int]]
    bounds: tuple[float, float]


def distance_strata(
    visibility: np.ndarray,
    distances: np.ndarray,
    names: Sequence[str],
    *,
    low: float = STRATA_VISIBILITY[0],
    high: float = STRATA_VISIBILITY[1],
) -> DistanceStrata:
    """Pick a near, a mid and a far pair of regions of the same hemisphere.

    Regions with ``low <= visibility <= high`` are eligible, so that no pair is decided by an
    unusually weak or strong region. The distances of all eligible same-hemisphere pairs are split
    at their terciles and from each third the pair whose distance is closest to that third's
    median is taken (the first one in index order when tied). Region names must end in ``-lh`` or
    ``-rh``.

    Raises
    ------
    ValueError
        For a name without hemisphere and when fewer than 3 pairs are eligible.
    """
    hemispheres = []
    for name in names:
        if not name.endswith(("-lh", "-rh")):
            raise ValueError(f"Region name {name!r} does not end in -lh or -rh (no hemisphere).")
        hemispheres.append(name[-2:])
    visibility = np.asarray(visibility, dtype=float)
    eligible = [i for i in range(visibility.size) if low <= visibility[i] <= high]
    pairs = [
        (i, j) for i in eligible for j in eligible if i < j and hemispheres[i] == hemispheres[j]
    ]
    if len(pairs) < 3:
        raise ValueError(
            f"Only {len(pairs)} eligible same-hemisphere pairs; at least 3 are needed."
        )
    dist = np.array([distances[i, j] for i, j in pairs])
    first, second = (float(v) for v in np.quantile(dist, [1.0 / 3.0, 2.0 / 3.0]))
    masks = (dist <= first, (dist > first) & (dist <= second), dist > second)
    chosen = {}
    for label, mask in zip(STRATA_NAMES, masks, strict=True):
        members = np.flatnonzero(mask)
        median = np.median(dist[members])
        best = members[int(np.argmin(np.abs(dist[members] - median)))]
        chosen[label] = pairs[int(best)]
    return DistanceStrata(chosen, (first, second))
