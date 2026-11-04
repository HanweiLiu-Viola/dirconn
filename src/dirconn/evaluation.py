"""Metrics that score an estimated connectivity matrix against a known ground truth.

All metrics take the estimate and the truth as a square ``[source, target]`` matrix or as a
:class:`~dirconn.Connectivity` whose frequency and time axes have length 1 (reduce a spectral
result first, for example with ``reduce_freq``). The truth is binarised: any non-zero entry is an
edge, NaN counts as no edge. The diagonal is excluded by default, because self-connections are not
estimated.

Metrics that are mathematically undefined for the input (no edges in the truth, all edges, no
unidirectional pair) return NaN with a :class:`~dirconn.DirconnWarning`, never 0: "could not be
computed" must not look like "no better than chance".
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import rankdata

from dirconn.bands import BANDS
from dirconn.exceptions import DirconnWarning
from dirconn.network import Network
from dirconn.result import Connectivity

if TYPE_CHECKING:
    from dirconn.estimator import MethodInfo


def _as_matrix(values: Any, what: str) -> np.ndarray:
    if isinstance(values, Connectivity):
        n_freqs, n_times = values.values.shape[2:]
        if n_freqs != 1 or n_times != 1:
            raise ValueError(
                f"{what} has {n_freqs} frequencies and {n_times} time points; reduce it to one "
                "matrix first, for example with reduce_freq."
            )
        array = values.values[:, :, 0, 0]
    else:
        array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{what} must be real-valued; take the magnitude or reduce it first.")
    array = np.asarray(array, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(
            f"{what} must be a square (source, target) matrix, got shape {array.shape}."
        )
    return array


def _undefined(metric: str, reason: str) -> float:
    warnings.warn(
        f"{metric} is undefined for this input ({reason}); returning NaN.",
        DirconnWarning,
        stacklevel=3,
    )
    return float("nan")


def _cells(truth: Any, exclude_diagonal: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return the candidate-cell mask and the boolean edge matrix of the truth."""
    matrix = _as_matrix(truth, "truth")
    is_edge = np.nan_to_num(matrix, nan=0.0) != 0
    mask = ~np.eye(matrix.shape[0], dtype=bool) if exclude_diagonal else np.ones_like(is_edge)
    return mask, is_edge


def _flatten(scores: Any, truth: Any, exclude_diagonal: bool) -> tuple[np.ndarray, np.ndarray]:
    """Aligned score and edge vectors over the candidate cells; non-finite scores are dropped."""
    score_matrix = _as_matrix(scores, "scores")
    mask, is_edge = _cells(truth, exclude_diagonal)
    if score_matrix.shape != is_edge.shape:
        raise ValueError(
            f"scores {score_matrix.shape} and truth {is_edge.shape} must have the same shape."
        )
    finite = np.isfinite(score_matrix)
    dropped = int(np.sum(mask & ~finite))
    if dropped:
        warnings.warn(
            f"Dropping {dropped} cell(s) with non-finite scores before evaluation.",
            DirconnWarning,
            stacklevel=3,
        )
    keep = mask & finite
    return score_matrix[keep], is_edge[keep]


def auroc(scores: Any, truth: Any, *, exclude_diagonal: bool = True) -> float:
    """Area under the ROC curve: how often a true edge outranks a non-edge.

    0.5 is chance, 1 a perfect ranking, below 0.5 an inverted one; ties count one half. On sparse
    networks it is optimistic, because the many true negatives dominate; report :func:`auprc` too.
    """
    values, is_edge = _flatten(scores, truth, exclude_diagonal)
    n_pos = int(is_edge.sum())
    n_neg = int(is_edge.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return _undefined("auroc", "the truth is all edges or all non-edges")
    ranks = rankdata(values, method="average")
    return float((ranks[is_edge].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(scores: Any, truth: Any, *, exclude_diagonal: bool = True) -> float:
    """Average precision, the area under the precision-recall curve.

    Tied scores form one threshold, so the value does not depend on the order of the cells. Its
    chance level is not 0.5: it is about the edge density (:func:`edge_density`) and, on small
    sparse networks, noticeably higher (:func:`chance_auprc`).
    """
    values, is_edge = _flatten(scores, truth, exclude_diagonal)
    n_pos = int(is_edge.sum())
    if n_pos == 0:
        return _undefined("auprc", "the truth contains no edges")
    order = np.argsort(-values, kind="mergesort")
    sorted_values = values[order]
    hits = is_edge[order]
    true_positives = np.cumsum(hits)
    false_positives = np.cumsum(~hits)
    last_of_each_threshold = np.r_[np.flatnonzero(np.diff(sorted_values)), values.size - 1]
    tp = true_positives[last_of_each_threshold]
    fp = false_positives[last_of_each_threshold]
    precision = tp / (tp + fp)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def edge_density(truth: Any, *, exclude_diagonal: bool = True) -> float:
    """Fraction of candidate cells that are true edges."""
    mask, is_edge = _cells(truth, exclude_diagonal)
    if not mask.any():
        return _undefined("edge_density", "there are no candidate cells")
    return float(is_edge[mask].mean())


def chance_auprc(
    truth: Any,
    *,
    exclude_diagonal: bool = True,
    n_permutations: int = 2000,
    seed: int = 0,
) -> float:
    """Average precision expected from random scores on this truth, by permutation.

    The edge density is only the asymptotic chance level; for 3 edges among 20 cells the real
    baseline is about 0.26, not 0.15, so comparing with the density would overstate a method.
    """
    mask, is_edge = _cells(truth, exclude_diagonal)
    candidates = is_edge[mask]
    n_pos = int(candidates.sum())
    if n_pos == 0:
        return _undefined("chance_auprc", "the truth contains no edges")
    rng = np.random.default_rng(seed)
    positions = np.arange(1, candidates.size + 1, dtype=float)
    total = 0.0
    for _ in range(int(n_permutations)):
        hits = rng.permutation(candidates)
        total += float(np.sum((np.cumsum(hits) / positions)[hits]) / n_pos)
    return total / float(n_permutations)


def direction_accuracy(scores: Any, truth: Any) -> float:
    """Fraction of unidirectional true edges whose direction the scores recover.

    Only pairs that are an edge in exactly one direction are scored; for a bidirectional or an
    absent pair there is no right answer. A tie, or a pair with a non-finite score, is not a hit:
    ties count as misses and non-finite pairs are skipped. Chance is 0.5, so a value near 0.5 means
    no directional information even when :func:`auroc` looks respectable.
    """
    score_matrix = _as_matrix(scores, "scores")
    _, is_edge = _cells(truth, True)
    if score_matrix.shape != is_edge.shape:
        raise ValueError(
            f"scores {score_matrix.shape} and truth {is_edge.shape} must have the same shape."
        )
    correct = total = 0
    for i in range(is_edge.shape[0]):
        for j in range(i + 1, is_edge.shape[0]):
            if is_edge[i, j] == is_edge[j, i]:
                continue
            forward, backward = score_matrix[i, j], score_matrix[j, i]
            if not (np.isfinite(forward) and np.isfinite(backward)):
                continue
            total += 1
            if forward != backward:
                correct += int((forward > backward) == bool(is_edge[i, j]))
    if total == 0:
        return _undefined("direction_accuracy", "there is no unidirectional edge to score")
    return correct / total


def separation(scores: np.ndarray, truth: np.ndarray) -> float:
    """Weakest true edge divided by strongest non-edge, over the off-diagonal cells.

    Above 1 every true edge outscores every non-edge (the ranking is perfect) and the number says
    by how much; below 1 some non-edge, a spurious edge, is stronger than a true one. Unlike
    :func:`dirconn.evaluation.auroc` it shows how close a method is to the limit. NaN when the
    truth has no edge or no non-edge; infinite when every non-edge scores zero. Cells with a
    non-finite score are left out.
    """
    off_diagonal = ~np.eye(scores.shape[0], dtype=bool) & np.isfinite(scores)
    is_edge = np.nan_to_num(truth, nan=0.0) != 0
    true_scores = scores[off_diagonal & is_edge]
    false_scores = scores[off_diagonal & ~is_edge]
    if true_scores.size == 0 or false_scores.size == 0:
        return float("nan")
    strongest_false = float(np.max(false_scores))
    weakest_true = float(np.min(true_scores))
    return math.inf if strongest_false <= 0 else weakest_true / strongest_false


def scoring_band(
    network: Network, default: tuple[float, float] | None = None
) -> tuple[float, float]:
    """Return the band in which a network's edges act, for band-reduced scoring.

    It is the smallest band that contains the bands of all banded edges. Broadband edges do not
    define a band; a network without any banded edge needs an explicit ``default``.
    """
    bands = [edge.band for edge in network.edges if edge.band is not None]
    if not bands:
        if default is None:
            raise ValueError(
                "The network has only broadband edges, so it defines no scoring band; pass "
                "default=(low, high)."
            )
        return (float(default[0]), float(default[1]))
    return (min(b[0] for b in bands), max(b[1] for b in bands))


def _band_edges(band: Any) -> tuple[float, float]:
    if isinstance(band, str):
        key = band.lower()
        if key not in BANDS:
            raise ValueError(f"Unknown band {band!r}; the named bands are {sorted(BANDS)}.")
        low, high = BANDS[key]
    else:
        low, high = (float(b) for b in band)
    return float(low), float(high)


def _truth_by_frequency(truth: Any, result: Connectivity) -> np.ndarray:
    """Return the truth as ``(source, target, freq)``, repeating a matrix over the frequencies."""
    n = len(result.source)
    if isinstance(truth, Connectivity) and truth.is_frequency_resolved:
        if not np.array_equal(truth.freqs, result.freqs):
            raise ValueError(
                "The truth and the result have different frequency axes "
                f"({truth.freqs.shape[0]} and {result.freqs.shape[0]} frequencies)."
            )
        array = truth.values[:, :, :, 0]
    else:
        matrix = _as_matrix(truth, "truth")
        array = np.repeat(matrix[:, :, np.newaxis], result.freqs.shape[0], axis=2)
    if array.shape[:2] != (n, n):
        raise ValueError(f"truth has shape {array.shape[:2]}, expected {(n, n)}.")
    return array


def reduced_scores(
    result: Connectivity, band: tuple[float, float] | str | Sequence[float] | None = None
) -> np.ndarray:
    """Return the magnitudes of a result as one ``[source, target]`` matrix.

    A spectral result is averaged over ``band``, which is then required; a result without a
    frequency axis is returned as it is and refuses a band.
    """
    if result.values.shape[3] != 1:
        raise ValueError("The result needs one time point; reduce it first.")
    if result.is_frequency_resolved:
        if band is None:
            raise ValueError(
                "A spectral result needs band=(low, high) (or a band name) to be reduced to a "
                "matrix; use auroc_by_frequency to look at the whole spectrum."
            )
        reduced = result.reduce_freq(_band_edges(band), how="mean")
        return np.abs(reduced.values[:, :, 0, 0])
    if band is not None:
        raise ValueError("band was given, but the result has no frequency axis to reduce.")
    return np.abs(result.values[:, :, 0, 0])


def is_net(info: MethodInfo | None) -> bool:
    """Whether a method is a directed net flow (antisymmetric), scored against the net truth."""
    return info is not None and info.directed and info.antisymmetric


def has_convention(info: MethodInfo | None) -> bool:
    """Whether a method is scored with a convention of its own (undirected, signed or net)."""
    return info is not None and (info.signed or not info.directed or is_net(info))


def convention_scores(
    result: Connectivity,
    info: MethodInfo | None,
    band: tuple[float, float] | str | Sequence[float] | None = None,
) -> np.ndarray:
    """Return one ``[source, target]`` score matrix that respects the method's sign convention.

    Without ``info``, and for an ordinary measure, this is :func:`reduced_scores` (the magnitude
    after the band mean). An undirected or a signed measure (``MethodInfo.directed`` false or
    ``signed`` true) is turned into its magnitude *before* the band mean, because its sign is a
    phase or the sign of a coupling and would cancel in the mean. A directed net flow
    (``antisymmetric``) keeps its sign: the negative side is the other direction.
    """
    if not has_convention(info):
        return reduced_scores(result, band)
    if result.values.shape[3] != 1:
        raise ValueError("The result needs one time point; reduce it first.")
    signed = is_net(info)
    if result.is_frequency_resolved:
        if band is None:
            raise ValueError("A spectral result needs band=(low, high) (or a band name).")
        source = result if signed else _magnitude(result)
        return source.reduce_freq(_band_edges(band), how="mean").values[:, :, 0, 0]
    if band is not None:
        raise ValueError("band was given, but the result has no frequency axis to reduce.")
    values = result.values[:, :, 0, 0]
    return np.array(values if signed else np.abs(values), dtype=float)


def _magnitude(result: Connectivity) -> Connectivity:
    return Connectivity(
        values=np.abs(result.values),
        source=result.source,
        target=result.target,
        freqs=result.freqs,
        times=result.times,
        provenance=result.provenance.with_step({"op": "abs"}),
    )


def convention_metrics(
    scores: np.ndarray, truth: np.ndarray, info: MethodInfo | None
) -> dict[str, float]:
    """AUROC, AUPRC, direction accuracy and separation under the method's scoring convention.

    Ordinary and signed measures: the directed truth. Undirected measures: the undirected truth
    (an edge in either direction), every pair counted once (the lower triangle is left out), no
    direction accuracy. Directed net flows: a pair with true edges in both directions is left
    out, and a true edge whose score is below zero counts as zero in the separation.
    """
    scores = np.array(scores, dtype=float)
    truth = np.array(truth, dtype=float)
    directed = info is None or info.directed
    net = is_net(info)
    if not directed:
        truth = np.fmax(truth, truth.T)
        scores = np.where(np.tril(np.ones(scores.shape, dtype=bool)), np.nan, scores)
    elif net:
        both = (np.nan_to_num(truth) != 0) & (np.nan_to_num(truth.T) != 0)
        scores = np.where(both, np.nan, scores)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DirconnWarning)
        return {
            "auroc": auroc(scores, truth),
            "auprc": auprc(scores, truth),
            "direction_accuracy": direction_accuracy(scores, truth) if directed else math.nan,
            "separation": separation(np.where(truth == 1, np.maximum(scores, 0.0), scores), truth)
            if net
            else separation(scores, truth),
        }


def score_result(
    result: Connectivity,
    truth: Any,
    *,
    band: tuple[float, float] | str | Sequence[float] | None = None,
    info: MethodInfo | None = None,
) -> dict[str, Any]:
    """Score a connectivity result against the truth, after reducing it to one matrix.

    Magnitudes are scored. A result with a frequency axis is first averaged over ``band``
    (required); an edge of a frequency-resolved truth counts as present when it is present at
    any frequency of the band. Metrics that are undefined for the truth are NaN with a warning.

    Parameters
    ----------
    result : Connectivity
        Time-domain (no frequency axis) or spectral, with one time point.
    truth : ndarray or Connectivity
        A square ``[source, target]`` matrix, or a truth :class:`~dirconn.Connectivity`, with
        the nodes in the same order as ``result``.
    band : (float, float) or str, optional
        Frequency band in Hz or a name from :data:`dirconn.BANDS`; required for a spectral
        result and refused for a result without a frequency axis.
    info : MethodInfo, optional
        The estimator's description. With it, undirected, signed and net-flow measures are scored
        with their conventions (:func:`convention_scores`, :func:`convention_metrics`); an
        ordinary measure, or no ``info``, is scored as before.

    Returns
    -------
    dict
        ``auroc``, ``auprc``, ``direction_accuracy``, ``separation``, ``n_edges`` (true edges,
        off the diagonal) and ``band``.
    """
    scores = convention_scores(result, info, band)
    used_band: tuple[float, float] | None
    if result.is_frequency_resolved:
        low, high = _band_edges(band)
        by_frequency = _truth_by_frequency(truth, result)
        in_band = (result.freqs >= low) & (result.freqs <= high)
        truth_matrix = np.nan_to_num(by_frequency[:, :, in_band], nan=0.0).max(axis=2)
        np.fill_diagonal(truth_matrix, np.nan)
        used_band = (low, high)
    else:
        truth_matrix = _as_matrix(truth, "truth")
        used_band = None
    if scores.shape != truth_matrix.shape:
        raise ValueError(
            f"scores {scores.shape} and truth {truth_matrix.shape} must have the same shape."
        )
    off_diagonal = ~np.eye(len(scores), dtype=bool)
    n_edges = int((np.nan_to_num(truth_matrix, nan=0.0)[off_diagonal] != 0).sum())
    if not has_convention(info):
        metrics = {
            "auroc": auroc(scores, truth_matrix),
            "auprc": auprc(scores, truth_matrix),
            "direction_accuracy": direction_accuracy(scores, truth_matrix),
            "separation": separation(scores, truth_matrix),
        }
    else:
        metrics = convention_metrics(scores, truth_matrix, info)
    return {**metrics, "n_edges": n_edges, "band": used_band}


def auroc_by_frequency(result: Connectivity, truth: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the AUROC at every frequency of a spectral result.

    It answers whether the method sees the right band.

    Magnitudes are scored. Where the truth has no edge, or only edges, at a frequency the AUROC is
    undefined and NaN, without a warning; a banded truth therefore gives NaN outside the band.

    Parameters
    ----------
    result : Connectivity
        Spectral, with one time point.
    truth : ndarray or Connectivity
        A matrix used at every frequency, or a truth :class:`~dirconn.Connectivity` on the same
        frequency axis (for example ``network.truth(result.freqs)``).

    Returns
    -------
    freqs, auroc : ndarray
    """
    if not result.is_frequency_resolved:
        raise ValueError("auroc_by_frequency needs a result with a frequency axis.")
    if result.values.shape[3] != 1:
        raise ValueError("auroc_by_frequency needs a result with one time point.")
    by_frequency = _truth_by_frequency(truth, result)
    magnitudes = np.abs(result.values)[:, :, :, 0]
    off_diagonal = ~np.eye(len(result.source), dtype=bool)
    curve = np.full(result.freqs.shape[0], np.nan)
    for k in range(curve.size):
        truth_k = by_frequency[:, :, k]
        n_edges = int(((np.nan_to_num(truth_k, nan=0.0) != 0) & off_diagonal).sum())
        if n_edges == 0 or n_edges == int(off_diagonal.sum()):
            continue
        curve[k] = auroc(magnitudes[:, :, k], truth_k)
    return result.freqs.copy(), curve


# ----------------------------------------------------------------------------------------------
# Time-resolved evaluation (non-stationary scenarios)
# ----------------------------------------------------------------------------------------------
#
# A continuously updated estimate lags a switch: just after the coupling changes it still carries
# the previous state. Scoring those time points as if the method were wrong would charge the lag to
# the ranking, so the metrics below separate two things: how well the ranking is recovered while the
# estimate has settled (the *stable* segments, outside a guard band after every switch) and how long
# the estimate takes to follow a switch (the *tracking latency*). The two are reported separately
# and never mixed.

_SWITCH_PROBE_S = 1e-6


def _edge_state(network: Network, t: float, pair: tuple[int, int] | None) -> np.ndarray:
    """Boolean ``[source, target]`` matrix of the pairs with an acting edge at time ``t``."""
    index = {name: i for i, name in enumerate(network.names)}
    state = np.zeros((network.n_nodes, network.n_nodes), dtype=bool)
    for edge in network.edges:
        i, j = index[edge.source], index[edge.target]
        if pair is not None and (i, j) != pair:
            continue
        if bool(edge.is_active(t)):
            state[i, j] = True
    return state


def switch_times(
    network: Network, *, pair: tuple[str, str] | None = None, duration: float | None = None
) -> list[float]:
    """Return the times in seconds at which the set of coupled pairs of a network changes.

    The state at time 0 is the initial state, not a switch, and neither is the end of the
    recording; a change of the strength of one pair (an edge ends where another edge of the same
    pair begins) leaves the pair coupled and is not a switch either.

    Parameters
    ----------
    network : Network
        The network, whose edges may have ``active`` intervals.
    pair : (str, str), optional
        ``(source, target)``: only the switches of this pair.
    duration : float, optional
        Length of the recording in seconds; changes at this time or later are not reported.

    Returns
    -------
    list of float
        Ascending, without duplicates.
    """
    index = {name: i for i, name in enumerate(network.names)}
    cell: tuple[int, int] | None = None
    if pair is not None:
        missing = [name for name in pair if name not in index]
        if missing:
            raise ValueError(f"{missing} not in the network; its nodes are {network.names}.")
        cell = (index[pair[0]], index[pair[1]])
    boundaries = sorted(
        {
            limit
            for edge in network.edges
            for interval in (edge.active or ())
            for limit in interval
            if limit > 0.0 and (duration is None or limit < duration)
        }
    )
    out = []
    for b in boundaries:
        before = _edge_state(network, b - _SWITCH_PROBE_S, cell)
        after = _edge_state(network, b + _SWITCH_PROBE_S, cell)
        if (before != after).any():
            out.append(float(b))
    return out


def _truth_matrices(truth: Connectivity, result: Connectivity, band: Any) -> np.ndarray:
    """Return the truth as ``[source, target, time]`` for the band and time axis of ``result``."""
    if not isinstance(truth, Connectivity) or not truth.is_time_resolved:
        raise ValueError(
            "The truth must be a time-resolved Connectivity (network.truth(times=...))."
        )
    if truth.values.shape[3] != result.values.shape[3] or not np.allclose(
        truth.times, result.times, rtol=0.0, atol=1e-9
    ):
        raise ValueError(
            "The truth and the result have different time axes "
            f"({truth.values.shape[3]} and {result.values.shape[3]} time points); build the truth "
            "with network.truth(freqs=result.freqs, times=result.times)."
        )
    values = np.nan_to_num(truth.values, nan=0.0)
    if result.is_frequency_resolved and truth.is_frequency_resolved:
        if not np.array_equal(truth.freqs, result.freqs):
            raise ValueError("The truth and the result have different frequency axes.")
        low, high = _band_edges(band)
        in_band = (result.freqs >= low) & (result.freqs <= high)
        return np.asarray(values[:, :, in_band, :].max(axis=2))
    return np.asarray(values.max(axis=2))


def _scores_over_time(result: Connectivity, band: Any) -> np.ndarray:
    """Return magnitudes as ``[source, target, time]``, averaged over ``band`` if spectral."""
    if not result.is_time_resolved:
        raise ValueError("The result is not time-resolved; there is no time course to score.")
    if result.is_frequency_resolved:
        if band is None:
            raise ValueError(
                "A spectral result needs band=(low, high) (or a band name) to be reduced."
            )
        reduced = result.reduce_freq(_band_edges(band), how="mean")
        return np.asarray(np.abs(reduced.values[:, :, 0, :]))
    if band is not None:
        raise ValueError("band was given, but the result has no frequency axis to reduce.")
    return np.asarray(np.abs(result.values[:, :, 0, :]))


def _defined(truth_matrix: np.ndarray) -> bool:
    off_diagonal = ~np.eye(truth_matrix.shape[0], dtype=bool)
    n_edges = int(((truth_matrix != 0) & off_diagonal).sum())
    return 0 < n_edges < int(off_diagonal.sum())


def truth_switch_times(truth: Connectivity) -> list[float]:
    """Return the times at which the off-diagonal truth of a time-resolved truth changes.

    A switch is reported at the first time point that shows the new state, so it is only as precise
    as the time axis; use :func:`switch_times` on the network when the exact limits are known.
    """
    if not isinstance(truth, Connectivity) or not truth.is_time_resolved:
        raise ValueError("The truth must be a time-resolved Connectivity.")
    state = np.nan_to_num(truth.values, nan=0.0).max(axis=2) != 0  # [source, target, time]
    changed = (state[:, :, 1:] != state[:, :, :-1]).any(axis=(0, 1))
    return [float(t) for t in truth.times[1:][changed]]


def stable_mask(times: Any, switches: Sequence[float], guard_s: float) -> np.ndarray:
    """Return a boolean mask of the time points outside the guard band after every switch.

    A time point is unstable when ``switch <= t < switch + guard_s`` for some switch.
    """
    guard = float(guard_s)
    if not math.isfinite(guard) or guard < 0.0:
        raise ValueError(
            f"guard_s must be a finite non-negative number of seconds, got {guard_s!r}."
        )
    t = np.asarray(times, dtype=float)
    unstable = np.zeros(t.shape, dtype=bool)
    for switch in switches:
        unstable |= (t >= float(switch) - 1e-9) & (t < float(switch) + guard - 1e-9)
    return ~unstable


def guard_seconds(result: Connectivity) -> float:
    """Return the effective memory of the method behind ``result`` in seconds.

    the measured memory of the recursion for :class:`~dirconn.ADTF` and :class:`~dirconn.TPDC`, the
    window length for a windowed :class:`~dirconn.WDTF`. Raises ``ValueError`` for anything else
    (a forgetting factor of 1, segments, a static method): pass ``guard_s`` explicitly then.
    """
    prov = result.provenance
    details = prov.details
    if prov.method in ("adtf", "tpdc"):
        memory = details.get("memory_samples")
        sfreq = details.get("sfreq")
        if memory is not None and sfreq:
            return float(memory) / float(sfreq)
        raise ValueError(
            f"{prov.method} with a forgetting factor of 1 never forgets; pass guard_s explicitly."
        )
    if prov.method == "wdtf" and details.get("window_seconds") is not None:
        return float(details["window_seconds"])
    raise ValueError(
        f"The method {prov.method!r} has no memory to take the guard band from (segments and "
        "static estimators have none); pass guard_s explicitly."
    )


def auroc_by_time(
    result: Connectivity, truth: Connectivity, *, band: Any = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the AUROC at every time point of a time-resolved result.

    Magnitudes are scored, a spectral result is averaged over ``band`` first and an edge of a
    spectral truth counts as present when it is present at any frequency of the band, as in
    :func:`score_result`. Where the truth has no edge, or only edges, at a time point the AUROC is
    undefined and NaN, without a warning.

    Parameters
    ----------
    result : Connectivity
        Time-resolved, for example from :class:`~dirconn.ADTF`.
    truth : Connectivity
        The truth on the same axes, ``network.truth(freqs=result.freqs, times=result.times)``.
    band : (float, float) or str, optional
        Required for a spectral result.

    Returns
    -------
    times, auroc : ndarray
    """
    scores = _scores_over_time(result, band)
    truth_matrix = _truth_matrices(truth, result, band)
    curve = np.full(scores.shape[2], np.nan)
    for k in range(curve.size):
        if _defined(truth_matrix[:, :, k]):
            curve[k] = auroc(scores[:, :, k], truth_matrix[:, :, k])
    return result.times.copy(), curve


def _mean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def score_stable(
    result: Connectivity,
    truth: Connectivity,
    *,
    band: Any = None,
    guard_s: float | None = None,
    switches: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Score a time-resolved result inside its stable segments.

    Every time point outside the guard band after a switch is scored on its own with
    :func:`auroc`, :func:`direction_accuracy` and :func:`separation`, and the values are averaged
    over the time points at which they are defined (an infinite separation stays infinite). The
    whole AUROC curve is returned as well, and the guard band is never used for the latency (see
    :func:`tracking_latency`).

    Parameters
    ----------
    result, truth, band
        As for :func:`auroc_by_time`.
    guard_s : float, optional
        Seconds after each switch that are left out; default :func:`guard_seconds` of the result.
    switches : sequence of float, optional
        Switch times in seconds; default :func:`truth_switch_times` of the truth.

    Returns
    -------
    dict
        ``auroc``, ``direction_accuracy``, ``separation`` (means over the stable time points),
        ``n_stable`` (stable time points), ``n_defined`` (those with a defined AUROC), ``guard_s``,
        ``switches``, ``times``, ``stable`` (the mask), ``auroc_curve`` and ``band``.
    """
    scores = _scores_over_time(result, band)
    truth_matrix = _truth_matrices(truth, result, band)
    guard = guard_seconds(result) if guard_s is None else float(guard_s)
    used = truth_switch_times(truth) if switches is None else [float(s) for s in switches]
    stable = stable_mask(result.times, used, guard)
    curve = np.full(scores.shape[2], np.nan)
    aurocs: list[float] = []
    directions: list[float] = []
    separations: list[float] = []
    for k in range(curve.size):
        matrix = truth_matrix[:, :, k]
        if not _defined(matrix):
            continue
        curve[k] = auroc(scores[:, :, k], matrix)
        if not stable[k]:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DirconnWarning)
            aurocs.append(float(curve[k]))
            directions.append(direction_accuracy(scores[:, :, k], matrix))
            separations.append(separation(scores[:, :, k], matrix))
    if not aurocs:
        warnings.warn(
            "No time point of the stable segments has a defined AUROC (the truth has no edge, or "
            "only edges, there); the scores are NaN.",
            DirconnWarning,
            stacklevel=2,
        )
    return {
        "auroc": _mean(aurocs),
        "direction_accuracy": _mean(directions),
        "separation": _mean(separations),
        "n_stable": int(stable.sum()),
        "n_defined": len(aurocs),
        "guard_s": guard,
        "switches": used,
        "times": result.times.copy(),
        "stable": stable,
        "auroc_curve": curve,
        "band": None if band is None else _band_edges(band),
    }


def latency_from_course(
    times: Any,
    course: Any,
    switch: float,
    *,
    switches: Sequence[float],
    guard_s: float,
    hold_points: int = 1,
) -> dict[str, Any]:
    """Return the tracking latency of a score time course around one switch.

    This is the core of :func:`tracking_latency` on bare arrays, so that it can also be applied to a
    course that is not a single fit, for example the mean over many datasets. The level before the
    switch is the mean score over the stable part of the stretch that ends at the switch, and the
    level after it the mean over the stable part of the stretch that starts there (a stretch runs
    between neighbouring switches; its first ``guard_s`` seconds are left out). The latency is the
    time from the switch until the score has covered half the way from the first level to the
    second and, with ``hold_points`` larger than 1, stayed beyond that point for that many
    consecutive time points. There is no threshold on the score itself. If the two levels do not
    differ, a stable window is empty, or the half way point is never held, ``tracked`` is False and
    the latency NaN, with the ``reason``.

    Parameters
    ----------
    times, course : array_like
        Time points in seconds (ascending) and the score at each.
    switch : float
        The switch time in seconds; one of ``switches``.
    switches : sequence of float
        All switch times of the recording, which bound the stretches.
    guard_s : float
        Seconds after a switch that are left out of the stable windows.
    hold_points : int
        Consecutive time points that must stay beyond the half way point; default 1, the first
        crossing.

    Returns
    -------
    dict
        ``tracked``, ``latency_s``, ``pre_level``, ``post_level``, ``reason``, ``switch``,
        ``guard_s`` and ``hold_points``.
    """
    if isinstance(hold_points, bool) or not isinstance(hold_points, (int, np.integer)):
        raise ValueError(f"hold_points must be a positive integer, got {hold_points!r}.")
    if hold_points < 1:
        raise ValueError(f"hold_points must be a positive integer, got {hold_points!r}.")
    all_switches = sorted(float(s) for s in switches)
    if not any(math.isclose(float(switch), s, abs_tol=1e-9) for s in all_switches):
        raise ValueError(f"switch ({switch}) must be one of switches {all_switches}.")
    switch = float(switch)
    guard = float(guard_s)
    if not math.isfinite(guard) or guard < 0.0:
        raise ValueError(f"guard_s must be a finite non-negative number, got {guard_s!r}.")
    times = np.asarray(times, dtype=float)
    course = np.asarray(course, dtype=float)
    if times.ndim != 1 or times.shape != course.shape:
        raise ValueError("times and course must be one-dimensional and of the same length.")

    earlier = [s for s in all_switches if s < switch - 1e-9]
    later = [s for s in all_switches if s > switch + 1e-9]
    start = (earlier[-1] if earlier else float(times[0])) + guard
    stop = later[0] if later else math.inf

    out: dict[str, Any] = {
        "tracked": False,
        "latency_s": float("nan"),
        "pre_level": float("nan"),
        "post_level": float("nan"),
        "reason": None,
        "switch": switch,
        "guard_s": guard,
        "hold_points": int(hold_points),
    }
    pre = (times >= start - 1e-9) & (times < switch - 1e-9)
    post = (times >= switch + guard - 1e-9) & (times < stop - 1e-9)
    if not pre.any() or not post.any():
        side = "pre-switch" if not pre.any() else "post-switch"
        out["reason"] = (
            f"no stable {side} window: the stretch is shorter than the guard band of "
            f"{guard:g} s (or the time axis does not reach it)."
        )
        return out
    pre_level = float(np.nanmean(course[pre]))
    post_level = float(np.nanmean(course[post]))
    out["pre_level"], out["post_level"] = pre_level, post_level
    delta = post_level - pre_level
    if not math.isfinite(delta) or abs(delta) <= 1e-9 * max(
        abs(pre_level), abs(post_level), 1e-300
    ):
        out["reason"] = "the level of the score does not change across the switch."
        return out

    window = np.flatnonzero((times >= switch - 1e-9) & (times < stop - 1e-9))
    covered = (course[window] - pre_level) / delta >= 0.5
    for k in range(covered.size - hold_points + 1):
        if covered[k : k + hold_points].all():
            out["tracked"] = True
            out["latency_s"] = max(0.0, float(times[window[k]] - switch))
            return out
    out["reason"] = (
        f"the score never stayed beyond half the way between the levels for {hold_points} "
        "consecutive time point(s)."
    )
    return out


def tracking_latency(
    result: Connectivity,
    source: str,
    target: str,
    switch: float,
    *,
    switches: Sequence[float],
    band: Any = None,
    guard_s: float | None = None,
    hold_points: int = 1,
) -> dict[str, Any]:
    """Return how long the score of one edge takes to follow a switch of its coupling.

    The magnitude of the edge's score (averaged over ``band`` for a spectral result) is analysed by
    :func:`latency_from_course`, which defines the levels, the half way point and the ``tracked``
    outcome. A single fit is noisy: a score that is already beyond half the way when the switch
    happens counts as a latency of zero, so for noisy estimators look at the mean course over many
    datasets as well.

    Parameters
    ----------
    result : Connectivity
        Time-resolved, with the edge among its nodes.
    source, target : str
        The edge whose coupling switches.
    switch, switches, hold_points
        As for :func:`latency_from_course`.
    band : (float, float) or str, optional
        Required for a spectral result.
    guard_s : float, optional
        Default :func:`guard_seconds` of the result.

    Returns
    -------
    dict
        The keys of :func:`latency_from_course` and ``edge``.
    """
    for name, names in ((source, result.source), (target, result.target)):
        if name not in names:
            raise ValueError(f"{name!r} is not in the result; its nodes are {tuple(names)}.")
    guard = guard_seconds(result) if guard_s is None else float(guard_s)
    scores = _scores_over_time(result, band)
    course = scores[result.source.index(source), result.target.index(target), :]
    out = latency_from_course(
        result.times, course, switch, switches=switches, guard_s=guard, hold_points=hold_points
    )
    out["edge"] = (source, target)
    return out
