"""Greedy selection of past variables with permutation tests, for multivariate transfer entropy.

For one target ``j`` the candidates are past samples ``(channel, lag)``: the target's own past at
``target_lags`` and every other channel at ``source_lags``. The selection follows the
multivariate transfer entropy of IDTxl (Lizier and Rubinov 2012; Novelli et al. 2019):

1. **Target past.** Add the candidate with the largest ``I(candidate; j_t | selected)`` while a
   maximum-statistic permutation test finds it significant. When none is, the most recent target
   sample is kept so the target's own memory is always conditioned on.
2. **Sources.** The same greedy loop over the source candidates, conditioning on everything
   selected so far.
3. **Pruning.** Remove the selected source variable with the smallest
   ``I(variable; j_t | all other selected)`` while a minimum-statistic test finds it not
   significant.
4. **Omnibus test.** ``I(all selected source variables; j_t | selected target past)`` against
   permutations of the whole source set; when it fails, no source is kept.
5. **Sequential test per source.** For every source channel, the information of its selected
   variables given everything else selected, tested in decreasing order against the ranked
   maxima of the permutations (step-down, so the p-values never decrease along the order).

A permutation replaces the tested variables by their values at permuted rows. With enough trials
(``n_trials! >= n_permutations``) whole trials are permuted, otherwise every trial is shifted
circularly by an amount drawn between 10 % and 90 % of its rows. One permutation is shared by all
candidates of a test, so a maximum (or minimum) over candidates is a maximum over one surrogate
dataset. p-values are ``(count(null >= observed) + 1) / (n_permutations + 1)``; a test passes at
``p <= alpha``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from dirconn._ksg import ksg_cmi_many


@dataclass(frozen=True)
class GreedySettings:
    """Everything the selection of one target needs, picklable for joblib."""

    target_lags: tuple[int, ...]
    source_lags: tuple[int, ...]
    alpha: float
    n_perm_max_stat: int
    n_perm_min_stat: int
    n_perm_omnibus: int
    n_perm_max_seq: int
    k: int
    workers: int
    device: str | None
    dtype: str

    @property
    def start(self) -> int:
        """First usable sample of every trial: the longest lag of any candidate."""
        return max(max(self.target_lags), max(self.source_lags))


Variable = tuple[int, int]
"""A past sample: ``(channel, lag)``."""


class _Problem:
    """The embedded data of one target and the CMI and permutation helpers."""

    def __init__(
        self, series: np.ndarray, target: int, settings: GreedySettings, rng: np.random.Generator
    ) -> None:
        self.series = series
        self.settings = settings
        self.rng = rng
        self.n_trials = series.shape[0]
        self.rows = series.shape[-1] - settings.start
        self.present = self.column((target, 0))
        self._cache: dict[Variable, np.ndarray] = {}

    def column(self, variable: Variable) -> np.ndarray:
        channel, lag = variable
        start, n_times = self.settings.start, self.series.shape[-1]
        return self.series[:, channel, start - lag : n_times - lag].reshape(-1)

    def values(self, variables: Sequence[Variable]) -> np.ndarray | None:
        if not variables:
            return None
        out = []
        for v in variables:
            if v not in self._cache:
                self._cache[v] = self.column(v)
            out.append(self._cache[v])
        return np.stack(out, axis=1)

    def cmi(self, xs: Sequence[np.ndarray], condition: Sequence[Variable]) -> np.ndarray:
        s = self.settings
        return ksg_cmi_many(
            xs,
            self.present,
            self.values(condition),
            k=s.k,
            workers=s.workers,
            device=s.device,
            dtype=s.dtype,
        )

    def permutations(self, n_perm: int) -> list[np.ndarray]:
        rows, n_trials = self.rows, self.n_trials
        offsets = np.arange(n_trials)[:, None] * rows
        within = np.arange(rows)[None, :]
        out = []
        by_trial = n_trials > 1 and _factorial_at_least(n_trials, n_perm)
        low, high = max(1, rows // 10), max(2, rows - rows // 10)
        for _ in range(n_perm):
            if by_trial:
                order = self.rng.permutation(n_trials)[:, None]
                out.append((order * rows + within).reshape(-1))
            else:
                shift = self.rng.integers(low, high, size=n_trials)[:, None]
                out.append((offsets + (within + shift) % rows).reshape(-1))
        return out

    def scheme(self, n_perm: int) -> str:
        if self.n_trials > 1 and _factorial_at_least(self.n_trials, n_perm):
            return "trials"
        return "circular"


def _factorial_at_least(n: int, bound: int) -> bool:
    value = 1
    for i in range(2, n + 1):
        value *= i
        if value >= bound:
            return True
    return value >= bound


def _p_value(null: np.ndarray, observed: float) -> float:
    return float((np.count_nonzero(null >= observed) + 1) / (null.size + 1))


def _include(
    problem: _Problem, candidates: list[Variable], selected: list[Variable]
) -> tuple[list[Variable], list[dict[str, Any]]]:
    """Greedy inclusion with the maximum-statistic test; returns the added variables."""
    s = problem.settings
    added: list[Variable] = []
    steps: list[dict[str, Any]] = []
    remaining = list(candidates)
    while remaining:
        condition = selected + added
        columns = problem.values(remaining)
        assert columns is not None
        stats = problem.cmi(list(columns.T), condition)
        best = int(np.argmax(stats))
        perms = problem.permutations(s.n_perm_max_stat)
        null = np.full(len(perms), -np.inf)
        for c in range(len(remaining)):
            surrogate = problem.cmi([columns[p, c] for p in perms], condition)
            null = np.maximum(null, surrogate)
        p = _p_value(null, float(stats[best]))
        steps.append({"variable": list(remaining[best]), "cmi": float(stats[best]), "p": p})
        if p > s.alpha:
            break
        added.append(remaining.pop(best))
    return added, steps


def _prune(
    problem: _Problem, target_past: list[Variable], sources: list[Variable]
) -> tuple[list[Variable], list[dict[str, Any]]]:
    """Remove source variables with the minimum-statistic test."""
    s = problem.settings
    kept = list(sources)
    steps: list[dict[str, Any]] = []
    while kept:
        perms = problem.permutations(s.n_perm_min_stat)
        stats = np.empty(len(kept))
        null = np.full(len(perms), np.inf)
        for i, v in enumerate(kept):
            condition = target_past + [u for u in kept if u != v]
            column = problem.values([v])
            assert column is not None
            stats[i] = problem.cmi([column], condition)[0]
            null = np.minimum(null, problem.cmi([column[p] for p in perms], condition))
        worst = int(np.argmin(stats))
        p = _p_value(null, float(stats[worst]))
        steps.append({"variable": list(kept[worst]), "cmi": float(stats[worst]), "p": p})
        if p <= s.alpha:
            break
        kept.pop(worst)
    return kept, steps


def _omnibus(
    problem: _Problem, target_past: list[Variable], sources: list[Variable]
) -> dict[str, Any]:
    if not sources:
        return {"te": 0.0, "p": 1.0, "significant": False}
    s = problem.settings
    joint = problem.values(sources)
    assert joint is not None
    observed = float(problem.cmi([joint], target_past)[0])
    perms = problem.permutations(s.n_perm_omnibus)
    null = problem.cmi([joint[p] for p in perms], target_past)
    p = _p_value(null, observed)
    return {"te": observed, "p": p, "significant": p <= s.alpha}


def _sequential(
    problem: _Problem, target_past: list[Variable], sources: list[Variable]
) -> dict[int, tuple[float, float]]:
    """Per source channel: (TE of its selected variables, step-down p-value)."""
    s = problem.settings
    channels = sorted({c for c, _ in sources})
    perms = problem.permutations(s.n_perm_max_seq)
    stats = np.empty(len(channels))
    table = np.empty((len(channels), len(perms)))
    for i, channel in enumerate(channels):
        group = [v for v in sources if v[0] == channel]
        condition = target_past + [v for v in sources if v[0] != channel]
        joint = problem.values(group)
        assert joint is not None
        stats[i] = problem.cmi([joint], condition)[0]
        table[i] = problem.cmi([joint[p] for p in perms], condition)
    order = np.argsort(-stats, kind="stable").tolist()
    ranked_null = -np.sort(-table, axis=0)  # every permutation's values in decreasing order
    out: dict[int, tuple[float, float]] = {}
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, _p_value(ranked_null[rank], float(stats[i])))
        out[channels[i]] = (float(stats[i]), running)
    return out


def select_target(
    series: np.ndarray, target: int, settings: GreedySettings, seed: np.random.SeedSequence
) -> dict[str, Any]:
    """Run the five steps for one target; the result is JSON-able."""
    problem = _Problem(series, target, settings, np.random.default_rng(seed))
    n_channels = series.shape[1]
    target_candidates = [(target, lag) for lag in settings.target_lags]
    source_candidates = [
        (c, lag) for c in range(n_channels) if c != target for lag in settings.source_lags
    ]
    target_past, target_steps = _include(problem, target_candidates, [])
    if not target_past:
        target_past = [(target, min(settings.target_lags))]
    sources, source_steps = _include(problem, source_candidates, target_past)
    sources, prune_steps = _prune(problem, target_past, sources)
    omnibus = _omnibus(problem, target_past, sources)
    per_source = _sequential(problem, target_past, sources) if omnibus["significant"] else {}
    kept = [v for v in sources if v[0] in per_source and per_source[v[0]][1] <= settings.alpha]
    return {
        "target_past": [list(v) for v in sorted(target_past, key=lambda v: v[1])],
        "sources_tested": [list(v) for v in sources],
        "sources": [list(v) for v in sorted(kept)],
        "source_te": {str(c): te for c, (te, _) in per_source.items()},
        "source_p": {str(c): p for c, (_, p) in per_source.items()},
        "omnibus": omnibus,
        "steps": {"target": target_steps, "sources": source_steps, "prune": prune_steps},
        "n_rows": int(problem.present.size),
    }


def select_all(
    series: np.ndarray, settings: GreedySettings, seed: int, n_jobs: int
) -> list[dict[str, Any]]:
    """Run :func:`select_target` for every target, in parallel over targets when ``n_jobs != 1``.

    Each target gets its own child of ``SeedSequence(seed)``, so the result does not depend on
    ``n_jobs``.
    """
    n_channels = series.shape[1]
    seeds = np.random.SeedSequence(seed).spawn(n_channels)
    if n_jobs != 1:
        from joblib import Parallel, delayed

        return list(
            Parallel(n_jobs=n_jobs)(
                delayed(select_target)(series, j, settings, seeds[j]) for j in range(n_channels)
            )
        )
    return [select_target(series, j, settings, seeds[j]) for j in range(n_channels)]


def min_permutations(alpha: float) -> int:
    """Smallest number of permutations whose smallest p-value reaches ``alpha``."""
    return max(1, math.ceil(1.0 / alpha - 1.0 - 1e-12))
