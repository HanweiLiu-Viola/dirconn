"""Transfer entropy with the Kraskov-Stoegbauer-Grassberger (KSG) estimator.

Transfer entropy from ``i`` to ``j`` is the conditional mutual information
``TE(i -> j) = I(j_t ; i_past | j_past)``: how much the past of ``i`` tells about the present of
``j`` beyond what the past of ``j`` already tells (Schreiber 2000). It is estimated without a model
with the nearest-neighbour estimator of conditional mutual information (Frenzel and Pompe 2007,
the conditional form of Kraskov et al. 2004), so it also sees non-linear dependence. For jointly
Gaussian data it equals half the Granger causality (Barnett et al. 2009).

The raw estimate is biased (it depends on the number of samples, neighbours and dimensions), so
values are not comparable across pairs with different embeddings or data lengths; use
:func:`dirconn.significance_test` for inference.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from dirconn._greedy import GreedySettings, min_permutations, select_all
from dirconn._ksg import (  # noqa: F401  (re-exported: the estimator used to live here)
    _STRICT,
    _columns,
    _standardise,
    _workers,
    check_device,
    ksg_cmi,
)
from dirconn.correction import adjust_pvalues, check_alpha
from dirconn.estimator import Estimator, MethodInfo
from dirconn.mvar import check_order_spec, choose_order
from dirconn.result import Connectivity
from dirconn.timeseries import TimeSeries


def _embed(
    data: TimeSeries,
    source: int,
    target: int,
    delay: int,
    target_history: int,
    source_history: int,
    others: Sequence[int],
    start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Present of the target, past of the source and conditioning past, pooled over trials.

    Rows never cross a trial boundary: within every trial the rows start at ``start``, the
    longest history needed by any scanned delay, so every delay uses the same rows.
    """
    x = np.asarray(data.data)
    present, source_past, condition = [], [], []
    for trial in x:
        times = np.arange(start, trial.shape[-1])
        present.append(trial[target, times][:, np.newaxis])
        source_past.append(
            np.stack([trial[source, times - delay - m] for m in range(source_history)], axis=1)
        )
        rows = [trial[target, times - m] for m in range(1, target_history + 1)]
        rows += [trial[o, times - m] for o in others for m in range(1, target_history + 1)]
        condition.append(np.stack(rows, axis=1))
    return np.vstack(present), np.vstack(source_past), np.vstack(condition)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


MODES: tuple[str, ...] = ("multivariate", "pairwise", "conditional")

_MULTIVARIATE_ONLY = (
    "alpha",
    "n_perm_max_stat",
    "n_perm_min_stat",
    "n_perm_omnibus",
    "n_perm_max_seq",
    "fdr",
    "seed",
)
_SCAN_ONLY = ("target_history", "source_history")
_PERMUTATIONS = ("n_perm_max_stat", "n_perm_min_stat", "n_perm_omnibus", "n_perm_max_seq")


@dataclass(frozen=True)
class TE(Estimator):
    """Transfer entropy with the KSG estimator: multivariate (default), pairwise or conditional.

    **multivariate** (default): for every target the past variables that carry information
    about its present are selected greedily and tested by permutation, as in IDTxl (Novelli et al.
    2019): first the target's own past among lags ``1 .. max_order``, then the sources' past
    among lags ``1 .. max_delay``, then pruning, an omnibus test per target and a sequential test
    per source (the steps are described in ``dirconn._greedy``). ``values[i, j]`` is the
    information of the selected variables of ``i`` about ``j_t`` given the selected past of ``j``
    and of the other selected sources, in nats, and 0 for a source that is not selected. The tests
    are built in and use ``seed``, so the result is already a decision; running
    :func:`dirconn.significance_test` on top repeats the testing. This mode is much slower than the
    others (thousands of estimates per target); see ``workers`` and ``device``.

    **pairwise**: ``values[i, j] = I(j_t ; i_{t-u}, ..., i_{t-u-l+1} | j_{t-1}, ..., j_{t-k})``
    with ``k = target_history`` and ``l = source_history``, for the delay ``u`` in
    ``1 .. max_delay`` that gives the largest value (Wibral et al. 2013); the chosen delays and the
    values at every delay are in the provenance, and every delay uses the same rows. The estimate
    is raw and biased (the maximum over delays adds about 0.01 to 0.02 nats on non-edges): compare
    values with a surrogate test, not with each other across settings.

    **conditional**: as pairwise, with the past of every other channel (``target_history``
    samples each) in the condition. The nearest-neighbour estimator degrades quickly as the
    dimension of the condition grows, which the multivariate mode avoids by selecting variables.

    Rows are taken within trials and pooled. The diagonal is undefined (NaN). This is predictive
    information transfer, not interventional causality.

    Parameters
    ----------
    mode : {"multivariate", "pairwise", "conditional"}
        See above.
    target_history : int or {"aic", "bic", "hqc", "cv"}
        Pairwise and conditional modes: target history length ``k`` in samples (the condition),
        or the criterion that picks it as the order of the pooled VAR model of all channels (as
        for :class:`GC`), at most ``max_order``. Too short a history makes the source's past carry
        the target's own memory and can reverse the direction: on the oscillating reference
        scenarios a history of one sample got the direction wrong where the BIC order (2) got it
        right. The multivariate mode selects the target past itself;
        setting this there raises.
    max_order : int
        Largest target lag: the largest candidate of the criterion, or of the multivariate
        selection.
    source_history : int
        Pairwise and conditional modes: source history length ``l`` in samples.
    max_delay : int
        Largest source lag, in samples (from 1): the largest delay scanned, or the largest
        candidate of the multivariate selection.
    n_neighbors : int
        Neighbours of the KSG estimator (Kraskov et al. recommend about 4).
    workers : int
        Parallel work (``-1``: all cores); the values do not depend on it. In the pairwise and
        conditional modes, processes for the neighbour searches. In the multivariate mode on the
        CPU path, the targets run in parallel processes when joblib is installed (extra
        ``parallel``), otherwise the neighbour searches are parallel.
    alpha : float
        Multivariate mode: level of every permutation test.
    n_perm_max_stat, n_perm_min_stat, n_perm_omnibus, n_perm_max_seq : int
        Multivariate mode: permutations of the inclusion (maximum statistic), pruning (minimum
        statistic), omnibus and sequential tests; the defaults are IDTxl's. Each must reach
        ``alpha``: at least ``1 / alpha - 1``.
    fdr : bool
        Multivariate mode: correct the targets' omnibus p-values with Benjamini-Hochberg and drop
        the sources of the targets that fail (the hierarchical correction of Novelli et al. 2019).
    seed : int
        Multivariate mode: seed of the permutations; the default makes ``te(data)`` reproducible.
    device : str, optional
        ``None`` (default) uses scipy's ``cKDTree`` on the CPU. A torch device name (``"cpu"``,
        ``"cuda"``, ``"xpu"``, ``"mps"``) computes the neighbours by brute force with PyTorch
        (extra ``gpu``), which pays off for long data on a GPU.
    dtype : {"float64", "float32"}
        Precision on a torch device. float64 gives the values of the CPU path; float32 is faster
        on most GPUs but neighbours that tie at float32 precision change the values slightly.
    """

    info = MethodInfo(
        name="te",
        backend=(
            "own KSG conditional mutual information (Frenzel-Pompe) with scipy cKDTree or brute "
            "force in PyTorch; greedy multivariate selection after IDTxl"
        ),
        directed=True,
        min_nodes=2,
        value_range=(-math.inf, math.inf),
        scale_dependent=False,
        assumptions=("stationarity", "sampling_adequacy", "causal_sufficiency", "filtering"),
    )

    mode: Literal["multivariate", "pairwise", "conditional"] = "multivariate"
    target_history: int | str = "bic"
    max_order: int = 10
    source_history: int = 1
    max_delay: int = 5
    n_neighbors: int = 4
    workers: int = 1
    alpha: float = 0.05
    n_perm_max_stat: int = 200
    n_perm_min_stat: int = 200
    n_perm_omnibus: int = 500
    n_perm_max_seq: int = 500
    fdr: bool = False
    seed: int = 0
    device: str | None = None
    dtype: Literal["float64", "float32"] = "float64"

    def __post_init__(self) -> None:
        """Validate the parameters."""
        super().__post_init__()
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}.")
        check_order_spec(self.target_history, self.max_order, 0.0, 5)
        for name in ("source_history", "max_delay", "n_neighbors"):
            _positive_int(getattr(self, name), name)
        _workers(self.workers)
        check_device(self.device, self.dtype)
        check_alpha(self.alpha)
        for name in _PERMUTATIONS:
            n_perm = _positive_int(getattr(self, name), name)
            if n_perm < min_permutations(self.alpha):
                raise ValueError(
                    f"{name}={n_perm} cannot reach alpha={self.alpha}: the smallest p-value is "
                    f"1 / ({name} + 1); use at least {min_permutations(self.alpha)}."
                )
        if not isinstance(self.fdr, bool):
            raise TypeError(f"fdr must be a bool, got {self.fdr!r}.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise TypeError(f"seed must be an integer, got {self.seed!r}.")
        defaults = {f.name: f.default for f in dataclasses.fields(self)}
        unused = _SCAN_ONLY if self.mode == "multivariate" else _MULTIVARIATE_ONLY
        changed = [name for name in unused if getattr(self, name) != defaults[name]]
        if changed:
            verb = "does" if len(changed) == 1 else "do"
            raise ValueError(f"{', '.join(changed)} {verb} not apply to mode={self.mode!r}.")

    def _fit(self, data: TimeSeries) -> Connectivity:
        if self.mode == "multivariate":
            return self._fit_multivariate(data)
        return self._fit_scan(data)

    def _fit_scan(self, data: TimeSeries) -> Connectivity:
        n = data.n_channels
        conditional = self.mode == "conditional"
        delays = list(range(1, self.max_delay + 1))
        history, selection = choose_order(
            data, self.target_history, self.max_order, ridge=0.0, demean=True, n_folds=5
        )
        start = max(history, self.max_delay + self.source_history - 1)
        if data.n_times - start <= self.n_neighbors + 1:
            raise ValueError(
                f"Trials of {data.n_times} samples leave too few rows after the longest history "
                f"({start} samples)."
            )
        scan = np.full((n, n, len(delays)), np.nan)
        n_rows = 0
        for source in range(n):
            for target in range(n):
                if source == target:
                    continue
                others = [c for c in range(n) if c not in (source, target)]
                others = others if conditional else []
                for d, delay in enumerate(delays):
                    present, source_past, condition = _embed(
                        data,
                        source,
                        target,
                        delay,
                        history,
                        self.source_history,
                        others,
                        start,
                    )
                    scan[source, target, d] = ksg_cmi(
                        present,
                        source_past,
                        condition,
                        k=self.n_neighbors,
                        workers=self.workers,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    n_rows = int(present.shape[0])
        best = np.full((n, n), np.nan)
        values = np.full((n, n), np.nan)
        off = ~np.eye(n, dtype=bool)
        index = np.argmax(np.where(off[:, :, None], scan, -np.inf), axis=-1)
        rows, cols = np.nonzero(off)
        values[rows, cols] = scan[rows, cols, index[rows, cols]]
        best[rows, cols] = np.asarray(delays)[index[rows, cols]]
        provenance = self._provenance(
            data,
            model_order=history,
            details={
                "mode": self.mode,
                "delays": delays,
                "delay": [[None if math.isnan(v) else int(v) for v in row] for row in best],
                "scan": [
                    [[None if math.isnan(v) else float(v) for v in cell] for cell in row]
                    for row in scan
                ],
                "target_history": history,
                "order_selection": None if selection is None else selection.to_dict(),
                "condition_dims": history * (n - 1 if conditional else 1),
                "n_rows": n_rows,
            },
        )
        return Connectivity.from_array(values, source=list(data.names), provenance=provenance)

    def _fit_multivariate(self, data: TimeSeries) -> Connectivity:
        settings = GreedySettings(
            target_lags=tuple(range(1, self.max_order + 1)),
            source_lags=tuple(range(1, self.max_delay + 1)),
            alpha=float(self.alpha),
            n_perm_max_stat=self.n_perm_max_stat,
            n_perm_min_stat=self.n_perm_min_stat,
            n_perm_omnibus=self.n_perm_omnibus,
            n_perm_max_seq=self.n_perm_max_seq,
            k=self.n_neighbors,
            workers=1,
            device=self.device,
            dtype=self.dtype,
        )
        if data.n_times - settings.start <= self.n_neighbors + 1:
            raise ValueError(
                f"Trials of {data.n_times} samples leave too few rows after the longest lag "
                f"({settings.start} samples)."
            )
        n_jobs = 1
        if self.workers != 1 and self.device is None:
            try:
                import joblib  # noqa: F401

                n_jobs = self.workers
            except ImportError:
                settings = dataclasses.replace(settings, workers=self.workers)
        series = np.asarray(data.data, dtype=float)
        results = select_all(series, settings, int(self.seed), n_jobs)
        return self._multivariate_result(data, results)

    def _multivariate_result(self, data: TimeSeries, results: list[dict[str, Any]]) -> Connectivity:
        n = data.n_channels
        if self.fdr:
            omnibus_p = np.array(
                [r["omnibus"]["p"] if r["sources_tested"] else np.nan for r in results]
            )
            adjusted = adjust_pvalues(omnibus_p, "fdr_bh")
            for r, p in zip(results, adjusted, strict=True):
                r["omnibus"]["p_fdr"] = None if math.isnan(p) else float(p)
                if math.isnan(p) or p > self.alpha:
                    r["sources"] = []
        values = np.zeros((n, n))
        pvalues = np.full((n, n), np.nan)
        for j, r in enumerate(results):
            kept = {source for source, _ in r["sources"]}
            for key, p in r["source_p"].items():
                i = int(key)
                pvalues[i, j] = p
                if i in kept:
                    values[i, j] = r["source_te"][key]
        np.fill_diagonal(values, np.nan)
        provenance = self._provenance(
            data,
            seed=int(self.seed),
            details={
                "mode": self.mode,
                "target_lags": list(range(1, self.max_order + 1)),
                "source_lags": list(range(1, self.max_delay + 1)),
                "selected_target_lags": [[lag for _, lag in r["target_past"]] for r in results],
                "selected_sources": [r["sources"] for r in results],
                "p_values": [[None if math.isnan(v) else float(v) for v in row] for row in pvalues],
                "omnibus": [r["omnibus"] for r in results],
                "steps": [r["steps"] for r in results],
                "n_rows": results[0]["n_rows"],
                "device": self.device,
                "dtype": self.dtype,
            },
        )
        return Connectivity.from_array(values, source=list(data.names), provenance=provenance)


def te(data: TimeSeries, **params: Any) -> Connectivity:
    """Transfer entropy of ``data``; shorthand for ``TE(**params).fit(data)``."""
    return TE(**params).fit(data)
