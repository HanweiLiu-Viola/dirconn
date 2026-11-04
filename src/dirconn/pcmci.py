"""PCMCI (Runge et al. 2019) through tigramite, with the partial-correlation test.

PCMCI first selects, for every channel, a small set of lagged parents with a PC-type search, then
tests every lagged link ``i (t - tau) -> j (t)`` with the momentary conditional independence (MCI)
test, conditioning on the parents of both ends. With the partial-correlation test (ParCorr) the
dependence is linear. Only lagged links (``tau >= 1``) are tested; contemporaneous links
(PCMCI+) are not part of this estimator.

tigramite is an optional dependency (``pip install dirconn[pcmci]``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import metadata
from typing import Any

import numpy as np

from dirconn.correction import Correctable, check_alpha
from dirconn.estimator import Estimator, MethodInfo
from dirconn.mvar import check_order_spec, choose_order
from dirconn.result import Connectivity
from dirconn.timeseries import TimeSeries

NULL = (
    "momentary conditional independence (ParCorr): at every lag, the source's past is linearly "
    "independent of the target given the parents that PCMCI selected for both"
)
LIMITATIONS: tuple[str, ...] = (
    "Each cell's p-value is the smallest p-value over the lags 1..tau_max multiplied by the number "
    "of lags (Bonferroni over the lags, at most 1); the lag-resolved p-values are in the "
    "provenance.",
    "ParCorr assumes linear dependence with Gaussian-like residuals; non-linear coupling can be "
    "missed and heavy tails make the p-values wrong in either direction.",
    "The p-values are conditional on the selected parents and on tau_max; the parent selection "
    "(pc_alpha) and a data-driven tau_max are not accounted for.",
    "Only lagged links are tested; a zero-lag (contemporaneous) dependence, such as volume "
    "conduction, is not represented and can distort the lagged links.",
    "Causal sufficiency is assumed: an unrecorded common driver can produce a significant link.",
)


def _require_tigramite() -> tuple[Any, Any, Any]:
    try:
        from tigramite import data_processing
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI as TigramitePCMCI
    except ImportError:
        raise ImportError(
            "PCMCI needs tigramite: install it with `pip install dirconn[pcmci]`."
        ) from None
    return data_processing, ParCorr, TigramitePCMCI


def _as_list(matrix: np.ndarray) -> Any:
    """Return nested lists for the provenance, NaN written as None."""
    if matrix.ndim == 1:
        return [None if math.isnan(v) else float(v) for v in matrix]
    return [_as_list(row) for row in matrix]


@dataclass(frozen=True)
class PCMCI(Estimator):
    """PCMCI with the partial-correlation (ParCorr) test, lagged links only.

    ``values[i, j]`` is the MCI partial correlation of the link ``i (t - tau) -> j (t)`` at the lag
    ``tau`` whose MCI p-value is smallest; it is signed (a negative coupling gives a negative
    value) and lies between -1 and 1. The chosen lags, the lag-resolved values and p-values, and
    the per-cell p-values (smallest over the lags times the number of lags) are in the provenance;
    :func:`pcmci_test` returns those p-values as a test result. The diagonal (a channel's own past)
    is undefined (NaN) here, although PCMCI uses it as a condition. Several trials are analysed
    together with tigramite's multiple-dataset mode. This is predictive dependence given the
    selected conditions, not interventional causality.

    Parameters
    ----------
    tau_max : int or {"aic", "bic", "hqc", "cv"}
        Largest lag tested, in samples. A criterion picks the order of the pooled VAR model of all
        channels (as for :class:`GC`), and ``tau_max`` is that order or ``min_lag_seconds`` in
        samples (rounded up), whichever is larger, at most ``max_order``. Links at longer lags are
        not tested at all, and the VAR order describes the fit of a model, not the longest
        coupling delay: with the BIC order alone a 3-sample delay was missed in the reference
        scenarios (``docs/method_validation.md``, finding 13). An integer is used as given; give
        one that covers the delays you expect when you know them.
    max_order : int
        Largest candidate when ``tau_max`` is a criterion, and the cap of ``tau_max``.
    min_lag_seconds : float
        Shortest delay range that a criterion-based ``tau_max`` must cover, in seconds (default
        30 ms, a common cortico-cortical conduction delay; 0 switches the floor off). More lags
        cost run time and, through the Bonferroni over the lags, some power.
    pc_alpha : float
        Significance level of the parent selection (tigramite's default 0.2).
    """

    info = MethodInfo(
        name="pcmci",
        backend="tigramite PCMCI with ParCorr, lagged links (tau_min=1)",
        directed=True,
        signed=True,
        min_nodes=2,
        value_range=(-1.0, 1.0),
        scale_dependent=False,
        assumptions=("stationarity", "linearity", "causal_sufficiency", "filtering"),
    )

    tau_max: int | str = "bic"
    max_order: int = 10
    pc_alpha: float = 0.2
    min_lag_seconds: float = 0.03

    def __post_init__(self) -> None:
        """Validate the parameters."""
        super().__post_init__()
        check_order_spec(self.tau_max, self.max_order, 0.0, 5)
        if (
            isinstance(self.pc_alpha, bool)
            or not isinstance(self.pc_alpha, (int, float))
            or not 0.0 < self.pc_alpha < 1.0
        ):
            raise ValueError(f"pc_alpha must be strictly between 0 and 1, got {self.pc_alpha!r}.")
        if (
            isinstance(self.min_lag_seconds, bool)
            or not isinstance(self.min_lag_seconds, (int, float))
            or not math.isfinite(self.min_lag_seconds)
            or self.min_lag_seconds < 0
        ):
            raise ValueError(
                f"min_lag_seconds must be a finite number >= 0, got {self.min_lag_seconds!r}."
            )

    def _fit(self, data: TimeSeries) -> Connectivity:
        data_processing, parcorr, tigramite_pcmci = _require_tigramite()
        order, selection = choose_order(
            data, self.tau_max, self.max_order, ridge=0.0, demean=True, n_folds=5
        )
        floor = 0
        if selection is not None:
            floor = math.ceil(float(self.min_lag_seconds) * data.sfreq - 1e-9)
            tau_max = min(max(order, floor), int(self.max_order))
        else:
            tau_max = order
        x = np.asarray(data.data)
        if data.segmentation == "epoched":
            frame = data_processing.DataFrame(
                {k: x[k].T.copy() for k in range(data.n_trials)}, analysis_mode="multiple"
            )
        else:
            frame = data_processing.DataFrame(x[0].T.copy())
        runner = tigramite_pcmci(dataframe=frame, cond_ind_test=parcorr(), verbosity=0)
        out = runner.run_pcmci(tau_min=1, tau_max=tau_max, pc_alpha=float(self.pc_alpha))
        val = np.asarray(out["val_matrix"], dtype=float)[:, :, 1:]  # [source, target, lag 1..]
        pval = np.asarray(out["p_matrix"], dtype=float)[:, :, 1:]
        best = np.argmin(pval, axis=-1)
        rows, cols = np.indices(best.shape)
        values = val[rows, cols, best]
        pvalues = np.minimum(pval[rows, cols, best] * tau_max, 1.0)
        lags = (best + 1).astype(float)
        index = np.arange(data.n_channels)
        for matrix in (values, pvalues, lags):
            matrix[index, index] = np.nan
        provenance = self._provenance(
            data,
            model_order=tau_max,
            details={
                "tau_max": tau_max,
                "var_order": order,
                "min_lag_samples": floor,
                "tau_min": 1,
                "pc_alpha": float(self.pc_alpha),
                "order_selection": None if selection is None else selection.to_dict(),
                "lag": _as_list(lags),
                "pvalues": _as_list(pvalues),
                "val_matrix": _as_list(val),
                "p_matrix": _as_list(pval),
                "tigramite": metadata.version("tigramite"),
            },
        )
        return Connectivity.from_array(values, source=list(data.names), provenance=provenance)


def pcmci(data: TimeSeries, **params: Any) -> Connectivity:
    """PCMCI of ``data``; shorthand for ``PCMCI(**params).fit(data)``."""
    return PCMCI(**params).fit(data)


@dataclass(frozen=True, eq=False)
class PCMCITestResult(Correctable):
    """PCMCI's own MCI p-values, as a test result that the correction code can use.

    Attributes
    ----------
    observed : Connectivity
        The PCMCI result, with its provenance.
    pvalues : ndarray
        Shape ``(source, target, 1, 1)``: the smallest MCI p-value over the lags times the number
        of lags, at most 1; NaN on the diagonal.
    lags : ndarray
        The lag (in samples) of every cell's smallest p-value.
    null : str
        The null hypothesis.
    limitations : tuple of str
        Ways the p-values can be wrong or misleading.
    alpha : float
        Default level for :meth:`significant`.
    """

    observed: Connectivity
    pvalues: np.ndarray
    lags: np.ndarray
    null: str
    limitations: tuple[str, ...]
    alpha: float

    def __post_init__(self) -> None:
        """Make the arrays read-only."""
        for name in ("pvalues", "lags"):
            view = np.asarray(getattr(self, name), dtype=float).view()
            view.flags.writeable = False
            object.__setattr__(self, name, view)

    @property
    def method(self) -> str:
        """Name of the estimator."""
        return self.observed.provenance.method

    @property
    def defined(self) -> np.ndarray:
        """Boolean mask of the cells that have a p-value."""
        return ~np.isnan(self.pvalues)

    def __repr__(self) -> str:
        """Return a summary that includes the null hypothesis."""
        shape = "x".join(str(n) for n in self.pvalues.shape)
        return f"PCMCITestResult(method={self.method!r}, shape={shape}, null={self.null!r})"


def pcmci_test(estimator: PCMCI, data: TimeSeries, *, alpha: float = 0.05) -> PCMCITestResult:
    """Fit PCMCI and return its own MCI p-values as a test result.

    The p-values come from the ParCorr test inside PCMCI, not from resampling; see
    :data:`LIMITATIONS`. Use :meth:`PCMCITestResult.significant` for FDR or Bonferroni
    correction over the channel pairs.

    Raises
    ------
    TypeError
        If ``estimator`` is not a :class:`PCMCI`.
    """
    if not isinstance(estimator, PCMCI):
        raise TypeError(f"pcmci_test needs a PCMCI estimator, got {type(estimator).__name__}.")
    alpha = check_alpha(alpha)
    observed = estimator.fit(data)
    details = observed.provenance.details
    pvalues = np.array(details["pvalues"], dtype=float)[:, :, None, None]
    lags = np.array(details["lag"], dtype=float)[:, :, None, None]
    return PCMCITestResult(
        observed=observed,
        pvalues=pvalues,
        lags=lags,
        null=NULL,
        limitations=LIMITATIONS,
        alpha=alpha,
    )
