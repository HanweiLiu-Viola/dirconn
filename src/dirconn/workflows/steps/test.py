"""Significance tests of an estimate, one per band on shared surrogates, with correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dirconn.data.timeseries import TimeSeries
from dirconn.methods.base import Estimator
from dirconn.stats.correction import adjust_pvalues
from dirconn.stats.significance import significance_test, significance_tests
from dirconn.stats.surrogate_test import SurrogateTestResult
from dirconn.workflows.config import StatisticsConfig
from dirconn.workflows.steps.estimate import Estimate

__all__ = ["BUILTIN_TESTS", "BandTest", "test_significance", "untested_reason"]

#: Methods whose result carries p-values of their own, used instead of a surrogate test:
#: PCMCI's MCI p-values and the permutation p-values of multivariate TE.
BUILTIN_TESTS: tuple[str, ...] = ("pcmci", "te")


@dataclass(frozen=True)
class BandTest:
    """Significance test of one band (or of a method without frequency resolution).

    Parameters
    ----------
    band : str or None
        Band name, ``None`` for methods that are not frequency resolved.
    pvalues : ndarray
        Raw p-values, NaN where undefined (the diagonal).
    adjusted : ndarray
        Corrected p-values, same shape.
    significant : ndarray of bool
        ``adjusted < alpha``; False where NaN.
    test : str
        ``"surrogate"`` (any surrogate generator, block shuffling included), ``"analytic"``, or
        ``"builtin"`` (the p-values of the method itself, `BUILTIN_TESTS`).
    n_surrogates : int or None
        Number of surrogates, ``None`` for the analytic and built-in tests.
    seed : int or None
        Seed of this test, ``None`` for the analytic and built-in tests.
    generator : dict
        The surrogate generator as recorded by the test (``name``, ``null``, ``params`` with the
        block size for block shuffling); ``{"name": "analytic"}`` for the analytic test and
        ``{"name": "builtin", "source": ...}`` for a built-in one.
    observed_history : tuple of dict
        The data history the observed estimate was fitted on, including the
        ``block_shuffle_blocks`` step with its block counts for block shuffling.
    """

    band: str | None
    pvalues: np.ndarray
    adjusted: np.ndarray
    significant: np.ndarray
    test: str
    n_surrogates: int | None
    seed: int | None
    generator: dict[str, Any]
    observed_history: tuple[dict[str, Any], ...]


def _band_limits(est: Estimate, name: str) -> tuple[float, float]:
    """Read the band limits back from the reduction recorded in the provenance."""
    for step in reversed(est.bands[name].provenance.history):
        if step.get("op") == "reduce_freq":
            lo, hi = step["band"]
            return float(lo), float(hi)
    raise ValueError(f"band {name!r} carries no reduce_freq step in its provenance.")


def _band_test(
    band: str | None,
    raw: Any,
    cfg: StatisticsConfig,
    n_surrogates: int | None,
    seed: int | None,
    generator: dict[str, Any],
    observed_history: Any,
    test: str | None = None,
) -> BandTest:
    pvalues = np.asarray(raw, dtype=float)
    if pvalues.ndim == 4 and pvalues.shape[2:] == (1, 1):
        pvalues = pvalues[:, :, 0, 0]
    adjusted = adjust_pvalues(pvalues, cfg.correction)
    with np.errstate(invalid="ignore"):
        significant = np.where(np.isfinite(adjusted), adjusted < cfg.alpha, False)
    history = tuple(dict(step) for step in observed_history)
    return BandTest(
        band,
        pvalues,
        adjusted,
        significant,
        cfg.test if test is None else test,
        n_surrogates,
        seed,
        generator,
        history,
    )


def _history(result: Any) -> Any:
    return result.observed.provenance.preprocessing.get("data_history", [])


def _builtin_pvalues(est: Estimate) -> tuple[np.ndarray, str] | None:
    """Return the method's own p-values and their source, or ``None`` if it has none."""
    details = est.full.provenance.details
    if est.method == "pcmci":
        source = "PCMCI MCI p-values (ParCorr), smallest over the lags times the number of lags"
        return np.array(details["pvalues"], dtype=float), source
    if est.method == "te" and details.get("mode") == "multivariate":
        source = "multivariate TE permutation p-values of each source (greedy selection)"
        return np.array(details["p_values"], dtype=float), source
    return None


def _surrogate(ts: TimeSeries, cfg: StatisticsConfig) -> str:
    if cfg.surrogate == "auto":
        return "block_shuffle" if ts.segmentation == "continuous" else "trial_shuffle"
    return cfg.surrogate


def untested_reason(ts: TimeSeries, est: Estimate, cfg: StatisticsConfig) -> str | None:
    """Why `test_significance` does not test ``est``, or ``None`` when it does.

    A time-resolved result (adtf, tpdc, wdtf) is not tested with block shuffling: the shuffled
    record is re-segmented into blocks, so the observed value of the test is fitted on blocks
    and its p-values belong to no time point of the result. On a continuous record this is the
    ``"auto"`` surrogate, and the result is then descriptive. Epoched records (trial shuffling)
    are tested per time point.
    """
    if cfg.test != "surrogate" or est.full.values.shape[3] == 1:
        return None
    if est.method in BUILTIN_TESTS or _surrogate(ts, cfg) != "block_shuffle":
        return None
    return (
        "not tested: block shuffling fits the observed value on blocks of the record, so its "
        "p-values belong to no time point of a time-resolved result; the result is descriptive"
    )


def test_significance(
    ts: TimeSeries,
    estimator: Estimator,
    est: Estimate,
    cfg: StatisticsConfig,
    *,
    seed: int,
) -> tuple[BandTest, ...]:
    """Test an estimate and correct for multiple comparisons.

    Parameters
    ----------
    ts : TimeSeries
        The record the estimate came from.
    estimator : Estimator
        The estimator that produced ``est``.
    est : Estimate
        Supplies the band names and limits.
    cfg : StatisticsConfig
        Test kind, surrogate (``"auto"`` is block shuffling for continuous records and trial
        shuffling for epoched ones), block length, number of surrogates, level and correction.
    seed : int
        Seed of the surrogates. All bands share them: each surrogate is fitted once and reduced
        to every band (:func:`dirconn.stats.significance.significance_tests`).

    Returns
    -------
    tuple of BandTest
        Empty for ``test = "none"`` and when `untested_reason` gives a reason. The methods of
        `BUILTIN_TESTS` (te in its multivariate mode only) give a single ``band=None`` entry
        with their own p-values for ``"surrogate"`` and ``"analytic"``; no surrogate is
        fitted. The surrogate test gives one entry per band (a single ``band=None`` entry for
        methods that are not frequency resolved). The analytic test does not depend on a band
        and gives a single ``band=None`` entry with the estimator's native p-value shape;
        estimators without an analytic test let their exception propagate.
    """
    if cfg.test == "none" or untested_reason(ts, est, cfg) is not None:
        return ()
    builtin = _builtin_pvalues(est)
    if builtin is not None:
        pvalues, source = builtin
        generator = {"name": "builtin", "source": source}
        history = est.full.provenance.preprocessing.get("data_history", [])
        return (_band_test(None, pvalues, cfg, None, None, generator, history, "builtin"),)
    if cfg.test == "analytic":
        result = significance_test(estimator, ts, test="analytic", alpha=cfg.alpha)
        generator = {"name": "analytic"}
        return (_band_test(None, result.pvalues, cfg, None, None, generator, _history(result)),)
    surrogate = _surrogate(ts, cfg)
    if surrogate == "block_shuffle":
        kind: dict[str, Any] = {
            "test": "block_perm",
            "block_size": int(round(cfg.block_seconds * ts.sfreq)),
        }
    else:
        kind = {"test": "surrogate", "surrogate": surrogate}
    names: list[str | None] = list(est.bands) if est.bands else [None]
    reduces = {(name or ""): (None if name is None else _band_limits(est, name)) for name in names}
    results = significance_tests(
        estimator,
        ts,
        reduces=reduces,
        n_surrogates=cfg.n_surrogates,
        seed=seed,
        alpha=cfg.alpha,
        **kind,
    )
    out = []
    for name in names:
        result = results[name or ""]
        assert isinstance(result, SurrogateTestResult)
        out.append(
            _band_test(
                name,
                result.pvalues,
                cfg,
                cfg.n_surrogates,
                seed,
                dict(result.generator),
                _history(result),
            )
        )
    return tuple(out)
