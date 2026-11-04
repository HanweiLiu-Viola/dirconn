"""One entry point for the significance tests of a connectivity estimator."""

from __future__ import annotations

from typing import Any

from dirconn._resampling import Reduction
from dirconn.analytic import (
    AnalyticTestResult,
    AsymptoticTestResult,
    analytic_gc_test,
    analytic_pdc_test,
    analytic_rpdc_test,
)
from dirconn.estimator import Estimator
from dirconn.spectral_estimators import PDC, RPDC
from dirconn.surrogate_test import SurrogateTestResult, surrogate_test
from dirconn.surrogates import (
    AAFT,
    IAAFT,
    BlockShuffle,
    CircularShift,
    PhaseRandomize,
    SurrogateGenerator,
    TrialShuffle,
)
from dirconn.timeseries import TimeSeries

TESTS: tuple[str, ...] = ("surrogate", "block_perm", "analytic")

_NAMED_SURROGATES: dict[str, type[SurrogateGenerator]] = {
    cls.name: cls for cls in (PhaseRandomize, AAFT, IAAFT, CircularShift, TrialShuffle)
}


def significance_test(
    estimator: Estimator,
    data: TimeSeries,
    *,
    test: str,
    surrogate: SurrogateGenerator | str | None = None,
    n_surrogates: int | None = None,
    block_size: int | None = None,
    seed: int | None = None,
    reduce: Reduction = None,
    alpha: float = 0.05,
) -> SurrogateTestResult | AnalyticTestResult | AsymptoticTestResult:
    """Test the significance of an estimator's result.

    There is no default test: which null hypothesis you test is a choice, and each result states
    it (``result.null``) together with its limitations. Arguments that the chosen test does not
    use are refused rather than ignored.

    Parameters
    ----------
    estimator : Estimator
        The connectivity estimator.
    data : TimeSeries
        Input data.
    test : {"surrogate", "block_perm", "analytic"}
        ``"surrogate"``: compare with surrogates from ``surrogate`` (see
        :func:`dirconn.surrogate_test`); needs ``surrogate``, ``n_surrogates`` and ``seed``.
        ``"block_perm"``: block shuffling for continuous data (see
        :class:`dirconn.surrogates.BlockShuffle`); needs ``block_size``, ``n_surrogates`` and
        ``seed``. ``"analytic"``: the test that belongs to the estimator's class, the F test of
        :class:`~dirconn.GC` (:func:`dirconn.analytic_gc_test`) or the asymptotic per-frequency
        tests of :class:`~dirconn.PDC` and :class:`~dirconn.RPDC`
        (:func:`dirconn.analytic_pdc_test`, :func:`dirconn.analytic_rpdc_test`); takes no other
        test arguments.
    surrogate : SurrogateGenerator or str, optional
        A generator instance, or the name of one that needs no parameters: ``"phase_randomize"``,
        ``"aaft"``, ``"iaaft"``, ``"circular_shift"``, ``"trial_shuffle"``.
    n_surrogates : int, optional
        Number of surrogates.
    block_size : int, optional
        Block length in samples for ``"block_perm"``.
    seed : int, optional
        Seed of the random generators; required by the two resampling tests.
    reduce : callable, str or (float, float), optional
        Defines the tested family for the resampling tests (see :func:`dirconn.surrogate_test`).
    alpha : float
        Default level for ``result.significant()`` and the power check.

    Returns
    -------
    SurrogateTestResult, AnalyticTestResult or AsymptoticTestResult
        All offer ``pvalues``, ``null``, ``limitations``, ``adjusted()`` and ``significant()``.
    """
    if test not in TESTS:
        raise ValueError(f"test must be one of {TESTS}, got {test!r}.")
    given: dict[str, Any] = {
        "surrogate": surrogate,
        "n_surrogates": n_surrogates,
        "block_size": block_size,
        "seed": seed,
        "reduce": reduce,
    }
    used = {
        "surrogate": {"surrogate", "n_surrogates", "seed", "reduce"},
        "block_perm": {"block_size", "n_surrogates", "seed", "reduce"},
        "analytic": set(),
    }[test]
    required = {
        "surrogate": ("surrogate", "n_surrogates", "seed"),
        "block_perm": ("block_size", "n_surrogates", "seed"),
        "analytic": (),
    }[test]
    for name, value in given.items():
        if value is not None and name not in used:
            raise ValueError(f"{name} is not used by test={test!r}; remove it.")
    for name in required:
        if given[name] is None:
            raise ValueError(f"test={test!r} needs {name}.")

    if test == "analytic":
        if isinstance(estimator, RPDC):
            return analytic_rpdc_test(estimator, data, alpha=alpha)
        if isinstance(estimator, PDC):
            return analytic_pdc_test(estimator, data, alpha=alpha)
        # any other estimator makes analytic_gc_test raise its explanatory TypeError
        return analytic_gc_test(estimator, data, alpha=alpha)  # type: ignore[arg-type]

    assert n_surrogates is not None and seed is not None
    if test == "block_perm":
        assert block_size is not None
        generator: SurrogateGenerator = BlockShuffle(block_size=block_size)
    else:
        generator = _generator(surrogate)
    return surrogate_test(
        estimator,
        data,
        generator,
        n_surrogates=n_surrogates,
        seed=seed,
        reduce=reduce,
        alpha=alpha,
    )


def _generator(surrogate: SurrogateGenerator | str | None) -> SurrogateGenerator:
    if isinstance(surrogate, SurrogateGenerator):
        return surrogate
    if surrogate == BlockShuffle.name:
        raise ValueError(
            "block_shuffle needs a block size, so it cannot be chosen by name: use "
            "test='block_perm' with block_size=, or pass BlockShuffle(block_size=...)."
        )
    if isinstance(surrogate, str) and surrogate in _NAMED_SURROGATES:
        return _NAMED_SURROGATES[surrogate]()
    raise ValueError(
        f"unknown surrogate {surrogate!r}; use a SurrogateGenerator or one of "
        f"{sorted(_NAMED_SURROGATES)}."
    )
