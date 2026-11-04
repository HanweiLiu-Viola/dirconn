"""Surrogate-data significance test for any estimator.

The estimator is applied to the data and, with exactly the same settings, to ``n_surrogates``
surrogates of the data. Each cell's p-value is the rank of its observed magnitude among the
surrogate magnitudes. What a small p-value means is fixed by the surrogate: it rejects
``result.null`` and nothing more, and ``result.limitations`` lists how that can happen without
any coupling between the channels (see ``docs/surrogates.md``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from dirconn._resampling import (
    Reduction,
    make_reducer,
    power_check,
    reissue,
    sort_warnings,
    summary_check,
    without_data_checks,
)
from dirconn.correction import Correctable, check_alpha
from dirconn.estimator import Estimator
from dirconn.provenance import AssumptionChecklist, array_hash
from dirconn.result import Connectivity
from dirconn.rng import RNGController, _as_count
from dirconn.surrogates import SurrogateGenerator
from dirconn.timeseries import TimeSeries


@dataclass(frozen=True, eq=False)
class SurrogateTestResult(Correctable):
    """Outcome of a surrogate test, with the premise it rests on.

    All arrays have the shape of ``observed.values``, ``(source, target, freq, time)``. Cells
    that are undefined in the observed result (for example the diagonal) are NaN in ``pvalues``,
    ``effect`` and ``zscore``, and are not part of any multiple-comparison family.

    Attributes
    ----------
    observed : Connectivity
        The tested result (after ``reduce``), with its provenance.
    pvalues : ndarray
        ``(count(|null| >= |observed|) + 1) / (n_surrogates + 1)``, two-sided on magnitudes. The
        smallest possible value is ``1 / (n_surrogates + 1)``.
    effect : ndarray
        ``|observed|`` minus the mean surrogate magnitude.
    zscore : ndarray
        ``effect`` divided by the standard deviation (ddof 1) of the surrogate magnitudes; NaN
        where that is zero or undefined.
    null : str
        The null hypothesis that the surrogates represent.
    limitations : tuple of str
        Ways that null can be rejected without coupling between the channels.
    generator : dict
        ``name``, ``null`` and ``params`` of the surrogate generator.
    n_surrogates, seed, alpha
        How the test was run. ``alpha`` only enters the power check.
    checks : AssumptionChecklist
        Diagnostics of the test itself: ``surrogate_power``, ``surrogate_generation`` and
        ``surrogate_fits``.
    """

    observed: Connectivity
    pvalues: np.ndarray
    effect: np.ndarray
    zscore: np.ndarray
    null: str
    limitations: tuple[str, ...]
    generator: dict[str, Any]
    n_surrogates: int
    seed: int
    alpha: float
    checks: AssumptionChecklist

    def __post_init__(self) -> None:
        """Make the arrays read-only."""
        for name in ("pvalues", "effect", "zscore"):
            view = np.asarray(getattr(self, name), dtype=float).view()
            view.flags.writeable = False
            object.__setattr__(self, name, view)

    @property
    def method(self) -> str:
        """Name of the estimator that was tested."""
        return self.observed.provenance.method

    @property
    def defined(self) -> np.ndarray:
        """Boolean mask of the cells that have a p-value."""
        return ~np.isnan(self.observed.values)

    def __repr__(self) -> str:
        """Return a summary that includes the null hypothesis."""
        shape = "x".join(str(n) for n in self.pvalues.shape)
        return (
            f"SurrogateTestResult(method={self.method!r}, shape={shape}, "
            f"surrogate={self.generator['name']!r}, n_surrogates={self.n_surrogates}, "
            f"null={self.null!r})"
        )


def surrogate_test(
    estimator: Estimator,
    data: TimeSeries,
    generator: SurrogateGenerator,
    *,
    n_surrogates: int,
    seed: int,
    reduce: Reduction = None,
    alpha: float = 0.05,
) -> SurrogateTestResult:
    """Test an estimator's result against surrogates of the data.

    The whole pipeline of ``estimator`` is repeated on every surrogate, including a model order
    chosen by a criterion, so that the null distribution reflects the same procedure as the
    observed value.

    Parameters
    ----------
    estimator : Estimator
        Any dirconn estimator.
    data : TimeSeries
        Input data.
    generator : SurrogateGenerator
        Defines the null hypothesis; see :mod:`dirconn.surrogates`.
    n_surrogates : int
        Number of surrogates. The smallest p-value is ``1 / (n_surrogates + 1)``.
    seed : int
        Required. Surrogate ``i`` uses ``RNGController(seed).child(i)``.
    reduce : callable, str or (float, float), optional
        Applied to the result of every fit before testing; this defines the family of tests.
        A callable maps a :class:`~dirconn.Connectivity` to another one, a band name or a pair
        of frequencies reduces the frequency axis with the mean (:meth:`Connectivity.reduce_freq`).
    alpha : float
        Significance level used for the power check: a Bonferroni-type family of ``m`` defined
        cells needs ``n_surrogates >= m / alpha - 1``, otherwise no cell can reach ``alpha / m``.

    Returns
    -------
    SurrogateTestResult

    Raises
    ------
    ValueError
        If a surrogate fit leaves a cell undefined that is defined on the data (a p-value there
        would be meaningless), or for invalid arguments.
    """
    if not isinstance(estimator, Estimator):
        raise TypeError(f"estimator must be an Estimator, got {type(estimator).__name__}.")
    if not isinstance(generator, SurrogateGenerator):
        raise TypeError(f"generator must be a SurrogateGenerator, got {type(generator).__name__}.")
    if not isinstance(data, TimeSeries):
        raise TypeError(f"data must be a TimeSeries, got {type(data).__name__}.")
    n_surrogates = _as_count(n_surrogates, "n_surrogates", minimum=1)
    controller = RNGController(seed)
    alpha = check_alpha(alpha)
    reducer = make_reducer(reduce)

    # A generator that re-segments the record also defines what the observed value is fitted on.
    original = data
    data = generator.prepare(original)
    observed = reducer(estimator.fit(data))
    if data is not original:
        # the fit saw the prepared data, so keep a way back to the user's own input
        observed = observed.with_provenance(
            observed.provenance.with_step(
                {
                    "step": "prepare",
                    "generator": generator.name,
                    "params": generator.params(),
                    "original_input_hash": array_hash(original.data),
                }
            )
        )
    magnitude = np.abs(observed.values)
    defined = ~np.isnan(magnitude)
    n_defined = int(defined.sum())

    exceed = np.zeros(magnitude.shape)
    mean = np.zeros(magnitude.shape)
    m2 = np.zeros(magnitude.shape)
    # One entry per surrogate that raised at least one diagnostic (the first message).
    generation_notes: list[str] = []
    fit_notes: list[str] = []
    other: dict[tuple[type[Warning], str], warnings.WarningMessage] = {}

    light = without_data_checks(estimator)  # the model checks belong to the fit on the data
    for i in range(n_surrogates):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            surrogate = generator.generate_one(data, controller.child(i))
        generation_notes.extend(sort_warnings(caught, other)[:1])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = reducer(light.fit(surrogate))
        fit_notes.extend(sort_warnings(caught, other)[:1])

        if result.values.shape != magnitude.shape:
            raise ValueError(
                f"surrogate {i} gave a result of shape {result.values.shape}, "
                f"expected {magnitude.shape}."
            )
        null = np.abs(result.values)
        lost = defined & np.isnan(null)
        if lost.any():
            raise ValueError(
                f"surrogate {i} left {int(lost.sum())} cell(s) undefined that are defined on "
                "the data; refusing to compute p-values from a degraded null distribution."
            )
        with np.errstate(invalid="ignore"):
            exceed += np.where(defined, null >= magnitude, 0)
        delta = np.where(defined, null - mean, 0.0)
        mean += delta / (i + 1)
        m2 += delta * (np.where(defined, null, 0.0) - mean)

    reissue(other)

    pvalues = np.where(defined, (exceed + 1.0) / (n_surrogates + 1.0), np.nan)
    effect = np.where(defined, magnitude - mean, np.nan)
    if n_surrogates > 1:
        std = np.sqrt(m2 / (n_surrogates - 1))
    else:
        std = np.zeros(magnitude.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = np.where(defined & (std > 0), effect / std, np.nan)

    checks = AssumptionChecklist(
        (
            power_check(
                n_defined, n_surrogates, alpha, check="surrogate_power", argument="n_surrogates"
            ),
            summary_check(
                "surrogate_generation",
                generation_notes,
                n_surrogates,
                f"the {generator.name} generator",
            ),
            summary_check(
                "surrogate_fits",
                fit_notes,
                n_surrogates,
                f"the {estimator.info.name} fits on surrogates",
            ),
        )
    )
    return SurrogateTestResult(
        observed=observed,
        pvalues=pvalues,
        effect=effect,
        zscore=zscore,
        null=generator.null,
        limitations=generator.limitations,
        generator=generator.describe(),
        n_surrogates=n_surrogates,
        seed=controller.seed,
        alpha=alpha,
        checks=checks,
    )
