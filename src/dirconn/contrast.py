"""Permutation test for a difference in connectivity between two conditions.

The estimator is applied to each condition and the difference of the magnitudes is the test
statistic. Under the null the condition label carries no information, so the pooled trials are
relabelled at random, keeping the group sizes, and the whole procedure is repeated. This tests a
difference between *estimates*; it does not say which process produced it (see
``ContrastResult.limitations``). Baseline normalisation is a separate step, see
:func:`dirconn.baseline_normalise`.

With ``paired=True`` the two data sets are two segments of the *same* trials (for example a
baseline and a stimulus period cut from every epoch). The trials are then not exchangeable between
the segments, only the two segments of one trial are: every relabelling swaps the segments of each
trial independently with probability 1/2.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

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
from dirconn.exceptions import DirconnWarning
from dirconn.provenance import AssumptionCheck, AssumptionChecklist
from dirconn.result import Connectivity
from dirconn.rng import RNGController, _as_count
from dirconn.timeseries import TimeSeries

NULL = "connectivity is the same in both conditions (trials are exchangeable between conditions)"
NULL_PAIRED = (
    "connectivity is the same in both segments (the two segments of a trial are exchangeable: "
    "their labels are swapped independently within each trial)"
)
LIMITATIONS: tuple[str, ...] = (
    "Connectivity estimates depend on signal power and noise level, so a difference in "
    "amplitude, signal-to-noise ratio or noise structure between the conditions can reject the "
    "null without any change in coupling.",
    "Trials are treated as independent and exchangeable: a drift, learning or order effect "
    "across trials makes the conditions differ for reasons unrelated to the condition.",
    "The statistic is the difference of magnitudes per cell; a significant cell says the "
    "estimates differ, not which underlying process changed.",
)
PAIRED_LIMITATIONS: tuple[str, ...] = (
    "Connectivity estimates depend on signal power and noise level, so a difference in "
    "amplitude, signal-to-noise ratio or noise structure between the two segments can reject the "
    "null without any change in coupling.",
    "The segments of a trial are assumed to differ only through the labels under the null: a "
    "time-order effect inside a trial (adaptation, fatigue, drift, a slow trend) makes the later "
    "segment differ from the earlier one for reasons unrelated to coupling, and this test cannot "
    "tell the two apart.",
    "Trials are treated as independent of each other; the pairing only removes what the two "
    "segments of a trial share, such as a trial's own coupling strength or noise level.",
    "The statistic is the difference of magnitudes per cell; a significant cell says the "
    "estimates differ, not which underlying process changed.",
    "Each segment is fitted on its own, so it must be long enough for the estimator; a continuous "
    "time-varying estimate (for example ADTF) cannot be cut into segments and swapped, and "
    "its segment means have no test here.",
)


@dataclass(frozen=True, eq=False)
class ContrastResult(Correctable):
    """Outcome of a condition contrast, with the premise it rests on.

    All arrays have the shape of ``observed_a.values``. Cells that are undefined in either
    condition are NaN and are not part of any multiple-comparison family.

    Attributes
    ----------
    observed_a, observed_b : Connectivity
        The estimates for each condition (after ``reduce``).
    pvalues : ndarray
        ``(count(|null difference| >= |observed difference|) + 1) / (n_permutations + 1)``.
    effect : ndarray
        Observed difference ``|a| - |b|``.
    zscore : ndarray
        ``effect`` minus the mean permutation difference, over its standard deviation (ddof 1);
        NaN where that is zero or undefined.
    null : str
        The null hypothesis.
    limitations : tuple of str
        Ways the null can be rejected without a change in coupling.
    n_permutations, seed, alpha, n_trials_a, n_trials_b
        How the test was run.
    checks : AssumptionChecklist
        ``permutation_power``, ``permutation_count`` and ``contrast_fits``.
    paired : bool
        Whether the two data sets are two segments of the same trials (segments swapped within
        trials) rather than two groups of trials.
    """

    observed_a: Connectivity
    observed_b: Connectivity
    pvalues: np.ndarray
    effect: np.ndarray
    zscore: np.ndarray
    null: str
    limitations: tuple[str, ...]
    n_permutations: int
    seed: int
    alpha: float
    n_trials_a: int
    n_trials_b: int
    checks: AssumptionChecklist
    paired: bool = False

    def __post_init__(self) -> None:
        """Make the arrays read-only."""
        for name in ("pvalues", "effect", "zscore"):
            view = np.asarray(getattr(self, name), dtype=float).view()
            view.flags.writeable = False
            object.__setattr__(self, name, view)

    @property
    def method(self) -> str:
        """Name of the estimator."""
        return self.observed_a.provenance.method

    @property
    def defined(self) -> np.ndarray:
        """Boolean mask of the cells that have a p-value."""
        return ~np.isnan(self.effect)

    def __repr__(self) -> str:
        """Return a summary that includes the null hypothesis."""
        shape = "x".join(str(n) for n in self.pvalues.shape)
        design = "paired=True, " if self.paired else ""
        return (
            f"ContrastResult(method={self.method!r}, shape={shape}, {design}"
            f"trials={self.n_trials_a}/{self.n_trials_b}, "
            f"n_permutations={self.n_permutations}, null={self.null!r})"
        )


def contrast_test(
    estimator: Estimator,
    data_a: TimeSeries,
    data_b: TimeSeries,
    *,
    n_permutations: int,
    seed: int,
    reduce: Reduction = None,
    alpha: float = 0.05,
    paired: bool = False,
) -> ContrastResult:
    """Test whether connectivity differs between two conditions or two segments of the same trials.

    Permutation ``i`` uses ``RNGController(seed).child(i)`` to shuffle the pooled trials and
    assigns the first ``n_a`` to condition A, so the group sizes are kept and results do not
    depend on how the work is split. The estimator's whole procedure, including a model order
    chosen by a criterion, is repeated on every relabelling.

    With ``paired=True`` the data sets are two segments of the same trials, trial ``k`` of
    ``data_a`` and trial ``k`` of ``data_b`` being cut from the same trial. Permutation ``i`` then
    draws one fair coin per trial from ``RNGController(seed).child(i).integers(0, 2, n_trials)`` and
    swaps the two segments of that trial when it is 1, so every trial keeps both of its segments
    and the groups keep their size. A relabelling and its complement give the same ``|statistic|``,
    so a p-value cannot be finer than ``1 / 2**(n_trials - 1)``. An unpaired test on such data would
    ignore what the two segments of a trial share and compare a null that is too wide.

    Parameters
    ----------
    estimator : Estimator
        Any dirconn estimator.
    data_a, data_b : TimeSeries
        Epoched data with the same channels, sampling frequency and number of samples, and at
        least two trials each. The trial counts may differ; with ``paired=True`` they must be
        equal and the segments of equal length.
    n_permutations : int
        Number of random relabellings; the smallest p-value is ``1 / (n_permutations + 1)``.
        A warning says so when fewer distinct relabellings exist than permutations drawn.
    seed : int
        Required.
    reduce : callable, str or (float, float), optional
        Applied to every fit before comparing (see :func:`surrogate_test`).
    alpha : float
        Level used for the power check, as in :func:`surrogate_test`.
    paired : bool
        ``True`` when ``data_a`` and ``data_b`` are two segments of the same trials (see above).

    Returns
    -------
    ContrastResult

    Raises
    ------
    ValueError
        For incompatible conditions, invalid arguments, or a relabelled fit that leaves a cell
        undefined that is defined in both conditions.
    """
    if not isinstance(estimator, Estimator):
        raise TypeError(f"estimator must be an Estimator, got {type(estimator).__name__}.")
    for label, data in (("data_a", data_a), ("data_b", data_b)):
        if not isinstance(data, TimeSeries):
            raise TypeError(f"{label} must be a TimeSeries, got {type(data).__name__}.")
    if not isinstance(paired, bool):
        raise TypeError(f"paired must be a bool, got {paired!r}.")
    _check_conditions(data_a, data_b, paired=paired)
    n_permutations = _as_count(n_permutations, "n_permutations", minimum=1)
    controller = RNGController(seed)
    alpha = check_alpha(alpha)
    reducer = make_reducer(reduce)

    observed_a = reducer(estimator.fit(data_a))
    observed_b = reducer(estimator.fit(data_b))
    if observed_a.values.shape != observed_b.values.shape:
        raise ValueError(
            f"the two conditions gave results of shapes {observed_a.values.shape} and "
            f"{observed_b.values.shape}; they cannot be compared."
        )
    difference = np.abs(observed_a.values) - np.abs(observed_b.values)
    defined = ~np.isnan(difference)
    n_defined = int(defined.sum())

    pool = np.concatenate([data_a.data, data_b.data], axis=0)
    n_a = data_a.n_trials
    # the masks travel with the trials; a missing mask is all true
    masked = data_a.has_mask or data_b.has_mask

    def mask(data: TimeSeries, name: str) -> np.ndarray:
        value = getattr(data, name)
        return np.ones((data.n_trials, data.n_times), dtype=bool) if value is None else value

    valid_a, valid_b = mask(data_a, "valid"), mask(data_b, "valid")
    predict_a, predict_b = mask(data_a, "predict"), mask(data_b, "predict")
    valid_pool = np.concatenate([valid_a, valid_b], axis=0)
    predict_pool = np.concatenate([predict_a, predict_b], axis=0)
    exceed = np.zeros(difference.shape)
    mean = np.zeros(difference.shape)
    m2 = np.zeros(difference.shape)
    notes: list[str] = []
    other: dict[tuple[type[Warning], str], warnings.WarningMessage] = {}

    light = without_data_checks(estimator)  # the model checks belong to the fits on the data
    first_masks: tuple[np.ndarray | None, np.ndarray | None]
    second_masks: tuple[np.ndarray | None, np.ndarray | None]
    for i in range(n_permutations):
        if paired:
            swap = controller.child(i).integers(0, 2, n_a).astype(bool)[:, None, None]
            first_data = np.where(swap, data_b.data, data_a.data)
            second_data = np.where(swap, data_a.data, data_b.data)
            pick = swap[:, 0]
            first_masks = (np.where(pick, valid_b, valid_a), np.where(pick, predict_b, predict_a))
            second_masks = (np.where(pick, valid_a, valid_b), np.where(pick, predict_a, predict_b))
        else:
            order = controller.child(i).permutation(pool.shape[0])
            first_data, second_data = pool[order[:n_a]], pool[order[n_a:]]
            first_masks = (valid_pool[order[:n_a]], predict_pool[order[:n_a]])
            second_masks = (valid_pool[order[n_a:]], predict_pool[order[n_a:]])
        if not masked:
            first_masks = second_masks = (None, None)
        first = TimeSeries(
            first_data,
            data_a.sfreq,
            data_a.channels,
            "epoched",
            data_a.tmin,
            valid=first_masks[0],
            predict=first_masks[1],
        )
        second = TimeSeries(
            second_data,
            data_a.sfreq,
            data_a.channels,
            "epoched",
            data_a.tmin,
            valid=second_masks[0],
            predict=second_masks[1],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit_first = reducer(light.fit(first))
            fit_second = reducer(light.fit(second))
        notes.extend(sort_warnings(caught, other)[:1])
        null = np.abs(fit_first.values) - np.abs(fit_second.values)
        if null.shape != difference.shape:
            raise ValueError(
                f"permutation {i} gave a result of shape {null.shape}, expected {difference.shape}."
            )
        lost = defined & np.isnan(null)
        if lost.any():
            raise ValueError(
                f"permutation {i} left {int(lost.sum())} cell(s) undefined that are defined in "
                "both conditions; refusing to compute p-values from a degraded null distribution."
            )
        with np.errstate(invalid="ignore"):
            exceed += np.where(defined, np.abs(null) >= np.abs(difference), 0)
        delta = np.where(defined, null - mean, 0.0)
        mean += delta / (i + 1)
        m2 += delta * (np.where(defined, null, 0.0) - mean)
    reissue(other)

    pvalues = np.where(defined, (exceed + 1.0) / (n_permutations + 1.0), np.nan)
    std = np.sqrt(m2 / (n_permutations - 1)) if n_permutations > 1 else np.zeros(difference.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = np.where(defined & (std > 0), (difference - mean) / std, np.nan)

    checks = AssumptionChecklist(
        (
            power_check(
                n_defined,
                n_permutations,
                alpha,
                check="permutation_power",
                argument="n_permutations",
            ),
            _paired_count_check(n_a, n_permutations)
            if paired
            else _count_check(n_a, data_b.n_trials, n_permutations),
            summary_check(
                "contrast_fits",
                notes,
                n_permutations,
                f"the {estimator.info.name} fits on relabelled trials",
            ),
        )
    )
    return ContrastResult(
        observed_a=observed_a,
        observed_b=observed_b,
        pvalues=pvalues,
        effect=np.where(defined, difference, np.nan),
        zscore=zscore,
        null=NULL_PAIRED if paired else NULL,
        limitations=PAIRED_LIMITATIONS if paired else LIMITATIONS,
        n_permutations=n_permutations,
        seed=controller.seed,
        alpha=alpha,
        n_trials_a=n_a,
        n_trials_b=data_b.n_trials,
        checks=checks,
        paired=paired,
    )


def _check_conditions(a: TimeSeries, b: TimeSeries, *, paired: bool = False) -> None:
    for label, data in (("data_a", a), ("data_b", b)):
        if data.segmentation != "epoched":
            raise ValueError(
                f"{label} must be epoched: the test relabels whole trials, and continuous "
                "data have a single trial."
            )
        if data.n_trials < 2:
            raise ValueError(f"{label} needs at least 2 trials, got {data.n_trials}.")
    if paired and a.n_trials != b.n_trials:
        raise ValueError(
            "paired=True needs the same number of trials in both segments (trial k of data_a and "
            f"trial k of data_b come from the same trial), got {a.n_trials} and {b.n_trials}."
        )
    if a.names != b.names:
        raise ValueError(f"the conditions have different channels: {a.names} and {b.names}.")
    if a.sfreq != b.sfreq:
        raise ValueError(f"the conditions have different sfreq: {a.sfreq} and {b.sfreq}.")
    if a.n_times != b.n_times:
        raise ValueError(
            f"the conditions have different n_times: {a.n_times} and {b.n_times}; "
            + (
                "the segments of a trial must be swappable."
                if paired
                else "trials must be exchangeable."
            )
        )


def _count_check(n_a: int, n_b: int, n_permutations: int) -> AssumptionCheck:
    distinct = math.comb(n_a + n_b, n_a)
    if distinct <= n_permutations:
        detail = (
            f"only {distinct} distinct relabellings of {n_a} + {n_b} trials exist, fewer than "
            f"the {n_permutations} drawn; permutations repeat and a p-value cannot be finer "
            f"than 1/{distinct}."
        )
        warnings.warn(detail, DirconnWarning, stacklevel=3)
        return AssumptionCheck("permutation_count", "warn", detail=detail, value=float(distinct))
    return AssumptionCheck(
        "permutation_count",
        "pass",
        detail=f"{distinct} distinct relabellings of {n_a} + {n_b} trials.",
        value=float(distinct),
    )


def _paired_count_check(n_trials: int, n_permutations: int) -> AssumptionCheck:
    distinct = 2 ** (n_trials - 1)
    if distinct <= n_permutations:
        detail = (
            f"only {distinct} distinct relabellings of the segments of {n_trials} trials exist "
            "(a relabelling and its complement give the same |statistic|), fewer than the "
            f"{n_permutations} drawn; permutations repeat and a p-value cannot be finer than "
            f"1/{distinct}."
        )
        warnings.warn(detail, DirconnWarning, stacklevel=3)
        return AssumptionCheck("permutation_count", "warn", detail=detail, value=float(distinct))
    return AssumptionCheck(
        "permutation_count",
        "pass",
        detail=f"{distinct} distinct relabellings of the segments of {n_trials} trials.",
        value=float(distinct),
    )
