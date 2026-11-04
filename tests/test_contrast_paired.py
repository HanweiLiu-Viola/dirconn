"""``contrast_test(paired=True)``: two segments of the same trials, swapped within each trial."""

import numpy as np
import pytest

from dirconn import DTF, DirconnWarning, TimeSeries
from dirconn.contrast import contrast_test
from dirconn.rng import RNGController
from helpers import MeanDifference, Spy, epochs

SFREQ = 100.0
pytestmark = pytest.mark.filterwarnings("ignore:n_permutations=:dirconn.exceptions.DirconnWarning")


def swapped(a, b, n_permutations, seed):
    """The relabellings the test must draw: one coin per trial, ``RNGController(seed).child(i)``."""
    for i in range(n_permutations):
        swap = RNGController(seed).child(i).integers(0, 2, a.n_trials).astype(bool)
        first = np.where(swap[:, None, None], b.data, a.data)
        second = np.where(swap[:, None, None], a.data, b.data)
        yield swap, first, second


def as_series(data, like):
    return TimeSeries(data, like.sfreq, like.channels, "epoched", like.tmin)


def manual_pvalues(estimator, a, b, n, seed):
    diff = np.abs(estimator.fit(a).values) - np.abs(estimator.fit(b).values)
    null = []
    for _, first, second in swapped(a, b, n, seed):
        f = np.abs(estimator.fit(as_series(first, a)).values)
        s = np.abs(estimator.fit(as_series(second, a)).values)
        null.append(f - s)
    null = np.stack(null)
    p = (np.sum(np.abs(null) >= np.abs(diff), axis=0) + 1) / (n + 1)
    return np.where(np.isnan(diff), np.nan, p), diff


# --------------------------------------------------------------------------
# The statistic and the relabelling
# --------------------------------------------------------------------------


def test_pvalues_follow_the_rank_rule_with_a_swap_within_every_trial():
    a = epochs(n_trials=10, seed=1)
    b = epochs(n_trials=10, means=(0.0, 1.0, 0.5), seed=2)
    result = contrast_test(MeanDifference(), a, b, n_permutations=40, seed=5, paired=True)
    p, diff = manual_pvalues(MeanDifference(), a, b, 40, 5)
    np.testing.assert_array_equal(result.pvalues, p)
    np.testing.assert_allclose(result.effect, diff)
    assert result.paired is True


def test_trials_keep_their_pairs_in_every_relabelling():
    """Each fitted dataset holds, for trial k, either (a_k, b_k) or (b_k, a_k), never a mix."""
    a = epochs(n_trials=6, seed=3)
    b = epochs(n_trials=6, means=(0.0, 1.0, 0.5), seed=4)
    spy = Spy()
    contrast_test(spy, a, b, n_permutations=8, seed=0, paired=True)
    seen = spy.seen[2:]  # the first two fits are the observed segments
    assert len(seen) == 16
    for first, second in zip(seen[0::2], seen[1::2], strict=True):
        for k in range(6):
            from_a = np.array_equal(first.data[k], a.data[k])
            from_b = np.array_equal(first.data[k], b.data[k])
            assert from_a != from_b
            partner = b.data[k] if from_a else a.data[k]
            np.testing.assert_array_equal(second.data[k], partner)


def test_both_groups_always_keep_the_number_of_trials():
    a, b = epochs(n_trials=7, seed=3), epochs(n_trials=7, seed=4)
    spy = Spy()
    contrast_test(spy, a, b, n_permutations=5, seed=0, paired=True)
    assert all(d.n_trials == 7 for d in spy.seen)


def test_the_result_does_not_depend_on_anything_but_the_seed():
    a, b = epochs(n_trials=8, seed=5), epochs(n_trials=8, means=(0.0, 1.0, 2.0), seed=6)
    first = contrast_test(MeanDifference(), a, b, n_permutations=20, seed=2, paired=True)
    again = contrast_test(MeanDifference(), a, b, n_permutations=20, seed=2, paired=True)
    other = contrast_test(MeanDifference(), a, b, n_permutations=20, seed=3, paired=True)
    np.testing.assert_array_equal(first.pvalues, again.pvalues)
    assert not np.array_equal(first.pvalues, other.pvalues)


def test_a_real_paired_difference_reaches_the_floor_and_an_unchanged_pair_does_not():
    a = epochs(n_trials=12, means=(0.0, 1.0, 3.0), seed=3)
    b = epochs(n_trials=12, means=(0.0, 1.0, 0.4), seed=4)
    n = 99
    result = contrast_test(MeanDifference(), a, b, n_permutations=n, seed=0, paired=True)
    assert result.pvalues[0, 2, 0, 0] == pytest.approx(1 / (n + 1))
    assert result.pvalues[0, 1, 0, 0] > 0.05


def test_unpaired_is_still_the_default_and_unchanged():
    a, b = epochs(n_trials=9, seed=1), epochs(n_trials=11, means=(0.0, 1.0, 0.5), seed=2)
    default = contrast_test(MeanDifference(), a, b, n_permutations=10, seed=4)
    explicit = contrast_test(MeanDifference(), a, b, n_permutations=10, seed=4, paired=False)
    np.testing.assert_array_equal(default.pvalues, explicit.pvalues)
    assert default.paired is False
    assert "exchangeable between conditions" in default.null


# --------------------------------------------------------------------------
# The premise it states
# --------------------------------------------------------------------------


def test_the_null_and_the_limitations_describe_the_paired_design():
    a, b = epochs(n_trials=6, seed=1), epochs(n_trials=6, seed=2)
    result = contrast_test(MeanDifference(), a, b, n_permutations=5, seed=0, paired=True)
    assert "within each trial" in result.null
    text = " ".join(result.limitations)
    assert "time-order" in text and "adaptation" in text and "fatigue" in text
    assert "paired=True" in repr(result)
    # the unpaired premise (trials exchangeable between conditions) is not claimed here
    assert "exchangeable between conditions" not in result.null


def test_the_trial_counts_are_equal_by_construction():
    a, b = epochs(n_trials=6, seed=1), epochs(n_trials=6, seed=2)
    result = contrast_test(MeanDifference(), a, b, n_permutations=5, seed=0, paired=True)
    assert result.n_trials_a == result.n_trials_b == 6


def test_few_trials_have_few_distinct_relabellings_and_warn():
    a, b = epochs(n_trials=4, seed=1), epochs(n_trials=4, seed=2)
    with pytest.warns(DirconnWarning, match="distinct"):
        result = contrast_test(MeanDifference(), a, b, n_permutations=50, seed=0, paired=True)
    check = result.checks.get("permutation_count")
    assert check.status == "warn"
    assert check.value == 2 ** (4 - 1)  # a labelling and its complement give the same |statistic|


def test_enough_trials_pass_the_count_check():
    a, b = epochs(n_trials=12, seed=1), epochs(n_trials=12, seed=2)
    result = contrast_test(MeanDifference(), a, b, n_permutations=50, seed=0, paired=True)
    assert result.checks.get("permutation_count").status == "pass"


# --------------------------------------------------------------------------
# Why pairing matters, and that it stays honest
# --------------------------------------------------------------------------


def coupled_trials(seed, *, extra, n_trials=12, n_times=200):
    """Two segments of the same trials; trial k has its own coupling ``c_k``, B adds ``extra``."""
    rng = np.random.default_rng(seed)
    gains = rng.uniform(0.0, 0.6, n_trials)

    def trial(c):
        a = np.array([[0.4, 0.0], [c, 0.3]])
        x = np.zeros((2, n_times + 50))
        for t in range(1, n_times + 50):
            x[:, t] = a @ x[:, t - 1] + rng.standard_normal(2)
        return x[:, 50:]

    first = np.stack([trial(c) for c in gains])
    second = np.stack([trial(c + extra) for c in gains])
    make = lambda d: TimeSeries.from_array(d, SFREQ, segmentation="epoched")  # noqa: E731
    return make(first), make(second)


ESTIMATOR = DTF(order=1, n_freqs=9, check_order_stability=False)


def test_pairing_removes_the_between_trial_variance_that_the_unpaired_null_keeps():
    """Trials differ a lot in coupling, segment B adds a little; every segment pair shares its
    trial's coupling. The unpaired relabelling mixes strong and weak trials between the groups and
    gets a much wider null; the paired one keeps the trials together."""
    paired, unpaired = [], []
    for seed in range(4):
        a, b = coupled_trials(seed, extra=0.2)
        kwargs = dict(n_permutations=99, seed=0, reduce=(0.0, 50.0))
        paired.append(contrast_test(ESTIMATOR, a, b, paired=True, **kwargs).pvalues[0, 1, 0, 0])
        unpaired.append(contrast_test(ESTIMATOR, a, b, **kwargs).pvalues[0, 1, 0, 0])
    # measured over seeds 0-5: paired 0.01 every time; unpaired 0.03-0.10
    assert max(paired) == pytest.approx(0.01)
    assert min(unpaired) > max(paired)


def test_under_a_true_null_the_paired_test_rejects_about_as_often_as_it_should():
    """No segment effect at all: the two segments of a trial are exchangeable by construction."""
    rejections = 0
    n = 40
    for seed in range(2000, 2000 + n):
        a, b = coupled_trials(seed, extra=0.0, n_trials=10, n_times=150)
        result = contrast_test(
            ESTIMATOR, a, b, n_permutations=49, seed=1, paired=True, reduce=(0.0, 50.0)
        )
        rejections += result.pvalues[0, 1, 0, 0] <= 0.1
    # calibration of exactly this configuration: over 300 fresh datasets the
    # rejection rate at 0.05 / 0.1 / 0.2 / 0.5 was 3.3 % / 9.7 % / 20.7 % / 49 %; these 40 fixed
    # seeds give 3 rejections (Binomial(40, 0.1) has mean 4 and exceeds 8 about 2 % of the time).
    assert rejections <= 8


# --------------------------------------------------------------------------
# Input checks
# --------------------------------------------------------------------------


def test_a_paired_test_needs_the_same_number_of_trials():
    a, b = epochs(n_trials=6, seed=1), epochs(n_trials=7, seed=2)
    with pytest.raises(ValueError, match="same number of trials"):
        contrast_test(MeanDifference(), a, b, n_permutations=5, seed=0, paired=True)


def test_paired_segments_must_have_the_same_length():
    a, b = epochs(n_trials=6, n_times=50, seed=1), epochs(n_trials=6, n_times=60, seed=2)
    with pytest.raises(ValueError, match="n_times"):
        contrast_test(MeanDifference(), a, b, n_permutations=5, seed=0, paired=True)


def test_paired_segments_must_be_epoched_with_at_least_two_trials():
    rng = np.random.default_rng(0)
    continuous = TimeSeries.from_array(rng.standard_normal((3, 100)), SFREQ)
    with pytest.raises(ValueError, match="epoched"):
        contrast_test(
            MeanDifference(), continuous, continuous, n_permutations=5, seed=0, paired=True
        )
    single = TimeSeries.from_array(rng.standard_normal((1, 3, 100)), SFREQ, segmentation="epoched")
    with pytest.raises(ValueError, match="at least 2 trials"):
        contrast_test(MeanDifference(), single, single, n_permutations=5, seed=0, paired=True)


@pytest.mark.parametrize("bad", [1, 0, "yes", None])
def test_paired_must_be_a_bool(bad):
    a, b = epochs(n_trials=6, seed=1), epochs(n_trials=6, seed=2)
    with pytest.raises(TypeError, match="paired must be a bool"):
        contrast_test(MeanDifference(), a, b, n_permutations=5, seed=0, paired=bad)
