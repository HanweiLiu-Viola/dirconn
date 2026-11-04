import warnings
from dataclasses import dataclass, field

import numpy as np
import pytest

from dirconn import Connectivity, DirconnWarning, TimeSeries
from dirconn.contrast import ContrastResult, contrast_test
from dirconn.estimator import Estimator, MethodInfo
from dirconn.rng import RNGController
from helpers import MeanDifference, epochs

# Most tests use few permutations for speed; the power check has its own tests below.
pytestmark = pytest.mark.filterwarnings("ignore:n_permutations=:dirconn.exceptions.DirconnWarning")


def pooled(a, b):
    return np.concatenate([a.data, b.data], axis=0)


def manual_pvalues(estimator, a, b, n, seed):
    pool = pooled(a, b)
    n_a = a.n_trials
    diff = np.abs(estimator.fit(a).values) - np.abs(estimator.fit(b).values)
    null = []
    for i in range(n):
        order = RNGController(seed).child(i).permutation(pool.shape[0])
        first = TimeSeries(pool[order[:n_a]], a.sfreq, a.channels, "epoched", a.tmin)
        second = TimeSeries(pool[order[n_a:]], a.sfreq, a.channels, "epoched", a.tmin)
        null.append(np.abs(estimator.fit(first).values) - np.abs(estimator.fit(second).values))
    null = np.stack(null)
    p = (np.sum(np.abs(null) >= np.abs(diff), axis=0) + 1) / (n + 1)
    return np.where(np.isnan(diff), np.nan, p), diff, null


class TestStatistic:
    def test_pvalues_follow_the_rank_rule_on_the_difference_of_magnitudes(self):
        a, b = epochs(n_trials=12, seed=1), epochs(n_trials=15, means=(0.0, 1.0, 0.5), seed=2)
        result = contrast_test(MeanDifference(), a, b, n_permutations=30, seed=4)
        p, diff, null = manual_pvalues(MeanDifference(), a, b, 30, 4)
        np.testing.assert_array_equal(result.pvalues, p)
        np.testing.assert_allclose(result.effect, diff)
        ok = result.defined
        np.testing.assert_allclose(
            result.zscore[ok], ((diff - null.mean(axis=0)) / null.std(axis=0, ddof=1))[ok]
        )

    def test_a_real_difference_reaches_the_floor_and_an_unchanged_pair_does_not(self):
        # Only channel 2 changes between the conditions (mean 3 against 0.4).
        a = epochs(n_trials=15, means=(0.0, 1.0, 3.0), seed=3)
        b = epochs(n_trials=15, means=(0.0, 1.0, 0.4), seed=4)
        n = 99
        result = contrast_test(MeanDifference(), a, b, n_permutations=n, seed=0)
        assert result.pvalues[0, 2, 0, 0] == pytest.approx(1 / (n + 1))
        assert result.pvalues[1, 2, 0, 0] == pytest.approx(1 / (n + 1))
        assert result.pvalues[0, 1, 0, 0] > 0.05

    def test_a_decrease_is_found_as_well_as_an_increase(self):
        # The test is two-sided: condition A is much weaker than B in the cells of channel 2.
        a = epochs(n_trials=15, means=(0.0, 1.0, 0.4), seed=3)
        b = epochs(n_trials=15, means=(0.0, 1.0, 3.0), seed=4)
        result = contrast_test(MeanDifference(), a, b, n_permutations=99, seed=0)
        assert result.effect[0, 2, 0, 0] < 0
        assert result.pvalues[0, 2, 0, 0] == pytest.approx(1 / 100)

    def test_the_diagonal_is_undefined(self):
        result = contrast_test(MeanDifference(), epochs(), epochs(seed=1), n_permutations=9, seed=0)
        diagonal = np.broadcast_to(np.eye(3, dtype=bool)[:, :, None, None], result.pvalues.shape)
        for array in (result.pvalues, result.effect, result.zscore):
            assert np.isnan(array[diagonal]).all()
            assert np.isfinite(array[~diagonal]).all()

    def test_unequal_group_sizes_are_kept_in_every_relabelling(self):
        @dataclass(frozen=True)
        class Sizes(Estimator):
            info = MethodInfo(name="sizes", backend="stub", directed=False)
            seen: list = field(default_factory=list, init=False, compare=False)

            def _fit(self, data):
                self.seen.append(data.n_trials)
                values = np.ones((data.n_channels, data.n_channels))
                np.fill_diagonal(values, np.nan)
                return Connectivity.from_array(
                    values, source=data.names, provenance=self._provenance(data)
                )

        spy = Sizes()
        contrast_test(
            spy, epochs(n_trials=8), epochs(n_trials=13, seed=1), n_permutations=5, seed=0
        )
        assert spy.seen == [8, 13] * 6  # the data, then five relabellings

    def test_every_relabelling_partitions_the_pooled_trials(self):
        @dataclass(frozen=True)
        class Rows(Estimator):
            info = MethodInfo(name="rows", backend="stub", directed=False)
            seen: list = field(default_factory=list, init=False, compare=False)

            def _fit(self, data):
                self.seen.append(sorted(float(t[0, 0]) for t in data.data))
                values = np.ones((data.n_channels, data.n_channels))
                np.fill_diagonal(values, np.nan)
                return Connectivity.from_array(
                    values, source=data.names, provenance=self._provenance(data)
                )

        a, b = epochs(n_trials=6), epochs(n_trials=7, seed=1)
        spy = Rows()
        contrast_test(spy, a, b, n_permutations=4, seed=0)
        everything = sorted(float(t[0, 0]) for t in pooled(a, b))
        for k in range(1, 5):
            first, second = spy.seen[2 * k], spy.seen[2 * k + 1]
            assert sorted(first + second) == everything


class TestReproducibility:
    def test_same_seed_same_result_and_the_seed_is_required(self):
        a, b = epochs(seed=1), epochs(seed=2)
        run = lambda seed: contrast_test(  # noqa: E731
            MeanDifference(), a, b, n_permutations=20, seed=seed
        )
        np.testing.assert_array_equal(run(3).pvalues, run(3).pvalues)
        assert not np.array_equal(run(3).zscore, run(4).zscore, equal_nan=True)
        with pytest.raises(TypeError):
            contrast_test(MeanDifference(), a, b, n_permutations=20)  # type: ignore[call-arg]


@pytest.fixture(scope="module")
def contrast_record():
    a, b = epochs(n_trials=10, seed=1), epochs(n_trials=14, seed=2)
    return contrast_test(MeanDifference(), a, b, n_permutations=19, seed=6, alpha=0.1)


class TestResultRecord:
    @pytest.fixture
    def record(self, contrast_record):
        return contrast_record

    def test_it_states_the_null_and_the_limitations(self, record):
        assert isinstance(record, ContrastResult)
        assert "both conditions" in record.null
        assert len(record.limitations) >= 3 and all(s.strip() for s in record.limitations)
        assert record.null in repr(record)

    def test_it_records_how_the_test_was_run(self, record):
        assert (record.n_permutations, record.seed, record.alpha) == (19, 6, 0.1)
        assert (record.n_trials_a, record.n_trials_b) == (10, 14)
        assert record.method == "meandiff"

    def test_it_keeps_both_observed_results(self, record):
        a, b = epochs(n_trials=10, seed=1), epochs(n_trials=14, seed=2)
        np.testing.assert_array_equal(record.observed_a.values, MeanDifference().fit(a).values)
        np.testing.assert_array_equal(record.observed_b.values, MeanDifference().fit(b).values)

    def test_correction_works_on_it(self, record):
        assert record.significant().shape == record.pvalues.shape
        assert not record.significant(method="none")[~record.defined].any()


class TestChecks:
    def test_the_power_check_counts_the_defined_cells(self):
        a, b = epochs(), epochs(seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DirconnWarning)
            ok = contrast_test(MeanDifference(), a, b, n_permutations=119, seed=0)
        assert ok.checks.get("permutation_power").status == "pass"
        with pytest.warns(DirconnWarning, match="n_permutations"):
            low = contrast_test(MeanDifference(), a, b, n_permutations=118, seed=0)
        assert low.checks.get("permutation_power").status == "warn"

    def test_few_distinct_relabellings_are_reported(self):
        # 4 + 4 trials have only 70 distinct splits: a p-value cannot be finer than 1/70.
        a, b = epochs(n_trials=4), epochs(n_trials=4, seed=1)
        with pytest.warns(DirconnWarning, match="distinct"):
            result = contrast_test(MeanDifference(), a, b, n_permutations=199, seed=0, alpha=0.9)
        assert result.checks.get("permutation_count").status == "warn"

    def test_many_trials_have_plenty_of_relabellings(self):
        result = contrast_test(
            MeanDifference(), epochs(), epochs(seed=1), n_permutations=19, seed=0, alpha=0.9
        )
        assert result.checks.get("permutation_count").status == "pass"

    def test_diagnostics_inside_the_fits_are_summarised_once(self):
        @dataclass(frozen=True)
        class Noisy(Estimator):
            info = MethodInfo(name="noisy", backend="stub", directed=False)

            def _fit(self, data):
                warnings.warn("about the data", DirconnWarning, stacklevel=2)
                values = np.ones((data.n_channels, data.n_channels))
                np.fill_diagonal(values, np.nan)
                return Connectivity.from_array(
                    values, source=data.names, provenance=self._provenance(data)
                )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = contrast_test(
                Noisy(), epochs(), epochs(seed=1), n_permutations=10, seed=0, alpha=0.9
            )
        messages = [str(w.message) for w in caught if issubclass(w.category, DirconnWarning)]
        # two fits on the data (one message each) and one summary for the permutations
        assert len(messages) == 3 and any("10 of 10" in m for m in messages)
        assert result.checks.get("contrast_fits").status == "warn"

    def test_a_relabelling_that_loses_a_cell_is_an_error(self):
        @dataclass(frozen=True)
        class LosesACell(Estimator):
            info = MethodInfo(name="loses", backend="stub", directed=False)
            calls: list = field(default_factory=list, init=False, compare=False)

            def _fit(self, data):
                self.calls.append(1)
                values = np.ones((data.n_channels, data.n_channels))
                if len(self.calls) > 2:
                    values[0, 1] = np.nan
                np.fill_diagonal(values, np.nan)
                return Connectivity.from_array(
                    values, source=data.names, provenance=self._provenance(data)
                )

        with pytest.raises(ValueError, match="undefined"):
            contrast_test(LosesACell(), epochs(), epochs(seed=1), n_permutations=5, seed=0)


class TestArguments:
    def test_the_conditions_must_have_the_same_layout(self):
        base = epochs()
        other_channels = TimeSeries.from_array(
            base.data, base.sfreq, names=["x", "y", "z"], segmentation="epoched"
        )
        other_rate = TimeSeries.from_array(base.data, 200.0, segmentation="epoched")
        other_length = epochs(n_times=60)
        for other, text in [
            (other_channels, "channels"),
            (other_rate, "sfreq"),
            (other_length, "n_times"),
        ]:
            with pytest.raises(ValueError, match=text):
                contrast_test(MeanDifference(), base, other, n_permutations=5, seed=0)

    def test_continuous_data_and_single_trials_are_refused(self):
        continuous = TimeSeries.from_array(np.random.default_rng(0).standard_normal((3, 50)), 100.0)
        with pytest.raises(ValueError, match="epoched"):
            contrast_test(MeanDifference(), continuous, epochs(), n_permutations=5, seed=0)
        with pytest.raises(ValueError, match="at least 2 trials"):
            contrast_test(MeanDifference(), epochs(n_trials=1), epochs(), n_permutations=5, seed=0)

    @pytest.mark.parametrize("n", [0, -1, 2.5, True])
    def test_the_number_of_permutations_must_be_a_positive_integer(self, n):
        with pytest.raises((TypeError, ValueError)):
            contrast_test(MeanDifference(), epochs(), epochs(), n_permutations=n, seed=0)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, float("nan")])
    def test_alpha_must_lie_strictly_between_zero_and_one(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            contrast_test(
                MeanDifference(), epochs(), epochs(), n_permutations=5, seed=0, alpha=alpha
            )

    def test_types_are_checked(self):
        with pytest.raises(TypeError, match="Estimator"):
            contrast_test(lambda d: d, epochs(), epochs(), n_permutations=5, seed=0)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="TimeSeries"):
            contrast_test(MeanDifference(), np.zeros((3, 10)), epochs(), n_permutations=5, seed=0)  # type: ignore[arg-type]


class TestTypeIErrorCalibration:
    """Two samples from one distribution give uniform p-values.

    Measured over 1500 datasets (10 trials each, 19 permutations) for three cells: the share of
    p-values at or below 0.10 was 0.107, 0.092 and 0.095 and at or below 0.05 it was between 0.044
    and 0.054, as they should be. With 19 permutations the p-values take the values 1/20, ..., 1,
    so their mean under the null is 0.525, not 0.5. One cell per dataset is used here because the
    cells of one dataset are dependent; 60 values have a standard deviation of about 0.04 around
    that mean.
    """

    def test_identical_conditions_give_uniform_pvalues(self):
        p = []
        for seed in range(60):
            a = epochs(n_trials=10, n_times=20, seed=2 * seed)
            b = epochs(n_trials=10, n_times=20, seed=2 * seed + 1)
            result = contrast_test(MeanDifference(), a, b, n_permutations=19, seed=seed, alpha=0.9)
            p.append(result.pvalues[0, 1, 0, 0])
        p = np.asarray(p)
        assert 0.4 < p.mean() < 0.65
        assert (p <= 0.1).mean() < 0.25
