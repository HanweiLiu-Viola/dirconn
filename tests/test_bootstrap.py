import warnings
from dataclasses import dataclass, field

import numpy as np
import pytest

from dirconn import Connectivity, DirconnWarning, TimeSeries
from dirconn.bootstrap import BootstrapResult, bootstrap
from dirconn.estimator import Estimator, MethodInfo
from dirconn.rng import RNGController
from helpers import MeanDifference, coupled, epochs


@dataclass(frozen=True)
class Spy(Estimator):
    """Records the shape and content of every dataset it is fitted on."""

    info = MethodInfo(name="spy", backend="test stub", directed=False)
    seen: list = field(default_factory=list, init=False, compare=False)

    def _fit(self, data: TimeSeries) -> Connectivity:
        self.seen.append(data)
        values = np.ones((data.n_channels, data.n_channels))
        np.fill_diagonal(values, np.nan)
        return Connectivity.from_array(values, source=data.names, provenance=self._provenance(data))


def manual_samples(estimator, data, n_boot, seed):
    controller = RNGController(seed)
    out = []
    for i in range(n_boot):
        index = controller.child(i).integers(0, data.n_trials, size=data.n_trials)
        resampled = TimeSeries(
            data.data[index], data.sfreq, data.channels, data.segmentation, data.tmin
        )
        out.append(np.abs(estimator.fit(resampled).values))
    return np.stack(out)


class TestTrialBootstrap:
    def test_the_bands_are_the_percentiles_of_the_resampled_magnitudes(self):
        data = epochs()
        result = bootstrap(MeanDifference(), data, n_boot=60, seed=3, ci=0.9)
        samples = manual_samples(MeanDifference(), data, 60, 3)
        ok = result.defined
        np.testing.assert_allclose(result.lower[ok], np.quantile(samples, 0.05, axis=0)[ok])
        np.testing.assert_allclose(result.upper[ok], np.quantile(samples, 0.95, axis=0)[ok])
        np.testing.assert_allclose(result.std[ok], samples.std(axis=0, ddof=1)[ok])

    def test_the_observed_result_is_the_fit_on_the_original_data(self):
        data = epochs()
        result = bootstrap(MeanDifference(), data, n_boot=10, seed=0)
        np.testing.assert_array_equal(result.observed.values, MeanDifference().fit(data).values)

    def test_undefined_cells_have_no_band(self):
        result = bootstrap(MeanDifference(), epochs(), n_boot=10, seed=0)
        diagonal = np.eye(3, dtype=bool)[:, :, None, None]
        for array in (result.lower, result.upper, result.std):
            assert np.isnan(array[np.broadcast_to(diagonal, array.shape)]).all()
            assert np.isfinite(array[~np.broadcast_to(diagonal, array.shape)]).all()

    def test_the_band_contains_the_observed_value_most_of_the_time_and_is_ordered(self):
        result = bootstrap(MeanDifference(), epochs(), n_boot=200, seed=1)
        ok = result.defined
        magnitude = np.abs(result.observed.values)
        assert (result.lower[ok] <= result.upper[ok]).all()
        assert ((result.lower[ok] <= magnitude[ok]) & (magnitude[ok] <= result.upper[ok])).all()

    def test_every_resample_draws_trials_with_replacement_from_the_originals(self):
        data = epochs(n_trials=25)
        spy = Spy()
        bootstrap(spy, data, n_boot=8, seed=0)
        original = {tuple(t[0]): i for i, t in enumerate(data.data)}
        assert len(spy.seen) == 1 + 8  # the data itself, then the resamples
        for sample in spy.seen[1:]:
            assert sample.n_trials == 25 and sample.n_times == data.n_times
            assert all(tuple(t[0]) in original for t in sample.data)
        assert any(len({tuple(t[0]) for t in s.data}) < 25 for s in spy.seen[1:]), (
            "with replacement some trial must repeat"
        )

    def test_it_is_reproducible_and_the_seed_is_required(self):
        data = epochs()
        a = bootstrap(MeanDifference(), data, n_boot=20, seed=5)
        b = bootstrap(MeanDifference(), data, n_boot=20, seed=5)
        c = bootstrap(MeanDifference(), data, n_boot=20, seed=6)
        np.testing.assert_array_equal(a.lower, b.lower)
        assert not np.array_equal(a.lower, c.lower, equal_nan=True)
        with pytest.raises(TypeError):
            bootstrap(MeanDifference(), data, n_boot=20)  # type: ignore[call-arg]

    def test_it_records_how_it_was_run(self):
        result = bootstrap(MeanDifference(), epochs(), n_boot=15, seed=2, ci=0.8)
        assert isinstance(result, BootstrapResult)
        assert (result.n_boot, result.seed, result.ci) == (15, 2, 0.8)
        assert result.method == "meandiff"
        assert result.resampling["unit"] == "trial" and result.resampling["n_units"] == 30
        assert result.interval == "percentile"


class TestBlockBootstrapOfContinuousData:
    @staticmethod
    def continuous(n_times=1000):
        x = np.random.default_rng(0).standard_normal((3, n_times))
        return TimeSeries.from_array(x, 100.0)

    def test_it_needs_a_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            bootstrap(MeanDifference(), self.continuous(), n_boot=5, seed=0)

    def test_epoched_data_refuse_a_block_size_instead_of_ignoring_it(self):
        with pytest.raises(ValueError, match="continuous"):
            bootstrap(MeanDifference(), epochs(), n_boot=5, seed=0, block_size=10)

    def test_resamples_are_blocks_stored_as_separate_trials(self):
        # Blocks become trials, so a model fit never regresses across a seam between blocks.
        data = self.continuous(1000)
        spy = Spy()
        result = bootstrap(spy, data, n_boot=6, seed=0, block_size=50)
        blocks = {tuple(data.data[0, :, k * 50 : (k + 1) * 50].ravel()) for k in range(20)}
        for sample in spy.seen[1:]:
            assert sample.segmentation == "epoched"
            assert (sample.n_trials, sample.n_times) == (20, 50)
            assert all(tuple(t.ravel()) in blocks for t in sample.data)
        assert result.resampling["unit"] == "block"
        assert result.resampling["block_size"] == 50 and result.resampling["n_units"] == 20

    def test_the_samples_left_over_after_the_last_whole_block_are_reported(self):
        result = bootstrap(MeanDifference(), self.continuous(1030), n_boot=5, seed=0, block_size=50)
        assert result.resampling["dropped_samples"] == 30

    def test_at_least_two_blocks_are_needed(self):
        with pytest.raises(ValueError, match="two blocks"):
            bootstrap(MeanDifference(), self.continuous(90), n_boot=5, seed=0, block_size=50)

    @pytest.mark.parametrize("block_size", [0, -5, 2.5, True])
    def test_the_block_size_must_be_a_positive_integer(self, block_size):
        with pytest.raises((TypeError, ValueError)):
            bootstrap(MeanDifference(), self.continuous(), n_boot=5, seed=0, block_size=block_size)


class TestFewUnits:
    def test_fewer_than_twenty_trials_warn_and_are_recorded(self):
        with pytest.warns(DirconnWarning, match="at least 20"):
            result = bootstrap(MeanDifference(), epochs(n_trials=12), n_boot=10, seed=0)
        assert result.checks.get("bootstrap_units").status == "warn"

    def test_twenty_trials_are_enough(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DirconnWarning)
            result = bootstrap(MeanDifference(), epochs(n_trials=20), n_boot=10, seed=0)
        assert result.checks.get("bootstrap_units").status == "pass"

    def test_one_trial_cannot_be_bootstrapped(self):
        data = TimeSeries.from_array(
            np.random.default_rng(0).standard_normal((1, 3, 50)), 100.0, segmentation="epoched"
        )
        with pytest.raises(ValueError, match="at least 2"):
            bootstrap(MeanDifference(), data, n_boot=5, seed=0)


class TestDiagnosticsAndFailures:
    def test_a_resample_that_loses_a_cell_is_an_error(self):
        @dataclass(frozen=True)
        class LosesACell(Estimator):
            info = MethodInfo(name="loses", backend="stub", directed=False)
            calls: list = field(default_factory=list, compare=False)

            def _fit(self, data):
                self.calls.append(1)
                values = np.ones((data.n_channels, data.n_channels))
                if len(self.calls) > 1:
                    values[0, 1] = np.nan
                np.fill_diagonal(values, np.nan)
                return Connectivity.from_array(
                    values, source=data.names, provenance=self._provenance(data)
                )

        with pytest.raises(ValueError, match="undefined"):
            bootstrap(LosesACell(), epochs(), n_boot=5, seed=0)

    def test_warnings_inside_the_resampled_fits_are_summarised_once(self):
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
            result = bootstrap(Noisy(), epochs(), n_boot=12, seed=0)
        messages = [str(w.message) for w in caught if issubclass(w.category, DirconnWarning)]
        assert len(messages) == 2 and any("12 of 12" in m for m in messages)
        assert result.checks.get("bootstrap_fits").status == "warn"


class TestArguments:
    @pytest.mark.parametrize("n_boot", [0, -1, 2.5, True])
    def test_the_number_of_resamples_must_be_a_positive_integer(self, n_boot):
        with pytest.raises((TypeError, ValueError)):
            bootstrap(MeanDifference(), epochs(), n_boot=n_boot, seed=0)

    @pytest.mark.parametrize("ci", [0.0, 1.0, -0.2, 1.5, float("nan")])
    def test_the_level_must_lie_strictly_between_zero_and_one(self, ci):
        with pytest.raises(ValueError, match="ci"):
            bootstrap(MeanDifference(), epochs(), n_boot=5, seed=0, ci=ci)

    def test_types_are_checked(self):
        with pytest.raises(TypeError, match="Estimator"):
            bootstrap(lambda d: d, epochs(), n_boot=5, seed=0)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="TimeSeries"):
            bootstrap(MeanDifference(), np.zeros((3, 10)), n_boot=5, seed=0)  # type: ignore[arg-type]

    def test_reduce_applies_to_every_fit(self):
        from dirconn import DTF

        data = coupled(n_trials=25, n_times=200)
        result = bootstrap(DTF(order=2), data, n_boot=8, seed=0, reduce=(10.0, 20.0))
        assert result.lower.shape == result.observed.values.shape
        assert result.lower.shape[2] == 1


class TestCoverage:
    """The percentile band covers the true value slightly less often than its nominal level.

    ``MeanDifference`` on channels with means 0, 1 and 3 has the true magnitudes 1, 3 and 2.
    Measured over 300 datasets of 30 trials with 200 resamples: a nominal 90% band covered 86.2%
    (86.7% over the first 60 datasets, the ones used here). That under-coverage is the known
    behaviour of percentile intervals at small sample sizes, so the test demands the measured
    range, not the nominal level.
    """

    def test_a_ninety_percent_band_covers_about_ninety_percent_of_the_time(self):
        truth = np.abs(np.array([0.0, 1.0, 3.0])[:, None] - np.array([0.0, 1.0, 3.0])[None, :])
        hits, total = 0, 0
        for seed in range(60):
            result = bootstrap(
                MeanDifference(), epochs(n_trials=30, seed=seed), n_boot=200, seed=seed, ci=0.9
            )
            ok = result.defined[:, :, 0, 0]
            inside = (result.lower[:, :, 0, 0] <= truth) & (truth <= result.upper[:, :, 0, 0])
            hits += int(inside[ok].sum())
            total += int(ok.sum())
        assert 0.82 < hits / total < 0.92
