import warnings
from dataclasses import dataclass, field

import numpy as np
import pytest

from dirconn import DTF, GC, Connectivity, DirconnWarning, TimeSeries
from dirconn.estimator import Estimator, MethodInfo
from dirconn.rng import RNGController
from dirconn.surrogate_test import SurrogateTestResult, surrogate_test
from dirconn.surrogates import AAFT, BlockShuffle, CircularShift, PhaseRandomize, TrialShuffle
from helpers import AbsCorr, coupled, independent

# Most tests use few surrogates for speed; the power check has its own tests below.
pytestmark = pytest.mark.filterwarnings("ignore:n_surrogates=:dirconn.exceptions.DirconnWarning")


@dataclass(frozen=True)
class Noisy(Estimator):
    """Warns on every fit, as a diagnostic would."""

    info = MethodInfo(name="noisy", backend="test stub", directed=False)

    def _fit(self, data: TimeSeries) -> Connectivity:
        warnings.warn("something about the data", DirconnWarning, stacklevel=2)
        values = np.ones((data.n_channels, data.n_channels))
        np.fill_diagonal(values, np.nan)
        return Connectivity.from_array(values, source=data.names, provenance=self._provenance(data))


@dataclass(frozen=True)
class FailsOnSurrogates(Estimator):
    """Defined on the data, NaN on anything else: a surrogate fit that silently degrades."""

    info = MethodInfo(name="fails", backend="test stub", directed=False)
    calls: list[int] = field(default_factory=list)

    def _fit(self, data: TimeSeries) -> Connectivity:
        values = np.ones((data.n_channels, data.n_channels))
        self.calls.append(1)
        if len(self.calls) > 1:  # the first call is the fit on the data itself
            values[0, 1] = np.nan
        np.fill_diagonal(values, np.nan)
        return Connectivity.from_array(values, source=data.names, provenance=self._provenance(data))


def manual_pvalues(estimator, data, generator, n, seed):
    observed = np.abs(estimator.fit(data).values)
    null = np.stack(
        [
            np.abs(estimator.fit(generator.generate_one(data, rng)).values)
            for rng in RNGController(seed).spawn(n)
        ]
    )
    p = (np.sum(null >= observed, axis=0) + 1) / (n + 1)
    # An undefined cell has no p-value (comparing NaN is False, which would give the floor).
    return np.where(np.isnan(observed), np.nan, p)


class TestPValues:
    def test_they_follow_the_rank_formula_on_magnitudes(self):
        data = coupled()
        result = surrogate_test(AbsCorr(), data, PhaseRandomize(), n_surrogates=30, seed=4)
        expected = manual_pvalues(AbsCorr(), data, PhaseRandomize(), 30, 4)
        np.testing.assert_array_equal(result.pvalues, expected)

    def test_they_lie_between_the_floor_and_one(self):
        result = surrogate_test(AbsCorr(), coupled(), PhaseRandomize(), n_surrogates=49, seed=0)
        finite = result.pvalues[np.isfinite(result.pvalues)]
        assert finite.min() >= 1 / 50
        assert finite.max() <= 1.0

    def test_strong_coupling_reaches_the_floor_and_independence_does_not(self):
        n = 99
        result = surrogate_test(AbsCorr(), coupled(), PhaseRandomize(), n_surrogates=n, seed=0)
        assert result.pvalues[0, 1, 0, 0] == pytest.approx(1 / (n + 1))
        assert result.pvalues[0, 2, 0, 0] > 0.05

    def test_the_diagonal_is_undefined_and_everything_derived_from_it_too(self):
        result = surrogate_test(AbsCorr(), coupled(), PhaseRandomize(), n_surrogates=19, seed=0)
        diagonal = np.eye(3, dtype=bool)[:, :, None, None]
        for array in (result.pvalues, result.effect, result.zscore):
            assert np.isnan(array[np.broadcast_to(diagonal, array.shape)]).all()
            assert np.isfinite(array[~np.broadcast_to(diagonal, array.shape)]).all()
        np.testing.assert_array_equal(
            result.defined, ~np.broadcast_to(diagonal, result.pvalues.shape)
        )

    def test_effect_and_zscore_are_measured_against_the_surrogates(self):
        data = coupled()
        n, seed = 40, 2
        result = surrogate_test(AbsCorr(), data, PhaseRandomize(), n_surrogates=n, seed=seed)
        null = np.stack(
            [
                np.abs(AbsCorr().fit(PhaseRandomize().generate_one(data, g)).values)
                for g in RNGController(seed).spawn(n)
            ]
        )
        observed = np.abs(AbsCorr().fit(data).values)
        np.testing.assert_allclose(result.effect, observed - null.mean(axis=0))
        np.testing.assert_allclose(result.zscore, result.effect / null.std(axis=0, ddof=1))

    @pytest.mark.filterwarnings("ignore::dirconn.exceptions.DirconnWarning")
    def test_a_surrogate_distribution_without_spread_gives_no_zscore(self):
        # Noisy returns the constant 1 for every surrogate: effect 0, z undefined, p = 1.
        result = surrogate_test(
            Noisy(), independent(), PhaseRandomize(), n_surrogates=9, seed=0, alpha=0.9
        )
        off = ~np.eye(3, dtype=bool)
        assert np.isnan(result.zscore[off]).all()
        assert (result.effect[off] == 0).all()
        assert (result.pvalues[off] == 1.0).all()

    def test_complex_values_are_compared_by_magnitude(self):
        @dataclass(frozen=True)
        class Phasor(Estimator):
            info = MethodInfo(name="phasor", backend="stub", directed=False, complex_valued=True)

            def _fit(self, data):
                z = np.corrcoef(np.moveaxis(data.data, 1, 0).reshape(data.n_channels, -1)) * 1j
                np.fill_diagonal(z, np.nan)
                return Connectivity.from_array(
                    z, source=data.names, provenance=self._provenance(data)
                )

        result = surrogate_test(Phasor(), coupled(), PhaseRandomize(), n_surrogates=19, seed=0)
        assert np.isrealobj(result.pvalues) and np.isrealobj(result.effect)
        assert result.pvalues[0, 1, 0, 0] == pytest.approx(1 / 20)


class TestReproducibility:
    def test_same_seed_same_result_other_seed_other_result(self):
        data = coupled()
        run = lambda seed: surrogate_test(  # noqa: E731
            AbsCorr(), data, PhaseRandomize(), n_surrogates=29, seed=seed
        )
        np.testing.assert_array_equal(run(1).pvalues, run(1).pvalues)
        np.testing.assert_array_equal(run(1).zscore, run(1).zscore)
        assert not np.array_equal(run(1).zscore, run(2).zscore, equal_nan=True)

    def test_more_surrogates_extend_the_earlier_ones(self):
        # The first 20 surrogates of a 40-surrogate run are those of a 20-surrogate run, so the
        # 20-surrogate p-values count at most as many exceedances as the 40-surrogate ones do.
        data = coupled()
        small = surrogate_test(AbsCorr(), data, PhaseRandomize(), n_surrogates=20, seed=3)
        large = surrogate_test(AbsCorr(), data, PhaseRandomize(), n_surrogates=40, seed=3)
        ok = small.defined
        exceed_small = small.pvalues[ok] * 21 - 1
        exceed_large = large.pvalues[ok] * 41 - 1
        assert (exceed_large >= exceed_small - 1e-9).all()

    def test_the_seed_is_required(self):
        with pytest.raises(TypeError):
            surrogate_test(AbsCorr(), coupled(), PhaseRandomize(), n_surrogates=9)  # type: ignore[call-arg]


@pytest.fixture(scope="module")
def record():
    return surrogate_test(
        AbsCorr(), coupled(), CircularShift(min_shift=5), n_surrogates=19, seed=7, alpha=0.1
    )


class TestResultRecord:
    @pytest.fixture
    def result(self, record):
        return record

    def test_it_states_the_null_hypothesis_and_the_limitations(self, result):
        assert isinstance(result, SurrogateTestResult)
        generator = CircularShift(min_shift=5)
        assert result.null == generator.null
        assert result.limitations == generator.limitations
        assert result.null.strip() and result.limitations

    def test_it_records_how_the_test_was_run(self, result):
        assert result.n_surrogates == 19
        assert result.seed == 7
        assert result.alpha == 0.1
        assert result.generator == CircularShift(min_shift=5).describe()
        assert result.method == "abscorr"

    def test_it_keeps_the_observed_result_with_its_provenance(self, result):
        observed = AbsCorr().fit(coupled())
        np.testing.assert_array_equal(result.observed.values, observed.values)
        assert result.observed.provenance.input_hash == observed.provenance.input_hash

    def test_p_values_carry_the_axes_of_the_observed_result(self, result):
        assert result.pvalues.shape == result.observed.values.shape
        assert result.effect.shape == result.zscore.shape == result.observed.values.shape

    def test_the_summary_names_the_null(self, result):
        assert result.null in repr(result)


class TestPowerCheck:
    def test_it_counts_only_the_defined_cells(self):
        # 3 channels: 6 defined cells, 9 in all. alpha = 0.05 needs N >= 6 / 0.05 - 1 = 119.
        data = independent(n_times=100)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DirconnWarning)
            result = surrogate_test(AbsCorr(), data, PhaseRandomize(), n_surrogates=119, seed=0)
        assert result.checks.get("surrogate_power").status == "pass"
        with pytest.warns(DirconnWarning, match="n_surrogates"):
            result = surrogate_test(AbsCorr(), data, PhaseRandomize(), n_surrogates=118, seed=0)
        assert result.checks.get("surrogate_power").status == "warn"

    def test_it_counts_the_cells_of_the_reduced_result(self):
        data = coupled(n_times=400)
        full = DTF(order=2)
        with pytest.warns(DirconnWarning, match="n_surrogates"):
            wide = surrogate_test(full, data, PhaseRandomize(), n_surrogates=19, seed=0)
        assert wide.pvalues.shape[2] > 1
        with warnings.catch_warnings():
            warnings.simplefilter("error", DirconnWarning)
            # one band: 6 defined cells need 119 surrogates at alpha 0.05; use alpha 0.5.
            narrow = surrogate_test(
                full,
                data,
                PhaseRandomize(),
                n_surrogates=19,
                seed=0,
                reduce=(10.0, 20.0),
                alpha=0.5,
            )
        assert narrow.pvalues.shape[2] == 1
        assert narrow.observed.is_frequency_resolved


class TestDiagnosticsOfSurrogateFits:
    def test_repeated_warnings_from_the_surrogate_fits_are_summarised_once(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = surrogate_test(
                Noisy(), independent(), PhaseRandomize(), n_surrogates=25, seed=0, alpha=0.9
            )
        messages = [str(w.message) for w in caught if issubclass(w.category, DirconnWarning)]
        # one from the fit on the data itself, one summary for the 25 surrogate fits
        assert len(messages) == 2
        assert any("25 of 25" in m for m in messages)
        assert result.checks.get("surrogate_fits").status == "warn"

    def test_clean_surrogate_fits_leave_a_passing_record(self):
        result = surrogate_test(
            AbsCorr(), independent(), PhaseRandomize(), n_surrogates=9, seed=0, alpha=0.9
        )
        assert result.checks.get("surrogate_fits").status == "pass"
        assert result.checks.get("surrogate_generation").status == "pass"

    def test_generator_warnings_are_summarised_and_recorded(self):
        from dirconn.surrogates import IAAFT

        with pytest.warns(DirconnWarning, match="converge") as caught:
            result = surrogate_test(
                AbsCorr(), coupled(), IAAFT(n_iter=1), n_surrogates=9, seed=0, alpha=0.9
            )
        assert sum("converge" in str(w.message) for w in caught) == 1
        assert result.checks.get("surrogate_generation").status == "warn"

    def test_a_surrogate_that_loses_a_cell_is_an_error_not_a_pvalue(self):
        with pytest.raises(ValueError, match="undefined"):
            surrogate_test(FailsOnSurrogates(), coupled(), PhaseRandomize(), n_surrogates=9, seed=0)


class TestArguments:
    @pytest.mark.parametrize("n", [0, -1, 2.5, True])
    def test_the_number_of_surrogates_must_be_a_positive_integer(self, n):
        with pytest.raises((TypeError, ValueError)):
            surrogate_test(AbsCorr(), coupled(), PhaseRandomize(), n_surrogates=n, seed=0)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 2.0, float("nan")])
    def test_alpha_must_lie_strictly_between_zero_and_one(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            surrogate_test(
                AbsCorr(), coupled(), PhaseRandomize(), n_surrogates=9, seed=0, alpha=alpha
            )

    def test_the_generator_must_be_a_surrogate_generator(self):
        with pytest.raises(TypeError, match="SurrogateGenerator"):
            surrogate_test(AbsCorr(), coupled(), "phase", n_surrogates=9, seed=0)  # type: ignore[arg-type]

    def test_the_estimator_must_be_an_estimator(self):
        with pytest.raises(TypeError, match="Estimator"):
            surrogate_test(lambda d: d, coupled(), PhaseRandomize(), n_surrogates=9, seed=0)  # type: ignore[arg-type]

    def test_the_data_must_be_a_timeseries(self):
        with pytest.raises(TypeError, match="TimeSeries"):
            surrogate_test(AbsCorr(), np.zeros((3, 10)), PhaseRandomize(), n_surrogates=9, seed=0)  # type: ignore[arg-type]

    def test_a_generator_that_cannot_run_on_the_data_raises_its_own_error(self):
        data = TimeSeries.from_array(np.random.default_rng(0).standard_normal((3, 200)), 100.0)
        with pytest.raises(ValueError, match="at least 2 trials"):
            surrogate_test(AbsCorr(), data, TrialShuffle(), n_surrogates=9, seed=0)


class TestOnTheRealEstimators:
    def test_gc_finds_the_driven_direction_with_a_driver_and_not_in_reverse(self):
        rng = np.random.default_rng(1)
        n_trials, n_times = 3, 400
        x = rng.standard_normal((n_trials, 3, n_times))
        for t in range(1, n_times):
            x[:, 1, t] += 0.6 * x[:, 0, t - 1]
        data = TimeSeries.from_array(x, 100.0, segmentation="epoched")
        result = surrogate_test(GC(order=2), data, AAFT(), n_surrogates=99, seed=0, alpha=0.5)
        assert result.pvalues[0, 1, 0, 0] == pytest.approx(1 / 100)
        assert result.pvalues[1, 0, 0, 0] > 0.05
        assert result.pvalues[2, 0, 0, 0] > 0.05


class TestTypeIErrorCalibration:
    """Under a true null the p-values are uniform.

    Full experiment (scripts/calibrate_surrogate_test.py; 100 datasets, 39 surrogates, 200 p-values
    per generator, GC of order 2 on independent AR(1) channels): rejections at 0.05 were 5
    (phase_randomize) and 7 (circular_shift) against 10 expected, at 0.10 they were 19 and 16
    against 20, and the mean p-value was 0.497 and 0.495. This is the reduced version for the fast
    suite: with 19 surrogates the p-values are 1/20, ..., 1 and average 0.525 under the null; with
    100 of them the mean has a standard deviation of about 0.03, so 0.4 to 0.6 is between four
    deviations below and two and a half above.
    """

    @pytest.mark.parametrize("generator", [PhaseRandomize(), CircularShift(min_shift=10)])
    def test_independent_channels_give_uniform_pvalues(self, generator):
        p = []
        for seed in range(50):
            rng = np.random.default_rng(seed)
            x = rng.standard_normal((2, 2, 150))
            for t in range(1, 150):
                x[..., t] += 0.5 * x[..., t - 1]
            data = TimeSeries.from_array(x, 100.0, segmentation="epoched")
            result = surrogate_test(
                GC(order=2), data, generator, n_surrogates=19, seed=seed, alpha=0.9
            )
            p.extend(result.pvalues[~np.eye(2, dtype=bool)].ravel())
        p = np.asarray(p)
        assert 0.4 < p.mean() < 0.6
        # With 19 surrogates p <= 0.1 happens with probability exactly 0.1 under the null.
        assert (p <= 0.1).mean() < 0.25


class TestBlockShuffleInATest:
    @staticmethod
    def continuous_pair(seed=0, n_times=600, coupled_pair=False):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((3, n_times))
        for t in range(1, n_times):
            x[:, t] += 0.5 * x[:, t - 1]
        if coupled_pair:
            x[1] = 0.8 * x[0] + 0.6 * x[1]
        return TimeSeries.from_array(x, 100.0)

    def test_the_observed_result_is_computed_on_the_same_blocks_as_the_surrogates(self):
        from helpers import Spy

        spy = Spy()
        surrogate_test(
            spy, self.continuous_pair(), BlockShuffle(block_size=50), n_surrogates=4, seed=0
        )
        assert len(spy.seen) == 1 + 4
        assert all((d.n_trials, d.n_times) == (12, 50) for d in spy.seen)

    def test_a_coupled_pair_reaches_the_floor_and_independent_pairs_do_not(self):
        data = self.continuous_pair(coupled_pair=True)
        result = surrogate_test(
            AbsCorr(), data, BlockShuffle(block_size=50), n_surrogates=99, seed=0, alpha=0.9
        )
        assert result.pvalues[0, 1, 0, 0] == pytest.approx(1 / 100)
        assert result.pvalues[0, 2, 0, 0] > 0.05
        assert result.generator["params"] == {"block_size": 50}

    def test_the_whole_test_is_reproducible(self):
        run = lambda: surrogate_test(  # noqa: E731
            AbsCorr(),
            self.continuous_pair(),
            BlockShuffle(block_size=50),
            n_surrogates=19,
            seed=2,
            alpha=0.9,
        )
        np.testing.assert_array_equal(run().pvalues, run().pvalues)

    def test_independent_channels_give_uniform_pvalues(self):
        # Full experiment: 150 datasets, GC of order 2, 600 samples in blocks of 50, 19
        # surrogates, 300 p-values: mean 0.526 (0.525 expected), share <= 0.10 was 0.12 and
        # share <= 0.05 was 0.053.
        p = []
        for seed in range(60):
            data = TimeSeries.from_array(self.continuous_pair(seed).data[0, :2], 100.0)
            result = surrogate_test(
                GC(order=2),
                data,
                BlockShuffle(block_size=50),
                n_surrogates=19,
                seed=seed,
                alpha=0.9,
            )
            p.extend(result.pvalues[~np.eye(2, dtype=bool)].ravel())
        p = np.asarray(p)
        assert 0.4 < p.mean() < 0.65
        assert (p <= 0.1).mean() < 0.25


class TestPreparedData:
    @staticmethod
    def continuous():
        x = np.random.default_rng(0).standard_normal((3, 600))
        return TimeSeries.from_array(x, 100.0)

    def test_the_provenance_records_that_the_data_were_cut_into_blocks(self):
        from dirconn.provenance import array_hash

        data = self.continuous()
        result = surrogate_test(
            AbsCorr(), data, BlockShuffle(block_size=50), n_surrogates=5, seed=0, alpha=0.9
        )
        step = result.observed.provenance.history[-1]
        assert step["step"] == "prepare" and step["generator"] == "block_shuffle"
        assert step["params"] == {"block_size": 50}
        # the hash of the caller's own input is recoverable even though the fit saw the blocks
        assert step["original_input_hash"] == array_hash(data.data)
        assert result.observed.provenance.input_hash != array_hash(data.data)

    def test_generators_that_keep_the_data_add_no_step(self):
        result = surrogate_test(
            AbsCorr(), coupled(), PhaseRandomize(), n_surrogates=5, seed=0, alpha=0.9
        )
        assert all(s.get("step") != "prepare" for s in result.observed.provenance.history)

    def test_the_data_must_be_a_timeseries_before_any_generator_touches_them(self):
        with pytest.raises(TypeError, match="TimeSeries"):
            surrogate_test(
                AbsCorr(),
                np.zeros((3, 600)),  # type: ignore[arg-type]
                BlockShuffle(block_size=50),
                n_surrogates=5,
                seed=0,
            )
