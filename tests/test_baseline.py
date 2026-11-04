import warnings

import numpy as np
import pytest

from dirconn import Connectivity, DirconnWarning, TimeSeries, baseline_normalise, bootstrap
from helpers import MeanDifference, epochs


def pair(active_means=(0.0, 1.0, 3.0), baseline_means=(0.0, 1.0, 0.5), n_trials=25):
    return (
        epochs(n_trials=n_trials, means=active_means, seed=1),
        epochs(n_trials=n_trials, means=baseline_means, seed=2),
    )


def magnitudes(data):
    return np.abs(MeanDifference().fit(data).values)


class TestDifferenceAndRatio:
    def test_difference_is_active_minus_baseline_in_magnitude(self):
        active, baseline = pair()
        result = baseline_normalise(MeanDifference(), active, baseline, how="difference")
        expected = magnitudes(active) - magnitudes(baseline)
        np.testing.assert_allclose(result.values[:, :, 0, 0], expected[:, :, 0, 0])

    def test_ratio_is_active_over_baseline_in_magnitude(self):
        active, baseline = pair()
        result = baseline_normalise(MeanDifference(), active, baseline, how="ratio")
        expected = magnitudes(active) / magnitudes(baseline)
        np.testing.assert_allclose(result.values[:, :, 0, 0], expected[:, :, 0, 0])

    def test_a_cell_equal_to_the_baseline_gives_zero_and_one(self):
        active, _ = pair()
        diff = baseline_normalise(MeanDifference(), active, active, how="difference")
        ratio = baseline_normalise(MeanDifference(), active, active, how="ratio")
        off = ~np.eye(3, dtype=bool)
        assert (diff.values[off][:, 0, 0] == 0).all()
        assert (ratio.values[off][:, 0, 0] == 1).all()

    def test_a_zero_baseline_makes_the_ratio_undefined_with_a_warning(self):
        # A constant baseline has identical channel means, so every magnitude is exactly zero.
        n_times = 30
        flat = TimeSeries.from_array(np.zeros((6, 3, n_times)), 100.0, segmentation="epoched")
        active, _ = pair()
        with pytest.warns(DirconnWarning, match="zero"):
            result = baseline_normalise(MeanDifference(), active, flat, how="ratio")
        off = ~np.eye(3, dtype=bool)
        assert np.isnan(result.values[off][:, 0, 0]).all()

    def test_the_diagonal_stays_undefined(self):
        active, baseline = pair()
        for how in ("difference", "ratio"):
            result = baseline_normalise(MeanDifference(), active, baseline, how=how)
            assert np.isnan(np.diagonal(result.values[:, :, 0, 0])).all()


class TestZScore:
    def test_it_scales_by_the_bootstrap_spread_of_the_baseline(self):
        active, baseline = pair()
        result = baseline_normalise(
            MeanDifference(), active, baseline, how="zscore", n_boot=80, seed=3
        )
        spread = bootstrap(MeanDifference(), baseline, n_boot=80, seed=3).std
        expected = (magnitudes(active) - magnitudes(baseline)) / spread
        ok = ~np.isnan(spread)
        np.testing.assert_allclose(result.values[ok], expected[ok])

    def test_it_needs_a_bootstrap_size_and_a_seed(self):
        active, baseline = pair()
        with pytest.raises(ValueError, match="n_boot"):
            baseline_normalise(MeanDifference(), active, baseline, how="zscore", seed=0)
        with pytest.raises(ValueError, match="seed"):
            baseline_normalise(MeanDifference(), active, baseline, how="zscore", n_boot=10)

    @pytest.mark.parametrize("how", ["difference", "ratio"])
    def test_the_bootstrap_arguments_are_refused_when_not_used(self, how):
        active, baseline = pair()
        with pytest.raises(ValueError, match="zscore"):
            baseline_normalise(MeanDifference(), active, baseline, how=how, n_boot=10, seed=0)

    def test_a_continuous_baseline_needs_a_block_size(self):
        active = TimeSeries.from_array(np.random.default_rng(0).standard_normal((3, 1000)), 100.0)
        baseline = TimeSeries.from_array(np.random.default_rng(1).standard_normal((3, 1000)), 100.0)
        with pytest.raises(ValueError, match="block_size"):
            baseline_normalise(MeanDifference(), active, baseline, how="zscore", n_boot=10, seed=0)
        result = baseline_normalise(
            MeanDifference(), active, baseline, how="zscore", n_boot=10, seed=0, block_size=50
        )
        assert result.values.shape == (3, 3, 1, 1)

    def test_the_bootstrap_warnings_about_few_trials_come_through(self):
        active, baseline = pair(n_trials=8)
        with pytest.warns(DirconnWarning, match="at least 20"):
            baseline_normalise(MeanDifference(), active, baseline, how="zscore", n_boot=10, seed=0)


class TestTheReturnedConnectivity:
    def test_it_is_a_connectivity_with_the_axes_of_the_active_result(self):
        active, baseline = pair()
        result = baseline_normalise(MeanDifference(), active, baseline)
        assert isinstance(result, Connectivity)
        assert result.source == MeanDifference().fit(active).source
        assert result.values.shape == (3, 3, 1, 1)
        assert np.isrealobj(result.values)

    def test_the_provenance_records_the_normalisation(self):
        active, baseline = pair()
        result = baseline_normalise(MeanDifference(), active, baseline, how="ratio")
        step = result.provenance.history[-1]
        assert step["step"] == "baseline_normalise" and step["how"] == "ratio"
        assert step["baseline_input_hash"] != result.provenance.input_hash
        assert result.provenance.method == "meandiff"

    def test_it_reduces_both_results_the_same_way(self):
        from dirconn import DTF
        from helpers import coupled

        active, baseline = (
            coupled(n_trials=10, n_times=200),
            coupled(n_trials=10, n_times=200, seed=5),
        )
        result = baseline_normalise(
            DTF(order=2), active, baseline, how="difference", reduce=(10.0, 20.0)
        )
        assert result.values.shape[2] == 1 and result.is_frequency_resolved


class TestArguments:
    def test_an_unknown_normalisation_names_the_choices(self):
        active, baseline = pair()
        with pytest.raises(ValueError, match="difference"):
            baseline_normalise(MeanDifference(), active, baseline, how="percent")

    def test_the_channels_and_rate_must_match(self):
        active, baseline = pair()
        renamed = TimeSeries.from_array(
            baseline.data, baseline.sfreq, names=["x", "y", "z"], segmentation="epoched"
        )
        faster = TimeSeries.from_array(baseline.data, 200.0, segmentation="epoched")
        with pytest.raises(ValueError, match="channels"):
            baseline_normalise(MeanDifference(), active, renamed)
        with pytest.raises(ValueError, match="sfreq"):
            baseline_normalise(MeanDifference(), active, faster)

    def test_the_windows_may_differ_in_length(self):
        active = epochs(n_times=60)
        baseline = epochs(n_times=30, seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DirconnWarning)
            result = baseline_normalise(MeanDifference(), active, baseline)
        assert np.isfinite(result.values[~np.eye(3, dtype=bool)]).all()

    def test_types_are_checked(self):
        active, baseline = pair()
        with pytest.raises(TypeError, match="Estimator"):
            baseline_normalise(lambda d: d, active, baseline)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="TimeSeries"):
            baseline_normalise(MeanDifference(), np.zeros((3, 10)), baseline)  # type: ignore[arg-type]
