import warnings

import numpy as np
import pytest

import dirconn.order_stability as order_stability
from dirconn import DTF, GC, GPDC, PDC, DirconnWarning, TimeSeries, surrogate_test
from dirconn.benchmark import constructed_pair, evaluate_pair
from dirconn.order_stability import candidate_orders, order_stability_check
from dirconn.surrogates import PhaseRandomize
from helpers import SFREQ, fake_recording


def reversed_pair(strength=3.0):
    """Source drives target through an 8 to 12 Hz influence on a 1/f background."""
    series, _ = constructed_pair(
        fake_recording(),
        source_channel="E0",
        noise_channel="E3",
        source_start=0.0,
        noise_start=100.0,
        duration=40.0,
        delay_ms=20.0,
        strength=strength,
    )
    return series


def white(n_times=6000, n_channels=2, seed=0, n_trials=1):
    x = np.random.default_rng(seed).standard_normal((n_trials, n_channels, n_times))
    return TimeSeries.from_array(x, SFREQ, segmentation="epoched" if n_trials > 1 else "continuous")


class TestCandidateOrders:
    def test_it_is_a_ladder_of_multiples_of_the_order(self):
        # 128 * 2 channels = 256 regressors exceeds the limit of 160
        assert candidate_orders(4, n_channels=2, n_times=6000, n_trials=1) == [8, 16, 32, 64]

    def test_the_sampling_rule_limits_it(self):
        # at most one coefficient per ten observations: k * 2 channels <= 0.1 * (600 - k)
        assert candidate_orders(4, n_channels=2, n_times=600, n_trials=1) == [8, 16]

    def test_the_size_of_the_regression_limits_it(self):
        # 160 * 2 channels = 320 regressors per equation exceeds the limit of 160
        assert candidate_orders(10, n_channels=2, n_times=10**6, n_trials=1) == [20, 40, 80]

    def test_trials_count_as_observations_but_never_cross_a_boundary(self):
        one = candidate_orders(4, n_channels=2, n_times=600, n_trials=1)
        many = candidate_orders(4, n_channels=2, n_times=600, n_trials=10)
        assert len(many) >= len(one)
        assert all(k < 600 for k in many)

    def test_nothing_fits_in_short_data(self):
        assert candidate_orders(8, n_channels=2, n_times=150, n_trials=1) == []


class TestTheCheck:
    """Measured on real EEG (24 to 30 random channel pairs, alpha 0.001, ladder to 32 times).

    Independent pairs were flagged 0 of 30 times. Where the direction at the chosen order was
    wrong, the check flagged it in 13 of 17 (order 4) and 17 of 23 (order 8) cases at strength 1 and
    6 of 10 and 14 of 19 at strength 3; of the 90 cases where the direction was right it flagged 1.
    """

    def test_a_reversed_direction_is_flagged_with_the_pair_and_the_order_to_use(self):
        series = reversed_pair()
        with pytest.warns(DirconnWarning, match="direction") as caught:
            check = order_stability_check(series, 8)
        assert check.name == "order_stability" and check.status == "warn"
        assert check.value >= 1
        text = str(caught[0].message)
        assert "source" in text and "target" in text
        assert "at least" in text  # the order at which the opposite direction appears

    def test_a_long_enough_model_passes_quietly(self):
        series = reversed_pair()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DirconnWarning)
            check = order_stability_check(series, 32)
        assert check.status == "pass" and check.value == 0

    def test_the_longest_order_that_can_be_checked_cannot_be_called_stable(self):
        # order 64 with two channels needs order 128 for the comparison: 256 regressors, too many
        check = order_stability_check(reversed_pair(), 64)
        assert check.status == "not_checked" and "longer" in check.detail

    @pytest.mark.parametrize("order", [2, 8])
    def test_independent_channels_are_not_flagged(self, order):
        flagged = 0
        for seed in range(20):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DirconnWarning)
                flagged += order_stability_check(white(seed=seed), order).status == "warn"
        assert flagged == 0

    def test_short_data_are_reported_as_not_checked_not_as_stable(self):
        series = white(n_times=150)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DirconnWarning)
            check = order_stability_check(series, 8)
        assert check.status == "not_checked"
        assert "longer" in check.detail

    def test_epoched_data_are_pooled(self):
        series = white(n_times=400, n_trials=8)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DirconnWarning)
            check = order_stability_check(series, 4)
        assert check.status in ("pass", "not_checked")

    def test_the_check_never_changes_the_input(self):
        series = reversed_pair()
        before = series.data.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DirconnWarning)
            order_stability_check(series, 8)
        np.testing.assert_array_equal(series.data, before)

    @pytest.mark.parametrize("order", [0, -1, 2.5, True])
    def test_the_order_must_be_a_positive_integer(self, order):
        with pytest.raises((TypeError, ValueError)):
            order_stability_check(reversed_pair(), order)


class TestInTheEstimators:
    @pytest.mark.parametrize("cls", [GC, DTF, PDC, GPDC])
    def test_it_runs_by_default_and_is_recorded_in_the_checklist(self, cls):
        with pytest.warns(DirconnWarning, match="direction"):
            result = cls(order=8).fit(reversed_pair())
        check = result.provenance.assumptions.get("order_stability")
        assert check.status == "warn" and result.provenance.assumptions.has_warnings

    @pytest.mark.parametrize("cls", [GC, DTF, PDC, GPDC])
    def test_it_can_be_switched_off(self, cls):
        with warnings.catch_warnings():
            # the residuals at order 8 are not white (largest autocorrelation 0.11), a warning of
            # its own; only the direction warning of the order check must stay away
            warnings.simplefilter("ignore", DirconnWarning)
            warnings.filterwarnings("error", message=".*direction", category=DirconnWarning)
            result = cls(order=8, check_order_stability=False).fit(reversed_pair())
        with pytest.raises(KeyError):
            result.provenance.assumptions.get("order_stability")

    @pytest.mark.parametrize("cls", [GC, DTF, PDC, GPDC])
    def test_the_switch_must_be_a_bool(self, cls):
        with pytest.raises(TypeError, match="check_order_stability"):
            cls(order=8, check_order_stability="yes")  # type: ignore[arg-type]

    def test_a_correct_direction_leaves_a_passing_record(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DirconnWarning)
            result = GC(order=32).fit(reversed_pair())
        assert result.provenance.assumptions.get("order_stability").status == "pass"

    def test_the_diagnostic_names_are_collected_by_the_pair_evaluation(self):
        series = reversed_pair()
        truth = np.array([[np.nan, 1.0], [0.0, np.nan]])
        out = evaluate_pair(GC(order=8), series, truth)
        assert "order_stability" in out["diagnostics"]
        assert out["direction_accuracy"] == 0.0  # and the direction really is wrong


class TestResamplingStaysCheap:
    def test_the_extra_fits_are_made_for_the_data_only_not_for_every_surrogate(self, monkeypatch):
        calls = []
        real = order_stability.order_stability_check

        def spy(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(order_stability, "order_stability_check", spy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DirconnWarning)
            surrogate_test(
                GC(order=8), reversed_pair(), PhaseRandomize(), n_surrogates=5, seed=0, alpha=0.9
            )
        assert len(calls) == 1
