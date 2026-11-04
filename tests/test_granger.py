import dataclasses
import warnings

import numpy as np
import pytest
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.vector_ar.var_model import VAR

from dirconn import (
    GC,
    Connectivity,
    DirconnWarning,
    Edge,
    Network,
    NodeDynamics,
    TimeSeries,
    dk_node,
    gc,
    simulate,
)


def make_series(coefs, n_times, *, n_trials=1, seed=0, names=None):
    """Simulate ``x[t] = sum_k coefs[k - 1] @ x[t - k] + e[t]`` with unit-variance innovations."""
    rng = np.random.default_rng(seed)
    order, n, _ = coefs.shape
    out = np.zeros((n_trials, n, n_times))
    for trial in range(n_trials):
        x = np.zeros((n, n_times + 200))
        noise = rng.standard_normal(x.shape)
        for t in range(order, x.shape[1]):
            x[:, t] = noise[:, t] + sum(coefs[k] @ x[:, t - k - 1] for k in range(order))
        out[trial] = x[:, 200:]
    segmentation = "continuous" if n_trials == 1 else "epoched"
    return TimeSeries.from_array(out, 100.0, names=names, segmentation=segmentation)


def three_channel_coefs():
    coefs = np.zeros((2, 3, 3))
    coefs[0] = [[0.5, 0.0, 0.0], [0.4, 0.3, 0.0], [0.0, 0.3, 0.2]]  # 0 -> 1 -> 2
    coefs[1] = [[-0.2, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, -0.1]]
    return coefs


def reference_gc(ts, order, *, conditional):
    """Granger causality written from the definition with explicit least-squares fits."""
    x = ts.data - ts.data.mean(axis=(0, 2), keepdims=True)
    n = x.shape[1]

    def rows(channels):
        ys, xs = [], []
        for trial in x:
            for t in range(order, trial.shape[1]):
                ys.append(trial[channels, t])
                xs.append(np.concatenate([trial[channels, t - k] for k in range(1, order + 1)]))
        return np.array(ys), np.array(xs)

    def residual_ss(channels, target):
        y, design = rows(channels)
        column = channels.index(target)
        beta = np.linalg.lstsq(design, y[:, column], rcond=None)[0]
        return np.sum((y[:, column] - design @ beta) ** 2)

    values = np.full((n, n), np.nan)
    for source in range(n):
        for target in range(n):
            if source == target:
                continue
            if conditional:
                everyone = list(range(n))
                others = [c for c in everyone if c != source]
                values[source, target] = np.log(
                    residual_ss(others, target) / residual_ss(everyone, target)
                )
            else:
                pair = [source, target]
                values[source, target] = np.log(
                    residual_ss([target], target) / residual_ss(pair, target)
                )
    return values


class TestClosedForm:
    def test_matches_the_population_value_of_a_simple_system(self):
        # x[t] = a x[t-1] + e, y[t] = c x[t-1] + e': with order 1 the reduced model of y uses only
        # y[t-1]. var(x) = 1/(1 - a^2), var(y) = 1 + c^2 var(x), cov(y[t], y[t-1]) = c^2 a var(x),
        # so the reduced residual variance is var(y) - cov^2 / var(y) and the full one is 1.
        a, c = 0.6, 0.8
        var_x = 1.0 / (1.0 - a**2)
        var_y = 1.0 + c**2 * var_x
        reduced = var_y - (c**2 * a * var_x) ** 2 / var_y
        coefs = np.array([[[a, 0.0], [c, 0.0]]])
        result = GC(order=1).fit(make_series(coefs, 300000, seed=1))
        assert result.values[0, 1, 0, 0] == pytest.approx(np.log(reduced), abs=0.02)
        assert result.values[1, 0, 0, 0] == pytest.approx(0.0, abs=0.01)


class TestAgainstIndependentReferences:
    @pytest.mark.parametrize("conditional", [True, False])
    def test_matches_a_hand_written_regression_reference(self, conditional):
        ts = make_series(three_channel_coefs(), 400, n_trials=3, seed=2)
        result = GC(order=2, conditional=conditional).fit(ts)
        np.testing.assert_allclose(
            result.values[:, :, 0, 0],
            reference_gc(ts, 2, conditional=conditional),
            rtol=1e-9,
            equal_nan=True,
        )

    def test_conditional_gc_matches_statsmodels_var_fits(self):
        # The models have no intercept, so the innovation variance is the mean squared residual
        # (zero-mean maximum likelihood), not np.var, which would remove the residual mean.
        ts = make_series(three_channel_coefs(), 1200, seed=3)
        x = ts.data[0].T - ts.data[0].T.mean(axis=0)
        full = VAR(x).fit(maxlags=2, ic=None, trend="n")
        expected = np.full((3, 3), np.nan)
        for source in range(3):
            others = [c for c in range(3) if c != source]
            reduced = VAR(x[:, others]).fit(maxlags=2, ic=None, trend="n")
            for target in others:
                reduced_variance = np.mean(reduced.resid[:, others.index(target)] ** 2)
                expected[source, target] = np.log(
                    reduced_variance / np.mean(full.resid[:, target] ** 2)
                )
        result = GC(order=2).fit(ts)
        np.testing.assert_allclose(result.values[:, :, 0, 0], expected, rtol=1e-8, equal_nan=True)

    def test_pairwise_gc_matches_statsmodels_bivariate_and_univariate_fits(self):
        ts = make_series(three_channel_coefs(), 1200, seed=4)
        x = ts.data[0].T - ts.data[0].T.mean(axis=0)
        expected = np.full((3, 3), np.nan)
        for source in range(3):
            for target in range(3):
                if source == target:
                    continue
                pair = VAR(x[:, [source, target]]).fit(maxlags=2, ic=None, trend="n")
                alone = AutoReg(x[:, target], lags=2, trend="n").fit()
                expected[source, target] = np.log(
                    np.mean(alone.resid**2) / np.mean(pair.resid[:, 1] ** 2)
                )
        result = GC(order=2, conditional=False).fit(ts)
        np.testing.assert_allclose(result.values[:, :, 0, 0], expected, rtol=1e-8, equal_nan=True)

    def test_with_two_channels_the_modes_agree(self):
        ts = make_series(three_channel_coefs()[:, :2, :2], 800, seed=5)
        conditional = GC(order=2, conditional=True).fit(ts)
        pairwise = GC(order=2, conditional=False).fit(ts)
        np.testing.assert_allclose(conditional.values, pairwise.values, rtol=1e-12, equal_nan=True)


class TestOutput:
    def test_the_result_is_a_broadband_directed_matrix_with_an_undefined_diagonal(self):
        ts = make_series(three_channel_coefs(), 600, seed=6, names=["a", "b", "c"])
        result = GC(order=2).fit(ts)
        assert isinstance(result, Connectivity)
        assert result.values.shape == (3, 3, 1, 1)
        assert result.source == result.target == ("a", "b", "c")
        assert not result.is_frequency_resolved and not result.is_time_resolved
        assert np.isnan(np.diag(result.values[:, :, 0, 0])).all()

    def test_values_are_never_negative_for_nested_least_squares_models(self):
        ts = make_series(three_channel_coefs(), 300, n_trials=2, seed=7)
        for conditional in (True, False):
            off_diagonal = GC(order=2, conditional=conditional).fit(ts).values[:, :, 0, 0]
            assert np.nanmin(off_diagonal) >= -1e-12

    def test_direction_follows_the_coupling(self):
        ts = make_series(three_channel_coefs(), 20000, seed=8)
        matrix = GC(order=2).fit(ts).values[:, :, 0, 0]
        assert matrix[0, 1] > 10 * matrix[1, 0]
        assert matrix[1, 2] > 10 * matrix[2, 1]

    def test_scaling_one_channel_changes_nothing(self):
        ts = make_series(three_channel_coefs(), 500, seed=9)
        scaled = TimeSeries.from_array(
            ts.data * np.array([1.0, 10.0, 0.1])[None, :, None], 100.0, segmentation="continuous"
        )
        np.testing.assert_allclose(
            GC(order=2).fit(ts).values, GC(order=2).fit(scaled).values, rtol=1e-6, equal_nan=True
        )

    def test_permuting_the_channels_permutes_the_matrix(self):
        ts = make_series(three_channel_coefs(), 500, seed=10)
        permutation = [2, 0, 1]
        permuted = ts.pick(permutation)
        original = GC(order=2).fit(ts).values[:, :, 0, 0]
        result = GC(order=2).fit(permuted).values[:, :, 0, 0]
        np.testing.assert_allclose(
            result, original[np.ix_(permutation, permutation)], rtol=1e-9, equal_nan=True
        )


class TestOrder:
    def test_an_integer_order_is_used_as_it_is(self):
        ts = make_series(three_channel_coefs(), 800, seed=11)
        result = GC(order=3).fit(ts)
        assert result.provenance.model_order == 3
        assert result.provenance.details["order_selection"] is None

    def test_a_criterion_name_selects_the_order_once_on_the_full_model(self):
        ts = make_series(three_channel_coefs(), 6000, seed=12)
        result = GC(order="bic", max_order=6).fit(ts)
        assert result.provenance.model_order == 2
        selection = result.provenance.details["order_selection"]
        assert selection["criterion"] == "bic"
        assert selection["orders"] == [1, 2, 3, 4, 5, 6]
        assert len(selection["bic"]) == 6 and selection["cv"] is None

    def test_the_default_is_bic_with_ten_candidate_orders(self):
        params = GC().get_params()
        assert params["order"] == "bic" and params["max_order"] == 10

    def test_cross_validation_is_available_and_recorded(self):
        ts = make_series(three_channel_coefs(), 3000, seed=13)
        result = GC(order="cv", max_order=4, n_folds=4).fit(ts)
        selection = result.provenance.details["order_selection"]
        assert selection["criterion"] == "cv"
        assert selection["cv_scheme"] == "blocks" and selection["n_folds"] == 4
        assert len(selection["cv"]) == 4

    def test_pairwise_mode_uses_the_same_single_order(self):
        ts = make_series(three_channel_coefs(), 3000, seed=14)
        conditional = GC(order="bic", max_order=5).fit(ts)
        pairwise = GC(order="bic", max_order=5, conditional=False).fit(ts)
        assert pairwise.provenance.model_order == conditional.provenance.model_order

    def test_max_order_bounds_the_search(self):
        ts = make_series(three_channel_coefs(), 3000, seed=15)
        assert GC(order="aic", max_order=1).fit(ts).provenance.model_order == 1


class TestProvenance:
    def test_the_result_records_how_it_was_made(self):
        ts = make_series(three_channel_coefs(), 600, seed=16)
        prov = GC(order=2, ridge=0.0).fit(ts).provenance
        assert prov.method == "gc"
        assert "OLS" in prov.backend
        assert prov.parameters["conditional"] is True
        assert prov.parameters["order"] == 2
        assert prov.details["n_obs"] == 600 - 2
        assert prov.details["conditional"] is True
        assert prov.claim == "predictive, not interventional"

    def test_the_sampling_adequacy_check_is_filled_in(self):
        ts = make_series(three_channel_coefs(), 3000, seed=17)
        check = GC(order=2).fit(ts).provenance.assumptions.get("sampling_adequacy")
        assert check.status == "pass"
        assert check.value == pytest.approx(3 * 2 / (3000 - 2))

    def test_pairwise_mode_counts_the_coefficients_of_a_bivariate_model(self):
        ts = make_series(three_channel_coefs(), 3000, seed=17)
        prov = GC(order=2, conditional=False).fit(ts).provenance
        assert prov.assumptions.get("sampling_adequacy").value == pytest.approx(2 * 2 / (3000 - 2))

    def test_too_many_coefficients_warn_and_are_recorded(self):
        rng = np.random.default_rng(18)
        ts = TimeSeries.from_array(rng.standard_normal((6, 120)), 100.0)
        with pytest.warns(DirconnWarning, match="coefficients"):
            result = GC(order=3).fit(ts)
        assert result.provenance.assumptions.get("sampling_adequacy").status == "warn"
        assert result.provenance.assumptions.has_warnings

    def test_the_input_hash_identifies_the_data(self):
        ts = make_series(three_channel_coefs(), 600, seed=19)
        other = make_series(three_channel_coefs(), 600, seed=20)
        assert (
            GC(order=2).fit(ts).provenance.input_hash == GC(order=2).fit(ts).provenance.input_hash
        )
        assert (
            GC(order=2).fit(ts).provenance.input_hash
            != GC(order=2).fit(other).provenance.input_hash
        )


class TestFailures:
    def test_a_duplicated_channel_raises_and_names_the_problem(self):
        data = make_series(three_channel_coefs(), 400, seed=21).data
        ts = TimeSeries.from_array(
            np.concatenate([data, data[:, :1]], axis=1), 100.0, segmentation="continuous"
        )
        with pytest.raises(ValueError, match="rank"):
            GC(order=2).fit(ts)

    def test_ridge_makes_collinear_channels_estimable(self):
        data = make_series(three_channel_coefs(), 400, seed=21).data
        ts = TimeSeries.from_array(
            np.concatenate([data, data[:, :1]], axis=1), 100.0, segmentation="continuous"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = GC(order=2, ridge=1e-3).fit(ts)
        off_diagonal = ~np.eye(4, dtype=bool)
        assert np.isfinite(result.values[:, :, 0, 0][off_diagonal]).all()
        assert np.isnan(np.diag(result.values[:, :, 0, 0])).all()

    def test_a_single_channel_is_rejected_by_the_node_limit(self):
        ts = TimeSeries.from_array(np.random.default_rng(0).standard_normal((1, 300)), 100.0)
        with pytest.raises(ValueError, match="at least 2 channels"):
            GC(order=2).fit(ts)

    def test_too_few_samples_raise(self):
        ts = TimeSeries.from_array(np.random.default_rng(0).standard_normal((3, 14)), 100.0)
        with pytest.raises(ValueError, match="samples"):
            GC(order=5).fit(ts)

    def test_a_perfectly_predicted_channel_raises_instead_of_returning_infinity(self):
        rng = np.random.default_rng(22)
        a = rng.standard_normal(500)
        b = np.roll(a, 1)  # b[t] = a[t - 1] exactly
        ts = TimeSeries.from_array(np.vstack([a, b]), 100.0)
        with pytest.raises(ValueError, match="perfect"):
            GC(order=1).fit(ts)


class TestDefinition:
    def test_it_is_a_frozen_estimator_with_typed_parameters(self):
        est = GC(order=3, conditional=False)
        with pytest.raises(dataclasses.FrozenInstanceError):
            est.order = 4  # type: ignore[misc]
        with pytest.raises(TypeError):
            GC(conditionall=True)  # type: ignore[call-arg]

    def test_the_method_name_is_the_short_acronym(self):
        assert GC.info.name == "gc"
        assert GC.info.directed and GC.info.min_nodes == 2
        assert not GC.info.scale_dependent

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(order=0), "order"),
            (dict(order="fpe"), "order"),
            (dict(order=2.5), "order"),
            (dict(max_order=0), "max_order"),
            (dict(ridge=-1.0), "ridge"),
            (dict(n_folds=1), "n_folds"),
        ],
    )
    def test_invalid_parameters_fail_at_construction(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            GC(**kwargs)

    def test_the_convenience_function_equals_the_estimator(self):
        ts = make_series(three_channel_coefs(), 500, seed=23)
        np.testing.assert_array_equal(gc(ts, order=2).values, GC(order=2).fit(ts).values)
        assert gc(ts, order=2, conditional=False).provenance.parameters["conditional"] is False


class TestStructure:
    # Thresholds come from observed values over several seeds (see the comments), with margin.

    def test_conditioning_removes_the_indirect_influence_but_keeps_the_direct_ones(self):
        # chain 0 -> 1 -> 2, 20000 samples. Observed over 5 seeds: pairwise 0 -> 2 between 0.0165
        # and 0.0237, conditional 0 -> 2 below 0.00034, direct edge 0 -> 1 equal in both modes to
        # within 0.0002.
        for seed in range(3):
            ts = make_series(three_channel_coefs(), 20000, seed=100 + seed)
            pairwise = GC(order=2, conditional=False).fit(ts).values[:, :, 0, 0]
            conditional = GC(order=2, conditional=True).fit(ts).values[:, :, 0, 0]
            assert pairwise[0, 2] > 0.01
            assert conditional[0, 2] < 0.002
            assert conditional[0, 2] < pairwise[0, 2] / 10
            assert abs(conditional[0, 1] - pairwise[0, 1]) < 0.005
            assert conditional[1, 2] < pairwise[1, 2]

    def test_it_recovers_the_known_network_of_the_simulator(self):
        # A -> B with an 8 ms delay (2 samples at 250 Hz), C unconnected, white node dynamics.
        # Observed over 4 seeds: A -> B between 0.216 and 0.227, every other direction below 0.0013,
        # and BIC chose order 2, the lag of the delay.
        a, b, c = "bankssts-lh", "superiorparietal-lh", "precuneus-rh"
        network = Network(
            nodes=[dk_node(a), dk_node(b), dk_node(c)],
            edges=[Edge(a, b, strength=0.5, delay_ms=8.0)],
        )
        white = NodeDynamics(kind="white")
        for seed in range(3):
            sim = simulate(
                network,
                sfreq=250.0,
                n_times=8000,
                dynamics={a: white, b: white, c: white},
                seed=seed,
                n_sensors=6,
            )
            result = GC(order="bic", max_order=6).fit(sim.sources)
            matrix = result.values[:, :, 0, 0]
            assert result.provenance.model_order >= 2
            assert matrix[0, 1] > 0.1
            off_diagonal = ~np.eye(3, dtype=bool)
            true_edge = sim.truth.values[:, :, 0, 0] == 1.0
            assert true_edge.sum() == 1 and true_edge[0, 1]
            assert np.max(matrix[off_diagonal & ~true_edge]) < 0.01


class TestOrderBoundary:
    def test_a_selection_at_max_order_warns_and_is_recorded(self):
        ts = make_series(three_channel_coefs(), 6000, seed=12)  # true order 2
        with pytest.warns(DirconnWarning, match="max_order"):
            result = GC(order="bic", max_order=2).fit(ts)
        assert result.provenance.assumptions.get("order_selection").status == "warn"

    def test_an_interior_selection_is_recorded_as_a_pass(self):
        ts = make_series(three_channel_coefs(), 6000, seed=12)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DirconnWarning)
            result = GC(order="bic", max_order=6).fit(ts)
        assert result.provenance.assumptions.get("order_selection").status == "pass"

    def test_a_fixed_order_has_no_selection_to_check(self):
        result = GC(order=2).fit(make_series(three_channel_coefs(), 600, seed=12))
        with pytest.raises(KeyError):
            result.provenance.assumptions.get("order_selection")
