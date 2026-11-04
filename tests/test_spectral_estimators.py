import dataclasses
import warnings

import numpy as np
import pytest

from dirconn import (
    DTF,
    GPDC,
    PDC,
    Connectivity,
    DirconnWarning,
    Edge,
    Network,
    NodeDynamics,
    TimeSeries,
    dk_node,
    dtf,
    fit_var,
    gpdc,
    pdc,
    simulate,
)
from dirconn.spectral import dtf_from_coefs, frequency_grid, pdc_from_coefs

COEFS = np.zeros((2, 3, 3))
COEFS[0] = [[0.5, 0.0, 0.0], [0.4, 0.3, 0.0], [0.0, 0.3, 0.2]]  # 0 -> 1 -> 2, [target, source]
COEFS[1] = [[-0.2, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, -0.1]]


def make_series(n_times=4000, *, n_trials=1, seed=0, scales=None, names=None, coefs=COEFS):
    rng = np.random.default_rng(seed)
    order, n, _ = coefs.shape
    out = np.zeros((n_trials, n, n_times))
    for trial in range(n_trials):
        x = np.zeros((n, n_times + 200))
        noise = rng.standard_normal(x.shape)
        for t in range(order, x.shape[1]):
            x[:, t] = noise[:, t] + sum(coefs[k] @ x[:, t - k - 1] for k in range(order))
        out[trial] = x[:, 200:]
    if scales is not None:
        out = out * np.asarray(scales, dtype=float)[None, :, None]
    segmentation = "continuous" if n_trials == 1 else "epoched"
    return TimeSeries.from_array(out, 200.0, names=names, segmentation=segmentation)


class TestBuildingBlocks:
    def test_dtf_is_the_transfer_function_measure_of_the_fitted_model(self):
        ts = make_series(seed=1)
        freqs = frequency_grid(200.0, n_freqs=33)
        fit = fit_var(ts, 2)
        result = DTF(order=2, n_freqs=33, standardize=False).fit(ts)
        np.testing.assert_allclose(
            result.values[:, :, :, 0], dtf_from_coefs(fit.coefs, freqs, 200.0), rtol=1e-12
        )

    @pytest.mark.parametrize("squared", [True, False])
    def test_pdc_is_the_ar_polynomial_measure_of_the_fitted_model(self, squared):
        ts = make_series(seed=2)
        freqs = frequency_grid(200.0, n_freqs=33)
        fit = fit_var(ts, 2)
        result = PDC(order=2, n_freqs=33, standardize=False, squared=squared).fit(ts)
        np.testing.assert_allclose(
            result.values[:, :, :, 0],
            pdc_from_coefs(fit.coefs, freqs, 200.0, squared=squared),
            rtol=1e-12,
        )

    def test_gpdc_uses_the_innovation_variances_of_the_fit(self):
        ts = make_series(seed=3, scales=[1.0, 5.0, 0.3])
        freqs = frequency_grid(200.0, n_freqs=33)
        fit = fit_var(ts, 2)
        result = GPDC(order=2, n_freqs=33).fit(ts)
        expected = pdc_from_coefs(fit.coefs, freqs, 200.0, noise_var=np.diag(fit.noise_cov))
        np.testing.assert_allclose(result.values[:, :, :, 0], expected, rtol=1e-12)


class TestOutput:
    @pytest.mark.parametrize("estimator", [DTF, PDC, GPDC])
    def test_the_result_is_a_frequency_resolved_directed_array(self, estimator):
        ts = make_series(seed=4, names=["a", "b", "c"])
        result = estimator(order=2, n_freqs=17).fit(ts)
        assert isinstance(result, Connectivity)
        assert result.values.shape == (3, 3, 17, 1)
        assert result.source == result.target == ("a", "b", "c")
        assert result.is_frequency_resolved and not result.is_time_resolved
        np.testing.assert_allclose(result.freqs, np.linspace(0.0, 100.0, 17))
        assert np.isfinite(result.values).all()

    def test_the_default_grid_is_zero_to_nyquist_with_128_points(self):
        result = PDC(order=2).fit(make_series(seed=5))
        assert result.freqs.shape == (128,)
        assert result.freqs[0] == 0.0 and result.freqs[-1] == 100.0

    def test_fmax_and_explicit_frequencies(self):
        ts = make_series(seed=5)
        np.testing.assert_allclose(
            DTF(order=2, n_freqs=11, fmax=50.0).fit(ts).freqs, np.linspace(0.0, 50.0, 11)
        )
        explicit = PDC(order=2, freqs=[2.0, 10.0, 40.0], n_freqs=999).fit(ts)
        np.testing.assert_array_equal(explicit.freqs, [2.0, 10.0, 40.0])

    def test_a_frequency_above_nyquist_is_an_error(self):
        with pytest.raises(ValueError, match="Nyquist"):
            DTF(order=2, fmax=150.0).fit(make_series(seed=5))

    def test_dtf_sums_to_one_over_the_sources_of_every_target(self):
        values = DTF(order=2).fit(make_series(seed=6)).values[:, :, :, 0]
        np.testing.assert_allclose(values.sum(axis=0), 1.0, rtol=1e-10)

    @pytest.mark.parametrize("estimator", [PDC, GPDC])
    def test_squared_pdc_sums_to_one_over_the_targets_of_every_source(self, estimator):
        values = estimator(order=2).fit(make_series(seed=7)).values[:, :, :, 0]
        np.testing.assert_allclose(values.sum(axis=1), 1.0, rtol=1e-10)

    @pytest.mark.parametrize("estimator", [PDC, GPDC])
    def test_the_amplitude_form_is_the_square_root(self, estimator):
        ts = make_series(seed=8)
        squared = estimator(order=2).fit(ts).values
        amplitude = estimator(order=2, squared=False).fit(ts).values
        np.testing.assert_allclose(amplitude, np.sqrt(squared), rtol=1e-12)

    def test_the_result_can_be_reduced_to_a_named_band(self):
        reduced = PDC(order=2).fit(make_series(seed=9)).reduce_freq("beta")
        assert reduced.values.shape == (3, 3, 1, 1)
        assert reduced.provenance.history[-1]["band_name"] == "beta"

    def test_epoched_data_are_pooled(self):
        ts = make_series(n_times=300, n_trials=8, seed=10)
        result = DTF(order=2).fit(ts)
        assert result.provenance.details["n_obs"] == 8 * (300 - 2)


class TestScale:
    def test_standardising_makes_dtf_and_pdc_independent_of_channel_scales(self):
        base = make_series(seed=11)
        scaled = make_series(seed=11, scales=[1.0, 20.0, 0.05])
        for estimator in (DTF, PDC):
            np.testing.assert_allclose(
                estimator(order=2, n_freqs=33).fit(base).values,
                estimator(order=2, n_freqs=33).fit(scaled).values,
                rtol=1e-8,
                atol=1e-12,
            )

    def test_without_standardising_the_relative_scale_changes_the_result(self):
        base = make_series(seed=11)
        scaled = make_series(seed=11, scales=[1.0, 20.0, 0.05])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DirconnWarning)
            for estimator in (DTF, PDC):
                a = estimator(order=2, standardize=False).fit(base).values
                b = estimator(order=2, standardize=False).fit(scaled).values
                assert not np.allclose(a, b, atol=1e-3)

    def test_gpdc_never_depends_on_the_scale(self):
        base = make_series(seed=12)
        scaled = make_series(seed=12, scales=[1.0, 20.0, 0.05])
        np.testing.assert_allclose(
            GPDC(order=2, n_freqs=33).fit(base).values,
            GPDC(order=2, n_freqs=33).fit(scaled).values,
            rtol=1e-8,
            atol=1e-12,
        )

    def test_standardising_is_recorded_in_the_provenance(self):
        scales = [1.0, 20.0, 0.05]
        prov = DTF(order=2).fit(make_series(seed=13, scales=scales)).provenance
        assert prov.preprocessing["standardized"] is True
        assert len(prov.preprocessing["scale"]) == 3
        assert prov.preprocessing["scale"][1] > prov.preprocessing["scale"][0]
        plain = DTF(order=2, standardize=False).fit(make_series(seed=13)).provenance
        assert plain.preprocessing == {"standardized": False}
        assert GPDC(order=2).fit(make_series(seed=13)).provenance.preprocessing == {}

    def test_unbalanced_scales_warn_only_when_not_standardising(self):
        ts = make_series(seed=14, scales=[1.0, 100.0, 1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            DTF(order=2).fit(ts)
            PDC(order=2).fit(ts)
            GPDC(order=2).fit(ts)
        with pytest.warns(DirconnWarning, match="standardi"):
            result = PDC(order=2, standardize=False).fit(ts)
        assert result.provenance.assumptions.get("channel_scale").status == "warn"

    def test_a_flat_channel_cannot_be_standardised(self):
        data = make_series(seed=15).data[0].copy()
        data[1] = 0.0
        ts = TimeSeries.from_array(data, 200.0)
        with pytest.raises(ValueError, match="variance"):
            DTF(order=2).fit(ts)


class TestOrder:
    def test_a_criterion_selects_the_order_once_and_records_the_curves(self):
        ts = make_series(n_times=8000, seed=16)
        result = DTF(order="bic", max_order=6).fit(ts)
        assert result.provenance.model_order == 2
        selection = result.provenance.details["order_selection"]
        assert selection["criterion"] == "bic" and selection["orders"] == [1, 2, 3, 4, 5, 6]

    def test_a_fixed_order_has_no_selection(self):
        result = PDC(order=3).fit(make_series(seed=17))
        assert result.provenance.model_order == 3
        assert result.provenance.details["order_selection"] is None

    def test_cross_validation_is_available(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DirconnWarning)  # cv may stop at max_order
            result = GPDC(order="cv", max_order=3, n_folds=4).fit(make_series(seed=18))
        assert result.provenance.details["order_selection"]["cv_scheme"] == "blocks"


class TestDiagnostics:
    def test_a_stable_model_passes_the_stationarity_check(self):
        check = DTF(order=2).fit(make_series(seed=19)).provenance.assumptions.get("stationarity")
        assert check.status == "pass" and 0 < check.value < 0.99

    def test_the_sampling_check_counts_channels_times_order(self):
        prov = DTF(order=2).fit(make_series(n_times=3000, seed=20)).provenance
        assert prov.assumptions.get("sampling_adequacy").value == pytest.approx(3 * 2 / 2998)

    def test_too_many_coefficients_warn(self):
        ts = TimeSeries.from_array(np.random.default_rng(21).standard_normal((6, 120)), 200.0)
        with pytest.warns(DirconnWarning, match="coefficients"):
            DTF(order=3).fit(ts)

    def test_an_explosive_process_is_reported_as_unstable_but_still_returned(self):
        rng = np.random.default_rng(22)
        x = np.zeros((2, 300))
        for t in range(1, 300):
            x[:, t] = 1.04 * x[:, t - 1] + rng.standard_normal(2)
        ts = TimeSeries.from_array(x, 200.0)
        # the F tests of the order-stability check are not valid for an explosive process
        with pytest.warns(DirconnWarning, match="unstable"):
            result = PDC(order=1, check_order_stability=False).fit(ts)
        check = result.provenance.assumptions.get("stationarity")
        assert check.status == "fail" and check.value >= 1.0
        assert result.provenance.assumptions.has_warnings

    def test_a_random_walk_is_flagged_as_near_a_unit_root(self):
        rng = np.random.default_rng(23)
        ts = TimeSeries.from_array(np.cumsum(rng.standard_normal((2, 3000)), axis=1), 200.0)
        with pytest.warns(DirconnWarning):
            result = PDC(order=1).fit(ts)
        assert result.provenance.assumptions.get("stationarity").status in ("warn", "fail")


class TestFailures:
    def test_a_duplicated_channel_raises(self):
        data = make_series(seed=24).data[0]
        ts = TimeSeries.from_array(np.vstack([data, data[:1]]), 200.0)
        with pytest.raises(ValueError, match="rank"):
            DTF(order=2).fit(ts)

    def test_ridge_makes_it_estimable(self):
        data = make_series(seed=24).data[0]
        ts = TimeSeries.from_array(np.vstack([data, data[:1]]), 200.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DirconnWarning)
            result = PDC(order=2, ridge=1e-3).fit(ts)
        assert np.isfinite(result.values).all()

    def test_one_channel_is_rejected_by_the_node_limit(self):
        ts = TimeSeries.from_array(np.random.default_rng(0).standard_normal((1, 500)), 200.0)
        with pytest.raises(ValueError, match="at least 2 channels"):
            PDC(order=2).fit(ts)

    def test_too_few_samples_raise(self):
        ts = TimeSeries.from_array(np.random.default_rng(0).standard_normal((3, 14)), 200.0)
        with pytest.raises(ValueError, match="samples"):
            DTF(order=5).fit(ts)


class TestDefinition:
    @pytest.mark.parametrize(
        "estimator, name, scale_dependent",
        [(DTF, "dtf", True), (PDC, "pdc", True), (GPDC, "gpdc", False)],
    )
    def test_short_acronym_names_and_metadata(self, estimator, name, scale_dependent):
        assert estimator.info.name == name
        assert estimator.info.directed and estimator.info.frequency_resolved
        assert estimator.info.value_range == (0.0, 1.0)
        assert estimator.info.scale_dependent is scale_dependent

    def test_provenance_names_the_method_and_the_backend(self):
        prov = GPDC(order=2).fit(make_series(seed=25)).provenance
        assert prov.method == "gpdc" and "OLS" in prov.backend
        assert prov.claim == "predictive, not interventional"
        assert prov.details["n_obs"] == 4000 - 2
        assert 0 < prov.details["spectral_radius"] < 1

    def test_dtf_has_no_squared_switch_and_unknown_arguments_fail(self):
        with pytest.raises(TypeError):
            DTF(squared=False)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            PDC(standardise=True)  # type: ignore[call-arg]
        assert PDC().squared is True and GPDC().squared is True

    def test_estimators_are_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            PDC().order = 3  # type: ignore[misc]

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(order=0), "order"),
            (dict(order="fpe"), "order"),
            (dict(max_order=0), "max_order"),
            (dict(ridge=-1.0), "ridge"),
            (dict(n_folds=1), "n_folds"),
            (dict(n_freqs=1), "n_freqs"),
            (dict(fmax=0.0), "fmax"),
            (dict(freqs=[1.0, 2.0], fmax=50.0), "either"),
        ],
    )
    def test_invalid_parameters_fail_at_construction(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            DTF(**kwargs)

    def test_convenience_functions_equal_the_estimators(self):
        ts = make_series(seed=26)
        np.testing.assert_array_equal(
            dtf(ts, order=2, n_freqs=9).values, DTF(order=2, n_freqs=9).fit(ts).values
        )
        np.testing.assert_array_equal(
            pdc(ts, order=2, n_freqs=9, squared=False).values,
            PDC(order=2, n_freqs=9, squared=False).fit(ts).values,
        )
        np.testing.assert_array_equal(
            gpdc(ts, order=2, n_freqs=9).values, GPDC(order=2, n_freqs=9).fit(ts).values
        )


class TestOrderBoundary:
    @pytest.mark.parametrize("estimator", [DTF, PDC, GPDC])
    def test_a_selection_at_max_order_warns_and_is_recorded(self, estimator):
        ts = make_series(n_times=8000, seed=16)  # true order 2
        with pytest.warns(DirconnWarning, match="max_order"):
            result = estimator(order="bic", max_order=2).fit(ts)
        assert result.provenance.assumptions.get("order_selection").status == "warn"

    def test_a_fixed_order_has_no_selection_to_check(self):
        result = DTF(order=2).fit(make_series(seed=16))
        with pytest.raises(KeyError):
            result.provenance.assumptions.get("order_selection")


class TestRecoveryOfASimulatedNetwork:
    # A -> B acts only in 13-30 Hz (delay 16 ms = 4 samples at 250 Hz), C is unconnected, the node
    # dynamics are white. Observed with 30000 samples over 4 seeds at order 16: in the beta band
    # A -> B between 0.361 and 0.419, outside it (60-110 Hz) at most 0.004, every other pair at most
    # 0.002. The thresholds below keep a wide margin around these values.

    @pytest.mark.parametrize("estimator", [DTF, PDC, GPDC])
    def test_the_measure_finds_the_edge_in_its_band_and_nowhere_else(self, estimator):
        a, b, c = "bankssts-lh", "superiorparietal-lh", "precuneus-rh"
        network = Network(
            nodes=[dk_node(a), dk_node(b), dk_node(c)],
            edges=[Edge(a, b, strength=1.0, delay_ms=16.0, band=(13, 30))],
        )
        white = NodeDynamics(kind="white")
        for seed in range(2):
            sim = simulate(
                network,
                sfreq=250.0,
                n_times=30000,
                dynamics={a: white, b: white, c: white},
                seed=seed,
                n_sensors=6,
            )
            result = estimator(order=16).fit(sim.sources)
            beta = result.reduce_freq("beta").values[:, :, 0, 0]
            high = result.reduce_freq((60.0, 110.0)).values[:, :, 0, 0]
            assert beta[0, 1] > 0.25
            assert high[0, 1] < 0.02
            assert beta[0, 1] > 20 * high[0, 1]
            others = ~np.eye(3, dtype=bool)
            others[0, 1] = False
            assert beta[others].max() < 0.02
