"""Granger causality from the state-space (innovations) form of a VAR (Barnett & Seth 2015)."""

import dataclasses
import warnings

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.signal import lfilter

from dirconn import (
    GC,
    SSGC,
    Connectivity,
    DirconnWarning,
    TimeSeries,
    analytic_gc_test,
    fit_var,
    ssgc,
)
from dirconn.multitaper_granger import geweke_spectral_gc
from dirconn.spectral import transfer_function
from dirconn.statespace import (
    reduced_model,
    spectral_state_space_gc,
    state_space_gc,
    var_to_state_space,
)

SFREQ = 200.0


def oscillator_chain(order_of_second_link=3):
    """Three 20 Hz oscillators (pole radius 0.9), x0 -> x1 at lag 1, x1 -> x2 at a later lag.

    ``coefs[lag - 1, target, source]``; true order 3 with the default lag.
    """
    radius, peak = 0.9, 20.0
    a1, a2 = 2 * radius * np.cos(2 * np.pi * peak / SFREQ), -(radius**2)
    coefs = np.zeros((max(2, order_of_second_link), 3, 3))
    for node in range(3):
        coefs[0, node, node], coefs[1, node, node] = a1, a2
    coefs[0, 1, 0] = 0.4
    coefs[order_of_second_link - 1, 2, 1] = 0.4
    return coefs


def simulate_chain(n_times, seed):
    """Simulate :func:`oscillator_chain` exactly; without feedback, filter node by node."""
    coefs = oscillator_chain()
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((3, n_times + 1000))
    ar = [1.0, -coefs[0, 0, 0], -coefs[1, 0, 0]]
    x = np.zeros_like(noise)
    x[0] = lfilter([1.0], ar, noise[0])
    drive = noise[1].copy()
    drive[1:] += 0.4 * x[0, :-1]
    x[1] = lfilter([1.0], ar, drive)
    drive = noise[2].copy()
    drive[3:] += 0.4 * x[1, :-3]
    x[2] = lfilter([1.0], ar, drive)
    return TimeSeries.from_array(
        x[None, :, 1000:], SFREQ, names=["x0", "x1", "x2"], segmentation="continuous"
    )


def random_model(n_channels, order, seed):
    """A stable VAR model with correlated innovations."""
    rng = np.random.default_rng(seed)
    coefs = rng.normal(
        scale=0.25 / np.sqrt(n_channels * order), size=(order, n_channels, n_channels)
    )
    coefs[0] += 0.4 * np.eye(n_channels)
    mixing = rng.normal(size=(n_channels, n_channels))
    noise_cov = mixing @ mixing.T + n_channels * np.eye(n_channels)
    return coefs, noise_cov


def full_circle_mean(values, freqs):
    """Mean over the whole unit circle from a symmetric grid 0 .. Nyquist (trapezoid rule)."""
    return trapezoid(values, freqs, axis=-1) / (freqs[-1] - freqs[0])


FREQS = np.linspace(0.0, SFREQ / 2, 513)
QUIET = {"check_order_stability": False, "check_whiteness": False}


class TestStateSpaceForm:
    def test_the_reduced_model_reproduces_the_spectrum_of_the_subprocess(self):
        coefs, noise_cov = random_model(4, 3, seed=1)
        model = var_to_state_space(coefs, noise_cov)
        h = transfer_function(coefs, FREQS, SFREQ)
        spectrum = h @ noise_cov @ np.conj(np.swapaxes(h, 1, 2))
        for channels in ([0, 2], [1, 2, 3], [3]):
            sub = reduced_model(model, channels)
            hr = sub.transfer_function(FREQS, SFREQ)
            sub_spectrum = hr @ sub.noise_cov @ np.conj(np.swapaxes(hr, 1, 2))
            np.testing.assert_allclose(
                sub_spectrum, spectrum[:, channels][:, :, channels], rtol=1e-10, atol=1e-12
            )

    def test_the_full_model_has_the_transfer_function_of_the_var(self):
        coefs, noise_cov = random_model(3, 4, seed=2)
        model = var_to_state_space(coefs, noise_cov)
        np.testing.assert_allclose(
            model.transfer_function(FREQS, SFREQ),
            transfer_function(coefs, FREQS, SFREQ),
            atol=1e-12,
        )
        assert model.channels == (0, 1, 2)

    def test_an_unstable_model_is_rejected(self):
        coefs = np.array([[[1.02, 0.0], [0.3, 0.5]]])
        with pytest.raises(ValueError, match="stable"):
            var_to_state_space(coefs, np.eye(2))
        with pytest.raises(ValueError, match="stable"):
            state_space_gc(coefs, np.eye(2))

    def test_the_innovation_covariance_must_be_positive_definite(self):
        coefs, _ = random_model(2, 1, seed=3)
        with pytest.raises(ValueError, match="positive definite"):
            var_to_state_space(coefs, np.array([[1.0, 1.0], [1.0, 1.0]]))


class TestAgainstTheTheory:
    @pytest.mark.parametrize("conditional", [True, False])
    @pytest.mark.parametrize("seed", [4, 5])
    def test_the_spectral_gc_averages_to_the_time_domain_value(self, conditional, seed):
        coefs, noise_cov = random_model(4, 3, seed=seed)
        time = state_space_gc(coefs, noise_cov, conditional=conditional)
        spectral = spectral_state_space_gc(coefs, noise_cov, FREQS, SFREQ, conditional=conditional)
        assert spectral.shape == (4, 4, FREQS.size)
        np.testing.assert_allclose(
            full_circle_mean(spectral, FREQS), time, atol=1e-12, equal_nan=True
        )
        off = ~np.eye(4, dtype=bool)
        assert np.all(time[off] > 0) and np.isnan(np.diag(time)).all()
        assert np.isnan(spectral[np.arange(4), np.arange(4)]).all()

    def test_the_oscillator_chain_averages_to_the_time_domain_value(self):
        coefs = oscillator_chain()
        time = state_space_gc(coefs, np.eye(3))
        spectral = spectral_state_space_gc(coefs, np.eye(3), FREQS, SFREQ)
        np.testing.assert_allclose(
            full_circle_mean(spectral, FREQS), time, atol=1e-12, equal_nan=True
        )

    @pytest.mark.parametrize("conditional", [True, False])
    def test_two_channels_give_gewekes_formula_at_every_frequency(self, conditional):
        coefs, noise_cov = random_model(2, 4, seed=6)
        expected = geweke_spectral_gc(transfer_function(coefs, FREQS, SFREQ), noise_cov)
        spectral = spectral_state_space_gc(coefs, noise_cov, FREQS, SFREQ, conditional=conditional)
        np.testing.assert_allclose(spectral, expected, atol=1e-12, equal_nan=True)

    def test_an_indirect_path_has_no_conditional_influence(self):
        coefs = oscillator_chain()
        time = state_space_gc(coefs, np.eye(3))
        spectral = spectral_state_space_gc(coefs, np.eye(3), FREQS, SFREQ)
        # x0 -> x2 runs through x1; nothing flows back
        for source, target in ((0, 2), (1, 0), (2, 0), (2, 1)):
            assert abs(time[source, target]) < 1e-12
            assert np.max(np.abs(spectral[source, target])) < 1e-12
        pairwise = state_space_gc(coefs, np.eye(3), conditional=False)
        assert pairwise[0, 2] > 0.1 and abs(pairwise[2, 0]) < 1e-12

    def test_the_values_do_not_depend_on_the_scale_of_the_channels(self):
        coefs, noise_cov = random_model(3, 2, seed=7)
        scale = np.diag([1.0, 1e3, 1e-2])
        scaled_coefs = np.einsum("ij,kjl,lm->kim", scale, coefs, np.linalg.inv(scale))
        scaled_cov = scale @ noise_cov @ scale
        for conditional in (True, False):
            np.testing.assert_allclose(
                spectral_state_space_gc(
                    scaled_coefs, scaled_cov, FREQS, SFREQ, conditional=conditional
                ),
                spectral_state_space_gc(coefs, noise_cov, FREQS, SFREQ, conditional=conditional),
                rtol=1e-8,
                atol=1e-12,
                equal_nan=True,
            )


@pytest.fixture(scope="module")
def chain_data():
    return simulate_chain(100_000, seed=0)


class TestGrangerMethod:
    def test_refit_is_the_default_and_is_unchanged(self):
        ts = simulate_chain(3000, seed=1)
        default = GC(order=3, **QUIET).fit(ts)
        explicit = GC(order=3, method="refit", **QUIET).fit(ts)
        assert GC().method == "refit"
        np.testing.assert_array_equal(default.values, explicit.values)
        assert default.provenance.details["method"] == "refit"

    @pytest.mark.parametrize("conditional", [True, False])
    def test_state_space_uses_the_one_full_fit(self, conditional):
        ts = simulate_chain(3000, seed=2)
        result = GC(order=3, method="state_space", conditional=conditional, **QUIET).fit(ts)
        fit = fit_var(ts, 3)
        np.testing.assert_allclose(
            result.values[:, :, 0, 0],
            state_space_gc(fit.coefs, fit.noise_cov, conditional=conditional),
            rtol=1e-12,
            equal_nan=True,
        )
        details = result.provenance.details
        assert details["method"] == "state_space" and details["conditional"] is conditional
        assert details["n_obs"] == fit.n_obs
        assert result.provenance.assumptions.get("stationarity").status == "pass"

    def test_state_space_at_the_true_order_matches_the_refit_at_a_long_order(self, chain_data):
        # The reduced process is VARMA, not VAR(3): the refit at the true order overestimates
        # x1 -> x2 (lag 3) by about 60 %, the refit converges only at long orders, and the
        # state-space value of the order-3 fit is right at once.
        truth = state_space_gc(oscillator_chain(), np.eye(3))
        state_space = GC(order=3, method="state_space", **QUIET).fit(chain_data).values[:, :, 0, 0]
        refit_short = GC(order=3, **QUIET).fit(chain_data).values[:, :, 0, 0]
        refit_long = GC(order=60, **QUIET).fit(chain_data).values[:, :, 0, 0]
        off = ~np.eye(3, dtype=bool)
        np.testing.assert_allclose(state_space[off], refit_long[off], atol=0.01)
        np.testing.assert_allclose(state_space[off], truth[off], atol=0.01)
        assert refit_short[1, 2] > 1.5 * truth[1, 2]

    def test_a_ridge_penalty_uses_the_penalised_fit(self):
        ts = simulate_chain(2000, seed=3)
        result = GC(order=4, method="state_space", ridge=0.1, **QUIET).fit(ts)
        fit = fit_var(ts, 4, ridge=0.1)
        expected = state_space_gc(fit.coefs, fit.noise_cov)
        np.testing.assert_allclose(result.values[:, :, 0, 0], expected, rtol=1e-12, equal_nan=True)
        # unlike the refit with a penalty, the values come from one model and stay non-negative
        off = ~np.eye(3, dtype=bool)
        assert np.all(result.values[:, :, 0, 0][off] >= -1e-12)
        assert result.provenance.parameters["ridge"] == 0.1

    def test_an_unstable_fit_raises(self):
        rng = np.random.default_rng(22)
        x = np.zeros((2, 300))
        for t in range(1, 300):
            x[:, t] = 1.04 * x[:, t - 1] + rng.standard_normal(2)
        ts = TimeSeries.from_array(x, SFREQ)
        with (
            pytest.warns(DirconnWarning, match="unstable"),
            pytest.raises(ValueError, match="stable"),
        ):
            GC(order=1, method="state_space", **QUIET).fit(ts)

    def test_an_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="method"):
            GC(method="exact")

    def test_the_f_test_is_only_for_the_refit(self):
        ts = simulate_chain(2000, seed=4)
        with pytest.raises(ValueError, match="refit"):
            analytic_gc_test(GC(order=3, method="state_space", **QUIET), ts)

    def test_the_parameters_round_trip(self):
        estimator = GC(order=3, method="state_space")
        assert dataclasses.replace(estimator).get_params()["method"] == "state_space"


class TestSSGC:
    def test_the_result_is_frequency_resolved_and_matches_the_fitted_model(self):
        ts = simulate_chain(4000, seed=5)
        result = SSGC(order=3, n_freqs=65, **QUIET).fit(ts)
        assert isinstance(result, Connectivity)
        assert result.values.shape == (3, 3, 65, 1)
        assert result.provenance.method == "ssgc"
        assert result.provenance.details["conditional"] is True
        fit = fit_var(ts, 3)
        np.testing.assert_allclose(
            result.values[:, :, :, 0],
            spectral_state_space_gc(fit.coefs, fit.noise_cov, result.freqs, SFREQ),
            rtol=1e-12,
            equal_nan=True,
        )
        assert np.isnan(result.values[[0, 1, 2], [0, 1, 2]]).all()

    @pytest.mark.parametrize("conditional", [True, False])
    def test_the_mean_over_frequency_is_the_time_domain_gc(self, conditional):
        ts = simulate_chain(4000, seed=6)
        spectral = SSGC(order=3, n_freqs=257, conditional=conditional, **QUIET).fit(ts)
        time = GC(order=3, method="state_space", conditional=conditional, **QUIET).fit(ts)
        np.testing.assert_allclose(
            full_circle_mean(spectral.values[:, :, :, 0], np.asarray(spectral.freqs)),
            time.values[:, :, 0, 0],
            atol=1e-10,
            equal_nan=True,
        )

    def test_the_shorthand_and_the_order_selection(self):
        ts = simulate_chain(4000, seed=7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DirconnWarning)
            result = ssgc(ts, max_order=6, n_freqs=16, check_order_stability=False)
            expected = SSGC(max_order=6, n_freqs=16, check_order_stability=False).fit(ts)
        np.testing.assert_array_equal(result.values, expected.values)
        assert result.provenance.details["order_selection"]["criterion"] == "bic"
        checks = result.provenance.assumptions
        for name in ("sampling_adequacy", "stationarity", "residual_whiteness"):
            assert checks.get(name).status in ("pass", "warn")

    def test_the_values_do_not_depend_on_the_scale_of_the_channels(self):
        ts = simulate_chain(3000, seed=8)
        scaled = TimeSeries.from_array(
            ts.data * np.array([1.0, 1e3, 1e-3])[None, :, None],
            SFREQ,
            names=list(ts.names),
            segmentation="continuous",
        )
        first = SSGC(order=3, n_freqs=33, **QUIET).fit(ts)
        second = SSGC(order=3, n_freqs=33, **QUIET).fit(scaled)
        np.testing.assert_allclose(
            first.values, second.values, rtol=1e-6, atol=1e-10, equal_nan=True
        )
        names = [check.name for check in second.provenance.assumptions.checks]
        assert "channel_scale" not in names

    def test_an_unstable_fit_raises(self):
        rng = np.random.default_rng(22)
        x = np.zeros((2, 300))
        for t in range(1, 300):
            x[:, t] = 1.04 * x[:, t - 1] + rng.standard_normal(2)
        ts = TimeSeries.from_array(x, SFREQ)
        with (
            pytest.warns(DirconnWarning, match="unstable"),
            pytest.raises(ValueError, match="stable"),
        ):
            SSGC(order=1, **QUIET).fit(ts)

    def test_conditional_must_be_a_bool(self):
        with pytest.raises(TypeError, match="conditional"):
            SSGC(conditional=1)
