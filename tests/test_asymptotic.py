"""Asymptotic tests of PDC and rPDC (Schelter et al. 2006, 2009) and the RPDC estimator.

The references are independent of dirconn's spectral code: statsmodels estimates the VAR model and
the covariance of its coefficients, and the real and imaginary parts of ``Abar_ij(f)`` are written
as explicit linear maps of those coefficients.
"""

import functools
import warnings

import numpy as np
import pytest
from scipy import stats
from statsmodels.tsa.vector_ar.var_model import VAR

from dirconn import (
    DTF,
    GC,
    GPDC,
    PDC,
    RPDC,
    AsymptoticTestResult,
    DirconnWarning,
    TimeSeries,
    analytic_pdc_test,
    analytic_rpdc_test,
    rpdc,
    significance_test,
)

COEFS = np.zeros((2, 3, 3))  # [lag - 1, target, source]: channel 0 drives channel 1
COEFS[0] = [[0.5, 0.0, 0.0], [0.4, 0.3, 0.0], [0.0, 0.0, 0.6]]
COEFS[1] = [[-0.3, 0.0, 0.0], [0.2, -0.2, 0.0], [0.0, 0.0, -0.4]]
COEFS_1 = np.array([[[0.5, 0.0], [0.4, 0.3]]])


def simulate_var(coefs, n_times, *, n_trials=1, seed=0, burn=200):
    """``(n_trials, n, n_times)`` from ``x[t] = sum_k coefs[k-1] @ x[t-k] + e[t]``."""
    rng = np.random.default_rng(seed)
    order, n, _ = coefs.shape
    out = np.zeros((n_trials, n, n_times))
    for trial in range(n_trials):
        x = np.zeros((n, n_times + burn))
        noise = rng.standard_normal((n, n_times + burn))
        for t in range(order, n_times + burn):
            x[:, t] = noise[:, t] + sum(coefs[k] @ x[:, t - k - 1] for k in range(order))
        out[trial] = x[:, burn:]
    return out


def series(coefs=COEFS, n_times=1000, seed=0, scale=None):
    x = simulate_var(coefs, n_times, seed=seed)
    if scale is not None:
        x = x * np.asarray(scale, dtype=float)[None, :, None]
    return TimeSeries.from_array(x, 100.0, segmentation="continuous")


def statsmodels_fit(data, order):
    x = data.data[0].T
    return VAR(x - x.mean(axis=0)).fit(maxlags=order, ic=None, trend="n")


def z_and_cov(res, order, n_channels, target, source, freq, sfreq):
    """``Z = [Re, Im] Abar[target, source](freq)`` and its covariance, from statsmodels."""
    w = 2 * np.pi * freq / sfreq
    lags = np.arange(1, order + 1)
    a = np.asarray(res.coefs)[:, target, source]
    delta = 1.0 if target == source else 0.0
    z = np.array([delta - np.sum(a * np.cos(lags * w)), np.sum(a * np.sin(lags * w))])
    # cov_params is indexed (regressor, equation), regressor (k - 1) * n_channels + source
    cov = np.asarray(res.cov_params())
    rows = [(k - 1) * n_channels * n_channels + source * n_channels + target for k in lags]
    block = cov[np.ix_(rows, rows)]
    jacobian = np.vstack([-np.cos(lags * w), np.sin(lags * w)])
    return z, jacobian @ block @ jacobian.T


class TestAnalyticPDCTest:
    def test_at_order_one_the_statistic_is_the_squared_t_value_of_statsmodels(self):
        data = series(COEFS_1, n_times=800, seed=1)
        result = analytic_pdc_test(
            PDC(order=1, standardize=False, n_freqs=9, check_order_stability=False), data
        )
        t = np.asarray(statsmodels_fit(data, 1).tvalues)  # [regressor = source, equation = target]
        for source, target in [(0, 1), (1, 0)]:
            np.testing.assert_allclose(
                result.statistic[source, target, :, 0], t[source, target] ** 2, rtol=1e-9
            )
        np.testing.assert_allclose(
            result.pvalues[:, :, :, 0], stats.chi2.sf(result.statistic[:, :, :, 0], 1), rtol=1e-12
        )

    def test_the_statistic_divides_the_power_by_the_variance_of_the_real_and_imaginary_part(self):
        data = series(n_times=1500, seed=2)
        estimator = PDC(
            order=2, standardize=False, freqs=[0.0, 7.5, 21.0, 50.0], check_order_stability=False
        )
        result = analytic_pdc_test(estimator, data)
        res = statsmodels_fit(data, 2)
        for source, target in [(0, 1), (1, 0), (2, 1)]:
            for f_index, freq in enumerate(estimator.freqs):
                z, cov = z_and_cov(res, 2, 3, target, source, freq, 100.0)
                expected = z @ z / np.trace(cov)
                np.testing.assert_allclose(
                    result.statistic[source, target, f_index, 0], expected, rtol=1e-8
                )

    def test_the_threshold_follows_schelter_2006(self):
        data = series(n_times=1200, seed=3)
        estimator = PDC(
            order=2, standardize=False, freqs=[5.0, 20.0, 40.0], check_order_stability=False
        )
        result = analytic_pdc_test(estimator, data, alpha=0.01)
        res = statsmodels_fit(data, 2)
        for source, target in [(0, 1), (1, 2)]:
            for f_index, freq in enumerate(estimator.freqs):
                _, cov = z_and_cov(res, 2, 3, target, source, freq, 100.0)
                # sum over all targets k of |Abar_k,source(f)|^2, the PDC denominator
                outflow = sum(
                    np.sum(z_and_cov(res, 2, 3, k, source, freq, 100.0)[0] ** 2) for k in range(3)
                )
                # C_ij / N is the trace of the covariance of [Re, Im] Abar_ij
                expected = np.trace(cov) * stats.chi2.ppf(0.99, 1) / outflow
                np.testing.assert_allclose(
                    result.threshold[source, target, f_index, 0], expected, rtol=1e-8
                )

    def test_a_cell_is_significant_exactly_when_the_pdc_exceeds_the_threshold(self):
        data = series(n_times=1000, seed=4)
        result = analytic_pdc_test(
            PDC(order=2, n_freqs=33, check_order_stability=False), data, alpha=0.05
        )
        above = result.observed.values > result.threshold
        below_alpha = result.pvalues < 0.05
        defined = result.defined
        np.testing.assert_array_equal(above[defined], below_alpha[defined])

    def test_the_amplitude_form_has_the_square_root_threshold_and_the_same_pvalues(self):
        data = series(n_times=800, seed=5)
        squared = analytic_pdc_test(PDC(order=2, n_freqs=17, check_order_stability=False), data)
        amplitude = analytic_pdc_test(
            PDC(order=2, n_freqs=17, squared=False, check_order_stability=False), data
        )
        np.testing.assert_allclose(amplitude.threshold, np.sqrt(squared.threshold), rtol=1e-12)
        np.testing.assert_array_equal(amplitude.pvalues, squared.pvalues)

    @pytest.mark.filterwarnings("ignore:pdc depends on channel amplitudes")
    def test_standardisation_changes_the_threshold_but_not_the_pvalues(self):
        data = series(n_times=800, seed=6, scale=[1.0, 50.0, 0.02])
        plain = analytic_pdc_test(
            PDC(order=2, n_freqs=17, standardize=False, check_order_stability=False), data
        )
        standardised = analytic_pdc_test(
            PDC(order=2, n_freqs=17, check_order_stability=False), data
        )
        np.testing.assert_allclose(plain.pvalues, standardised.pvalues, rtol=1e-8)
        assert not np.allclose(plain.threshold, standardised.threshold, equal_nan=True)

    def test_the_tested_values_are_those_of_the_estimator(self):
        data = series(n_times=800, seed=7)
        estimator = PDC(order="bic", max_order=6, n_freqs=17, check_order_stability=False)
        result = analytic_pdc_test(estimator, data)
        np.testing.assert_array_equal(result.observed.values, estimator.fit(data).values)
        assert result.observed.provenance.model_order == 2

    def test_the_result_has_the_estimator_shape_nan_diagonal_and_states_its_null(self):
        data = series(n_times=600, seed=8)
        result = analytic_pdc_test(PDC(order=2, n_freqs=11, check_order_stability=False), data)
        assert isinstance(result, AsymptoticTestResult)
        for array in (result.pvalues, result.statistic, result.threshold):
            assert array.shape == (3, 3, 11, 1)
            assert np.isnan(array[np.arange(3), np.arange(3)]).all()
            with pytest.raises(ValueError, match="read-only"):
                array[0, 1, 0, 0] = 0.0
        np.testing.assert_array_equal(result.df, np.ones(11))
        assert result.n_obs == 598
        assert "direct" in result.null and "null" in repr(result)
        assert result.limitations and result.alpha == 0.05

    def test_the_family_is_every_pair_and_frequency_and_a_band_is_chosen_by_the_grid(self):
        data = series(n_times=1000, seed=9)
        band = analytic_pdc_test(
            PDC(order=2, freqs=[8.0, 10.0, 12.0], check_order_stability=False), data
        )
        assert band.pvalues.shape == (3, 3, 3, 1) and band.defined.sum() == 6 * 3
        adjusted = band.adjusted("bonferroni")
        np.testing.assert_allclose(
            adjusted[band.defined], np.minimum(band.pvalues[band.defined] * 18, 1.0), rtol=1e-12
        )

    def test_significance_test_reaches_it(self):
        data = series(n_times=600, seed=10)
        via = significance_test(
            PDC(order=2, n_freqs=9, check_order_stability=False), data, test="analytic", alpha=0.1
        )
        direct = analytic_pdc_test(
            PDC(order=2, n_freqs=9, check_order_stability=False), data, alpha=0.1
        )
        assert isinstance(via, AsymptoticTestResult) and via.alpha == 0.1
        np.testing.assert_array_equal(via.pvalues, direct.pvalues)

    @pytest.mark.parametrize("estimator", [GPDC(order=2), DTF(order=2), GC(order=2)])
    def test_other_estimators_are_refused_with_a_pointer_to_the_surrogates(self, estimator):
        with pytest.raises(TypeError, match="surrogate"):
            analytic_pdc_test(estimator, series(n_times=300))  # type: ignore[arg-type]

    def test_a_ridge_penalty_is_refused(self):
        with pytest.raises(ValueError, match="ridge"):
            analytic_pdc_test(PDC(order=2, ridge=0.1), series(n_times=300))


class TestRPDC:
    def test_at_order_one_n_lambda_is_the_squared_t_value_at_every_frequency(self):
        data = series(COEFS_1, n_times=800, seed=11)
        result = RPDC(order=1, n_freqs=9, check_order_stability=False).fit(data)
        t = np.asarray(statsmodels_fit(data, 1).tvalues)
        n_obs = result.provenance.details["n_obs"]
        for source, target in [(0, 1), (1, 0)]:
            np.testing.assert_allclose(
                n_obs * result.values[source, target, :, 0], t[source, target] ** 2, rtol=1e-9
            )

    def test_n_lambda_is_the_wald_statistic_of_the_real_and_imaginary_part(self):
        data = series(n_times=1500, seed=12)
        freqs = [0.0, 3.0, 12.5, 31.0, 50.0]
        result = RPDC(order=2, freqs=freqs, check_order_stability=False).fit(data)
        res = statsmodels_fit(data, 2)
        n_obs = result.provenance.details["n_obs"]
        for source, target in [(0, 1), (1, 0), (2, 0)]:
            for f_index, freq in enumerate(freqs):
                z, cov = z_and_cov(res, 2, 3, target, source, freq, 100.0)
                if freq in (0.0, 50.0):  # only the real part is tested at both ends
                    expected = z[0] ** 2 / cov[0, 0]
                else:
                    expected = z @ np.linalg.solve(cov, z)
                np.testing.assert_allclose(
                    n_obs * result.values[source, target, f_index, 0], expected, rtol=1e-7
                )

    def test_at_order_two_it_is_the_joint_time_domain_test_off_both_ends(self):
        # [Re, Im] is an invertible linear map of the two coefficients, and the Wald statistic
        # does not change under it: rPDC resolves frequency only from order 3 on
        data = series(n_times=1500, seed=12)
        result = RPDC(order=2, freqs=[0.0, 3.0, 12.5, 31.0, 50.0], check_order_stability=False).fit(
            data
        )
        res = statsmodels_fit(data, 2)
        cov = np.asarray(res.cov_params())
        n_obs = result.provenance.details["n_obs"]
        for source, target in [(0, 1), (1, 0)]:
            rows = [lag * 9 + source * 3 + target for lag in range(2)]
            a = np.asarray(res.coefs)[:, target, source]
            wald = a @ np.linalg.solve(cov[np.ix_(rows, rows)], a)
            interior = n_obs * result.values[source, target, 1:4, 0]
            np.testing.assert_allclose(interior, wald, rtol=1e-7)
        third = rpdc(data, order=3, n_freqs=9, check_order_stability=False).values[0, 1, 1:-1, 0]
        assert np.ptp(third) > 0.01 * third.mean()

    @pytest.mark.filterwarnings("ignore:The residuals of the VAR fit are not white")  # order 1
    def test_the_degrees_of_freedom_drop_to_one_where_v_has_rank_one(self):
        data = series(n_times=600, seed=13)
        result = analytic_rpdc_test(RPDC(order=2, n_freqs=6, check_order_stability=False), data)
        np.testing.assert_array_equal(result.df, [1, 2, 2, 2, 2, 1])
        details = result.observed.provenance.details
        assert details["rank_one_freqs"] == [0.0, 50.0]
        order_one = analytic_rpdc_test(RPDC(order=1, n_freqs=6, check_order_stability=False), data)
        np.testing.assert_array_equal(order_one.df, np.ones(6))

    def test_pvalues_and_threshold_use_n_lambda_and_chi2_over_n(self):
        data = series(n_times=900, seed=14)
        result = analytic_rpdc_test(
            RPDC(order=2, n_freqs=8, check_order_stability=False), data, alpha=0.01
        )
        n_obs, df = result.n_obs, result.df
        np.testing.assert_allclose(result.statistic, n_obs * result.observed.values, rtol=1e-12)
        np.testing.assert_allclose(
            result.pvalues[0, 1, :, 0], stats.chi2.sf(result.statistic[0, 1, :, 0], df)
        )
        np.testing.assert_allclose(
            result.threshold[0, 1, :, 0], stats.chi2.ppf(0.99, df) / n_obs, rtol=1e-12
        )
        assert np.isnan(result.threshold[np.arange(3), np.arange(3)]).all()

    def test_it_does_not_depend_on_the_scale_of_the_channels(self):
        plain = rpdc(series(n_times=800, seed=15), order=2, n_freqs=17, check_order_stability=False)
        scaled = rpdc(
            series(n_times=800, seed=15, scale=[1.0, 1e3, 1e-2]),
            check_order_stability=False,
            order=2,
            n_freqs=17,
        )
        # atol for values near zero, where the quadratic form cancels
        np.testing.assert_allclose(
            scaled.values, plain.values, rtol=1e-8, atol=1e-14, equal_nan=True
        )

    def test_permuting_the_channels_permutes_the_result(self):
        data = series(n_times=800, seed=16)
        order = [2, 0, 1]
        permuted = TimeSeries.from_array(data.data[:, order], 100.0, segmentation="continuous")
        a = rpdc(data, order=2, n_freqs=9, check_order_stability=False).values
        b = rpdc(permuted, order=2, n_freqs=9, check_order_stability=False).values
        np.testing.assert_allclose(b, a[np.ix_(order, order)], rtol=1e-9, equal_nan=True)

    # the order is adequate, but at 600 samples the whiteness check warns by chance in about
    # 6.5 % of fits, and this seed is one of them
    @pytest.mark.filterwarnings("ignore:The residuals of the VAR fit are not white")
    def test_values_are_finite_off_the_diagonal_at_both_ends_of_the_default_grid(self):
        result = rpdc(series(n_times=600, seed=17), order=3, check_order_stability=False)
        off = ~np.eye(3, dtype=bool)
        assert np.isfinite(result.values[off]).all()
        assert (result.values[off] >= 0).all()
        assert np.isnan(result.values[np.arange(3), np.arange(3)]).all()
        assert result.freqs[0] == 0.0 and result.freqs[-1] == 50.0

    def test_a_true_edge_stands_out_and_an_absent_one_does_not(self):
        result = analytic_rpdc_test(
            RPDC(order=2, n_freqs=17, check_order_stability=False), series(n_times=4000, seed=18)
        )
        assert (result.pvalues[0, 1] < 1e-6).all()  # 0 -> 1 is in the model
        assert np.median(result.pvalues[1, 0]) > 0.05  # 1 -> 0 is not

    @pytest.mark.parametrize("name", ["ridge", "standardize", "squared"])
    def test_it_has_no_ridge_no_standardisation_and_no_amplitude_form(self, name):
        with pytest.raises(TypeError):
            RPDC(**{name: 0.1})
        assert name not in RPDC().get_params()

    def test_its_info_says_scale_free_and_unbounded(self):
        info = RPDC.info
        assert (info.name, info.directed, info.scale_dependent) == ("rpdc", True, False)
        assert info.value_range == (0.0, np.inf) and info.frequency_resolved

    def test_significance_test_reaches_its_test(self):
        data = series(n_times=600, seed=19)
        via = significance_test(
            RPDC(order=2, n_freqs=9, check_order_stability=False), data, test="analytic"
        )
        direct = analytic_rpdc_test(RPDC(order=2, n_freqs=9, check_order_stability=False), data)
        np.testing.assert_array_equal(via.pvalues, direct.pvalues)

    def test_the_rpdc_test_refuses_other_estimators(self):
        with pytest.raises(TypeError, match="surrogate"):
            analytic_rpdc_test(PDC(order=2), series(n_times=300))  # type: ignore[arg-type]


NULL_CELLS = [(1, 0), (2, 0), (0, 2), (1, 2), (2, 1)]  # [source, target] without an edge in COEFS
NULL_FREQS = [0.0, 12.5, 50.0]


@functools.cache
def null_rejection_rates(n_datasets, n_times):
    """Rejection rates at 0.05 of both tests over fixed seeds, pooled over ``NULL_CELLS``.

    Returns ``{"pdc": rates, "rpdc": rates}``, one rate per frequency of ``NULL_FREQS``.
    """
    rejections = {"pdc": [], "rpdc": []}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DirconnWarning)  # chance whiteness warnings
        for seed in range(n_datasets):
            data = series(n_times=n_times, seed=1000 + seed)
            for name, test, estimator in [
                ("pdc", analytic_pdc_test, PDC),
                ("rpdc", analytic_rpdc_test, RPDC),
            ]:
                result = test(
                    estimator(order=2, freqs=NULL_FREQS, check_order_stability=False), data
                )
                p = np.array([result.pvalues[s, t, :, 0] for s, t in NULL_CELLS])
                rejections[name].append(p < 0.05)
    return {name: np.mean(r, axis=(0, 1)) for name, r in rejections.items()}


class TestNullCalibration:
    """Pointwise rejection rates at 0.05 on a VAR(2) without the tested edges.

    Measured on the five absent edges of ``COEFS`` at order 2, pooled over the edges, 4000 fresh
    datasets each of 300, 1000 and 3000 samples: rPDC 0.048-0.054 at every frequency; PDC
    0.048-0.054 at 0 Hz and the Nyquist frequency, where its chi-square approximation is exact, and
    0.027-0.031 at 12.5 and 25 Hz, where it is conservative (the exact rate of a sum of two equally
    weighted terms is 0.021). Over-fitted at order 4 (1000 samples): rPDC 0.048-0.052, PDC
    0.023-0.025 at 12.5 to 37.5 Hz. The 500 fixed datasets of 300 samples below give PDC 0.050,
    0.030, 0.056 and rPDC 0.050, 0.056, 0.056 at 0, 12.5 and 50 Hz.
    """

    def test_the_rpdc_test_holds_its_level_at_every_frequency(self):
        rates = null_rejection_rates(500, 300)["rpdc"]
        assert ((0.035 < rates) & (rates < 0.07)).all()

    def test_the_pdc_test_holds_its_level_at_both_ends_and_is_not_liberal_between(self):
        rates = null_rejection_rates(500, 300)["pdc"]
        assert 0.035 < rates[0] < 0.07 and 0.035 < rates[2] < 0.07
        assert 0.01 < rates[1] < 0.05
