import warnings

import numpy as np
import pytest
from scipy import stats

from dirconn import GC, DirconnWarning, TimeSeries
from dirconn.analytic import AnalyticTestResult, analytic_gc_test


def var_data(n_trials=1, n_times=300, n_channels=2, seed=0, drive=0.0):
    """White noise where channel 0 drives channel 1 through lag 1 with strength ``drive``."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_trials, n_channels, n_times))
    for t in range(1, n_times):
        x[:, 1, t] += drive * x[:, 0, t - 1]
    return TimeSeries.from_array(x, 100.0, segmentation="epoched")


def reference(data, order, demean, conditional):
    """F test of the nested models by direct least squares; independent of dirconn.mvar."""
    x = data.data.copy()
    if demean:
        x = x - x.mean(axis=(0, 2), keepdims=True)
    n_trials, n_channels, n_times = x.shape
    rows_y, rows_x = [], []
    for trial in range(n_trials):
        for t in range(order, n_times):
            rows_y.append(x[trial, :, t])
            rows_x.append(np.concatenate([x[trial, :, t - lag] for lag in range(1, order + 1)]))
    y, design = np.array(rows_y), np.array(rows_x)
    n_obs = y.shape[0]

    def sse(columns, target):
        beta = np.linalg.lstsq(design[:, columns], y[:, target], rcond=None)[0]
        return float(np.sum((y[:, target] - design[:, columns] @ beta) ** 2))

    p = np.full((n_channels, n_channels), np.nan)
    for source in range(n_channels):
        for target in range(n_channels):
            if source == target:
                continue
            if conditional:
                full = list(range(n_channels * order))
                reduced = [
                    lag * n_channels + c
                    for lag in range(order)
                    for c in range(n_channels)
                    if c != source
                ]
                n_regressors = n_channels * order
            else:  # bivariate model of the two channels
                pair = [source, target]
                full = [lag * n_channels + c for lag in range(order) for c in pair]
                reduced = [lag * n_channels + target for lag in range(order)]
                n_regressors = 2 * order
            df2 = n_obs - n_regressors - (1 if demean else 0)
            sse_full, sse_reduced = sse(full, target), sse(reduced, target)
            f = (sse_reduced - sse_full) / order / (sse_full / df2)
            p[source, target] = stats.f.sf(f, order, df2)
    return p


@pytest.mark.parametrize("demean", [True, False])
@pytest.mark.parametrize("conditional", [True, False])
@pytest.mark.parametrize("n_trials", [1, 4])
def test_pvalues_equal_a_direct_least_squares_f_test(demean, conditional, n_trials):
    data = var_data(n_trials=n_trials, n_times=120, n_channels=3, seed=2, drive=0.3)
    estimator = GC(order=2, demean=demean, conditional=conditional)
    result = analytic_gc_test(estimator, data)
    expected = reference(data, 2, demean, conditional)
    np.testing.assert_allclose(
        result.pvalues[:, :, 0, 0], expected, rtol=1e-8, atol=1e-300, equal_nan=True
    )


def test_it_agrees_with_statsmodels_on_a_bivariate_series():
    # statsmodels always includes a constant; dirconn removes the mean instead, so the two
    # differ at the order of 1/n (relative 2e-4 on F in the cases checked).
    grangercausalitytests = pytest.importorskip("statsmodels.tsa.stattools").grangercausalitytests

    for seed in range(3):
        data = var_data(n_times=300, seed=seed, drive=0.2)
        result = analytic_gc_test(GC(order=2), data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theirs = grangercausalitytests(data.data[0][[1, 0]].T, maxlag=2)[2][0]["ssr_ftest"]
        assert result.statistic[0, 1, 0, 0] == pytest.approx(theirs[0], rel=2e-3)
        assert result.pvalues[0, 1, 0, 0] == pytest.approx(theirs[1], rel=5e-3, abs=1e-12)
        assert result.df == (2, int(theirs[2]))


def test_the_diagonal_is_undefined_and_the_axes_are_the_estimators():
    result = analytic_gc_test(GC(order=2), var_data(n_channels=3))
    assert result.pvalues.shape == (3, 3, 1, 1)
    diagonal = np.broadcast_to(np.eye(3, dtype=bool)[:, :, None, None], result.pvalues.shape)
    for array in (result.pvalues, result.statistic):
        assert np.isnan(array[diagonal]).all() and np.isfinite(array[~diagonal]).all()
    assert result.observed.source == ("ch0", "ch1", "ch2")
    np.testing.assert_array_equal(result.defined, ~diagonal)


def test_a_real_influence_is_found_and_the_reverse_is_not():
    result = analytic_gc_test(GC(order=1), var_data(n_times=500, drive=0.5, seed=4))
    assert result.pvalues[0, 1, 0, 0] < 1e-6
    assert result.pvalues[1, 0, 0, 0] > 0.01


def test_the_degrees_of_freedom_follow_the_model():
    data = var_data(n_trials=3, n_times=100, n_channels=3)
    conditional = analytic_gc_test(GC(order=2), data)
    pairwise = analytic_gc_test(GC(order=2, conditional=False), data)
    n_obs = 3 * (100 - 2)
    assert conditional.df == (2, n_obs - 3 * 2 - 1)
    assert pairwise.df == (2, n_obs - 2 * 2 - 1)
    assert analytic_gc_test(GC(order=2, demean=False), data).df == (2, n_obs - 3 * 2)


def test_it_states_its_null_and_limitations_and_works_with_correction():
    result = analytic_gc_test(GC(order=2), var_data(drive=0.4))
    assert isinstance(result, AnalyticTestResult)
    assert "past of the source" in result.null
    assert len(result.limitations) >= 3
    assert result.significant(alpha=0.05, method="none")[0, 1, 0, 0]
    assert not result.significant(method="none")[~result.defined].any()
    assert result.null in repr(result)


def test_a_ridge_penalty_is_refused_because_the_f_distribution_no_longer_applies():
    with pytest.raises(ValueError, match="ridge"):
        analytic_gc_test(GC(order=2, ridge=0.1), var_data())


def test_too_few_observations_for_the_residual_degrees_of_freedom_are_an_error():
    data = var_data(n_times=8, n_channels=3)
    with pytest.raises(ValueError):
        analytic_gc_test(GC(order=2), data)


def test_only_gc_has_an_analytic_test_and_the_message_points_elsewhere():
    from dirconn import DTF

    with pytest.raises(TypeError, match="surrogate"):
        analytic_gc_test(DTF(order=2), var_data())  # type: ignore[arg-type]


def test_the_diagnostics_of_the_fit_still_come_through():
    data = var_data(n_times=40, n_channels=3)
    with pytest.warns(DirconnWarning):
        analytic_gc_test(GC(order=6), data)


@pytest.mark.parametrize("alpha", [0.0, 1.0, float("nan")])
def test_alpha_must_lie_strictly_between_zero_and_one(alpha):
    with pytest.raises(ValueError, match="alpha"):
        analytic_gc_test(GC(order=2), var_data(), alpha=alpha)


class TestTypeIErrorCalibration:
    """Under independence the F-test p-values are uniform.

    Measured over 1500 datasets of order-2 GC on independent white noise: one trial of 150
    samples gave a mean p of 0.510, 4.4% of p-values at or below 0.05 and a Kolmogorov-Smirnov
    p-value of 0.21 against the uniform distribution; five trials of 60 samples gave 0.494, 5.5%
    and 0.92. The test below uses the first 400 datasets and requires only that uniformity is not
    rejected at the 1% level.
    """

    @pytest.mark.parametrize("kwargs", [{"n_times": 150}, {"n_trials": 5, "n_times": 60}])
    def test_independent_channels_give_uniform_pvalues(self, kwargs):
        p = np.array(
            [
                analytic_gc_test(GC(order=2), var_data(seed=seed, **kwargs)).pvalues[0, 1, 0, 0]
                for seed in range(400)
            ]
        )
        assert stats.kstest(p, "uniform").pvalue > 0.01
        assert 0.03 < (p <= 0.05).mean() < 0.08
