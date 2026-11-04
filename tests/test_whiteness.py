"""The residual whiteness check of the static VAR methods (multivariate portmanteau test)."""

import warnings

import numpy as np
import pytest
from scipy import stats
from statsmodels.tsa.vector_ar.var_model import VAR

from dirconn import DTF, GC, GPDC, PDC, RPDC, TRGC, DirconnWarning, TimeSeries, surrogate_test
from dirconn._resampling import without_data_checks
from dirconn.estimator import (
    WHITENESS_EXTRA_LAGS,
    WHITENESS_MIN_AUTOCORR,
    residual_whiteness_check,
)
from dirconn.mvar import fit_var
from dirconn.surrogates import PhaseRandomize

COEFS = np.zeros((2, 3, 3))  # [lag - 1, target, source]
COEFS[0] = [[1.2, 0.0, 0.0], [0.4, 0.3, 0.0], [0.0, 0.0, 0.6]]
COEFS[1] = [[-0.6, 0.0, 0.0], [0.2, -0.2, 0.0], [0.0, 0.0, -0.4]]


def simulate_var(coefs, n_times, *, n_trials=1, seed=0, burn=200):
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


def series(n_times=3000, n_trials=1, seed=0):
    segmentation = "continuous" if n_trials == 1 else "epoched"
    data = simulate_var(COEFS, n_times, n_trials=n_trials, seed=seed)
    return TimeSeries.from_array(data, 100.0, segmentation=segmentation)


def reference_pvalue(residuals, n_trials, order):
    """Hosking's modified statistic with an explicit loop over trials and lags."""
    u = residuals - residuals.mean(axis=0)
    n_obs, k = u.shape
    trials = np.split(u, n_trials)
    lags = order + WHITENESS_EXTRA_LAGS
    c0 = sum(t.T @ t for t in trials) / n_obs
    c0_inv = np.linalg.inv(c0)
    q = 0.0
    for s in range(1, lags + 1):
        cs = sum(t[s:].T @ t[:-s] for t in trials) / n_obs
        q += np.trace(cs.T @ c0_inv @ cs @ c0_inv) / (n_obs - n_trials * s)
    return stats.chi2.sf(n_obs**2 * q, k * k * (lags - order))


def test_one_trial_equals_statsmodels_adjusted_test_whiteness():
    data = series(n_times=1500, seed=1)
    fit = fit_var(data, order=2)
    check = residual_whiteness_check(fit.residuals, 1, 2)
    x = data.data[0].T
    res = VAR(x - x.mean(axis=0)).fit(maxlags=2, ic=None, trend="n")
    expected = res.test_whiteness(nlags=2 + WHITENESS_EXTRA_LAGS, adjusted=True)
    assert check.value == pytest.approx(expected.pvalue, rel=1e-9)
    assert f"(df {expected.df}," in check.detail


def test_autocovariances_stay_within_trials():
    data = series(n_times=200, n_trials=5, seed=2)
    fit = fit_var(data, order=2)
    check = residual_whiteness_check(fit.residuals, 5, 2)
    assert check.value == pytest.approx(reference_pvalue(fit.residuals, 5, 2), rel=1e-9)


def test_white_residuals_pass():
    rng = np.random.default_rng(3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check = residual_whiteness_check(rng.standard_normal((3000, 3)), 1, 2)
    assert check.name == "residual_whiteness" and check.status == "pass"


def test_a_too_low_order_warns():
    fit = fit_var(series(n_times=3000, seed=4), order=1)
    with pytest.warns(DirconnWarning, match="not white"):
        check = residual_whiteness_check(fit.residuals, 1, 1)
    assert check.status == "warn" and check.value < 0.05


def test_a_significant_but_negligible_autocorrelation_does_not_warn():
    rng = np.random.default_rng(5)
    e = rng.standard_normal((200_000, 3))
    u = e.copy()
    u[1:] += 0.03 * e[:-1]  # residual autocorrelation of about 0.03 at lag 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check = residual_whiteness_check(u, 1, 2)
    assert check.status == "pass" and check.value < 1e-6
    assert 0.02 < float(check.detail.split("autocorrelation ")[1].split(" ")[0])
    assert WHITENESS_MIN_AUTOCORR > 0.03


def test_trials_shorter_than_the_lags_are_not_checked():
    check = residual_whiteness_check(np.ones((40, 2)), 4, 2)
    assert check.status == "not_checked" and check.value is None


def test_rows_that_do_not_split_into_trials_are_an_error():
    with pytest.raises(ValueError, match="equal trials"):
        residual_whiteness_check(np.zeros((101, 2)), 4, 1)


@pytest.mark.parametrize(
    "estimator",
    [
        GC(order=2, check_order_stability=False),
        GC(order=2, conditional=False, check_order_stability=False),
        DTF(order=2, check_order_stability=False),
        PDC(order=2, check_order_stability=False),
        GPDC(order=2, check_order_stability=False),
        RPDC(order=2, check_order_stability=False),
        TRGC(order=2, check_order_stability=False),
    ],
    ids=["gc", "gc_pairwise", "dtf", "pdc", "gpdc", "rpdc", "trgc"],
)
def test_the_static_var_methods_record_the_check(estimator):
    data = series(n_times=1000, n_trials=2, seed=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = estimator.fit(data)
    checks = {c.name: c for c in result.provenance.assumptions.checks}
    assert checks["residual_whiteness"].status == "pass"


def test_a_too_low_order_warns_through_an_estimator():
    with pytest.warns(DirconnWarning, match="not white"):
        result = PDC(order=1, check_order_stability=False).fit(series(n_times=3000, seed=7))
    checks = {c.name: c for c in result.provenance.assumptions.checks}
    assert checks["residual_whiteness"].status == "warn"


@pytest.mark.parametrize(
    "estimator",
    [
        GC(order=1, check_order_stability=False, check_whiteness=False),
        GC(order=1, conditional=False, check_order_stability=False, check_whiteness=False),
        DTF(order=1, check_order_stability=False, check_whiteness=False),
        PDC(order=1, check_order_stability=False, check_whiteness=False),
        GPDC(order=1, check_order_stability=False, check_whiteness=False),
        RPDC(order=1, check_order_stability=False, check_whiteness=False),
        TRGC(order=1, check_order_stability=False, check_whiteness=False),
    ],
    ids=["gc", "gc_pairwise", "dtf", "pdc", "gpdc", "rpdc", "trgc"],
)
def test_it_can_be_switched_off(estimator):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = estimator.fit(series(n_times=3000, seed=7))  # order 1 is too low
    assert "residual_whiteness" not in {c.name for c in result.provenance.assumptions.checks}


@pytest.mark.parametrize("cls", [GC, DTF, PDC, GPDC, RPDC, TRGC])
def test_the_switch_must_be_a_bool(cls):
    with pytest.raises(TypeError, match="check_whiteness"):
        cls(order=2, check_whiteness="yes")


def test_trgc_checks_the_forward_fit_only():
    with pytest.warns(DirconnWarning, match="not white") as record:
        result = TRGC(order=1, check_order_stability=False).fit(series(n_times=3000, seed=7))
    assert sum("not white" in str(w.message) for w in record) == 1
    assert result.provenance.assumptions.get("residual_whiteness").status == "warn"


def test_the_resampled_fits_of_a_surrogate_test_skip_it():
    # order 1 is too low for the data and for its surrogates: only the fit on the data may warn
    estimator = PDC(order=1, n_freqs=5, check_order_stability=False)
    with pytest.warns(DirconnWarning, match="not white") as record:
        result = surrogate_test(
            estimator, series(n_times=1000, seed=8), PhaseRandomize(), n_surrogates=9, seed=0
        )
    assert sum("not white" in str(w.message) for w in record) == 1
    assert result.checks.get("surrogate_fits").status == "pass"


def test_without_data_checks_switches_off_both_checks():
    light = without_data_checks(PDC(order=2))
    assert light.check_order_stability is False and light.check_whiteness is False
    assert without_data_checks(light) is light
