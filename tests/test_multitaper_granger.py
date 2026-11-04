"""Tests of the Wilson factorisation and the multitaper Granger causality."""

import warnings

import numpy as np
import pytest

from dirconn import GC, DirconnWarning, TimeSeries
from dirconn._csd import two_sided_spectra
from dirconn.multitaper_granger import (
    MTGC,
    full_circle,
    geweke_spectral_gc,
    mtgc,
    wilson_factorisation,
)
from dirconn.spectral import transfer_function

SFREQ = 100.0
# bivariate VAR(2), layout [lag, target, source]: channel 0 drives channel 1 at lag 2
COEFS = np.zeros((2, 2, 2))
COEFS[0] = [[0.5, 0.0], [0.0, 0.4]]
COEFS[1] = [[-0.3, 0.0], [0.4, 0.0]]
SIGMA = np.array([[1.0, 0.3], [0.3, 0.8]])


def _analytic(n_fft):
    freqs = np.fft.fftfreq(n_fft, 1 / SFREQ)
    transfer = transfer_function(COEFS, freqs, SFREQ)
    return transfer, transfer @ SIGMA @ np.conj(np.swapaxes(transfer, 1, 2))


def _var(seed, n_times=20000, extra=False):
    rng = np.random.default_rng(seed)
    x = np.zeros((2, n_times + 100))
    e = rng.multivariate_normal([0, 0], SIGMA, n_times + 100).T
    for t in range(2, n_times + 100):
        x[:, t] = COEFS[0] @ x[:, t - 1] + COEFS[1] @ x[:, t - 2] + e[:, t]
    x = x[:, 100:]
    if extra:  # a third, independent AR(1) channel
        z = np.zeros(n_times)
        noise = rng.standard_normal(n_times)
        for t in range(1, n_times):
            z[t] = 0.7 * z[t - 1] + noise[t]
        x = np.vstack([x, z])
    return TimeSeries.from_array(x, SFREQ)


def _fit(estimator, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DirconnWarning)
        return estimator.fit(data)


# --- full circle and Wilson ----------------------------------------------------------------


@pytest.mark.parametrize("n_fft", [8, 9])
def test_full_circle_mirrors_the_positive_frequencies(n_fft):
    rng = np.random.default_rng(0)
    full = rng.standard_normal((n_fft, 2, 2)) + 1j * rng.standard_normal((n_fft, 2, 2))
    full = full + np.conj(np.roll(full[::-1], 1, axis=0))  # S(-f) = conj(S(f)) on the circle
    positive = np.moveaxis(full[: n_fft // 2 + 1], 0, -1)
    np.testing.assert_allclose(full_circle(positive, n_fft), full, atol=1e-15)
    with pytest.raises(ValueError, match="do not belong"):
        full_circle(positive, n_fft + 2)


def test_wilson_recovers_the_transfer_function_and_noise_of_a_var():
    transfer, spectrum = _analytic(256)
    factor = wilson_factorisation(spectrum)
    np.testing.assert_allclose(factor.transfer, transfer, atol=1e-9)
    np.testing.assert_allclose(factor.noise_cov, SIGMA, atol=1e-9)
    product = factor.transfer @ factor.noise_cov @ np.conj(np.swapaxes(factor.transfer, 1, 2))
    np.testing.assert_allclose(product, spectrum, rtol=1e-8, atol=1e-12)
    assert factor.error < 1e-10 and factor.n_iter < 100  # measured: 33 iterations


def test_wilson_refuses_instead_of_returning_a_partial_factor():
    _, spectrum = _analytic(256)
    with pytest.raises(ValueError, match="did not converge"):
        wilson_factorisation(spectrum, max_iter=3)
    # an estimated spectrum on its native grid stalls well above 1e-6 (measured 3.5e-4)
    data = _var(0, n_times=6000)
    native = two_sided_spectra(data, segment_length=2.0)
    with pytest.raises(ValueError, match="stalled"):
        wilson_factorisation(full_circle(native.values, 200))
    singular = np.zeros((16, 2, 2), dtype=complex)
    with pytest.raises(ValueError, match="positive definite"):
        wilson_factorisation(singular)


def test_geweke_is_zero_without_influence_and_integrates_to_time_domain_granger():
    transfer, _ = _analytic(4000)
    values = geweke_spectral_gc(transfer, SIGMA)
    assert np.isnan(values[0, 0]).all() and np.isnan(values[1, 1]).all()
    np.testing.assert_allclose(values[1, 0], 0.0, atol=1e-12)  # channel 1 does not drive 0
    assert (values[0, 1] >= 0).all()
    # Geweke: the mean over the circle is the time-domain Granger causality (0.2136 here);
    # pairwise GC with the true order on 200000 samples: measured 0.2146 to 0.2171 (3 seeds)
    long = _fit(GC(order=2, conditional=False), _var(1, n_times=200000)).values[0, 1, 0, 0]
    assert values[0, 1].mean() == pytest.approx(long, abs=0.01)
    with pytest.raises(ValueError, match="two-channel"):
        geweke_spectral_gc(np.zeros((4, 3, 3)), np.eye(3))


# --- MTGC ----------------------------------------------------------------------------------


def test_mtgc_approaches_the_analytic_spectral_granger_causality():
    # Measured over 8 seeds (20000 samples, 2 s segments): largest deviation from the analytic
    # spectrum 0.044 to 0.069 (peak 0.52), reverse direction at most 0.0055, integral 0.1995 to
    # 0.216 against 0.2136.
    for seed in range(2):
        result = _fit(MTGC(segment_length=2.0), _var(seed))
        transfer = transfer_function(COEFS, result.freqs, SFREQ)
        reference = geweke_spectral_gc(transfer, SIGMA)
        assert np.max(np.abs(result.values[0, 1, :, 0] - reference[0, 1])) < 0.12
        assert np.max(result.values[1, 0, :, 0]) < 0.012
        integral = np.array(result.provenance.details["integral"], dtype=float)
        assert integral[0, 1] == pytest.approx(0.2136, abs=0.025)


def test_mtgc_is_pairwise():
    """A pair's values do not depend on the other channels."""
    data = _var(2, n_times=6000, extra=True)
    three = _fit(MTGC(segment_length=2.0), data)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        alone = _fit(MTGC(segment_length=2.0), data.pick([i, j]))
        np.testing.assert_allclose(three.values[[i, j]][:, [i, j]], alone.values, rtol=1e-10)
    assert np.isnan(three.values[[0, 1, 2], [0, 1, 2]]).all()


def test_mtgc_grid_provenance_and_shorthand():
    data = _var(3, n_times=4000)
    result = _fit(MTGC(segment_length=2.0, fmin=0.0, fmax=20.0), data)
    np.testing.assert_allclose(result.freqs, np.arange(0.0, 20.0 + 1e-9, 0.5))  # native, 0 Hz
    details = result.provenance.details
    assert details["pairwise"] is True
    assert details["pad_factor"] == 16
    assert details["cross_spectra"]["n_fft"] == 3200
    assert details["wilson_error_max"] <= details["wilson_max_residual"]
    assert result.provenance.method == "mtgc"
    np.testing.assert_array_equal(mtgc(data, segment_length=2.0, fmax=20.0).values, result.values)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({}, ValueError, "segment_length is required"),
        ({"segment_length": 2.0, "fmin": 30.0, "fmax": 10.0}, ValueError, "fmin < fmax"),
        ({"segment_length": 2.0, "fmax": 80.0}, ValueError, "Nyquist"),
        ({"segment_length": 2.0, "fmin": 10.1, "fmax": 10.2}, ValueError, "No frequency bin"),
    ],
)
def test_mtgc_errors(kwargs, error, match):
    with pytest.raises(error, match=match):
        _fit(MTGC(**kwargs), _var(4, n_times=2000))


def test_mtgc_refuses_the_vector_autoregressive_grid_arguments():
    with pytest.raises(TypeError):
        MTGC(segment_length=2.0, n_freqs=64)
    with pytest.raises(TypeError):
        MTGC(segment_length=2.0, order=3)
