"""The time-resolved spectral transforms must equal the per-slice transforms."""

import numpy as np
import pytest

from dirconn import spectral
from dirconn.spectral import (
    dtf_from_coefs,
    dtf_from_coefs_series,
    frequency_grid,
    pdc_from_coefs,
    pdc_from_coefs_series,
)

SFREQ = 100.0


def random_series(n_times=7, order=2, n=3, seed=0):
    """A drifting, stable coefficient trajectory ``(n_times, order, n, n)``."""
    rng = np.random.default_rng(seed)
    base = 0.25 * rng.standard_normal((order, n, n))
    drift = 0.05 * rng.standard_normal((n_times, order, n, n)).cumsum(axis=0)
    return base[None] + drift


@pytest.mark.parametrize("chunk", [None, 1, 3])
def test_dtf_series_equals_the_slice_by_slice_transform(monkeypatch, chunk):
    if chunk is not None:  # force several chunks
        monkeypatch.setattr(spectral, "SERIES_CHUNK_ELEMENTS", chunk * 9 * 11)
    coefs = random_series()
    freqs = frequency_grid(SFREQ, n_freqs=11)
    out = dtf_from_coefs_series(coefs, freqs, SFREQ)
    assert out.shape == (3, 3, 11, 7)
    for t in range(7):
        np.testing.assert_allclose(out[..., t], dtf_from_coefs(coefs[t], freqs, SFREQ), rtol=1e-12)


@pytest.mark.parametrize("squared", [True, False])
@pytest.mark.parametrize("chunk", [None, 2])
def test_pdc_series_equals_the_slice_by_slice_transform(monkeypatch, chunk, squared):
    if chunk is not None:
        monkeypatch.setattr(spectral, "SERIES_CHUNK_ELEMENTS", chunk * 9 * 11)
    coefs = random_series(seed=1)
    freqs = frequency_grid(SFREQ, n_freqs=11)
    out = pdc_from_coefs_series(coefs, freqs, SFREQ, squared=squared)
    for t in range(7):
        np.testing.assert_allclose(
            out[..., t], pdc_from_coefs(coefs[t], freqs, SFREQ, squared=squared), rtol=1e-12
        )


def test_the_layout_is_source_target_freq_time():
    coefs = np.zeros((4, 1, 2, 2))
    coefs[:, 0, 1, 0] = 0.5  # channel 0 drives channel 1
    out = dtf_from_coefs_series(coefs, frequency_grid(SFREQ, n_freqs=5), SFREQ)
    assert (out[0, 1] > out[1, 0]).all()
    assert out[0, 1].shape == (5, 4)


def test_values_sum_to_one_over_the_right_axis():
    coefs = random_series(seed=2)
    freqs = frequency_grid(SFREQ, n_freqs=9)
    np.testing.assert_allclose(dtf_from_coefs_series(coefs, freqs, SFREQ).sum(axis=0), 1.0)
    np.testing.assert_allclose(pdc_from_coefs_series(coefs, freqs, SFREQ).sum(axis=1), 1.0)


def test_a_unit_root_slice_raises_like_the_static_transform():
    coefs = random_series(order=1, n=2)
    coefs[3] = np.eye(2)  # I - A(0) is singular at f = 0
    with pytest.raises(ValueError, match="unit root"):
        dtf_from_coefs_series(coefs, np.array([0.0, 10.0]), SFREQ)


def test_non_finite_slices_give_nan_instead_of_an_error():
    coefs = random_series(order=1, n=2)
    coefs[2] = np.nan
    freqs = frequency_grid(SFREQ, n_freqs=5)
    out = dtf_from_coefs_series(coefs, freqs, SFREQ)
    assert np.isnan(out[..., 2]).all()
    assert np.isfinite(out[..., [0, 1, 3, 4, 5, 6]]).all()


@pytest.mark.parametrize(
    "bad", [np.zeros((3, 2, 2)), np.zeros((2, 1, 2, 3)), np.zeros((0, 1, 2, 2))]
)
def test_the_coefficient_shape_is_checked(bad):
    with pytest.raises(ValueError, match="coefs"):
        dtf_from_coefs_series(bad, np.array([0.0, 1.0]), SFREQ)
