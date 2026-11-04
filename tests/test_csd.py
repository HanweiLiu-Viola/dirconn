"""Tests of the shared multitaper cross-spectra."""

import mne
import numpy as np
import pytest
from mne.time_frequency import csd_array_multitaper

from dirconn import TimeSeries
from dirconn._csd import CrossSpectra, cross_spectra, two_sided_spectra

SFREQ = 100.0


def _continuous(x):
    return TimeSeries.from_array(x[np.newaxis], SFREQ, segmentation="continuous")


def _delayed_pair(n_times, delay, seed=0, n_trials=None):
    """Channel 1 is channel 0 delayed by ``delay`` samples (white noise source)."""
    rng = np.random.default_rng(seed)
    if n_trials is None:
        src = rng.standard_normal(n_times + delay)
        return np.stack([src[delay:], src[:n_times]])
    src = rng.standard_normal((n_trials, n_times + delay))
    return np.stack([src[:, delay:], src[:, :n_times]], axis=1)


def _mne_values(segments, **kwargs):
    csd = csd_array_multitaper(segments, SFREQ, verbose=False, **kwargs)
    values = np.stack([csd.get_data(index=k) for k in range(len(csd.frequencies))], axis=-1)
    return values, np.asarray(csd.frequencies)


def test_shape_and_hermitian():
    rng = np.random.default_rng(1)
    data = _continuous(rng.standard_normal((3, 2000)))
    out = cross_spectra(data, segment_length=2.0)
    assert isinstance(out, CrossSpectra)
    assert out.values.shape == (3, 3, out.freqs.size)
    assert out.values.dtype == np.complex128
    np.testing.assert_array_equal(out.values, np.conj(np.swapaxes(out.values, 0, 1)))  # exactly
    diag = np.einsum("iif->if", out.values)
    np.testing.assert_allclose(diag.imag, 0.0, atol=1e-15)
    assert (diag.real > 0).all()


def test_continuous_matches_direct_mne_call_on_the_same_segments():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 1050))
    out = cross_spectra(_continuous(x), segment_length=2.0)
    segments = x[:, :1000].reshape(2, 5, 200).transpose(1, 0, 2)
    ref, freqs = _mne_values(segments)
    np.testing.assert_allclose(out.freqs, freqs)
    np.testing.assert_allclose(out.values, ref, rtol=1e-12, atol=1e-15)
    assert out.record["n_segments"] == 5
    assert out.record["segment_samples"] == 200
    assert out.record["discarded_samples"] == 50


def test_epoched_is_the_mean_of_per_trial_spectra():
    """Each trial is one segment: nothing crosses a trial boundary."""
    x = _delayed_pair(300, 2, seed=3, n_trials=4)
    x[1] += 5.0  # a trial-specific offset; the mean is removed within every segment
    data = TimeSeries.from_array(x, SFREQ, segmentation="epoched")
    out = cross_spectra(data)
    per_trial = [
        cross_spectra(TimeSeries.from_array(x[k : k + 1], SFREQ, segmentation="epoched")).values
        for k in range(4)
    ]
    np.testing.assert_allclose(out.values, np.mean(per_trial, axis=0), rtol=1e-12, atol=1e-15)
    assert out.record["n_segments"] == 4
    assert out.record["discarded_samples"] == 0


def test_white_noise_is_flat_with_one_sided_density():
    rng = np.random.default_rng(4)
    data = _continuous(rng.standard_normal((1, 60000)))
    out = cross_spectra(data, segment_length=2.0)
    psd = out.values[0, 0].real
    # one-sided density of unit-variance white noise: 2 / sfreq
    assert abs(psd.mean() / (2 / SFREQ) - 1) < 0.02
    half = psd.size // 2
    assert abs(psd[:half].mean() / psd[half:].mean() - 1) < 0.05


def test_delayed_copy_has_phase_plus_two_pi_f_d():
    """values[i, j] = <X_i conj(X_j)>: when j lags i by d samples the phase is +2 pi f d / sfreq."""
    delay = 3
    data = _continuous(_delayed_pair(40000, delay, seed=5))
    out = cross_spectra(data, segment_length=2.0)
    keep = out.freqs < 0.4 * SFREQ / 2
    expected = np.angle(np.exp(2j * np.pi * out.freqs[keep] * delay / SFREQ))
    error = np.angle(out.values[0, 1, keep] * np.exp(-1j * expected))
    assert np.abs(error).max() < 0.05


def test_a_per_segment_offset_has_no_effect():
    """MNE removes the mean of every segment, so a constant offset per segment changes nothing."""
    rng = np.random.default_rng(6)
    x = rng.standard_normal((2, 1000))
    shifted = x + np.repeat(rng.normal(0, 3, (2, 5)), 200, axis=1)
    a = cross_spectra(_continuous(x), segment_length=2.0)
    b = cross_spectra(_continuous(shifted), segment_length=2.0)
    np.testing.assert_allclose(a.values, b.values, rtol=1e-10, atol=1e-14)
    assert a.record["mean_removed_per_segment"] is True


def test_frequency_range_and_resolution():
    rng = np.random.default_rng(7)
    data = _continuous(rng.standard_normal((2, 2000)))
    out = cross_spectra(data, segment_length=4.0, fmin=8.0, fmax=30.0)
    assert out.freqs.min() >= 8.0 and out.freqs.max() <= 30.0
    np.testing.assert_allclose(np.diff(out.freqs), 0.25)
    assert out.record["frequency_resolution_hz"] == 0.25
    full = cross_spectra(data, segment_length=4.0)
    assert full.freqs.max() == SFREQ / 2
    assert full.freqs.min() > 0  # 0 Hz is not returned


@pytest.mark.parametrize(
    ("segment_length", "nw"), [(2.0, 1.0), (2.0, 3.0), (2.56, 2.5), (5.0, 4.0)]
)
def test_record_names_tapers_and_bandwidth(segment_length, nw):
    rng = np.random.default_rng(8)
    data = _continuous(rng.standard_normal((2, 2000)))
    out = cross_spectra(data, segment_length=segment_length, time_halfbandwidth=nw)
    rec = out.record
    assert rec["time_halfbandwidth"] == nw
    assert rec["bandwidth_hz"] == pytest.approx(2 * nw / segment_length)
    tapers, _, _ = mne.time_frequency.multitaper._compute_mt_params(
        rec["segment_samples"],
        SFREQ,
        rec["bandwidth_hz"],
        low_bias=True,
        adaptive=False,
        verbose=False,
    )
    assert rec["n_tapers"] == tapers.shape[0]
    assert rec["segmentation"] == "continuous"
    assert rec["segment_length_s"] == segment_length
    assert "own multitaper" in rec["backend"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "segment_length is required"),
        ({"segment_length": 30.0}, "longer than the data"),
        ({"segment_length": 0.015}, "multiple of the sampling interval"),
        ({"segment_length": 2.0, "fmin": 20.0, "fmax": 10.0}, "fmin"),
        ({"segment_length": 2.0, "fmax": 60.0}, "Nyquist"),
        ({"segment_length": 2.0, "fmin": -1.0}, "fmin"),
        ({"segment_length": 2.0, "fmin": 10.1, "fmax": 10.3}, "no frequency"),
        ({"segment_length": 2.0, "time_halfbandwidth": 0.5}, "time_halfbandwidth"),
    ],
)
def test_continuous_errors(kwargs, match):
    rng = np.random.default_rng(9)
    data = _continuous(rng.standard_normal((2, 2000)))
    with pytest.raises(ValueError, match=match):
        cross_spectra(data, **kwargs)


def test_epoched_with_segment_length_cuts_every_trial_on_its_own():
    """Segments are taken inside each trial; the tail of every trial is dropped and counted."""
    x = _delayed_pair(250, 1, seed=10, n_trials=3)
    data = TimeSeries.from_array(x, SFREQ, segmentation="epoched")
    out = cross_spectra(data, segment_length=1.0)
    segments = x[:, :, :200].reshape(3, 2, 2, 100).transpose(0, 2, 1, 3).reshape(6, 2, 100)
    ref, _ = _mne_values(segments)
    np.testing.assert_allclose(out.values, ref, rtol=1e-12, atol=1e-15)
    assert out.record["n_segments"] == 6
    assert out.record["discarded_samples"] == 150
    # a segment can never join the end of one trial to the start of the next
    joined = np.concatenate([x[0, :, 200:250], x[1, :, :50]], axis=-1)
    assert not any(np.array_equal(s, joined) for s in segments)


def test_epoched_segment_longer_than_a_trial_raises():
    data = TimeSeries.from_array(_delayed_pair(200, 1, n_trials=3), SFREQ, segmentation="epoched")
    with pytest.raises(ValueError, match="longer than a trial"):
        cross_spectra(data, segment_length=3.0)


# --- two-sided spectra for the factorisation -----------------------------------------------


def test_two_sided_spectra_cover_zero_to_nyquist_and_halve_the_one_sided_ones():
    rng = np.random.default_rng(11)
    data = _continuous(rng.standard_normal((2, 2000)))
    two = two_sided_spectra(data, segment_length=2.0)
    one = cross_spectra(data, segment_length=2.0)
    np.testing.assert_allclose(two.freqs, np.fft.rfftfreq(200, 1 / SFREQ))
    assert two.freqs[0] == 0.0 and two.freqs[-1] == SFREQ / 2
    assert two.record["two_sided"] is True and one.record["two_sided"] is False
    interior = slice(1, -1)  # away from 0 Hz and Nyquist the one-sided density is twice as large
    np.testing.assert_allclose(one.values[:, :, :-1], 2 * two.values[:, :, interior], rtol=1e-12)
    np.testing.assert_allclose(one.values[:, :, -1], two.values[:, :, -1], rtol=1e-12)
    np.testing.assert_allclose(two.values, np.conj(np.swapaxes(two.values, 0, 1)), atol=1e-15)


def test_two_sided_white_noise_density_and_variance():
    """Unit-variance white noise: two-sided density 1 / sfreq, whose integral is the variance."""
    rng = np.random.default_rng(12)
    data = _continuous(rng.standard_normal((1, 60000)))
    two = two_sided_spectra(data, segment_length=2.0)
    psd = two.values[0, 0].real
    assert abs(psd[1:-1].mean() * SFREQ - 1) < 0.02
    full_circle = np.concatenate([psd, psd[-2:0:-1]])  # negative frequencies mirror the positive
    assert abs(full_circle.mean() * SFREQ - 1) < 0.02  # mean density x sfreq = variance


def test_mne_is_not_needed_at_run_time(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "mne", None)
    monkeypatch.setitem(sys.modules, "mne.time_frequency", None)
    rng = np.random.default_rng(13)
    out = cross_spectra(_continuous(rng.standard_normal((2, 1000))), segment_length=2.0)
    assert out.values.shape[0] == 2
