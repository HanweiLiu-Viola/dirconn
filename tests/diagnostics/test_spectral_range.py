import json
import warnings

import numpy as np
import pytest
from scipy import signal

from dirconn import TimeSeries
from dirconn.diagnostics.spectral_range import (
    SPECTRAL_RANGE_WARN_DB,
    spectral_dynamic_range,
    spectral_range_check,
)

SFREQ = 250.0
BANDS = {"low": (4.0, 16.0), "high": (60.0, 100.0)}


def _white(n_channels=2, seed=0, seconds=20):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_channels, int(seconds * SFREQ)))


def test_white_noise_has_small_dynamic_range():
    ts = TimeSeries.from_array(_white(), SFREQ)
    sr = spectral_dynamic_range(ts, BANDS)
    assert sr.worst_db < 10.0
    assert spectral_range_check(sr).status == "pass"
    assert sr.dynamic_range_db.shape == (2,)


def test_lowpassed_signal_warns():
    b, a = signal.butter(4, 20, fs=SFREQ)
    ts = TimeSeries.from_array(signal.filtfilt(b, a, _white(), axis=-1), SFREQ)
    sr = spectral_dynamic_range(ts, BANDS)
    check = spectral_range_check(sr)
    assert sr.worst_db > SPECTRAL_RANGE_WARN_DB
    assert check.status == "warn"
    assert check.name == "spectral_dynamic_range"
    assert check.value == sr.worst_db
    assert sr.worst_channel in check.detail


def test_invalid_samples_are_skipped():
    clean = _white(seconds=30)
    valid = np.ones(clean.shape[-1], dtype=bool)
    valid[3000:3500] = False
    dirty = clean.copy()
    dirty[:, ~valid] = 1e6
    sr_dirty = spectral_dynamic_range(TimeSeries.from_array(dirty, SFREQ, valid=valid), BANDS)
    sr_clean = spectral_dynamic_range(TimeSeries.from_array(clean, SFREQ, valid=valid), BANDS)
    assert np.allclose(sr_dirty.psd, sr_clean.psd)
    assert sr_dirty.worst_db < 10.0


def test_no_long_enough_run_raises():
    ts = TimeSeries.from_array(_white(seconds=2), SFREQ, valid=np.arange(500) % 100 < 50)
    with pytest.raises(ValueError, match="valid"):
        spectral_dynamic_range(ts, BANDS)


def test_to_dict_rounds_and_is_plain():
    sr = spectral_dynamic_range(TimeSeries.from_array(_white(), SFREQ), BANDS)
    d = sr.to_dict()
    assert isinstance(d["worst_db"], float)
    assert isinstance(d["psd"], list)
    assert d["worst_db"] == float(f"{sr.worst_db:.6g}")


def test_flat_channel_gives_finite_warning_and_json():
    data = _white(n_channels=2)
    data[0] = 3.0
    ts = TimeSeries.from_array(data, SFREQ)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sr = spectral_dynamic_range(ts, BANDS)
        check = spectral_range_check(sr)
    assert np.all(np.isfinite(sr.dynamic_range_db))
    assert check.status == "warn"
    assert sr.worst_channel == ts.names[0]
    json.dumps(sr.to_dict(), allow_nan=False)


def test_runs_are_weighted_by_length():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 2000)) * np.linspace(1, 5, 2000)
    valid = np.zeros(2000, dtype=bool)
    valid[0:600] = True
    valid[800:1800] = True
    ts = TimeSeries.from_array(x, SFREQ, valid=valid)
    sr = spectral_dynamic_range(ts, {"all": (1.0, 100.0)}, nperseg=500)
    _, pa = signal.welch(x[:, 0:600], fs=SFREQ, nperseg=500, axis=-1)
    _, pb = signal.welch(x[:, 800:1800], fs=SFREQ, nperseg=500, axis=-1)
    assert np.allclose(sr.psd, (pa * 600 + pb * 1000) / 1600)
    assert not np.allclose(sr.psd, (pa + pb) / 2)


def test_short_epochs_are_used_whole():
    rng = np.random.default_rng(2)
    epochs = rng.standard_normal((6, 2, int(1.5 * SFREQ)))
    ts = TimeSeries.from_array(epochs, SFREQ, segmentation="epoched")
    sr = spectral_dynamic_range(ts, BANDS)
    assert sr.freqs.size == epochs.shape[-1] // 2 + 1
    _, pxx = signal.welch(epochs, fs=SFREQ, nperseg=epochs.shape[-1], axis=-1)
    assert np.allclose(sr.psd, pxx.mean(axis=0))


def test_default_nperseg_is_capped_at_the_longest_valid_run():
    x = _white(seconds=4)
    valid = np.zeros(x.shape[-1], dtype=bool)
    valid[100:400] = True
    valid[500:700] = True
    sr = spectral_dynamic_range(TimeSeries.from_array(x, SFREQ, valid=valid), BANDS)
    _, pxx = signal.welch(x[:, 100:400], fs=SFREQ, nperseg=300, axis=-1)
    assert np.allclose(sr.psd, pxx)


def test_a_run_of_exactly_nperseg_samples_is_used():
    x = _white(seconds=4)
    sr = spectral_dynamic_range(TimeSeries.from_array(x, SFREQ), BANDS, nperseg=x.shape[-1])
    assert sr.freqs.size == x.shape[-1] // 2 + 1
