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
