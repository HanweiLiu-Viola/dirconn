import mne
import numpy as np
import pytest

from dirconn import TimeSeries


def _info(ch_names, ch_types, sfreq=100.0):
    return mne.create_info(ch_names, sfreq, ch_types)


@pytest.fixture
def rng():
    return np.random.default_rng(1)


def test_raw_becomes_continuous_time_series(rng):
    data = rng.standard_normal((3, 300)) * 1e-6
    raw = mne.io.RawArray(data, _info(["Fz", "Cz", "Pz"], "eeg"), verbose="error")
    ts = TimeSeries.from_mne(raw)
    assert ts.segmentation == "continuous"
    assert ts.sfreq == 100.0
    assert ts.names == ("Fz", "Cz", "Pz")
    assert {ch.type for ch in ts.channels} == {"eeg"}
    np.testing.assert_allclose(ts.data[0], data)


def test_epochs_become_epoched_time_series(rng):
    data = rng.standard_normal((5, 3, 120)) * 1e-6
    epochs = mne.EpochsArray(data, _info(["Fz", "Cz", "Pz"], "eeg"), tmin=-0.2, verbose="error")
    ts = TimeSeries.from_mne(epochs)
    assert ts.segmentation == "epoched"
    assert ts.data.shape == (5, 3, 120)
    assert ts.tmin == pytest.approx(-0.2)
    np.testing.assert_allclose(ts.data, data)


def test_single_epoch_stays_epoched(rng):
    data = rng.standard_normal((1, 2, 60)) * 1e-6
    epochs = mne.EpochsArray(data, _info(["a", "b"], "eeg"), verbose="error")
    assert TimeSeries.from_mne(epochs).segmentation == "epoched"


def test_bad_channels_and_non_neural_channels_are_excluded(rng):
    data = rng.standard_normal((4, 100)) * 1e-6
    info = _info(["Fz", "Cz", "STI 014", "EOG 061"], ["eeg", "eeg", "stim", "eog"])
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.info["bads"] = ["Cz"]
    ts = TimeSeries.from_mne(raw)
    assert ts.names == ("Fz",)


def test_misc_channels_are_kept_because_roi_series_use_that_type(rng):
    data = rng.standard_normal((3, 100))
    raw = mne.io.RawArray(
        data,
        _info(["inferiortemporal-lh", "STN-L", "STN-R"], "misc"),
        verbose="error",
    )
    ts = TimeSeries.from_mne(raw)
    assert ts.names == ("inferiortemporal-lh", "STN-L", "STN-R")
    assert {ch.type for ch in ts.channels} == {"misc"}


def test_picks_select_a_channel_subset(rng):
    data = rng.standard_normal((3, 100)) * 1e-6
    raw = mne.io.RawArray(data, _info(["Fz", "Cz", "Pz"], "eeg"), verbose="error")
    ts = TimeSeries.from_mne(raw, picks=["Pz", "Fz"])
    assert ts.names == ("Pz", "Fz")
    np.testing.assert_allclose(ts.data[0, 0], data[2])


def test_montage_positions_are_carried_over(rng):
    names = ["Fz", "Cz", "Pz"]
    raw = mne.io.RawArray(
        rng.standard_normal((3, 100)) * 1e-6, _info(names, "eeg"), verbose="error"
    )
    raw.set_montage("standard_1020", verbose="error")
    ts = TimeSeries.from_mne(raw)
    assert all(ch.pos is not None for ch in ts.channels)


def test_channels_without_positions_have_none(rng):
    raw = mne.io.RawArray(
        rng.standard_normal((2, 100)) * 1e-6, _info(["a", "b"], "eeg"), verbose="error"
    )
    ts = TimeSeries.from_mne(raw)
    assert all(ch.pos is None for ch in ts.channels)


def test_non_mne_object_is_rejected():
    with pytest.raises(TypeError, match="Raw or Epochs"):
        TimeSeries.from_mne(np.zeros((2, 10)))


def test_empty_selection_is_rejected(rng):
    info = _info(["STI 014"], ["stim"])
    raw = mne.io.RawArray(np.zeros((1, 50)), info, verbose="error")
    with pytest.raises(ValueError, match="No data channels"):
        TimeSeries.from_mne(raw)
