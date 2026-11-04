import numpy as np
import pytest

from dirconn import ChannelMeta, TimeSeries


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_two_dimensional_input_is_continuous(rng):
    data = rng.standard_normal((3, 200))
    ts = TimeSeries.from_array(data, sfreq=100.0)
    assert ts.segmentation == "continuous"
    assert ts.data.shape == (1, 3, 200)
    assert (ts.n_trials, ts.n_channels, ts.n_times) == (1, 3, 200)
    np.testing.assert_array_equal(ts.data[0], data)


def test_default_channel_names_are_generated(rng):
    ts = TimeSeries.from_array(rng.standard_normal((3, 50)), sfreq=10.0)
    assert ts.names == ("ch0", "ch1", "ch2")


def test_three_dimensional_input_requires_explicit_segmentation(rng):
    with pytest.raises(ValueError, match="segmentation"):
        TimeSeries.from_array(rng.standard_normal((4, 3, 50)), sfreq=10.0)


def test_single_trial_three_dimensional_input_is_not_inferred_as_continuous(rng):
    ts = TimeSeries.from_array(rng.standard_normal((1, 3, 50)), sfreq=10.0, segmentation="epoched")
    assert ts.segmentation == "epoched"
    assert ts.n_trials == 1


def test_continuous_data_cannot_have_several_trials(rng):
    with pytest.raises(ValueError, match="continuous"):
        TimeSeries.from_array(
            rng.standard_normal((2, 3, 50)), sfreq=10.0, segmentation="continuous"
        )


def test_unknown_segmentation_is_rejected(rng):
    with pytest.raises(ValueError, match="segmentation"):
        TimeSeries.from_array(rng.standard_normal((3, 50)), sfreq=10.0, segmentation="windows")


@pytest.mark.parametrize("shape", [(50,), (2, 2, 3, 50)])
def test_unsupported_number_of_dimensions_is_rejected(shape):
    with pytest.raises(ValueError, match="dimension"):
        TimeSeries.from_array(np.zeros(shape), sfreq=10.0, segmentation="epoched")


@pytest.mark.parametrize("sfreq", [0.0, -1.0, float("nan"), float("inf")])
def test_sampling_frequency_must_be_positive_and_finite(rng, sfreq):
    with pytest.raises(ValueError, match="sfreq"):
        TimeSeries.from_array(rng.standard_normal((2, 20)), sfreq=sfreq)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_samples_are_rejected(rng, bad):
    data = rng.standard_normal((2, 20))
    data[1, 5] = bad
    with pytest.raises(ValueError, match="finite"):
        TimeSeries.from_array(data, sfreq=10.0)


def test_duplicate_channel_names_are_rejected(rng):
    with pytest.raises(ValueError, match="unique"):
        TimeSeries.from_array(rng.standard_normal((2, 20)), sfreq=10.0, names=["a", "a"])


def test_number_of_names_must_match_number_of_channels(rng):
    with pytest.raises(ValueError, match="channels"):
        TimeSeries.from_array(rng.standard_normal((3, 20)), sfreq=10.0, names=["a", "b"])


def test_names_and_channels_cannot_both_be_given(rng):
    with pytest.raises(ValueError, match="names"):
        TimeSeries.from_array(
            rng.standard_normal((1, 20)),
            sfreq=10.0,
            names=["a"],
            channels=[ChannelMeta("a")],
        )


def test_channel_metadata_is_kept(rng):
    channels = [
        ChannelMeta("inferiortemporal-lh", node_kind="cortical", region="inferiortemporal"),
        ChannelMeta("STN-L", node_kind="direct", region="STN"),
    ]
    ts = TimeSeries.from_array(rng.standard_normal((2, 40)), sfreq=10.0, channels=channels)
    assert ts.channels == tuple(channels)
    assert ts.names == ("inferiortemporal-lh", "STN-L")


def test_integer_data_is_converted_to_float64():
    ts = TimeSeries.from_array(np.arange(40).reshape(2, 20), sfreq=10.0)
    assert ts.data.dtype == np.float64


def test_data_is_read_only_and_not_copied_when_already_float64(rng):
    data = rng.standard_normal((2, 20))
    ts = TimeSeries.from_array(data, sfreq=10.0)
    assert np.shares_memory(ts.data, data)
    with pytest.raises(ValueError, match="read-only"):
        ts.data[0, 0, 0] = 1.0
    data[0, 0] = 5.0  # the caller's array stays writable
    assert data[0, 0] == 5.0


def test_duration_and_tmin(rng):
    ts = TimeSeries.from_array(rng.standard_normal((2, 250)), sfreq=100.0, tmin=-0.5)
    assert ts.duration == pytest.approx(2.5)
    assert ts.tmin == -0.5


def test_tmin_defaults_to_zero(rng):
    assert TimeSeries.from_array(rng.standard_normal((2, 20)), sfreq=10.0).tmin == 0.0
