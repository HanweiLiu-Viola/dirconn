import numpy as np
import pytest

from dirconn import ChannelMeta, TimeSeries

CHANNELS = [
    ChannelMeta("Fz", type="eeg"),
    ChannelMeta("inferiortemporal-lh", node_kind="cortical", region="inferiortemporal"),
    ChannelMeta("STN-L", node_kind="direct", region="STN", hemisphere="lh"),
    ChannelMeta("Pz", type="eeg"),
]


@pytest.fixture
def ts():
    data = np.random.default_rng(0).standard_normal((3, 4, 50))
    return TimeSeries.from_array(
        data, sfreq=250.0, channels=CHANNELS, segmentation="epoched", tmin=-0.1
    )


def test_picking_by_name_keeps_the_given_order_and_the_data(ts):
    picked = ts.pick(["STN-L", "Fz"])
    assert picked.names == ("STN-L", "Fz")
    np.testing.assert_array_equal(picked.data[:, 0], ts.data[:, 2])
    np.testing.assert_array_equal(picked.data[:, 1], ts.data[:, 0])


def test_picking_by_index(ts):
    picked = ts.pick([3, 1])
    assert picked.names == ("Pz", "inferiortemporal-lh")
    np.testing.assert_array_equal(picked.data, ts.data[:, [3, 1]])


def test_names_and_indices_can_be_mixed(ts):
    assert ts.pick(["Fz", 2]).names == ("Fz", "STN-L")


def test_numpy_integers_are_accepted(ts):
    assert ts.pick(np.array([1, 2])).names == ("inferiortemporal-lh", "STN-L")


def test_channel_metadata_travels_with_the_channels(ts):
    picked = ts.pick(["STN-L", "inferiortemporal-lh"])
    assert picked.channels == (CHANNELS[2], CHANNELS[1])


def test_everything_else_is_unchanged(ts):
    picked = ts.pick(["Fz", "Pz"])
    assert (picked.sfreq, picked.segmentation, picked.tmin) == (250.0, "epoched", -0.1)
    assert (picked.n_trials, picked.n_times) == (3, 50)


def test_a_continuous_series_stays_continuous():
    series = TimeSeries.from_array(np.zeros((3, 20)) + 1.0, sfreq=10.0, names=["a", "b", "c"])
    assert series.pick(["c", "a"]).segmentation == "continuous"


def test_the_result_is_a_read_only_independent_series(ts):
    picked = ts.pick(["Fz", "Pz"])
    assert not np.shares_memory(picked.data, ts.data)
    with pytest.raises(ValueError, match="read-only"):
        picked.data[0, 0, 0] = 1.0


def test_picking_everything_in_the_original_order_reproduces_the_series(ts):
    picked = ts.pick(list(ts.names))
    np.testing.assert_array_equal(picked.data, ts.data)
    assert picked.channels == ts.channels


def test_picks_can_be_chained(ts):
    assert ts.pick(["Fz", "STN-L", "Pz"]).pick(["Pz", "Fz"]).names == ("Pz", "Fz")


def test_an_unknown_channel_name_is_reported(ts):
    with pytest.raises(ValueError, match="Cz"):
        ts.pick(["Fz", "Cz"])


@pytest.mark.parametrize("index", [4, 10, -1])
def test_an_index_outside_the_channels_is_rejected(ts, index):
    with pytest.raises(ValueError, match="index"):
        ts.pick([index])


def test_duplicates_are_rejected_also_across_names_and_indices(ts):
    with pytest.raises(ValueError, match="duplicate"):
        ts.pick(["Fz", "Fz"])
    with pytest.raises(ValueError, match="duplicate"):
        ts.pick(["Fz", 0])


def test_an_empty_selection_is_rejected(ts):
    with pytest.raises(ValueError, match="at least one"):
        ts.pick([])


def test_a_single_string_is_not_treated_as_a_sequence_of_characters(ts):
    with pytest.raises(TypeError, match="sequence"):
        ts.pick("Fz")  # type: ignore[arg-type]


@pytest.mark.parametrize("entry", [True, 1.5, None])
def test_entries_must_be_names_or_integer_indices(ts, entry):
    with pytest.raises(TypeError, match="channel"):
        ts.pick([entry])  # type: ignore[list-item]
