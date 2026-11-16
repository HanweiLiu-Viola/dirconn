import pytest

from dirconn.io.readers import READERS, read_unit

ROI = "derivatives/dirconn-roi"
KINDS_ALL = ("cortical", "direct")


def _roi(bids_tiny, node_kinds):
    return read_unit(
        bids_tiny / ROI,
        subject="01",
        session=None,
        task="move",
        run=None,
        datatype="eeg",
        level="roi",
        description="DK",
        node_kinds=node_kinds,
    )


def test_roi_reader_returns_cortical_and_direct_nodes(bids_tiny):
    ts = _roi(bids_tiny, KINDS_ALL)
    assert list(ts.names) == ["precuneus-lh", "postcentral-lh", "superiorfrontal-lh", "STN-L"]
    assert {ch.node_kind for ch in ts.channels} == {"cortical", "direct"}


def test_node_kinds_filter_drops_direct_nodes(bids_tiny):
    ts = _roi(bids_tiny, ("cortical",))
    assert list(ts.names) == ["precuneus-lh", "postcentral-lh", "superiorfrontal-lh"]
    with pytest.raises(ValueError, match="cortical"):
        _roi(bids_tiny, ("direct",))


def test_sensor_reader_marks_bad_segments_invalid(bids_tiny):
    ts = read_unit(
        bids_tiny,
        subject="01",
        session=None,
        task="rest",
        run=None,
        datatype="eeg",
        level="sensor",
        description=None,
        node_kinds=(),
    )
    assert ts.has_mask
    summary = ts.mask_summary()
    assert 748 <= summary["n_samples"] - summary["n_valid"] <= 752


def test_unknown_pair_lists_known_readers(bids_tiny):
    assert ("eeg", "roi") in READERS
    with pytest.raises(KeyError, match=r"\('meg', 'sensor'\)"):
        read_unit(
            bids_tiny,
            subject="01",
            session=None,
            task="rest",
            run=None,
            datatype="emg",
            level="sensor",
            description=None,
            node_kinds=(),
        )
