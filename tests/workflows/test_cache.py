"""Tests for the prepared-series cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dirconn.data.channels import ChannelMeta
from dirconn.data.timeseries import TimeSeries
from dirconn.workflows._cache import load_timeseries, save_timeseries


def test_cache_round_trips_a_masked_timeseries(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n_trials, n_channels, n_times = 2, 3, 50
    valid = np.ones((n_trials, n_times), dtype=bool)
    valid[0, 10:20] = False
    predict = np.ones((n_trials, n_times), dtype=bool)
    predict[1, :5] = False
    channels = (
        ChannelMeta("Cz", type="eeg"),
        ChannelMeta(
            "precentral-lh",
            node_kind="cortical",
            region="precentral",
            hemisphere="lh",
            atlas="DK",
            pos=(0.01, -0.02, 0.05),
        ),
        ChannelMeta("STN-L", node_kind="direct", region="STN", hemisphere="lh"),
    )
    ts = TimeSeries(
        rng.standard_normal((n_trials, n_channels, n_times)),
        sfreq=250.0,
        channels=channels,
        segmentation="epoched",
        tmin=-0.5,
        valid=valid,
        predict=predict,
        history=({"step": "zscore"}, {"step": "lock", "events": ["go"], "n": 2}),
    )
    path = tmp_path / "prepared.npz"
    save_timeseries(ts, path)
    assert path.is_file()
    assert path.with_suffix(".json").is_file()

    back = load_timeseries(path)
    np.testing.assert_array_equal(back.data, ts.data)
    assert back.sfreq == ts.sfreq
    assert back.segmentation == ts.segmentation
    assert back.tmin == ts.tmin
    assert back.channels == ts.channels
    assert back.valid is not None and back.predict is not None
    np.testing.assert_array_equal(back.valid, valid)
    np.testing.assert_array_equal(back.predict, predict)
    assert back.history == ts.history


def test_cache_round_trips_an_unmasked_timeseries(tmp_path: Path) -> None:
    ts = TimeSeries(
        np.arange(12, dtype=float).reshape(1, 2, 6),
        sfreq=100.0,
        channels=(ChannelMeta("a"), ChannelMeta("b")),
        segmentation="continuous",
    )
    path = tmp_path / "plain.npz"
    save_timeseries(ts, path)
    back = load_timeseries(path)
    np.testing.assert_array_equal(back.data, ts.data)
    assert back.valid is None and back.predict is None
    assert back.history == ()
