import numpy as np
import pytest
from scipy import signal

from dirconn import GC, TimeSeries
from dirconn.io.readers import read_unit
from dirconn.workflows.config import LockConfig, PrepareConfig
from dirconn.workflows.steps.estimate import refuse_masked
from dirconn.workflows.steps.prepare import lock_events, pad_invalid, prepare, resample, zscore


def _synthetic(n=1000, sfreq=500.0, valid=None, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((3, n)) * [[1.0], [2.0], [3.0]] + [[1.0], [2.0], [3.0]]
    return TimeSeries.from_array(data, sfreq, valid=valid)


def _fixture_ts(root):
    return read_unit(
        root,
        subject="01",
        session=None,
        task="rest",
        run=None,
        datatype="eeg",
        level="sensor",
        description=None,
        node_kinds=(),
    )


def _events(root):
    return root / "sub-01" / "eeg" / "sub-01_task-rest_events.tsv"


def test_resample_halves_the_rate_and_cuts_the_edge_transients():
    ts = _synthetic(10000)
    out = resample(ts, 250.0)
    assert out.data.shape[-1] == 5000 - 2 * 10
    assert out.sfreq == 250.0
    assert out.tmin == pytest.approx(10 / 250.0)
    assert out.valid is None
    full = signal.resample_poly(np.asarray(ts.data), 1, 2, axis=-1)
    np.testing.assert_array_equal(out.data, full[..., 10:-10])
    assert out.history[-1] == {
        "step": "resample",
        "from": 500.0,
        "to": 250.0,
        "up": 1,
        "down": 2,
        "guard_samples": 10,
        "tmin": pytest.approx(0.04),
    }
    assert resample(ts, 500.0) is ts


def test_resample_keeps_invalid_regions_invalid():
    valid = np.ones(1000, dtype=bool)
    valid[400:500] = False
    out = resample(_synthetic(1000, valid=valid), 250.0)
    half = out.history[-1]["guard_samples"]
    expected = np.ones(500, dtype=bool)
    expected[200 - half : 250 + half] = False
    np.testing.assert_array_equal(out.valid[0], expected[half : 500 - half])


def test_resample_does_not_leak_artefacts_from_invalid_samples():
    data = np.zeros((2, 2000))
    data[:, 1001] = 1e6
    valid = np.ones(2000, dtype=bool)
    valid[1001] = False
    ts = TimeSeries.from_array(data, 500.0, valid=valid)
    out = resample(ts, 250.0)
    half = out.history[-1]["guard_samples"]
    assert np.abs(out.data[0][:, out.valid[0]]).max() < 1e-6
    # indices are shifted by the samples cut from the start
    assert not out.valid[0, 500 - 2 * half : 500 + 1].any()
    assert out.valid[0, 500 - 2 * half - 1] and out.valid[0, 500 + 1]
    # unmasked input gets no mask: the edge transients are cut, not masked
    plain = resample(_synthetic(1000), 250.0)
    assert plain.valid is None
    assert plain.n_times == 500 - 2 * half


def test_resample_2048_to_250_is_exact():
    out = resample(_synthetic(2048 * 2, sfreq=2048.0), 250.0)
    assert out.data.shape[-1] == 500 - 2 * out.history[-1]["guard_samples"]
    assert out.sfreq == 250.0
    assert out.history[-1]["up"] == 125
    assert out.history[-1]["down"] == 1024


def test_resample_rejects_inexact_ratio():
    with pytest.raises(ValueError, match="nearest representable"):
        resample(_synthetic(1000, sfreq=1000.0), 999.9999)


def test_resample_rejects_a_trial_too_short_to_keep_a_sample():
    with pytest.raises(ValueError, match="keeps no sample"):
        resample(_synthetic(80, sfreq=1000.0), 250.0)


def test_epochs_locked_after_cropping_match_epochs_locked_without_it():
    rng = np.random.default_rng(3)
    data = rng.standard_normal((2, 10000))
    ts = TimeSeries.from_array(data, 1000.0, tmin=1.5)
    events = [3.0, 6.25, 9.5]
    cropped = resample(ts, 250.0).lock(events, -0.5, 0.5)
    full = signal.resample_poly(data, 1, 4, axis=-1)
    reference = TimeSeries.from_array(full, 250.0, tmin=1.5).lock(events, -0.5, 0.5)
    np.testing.assert_array_equal(cropped.data, reference.data)
    assert cropped.tmin == reference.tmin


def test_resample_cuts_every_epoch_and_moves_tmin():
    data = np.random.default_rng(4).standard_normal((3, 2, 1000))
    ts = TimeSeries.from_array(data, 1000.0, segmentation="epoched", tmin=-0.5)
    out = resample(ts, 250.0)
    half = out.history[-1]["guard_samples"]
    assert out.data.shape == (3, 2, 250 - 2 * half)
    assert out.tmin == pytest.approx(-0.5 + half / 250.0)
    assert out.valid is None


def test_prepare_without_bad_segments_leaves_no_mask_for_unmasked_methods():
    rng = np.random.default_rng(5)
    x = np.zeros((2, 8000))
    for t in range(2, 8000):
        x[0, t] = 0.5 * x[0, t - 1] + rng.standard_normal()
        x[1, t] = 0.4 * x[1, t - 1] + 0.3 * x[0, t - 2] + rng.standard_normal()
    ts = TimeSeries.from_array(x, 1000.0)
    out = prepare(ts, PrepareConfig(resample_hz=250.0), events_tsv=None)
    assert out.valid is None
    refuse_masked(out, "gc", supports_mask=False)
    GC(order=3).fit(out)


def test_bad_segments_away_from_the_edges_stay_masked():
    valid = np.ones(4000, dtype=bool)
    valid[2000:2100] = False
    out = prepare(_synthetic(4000, valid=valid), PrepareConfig(resample_hz=250.0), events_tsv=None)
    assert out.valid is not None
    with pytest.raises(ValueError, match="does not support masked data"):
        refuse_masked(out, "gc", supports_mask=False)


def test_zscore_uses_only_valid_samples():
    valid = np.ones(1000, dtype=bool)
    valid[:300] = False
    data = np.array(_synthetic(1000).data[0])
    data[:, :300] += 100.0
    ts = TimeSeries.from_array(data, 500.0, valid=valid)
    out = zscore(ts)
    kept = out.data[0][:, valid]
    np.testing.assert_allclose(kept.mean(axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose(kept.std(axis=1), 1.0)
    assert out.history[-1] == {"step": "zscore", "scope": "per_channel_valid_samples"}


def test_zscore_rejects_flat_channel():
    data = np.random.default_rng(1).standard_normal((2, 100))
    data[1] = 4.0
    ts = TimeSeries.from_array(data, 100.0, names=["a", "flat"])
    with pytest.raises(ValueError, match="flat"):
        zscore(ts)


def test_bad_annotations_become_invalid_samples_with_padding(bids_tiny):
    ts = _fixture_ts(bids_tiny)
    cfg = PrepareConfig(resample_hz=None, zscore=False, pad_bad_seconds=0.5)
    out = prepare(ts, cfg, events_tsv=None)
    summary = out.mask_summary()
    assert abs(summary["n_samples"] - summary["n_valid"] - (750 + 4 * 250)) <= 2
    assert out.history[-1]["step"] == "pad_invalid"
    assert pad_invalid(_synthetic(), 0.5).history == ()


def test_lock_events_cuts_epochs_from_events_tsv(bids_tiny):
    ts = _fixture_ts(bids_tiny)
    out = lock_events(ts, _events(bids_tiny), LockConfig(events=("cue",), tmin=-0.1, tmax=0.4))
    assert out.data.shape[0] == 5
    assert out.segmentation == "epoched"
    assert lock_events(ts, None, LockConfig()) is ts
    with pytest.raises(ValueError, match="events.tsv"):
        lock_events(ts, None, LockConfig(events=("cue",)))
    with pytest.raises(ValueError, match="continuous"):
        lock_events(out, _events(bids_tiny), LockConfig(events=("cue",)))


def test_history_records_every_step_in_order(bids_tiny):
    ts = _fixture_ts(bids_tiny)
    cfg = PrepareConfig(
        resample_hz=250.0,
        zscore=True,
        pad_bad_seconds=0.1,
        lock=LockConfig(events=("cue",), tmin=-0.1, tmax=0.4),
    )
    out = prepare(ts, cfg, events_tsv=_events(bids_tiny))
    steps = [h["step"] for h in out.history[len(ts.history) :]]
    assert steps == ["pad_invalid", "resample", "zscore", "lock"]
    assert out.segmentation == "epoched"


def test_ignore_mode_keeps_the_data_and_drops_the_mask():
    valid = np.ones(1000, dtype=bool)
    valid[100:200] = False
    ts = _synthetic(1000, valid=valid)
    cfg = PrepareConfig(resample_hz=None, zscore=False, bad_segments="ignore", pad_bad_seconds=1.0)
    out = prepare(ts, cfg, events_tsv=None)
    assert out.valid is None
    np.testing.assert_array_equal(out.data, ts.data)
    assert out.history[-1] == {"step": "ignore_bad_segments", "n_invalid": 100}
    assert ts.valid is not None


def test_ignore_mode_has_nothing_to_drop_after_resampling_clean_data():
    ts = _synthetic(1000, sfreq=500.0)
    cfg = PrepareConfig(resample_hz=250.0, zscore=False, bad_segments="ignore")
    out = prepare(ts, cfg, events_tsv=None)
    assert out.sfreq == 250.0
    assert out.valid is None
    assert [h["step"] for h in out.history[len(ts.history) :]] == ["resample"]
