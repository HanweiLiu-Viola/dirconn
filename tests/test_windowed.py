"""WDTF: the sliding-window (or segment-wise) DTF, the cheap time-resolved baseline."""

import dataclasses

import numpy as np
import pytest

from dirconn import DTF, WDTF, Connectivity, DirconnWarning, TimeSeries, wdtf

SFREQ = 100.0


def chain(n_times, *, switch=None, seed=0, n_trials=1, tmin=0.0):
    """Two nodes; 0 -> 1 before ``switch`` (the whole recording if ``None``), 1 -> 0 after."""
    rng = np.random.default_rng(seed)
    before = np.array([[0.5, 0.0], [0.6, 0.4]])
    after = np.array([[0.5, 0.6], [0.0, 0.4]])
    out = np.zeros((n_trials, 2, n_times))
    for k in range(n_trials):
        x = np.zeros((2, n_times))
        for t in range(1, n_times):
            a = before if switch is None or t < switch else after
            x[:, t] = a @ x[:, t - 1] + rng.standard_normal(2)
        out[k] = x
    seg = "continuous" if n_trials == 1 else "epoched"
    return TimeSeries.from_array(
        out if n_trials > 1 else out[0], SFREQ, segmentation=seg, tmin=tmin
    )


def crop(ts, first, last):
    """The samples ``first .. last`` (inclusive) of every trial, as a TimeSeries."""
    return TimeSeries(
        data=ts.data[:, :, first : last + 1],
        sfreq=ts.sfreq,
        channels=ts.channels,
        segmentation=ts.segmentation,
        tmin=ts.tmin + first / ts.sfreq,
    )


# --------------------------------------------------------------------------
# What it computes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_trials", [1, 3])
def test_one_window_over_the_whole_recording_is_the_static_dtf(n_trials):
    ts = chain(800, seed=1, n_trials=n_trials)
    result = WDTF(window=8.0, order=2, n_freqs=17).fit(ts)
    static = DTF(order=2, n_freqs=17, check_order_stability=False).fit(ts)
    assert result.values.shape == (2, 2, 17, 1)
    np.testing.assert_allclose(result.values, static.values, rtol=1e-12)
    np.testing.assert_allclose(result.times, [(799 / 2) / SFREQ])


def test_every_window_is_the_dtf_of_that_window():
    ts = chain(1000, seed=2)
    result = WDTF(window=2.0, step=1.0, order=1, n_freqs=9, standardize=False).fit(ts)
    assert result.values.shape == (2, 2, 9, 9)
    for k, start in enumerate(range(0, 801, 100)):
        expected = DTF(order=1, n_freqs=9, standardize=False, check_order_stability=False).fit(
            crop(ts, start, start + 199)
        )
        np.testing.assert_allclose(result.values[..., k], expected.values[..., 0], rtol=1e-12)


def test_windows_and_their_centres_in_seconds():
    ts = chain(1000, seed=3, tmin=-2.0)
    result = WDTF(window=2.0, step=0.5, order=1, n_freqs=5).fit(ts)
    starts = np.arange(0, 801, 50)
    np.testing.assert_allclose(result.times, -2.0 + (starts + 99.5) / SFREQ)
    assert result.is_time_resolved and result.is_frequency_resolved
    assert result.provenance.details["n_windows"] == starts.size
    assert result.provenance.details["window_samples"] == 200
    assert result.provenance.details["step_samples"] == 50


def test_the_default_step_is_half_the_window():
    ts = chain(1000, seed=3)
    result = WDTF(window=2.0, order=1, n_freqs=5).fit(ts)
    assert result.provenance.details["step_samples"] == 100
    assert result.times.size == 9


@pytest.mark.filterwarnings("ignore:the smallest window:dirconn.exceptions.DirconnWarning")
def test_segments_are_fitted_one_by_one_and_labelled_by_their_centre():
    ts = chain(1000, seed=4, tmin=-1.0)
    segments = [(-0.5, 1.49), (2.0, 4.99)]
    result = WDTF(segments=segments, order=2, n_freqs=9, standardize=False).fit(ts)
    first = DTF(order=2, n_freqs=9, standardize=False, check_order_stability=False).fit(
        crop(ts, 50, 249)
    )
    second = DTF(order=2, n_freqs=9, standardize=False, check_order_stability=False).fit(
        crop(ts, 300, 599)
    )
    np.testing.assert_allclose(result.values[..., 0], first.values[..., 0], rtol=1e-12)
    np.testing.assert_allclose(result.values[..., 1], second.values[..., 0], rtol=1e-12)
    np.testing.assert_allclose(result.times, [-1.0 + 149.5 / SFREQ, -1.0 + 449.5 / SFREQ])
    assert result.provenance.details["segments"] == [[-0.5, 1.49], [2.0, 4.99]]


def test_both_ends_of_a_segment_are_included():
    ts = chain(600, seed=5)
    result = WDTF(segments=[(1.0, 2.0)], order=1, n_freqs=5, standardize=False).fit(ts)
    expected = DTF(order=1, n_freqs=5, standardize=False, check_order_stability=False).fit(
        crop(ts, 100, 200)  # 101 samples
    )
    np.testing.assert_allclose(result.values[..., 0], expected.values[..., 0], rtol=1e-12)


def test_the_order_is_chosen_once_on_the_whole_recording():
    ts = chain(1500, seed=6)
    chosen = DTF(order="bic", max_order=5, check_order_stability=False).fit(ts)
    result = WDTF(window=3.0, n_freqs=5).fit(ts)
    assert result.provenance.model_order == chosen.provenance.model_order
    assert result.provenance.details["order_selection"]["criterion"] == "bic"
    assert max(result.provenance.details["order_selection"]["orders"]) == 5


def test_standardising_uses_one_scale_for_all_windows():
    data = chain(1000, seed=7).data.copy()
    data[0, 1] *= 20.0
    ts = TimeSeries.from_array(data, SFREQ, segmentation="continuous")
    result = WDTF(window=2.0, order=1, n_freqs=5).fit(ts)
    assert len(result.provenance.preprocessing["scale"]) == 2
    assert result.provenance.preprocessing["standardized"] is True


@pytest.mark.filterwarnings("ignore:the smallest window:dirconn.exceptions.DirconnWarning")
@pytest.mark.parametrize("order", [1, "bic"])
def test_the_direction_follows_a_mid_recording_switch(order):
    n_times = 6000
    switch = n_times // 2
    ts = chain(n_times, switch=switch, seed=8)
    result = WDTF(window=2.0, step=0.5, order=order, n_freqs=17).fit(ts)
    band = result.values[..., 1:9, :].mean(axis=2)  # [source, target, time]
    before = result.times < (switch - 200) / SFREQ - 1.0
    after = result.times > (switch + 200) / SFREQ + 1.0
    assert band[0, 1, before].mean() > band[1, 0, before].mean()
    assert band[1, 0, after].mean() > band[0, 1, after].mean()


# --------------------------------------------------------------------------
# Provenance, checklist, warnings
# --------------------------------------------------------------------------


def test_the_checklist_and_details():
    ts = chain(3000, seed=9, n_trials=1)
    result = WDTF(window=5.0, order=1, n_freqs=5).fit(ts)
    prov = result.provenance
    assert prov.method == "wdtf"
    names = {c.name: c for c in prov.assumptions.checks}
    for key in ("stationarity", "sampling_adequacy", "order_source"):
        assert key in names
    assert names["sampling_adequacy"].status == "pass"
    assert names["order_source"].status == "not_checked"
    assert "trial_locking" not in names
    assert len(prov.details["spectral_radius"]) == prov.details["n_windows"]
    assert prov.details["min_rows"] == 500 - 1


def test_several_trials_are_pooled_within_every_window_and_the_assumption_is_listed():
    ts = chain(600, seed=10, n_trials=4)
    result = WDTF(window=2.0, order=1, n_freqs=5).fit(ts)
    names = {c.name: c for c in result.provenance.assumptions.checks}
    assert "trial_locking" in names
    assert "pooled" in names["trial_locking"].detail
    assert result.provenance.details["min_rows"] == 4 * (200 - 1)


def test_fewer_than_50_samples_per_coefficient_warns_but_runs():
    ts = chain(2000, seed=11)
    # 100 samples, order 1, 2 channels: 99 rows < 50 * 1 * 2
    with pytest.warns(DirconnWarning, match="rows"):
        result = WDTF(window=1.0, order=1, n_freqs=5).fit(ts)
    check = result.provenance.assumptions.get("sampling_adequacy")
    assert check.status == "warn"
    assert check.value == pytest.approx(2 / 99)


@pytest.mark.filterwarnings("ignore:.*:dirconn.exceptions.DirconnWarning")
def test_unstable_windows_are_counted_and_warn_above_five_percent():
    """Two noiseless-ish exponentials: every window fits coefficients 1.05 and 1.06."""
    rng = np.random.default_rng(0)
    t = np.arange(600)
    growing = np.stack([1.05**t, 1.06**t]) + 0.01 * rng.standard_normal((2, 600))
    ts = TimeSeries.from_array(growing, SFREQ)
    with pytest.warns(DirconnWarning, match="windows"):
        result = WDTF(window=2.0, order=1, n_freqs=5, standardize=False, demean=False).fit(ts)
    assert result.provenance.assumptions.get("stationarity").status == "warn"
    assert min(result.provenance.details["spectral_radius"]) > 1.0


def test_a_window_too_short_for_the_model_raises():
    ts = chain(1000, seed=12)
    with pytest.raises(ValueError, match="Too few samples"):
        WDTF(window=0.06, order=2, n_freqs=5).fit(ts)


def test_results_are_reproducible():
    ts = chain(1500, seed=13)
    a = WDTF(window=3.0, order=1, n_freqs=5).fit(ts)
    b = wdtf(ts, window=3.0, order=1, n_freqs=5)
    assert isinstance(b, Connectivity)
    np.testing.assert_array_equal(a.values, b.values)


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({}, "exactly one of window or segments"),
        ({"window": 1.0, "segments": [(0.0, 1.0)]}, "exactly one of window or segments"),
        ({"segments": [(0.0, 1.0)], "step": 0.5}, "step is only used with window"),
        ({"window": 0.0}, "window must be"),
        ({"window": -1.0}, "window must be"),
        ({"window": "long"}, "window must be"),
        ({"window": 1.0, "step": 0.0}, "step must be"),
        ({"window": 1.0, "step": -1.0}, "step must be"),
        ({"segments": []}, "segments must"),
        ({"segments": [(1.0,)]}, "segments must"),
        ({"segments": [(2.0, 1.0)]}, "start must not exceed"),
        ({"segments": [(0.0, 2.0), (1.0, 3.0)]}, "ascending and must not overlap"),
        ({"segments": [(2.0, 3.0), (0.0, 1.0)]}, "ascending and must not overlap"),
        ({"segments": [(0.0, float("nan"))]}, "finite"),
        ({"window": 1.0, "order": 0}, "order must be"),
        ({"window": 1.0, "n_freqs": 1}, "n_freqs must be"),
        ({"window": 1.0, "freqs": [1.0], "fmax": 5.0}, "either freqs or fmax"),
        ({"window": 1.0, "standardize": "yes"}, "standardize must be a bool"),
        ({"window": 1.0, "ridge": -1.0}, "ridge must be"),
    ],
)
def test_invalid_parameters_are_rejected_when_the_estimator_is_built(kwargs, message):
    with pytest.raises((ValueError, TypeError), match=message):
        WDTF(**kwargs)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"window": 0.015}, "multiple of the sampling interval"),
        ({"window": 1.0, "step": 0.015}, "multiple of the sampling interval"),
        ({"window": 20.0}, "longer than the recording"),
        ({"segments": [(-1.0, 1.0)]}, "outside the recording"),
        ({"segments": [(1.0, 99.0)]}, "outside the recording"),
        ({"segments": [(1.0, 1.001)]}, "no sample|only one sample|at least two"),
    ],
)
def test_parameters_that_do_not_fit_the_recording_raise_when_fitting(kwargs, message):
    ts = chain(1000, seed=14)
    with pytest.raises(ValueError, match=message):
        WDTF(order=1, n_freqs=5, **kwargs).fit(ts)


def test_segments_are_stored_as_tuples_and_the_estimator_is_frozen():
    estimator = WDTF(segments=[[0.0, 1.0], [2.0, 3.0]], order=1)
    assert estimator.segments == ((0.0, 1.0), (2.0, 3.0))
    assert estimator.get_params()["segments"] == ((0.0, 1.0), (2.0, 3.0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        estimator.window = 1.0  # type: ignore[misc]


def test_one_channel_is_rejected():
    ts = TimeSeries.from_array(np.random.default_rng(0).normal(size=(1, 500)), SFREQ)
    with pytest.raises(ValueError, match="at least 2 channels"):
        WDTF(window=2.0, order=1).fit(ts)


def test_the_info():
    info = WDTF.info
    assert info.name == "wdtf"
    assert info.directed and info.time_resolved and info.frequency_resolved
    assert info.value_range == (0.0, 1.0)
