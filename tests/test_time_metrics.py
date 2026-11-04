"""Time-resolved metrics: AUROC over time, the guard band, stable scores, tracking latency."""

import dataclasses

import numpy as np
import pytest

from dirconn import ADTF, DTF, WDTF, Connectivity, DirconnWarning, Provenance, TimeSeries
from dirconn.evaluation import (
    auroc_by_time,
    guard_seconds,
    score_stable,
    stable_mask,
    switch_times,
    tracking_latency,
    truth_switch_times,
)
from dirconn.network import Edge, Network, dk_node

A, B, C = "bankssts-lh", "superiorparietal-lh", "precuneus-rh"
SFREQ = 100.0

# The default settings remember only a few samples, so the ADTF fits here carry
# the sampling warning; it is not what these tests look at.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*coefficients per equation:dirconn.exceptions.DirconnWarning"
)


def net(*edges, nodes=(A, B, C)):
    return Network(tuple(dk_node(n) for n in nodes), tuple(edges), name="t")


def result_from(values, times, *, freqs=None, names=(A, B, C)):
    """A time-resolved result from ``values[source, target, freq, time]`` (3 nodes)."""
    values = np.asarray(values, dtype=float)
    return Connectivity.from_array(
        values,
        source=list(names),
        freqs=freqs,
        times=times,
        provenance=Provenance(method="adtf", parameters={}),
    )


def following(truth, noise=0.0, seed=0):
    """A result that is exactly the truth (plus noise), so every ranking is perfect."""
    rng = np.random.default_rng(seed)
    values = np.nan_to_num(truth.values, nan=0.0) + noise * rng.random(truth.values.shape)
    return result_from(values, truth.times, freqs=None if truth.freqs.size == 1 else truth.freqs)


# --------------------------------------------------------------------------
# Switch times
# --------------------------------------------------------------------------


def test_the_initial_state_and_the_end_of_the_recording_are_not_switches():
    network = net(Edge(A, B, 0.4, active=[(0.0, 10.0)]), Edge(B, A, 0.4, active=[(10.0, 20.0)]))
    assert switch_times(network, duration=20.0) == [10.0]
    assert switch_times(network) == [10.0, 20.0]  # without the duration the turn-off is a switch


def test_a_burst_has_two_switches():
    network = net(Edge(A, B, 0.4, active=[(5.0, 6.0)]))
    assert switch_times(network, duration=30.0) == [5.0, 6.0]


def test_switches_of_one_pair_only():
    network = net(Edge(A, B, 0.4, active=[(5.0, 6.0)]), Edge(B, C, 0.4, active=[(8.0, 9.0)]))
    assert switch_times(network, pair=(A, B)) == [5.0, 6.0]
    assert switch_times(network, pair=(B, C)) == [8.0, 9.0]
    assert switch_times(network, pair=(C, A)) == []


def test_a_change_of_strength_of_one_pair_is_not_a_switch():
    network = net(Edge(A, B, 0.6, active=[(0.0, 5.0)]), Edge(A, B, 0.2, active=[(5.0, 10.0)]))
    assert switch_times(network, duration=10.0) == []


def test_a_static_network_has_no_switches():
    assert switch_times(net(Edge(A, B, 0.4))) == []


def test_an_unknown_pair_is_rejected():
    with pytest.raises(ValueError, match="not in the network"):
        switch_times(net(Edge(A, B, 0.4)), pair=(A, "nowhere"))


def test_switches_can_be_read_off_a_truth_over_time():
    network = net(Edge(A, B, 0.4, active=[(0.0, 3.0)]), Edge(B, A, 0.4, active=[(3.0, 6.0)]))
    truth = network.truth(times=np.arange(6.0))
    assert truth_switch_times(truth) == [3.0]  # the first time point of the new state


# --------------------------------------------------------------------------
# The guard band
# --------------------------------------------------------------------------


def test_the_guard_band_follows_each_switch():
    times = np.arange(0.0, 10.0, 0.5)
    mask = stable_mask(times, [3.0, 7.0], 1.0)
    unstable = (times >= 3.0) & (times < 4.0) | (times >= 7.0) & (times < 8.0)
    np.testing.assert_array_equal(mask, ~unstable)


def test_no_switches_or_no_guard_leaves_everything_stable():
    times = np.arange(0.0, 5.0, 0.5)
    assert stable_mask(times, [], 2.0).all()
    assert stable_mask(times, [2.0], 0.0).all()


def test_the_guard_must_not_be_negative():
    with pytest.raises(ValueError, match="guard_s"):
        stable_mask([0.0, 1.0], [0.5], -1.0)


def chain_data(n_times=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((2, n_times))
    a = np.array([[0.5, 0.0], [0.6, 0.4]])
    for t in range(1, n_times):
        x[:, t] = a @ x[:, t - 1] + rng.standard_normal(2)
    return TimeSeries.from_array(x, SFREQ)


def test_the_guard_is_the_measured_memory_of_the_method_in_seconds():
    ts = chain_data()
    adtf = ADTF(order=1, forgetting=0.99, n_freqs=5).fit(ts)
    memory = adtf.provenance.details["memory_samples"]
    assert guard_seconds(adtf) == pytest.approx(memory / SFREQ)
    assert 2.0 < memory < 100.0  # a few samples: far below the nominal 1 / (1 - 0.99) = 100
    wdtf = WDTF(window=2.0, order=1, n_freqs=5).fit(ts)
    assert guard_seconds(wdtf) == pytest.approx(2.0)


def test_a_method_without_a_memory_needs_an_explicit_guard():
    ts = chain_data()
    with pytest.raises(ValueError, match="guard_s"):
        guard_seconds(WDTF(segments=[(1.0, 5.0)], order=1, n_freqs=5).fit(ts))
    with pytest.raises(ValueError, match="guard_s"):
        guard_seconds(DTF(order=1, n_freqs=5, check_order_stability=False).fit(ts))


def test_a_filter_that_never_updates_has_no_finite_memory():
    ts = chain_data(1500)
    result = ADTF(order=1, n_freqs=5).fit(ts)
    details = {**result.provenance.details, "memory_samples": None}
    never = result.with_provenance(dataclasses.replace(result.provenance, details=details))
    with pytest.raises(ValueError, match="guard_s"):
        guard_seconds(never)


# --------------------------------------------------------------------------
# AUROC over time
# --------------------------------------------------------------------------


def test_a_result_that_follows_the_truth_has_auroc_one_where_it_is_defined():
    network = net(Edge(A, B, 0.4, active=[(2.0, 4.0)]))
    truth = network.truth(times=np.arange(6.0))
    times, curve = auroc_by_time(following(truth), truth)
    np.testing.assert_array_equal(times, truth.times)
    # no edge at all outside [2, 4): undefined there, never 0
    np.testing.assert_array_equal(curve[[0, 1, 4, 5]], np.full(4, np.nan))
    np.testing.assert_array_equal(curve[[2, 3]], [1.0, 1.0])


def test_a_result_that_lags_the_truth_scores_below_one_around_the_switch():
    network = net(Edge(A, B, 0.4), Edge(B, C, 0.4, active=[(3.0, 6.0)]))
    truth = network.truth(times=np.arange(6.0))
    values = np.nan_to_num(truth.values, nan=0.0).copy()
    values[1, 2, 0, :] = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0]  # B -> C appears one step late
    values[2, 0, 0, :] = 0.3  # a false edge that is stronger than the late one for a while
    values[0, 1, 0, :] = 1.0
    times, curve = auroc_by_time(result_from(values, truth.times), truth)
    assert curve[3] < 1.0
    np.testing.assert_array_equal(curve[[4, 5]], [1.0, 1.0])


def test_a_band_is_averaged_and_the_truth_collapsed_like_score_result():
    network = net(Edge(A, B, 0.4, delay_ms=10.0, band=(8.0, 12.0), active=[(2.0, 4.0)]))
    freqs = np.array([5.0, 10.0, 20.0])
    times = np.arange(6.0)
    truth = network.truth(freqs=freqs, times=times)
    values = np.zeros((3, 3, 3, 6))
    values[0, 1, 1, 2:4] = 1.0  # only at 10 Hz and only while the edge acts
    values[1, 0, 0, :] = 0.3  # a false edge outside the band
    result = result_from(values, times, freqs=freqs)
    _, curve = auroc_by_time(result, truth, band=(8.0, 12.0))
    np.testing.assert_array_equal(curve[[2, 3]], [1.0, 1.0])
    assert np.isnan(curve[[0, 1, 4, 5]]).all()


def test_the_time_axes_of_result_and_truth_must_match():
    network = net(Edge(A, B, 0.4, active=[(2.0, 4.0)]))
    truth = network.truth(times=np.arange(6.0))
    other = following(network.truth(times=np.arange(6.0) + 0.5))
    with pytest.raises(ValueError, match="time axes"):
        auroc_by_time(other, truth)


def test_a_result_without_a_time_axis_is_refused():
    network = net(Edge(A, B, 0.4))
    static = Connectivity.from_array(
        np.ones((3, 3)), source=[A, B, C], provenance=Provenance(method="dtf", parameters={})
    )
    with pytest.raises(ValueError, match="time-resolved"):
        auroc_by_time(static, network.truth())


def test_a_spectral_result_needs_a_band():
    network = net(Edge(A, B, 0.4, active=[(1.0, 3.0)]))
    freqs = np.array([5.0, 10.0])
    truth = network.truth(freqs=freqs, times=np.arange(4.0))
    result = result_from(np.ones((3, 3, 2, 4)), truth.times, freqs=freqs)
    with pytest.raises(ValueError, match="band"):
        auroc_by_time(result, truth)


# --------------------------------------------------------------------------
# Scores inside the stable segments
# --------------------------------------------------------------------------


def switching_case():
    """A -> B for 6 s, then B -> A; the scores follow the truth except just after 6 s."""
    network = net(
        Edge(A, B, 0.4, active=[(0.0, 6.0)]),
        Edge(B, A, 0.4, active=[(6.0, 12.0)]),
        Edge(B, C, 0.4),
    )
    times = np.arange(0.0, 12.0, 0.5)
    truth = network.truth(times=times)
    values = np.nan_to_num(truth.values, nan=0.0).copy()
    after = (times >= 6.0) & (times < 8.0)
    values[0, 1, 0, after] = 0.9  # the old direction lingers: wrong during the transition
    values[1, 0, 0, after] = 0.1
    values[1, 2, 0, :] = 1.0
    return network, truth, result_from(values, times)


def test_the_stable_segments_hide_the_transition_that_the_full_curve_shows():
    network, truth, result = switching_case()
    switches = switch_times(network, duration=12.0)
    full = score_stable(result, truth, guard_s=0.0, switches=switches)
    guarded = score_stable(result, truth, guard_s=2.0, switches=switches)
    assert guarded["direction_accuracy"] == 1.0
    assert full["direction_accuracy"] < 1.0
    assert guarded["n_stable"] == 20
    assert full["n_stable"] == 24
    assert guarded["guard_s"] == 2.0
    assert not guarded["stable"][(guarded["times"] >= 6.0) & (guarded["times"] < 8.0)].any()


def test_every_metric_is_reported_with_the_whole_curve_kept():
    network, truth, result = switching_case()
    scores = score_stable(result, truth, guard_s=2.0, switches=switch_times(network, duration=12.0))
    assert set(scores) >= {
        "auroc",
        "direction_accuracy",
        "separation",
        "n_stable",
        "n_defined",
        "guard_s",
        "times",
        "stable",
        "auroc_curve",
    }
    assert scores["auroc"] == pytest.approx(1.0)
    assert scores["separation"] > 1.0
    assert scores["auroc_curve"].shape == scores["times"].shape


def test_the_switches_default_to_those_of_the_truth_and_the_guard_to_the_memory():
    ts = chain_data(4000, seed=1)
    network = net(
        Edge(A, B, 0.4, active=[(0.0, 20.0)]), Edge(B, A, 0.4, active=[(20.0, 40.0)]), nodes=(A, B)
    )
    result = ADTF(order=1, forgetting=0.98, n_freqs=5, time_step=0.1).fit(ts)
    truth = network.truth(freqs=result.freqs, times=result.times)
    scores = score_stable(result, truth, band=(0.0, 50.0))
    assert scores["guard_s"] == pytest.approx(result.provenance.details["memory_samples"] / SFREQ)
    assert scores["switches"] == [truth.times[np.argmax(truth.values[1, 0, 0] > 0)]]


def test_undefined_time_points_are_skipped_without_a_warning():
    network = net(Edge(A, B, 0.4, active=[(2.0, 4.0)]))
    truth = network.truth(times=np.arange(6.0))
    scores = score_stable(following(truth), truth, guard_s=0.0, switches=[])
    assert scores["n_defined"] == 2 and scores["n_stable"] == 6
    assert scores["auroc"] == 1.0


def test_with_nothing_defined_the_metrics_are_nan_not_zero():
    network = net(Edge(A, B, 0.4, active=[(10.0, 12.0)]))
    truth = network.truth(times=np.arange(6.0))
    with pytest.warns(DirconnWarning, match="No time point"):
        scores = score_stable(following(truth), truth, guard_s=0.0, switches=[])
    assert np.isnan(scores["auroc"]) and np.isnan(scores["direction_accuracy"])
    assert scores["n_defined"] == 0


# --------------------------------------------------------------------------
# Tracking latency
# --------------------------------------------------------------------------


def step_result(times, scores_ab, *, others=0.0):
    values = np.full((3, 3, 1, times.size), others)
    values[0, 1, 0, :] = scores_ab
    return result_from(values, times)


def test_the_latency_is_the_time_to_cover_half_the_way_between_the_two_levels():
    times = np.arange(0.0, 30.0, 0.1)
    scores = np.where(times < 10.0, 0.2, 0.8 - 0.6 * np.exp(-(times - 10.0) / 1.0))
    out = tracking_latency(step_result(times, scores), A, B, 10.0, switches=[10.0], guard_s=4.0)
    assert out["tracked"] is True
    assert out["latency_s"] == pytest.approx(np.log(2.0), abs=0.11)  # progress 0.5 at ln 2 s
    assert out["pre_level"] == pytest.approx(0.2)
    assert out["post_level"] == pytest.approx(0.8, abs=0.02)
    assert out["edge"] == (A, B) and out["switch"] == 10.0


def test_a_switch_down_is_measured_the_same_way():
    times = np.arange(0.0, 30.0, 0.1)
    scores = np.where(times < 10.0, 0.8, 0.2 + 0.6 * np.exp(-(times - 10.0) / 2.0))
    out = tracking_latency(step_result(times, scores), A, B, 10.0, switches=[10.0], guard_s=8.0)
    assert out["tracked"] is True
    assert out["latency_s"] == pytest.approx(2.0 * np.log(2.0), abs=0.11)
    assert out["post_level"] < out["pre_level"]


def test_a_late_arrival_is_a_long_latency_and_a_flat_score_is_not_tracked():
    times = np.arange(0.0, 30.0, 0.1)
    scores = np.full(times.size, 0.2)
    scores[(times >= 25.0)] = 0.8  # arrives, but only late in the post window
    scores[(times > 10.0) & (times < 25.0)] = 0.25
    out = tracking_latency(step_result(times, scores), A, B, 10.0, switches=[10.0], guard_s=6.0)
    assert out["tracked"] is True and out["latency_s"] > 14.0
    flat = tracking_latency(
        step_result(times, np.where(times < 10.0, 0.2, 0.2 + 1e-12)),
        A,
        B,
        10.0,
        switches=[10.0],
        guard_s=4.0,
    )
    assert flat["tracked"] is False and np.isnan(flat["latency_s"])


def test_a_score_that_never_stays_beyond_half_way_is_not_tracked():
    times = np.arange(0.0, 30.0, 0.1)
    scores = np.where(times < 10.0, 0.2, 0.8)
    post = times >= 10.0
    scores[post] = np.where(
        np.arange(post.sum()) % 2 == 0, 1.4, 0.2
    )  # mean 0.8, never two in a row
    out = tracking_latency(
        step_result(times, scores), A, B, 10.0, switches=[10.0], guard_s=4.0, hold_points=2
    )
    assert out["tracked"] is False and np.isnan(out["latency_s"])
    assert "consecutive" in out["reason"]
    assert out["post_level"] == pytest.approx(0.8, abs=0.02)


def test_a_level_that_does_not_change_cannot_be_tracked():
    times = np.arange(0.0, 30.0, 0.1)
    out = tracking_latency(
        step_result(times, np.full(times.size, 0.5)), A, B, 10.0, switches=[10.0], guard_s=4.0
    )
    assert out["tracked"] is False
    assert "does not change" in out["reason"]


def test_the_levels_use_only_the_stable_part_between_neighbouring_switches():
    times = np.arange(0.0, 40.0, 0.1)
    # levels 0.2 | 0.8 | 0.4 with switches at 10 and 25
    scores = np.where(times < 10.0, 0.2, np.where(times < 25.0, 0.8, 0.4))
    out = tracking_latency(
        step_result(times, scores), A, B, 25.0, switches=[10.0, 25.0], guard_s=3.0
    )
    assert out["pre_level"] == pytest.approx(0.8)
    assert out["post_level"] == pytest.approx(0.4)
    assert out["latency_s"] == pytest.approx(0.0, abs=0.11)


def test_an_empty_stable_window_is_reported_not_guessed():
    times = np.arange(0.0, 30.0, 0.1)
    out = tracking_latency(
        step_result(times, np.where(times < 10.0, 0.2, 0.8)),
        A,
        B,
        10.0,
        switches=[9.0, 10.0],
        guard_s=5.0,
    )
    assert out["tracked"] is False
    assert "stable" in out["reason"]


def test_a_noise_spike_counts_only_when_it_is_held():
    times = np.arange(0.0, 30.0, 0.1)
    scores = np.where(times < 10.0, 0.2, 0.8 - 0.6 * np.exp(-(times - 10.0) / 2.0))
    scores[np.argmin(np.abs(times - 10.2))] = 0.9  # one stray point
    quick = tracking_latency(
        step_result(times, scores), A, B, 10.0, switches=[10.0], guard_s=8.0, hold_points=1
    )
    held = tracking_latency(
        step_result(times, scores), A, B, 10.0, switches=[10.0], guard_s=8.0, hold_points=3
    )
    assert quick["latency_s"] == pytest.approx(0.2, abs=0.11)
    assert held["latency_s"] == pytest.approx(2.0 * np.log(2.0), abs=0.15)


def test_a_band_is_reduced_before_the_latency_is_measured():
    freqs = np.array([5.0, 10.0, 20.0])
    times = np.arange(0.0, 30.0, 0.1)
    values = np.zeros((3, 3, 3, times.size))
    values[0, 1, 1, :] = np.where(times < 10.0, 0.2, 0.8)
    values[0, 1, 0, :] = 5.0  # outside the band, must be ignored
    out = tracking_latency(
        result_from(values, times, freqs=freqs),
        A,
        B,
        10.0,
        switches=[10.0],
        guard_s=4.0,
        band=(8.0, 12.0),
    )
    assert out["tracked"] is True and out["pre_level"] == pytest.approx(0.2)


def test_unknown_edges_and_bad_parameters_are_rejected():
    times = np.arange(0.0, 30.0, 0.1)
    result = step_result(times, np.zeros(times.size))
    with pytest.raises(ValueError, match="not in the result"):
        tracking_latency(result, A, "nowhere", 10.0, switches=[10.0], guard_s=1.0)
    with pytest.raises(ValueError, match="hold_points"):
        tracking_latency(result, A, B, 10.0, switches=[10.0], guard_s=1.0, hold_points=0)
    with pytest.raises(ValueError, match="switch"):
        tracking_latency(result, A, B, 10.0, switches=[20.0], guard_s=1.0)


# --------------------------------------------------------------------------
# The same latency on a bare course (for example the mean over many datasets)
# --------------------------------------------------------------------------


def test_latency_from_course_equals_tracking_latency_on_the_same_course():
    from dirconn.evaluation import latency_from_course

    times = np.arange(0.0, 30.0, 0.1)
    scores = np.where(times < 10.0, 0.2, 0.8 - 0.6 * np.exp(-(times - 10.0) / 1.0))
    full = tracking_latency(step_result(times, scores), A, B, 10.0, switches=[10.0], guard_s=4.0)
    bare = latency_from_course(times, scores, 10.0, switches=[10.0], guard_s=4.0)
    for key in ("tracked", "latency_s", "pre_level", "post_level", "reason", "switch", "guard_s"):
        assert bare[key] == full[key] or (np.isnan(bare[key]) and np.isnan(full[key]))
    assert "edge" not in bare and full["edge"] == (A, B)


def test_latency_from_course_checks_its_arguments():
    from dirconn.evaluation import latency_from_course

    times = np.arange(0.0, 30.0, 0.1)
    with pytest.raises(ValueError, match="same length"):
        latency_from_course(times, np.zeros(3), 10.0, switches=[10.0], guard_s=1.0)
    with pytest.raises(ValueError, match="guard_s"):
        latency_from_course(times, np.zeros(times.size), 10.0, switches=[10.0], guard_s=-1.0)


def test_averaging_many_noisy_courses_gives_a_latency_the_single_ones_cannot():
    """Single noisy courses cross half way by chance at once; their mean follows the step."""
    from dirconn.evaluation import latency_from_course

    times = np.arange(0.0, 40.0, 0.1)
    truth = np.where(times < 20.0, 0.1, 0.1 + 0.2 * (1 - np.exp(-(times - 20.0) / 3.0)))
    rng = np.random.default_rng(0)
    courses = truth + 0.15 * rng.standard_normal((30, times.size))
    single = [
        latency_from_course(times, c, 20.0, switches=[20.0], guard_s=8.0, hold_points=1)
        for c in courses
    ]
    mean = latency_from_course(times, courses.mean(axis=0), 20.0, switches=[20.0], guard_s=8.0)
    assert min(s["latency_s"] for s in single if s["tracked"]) < 0.3  # chance crossings
    assert mean["latency_s"] == pytest.approx(3.0 * np.log(2.0), abs=0.6)
