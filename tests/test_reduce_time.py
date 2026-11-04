"""``Connectivity.reduce_time``: reduce the time axis over segments, like ``reduce_freq``."""

import numpy as np
import pytest

from dirconn import Connectivity, Provenance
from dirconn.result import TIME_REDUCTIONS


@pytest.fixture
def prov():
    return Provenance(method="adtf", parameters={"order": 2}, model_order=2, seed=None)


def movie(prov, *, n_times=11, n_freqs=3, values=None, times=None):
    """A time-resolved result with times 0.0, 0.1, ... and ``values[s, t, f, k]``."""
    shape = (2, 3, n_freqs, n_times)
    if values is None:
        values = np.arange(np.prod(shape), dtype=float).reshape(shape)
    return Connectivity.from_array(
        values,
        source=["A", "B"],
        target=["X", "Y", "Z"],
        freqs=np.arange(n_freqs) * 5.0,
        times=np.arange(n_times) * 0.1 if times is None else times,
        provenance=prov,
    )


def test_the_reductions_mirror_reduce_freq_with_peak_time():
    assert TIME_REDUCTIONS == ("mean", "median", "max", "peak_time")


def test_mean_uses_a_segment_with_both_ends_included(prov):
    conn = movie(prov)
    out = conn.reduce_time((0.2, 0.5))
    expected = conn.values[..., 2:6].mean(axis=3, keepdims=True)  # 0.2, 0.3, 0.4, 0.5
    np.testing.assert_allclose(out.values, expected)
    assert out.values.shape == (2, 3, 3, 1)


def test_the_time_coordinate_is_the_segment_centre_and_the_frequencies_are_kept(prov):
    conn = movie(prov)
    out = conn.reduce_time((0.2, 0.5))
    np.testing.assert_allclose(out.times, [0.35])
    assert out.is_time_resolved
    np.testing.assert_array_equal(out.freqs, conn.freqs)
    assert out.source == conn.source and out.target == conn.target


def test_several_segments_give_one_time_point_each(prov):
    conn = movie(prov)
    out = conn.reduce_time([(0.0, 0.2), (0.4, 0.5), (0.7, 1.0)])
    assert out.values.shape == (2, 3, 3, 3)
    np.testing.assert_allclose(out.times, [0.1, 0.45, 0.85])
    np.testing.assert_allclose(out.values[..., 1], conn.values[..., 4:6].mean(axis=3))
    np.testing.assert_allclose(out.values[..., 2], conn.values[..., 7:11].mean(axis=3))


def test_a_segment_that_is_one_sample_is_allowed(prov):
    out = movie(prov).reduce_time((0.3, 0.3))
    np.testing.assert_allclose(out.values[..., 0], movie(prov).values[..., 3])
    np.testing.assert_allclose(out.times, [0.3])


def test_the_edges_tolerate_floating_point_noise(prov):
    conn = movie(prov)
    out = conn.reduce_time((0.3, 0.6))  # the samples are 0.30000000000000004 .. 0.6000000000000001
    np.testing.assert_allclose(out.values[..., 0], conn.values[..., 3:7].mean(axis=3))


def test_median_max_and_peak_time_for_real_values(prov):
    values = np.zeros((2, 3, 3, 11))
    values[0, 0, 0, :] = [0.1, 0.9, 0.2, 0.5, 0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0]
    conn = movie(prov, values=values)
    segment = (0.1, 0.5)  # samples 1..5
    assert conn.reduce_time(segment, how="median").values[0, 0, 0, 0] == pytest.approx(0.5)
    assert conn.reduce_time(segment, how="max").values[0, 0, 0, 0] == 0.9
    peak = conn.reduce_time(segment, how="peak_time")
    assert peak.values[0, 0, 0, 0] == pytest.approx(0.1)  # seconds
    assert peak.provenance.history[-1]["values_unit"] == "s"


def test_peak_time_picks_the_first_time_on_ties(prov):
    values = np.zeros((2, 3, 3, 11))
    values[0, 0, 0, :] = [0.0, 1.0, 0.0, 1.0] + [0.0] * 7
    out = movie(prov, values=values).reduce_time((0.0, 1.0), how="peak_time")
    assert out.values[0, 0, 0, 0] == pytest.approx(0.1)


def test_complex_values_are_reduced_in_the_complex_plane(prov):
    rng = np.random.default_rng(0)
    values = rng.standard_normal((2, 3, 3, 11)) + 1j * rng.standard_normal((2, 3, 3, 11))
    conn = movie(prov, values=values)
    mean = conn.reduce_time((0.2, 0.6)).values
    np.testing.assert_allclose(mean[..., 0], values[..., 2:7].mean(axis=3))
    median = conn.reduce_time((0.2, 0.6), how="median").values
    np.testing.assert_allclose(median[..., 0].real, np.median(values[..., 2:7].real, axis=3))
    np.testing.assert_allclose(median[..., 0].imag, np.median(values[..., 2:7].imag, axis=3))
    largest = conn.reduce_time((0.2, 0.6), how="max").values[..., 0]
    magnitude = np.abs(values[..., 2:7])
    expected = np.take_along_axis(values[..., 2:7], magnitude.argmax(axis=3)[..., None], 3)[..., 0]
    np.testing.assert_allclose(largest, expected)


def test_nan_entries_are_ignored_and_an_all_nan_cell_stays_nan(prov):
    values = np.arange(2 * 3 * 3 * 11, dtype=float).reshape(2, 3, 3, 11)
    values[0, 0, 0, 3] = np.nan
    values[1, 1, 1, :] = np.nan
    out = movie(prov, values=values).reduce_time((0.2, 0.5))
    expected = np.nanmean(values[0, 0, 0, 2:6])
    assert out.values[0, 0, 0, 0] == pytest.approx(expected)
    assert np.isnan(out.values[1, 1, 1, 0])
    for how in ("median", "max", "peak_time"):
        assert np.isnan(
            movie(prov, values=values).reduce_time((0.2, 0.5), how=how).values[1, 1, 1, 0]
        )


def test_the_mean_reductions_over_time_and_frequency_commute(prov):
    conn = movie(prov, n_freqs=5)
    a = conn.reduce_time((0.2, 0.7)).reduce_freq((5.0, 15.0))
    b = conn.reduce_freq((5.0, 15.0)).reduce_time((0.2, 0.7))
    np.testing.assert_allclose(a.values, b.values)
    np.testing.assert_allclose(a.times, b.times)
    np.testing.assert_allclose(a.freqs, b.freqs)


def test_the_original_is_untouched_and_the_step_is_recorded(prov):
    conn = movie(prov)
    before = conn.values.copy()
    out = conn.reduce_time([(0.0, 0.2), (0.5, 0.9)], how="median")
    np.testing.assert_array_equal(conn.values, before)
    step = out.provenance.history[-1]
    assert step["op"] == "reduce_time"
    assert step["segments"] == [[0.0, 0.2], [0.5, 0.9]]
    assert step["how"] == "median"
    assert step["n_times"] == [3, 5]
    assert len(out.provenance.history) == len(conn.provenance.history) + 1


def test_a_list_of_lists_and_numpy_pairs_are_accepted(prov):
    conn = movie(prov)
    a = conn.reduce_time([[0.0, 0.2], [0.5, 0.9]])
    b = conn.reduce_time(np.array([[0.0, 0.2], [0.5, 0.9]]))
    np.testing.assert_array_equal(a.values, b.values)


def test_it_works_as_the_reduce_argument_of_the_statistics_functions(prov):
    from dirconn._resampling import make_reducer

    reducer = make_reducer(lambda r: r.reduce_freq((0.0, 10.0)).reduce_time((0.2, 0.5)))
    out = reducer(movie(prov))
    assert out.values.shape == (2, 3, 1, 1)


@pytest.mark.parametrize(
    "segments,message",
    [
        ([], "segments must"),
        ((0.1,), "segments must"),
        ("abc", "segments must"),
        ([(0.1, 0.2, 0.3)], "segments must"),
        ((0.5, 0.2), "start must not exceed"),
        ([(0.0, float("nan"))], "finite"),
        ([(0.0, 0.3), (0.2, 0.5)], "ascending and must not overlap"),
        ([(0.4, 0.5), (0.0, 0.1)], "ascending and must not overlap"),
        ([(0.0, 0.3), (0.3, 0.5)], "ascending and must not overlap"),
    ],
)
def test_invalid_segments_are_rejected(prov, segments, message):
    with pytest.raises(ValueError, match=message):
        movie(prov).reduce_time(segments)


@pytest.mark.parametrize(
    "segment,message",
    [
        ((-0.2, 0.3), "outside the time range"),
        ((0.5, 1.2), "outside the time range"),
        ((1.5, 2.0), "outside the time range"),
        ((0.31, 0.39), "no time point"),
    ],
)
def test_segments_outside_the_result_or_without_a_time_point_raise(prov, segment, message):
    with pytest.raises(ValueError, match=message):
        movie(prov).reduce_time(segment)


def test_a_result_without_a_time_axis_cannot_be_reduced(prov):
    conn = Connectivity.from_array(
        np.ones((2, 2, 3)), source=["A", "B"], freqs=[1.0, 2.0, 3.0], provenance=prov
    )
    with pytest.raises(ValueError, match="not time-resolved"):
        conn.reduce_time((0.0, 1.0))


def test_an_unknown_reduction_is_rejected(prov):
    with pytest.raises(ValueError, match="how must be one of"):
        movie(prov).reduce_time((0.0, 0.5), how="peak_freq")
