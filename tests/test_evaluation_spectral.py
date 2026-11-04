import warnings

import numpy as np
import pytest

from dirconn import Connectivity, DirconnWarning, Edge, Network, Provenance, dk_node
from dirconn.evaluation import (
    auprc,
    auroc,
    auroc_by_frequency,
    direction_accuracy,
    score_result,
    scoring_band,
    separation,
)

FREQS = np.arange(0.0, 51.0, 1.0)  # 0 to 50 Hz in 1 Hz steps


def spectral(values_fn, n=3):
    """A spectral result whose value at (i, j, f) is values_fn(i, j, f)."""
    values = np.zeros((n, n, FREQS.size, 1))
    for i in range(n):
        for j in range(n):
            for k, f in enumerate(FREQS):
                values[i, j, k, 0] = values_fn(i, j, f)
    values[np.arange(n), np.arange(n)] = np.nan
    return Connectivity.from_array(
        values,
        source=[f"n{i}" for i in range(n)],
        freqs=FREQS,
        provenance=Provenance(method="test"),
    )


def network(band=(8.0, 12.0)):
    names = ("a-lh", "b-lh", "c-lh")
    return Network(
        nodes=[dk_node(n) for n in names],
        edges=[Edge("a-lh", "b-lh", 0.4, 10.0, band), Edge("b-lh", "c-lh", 0.4, 10.0, band)],
    )


def peaked(i, j, f, band=(8.0, 12.0)):
    """High scores on the true edges inside the band, low noise elsewhere."""
    is_edge = (i, j) in ((0, 1), (1, 2))
    in_band = band[0] <= f <= band[1]
    return 1.0 + 0.01 * ((i * 7 + j * 3) % 5) if (is_edge and in_band) else 0.05 * ((i + 2 * j) % 4)


class TestScoreResult:
    def test_a_time_domain_result_is_scored_like_the_plain_metrics(self):
        scores = np.array([[np.nan, 0.9, 0.1], [0.2, np.nan, 0.8], [0.1, 0.3, np.nan]])
        truth = np.array([[np.nan, 1, 0], [0, np.nan, 1], [0, 0, np.nan]])
        result = Connectivity.from_array(
            scores, source=["a", "b", "c"], provenance=Provenance(method="t")
        )
        card = score_result(result, truth)
        assert card["auroc"] == auroc(scores, truth)
        assert card["auprc"] == auprc(scores, truth)
        assert card["direction_accuracy"] == direction_accuracy(scores, truth)
        assert card["separation"] == separation(scores, truth)
        assert card["n_edges"] == 2

    def test_a_spectral_result_is_reduced_to_the_band_first(self):
        result = spectral(peaked)
        truth = network().truth(FREQS)
        card = score_result(result, truth, band=(8.0, 12.0))
        reduced = result.reduce_freq((8.0, 12.0), "mean").values[:, :, 0, 0]
        edges = np.array([[np.nan, 1, 0], [0, np.nan, 1], [0, 0, np.nan]])
        assert card["auroc"] == auroc(reduced, edges)
        assert card["auroc"] == 1.0
        assert card["band"] == (8.0, 12.0)

    def test_a_band_where_the_truth_has_no_edge_is_undefined_not_a_number(self):
        # the edges only exist at 8 to 12 Hz, so there is nothing to find at 30 to 45 Hz
        result = spectral(peaked)
        truth = network().truth(FREQS)
        with pytest.warns(DirconnWarning):
            card = score_result(result, truth, band=(30.0, 45.0))
        assert np.isnan(card["auroc"]) and card["n_edges"] == 0

    def test_a_spectral_result_without_a_band_is_an_error(self):
        with pytest.raises(ValueError, match="band"):
            score_result(spectral(peaked), np.zeros((3, 3)))

    def test_a_band_is_refused_for_a_result_without_a_frequency_axis(self):
        result = Connectivity.from_array(
            np.ones((3, 3)), source=["a", "b", "c"], provenance=Provenance(method="t")
        )
        with pytest.raises(ValueError, match="frequency"):
            score_result(result, np.zeros((3, 3)), band=(8.0, 12.0))

    def test_a_truth_with_a_frequency_axis_counts_an_edge_present_anywhere_in_the_band(self):
        result = spectral(peaked)
        truth = network(band=(10.0, 20.0)).truth(FREQS)  # present only part of the scoring band
        card = score_result(result, truth, band=(8.0, 12.0))
        assert card["n_edges"] == 2

    def test_a_truth_with_another_frequency_axis_is_refused(self):
        result = spectral(peaked)
        truth = network().truth(np.arange(0.0, 40.0, 2.0))
        with pytest.raises(ValueError, match="frequenc"):
            score_result(result, truth, band=(8.0, 12.0))

    def test_shapes_must_agree(self):
        result = spectral(peaked)
        with pytest.raises(ValueError, match="shape"):
            score_result(result, np.zeros((4, 4)), band=(8.0, 12.0))

    def test_undefined_metrics_are_nan_and_warn_once_each(self):
        result = spectral(peaked)
        with pytest.warns(DirconnWarning):
            card = score_result(result, np.zeros((3, 3)), band=(8.0, 12.0))
        assert np.isnan(card["auroc"]) and np.isnan(card["auprc"]) and card["n_edges"] == 0


class TestAurocByFrequency:
    def test_the_curve_is_high_inside_the_band_and_not_outside(self):
        result = spectral(peaked)
        truth = network().truth(FREQS)
        freqs, curve = auroc_by_frequency(result, truth)
        np.testing.assert_array_equal(freqs, FREQS)
        inside = (freqs >= 8) & (freqs <= 12)
        assert (curve[inside] == 1.0).all()
        # outside the band the truth has no edge at all: undefined, not 0.5
        assert np.isnan(curve[~inside]).all()

    def test_it_does_not_warn_for_the_frequencies_where_it_is_undefined(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DirconnWarning)
            auroc_by_frequency(spectral(peaked), network().truth(FREQS))

    def test_a_broadband_truth_matrix_is_used_at_every_frequency(self):
        edges = np.array([[np.nan, 1, 0], [0, np.nan, 1], [0, 0, np.nan]])
        freqs, curve = auroc_by_frequency(spectral(peaked), edges)
        inside = (freqs >= 8) & (freqs <= 12)
        assert (curve[inside] == 1.0).all()
        assert (curve[~inside] < 1.0).all()  # the edges exist but the method does not see them

    def test_a_wrong_band_shows_up_as_a_low_curve_where_the_truth_is(self):
        # scores peak at 20 to 24 Hz but the edges live at 8 to 12 Hz
        result = spectral(lambda i, j, f: peaked(i, j, f, band=(20.0, 24.0)))
        freqs, curve = auroc_by_frequency(result, network().truth(FREQS))
        inside = (freqs >= 8) & (freqs <= 12)
        assert (curve[inside] < 0.8).all()

    def test_a_result_without_a_frequency_axis_is_refused(self):
        result = Connectivity.from_array(
            np.ones((3, 3)), source=["a", "b", "c"], provenance=Provenance(method="t")
        )
        with pytest.raises(ValueError, match="frequency"):
            auroc_by_frequency(result, np.zeros((3, 3)))


class TestScoringBand:
    def test_it_is_the_hull_of_the_edge_bands(self):
        names = ("a-lh", "b-lh", "c-lh")
        net = Network(
            nodes=[dk_node(n) for n in names],
            edges=[
                Edge("a-lh", "b-lh", 0.3, 10.0, (4.0, 8.0)),
                Edge("b-lh", "c-lh", 0.3, 10.0, (10.0, 20.0)),
            ],
        )
        assert scoring_band(net) == (4.0, 20.0)

    def test_a_broadband_network_needs_an_explicit_default(self):
        names = ("a-lh", "b-lh")
        net = Network(nodes=[dk_node(n) for n in names], edges=[Edge("a-lh", "b-lh", 0.3, 10.0)])
        assert scoring_band(net, default=(1.0, 40.0)) == (1.0, 40.0)
        with pytest.raises(ValueError, match="broadband"):
            scoring_band(net)

    def test_a_mixed_network_uses_the_banded_edges_only(self):
        names = ("a-lh", "b-lh", "c-lh")
        net = Network(
            nodes=[dk_node(n) for n in names],
            edges=[Edge("a-lh", "b-lh", 0.3, 10.0), Edge("b-lh", "c-lh", 0.3, 10.0, (8.0, 12.0))],
        )
        assert scoring_band(net) == (8.0, 12.0)
