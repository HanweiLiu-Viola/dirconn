import warnings

import numpy as np
import pytest

from dirconn import Connectivity, DirconnWarning, Edge, Network, Provenance, dk_node
from dirconn.evaluation import auprc, auroc, chance_auprc, direction_accuracy, edge_density


def off_diagonal(matrix):
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


def brute_force_auroc(scores, truth):
    """Probability that a random true edge outranks a random non-edge; ties count one half."""
    positives = scores[truth]
    negatives = scores[~truth]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def brute_force_average_precision(scores, truth):
    order = np.argsort(-scores, kind="mergesort")
    hits = truth[order]
    precision = [hits[: k + 1].mean() for k in range(len(hits)) if hits[k]]
    return float(np.mean(precision))


def make_truth(n=5, edges=((0, 1), (1, 2), (3, 2))):
    truth = np.zeros((n, n))
    for i, j in edges:
        truth[i, j] = 1.0
    return truth


class TestAuroc:
    def test_a_perfect_ranking_scores_one_and_an_inverted_one_zero(self):
        truth = make_truth()
        perfect = truth * 0.9 + 0.05
        assert auroc(perfect, truth) == 1.0
        assert auroc(1.0 - perfect, truth) == 0.0

    def test_it_matches_a_brute_force_count_on_random_scores(self):
        rng = np.random.default_rng(0)
        truth = make_truth(6, ((0, 1), (1, 2), (3, 2), (4, 5)))
        for _ in range(5):
            scores = rng.random((6, 6))
            expected = brute_force_auroc(off_diagonal(scores), off_diagonal(truth) != 0)
            assert auroc(scores, truth) == pytest.approx(expected, rel=1e-12)

    def test_ties_count_one_half(self):
        truth = make_truth()
        assert auroc(np.ones((5, 5)), truth) == pytest.approx(0.5)

    def test_the_diagonal_is_ignored_unless_asked_for(self):
        truth = make_truth()
        scores = truth * 0.9 + 0.05
        loud = scores + np.eye(5) * 100.0
        assert auroc(loud, truth) == 1.0
        assert auroc(loud, truth, exclude_diagonal=False) < 1.0

    def test_non_finite_scores_are_dropped_with_a_warning(self):
        truth = make_truth()
        scores = truth * 0.9 + 0.05
        scores[4, 0] = np.nan
        with pytest.warns(DirconnWarning, match="non-finite"):
            assert auroc(scores, truth) == 1.0

    def test_an_undefined_metric_is_nan_with_a_warning_never_zero(self):
        with pytest.warns(DirconnWarning, match="undefined"):
            assert np.isnan(auroc(np.random.default_rng(0).random((4, 4)), np.zeros((4, 4))))
        with pytest.warns(DirconnWarning, match="undefined"):
            assert np.isnan(auroc(np.random.default_rng(0).random((3, 3)), np.ones((3, 3))))

    def test_shapes_must_agree_and_be_square(self):
        with pytest.raises(ValueError, match="shape"):
            auroc(np.zeros((3, 3)), np.zeros((4, 4)))
        with pytest.raises(ValueError, match="square"):
            auroc(np.zeros((3, 4)), np.zeros((3, 4)))


class TestConnectivityInputs:
    def provenance(self):
        return Provenance(method="x")

    def test_a_broadband_result_and_the_network_truth_can_be_used_directly(self):
        net = Network(
            nodes=[dk_node(f"{n}-lh") for n in ("bankssts", "insula", "precuneus")],
            edges=[Edge("bankssts-lh", "insula-lh", strength=1.0)],
        )
        truth = net.truth()
        scores = Connectivity.from_array(
            np.array([[0.0, 0.9, 0.1], [0.1, 0.0, 0.1], [0.1, 0.1, 0.0]]),
            source=truth.source,
            provenance=self.provenance(),
        )
        assert auroc(scores, truth) == 1.0
        assert auprc(scores, truth) == 1.0
        assert direction_accuracy(scores, truth) == 1.0

    def test_a_spectral_result_must_be_reduced_first(self):
        values = np.random.default_rng(0).random((3, 3, 4, 1))
        scores = Connectivity.from_array(
            values, source=["a", "b", "c"], freqs=[1.0, 2.0, 3.0, 4.0], provenance=self.provenance()
        )
        truth = make_truth(3, ((0, 1),))
        with pytest.raises(ValueError, match="reduce_freq"):
            auroc(scores, truth)
        reduced = scores.reduce_freq((1.0, 4.0))
        assert np.isfinite(auroc(reduced, truth))


class TestAuprc:
    def test_a_perfect_ranking_scores_one(self):
        truth = make_truth()
        assert auprc(truth * 0.9 + 0.05, truth) == 1.0

    def test_it_matches_a_brute_force_average_precision(self):
        rng = np.random.default_rng(1)
        truth = make_truth(6, ((0, 1), (1, 2), (3, 2), (4, 5)))
        for _ in range(5):
            scores = rng.random((6, 6))
            expected = brute_force_average_precision(off_diagonal(scores), off_diagonal(truth) != 0)
            assert auprc(scores, truth) == pytest.approx(expected, rel=1e-12)

    def test_without_edges_it_is_undefined(self):
        with pytest.warns(DirconnWarning, match="undefined"):
            assert np.isnan(auprc(np.random.default_rng(0).random((4, 4)), np.zeros((4, 4))))

    def test_tied_scores_are_one_threshold_not_an_arbitrary_order(self):
        truth = make_truth()
        constant = np.ones((5, 5))
        assert auprc(constant, truth) == pytest.approx(edge_density(truth))
        assert auprc(constant.T, truth.T) == pytest.approx(edge_density(truth))


class TestChance:
    def test_edge_density_counts_off_diagonal_cells(self):
        assert edge_density(make_truth()) == pytest.approx(3 / 20)

    def test_chance_auprc_is_above_the_density_on_small_sparse_networks(self):
        chance = chance_auprc(make_truth(), seed=0)
        assert edge_density(make_truth()) < chance < 0.35

    def test_chance_auprc_is_reproducible(self):
        assert chance_auprc(make_truth(), seed=3) == chance_auprc(make_truth(), seed=3)

    def test_chance_auprc_without_edges_is_undefined(self):
        with pytest.warns(DirconnWarning, match="undefined"):
            assert np.isnan(chance_auprc(np.zeros((4, 4))))


class TestDirectionAccuracy:
    def test_correct_and_reversed_directions(self):
        truth = make_truth()
        right = truth + 0.1
        assert direction_accuracy(right, truth) == 1.0
        assert direction_accuracy(right.T, truth) == 0.0

    def test_only_unidirectional_pairs_are_scored(self):
        truth = make_truth(4, ((0, 1), (1, 0), (2, 3)))  # 0 <-> 1 has no direction to recover
        scores = np.zeros((4, 4))
        scores[2, 3] = 1.0
        assert direction_accuracy(scores, truth) == 1.0

    def test_a_tie_is_a_miss(self):
        truth = make_truth(3, ((0, 1),))
        assert direction_accuracy(np.ones((3, 3)), truth) == 0.0

    def test_pairs_with_non_finite_scores_are_skipped(self):
        truth = make_truth(4, ((0, 1), (2, 3)))
        scores = truth + 0.1
        scores[0, 1] = np.nan
        assert direction_accuracy(scores, truth) == 1.0

    def test_without_a_unidirectional_edge_it_is_undefined(self):
        with pytest.warns(DirconnWarning, match="undefined"):
            assert np.isnan(direction_accuracy(np.ones((3, 3)), np.zeros((3, 3))))


def test_clean_inputs_raise_no_warnings():
    truth = make_truth()
    scores = truth * 0.9 + 0.05
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        auroc(scores, truth)
        auprc(scores, truth)
        direction_accuracy(scores, truth)


class TestConventions:
    """score_result with a MethodInfo scores net, undirected and signed measures by convention."""

    @staticmethod
    def _result(values, freqs=None):
        from dirconn import Connectivity, Provenance

        values = np.array(values, dtype=float)
        return Connectivity.from_array(
            values,
            source=[f"n{i}" for i in range(values.shape[0])],
            freqs=freqs,
            provenance=Provenance(method="stub", parameters={}),
        )

    @staticmethod
    def _info(**kwargs):
        from dirconn.estimator import MethodInfo

        return MethodInfo(name="stub", backend="test", **kwargs)

    def test_a_net_flow_keeps_its_sign_and_leaves_a_pair_coupled_both_ways_out(self):
        from dirconn.evaluation import score_result

        truth = np.array([[np.nan, 1, 0], [1, np.nan, 1], [0, 0, np.nan]])  # 0 <-> 1, 1 -> 2
        values = np.array([[np.nan, 0.5, 0.1], [-0.5, np.nan, 0.4], [-0.1, -0.4, np.nan]])
        net = score_result(
            self._result(values), truth, info=self._info(directed=True, antisymmetric=True)
        )
        plain = score_result(self._result(values), truth)
        assert net["auroc"] == 1.0  # 1 -> 2 above 0 -> 2 and the negative cells; 0 <-> 1 left out
        assert net["direction_accuracy"] == 1.0
        assert plain["direction_accuracy"] < 1.0  # magnitudes cannot tell 1 -> 2 from 2 -> 1

    def test_an_undirected_measure_is_scored_on_its_magnitude_before_the_band_mean(self):
        from dirconn.evaluation import convention_scores, score_result

        freqs = np.array([5.0, 10.0])
        values = np.zeros((2, 2, 2))
        values[0, 1] = [0.4, -0.4]  # a phase that changes sign: the plain mean is 0
        values[1, 0] = -values[0, 1]
        values[[0, 1], [0, 1]] = np.nan
        result = self._result(values, freqs)
        info = self._info(directed=False, antisymmetric=True)
        np.testing.assert_allclose(convention_scores(result, info, (4.0, 11.0))[0, 1], 0.4)
        np.testing.assert_allclose(convention_scores(result, None, (4.0, 11.0))[0, 1], 0.0)
        card = score_result(
            result, np.array([[np.nan, 1], [0, np.nan]]), band=(4.0, 11.0), info=info
        )
        assert np.isnan(card["direction_accuracy"])

    def test_a_signed_measure_is_scored_on_its_magnitude(self):
        from dirconn.evaluation import score_result

        truth = np.array([[np.nan, 1, 0], [0, np.nan, 0], [0, 0, np.nan]])
        values = np.array([[np.nan, -0.6, 0.1], [0.05, np.nan, 0.0], [0.1, 0.0, np.nan]])
        card = score_result(
            self._result(values), truth, info=self._info(directed=True, signed=True)
        )
        assert card["auroc"] == 1.0 and card["separation"] == pytest.approx(6.0)

    def test_an_ordinary_measure_is_scored_as_before(self):
        from dirconn.evaluation import score_result

        truth = np.array([[np.nan, 1], [0, np.nan]])
        values = np.array([[np.nan, 0.3], [0.1, np.nan]])
        assert score_result(self._result(values), truth, info=self._info(directed=True)) == (
            score_result(self._result(values), truth)
        )
