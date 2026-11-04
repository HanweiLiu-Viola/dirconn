"""Region selection for the scenarios, on made-up arrays (no head model needed)."""

import numpy as np
import pytest

from dirconn.regions import (
    distance_strata,
    distant_region,
    leakage_pairs,
    pool,
)


def line_distances(n, step=10.0):
    x = np.arange(n) * step
    return np.abs(x[:, None] - x[None])


class TestPool:
    def test_regions_are_taken_by_decreasing_visibility_and_kept_apart(self):
        visibility = np.array([1.2, 1.7, 1.5, 1.4, 0.5, 1.9])
        distances = line_distances(6, step=30.0)
        # 5 is too visible and 4 too hidden; 1 is taken first, 2 and 0 are 30 mm from it or
        # from each other, 3 is 60 mm from 1
        assert pool(visibility, distances, low=1.0, high=1.8, min_distance=60.0) == [1, 3]

    def test_the_distance_limit_is_inclusive(self):
        visibility = np.array([1.5, 1.4])
        assert pool(visibility, line_distances(2, 60.0), min_distance=60.0) == [0, 1]
        assert pool(visibility, line_distances(2, 59.9), min_distance=60.0) == [0]

    def test_nothing_qualifies(self):
        assert pool(np.array([0.2, 0.3]), line_distances(2)) == []


def leaky(n, entries, background=0.1):
    """A leakage matrix with ``matrix[receiver, source]`` set for the given entries."""
    matrix = np.full((n, n), background)
    np.fill_diagonal(matrix, 1.0)
    for (receiver, source), value in entries.items():
        matrix[receiver, source] = value
    return matrix


class TestLeakagePairs:
    def test_the_leak_of_x_into_the_series_of_y_is_entry_y_x(self):
        m = leaky(10, {(3, 1): 5.0})
        (first, *_) = leakage_pairs({"eeg": m, "grad": m}, np.full(10, 1.5))
        assert (first.source, first.receiver) == (1, 3)

    def test_a_pair_must_be_in_the_top_tenth_on_every_modality(self):
        n = 12
        eeg = leaky(n, {(3, 1): 5.0, (4, 2): 5.0})
        grad = leaky(n, {(3, 1): 5.0})  # (4, 2) is ordinary on gradiometers
        pairs = leakage_pairs({"eeg": eeg, "grad": grad}, np.full(n, 1.5))
        assert [(p.source, p.receiver) for p in pairs] == [(1, 3)]

    def test_pairs_are_ranked_by_the_weaker_of_the_two_modalities(self):
        n = 12
        eeg = leaky(n, {(3, 1): 5.0, (4, 2): 9.0})
        grad = leaky(n, {(3, 1): 9.0, (4, 2): 6.0})
        pairs = leakage_pairs({"eeg": eeg, "grad": grad}, np.full(n, 1.5), quantile=0.97)
        # (1 -> 3): ranks (second, first); (2 -> 4): ranks (first, second); both have the same
        # weaker rank, so the order between them is by source index
        assert {(p.source, p.receiver) for p in pairs} == {(1, 3), (2, 4)}

    def test_the_score_is_the_smaller_percentile_rank_and_lies_in_the_unit_interval(self):
        n = 12
        m = leaky(n, {(3, 1): 5.0})
        (first, *_) = leakage_pairs({"a": m, "b": m}, np.full(n, 1.5))
        assert first.score == pytest.approx(1.0)
        assert 0.0 <= first.score <= 1.0

    def test_hidden_regions_are_left_out(self):
        n = 10
        m = leaky(n, {(3, 1): 5.0})
        visibility = np.full(n, 1.5)
        visibility[3] = 0.5
        assert leakage_pairs({"a": m}, visibility) == []

    def test_the_diagonal_never_counts(self):
        n = 10
        m = leaky(n, {(3, 1): 5.0})
        np.fill_diagonal(m, 100.0)
        pairs = leakage_pairs({"a": m}, np.full(n, 1.5))
        assert [(p.source, p.receiver) for p in pairs] == [(1, 3)]

    def test_a_matrix_of_the_wrong_shape_is_refused(self):
        with pytest.raises(ValueError, match="matrix"):
            leakage_pairs({"a": np.ones((3, 4))}, np.ones(3))

    def test_the_sign_of_a_leak_does_not_matter(self):
        n = 10
        m = leaky(n, {(3, 1): -5.0})
        (first, *_) = leakage_pairs({"a": m}, np.full(n, 1.5))
        assert (first.source, first.receiver) == (1, 3)


class TestDistantRegion:
    def setup_method(self):
        self.n = 8
        self.distances = line_distances(self.n, step=40.0)
        self.visibility = np.array([1.5, 1.5, 1.1, 1.7, 1.3, 1.2, 0.5, 1.6])
        self.m = leaky(self.n, {(1, 0): 5.0})

    def test_it_takes_the_most_visible_region_that_is_far_and_quiet(self):
        # 0 and 1 are the pair; 2 (80 mm) and up are far enough; 3 is the most visible
        assert distant_region((0, 1), {"a": self.m}, self.visibility, self.distances) == 3

    def test_a_region_more_visible_than_the_window_is_skipped(self):
        visibility = self.visibility.copy()
        visibility[3] = 2.5
        assert distant_region((0, 1), {"a": self.m}, visibility, self.distances) == 7

    def test_regions_too_close_to_either_member_are_skipped(self):
        # 3 is 120 mm from the pair; make it 40 mm from the receiver
        distances = self.distances.copy()
        distances[3, 1] = distances[1, 3] = 40.0
        assert distant_region((0, 1), {"a": self.m}, self.visibility, distances) == 7

    def test_a_region_that_leaks_into_the_pair_is_skipped(self):
        m = self.m.copy()
        m[0, 3] = 5.0  # region 3 leaks into the series of region 0
        assert distant_region((0, 1), {"a": m}, self.visibility, self.distances) == 7

    def test_hidden_regions_are_skipped(self):
        visibility = self.visibility.copy()
        visibility[[3, 7, 2, 4, 5]] = 0.1
        with pytest.raises(ValueError, match="No region"):
            distant_region((0, 1), {"a": self.m}, visibility, self.distances)


class TestDistanceStrata:
    def make(self, n=12):
        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 100, size=(n, 3))
        distances = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        names = tuple(f"r{i}-{'lh' if i % 2 == 0 else 'rh'}" for i in range(n))
        visibility = np.full(n, 1.2)
        return visibility, distances, names

    def test_three_pairs_of_the_same_hemisphere_in_increasing_distance(self):
        visibility, distances, names = self.make()
        result = distance_strata(visibility, distances, names)
        assert list(result.pairs) == ["near", "mid", "far"]
        d = [distances[i, j] for i, j in result.pairs.values()]
        assert d[0] < d[1] < d[2]
        for i, j in result.pairs.values():
            assert names[i][-2:] == names[j][-2:]
            assert i != j

    def test_each_pair_lies_in_its_tercile_and_is_the_one_nearest_the_median(self):
        visibility, distances, names = self.make()
        result = distance_strata(visibility, distances, names)
        low, high = result.bounds
        near, mid, far = (distances[i, j] for i, j in result.pairs.values())
        assert near <= low < mid <= high < far
        eligible = [
            distances[i, j]
            for i in range(12)
            for j in range(i + 1, 12)
            if names[i][-2:] == names[j][-2:]
        ]
        in_mid = [d for d in eligible if low < d <= high]
        assert abs(mid - np.median(in_mid)) == min(abs(d - np.median(in_mid)) for d in in_mid)

    def test_regions_outside_the_visibility_window_are_not_used(self):
        visibility, distances, names = self.make()
        visibility[:6] = 0.2
        result = distance_strata(visibility, distances, names)
        used = {k for pair in result.pairs.values() for k in pair}
        assert used and min(used) >= 6

    def test_too_few_regions_are_refused(self):
        visibility, distances, names = self.make(4)
        with pytest.raises(ValueError, match="pairs"):
            distance_strata(visibility, distances, names)

    def test_a_name_without_a_hemisphere_is_refused(self):
        visibility, distances, names = self.make()
        with pytest.raises(ValueError, match="hemisphere"):
            distance_strata(visibility, distances, ("x",) + names[1:])
