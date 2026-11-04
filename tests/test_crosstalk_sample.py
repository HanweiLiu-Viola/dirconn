"""Tests of the leakage between regions that need the MNE sample dataset (``-m sample_data``)."""

import numpy as np
import pytest
from scipy.stats import spearmanr

from dirconn.headmodel import load_head_model
from dirconn.inverse import crosstalk

pytestmark = pytest.mark.sample_data


def centre_distances(head):
    positions = np.concatenate([s["rr"][s["vertno"]] for s in head.forward["src"]])
    centres = positions[head.roi_centre]
    return np.linalg.norm(centres[:, None] - centres[None], axis=2)


@pytest.fixture(scope="module")
def grad_head():
    return load_head_model("grad")


@pytest.fixture(scope="module")
def grad_mne(grad_head):
    return crosstalk(grad_head, method="mne", seed=0)


class TestCrosstalk:
    def test_the_diagonal_is_one_and_the_shape_is_regions_by_regions(self, grad_head, grad_mne):
        assert grad_mne.matrix.shape == (grad_head.n_rois, grad_head.n_rois)
        assert grad_mne.rois == tuple(grad_head.roi_names)
        np.testing.assert_allclose(np.diag(grad_mne.matrix), 1.0)
        assert np.isfinite(grad_mne.matrix).all()

    def test_neighbouring_regions_leak_into_each_other_far_ones_do_not(self, grad_head, grad_mne):
        # Measured (minimum norm and LCMV, gradiometers and EEG): the nearest tenth of the region
        # pairs leaks 0.38 to 0.69 of the own response, the farthest tenth 0.02 to 0.08, and the
        # Spearman correlation between distance and |leakage| is -0.34 to -0.47.
        distance = centre_distances(grad_head)
        off = ~np.eye(grad_head.n_rois, dtype=bool)
        leakage = np.abs(grad_mne.matrix)
        rho = spearmanr(distance[off], leakage[off]).statistic
        assert rho < -0.25
        near = np.median(leakage[off & (distance < np.percentile(distance[off], 10))])
        far = np.median(leakage[off & (distance > np.percentile(distance[off], 90))])
        assert near > 5 * far

    def test_a_subset_of_regions_has_leakage_from_the_ones_left_out(self, grad_head):
        subset = ("precuneus-rh", "superiorparietal-rh", "lateraloccipital-lh")
        ct = crosstalk(grad_head, subset, method="mne", seed=0)
        assert ct.rois == subset and ct.matrix.shape == (3, 3)
        assert (ct.outside > 0).all()

    def test_it_is_reproducible_and_the_seed_is_required(self, grad_head):
        subset = ("precuneus-rh", "lateraloccipital-lh")
        a = crosstalk(grad_head, subset, seed=3)
        b = crosstalk(grad_head, subset, seed=3)
        np.testing.assert_array_equal(a.matrix, b.matrix)
        with pytest.raises(TypeError):
            crosstalk(grad_head, subset)  # type: ignore[call-arg]

    def test_unknown_regions_are_named(self, grad_head):
        with pytest.raises(ValueError, match="notaregion-lh"):
            crosstalk(grad_head, ("notaregion-lh",), seed=0)
