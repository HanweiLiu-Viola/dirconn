import numpy as np
import pytest

from dirconn.inverse import INVERSE_METHODS, ROI_MODES, extract_roi


def sources(n_trials=1, n_idx=7, n_times=300, seed=0):
    return np.random.default_rng(seed).standard_normal((n_trials, n_idx, n_times))


def reference_pca_flip(data, flip):
    """MNE's definition for one trial, written out: first right singular vector, signed by flips."""
    u, s, vt = np.linalg.svd(data, full_matrices=False)
    sign = np.sign(u[:, 0] @ flip)
    scale = np.linalg.norm(s) / np.sqrt(data.shape[0])
    return sign * scale * vt[0]


class TestExtractRoi:
    def test_the_modes_and_methods_are_the_documented_ones(self):
        assert ROI_MODES == ("pca_flip", "mean_flip")
        assert INVERSE_METHODS == ("lcmv", "mne", "dspm", "eloreta")

    def test_mean_flip_averages_the_sources_after_flipping_their_signs(self):
        data = sources()
        flip = np.array([1, -1, 1, 1, -1, -1, 1])
        out = extract_roi(data, flip, "mean_flip")
        np.testing.assert_allclose(out[0], (flip[:, None] * data[0]).mean(axis=0))

    def test_pca_flip_of_one_trial_is_mnes_definition(self):
        data = sources(seed=1)
        flip = np.array([1, 1, -1, 1, -1, 1, 1])
        out = extract_roi(data, flip, "pca_flip")
        np.testing.assert_allclose(out[0], reference_pca_flip(data[0], flip), rtol=1e-10)

    def test_the_sign_follows_the_flips_not_the_arbitrary_sign_of_the_svd(self):
        # sources that all carry the same time course with the signs given by the flips
        t = np.sin(np.linspace(0, 20, 400))
        flip = np.array([1, -1, 1, -1, 1])
        data = (flip[:, None] * t[None, :])[None] * np.array([1.0, 2.0, 0.5, 1.5, 1.0])[
            None, :, None
        ]
        out = extract_roi(data, flip, "pca_flip")[0]
        assert np.corrcoef(out, t)[0, 1] == pytest.approx(1.0, abs=1e-9)
        reversed_out = extract_roi(data, -flip, "pca_flip")[0]
        assert np.corrcoef(reversed_out, t)[0, 1] == pytest.approx(-1.0, abs=1e-9)

    def test_the_power_is_the_average_power_of_the_sources(self):
        data = sources(seed=2)
        flip = np.ones(7)
        out = extract_roi(data, flip, "pca_flip")[0]
        assert np.mean(out**2) == pytest.approx(np.mean(data[0] ** 2), rel=0.3)

    def test_trials_share_one_pattern_so_their_signs_agree(self):
        # one pattern, scaled by +1 and -1 in two trials: a per-trial SVD would flip the second
        # trial's sign relative to the first; the pooled decomposition keeps them as they are
        rng = np.random.default_rng(3)
        pattern = rng.standard_normal(6)
        t1 = rng.standard_normal(500)
        data = np.stack([np.outer(pattern, t1), -np.outer(pattern, t1)])
        out = extract_roi(data, np.sign(pattern), "pca_flip")
        np.testing.assert_allclose(out[1], -out[0], atol=1e-10)

    def test_a_single_source_is_returned_flipped_for_both_modes(self):
        data = sources(n_idx=1)
        for mode in ROI_MODES:
            out = extract_roi(data, np.array([-1]), mode)
            np.testing.assert_allclose(np.abs(out[0]), np.abs(data[0, 0]), rtol=1e-10)

    def test_it_does_not_change_its_input(self):
        data = sources()
        before = data.copy()
        extract_roi(data, np.ones(7), "pca_flip")
        np.testing.assert_array_equal(data, before)

    def test_the_output_has_one_row_per_trial(self):
        assert extract_roi(sources(n_trials=3), np.ones(7), "mean_flip").shape == (3, 300)

    def test_invalid_input_is_refused_with_the_reason(self):
        with pytest.raises(ValueError, match="mode"):
            extract_roi(sources(), np.ones(7), "max")
        with pytest.raises(ValueError, match="flip"):
            extract_roi(sources(), np.ones(5), "mean_flip")
        with pytest.raises(ValueError, match="flip"):
            extract_roi(sources(), np.array([1, 1, 0, 1, 1, 1, 1]), "mean_flip")
        with pytest.raises(ValueError, match="3 dimensions"):
            extract_roi(np.ones((7, 300)), np.ones(7), "mean_flip")
        with pytest.raises(ValueError, match="no source"):
            extract_roi(np.ones((1, 0, 300)), np.ones(0), "mean_flip")
