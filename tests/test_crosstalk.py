import numpy as np
import pytest

from dirconn.inverse import crosstalk_from_series


def probe(n_sources=5, n_times=20000, seed=0):
    return np.random.default_rng(seed).standard_normal((n_sources, n_times))


class TestCrosstalkFromSeries:
    def test_it_recovers_a_known_leakage_matrix_row_normalised(self):
        # extracted series 0 and 1 are mixtures of the sources 0 to 4
        sources = probe()
        mixing = np.array([[2.0, 0.5, 0.0, -0.2, 0.1], [0.3, -4.0, 1.0, 0.0, 0.0]])
        extracted = mixing @ sources
        out = crosstalk_from_series(extracted, sources, own=[0, 1])
        expected = np.array([[1.0, 0.5 / 2.0], [0.3 / -4.0, 1.0]])
        np.testing.assert_allclose(out, expected, atol=0.02)

    def test_the_diagonal_is_one_and_the_scale_of_a_row_does_not_matter(self):
        sources = probe()
        extracted = np.array([[1.0, 0.4, 0.0, 0.0, 0.0], [0.0, 1.0, 0.2, 0.0, 0.0]]) @ sources
        a = crosstalk_from_series(extracted, sources, own=[0, 1])
        b = crosstalk_from_series(extracted * np.array([[7.0], [0.01]]), sources, own=[0, 1])
        np.testing.assert_allclose(np.diag(a), 1.0)
        np.testing.assert_allclose(a, b, atol=1e-10)

    def test_a_negative_own_coefficient_keeps_its_sign_convention(self):
        # extraction can flip the sign of a series; the leakage is relative to the own response, so
        # flipping a whole series leaves the row unchanged
        sources = probe()
        extracted = np.array([[1.0, 0.4, 0.0, 0.0, 0.0], [0.0, 1.0, 0.2, 0.0, 0.0]]) @ sources
        a = crosstalk_from_series(extracted, sources, own=[0, 1])
        b = crosstalk_from_series(-extracted, sources, own=[0, 1])
        np.testing.assert_allclose(a, b, atol=1e-10)

    def test_only_the_requested_columns_are_returned(self):
        sources = probe(n_sources=6)
        extracted = np.eye(6)[[1, 4]] @ sources + 0.1 * np.eye(6)[[0, 0]] @ sources
        out = crosstalk_from_series(extracted, sources, own=[1, 4])
        assert out.shape == (2, 2)

    def test_leakage_from_other_active_sources_is_returned_separately(self):
        sources = probe()
        extracted = np.array([[1.0, 0.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.0, 0.25]]) @ sources
        matrix, outside = crosstalk_from_series(extracted, sources, own=[0, 1], return_outside=True)
        np.testing.assert_allclose(matrix, np.eye(2), atol=0.02)
        # sources 2, 3 and 4 are not extracted: their leakage is the rms of the row's coefficients
        np.testing.assert_allclose(outside, [0.5 / np.sqrt(3), 0.25 / np.sqrt(3)], atol=0.02)

    def test_a_series_that_does_not_respond_to_its_own_source_is_an_error(self):
        sources = probe()
        extracted = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]]) @ sources
        with pytest.raises(ValueError, match="own"):
            crosstalk_from_series(extracted, sources, own=[0])

    def test_shapes_are_checked(self):
        sources = probe()
        with pytest.raises(ValueError, match="same number of samples"):
            crosstalk_from_series(sources[:2, :100], sources, own=[0, 1])
        with pytest.raises(ValueError, match="own"):
            crosstalk_from_series(sources[:2], sources, own=[0])
        with pytest.raises(ValueError, match="own"):
            crosstalk_from_series(sources[:2], sources, own=[0, 9])
