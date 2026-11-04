import numpy as np
import pytest

from dirconn import surrogate_test
from dirconn.correction import adjust_pvalues
from dirconn.surrogates import PhaseRandomize
from helpers import AbsCorr, coupled

multipletests = pytest.importorskip("statsmodels.stats.multitest").multipletests


def with_holes(seed=0, shape=(4, 4, 3, 1), n_nan=5):
    rng = np.random.default_rng(seed)
    p = rng.random(shape) ** 2  # skewed towards small values so some survive correction
    p.flat[rng.choice(p.size, n_nan, replace=False)] = np.nan
    return p


@pytest.mark.parametrize("method, reference", [("fdr_bh", "fdr_bh"), ("bonferroni", "bonferroni")])
@pytest.mark.parametrize("seed", range(5))
def test_adjusted_values_equal_statsmodels_on_the_defined_cells(method, reference, seed):
    p = with_holes(seed)
    ok = ~np.isnan(p)
    expected = multipletests(p[ok], method=reference)[1]
    adjusted = adjust_pvalues(p, method)
    np.testing.assert_allclose(adjusted[ok], expected, rtol=1e-12)
    assert np.isnan(adjusted[~ok]).all()


def test_the_family_is_the_defined_cells_only():
    # The old package counted every cell, including the diagonal: 9 here instead of 6.
    p = np.full((3, 3, 1, 1), np.nan)
    off = ~np.eye(3, dtype=bool)
    p[off, 0, 0] = 0.01
    np.testing.assert_allclose(adjust_pvalues(p, "bonferroni")[off, 0, 0], 0.06)
    np.testing.assert_allclose(adjust_pvalues(p, "fdr_bh")[off, 0, 0], 0.01)


def test_the_textbook_example():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adjust_pvalues(p, "fdr_bh"), [0.02, 0.04, 0.04, 0.02])
    np.testing.assert_allclose(adjust_pvalues(p, "bonferroni"), [0.04, 0.16, 0.12, 0.02])


def test_adjusted_values_never_exceed_one_and_never_fall_below_the_raw_ones():
    p = with_holes(3)
    ok = ~np.isnan(p)
    for method in ("fdr_bh", "bonferroni"):
        adjusted = adjust_pvalues(p, method)
        assert (adjusted[ok] <= 1.0).all()
        assert (adjusted[ok] >= p[ok]).all()


def test_none_leaves_the_values_alone():
    p = with_holes(1)
    np.testing.assert_array_equal(adjust_pvalues(p, "none"), p)


def test_the_input_is_not_changed_and_the_shape_is_kept():
    p = with_holes(2)
    before = p.copy()
    out = adjust_pvalues(p, "fdr_bh")
    np.testing.assert_array_equal(p, before)
    assert out.shape == p.shape


def test_with_no_defined_cell_nothing_is_adjusted_and_nothing_fails():
    p = np.full((2, 2, 1, 1), np.nan)
    assert np.isnan(adjust_pvalues(p, "fdr_bh")).all()


@pytest.mark.parametrize("bad", [-0.1, 1.5, np.inf])
def test_values_outside_zero_one_are_rejected(bad):
    with pytest.raises(ValueError, match="between 0 and 1"):
        adjust_pvalues(np.array([0.1, bad]), "fdr_bh")


def test_an_unknown_method_names_the_choices():
    with pytest.raises(ValueError, match="fdr_bh"):
        adjust_pvalues(np.array([0.1]), "holm-ish")


@pytest.fixture(scope="module")
def tested():
    return surrogate_test(
        AbsCorr(), coupled(), PhaseRandomize(), n_surrogates=99, seed=0, alpha=0.9
    )


class TestOnATestResult:
    def test_significance_follows_the_adjusted_values(self, tested):
        for method in ("fdr_bh", "bonferroni", "none"):
            adjusted = tested.adjusted(method)
            expected = np.where(np.isnan(adjusted), False, adjusted <= 0.05)
            np.testing.assert_array_equal(tested.significant(alpha=0.05, method=method), expected)

    def test_undefined_cells_are_never_significant(self, tested):
        assert not tested.significant(method="none")[~tested.defined].any()

    def test_fdr_flags_no_more_than_the_raw_values_and_no_fewer_than_bonferroni(self, tested):
        fdr = tested.significant(method="fdr_bh")
        assert fdr.sum() <= tested.significant(method="none").sum()
        assert fdr.sum() >= tested.significant(method="bonferroni").sum()

    def test_the_driven_cell_is_found_and_independent_ones_are_not(self, tested):
        significant = tested.significant(alpha=0.05)
        assert significant[0, 1, 0, 0] and significant[1, 0, 0, 0]
        assert not significant[0, 2, 0, 0] and not significant[2, 1, 0, 0]

    def test_alpha_defaults_to_the_one_of_the_test_and_can_be_overridden(self, tested):
        np.testing.assert_array_equal(tested.significant(), tested.significant(alpha=tested.alpha))
        assert tested.significant(alpha=0.9).sum() >= tested.significant(alpha=0.01).sum()

    def test_the_default_method_is_fdr(self, tested):
        np.testing.assert_array_equal(tested.significant(), tested.significant(method="fdr_bh"))

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -1.0])
    def test_alpha_must_lie_strictly_between_zero_and_one(self, tested, alpha):
        with pytest.raises(ValueError, match="alpha"):
            tested.significant(alpha=alpha)
