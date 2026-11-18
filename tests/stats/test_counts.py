import math

import pytest

from dirconn.stats.counts import mcnemar_exact, wilson_interval


class TestWilsonInterval:
    def test_zero_of_ten(self):
        low, high = wilson_interval(0, 10)
        assert low == 0.0
        assert high == pytest.approx(0.2775, abs=5e-4)  # 27.8%

    def test_five_of_ten_is_symmetric_about_one_half(self):
        low, high = wilson_interval(5, 10)
        assert low == pytest.approx(0.2366, abs=5e-4)
        assert high == pytest.approx(0.7634, abs=5e-4)

    def test_all_of_ten_mirrors_none_of_ten(self):
        low, high = wilson_interval(10, 10)
        assert high == 1.0
        assert low == pytest.approx(1 - wilson_interval(0, 10)[1])

    def test_equals_statsmodels(self):
        proportion = pytest.importorskip("statsmodels.stats.proportion")
        for k, n in ((3, 45), (44, 45), (17, 50), (1, 1)):
            expected = proportion.proportion_confint(k, n, alpha=0.05, method="wilson")
            assert wilson_interval(k, n) == pytest.approx(expected, abs=1e-12)

    def test_wider_for_higher_confidence(self):
        narrow, wide = wilson_interval(4, 20, 0.8), wilson_interval(4, 20, 0.99)
        assert wide[0] < narrow[0] and wide[1] > narrow[1]

    def test_no_trials_gives_nan(self):
        assert all(math.isnan(v) for v in wilson_interval(0, 0))

    @pytest.mark.parametrize("k, n, confidence", [(-1, 5, 0.95), (6, 5, 0.95), (1, 5, 1.0)])
    def test_bad_arguments_are_refused(self, k, n, confidence):
        with pytest.raises(ValueError):
            wilson_interval(k, n, confidence)


class TestMcNemarExact:
    def test_eleven_against_none(self):
        assert mcnemar_exact(11, 0) == pytest.approx(2 * 0.5**11)
        assert mcnemar_exact(0, 11) == mcnemar_exact(11, 0)

    def test_no_discordant_pairs_gives_one(self):
        assert mcnemar_exact(0, 0) == 1.0

    def test_balanced_counts_are_capped_at_one(self):
        assert mcnemar_exact(3, 3) == 1.0

    def test_equals_the_binomial_test(self):
        binomtest = pytest.importorskip("scipy.stats").binomtest
        for b, c in ((2, 9), (5, 1), (20, 7)):
            assert mcnemar_exact(b, c) == pytest.approx(binomtest(b, b + c, 0.5).pvalue)

    def test_negative_counts_are_refused(self):
        with pytest.raises(ValueError):
            mcnemar_exact(-1, 2)
