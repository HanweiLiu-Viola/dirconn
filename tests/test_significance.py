import numpy as np
import pytest

from dirconn import DTF, GC, TimeSeries, significance_test, surrogate_test
from dirconn.analytic import AnalyticTestResult, analytic_gc_test
from dirconn.surrogate_test import SurrogateTestResult
from dirconn.surrogates import AAFT, BlockShuffle, PhaseRandomize
from helpers import AbsCorr, coupled

pytestmark = pytest.mark.filterwarnings("ignore:n_surrogates=:dirconn.exceptions.DirconnWarning")


def continuous(n_times=600, seed=0):
    x = np.random.default_rng(seed).standard_normal((3, n_times))
    return TimeSeries.from_array(x, 100.0)


class TestDispatch:
    def test_surrogate_equals_the_direct_call(self):
        data = coupled()
        via = significance_test(
            AbsCorr(), data, test="surrogate", surrogate=AAFT(), n_surrogates=19, seed=3
        )
        direct = surrogate_test(AbsCorr(), data, AAFT(), n_surrogates=19, seed=3)
        assert isinstance(via, SurrogateTestResult)
        np.testing.assert_array_equal(via.pvalues, direct.pvalues)

    def test_a_surrogate_can_be_given_by_name(self):
        data = coupled()
        by_name = significance_test(
            AbsCorr(), data, test="surrogate", surrogate="phase_randomize", n_surrogates=9, seed=1
        )
        assert by_name.generator["name"] == "phase_randomize"
        direct = surrogate_test(AbsCorr(), data, PhaseRandomize(), n_surrogates=9, seed=1)
        np.testing.assert_array_equal(by_name.pvalues, direct.pvalues)

    def test_block_perm_equals_block_shuffle_through_surrogate_test(self):
        data = continuous()
        via = significance_test(
            AbsCorr(), data, test="block_perm", block_size=50, n_surrogates=19, seed=2
        )
        direct = surrogate_test(
            AbsCorr(), data, BlockShuffle(block_size=50), n_surrogates=19, seed=2
        )
        assert via.generator["name"] == "block_shuffle"
        np.testing.assert_array_equal(via.pvalues, direct.pvalues)

    def test_analytic_equals_the_direct_call(self):
        data = coupled(n_trials=4, n_times=100)
        via = significance_test(GC(order=2), data, test="analytic")
        direct = analytic_gc_test(GC(order=2), data)
        assert isinstance(via, AnalyticTestResult)
        np.testing.assert_array_equal(via.pvalues, direct.pvalues)

    def test_alpha_and_reduce_are_passed_on(self):
        data = coupled(n_trials=6, n_times=200)
        result = significance_test(
            DTF(order=2),
            data,
            test="surrogate",
            surrogate=PhaseRandomize(),
            n_surrogates=5,
            seed=0,
            reduce=(10.0, 20.0),
            alpha=0.3,
        )
        assert result.pvalues.shape[2] == 1 and result.alpha == 0.3
        assert significance_test(GC(order=2), data, test="analytic", alpha=0.2).alpha == 0.2

    def test_every_result_can_be_corrected_and_states_its_null(self):
        data = coupled(n_trials=4, n_times=100)
        results = [
            significance_test(GC(order=2), data, test="analytic"),
            significance_test(
                AbsCorr(),
                data,
                test="surrogate",
                surrogate="circular_shift",
                n_surrogates=9,
                seed=0,
            ),
            significance_test(
                AbsCorr(), continuous(), test="block_perm", block_size=50, n_surrogates=9, seed=0
            ),
        ]
        for result in results:
            assert result.null.strip() and result.limitations
            assert result.significant(method="none").shape == result.pvalues.shape


class TestArguments:
    def test_the_test_must_be_chosen(self):
        with pytest.raises(TypeError):
            significance_test(AbsCorr(), coupled())  # type: ignore[call-arg]

    def test_an_unknown_test_names_the_choices(self):
        with pytest.raises(ValueError, match="surrogate.*block_perm.*analytic"):
            significance_test(AbsCorr(), coupled(), test="ttest")

    def test_surrogate_needs_a_surrogate_a_size_and_a_seed(self):
        data = coupled()
        with pytest.raises(ValueError, match="surrogate"):
            significance_test(AbsCorr(), data, test="surrogate", n_surrogates=9, seed=0)
        with pytest.raises(ValueError, match="n_surrogates"):
            significance_test(AbsCorr(), data, test="surrogate", surrogate="aaft", seed=0)
        with pytest.raises(ValueError, match="seed"):
            significance_test(AbsCorr(), data, test="surrogate", surrogate="aaft", n_surrogates=9)

    def test_block_perm_needs_a_block_size_a_size_and_a_seed(self):
        data = continuous()
        with pytest.raises(ValueError, match="block_size"):
            significance_test(AbsCorr(), data, test="block_perm", n_surrogates=9, seed=0)
        with pytest.raises(ValueError, match="n_surrogates"):
            significance_test(AbsCorr(), data, test="block_perm", block_size=50, seed=0)
        with pytest.raises(ValueError, match="seed"):
            significance_test(AbsCorr(), data, test="block_perm", block_size=50, n_surrogates=9)

    @pytest.mark.parametrize(
        "test, extra",
        [
            ("surrogate", {"block_size": 50}),
            ("block_perm", {"surrogate": "aaft"}),
            ("analytic", {"n_surrogates": 9}),
            ("analytic", {"seed": 0}),
            ("analytic", {"surrogate": "aaft"}),
            ("analytic", {"block_size": 10}),
            ("analytic", {"reduce": (10.0, 20.0)}),
        ],
    )
    def test_an_argument_a_test_does_not_use_is_refused_not_ignored(self, test, extra):
        base = {
            "surrogate": {"surrogate": "aaft", "n_surrogates": 9, "seed": 0},
            "block_perm": {"block_size": 50, "n_surrogates": 9, "seed": 0},
            "analytic": {},
        }[test]
        estimator = GC(order=2) if test == "analytic" else AbsCorr()
        data = continuous() if test == "block_perm" else coupled(n_trials=4, n_times=100)
        with pytest.raises(ValueError, match=f"{next(iter(extra))}.*{test}"):
            significance_test(estimator, data, test=test, **base, **extra)

    def test_an_unknown_surrogate_name_lists_the_known_ones(self):
        with pytest.raises(ValueError, match="phase_randomize"):
            significance_test(
                AbsCorr(), coupled(), test="surrogate", surrogate="aaaft", n_surrogates=9, seed=0
            )

    def test_block_shuffle_is_not_available_by_name_and_the_message_says_why(self):
        with pytest.raises(ValueError, match="block_perm"):
            significance_test(
                AbsCorr(),
                coupled(),
                test="surrogate",
                surrogate="block_shuffle",
                n_surrogates=9,
                seed=0,
            )

    def test_analytic_is_only_for_gc(self):
        with pytest.raises(TypeError, match="surrogate"):
            significance_test(DTF(order=2), coupled(), test="analytic")
