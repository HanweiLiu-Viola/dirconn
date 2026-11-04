import json
import warnings

import numpy as np
import pytest

from dirconn import DirconnWarning, TimeSeries
from dirconn.estimator import STABILITY_RATIO_WARN, order_boundary_check, stability_check
from dirconn.mvar import check_order_spec, choose_order, select_order


def make_series(n_times=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((2, n_times + 100))
    noise = rng.standard_normal(x.shape)
    for t in range(2, x.shape[1]):
        x[0, t] = noise[0, t] + 0.5 * x[0, t - 1] - 0.2 * x[0, t - 2]
        x[1, t] = noise[1, t] + 0.4 * x[0, t - 1] + 0.3 * x[1, t - 1]
    return TimeSeries.from_array(x[:, 100:], 100.0)


class TestChooseOrder:
    def test_a_fixed_order_is_returned_without_a_selection(self):
        order, selection = choose_order(make_series(), 3, 10, ridge=0.0, demean=True, n_folds=5)
        assert order == 3 and selection is None

    def test_a_criterion_runs_select_order(self):
        ts = make_series()
        order, selection = choose_order(ts, "bic", 6, ridge=0.0, demean=True, n_folds=5)
        expected = select_order(ts, 6, criterion="bic")
        assert order == expected.order == selection.order
        np.testing.assert_array_equal(selection.bic, expected.bic)

    def test_cross_validation_gets_the_number_of_folds(self):
        _, selection = choose_order(make_series(), "cv", 4, ridge=0.0, demean=True, n_folds=3)
        assert selection.cv_scheme == "blocks" and selection.n_folds == 3


class TestSelectionToDict:
    def test_it_is_json_safe_and_complete(self):
        selection = select_order(make_series(), 4, criterion="aic")
        data = selection.to_dict()
        assert json.loads(json.dumps(data)) == data
        assert data["criterion"] == "aic"
        assert data["orders"] == [1, 2, 3, 4]
        assert len(data["aic"]) == len(data["bic"]) == len(data["hqc"]) == 4
        assert data["cv"] is None and data["cv_scheme"] is None and data["n_folds"] is None
        assert data["n_obs"] == selection.n_obs

    def test_cross_validation_curves_are_included_when_computed(self):
        data = select_order(make_series(), 3, criterion="cv").to_dict()
        assert len(data["cv"]) == 3 and data["cv_scheme"] == "blocks"


class TestCheckOrderSpec:
    def test_valid_specifications_pass(self):
        check_order_spec("bic", 10, 0.0, 5)
        check_order_spec(3, 10, 0.5, 2)
        check_order_spec("cv", 1, 0.0, 2)

    @pytest.mark.parametrize(
        "args, match",
        [
            ((0, 10, 0.0, 5), "order"),
            (("fpe", 10, 0.0, 5), "order"),
            ((2.5, 10, 0.0, 5), "order"),
            (("bic", 0, 0.0, 5), "max_order"),
            (("bic", 10, -1.0, 5), "ridge"),
            (("bic", 10, 0.0, 1), "n_folds"),
        ],
    )
    def test_invalid_specifications_raise(self, args, match):
        with pytest.raises(ValueError, match=match):
            check_order_spec(*args)


class TestStabilityCheck:
    def test_a_stable_model_passes_silently_and_reports_the_radius(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check = stability_check(np.array([[[0.5]]]))
        assert check.name == "stationarity"
        assert check.status == "pass"
        assert check.value == pytest.approx(0.5)

    def test_the_limit_is_a_radius_of_0_99(self):
        assert STABILITY_RATIO_WARN == 0.99

    def test_a_near_unit_root_warns(self):
        with pytest.warns(DirconnWarning, match="spectral radius"):
            check = stability_check(np.array([[[0.995]]]))
        assert check.status == "warn"
        assert check.value == pytest.approx(0.995)

    def test_an_unstable_model_fails_and_warns(self):
        with pytest.warns(DirconnWarning, match="unstable"):
            check = stability_check(np.array([[[1.05]]]))
        assert check.status == "fail"
        assert check.value == pytest.approx(1.05)


class TestOrderBoundaryCheck:
    def test_an_interior_choice_passes_silently(self):
        selection = select_order(make_series(), 6, criterion="bic")
        assert selection.order < 6
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check = order_boundary_check(selection)
        assert check.name == "order_selection"
        assert check.status == "pass"
        assert check.value == selection.order

    def test_a_choice_at_the_largest_candidate_warns_and_is_recorded(self):
        selection = select_order(make_series(), 2, criterion="aic")
        assert selection.order == selection.max_order == 2
        with pytest.warns(DirconnWarning, match="max_order"):
            check = order_boundary_check(selection)
        assert check.status == "warn"
        assert check.value == 2

    def test_a_single_candidate_is_no_choice_and_does_not_warn(self):
        selection = select_order(make_series(), 1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert order_boundary_check(selection).status == "pass"
