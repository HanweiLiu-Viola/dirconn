import warnings

import pytest

from dirconn import DirconnWarning
from dirconn.estimator import SAMPLING_RATIO_WARN, sampling_check


def test_the_limit_is_one_coefficient_per_ten_observations():
    assert SAMPLING_RATIO_WARN == 0.1


def test_plenty_of_data_passes_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check = sampling_check(n_coefficients=20, n_obs=2000)
    assert check.name == "sampling_adequacy"
    assert check.status == "pass"
    assert check.value == pytest.approx(0.01)


def test_the_limit_itself_still_passes():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert sampling_check(n_coefficients=10, n_obs=100).status == "pass"


def test_too_many_coefficients_warn_and_are_recorded():
    with pytest.warns(DirconnWarning, match="coefficients"):
        check = sampling_check(n_coefficients=60, n_obs=300)
    assert check.status == "warn"
    assert check.value == pytest.approx(0.2)
    assert "300" in check.detail and "60" in check.detail


@pytest.mark.parametrize("n_coefficients, n_obs", [(0, 100), (10, 0), (-1, 10)])
def test_nonsensical_counts_are_rejected(n_coefficients, n_obs):
    with pytest.raises(ValueError, match="positive"):
        sampling_check(n_coefficients=n_coefficients, n_obs=n_obs)
