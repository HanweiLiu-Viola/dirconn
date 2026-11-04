"""Tests of the non-linear generator, the KSG estimator and transfer entropy."""

import warnings

import numpy as np
import pytest

from dirconn import GC, DirconnWarning, TimeSeries
from dirconn.simulation import coupled_nonlinear
from dirconn.transfer_entropy import TE, ksg_cmi, te

SFREQ = 100.0


def _fit(estimator, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DirconnWarning)
        return estimator.fit(data)


def _var1(seed, n_times=10000):
    """Bivariate VAR(1): channel 0 drives channel 1 at lag 1."""
    rng = np.random.default_rng(seed)
    x = np.zeros((2, n_times))
    e = rng.standard_normal((2, n_times))
    for t in range(1, n_times):
        x[0, t] = 0.5 * x[0, t - 1] + e[0, t]
        x[1, t] = 0.4 * x[1, t - 1] + 0.4 * x[0, t - 1] + e[1, t]
    return TimeSeries.from_array(x, SFREQ)


# --- generator -----------------------------------------------------------------------------


def test_generator_is_deterministic_and_returns_its_truth():
    a, truth = coupled_nonlinear(500, sfreq=SFREQ, seed=3)
    b, _ = coupled_nonlinear(500, sfreq=SFREQ, seed=3)
    np.testing.assert_array_equal(a.data, b.data)
    assert a.names == ("X", "Y", "Z") and a.segmentation == "continuous"
    expected = np.array([[np.nan, 1, 0], [0, np.nan, 0], [0, 0, np.nan]])
    np.testing.assert_array_equal(truth.values[:, :, 0, 0], expected)
    assert truth.provenance.seed == 3 and truth.provenance.parameters["coupling"] == "quadratic"
    epoched, _ = coupled_nonlinear(200, sfreq=SFREQ, seed=1, n_trials=4)
    assert epoched.segmentation == "epoched" and epoched.data.shape == (4, 3, 200)


def test_generator_stays_bounded_and_the_quadratic_coupling_is_uncorrelated():
    data, _ = coupled_nonlinear(20000, sfreq=SFREQ, seed=0, strength=1.0)
    x = np.asarray(data.data)[0]
    assert np.all(np.isfinite(x)) and np.abs(x).max() < 30
    # the coupling g(X) has zero linear correlation with X: lagged correlation X(t-2), Y(t) ~ 0
    assert abs(np.corrcoef(x[0, :-2], x[1, 2:])[0, 1]) < 0.03


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"coupling": "cubic"}, "coupling"),
        ({"delay": 0}, "delay"),
        ({"n_trials": 0}, "n_trials"),
        ({"strength": float("nan")}, "strength"),
    ],
)
def test_generator_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        coupled_nonlinear(100, sfreq=SFREQ, seed=0, **kwargs)


# --- KSG -----------------------------------------------------------------------------------


@pytest.mark.parametrize("rho", [0.3, 0.6])
def test_ksg_matches_the_gaussian_closed_forms(rho):
    # Measured over 8 seeds (5000 samples, k = 4): rho 0.3 -> MI 0.032 to 0.064 and CMI 0.032 to
    # 0.073 (true 0.047); rho 0.6 -> 0.209 to 0.249 and 0.215 to 0.260 (true 0.223).
    true = -0.5 * np.log(1 - rho**2)
    rng = np.random.default_rng(0)
    z, x = rng.standard_normal((2, 5000))
    y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(5000)
    assert ksg_cmi(x, y) == pytest.approx(true, abs=0.05)
    assert ksg_cmi(x + z, y + z, z) == pytest.approx(true, abs=0.05)  # given z, correlation rho


def test_ksg_agrees_with_tigramite_cmiknn():
    cmiknn = pytest.importorskip("tigramite.independence_tests.cmiknn")
    rng = np.random.default_rng(1)
    z, x, noise = rng.standard_normal((3, 3000))
    x = x + 0.5 * z
    y = 0.4 * x + noise + 0.5 * z
    reference = cmiknn.CMIknn(knn=4, transform="standardize", workers=1).get_dependence_measure(
        np.vstack([x, y, z]), np.array([0, 1, 2])
    )
    # tigramite adds noise of 1e-6 standard deviations to break ties; measured difference 9e-7
    assert ksg_cmi(x, y, z, k=4) == pytest.approx(reference, abs=1e-4)


def test_ksg_is_invariant_to_affine_rescaling_and_needs_variance():
    rng = np.random.default_rng(2)
    x, y = rng.standard_normal((2, 2000))
    y = y + 0.5 * x
    np.testing.assert_allclose(ksg_cmi(3 * x + 7, -2 * y), ksg_cmi(x, y), rtol=1e-12)
    with pytest.raises(ValueError, match="constant"):
        ksg_cmi(np.ones(100), y[:100])
    with pytest.raises(ValueError, match="positive integer"):
        ksg_cmi(x, y, k=0)
    with pytest.raises(ValueError, match="samples"):
        ksg_cmi(x[:5], y[:5], k=4)


# --- TE ------------------------------------------------------------------------------------


def test_te_is_half_the_granger_causality_for_linear_gaussian_data():
    # Measured over 8 seeds (10000 samples): TE - GC/2 between -0.008 and 0.010
    for seed in range(2):
        data = _var1(seed)
        transfer = _fit(TE(mode="pairwise", target_history=1, max_delay=1), data).values[:, :, 0, 0]
        granger = _fit(GC(order=1, check_order_stability=False), data).values[:, :, 0, 0]
        assert transfer[0, 1] == pytest.approx(granger[0, 1] / 2, abs=0.02)
        assert abs(transfer[1, 0]) < 0.02


def test_te_sees_the_quadratic_coupling_that_granger_causality_misses():
    # Measured over 5 seeds (5000 samples): GC X -> Y 0.0002 to 0.0004; TE X -> Y 0.089 to
    # 0.105 with every other pair at most 0.023, the delay found as 2.
    data, _ = coupled_nonlinear(5000, sfreq=SFREQ, seed=0)
    granger = _fit(GC(order="bic", check_order_stability=False), data).values[:, :, 0, 0]
    result = _fit(TE(mode="pairwise"), data)
    transfer = result.values[:, :, 0, 0]
    assert granger[0, 1] < 0.005
    assert transfer[0, 1] > 0.06
    others = transfer[~np.eye(3, dtype=bool)]
    assert np.sort(others)[-2] < 0.04  # everything but X -> Y
    assert result.provenance.details["delay"][0][1] == 2


def test_te_scan_provenance_rows_and_conditional_mode():
    data, _ = coupled_nonlinear(2000, sfreq=SFREQ, seed=1, delay=3)
    result = _fit(TE(mode="pairwise", target_history=1, max_delay=4), data)
    details = result.provenance.details
    scan = np.array(details["scan"], dtype=float)
    assert scan.shape == (3, 3, 4)
    off = ~np.eye(3, dtype=bool)
    np.testing.assert_allclose(result.values[:, :, 0, 0][off], scan[off].max(axis=-1))
    assert details["delay"][0][1] == 3
    assert details["n_rows"] == 2000 - 4  # every delay uses the rows after the longest history
    assert details["condition_dims"] == 1
    conditional = _fit(TE(mode="conditional", target_history=1, max_delay=4), data)
    assert conditional.provenance.details["condition_dims"] == 2
    assert not np.allclose(conditional.values, result.values, equal_nan=True)
    assert np.isnan(np.diag(result.values[:, :, 0, 0])).all()


def test_te_pools_trials_without_crossing_their_boundaries():
    data, _ = coupled_nonlinear(400, sfreq=SFREQ, seed=2, n_trials=5)
    result = _fit(TE(mode="pairwise", max_delay=2), data)
    assert result.provenance.details["n_rows"] == 5 * (400 - 2)
    # the same trials glued into one record give more rows and a different estimate
    glued = TimeSeries.from_array(np.concatenate(list(np.asarray(data.data)), axis=-1), SFREQ)
    assert _fit(TE(mode="pairwise", max_delay=2), glued).provenance.details["n_rows"] == 2000 - 2


def test_te_shorthand_parameters_and_short_data():
    data, _ = coupled_nonlinear(600, sfreq=SFREQ, seed=4)
    np.testing.assert_array_equal(
        te(data, mode="pairwise", max_delay=2).values,
        _fit(TE(mode="pairwise", max_delay=2), data).values,
    )
    for name in ("source_history", "max_delay", "n_neighbors"):
        with pytest.raises(ValueError, match=name):
            TE(**{name: 0})
    with pytest.raises(ValueError):
        TE(mode="pairwise", target_history=0)
    with pytest.raises(ValueError):
        TE(mode="pairwise", target_history="aicc")
    with pytest.raises(ValueError, match="mode"):
        TE(mode="yes")
    with pytest.raises(TypeError):
        TE(k=2)
    with pytest.raises(ValueError, match="too few rows"):
        _fit(
            TE(mode="pairwise", target_history=1, max_delay=5),
            TimeSeries.from_array(np.asarray(data.data)[0, :, :8], SFREQ),
        )
    assert TE.info.name == "te" and TE.info.directed


def test_te_target_history_from_the_var_order():
    data, _ = coupled_nonlinear(2000, sfreq=SFREQ, seed=5)
    result = _fit(TE(mode="pairwise", max_delay=2), data)
    details = result.provenance.details
    assert details["order_selection"]["criterion"] == "bic"
    assert details["target_history"] == result.provenance.model_order >= 1
    fixed = _fit(TE(mode="pairwise", target_history=3, max_delay=2), data)
    assert fixed.provenance.details["target_history"] == 3
    assert fixed.provenance.details["order_selection"] is None


def test_parallel_neighbour_searches_give_the_same_values():
    data, _ = coupled_nonlinear(1500, sfreq=SFREQ, seed=6)
    serial = _fit(TE(mode="pairwise", max_delay=2), data).values
    parallel = _fit(TE(mode="pairwise", max_delay=2, workers=-1), data).values
    np.testing.assert_array_equal(parallel, serial)
    rng = np.random.default_rng(0)
    x, y, z = rng.standard_normal((3, 500))
    assert ksg_cmi(x, y, z, workers=2) == ksg_cmi(x, y, z)
    for bad in (0, -2, 1.5, True):
        with pytest.raises(ValueError, match="workers"):
            TE(workers=bad)
        with pytest.raises(ValueError, match="workers"):
            ksg_cmi(x, y, workers=bad)
