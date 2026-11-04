"""Tests of the multivariate (greedy, permutation-tested) transfer entropy and the KSG kernel."""

import warnings

import numpy as np
import pytest

from dirconn import DirconnWarning, TimeSeries, te
from dirconn._greedy import _factorial_at_least, _p_value, min_permutations
from dirconn._ksg import ksg_cmi, ksg_cmi_many
from dirconn.simulation import coupled_nonlinear
from dirconn.transfer_entropy import TE

SFREQ = 100.0
FAST = {"n_perm_max_stat": 19, "n_perm_min_stat": 19, "n_perm_omnibus": 19, "n_perm_max_seq": 19}
"""The fewest permutations that reach alpha = 0.05, to keep the tests short."""


def _fit(estimator, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DirconnWarning)
        return estimator.fit(data)


def _chain(seed, n_times=3000):
    """Linear Gaussian chain X -> Y -> Z at lag 1, every channel with its own memory."""
    rng = np.random.default_rng(seed)
    x = np.zeros((3, n_times))
    e = rng.standard_normal((3, n_times))
    for t in range(1, n_times):
        x[0, t] = 0.5 * x[0, t - 1] + e[0, t]
        x[1, t] = 0.3 * x[1, t - 1] + 0.7 * x[0, t - 1] + e[1, t]
        x[2, t] = 0.3 * x[2, t - 1] + 0.7 * x[1, t - 1] + e[2, t]
    return TimeSeries.from_array(x, SFREQ, names=["X", "Y", "Z"])


def _common_driver(seed, n_times=3000):
    """Z drives X at lag 1 and Y at lag 2; X and Y are not coupled."""
    rng = np.random.default_rng(seed)
    x = np.zeros((3, n_times))
    e = rng.standard_normal((3, n_times))
    for t in range(2, n_times):
        x[2, t] = 0.6 * x[2, t - 1] + e[2, t]
        x[0, t] = 0.8 * x[2, t - 1] + e[0, t]
        x[1, t] = 0.8 * x[2, t - 2] + e[1, t]
    return TimeSeries.from_array(x, SFREQ, names=["X", "Y", "Z"])


# --- kernel ---------------------------------------------------------------------------------


def test_many_equals_one_at_a_time():
    rng = np.random.default_rng(0)
    y, z = rng.standard_normal((2, 800))
    xs = [
        rng.standard_normal(800) + 0.5 * y,
        rng.standard_normal((800, 2)),
        rng.standard_normal(800),
    ]
    many = ksg_cmi_many(xs, y, z, k=4)
    one = [ksg_cmi(x, y, z, k=4) for x in xs]
    np.testing.assert_allclose(many, one, rtol=0, atol=0)
    assert ksg_cmi_many([], y, z).shape == (0,)
    # an empty condition is the mutual information
    assert ksg_cmi_many(xs[:1], y, np.empty((800, 0)))[0] == ksg_cmi(xs[0], y)


def test_device_and_dtype_are_validated():
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal((2, 50))
    with pytest.raises(ValueError, match="dtype"):
        ksg_cmi(x, y, dtype="float16")
    with pytest.raises(ValueError, match="torch device"):
        ksg_cmi(x, y, dtype="float32")
    with pytest.raises(ValueError, match="device"):
        ksg_cmi(x, y, device="")


# --- helpers of the tests --------------------------------------------------------------------


def test_permutation_helpers():
    assert min_permutations(0.05) == 19 and min_permutations(0.01) == 99
    assert _factorial_at_least(5, 120) and not _factorial_at_least(4, 25)
    assert not _factorial_at_least(1, 2)
    # (count + 1) / (n + 1)
    assert _p_value(np.array([0.1, 0.2, 0.3]), 0.25) == pytest.approx(2 / 4)
    assert _p_value(np.zeros(19), 1.0) == pytest.approx(1 / 20)


# --- multivariate TE --------------------------------------------------------------------------


def test_multivariate_removes_the_indirect_link_of_a_chain():
    data = _chain(0)
    result = _fit(TE(max_order=3, max_delay=2, **FAST), data)
    values = result.values[:, :, 0, 0]
    assert values[0, 1] > 0.1 and values[1, 2] > 0.1  # X -> Y, Y -> Z
    off = ~np.eye(3, dtype=bool)
    off[0, 1] = off[1, 2] = False
    np.testing.assert_array_equal(values[off], 0.0)  # in particular X -> Z
    details = result.provenance.details
    assert details["selected_sources"][2] == [[1, 1]]
    assert details["selected_sources"][1] == [[0, 1]]
    assert details["omnibus"][2]["significant"] and details["omnibus"][0]["p"] == 1.0
    # pairwise TE sees the indirect link
    pairwise = _fit(TE(mode="pairwise", target_history=1, max_delay=2), data).values[:, :, 0, 0]
    assert pairwise[0, 2] > 0.05


def test_multivariate_does_not_link_the_two_driven_channels():
    data = _common_driver(1)
    values = _fit(TE(max_order=3, max_delay=3, **FAST), data).values[:, :, 0, 0]
    assert values[2, 0] > 0.1 and values[2, 1] > 0.1
    assert values[0, 1] == 0.0 and values[1, 0] == 0.0


def test_multivariate_sees_the_quadratic_coupling():
    data, _ = coupled_nonlinear(3000, sfreq=SFREQ, seed=0)
    result = _fit(TE(max_order=3, max_delay=3, **FAST), data)
    values = result.values[:, :, 0, 0]
    assert values[0, 1] > 0.05
    assert result.provenance.details["selected_sources"][1] == [[0, 2]]  # delay 2


def test_multivariate_is_reproducible_and_independent_of_workers():
    pytest.importorskip("joblib")
    data = _chain(2, n_times=1500)
    estimator = TE(max_order=2, max_delay=2, **FAST)
    first = _fit(estimator, data)
    again = _fit(estimator, data)
    parallel = _fit(TE(max_order=2, max_delay=2, workers=2, **FAST), data)
    np.testing.assert_array_equal(first.values, again.values)
    np.testing.assert_array_equal(first.values, parallel.values)
    assert first.provenance.details == parallel.provenance.details
    assert first.provenance.seed == 0
    other = _fit(TE(max_order=2, max_delay=2, seed=5, **FAST), data)
    assert other.provenance.seed == 5


def test_multivariate_provenance_and_trials():
    data, _ = coupled_nonlinear(300, sfreq=SFREQ, seed=3, n_trials=6)
    result = _fit(TE(max_order=2, max_delay=3, **FAST), data)
    details = result.provenance.details
    assert details["mode"] == "multivariate"
    assert details["n_rows"] == 6 * (300 - 3)
    assert details["target_lags"] == [1, 2] and details["source_lags"] == [1, 2, 3]
    for lags in details["selected_target_lags"]:
        assert 1 <= len(lags) and set(lags) <= {1, 2}
    p = np.array(details["p_values"], dtype=float)
    assert p.shape == (3, 3) and np.isnan(np.diag(p)).all()
    assert np.isnan(np.diag(result.values[:, :, 0, 0])).all()
    assert result.provenance.parameters["mode"] == "multivariate"


def test_fdr_can_only_remove_links():
    data = _chain(4, n_times=1500)
    plain = _fit(TE(max_order=2, max_delay=2, **FAST), data).values[:, :, 0, 0]
    result = _fit(TE(max_order=2, max_delay=2, fdr=True, **FAST), data)
    corrected = result.values[:, :, 0, 0]
    off = ~np.eye(3, dtype=bool)
    assert np.all((corrected[off] == plain[off]) | (corrected[off] == 0.0))
    assert "p_fdr" in result.provenance.details["omnibus"][1]


def test_independent_channels_give_an_empty_network():
    rng = np.random.default_rng(7)
    data = TimeSeries.from_array(rng.standard_normal((3, 1500)), SFREQ)
    values = _fit(TE(max_order=2, max_delay=2, **FAST), data).values[:, :, 0, 0]
    np.testing.assert_array_equal(values[~np.eye(3, dtype=bool)], 0.0)


def test_parameters_that_do_not_apply_raise():
    with pytest.raises(ValueError, match="target_history"):
        TE(target_history=2)
    with pytest.raises(ValueError, match="source_history"):
        TE(source_history=2)
    with pytest.raises(ValueError, match="alpha, seed do not apply"):
        TE(mode="pairwise", alpha=0.01, seed=3)
    with pytest.raises(ValueError, match="fdr"):
        TE(mode="conditional", fdr=True)
    with pytest.raises(ValueError, match="cannot reach alpha"):
        TE(n_perm_omnibus=10)
    with pytest.raises(ValueError, match="alpha"):
        TE(alpha=1.5)
    with pytest.raises(TypeError, match="fdr"):
        TE(fdr=1)
    with pytest.raises(TypeError, match="seed"):
        TE(seed=1.5)
    with pytest.raises(ValueError, match="dtype"):
        TE(dtype="float32")  # needs a torch device
    assert TE().mode == "multivariate"
    TE(mode="pairwise", seed=0)  # the default value is fine


def test_shorthand_and_short_data():
    data = _chain(5, n_times=600)
    np.testing.assert_array_equal(
        te(data, max_order=2, max_delay=1, **FAST).values,
        _fit(TE(max_order=2, max_delay=1, **FAST), data).values,
    )
    short = TimeSeries.from_array(np.asarray(data.data)[0, :, :12], SFREQ)
    with pytest.raises(ValueError, match="too few rows"):
        _fit(TE(**FAST), short)


# --- torch backend ----------------------------------------------------------------------------


def _torch():
    try:
        import torch
    except ImportError:
        pytest.skip("torch is not installed")
    except OSError as error:  # conda's MKL numpy loaded first and pip's torch cannot load
        pytest.skip(f"torch cannot be loaded in this process: {error}")
    return torch


@pytest.mark.parametrize("dims", [(1, 1, 0), (1, 1, 2), (2, 1, 3)])
def test_torch_float64_matches_the_scipy_path(dims):
    _torch()
    rng = np.random.default_rng(10)
    dx, dy, dz = dims
    n = 700
    z = rng.standard_normal((n, dz)) if dz else None
    x = rng.standard_normal((n, dx))
    y = rng.standard_normal((n, dy)) + 0.4 * x[:, :1]
    if z is not None:
        x = x + 0.5 * z[:, :1]
        y = y + 0.5 * z[:, :1]
    reference = ksg_cmi(x, y, z)
    assert ksg_cmi(x, y, z, device="cpu") == pytest.approx(reference, abs=1e-10)


def test_torch_float32_agrees_statistically():
    _torch()
    rng = np.random.default_rng(11)
    z, x, noise = rng.standard_normal((3, 2000))
    y = 0.6 * x + noise + 0.5 * z
    reference = ksg_cmi(x, y, z)
    assert ksg_cmi(x, y, z, device="cpu", dtype="float32") == pytest.approx(reference, rel=0.01)


def test_torch_blocks_give_the_same_counts(monkeypatch):
    _torch()
    import dirconn._ksg as ksg

    rng = np.random.default_rng(12)
    x, y, z = rng.standard_normal((3, 300))
    whole = ksg.ksg_cmi(x, y, z, device="cpu")
    monkeypatch.setattr(ksg, "_BLOCK_BYTES", 6 * 300 * 8 * 7)  # blocks of 7 rows
    assert ksg.ksg_cmi(x, y, z, device="cpu") == whole


def test_torch_te_matches_the_scipy_path_in_every_mode():
    _torch()
    data = _chain(6, n_times=800)
    for params in (
        {"mode": "pairwise", "target_history": 1, "max_delay": 2},
        {"mode": "conditional", "target_history": 1, "max_delay": 1},
        {"max_order": 2, "max_delay": 1, **FAST},
    ):
        cpu = _fit(TE(**params), data)
        torch = _fit(TE(device="cpu", **params), data)
        np.testing.assert_allclose(torch.values, cpu.values, atol=1e-10, equal_nan=True)
        assert torch.provenance.details["mode"] == cpu.provenance.details["mode"]


@pytest.mark.gpu
def test_cuda_float64_matches_the_scipy_path():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(13)
    z, x, noise = rng.standard_normal((3, 5000))
    y = 0.4 * x + noise + 0.5 * z
    assert ksg_cmi(x, y, z, device="cuda") == pytest.approx(ksg_cmi(x, y, z), abs=1e-10)


def test_missing_torch_names_the_extra(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    rng = np.random.default_rng(14)
    x, y = rng.standard_normal((2, 50))
    with pytest.raises(ImportError, match=r"dirconn\[gpu\]"):
        ksg_cmi(x, y, device="cpu")
