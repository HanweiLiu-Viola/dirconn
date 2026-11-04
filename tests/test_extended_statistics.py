"""The further estimators work unchanged with the significance tests and the bootstrap."""

import warnings

import numpy as np
import pytest

from dirconn import (
    MTGC,
    PCMCI,
    PSI,
    TE,
    TRGC,
    BootstrapResult,
    DirconnWarning,
    ImCoh,
    SurrogateTestResult,
    TimeSeries,
    bootstrap,
    significance_test,
)

SFREQ = 100.0


def _lagged(seed=0, n_times=4000):
    """Channel 0 drives channel 1 at lag 2 (VAR(2)); channel 2 is independent."""
    rng = np.random.default_rng(seed)
    x = np.zeros((3, n_times + 100))
    e = rng.standard_normal(x.shape)
    for t in range(2, x.shape[-1]):
        x[0, t] = 0.5 * x[0, t - 1] - 0.3 * x[0, t - 2] + e[0, t]
        x[1, t] = 0.4 * x[1, t - 1] + 0.4 * x[0, t - 2] + e[1, t]
        x[2, t] = 0.6 * x[2, t - 1] + e[2, t]
    return TimeSeries.from_array(x[:, 100:], SFREQ)


ESTIMATORS = {
    "imcoh": lambda: ImCoh(segment_length=1.0, fmin=2.0, fmax=20.0),
    "psi": lambda: PSI(segment_length=1.0, bands=[(2.0, 20.0)]),
    "trgc": lambda: TRGC(order=2, check_order_stability=False),
    "mtgc": lambda: MTGC(segment_length=1.0, fmin=2.0, fmax=20.0),
    "pcmci": lambda: PCMCI(tau_max=3),
    "te": lambda: TE(mode="pairwise", max_delay=3),
}


def _quiet(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DirconnWarning)
        return func(*args, **kwargs)


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_surrogate_test_finds_the_lagged_pair(name):
    """Phase randomisation keeps each spectrum and destroys the coupling.

    The test is on magnitudes, so for the antisymmetric measures both cells of a pair get the
    same p-value; the sign of ``observed`` gives the direction.
    """
    result = _quiet(
        significance_test,
        ESTIMATORS[name](),
        _lagged(),
        test="surrogate",
        surrogate="phase_randomize",
        n_surrogates=19,
        seed=1,
        reduce=(2.0, 20.0) if name in ("imcoh", "mtgc") else None,
    )
    assert isinstance(result, SurrogateTestResult)
    p = result.pvalues[:, :, 0, 0]
    assert p[0, 1] == pytest.approx(1 / 20)
    assert np.isnan(np.diag(p)).all()
    assert result.observed.provenance.method == name


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_block_permutation_runs_and_keeps_the_shape(name):
    data = _lagged(n_times=3000)
    result = _quiet(
        significance_test,
        ESTIMATORS[name](),
        data,
        test="block_perm",
        block_size=300,
        n_surrogates=9,
        seed=2,
    )
    observed = _quiet(ESTIMATORS[name]().fit, data)
    assert result.pvalues.shape == observed.values.shape
    assert np.isfinite(result.pvalues[0, 1]).all()


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_bootstrap_of_continuous_data_runs_on_the_blocks(name):
    """Bootstrap blocks become epoched trials; the spectral methods cut each block into segments."""
    data = _lagged(n_times=3000)
    result = _quiet(bootstrap, ESTIMATORS[name](), data, n_boot=8, seed=3, block_size=300)
    assert isinstance(result, BootstrapResult)
    observed = result.observed.values[0, 1]
    assert (result.lower[0, 1] <= result.upper[0, 1]).all()
    assert np.isfinite(observed).all()
    assert result.lower.shape == result.observed.values.shape
