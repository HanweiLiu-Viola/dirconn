"""Tests of the PCMCI estimator and its p-values."""

import sys
import warnings

import numpy as np
import pytest

from dirconn import DirconnWarning, TimeSeries

tigramite = pytest.importorskip("tigramite")
from tigramite import data_processing  # noqa: E402
from tigramite.independence_tests.parcorr import ParCorr  # noqa: E402
from tigramite.pcmci import PCMCI as TigramitePCMCI  # noqa: E402

from dirconn.pcmci import LIMITATIONS, NULL, PCMCI, PCMCITestResult, pcmci, pcmci_test  # noqa: E402

SFREQ = 100.0


def _chain(seed=0, n_times=3000):
    """Channel 0 drives 1 at lag 2 (+0.5); channel 1 drives 2 at lag 3 (-0.4); (time, channel)."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n_times, 3))
    e = rng.standard_normal((n_times, 3))
    for t in range(3, n_times):
        x[t, 0] = 0.5 * x[t - 1, 0] + e[t, 0]
        x[t, 1] = 0.4 * x[t - 1, 1] + 0.5 * x[t - 2, 0] + e[t, 1]
        x[t, 2] = 0.3 * x[t - 1, 2] - 0.4 * x[t - 3, 1] + e[t, 2]
    return x


def _fit(estimator, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DirconnWarning)
        return estimator.fit(data)


def _reference(frame, tau_max, pc_alpha=0.2):
    out = TigramitePCMCI(dataframe=frame, cond_ind_test=ParCorr(), verbosity=0).run_pcmci(
        tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha
    )
    return out["val_matrix"], out["p_matrix"]


def test_a_known_lagged_link_lands_at_source_target_with_its_lag():
    result = _fit(PCMCI(tau_max=4), TimeSeries.from_array(_chain().T, SFREQ))
    v = result.values[:, :, 0, 0]
    lag = np.array(result.provenance.details["lag"], dtype=float)
    assert v[0, 1] > 0.4 and lag[0, 1] == 2  # 0 -> 1 at lag 2, row 0 column 1
    assert v[1, 2] < -0.3 and lag[1, 2] == 3  # negative coupling keeps its sign
    assert abs(v[1, 0]) < 0.1 and abs(v[2, 1]) < 0.1  # no link backwards
    assert np.isnan(np.diag(v)).all() and np.isnan(np.diag(lag)).all()


def test_values_and_pvalues_are_read_from_tigramite_at_the_chosen_lag():
    x = _chain(1)
    result = _fit(PCMCI(tau_max=4), TimeSeries.from_array(x.T, SFREQ))
    val, pval = _reference(data_processing.DataFrame(x), 4)
    details = result.provenance.details
    np.testing.assert_allclose(np.array(details["val_matrix"], dtype=float), val[:, :, 1:])
    np.testing.assert_allclose(np.array(details["p_matrix"], dtype=float), pval[:, :, 1:])
    off = ~np.eye(3, dtype=bool)
    best = np.argmin(pval[:, :, 1:], axis=-1)
    rows, cols = np.indices(best.shape)
    np.testing.assert_allclose(result.values[:, :, 0, 0][off], val[:, :, 1:][rows, cols, best][off])
    expected_p = np.minimum(pval[:, :, 1:][rows, cols, best] * 4, 1.0)
    np.testing.assert_allclose(np.array(details["pvalues"], dtype=float)[off], expected_p[off])


def test_trials_are_analysed_with_the_multiple_dataset_mode():
    x = _chain(2)
    trials = x.T.reshape(3, 6, 500).transpose(1, 0, 2)
    result = _fit(PCMCI(tau_max=4), TimeSeries.from_array(trials, SFREQ, segmentation="epoched"))
    frame = data_processing.DataFrame(
        {k: trials[k].T.copy() for k in range(6)}, analysis_mode="multiple"
    )
    val, _ = _reference(frame, 4)
    np.testing.assert_allclose(
        np.array(result.provenance.details["val_matrix"], float), val[:, :, 1:]
    )


def test_tau_max_comes_from_the_var_order_or_is_given():
    data = TimeSeries.from_array(_chain(3).T, SFREQ)
    chosen = _fit(PCMCI(), data)
    assert chosen.provenance.details["tau_max"] == chosen.provenance.model_order
    assert chosen.provenance.details["order_selection"]["criterion"] == "bic"
    assert len(chosen.provenance.details["val_matrix"][0][1]) == chosen.provenance.model_order
    fixed = _fit(PCMCI(tau_max=5), data)
    assert fixed.provenance.model_order == 5 and fixed.provenance.details["order_selection"] is None
    assert fixed.provenance.method == "pcmci"
    np.testing.assert_array_equal(pcmci(data, tau_max=5).values, fixed.values)


def test_pcmci_test_returns_the_corrected_mci_pvalues():
    data = TimeSeries.from_array(_chain(4).T, SFREQ)
    result = pcmci_test(PCMCI(tau_max=4), data, alpha=0.05)
    assert isinstance(result, PCMCITestResult)
    assert result.null == NULL and result.limitations == LIMITATIONS
    assert result.pvalues.shape == (3, 3, 1, 1)
    np.testing.assert_array_equal(result.defined[:, :, 0, 0], ~np.eye(3, dtype=bool))
    significant = result.significant()[:, :, 0, 0]
    assert significant[0, 1] and significant[1, 2]
    assert not significant[1, 0] and not significant[2, 0]
    assert result.lags[0, 1, 0, 0] == 2
    assert "pcmci" in repr(result)
    with pytest.raises(TypeError, match="PCMCI estimator"):
        pcmci_test(object(), data)


def test_the_per_cell_pvalues_hold_their_level_on_independent_channels():
    # Measured over 40 seeds (3 independent AR(1) channels, 2000 samples, tau_max=4): 5.4 % of
    # the 240 cells have p < 0.05 with the Bonferroni over lags, 20.8 % with the raw smallest p.
    rejected = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        e = rng.standard_normal((2000, 3))
        x = np.zeros((2000, 3))
        for t in range(1, 2000):
            x[t] = np.array([0.7, 0.5, 0.3]) * x[t - 1] + e[t]
        p = pcmci_test(PCMCI(tau_max=4), TimeSeries.from_array(x.T, SFREQ)).pvalues[:, :, 0, 0]
        rejected += list(p[~np.eye(3, dtype=bool)] < 0.05)
    assert np.mean(rejected) <= 0.15  # 60 cells; 0.15 is far above 5 % and far below 21 %


def test_parameters_and_a_missing_tigramite(monkeypatch):
    with pytest.raises(ValueError, match="pc_alpha"):
        PCMCI(pc_alpha=1.5)
    with pytest.raises(TypeError):
        PCMCI(tau_min=0)
    monkeypatch.setitem(sys.modules, "tigramite", None)
    monkeypatch.setitem(sys.modules, "tigramite.pcmci", None)
    with pytest.raises(ImportError, match="dirconn\\[pcmci\\]"):
        PCMCI(tau_max=2).fit(TimeSeries.from_array(_chain(5, 300).T, SFREQ))


def test_signed_flag():
    assert PCMCI.info.signed and PCMCI.info.directed and not PCMCI.info.antisymmetric


def test_a_criterion_tau_max_covers_the_minimum_delay():
    # at 100 Hz the default 30 ms floor is 3 samples; 250 Hz would give 8
    data = TimeSeries.from_array(_chain(6, 2000).T, SFREQ)
    floored = _fit(PCMCI(tau_max="bic", min_lag_seconds=0.08), data)
    details = floored.provenance.details
    assert details["min_lag_samples"] == 8
    assert details["tau_max"] == max(details["var_order"], 8) == floored.provenance.model_order
    assert len(details["val_matrix"][0][1]) == details["tau_max"]
    off = _fit(PCMCI(tau_max="bic", min_lag_seconds=0.0), data).provenance.details
    assert off["tau_max"] == off["var_order"] and off["min_lag_samples"] == 0
    capped = _fit(PCMCI(tau_max="bic", max_order=4, min_lag_seconds=0.2), data)
    assert capped.provenance.details["tau_max"] == 4
    fixed = _fit(PCMCI(tau_max=2, min_lag_seconds=0.2), data).provenance.details
    assert fixed["tau_max"] == 2 and fixed["min_lag_samples"] == 0  # an integer is used as given
    assert PCMCI().min_lag_seconds == 0.03
    for bad in (-0.1, float("nan"), True):
        with pytest.raises(ValueError, match="min_lag_seconds"):
            PCMCI(min_lag_seconds=bad)
