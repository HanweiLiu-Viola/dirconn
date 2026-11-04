"""Tests of time-reversed Granger causality and the asymmetry diagnostic."""

import warnings
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.signal import lfilter

from dirconn import GC, Connectivity, DirconnWarning, TimeSeries
from dirconn.estimator import Estimator, MethodInfo
from dirconn.time_reversal import (
    TRGC,
    TimeReversalAsymmetry,
    reverse_time,
    time_reversal_asymmetry,
    trgc,
)

SFREQ = 100.0
MIX = np.array([[1.0, 0.8], [0.6, 1.0]])


def _lagged_var(seed, n_trials=1, n_times=6000):
    """Bivariate VAR(2): channel 0 drives channel 1 at lag 2, nothing the other way."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n_trials, 2, n_times + 100))
    e = rng.standard_normal(x.shape)
    for t in range(2, x.shape[-1]):
        x[:, 0, t] = 0.5 * x[:, 0, t - 1] - 0.3 * x[:, 0, t - 2] + e[:, 0, t]
        x[:, 1, t] = 0.4 * x[:, 1, t - 1] + 0.4 * x[:, 0, t - 2] + e[:, 1, t]
    return x[:, :, 100:]


def _lagged(seed, **kwargs):
    x = _lagged_var(seed, **kwargs)
    if x.shape[0] == 1:
        return TimeSeries.from_array(x[0], SFREQ)
    return TimeSeries.from_array(x, SFREQ, segmentation="epoched")


def _mixed_ar(seed, n_times=6000):
    """Two independent AR(1) sources (0.9 and 0.3) mixed instantaneously: GC sees false edges."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((2, n_times))
    s = np.stack([lfilter([1], [1, -0.9], e[0]), lfilter([1], [1, -0.3], e[1])])
    return TimeSeries.from_array(MIX @ s, SFREQ)


def _fit(estimator, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DirconnWarning)
        return estimator.fit(data)


# --- TRGC ----------------------------------------------------------------------------------


def test_reverse_time_reverses_every_trial():
    data = _lagged(0, n_trials=3, n_times=50)
    back = reverse_time(data)
    np.testing.assert_array_equal(back.data, np.asarray(data.data)[:, :, ::-1])
    assert back.segmentation == "epoched" and back.names == data.names
    np.testing.assert_array_equal(reverse_time(back).data, data.data)


def test_trgc_antisymmetric_and_built_from_the_recorded_components():
    noise = np.random.default_rng(1).standard_normal((1, 3000))
    data = TimeSeries.from_array(np.vstack([_lagged_var(1, n_times=3000)[0], noise]), SFREQ)
    result = _fit(TRGC(), data)
    v = result.values[:, :, 0, 0]
    assert np.isnan(np.diag(v)).all()
    off = ~np.eye(3, dtype=bool)
    np.testing.assert_allclose(v[off], -v.T[off], atol=1e-15)
    d = result.provenance.details
    net_f = np.array(d["net_forward"], dtype=float)
    net_r = np.array(d["net_reversed"], dtype=float)
    np.testing.assert_allclose(v[off], (net_f - net_r)[off], rtol=1e-12)
    gc_f = np.array(d["gc_forward"], dtype=float)
    np.testing.assert_allclose(net_f[off], (gc_f - gc_f.T)[off], rtol=1e-12)
    assert result.provenance.method == "trgc"
    assert d["conditional"] is True


def test_trgc_cancels_an_instantaneous_mixture_where_gc_does_not():
    # Measured over 20 seeds: the larger GC of the pair is 0.153 to 0.186, TRGC[0, 1] is
    # -0.0012 to 0.0034; for the lagged VAR (no mixing) TRGC[0, 1] is 0.31 to 0.41.
    for seed in range(4):
        data = _mixed_ar(seed)
        g = _fit(GC(), data).values[:, :, 0, 0]
        assert max(g[0, 1], g[1, 0]) > 0.1
        assert abs(_fit(TRGC(), data).values[0, 1, 0, 0]) < 0.01
        assert _fit(TRGC(), _lagged(seed)).values[0, 1, 0, 0] > 0.25


def test_trgc_reuses_the_forward_order_and_passes_conditional():
    data = _lagged(2, n_times=3000)
    result = _fit(TRGC(order="bic", max_order=8), data)
    order = result.provenance.model_order
    reversed_gc = _fit(GC(order=order, max_order=8), reverse_time(data)).values[:, :, 0, 0]
    np.testing.assert_allclose(
        np.array(result.provenance.details["gc_reversed"], dtype=float)[0, 1], reversed_gc[0, 1]
    )
    assert result.provenance.details["order_selection"] is not None


def test_trgc_selects_the_order_only_once(monkeypatch):
    """The reversed fit reuses the forward order instead of selecting its own."""
    import dirconn.granger

    calls = []
    original = dirconn.granger.choose_order

    def spy(data, order, *args, **kwargs):
        calls.append(order)
        return original(data, order, *args, **kwargs)

    monkeypatch.setattr(dirconn.granger, "choose_order", spy)
    result = _fit(TRGC(order="bic", max_order=8, check_order_stability=False), _lagged(2))
    assert calls == ["bic", result.provenance.model_order]


def test_trgc_passes_conditional_to_both_fits():
    noise = np.random.default_rng(5).standard_normal((1, 3000))
    data = TimeSeries.from_array(np.vstack([_lagged_var(5, n_times=3000)[0], noise]), SFREQ)
    for conditional in (True, False):
        result = _fit(TRGC(order=2, conditional=conditional), data)
        assert result.provenance.details["conditional"] is conditional
        expected = _fit(GC(order=2, conditional=conditional), data).values[:, :, 0, 0]
        got = np.array(result.provenance.details["gc_forward"], dtype=float)
        np.testing.assert_allclose(got, expected, rtol=1e-12)
        back = _fit(GC(order=2, conditional=conditional), reverse_time(data)).values[:, :, 0, 0]
        got = np.array(result.provenance.details["gc_reversed"], dtype=float)
        np.testing.assert_allclose(got, back, rtol=1e-12)
    a = np.array(_fit(TRGC(order=2), data).provenance.details["gc_forward"], dtype=float)
    b = np.array(
        _fit(TRGC(order=2, conditional=False), data).provenance.details["gc_forward"], dtype=float
    )
    assert not np.allclose(a, b, equal_nan=True)  # three channels: the modes differ


def test_trgc_shorthand_and_parameter_checks():
    data = _lagged(3, n_times=2000)
    np.testing.assert_array_equal(_fit(TRGC(order=2), data).values, trgc(data, order=2).values)
    with pytest.raises(TypeError):
        TRGC(check_order_stability="yes")
    with pytest.raises(TypeError):
        TRGC(n_freqs=10)


def test_trgc_close_to_mne_connectivity_gc_tr():
    """Two channels: mne-connectivity's state-space GC and gc_tr, averaged over frequency."""
    mne_connectivity = pytest.importorskip("mne_connectivity")
    # Measured over 8 seeds (40 trials of 5 s): ours is 0.9 % to 3.4 % above mne-connectivity's,
    # whose spectra start at 1 Hz and come from a 20-lag model of multitaper cross-spectra.
    for seed in range(2):
        x = _lagged_var(seed, n_trials=40, n_times=500)
        ours = _fit(TRGC(order=2), TimeSeries.from_array(x, SFREQ, segmentation="epoched"))
        means = []
        for indices in (([[0]], [[1]]), ([[1]], [[0]])):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cons = mne_connectivity.spectral_connectivity_epochs(
                    x,
                    method=["gc", "gc_tr"],
                    indices=indices,
                    sfreq=SFREQ,
                    mode="multitaper",
                    fmin=1.0,
                    fmax=SFREQ / 2,
                    gc_n_lags=20,
                    verbose=False,
                )
            means.append([float(c.get_data()[0].mean()) for c in cons])
        (gc01, tr01), (gc10, tr10) = means
        reference = (gc01 - gc10) - (tr01 - tr10)
        assert ours.values[0, 1, 0, 0] == pytest.approx(reference, rel=0.07)


# --- asymmetry diagnostic ------------------------------------------------------------------

ORIGINAL = np.array([[0.0, 0.30, 0.05], [0.02, 0.0, 0.4], [0.1, 0.1, 0.0]])
REVERSED = np.array([[0.0, 0.01, 0.05], [0.25, 0.0, 0.1], [0.3, 0.1, 0.0]])
# the old package's asymmetry() on these two matrices (its diagonal was 0 / 1e-12 = 0)
OLD_ASYMMETRY = np.array([[0.0, 0.93548387, 0.0], [-0.85185185, 0.0, 0.6], [-0.5, 0.0, 0.0]])


@dataclass(frozen=True)
class _ByDirection(Estimator):
    """Returns ORIGINAL when the first channel rises over time and REVERSED when it falls."""

    info = MethodInfo(name="by_direction", backend="test stub", directed=True)

    def _fit(self, data):
        x = np.asarray(data.data)
        values = ORIGINAL if x[0, 0, -1] > x[0, 0, 0] else REVERSED
        return Connectivity.from_array(
            values.copy(), source=data.names, provenance=self._provenance(data)
        )


def _ramp():
    t = np.linspace(0, 1, 50)
    return TimeSeries.from_array(np.stack([t, t**2, np.sqrt(t)]), SFREQ)


def test_asymmetry_matches_the_old_package_and_marks_zero_cells_undefined():
    out = time_reversal_asymmetry(_ByDirection(), _ramp())
    assert isinstance(out, TimeReversalAsymmetry)
    score = out.asymmetry.values[:, :, 0, 0]
    off = ~np.eye(3, dtype=bool)
    np.testing.assert_allclose(score[off], OLD_ASYMMETRY[off], rtol=1e-7)
    assert np.isnan(np.diag(score)).all()  # both zero: NaN, where the old code gave 0
    np.testing.assert_allclose(out.robustness[:, :, 0, 0][off], np.abs(OLD_ASYMMETRY)[off], 1e-7)
    np.testing.assert_array_equal(out.original.values[:, :, 0, 0], ORIGINAL)
    np.testing.assert_array_equal(out.reversed.values[:, :, 0, 0], REVERSED)
    assert out.asymmetry.provenance.history[-1]["op"] == "time_reversal_asymmetry"


def test_asymmetry_with_gc_separates_lag_from_mixture():
    lagged = time_reversal_asymmetry(GC(order=2, check_order_stability=False), _lagged(4))
    assert lagged.asymmetry.values[0, 1, 0, 0] > 0.9
    mixed = time_reversal_asymmetry(GC(order=2, check_order_stability=False), _mixed_ar(4))
    assert np.nanmax(np.abs(mixed.asymmetry.values)) < 0.2


def test_asymmetry_refuses_complex_results_and_non_estimators():
    @dataclass(frozen=True)
    class _Complex(Estimator):
        info = MethodInfo(name="cplx", backend="test stub", directed=True, complex_valued=True)

        def _fit(self, data):
            values = np.full((3, 3), 1 + 1j)
            return Connectivity.from_array(
                values, source=data.names, provenance=self._provenance(data)
            )

    with pytest.raises(ValueError, match="real values"):
        time_reversal_asymmetry(_Complex(), _ramp())
    with pytest.raises(TypeError, match="Estimator"):
        time_reversal_asymmetry(lambda d: d, _ramp())
