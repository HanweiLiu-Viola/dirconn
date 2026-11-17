import numpy as np
import pytest

from dirconn import TimeSeries
from dirconn.io.readers import read_unit
from dirconn.workflows.config import (
    ControlsConfig,
    EstimationConfig,
    PrepareConfig,
    StatisticsConfig,
)
from dirconn.workflows.steps.control import time_reversal_control
from dirconn.workflows.steps.estimate import build_estimator, estimate
from dirconn.workflows.steps.prepare import prepare
from dirconn.workflows.steps.test import test_significance as run_significance

BANDS = {"theta": (4.0, 8.0), "beta": (13.0, 30.0)}


@pytest.fixture(scope="module")
def ts(bids_tiny):
    raw = read_unit(
        root=bids_tiny / "derivatives" / "dirconn-roi",
        subject="01",
        session=None,
        task="rest",
        run="1",
        datatype="eeg",
        level="roi",
        description="DK",
        node_kinds=("cortical", "direct"),
    )
    return prepare(raw, PrepareConfig(), events_tsv=None)


def _cfg(**kw):
    kw.setdefault("order", 3)
    kw.setdefault("bands", BANDS)
    return EstimationConfig(**kw)


def test_gc_estimate_has_no_bands(ts):
    est = estimate(ts, "gc", _cfg())
    assert est.method == "gc"
    assert est.bands == {}
    assert est.full.values.shape[:2] == (4, 4)


def test_rpdc_estimate_reduces_every_band(ts):
    est = estimate(ts, "rpdc", _cfg())
    assert set(est.bands) == set(BANDS)
    assert est.full.values.shape[2] > 1
    for conn in est.bands.values():
        assert conn.values.shape[2] == 1


def test_exclude_freqs_sets_bins_to_nan_before_reduction(ts):
    base = estimate(ts, "pdc", _cfg())
    est = estimate(ts, "pdc", _cfg(exclude_freqs=((5.0, 6.0),)))
    freqs = np.asarray(est.full.freqs)
    hit = (freqs >= 5.0) & (freqs <= 6.0)
    assert hit.any()
    assert np.isnan(est.full.values[:, :, hit, :]).all()
    np.testing.assert_array_equal(est.full.values[:, :, ~hit, :], base.full.values[:, :, ~hit, :])
    assert est.full.provenance.history[-1] == {"step": "exclude_freqs", "ranges": [[5.0, 6.0]]}
    assert not np.allclose(est.bands["theta"].values, base.bands["theta"].values, equal_nan=True)


def test_params_override_order(ts):
    cfg = _cfg(order=3, params={"gc": {"order": 5}})
    assert build_estimator("gc", cfg).order == 5
    assert estimate(ts, "gc", cfg).full.provenance.model_order == 5
    assert build_estimator("gc", _cfg(order=3)).order == 3


def test_time_reversal_control_returns_trgc(ts):
    cfg = _cfg()
    assert time_reversal_control(ts, cfg, ControlsConfig(time_reversal=False)) is None
    out = time_reversal_control(ts, cfg, ControlsConfig(time_reversal=True))
    assert out is not None
    assert out.values.shape[:2] == (4, 4)
    assert out.provenance.method == "trgc"


def test_surrogate_test_per_band_with_distinct_seeds(ts):
    cfg = _cfg()
    est = estimate(ts, "rpdc", cfg)
    stats = StatisticsConfig(test="surrogate", n_surrogates=10)
    tests = run_significance(ts, build_estimator("rpdc", cfg), est, stats, seed=7)
    assert [t.band for t in tests] == ["theta", "beta"]
    assert [t.seed for t in tests] == [7, 8]
    for t in tests:
        assert t.test == "surrogate" and t.n_surrogates == 10
        assert t.pvalues.shape == t.adjusted.shape == t.significant.shape == (4, 4)
        assert np.isnan(np.diag(t.pvalues)).all()
        assert np.isnan(np.diag(t.adjusted)).all()
        assert t.significant.dtype == bool
        assert not np.diag(t.significant).any()
        assert t.generator["name"] == "block_shuffle"
        assert t.generator["params"] == {"block_size": 1250}
        steps = [h["step"] for h in t.observed_history]
        assert steps[-1] == "block_shuffle_blocks"
        assert t.observed_history[-1]["n_blocks_kept"] == 8


def test_observed_history_carries_the_block_counts_on_masked_data():
    rng = np.random.default_rng(0)
    valid = np.ones(20000, dtype=bool)
    valid[10000:10500] = False  # a bad segment mid-record; the record edges are cut, not masked
    raw = TimeSeries.from_array(rng.standard_normal((3, 20000)), 500.0, valid=valid)
    prepared = prepare(raw, PrepareConfig(), events_tsv=None)
    assert prepared.valid is not None
    cfg = _cfg()
    est = estimate(prepared, "gc", cfg)
    stats = StatisticsConfig(test="surrogate", n_surrogates=5)
    (test,) = run_significance(prepared, build_estimator("gc", cfg), est, stats, seed=0)
    steps = [h["step"] for h in test.observed_history]
    assert steps == ["resample", "zscore", "block_shuffle_blocks"]
    assert test.observed_history[-1] == {
        "step": "block_shuffle_blocks",
        "n_blocks_total": 7,
        "n_blocks_kept": 6,
    }


def test_statistics_none_gives_no_tests(ts):
    cfg = _cfg()
    est = estimate(ts, "gc", cfg)
    stats = StatisticsConfig(test="none")
    assert run_significance(ts, build_estimator("gc", cfg), est, stats, seed=0) == ()


def test_masked_data_with_unsupported_method_gets_a_clear_error(ts):
    import dataclasses

    valid = np.ones((ts.data.shape[0], ts.data.shape[2]), bool)
    valid[:, :50] = False
    masked = dataclasses.replace(ts, valid=valid)
    with pytest.raises(ValueError, match="does not support masked data"):
        estimate(masked, "te", _cfg())
    out = time_reversal_control(masked, _cfg(), ControlsConfig(time_reversal=True))
    assert out is not None
    assert out.provenance.details["masks"]["n_valid"] < masked.n_times * masked.n_trials
