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


def test_surrogate_test_of_every_band_shares_the_surrogates(ts):
    cfg = _cfg()
    est = estimate(ts, "rpdc", cfg)
    stats = StatisticsConfig(test="surrogate", n_surrogates=10)
    tests = run_significance(ts, build_estimator("rpdc", cfg), est, stats, seed=7)
    assert [t.band for t in tests] == ["theta", "beta"]
    assert [t.seed for t in tests] == [7, 7]
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


@pytest.mark.parametrize("method", ["mtgc", "imcoh", "psi", "wdtf"])
def test_methods_without_library_defaults_run_on_continuous_data(ts, method):
    est = estimate(ts, method, _cfg())
    assert set(est.bands) == set(BANDS)


def test_pipeline_defaults_depend_on_the_segmentation(ts):
    from dirconn.workflows.steps.estimate import method_kwargs

    assert method_kwargs("imcoh", _cfg(), ts)["segment_length"] == 2.0
    assert method_kwargs("wdtf", _cfg(), ts)["window"] == 2.0
    epoched = TimeSeries.from_array(
        np.zeros((3, 4, 200)), sfreq=250.0, names=list(ts.names), segmentation="epoched"
    )
    assert epoched.segmentation == "epoched"
    assert "segment_length" not in method_kwargs("imcoh", _cfg(), epoched)
    assert method_kwargs("wdtf", _cfg(), epoched)["window"] == 0.5
    assert "segment_length" not in method_kwargs("imcoh", _cfg())  # no record: no defaults


def test_params_override_pipeline_defaults(ts):
    from dirconn.workflows.steps.estimate import method_kwargs

    cfg = _cfg(methods=("imcoh",), params={"imcoh": {"segment_length": 4.0}})
    assert method_kwargs("imcoh", cfg, ts)["segment_length"] == 4.0


def test_psi_gets_the_configured_bands(ts):
    from dirconn import PSI

    est = estimate(ts, "psi", _cfg())
    direct = PSI(segment_length=2.0, bands=[BANDS["theta"], BANDS["beta"]]).fit(ts)
    np.testing.assert_allclose(est.bands["theta"].values[:, :, 0, 0], direct.values[:, :, 0, 0])
    np.testing.assert_allclose(est.bands["beta"].values[:, :, 0, 0], direct.values[:, :, 1, 0])


def test_psi_refuses_a_band_holding_two_centres(ts):
    with pytest.raises(ValueError, match="band centres"):
        estimate(ts, "psi", _cfg(bands={"wide": (4.0, 30.0), "beta": (13.0, 30.0)}))


def test_psi_bands_in_params_are_refused():
    with pytest.raises(ValueError, match="estimation.bands"):
        EstimationConfig(methods=("psi",), params={"psi": {"bands": [[8.0, 12.0]]}})


def test_one_order_selection_serves_every_var_method(ts):
    from dirconn.workflows.steps.estimate import select_shared_order

    cfg = _cfg(order="bic", order_max=8)
    selection = select_shared_order(ts, cfg)
    assert selection is not None and 1 <= selection.order <= 8
    for method in ("gc", "dtf", "pdc", "adtf"):
        est = estimate(ts, method, cfg, selection)
        prov = est.full.provenance
        assert prov.model_order == selection.order
        assert prov.details["order_selection"]["criterion"] == "bic"
        assert prov.history[-1]["step"] == "shared_order_selection"
        assert prov.assumptions.get("order_selection") is not None


def test_shared_order_gives_the_same_result_as_the_estimator_own_selection(ts):
    from dirconn.workflows.steps.estimate import select_shared_order

    cfg = _cfg(order="bic", order_max=8)
    shared = estimate(ts, "pdc", cfg, select_shared_order(ts, cfg))
    own = estimate(ts, "pdc", cfg)
    assert shared.full.provenance.model_order == own.full.provenance.model_order
    np.testing.assert_allclose(shared.full.values, own.full.values)


def test_the_estimator_for_the_surrogates_has_a_fixed_order(ts):
    cfg = _cfg(order="bic")
    assert build_estimator("gc", cfg, ts, order=7).order == 7
    fixed = _cfg(order="bic", methods=("gc",), params={"gc": {"order": 3}})
    assert build_estimator("gc", fixed, ts, order=7).order == 3
    assert "order" not in build_estimator("imcoh", cfg, ts, order=7).__dataclass_fields__


def test_a_fixed_order_needs_no_selection():
    from dirconn.workflows.steps.estimate import select_shared_order

    assert select_shared_order(None, _cfg(order=4)) is None  # type: ignore[arg-type]


def test_methods_without_an_order_keep_their_provenance(ts):
    from dirconn.workflows.steps.estimate import select_shared_order

    cfg = _cfg(order="bic", order_max=8)
    est = estimate(ts, "imcoh", cfg, select_shared_order(ts, cfg))
    assert "order_selection" not in est.full.provenance.details


def test_time_resolved_result_is_not_block_shuffled(ts):
    from dirconn.workflows.steps.test import untested_reason

    cfg = _cfg(order=2, methods=("adtf",), params={"adtf": {"n_freqs": 8, "fmax": 40.0}})
    est = estimate(ts, "adtf", cfg)
    assert est.full.values.shape[3] > 1
    stats = StatisticsConfig(test="surrogate", n_surrogates=5)
    assert "block shuffling" in untested_reason(ts, est, stats)
    assert run_significance(ts, build_estimator("adtf", cfg, ts), est, stats, seed=0) == ()
    # another surrogate keeps the time axis, so the test runs
    other = StatisticsConfig(test="surrogate", surrogate="circular_shift", n_surrogates=5)
    assert untested_reason(ts, est, other) is None


def test_pcmci_uses_its_own_p_values(ts):
    pytest.importorskip("tigramite")
    cfg = _cfg(methods=("pcmci",), params={"pcmci": {"tau_max": 2}})
    est = estimate(ts, "pcmci", cfg)
    stats = StatisticsConfig(test="surrogate", n_surrogates=5)
    (test,) = run_significance(ts, build_estimator("pcmci", cfg, ts), est, stats, seed=0)
    assert test.test == "builtin" and test.band is None
    assert test.n_surrogates is None and test.seed is None
    assert test.generator["name"] == "builtin"
    expected = np.array(est.full.provenance.details["pvalues"], dtype=float)
    np.testing.assert_array_equal(test.pvalues, expected)


def test_multivariate_te_uses_its_own_p_values(ts):
    import dataclasses

    from dirconn.workflows.steps.estimate import Estimate

    # A stand-in for a multivariate TE result (a real fit takes minutes): the test step only
    # reads the p-values from the provenance and fits nothing.
    gc = estimate(ts, "gc", _cfg()).full
    n = len(gc.source)
    p_values = [[None if i == j else 0.01 * (i + j) for j in range(n)] for i in range(n)]
    details = {"mode": "multivariate", "p_values": p_values}
    prov = dataclasses.replace(gc.provenance, details=details)
    est = Estimate("te", gc.with_provenance(prov), {})
    stats = StatisticsConfig(test="surrogate", n_surrogates=2)
    (test,) = run_significance(ts, None, est, stats, seed=0)
    assert test.test == "builtin"
    np.testing.assert_array_equal(test.pvalues, np.array(p_values, dtype=float))
    assert np.isnan(np.diag(test.pvalues)).all()
