"""Per-sample ``valid`` and ``predict`` masks on TimeSeries and in the VAR core (issue #2)."""

import dataclasses
import warnings
from dataclasses import dataclass, field

import numpy as np
import pytest

from dirconn import (
    ADTF,
    DTF,
    GC,
    GPDC,
    MTGC,
    PCMCI,
    PDC,
    PSI,
    RPDC,
    TE,
    TPDC,
    TRGC,
    WDTF,
    Connectivity,
    DirconnWarning,
    ImCoh,
    TimeSeries,
    bootstrap,
    contrast_test,
    fit_var,
    select_order,
    surrogate_test,
)
from dirconn.estimator import Estimator, MethodInfo, residual_whiteness_check
from dirconn.mvar import cv_splits, lagged_design, row_mask
from dirconn.surrogates import (
    AAFT,
    IAAFT,
    BlockShuffle,
    CircularShift,
    PhaseRandomize,
    TrialShuffle,
)

COEFS = np.zeros((2, 3, 3))
COEFS[0] = [[0.5, 0.0, 0.0], [0.4, 0.3, 0.0], [0.0, 0.3, 0.2]]  # [target, source]
COEFS[1] = [[-0.2, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, -0.1]]


def simulate(n_times, *, n_trials=1, seed=0, burn=100):
    rng = np.random.default_rng(seed)
    order, n, _ = COEFS.shape
    out = np.zeros((n_trials, n, n_times))
    for trial in range(n_trials):
        x = np.zeros((n, n_times + burn))
        noise = rng.standard_normal((n, n_times + burn))
        for t in range(order, n_times + burn):
            x[:, t] = noise[:, t] + sum(COEFS[k] @ x[:, t - k - 1] for k in range(order))
        out[trial] = x[:, burn:]
    return out


def continuous(n_times=3000, seed=0, **masks):
    return TimeSeries.from_array(simulate(n_times, seed=seed)[0], 100.0, **masks)


def epoched(n_trials=8, n_times=300, seed=0, **masks):
    return TimeSeries.from_array(
        simulate(n_times, n_trials=n_trials, seed=seed), 100.0, segmentation="epoched", **masks
    )


@dataclass(frozen=True)
class MaskSpy(Estimator):
    """Records every dataset it is fitted on; accepts masks."""

    info = MethodInfo(name="maskspy", backend="test stub", directed=False, supports_mask=True)
    seen: list = field(default_factory=list, init=False, compare=False)

    def _fit(self, data: TimeSeries) -> Connectivity:
        self.seen.append(data)
        values = np.ones((data.n_channels, data.n_channels))
        np.fill_diagonal(values, np.nan)
        return Connectivity.from_array(values, source=data.names, provenance=self._provenance(data))


# --------------------------------------------------------------------------------------------
# The container
# --------------------------------------------------------------------------------------------


class TestTimeSeriesMasks:
    def test_masks_default_to_none(self):
        ts = continuous(200)
        assert ts.valid is None and ts.predict is None
        assert not ts.has_mask

    def test_masks_are_stored_read_only(self):
        valid = np.ones((1, 200), dtype=bool)
        valid[0, 50:60] = False
        ts = continuous(200, valid=valid)
        assert ts.has_mask
        np.testing.assert_array_equal(ts.valid, valid)
        assert not ts.valid.flags.writeable
        valid[0, 0] = False  # the series keeps its own copy
        assert ts.valid[0, 0]

    def test_masks_that_are_all_true_are_dropped(self):
        ones = np.ones((1, 200), dtype=bool)
        ts = continuous(200, valid=ones, predict=ones)
        assert ts.valid is None and ts.predict is None
        assert not ts.has_mask

    def test_one_dimensional_masks_are_accepted_for_two_dimensional_input(self):
        predict = np.zeros(200, dtype=bool)
        predict[100:150] = True
        ts = continuous(200, predict=predict)
        assert ts.predict.shape == (1, 200)
        np.testing.assert_array_equal(ts.predict[0], predict)

    @pytest.mark.parametrize("name", ["valid", "predict"])
    def test_a_mask_must_have_the_shape_of_trials_by_times(self, name):
        with pytest.raises(ValueError, match=name):
            continuous(200, **{name: np.ones((1, 199), dtype=bool)})

    @pytest.mark.parametrize("name", ["valid", "predict"])
    def test_a_mask_must_be_boolean(self, name):
        with pytest.raises(TypeError, match=name):
            continuous(200, **{name: np.ones((1, 200))})

    def test_nan_is_still_rejected_by_default(self):
        x = simulate(200)[0]
        x[1, 20] = np.nan
        with pytest.raises(ValueError, match="nan='mask'"):
            TimeSeries.from_array(x, 100.0)

    def test_nan_mask_replaces_nan_by_zero_and_marks_the_sample_invalid(self):
        x = simulate(200)[0]
        x[1, 20] = np.nan
        x[:, 100:110] = np.nan
        ts = TimeSeries.from_array(x, 100.0, nan="mask")
        assert np.isfinite(ts.data).all()
        assert ts.data[0, 1, 20] == 0.0
        assert ts.data[0, 0, 20] == x[0, 20]  # the other channels keep their values
        expected = np.ones(200, dtype=bool)
        expected[20] = False
        expected[100:110] = False
        np.testing.assert_array_equal(ts.valid[0], expected)

    def test_nan_mask_is_combined_with_a_given_valid_mask(self):
        x = simulate(200)[0]
        x[0, 5] = np.nan
        valid = np.ones(200, dtype=bool)
        valid[150] = False
        ts = TimeSeries.from_array(x, 100.0, nan="mask", valid=valid)
        assert not ts.valid[0, 5] and not ts.valid[0, 150]
        assert ts.valid.sum() == 198

    def test_nan_mask_without_nan_gives_no_mask(self):
        ts = TimeSeries.from_array(simulate(200)[0], 100.0, nan="mask")
        assert ts.valid is None

    def test_an_unknown_nan_policy_is_rejected(self):
        with pytest.raises(ValueError, match="nan"):
            TimeSeries.from_array(simulate(200)[0], 100.0, nan="zero")

    def test_pick_keeps_the_masks(self):
        valid = np.ones(200, dtype=bool)
        valid[10] = False
        predict = np.zeros(200, dtype=bool)
        predict[50:] = True
        ts = continuous(200, valid=valid, predict=predict).pick(["ch2", "ch0"])
        np.testing.assert_array_equal(ts.valid[0], valid)
        np.testing.assert_array_equal(ts.predict[0], predict)

    def test_repr_mentions_masks_only_when_there_are_some(self):
        assert "mask" not in repr(continuous(200))
        predict = np.zeros(200, dtype=bool)
        predict[50:] = True
        assert "masked" in repr(continuous(200, predict=predict))


# --------------------------------------------------------------------------------------------
# Row selection in the VAR core
# --------------------------------------------------------------------------------------------


class TestRowSelection:
    def test_all_true_masks_reproduce_the_design_bit_for_bit(self):
        x = simulate(300, n_trials=3)
        ones = np.ones((3, 300), dtype=bool)
        y0, d0 = lagged_design(x, 4)
        y1, d1 = lagged_design(x, 4, valid=ones, predict=ones)
        assert np.array_equal(y0, y1) and np.array_equal(d0, d1)
        y0, d0 = lagged_design(x, 4, start=7)
        y1, d1 = lagged_design(x, 4, start=7, valid=ones, predict=ones)
        assert np.array_equal(y0, y1) and np.array_equal(d0, d1)

    def test_a_row_is_used_iff_predict_and_its_whole_window_is_valid(self):
        x = np.arange(1 * 2 * 12, dtype=float).reshape(1, 2, 12)
        valid = np.ones((1, 12), dtype=bool)
        valid[0, 5] = False
        predict = np.ones((1, 12), dtype=bool)
        predict[0, 10] = False
        y, design = lagged_design(x, 2, valid=valid, predict=predict)
        # candidates t = 2..11; t = 5, 6, 7 touch sample 5; t = 10 is not predicted
        used = [2, 3, 4, 8, 9, 11]
        np.testing.assert_array_equal(y, x[0][:, used].T)
        np.testing.assert_array_equal(design[:, :2], x[0][:, [t - 1 for t in used]].T)
        np.testing.assert_array_equal(design[:, 2:], x[0][:, [t - 2 for t in used]].T)

    def test_row_mask_matches_the_rule(self):
        valid = np.ones((2, 10), dtype=bool)
        valid[1, 4] = False
        predict = np.zeros((2, 10), dtype=bool)
        predict[:, 6:] = True
        rows = row_mask(valid, predict, order=2, n_times=10)
        assert rows.shape == (2, 8)  # t = 2..9
        np.testing.assert_array_equal(rows[0], [False] * 4 + [True] * 4)
        np.testing.assert_array_equal(rows[1], [False] * 4 + [False, True, True, True])
        assert row_mask(None, None, order=2, n_times=10) is None

    def test_masks_must_match_the_data(self):
        with pytest.raises(ValueError, match="shape"):
            lagged_design(simulate(50), 2, valid=np.ones((1, 49), dtype=bool))


def tiles_and_mask(x, onsets, order, k):
    """Tiles of ``order`` history + ``k`` predicted samples, and the equivalent predict mask."""
    tiles = np.stack([x[:, s - order : s + k] for s in onsets])
    predict = np.zeros(x.shape[1], dtype=bool)
    for s in onsets:
        predict[s : s + k] = True
    return tiles, predict


class TestTilesEqualMasks:
    """Tiles of length ``order + k`` as trials are the same regression as a predict mask."""

    ORDER, K = 3, 10

    def data(self):
        x = simulate(4000, seed=3)[0]
        onsets = np.arange(50, 3900, 37)  # predicted stretches overlap other tiles' history
        tiles, predict = tiles_and_mask(x, onsets, self.ORDER, self.K)
        tiled = TimeSeries.from_array(tiles, 100.0, segmentation="epoched")
        masked = TimeSeries.from_array(x, 100.0, predict=predict)
        return tiled, masked

    def test_var_coefficients_are_the_same(self):
        tiled, masked = self.data()
        a = fit_var(tiled, self.ORDER, demean=False)
        b = fit_var(masked, self.ORDER, demean=False)
        assert a.n_obs == b.n_obs == tiled.n_trials * self.K
        np.testing.assert_allclose(a.coefs, b.coefs, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(a.noise_cov, b.noise_cov, rtol=1e-12)
        np.testing.assert_allclose(a.regressor_cov, b.regressor_cov, rtol=1e-12)

    def test_order_selection_sees_the_same_rows(self):
        tiled, masked = self.data()
        a = select_order(tiled, self.ORDER, demean=False)
        b = select_order(masked, self.ORDER, demean=False)
        assert a.n_obs == b.n_obs
        np.testing.assert_allclose(a.bic, b.bic, rtol=1e-12)
        assert a.order == b.order

    @pytest.mark.parametrize("estimator", [GC, GPDC], ids=["gc", "gpdc"])
    def test_estimators_give_the_same_values(self, estimator):
        tiled, masked = self.data()
        est = estimator(order=self.ORDER, demean=False, check_order_stability=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DirconnWarning)
            a, b = est.fit(tiled), est.fit(masked)
        np.testing.assert_allclose(a.values, b.values, rtol=1e-10, atol=1e-14, equal_nan=True)
        assert a.provenance.details["n_obs"] == b.provenance.details["n_obs"]


class TestBitForBit:
    """Data without masks, or with masks that are all True, give exactly the old results."""

    @pytest.mark.parametrize("estimator", [GC, DTF, GPDC, RPDC], ids=["gc", "dtf", "gpdc", "rpdc"])
    def test_all_true_masks_change_nothing(self, estimator):
        plain = epoched(n_trials=4, n_times=200)
        ones = np.ones((4, 200), dtype=bool)
        masked = TimeSeries(
            plain.data, plain.sfreq, plain.channels, "epoched", valid=ones, predict=ones
        )
        est = estimator(order="bic", max_order=4)
        a, b = est.fit(plain), est.fit(masked)
        assert np.array_equal(a.values, b.values, equal_nan=True)
        assert a.provenance.input_hash == b.provenance.input_hash
        assert a.provenance.details == b.provenance.details

    def test_cv_splits_are_unchanged(self):
        for n_trials in (1, 3, 6):
            scheme, splits = cv_splits(n_trials, 40, 5, gap=3)
            rows = np.arange(n_trials * 40).reshape(n_trials, 40)
            if n_trials >= 4:
                assert scheme == "trials"
                for fold, (_, test) in enumerate(splits):
                    np.testing.assert_array_equal(test, rows[fold::5].ravel())
            else:
                assert scheme == "blocks"
                assert sum(t.size for _, t in splits) == n_trials * 40


# --------------------------------------------------------------------------------------------
# Gaps
# --------------------------------------------------------------------------------------------


class TestGaps:
    def gapped(self, fill):
        x = simulate(3000, seed=5)[0]
        x[:, 1000:1040] = fill
        x[2, 2000] = fill
        return x

    def test_the_content_of_invalid_samples_does_not_matter(self):
        a = TimeSeries.from_array(self.gapped(np.nan), 100.0, nan="mask")
        b = TimeSeries.from_array(self.gapped(1e6), 100.0, valid=a.valid[0])
        for demean in (False, True):
            fa, fb = fit_var(a, 4, demean=demean), fit_var(b, 4, demean=demean)
            np.testing.assert_allclose(fa.coefs, fb.coefs, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(fa.mean, fb.mean, rtol=1e-12, atol=1e-14)

    def test_the_mean_is_taken_over_valid_samples(self):
        x = self.gapped(np.nan)
        ts = TimeSeries.from_array(x, 100.0, nan="mask")
        valid = ts.valid[0]
        fit = fit_var(ts, 2)
        np.testing.assert_allclose(fit.mean, x[:, valid].mean(axis=1), rtol=1e-12)

    def test_rows_touching_a_gap_are_dropped(self):
        ts = TimeSeries.from_array(self.gapped(np.nan), 100.0, nan="mask")
        fit = fit_var(ts, 4)
        # 2996 candidate rows; the 40-sample gap removes 44 of them, the single sample 5
        assert fit.n_obs == 2996 - 44 - 5

    def test_gc_reports_the_selected_rows(self):
        ts = TimeSeries.from_array(self.gapped(np.nan), 100.0, nan="mask")
        result = GC(order=4).fit(ts)
        assert result.provenance.details["n_obs"] == 2996 - 49
        masks = result.provenance.details["masks"]
        assert masks["n_valid"] == 3000 - 41 and masks["n_predict"] == 3000
        sampling = result.provenance.assumptions.get("sampling_adequacy")
        assert f"{2996 - 49} observations" in sampling.detail

    def test_cross_validation_works_on_masked_data(self):
        ts = TimeSeries.from_array(self.gapped(np.nan), 100.0, nan="mask")
        selection = select_order(ts, 5, criterion="cv")
        assert selection.cv_scheme == "blocks" and selection.cv is not None
        masked_epochs = epoched(
            n_trials=6, n_times=200, predict=np.tile(np.arange(200) > 40, (6, 1))
        )
        selection = select_order(masked_epochs, 4, criterion="cv")
        assert selection.cv_scheme == "trials" and selection.n_folds == 5

    def test_the_hash_depends_on_the_masks(self):
        x = simulate(1000)[0]
        early = np.arange(1000) < 500
        a = GC(order=2).fit(TimeSeries.from_array(x, 100.0, predict=early))
        b = GC(order=2).fit(TimeSeries.from_array(x, 100.0, predict=~early))
        assert a.provenance.input_hash != b.provenance.input_hash


class TestWhiteness:
    def test_a_full_row_mask_changes_nothing(self):
        rng = np.random.default_rng(0)
        u = rng.standard_normal((3 * 200, 2))
        a = residual_whiteness_check(u, 3, 2)
        b = residual_whiteness_check(u, 3, 2, rows=np.ones((3, 200), dtype=bool))
        assert (a.status, a.value) == (b.status, b.value)

    def test_gaps_are_handled(self):
        ts = TimeSeries.from_array(TestGaps().gapped(np.nan), 100.0, nan="mask")
        check = GC(order=2).fit(ts).provenance.assumptions.get("residual_whiteness")
        assert check.status in ("pass", "warn")

    def test_short_stretches_are_not_checked(self):
        x = simulate(3000)[0]
        onsets = np.arange(50, 2900, 37)
        _, predict = tiles_and_mask(x, onsets, 2, 10)
        result = GC(order=2).fit(TimeSeries.from_array(x, 100.0, predict=predict))
        assert result.provenance.assumptions.get("residual_whiteness").status == "not_checked"


# --------------------------------------------------------------------------------------------
# Which estimators accept masks
# --------------------------------------------------------------------------------------------


def masked_epochs():
    predict = np.zeros((8, 300), dtype=bool)
    predict[:, 100:] = True
    return epoched(predict=predict)


class TestSupport:
    @pytest.mark.parametrize("estimator", [GC, DTF, PDC, GPDC, RPDC])
    def test_static_var_estimators_accept_masks(self, estimator):
        assert estimator.info.supports_mask
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DirconnWarning)
            result = estimator(order=2).fit(masked_epochs())
        assert result.provenance.details["n_obs"] == 8 * 200

    @pytest.mark.parametrize(
        "estimator",
        [
            MTGC(),
            TE(),
            PCMCI(),
            PSI(bands=[(8, 13)]),
            ImCoh(),
            TRGC(),
            ADTF(),
            TPDC(),
            WDTF(window=1.0),
        ],
        ids=lambda e: type(e).__name__,
    )
    def test_other_estimators_refuse_masked_data(self, estimator):
        assert not estimator.info.supports_mask
        with pytest.raises(ValueError, match="mask"):
            estimator.fit(masked_epochs())

    def test_method_info_defaults_to_no_mask_support(self):
        assert not MethodInfo(name="x", backend="y", directed=True).supports_mask

    def test_standardisation_uses_valid_samples_only(self):
        a = TimeSeries.from_array(TestGaps().gapped(np.nan), 100.0, nan="mask")
        b = TimeSeries.from_array(TestGaps().gapped(1e3), 100.0, valid=a.valid[0])
        ra, rb = DTF(order=3).fit(a), DTF(order=3).fit(b)
        np.testing.assert_allclose(ra.values, rb.values, rtol=1e-10)
        np.testing.assert_allclose(
            ra.provenance.preprocessing["scale"], rb.provenance.preprocessing["scale"]
        )


# --------------------------------------------------------------------------------------------
# Surrogates and resampling
# --------------------------------------------------------------------------------------------


def gappy_epochs(n_trials=6, n_times=120, seed=0):
    rng = np.random.default_rng(seed)
    valid = np.ones((n_trials, n_times), dtype=bool)
    predict = np.zeros((n_trials, n_times), dtype=bool)
    for k in range(n_trials):
        start = int(rng.integers(0, n_times - 10))
        valid[k, start : start + 5] = False
        predict[k, int(rng.integers(10, 60)) :] = True
    return epoched(n_trials, n_times, seed=seed, valid=valid, predict=predict)


class TestSurrogates:
    @pytest.mark.parametrize(
        "generator",
        [PhaseRandomize(), AAFT(), IAAFT(), CircularShift(min_shift=5)],
        ids=lambda g: g.name,
    )
    def test_generators_that_move_samples_within_a_trial_refuse_masks(self, generator):
        with pytest.raises(ValueError, match="mask"):
            generator.generate_one(gappy_epochs(), np.random.default_rng(0))

    def test_block_shuffle_refuses_masks(self):
        with pytest.raises(ValueError, match="mask"):
            BlockShuffle(block_size=20).prepare(gappy_epochs())

    def test_trial_shuffle_carries_the_masks_with_the_trials(self):
        data = gappy_epochs()
        surrogate = TrialShuffle().generate_one(data, np.random.default_rng(4))
        # find the source trial of every channel and surrogate trial
        source = np.empty((data.n_trials, data.n_channels), dtype=int)
        for k in range(data.n_trials):
            for c in range(data.n_channels):
                match = [
                    j
                    for j in range(data.n_trials)
                    if np.array_equal(surrogate.data[k, c], data.data[j, c])
                ]
                assert len(match) == 1
                source[k, c] = match[0]
        expected_valid = np.logical_and.reduce([data.valid[source[:, c]] for c in range(3)])
        expected_predict = np.logical_and.reduce([data.predict[source[:, c]] for c in range(3)])
        np.testing.assert_array_equal(surrogate.valid, expected_valid)
        np.testing.assert_array_equal(surrogate.predict, expected_predict)

    def test_trial_shuffle_with_shared_masks_keeps_them(self):
        predict = np.tile(np.arange(120) >= 30, (6, 1))
        data = epoched(6, 120, predict=predict)
        surrogate = TrialShuffle().generate_one(data, np.random.default_rng(1))
        np.testing.assert_array_equal(surrogate.predict, predict)
        assert surrogate.valid is None

    @pytest.mark.filterwarnings("ignore::dirconn.DirconnWarning")
    def test_surrogate_test_runs_on_masked_data(self):
        data = gappy_epochs(n_trials=8, n_times=200)
        result = surrogate_test(GC(order=2), data, TrialShuffle(), n_surrogates=5, seed=0)
        assert result.observed.provenance.details["n_obs"] == int(
            row_mask(data.valid, data.predict, 2, data.n_times).sum()
        )


@pytest.mark.filterwarnings("ignore::dirconn.DirconnWarning")
class TestResampling:
    def test_bootstrap_carries_the_masks_with_the_trials(self):
        data = gappy_epochs()
        spy = MaskSpy()
        bootstrap(spy, data, n_boot=3, seed=0)
        observed, resamples = spy.seen[0], spy.seen[1:]
        assert observed is data
        for resample in resamples:
            for k in range(resample.n_trials):
                j = next(
                    j
                    for j in range(data.n_trials)
                    if np.array_equal(resample.data[k], data.data[j])
                )
                np.testing.assert_array_equal(resample.valid[k], data.valid[j])
                np.testing.assert_array_equal(resample.predict[k], data.predict[j])

    def test_block_bootstrap_carries_the_masks(self):
        valid = np.ones(400, dtype=bool)
        valid[130:135] = False
        data = continuous(400, valid=valid)
        spy = MaskSpy()
        bootstrap(spy, data, n_boot=4, seed=1, block_size=100)
        blocks = valid.reshape(4, 100)
        for resample in spy.seen[1:]:
            for k in range(resample.n_trials):
                j = next(
                    j
                    for j in range(4)
                    if np.array_equal(resample.data[k], data.data[0][:, j * 100 : (j + 1) * 100])
                )
                expected = blocks[j]
                got = resample.valid[k] if resample.valid is not None else np.ones(100, bool)
                np.testing.assert_array_equal(got, expected)

    def test_bootstrap_of_an_estimator_without_mask_support_raises(self):
        with pytest.raises(ValueError, match="mask"):
            bootstrap(TRGC(), gappy_epochs(), n_boot=2, seed=0)

    @pytest.mark.parametrize("paired", [False, True])
    def test_contrast_test_carries_the_masks(self, paired):
        a, b = gappy_epochs(seed=1), gappy_epochs(seed=2)
        spy = MaskSpy()
        contrast_test(spy, a, b, n_permutations=3, seed=0, paired=paired)
        pool = [(a.data[k], a.valid[k], a.predict[k]) for k in range(a.n_trials)] + [
            (b.data[k], b.valid[k], b.predict[k]) for k in range(b.n_trials)
        ]
        for relabelled in spy.seen[2:]:
            for k in range(relabelled.n_trials):
                data, valid, predict = next(
                    p for p in pool if np.array_equal(p[0], relabelled.data[k])
                )
                np.testing.assert_array_equal(relabelled.valid[k], valid)
                np.testing.assert_array_equal(relabelled.predict[k], predict)


def test_replace_keeps_the_masks():
    data = gappy_epochs()
    copy = dataclasses.replace(data, data=data.data * 2)
    np.testing.assert_array_equal(copy.valid, data.valid)


@pytest.mark.filterwarnings("ignore::dirconn.DirconnWarning")
@pytest.mark.parametrize("test", ["pdc", "rpdc"])
def test_asymptotic_tests_use_the_selected_rows(test):
    from dirconn import analytic_pdc_test, analytic_rpdc_test

    data = gappy_epochs(n_trials=8, n_times=300)
    rows = int(row_mask(data.valid, data.predict, 2, data.n_times).sum())
    if test == "pdc":
        result = analytic_pdc_test(PDC(order=2), data)
    else:
        result = analytic_rpdc_test(RPDC(order=2), data)
    assert result.n_obs == rows
