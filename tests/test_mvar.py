import numpy as np
import pytest
from statsmodels.tsa.vector_ar.var_model import VAR

from dirconn import TimeSeries
from dirconn.mvar import OrderSelection, VarFit, cv_splits, fit_var, lagged_design, select_order

COEFS = np.zeros((2, 2, 2))
COEFS[0] = [[0.5, 0.0], [0.4, 0.3]]  # lag 1, [target, source]: channel 0 drives channel 1
COEFS[1] = [[-0.2, 0.0], [0.0, -0.1]]


def simulate_var(coefs, n_times, *, n_trials=1, seed=0, burn=200):
    """Generate ``(n_trials, n, n_times)`` from ``x[t] = sum_k coefs[k-1] @ x[t-k] + e[t]``."""
    rng = np.random.default_rng(seed)
    order, n, _ = coefs.shape
    out = np.zeros((n_trials, n, n_times))
    for trial in range(n_trials):
        x = np.zeros((n, n_times + burn))
        noise = rng.standard_normal((n, n_times + burn))
        for t in range(order, n_times + burn):
            x[:, t] = noise[:, t] + sum(coefs[k] @ x[:, t - k - 1] for k in range(order))
        out[trial] = x[:, burn:]
    return out


def series(data, segmentation=None):
    if segmentation is None:
        segmentation = "continuous" if data.shape[0] == 1 else "epoched"
    return TimeSeries.from_array(data, sfreq=100.0, segmentation=segmentation)


class TestLaggedDesign:
    def test_rows_and_columns_follow_the_documented_layout(self):
        x = np.arange(2 * 2 * 6, dtype=float).reshape(2, 2, 6)  # 2 trials, 2 channels, 6 samples
        y, design = lagged_design(x, order=2)
        assert y.shape == (2 * 4, 2)
        assert design.shape == (2 * 4, 2 * 2)
        np.testing.assert_array_equal(y[:4], x[0, :, 2:].T)
        np.testing.assert_array_equal(design[:4, 0:2], x[0, :, 1:5].T)  # lag 1 block
        np.testing.assert_array_equal(design[:4, 2:4], x[0, :, 0:4].T)  # lag 2 block
        np.testing.assert_array_equal(y[4:], x[1, :, 2:].T)

    def test_no_row_mixes_samples_from_different_trials(self):
        t = np.arange(8, dtype=float)
        x = np.stack([100.0 * b + t + np.zeros((3, 1)) for b in range(3)])  # (trial, channel, time)
        y, design = lagged_design(x, order=3)
        trial_of_target = np.floor(y[:, 0] / 100.0)
        for column in range(design.shape[1]):
            np.testing.assert_array_equal(np.floor(design[:, column] / 100.0), trial_of_target)

    def test_start_moves_the_first_predicted_sample(self):
        x = np.random.default_rng(0).standard_normal((1, 2, 12))
        y, design = lagged_design(x, order=2, start=5)
        np.testing.assert_array_equal(y, x[0, :, 5:].T)
        np.testing.assert_array_equal(design[:, 0:2], x[0, :, 4:-1].T)
        np.testing.assert_array_equal(design[:, 2:4], x[0, :, 3:-2].T)

    @pytest.mark.parametrize(
        "kwargs", [dict(order=0), dict(order=2, start=1), dict(order=2, start=12), dict(order=12)]
    )
    def test_invalid_arguments_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            lagged_design(np.zeros((1, 2, 12)), **kwargs)


class TestFitVar:
    def test_recovers_the_coefficients_of_a_known_system(self):
        fit = fit_var(series(simulate_var(COEFS, 30000, seed=1)), order=2)
        np.testing.assert_allclose(fit.coefs, COEFS, atol=0.03)

    def test_matches_statsmodels_exactly(self):
        data = simulate_var(COEFS, 1500, seed=2)
        fit = fit_var(series(data), order=2)
        centred = data[0].T - data[0].T.mean(axis=0)
        reference = VAR(centred).fit(maxlags=2, ic=None, trend="n")
        np.testing.assert_allclose(fit.coefs, reference.coefs, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(fit.noise_cov, reference.sigma_u, rtol=1e-9)

    def test_pooled_trials_equal_the_manual_stacked_regression(self):
        data = simulate_var(COEFS, 60, n_trials=4, seed=3)
        fit = fit_var(series(data), order=2)
        centred = data - data.mean(axis=(0, 2), keepdims=True)
        rows_y, rows_x = [], []
        for trial in centred:
            for t in range(2, trial.shape[1]):
                rows_y.append(trial[:, t])
                rows_x.append(np.concatenate([trial[:, t - 1], trial[:, t - 2]]))
        beta, *_ = np.linalg.lstsq(np.array(rows_x), np.array(rows_y), rcond=None)
        expected = beta.T.reshape(2, 2, 2).transpose(1, 0, 2)
        np.testing.assert_allclose(fit.coefs, expected, rtol=1e-9, atol=1e-12)
        assert fit.n_obs == 4 * (60 - 2)

    def test_pooling_differs_from_naive_concatenation(self):
        data = simulate_var(COEFS, 30, n_trials=5, seed=4)
        pooled = fit_var(series(data), order=2)
        naive = fit_var(series(np.concatenate(list(data), axis=1)[None]), order=2)
        assert np.abs(pooled.coefs - naive.coefs).max() > 1e-6
        assert naive.n_obs == 5 * 30 - 2

    def test_a_single_epoch_gives_the_same_fit_as_the_continuous_record(self):
        data = simulate_var(COEFS, 400, seed=5)
        continuous = fit_var(series(data, "continuous"), order=2)
        epoched = fit_var(series(data, "epoched"), order=2)
        np.testing.assert_array_equal(continuous.coefs, epoched.coefs)

    def test_residuals_and_noise_covariance_are_consistent(self):
        fit = fit_var(series(simulate_var(COEFS, 500, n_trials=2, seed=6)), order=2)
        assert fit.residuals.shape == (2 * 498, 2)
        dof = fit.n_obs - 2 * 2
        np.testing.assert_allclose(fit.noise_cov, fit.residuals.T @ fit.residuals / dof)
        np.testing.assert_allclose(fit.noise_cov, fit.noise_cov.T)
        assert np.linalg.eigvalsh(fit.noise_cov).min() > 0

    def test_a_constant_offset_is_removed_by_demeaning(self):
        data = simulate_var(COEFS, 800, seed=7)
        base = fit_var(series(data), order=2)
        shifted = fit_var(series(data + np.array([[[50.0], [-30.0]]])), order=2)
        np.testing.assert_allclose(shifted.coefs, base.coefs, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(shifted.mean, base.mean + [50.0, -30.0])
        assert shifted.demeaned

    def test_demeaning_can_be_switched_off(self):
        data = simulate_var(COEFS, 800, seed=7) + np.array([[[50.0], [-30.0]]])
        raw = fit_var(series(data), order=2, demean=False)
        centred = fit_var(series(data), order=2)
        assert not raw.demeaned
        np.testing.assert_array_equal(raw.mean, np.zeros(2))
        assert np.abs(raw.coefs - centred.coefs).max() > 0.01

    def test_zero_ridge_is_ols_and_a_positive_ridge_shrinks(self):
        ts = series(simulate_var(COEFS, 300, seed=8))
        ols = fit_var(ts, order=2)
        assert np.array_equal(fit_var(ts, order=2, ridge=0.0).coefs, ols.coefs)
        ridged = fit_var(ts, order=2, ridge=0.5)
        assert np.linalg.norm(ridged.coefs) < np.linalg.norm(ols.coefs)
        assert ridged.ridge == 0.5

    def test_ridge_is_scale_free(self):
        data = simulate_var(COEFS, 300, seed=9)
        small = fit_var(series(data), order=2, ridge=0.2)
        large = fit_var(series(data * 10.0), order=2, ridge=0.2)
        np.testing.assert_allclose(small.coefs, large.coefs, rtol=1e-9, atol=1e-12)

    def test_negative_ridge_is_rejected(self):
        with pytest.raises(ValueError, match="ridge"):
            fit_var(series(simulate_var(COEFS, 100)), order=1, ridge=-0.1)

    def test_a_duplicated_channel_is_reported_as_rank_deficient(self):
        data = simulate_var(COEFS, 200, seed=10)
        duplicated = np.concatenate([data, data[:, :1]], axis=1)
        with pytest.raises(ValueError, match="rank"):
            fit_var(series(duplicated), order=2)

    def test_ridge_makes_a_rank_deficient_problem_solvable(self):
        data = simulate_var(COEFS, 200, seed=10)
        duplicated = np.concatenate([data, data[:, :1]], axis=1)
        fit = fit_var(series(duplicated), order=2, ridge=1e-3)
        assert np.isfinite(fit.coefs).all()

    def test_too_few_samples_are_rejected(self):
        data = np.random.default_rng(0).standard_normal((1, 3, 12))
        with pytest.raises(ValueError, match="samples"):
            fit_var(series(data), order=5)

    @pytest.mark.parametrize("order", [0, -1, 1.5, True])
    def test_the_order_must_be_a_positive_integer(self, order):
        with pytest.raises(ValueError, match="order"):
            fit_var(series(simulate_var(COEFS, 100)), order=order)

    def test_the_input_must_be_a_time_series(self):
        with pytest.raises(TypeError, match="TimeSeries"):
            fit_var(np.zeros((2, 100)), order=1)  # type: ignore[arg-type]

    def test_the_fit_carries_its_context_and_is_read_only(self):
        ts = TimeSeries.from_array(simulate_var(COEFS, 200)[0], sfreq=250.0, names=["a", "b"])
        fit = fit_var(ts, order=2)
        assert isinstance(fit, VarFit)
        assert (fit.names, fit.sfreq, fit.order, fit.n_channels) == (("a", "b"), 250.0, 2, 2)
        assert np.isfinite(fit.coefs).all() and np.isfinite(fit.residuals).all()
        with pytest.raises(ValueError, match="read-only"):
            fit.coefs[0, 0, 0] = 1.0
        with pytest.raises(ValueError, match="read-only"):
            fit.residuals[0, 0] = 1.0
        with pytest.raises(ValueError, match="read-only"):
            fit.regressor_cov[0, 0] = 1.0

    @pytest.mark.parametrize("n_trials", [1, 3])
    @pytest.mark.parametrize("demean", [True, False])
    def test_regressor_covariance_is_the_gram_matrix_of_the_lagged_design(self, n_trials, demean):
        data = simulate_var(COEFS, 80, n_trials=n_trials, seed=8) + 0.5
        fit = fit_var(series(data), order=3, demean=demean)
        x = data - data.mean(axis=(0, 2), keepdims=True) if demean else data
        _, design = lagged_design(x, order=3)
        assert fit.regressor_cov.shape == (3 * 2, 3 * 2)
        np.testing.assert_allclose(fit.regressor_cov, design.T @ design / fit.n_obs, rtol=1e-12)

    def test_regressor_covariance_does_not_depend_on_the_ridge(self):
        ts = series(simulate_var(COEFS, 300, seed=9))
        plain = fit_var(ts, order=2)
        penalised = fit_var(ts, order=2, ridge=0.5)
        np.testing.assert_array_equal(plain.regressor_cov, penalised.regressor_cov)


class TestSelectOrder:
    def reference_criteria(self, data, max_order):
        centred = data[0].T - data[0].T.mean(axis=0)
        return VAR(centred).select_order(maxlags=max_order, trend="n").ics

    def test_criteria_match_statsmodels_exactly(self):
        data = simulate_var(COEFS, 1500, seed=11)
        selection = select_order(series(data), max_order=5)
        reference = self.reference_criteria(data, 5)
        np.testing.assert_allclose(selection.aic, reference["aic"], rtol=1e-9)
        np.testing.assert_allclose(selection.bic, reference["bic"], rtol=1e-9)
        np.testing.assert_allclose(selection.hqc, reference["hqic"], rtol=1e-9)

    def test_all_orders_share_one_observation_window(self):
        data = simulate_var(COEFS, 200, n_trials=3, seed=12)
        selection = select_order(series(data), max_order=6)
        assert selection.n_obs == 3 * (200 - 6)
        np.testing.assert_array_equal(selection.orders, np.arange(1, 7))

    def test_bic_recovers_the_true_order(self):
        selection = select_order(series(simulate_var(COEFS, 8000, seed=13)), max_order=6)
        assert isinstance(selection, OrderSelection)
        assert selection.criterion == "bic"
        assert selection.order == 2

    def test_the_selected_order_is_the_minimum_of_the_chosen_criterion(self):
        ts = series(simulate_var(COEFS, 3000, seed=14))
        for criterion in ("aic", "bic", "hqc"):
            selection = select_order(ts, max_order=6, criterion=criterion)
            curve = getattr(selection, criterion)
            assert selection.order == int(np.argmin(curve)) + 1

    def test_aic_never_selects_a_smaller_model_than_bic(self):
        ts = series(simulate_var(COEFS, 3000, seed=15))
        assert (
            select_order(ts, max_order=8, criterion="aic").order
            >= select_order(ts, max_order=8, criterion="bic").order
        )

    def test_pooled_trials_are_supported(self):
        selection = select_order(series(simulate_var(COEFS, 300, n_trials=6, seed=16)), max_order=5)
        assert selection.order == 2

    def test_ridge_is_passed_through_to_the_fits(self):
        ts = series(simulate_var(COEFS, 500, seed=17))
        assert not np.allclose(
            select_order(ts, max_order=4).bic, select_order(ts, max_order=4, ridge=0.5).bic
        )

    def test_unknown_criterion_is_rejected(self):
        with pytest.raises(ValueError, match="criterion"):
            select_order(series(simulate_var(COEFS, 300)), max_order=3, criterion="fpe")

    def test_too_large_a_maximum_order_is_rejected(self):
        with pytest.raises(ValueError, match="samples"):
            select_order(series(simulate_var(COEFS, 30)), max_order=20)

    @pytest.mark.parametrize("max_order", [0, 1.5])
    def test_the_maximum_order_must_be_a_positive_integer(self, max_order):
        with pytest.raises(ValueError, match="max_order"):
            select_order(series(simulate_var(COEFS, 300)), max_order=max_order)


def manual_cv(data, max_order, n_folds, scheme):
    """Independent cross-validation of the normalised prediction error, written with loops."""
    x = data - data.mean(axis=(0, 2), keepdims=True)
    n_trials, n_channels, n_times = x.shape
    rows_per_trial = n_times - max_order
    samples = []  # (trial, row inside the trial, target, [lag 1 .. lag max_order])
    for b in range(n_trials):
        for t in range(max_order, n_times):
            lags = [x[b, :, t - k] for k in range(1, max_order + 1)]
            samples.append((b, t - max_order, x[b, :, t], lags))
    total = len(samples)
    variance = np.mean(np.array([s[2] for s in samples]) ** 2, axis=0)

    if scheme == "trials":
        n_splits = min(n_folds, n_trials)

        def fold_of(sample):
            return sample[0] % n_splits

    else:
        n_splits = n_folds
        edges = np.array_split(np.arange(rows_per_trial), n_folds)
        block_of = np.empty(rows_per_trial, dtype=int)
        for index, edge in enumerate(edges):
            block_of[edge] = index

        def fold_of(sample):
            return block_of[sample[1]]

    curve = []
    for order in range(1, max_order + 1):
        sse = np.zeros(n_channels)
        for fold in range(n_splits):
            train, test = [], []
            for sample in samples:
                if fold_of(sample) == fold:
                    test.append(sample)
                elif scheme == "trials":
                    train.append(sample)
                else:
                    first, last = edges[fold][0], edges[fold][-1] + 1
                    if sample[1] < first - max_order or sample[1] > last + max_order - 1:
                        train.append(sample)
            design = np.array([np.concatenate(s[3][:order]) for s in train])
            beta = np.linalg.lstsq(design, np.array([s[2] for s in train]), rcond=None)[0]
            held = np.array([np.concatenate(s[3][:order]) for s in test])
            error = np.array([s[2] for s in test]) - held @ beta
            sse += (error**2).sum(axis=0)
        curve.append(np.mean(sse / total / variance))
    return np.array(curve)


class TestCrossValidation:
    def test_trial_folds_match_an_independent_implementation(self):
        data = simulate_var(COEFS, 60, n_trials=7, seed=21)
        selection = select_order(series(data), max_order=4, criterion="cv", n_folds=3)
        np.testing.assert_allclose(selection.cv, manual_cv(data, 4, 3, "trials"), rtol=1e-9)
        assert selection.cv_scheme == "trials"
        assert selection.n_folds == 3

    def test_block_folds_match_an_independent_implementation_for_one_trial(self):
        data = simulate_var(COEFS, 300, seed=22)
        selection = select_order(series(data), max_order=4, criterion="cv", n_folds=4)
        np.testing.assert_allclose(selection.cv, manual_cv(data, 4, 4, "blocks"), rtol=1e-9)
        assert selection.cv_scheme == "blocks"
        assert selection.n_folds == 4

    def test_two_or_three_trials_use_blocks_inside_every_trial(self):
        for n_trials in (2, 3):
            data = simulate_var(COEFS, 200, n_trials=n_trials, seed=23)
            selection = select_order(series(data), max_order=3, criterion="cv", n_folds=4)
            assert selection.cv_scheme == "blocks"
            np.testing.assert_allclose(selection.cv, manual_cv(data, 3, 4, "blocks"), rtol=1e-9)

    def test_the_number_of_trial_folds_never_exceeds_the_number_of_trials(self):
        data = simulate_var(COEFS, 80, n_trials=4, seed=24)
        selection = select_order(series(data), max_order=3, criterion="cv", n_folds=5)
        assert selection.cv_scheme == "trials"
        assert selection.n_folds == 4

    @pytest.mark.parametrize("n_trials, n_times", [(1, 6000), (8, 750)])
    def test_the_curve_drops_up_to_the_true_order_and_then_flattens(self, n_trials, n_times):
        # Cross-validation has no penalty term, so beyond the true order the curve is nearly flat
        # and the minimum can sit a lag or more later; the structure is what must hold. Thresholds
        # come from the observed curves (drop 0.014 to 0.022, plateau spread up to 0.0034).
        orders = []
        for seed in range(5):
            data = simulate_var(COEFS, n_times, n_trials=n_trials, seed=40 + seed)
            selection = select_order(series(data), max_order=6, criterion="cv")
            curve = selection.cv
            assert selection.criterion == "cv"
            assert curve[0] - curve[1] > 0.01  # the second lag matters
            assert selection.order >= 2
            assert curve[1:].max() - curve.min() < 0.01  # and the plateau is flat
            orders.append(selection.order)
        assert np.median(orders) == 2

    def test_white_noise_gets_the_smallest_order_for_most_seeds(self):
        orders = [
            select_order(
                series(np.random.default_rng(seed).standard_normal((1, 2, 4000))),
                max_order=5,
                criterion="cv",
            ).order
            for seed in range(5)
        ]
        assert np.median(orders) == 1

    def test_the_curve_is_a_normalised_error_below_one_for_a_predictable_system(self):
        selection = select_order(
            series(simulate_var(COEFS, 4000, seed=28)), max_order=4, criterion="cv"
        )
        assert selection.cv.shape == (4,)
        assert np.all(selection.cv > 0)
        assert selection.cv.min() < 1.0

    def test_cv_is_scale_free(self):
        data = simulate_var(COEFS, 800, seed=29)
        small = select_order(series(data), max_order=3, criterion="cv").cv
        large = select_order(series(data * 25.0), max_order=3, criterion="cv").cv
        np.testing.assert_allclose(small, large, rtol=1e-9)

    def test_the_default_criterion_does_not_compute_cv(self):
        selection = select_order(series(simulate_var(COEFS, 500, seed=30)), max_order=3)
        assert selection.cv is None and selection.cv_scheme is None and selection.n_folds is None

    def test_information_criteria_are_still_reported_with_cv(self):
        ts = series(simulate_var(COEFS, 800, seed=31))
        with_cv = select_order(ts, max_order=3, criterion="cv")
        without = select_order(ts, max_order=3)
        np.testing.assert_array_equal(with_cv.bic, without.bic)

    def test_ridge_is_applied_inside_the_folds(self):
        ts = series(simulate_var(COEFS, 800, seed=32))
        plain = select_order(ts, max_order=3, criterion="cv").cv
        ridged = select_order(ts, max_order=3, criterion="cv", ridge=1.0).cv
        assert not np.allclose(plain, ridged)

    @pytest.mark.parametrize("n_folds", [1, 0, 2.5, True])
    def test_the_number_of_folds_must_be_an_integer_of_at_least_two(self, n_folds):
        with pytest.raises(ValueError, match="n_folds"):
            select_order(
                series(simulate_var(COEFS, 300)), max_order=2, criterion="cv", n_folds=n_folds
            )

    def test_too_little_data_for_the_folds_is_an_error(self):
        data = simulate_var(COEFS, 40, seed=33)
        with pytest.raises(ValueError, match="cross-validation"):
            select_order(series(data), max_order=10, criterion="cv", n_folds=5)


class TestCvSplits:
    def test_trial_splits_hold_out_whole_trials(self):
        scheme, splits = cv_splits(n_trials=6, rows_per_trial=10, n_folds=3, gap=2)
        assert scheme == "trials"
        assert len(splits) == 3
        all_rows = np.arange(60)
        tested = np.concatenate([test for _, test in splits])
        np.testing.assert_array_equal(np.sort(tested), all_rows)
        for train, test in splits:
            assert len(np.intersect1d(train, test)) == 0
            assert len(train) + len(test) == 60
            held_trials = np.unique(test // 10)
            assert len(test) == 10 * len(held_trials)
            assert not np.isin(train // 10, held_trials).any()

    def test_block_splits_leave_a_gap_around_the_held_out_block(self):
        scheme, splits = cv_splits(n_trials=2, rows_per_trial=20, n_folds=4, gap=3)
        assert scheme == "blocks"
        tested = np.concatenate([test for _, test in splits])
        np.testing.assert_array_equal(np.sort(tested), np.arange(40))
        for train, test in splits:
            assert len(np.intersect1d(train, test)) == 0
            for trial in (0, 1):
                held = test[test // 20 == trial] % 20
                kept = train[train // 20 == trial] % 20
                distance = np.abs(kept[:, None] - held[None, :]).min()
                assert distance >= 3 + 1

    def test_blocks_are_contiguous_and_cover_the_trial(self):
        _, splits = cv_splits(n_trials=1, rows_per_trial=23, n_folds=5, gap=2)
        for _, test in splits:
            assert np.all(np.diff(test) == 1)
        assert sum(len(test) for _, test in splits) == 23

    def test_more_block_folds_than_rows_is_rejected(self):
        with pytest.raises(ValueError, match="rows"):
            cv_splits(n_trials=1, rows_per_trial=3, n_folds=5, gap=1)

    def test_a_negative_gap_is_rejected(self):
        with pytest.raises(ValueError, match="gap"):
            cv_splits(n_trials=1, rows_per_trial=30, n_folds=3, gap=-1)
