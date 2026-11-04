"""Tests for the time-varying MVAR core (Kalman, RLS and DEKF recursions)."""

import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from dirconn import DirconnWarning, TimeSeries
from dirconn.mvar import OrderSelection
from dirconn.tvmvar import TvarFit, fit_tvar, tvar_checks

SFREQ = 100.0
FIXTURES = Path(__file__).parent / "data" / "tvar_neurrow.npz"


def simulate_var1(coefs_at, n_times, *, seed=0, noise=0.3):
    """Generate a VAR(1) whose coefficient matrix is ``coefs_at(t)``; shape (n_nodes, n_times)."""
    rng = np.random.default_rng(seed)
    n_nodes = coefs_at(0).shape[0]
    x = np.zeros((n_nodes, n_times))
    for t in range(1, n_times):
        x[:, t] = coefs_at(t) @ x[:, t - 1] + noise * rng.standard_normal(n_nodes)
    return x


def stationary_a():
    return np.array([[0.6, 0.0], [0.7, 0.4]])


def switched_a():
    """Same nodes, coupling reversed: 1 -> 0 instead of 0 -> 1."""
    return np.array([[0.6, 0.7], [0.0, 0.4]])


def series(x, *, tmin=0.0):
    """Wrap ``(n_nodes, n_times)`` or ``(n_trials, n_nodes, n_times)`` as a TimeSeries."""
    x = np.asarray(x, dtype=float)
    segmentation = "continuous" if x.ndim == 2 else "epoched"
    return TimeSeries.from_array(x, sfreq=SFREQ, segmentation=segmentation, tmin=tmin)


def fit(x, **kwargs):
    """``fit_tvar`` with the settings the recovery tests share: order 1, no standardising."""
    kwargs.setdefault("order", 1)
    kwargs.setdefault("standardize", None)
    return fit_tvar(series(x), **kwargs)


# --------------------------------------------------------------------------
# Recovery of a known trajectory
# --------------------------------------------------------------------------


def test_recovers_stationary_coefficients():
    truth = stationary_a()
    x = simulate_var1(lambda t: truth, 6000, seed=1)
    result = fit(x, forgetting=0.999)
    estimated = result.coefs[len(result.coefs) // 2 :, 0].mean(axis=0)
    np.testing.assert_allclose(estimated, truth, atol=0.08)


def test_tracks_a_mid_recording_switch():
    n_times = 8000
    switch = n_times // 2
    before, after = stationary_a(), switched_a()
    x = simulate_var1(lambda t: before if t < switch else after, n_times, seed=2)

    result = fit(x, forgetting=0.99)
    early = result.sample_index < switch - 500
    late = result.sample_index > switch + 1500

    # a[1, 0] carries 0 -> 1 and is present only before the switch; a[0, 1] carries 1 -> 0 after.
    assert result.coefs[early, 0, 1, 0].mean() > 0.4
    assert abs(result.coefs[early, 0, 0, 1].mean()) < 0.2
    assert result.coefs[late, 0, 0, 1].mean() > 0.4
    assert abs(result.coefs[late, 0, 1, 0].mean()) < 0.2


def _first_stable_reversal(result, *, margin=0.2, hold=200):
    """First time ``0 -> 1`` overtakes ``1 -> 0`` by ``margin`` and stays there.

    A bare zero-crossing is fooled by the burn-in transient, when both coefficients are still near
    zero; the margin and the hold requirement select the genuine, sustained reversal.
    """
    difference = result.coefs[:, 0, 0, 1] - result.coefs[:, 0, 1, 0]
    hold_samples = max(1, int(hold / max(1, np.diff(result.sample_index).min())))
    for i in range(difference.size - hold_samples):
        if difference[i] > margin and np.all(difference[i : i + hold_samples] > 0):
            return result.sample_index[i]
    return None


def test_switch_is_located_within_a_tolerance_set_by_forgetting():
    """The lag scales with the memory ``1 / (1 - forgetting)``, not with the seed."""
    n_times = 8000
    switch = n_times // 2
    forgetting = 0.99
    x = simulate_var1(lambda t: stationary_a() if t < switch else switched_a(), n_times, seed=3)
    result = fit(x, forgetting=forgetting)

    detected = _first_stable_reversal(result)
    assert detected is not None, "the dominant direction never reversed"
    assert detected >= switch - 500, "reversal detected before the switch happened"
    assert abs(detected - switch) < 5 / (1.0 - forgetting)


def test_a_faster_forgetting_factor_detects_the_switch_sooner():
    n_times = 8000
    switch = n_times // 2
    x = simulate_var1(lambda t: stationary_a() if t < switch else switched_a(), n_times, seed=4)

    def detection_lag(forgetting):
        detected = _first_stable_reversal(fit(x, forgetting=forgetting))
        assert detected is not None
        return detected - switch

    assert detection_lag(0.98) <= detection_lag(0.999)


# --------------------------------------------------------------------------
# Shapes, axes, subsampling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2, 3])
def test_coefficient_layout_is_the_var_layout(order):
    """(time, lag, target, source) per slice: every time slice is a ``VarFit.coefs``."""
    x = simulate_var1(lambda t: stationary_a(), 2000, seed=5)
    result = fit(x, order=order)
    assert result.coefs.shape[1:] == (order, 2, 2)
    assert result.order == order
    assert result.n_nodes == 2


def test_time_step_subsamples_the_returned_trajectory():
    x = simulate_var1(lambda t: stationary_a(), 2000, seed=6)
    dense = fit(x, time_step=1)
    sparse = fit(x, time_step=10)
    assert sparse.n_times == pytest.approx(dense.n_times / 10, rel=0.02)
    np.testing.assert_array_equal(sparse.sample_index, dense.sample_index[::10])
    np.testing.assert_array_equal(sparse.coefs, dense.coefs[::10])


def test_without_a_time_step_about_200_points_are_returned():
    x = simulate_var1(lambda t: stationary_a(), 8000, seed=6)
    result = fit(x, time_step=None)
    assert 150 <= result.n_times <= 200
    assert result.time_step == int(np.diff(result.sample_index)[0]) > 1
    short = fit(simulate_var1(lambda t: stationary_a(), 150, seed=6), time_step=None, burn_in=20)
    assert short.time_step == 1 and short.n_times == 130


def test_sample_index_and_times_are_increasing_and_inside_the_recording():
    x = simulate_var1(lambda t: stationary_a(), 1500, seed=7)
    result = fit_tvar(series(x, tmin=-0.5), order=1, standardize=None)
    assert np.all(np.diff(result.sample_index) > 0)
    assert result.sample_index[0] >= result.burn_in
    assert result.sample_index[-1] == 1499
    np.testing.assert_allclose(result.times, -0.5 + result.sample_index / SFREQ)


def test_noise_covariance_shape_and_positivity():
    x = simulate_var1(lambda t: stationary_a(), 2000, seed=8)
    result = fit(x, update="dual")
    assert result.noise_cov.shape == (result.n_times, 2, 2)
    assert np.all(np.diagonal(result.noise_cov, axis1=1, axis2=2) >= 0)


def test_the_arrays_are_read_only():
    result = fit(simulate_var1(lambda t: stationary_a(), 600, seed=8))
    for name in ("coefs", "noise_cov", "sample_index", "times"):
        assert not getattr(result, name).flags.writeable


# --------------------------------------------------------------------------
# Failures raise; nothing is replaced silently
# --------------------------------------------------------------------------


def test_a_flat_channel_cannot_be_standardised():
    x = simulate_var1(lambda t: stationary_a(), 1200, seed=9)
    x = np.vstack([x, np.zeros((1, x.shape[1]))])
    with pytest.raises(ValueError, match="zero variance"):
        fit_tvar(series(x), order=1, standardize="zscore")


def test_a_failing_seed_fit_raises_instead_of_starting_from_zero():
    """A flat channel makes the seed regression rank deficient: no silent zero start."""
    x = simulate_var1(lambda t: stationary_a(), 1200, seed=9)
    x = np.vstack([x, np.zeros((1, x.shape[1]))])
    with pytest.raises(ValueError, match="rank"):
        fit(x, init="var")


def test_exactly_collinear_channels_raise_with_the_seed_fit():
    base = simulate_var1(lambda t: stationary_a(), 1200, seed=10)
    with pytest.raises(ValueError, match="rank"):
        fit(np.vstack([base, base[0:1]]), init="var")


@pytest.mark.parametrize("update", ["dual", "rls", "dekf"])
def test_an_explicit_zero_start_stays_finite_for_degenerate_input(update):
    """With ``init="zeros"`` (an explicit choice) flat and duplicated channels do not poison it."""
    base = simulate_var1(lambda t: stationary_a(), 1200, seed=10)
    for x in (np.vstack([base, np.zeros((1, base.shape[1]))]), np.vstack([base, base[0:1]])):
        result = fit(x, update=update, init="zeros")
        assert np.all(np.isfinite(result.coefs))
        assert np.all(np.isfinite(result.noise_cov))
        assert result.init == "zeros"


@pytest.mark.parametrize("update", ["dual", "rls", "dekf"])
def test_is_deterministic(update):
    x = simulate_var1(lambda t: stationary_a(), 1500, seed=11)
    first = fit(x, update=update)
    second = fit(x, update=update)
    np.testing.assert_array_equal(first.coefs, second.coefs)


def test_state_covariance_stays_well_behaved_over_a_long_run():
    """Joseph-form updates plus symmetrisation; the naive form drifts and goes indefinite."""
    x = simulate_var1(lambda t: stationary_a(), 20000, seed=12)
    result = fit(x, order=2)
    assert np.all(np.isfinite(result.coefs))
    assert result.unstable_fraction < 0.5
    assert result.n_skipped_updates == 0


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------


def test_rls_and_dual_agree_on_a_stationary_signal():
    x = simulate_var1(lambda t: stationary_a(), 6000, seed=13)
    dual = fit(x, update="dual", forgetting=0.999)
    rls = fit(x, update="rls", forgetting=0.999)
    half = dual.n_times // 2
    np.testing.assert_allclose(
        dual.coefs[half:, 0].mean(axis=0), rls.coefs[half:, 0].mean(axis=0), atol=0.08
    )


def test_dual_and_rls_have_identical_coefficients_and_differ_only_in_the_covariance():
    """The coefficient filter reads only the diagonal of the innovation covariance, which both
    recursions update identically; ``dual`` also fills the off-diagonal entries."""
    x = np.vstack(
        [
            simulate_var1(lambda t: stationary_a(), 3000, seed=40),
            simulate_var1(lambda t: stationary_a(), 3000, seed=41)[:1],
        ]
    )
    dual = fit(x, order=2, update="dual", forgetting=0.97)
    rls = fit(x, order=2, update="rls", forgetting=0.97)
    np.testing.assert_array_equal(dual.coefs, rls.coefs)
    np.testing.assert_array_equal(
        np.diagonal(dual.noise_cov, axis1=1, axis2=2), np.diagonal(rls.noise_cov, axis1=1, axis2=2)
    )
    off = ~np.eye(3, dtype=bool)
    assert np.abs(dual.noise_cov[:, off]).max() > 0.01
    assert np.abs(rls.noise_cov[:, off]).max() == 0.0


def test_standardisation_is_recorded_and_applied_to_all_trials_at_once():
    x = 5.0 * simulate_var1(lambda t: stationary_a(), 1500, seed=14) + 2.0
    zscored = fit_tvar(series(x), order=1)
    assert zscored.standardize == "zscore"
    np.testing.assert_allclose(zscored.scale, x.std(axis=1))
    assert fit_tvar(series(x), order=1, standardize=None).standardize is None
    assert fit_tvar(series(x), order=1, standardize="demean").standardize == "demean"


def test_the_settings_actually_used_are_recorded():
    x = simulate_var1(lambda t: stationary_a(), 2000, seed=15)
    result = fit(x, forgetting=0.98, update="dual")
    assert result.update == "dual"
    assert result.forgetting == 0.98
    assert result.process_noise == pytest.approx(1 / 0.98 - 1)
    assert result.init == "var"
    assert result.init_samples == min(2000 // 4, 20 * 2)
    assert result.burn_in == max(result.init_samples, 10)
    assert result.measurement_noise is None
    dekf = fit(x, update="dekf")
    np.testing.assert_allclose(dekf.measurement_noise, 0.1 * x.var(axis=1))


def test_the_memory_is_measured_and_much_shorter_than_one_over_one_minus_forgetting():
    """The forgetting factor only sets the process noise 1 / lambda - 1: a
    recursive least-squares filter would remember 1 / (1 - lambda) samples, this one does not."""
    x = simulate_var1(lambda t: stationary_a(), 6000, seed=42)
    memory = {lam: fit(x, forgetting=lam).memory_samples for lam in (0.95, 0.99, 0.999, 0.9999)}
    values = list(memory.values())
    assert values == sorted(values)  # a larger factor remembers longer
    assert all(np.isfinite(v) and v > 1 for v in values)
    for lam in (0.99, 0.999, 0.9999):
        assert memory[lam] < 0.1 / (
            1.0 - lam
        )  # measured: 5, 15 and 43 samples against 100, 1000, 10000
    assert fit(x, forgetting=0.99, update="dekf").memory_samples == pytest.approx(
        memory[0.99], rel=0.3
    )


def test_the_measured_memory_matches_how_fast_a_switch_is_followed():
    n_times, switch = 6000, 3000
    x = simulate_var1(lambda t: stationary_a() if t < switch else switched_a(), n_times, seed=43)
    for forgetting in (0.999, 0.99999):
        result = fit(x, forgetting=forgetting)
        late = result.coefs[:, 0, 0, 1]  # 1 -> 0 exists only after the switch
        before = late[(result.sample_index > 1500) & (result.sample_index < switch)].mean()
        after = late[result.sample_index > n_times - 1000].mean()
        progress = (late - before) / (after - before)
        crossed = result.sample_index[(result.sample_index >= switch) & (progress >= 0.5)]
        half_way = crossed[0] - switch
        assert half_way <= 2.5 * result.memory_samples + 5  # measured: 11 and 193 for 15 and 126


def test_details_are_json_serialisable():
    x = simulate_var1(lambda t: stationary_a(), 1000, seed=15)
    details = fit(x, update="dekf").details()
    assert json.loads(json.dumps(details)) == details
    for key in (
        "update",
        "forgetting",
        "init",
        "burn_in",
        "n_trials",
        "unstable_fraction",
        "memory_samples",
        "sfreq",
    ):
        assert key in details


# --------------------------------------------------------------------------
# Order
# --------------------------------------------------------------------------


def test_the_default_order_comes_from_bic_on_a_stationary_fit_and_is_capped_at_5():
    x = simulate_var1(lambda t: stationary_a(), 3000, seed=21)
    result = fit_tvar(series(x), standardize=None)
    assert isinstance(result.order_selection, OrderSelection)
    assert result.order_selection.criterion == "bic"
    assert result.order_selection.max_order == 5
    assert result.order == result.order_selection.order
    assert 1 <= result.order <= 5
    assert result.coefs.shape[1] == result.order


def test_a_fixed_order_skips_the_selection():
    x = simulate_var1(lambda t: stationary_a(), 1000, seed=21)
    result = fit(x, order=2)
    assert result.order == 2
    assert result.order_selection is None


# --------------------------------------------------------------------------
# Several trials
# --------------------------------------------------------------------------


def _trials(n_trials, n_times=600, *, switch=None, seed=30):
    out = []
    for k in range(n_trials):
        if switch is None:
            out.append(simulate_var1(lambda t: stationary_a(), n_times, seed=seed + k))
        else:
            out.append(
                simulate_var1(
                    lambda t: stationary_a() if t < switch else switched_a(), n_times, seed=seed + k
                )
            )
    return np.stack(out)


@pytest.mark.parametrize("update", ["dual", "rls", "dekf"])
def test_trials_are_filtered_separately_and_their_coefficients_averaged(update):
    data = _trials(3)
    kwargs = dict(order=2, standardize=None, init="zeros", update=update, burn_in=60)
    if update == "dekf":
        kwargs["measurement_noise"] = 0.05  # the default is a fraction of the variance of the data
    together = fit_tvar(series(data), **kwargs)
    singles = [fit_tvar(series(data[k]), **kwargs) for k in range(3)]
    np.testing.assert_allclose(
        together.coefs, np.mean([s.coefs for s in singles], axis=0), atol=1e-10
    )
    np.testing.assert_allclose(
        together.noise_cov, np.mean([s.noise_cov for s in singles], axis=0), atol=1e-10
    )
    assert together.n_trials == 3
    assert singles[0].n_trials == 1


def test_a_time_locked_switch_is_tracked_across_trials():
    data = _trials(8, n_times=500, switch=250)
    result = fit_tvar(series(data), order=1, standardize=None, forgetting=0.97)
    early = result.sample_index < 220
    late = result.sample_index > 400
    assert result.coefs[early, 0, 1, 0].mean() > 0.4
    assert result.coefs[late, 0, 0, 1].mean() > 0.4


def test_all_trials_share_one_seed_fit():
    data = _trials(4, n_times=400)
    result = fit_tvar(series(data), order=1, standardize=None)
    # The seed regression is pooled over trials: its row count is trials x (init - order).
    assert result.details()["seed_rows"] == 4 * (result.init_samples - 1)


# --------------------------------------------------------------------------
# DEKF (dual extended Kalman filter)
# --------------------------------------------------------------------------


def test_dekf_recovers_stationary_coefficients():
    truth = stationary_a()
    x = simulate_var1(lambda t: truth, 6000, seed=1)
    result = fit(x, update="dekf", forgetting=0.999)
    estimated = result.coefs[len(result.coefs) // 2 :, 0].mean(axis=0)
    np.testing.assert_allclose(estimated, truth, atol=0.1)


def test_dekf_with_zero_measurement_noise_matches_dual():
    """With r = 0 the state filter returns the observations exactly, so DEKF reduces to dual.

    Any indexing or ordering mistake in the DEKF branch breaks this identity.
    """
    x = simulate_var1(lambda t: stationary_a(), 3000, seed=17)
    dual = fit(x, order=2, update="dual")
    dekf = fit(x, order=2, update="dekf", measurement_noise=0.0)
    np.testing.assert_allclose(dekf.coefs, dual.coefs, atol=1e-8)
    np.testing.assert_allclose(dekf.noise_cov, dual.noise_cov, atol=1e-8)


def test_dekf_beats_dual_under_observation_noise():
    """Observation noise biases the plain coefficient filter toward zero (regression attenuation);
    the denoised regressors of the state filter remove most of that bias."""
    truth = stationary_a()
    x = simulate_var1(lambda t: truth, 8000, seed=18)
    x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    y = x + 0.5 * np.random.default_rng(19).standard_normal(x.shape)  # var(n) / var(x) = 0.25

    def late_mse(update, **kwargs):
        result = fit(y, update=update, forgetting=0.999, **kwargs)
        estimated = result.coefs[len(result.coefs) // 2 :, 0].mean(axis=0)
        return float(((estimated - truth) ** 2).mean())

    assert late_mse("dekf", measurement_noise=0.25) < 0.8 * late_mse("dual")


def test_dekf_tracks_a_mid_recording_switch():
    n_times = 8000
    switch = n_times // 2
    x = simulate_var1(lambda t: stationary_a() if t < switch else switched_a(), n_times, seed=2)
    result = fit(x, update="dekf", forgetting=0.99)
    early = result.sample_index < switch - 500
    late = result.sample_index > switch + 1500
    assert result.coefs[early, 0, 1, 0].mean() > 0.4
    assert abs(result.coefs[early, 0, 0, 1].mean()) < 0.2
    assert result.coefs[late, 0, 0, 1].mean() > 0.4
    assert abs(result.coefs[late, 0, 1, 0].mean()) < 0.2


def test_dekf_noise_cov_shape_and_positivity():
    x = simulate_var1(lambda t: stationary_a(), 2000, seed=8)
    result = fit(x, update="dekf")
    assert result.noise_cov.shape == (result.n_times, 2, 2)
    assert np.all(np.diagonal(result.noise_cov, axis1=1, axis2=2) >= 0)


def _reference_dekf(y, *, order, forgetting, measurement_noise, obs_noise, state_cov_init, burn_in):
    """Slow full-matrix DEKF written straight from the Wan and Nelson equations.

    The weight filter carries one joint state of length ``p * N**2`` with the Kronecker
    observation matrix ``I_N (x) phi^T`` and a full vector observation update; the state filter
    uses an explicit companion matrix and plain matrix inverses. None of the production code's
    decoupling or batching is reused, so agreement validates them. Zero start, every sample kept.
    """
    n_nodes, n_times = y.shape
    state_size = order * n_nodes
    lam = float(forgetting)
    q_w = 1.0 / lam - 1.0

    weights = np.zeros(n_nodes * state_size)
    weight_cov = np.eye(n_nodes * state_size) * state_cov_init
    noise = np.eye(n_nodes)
    obs_variance = np.full(n_nodes, float(obs_noise))
    r = np.full(n_nodes, float(measurement_noise))

    z = np.concatenate([y[:, order - 1 - lag] for lag in range(order)])
    z_cov = np.diag(np.tile(r, order))
    companion = np.zeros((state_size, state_size))
    if order > 1:
        companion[n_nodes:, : state_size - n_nodes] = np.eye(state_size - n_nodes)
    h_state = np.zeros((n_nodes, state_size))
    h_state[:, :n_nodes] = np.eye(n_nodes)

    kept = []
    for t in range(order, n_times):
        phi = z.copy()
        h_w = np.kron(np.eye(n_nodes), phi[None, :])
        weight_cov = weight_cov + q_w * np.eye(weight_cov.shape[0])
        innovation = y[:, t] - h_w @ weights
        s_w = h_w @ weight_cov @ h_w.T + np.diag(obs_variance)
        k_w = weight_cov @ h_w.T @ np.linalg.inv(s_w)
        weights = weights + k_w @ innovation
        factor_w = np.eye(weight_cov.shape[0]) - k_w @ h_w
        weight_cov = factor_w @ weight_cov @ factor_w.T + k_w @ np.diag(obs_variance) @ k_w.T
        weight_cov = 0.5 * (weight_cov + weight_cov.T)

        noise = lam * noise + (1.0 - lam) * np.outer(innovation, innovation)
        obs_variance = np.maximum(np.diag(noise).copy(), np.finfo(float).tiny)

        driving = np.maximum(obs_variance - r, np.finfo(float).tiny)
        companion[:n_nodes] = weights.reshape(n_nodes, state_size)
        z_pred = companion @ z
        cov_pred = companion @ z_cov @ companion.T
        cov_pred[:n_nodes, :n_nodes] += np.diag(driving)
        s_x = h_state @ cov_pred @ h_state.T + np.diag(r)
        k_x = cov_pred @ h_state.T @ np.linalg.inv(s_x)
        z = z_pred + k_x @ (y[:, t] - h_state @ z_pred)
        factor_x = np.eye(state_size) - k_x @ h_state
        z_cov = factor_x @ cov_pred @ factor_x.T + k_x @ np.diag(r) @ k_x.T
        z_cov = 0.5 * (z_cov + z_cov.T)

        if t < burn_in:
            continue
        rows = weights.reshape(n_nodes, state_size)
        kept.append(
            np.stack([rows[:, lag * n_nodes : (lag + 1) * n_nodes] for lag in range(order)])
        )
    return np.stack(kept)


def test_dekf_matches_a_full_joint_reference_implementation():
    """With a diagonal observation noise and a block-diagonal initial weight covariance the joint
    vector update equals the N decoupled scalar updates, so the two agree to numerical precision."""
    x = simulate_var1(lambda t: stationary_a(), 800, seed=20)
    kwargs = dict(
        order=2,
        forgetting=0.98,
        measurement_noise=0.3,
        obs_noise=1.0,
        state_cov_init=1.0,
        burn_in=50,
    )
    fast = fit(x, update="dekf", init="zeros", **kwargs)
    np.testing.assert_allclose(fast.coefs, _reference_dekf(x, **kwargs), atol=1e-8)


def test_the_batched_filter_matches_the_reference_for_several_trials():
    data = _trials(3, n_times=300)
    kwargs = dict(
        order=2,
        forgetting=0.98,
        measurement_noise=0.3,
        obs_noise=1.0,
        state_cov_init=1.0,
        burn_in=40,
    )
    together = fit_tvar(series(data), update="dekf", init="zeros", standardize=None, **kwargs)
    reference = np.mean([_reference_dekf(data[k], **kwargs) for k in range(3)], axis=0)
    np.testing.assert_allclose(together.coefs, reference, atol=1e-8)


# --------------------------------------------------------------------------
# Filter health and the checklist
# --------------------------------------------------------------------------


def test_a_healthy_fit_has_no_skipped_updates_and_passing_checks():
    x = simulate_var1(lambda t: stationary_a(), 3000, seed=22)
    result = fit(x, update="dekf")
    assert result.n_skipped_updates == 0
    assert result.n_state_resets == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", DirconnWarning)
        checks = {c.name: c for c in tvar_checks(result)}
    assert checks["filter_health"].status == "pass"
    assert checks["stationarity"].status == "pass"


def test_the_checklist_states_where_the_order_came_from_and_the_trial_assumption():
    data = _trials(3, n_times=500)
    result = fit_tvar(series(data), standardize=None)
    checks = {c.name: c for c in tvar_checks(result)}
    assert "stationary" in checks["order_source"].detail
    assert checks["order_source"].value == result.order
    assert "time-locked" in checks["trial_locking"].detail
    assert "trial_locking" not in {c.name for c in tvar_checks(fit(data[0]))}


def test_an_unstable_trajectory_warns_once_and_is_recorded():
    x = simulate_var1(lambda t: stationary_a(), 1500, seed=23)
    bad = dataclasses.replace(fit(x), n_unstable=800)  # more than 5 % of the slices
    with pytest.warns(DirconnWarning, match="spectral radius") as caught:
        checks = {c.name: c for c in tvar_checks(bad)}
    assert len(caught) == 1
    assert checks["stationarity"].status == "warn"
    assert checks["stationarity"].value == pytest.approx(bad.unstable_fraction)


def test_skipped_updates_and_state_resets_are_reported_not_hidden():
    x = simulate_var1(lambda t: stationary_a(), 1500, seed=23)
    bad = dataclasses.replace(fit(x), n_skipped_updates=3, n_state_resets=2)
    with pytest.warns(DirconnWarning, match="skipped"):
        checks = {c.name: c for c in tvar_checks(bad)}
    assert checks["filter_health"].status == "warn"
    assert "3" in checks["filter_health"].detail and "2" in checks["filter_health"].detail


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"order": 0}, "order must be"),
        ({"order": "bogus"}, "order must be"),
        ({"update": "bogus"}, "update must be one of"),
        ({"forgetting": 0.0}, "forgetting must be in"),
        ({"forgetting": 1.5}, "forgetting must be in"),
        ({"standardize": "bogus"}, "standardize must be one of"),
        ({"init": "bogus"}, "init must be"),
        ({"time_step": 0}, "time_step must be"),
        ({"burn_in": -1}, "burn_in must be"),
        ({"process_noise": -0.1}, "process_noise must be"),
        ({"obs_noise": 0.0}, "obs_noise must be"),
        ({"state_cov_init": 0.0}, "state_cov_init must be"),
        ({"measurement_noise": -1.0, "update": "dekf"}, "measurement_noise must be"),
        ({"measurement_noise": 0.1}, "only meaningful for update='dekf'"),
    ],
)
def test_invalid_options_are_rejected(kwargs, message):
    x = simulate_var1(lambda t: stationary_a(), 500, seed=16)
    kwargs = {"order": 1, **kwargs}
    with pytest.raises(ValueError, match=message):
        fit_tvar(series(x), **kwargs)


def test_rejects_something_that_is_not_a_time_series():
    with pytest.raises(TypeError, match="TimeSeries"):
        fit_tvar(np.zeros((2, 100)), order=1)


def test_rejects_a_recording_shorter_than_the_model():
    with pytest.raises(ValueError, match="too few"):
        fit_tvar(series(np.random.default_rng(0).standard_normal((2, 6))), order=5, init="zeros")


def test_a_burn_in_that_leaves_nothing_raises():
    x = simulate_var1(lambda t: stationary_a(), 300, seed=16)
    with pytest.raises(ValueError, match="burn_in"):
        fit(x, burn_in=300)


def test_tvarfit_is_the_documented_type():
    x = simulate_var1(lambda t: stationary_a(), 400, seed=16)
    assert isinstance(fit(x), TvarFit)


# --------------------------------------------------------------------------
# The old package as a numerical reference (fixtures made by scripts/make_tvar_fixtures.py)
# --------------------------------------------------------------------------


def _fixture_cases():
    with np.load(FIXTURES) as archive:
        return list(json.loads(str(archive["cases"])))


@pytest.mark.parametrize("name", _fixture_cases())
def test_agrees_with_the_old_package_from_a_common_zero_start(name):
    """Same input, same explicit start (zero coefficients): the recursions must coincide."""
    with np.load(FIXTURES) as archive:
        case = json.loads(str(archive["cases"]))[name]
        x = archive[f"{name}__x"]
        expected_coefs = archive[f"{name}__coefs"]
        expected_noise = archive[f"{name}__noise_cov"]
        expected_index = archive[f"{name}__sample_index"]
    result = fit_tvar(series(x), init="zeros", standardize=None, **case)
    np.testing.assert_array_equal(result.sample_index, expected_index)
    np.testing.assert_allclose(result.coefs, expected_coefs, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(result.noise_cov, expected_noise, atol=1e-9, rtol=1e-9)


# --------------------------------------------------------------------------
# Several forgetting factors in one pass
# --------------------------------------------------------------------------


@pytest.mark.parametrize("update", ["dual", "rls", "dekf"])
@pytest.mark.parametrize("n_trials", [1, 3])
def test_a_grid_of_forgetting_factors_equals_separate_fits_bit_for_bit(update, n_trials):
    from dirconn.tvmvar import fit_tvar_grid

    data = _trials(n_trials, n_times=500)
    ts = series(data if n_trials > 1 else data[0])
    factors = [0.95, 0.99, 0.9999, 1.0]
    kwargs = dict(order=2, standardize=None, update=update, time_step=3)
    together = fit_tvar_grid(ts, forgetting=factors, **kwargs)
    assert [f.forgetting for f in together] == factors
    for factor, joint in zip(factors, together, strict=True):
        alone = fit_tvar(ts, forgetting=factor, **kwargs)
        np.testing.assert_array_equal(joint.coefs, alone.coefs)
        np.testing.assert_array_equal(joint.noise_cov, alone.noise_cov)
        np.testing.assert_array_equal(joint.sample_index, alone.sample_index)
        assert joint.memory_samples == alone.memory_samples
        assert joint.process_noise == alone.process_noise
        assert joint.n_unstable == alone.n_unstable
        assert joint.n_skipped_updates == alone.n_skipped_updates


def test_a_grid_shares_the_order_selection_and_the_seed_fit():
    from dirconn.tvmvar import fit_tvar_grid

    ts = series(simulate_var1(lambda t: stationary_a(), 3000, seed=50))
    fits = fit_tvar_grid(ts, forgetting=[0.98, 0.9999], standardize=None)
    assert fits[0].order == fits[1].order
    assert fits[0].order_selection is fits[1].order_selection
    assert fits[0].seed_rows == fits[1].seed_rows > 0


def test_a_grid_needs_valid_factors():
    from dirconn.tvmvar import fit_tvar_grid

    ts = series(simulate_var1(lambda t: stationary_a(), 600, seed=51))
    with pytest.raises(ValueError, match="at least one"):
        fit_tvar_grid(ts, forgetting=[], order=1)
    with pytest.raises(ValueError, match="forgetting must be in"):
        fit_tvar_grid(ts, forgetting=[0.99, 1.5], order=1)


def test_the_default_forgetting_is_0_99999():
    import inspect

    from dirconn.tvmvar import fit_tvar_grid

    assert inspect.signature(fit_tvar).parameters["forgetting"].default == 0.99999
    assert inspect.signature(fit_tvar_grid).parameters["forgetting"].default == (0.99999,)
    x = simulate_var1(lambda t: stationary_a(), 1500, seed=60)
    result = fit_tvar(series(x), order=1, standardize=None)
    assert result.forgetting == 0.99999 and result.process_noise == pytest.approx(1 / 0.99999 - 1)
