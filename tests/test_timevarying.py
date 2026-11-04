"""ADTF and TPDC: time-resolved DTF and PDC from the time-varying MVAR."""

import dataclasses
import warnings

import numpy as np
import pytest

from dirconn import (
    ADTF,
    DTF,
    PDC,
    TPDC,
    Connectivity,
    DirconnWarning,
    TimeSeries,
    adtf,
    tpdc,
)
from dirconn.spectral import dtf_from_coefs_series, frequency_grid, pdc_from_coefs_series
from dirconn.tvmvar import fit_tvar

SFREQ = 100.0

# The default settings remember only a few samples, so most fits here carry
# the sampling warning; the tests that look at warnings say so with pytest.warns.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*coefficients per equation:dirconn.exceptions.DirconnWarning"
)


def chain(n_times, *, switch=None, seed=0, noise=1.0, n_trials=1):
    """Two nodes; 0 -> 1 before ``switch`` (the whole recording if ``None``), 1 -> 0 after."""
    rng = np.random.default_rng(seed)
    before = np.array([[0.5, 0.0], [0.6, 0.4]])
    after = np.array([[0.5, 0.6], [0.0, 0.4]])
    out = np.zeros((n_trials, 2, n_times))
    for k in range(n_trials):
        x = np.zeros((2, n_times))
        for t in range(1, n_times):
            a = before if switch is None or t < switch else after
            x[:, t] = a @ x[:, t - 1] + noise * rng.standard_normal(2)
        out[k] = x
    seg = "continuous" if n_trials == 1 else "epoched"
    return TimeSeries.from_array(out if n_trials > 1 else out[0], SFREQ, segmentation=seg)


# --------------------------------------------------------------------------
# What the estimators compute
# --------------------------------------------------------------------------


def test_adtf_is_the_dtf_of_the_coefficient_trajectory():
    ts = chain(3000, seed=1)
    result = ADTF(order=1, n_freqs=17, time_step=0.1).fit(ts)
    fit = fit_tvar(ts, order=1, time_step=10)
    freqs = frequency_grid(SFREQ, n_freqs=17)
    np.testing.assert_allclose(
        result.values, dtf_from_coefs_series(fit.coefs, freqs, SFREQ), rtol=1e-12
    )
    np.testing.assert_allclose(result.times, fit.times)
    np.testing.assert_allclose(result.freqs, freqs)


@pytest.mark.parametrize("squared", [True, False])
def test_tpdc_is_the_pdc_of_the_coefficient_trajectory(squared):
    ts = chain(3000, seed=2)
    result = TPDC(order=1, n_freqs=17, time_step=0.1, squared=squared).fit(ts)
    fit = fit_tvar(ts, order=1, update="dekf", time_step=10)
    freqs = frequency_grid(SFREQ, n_freqs=17)
    np.testing.assert_allclose(
        result.values, pdc_from_coefs_series(fit.coefs, freqs, SFREQ, squared=squared), rtol=1e-12
    )


def test_the_defaults_differ_as_documented():
    assert ADTF().update == "dual"
    assert TPDC().update == "dekf"
    assert ADTF().n_freqs == TPDC().n_freqs == 64
    assert ADTF().forgetting == TPDC().forgetting == 0.99999  # decided 2026-10-04
    assert ADTF().order == "bic" and ADTF().max_order == 5
    assert TPDC().squared is True


def test_a_stationary_signal_gives_nearly_the_stationary_dtf_and_pdc():
    # Thresholds from the distribution of the largest absolute difference over 10 seeds (3-12) of
    # this setting: ADTF 0.029-0.048, TPDC 0.053-0.091 (its DEKF assumes 10 % measurement noise).
    ts = chain(8000, seed=3)
    static = DTF(order=1, n_freqs=17, check_order_stability=False).fit(ts)
    moving = ADTF(order=1, n_freqs=17, forgetting=0.999).fit(ts)
    half = moving.values.shape[3] // 2
    np.testing.assert_allclose(
        moving.values[..., half:].mean(axis=3)[..., None], static.values, atol=0.08
    )
    static_pdc = PDC(order=1, n_freqs=17, check_order_stability=False).fit(ts)
    moving_pdc = TPDC(order=1, n_freqs=17, forgetting=0.999).fit(ts)
    np.testing.assert_allclose(
        moving_pdc.values[..., half:].mean(axis=3)[..., None], static_pdc.values, atol=0.12
    )


@pytest.mark.parametrize("estimator_class", [ADTF, TPDC])
def test_the_direction_follows_a_mid_recording_switch(estimator_class):
    n_times = 6000
    switch = n_times // 2
    ts = chain(n_times, switch=switch, seed=4)
    result = estimator_class(order=1, forgetting=0.99, n_freqs=17).fit(ts)
    band = result.values[..., 1:9, :].mean(axis=2)  # [source, target, time]
    before = (result.times < (switch - 400) / SFREQ) & (result.times > 100 / SFREQ)
    after = result.times > (switch + 600) / SFREQ
    assert band[0, 1, before].mean() > band[1, 0, before].mean()
    assert band[1, 0, after].mean() > band[0, 1, after].mean()


# --------------------------------------------------------------------------
# Axes
# --------------------------------------------------------------------------


def test_the_result_is_time_resolved_in_seconds_and_frequency_resolved():
    ts = TimeSeries.from_array(chain(2000, seed=5).data[0], SFREQ, tmin=-1.0)
    result = adtf(ts, order=1, time_step=0.05)
    assert isinstance(result, Connectivity)
    assert result.is_time_resolved and result.is_frequency_resolved
    assert result.values.shape == (2, 2, 64, result.times.size)
    np.testing.assert_allclose(np.diff(result.times), 0.05)
    assert result.times[0] >= -1.0 + result.provenance.details["burn_in"] / SFREQ - 1e-9
    assert result.times[-1] > -1.0 + (1999 - 5) / SFREQ - 1e-9  # the last step fits in the record
    assert result.times[-1] <= -1.0 + 1999 / SFREQ + 1e-9
    assert result.freqs[0] == 0.0 and result.freqs[-1] == pytest.approx(SFREQ / 2)


def test_without_a_time_step_about_200_points_are_returned():
    ts = chain(8000, seed=6)
    result = ADTF(order=1).fit(ts)
    assert 150 <= result.times.size <= 200
    assert result.provenance.details["time_step"] > 1


def test_a_short_recording_keeps_every_sample_when_it_has_fewer_than_200():
    result = ADTF(order=1, burn_in=20).fit(chain(150, seed=6))
    assert result.provenance.details["time_step"] == 1
    assert result.times.size == 130


def test_the_time_step_must_be_a_multiple_of_the_sampling_interval():
    with pytest.raises(ValueError, match="multiple of the sampling interval"):
        ADTF(order=1, time_step=0.015).fit(chain(1000, seed=7))


def test_an_explicit_frequency_axis_is_used():
    result = ADTF(order=1, freqs=[5.0, 10.0, 20.0], time_step=0.1).fit(chain(2000, seed=7))
    np.testing.assert_allclose(result.freqs, [5.0, 10.0, 20.0])


# --------------------------------------------------------------------------
# Provenance, checklist, warnings
# --------------------------------------------------------------------------


def test_provenance_records_the_method_the_order_and_the_settings():
    ts = chain(3000, seed=8)
    result = ADTF(order="bic", forgetting=0.98, update="rls", time_step=0.1).fit(ts)
    prov = result.provenance
    assert prov.method == "adtf"
    assert prov.model_order == prov.details["order"]
    assert prov.parameters["forgetting"] == 0.98
    assert prov.parameters["update"] == "rls"
    for key in ("update", "init", "burn_in", "time_step", "n_trials", "unstable_fraction"):
        assert key in prov.details
    assert prov.details["time_step_seconds"] == pytest.approx(0.1)
    assert prov.preprocessing["standardized"] is True
    assert len(prov.preprocessing["scale"]) == 2
    names = {c.name: c for c in prov.assumptions.checks}
    for key in ("stationarity", "filter_health", "order_source", "sampling_adequacy"):
        assert key in names
    assert "trial_locking" not in names
    assert names["order_source"].status == "not_checked"


def test_several_trials_are_averaged_and_the_assumption_is_listed():
    ts = chain(600, switch=300, seed=9, n_trials=6)
    result = ADTF(order=1, forgetting=0.97, n_freqs=9).fit(ts)
    assert result.provenance.details["n_trials"] == 6
    assert "trial_locking" in {c.name for c in result.provenance.assumptions.checks}


@pytest.mark.filterwarnings("ignore:.*companion spectral radius:dirconn.exceptions.DirconnWarning")
def test_a_short_memory_with_many_coefficients_warns_about_sampling():
    ts = chain(2000, seed=10)
    with pytest.warns(DirconnWarning, match="coefficients per equation"):
        result = ADTF(order=3, forgetting=0.9).fit(ts)
    assert result.provenance.assumptions.get("sampling_adequacy").status == "warn"


def test_a_healthy_run_raises_no_dirconn_warning():
    ts = chain(4000, seed=11)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DirconnWarning)
        ADTF(order=1, forgetting=0.99999, n_freqs=9).fit(ts)  # a memory of about 100 samples


@pytest.mark.filterwarnings("ignore:.*companion spectral radius:dirconn.exceptions.DirconnWarning")
def test_without_standardising_unequal_channel_scales_are_flagged():
    data = chain(2000, seed=12).data.copy()
    data[0, 1] *= 50.0
    ts = TimeSeries.from_array(data, SFREQ, segmentation="continuous")
    with pytest.warns(DirconnWarning, match="standard deviations differ"):
        result = ADTF(order=1, standardize=False, n_freqs=5).fit(ts)
    assert result.provenance.preprocessing["standardized"] is False


def test_the_convenience_functions_equal_the_classes():
    ts = chain(1500, seed=14)
    np.testing.assert_array_equal(
        adtf(ts, order=1, n_freqs=5).values, ADTF(order=1, n_freqs=5).fit(ts).values
    )
    np.testing.assert_array_equal(
        tpdc(ts, order=1, n_freqs=5).values, TPDC(order=1, n_freqs=5).fit(ts).values
    )


def test_results_are_reproducible():
    ts = chain(2000, seed=13)
    a = ADTF(order=1, time_step=0.1, n_freqs=9).fit(ts)
    b = ADTF(order=1, time_step=0.1, n_freqs=9).fit(ts)
    np.testing.assert_array_equal(a.values, b.values)


def test_one_channel_is_rejected():
    ts = TimeSeries.from_array(
        np.zeros((1, 500)) + np.random.default_rng(0).normal(size=(1, 500)), SFREQ
    )
    with pytest.raises(ValueError, match="at least 2 channels"):
        ADTF(order=1).fit(ts)


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------


@pytest.mark.parametrize("estimator_class", [ADTF, TPDC])
@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"update": "bogus"}, "update must be one of"),
        ({"forgetting": 1.2}, "forgetting must be in"),
        ({"order": 0}, "order must be"),
        ({"time_step": -0.1}, "time_step must be"),
        ({"time_step": 0.0}, "time_step must be"),
        ({"time_step": "fast"}, "time_step must be"),
        ({"init": "bogus"}, "init must be"),
        ({"n_freqs": 1}, "n_freqs must be"),
        ({"freqs": [1.0, 2.0], "fmax": 10.0}, "either freqs or fmax"),
        ({"standardize": "yes"}, "standardize must be a bool"),
        ({"burn_in": -3}, "burn_in must be"),
    ],
)
def test_invalid_parameters_are_rejected_when_the_estimator_is_built(
    estimator_class, kwargs, message
):
    with pytest.raises((ValueError, TypeError), match=message):
        estimator_class(**kwargs)


def test_measurement_noise_belongs_to_dekf_only():
    assert TPDC(measurement_noise=0.2).measurement_noise == 0.2
    assert ADTF(update="dekf", measurement_noise=0.2).measurement_noise == 0.2
    with pytest.raises(ValueError, match="only meaningful for update='dekf'"):
        ADTF(measurement_noise=0.2)
    with pytest.raises(ValueError, match="only meaningful for update='dekf'"):
        TPDC(update="dual", measurement_noise=0.2)


def test_the_estimator_is_a_frozen_dataclass_with_parameters():
    estimator = ADTF(order=2, forgetting=0.95)
    assert estimator.get_params()["forgetting"] == 0.95
    with pytest.raises(dataclasses.FrozenInstanceError):
        estimator.forgetting = 0.9  # type: ignore[misc]


def test_the_info_describes_a_time_and_frequency_resolved_directed_measure():
    for cls in (ADTF, TPDC):
        info = cls.info
        assert info.directed and info.time_resolved and info.frequency_resolved
        assert info.value_range == (0.0, 1.0)
    assert ADTF.info.name == "adtf" and TPDC.info.name == "tpdc"


def test_a_segment_reduction_of_adtf_has_the_shape_and_axes_of_wdtf_with_segments():
    """The two ways of getting one value per segment give results that can be compared directly."""
    from dirconn import WDTF

    ts = chain(3000, seed=15)
    segments = [(5.0, 9.0), (15.0, 19.0)]
    reduced = ADTF(order=1, n_freqs=9, time_step=0.1).fit(ts).reduce_time(segments)
    fitted = WDTF(segments=segments, order=1, n_freqs=9).fit(ts)
    assert reduced.values.shape == fitted.values.shape == (2, 2, 9, 2)
    np.testing.assert_allclose(reduced.times, fitted.times)
    np.testing.assert_array_equal(reduced.freqs, fitted.freqs)


# --------------------------------------------------------------------------
# A result from a precomputed fit (the sweep computes many forgetting factors at once)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("estimator_class", [ADTF, TPDC])
def test_a_result_from_a_grid_fit_equals_the_estimator_fit(estimator_class):
    from dirconn.tvmvar import fit_tvar_grid

    ts = chain(2000, seed=16)
    factors = [0.98, 0.9999]
    base = estimator_class(order=1, n_freqs=9, time_step=0.1)
    fits = fit_tvar_grid(ts, forgetting=factors, **base.tvar_options(ts))
    for factor, fit in zip(factors, fits, strict=True):
        estimator = estimator_class(order=1, n_freqs=9, time_step=0.1, forgetting=factor)
        joint = estimator.from_fit(ts, fit)
        alone = estimator.fit(ts)
        np.testing.assert_array_equal(joint.values, alone.values)
        np.testing.assert_array_equal(joint.times, alone.times)
        assert joint.provenance.details == alone.provenance.details
        assert [c.name for c in joint.provenance.assumptions.checks] == [
            c.name for c in alone.provenance.assumptions.checks
        ]


def test_from_fit_refuses_a_fit_of_other_settings():
    from dirconn.tvmvar import fit_tvar_grid

    ts = chain(1500, seed=17)
    estimator = ADTF(order=1, n_freqs=5, forgetting=0.99)
    (fit,) = fit_tvar_grid(ts, forgetting=[0.95], **estimator.tvar_options(ts))
    with pytest.raises(ValueError, match="forgetting"):
        estimator.from_fit(ts, fit)
    (other,) = fit_tvar_grid(ts, forgetting=[0.99], **ADTF(order=1, update="dekf").tvar_options(ts))
    with pytest.raises(ValueError, match="update"):
        estimator.from_fit(ts, other)
