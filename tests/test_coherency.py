"""Tests of imaginary coherence and the phase slope index."""

import numpy as np
import pytest

from dirconn import TimeSeries
from dirconn._csd import cross_spectra
from dirconn.coherency import PSI, ImCoh, imcoh, psi

SFREQ = 100.0
MIX = np.array([[1.0, 0.8], [0.6, 1.0]])


def _mixture(seed, n_times=6000, mixing=MIX):
    """Instantaneous mixture of two independent white sources: no lag anywhere."""
    rng = np.random.default_rng(seed)
    return TimeSeries.from_array(mixing @ rng.standard_normal((2, n_times)), SFREQ)


def _lagged(seed, n_times=6000, delay=3, mixing=None):
    """Channel 1 is channel 0 delayed by ``delay`` samples, plus independent noise."""
    rng = np.random.default_rng(seed)
    src = rng.standard_normal(n_times + delay)
    x = np.stack([src[delay:], src[:n_times]]) + 0.7 * rng.standard_normal((2, n_times))
    if mixing is not None:
        x = mixing @ x
    return TimeSeries.from_array(x, SFREQ)


def _epoched(seed=0, n_trials=30, n_times=200, delay=3):
    rng = np.random.default_rng(seed)
    src = rng.standard_normal((n_trials, n_times + delay))
    x = np.stack([src[:, delay:], src[:, :n_times]], axis=1)
    x = x + 0.5 * rng.standard_normal((n_trials, 2, n_times))
    return TimeSeries.from_array(x, SFREQ, segmentation="epoched")


# --- imaginary coherence -------------------------------------------------------------------


def test_imcoh_shape_antisymmetry_and_range():
    data = TimeSeries.from_array(np.random.default_rng(0).standard_normal((3, 4000)), SFREQ)
    result = ImCoh(segment_length=2.0).fit(data)
    spectra = cross_spectra(data, segment_length=2.0)
    np.testing.assert_array_equal(result.freqs, spectra.freqs)
    v = result.values[:, :, :, 0]
    assert np.isnan(v[[0, 1, 2], [0, 1, 2]]).all()
    off = ~np.eye(3, dtype=bool)
    np.testing.assert_allclose(v[off], -np.swapaxes(v, 0, 1)[off], atol=1e-15)
    assert np.nanmax(np.abs(v)) <= 1.0
    assert result.provenance.method == "imcoh"
    assert result.provenance.details["cross_spectra"]["segment_samples"] == 200
    assert not ImCoh.info.directed


def test_imcoh_is_positive_when_the_target_lags_below_half_a_cycle():
    """values[i, j] > 0 while the delay of j is under half a period (f < sfreq / (2 d))."""
    result = ImCoh(segment_length=2.0).fit(_lagged(1))
    low = (result.freqs > 1) & (result.freqs < 0.8 * SFREQ / (2 * 3))
    assert (result.values[0, 1, low, 0] > 0).all()
    assert (result.values[1, 0, low, 0] < 0).all()


def test_imcoh_is_small_for_an_instantaneous_mixture():
    # Measured over 30 seeds (6000 samples, 2 s segments): the mean |imcoh| below 20 Hz is
    # 0.010 to 0.022 for the mixture and 0.155 to 0.178 for a lagged pair under the same mixing.
    for seed in range(5):
        mixed = ImCoh(segment_length=2.0).fit(_mixture(seed))
        lagged = ImCoh(segment_length=2.0).fit(_lagged(seed, mixing=MIX))
        low = mixed.freqs < 20
        assert np.mean(np.abs(mixed.values[0, 1, low, 0])) < 0.04
        assert np.mean(np.abs(lagged.values[0, 1, low, 0])) > 0.1


def test_imcoh_matches_mne_connectivity():
    mne_connectivity = pytest.importorskip("mne_connectivity")
    data = _epoched()
    ours = ImCoh(fmin=2.5).fit(data)  # mne-connectivity warns below 5 cycles per epoch
    con = mne_connectivity.spectral_connectivity_epochs(
        np.asarray(data.data),
        method="imcoh",
        mode="multitaper",
        sfreq=SFREQ,
        indices=(np.array([0]), np.array([1])),
        fmin=2.5,
        fmax=SFREQ / 2,
        verbose=False,
    )
    np.testing.assert_allclose(np.asarray(con.freqs), ours.freqs)
    np.testing.assert_allclose(con.get_data()[0], ours.values[0, 1, :, 0], atol=1e-12)


def test_imcoh_frequency_range_and_shorthand():
    data = _lagged(2)
    result = imcoh(data, segment_length=2.0, fmin=5.0, fmax=20.0)
    assert result.freqs.min() >= 5.0 and result.freqs.max() <= 20.0
    np.testing.assert_array_equal(
        result.values, ImCoh(segment_length=2.0, fmin=5.0, fmax=20.0).fit(data).values
    )


@pytest.mark.parametrize("kwargs", [{"n_freqs": 64}, {"freqs": [10.0, 20.0]}])
def test_spectral_grid_arguments_are_refused(kwargs):
    with pytest.raises(TypeError):
        ImCoh(segment_length=2.0, **kwargs)
    with pytest.raises(TypeError):
        PSI(segment_length=2.0, bands=[(5, 20)], **kwargs)


def test_continuous_input_needs_segment_length():
    with pytest.raises(ValueError, match="segment_length is required"):
        ImCoh().fit(_lagged(0))


# --- phase slope index ---------------------------------------------------------------------


def test_psi_antisymmetric_and_positive_when_the_source_leads():
    result = PSI(segment_length=2.0, bands=[(2, 20)]).fit(_lagged(3))
    v = result.values[:, :, 0, 0]
    assert v[0, 1] > 0
    assert v[1, 0] == pytest.approx(-v[0, 1])
    assert np.isnan(v[0, 0]) and np.isnan(v[1, 1])


def test_psi_is_small_for_an_instantaneous_mixture():
    # Measured over 30 seeds: PSI in 2 to 20 Hz is -0.023 to 0.043 for the mixture and 1.34 to
    # 1.58 for the unmixed lagged pair.
    for seed in range(5):
        assert abs(PSI(segment_length=2.0, bands=[(2, 20)]).fit(_mixture(seed)).values[0, 1]) < 0.1
        assert PSI(segment_length=2.0, bands=[(2, 20)]).fit(_lagged(seed)).values[0, 1] > 1.0


def test_psi_sign_can_flip_under_strong_mixing_of_a_coupled_pair():
    """The documented limit: mixing *coupled* sources distorts PSI, here enough to flip it.

    Measured over 20 seeds with this mixing: PSI in 2 to 20 Hz is -0.16 to -0.07, while imcoh
    keeps its sign. Not a property to rely on, a warning to show.
    """
    for seed in range(5):
        data = _lagged(seed, mixing=MIX)
        assert PSI(segment_length=2.0, bands=[(2, 20)]).fit(data).values[0, 1] < 0
        result = ImCoh(segment_length=2.0).fit(data)
        assert result.values[0, 1, (result.freqs > 2) & (result.freqs < 8), 0].mean() > 0


def test_psi_bands_centres_and_provenance():
    result = psi(_lagged(4), segment_length=2.0, bands=["theta", (13, 30)])
    np.testing.assert_allclose(result.freqs, [6.0, 21.5])
    details = result.provenance.details
    assert details["bands"] == [[4.0, 8.0], [13.0, 30.0]]
    assert details["bins_per_band"] == [9, 35]
    assert result.provenance.method == "psi"


def test_psi_matches_mne_connectivity():
    mne_connectivity = pytest.importorskip("mne_connectivity")
    data = _epoched(seed=5)
    bands = [(4.0, 8.0), (10.0, 25.0)]
    ours = PSI(bands=bands).fit(data)
    half_bin = 0.25  # mne-connectivity excludes both band limits, dirconn includes them
    for b, (lo, hi) in enumerate(bands):
        ref = mne_connectivity.phase_slope_index(
            np.asarray(data.data),
            mode="multitaper",
            sfreq=SFREQ,
            indices=(np.array([0]), np.array([1])),
            fmin=lo - half_bin,
            fmax=hi + half_bin,
            verbose=False,
        )
        assert ours.values[0, 1, b, 0] == pytest.approx(float(ref.get_data()[0, 0]), rel=1e-10)


@pytest.mark.parametrize(
    ("bands", "match"),
    [
        (None, "bands is required"),
        ("alpha", "sequence of bands"),
        ([], "at least one band"),
        (["nope"], "Unknown band"),
        ([(10, 5)], "fmin < fmax"),
        ([(13, 30), (4, 8)], "increase"),
        ([(1, 2, 3)], "name or"),
    ],
)
def test_psi_band_errors(bands, match):
    with pytest.raises(ValueError, match=match):
        PSI(segment_length=2.0, bands=bands)


def test_psi_band_too_narrow_or_above_nyquist():
    data = _lagged(0)
    with pytest.raises(ValueError, match="at least 2"):
        PSI(segment_length=2.0, bands=[(10.0, 10.4)]).fit(data)
    with pytest.raises(ValueError, match="Nyquist"):
        PSI(segment_length=2.0, bands=[(40.0, 60.0)]).fit(data)
