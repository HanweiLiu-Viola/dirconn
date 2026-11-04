import numpy as np
import pytest

from dirconn.spectral import (
    ar_polynomial,
    dtf_from_coefs,
    frequency_grid,
    pdc_from_coefs,
    spectral_radius,
    transfer_function,
)

FS = 200.0
COEFS3 = np.array(
    [
        [[0.4, 0.1, 0.0], [0.2, 0.3, -0.1], [0.0, 0.25, 0.2]],  # lag 1, [target, source]
        [[-0.1, 0.0, 0.05], [0.0, -0.15, 0.0], [0.1, 0.0, -0.1]],  # lag 2
    ]
)
FREQS = np.array([0.0, 3.0, 10.0, 25.5, 40.0, 77.7, 100.0])


def reference(coefs, freqs, fs, noise_var=None):
    """DTF, PDC and gPDC written from the definitions, with loops and explicit matrix inverses.

    Layout ``[source, target, freq]``. ``abar`` and ``h`` are indexed ``[target, source]``.
    """
    order, n, _ = coefs.shape
    dtf = np.empty((n, n, len(freqs)))
    pdc = np.empty((n, n, len(freqs)))
    gpdc = np.empty((n, n, len(freqs)))
    weight = np.ones(n) if noise_var is None else 1.0 / np.asarray(noise_var)
    for k, f in enumerate(freqs):
        a_f = np.zeros((n, n), dtype=complex)
        for lag in range(1, order + 1):
            a_f += coefs[lag - 1] * np.exp(-2j * np.pi * f * lag / fs)
        abar = np.eye(n) - a_f
        h = np.linalg.inv(abar)
        for target in range(n):
            for source in range(n):
                dtf[source, target, k] = abs(h[target, source]) ** 2 / sum(
                    abs(h[target, m]) ** 2 for m in range(n)
                )
                pdc[source, target, k] = abs(abar[target, source]) ** 2 / sum(
                    abs(abar[m, source]) ** 2 for m in range(n)
                )
                gpdc[source, target, k] = (
                    weight[target]
                    * abs(abar[target, source]) ** 2
                    / sum(weight[m] * abs(abar[m, source]) ** 2 for m in range(n))
                )
    return dtf, pdc, gpdc


class TestFrequencyGrid:
    def test_default_runs_from_zero_to_the_nyquist_frequency(self):
        grid = frequency_grid(250.0)
        assert grid.shape == (128,)
        np.testing.assert_allclose(grid, np.linspace(0.0, 125.0, 128))

    def test_fmax_and_n_freqs_set_the_grid(self):
        np.testing.assert_allclose(
            frequency_grid(250.0, n_freqs=11, fmax=50.0), np.linspace(0.0, 50.0, 11)
        )

    def test_explicit_frequencies_are_used_as_they_are(self):
        np.testing.assert_array_equal(frequency_grid(250.0, freqs=[2, 5.5, 40]), [2.0, 5.5, 40.0])

    def test_the_grid_is_read_only(self):
        with pytest.raises(ValueError, match="read-only"):
            frequency_grid(250.0)[0] = 1.0

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(n_freqs=1), "n_freqs"),
            (dict(n_freqs=2.5), "n_freqs"),
            (dict(fmax=0.0), "fmax"),
            (dict(fmax=130.0), "Nyquist"),
            (dict(freqs=[5.0, 3.0]), "increasing"),
            (dict(freqs=[-1.0, 3.0]), "non-negative"),
            (dict(freqs=[1.0, 126.0]), "Nyquist"),
            (dict(freqs=[1.0, np.nan]), "finite"),
            (dict(freqs=[]), "freqs"),
            (dict(freqs=[1.0, 2.0], fmax=50.0), "either"),
        ],
    )
    def test_invalid_grids_are_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            frequency_grid(250.0, **kwargs)

    def test_sfreq_must_be_positive(self):
        with pytest.raises(ValueError, match="sfreq"):
            frequency_grid(0.0)


class TestAgainstTheReference:
    def test_dtf_pdc_and_gpdc_match_the_loop_reference(self):
        noise_var = np.array([1.0, 2.5, 0.4])
        dtf, pdc, gpdc = reference(COEFS3, FREQS, FS, noise_var)
        np.testing.assert_allclose(dtf_from_coefs(COEFS3, FREQS, FS), dtf, rtol=1e-10)
        np.testing.assert_allclose(pdc_from_coefs(COEFS3, FREQS, FS), pdc, rtol=1e-10)
        np.testing.assert_allclose(
            pdc_from_coefs(COEFS3, FREQS, FS, noise_var=noise_var), gpdc, rtol=1e-10
        )

    def test_the_transfer_function_is_the_inverse_of_the_ar_polynomial(self):
        abar = ar_polynomial(COEFS3, FREQS, FS)
        h = transfer_function(COEFS3, FREQS, FS)
        assert abar.shape == h.shape == (len(FREQS), 3, 3)
        for k in range(len(FREQS)):
            np.testing.assert_allclose(abar[k] @ h[k], np.eye(3), atol=1e-12)

    def test_the_ar_polynomial_follows_the_definition(self):
        abar = ar_polynomial(COEFS3, np.array([10.0]), FS)[0]
        expected = np.eye(3) - sum(
            COEFS3[lag - 1] * np.exp(-2j * np.pi * 10.0 * lag / FS) for lag in (1, 2)
        )
        np.testing.assert_allclose(abar, expected, rtol=1e-12)

    def test_layout_is_source_target_freq(self):
        dtf = dtf_from_coefs(COEFS3, FREQS, FS)
        assert dtf.shape == (3, 3, len(FREQS))
        # channel 0 does not drive channel 2 directly (COEFS3[:, 2, 0] is zero at lag 1) but
        # channel 1 does: the reference fixes the orientation of every cell.
        reference_dtf, _, _ = reference(COEFS3, FREQS, FS)
        assert not np.allclose(dtf, dtf.transpose(1, 0, 2))
        np.testing.assert_allclose(dtf, reference_dtf, rtol=1e-10)


class TestClosedForms:
    # x1[t] = a x1[t-1] + e1,  x2[t] = b x2[t-1] + c x1[t-1] + e2   (coefs[0] = [[a, 0], [c, b]])
    A, B, C = 0.6, 0.3, 0.8
    COEFS = np.array([[[A, 0.0], [C, B]]])
    FREQS = np.linspace(0.0, 100.0, 9)

    def z(self):
        return np.exp(-2j * np.pi * self.FREQS / FS)

    def test_dtf_from_the_driver_to_the_driven_channel(self):
        expected = self.C**2 / (self.C**2 + np.abs(1 - self.A * self.z()) ** 2)
        dtf = dtf_from_coefs(self.COEFS, self.FREQS, FS)
        np.testing.assert_allclose(dtf[0, 1], expected, rtol=1e-12)
        np.testing.assert_allclose(dtf[1, 0], 0.0, atol=1e-15)  # nothing flows from 2 to 1
        np.testing.assert_allclose(dtf[0, 0], 1.0)  # channel 1 receives only from itself

    def test_pdc_from_the_driver_to_the_driven_channel(self):
        denominator = np.abs(1 - self.A * self.z()) ** 2 + self.C**2
        pdc = pdc_from_coefs(self.COEFS, self.FREQS, FS)
        np.testing.assert_allclose(pdc[0, 1], self.C**2 / denominator, rtol=1e-12)
        np.testing.assert_allclose(pdc[1, 0], 0.0, atol=1e-15)
        np.testing.assert_allclose(pdc[0, 0], np.abs(1 - self.A * self.z()) ** 2 / denominator)

    def test_generalised_pdc_weights_by_the_inverse_innovation_variance(self):
        var1, var2 = 1.0, 4.0
        abar = np.abs(1 - self.A * self.z()) ** 2
        expected = (self.C**2 / var2) / (abar / var1 + self.C**2 / var2)
        gpdc = pdc_from_coefs(self.COEFS, self.FREQS, FS, noise_var=[var1, var2])
        np.testing.assert_allclose(gpdc[0, 1], expected, rtol=1e-12)

    def test_the_zero_frequency_values(self):
        dtf = dtf_from_coefs(self.COEFS, np.array([0.0]), FS)
        assert dtf[0, 1, 0] == pytest.approx(self.C**2 / (self.C**2 + (1 - self.A) ** 2))


class TestProperties:
    def test_dtf_sums_to_one_over_the_sources_of_every_target(self):
        dtf = dtf_from_coefs(COEFS3, FREQS, FS)
        np.testing.assert_allclose(dtf.sum(axis=0), 1.0, rtol=1e-12)
        assert dtf.min() >= 0.0

    def test_squared_pdc_sums_to_one_over_the_targets_of_every_source(self):
        pdc = pdc_from_coefs(COEFS3, FREQS, FS)
        np.testing.assert_allclose(pdc.sum(axis=1), 1.0, rtol=1e-12)

    def test_the_amplitude_form_is_the_square_root_and_its_squares_sum_to_one(self):
        squared = pdc_from_coefs(COEFS3, FREQS, FS, squared=True)
        amplitude = pdc_from_coefs(COEFS3, FREQS, FS, squared=False)
        np.testing.assert_allclose(amplitude, np.sqrt(squared), rtol=1e-12)
        np.testing.assert_allclose((amplitude**2).sum(axis=1), 1.0, rtol=1e-12)

    def test_equal_or_uniformly_scaled_noise_variances_give_plain_pdc(self):
        plain = pdc_from_coefs(COEFS3, FREQS, FS)
        np.testing.assert_allclose(
            pdc_from_coefs(COEFS3, FREQS, FS, noise_var=[2.0, 2.0, 2.0]), plain, rtol=1e-12
        )
        np.testing.assert_allclose(
            pdc_from_coefs(COEFS3, FREQS, FS, noise_var=[7.0, 7.0, 7.0]), plain, rtol=1e-12
        )

    def test_channel_permutation_permutes_the_measures(self):
        permutation = [2, 0, 1]
        permuted = COEFS3[:, permutation][:, :, permutation]
        for function in (dtf_from_coefs, pdc_from_coefs):
            original = function(COEFS3, FREQS, FS)
            np.testing.assert_allclose(
                function(permuted, FREQS, FS),
                original[np.ix_(permutation, permutation, range(len(FREQS)))],
                rtol=1e-12,
            )

    def test_results_are_real_and_finite(self):
        for result in (dtf_from_coefs(COEFS3, FREQS, FS), pdc_from_coefs(COEFS3, FREQS, FS)):
            assert result.dtype == np.float64 and np.isfinite(result).all()

    def test_the_frequency_axis_is_independent_of_the_other_frequencies(self):
        alone = dtf_from_coefs(COEFS3, FREQS[2:3], FS)
        together = dtf_from_coefs(COEFS3, FREQS, FS)
        np.testing.assert_allclose(alone[:, :, 0], together[:, :, 2], rtol=1e-12)


class TestFailures:
    def test_a_unit_root_makes_the_transfer_function_singular_and_raises(self):
        coefs = np.array([np.eye(2)])
        with pytest.raises(ValueError, match="singular"):
            transfer_function(coefs, np.array([0.0, 10.0]), FS)
        with pytest.raises(ValueError, match="singular"):
            dtf_from_coefs(coefs, np.array([0.0]), FS)

    def test_pdc_needs_no_inverse_and_marks_a_vanishing_denominator_as_undefined(self):
        coefs = np.array([[[1.0, 0.0], [0.0, 0.5]]])  # channel 0 has a unit root at f = 0
        pdc = pdc_from_coefs(coefs, np.array([0.0, 40.0]), FS)
        assert np.isnan(pdc[0, :, 0]).all()
        assert np.isfinite(pdc[0, :, 1]).all()
        assert np.isfinite(pdc[1, :, :]).all()

    @pytest.mark.parametrize(
        "noise_var", [[1.0], [1.0, 0.0, 1.0], [1.0, -1.0, 1.0], [1, np.nan, 1]]
    )
    def test_invalid_noise_variances_are_rejected(self, noise_var):
        with pytest.raises(ValueError, match="noise_var"):
            pdc_from_coefs(COEFS3, FREQS, FS, noise_var=noise_var)

    def test_coefficients_must_be_three_dimensional_and_square(self):
        with pytest.raises(ValueError, match="coefs"):
            dtf_from_coefs(np.zeros((3, 3)), FREQS, FS)
        with pytest.raises(ValueError, match="coefs"):
            dtf_from_coefs(np.zeros((2, 3, 4)), FREQS, FS)

    def test_frequencies_must_be_a_vector(self):
        with pytest.raises(ValueError, match="freqs"):
            dtf_from_coefs(COEFS3, np.zeros((2, 2)), FS)


class TestSpectralRadius:
    def test_a_scalar_ar1_process(self):
        assert spectral_radius(np.array([[[0.5]]])) == pytest.approx(0.5)

    def test_matches_the_roots_of_the_ar2_characteristic_polynomial(self):
        a1, a2 = 0.5, -0.2
        roots = np.roots([1.0, -a1, -a2])
        coefs = np.array([[[a1]], [[a2]]])
        assert spectral_radius(coefs) == pytest.approx(np.abs(roots).max())

    def test_a_unit_root_system_has_radius_one(self):
        assert spectral_radius(np.array([np.eye(2)])) == pytest.approx(1.0)

    def test_the_example_system_is_stable(self):
        assert spectral_radius(COEFS3) < 1.0
