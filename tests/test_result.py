import dataclasses
import warnings

import numpy as np
import pytest
import xarray as xr

from dirconn import Connectivity, Provenance
from dirconn.result import DIMS


@pytest.fixture
def prov():
    return Provenance(method="pdc", parameters={"order": 3}, model_order=3, seed=0)


def spectrum(prov, *, freqs=(4.0, 6.0, 8.0, 10.0, 12.0, 14.0), n_times=1, values=None):
    n_src, n_tgt = 2, 3
    freqs = np.asarray(freqs, dtype=float)
    shape = (n_src, n_tgt, len(freqs), n_times)
    if values is None:
        values = np.arange(np.prod(shape), dtype=float).reshape(shape)
    return Connectivity.from_array(
        values,
        source=["A", "B"],
        target=["X", "Y", "Z"],
        freqs=freqs,
        times=None if n_times == 1 else np.arange(n_times) * 0.1,
        provenance=prov,
    )


class TestConstruction:
    def test_dims_are_fixed_and_always_four(self, prov):
        conn = Connectivity.from_array(np.ones((2, 2)), source=["A", "B"], provenance=prov)
        assert DIMS == ("source", "target", "freq", "time")
        assert conn.values.shape == (2, 2, 1, 1)
        assert conn.to_xarray().dims == DIMS

    def test_target_defaults_to_source(self, prov):
        conn = Connectivity.from_array(np.ones((2, 2)), source=["A", "B"], provenance=prov)
        assert conn.target == ("A", "B")

    def test_unused_dimensions_have_nan_coordinates_and_are_not_resolved(self, prov):
        conn = Connectivity.from_array(np.ones((2, 2)), source=["A", "B"], provenance=prov)
        assert np.isnan(conn.freqs).all() and conn.freqs.shape == (1,)
        assert np.isnan(conn.times).all() and conn.times.shape == (1,)
        assert not conn.is_frequency_resolved
        assert not conn.is_time_resolved

    def test_three_dimensional_input_is_source_target_freq(self, prov):
        conn = Connectivity.from_array(
            np.ones((2, 2, 5)),
            source=["A", "B"],
            freqs=[1, 2, 3, 4, 5],
            provenance=prov,
        )
        assert conn.values.shape == (2, 2, 5, 1)
        assert conn.is_frequency_resolved
        assert not conn.is_time_resolved

    def test_time_resolved_broadband_needs_four_dimensions(self, prov):
        conn = Connectivity.from_array(
            np.ones((2, 2, 1, 4)),
            source=["A", "B"],
            times=[0.0, 0.1, 0.2, 0.3],
            provenance=prov,
        )
        assert conn.is_time_resolved
        assert not conn.is_frequency_resolved

    def test_a_single_frequency_with_a_real_coordinate_is_resolved(self, prov):
        conn = Connectivity.from_array(
            np.ones((2, 2, 1, 1)), source=["A", "B"], freqs=[10.0], provenance=prov
        )
        assert conn.is_frequency_resolved

    def test_provenance_is_required(self):
        with pytest.raises(TypeError):
            Connectivity.from_array(np.ones((2, 2)), source=["A", "B"])  # type: ignore[call-arg]

    def test_values_are_read_only_and_object_is_immutable(self, prov):
        conn = spectrum(prov)
        with pytest.raises(ValueError, match="read-only"):
            conn.values[0, 0, 0, 0] = 1.0
        with pytest.raises(dataclasses.FrozenInstanceError):
            conn.source = ("a", "b")  # type: ignore[misc]

    def test_integer_values_become_float64(self, prov):
        values = np.ones((2, 2), dtype=int)
        conn = Connectivity.from_array(values, source=["A", "B"], provenance=prov)
        assert conn.values.dtype == np.float64

    def test_complex_values_are_kept(self, prov):
        conn = Connectivity.from_array(np.full((2, 2), 1 + 2j), source=["A", "B"], provenance=prov)
        assert conn.values.dtype == np.complex128
        assert conn.is_complex

    def test_nan_is_allowed_for_undefined_entries_but_inf_is_not(self, prov):
        values = np.ones((2, 2))
        values[0, 0] = np.nan
        Connectivity.from_array(values, source=["A", "B"], provenance=prov)
        values[0, 1] = np.inf
        with pytest.raises(ValueError, match="infinite"):
            Connectivity.from_array(values, source=["A", "B"], provenance=prov)

    @pytest.mark.parametrize("shape", [(2,), (2, 2, 1, 1, 1)])
    def test_unsupported_number_of_dimensions_is_rejected(self, prov, shape):
        with pytest.raises(ValueError, match="dimension"):
            Connectivity.from_array(np.ones(shape), source=["A", "B"], provenance=prov)

    def test_name_count_must_match_the_data(self, prov):
        with pytest.raises(ValueError, match="source"):
            Connectivity.from_array(np.ones((2, 2)), source=["A"], provenance=prov)
        with pytest.raises(ValueError, match="target"):
            Connectivity.from_array(
                np.ones((2, 2)), source=["A", "B"], target=["X"], provenance=prov
            )

    def test_names_must_be_unique(self, prov):
        with pytest.raises(ValueError, match="unique"):
            Connectivity.from_array(np.ones((2, 2)), source=["A", "A"], provenance=prov)

    def test_freqs_must_match_and_increase(self, prov):
        with pytest.raises(ValueError, match="freqs"):
            Connectivity.from_array(
                np.ones((2, 2, 3)), source=["A", "B"], freqs=[1.0, 2.0], provenance=prov
            )
        with pytest.raises(ValueError, match="increasing"):
            Connectivity.from_array(
                np.ones((2, 2, 3)), source=["A", "B"], freqs=[1.0, 3.0, 2.0], provenance=prov
            )

    def test_negative_or_missing_frequencies_are_rejected_when_resolved(self, prov):
        with pytest.raises(ValueError, match="freqs"):
            Connectivity.from_array(
                np.ones((2, 2, 2)), source=["A", "B"], freqs=[-1.0, 2.0], provenance=prov
            )
        with pytest.raises(ValueError, match="freqs"):
            Connectivity.from_array(
                np.ones((2, 2, 2)), source=["A", "B"], freqs=[np.nan, 2.0], provenance=prov
            )

    def test_frequency_axis_longer_than_one_needs_coordinates(self, prov):
        with pytest.raises(ValueError, match="freqs"):
            Connectivity.from_array(np.ones((2, 2, 3)), source=["A", "B"], provenance=prov)

    def test_times_must_match_and_increase(self, prov):
        with pytest.raises(ValueError, match="times"):
            Connectivity.from_array(
                np.ones((2, 2, 1, 3)), source=["A", "B"], times=[0.0, 1.0], provenance=prov
            )
        with pytest.raises(ValueError, match="increasing"):
            Connectivity.from_array(
                np.ones((2, 2, 1, 3)), source=["A", "B"], times=[0.0, 2.0, 1.0], provenance=prov
            )

    def test_time_axis_longer_than_one_needs_coordinates(self, prov):
        with pytest.raises(ValueError, match="times"):
            Connectivity.from_array(np.ones((2, 2, 1, 3)), source=["A", "B"], provenance=prov)

    def test_negative_times_are_allowed(self, prov):
        conn = Connectivity.from_array(
            np.ones((2, 2, 1, 3)), source=["A", "B"], times=[-0.2, 0.0, 0.2], provenance=prov
        )
        assert conn.times[0] == -0.2

    def test_to_xarray_has_named_coordinates_and_returns_a_copy(self, prov):
        conn = spectrum(prov, n_times=2)
        da = conn.to_xarray()
        assert isinstance(da, xr.DataArray)
        assert list(da["source"].values) == ["A", "B"]
        assert list(da["target"].values) == ["X", "Y", "Z"]
        np.testing.assert_array_equal(da["freq"].values, conn.freqs)
        da.values[0, 0, 0, 0] = -1.0
        assert conn.values[0, 0, 0, 0] != -1.0

    def test_with_provenance_returns_a_new_result_sharing_the_values(self, prov):
        conn = spectrum(prov)
        other = Provenance(method="other")
        replaced = conn.with_provenance(other)
        assert replaced.provenance.method == "other"
        assert conn.provenance.method == "pdc"
        np.testing.assert_array_equal(replaced.values, conn.values)
        assert replaced.source == conn.source

    def test_repr_does_not_print_the_values(self, prov):
        text = repr(spectrum(prov))
        assert "Connectivity" in text and "pdc" in text and "2x3x6x1" in text


class TestReduceFreq:
    def test_mean_uses_the_inclusive_band(self, prov):
        conn = spectrum(prov)
        out = conn.reduce_freq((6.0, 10.0), how="mean")
        expected = conn.values[:, :, 1:4, :].mean(axis=2, keepdims=True)
        np.testing.assert_allclose(out.values, expected)
        assert out.values.shape == (2, 3, 1, 1)

    def test_result_frequency_coordinate_is_the_band_centre(self, prov):
        out = spectrum(prov).reduce_freq((6.0, 10.0))
        np.testing.assert_allclose(out.freqs, [8.0])
        assert out.is_frequency_resolved

    def test_median_and_max_for_real_values(self, prov):
        values = np.zeros((2, 3, 6, 1))
        values[0, 0, :, 0] = [0.1, 0.9, 0.2, 0.5, 0.3, 0.7]
        conn = spectrum(prov, values=values)
        median = conn.reduce_freq((6.0, 12.0), how="median").values[0, 0, 0, 0]
        np.testing.assert_allclose(median, np.median([0.9, 0.2, 0.5, 0.3]))
        assert conn.reduce_freq((6.0, 12.0), how="max").values[0, 0, 0, 0] == 0.9

    def test_peak_freq_returns_the_frequency_of_the_maximum(self, prov):
        values = np.zeros((2, 3, 6, 1))
        values[0, 0, :, 0] = [0.1, 0.9, 0.2, 0.5, 0.3, 0.7]
        values[1, 2, :, 0] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        out = spectrum(prov, values=values).reduce_freq((4.0, 14.0), how="peak_freq")
        assert out.values[0, 0, 0, 0] == 6.0
        assert out.values[1, 2, 0, 0] == 12.0

    def test_peak_freq_picks_the_first_frequency_on_ties(self, prov):
        values = np.zeros((2, 3, 6, 1))
        values[0, 0, :, 0] = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        out = spectrum(prov, values=values).reduce_freq((4.0, 14.0), how="peak_freq")
        assert out.values[0, 0, 0, 0] == 6.0

    def test_complex_mean_is_taken_in_the_complex_plane(self, prov):
        z = np.zeros((2, 3, 6, 1), dtype=complex)
        z[0, 0, :, 0] = [1, -1, 1j, -1j, 1, 1]
        out = spectrum(prov, values=z).reduce_freq((4.0, 8.0), how="mean")
        # |mean(1, -1, 1j)| = 1/3, whereas mean(|.|) would be 1
        assert out.values[0, 0, 0, 0] == pytest.approx(1j / 3)

    def test_complex_max_keeps_the_value_with_the_largest_magnitude(self, prov):
        z = np.zeros((2, 3, 6, 1), dtype=complex)
        z[0, 0, :, 0] = [0.1, -0.2j, 3 - 4j, 1, 0, 0]
        out = spectrum(prov, values=z).reduce_freq((4.0, 14.0), how="max")
        assert out.values[0, 0, 0, 0] == 3 - 4j
        assert out.is_complex

    def test_complex_median_is_componentwise(self, prov):
        z = np.zeros((2, 3, 6, 1), dtype=complex)
        z[0, 0, :3, 0] = [1 + 10j, 2 + 20j, 100 + 30j]
        out = spectrum(prov, values=z).reduce_freq((4.0, 8.0), how="median")
        assert out.values[0, 0, 0, 0] == 2 + 20j

    def test_complex_peak_freq_uses_the_magnitude_and_returns_real_hz(self, prov):
        z = np.zeros((2, 3, 6, 1), dtype=complex)
        z[0, 0, :, 0] = [0.1, 0.2, 0.0, -3j, 1.0, 0.0]
        out = spectrum(prov, values=z).reduce_freq((4.0, 14.0), how="peak_freq")
        assert out.values[0, 0, 0, 0] == 10.0
        assert not out.is_complex

    @pytest.mark.parametrize("how", ["mean", "median", "max", "peak_freq"])
    def test_undefined_entries_stay_nan_without_warnings(self, prov, how):
        values = np.ones((2, 3, 6, 1))
        values[0, 0, :, :] = np.nan
        values[1, 1, 2, 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = spectrum(prov, values=values).reduce_freq((4.0, 14.0), how=how)
        assert np.isnan(out.values[0, 0, 0, 0])
        assert np.isfinite(out.values[1, 1, 0, 0])

    def test_time_axis_is_preserved(self, prov):
        conn = spectrum(prov, n_times=4)
        out = conn.reduce_freq((4.0, 8.0))
        assert out.values.shape == (2, 3, 1, 4)
        np.testing.assert_allclose(out.times, conn.times)

    def test_provenance_records_the_step_and_the_original_is_untouched(self, prov):
        conn = spectrum(prov)
        out = conn.reduce_freq((6.0, 10.0), how="peak_freq")
        assert conn.provenance.history == ()
        step = out.provenance.history[-1]
        assert step["op"] == "reduce_freq"
        assert step["band"] == [6.0, 10.0]
        assert step["how"] == "peak_freq"
        assert step["values_unit"] == "Hz"
        assert out.provenance.method == "pdc"

    def test_not_frequency_resolved_data_cannot_be_reduced(self, prov):
        conn = Connectivity.from_array(np.ones((2, 2)), source=["A", "B"], provenance=prov)
        with pytest.raises(ValueError, match="frequency"):
            conn.reduce_freq((1.0, 4.0))

    def test_band_without_frequencies_is_rejected(self, prov):
        with pytest.raises(ValueError, match="No frequencies"):
            spectrum(prov).reduce_freq((20.0, 30.0))

    def test_invalid_band_is_rejected(self, prov):
        conn = spectrum(prov)
        with pytest.raises(ValueError, match="band"):
            conn.reduce_freq((10.0, 6.0))
        with pytest.raises(ValueError, match="band"):
            conn.reduce_freq((6.0,))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="band"):
            conn.reduce_freq((np.nan, 6.0))

    def test_unknown_reduction_is_rejected(self, prov):
        with pytest.raises(ValueError, match="how"):
            spectrum(prov).reduce_freq((6.0, 10.0), how="sum")


class TestNetcdf:
    def test_real_round_trip(self, prov, tmp_path):
        conn = spectrum(prov, n_times=3)
        path = tmp_path / "result.nc"
        conn.to_netcdf(path)
        loaded = Connectivity.from_netcdf(path)
        np.testing.assert_array_equal(loaded.values, conn.values)
        assert loaded.source == conn.source
        assert loaded.target == conn.target
        np.testing.assert_array_equal(loaded.freqs, conn.freqs)
        np.testing.assert_array_equal(loaded.times, conn.times)
        assert loaded.provenance == conn.provenance

    def test_nan_values_and_nan_coordinates_survive(self, prov, tmp_path):
        values = np.ones((2, 2))
        values[0, 0] = np.nan
        conn = Connectivity.from_array(values, source=["A", "B"], provenance=prov)
        conn.to_netcdf(tmp_path / "r.nc")
        loaded = Connectivity.from_netcdf(str(tmp_path / "r.nc"))
        assert np.isnan(loaded.values[0, 0, 0, 0])
        assert np.isnan(loaded.freqs).all() and np.isnan(loaded.times).all()
        assert not loaded.is_frequency_resolved

    def test_complex_round_trip(self, prov, tmp_path):
        rng = np.random.default_rng(0)
        z = rng.standard_normal((2, 3, 6, 1)) + 1j * rng.standard_normal((2, 3, 6, 1))
        conn = spectrum(prov, values=z)
        conn.to_netcdf(tmp_path / "c.nc")
        loaded = Connectivity.from_netcdf(tmp_path / "c.nc")
        assert loaded.is_complex
        np.testing.assert_array_equal(loaded.values, z)

    def test_provenance_history_round_trips(self, prov, tmp_path):
        conn = spectrum(prov).reduce_freq((6.0, 10.0))
        conn.to_netcdf(tmp_path / "h.nc")
        assert Connectivity.from_netcdf(tmp_path / "h.nc").provenance == conn.provenance

    def test_file_is_a_plain_netcdf_readable_with_xarray(self, prov, tmp_path):
        spectrum(prov).to_netcdf(tmp_path / "p.nc")
        with xr.open_dataset(tmp_path / "p.nc") as ds:
            assert ds.attrs["dirconn_format_version"] == "1"
            assert "provenance" in ds.attrs
            assert ds["values"].dims == DIMS

    def test_file_without_provenance_is_rejected(self, tmp_path):
        ds = xr.Dataset(
            {"values": (DIMS, np.ones((1, 1, 1, 1)))},
            coords={"source": ["A"], "target": ["B"], "freq": [np.nan], "time": [np.nan]},
            attrs={"dirconn_format_version": "1"},
        )
        ds.to_netcdf(tmp_path / "bad.nc")
        with pytest.raises(ValueError, match="provenance"):
            Connectivity.from_netcdf(tmp_path / "bad.nc")

    def test_unknown_format_version_is_rejected(self, prov, tmp_path):
        spectrum(prov).to_netcdf(tmp_path / "v.nc")
        with xr.open_dataset(tmp_path / "v.nc") as ds:
            loaded = ds.load()
        loaded.attrs["dirconn_format_version"] = "99"
        loaded.to_netcdf(tmp_path / "v99.nc")
        with pytest.raises(ValueError, match="format version"):
            Connectivity.from_netcdf(tmp_path / "v99.nc")

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Connectivity.from_netcdf(tmp_path / "nope.nc")
