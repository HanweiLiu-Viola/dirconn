import numpy as np
import pytest

import dirconn
from dirconn import BANDS, Connectivity, Provenance


def spectrum():
    freqs = np.arange(0.0, 101.0, 2.0)  # 0, 2, ..., 100 Hz
    rng = np.random.default_rng(0)
    values = rng.random((2, 2, len(freqs), 1))
    return Connectivity.from_array(
        values, source=["a", "b"], freqs=freqs, provenance=Provenance(method="pdc")
    )


def test_the_named_bands_are_the_documented_ones():
    assert dict(BANDS) == {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 80.0),
    }


def test_the_bands_are_valid_ordered_ranges_and_cannot_be_changed():
    edges = list(BANDS.values())
    assert all(0 <= low < high for low, high in edges)
    assert edges == sorted(edges)
    with pytest.raises(TypeError):
        BANDS["beta"] = (12.0, 35.0)  # type: ignore[index]


def test_bands_are_exported_at_the_top_level():
    assert dirconn.BANDS is BANDS


def test_a_band_name_is_the_same_as_its_edges():
    conn = spectrum()
    by_name = conn.reduce_freq("beta")
    by_edges = conn.reduce_freq((13.0, 30.0))
    np.testing.assert_array_equal(by_name.values, by_edges.values)
    np.testing.assert_array_equal(by_name.freqs, by_edges.freqs)


def test_the_provenance_records_the_name_and_the_edges():
    step = spectrum().reduce_freq("alpha", how="max").provenance.history[-1]
    assert step["band"] == [8.0, 13.0]
    assert step["band_name"] == "alpha"
    assert step["how"] == "max"


def test_explicit_edges_leave_no_band_name():
    step = spectrum().reduce_freq((8.0, 13.0)).provenance.history[-1]
    assert "band_name" not in step


def test_names_are_not_case_sensitive():
    conn = spectrum()
    np.testing.assert_array_equal(conn.reduce_freq("Beta").values, conn.reduce_freq("beta").values)
    assert conn.reduce_freq(" GAMMA ").provenance.history[-1]["band_name"] == "gamma"


def test_an_unknown_name_lists_the_known_ones():
    with pytest.raises(ValueError, match="theta") as error:
        spectrum().reduce_freq("sigma")
    assert "sigma" in str(error.value)
