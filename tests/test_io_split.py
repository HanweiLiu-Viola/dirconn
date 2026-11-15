import subprocess
import sys
from pathlib import Path

import mne
import numpy as np

import dirconn
from dirconn import Connectivity, Provenance, TimeSeries
from dirconn.io.mne import timeseries_from_mne
from dirconn.io.netcdf import read_connectivity, write_connectivity

FIXTURE = Path(__file__).parent / "data" / "connectivity_main_0e2b04e.nc"


def _conn(complex_values=False):
    values = np.arange(2 * 2 * 3 * 2, dtype=float).reshape(2, 2, 3, 2)
    if complex_values:
        values = values + 1j * values[::-1]
    return Connectivity.from_array(
        values,
        source=["A", "B"],
        freqs=np.array([4.0, 8.0, 12.0]),
        times=np.array([0.0, 0.5]),
        provenance=Provenance(method="pdc", parameters={"order": 3}, model_order=3, seed=0),
    )


def test_netcdf_round_trip_matches_method(tmp_path):
    for is_complex in (False, True):
        conn = _conn(is_complex)
        path = tmp_path / f"c{is_complex}.nc"
        write_connectivity(conn, path)
        direct = read_connectivity(path)
        method = Connectivity.from_netcdf(path)
        assert np.array_equal(direct.values, method.values)
        assert np.array_equal(direct.values, conn.values)
        assert direct.provenance.to_dict() == method.provenance.to_dict()
        assert direct.source == method.source


def test_to_netcdf_method_delegates(tmp_path):
    conn = _conn()
    conn.to_netcdf(tmp_path / "a.nc")
    write_connectivity(conn, tmp_path / "b.nc")
    assert np.array_equal(
        read_connectivity(tmp_path / "a.nc").values, read_connectivity(tmp_path / "b.nc").values
    )


def test_pre_restructure_fixture_loads():
    conn = read_connectivity(FIXTURE)
    assert np.array_equal(conn.values, Connectivity.from_netcdf(FIXTURE).values, equal_nan=True)
    assert conn.provenance.method == "gc"


def test_timeseries_from_mne_matches_method():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((3, 200)) * 1e-6
    info = mne.create_info(["Fz", "Cz", "Pz"], 100.0, "eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    a = timeseries_from_mne(raw)
    b = TimeSeries.from_mne(raw)
    assert np.array_equal(a.data, b.data)
    assert a.names == b.names
    assert a.sfreq == b.sfreq
    assert a.segmentation == b.segmentation


def test_simulation_bids_names_are_the_same_objects():
    import dirconn.io.bids as bids
    import dirconn.io.simulation_bids as sb

    assert sb.write_simulation_bids is dirconn.write_simulation_bids
    assert sb.read_simulation_bids is dirconn.read_simulation_bids
    assert sb.SimulationRecord is dirconn.SimulationRecord
    assert bids.write_simulation_bids is sb.write_simulation_bids
    assert bids.SimulationRecord is sb.SimulationRecord


def test_data_model_does_not_load_io_modules_eagerly():
    code = (
        "import sys, dirconn\n"
        "assert 'dirconn.io.netcdf' not in sys.modules\n"
        "assert 'dirconn.io.mne' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_import_order_of_bids_modules():
    for first, second in (("simulation_bids", "bids"), ("bids", "simulation_bids")):
        code = f"import dirconn.io.{first}; import dirconn.io.{second}"
        subprocess.run([sys.executable, "-c", code], check=True)
