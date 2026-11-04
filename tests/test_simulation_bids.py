import json

import numpy as np
import pytest

from dirconn import (
    Edge,
    Network,
    NodeDynamics,
    direct_node,
    dk_node,
    read_bids,
    read_simulation_bids,
    simulate,
    write_simulation_bids,
)
from dirconn.bids import SimulationRecord

A, B = "bankssts-lh", "superiorparietal-lh"
N_SENSORS = 6


def make_sim(n_trials=1, with_direct=True, seed=0):
    nodes = [dk_node(A), dk_node(B)]
    if with_direct:
        nodes.append(direct_node("STN-L", region="STN", hemisphere="lh"))
    net = Network(nodes=nodes, edges=[Edge(A, B, strength=0.3, delay_ms=8.0)])
    return simulate(
        net,
        sfreq=250.0,
        n_times=400,
        n_trials=n_trials,
        seed=seed,
        n_sensors=N_SENSORS,
        n_background=3,
        dynamics={A: NodeDynamics(freq_hz=10.0), B: NodeDynamics(freq_hz=20.0)},
    )


def write(sim, root, **kwargs):
    options = dict(subject="sim01", task="sim")
    options.update(kwargs)
    return write_simulation_bids(sim, root, **options)


def read(root, **kwargs):
    options = dict(subject="sim01", task="sim")
    options.update(kwargs)
    return read_simulation_bids(root, **options)


def sensor_values(sim):
    return sim.data.data[:, :N_SENSORS]


class TestWrite:
    def test_creates_a_raw_sensor_dataset_and_a_derivatives_dataset(self, tmp_path):
        paths = write(make_sim(), tmp_path)
        assert (tmp_path / "sub-sim01" / "eeg" / "sub-sim01_task-sim_eeg.vhdr").is_file()
        derivatives = tmp_path / "derivatives" / "dirconn-simulation"
        description = json.loads((derivatives / "dataset_description.json").read_text())
        assert description["DatasetType"] == "derivative"
        for key in ("raw", "sources", "direct", "truth", "leadfield", "simulation"):
            assert paths[key].exists(), key

    def test_derivative_file_names(self, tmp_path):
        paths = write(make_sim(), tmp_path)
        eeg = tmp_path / "derivatives" / "dirconn-simulation" / "sub-sim01" / "eeg"
        assert paths["sources"] == eeg / "sub-sim01_task-sim_desc-sources_eeg.fif"
        assert paths["direct"] == eeg / "sub-sim01_task-sim_desc-direct_eeg.fif"
        assert paths["truth"] == eeg / "sub-sim01_task-sim_desc-truth_connectivity.nc"
        assert paths["leadfield"] == eeg / "sub-sim01_task-sim_desc-leadfield_leadfield.nc"
        assert paths["simulation"] == eeg / "sub-sim01_task-sim_desc-simulation_simulation.json"

    def test_continuous_data_have_no_events_and_no_segmentation_entry(self, tmp_path):
        write(make_sim(n_trials=1), tmp_path)
        raw_dir = tmp_path / "sub-sim01" / "eeg"
        assert not list(raw_dir.glob("*_events.tsv"))
        sidecar = json.loads((raw_dir / "sub-sim01_task-sim_eeg.json").read_text())
        assert "Dirconn" not in sidecar

    def test_epoched_data_are_described_by_events_and_the_sidecar(self, tmp_path):
        write(make_sim(n_trials=3), tmp_path)
        raw_dir = tmp_path / "sub-sim01" / "eeg"
        events = (raw_dir / "sub-sim01_task-sim_events.tsv").read_text().splitlines()
        assert len(events) == 1 + 3
        meta = json.loads((raw_dir / "sub-sim01_task-sim_eeg.json").read_text())["Dirconn"]
        assert meta["Segmentation"] == "epoched"
        assert meta["NTrials"] == 3
        assert meta["EpochLength"] == 400
        assert meta["Tmin"] == 0.0

    def test_without_direct_nodes_no_direct_files_are_written(self, tmp_path):
        paths = write(make_sim(with_direct=False), tmp_path)
        assert "direct" not in paths
        assert not list((tmp_path / "derivatives").rglob("*desc-direct*"))

    def test_existing_data_are_not_overwritten_by_default(self, tmp_path):
        write(make_sim(), tmp_path)
        with pytest.raises(FileExistsError):
            write(make_sim(seed=1), tmp_path)

    def test_overwrite_replaces_everything(self, tmp_path):
        write(make_sim(seed=0), tmp_path)
        second = make_sim(seed=1)
        write(second, tmp_path, overwrite=True)
        record = read(tmp_path)
        np.testing.assert_array_equal(record.sources.data, second.sources.data)
        assert record.provenance.seed == 1

    def test_a_failed_overwrite_check_leaves_the_raw_dataset_untouched(self, tmp_path):
        write(make_sim(seed=0), tmp_path)
        raw_file = tmp_path / "sub-sim01" / "eeg" / "sub-sim01_task-sim_eeg.eeg"
        before = raw_file.read_bytes()
        with pytest.raises(FileExistsError):
            write(make_sim(seed=1), tmp_path)
        assert raw_file.read_bytes() == before

    def test_session_and_run_are_part_of_the_names(self, tmp_path):
        write(make_sim(), tmp_path, session="01", run="02")
        assert (
            tmp_path / "sub-sim01" / "ses-01" / "eeg" / "sub-sim01_ses-01_task-sim_run-02_eeg.vhdr"
        ).is_file()
        record = read(tmp_path, session="01", run="02")
        assert record.sources.names[0] == A


class TestRoundTrip:
    def test_continuous_sensor_data_come_back_in_volts(self, tmp_path):
        sim = make_sim()
        write(sim, tmp_path)
        record = read(tmp_path)
        expected = sensor_values(sim) * 1e-6  # one simulated unit is written as one microvolt
        assert record.sensors.segmentation == "continuous"
        assert record.sensors.names == ("S01", "S02", "S03", "S04", "S05", "S06")
        assert {ch.type for ch in record.sensors.channels} == {"eeg"}
        assert record.sensors.sfreq == 250.0
        np.testing.assert_allclose(
            record.sensors.data, expected, rtol=0, atol=1e-6 * np.abs(expected).max()
        )

    def test_epoched_sensor_data_are_restored_as_epochs(self, tmp_path):
        sim = make_sim(n_trials=3)
        write(sim, tmp_path)
        record = read(tmp_path)
        expected = sensor_values(sim) * 1e-6
        assert record.sensors.segmentation == "epoched"
        assert record.sensors.data.shape == (3, N_SENSORS, 400)
        np.testing.assert_allclose(
            record.sensors.data, expected, rtol=0, atol=1e-6 * np.abs(expected).max()
        )

    def test_the_standard_reader_also_restores_epochs(self, tmp_path):
        write(make_sim(n_trials=2), tmp_path)
        loaded = read_bids(tmp_path, subject="sim01", task="sim")
        assert loaded.segmentation == "epoched"
        assert loaded.data.shape == (2, N_SENSORS, 400)

    def test_sources_and_direct_channels_are_exact(self, tmp_path):
        sim = make_sim(n_trials=2)
        write(sim, tmp_path)
        record = read(tmp_path)
        np.testing.assert_array_equal(record.sources.data, sim.sources.data)
        assert record.sources.channels == sim.sources.channels
        assert record.sources.segmentation == "epoched"
        assert record.direct is not None
        np.testing.assert_array_equal(record.direct.data, sim.data.data[:, N_SENSORS:])
        assert record.direct.names == ("STN-L",)
        assert record.direct.channels[0].node_kind == "direct"

    def test_without_direct_nodes_the_record_has_no_direct_data(self, tmp_path):
        write(make_sim(with_direct=False), tmp_path)
        assert read(tmp_path).direct is None

    def test_truth_leadfield_network_and_provenance_round_trip(self, tmp_path):
        sim = make_sim()
        write(sim, tmp_path)
        record = read(tmp_path)
        assert isinstance(record, SimulationRecord)
        np.testing.assert_array_equal(record.truth.values, sim.truth.values)
        assert record.truth.source == sim.truth.source
        np.testing.assert_array_equal(record.leadfield, sim.leadfield)
        assert record.network == sim.network
        assert record.provenance == sim.provenance

    def test_truth_can_be_rebuilt_on_any_frequency_axis(self, tmp_path):
        sim = make_sim()
        write(sim, tmp_path)
        record = read(tmp_path)
        freqs = [5.0, 10.0, 20.0]
        np.testing.assert_array_equal(
            record.truth_at(freqs=freqs).values, sim.truth_at(freqs=freqs).values
        )

    def test_missing_data_raise_file_not_found(self, tmp_path):
        write(make_sim(), tmp_path)
        with pytest.raises(FileNotFoundError):
            read(tmp_path, subject="other")
        with pytest.raises(FileNotFoundError):
            read(tmp_path / "nowhere")
