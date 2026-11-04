import json
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
from mne_bids import BIDSPath, write_raw_bids

import dirconn
from dirconn import ChannelMeta, TimeSeries, read_bids, write_roi_bids

ROI_CHANNELS = [
    ChannelMeta(
        "inferiortemporal-lh",
        type="misc",
        node_kind="cortical",
        region="inferiortemporal",
        hemisphere="lh",
        atlas="DK",
    ),
    ChannelMeta(
        "superiorparietal-rh",
        type="misc",
        node_kind="cortical",
        region="superiorparietal",
        hemisphere="rh",
        atlas="DK",
    ),
    ChannelMeta("STN-L", type="misc", node_kind="direct", region="STN", hemisphere="lh"),
]


def roi_series(n_trials=1, n_times=120, segmentation=None, tmin=0.0, seed=0, channels=None):
    rng = np.random.default_rng(seed)
    channels = channels or ROI_CHANNELS
    data = rng.standard_normal((n_trials, len(channels), n_times))
    if segmentation is None:
        segmentation = "continuous" if n_trials == 1 else "epoched"
    return TimeSeries.from_array(
        data, sfreq=250.0, channels=channels, segmentation=segmentation, tmin=tmin
    )


def write_raw_dataset(root, *, datatype="eeg", subject="01", task="rest", seed=0):
    rng = np.random.default_rng(seed)
    names = ["Fz", "Cz", "Pz"]
    data = rng.standard_normal((3, 600)) * 1e-6
    raw = mne.io.RawArray(data, mne.create_info(names, 100.0, "eeg"), verbose="error")
    raw.info["line_freq"] = 50.0
    path = BIDSPath(subject=subject, task=task, root=root, datatype=datatype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_raw_bids(
            raw, path, format="BrainVision", allow_preload=True, overwrite=True, verbose="error"
        )
    return data, names


class TestWriteRoi:
    def test_creates_a_derivative_dataset_with_the_expected_files(self, tmp_path):
        path = write_roi_bids(roi_series(), tmp_path, subject="01", task="rest", description="dk")
        assert isinstance(path, BIDSPath)
        base = tmp_path / "sub-01" / "eeg" / "sub-01_task-rest_desc-dk"
        for suffix in ("_eeg.fif", "_eeg.json", "_channels.tsv", "_channels.json"):
            assert Path(str(base) + suffix).is_file(), suffix
        description = json.loads((tmp_path / "dataset_description.json").read_text())
        assert description["DatasetType"] == "derivative"
        assert description["GeneratedBy"][0]["Name"] == "dirconn"
        assert description["GeneratedBy"][0]["Version"] == dirconn.__version__

    def test_events_file_is_written_only_for_epoched_data(self, tmp_path):
        write_roi_bids(roi_series(), tmp_path / "a", subject="01", task="rest", description="dk")
        assert not list((tmp_path / "a").rglob("*_events.tsv"))
        series = roi_series(n_trials=4, n_times=100)
        write_roi_bids(series, tmp_path / "b", subject="01", task="rest", description="dk")
        events = pd.read_csv(
            tmp_path / "b" / "sub-01" / "eeg" / "sub-01_task-rest_desc-dk_events.tsv", sep="\t"
        )
        assert list(events.columns) == ["onset", "duration", "trial_type"]
        np.testing.assert_allclose(events["onset"], np.arange(4) * 100 / 250.0)
        np.testing.assert_allclose(events["duration"], 100 / 250.0)
        assert set(events["trial_type"]) == {"epoch"}

    def test_channels_tsv_carries_the_roi_columns(self, tmp_path):
        write_roi_bids(roi_series(), tmp_path, subject="01", task="rest", description="dk")
        table = pd.read_csv(
            tmp_path / "sub-01" / "eeg" / "sub-01_task-rest_desc-dk_channels.tsv",
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )
        assert list(table.columns[:3]) == ["name", "type", "units"]
        assert {"region", "hemisphere", "atlas", "node_kind"} <= set(table.columns)
        assert list(table["name"]) == ["inferiortemporal-lh", "superiorparietal-rh", "STN-L"]
        assert set(table["type"]) == {"MISC"}
        assert list(table["node_kind"]) == ["cortical", "cortical", "direct"]
        assert table.loc[2, "atlas"] == "n/a"

    def test_channels_json_documents_the_extra_columns(self, tmp_path):
        write_roi_bids(roi_series(), tmp_path, subject="01", task="rest", description="dk")
        sidecar = json.loads(
            (tmp_path / "sub-01" / "eeg" / "sub-01_task-rest_desc-dk_channels.json").read_text()
        )
        for column in ("region", "hemisphere", "atlas", "node_kind"):
            assert "Description" in sidecar[column]
        assert set(sidecar["node_kind"]["Levels"]) == {"sensor", "cortical", "direct"}

    def test_sidecar_states_the_convention_and_the_segmentation(self, tmp_path):
        series = roi_series(n_trials=3, n_times=80, tmin=-0.1)
        write_roi_bids(
            series,
            tmp_path,
            subject="01",
            task="rest",
            description="dk",
            source_space="surface",
            inverse_method="dSPM",
            roi_extraction="mean_flip",
            source_orientation="fixed",
            leakage_correction="symmetric orthogonalisation",
        )
        sidecar = json.loads(
            (tmp_path / "sub-01" / "eeg" / "sub-01_task-rest_desc-dk_eeg.json").read_text()
        )
        assert sidecar["TaskName"] == "rest"
        assert sidecar["SamplingFrequency"] == 250.0
        meta = sidecar["Dirconn"]
        assert meta["Convention"] == "0"
        assert meta["Atlas"] == "DK"
        assert meta["Segmentation"] == "epoched"
        assert meta["NTrials"] == 3
        assert meta["EpochLength"] == 80
        assert meta["Tmin"] == -0.1
        assert meta["SourceSpace"] == "surface"
        assert meta["InverseMethod"] == "dSPM"
        assert meta["RoiExtraction"] == "mean_flip"
        assert meta["SourceOrientation"] == "fixed"
        assert meta["LeakageCorrection"] == "symmetric orthogonalisation"

    def test_unset_processing_fields_are_left_out_of_the_sidecar(self, tmp_path):
        write_roi_bids(roi_series(), tmp_path, subject="01", task="rest", description="dk")
        sidecar = json.loads(
            (tmp_path / "sub-01" / "eeg" / "sub-01_task-rest_desc-dk_eeg.json").read_text()
        )
        assert "InverseMethod" not in sidecar["Dirconn"]
        assert "SourceSpace" not in sidecar["Dirconn"]

    def test_several_atlases_are_listed(self, tmp_path):
        channels = [
            ChannelMeta("a", node_kind="cortical", atlas="DK"),
            ChannelMeta("b", node_kind="cortical", atlas="HCPMMP"),
        ]
        write_roi_bids(
            roi_series(channels=channels), tmp_path, subject="01", task="rest", description="x"
        )
        sidecar = json.loads(
            (tmp_path / "sub-01" / "eeg" / "sub-01_task-rest_desc-x_eeg.json").read_text()
        )
        assert sidecar["Dirconn"]["Atlas"] == ["DK", "HCPMMP"]

    def test_existing_data_is_not_overwritten_by_default(self, tmp_path):
        write_roi_bids(roi_series(), tmp_path, subject="01", task="rest", description="dk")
        with pytest.raises(FileExistsError):
            write_roi_bids(
                roi_series(seed=1), tmp_path, subject="01", task="rest", description="dk"
            )
        write_roi_bids(
            roi_series(seed=1),
            tmp_path,
            subject="01",
            task="rest",
            description="dk",
            overwrite=True,
        )
        loaded = read_bids(tmp_path, subject="01", task="rest", level="roi", description="dk")
        np.testing.assert_array_equal(loaded.data, roi_series(seed=1).data)

    def test_meg_datatype_uses_the_meg_directory(self, tmp_path):
        write_roi_bids(
            roi_series(), tmp_path, subject="01", task="rest", description="dk", datatype="meg"
        )
        assert (tmp_path / "sub-01" / "meg" / "sub-01_task-rest_desc-dk_meg.fif").is_file()

    def test_session_and_run_are_part_of_the_path(self, tmp_path):
        write_roi_bids(
            roi_series(),
            tmp_path,
            subject="01",
            session="02",
            run="03",
            task="rest",
            description="dk",
        )
        expected = "sub-01/ses-02/eeg/sub-01_ses-02_task-rest_run-03_desc-dk_eeg.fif"
        assert (tmp_path / expected).is_file()

    def test_run_entities_are_formatted_by_mne_bids_without_padding(self, tmp_path):
        write_roi_bids(roi_series(), tmp_path, subject="01", run=3, task="rest", description="dk")
        assert (tmp_path / "sub-01/eeg/sub-01_task-rest_run-3_desc-dk_eeg.fif").is_file()

    def test_unknown_datatype_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="datatype"):
            write_roi_bids(
                roi_series(), tmp_path, subject="01", task="rest", description="dk", datatype="fmri"
            )

    def test_description_is_required(self, tmp_path):
        with pytest.raises(ValueError, match="description"):
            write_roi_bids(roi_series(), tmp_path, subject="01", task="rest", description="")


class TestReadRoi:
    def read(self, root, **kwargs):
        options = dict(subject="01", task="rest", level="roi", description="dk")
        options.update(kwargs)
        return read_bids(root, **options)

    def test_continuous_round_trip_is_exact(self, tmp_path):
        series = roi_series()
        write_roi_bids(series, tmp_path, subject="01", task="rest", description="dk")
        loaded = self.read(tmp_path)
        assert loaded.segmentation == "continuous"
        assert loaded.sfreq == 250.0
        assert loaded.tmin == 0.0
        np.testing.assert_array_equal(loaded.data, series.data)

    def test_epoched_round_trip_is_exact_and_keeps_tmin(self, tmp_path):
        series = roi_series(n_trials=5, n_times=90, tmin=-0.2)
        write_roi_bids(series, tmp_path, subject="01", task="rest", description="dk")
        loaded = self.read(tmp_path)
        assert loaded.segmentation == "epoched"
        assert loaded.data.shape == (5, 3, 90)
        assert loaded.tmin == pytest.approx(-0.2)
        np.testing.assert_array_equal(loaded.data, series.data)

    def test_a_single_epoch_stays_epoched(self, tmp_path):
        series = roi_series(n_trials=1, segmentation="epoched")
        write_roi_bids(series, tmp_path, subject="01", task="rest", description="dk")
        assert self.read(tmp_path).segmentation == "epoched"

    def test_channel_metadata_round_trips(self, tmp_path):
        series = roi_series()
        write_roi_bids(series, tmp_path, subject="01", task="rest", description="dk")
        assert self.read(tmp_path).channels == series.channels

    def test_session_run_and_meg_datatype_are_found_again(self, tmp_path):
        series = roi_series()
        write_roi_bids(
            series,
            tmp_path,
            subject="01",
            session="02",
            run=1,
            task="rest",
            description="dk",
            datatype="meg",
        )
        loaded = self.read(tmp_path, session="02", run=1, datatype="meg")
        np.testing.assert_array_equal(loaded.data, series.data)

    def test_datatype_is_inferred_when_only_one_exists(self, tmp_path):
        series = roi_series()
        write_roi_bids(
            series, tmp_path, subject="01", task="rest", description="dk", datatype="meg"
        )
        np.testing.assert_array_equal(self.read(tmp_path).data, series.data)

    def test_ambiguous_datatype_must_be_given(self, tmp_path):
        for datatype in ("eeg", "meg"):
            write_roi_bids(
                roi_series(),
                tmp_path,
                subject="01",
                task="rest",
                description="dk",
                datatype=datatype,
            )
        with pytest.raises(ValueError, match="datatype"):
            self.read(tmp_path)

    def test_a_file_without_the_convention_entry_is_rejected(self, tmp_path):
        write_roi_bids(roi_series(), tmp_path, subject="01", task="rest", description="dk")
        sidecar_path = tmp_path / "sub-01" / "eeg" / "sub-01_task-rest_desc-dk_eeg.json"
        sidecar = json.loads(sidecar_path.read_text())
        del sidecar["Dirconn"]
        sidecar_path.write_text(json.dumps(sidecar))
        with pytest.raises(ValueError, match="Dirconn"):
            self.read(tmp_path)

    def test_an_unknown_convention_version_is_rejected(self, tmp_path):
        write_roi_bids(roi_series(), tmp_path, subject="01", task="rest", description="dk")
        sidecar_path = tmp_path / "sub-01" / "eeg" / "sub-01_task-rest_desc-dk_eeg.json"
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["Dirconn"]["Convention"] = "99"
        sidecar_path.write_text(json.dumps(sidecar))
        with pytest.raises(ValueError, match="convention"):
            self.read(tmp_path)

    def test_missing_data_raises_file_not_found(self, tmp_path):
        write_roi_bids(roi_series(), tmp_path, subject="01", task="rest", description="dk")
        with pytest.raises(FileNotFoundError):
            self.read(tmp_path, description="other")
        with pytest.raises(FileNotFoundError):
            self.read(tmp_path, subject="02", datatype="eeg")

    def test_channel_table_must_match_the_data(self, tmp_path):
        write_roi_bids(roi_series(), tmp_path, subject="01", task="rest", description="dk")
        table_path = tmp_path / "sub-01" / "eeg" / "sub-01_task-rest_desc-dk_channels.tsv"
        table = pd.read_csv(table_path, sep="\t", dtype=str, keep_default_na=False)
        table["name"] = ["x", "y", "z"]
        table.to_csv(table_path, sep="\t", index=False)
        with pytest.raises(ValueError, match="channels.tsv"):
            self.read(tmp_path)


class TestReadSensor:
    def test_raw_bids_dataset_is_read_as_continuous_time_series(self, tmp_path):
        data, names = write_raw_dataset(tmp_path)
        loaded = read_bids(tmp_path, subject="01", task="rest")
        assert loaded.segmentation == "continuous"
        assert loaded.sfreq == 100.0
        assert loaded.names == tuple(names)
        np.testing.assert_allclose(loaded.data[0], data, rtol=1e-5, atol=1e-12)

    def test_datatype_can_be_given_explicitly(self, tmp_path):
        write_raw_dataset(tmp_path)
        assert read_bids(tmp_path, subject="01", task="rest", datatype="eeg").n_channels == 3

    def test_missing_subject_raises_file_not_found(self, tmp_path):
        write_raw_dataset(tmp_path)
        with pytest.raises(FileNotFoundError):
            read_bids(tmp_path, subject="02", task="rest", datatype="eeg")


class TestReadValidation:
    def test_unknown_level_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="level"):
            read_bids(tmp_path, subject="01", task="rest", level="source")

    def test_roi_level_needs_a_description(self, tmp_path):
        with pytest.raises(ValueError, match="description"):
            read_bids(tmp_path, subject="01", task="rest", level="roi")

    def test_unknown_datatype_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="datatype"):
            read_bids(tmp_path, subject="01", task="rest", datatype="fmri")
