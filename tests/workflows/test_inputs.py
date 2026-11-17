"""Tests for BIDS input discovery."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from dirconn.workflows.inputs import (
    InputDiscoveryError,
    InputUnit,
    derivatives_root,
    discover,
)

ROI_LABELS = [
    "sub-01_task-move",
    "sub-01_task-rest_run-1",
    "sub-01_task-rest_run-2",
    "sub-02_ses-01_task-rest",
    "sub-02_ses-02_task-move",
]


def _roi(bids: Path, **kwargs: object) -> list[InputUnit]:
    return discover(bids, level="roi", datatypes=["eeg"], description="DK", **kwargs)  # type: ignore[arg-type]


def test_roi_units_are_found(bids_tiny: Path) -> None:
    units = _roi(bids_tiny)
    assert [u.label() for u in units] == ROI_LABELS
    first = units[0]
    assert first.level == "roi"
    assert first.datatype == "eeg"
    assert first.root == derivatives_root(bids_tiny)
    assert [f.suffix for f in first.files] == [".fif", ".json", ".tsv"]
    assert all(f.is_absolute() and f.exists() for f in first.files)
    assert len(first.sha256) == 64


def test_sessions_and_runs_become_separate_units(bids_tiny: Path) -> None:
    units = {u.label(): u for u in _roi(bids_tiny)}
    assert units["sub-02_ses-01_task-rest"].entities() == {
        "sub": "02",
        "ses": "01",
        "task": "rest",
    }
    assert units["sub-01_task-rest_run-1"].entities() == {
        "sub": "01",
        "task": "rest",
        "run": "1",
    }
    assert units["sub-01_task-rest_run-1"].files != units["sub-01_task-rest_run-2"].files


def test_sensor_level_finds_the_eeg_recording(bids_tiny: Path) -> None:
    units = discover(bids_tiny, level="sensor", datatypes=["eeg"])
    assert [u.label() for u in units] == ["sub-01_task-rest"]
    assert units[0].root == bids_tiny
    assert units[0].files[0].suffix == ".vhdr"
    assert units[0].level == "sensor"


def test_participant_label_filters_with_or_without_prefix(bids_tiny: Path) -> None:
    bare = _roi(bids_tiny, participant_labels=["02"])
    prefixed = _roi(bids_tiny, participant_labels=["sub-02"])
    assert [u.label() for u in bare] == [u.label() for u in prefixed]
    assert [u.label() for u in bare] == ROI_LABELS[3:]
    by_session = _roi(bids_tiny, session_ids=["ses-02"])
    assert [u.label() for u in by_session] == ["sub-02_ses-02_task-move"]
    by_task_run = _roi(bids_tiny, task_ids=["task-rest"], run_ids=["run-2"])
    assert [u.label() for u in by_task_run] == ["sub-01_task-rest_run-2"]


def test_filter_file_intersects_with_cli_filters(bids_tiny: Path, tmp_path: Path) -> None:
    ff = tmp_path / "filter.json"
    ff.write_text(json.dumps({"subject": ["01"], "task": ["rest"]}), encoding="utf-8")
    assert [u.label() for u in _roi(bids_tiny, filter_file=ff)] == ROI_LABELS[1:3]
    both = _roi(bids_tiny, filter_file=ff, run_ids=["2"])
    assert [u.label() for u in both] == ["sub-01_task-rest_run-2"]
    with pytest.raises(InputDiscoveryError):
        _roi(bids_tiny, filter_file=ff, participant_labels=["02"])


def test_no_match_raises_with_the_filters_in_the_message(bids_tiny: Path) -> None:
    with pytest.raises(InputDiscoveryError) as exc:
        _roi(bids_tiny, participant_labels=["99"], task_ids=["rest"])
    msg = str(exc.value)
    assert "99" in msg
    assert "rest" in msg
    assert "roi" in msg


def test_sha256_changes_when_data_changes(bids_tiny_copy: Path) -> None:
    before = {u.label(): u.sha256 for u in _roi(bids_tiny_copy)}
    target = _roi(bids_tiny_copy)[0].files[0]
    raw = bytearray(target.read_bytes())
    raw[-1] ^= 0x01
    target.write_bytes(bytes(raw))
    after = {u.label(): u.sha256 for u in _roi(bids_tiny_copy)}
    assert after[ROI_LABELS[0]] != before[ROI_LABELS[0]]
    assert {k: v for k, v in after.items() if k != ROI_LABELS[0]} == {
        k: v for k, v in before.items() if k != ROI_LABELS[0]
    }


def test_sensor_hash_covers_brainvision_companions(bids_tiny_copy: Path) -> None:
    (unit,) = discover(bids_tiny_copy, level="sensor", datatypes=["eeg"])
    assert [f.suffix for f in unit.files[:3]] == [".vhdr", ".eeg", ".vmrk"]
    eeg = next(f for f in unit.files if f.suffix == ".eeg")
    raw = bytearray(eeg.read_bytes())
    raw[-1] ^= 0x01
    eeg.write_bytes(bytes(raw))
    (changed,) = discover(bids_tiny_copy, level="sensor", datatypes=["eeg"])
    assert changed.sha256 != unit.sha256


def test_sensor_hash_covers_the_events_file(bids_tiny_copy: Path) -> None:
    (unit,) = discover(bids_tiny_copy, level="sensor", datatypes=["eeg"])
    assert unit.files[-1].name == "sub-01_task-rest_events.tsv"
    events = unit.files[-1]
    events.write_text(events.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    (changed,) = discover(bids_tiny_copy, level="sensor", datatypes=["eeg"])
    assert changed.sha256 != unit.sha256


def test_two_dirconn_derivatives_is_an_error(bids_tiny_copy: Path) -> None:
    deriv = bids_tiny_copy / "derivatives"
    shutil.copytree(deriv / "dirconn-roi", deriv / "dirconn-roi-2")
    with pytest.raises(InputDiscoveryError) as exc:
        derivatives_root(bids_tiny_copy)
    assert "dirconn-roi" in str(exc.value)
    assert "dirconn-roi-2" in str(exc.value)
    with pytest.raises(InputDiscoveryError):
        _roi(bids_tiny_copy)


def _output_dataset(path: Path) -> None:
    path.mkdir(parents=True)
    description = {
        "Name": "dirconn derivatives",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "dirconn", "Version": "0.2.0"}],
        "Dirconn": {"DerivativesVersion": "1"},
    }
    (path / "dataset_description.json").write_text(json.dumps(description), encoding="utf-8")


def test_connectivity_output_dataset_is_not_an_input(bids_tiny_copy: Path) -> None:
    deriv = bids_tiny_copy / "derivatives"
    _output_dataset(deriv / "dirconn")
    assert derivatives_root(bids_tiny_copy) == deriv / "dirconn-roi"
    assert [u.label() for u in _roi(bids_tiny_copy)] == ROI_LABELS


def test_excluded_dataset_is_skipped(bids_tiny_copy: Path) -> None:
    deriv = bids_tiny_copy / "derivatives"
    shutil.copytree(deriv / "dirconn-roi", deriv / "dirconn-roi-2")
    # compared as resolved paths, so a relative or unnormalised spelling works too
    exclude = deriv / "dirconn-roi-2" / ".." / "dirconn-roi-2"
    assert derivatives_root(bids_tiny_copy, exclude=exclude) == deriv / "dirconn-roi"
    units = _roi(bids_tiny_copy, exclude=exclude)
    assert {u.root for u in units} == {deriv / "dirconn-roi"}
