"""The committed synthetic BIDS dataset tests/fixtures/bids_tiny."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from mne_bids import BIDSPath, read_raw_bids

from dirconn import read_bids

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "make_bids_fixture.py"

ROI_UNITS = [
    {"subject": "01", "task": "rest", "run": 1},
    {"subject": "01", "task": "rest", "run": 2},
    {"subject": "01", "task": "move"},
    {"subject": "02", "task": "rest", "session": "01"},
    {"subject": "02", "task": "move", "session": "02"},
]


def load_script():
    spec = importlib.util.spec_from_file_location("make_bids_fixture", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_has_the_five_roi_units_and_one_sensor_recording(bids_tiny):
    expected = load_script().EXPECTED_FILES
    found = sorted(p.relative_to(bids_tiny) for p in bids_tiny.rglob("*") if p.is_file())
    assert found == sorted(Path(p) for p in expected)
    assert len([p for p in found if p.name.endswith("_eeg.fif")]) == 5
    assert len([p for p in found if p.suffix == ".vhdr"]) == 1


@pytest.mark.parametrize("unit", ROI_UNITS, ids=lambda u: "-".join(map(str, u.values())))
def test_roi_units_read_as_timeseries(bids_tiny, unit):
    ts = read_bids(
        bids_tiny / "derivatives" / "dirconn-roi",
        datatype="eeg",
        level="roi",
        description="DK",
        **unit,
    )
    assert ts.n_channels == 4
    assert list(ts.names) == ["precuneus-lh", "postcentral-lh", "superiorfrontal-lh", "STN-L"]
    assert ts.sfreq == 250.0
    assert ts.n_times == 10000


def test_sensor_recording_carries_bad_annotations(bids_tiny):
    path = BIDSPath(root=bids_tiny, subject="01", task="rest", datatype="eeg", suffix="eeg")
    raw = read_raw_bids(path, verbose="error")
    notes = raw.annotations
    bad = [a for a in notes if a["description"] == "BAD_segment"]
    assert [a["onset"] - raw.first_time for a in bad] == pytest.approx([5.0, 12.0])
    assert [a["duration"] for a in bad] == pytest.approx([0.8, 0.7])
    assert len(bad) == 2
    assert sum(a["description"] == "cue" for a in notes) == 5
    ts = read_bids(bids_tiny, subject="01", task="rest", level="sensor", nan="mask")
    assert ts.n_channels == 8
    assert ts.sfreq == 500.0
    assert ts.n_times == 10000
    # Default stays backward compatible: BAD_ annotations are ignored.
    assert ts.valid is None
    masked = read_bids(
        bids_tiny, subject="01", task="rest", level="sensor", nan="mask", bad_annotations="mask"
    )
    summary = masked.mask_summary()
    assert 748 <= summary["n_samples"] - summary["n_valid"] <= 752


def test_events_tsv_lists_five_cues_and_two_bad_segments(bids_tiny):
    events = pd.read_csv(bids_tiny / "sub-01" / "eeg" / "sub-01_task-rest_events.tsv", sep="	")
    assert events["onset"].is_monotonic_increasing
    cues = events[events["trial_type"] == "cue"]
    assert cues["onset"].tolist() == [2, 6, 10, 14, 18]
    assert cues["duration"].tolist() == [0] * 5
    bad = events[events["trial_type"] == "BAD_segment"]
    assert bad["onset"].tolist() == [5.0, 12.0]
    assert bad["duration"].tolist() == pytest.approx([0.8, 0.7])
    assert len(events) == 7


def test_fixture_is_under_five_megabytes(bids_tiny):
    total = sum(p.stat().st_size for p in bids_tiny.rglob("*") if p.is_file())
    assert total < 5_000_000


def test_make_check_passes(bids_tiny):
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", str(bids_tiny)], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout + done.stderr
