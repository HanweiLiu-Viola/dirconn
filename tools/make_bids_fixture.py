"""Generate (or check) the tiny synthetic BIDS dataset ``tests/fixtures/bids_tiny``.

The dataset has five region-level (ROI) units in ``derivatives/dirconn-roi`` and one raw
BrainVision EEG recording. Everything is deterministic except the dates BrainVision writes into
its header, so ``--check`` compares the file list and the total size, not hashes.

Usage::

    python tools/make_bids_fixture.py --out tests/fixtures/bids_tiny [--force]
    python tools/make_bids_fixture.py --check tests/fixtures/bids_tiny
"""

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

BASE_SEED = 20261116
MAX_BYTES = 5_000_000

#: Every file of the dataset, relative to its root.
EXPECTED_FILES: list[str] = [
    "README",
    "dataset_description.json",
    "participants.json",
    "participants.tsv",
    "sub-01/sub-01_scans.tsv",
    "sub-01/eeg/sub-01_space-CapTrak_coordsystem.json",
    "sub-01/eeg/sub-01_space-CapTrak_electrodes.json",
    "sub-01/eeg/sub-01_space-CapTrak_electrodes.tsv",
    "sub-01/eeg/sub-01_task-rest_channels.tsv",
    "sub-01/eeg/sub-01_task-rest_eeg.eeg",
    "sub-01/eeg/sub-01_task-rest_eeg.json",
    "sub-01/eeg/sub-01_task-rest_eeg.vhdr",
    "sub-01/eeg/sub-01_task-rest_eeg.vmrk",
    "sub-01/eeg/sub-01_task-rest_events.json",
    "sub-01/eeg/sub-01_task-rest_events.tsv",
    "derivatives/dirconn-roi/dataset_description.json",
    *(
        f"derivatives/dirconn-roi/{directory}/eeg/{stem}_desc-DK_{suffix}"
        for directory, stem in [
            ("sub-01", "sub-01_task-rest_run-1"),
            ("sub-01", "sub-01_task-rest_run-2"),
            ("sub-01", "sub-01_task-move"),
            ("sub-02/ses-01", "sub-02_ses-01_task-rest"),
            ("sub-02/ses-02", "sub-02_ses-02_task-move"),
        ]
        for suffix in ("channels.json", "channels.tsv", "eeg.fif", "eeg.json")
    ),
]

#: (subject, session, task, run) of the five ROI units, in seed order.
ROI_UNITS = [
    ("01", None, "rest", 1),
    ("01", None, "rest", 2),
    ("01", None, "move", None),
    ("02", "01", "rest", None),
    ("02", "02", "move", None),
]
SENSOR_NAMES = ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "P3", "P4"]
CUE_ONSETS = [2, 6, 10, 14, 18]
BAD_SEGMENTS = [(5.0, 0.8), (12.0, 0.7)]


def _write_roi_units(root: Path) -> None:
    from dirconn import Edge, Network, direct_node, dk_node, write_roi_bids
    from dirconn.simulation import NodeDynamics, simulate

    cortical = ["precuneus-lh", "postcentral-lh", "superiorfrontal-lh"]
    nodes = [
        dk_node(name, region=name.removesuffix("-lh"), hemisphere="lh", atlas="DK")
        for name in cortical
    ]
    nodes.append(direct_node("STN-L", region="STN", hemisphere="lh"))
    dynamics = {
        name: NodeDynamics(kind="oscillator", freq_hz=freq)
        for name, freq in zip([*cortical, "STN-L"], (10.0, 20.0, 15.0, 8.0), strict=True)
    }
    for index, (subject, session, task, run) in enumerate(ROI_UNITS):
        seed = BASE_SEED + index
        if task == "move":
            link = Edge("STN-L", "postcentral-lh", strength=0.4, delay_ms=12.0)
        else:
            link = Edge("postcentral-lh", "STN-L", strength=0.4, delay_ms=12.0)
        network = Network(
            nodes=nodes,
            edges=[Edge("precuneus-lh", "postcentral-lh", strength=0.4, delay_ms=12.0), link],
            name=f"tiny-{task}",
        )
        sim = simulate(
            network,
            sfreq=250.0,
            n_times=10000,
            seed=seed,
            dynamics=dynamics,
            snr_db=10.0,
            n_background=0,
            node_scale="unit",
        )
        series = sim.roi_observations(snr_db=10.0, seed=seed)
        write_roi_bids(
            series,
            root / "derivatives" / "dirconn-roi",
            subject=subject,
            session=session,
            task=task,
            run=run,
            description="DK",
            datatype="eeg",
        )


def _write_bad_markers(vmrk: Path, sfreq: float) -> None:
    """Replace the markers mne-bids wrote with ones that keep the BAD annotations.

    mne-bids stores annotations as bare ``S  1`` stimulus markers without name or duration. Here
    each marker has the segment's duration, and its type is ``BAD_segment`` so that
    ``mne.io.read_raw_brainvision`` returns an annotation whose description starts with ``BAD_``.
    """
    lines = [line for line in vmrk.read_text().splitlines() if not line.startswith("Mk")]
    for number, (onset, duration) in enumerate(BAD_SEGMENTS, start=1):
        position = int(onset * sfreq) + 1
        lines.append(f"Mk{number}=BAD_segment,,{position},{int(duration * sfreq)},0")
    vmrk.write_text("\n".join(lines) + "\n", newline="\n")


def _write_sensor_recording(root: Path) -> None:
    import mne
    from mne_bids import BIDSPath, write_raw_bids

    sfreq, seconds = 500.0, 20
    n_times = int(sfreq * seconds)
    rng = np.random.default_rng(BASE_SEED)
    t = np.arange(n_times) / sfreq
    data = 1e-5 * (rng.standard_normal((len(SENSOR_NAMES), n_times)) + np.sin(2 * np.pi * 10 * t))
    info = mne.create_info(SENSOR_NAMES, sfreq, "eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_montage(mne.channels.make_standard_montage("colin27_1020"), verbose="error")
    raw.info["line_freq"] = 50.0
    onsets, durations = zip(*BAD_SEGMENTS, strict=True)
    raw.set_annotations(
        mne.Annotations(list(onsets), list(durations), ["BAD_segment"] * len(BAD_SEGMENTS))
    )
    path = BIDSPath(root=root, subject="01", task="rest", datatype="eeg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_raw_bids(
            raw, path, format="BrainVision", allow_preload=True, overwrite=True, verbose="error"
        )
    _write_bad_markers(path.copy().update(suffix="eeg", extension=".vmrk").fpath, sfreq)
    events = pd.DataFrame(
        {"onset": CUE_ONSETS, "duration": 0, "trial_type": "cue"},
    )
    events_path = path.copy().update(suffix="events", extension=".tsv")
    events.to_csv(events_path.fpath, sep="\t", index=False, lineterminator="\n")


def generate(out: Path, *, force: bool = False) -> None:
    """Write the dataset into ``out``.

    Parameters
    ----------
    out : pathlib.Path
        Target directory; must not exist unless ``force`` is set.
    force : bool
        Delete an existing ``out`` first.
    """
    if out.exists():
        if not force:
            raise FileExistsError(f"{out} exists; pass --force to replace it.")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    _write_sensor_recording(out)
    _write_roi_units(out)
    # mne-bids writes both files for the recording; replace them with the fixture's own content.
    description = {"Name": "dirconn tiny fixture", "BIDSVersion": "1.9.0"}
    (out / "dataset_description.json").write_text(
        json.dumps(description, indent=4) + "\n", newline="\n"
    )
    participants = "participant_id\tage\tgroup\nsub-01\t61\tpd\nsub-02\t58\tpd\n"
    (out / "participants.tsv").write_text(participants, newline="\n")


def check(root: Path) -> list[str]:
    """Return the problems found in the dataset at ``root`` (empty when it is as expected).

    Parameters
    ----------
    root : pathlib.Path
        Dataset root.

    Returns
    -------
    list of str
        One message per missing file, unexpected file, or size violation.
    """
    found = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    problems = [f"missing: {name}" for name in EXPECTED_FILES if name not in found]
    problems += [f"unexpected: {name}" for name in found if name not in EXPECTED_FILES]
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if total >= MAX_BYTES:
        problems.append(f"total size {total} bytes is not below {MAX_BYTES}")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Run the command line; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path, help="generate the dataset here")
    group.add_argument("--check", type=Path, help="verify the dataset here")
    parser.add_argument("--force", action="store_true", help="replace an existing --out")
    args = parser.parse_args(argv)
    if args.check is not None:
        problems = check(args.check)
        for line in problems:
            print(line)
        print("ok" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    generate(args.out, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
