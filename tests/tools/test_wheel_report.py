"""The wheel carries the report's package data, and (opt-in) a clean install writes a report.

``test_wheel_contains_the_package_data`` builds the wheel offline with hatchling (the build backend,
in the ``dev`` extra). ``test_wheel_builds_a_report`` installs it into a new virtual environment
with ``pip`` and needs the network, so it runs only with ``-m wheel``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PACKAGE_DATA = (
    "dirconn/reports/_data/spec.yml",
    "dirconn/reports/_data/references.json",
    "dirconn/reports/_data/templates/participant.html",
    "dirconn/reports/_data/templates/report.css",
    "dirconn/workflows/_data/default_config.toml",
)
NOT_SHIPPED = (".superpowers", "_notes", ".claude", "tests/")


def _build_wheel(directory: Path) -> Path:
    """Build the wheel of this checkout into ``directory`` (no network, no build isolation)."""
    if importlib.util.find_spec("hatchling") is None:
        pytest.skip("hatchling is not installed (it is in the dev extra)")
    done = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(directory)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    (wheel,) = directory.glob("dirconn-*.whl")
    return wheel


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_wheel(tmp_path_factory.mktemp("wheel"))


def test_wheel_contains_the_package_data(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    missing = [name for name in PACKAGE_DATA if name not in names]
    assert not missing, f"missing from {wheel.name}: {missing}"
    stray = [n for n in names if any(part in n for part in NOT_SHIPPED)]
    assert not stray, f"files that must not ship: {stray}"
    assert "dirconn/reports/assemble.py" in names
    assert "dirconn/cli/run.py" in names


@pytest.mark.wheel
def test_wheel_builds_a_report(wheel: Path, bids_tiny: Path, tmp_path: Path) -> None:
    """Install the wheel with the ``app`` extra into a clean venv and run it on a fixture."""
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, timeout=300)
    bindir = venv / ("Scripts" if os.name == "nt" else "bin")
    python = bindir / ("python.exe" if os.name == "nt" else "python")
    install = subprocess.run(
        [str(python), "-m", "pip", "install", f"{wheel}[app]"],
        capture_output=True,
        text=True,
        timeout=1500,
    )
    assert install.returncode == 0, install.stderr[-3000:]
    bids = tmp_path / "bids_tiny"
    shutil.copytree(bids_tiny, bids)
    out = tmp_path / "out"
    command = bindir / ("dirconn-bids.exe" if os.name == "nt" else "dirconn-bids")
    done = subprocess.run(
        [
            str(command), str(bids), str(out), "participant",
            "--participant-label", "01",
            "--set", "prepare.resample_hz=null",
            "--set", "estimation.methods=gc",
            "--set", "estimation.order=3",
            "--set", "statistics.n_surrogates=19",
            "--skip-bids-validation",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1500,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-3000:]
    assert (out / "dirconn" / "sub-01.html").is_file()
