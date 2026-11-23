"""The installed ``dirconn-bids`` command writes a report for every subject and the boilerplate."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from ._offline import assert_offline

CONFIG = """\
[prepare]
resample_hz = "null"

[estimation]
methods = ["gc", "adtf"]
order = 3

[statistics]
n_surrogates = 19
"""


def _entry_point() -> Path:
    """The ``dirconn-bids`` script of the interpreter that runs the tests."""
    bindir = Path(sys.executable).parent
    for name in ("dirconn-bids.exe", "dirconn-bids"):
        if (bindir / name).is_file():
            return bindir / name
    found = shutil.which("dirconn-bids")
    assert found is not None, f"dirconn-bids is not installed next to {sys.executable}"
    return Path(found)


@dataclass(frozen=True)
class Run:
    root: Path
    proc: subprocess.CompletedProcess[str]


@pytest.fixture(scope="module")
def run(bids_tiny: Path, tmp_path_factory: pytest.TempPathFactory) -> Run:
    """Run the command once on a copy of ``bids_tiny`` (five recordings, two subjects)."""
    tmp = tmp_path_factory.mktemp("end_to_end")
    bids = tmp / "bids_tiny"
    shutil.copytree(bids_tiny, bids)
    config = tmp / "config.toml"
    config.write_text(CONFIG, encoding="utf-8")
    out = tmp / "out"
    proc = subprocess.run(
        [
            str(_entry_point()),
            str(bids),
            str(out),
            "participant",
            "--config",
            str(config),
            "--skip-bids-validation",
            "--nprocs",
            "2",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=900,
    )
    return Run(out / "dirconn", proc)


def test_cli_end_to_end_report(run: Run) -> None:
    assert run.proc.returncode == 0, run.proc.stderr
    pages = sorted(p.name for p in run.root.glob("sub-*.html"))
    assert pages == ["sub-01.html", "sub-02.html"]
    for page in run.root.glob("sub-*.html"):
        assert_offline(page.read_text(encoding="utf-8"))
        assert str(page) in run.proc.stderr
    citation = (run.root / "logs" / "CITATION.md").read_text(encoding="utf-8")
    assert "predictive, not interventional" in citation


def test_the_report_shows_both_methods_and_an_empty_errors_section(run: Run) -> None:
    text = (run.root / "sub-01.html").read_text(encoding="utf-8")
    for needle in ("section-summary", "section-errors", "GC", "ADTF"):
        assert needle in text, needle
    assert "No task failed and no reportlet is missing." in text
