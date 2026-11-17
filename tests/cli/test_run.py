"""Tests for the ``dirconn-bids`` command line (participant level and subcommands)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dirconn.methods.registry import REGISTRY
from dirconn.workflows.config import config_hash, load_config

TEST_CONFIG = """\
[prepare]
resample_hz = "null"

[estimation]
methods = ["gc"]
order = 3

[statistics]
n_surrogates = 10
"""

ROI_LABELS = [
    "sub-01_task-move",
    "sub-01_task-rest_run-1",
    "sub-01_task-rest_run-2",
    "sub-02_ses-01_task-rest",
    "sub-02_ses-02_task-move",
]


def _write_config(tmp: Path, text: str = TEST_CONFIG) -> Path:
    path = tmp / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _cli(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "dirconn.cli.run", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=900,
    )


def _participant(
    bids: Path, out: Path, config: Path, *extra: str | Path
) -> subprocess.CompletedProcess[str]:
    return _cli(bids, out, "participant", "--config", config, "--skip-bids-validation", *extra)


def _run_dirs(out: Path) -> list[Path]:
    return sorted((out / "dirconn" / "logs").glob("run-*"))


def _tasks(run_dir: Path) -> dict[str, int]:
    record = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    return record["tasks"]


def test_participant_run_exits_zero_and_writes_outputs(bids_tiny_copy: Path, tmp_path: Path):
    out = tmp_path / "out"
    proc = _participant(bids_tiny_copy, out, _write_config(tmp_path))
    assert proc.returncode == 0, proc.stderr
    root = out / "dirconn"
    assert (root / "dirconn.toml").is_file()
    assert (root / "dataset_description.json").is_file()
    for label in ROI_LABELS:
        assert list(root.rglob(f"{label}_*desc-gc_conn.json")), label
    (run_dir,) = _run_dirs(out)
    for name in ("dirconn.log", "progress.jsonl", "provenance.json", "dirconn.toml"):
        assert (run_dir / name).is_file(), name
    assert _tasks(run_dir) == {"done": 10, "skipped": 0, "failed": 0}
    assert "done 10, skipped 0, failed 0" in proc.stderr
    assert str(run_dir) in proc.stderr
    assert (out / "work").is_dir()  # the documented default of --work-dir


def test_rerun_is_all_skipped_and_exit_zero(bids_tiny_copy: Path, tmp_path: Path):
    out = tmp_path / "out"
    config = _write_config(tmp_path)
    assert _participant(bids_tiny_copy, out, config).returncode == 0
    proc = _participant(bids_tiny_copy, out, config, "--work-dir", tmp_path / "other-work")
    assert proc.returncode == 0, proc.stderr
    assert "done 0, skipped 10, failed 0" in proc.stderr
    assert _tasks(_run_dirs(out)[-1]) == {"done": 0, "skipped": 10, "failed": 0}


def test_cli_refuses_changed_config(bids_tiny_copy: Path, tmp_path: Path):
    out = tmp_path / "out"
    config = _write_config(tmp_path)
    subset = ("--participant-label", "sub-02", "--session-id", "01")
    assert _participant(bids_tiny_copy, out, config, *subset).returncode == 0
    proc = _participant(bids_tiny_copy, out, config, *subset, "--set", "statistics.n_surrogates=20")
    assert proc.returncode == 2
    assert "dirconn-bids: error: " in proc.stderr
    assert "dirconn.toml" in proc.stderr
    assert len(_run_dirs(out)) == 1


def test_cli_exit_code_1_when_one_task_fails(bids_tiny_copy: Path, tmp_path: Path):
    out = tmp_path / "out"
    proc = _participant(
        bids_tiny_copy,
        out,
        _write_config(tmp_path),
        "--participant-label",
        "02",
        "--session-id",
        "ses-01",
        "--set",
        "estimation.params.gc.order=100000",
    )
    assert proc.returncode == 1, proc.stderr
    assert "done 1, skipped 0, failed 1" in proc.stderr
    (run_dir,) = _run_dirs(out)
    assert _tasks(run_dir) == {"done": 1, "skipped": 0, "failed": 1}
    assert len(list(run_dir.glob("crash-*"))) == 1


def test_unknown_participant_exits_two(bids_tiny_copy: Path, tmp_path: Path):
    out = tmp_path / "out"
    proc = _participant(
        bids_tiny_copy, out, _write_config(tmp_path), "--participant-label", "sub-99"
    )
    assert proc.returncode == 2
    assert proc.stderr.startswith("dirconn-bids: error: ")
    assert not out.exists()


def test_group_level_exits_two_with_message(bids_tiny_copy: Path, tmp_path: Path):
    proc = _cli(bids_tiny_copy, tmp_path / "out", "group")
    assert proc.returncode == 2
    assert "available in dirconn 0.3.0" in proc.stderr
    assert not (tmp_path / "out").exists()


def test_reports_only_exits_two(bids_tiny_copy: Path, tmp_path: Path):
    for flag in ("--reports-only", "--boilerplate-only"):
        proc = _cli(bids_tiny_copy, tmp_path / "out", "participant", flag)
        assert proc.returncode == 2, flag
        assert "available in dirconn 0.3.0" in proc.stderr, flag
    assert not (tmp_path / "out").exists()


def test_validate_config_subcommand(tmp_path: Path):
    good = _write_config(tmp_path)
    proc = _cli("validate-config", good)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"OK {config_hash(load_config(good))}"

    bad = tmp_path / "bad.toml"
    bad.write_text("[estimation]\norder = -1\nunknown_key = 3\n", encoding="utf-8")
    proc = _cli("validate-config", bad)
    assert proc.returncode == 2
    assert "dirconn-bids: error: " in proc.stderr
    assert "order" in proc.stderr and "unknown_key" in proc.stderr

    proc = _cli("validate-config", tmp_path / "missing.toml")
    assert proc.returncode == 2


def test_list_methods_subcommand_lists_fifteen_methods():
    proc = _cli("list-methods")
    assert proc.returncode == 0, proc.stderr
    header, *rows = proc.stdout.strip().splitlines()
    assert header.split()[:4] == ["name", "report_kind", "frequency_resolved", "supports_mask"]
    assert len(rows) == 15 == len(REGISTRY)
    assert [row.split()[0] for row in rows] == sorted(REGISTRY)
    gc = next(row for row in rows if row.split()[0] == "gc")
    assert REGISTRY["gc"].info.backend in gc


def test_bad_set_override_exits_two(bids_tiny_copy: Path, tmp_path: Path):
    config = _write_config(tmp_path)
    for override in ("estimation.no_such_field=1", "estimation.order", "estimation.order=-5"):
        proc = _participant(bids_tiny_copy, tmp_path / "out", config, "--set", override)
        assert proc.returncode == 2, override
        assert "dirconn-bids: error: " in proc.stderr, override
    assert not (tmp_path / "out").exists()


def test_version_still_works():
    import dirconn

    proc = _cli("--version")
    assert proc.returncode == 0
    assert dirconn.__version__ in proc.stdout
