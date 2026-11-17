"""Tests for the ``dirconn-bids`` command line (participant level and subcommands)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import dirconn
from dirconn.cli import run as cli_run
from dirconn.methods.registry import REGISTRY
from dirconn.workflows import participant
from dirconn.workflows.config import config_hash, load_config
from dirconn.workflows.log import close_logging

# Order 3 is deliberately too low for the fixture; the model-fit warnings are expected.
pytestmark = pytest.mark.filterwarnings("ignore::dirconn.data.exceptions.DirconnWarning")

#: Subprocess options: decode as UTF-8 whatever the console code page.
_TEXT = {
    "capture_output": True,
    "text": True,
    "encoding": "utf-8",
    "errors": "replace",
    "env": {**os.environ, "PYTHONIOENCODING": "utf-8"},
}

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
        [sys.executable, "-m", "dirconn.cli.run", *map(str, args)], timeout=900, **_TEXT
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
    proc = _cli("--version")
    assert proc.returncode == 0
    assert dirconn.__version__ in proc.stdout


# One unit (sub-02, ses-01): two tasks, controls and gc.
SUBSET = ("--participant-label", "sub-02", "--session-id", "01")


def test_rerun_with_output_below_bids_derivatives(bids_tiny_copy: Path, tmp_path: Path):
    out = bids_tiny_copy / "derivatives"
    config = _write_config(tmp_path)
    first = _participant(bids_tiny_copy, out, config, *SUBSET)
    assert first.returncode == 0, first.stderr
    assert "done 2, skipped 0, failed 0" in first.stderr
    second = _participant(bids_tiny_copy, out, config, *SUBSET)
    assert second.returncode == 0, second.stderr
    assert "done 0, skipped 2, failed 0" in second.stderr
    assert (out / "dirconn" / "dataset_description.json").is_file()


NO_APP = """\
import sys
sys.modules["pydantic"] = None
from dirconn.cli.run import main
raise SystemExit(main(sys.argv[1:]))
"""


def _no_app(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", NO_APP, *map(str, args)], timeout=300, **_TEXT)


def test_cli_without_the_app_extra(tmp_path: Path):
    version = _no_app("--version")
    assert version.returncode == 0, version.stderr
    assert dirconn.__version__ in version.stdout
    assert _no_app("--help").returncode == 0
    message = (
        'dirconn-bids: error: the command line needs the app extra: pip install "dirconn[app]"'
    )
    for args in (
        (tmp_path, tmp_path / "out", "participant"),
        ("validate-config", _write_config(tmp_path)),
    ):
        proc = _no_app(*args)
        assert proc.returncode == 2, proc.stderr
        assert message in proc.stderr
    assert not (tmp_path / "out").exists()


# In-process tests of main().


@pytest.fixture(autouse=True)
def _restore_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Restore the thread variables that main() sets and the dirconn logger."""
    for name in cli_run.THREAD_VARIABLES:
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])
        else:
            monkeypatch.delenv(name, raising=False)
    logger = logging.getLogger("dirconn")
    state = (list(logger.handlers), logger.level, logger.propagate)
    yield
    close_logging()
    logger.handlers[:] = state[0]
    logger.setLevel(state[1])
    logger.propagate = state[2]


def _main(bids: Path, out: Path, config: Path, *extra: str | Path) -> int:
    args = [bids, out, "participant", "--config", config, *extra]
    return cli_run.main([str(a) for a in args])


def test_nonexistent_bids_dir_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    code = _main(tmp_path / "missing", tmp_path / "out", _write_config(tmp_path))
    assert code == 2
    assert "dirconn-bids: error: BIDS directory" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("where", [".", "sub-01/out", "code"])
def test_output_inside_the_bids_dir_is_refused(
    bids_tiny_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], where: str
):
    before = sorted(bids_tiny_copy.rglob("*"))
    code = _main(
        bids_tiny_copy, bids_tiny_copy / where, _write_config(tmp_path), "--skip-bids-validation"
    )
    assert code == 2
    assert "dirconn-bids: error: the output directory" in capsys.readouterr().err
    assert sorted(bids_tiny_copy.rglob("*")) == before


class _FakeValidator:
    def __init__(self, returncode: int | None) -> None:
        self.returncode = returncode
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append(cmd)
        if self.returncode is None:
            raise FileNotFoundError(cmd[0])
        return subprocess.CompletedProcess(cmd, self.returncode)


def _validator(monkeypatch: pytest.MonkeyPatch, returncode: int | None) -> _FakeValidator:
    fake = _FakeValidator(returncode)
    monkeypatch.setattr(shutil, "which", lambda name: f"/opt/{name}")
    monkeypatch.setattr(subprocess, "run", fake)
    return fake


def test_bids_validator_rejecting_exits_three(
    bids_tiny_copy: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    fake = _validator(monkeypatch, 1)
    assert _main(bids_tiny_copy, tmp_path / "out", _write_config(tmp_path)) == 3
    assert fake.calls == [["/opt/bids-validator", str(bids_tiny_copy.resolve())]]
    assert "dirconn-bids: error: bids-validator rejected" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_bids_validator_that_cannot_run_exits_two(
    bids_tiny_copy: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    _validator(monkeypatch, None)
    assert _main(bids_tiny_copy, tmp_path / "out", _write_config(tmp_path)) == 2
    assert "dirconn-bids: error: cannot run bids-validator" in capsys.readouterr().err


def test_bids_validator_accepting_continues(
    bids_tiny_copy: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    fake = _validator(monkeypatch, 0)
    # An unknown participant stops the run at input discovery, after the validation.
    code = _main(
        bids_tiny_copy, tmp_path / "out", _write_config(tmp_path), "--participant-label", "99"
    )
    assert code == 2
    assert len(fake.calls) == 1
    assert "No recordings found" in capsys.readouterr().err


def test_missing_bids_validator_warns_and_continues(
    bids_tiny_copy: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    code = _main(
        bids_tiny_copy, tmp_path / "out", _write_config(tmp_path), "--participant-label", "99"
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "dirconn-bids: warning: bids-validator not found" in err
    assert "No recordings found" in err


def test_os_error_setting_up_the_run_exits_two(
    bids_tiny_copy: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    def fail(*args: object, **kwargs: object) -> None:
        raise PermissionError("read-only file system")

    monkeypatch.setattr(participant, "run_participant", fail)
    code = _main(
        bids_tiny_copy, tmp_path / "out", _write_config(tmp_path), "--skip-bids-validation"
    )
    assert code == 2
    assert "dirconn-bids: error: cannot set up the run" in capsys.readouterr().err


def test_filter_file_resources_and_argv_reach_the_provenance(bids_tiny_copy: Path, tmp_path: Path):
    filters = tmp_path / "filters.json"
    filters.write_text(json.dumps({"subject": ["02"], "session": ["01"]}), encoding="utf-8")
    out = tmp_path / "out"
    args = [
        str(bids_tiny_copy),
        str(out),
        "participant",
        "--config",
        str(_write_config(tmp_path)),
        "--skip-bids-validation",
        "--bids-filter-file",
        str(filters),
        "--nprocs",
        "2",
        "--omp-nthreads",
        "2",
        "--mem-mb",
        "512",
    ]
    assert cli_run.main(args) == 0
    for name in cli_run.THREAD_VARIABLES:
        assert os.environ[name] == "2", name
    (run_dir,) = _run_dirs(out)
    record = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert record["command"] == ["dirconn-bids", *args]
    assert [unit["label"] for unit in record["inputs"]] == ["sub-02_ses-01_task-rest"]
    resources = record["resources"]
    assert (resources["nprocs"], resources["omp_nthreads"], resources["mem_mb"]) == (2, 2, 512)
    assert Path(resources["work_dir"]) == (out / "work").resolve()
    assert record["tasks"] == {"done": 2, "skipped": 0, "failed": 0}


def test_help_says_the_filter_file_is_intersected(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc:
        cli_run.main(["--help"])
    assert exc.value.code == 0
    assert "intersected with the other filters" in " ".join(capsys.readouterr().out.split())
