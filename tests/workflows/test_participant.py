"""Tests for the participant-level workflow."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import dirconn
from dirconn.io.derivatives import read_edges, result_task_key, unit_dir
from dirconn.workflows.config import PipelineConfig, analysis_hash, config_hash, load_config
from dirconn.workflows.executor import RunSummary, task_key
from dirconn.workflows.inputs import InputUnit, discover
from dirconn.workflows.log import setup_logging
from dirconn.workflows.participant import (
    ConfigMismatchError,
    UnitTask,
    events_tsv_for,
    plan_tasks,
    run_participant,
)
from dirconn.workflows.provenance import Resources

# Order 3 is deliberately too low for the fixture; the model-fit warnings are expected.
pytestmark = pytest.mark.filterwarnings("ignore::dirconn.data.exceptions.DirconnWarning")

BASE_CONFIG = """\
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


def _config(tmp: Path, extra: str = "") -> PipelineConfig:
    path = tmp / "config.toml"
    path.write_text(BASE_CONFIG + extra, encoding="utf-8")
    return load_config(path)


def _units(bids: Path, cfg: PipelineConfig) -> list[InputUnit]:
    return discover(
        bids,
        level=cfg.input.level,
        datatypes=cfg.input.datatypes,
        description=cfg.input.description,
    )


@dataclass
class Run:
    bids: Path
    out: Path
    cfg: PipelineConfig
    units: list[InputUnit]
    summary: RunSummary

    @property
    def root(self) -> Path:
        return self.out / "dirconn"


def _run(bids: Path, out: Path, cfg: PipelineConfig, *, workers: int = 1) -> Run:
    units = _units(bids, cfg)
    resources = Resources(nprocs=workers, omp_nthreads=1, mem_mb=None, work_dir=out / "work")
    summary = run_participant(
        cfg,
        bids_dir=bids,
        out_dir=out,
        units=units,
        resources=resources,
        argv=["dirconn", str(bids), str(out), "participant"],
        verbosity=1,
    )
    return Run(bids, out, cfg, units, summary)


@pytest.fixture(scope="module")
def first_run(bids_tiny: Path, tmp_path_factory: pytest.TempPathFactory) -> Run:
    """One run of the test configuration over the five ROI units (read-only for the tests)."""
    tmp = tmp_path_factory.mktemp("participant")
    bids = tmp / "bids_tiny"
    shutil.copytree(bids_tiny, bids)
    return _run(bids, tmp / "out", _config(tmp))


def test_plan_tasks_keys_labels_and_order(bids_tiny: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg = cfg.model_copy(
        update={"estimation": cfg.estimation.model_copy(update={"methods": ("gc", "pdc")})}
    )
    units = _units(bids_tiny, cfg)
    tasks = plan_tasks(units, cfg)
    assert [t.label for t in tasks] == [f"{lab}:{m}" for lab in ROI_LABELS for m in ("gc", "pdc")]
    first = tasks[0]
    assert first.payload == UnitTask(units[0], "gc")
    assert first.key == task_key(units[0].sha256, analysis_hash(cfg), "gc", dirconn.__version__)
    assert len({t.key for t in tasks}) == len(tasks)


def test_events_tsv_is_found_next_to_the_data_file(bids_tiny: Path) -> None:
    sensor = discover(bids_tiny, level="sensor", datatypes=["eeg"])
    rest = next(u for u in sensor if u.task == "rest")
    events = events_tsv_for(rest)
    assert events is not None and events.is_file()
    assert events.name == "sub-01_task-rest_events.tsv"
    roi = _units(bids_tiny, load_config(None))
    assert events_tsv_for(roi[0]) is None


def test_participant_run_writes_every_output(first_run: Run) -> None:
    assert first_run.summary == RunSummary(done=5, skipped=0, failed=0, failures=())
    root = first_run.root
    for name in ("dirconn.toml", "dataset_description.json", "provenance.json"):
        assert (root / name).is_file(), name
    assert (root / "environment-lock.txt").read_text(encoding="utf-8").strip()
    assert (root / "logs" / "dirconn.log").is_file()
    assert len((root / "logs" / "progress.jsonl").read_text(encoding="utf-8").splitlines()) == 5
    assert load_config(root / "dirconn.toml") == first_run.cfg
    for unit in first_run.units:
        directory = unit_dir(root, unit.entities(), unit.datatype)
        stem = unit.label()
        for suffix in ("conn.nc", "conn.json", "edges.tsv"):
            assert (directory / f"{stem}_desc-gc_{suffix}").is_file(), (stem, suffix)
        for suffix in ("conn.nc", "conn.json"):
            assert (directory / f"{stem}_desc-trgc_{suffix}").is_file(), (stem, suffix)
        sidecar = directory / f"{stem}_desc-gc_conn.json"
        task = plan_tasks([unit], first_run.cfg)[0]
        assert result_task_key(sidecar) == task.key
        assert result_task_key(directory / f"{stem}_desc-trgc_conn.json") == task.key
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        assert meta["Statistics"] and all(s["NSurrogates"] == 10 for s in meta["Statistics"])
    diagnostics = sorted(root.rglob("figures/*_diagnostics.json"))
    assert [p.name for p in diagnostics] == [f"{lab}_diagnostics.json" for lab in ROI_LABELS]
    assert "valid_fraction" in json.loads(diagnostics[0].read_text(encoding="utf-8"))
    assert len(list((first_run.out / "work").glob("prepared-*.npz"))) == 5


def test_provenance_lists_inputs_and_counts(first_run: Run) -> None:
    prov = json.loads((first_run.root / "provenance.json").read_text(encoding="utf-8"))
    assert set(prov) == {
        "command",
        "started",
        "finished",
        "hostname",
        "platform",
        "python",
        "versions",
        "image",
        "git",
        "inputs",
        "config_hash",
        "analysis_hash",
        "resources",
        "tasks",
    }
    assert prov["command"][0] == "dirconn"
    assert prov["started"] <= prov["finished"]
    assert prov["versions"]["dirconn"] == dirconn.__version__
    assert set(prov["image"]) == {"tag", "revision"}
    assert [i["label"] for i in prov["inputs"]] == ROI_LABELS
    assert prov["inputs"][0]["sha256"] == first_run.units[0].sha256
    assert prov["inputs"][0]["files"] == [f.as_posix() for f in first_run.units[0].files]
    assert prov["config_hash"] == config_hash(first_run.cfg)
    assert prov["analysis_hash"] == analysis_hash(first_run.cfg)
    assert prov["resources"]["nprocs"] == 1
    assert prov["resources"]["work_dir"] == str(first_run.out / "work")
    assert prov["tasks"] == {"done": 5, "skipped": 0, "failed": 0}


def test_second_run_skips_everything(bids_tiny_copy: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    out = tmp_path / "out"
    assert _run(bids_tiny_copy, out, cfg).summary.done == 5
    again = _run(bids_tiny_copy, out, cfg)
    assert again.summary.skipped == 5
    assert again.summary.done == 0
    prov = json.loads((again.root / "provenance.json").read_text(encoding="utf-8"))
    assert prov["tasks"] == {"done": 0, "skipped": 5, "failed": 0}


def test_rerun_with_different_config_is_refused(first_run: Run, tmp_path: Path) -> None:
    other = first_run.cfg.model_copy(
        update={"statistics": first_run.cfg.statistics.model_copy(update={"n_surrogates": 11})}
    )
    before = (first_run.root / "dirconn.toml").read_text(encoding="utf-8")
    with pytest.raises(ConfigMismatchError):
        _run(first_run.bids, first_run.out, other)
    assert (first_run.root / "dirconn.toml").read_text(encoding="utf-8") == before


def test_failing_method_is_isolated(bids_tiny_copy: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path, "\n[estimation.gc]\norder = 100000\n")
    run = _run(bids_tiny_copy, tmp_path / "out", cfg)
    assert run.summary.failed == 5
    assert run.summary.done == 0
    assert run.summary.exit_code == 1
    assert len(list((run.root / "logs").glob("crash-*.txt"))) == 5
    assert not list(run.root.rglob("*desc-gc*"))
    for name in ("dirconn.toml", "dataset_description.json", "provenance.json"):
        assert (run.root / name).is_file(), name
    prov = json.loads((run.root / "provenance.json").read_text(encoding="utf-8"))
    assert prov["tasks"] == {"done": 0, "skipped": 0, "failed": 5}


def test_fail_fast_still_writes_provenance(bids_tiny_copy: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path, "\n[execution]\nfail_fast = true\n\n[estimation.gc]\norder = 100000\n")
    run = _run(bids_tiny_copy, tmp_path / "out", cfg)
    assert run.summary.failed == 1
    assert run.summary.exit_code == 1
    prov = json.loads((run.root / "provenance.json").read_text(encoding="utf-8"))
    assert prov["tasks"] == {"done": 0, "skipped": 0, "failed": 1}


def test_workers_two_matches_workers_one(first_run: Run, tmp_path: Path) -> None:
    two = _run(first_run.bids, tmp_path / "out2", first_run.cfg, workers=2)
    assert two.summary.done == 5
    one_tables = sorted(first_run.root.rglob("*_desc-gc_edges.tsv"))
    two_tables = sorted(two.root.rglob("*_desc-gc_edges.tsv"))
    assert [p.name for p in one_tables] == [p.name for p in two_tables]
    assert len(one_tables) == 5
    for a, b in zip(one_tables, two_tables, strict=True):
        ea, eb = read_edges(a), read_edges(b)
        assert list(ea["source"]) == list(eb["source"])
        assert list(ea["band"]) == list(eb["band"])
        np.testing.assert_allclose(ea["value"], eb["value"])
        np.testing.assert_array_equal(ea["p"], eb["p"])


def test_setup_logging_is_idempotent(tmp_path: Path) -> None:
    logger = setup_logging(tmp_path / "a", 1)
    assert logger.name == "dirconn"
    assert logger.level == logging.INFO
    n = len(logger.handlers)
    assert setup_logging(tmp_path / "a", 2) is logger
    assert len(logger.handlers) == n
    assert logger.level == logging.DEBUG
    setup_logging(tmp_path / "b", 0)
    assert len(logger.handlers) == n
    assert logger.level == logging.WARNING
    files = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert [Path(h.baseFilename) for h in files] == [tmp_path / "b" / "dirconn.log"]
    logger.warning("hello")
    for h in files:
        h.flush()
    assert "WARNING dirconn: hello" in (tmp_path / "b" / "dirconn.log").read_text(encoding="utf-8")
    for h in files:
        logger.removeHandler(h)
        h.close()
