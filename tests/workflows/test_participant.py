"""Tests for the participant-level workflow."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import dirconn
from dirconn.io.derivatives import TIME_EDGE_COLUMNS, read_edges, result_task_key, unit_dir
from dirconn.workflows import participant
from dirconn.workflows.config import PipelineConfig, analysis_hash, config_hash, load_config
from dirconn.workflows.executor import task_key
from dirconn.workflows.inputs import InputUnit, discover
from dirconn.workflows.log import close_logging, setup_logging
from dirconn.workflows.participant import (
    ConfigMismatchError,
    ControlsTask,
    ParticipantSummary,
    UnitTask,
    _prepared,
    diagnostics_path,
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


@pytest.fixture(autouse=True)
def _restore_dirconn_logger() -> Iterator[None]:
    logger = logging.getLogger("dirconn")
    state = (list(logger.handlers), logger.level, logger.propagate)
    yield
    close_logging()
    logger.handlers[:] = state[0]
    logger.setLevel(state[1])
    logger.propagate = state[2]


def _config(tmp: Path, extra: str = "", base: str = BASE_CONFIG) -> PipelineConfig:
    path = tmp / "config.toml"
    path.write_text(base + extra, encoding="utf-8")
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
    summary: ParticipantSummary

    @property
    def root(self) -> Path:
        return self.out / "dirconn"

    def provenance(self) -> dict:
        return json.loads((self.summary.run_dir / "provenance.json").read_text(encoding="utf-8"))


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


def _controls_key(unit: InputUnit, cfg: PipelineConfig) -> str:
    return task_key(unit.sha256, analysis_hash(cfg), "controls", dirconn.__version__)


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
    assert [t.label for t in tasks] == [
        f"{lab}:{m}" for lab in ROI_LABELS for m in ("controls", "gc", "pdc")
    ]
    assert tasks[0].payload == ControlsTask(units[0])
    assert tasks[0].key == _controls_key(units[0], cfg)
    assert tasks[1].payload == UnitTask(units[0], "gc")
    assert tasks[1].key == task_key(units[0].sha256, analysis_hash(cfg), "gc", dirconn.__version__)
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
    summary = first_run.summary
    assert (summary.done, summary.skipped, summary.failed) == (10, 0, 0)
    assert summary.exit_code == 0
    root = first_run.root
    for name in ("dirconn.toml", "dataset_description.json"):
        assert (root / name).is_file(), name
    for name in ("provenance.json", "environment-lock.txt"):
        assert not (root / name).exists(), name
    assert load_config(root / "dirconn.toml") == first_run.cfg

    run_dir = summary.run_dir
    assert run_dir.parent == root / "logs"
    assert run_dir.name.startswith("run-")
    assert (run_dir / "environment-lock.txt").read_text(encoding="utf-8").strip()
    assert (run_dir / "provenance.json").is_file()
    assert load_config(run_dir / "dirconn.toml") == first_run.cfg
    progress = (run_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(progress) == 10
    assert "run started" in (run_dir / "dirconn.log").read_text(encoding="utf-8")

    for unit in first_run.units:
        directory = unit_dir(root, unit.entities(), unit.datatype)
        stem = unit.label()
        for suffix in ("conn.nc", "conn.json", "edges.tsv"):
            assert (directory / f"{stem}_desc-gc_{suffix}").is_file(), (stem, suffix)
        for suffix in ("conn.nc", "conn.json"):
            assert (directory / f"{stem}_desc-trgc_{suffix}").is_file(), (stem, suffix)
        gc_task = next(t for t in plan_tasks([unit], first_run.cfg) if t.label.endswith(":gc"))
        sidecar = directory / f"{stem}_desc-gc_conn.json"
        assert result_task_key(sidecar) == gc_task.key
        controls = _controls_key(unit, first_run.cfg)
        assert result_task_key(directory / f"{stem}_desc-trgc_conn.json") == controls
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        assert meta["Statistics"] and all(s["NSurrogates"] == 10 for s in meta["Statistics"])
        diag = json.loads(diagnostics_path(root, unit).read_text(encoding="utf-8"))
        assert diag["task_key"] == controls
        assert "valid_fraction" in diag
    diagnostics = sorted(root.rglob("figures/*_diagnostics.json"))
    assert [p.name for p in diagnostics] == [f"{lab}_diagnostics.json" for lab in ROI_LABELS]
    assert len(list((first_run.out / "work").glob("prepared-*.npz"))) == 5


def test_provenance_lists_inputs_and_counts(first_run: Run) -> None:
    prov = first_run.provenance()
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
    assert prov["tasks"] == {"done": 10, "skipped": 0, "failed": 0}


def test_second_run_skips_everything(bids_tiny_copy: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    out = tmp_path / "out"
    first = _run(bids_tiny_copy, out, cfg)
    assert first.summary.done == 10
    again = _run(bids_tiny_copy, out, cfg)
    assert again.summary.skipped == 10
    assert again.summary.done == 0
    assert again.summary.run_dir != first.summary.run_dir
    assert again.provenance()["tasks"] == {"done": 0, "skipped": 10, "failed": 0}
    assert first.provenance()["tasks"] == {"done": 10, "skipped": 0, "failed": 0}


def test_rerun_with_different_analysis_is_refused(first_run: Run) -> None:
    other = first_run.cfg.model_copy(
        update={"statistics": first_run.cfg.statistics.model_copy(update={"n_surrogates": 11})}
    )
    before = (first_run.root / "dirconn.toml").read_text(encoding="utf-8")
    runs_before = sorted((first_run.root / "logs").iterdir())
    with pytest.raises(ConfigMismatchError, match="analysis"):
        _run(first_run.bids, first_run.out, other)
    assert (first_run.root / "dirconn.toml").read_text(encoding="utf-8") == before
    assert sorted((first_run.root / "logs").iterdir()) == runs_before


def test_rerun_with_different_input_is_refused(first_run: Run) -> None:
    other = first_run.cfg.model_copy(
        update={"input": first_run.cfg.input.model_copy(update={"node_kinds": ("cortical",)})}
    )
    with pytest.raises(ConfigMismatchError, match="input"):
        _run(first_run.bids, first_run.out, other)


def test_rerun_with_different_execution_is_allowed(bids_tiny_copy: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    out = tmp_path / "out"
    _run(bids_tiny_copy, out, cfg)
    stored = (out / "dirconn" / "dirconn.toml").read_text(encoding="utf-8")
    other = cfg.model_copy(
        update={
            "execution": cfg.execution.model_copy(update={"fail_fast": True}),
            "output": cfg.output.model_copy(update={"long_table": False}),
        }
    )
    again = _run(bids_tiny_copy, out, other)
    assert again.summary.skipped == 10
    assert (out / "dirconn" / "dirconn.toml").read_text(encoding="utf-8") == stored
    assert load_config(again.summary.run_dir / "dirconn.toml") == other


def test_failing_method_is_isolated(bids_tiny_copy: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path, "\n[estimation.gc]\norder = 100000\n")
    run = _run(bids_tiny_copy, tmp_path / "out", cfg)
    assert run.summary.failed == 5
    assert run.summary.done == 5  # the controls tasks
    assert run.summary.exit_code == 1
    assert len(list(run.summary.run_dir.glob("crash-*.txt"))) == 5
    assert not list(run.root.rglob("*desc-gc*"))
    assert len(list(run.root.rglob("*_desc-trgc_conn.json"))) == 5
    assert len(list(run.root.rglob("figures/*_diagnostics.json"))) == 5
    for name in ("dirconn.toml", "dataset_description.json"):
        assert (run.root / name).is_file(), name
    assert run.provenance()["tasks"] == {"done": 5, "skipped": 0, "failed": 5}


def test_failing_method_leaves_the_other_methods(bids_tiny_copy: Path, tmp_path: Path) -> None:
    base = BASE_CONFIG.replace('methods = ["gc"]', 'methods = ["gc", "pdc"]')
    cfg = _config(tmp_path, "\n[estimation.gc]\norder = 100000\n", base=base)
    run = _run(bids_tiny_copy, tmp_path / "out", cfg)
    assert (run.summary.done, run.summary.failed) == (10, 5)
    assert run.summary.exit_code == 1
    assert not list(run.root.rglob("*desc-gc*"))
    assert len(list(run.root.rglob("*_desc-pdc_conn.json"))) == 5
    assert len(list(run.root.rglob("*_desc-pdc_edges.tsv"))) == 5
    assert len(list(run.root.rglob("*_desc-trgc_conn.json"))) == 5
    assert len(list(run.root.rglob("figures/*_diagnostics.json"))) == 5
    assert "run started" in (run.summary.run_dir / "dirconn.log").read_text(encoding="utf-8")


def test_fail_fast_still_writes_provenance(bids_tiny_copy: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path, "\n[execution]\nfail_fast = true\n\n[estimation.gc]\norder = 100000\n")
    run = _run(bids_tiny_copy, tmp_path / "out", cfg)
    assert (run.summary.done, run.summary.failed) == (1, 1)  # first controls task, then gc
    assert run.summary.exit_code == 1
    assert run.provenance()["tasks"] == {"done": 1, "skipped": 0, "failed": 1}


def test_workers_two_matches_workers_one(first_run: Run, tmp_path: Path) -> None:
    two = _run(first_run.bids, tmp_path / "out2", first_run.cfg, workers=2)
    assert two.summary.done == 10
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


def test_cache_key_changes_with_the_dirconn_version(
    bids_tiny: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    unit = _units(bids_tiny, cfg)[0]
    before = participant._prepare_key(unit, cfg)
    assert participant._prepare_key(unit, cfg) == before
    monkeypatch.setattr(participant, "DIRCONN_VERSION", "999.0.0")
    assert participant._prepare_key(unit, cfg) != before


def test_unreadable_cache_entry_is_recomputed(bids_tiny: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    unit = _units(bids_tiny, cfg)[0]
    work = tmp_path / "work"
    first = _prepared(unit, cfg, work)
    (entry,) = work.glob("prepared-*.npz")
    entry.write_bytes(b"not an npz file")
    again = _prepared(unit, cfg, work)
    np.testing.assert_array_equal(again.data, first.data)
    (entry_after,) = work.glob("prepared-*.npz")
    assert entry_after.read_bytes() != b"not an npz file"


def test_setup_logging_is_idempotent(tmp_path: Path) -> None:
    logger = setup_logging(tmp_path / "a", 1)
    assert logger.name == "dirconn"
    assert logger.level == logging.DEBUG
    assert logger.propagate is False
    n = len(logger.handlers)

    def levels() -> tuple[int, int]:
        (stream,) = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        (file,) = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        return stream.level, file.level

    assert levels() == (logging.INFO, logging.DEBUG)
    assert setup_logging(tmp_path / "a", 2) is logger
    assert len(logger.handlers) == n
    assert levels() == (logging.DEBUG, logging.DEBUG)
    setup_logging(tmp_path / "b", 0)
    assert len(logger.handlers) == n
    assert levels() == (logging.WARNING, logging.DEBUG)
    files = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert [Path(h.baseFilename) for h in files] == [tmp_path / "b" / "dirconn.log"]
    logger.getChild("x").debug("quiet on stderr, kept in the file")
    close_logging()
    assert not [h for h in logger.handlers if getattr(h, "_dirconn_handler", False)]
    text = (tmp_path / "b" / "dirconn.log").read_text(encoding="utf-8")
    assert "DEBUG dirconn.x: quiet on stderr, kept in the file" in text


def test_pipeline_sidecars_record_node_kinds_and_sampling_rate(first_run: Run) -> None:
    for unit in first_run.units:
        ts = _prepared(unit, first_run.cfg, first_run.out / "work")
        kinds = {ch.name: ch.node_kind for ch in ts.channels}
        directory = unit_dir(first_run.root, unit.entities(), unit.datatype)
        for desc in ("gc", "trgc"):
            path = directory / f"{unit.label()}_desc-{desc}_conn.json"
            meta = json.loads(path.read_text(encoding="utf-8"))
            assert meta["NodeKinds"] == [kinds[name] for name in meta["NodeLabels"]], path
            assert meta["SamplingFrequency"] == ts.sfreq, path
        assert "direct" in kinds.values()


def _run_units(bids: Path, out: Path, cfg: PipelineConfig, units: list[InputUnit]) -> Run:
    resources = Resources(nprocs=1, omp_nthreads=1, mem_mb=None, work_dir=out / "work")
    summary = run_participant(
        cfg,
        bids_dir=bids,
        out_dir=out,
        units=units,
        resources=resources,
        argv=["dirconn"],
        verbosity=1,
    )
    return Run(bids, out, cfg, units, summary)


TIME_VARYING_CONFIG = """\
[prepare]
resample_hz = "null"
bad_segments = "ignore"

[estimation]
methods = ["adtf"]
order = 2

[estimation.bands]
beta = [13.0, 30.0]

[estimation.adtf]
n_freqs = 16
fmax = 40.0

[statistics]
n_surrogates = 2
block_seconds = 2.0

[controls]
time_reversal = false
"""


def test_time_varying_method_on_a_continuous_record_is_not_tested(
    bids_tiny_copy: Path, tmp_path: Path
) -> None:
    cfg = _config(tmp_path, base=TIME_VARYING_CONFIG)
    units = _units(bids_tiny_copy, cfg)[:1]
    run = _run_units(bids_tiny_copy, tmp_path / "out", cfg, units)
    assert (run.summary.done, run.summary.failed) == (2, 0)
    (table,) = run.root.rglob("*_desc-adtf_edges.tsv")
    edges = read_edges(table)
    assert list(edges.columns) == list(TIME_EDGE_COLUMNS)
    n_times = edges["time"].nunique()
    assert n_times > 1
    assert len(edges) == 12 * len(edges["band"].unique()) * n_times
    # block shuffling would fit the test on 2 s blocks, another time axis: not tested
    assert edges["p"].isna().all()
    (sidecar,) = run.root.rglob("*_desc-adtf_conn.json")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["Statistics"] == []
    assert "block shuffling" in meta["StatisticsNote"]


EPOCHED_SENSOR_CONFIG = """\
[input]
level = "sensor"
datatypes = ["eeg"]

[prepare]
resample_hz = "null"

[prepare.lock]
events = ["cue"]
tmin = -0.5
tmax = 1.0

[estimation]
methods = ["gc"]
order = 3

[statistics]
n_surrogates = 5
"""


def test_short_epochs_run_the_controls_task(bids_tiny_copy: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path, base=EPOCHED_SENSOR_CONFIG)
    units = [u for u in _units(bids_tiny_copy, cfg) if u.task == "rest"]
    assert len(units) == 1
    run = _run_units(bids_tiny_copy, tmp_path / "out", cfg, units)
    assert (run.summary.done, run.summary.failed) == (2, 0)
    diag = json.loads(diagnostics_path(run.root, units[0]).read_text(encoding="utf-8"))
    assert diag["task_key"] == _controls_key(units[0], cfg)
    assert len(list(run.root.rglob("*_desc-trgc_conn.json"))) == 1


def test_time_varying_method_on_epochs_keeps_p_values_per_time_point(
    bids_tiny_copy: Path, tmp_path: Path
) -> None:
    base = EPOCHED_SENSOR_CONFIG.replace('methods = ["gc"]', 'methods = ["wdtf"]').replace(
        "order = 3", "order = 2\n\n[estimation.wdtf]\nwindow = 0.5\nstep = 0.25"
    )
    cfg = _config(tmp_path, base=base)
    units = [u for u in _units(bids_tiny_copy, cfg) if u.task == "rest"]
    run = _run_units(bids_tiny_copy, tmp_path / "out", cfg, units)
    assert run.summary.failed == 0
    (table,) = run.root.rglob("*_desc-wdtf_edges.tsv")
    edges = read_edges(table)
    assert edges["time"].nunique() > 1
    assert edges["p"].notna().all() and edges["p_adj"].notna().all()
