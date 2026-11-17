"""Participant-level workflow: every ``(unit, method)`` task of a run, from reading to the results.

`run_participant` prepares the derivatives dataset (configuration, dataset description, lock
file, logs), plans one task per unit and method, runs them with the idempotent executor and
records the run in ``provenance.json``. `run_unit_task` is one task: read (or load the cached
prepared series), prepare, diagnose, estimate, run the controls, test and write the result.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dirconn.data.result import Connectivity
from dirconn.data.timeseries import TimeSeries
from dirconn.io.derivatives import (
    ResultFiles,
    derivative_root,
    filename,
    result_task_key,
    unit_dir,
    write_dataset_description,
    write_result,
)
from dirconn.io.readers import read_unit
from dirconn.workflows import executor
from dirconn.workflows._cache import load_timeseries, save_timeseries
from dirconn.workflows.config import (
    PipelineConfig,
    analysis_hash,
    config_hash,
    dump_config,
    load_config,
)
from dirconn.workflows.executor import RunSummary, Task, task_key
from dirconn.workflows.inputs import InputUnit
from dirconn.workflows.log import setup_logging
from dirconn.workflows.provenance import (
    DIRCONN_VERSION,
    Resources,
    run_provenance,
    write_environment_lock,
)
from dirconn.workflows.steps.control import time_reversal_control
from dirconn.workflows.steps.diagnose import diagnose
from dirconn.workflows.steps.estimate import build_estimator, estimate
from dirconn.workflows.steps.prepare import prepare
from dirconn.workflows.steps.test import test_significance

__all__ = [
    "ConfigMismatchError",
    "UnitTask",
    "check_existing_config",
    "events_tsv_for",
    "plan_tasks",
    "run_participant",
    "run_unit_task",
]

_log = logging.getLogger("dirconn.workflows.participant")

CONFIG_FILE = "dirconn.toml"


class ConfigMismatchError(RuntimeError):
    """Raised when an output directory already holds a run with a different configuration."""


@dataclass(frozen=True)
class UnitTask:
    """Payload of one pipeline task.

    Parameters
    ----------
    unit : InputUnit
        The recording.
    method : str
        Registry name of the estimator.
    """

    unit: InputUnit
    method: str


def check_existing_config(root: Path, cfg: PipelineConfig) -> None:
    """Refuse to add results of a different configuration to an existing derivatives dataset.

    Parameters
    ----------
    root : Path
        Derivatives dataset root.
    cfg : PipelineConfig
        Configuration of the new run.

    Raises
    ------
    ConfigMismatchError
        If ``root/dirconn.toml`` exists and its ``config_hash`` differs from that of ``cfg``.
    """
    path = Path(root) / CONFIG_FILE
    if not path.is_file():
        return
    old = config_hash(load_config(path))
    new = config_hash(cfg)
    if old != new:
        raise ConfigMismatchError(
            f"{path} was written by a run with a different configuration (config hash {old}, "
            f"this run {new}). Use a new output directory, or run again with the configuration "
            f"stored in {path}."
        )


def plan_tasks(units: Sequence[InputUnit], cfg: PipelineConfig) -> list[Task]:
    """Return one task per unit and method.

    Parameters
    ----------
    units : sequence of InputUnit
        Inputs, in discovery order.
    cfg : PipelineConfig
        Supplies the methods (in configuration order) and the analysis hash.

    Returns
    -------
    list of Task
        Units in the given order, methods within a unit in configuration order. The key is
        ``task_key(unit.sha256, analysis_hash(cfg), method, dirconn.__version__)`` (read via
        `dirconn.workflows.provenance.DIRCONN_VERSION`), the label
        ``"<unit label>:<method>"`` and the payload a `UnitTask`.
    """
    digest = analysis_hash(cfg)
    return [
        Task(
            key=task_key(unit.sha256, digest, method, DIRCONN_VERSION),
            label=f"{unit.label()}:{method}",
            payload=UnitTask(unit, method),
        )
        for unit in units
        for method in cfg.estimation.methods
    ]


def events_tsv_for(unit: InputUnit) -> Path | None:
    """Return the ``_events.tsv`` file next to the unit's data file, or ``None`` when absent.

    The name is that of the data file with its trailing ``_<suffix>.<ext>`` replaced by
    ``_events.tsv``.
    """
    data_file = Path(unit.files[0])
    base, _, _ = data_file.stem.rpartition("_")
    if not base:
        return None
    events = data_file.with_name(f"{base}_events.tsv")
    return events if events.is_file() else None


def _prepare_key(unit: InputUnit, cfg: PipelineConfig) -> str:
    """Cache key of the prepared series: the unit's content, what is read of it, and ``prepare``."""
    material = {
        "prepare": cfg.prepare.model_dump(mode="json"),
        "node_kinds": list(cfg.input.node_kinds),
    }
    prepare_hash = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
    return hashlib.sha256((unit.sha256 + prepare_hash).encode()).hexdigest()[:16]


def _replace(source: Path, target: Path) -> None:
    """Move ``source`` onto ``target``; if another process holds ``target``, keep theirs."""
    try:
        os.replace(source, target)
    except PermissionError:
        if not target.exists():
            raise
        source.unlink(missing_ok=True)


def _prepared(unit: InputUnit, cfg: PipelineConfig, work_dir: Path) -> TimeSeries:
    """Read and prepare ``unit``, or load the cached result of an earlier task."""
    work_dir = Path(work_dir)
    cached = work_dir / f"prepared-{_prepare_key(unit, cfg)}.npz"
    if cached.is_file() and cached.with_suffix(".json").is_file():
        return load_timeseries(cached)
    raw = read_unit(
        unit.root,
        subject=unit.subject,
        session=unit.session,
        task=unit.task,
        run=unit.run,
        datatype=unit.datatype,
        level=unit.level,
        description=cfg.input.description,
        node_kinds=cfg.input.node_kinds,
    )
    ts = prepare(raw, cfg.prepare, events_tsv=events_tsv_for(unit))
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp = work_dir / f"{cached.stem}-{os.getpid()}-tmp.npz"
    save_timeseries(ts, tmp)
    # The sidecar goes first: a cache entry counts once its .npz exists.
    _replace(tmp.with_suffix(".json"), cached.with_suffix(".json"))
    _replace(tmp, cached)
    return ts


def run_unit_task(
    task: Task,
    *,
    cfg: PipelineConfig,
    bids_dir: Path,
    out_dir: Path,
    work_dir: Path,
    seed: int,
) -> ResultFiles:
    """Run one ``(unit, method)`` task and write its results.

    Parameters
    ----------
    task : Task
        A task of `plan_tasks`; its payload is a `UnitTask`.
    cfg : PipelineConfig
        Run configuration.
    bids_dir : Path
        BIDS root; the sources in the sidecars are relative to it.
    out_dir : Path
        Output directory; results go to ``derivative_root(out_dir)``.
    work_dir : Path
        Holds the prepared-series cache ``prepared-<hash>.npz``, shared by the methods of a unit.
    seed : int
        Run seed; the test of this task uses ``seed + int(task.key[:8], 16) % 10**6``.

    Returns
    -------
    ResultFiles
        Files of the method's result (``desc-<method>``).

    Notes
    -----
    The task of the first configured method also writes the unit's diagnostics
    (``<unit dir>/figures/<label>_diagnostics.json``) and the time-reversal control
    (``desc-trgc``, no tests), which do not depend on the method. The method's own sidecar is
    written last, so its ``TaskKey`` marks the whole task as done.
    """
    payload: UnitTask = task.payload
    unit, method = payload.unit, payload.method
    root = derivative_root(out_dir)
    entities = unit.entities()
    first = method == cfg.estimation.methods[0]

    ts = _prepared(unit, cfg, work_dir)
    if first:
        figures = unit_dir(root, entities, unit.datatype) / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        (figures / f"{unit.label()}_diagnostics.json").write_text(
            json.dumps(diagnose(ts, cfg).to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    def write(
        desc: str, conn: Connectivity, bands: Mapping[str, Connectivity], tests: Sequence[Any]
    ) -> ResultFiles:
        return write_result(
            root,
            entities,
            unit.datatype,
            desc=desc,
            conn=conn,
            bands=bands,
            tests=tests,
            sources=unit.files,
            bids_dir=Path(bids_dir),
            task_key=task.key,
            long_table=cfg.output.long_table,
        )

    est = estimate(ts, method, cfg.estimation)
    if first:
        control = time_reversal_control(ts, cfg.estimation, cfg.controls)
        if control is not None:
            write("trgc", control, {}, ())

    estimator = build_estimator(method, cfg.estimation)
    tests = test_significance(
        ts, estimator, est, cfg.statistics, seed=seed + int(task.key[:8], 16) % 10**6
    )
    return write(method, est.full, est.bands, tests)


def _run_task(
    task: Task,
    *,
    cfg: PipelineConfig,
    bids_dir: Path,
    out_dir: Path,
    work_dir: Path,
    seed: int,
) -> None:
    """Executor entry point (module level, so loky can pickle it with `functools.partial`)."""
    run_unit_task(task, cfg=cfg, bids_dir=bids_dir, out_dir=out_dir, work_dir=work_dir, seed=seed)


def _result_sidecar(root: Path, task: Task) -> Path:
    payload: UnitTask = task.payload
    entities = payload.unit.entities()
    name = filename(entities, desc=payload.method, suffix="conn", ext=".json")
    return unit_dir(root, entities, payload.unit.datatype) / name


def _container() -> dict[str, str | None] | None:
    tag = os.environ.get("DIRCONN_IMAGE_TAG")
    if not tag:
        return None
    return {"Type": "docker", "Tag": tag, "Revision": os.environ.get("DIRCONN_BUILD_REVISION")}


def run_participant(
    cfg: PipelineConfig,
    *,
    bids_dir: Path,
    out_dir: Path,
    units: Sequence[InputUnit],
    resources: Resources,
    argv: Sequence[str],
    verbosity: int,
) -> RunSummary:
    """Run the participant level over ``units`` and write the derivatives dataset.

    Parameters
    ----------
    cfg : PipelineConfig
        Run configuration.
    bids_dir : Path
        BIDS root of the inputs.
    out_dir : Path
        Output directory; the dataset is ``out_dir/dirconn``.
    units : sequence of InputUnit
        Inputs (from `dirconn.workflows.inputs.discover`).
    resources : Resources
        ``nprocs`` worker processes; ``work_dir`` holds the prepared-series cache.
    argv : sequence of str
        Command line, recorded in the provenance.
    verbosity : int
        Log level, see `dirconn.workflows.log.setup_logging`.

    Returns
    -------
    RunSummary
        Task counts. Tasks whose result already carries their key are skipped; failed tasks
        leave a crash file in ``logs/``. With ``execution.fail_fast`` the summary covers the
        tasks run before the stop.

    Raises
    ------
    ConfigMismatchError
        If the dataset was written with a different configuration; nothing is written then
        except the log.
    """
    started = datetime.now(timezone.utc)
    root = derivative_root(out_dir)
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(resources.work_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, verbosity)
    check_existing_config(root, cfg)

    dump_config(cfg, root / CONFIG_FILE)
    write_dataset_description(root, version=DIRCONN_VERSION, container=_container())
    write_environment_lock(root)

    tasks = plan_tasks(units, cfg)
    _log.info(
        "dirconn %s: %d units, %d tasks, %d worker(s), output %s",
        DIRCONN_VERSION,
        len(units),
        len(tasks),
        resources.nprocs,
        root,
    )
    fn = functools.partial(
        _run_task,
        cfg=cfg,
        bids_dir=Path(bids_dir),
        out_dir=Path(out_dir),
        work_dir=Path(resources.work_dir),
        seed=cfg.execution.seed,
    )
    try:
        summary = executor.run(
            tasks,
            fn,
            workers=resources.nprocs,
            fail_fast=cfg.execution.fail_fast,
            log_dir=log_dir,
            is_done=lambda t: result_task_key(_result_sidecar(root, t)) == t.key,
        )
    except RuntimeError as exc:
        partial = getattr(exc, "summary", None)
        if not isinstance(partial, RunSummary):
            raise
        _log.error("%s", exc)
        summary = partial

    for label, _ in summary.failures:
        _log.warning("task %s failed; see the crash file in %s", label, log_dir)
    _log.info("done %d, skipped %d, failed %d", summary.done, summary.skipped, summary.failed)
    record = run_provenance(
        argv=argv,
        cfg_hash=config_hash(cfg),
        analysis_hash=analysis_hash(cfg),
        units=units,
        started=started,
        finished=datetime.now(timezone.utc),
        resources=resources,
        summary=summary,
    )
    (root / "provenance.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return summary
