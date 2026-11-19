"""Participant-level workflow: every task of a run, from reading the inputs to the results.

`run_participant` makes sure the derivatives dataset exists (configuration, dataset description),
opens a run directory below ``logs/`` (log, progress, crash files, lock file, provenance), plans
the tasks, runs them with the idempotent executor and records the run in ``provenance.json``.

Each unit gets one task per method (`run_unit_task`: estimate, test, write ``desc-<method>``)
and one controls task (`run_controls_task`: diagnostics and the time-reversal control), so a
failing method never takes the controls of its unit down with it. All tasks of a unit share the
prepared series through a cache in the work directory.
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
from dirconn.workflows.log import close_logging, setup_logging
from dirconn.workflows.provenance import (
    DIRCONN_VERSION,
    Resources,
    run_provenance,
    write_environment_lock,
)
from dirconn.workflows.steps.control import time_reversal_control
from dirconn.workflows.steps.diagnose import diagnose
from dirconn.workflows.steps.estimate import build_estimator, estimate, select_shared_order
from dirconn.workflows.steps.prepare import prepare
from dirconn.workflows.steps.test import test_significance

__all__ = [
    "CONTROLS",
    "ConfigMismatchError",
    "ControlsTask",
    "ParticipantSummary",
    "UnitTask",
    "check_existing_config",
    "diagnostics_path",
    "events_tsv_for",
    "plan_tasks",
    "run_controls_task",
    "run_participant",
    "run_unit_task",
]

_log = logging.getLogger("dirconn.workflows.participant")

CONFIG_FILE = "dirconn.toml"
#: Name of the per-unit controls task in task keys and labels.
CONTROLS = "controls"


class ConfigMismatchError(RuntimeError):
    """Raised when an output directory already holds results of a different analysis."""


@dataclass(frozen=True)
class UnitTask:
    """Payload of a method task.

    Parameters
    ----------
    unit : InputUnit
        The recording.
    method : str
        Registry name of the estimator.
    """

    unit: InputUnit
    method: str


@dataclass(frozen=True)
class ControlsTask:
    """Payload of the controls task of a unit (diagnostics and time-reversal control).

    Parameters
    ----------
    unit : InputUnit
        The recording.
    """

    unit: InputUnit


@dataclass(frozen=True)
class ParticipantSummary(RunSummary):
    """`RunSummary` of a participant run, plus where the run recorded itself.

    Parameters
    ----------
    run_dir : Path
        The run directory ``<dataset>/logs/run-<YYYYmmddTHHMMSS>-<pid>[-<n>]`` (start time in
        UTC; ``-<n>`` only when a run of the same process started in the same second) with
        ``dirconn.log``, ``progress.jsonl``, crash files, ``environment-lock.txt``, the run's
        effective ``dirconn.toml`` and ``provenance.json``.
    """

    run_dir: Path


def _error_text(exc: ValueError) -> str:
    """One line per problem of a failed configuration load, joined with ``"; "``."""
    errors = getattr(exc, "errors", None)
    if callable(errors):  # pydantic ValidationError
        return "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in errors()
        )
    return str(exc)


def check_existing_config(root: Path, cfg: PipelineConfig) -> None:
    """Refuse to add results of a different analysis to an existing derivatives dataset.

    Parameters
    ----------
    root : Path
        Derivatives dataset root.
    cfg : PipelineConfig
        Configuration of the new run.

    Raises
    ------
    ConfigMismatchError
        If ``root/dirconn.toml`` exists and differs from ``cfg`` in its ``analysis_hash`` (the
        ``prepare``, ``estimation``, ``statistics`` and ``controls`` sections) or in its
        ``input`` section, or is not a valid configuration. Differences in ``output`` or
        ``execution`` are allowed.
    """
    path = Path(root) / CONFIG_FILE
    if not path.is_file():
        return
    try:
        stored = load_config(path)
    except ValueError as exc:  # TOML syntax and pydantic validation errors
        raise ConfigMismatchError(
            f"{path} is not a valid dirconn configuration ({_error_text(exc)}), so this run "
            "cannot be checked against the results there. Repair the file or use a new output "
            "directory."
        ) from exc
    differ = []
    if analysis_hash(stored) != analysis_hash(cfg):
        differ.append("analysis (prepare, estimation, statistics, controls)")
    if stored.input != cfg.input:
        differ.append("input")
    if differ:
        raise ConfigMismatchError(
            f"{path} was written by a run with a different {' and '.join(differ)} "
            "configuration. Use a new output directory, or run again with the configuration "
            f"stored in {path}."
        )


def plan_tasks(units: Sequence[InputUnit], cfg: PipelineConfig) -> list[Task]:
    """Return the controls task and one task per method for every unit.

    Parameters
    ----------
    units : sequence of InputUnit
        Inputs, in discovery order.
    cfg : PipelineConfig
        Supplies the methods (in configuration order) and the analysis hash.

    Returns
    -------
    list of Task
        Units in the given order; per unit the controls task, then the methods in
        configuration order. A method task has the key
        ``task_key(unit.sha256, analysis_hash(cfg), method, dirconn.__version__)``, the label
        ``"<unit label>:<method>"`` and a `UnitTask` payload; the controls task the same with
        ``"controls"`` in place of the method and a `ControlsTask` payload.
    """
    digest = analysis_hash(cfg)
    tasks: list[Task] = []
    for unit in units:
        tasks.append(
            Task(
                key=task_key(unit.sha256, digest, CONTROLS, DIRCONN_VERSION),
                label=f"{unit.label()}:{CONTROLS}",
                payload=ControlsTask(unit),
            )
        )
        tasks.extend(
            Task(
                key=task_key(unit.sha256, digest, method, DIRCONN_VERSION),
                label=f"{unit.label()}:{method}",
                payload=UnitTask(unit, method),
            )
            for method in cfg.estimation.methods
        )
    return tasks


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


def diagnostics_path(root: Path, unit: InputUnit) -> Path:
    """Return ``<unit dir>/figures/<label>_diagnostics.json`` of ``unit`` below ``root``."""
    figures = unit_dir(root, unit.entities(), unit.datatype) / "figures"
    return figures / f"{unit.label()}_diagnostics.json"


def _file_sha256(path: Path | None) -> str | None:
    return None if path is None else hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_key(unit: InputUnit, cfg: PipelineConfig) -> str:
    """Cache key of the prepared series: the unit's content and everything that shapes it."""
    material = {
        "prepare": cfg.prepare.model_dump(mode="json"),
        "node_kinds": list(cfg.input.node_kinds),
        "version": DIRCONN_VERSION,
        "events": _file_sha256(events_tsv_for(unit)),
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


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to a temporary file next to ``path`` and move it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    _replace(tmp, path)


def _prepared(unit: InputUnit, cfg: PipelineConfig, work_dir: Path) -> TimeSeries:
    """Read and prepare ``unit``, or load the cached result of an earlier task.

    A cache entry that cannot be loaded is deleted and recomputed.
    """
    work_dir = Path(work_dir)
    cached = work_dir / f"prepared-{_prepare_key(unit, cfg)}.npz"
    sidecar = cached.with_suffix(".json")
    if cached.is_file() and sidecar.is_file():
        try:
            return load_timeseries(cached)
        except Exception:
            _log.warning("discarding unreadable cache entry %s", cached, exc_info=True)
            cached.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
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
        bad_annotations="ignore" if cfg.prepare.bad_segments == "ignore" else "mask",
    )
    ts = prepare(raw, cfg.prepare, events_tsv=events_tsv_for(unit))
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp = work_dir / f"{cached.stem}-{os.getpid()}-tmp.npz"
    save_timeseries(ts, tmp)
    # The sidecar goes first: a cache entry counts once its .npz exists.
    _replace(tmp.with_suffix(".json"), sidecar)
    _replace(tmp, cached)
    return ts


def _write(
    root: Path,
    unit: InputUnit,
    task: Task,
    cfg: PipelineConfig,
    bids_dir: Path,
    *,
    ts: TimeSeries,
    desc: str,
    conn: Connectivity,
    bands: Mapping[str, Connectivity],
    tests: Sequence[Any],
) -> ResultFiles:
    """Write one result; node kinds and sampling rate come from the prepared series ``ts``."""
    kinds = {ch.name: ch.node_kind for ch in ts.channels}
    node_kinds = [kinds[name] for name in conn.source] if set(conn.source) <= set(kinds) else None
    return write_result(
        root,
        unit.entities(),
        unit.datatype,
        desc=desc,
        conn=conn,
        bands=bands,
        tests=tests,
        sources=unit.files,
        bids_dir=Path(bids_dir),
        task_key=task.key,
        long_table=cfg.output.long_table,
        node_kinds=node_kinds,
        sfreq=ts.sfreq,
    )


def run_controls_task(
    task: Task,
    *,
    cfg: PipelineConfig,
    bids_dir: Path,
    out_dir: Path,
    work_dir: Path,
) -> Path:
    """Run the controls task of a unit: diagnostics and the time-reversal control.

    Parameters
    ----------
    task : Task
        A controls task of `plan_tasks`; its payload is a `ControlsTask`.
    cfg : PipelineConfig
        Run configuration.
    bids_dir : Path
        BIDS root; the sources in the sidecars are relative to it.
    out_dir : Path
        Output directory; results go to ``derivative_root(out_dir)``.
    work_dir : Path
        Holds the prepared-series cache.

    Returns
    -------
    Path
        The diagnostics file (`diagnostics_path`), which records the task key as ``task_key``.

    Notes
    -----
    With ``controls.time_reversal`` the TRGC result is written as ``desc-trgc`` (no tests)
    after the diagnostics, and its sidecar's ``TaskKey`` marks the task as done; otherwise the
    diagnostics file does.
    """
    payload: ControlsTask = task.payload
    unit = payload.unit
    root = derivative_root(out_dir)
    ts = _prepared(unit, cfg, work_dir)
    selection = select_shared_order(ts, cfg.estimation)
    record = diagnose(ts, cfg).to_dict()
    record["order_selection"] = None if selection is None else selection.to_dict()
    record["task_key"] = task.key
    path = diagnostics_path(root, unit)
    _write_atomic(path, json.dumps(record, indent=2) + "\n")
    control = time_reversal_control(ts, cfg.estimation, cfg.controls, selection)
    if control is not None:
        _write(
            root, unit, task, cfg, bids_dir, ts=ts, desc="trgc", conn=control, bands={}, tests=()
        )
    return path


def run_unit_task(
    task: Task,
    *,
    cfg: PipelineConfig,
    bids_dir: Path,
    out_dir: Path,
    work_dir: Path,
    seed: int,
) -> ResultFiles:
    """Run one ``(unit, method)`` task: estimate, test and write ``desc-<method>``.

    Parameters
    ----------
    task : Task
        A method task of `plan_tasks`; its payload is a `UnitTask`.
    cfg : PipelineConfig
        Run configuration.
    bids_dir : Path
        BIDS root; the sources in the sidecars are relative to it.
    out_dir : Path
        Output directory; results go to ``derivative_root(out_dir)``.
    work_dir : Path
        Holds the prepared-series cache ``prepared-<hash>.npz``, shared by the tasks of a unit.
    seed : int
        Run seed; the test of this task uses ``seed + int(task.key[:8], 16) % 10**6``.

    Returns
    -------
    ResultFiles
        Files of the method's result; the sidecar carries the task key.
    """
    payload: UnitTask = task.payload
    unit, method = payload.unit, payload.method
    ts = _prepared(unit, cfg, work_dir)
    # one order for every method of the record, and for the surrogates of its test
    selection = select_shared_order(ts, cfg.estimation)
    order = None if selection is None else selection.order
    est = estimate(ts, method, cfg.estimation, selection)
    estimator = build_estimator(method, cfg.estimation, ts, order)
    tests = test_significance(
        ts, estimator, est, cfg.statistics, seed=seed + int(task.key[:8], 16) % 10**6
    )
    return _write(
        derivative_root(out_dir),
        unit,
        task,
        cfg,
        bids_dir,
        ts=ts,
        desc=method,
        conn=est.full,
        bands=est.bands,
        tests=tests,
    )


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
    if isinstance(task.payload, ControlsTask):
        run_controls_task(task, cfg=cfg, bids_dir=bids_dir, out_dir=out_dir, work_dir=work_dir)
    else:
        run_unit_task(
            task, cfg=cfg, bids_dir=bids_dir, out_dir=out_dir, work_dir=work_dir, seed=seed
        )


def _sidecar(root: Path, unit: InputUnit, desc: str) -> Path:
    entities = unit.entities()
    name = filename(entities, desc=desc, suffix="conn", ext=".json")
    return unit_dir(root, entities, unit.datatype) / name


def _diagnostics_key(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    key = data.get("task_key") if isinstance(data, dict) else None
    return key if isinstance(key, str) else None


def _is_done(root: Path, cfg: PipelineConfig, task: Task) -> bool:
    """Whether the file written last by ``task`` carries its key."""
    payload = task.payload
    if isinstance(payload, ControlsTask):
        if cfg.controls.time_reversal:
            return result_task_key(_sidecar(root, payload.unit, "trgc")) == task.key
        return _diagnostics_key(diagnostics_path(root, payload.unit)) == task.key
    return result_task_key(_sidecar(root, payload.unit, payload.method)) == task.key


def _container() -> dict[str, str | None] | None:
    tag = os.environ.get("DIRCONN_IMAGE_TAG")
    if not tag:
        return None
    return {"Type": "docker", "Tag": tag, "Revision": os.environ.get("DIRCONN_BUILD_REVISION")}


def _ensure_dataset_files(root: Path, cfg: PipelineConfig) -> None:
    """Write ``dirconn.toml`` and ``dataset_description.json`` when absent, atomically."""
    pid = os.getpid()
    config = root / CONFIG_FILE
    if not config.exists():
        tmp = root / f".{CONFIG_FILE}.{pid}.tmp"
        dump_config(cfg, tmp)
        _replace(tmp, config)
    description = root / "dataset_description.json"
    if not description.exists():
        tmp_dir = root / f".dataset_description.{pid}.tmp"
        write_dataset_description(tmp_dir, version=DIRCONN_VERSION, container=_container())
        _replace(tmp_dir / description.name, description)
        tmp_dir.rmdir()


def _make_run_dir(root: Path, started: datetime) -> Path:
    """Create ``root/logs/run-<YYYYmmddTHHMMSS>-<pid>``, with ``-<n>`` appended if taken."""
    base = f"run-{started.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        run_dir = logs / (base if n == 1 else f"{base}-{n}")
        try:
            run_dir.mkdir()
        except FileExistsError:
            n += 1
            continue
        return run_dir


def run_participant(
    cfg: PipelineConfig,
    *,
    bids_dir: Path,
    out_dir: Path,
    units: Sequence[InputUnit],
    resources: Resources,
    argv: Sequence[str],
    verbosity: int,
) -> ParticipantSummary:
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
        Level of the stderr log, see `dirconn.workflows.log.setup_logging`.

    Returns
    -------
    ParticipantSummary
        Task counts and the run directory. Tasks whose last file already carries their key are
        skipped; failed tasks leave a crash file in the run directory. With
        ``execution.fail_fast`` the counts cover the tasks run before the stop.

    Raises
    ------
    ConfigMismatchError
        If the dataset holds results of a different analysis or input configuration
        (`check_existing_config`); nothing is written then.

    Notes
    -----
    ``dirconn.toml`` and ``dataset_description.json`` are written only when absent. Every run
    gets its own directory below ``logs/`` (see `ParticipantSummary`). The log handlers are
    removed when the function returns.
    """
    started = datetime.now(timezone.utc)
    root = derivative_root(out_dir)
    check_existing_config(root, cfg)

    root.mkdir(parents=True, exist_ok=True)
    run_dir = _make_run_dir(root, started)
    setup_logging(run_dir, verbosity)
    try:
        Path(resources.work_dir).mkdir(parents=True, exist_ok=True)
        _ensure_dataset_files(root, cfg)
        write_environment_lock(run_dir)
        # execution/output may differ between runs; each run records its own configuration.
        dump_config(cfg, run_dir / CONFIG_FILE)

        tasks = plan_tasks(units, cfg)
        _log.info(
            "run started: dirconn %s, %d units, %d tasks, %d worker(s), output %s, run dir %s",
            DIRCONN_VERSION,
            len(units),
            len(tasks),
            resources.nprocs,
            root,
            run_dir,
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
                log_dir=run_dir,
                is_done=lambda t: _is_done(root, cfg, t),
            )
        except RuntimeError as exc:
            partial = getattr(exc, "summary", None)
            if not isinstance(partial, RunSummary):
                raise
            _log.error("%s", exc)
            summary = partial

        for label, _ in summary.failures:
            _log.warning("task %s failed; see the crash file in %s", label, run_dir)
        _log.info(
            "run finished: done %d, skipped %d, failed %d",
            summary.done,
            summary.skipped,
            summary.failed,
        )
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
        (run_dir / "provenance.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        close_logging()
    return ParticipantSummary(
        done=summary.done,
        skipped=summary.skipped,
        failed=summary.failed,
        failures=summary.failures,
        run_dir=run_dir,
    )
