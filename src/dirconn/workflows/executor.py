"""Idempotent, failure-isolating task executor.

The executor runs independent ``(unit, method)`` tasks, skips those already done, isolates failures
(a crash file per failed task) and appends one line per task to ``progress.jsonl``. It knows nothing
about the analysis it runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["RunSummary", "Task", "run", "task_key"]

_log = logging.getLogger("dirconn.workflows.executor")


@dataclass(frozen=True)
class Task:
    """One unit of work.

    Parameters
    ----------
    key : str
        Stable identifier, see `task_key`.
    label : str
        Human-readable name used in logs and messages.
    payload : Any
        Whatever the task function needs; must be picklable when ``workers > 1``.
    """

    key: str
    label: str
    payload: Any


@dataclass(frozen=True)
class RunSummary:
    """Outcome of `run`.

    Parameters
    ----------
    done, skipped, failed : int
        Task counts.
    failures : tuple of (str, str)
        ``(label, traceback text)`` of every failed task.
    """

    done: int
    skipped: int
    failed: int
    failures: tuple[tuple[str, str], ...]

    @property
    def exit_code(self) -> int:
        """Return 1 if any task failed, else 0."""
        return 1 if self.failed else 0


def task_key(*parts: str) -> str:
    """Return a stable 16-hex-digit key for ``parts`` (order sensitive).

    Parameters
    ----------
    *parts : str
        Components identifying the task.

    Returns
    -------
    str
        First 16 hex digits of the SHA-256 of the parts joined by the unit separator.
    """
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def _guarded(fn: Callable[[Task], Any], task: Task) -> tuple[str, str, float, str | None]:
    """Run ``fn(task)`` and never raise.

    Returns
    -------
    tuple
        ``(key, status, seconds, traceback_text)``; status is ``"done"`` or ``"failed"``.
    """
    start = time.perf_counter()
    try:
        fn(task)
    except Exception:
        return task.key, "failed", time.perf_counter() - start, traceback.format_exc()
    return task.key, "done", time.perf_counter() - start, None


def run(
    tasks: Sequence[Task],
    fn: Callable[[Task], Any],
    *,
    workers: int,
    fail_fast: bool,
    log_dir: Path,
    is_done: Callable[[Task], bool],
) -> RunSummary:
    """Run ``fn`` over ``tasks``, skipping finished ones and isolating failures.

    Parameters
    ----------
    tasks : sequence of Task
        Work to do.
    fn : callable
        ``fn(task)``; its return value is ignored. With ``workers > 1`` it must be a top-level
        picklable callable (loky pickles it to the worker processes).
    workers : int
        ``1`` runs a plain loop; more uses ``joblib.Parallel`` with the loky backend.
    fail_fast : bool
        Stop after the first failure. With ``workers > 1`` tasks run in batches of ``workers`` and
        the run stops after the first batch that contains a failure, so the other tasks of that
        batch complete (and are recorded) as well.
    log_dir : Path
        Created if missing; receives ``progress.jsonl`` (one JSON object per task with ``key``,
        ``label``, ``status``, ``seconds``, ``crash``, ``finished``) and ``crash-<key>.txt`` for
        each failure (label, ``repr(payload)``, traceback).
    is_done : callable
        ``is_done(task)`` is true for tasks whose results already exist; they are skipped (and
        logged as such).

    Returns
    -------
    RunSummary
        Counts and failures.

    Raises
    ------
    RuntimeError
        With ``fail_fast`` after a failure; the partial `RunSummary` is attached as ``.summary``.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    progress = log_dir / "progress.jsonl"
    by_key = {t.key: t for t in tasks}
    counts = {"done": 0, "skipped": 0, "failed": 0}
    failures: list[tuple[str, str]] = []

    def record(key: str, status: str, seconds: float, crash: str | None) -> None:
        task = by_key[key]
        crash_name = None
        if crash is not None:
            crash_name = f"crash-{key}.txt"
            (log_dir / crash_name).write_text(
                f"{task.label}\n{task.payload!r}\n\n{crash}", encoding="utf-8"
            )
            failures.append((task.label, crash))
        counts[status] += 1
        row = {
            "key": key,
            "label": task.label,
            "status": status,
            "seconds": seconds,
            "crash": crash_name,
            "finished": datetime.now(timezone.utc).isoformat(),
        }
        with progress.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        _log.info("%s: %s (%.2fs)", task.label, status, seconds)

    todo: list[Task] = []
    for task in tasks:
        if is_done(task):
            record(task.key, "skipped", 0.0, None)
        else:
            todo.append(task)

    def summary() -> RunSummary:
        return RunSummary(counts["done"], counts["skipped"], counts["failed"], tuple(failures))

    def stop(label: str) -> RuntimeError:
        err = RuntimeError(f"task {label} failed; stopping (fail_fast)")
        err.summary = summary()  # type: ignore[attr-defined]
        return err

    if workers <= 1:
        for task in todo:
            key, status, seconds, crash = _guarded(fn, task)
            record(key, status, seconds, crash)
            if status == "failed" and fail_fast:
                raise stop(task.label)
        return summary()

    import joblib

    size = workers if fail_fast else max(len(todo), 1)
    for start in range(0, len(todo), size):
        batch = todo[start : start + size]
        results = joblib.Parallel(n_jobs=workers, backend="loky")(
            joblib.delayed(_guarded)(fn, t) for t in batch
        )
        first_failed = None
        for key, status, seconds, crash in results:
            record(key, status, seconds, crash)
            if status == "failed" and first_failed is None:
                first_failed = by_key[key].label
        if first_failed is not None and fail_fast:
            raise stop(first_failed)
    return summary()
