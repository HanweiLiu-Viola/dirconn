"""Tests for the idempotent task executor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dirconn.workflows.executor import RunSummary, Task, run, task_key


def _double(task: Task) -> int:
    """Trivial top-level function so loky can pickle it."""
    return task.payload * 2


def _boom_on_bad(task: Task) -> int:
    if task.payload == "bad":
        raise ValueError("kaput")
    return 1


def _tasks(*payloads):
    return [Task(task_key(str(p)), f"t-{p}", p) for p in payloads]


def _progress(log_dir: Path) -> list[dict]:
    lines = (log_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_done_tasks_are_skipped_and_logged(tmp_path):
    tasks = _tasks(1, 2, 3)
    called = []

    def fn(task):
        called.append(task.label)

    summary = run(
        tasks,
        fn,
        workers=1,
        fail_fast=False,
        log_dir=tmp_path / "logs",
        is_done=lambda t: t.payload == 2,
    )
    assert summary == RunSummary(done=2, skipped=1, failed=0, failures=())
    assert summary.exit_code == 0
    assert called == ["t-1", "t-3"]
    rows = _progress(tmp_path / "logs")
    assert rows[0]["status"] == "skipped" and rows[0]["seconds"] == 0
    assert [r["status"] for r in rows] == ["skipped", "done", "done"]
    assert set(rows[0]) == {"key", "label", "status", "seconds", "crash", "finished"}


def test_failed_task_is_isolated_and_recorded(tmp_path):
    tasks = _tasks(1, "bad", 3)
    log_dir = tmp_path / "logs"
    summary = run(
        tasks, _boom_on_bad, workers=1, fail_fast=False, log_dir=log_dir, is_done=lambda t: False
    )
    assert (summary.done, summary.skipped, summary.failed) == (2, 0, 1)
    assert summary.exit_code == 1
    assert summary.failures[0][0] == "t-bad"
    assert "ValueError: kaput" in summary.failures[0][1]
    crash = (log_dir / f"crash-{tasks[1].key}.txt").read_text(encoding="utf-8")
    assert "t-bad" in crash and "'bad'" in crash and "kaput" in crash
    failed = [r for r in _progress(log_dir) if r["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["crash"]


def test_fail_fast_stops_after_the_failure(tmp_path):
    tasks = _tasks(1, "bad", 3)
    log_dir = tmp_path / "logs"
    with pytest.raises(RuntimeError, match="t-bad failed") as info:
        run(
            tasks, _boom_on_bad, workers=1, fail_fast=True, log_dir=log_dir, is_done=lambda t: False
        )
    summary = info.value.summary  # type: ignore[attr-defined]
    assert (summary.done, summary.failed) == (1, 1)
    assert [r["label"] for r in _progress(log_dir)] == ["t-1", "t-bad"]
    assert (log_dir / f"crash-{tasks[1].key}.txt").exists()


def test_two_workers_give_the_same_progress_as_one(tmp_path):
    tasks = _tasks(1, 2, 3)
    out = {}
    for workers in (1, 2):
        log_dir = tmp_path / f"w{workers}"
        run(
            tasks,
            _double,
            workers=workers,
            fail_fast=False,
            log_dir=log_dir,
            is_done=lambda t: False,
        )
        out[workers] = {(r["key"], r["status"]) for r in _progress(log_dir)}
    assert out[1] == out[2] and len(out[1]) == 3


def test_task_key_is_stable_and_order_sensitive():
    assert task_key("a", "b") == task_key("a", "b")
    assert task_key("a", "b") != task_key("b", "a")
    assert task_key("ab", "c") != task_key("a", "bc")
    assert len(task_key("x")) == 16
