"""Tests for ``dirconn.reports.collect``."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from dirconn.reports.collect import (
    collect_subject,
    list_subjects,
    selected_order,
)


def _all_units(root: Path) -> list[Any]:
    return [u for s in list_subjects(root) for u in collect_subject(root, s).units]


def test_list_subjects(pipeline_run: Any) -> None:
    assert list_subjects(pipeline_run.root) == ["01", "02"]
    assert list_subjects(pipeline_run.root / "missing") == []


def test_collect_finds_every_method_and_the_control(pipeline_run: Any) -> None:
    units = _all_units(pipeline_run.root)
    assert len(units) == 5
    for unit in units:
        assert set(unit.methods) == {"gc", "pdc", "adtf"}
        assert "trgc" not in unit.methods
        assert unit.control is not None and unit.control.desc == "trgc"
        assert unit.diagnostics is not None
        assert unit.diagnostics["task_key"]
        assert unit.diagnostics["order_selection"]["criterion"] == "bic"
        assert unit.figures_dir.name == "figures"


def test_collect_groups_units_by_subject(pipeline_run: Any) -> None:
    sub02 = collect_subject(pipeline_run.root, "02")
    assert [u.label for u in sub02.units] == ["sub-02_ses-01_task-rest", "sub-02_ses-02_task-move"]
    assert all("ses-" in u.label for u in sub02.units)
    assert sub02.units[0].entities == {"sub": "02", "ses": "01", "task": "rest"}
    assert sub02.units[0].datatype == sub02.units[0].figures_dir.parent.name
    sub01 = collect_subject(pipeline_run.root, "01")
    assert len(sub01.units) == 3
    assert all(u.label.startswith("sub-01_") for u in sub01.units)
    assert collect_subject(pipeline_run.root, "0").units == ()


def test_collect_reads_runs_and_config(pipeline_run: Any) -> None:
    sub = collect_subject(pipeline_run.root, "01")
    assert [r.run_dir for r in sub.runs] == [pipeline_run.summary.run_dir]
    assert sub.runs[0].provenance is not None
    assert sub.runs[0].bad_lines == 0
    assert sub.runs[0].progress
    assert sub.failures == ()
    assert sub.config is not None and sub.config["estimation"]["order"] == "bic"


def test_time_resolved_output_keeps_the_time_column(pipeline_run: Any) -> None:
    unit = collect_subject(pipeline_run.root, "01").units[0]
    adtf = unit.methods["adtf"]
    assert adtf.time_resolved
    table = adtf.edge_table()
    assert table is not None and "time" in table.columns
    assert adtf.statistics_note is not None and "block shuffling" in adtf.statistics_note
    gc = unit.methods["gc"]
    assert not gc.time_resolved
    assert gc.statistics_note is None
    assert gc.connectivity().times.shape[0] == 1


def test_selected_order_matches_the_model_order(pipeline_run: Any) -> None:
    for unit in _all_units(pipeline_run.root):
        selection = unit.diagnostics["order_selection"]
        order = selected_order(selection)
        assert order == unit.methods["gc"].connectivity().provenance.model_order


def test_selected_order_curves() -> None:
    assert selected_order({"criterion": "bic", "orders": [1, 2, 3], "bic": [3.0, 1.0, 2.0]}) == 2
    assert selected_order({"criterion": "cv", "orders": [1, 2, 3], "cv": [3.0, 2.0, 1.0]}) == 3
    nan = float("nan")
    assert selected_order({"criterion": "aic", "orders": [1, 2, 3], "aic": [nan, 5.0, 4.0]}) == 3
    with pytest.raises(ValueError):
        selected_order({"criterion": "xyz", "orders": [1], "xyz": [1.0]})
    with pytest.raises(ValueError):
        selected_order({"criterion": "bic", "orders": [], "bic": []})


def test_failures_come_from_the_last_progress_row(
    pipeline_run_with_failure: Any, tmp_path: Path
) -> None:
    root = pipeline_run_with_failure.root
    sub = collect_subject(root, "01")
    assert len(sub.failures) == 3
    assert all(f.label.endswith(":gc") for f in sub.failures)
    assert all(f.crash_file is not None and f.crash_file.exists() for f in sub.failures)
    assert len(collect_subject(root, "02").failures) == 2
    # Every unit still has its pdc result.
    assert all(set(u.methods) == {"pdc"} for u in sub.units)

    # A later run in which the task succeeded clears the failure.
    copy = tmp_path / "dirconn"
    shutil.copytree(root, copy)
    first = sorted((copy / "logs").glob("run-*"))[0]
    later = copy / "logs" / "run-99991231T235959-1"
    shutil.copytree(first, later)
    rows = [
        json.loads(line)
        for line in (later / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row["status"] = "done"
        row["crash"] = None
    (later / "progress.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    assert collect_subject(copy, "01").failures == ()
    assert collect_subject(copy, "02").failures == ()


def test_failures_ignore_other_subject_prefixes(tmp_path: Path) -> None:
    run = tmp_path / "logs" / "run-20260101T000000-1"
    run.mkdir(parents=True)
    rows = [
        {"key": "a", "label": "sub-1_task-x:gc", "status": "failed", "crash": "crash-a.txt"},
        {"key": "b", "label": "sub-10_task-x:gc", "status": "failed", "crash": None},
    ]
    (run / "progress.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows) + "not json\n[1]\n", encoding="utf-8"
    )
    sub = collect_subject(tmp_path, "1")
    assert [f.label for f in sub.failures] == ["sub-1_task-x:gc"]
    assert sub.failures[0].crash_file is None
    assert sub.runs[0].bad_lines == 2
    assert sub.runs[0].provenance is None
    assert sub.config is None


def test_latest_run_for_picks_the_newest_run(tmp_path: Path) -> None:
    for name, labels in (
        ("run-20260101T000000-1", ["sub-01_task-a", "sub-01_task-b"]),
        ("run-20260102T000000-1", ["sub-01_task-a"]),
        ("run-20260103T000000-1", ["sub-01_task-c"]),
    ):
        run = tmp_path / "logs" / name
        run.mkdir(parents=True)
        (run / "provenance.json").write_text(
            json.dumps({"inputs": [{"label": lab} for lab in labels]}), encoding="utf-8"
        )
    sub = collect_subject(tmp_path, "01")
    latest_a = sub.latest_run_for("sub-01_task-a")
    assert latest_a is not None and latest_a.run_dir.name == "run-20260102T000000-1"
    latest_b = sub.latest_run_for("sub-01_task-b")
    assert latest_b is not None and latest_b.run_dir.name == "run-20260101T000000-1"
    assert sub.latest_run_for("sub-01_task-z") is None


def test_missing_files_give_none(pipeline_run: Any, tmp_path: Path) -> None:
    copy = tmp_path / "dirconn"
    shutil.copytree(pipeline_run.root, copy)
    unit = collect_subject(copy, "01").units[0]
    pdc = unit.methods["pdc"]
    pdc.nc.with_suffix(".json").unlink()
    assert pdc.edges is not None
    unit.methods["gc"].edges.unlink()  # type: ignore[union-attr]
    (unit.figures_dir / f"{unit.label}_diagnostics.json").unlink()
    (copy / "logs").joinpath(pipeline_run.summary.run_dir.name, "provenance.json").unlink()
    after = collect_subject(copy, "01")
    unit2 = after.units[0]
    assert "pdc" not in unit2.methods
    assert unit2.methods["gc"].edges is None and unit2.methods["gc"].edge_table() is None
    assert unit2.diagnostics is None
    assert after.runs[0].provenance is None
    # A sidecar without its .nc file is not listed either.
    unit2.methods["adtf"].nc.unlink()
    assert "adtf" not in collect_subject(copy, "01").units[0].methods


def test_epoched_run_has_no_order_selection(pipeline_run_epoched: Any) -> None:
    (unit,) = _all_units(pipeline_run_epoched.root)
    assert unit.diagnostics is not None and unit.diagnostics["order_selection"] is None
    wdtf = unit.methods["wdtf"]
    assert wdtf.time_resolved
    assert wdtf.statistics_note is None
    table = wdtf.edge_table()
    assert table is not None and table["time"].nunique() > 1


def test_time_resolved_reads_the_edge_table_once(
    pipeline_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dirconn.reports import collect

    output = collect_subject(pipeline_run.root, "01").units[0].methods["gc"]
    calls: list[Any] = []
    original = collect.read_edges

    def counting(path: Any) -> Any:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(collect, "read_edges", counting)
    first = output.time_resolved
    assert output.time_resolved is first
    assert len(calls) == 1
