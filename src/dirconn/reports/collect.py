"""Collect one subject's outputs from a ``dirconn`` derivatives dataset.

Units are enumerated from the ``*_conn.json`` sidecars, which mark a result as complete. Run
directories below ``logs/`` supply the provenance and the task progress from which failures are
derived. Nothing here imports :mod:`dirconn.workflows`; file names follow the layout written by the
participant pipeline.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dirconn.data.result import Connectivity
from dirconn.io.derivatives import read_edges
from dirconn.io.netcdf import read_connectivity

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

__all__ = [
    "MethodOutput",
    "RunRecord",
    "SubjectOutputs",
    "TaskFailure",
    "UnitOutputs",
    "collect_subject",
    "list_subjects",
    "selected_order",
]

CONTROL_DESC = "trgc"
_SIDECAR_SUFFIX = "_conn.json"


@dataclass(frozen=True)
class MethodOutput:
    """The files of one method's result for one unit.

    Parameters
    ----------
    desc : str
        Value of the ``desc`` entity (the method name).
    sidecar : dict
        Parsed JSON sidecar.
    nc : Path
        The NetCDF file.
    edges : Path or None
        The ``_edges.tsv`` table, ``None`` when it is missing.
    """

    desc: str
    sidecar: dict[str, Any]
    nc: Path
    edges: Path | None

    def connectivity(self) -> Connectivity:
        """Read the NetCDF result (not cached)."""
        return read_connectivity(self.nc)

    def edge_table(self) -> pd.DataFrame | None:
        """Return the edge table, with its ``time`` column for time-resolved results."""
        return None if self.edges is None else read_edges(self.edges)

    @property
    def statistics_note(self) -> str | None:
        """Why the result is untested (``StatisticsNote`` of the sidecar), if recorded."""
        note = self.sidecar.get("StatisticsNote")
        return note if isinstance(note, str) else None

    @property
    def time_resolved(self) -> bool:
        """Whether the edge table has a ``time`` column or the NetCDF time axis has > 1 entry."""
        if self.edges is not None:
            return "time" in read_edges(self.edges).columns
        return int(self.connectivity().times.shape[0]) > 1


@dataclass(frozen=True)
class UnitOutputs:
    """All results of one unit (recording).

    Parameters
    ----------
    label : str
        Unit label, the file-name part before ``_desc-``.
    entities : dict of str to str
        ``sub``, ``ses``, ``task``, ``run`` and any other key-value parts of the label.
    datatype : str
        Name of the directory holding the files (``"eeg"``, ``"roi"``, ...).
    figures_dir : Path
        ``<unit dir>/figures``.
    methods : dict of str to MethodOutput
        Results by method; the ``trgc`` control is not included.
    control : MethodOutput or None
        The ``desc-trgc`` result.
    diagnostics : dict or None
        Parsed ``<label>_diagnostics.json`` (``order_selection`` is ``None`` for a fixed order).
    """

    label: str
    entities: dict[str, str]
    datatype: str
    figures_dir: Path
    methods: dict[str, MethodOutput]
    control: MethodOutput | None
    diagnostics: dict[str, Any] | None


@dataclass(frozen=True)
class RunRecord:
    """One run directory below ``logs/``.

    Parameters
    ----------
    run_dir : Path
        The run directory.
    provenance : dict or None
        Parsed ``provenance.json``; ``None`` when missing or unreadable.
    progress : list of dict
        Rows of ``progress.jsonl``.
    bad_lines : int
        Number of unparsable lines of ``progress.jsonl`` that were skipped.
    """

    run_dir: Path
    provenance: dict[str, Any] | None
    progress: list[dict[str, Any]]
    bad_lines: int


@dataclass(frozen=True)
class TaskFailure:
    """A task whose last progress row (over all runs) is ``failed``.

    Parameters
    ----------
    label : str
        Task label, ``"<unit label>:<method>"``.
    crash_file : Path or None
        The crash file in the run directory, ``None`` if not recorded or missing.
    run_dir : Path
        The run directory of the failing row.
    """

    label: str
    crash_file: Path | None
    run_dir: Path


@dataclass(frozen=True)
class SubjectOutputs:
    """Everything on disk for one subject.

    Parameters
    ----------
    root : Path
        Dataset root (``<out>/dirconn``).
    subject : str
        Subject label without ``sub-``.
    units : tuple of UnitOutputs
        Units of this subject, sorted by label.
    runs : tuple of RunRecord
        All run directories, sorted by name (names start with the UTC time).
    failures : tuple of TaskFailure
        This subject's tasks that still fail.
    config : dict or None
        Parsed ``dirconn.toml`` of the dataset.
    """

    root: Path
    subject: str
    units: tuple[UnitOutputs, ...]
    runs: tuple[RunRecord, ...]
    failures: tuple[TaskFailure, ...]
    config: dict[str, Any] | None

    def latest_run_for(self, label: str) -> RunRecord | None:
        """Return the newest run whose provenance ``inputs`` contain ``label``, else ``None``."""
        for run in reversed(self.runs):
            inputs = (run.provenance or {}).get("inputs") or []
            if any(isinstance(i, dict) and i.get("label") == label for i in inputs):
                return run
        return None


def list_subjects(root: Path) -> list[str]:
    """Return the subjects below ``root`` (``sub-*`` directories, prefix removed), sorted."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.name[4:] for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-"))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _parse_label(label: str) -> dict[str, str]:
    entities: dict[str, str] = {}
    for part in label.split("_"):
        key, sep, value = part.partition("-")
        if sep:
            entities[key] = value
    return entities


def _method_output(sidecar_path: Path, desc: str) -> MethodOutput | None:
    stem = sidecar_path.name[: -len(".json")]
    nc = sidecar_path.with_name(stem + ".nc")
    sidecar = _read_json(sidecar_path)
    if sidecar is None or not nc.is_file():
        return None
    edges = sidecar_path.with_name(stem[: -len("_conn")] + "_edges.tsv")
    return MethodOutput(desc, sidecar, nc, edges if edges.is_file() else None)


def _collect_units(root: Path, subject: str) -> tuple[UnitOutputs, ...]:
    base = root / f"sub-{subject}"
    groups: dict[tuple[Path, str], dict[str, Path]] = {}
    for sidecar in sorted(base.rglob(f"*{_SIDECAR_SUFFIX}")):
        label, sep, desc = sidecar.name[: -len(_SIDECAR_SUFFIX)].partition("_desc-")
        if sep:
            groups.setdefault((sidecar.parent, label), {})[desc] = sidecar
    units = []
    for (directory, label), by_desc in groups.items():
        outputs = {
            desc: out
            for desc, path in sorted(by_desc.items())
            if (out := _method_output(path, desc)) is not None
        }
        if not outputs:
            continue
        figures = directory / "figures"
        units.append(
            UnitOutputs(
                label=label,
                entities=_parse_label(label),
                datatype=directory.name,
                figures_dir=figures,
                methods={d: o for d, o in outputs.items() if d != CONTROL_DESC},
                control=outputs.get(CONTROL_DESC),
                diagnostics=_read_json(figures / f"{label}_diagnostics.json"),
            )
        )
    return tuple(sorted(units, key=lambda u: u.label))


def _read_run(run_dir: Path) -> RunRecord:
    provenance = _read_json(run_dir / "provenance.json")
    rows: list[dict[str, Any]] = []
    bad = 0
    try:
        text = (run_dir / "progress.jsonl").read_text(encoding="utf-8")
    except OSError:
        text = ""
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            bad += 1
    return RunRecord(run_dir, provenance, rows, bad)


def _failures(runs: tuple[RunRecord, ...], subject: str) -> tuple[TaskFailure, ...]:
    prefix = f"sub-{subject}_"
    last: dict[str, tuple[dict[str, Any], Path]] = {}
    for run in runs:
        for row in run.progress:
            key = row.get("key")
            if isinstance(key, str):
                last[key] = (row, run.run_dir)
    failures = []
    for row, run_dir in last.values():
        label = row.get("label")
        if row.get("status") != "failed" or not isinstance(label, str):
            continue
        if not label.startswith(prefix):
            continue
        crash = row.get("crash")
        crash_file = run_dir / crash if isinstance(crash, str) else None
        if crash_file is not None and not crash_file.is_file():
            crash_file = None
        failures.append(TaskFailure(label, crash_file, run_dir))
    return tuple(sorted(failures, key=lambda f: f.label))


def collect_subject(root: Path, subject: str) -> SubjectOutputs:
    """Gather the units, runs, failures and configuration of ``subject`` below ``root``.

    Parameters
    ----------
    root : Path
        Dataset root (``<out_dir>/dirconn``).
    subject : str
        Subject label without the ``sub-`` prefix.

    Returns
    -------
    SubjectOutputs
        Results are listed only when both the sidecar and the NetCDF file exist.
    """
    root = Path(root)
    logs = root / "logs"
    run_dirs = (
        sorted((p for p in logs.glob("run-*") if p.is_dir()), key=lambda p: p.name)
        if logs.is_dir()
        else []
    )
    runs = tuple(_read_run(p) for p in run_dirs)
    config_path = root / "dirconn.toml"
    config = None
    if config_path.is_file():
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return SubjectOutputs(
        root=root,
        subject=subject,
        units=_collect_units(root, subject),
        runs=runs,
        failures=_failures(runs, subject),
        config=config,
    )


def selected_order(selection: Mapping[str, Any]) -> int:
    """Return the model order picked by an ``order_selection`` record.

    Parameters
    ----------
    selection : mapping
        ``OrderSelection.to_dict()`` (``criterion``, ``orders`` and one curve per criterion).

    Returns
    -------
    int
        ``orders[argmin(curve)]``, ignoring NaN entries of the curve.

    Raises
    ------
    ValueError
        If the criterion is unknown or the curve is empty, all NaN or not matching ``orders``.
    """
    criterion = selection.get("criterion")
    curve_raw = selection.get(criterion) if isinstance(criterion, str) else None
    if criterion not in ("aic", "bic", "hqc", "cv") or curve_raw is None:
        raise ValueError(f"unknown order criterion {criterion!r}")
    curve = np.asarray(curve_raw, dtype=float)
    orders = list(selection.get("orders", ()))
    if curve.size == 0 or np.all(np.isnan(curve)):
        raise ValueError(f"the {criterion} curve is empty")
    if len(orders) != curve.size:
        raise ValueError("orders and the criterion curve differ in length")
    return int(orders[int(np.nanargmin(curve))])
